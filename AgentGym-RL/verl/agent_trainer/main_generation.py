# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
from collections import defaultdict
import json
from numbers import Integral
import os
from pathlib import Path
import ray
import numpy as np
import hydra
import verl.utils.torch_functional as verl_F

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.agentgym.client import init_env_client
from verl.utils.agentgym.rollout_context import AGENTMEMORY_STEP_RECORD_JSON


_TRAIN_ONLY_ROLLOUT_FLAGS = (
    "AGENTMEMORY_LATEST_OBS_SUFFIX_CREDIT",
)
_MAX_OPTIONAL_CATEGORY_FILE_BYTES = 64 * 1024 * 1024


def _scrub_training_rollout_flags():
    """Prevent a formal eval from inheriting train-only rollout protocols."""

    allow_diagnostic = os.environ.get(
        "AGENTMEMORY_ALLOW_TRAIN_ROLLOUT_FLAGS_IN_EVAL", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if allow_diagnostic:
        return []
    removed = []
    for name in _TRAIN_ONLY_ROLLOUT_FLAGS:
        if name in os.environ:
            removed.append(name)
            os.environ.pop(name, None)
    return removed


def _pad_dataproto_for_dp(data: DataProto, dp_size: int):
    """Repeat real rows until the eval batch is exactly divisible by DP size."""

    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}.")
    real_batch_size = len(data)
    if real_batch_size <= 0:
        raise ValueError("Eval batch must contain at least one real row.")
    dummy_data_size = (-real_batch_size) % dp_size
    if dummy_data_size == 0:
        return data, 0
    repeats = (dummy_data_size + real_batch_size - 1) // real_batch_size
    dummy_source = DataProto.concat([data] * repeats)
    return DataProto.concat([data, dummy_source[:dummy_data_size]]), dummy_data_size


def _read_json_records(path):
    """Read object records from either a JSON array or JSON Lines file."""

    def require_object(record, location):
        if not isinstance(record, dict):
            raise ValueError(
                f"{location}: expected a JSON object record, got {type(record).__name__}."
            )
        return record

    def read_json_lines(array_error=None):
        records = []
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise ValueError(f"{path}:{line_number}: blank JSON Lines record.")
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as line_error:
                        raise ValueError(
                            f"{path}:{line_number}: invalid JSON Lines record "
                            f"({line_error.msg} at column {line_error.colno})."
                        ) from line_error
                    records.append(require_object(record, f"{path}:{line_number}"))
            if not records:
                raise ValueError(f"{path}: JSON Lines file contains no records.")
        except ValueError as jsonl_error:
            if array_error is None:
                raise
            raise ValueError(
                f"Could not parse eval data {path} as a JSON array "
                f"({array_error.msg} at line {array_error.lineno}, "
                f"column {array_error.colno}) or JSON Lines ({jsonl_error})."
            ) from jsonl_error
        return records

    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as array_error:
        return read_json_lines(array_error)

    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"{path}: JSON array contains no records.")
        return [
            require_object(record, f"{path}: JSON array record {index}")
            for index, record in enumerate(payload)
        ]
    return read_json_lines()


def _resolve_eval_dataset_path(data_config, agentgym_config):
    """Resolve an explicit eval file, retaining the legacy filename fallback."""

    data_root = data_config.get("path")
    explicit_file = data_config.get("file")
    if explicit_file:
        dataset_path = Path(str(explicit_file)).expanduser()
        if not dataset_path.is_absolute():
            if not data_root:
                raise ValueError(
                    "data.file is relative but data.path is not configured."
                )
            dataset_path = Path(str(data_root)).expanduser() / dataset_path
    else:
        if not data_root:
            raise ValueError(
                "Eval data requires data.file or the legacy data.path directory."
            )
        task_name = str(agentgym_config.get("task_name", "")).strip()
        if not task_name:
            raise ValueError(
                "agentgym.task_name is required when data.file is omitted."
            )
        dataset_path = Path(str(data_root)).expanduser() / f"{task_name}_test.json"
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Eval dataset file does not exist: {dataset_path}")
    return dataset_path


def _load_category_map(data_root, dataset_path, task_name):
    """Load optional category labels from ordinary JSON/JSONL records only.

    Category labels are reporting metadata; malformed or non-record assets must
    not prevent a valid explicit eval file from running.  FAISS indexes,
    directories, and unrelated file formats are intentionally ignored.
    """

    if not data_root:
        return [], {}
    root = Path(str(data_root)).expanduser().resolve()
    if not root.is_dir():
        return [], {}
    dataset_path = Path(dataset_path).resolve()
    legacy_prefix = f"{str(task_name).strip()}_test" if task_name else None
    category_files = []
    category_map = {}
    for candidate in sorted(root.iterdir(), key=lambda item: item.name):
        if not candidate.is_file() or candidate.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if candidate.resolve() == dataset_path:
            continue
        if legacy_prefix and candidate.name.startswith(legacy_prefix):
            continue
        try:
            if candidate.stat().st_size > _MAX_OPTIONAL_CATEGORY_FILE_BYTES:
                # Category reporting is optional; never parse multi-gigabyte
                # environment assets (for example WebShop product corpora) as
                # speculative category maps.
                continue
        except OSError:
            continue
        try:
            records = _read_json_records(candidate)
        except (OSError, ValueError):
            # Category reporting is optional.  A non-record JSON asset should
            # not make an explicitly selected eval dataset unusable.
            continue
        item_records = [record for record in records if "item_id" in record]
        if not item_records:
            continue
        category_files.append(candidate.name)
        category = candidate.stem
        for record in item_records:
            category_map[str(record["item_id"])] = category
    return category_files, category_map


def _resolve_max_policy_turns(agentgym_config):
    """Read the canonical policy-turn ceiling with legacy max_rounds fallback."""

    value = agentgym_config.get("max_policy_turns")
    source = "max_policy_turns"
    if value is None:
        value = agentgym_config.get("max_rounds")
        source = "max_rounds"
    if value is None or isinstance(value, bool):
        raise ValueError(
            "agentgym.max_policy_turns (or legacy agentgym.max_rounds) must be set."
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"agentgym.{source} must be a positive integer, got {value!r}."
        ) from exc
    non_integral = isinstance(value, float) and not value.is_integer()
    if isinstance(value, str) and str(parsed) != value.strip():
        non_integral = True
    if parsed <= 0 or non_integral:
        raise ValueError(
            f"agentgym.{source} must be a positive integer, got {value!r}."
        )
    return parsed


def _resolve_eval_prompt_key(data_config, dataset):
    """Resolve the reporting id column across legacy and MemoryArena files."""

    configured = str(data_config.get("prompt_key", "")).strip()
    if configured and configured in dataset.columns:
        return configured
    # MemoryArena source files use ``id`` while older AgentGym eval files use
    # ``item_id``.  Prefer an explicit configured key, then deterministic
    # compatibility aliases; never silently choose an arbitrary column.
    for candidate in ("item_id", "id", "task_id", "source_id"):
        if candidate in dataset.columns:
            return candidate
    raise ValueError(
        f"Eval dataset has no configured prompt key {configured!r} and no "
        "supported id alias (item_id/id/task_id/source_id)."
    )


def _resolve_eval_data_indices(dataset, prompt_key):
    """Resolve reset positions, allowing only an explicit strict-int override."""

    del prompt_key  # Reporting identifiers never select environment rows.
    row_count = len(dataset)
    if "data_idx" not in dataset.columns:
        return list(range(row_count))

    values = []
    for row_index, candidate in enumerate(dataset["data_idx"].tolist()):
        if isinstance(candidate, bool) or not isinstance(candidate, Integral):
            raise ValueError(
                "Eval data_idx must be an integer environment position at "
                f"row {row_index}, got {candidate!r}."
            )
        data_idx = int(candidate)
        if data_idx < 0 or data_idx >= row_count:
            raise ValueError(
                "Eval data_idx is out of range at "
                f"row {row_index}: {data_idx}; expected 0 <= data_idx < {row_count}."
            )
        values.append(data_idx)
    return values


def _to_float_list(tensor_or_array):
    """Return one scalar score per generated row/action."""
    if hasattr(tensor_or_array, 'detach'):
        tensor_or_array = tensor_or_array.detach().cpu()
    if hasattr(tensor_or_array, 'tolist'):
        values = tensor_or_array.tolist()
    else:
        values = list(tensor_or_array)
    return [float(v) for v in values]


def _formal_step_records(output: DataProto, row_count: int):
    """Parse row-aligned formal environment records, when present."""

    if output.non_tensor_batch is None:
        return None
    raw_records = output.non_tensor_batch.get(AGENTMEMORY_STEP_RECORD_JSON)
    if raw_records is None:
        return None
    if len(raw_records) != row_count:
        raise ValueError(
            "AgentMemory eval formal evidence is misaligned: "
            f"scores={row_count} step_records={len(raw_records)}"
        )

    records = []
    for row_index, raw_record in enumerate(raw_records):
        try:
            record = json.loads(raw_record) if isinstance(raw_record, str) else raw_record
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"AgentMemory eval formal step record {row_index} is not valid JSON."
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"AgentMemory eval formal step record {row_index} is not an object."
            )
        records.append(record)
    return records


def _authoritative_episode_success_flags(output: DataProto, row_count: int):
    """Read per-action episode success from formal environment evidence."""

    records = _formal_step_records(output, row_count)
    if records is None:
        return None
    flags = []
    for row_index, record in enumerate(records):
        if "episode_success" in record:
            value = record["episode_success"]
        else:
            env_info_after = record.get("env_info_after")
            if not isinstance(env_info_after, dict) or "episode_success" not in env_info_after:
                raise ValueError(
                    "AgentMemory eval formal step record is missing authoritative "
                    f"episode_success at row {row_index}."
                )
            value = env_info_after["episode_success"]
        if type(value) is not bool:
            raise ValueError(
                "AgentMemory eval authoritative episode_success must be boolean at "
                f"row {row_index}."
            )
        flags.append(value)
    return flags


def _require_formal_boolean_flags(values, label):
    for row_index, value in enumerate(values):
        if type(value) is not bool:
            raise ValueError(
                f"AgentMemory eval {label} must be boolean at row {row_index}."
            )


def _formal_phase_progress_distribution(output: DataProto, real_batch_size: int):
    """Count formal trajectories by authoritative final/max phase reached.

    A missing phase index/count is reported as ``unknown``.  This helper never
    infers a denominator from reward magnitude, domain name, or task family.
    """

    action_scores = _to_float_list(output.batch['task_scores'].sum(dim=-1))
    records = _formal_step_records(output, len(action_scores))
    if records is None:
        return None
    non_tensor_batch = output.non_tensor_batch or {}
    parent_indices = non_tensor_batch.get('rollout_parent_indices')
    if parent_indices is None:
        if len(records) != real_batch_size:
            raise ValueError(
                "AgentMemory eval formal episode rows must align with the "
                f"real batch: rows={len(records)} real_batch={real_batch_size}"
            )
        parent_indices = list(range(real_batch_size))
    elif len(parent_indices) != len(records):
        raise ValueError(
            "AgentMemory eval phase evidence is misaligned: "
            f"records={len(records)} parents={len(parent_indices)}"
        )

    progress_by_parent = [None] * real_batch_size
    for row_index, (parent, record) in enumerate(zip(parent_indices, records)):
        try:
            parent_index = int(parent)
        except (TypeError, ValueError, OverflowError):
            continue
        if parent_index < 0 or parent_index >= real_batch_size:
            continue
        env_info_after = record.get("env_info_after")
        if not isinstance(env_info_after, dict):
            env_info_after = {}
        phase_index = record.get("phase_index_after")
        if phase_index is None:
            phase_index = record.get(
                "subtask_index_after",
                env_info_after.get(
                    "phase_index", env_info_after.get("current_subtask_index")
                ),
            )
        phase_count = record.get("phase_count")
        if phase_count is None:
            phase_count = env_info_after.get(
                "phase_count", env_info_after.get("subtask_count")
            )
        try:
            if isinstance(phase_index, bool) or isinstance(phase_count, bool):
                raise ValueError
            phase_index = int(phase_index)
            phase_count = int(phase_count)
        except (TypeError, ValueError, OverflowError):
            continue
        if phase_index < 0 or phase_count <= 0 or phase_index > phase_count:
            continue
        previous = progress_by_parent[parent_index]
        if previous is not None and previous[1] != phase_count:
            raise ValueError(
                "AgentMemory eval phase_count changed within one trajectory: "
                f"parent={parent_index} row={row_index} "
                f"before={previous[1]} after={phase_count}"
            )
        if previous is None or phase_index > previous[0]:
            progress_by_parent[parent_index] = (phase_index, phase_count)

    known_phase_counts = {
        progress[1] for progress in progress_by_parent if progress is not None
    }
    distribution = defaultdict(int)
    if len(known_phase_counts) == 1:
        phase_count = next(iter(known_phase_counts))
        for phase_index in range(phase_count + 1):
            distribution[f"{phase_index}/{phase_count}"] = 0
    for progress in progress_by_parent:
        key = "unknown" if progress is None else f"{progress[0]}/{progress[1]}"
        distribution[key] += 1
    return dict(distribution)


def _record_phase_progress_flag(record: dict):
    """Return whether one formal row advanced a phase, or ``None`` if unknown."""

    env_info_before = record.get("env_info_before")
    env_info_after = record.get("env_info_after")
    if not isinstance(env_info_before, dict):
        env_info_before = {}
    if not isinstance(env_info_after, dict):
        env_info_after = {}
    before = record.get("phase_index_before")
    if before is None:
        before = record.get(
            "subtask_index_before",
            env_info_before.get(
                "phase_index", env_info_before.get("current_subtask_index")
            ),
        )
    after = record.get("phase_index_after")
    if after is None:
        after = record.get(
            "subtask_index_after",
            env_info_after.get(
                "phase_index", env_info_after.get("current_subtask_index")
            ),
        )
    try:
        if isinstance(before, bool) or isinstance(after, bool):
            raise ValueError
        if before is None and after is None:
            return None
        if after is None:
            return None
        after = int(after)
        if before is None:
            return after > 0
        return after > int(before)
    except (TypeError, ValueError, OverflowError):
        return None


def _aggregate_episode_scores(output: DataProto, real_batch_size: int):
    """Aggregate rollout rows into episode-level eval metrics.

    Normal AgentGym generation returns one row per episode.  AgentMemoryGym's
    latest-observation rollout returns one row per *action*, plus
    ``rollout_parent_indices`` and ``rollout_done_flags``.  Counting positive
    action-level shaping rewards as Pass@k is wrong; Pass must be terminal
    episode success.  We keep Avg as episode return (sum of action rewards),
    and expose Progress separately.  Formal rows derive it from phase
    transitions; legacy rows retain the historical shaping fallback.
    """
    action_scores = _to_float_list(output.batch['task_scores'].sum(dim=-1))
    non_tensor_batch = output.non_tensor_batch or {}
    parent_indices = non_tensor_batch.get('rollout_parent_indices')
    has_formal_records = AGENTMEMORY_STEP_RECORD_JSON in non_tensor_batch
    # Formal rows carry the authoritative terminal outcome in the environment
    # evidence.  Even when an older/alternate worker omits parent indices, do
    # not silently turn a positive shaping return into a successful episode.
    if parent_indices is None:
        if has_formal_records:
            episode_success_flags = _authoritative_episode_success_flags(
                output, len(action_scores)
            )
            formal_records = _formal_step_records(output, len(action_scores))
            done_flags = non_tensor_batch.get('rollout_done_flags')
            if done_flags is None:
                raise ValueError(
                    "AgentMemory eval formal rows are missing rollout_done_flags."
                )
            if len(done_flags) != len(action_scores):
                raise ValueError(
                    "AgentMemory eval formal rows are misaligned: "
                    f"scores={len(action_scores)} done_flags={len(done_flags)}"
                )
            _require_formal_boolean_flags(done_flags, "rollout_done_flags")
            if len(action_scores) != real_batch_size:
                raise ValueError(
                    "AgentMemory eval formal episode rows must align with the "
                    f"real batch: rows={len(action_scores)} real_batch={real_batch_size}"
                )
            episode_scores = action_scores
            episode_pass = [
                done and success
                for done, success in zip(done_flags, episode_success_flags)
            ]
            progress_flags = [
                _record_phase_progress_flag(record) for record in formal_records
            ]
            # Keep the legacy boolean return shape, but explicitly mark rows
            # without phase evidence as unknown instead of using reward as a
            # proxy for progress.
            episode_progress = [flag is True for flag in progress_flags]
            return episode_scores, episode_pass, episode_progress, {
                'mode': 'formal_episode_rows',
                'rows': len(action_scores),
                'parents': real_batch_size,
                'pass_source': 'formal_episode_success',
                'progress_source': (
                    'formal_phase_progress'
                    if all(flag is not None for flag in progress_flags)
                    else 'unknown'
                ),
                'progress_unknown_count': sum(
                    flag is None for flag in progress_flags
                ),
            }
        episode_scores = action_scores[:real_batch_size]
        episode_pass = [score > 0 for score in episode_scores]
        episode_progress = [score >= 1.0 for score in episode_scores]
        return episode_scores, episode_pass, episode_progress, {
            'mode': 'episode_rows',
            'rows': len(action_scores),
            'parents': real_batch_size,
            'pass_source': 'legacy_score_positive',
        }

    done_flags = output.non_tensor_batch.get('rollout_done_flags')
    if done_flags is None:
        if has_formal_records:
            raise ValueError(
                "AgentMemory eval formal action rows are missing "
                "rollout_done_flags."
            )
        done_flags = [False] * len(action_scores)
    episode_success_flags = _authoritative_episode_success_flags(
        output, len(action_scores)
    )
    if has_formal_records and episode_success_flags is None:
        raise ValueError(
            "AgentMemory eval formal evidence is missing authoritative "
            "episode_success; refusing score-based Pass aggregation."
        )
    if len(parent_indices) != len(action_scores) or len(done_flags) != len(action_scores):
        raise ValueError(
            "AgentMemory eval action rows are misaligned: "
            f"scores={len(action_scores)} parents={len(parent_indices)} "
            f"done_flags={len(done_flags)}"
        )
    if has_formal_records:
        _require_formal_boolean_flags(done_flags, "rollout_done_flags")

    episode_scores = [0.0] * real_batch_size
    episode_pass = [False] * real_batch_size
    episode_progress = [False] * real_batch_size
    action_counts = [0] * real_batch_size
    ignored_rows = 0
    if episode_success_flags is None:
        success_flags = [None] * len(action_scores)
        pass_source = 'legacy_done_positive'
    else:
        success_flags = episode_success_flags
        pass_source = 'formal_episode_success'
    formal_records = _formal_step_records(output, len(action_scores))
    progress_flags_by_row = (
        [_record_phase_progress_flag(record) for record in formal_records]
        if formal_records is not None
        else None
    )

    for row_index, (parent, done, score, episode_success) in enumerate(
        zip(parent_indices, done_flags, action_scores, success_flags)
    ):
        try:
            parent_idx = int(parent)
        except Exception:
            ignored_rows += 1
            continue
        if parent_idx < 0 or parent_idx >= real_batch_size:
            ignored_rows += 1
            continue
        score = float(score)
        episode_scores[parent_idx] += score
        terminal_success = (
            done and episode_success
            if episode_success is not None
            else bool(done) and score > 0.0
        )
        episode_pass[parent_idx] = bool(
            episode_pass[parent_idx] or terminal_success
        )
        if progress_flags_by_row is None:
            row_progress = score >= 1.0
        else:
            row_progress = progress_flags_by_row[row_index]
        episode_progress[parent_idx] = bool(
            episode_progress[parent_idx] or row_progress is True
        )
        action_counts[parent_idx] += 1

    return episode_scores, episode_pass, episode_progress, {
        'mode': 'agentmemory_action_rows',
        'rows': len(action_scores),
        'parents': real_batch_size,
        'ignored_rows': ignored_rows,
        'pass_source': pass_source,
        'progress_source': (
            'legacy_score_positive'
            if progress_flags_by_row is None
            else 'formal_phase_progress'
            if all(flag is not None for flag in progress_flags_by_row)
            else 'unknown'
        ),
        'progress_unknown_count': (
            0
            if progress_flags_by_row is None
            else sum(flag is None for flag in progress_flags_by_row)
        ),
        'min_actions_per_parent': min(action_counts) if action_counts else 0,
        'max_actions_per_parent': max(action_counts) if action_counts else 0,
    }


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    removed_train_flags = _scrub_training_rollout_flags()
    if removed_train_flags:
        print(
            "Formal eval removed train-only rollout flags: "
            + ", ".join(removed_train_flags),
            flush=True,
        )
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    max_policy_turns = _resolve_max_policy_turns(config.agentgym)
    print(f"AgentMemory eval policy-turn ceiling: {max_policy_turns}", flush=True)

    # The selected file may be JSON or JSON Lines.  Keep the historical
    # ``<task_name>_test.json`` convention when data.file is absent.
    dataset_path = _resolve_eval_dataset_path(config.data, config.agentgym)
    dataset = pd.DataFrame.from_records(_read_json_records(dataset_path))
    prompt_key = _resolve_eval_prompt_key(config.data, dataset)
    item_ids = dataset[prompt_key].tolist()
    data_indices = _resolve_eval_data_indices(dataset, prompt_key)
    # Category files are optional reporting metadata.  Do not parse FAISS
    # indexes, directories, or unrelated assets as JSON.
    category_files, category_map = _load_category_map(
        config.data.get("path"),
        dataset_path,
        config.agentgym.get("task_name", ""),
    )

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    num_batch = (total_samples + config_batch_size - 1) // config_batch_size
    output_lst = [[] for _ in range(config.data.n_samples)]
    pass_lst = [[] for _ in range(config.data.n_samples)]
    progress_lst = [[] for _ in range(config.data.n_samples)]
    used_action_row_aggregation = False
    phase_progress_distribution = defaultdict(int)
    env_client = init_env_client(config.agentgym)

    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        start_idx = batch_idx * config_batch_size
        end_idx = min(total_samples, start_idx + config_batch_size)
        batch_item_ids = item_ids[start_idx: end_idx]
        prompt_with_chat_template = ["<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + env_client.conversation_start[0]["value"] + "<|im_end|>\n<|im_start|>assistant\n" + env_client.conversation_start[1]["value"] + "<|im_end|>" for _ in range(len(batch_item_ids))]
        messages = [[{"role": "user", "content": env_client.conversation_start[0]["value"]},
                     {"role": "assistant", "content": env_client.conversation_start[1]["value"]}] for _ in range(len(batch_item_ids))]

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=tokenizer,
                                                                         max_length=config.data.max_prompt_length,
                                                                         pad_token_id=tokenizer.pad_token_id,
                                                                         left_pad=True)
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        data.meta_info['global_steps'] = 'test_batch_' + str(batch_idx)
        # Propagate both keys so older rollout workers remain compatible.
        data.meta_info['max_policy_turns'] = max_policy_turns
        data.meta_info['max_rounds'] = max_policy_turns
        data.non_tensor_batch["item_id"] = np.array(
            [str(item_id) for item_id in batch_item_ids], dtype=object
        )
        real_batch_size = data.batch['input_ids'].shape[0]
        # Compute the padding count before adding selector metadata.  The
        # selector array is part of the input to the rollout worker and must
        # have one entry for every row, including DP padding rows.
        dummy_data_size = (-real_batch_size) % dp_size
        data.non_tensor_batch["rollout_data_indices"] = np.array(
            data_indices[start_idx:end_idx],
            dtype=object,
        )
        data.non_tensor_batch["raw_prompt"] = np.array(messages, dtype=object)
        data, dummy_data_size = _pad_dataproto_for_dp(data, dp_size)
        # Padding repeats real metadata; overwrite selectors so dummy rows are
        # never interpreted as a real environment reset (``-1`` is ignored by
        # the eval parent-index aggregation).
        data.non_tensor_batch["rollout_data_indices"] = np.array(
            data_indices[start_idx:end_idx] + [-1] * dummy_data_size,
            dtype=object,
        )
        if dummy_data_size:
            print(
                f'dp_size {dp_size} is not divisible by real_batch_size '
                f'{real_batch_size}, add {dummy_data_size} dummy data'
            )
        # AgentMemory latest-observation rollout returns one row per action and
        # runs on sharded worker-local batches, so worker-local row ids are not
        # enough to aggregate back to this eval batch.  Carry explicit parent ids
        # and mark padded dummy episodes as -1 so their action rows are ignored.
        data.non_tensor_batch["rollout_eval_parent_indices"] = np.array(
            list(range(real_batch_size)) + [-1] * dummy_data_size,
            dtype=object,
        )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')

        for i in range(config.data.n_samples):
            # Keep eval rollout logs traceable across n_samples. AgentMemory rollout
            # writes to rollout_log_dir/step{global_steps}; reusing test_batch_N
            # silently overwrites earlier samples and makes behavior audits non-reproducible.
            data.meta_info['global_steps'] = f'test_batch_{batch_idx}_sample_{i}'
            output = wg.generate_sequences(data)
            episode_scores, episode_pass, episode_progress, agg_info = _aggregate_episode_scores(output, real_batch_size)
            batch_phase_distribution = _formal_phase_progress_distribution(
                output, real_batch_size
            )
            if batch_phase_distribution is not None:
                for key, count in batch_phase_distribution.items():
                    phase_progress_distribution[key] += int(count)
            used_action_row_aggregation = used_action_row_aggregation or agg_info['mode'] == 'agentmemory_action_rows'
            print(f"[{batch_idx+1}/{num_batch}] sample {i+1}/{config.data.n_samples} aggregation: {agg_info}")

            output_lst[i].extend(episode_scores)
            pass_lst[i].extend(episode_pass)
            progress_lst[i].extend(episode_progress)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_np = np.array(output_lst, dtype=float)
    output_np = np.transpose(output_np, axes=(1, 0))
    pass_np = np.array(pass_lst, dtype=bool)
    pass_np = np.transpose(pass_np, axes=(1, 0))
    progress_np = np.array(progress_lst, dtype=bool)
    progress_np = np.transpose(progress_np, axes=(1, 0))
    output_lst = output_np.tolist()
    pass_lst_t = pass_np.tolist()
    progress_lst_t = progress_np.tolist()

    print("============Total Task Evaluation============")
    if used_action_row_aggregation:
        print(
            "AgentMemoryGym strict eval: Avg is summed episode return; "
            "formal Pass uses authoritative episode_success, not action-level shaping."
        )
    print(f"Avg@{config.data.n_samples}: {np.mean(output_np)}")
    print(f"Pass@{config.data.n_samples}: {np.mean(np.max(pass_np, axis=-1))}")
    if used_action_row_aggregation:
        print(f"Progress@{config.data.n_samples}: {np.mean(np.max(progress_np, axis=-1))}")
    if phase_progress_distribution:
        print(
            "FinalPhaseProgressDistribution: "
            + json.dumps(
                dict(phase_progress_distribution),
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    print("============Sub Task Evaluation============")
    
    category_success_bucket = defaultdict(list)
    category_pass_bucket = defaultdict(list)
    category_progress_bucket = defaultdict(list)
    uncategorized_seen = False
    for item_id, score, pass_flags, progress_flags in zip(item_ids, output_lst, pass_lst_t, progress_lst_t):
        category = category_map.get(str(item_id))
        if category is None:
            category = "uncategorized"
            uncategorized_seen = True
        category_success_bucket[category].append(score)
        category_pass_bucket[category].append(pass_flags)
        category_progress_bucket[category].append(progress_flags)
    if uncategorized_seen:
        category_files.append("uncategorized.json")
    for category_file in category_files:
        category = category_file.split(".")[0]
        print(f"Category: {category}")
        if not category_success_bucket[category]:
            print("No samples for this category; skip empty category bucket.")
            continue
        print(f"Avg@{config.data.n_samples}: {np.mean(np.array(category_success_bucket[category], dtype=float))}")
        print(f"Pass@{config.data.n_samples}: {np.mean(np.max(np.array(category_pass_bucket[category], dtype=bool), axis=-1))}")
        if used_action_row_aggregation:
            print(f"Progress@{config.data.n_samples}: {np.mean(np.max(np.array(category_progress_bucket[category], dtype=bool), axis=-1))}")



if __name__ == '__main__':
    main()
