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
import ray
import numpy as np
import hydra
import verl.utils.torch_functional as verl_F
import os

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


_TRAIN_ONLY_ROLLOUT_FLAGS = (
    "AGENTMEMORY_ACTION_ENUMERATION_ROLLOUT",
    "AGENTMEMORY_ACTION_SEQUENCE_ENUMERATION_ROLLOUT",
    "AGENTMEMORY_FORCE_BALANCED_CHOOSE_ROLLOUT",
    "AGENTMEMORY_FINAL_BUY_ENUMERATION_ROLLOUT",
    "AGENTMEMORY_FINAL_BUY_PAIRWISE_ROLLOUT",
    "AGENTMEMORY_LATEST_OBS_SUFFIX_CREDIT",
)


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


def _to_float_list(tensor_or_array):
    """Return one scalar score per generated row/action."""
    if hasattr(tensor_or_array, 'detach'):
        tensor_or_array = tensor_or_array.detach().cpu()
    if hasattr(tensor_or_array, 'tolist'):
        values = tensor_or_array.tolist()
    else:
        values = list(tensor_or_array)
    return [float(v) for v in values]


def _aggregate_episode_scores(output: DataProto, real_batch_size: int):
    """Aggregate rollout rows into episode-level eval metrics.

    Normal AgentGym generation returns one row per episode.  AgentMemoryGym's
    latest-observation rollout returns one row per *action*, plus
    ``rollout_parent_indices`` and ``rollout_done_flags``.  Counting positive
    action-level shaping rewards as Pass@k is wrong; Pass must be terminal
    episode success.  We keep Avg as episode return (sum of action rewards),
    and expose Progress separately for positive non-terminal BUY-like signals.
    """
    action_scores = _to_float_list(output.batch['task_scores'].sum(dim=-1))
    parent_indices = output.non_tensor_batch.get('rollout_parent_indices') if output.non_tensor_batch is not None else None
    if parent_indices is None:
        episode_scores = action_scores[:real_batch_size]
        episode_pass = [score > 0 for score in episode_scores]
        episode_progress = [score >= 1.0 for score in episode_scores]
        return episode_scores, episode_pass, episode_progress, {
            'mode': 'episode_rows',
            'rows': len(action_scores),
            'parents': real_batch_size,
        }

    done_flags = output.non_tensor_batch.get('rollout_done_flags')
    if done_flags is None:
        done_flags = [False] * len(action_scores)
    if len(parent_indices) != len(action_scores) or len(done_flags) != len(action_scores):
        raise ValueError(
            "AgentMemory eval action rows are misaligned: "
            f"scores={len(action_scores)} parents={len(parent_indices)} "
            f"done_flags={len(done_flags)}"
        )

    episode_scores = [0.0] * real_batch_size
    episode_pass = [False] * real_batch_size
    episode_progress = [False] * real_batch_size
    action_counts = [0] * real_batch_size
    ignored_rows = 0
    for parent, done, score in zip(parent_indices, done_flags, action_scores):
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
        episode_pass[parent_idx] = bool(episode_pass[parent_idx] or (bool(done) and score > 0.0))
        episode_progress[parent_idx] = bool(episode_progress[parent_idx] or score >= 1.0)
        action_counts[parent_idx] += 1

    return episode_scores, episode_pass, episode_progress, {
        'mode': 'agentmemory_action_rows',
        'rows': len(action_scores),
        'parents': real_batch_size,
        'ignored_rows': ignored_rows,
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

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset_path = os.path.join(config.data.path, f"{config.agentgym.task_name}_test.json")
    dataset = pd.DataFrame.from_records(_read_json_records(dataset_path))
    if config.data.prompt_key not in dataset.columns:
        raise ValueError(
            f"Eval data {dataset_path} is missing prompt key {config.data.prompt_key!r}."
        )
    item_ids = dataset[config.data.prompt_key].tolist()
    # load sub category test file
    category_files = os.listdir(config.data.path)
    category_files = [f for f in category_files if not f.startswith(f"{config.agentgym.task_name}_test")]
    category_map = {}
    for category_file in category_files:
        path = os.path.join(config.data.path, category_file)
        with open(path, "r") as f:
            datas = json.load(f)
            for data in datas:
                category_map[data["item_id"]] = category_file.split(".")[0]

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
        data.meta_info['max_rounds'] = config.agentgym.max_rounds
        data.non_tensor_batch["item_id"] = np.array(batch_item_ids, dtype=object)
        data.non_tensor_batch["raw_prompt"] = np.array(messages, dtype=object)
        real_batch_size = data.batch['input_ids'].shape[0]
        data, dummy_data_size = _pad_dataproto_for_dp(data, dp_size)
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
        print("AgentMemoryGym strict eval: Avg is summed episode return; Pass is terminal done&&positive, not action-level shaping.")
    print(f"Avg@{config.data.n_samples}: {np.mean(output_np)}")
    print(f"Pass@{config.data.n_samples}: {np.mean(np.max(pass_np, axis=-1))}")
    if used_action_row_aggregation:
        print(f"Progress@{config.data.n_samples}: {np.mean(np.max(progress_np, axis=-1))}")
    print("============Sub Task Evaluation============")
    
    category_success_bucket = defaultdict(list)
    category_pass_bucket = defaultdict(list)
    category_progress_bucket = defaultdict(list)
    uncategorized_seen = False
    for item_id, score, pass_flags, progress_flags in zip(item_ids, output_lst, pass_lst_t, progress_lst_t):
        category = category_map.get(item_id)
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
