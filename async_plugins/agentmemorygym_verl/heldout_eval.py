"""Task-neutral GPU runner for the frozen CAMG native held-out panel.

Heavy runtime dependencies are imported only inside :func:`run_evaluation` so
the launch/configuration contract can be audited with the system Python on a
CPU-only host.  Environment lifecycle remains owned by the outer run-scoped
orchestrator; this module owns only model serving, shared AgentLoop sampling,
and immutable result publication.
"""

from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .heldout_eval_contract import (
    AGENT_NAME,
    CANONICAL_ROUTES,
    RUN_SCHEMA,
    SCHEDULE_SCHEMA,
    atomic_write_json,
    commit_batch,
    finalize_run_metrics,
    initialize_run_contract,
    inspect_heldout_schedule,
    inspect_resume_state,
    materialize_generated_batch,
    pad_batch_rows,
    read_json,
    read_jsonl,
    require_positive_int,
    require_regular_file,
    require_sha256,
    sha256_file,
    verify_complete_split_contract,
    verify_swesmith_formal_eval_authority,
)
from .routes import load_route_registry
from .heldout_method_evidence import SUPPORTED_METHOD_IDS

MODEL_MANIFEST_SCHEMA = "camg_merged_hf_checkpoint_manifest_v1"
FROZEN_MODEL_MANIFEST_SCHEMA = "camg_frozen_hf_model_manifest_v1"
SUPPORTED_MODEL_KINDS = ("merged_checkpoint", "frozen_hf")
SCHEDULE_MANIFEST_SCHEMA = SCHEDULE_SCHEMA
EXPECTED_CHECKPOINT_STEP = 200
EXPECTED_BATCH_SIZE = 64
EXPECTED_NUM_GPUS = 8
PADDING_INDEX_BASE = 1_000_000_000
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
_DATASET_RUNTIME_FIELDS = frozenset(
    {
        "raw_prompt",
        "dummy_tensor",
        "tools_kwargs",
        "interaction_kwargs",
    }
)


@dataclass(frozen=True)
class VerifiedModel:
    path: Path
    manifest_path: Path
    manifest_sha256: str
    checkpoint_step: int
    training_run_id: str
    source_commits: dict[str, str]
    file_count: int
    model_kind: str = "merged_checkpoint"
    model_id: str = ""
    source_revision: str = ""


@dataclass(frozen=True)
class HeldoutEvalPlan:
    run_id: str
    run_dir: Path
    resolved_config_path: Path
    resolved_config_sha256: str
    schedule_path: Path
    schedule_sha256: str
    schedule_manifest_path: Path
    schedule_manifest_sha256: str
    route_registry_path: Path
    route_registry_sha256: str
    route_counts: dict[str, int]
    episode_count: int
    complete_split_authority: dict[str, Any]
    swesmith_formal_eval_authority: dict[str, Any]
    route_max_rounds: dict[str, int]
    route_attestations: dict[str, str]
    agent_loop_config_path: Path
    agent_loop_config_sha256: str
    model: VerifiedModel
    evaluator_outer_commit: str
    evaluator_inner_commit: str
    evaluator_verl_commit: str
    method_id: str = "agemem"
    model_kind: str = "merged_checkpoint"
    batch_size: int = EXPECTED_BATCH_SIZE
    num_gpus: int = EXPECTED_NUM_GPUS
    gpu_memory_utilization: float = 0.8


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _sequence(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return value


def _commit(value: Any, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if not _COMMIT.fullmatch(text):
        raise ValueError(f"{field} must be a full lowercase git commit")
    return text


def _absolute_regular(path: str | os.PathLike[str], *, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return require_regular_file(candidate, field=field).resolve()


def _absolute_directory(path: str | os.PathLike[str], *, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"{field} must be an absolute non-symlink directory")
    return candidate.resolve()


def _relative_payload_path(value: Any, *, field: str) -> Path:
    text = str(value or "")
    candidate = Path(text)
    if (
        not text
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate == Path(".")
    ):
        raise ValueError(f"{field} must be a safe relative payload path")
    return candidate


def _schedule_counts_and_coding_authority(
    schedule_metadata: dict[str, Any],
) -> tuple[dict[str, int], int, dict[str, Any], dict[str, Any]]:
    if tuple(schedule_metadata.get("route_order", ())) != CANONICAL_ROUTES:
        raise ValueError("held-out schedule manifest route order drift")
    raw_counts = _mapping(
        schedule_metadata.get("per_route_rows"),
        field="held-out schedule per_route_rows",
    )
    if set(raw_counts) != set(CANONICAL_ROUTES):
        raise ValueError("held-out schedule manifest route set drift")
    route_counts = {
        route_id: require_positive_int(
            raw_counts.get(route_id),
            field=f"held-out route count {route_id}",
        )
        for route_id in CANONICAL_ROUTES
    }
    split_authority = _mapping(
        schedule_metadata.get("complete_split_authority"),
        field="complete held-out split authority",
    )
    verified_split = verify_complete_split_contract(
        split_authority.get("path"),
        expected_sha256=split_authority.get("sha256"),
        expected_route_counts=route_counts,
    )
    if split_authority != verified_split:
        raise ValueError("complete held-out split authority summary drift")
    episode_count = sum(route_counts.values())
    if schedule_metadata.get("row_count") != episode_count:
        raise ValueError("held-out schedule manifest row count drift")
    sources = _mapping(
        schedule_metadata.get("sources"), field="held-out schedule sources"
    )
    coding_source = _mapping(
        sources.get("swesmith"), field="held-out SWE-smith schedule source"
    )
    authority = _mapping(
        coding_source.get("selection_authority"),
        field="SWE-smith formal Eval authority",
    )
    coding_schedule = _absolute_regular(
        coding_source.get("path"), field="held-out SWE-smith source schedule"
    )
    coding_schedule_hash = require_sha256(
        coding_source.get("schedule_sha256"),
        field="held-out SWE-smith source schedule expected sha256",
    )
    if sha256_file(coding_schedule) != coding_schedule_hash:
        raise ValueError("held-out SWE-smith source schedule sha256 mismatch")
    selected_coding_rows = tuple(read_jsonl(coding_schedule))
    if len(selected_coding_rows) != route_counts["swesmith"]:
        raise ValueError("held-out SWE-smith source schedule row count drift")
    verified = verify_swesmith_formal_eval_authority(
        authority.get("path"),
        expected_sha256=authority.get("sha256"),
        expected_routing_sha256=coding_schedule_hash,
        expected_admitted_task_count=route_counts["swesmith"],
        selected_rows=selected_coding_rows,
    )
    if authority != verified:
        raise ValueError("SWE-smith formal Eval authority summary drift")
    return route_counts, episode_count, verified_split, verified


def verify_runtime_dataset_row(
    expected: dict[str, Any],
    processed: dict[str, Any],
    *,
    expected_route_attestation_sha256: str,
    schedule_position: int,
) -> None:
    """Prove that dataset preprocessing did not change task identity.

    ``AMGTrajectoryDataset`` may attach only its four standard runtime fields
    and the route attestation pinned by the immutable route registry.  Every
    schedule-owned value, including the complete nested ``extra_info`` payload,
    remains byte-semantically equal after preprocessing.
    """

    if not isinstance(expected, dict) or not isinstance(processed, dict):
        raise RuntimeError(
            f"runtime dataset row {schedule_position} must remain an object"
        )
    unexpected = set(processed) - set(expected) - _DATASET_RUNTIME_FIELDS
    if unexpected:
        raise RuntimeError(
            f"runtime dataset row {schedule_position} has unexpected top-level "
            f"fields: {sorted(unexpected)!r}"
        )
    missing = set(expected) - set(processed)
    if missing:
        raise RuntimeError(
            f"runtime dataset row {schedule_position} lost schedule fields: "
            f"{sorted(missing)!r}"
        )

    expected_index = expected.get("index")
    if expected_index != schedule_position:
        raise RuntimeError(
            f"runtime dataset schedule/index drift at row {schedule_position}: "
            f"expected index {expected_index!r}"
        )
    for field in expected:
        if field == "extra_info":
            continue
        if processed.get(field) != expected[field]:
            raise RuntimeError(
                f"runtime dataset identity drift at schedule row "
                f"{schedule_position}: field {field!r}"
            )

    expected_extra_raw = expected.get("extra_info")
    processed_extra_raw = processed.get("extra_info")
    if not isinstance(expected_extra_raw, dict) or not isinstance(
        processed_extra_raw, dict
    ):
        raise RuntimeError(
            f"runtime dataset extra_info at schedule row {schedule_position} "
            "must remain an object"
        )
    attestation = require_sha256(
        expected_route_attestation_sha256,
        field=f"route attestation at schedule row {schedule_position}",
    )
    expected_extra = deepcopy(expected_extra_raw)
    existing_attestation = expected_extra.get("route_attestation_sha256")
    if existing_attestation is not None and existing_attestation != attestation:
        raise RuntimeError(
            f"runtime dataset extra_info route attestation drift at schedule row "
            f"{schedule_position}"
        )
    expected_extra["route_attestation_sha256"] = attestation
    if processed_extra_raw != expected_extra:
        raise RuntimeError(
            f"runtime dataset extra_info drift at schedule row {schedule_position}"
        )


def verify_model_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str,
    expected_checkpoint_step: int,
    expected_training_run_id: str,
    expected_source_commits: dict[str, str],
    model_kind: str = "merged_checkpoint",
) -> VerifiedModel:
    """Verify every byte in one immutable merged-HF checkpoint publication."""

    path = _absolute_regular(manifest_path, field="model manifest")
    expected_digest = require_sha256(
        expected_manifest_sha256, field="model manifest expected sha256"
    )
    observed_digest = sha256_file(path)
    if observed_digest != expected_digest:
        raise ValueError(
            f"model manifest sha256 mismatch: expected {expected_digest}, "
            f"got {observed_digest}"
        )
    payload = _mapping(read_json(path), field="model manifest")
    if model_kind not in SUPPORTED_MODEL_KINDS:
        raise ValueError(f"unsupported held-out model_kind {model_kind!r}")
    expected_schema = (
        MODEL_MANIFEST_SCHEMA
        if model_kind == "merged_checkpoint"
        else FROZEN_MODEL_MANIFEST_SCHEMA
    )
    if payload.get("schema") != expected_schema:
        raise ValueError(f"model manifest schema must be {expected_schema!r}")
    raw_step = payload.get("checkpoint_step")
    if isinstance(raw_step, bool) or not isinstance(raw_step, int) or raw_step < 0:
        raise ValueError("model checkpoint_step must be a non-negative integer")
    step = raw_step
    if step != expected_checkpoint_step:
        raise ValueError(
            f"model checkpoint step mismatch: expected {expected_checkpoint_step}, got {step}"
        )
    training_run_id = str(payload.get("training_run_id", ""))
    if training_run_id != expected_training_run_id:
        raise ValueError("model manifest training_run_id mismatch")
    normalized_source: dict[str, str] = {}
    model_id = ""
    source_revision = ""
    if model_kind == "merged_checkpoint":
        source = _mapping(payload.get("source_commits"), field="model source_commits")
        normalized_source = {
            key: _commit(source.get(key), field=f"model source commit {key}")
            for key in ("outer", "inner", "verl")
        }
        normalized_expected = {
            key: _commit(
                expected_source_commits.get(key),
                field=f"expected source commit {key}",
            )
            for key in ("outer", "inner", "verl")
        }
        if normalized_source != normalized_expected:
            raise ValueError("model manifest source commits differ from the eval contract")
    else:
        if step != 0:
            raise ValueError("frozen HF model manifest checkpoint_step must be zero")
        model_id = str(payload.get("model_id") or "").strip()
        source_revision = str(payload.get("source_revision") or "").strip().lower()
        if not model_id:
            raise ValueError("frozen HF model manifest requires model_id")
        if not _COMMIT.fullmatch(source_revision):
            raise ValueError("frozen HF model manifest requires a full source_revision")

    model_path = _absolute_directory(payload.get("model_path"), field="model path")
    if path == model_path or model_path in path.parents:
        raise ValueError("model manifest must live outside the model payload directory")
    raw_files = _sequence(payload.get("files"), field="model manifest files")
    if not raw_files:
        raise ValueError("model manifest files must not be empty")
    declared: dict[str, dict[str, Any]] = {}
    for position, raw_entry in enumerate(raw_files):
        entry = _mapping(raw_entry, field=f"model file entry {position}")
        relative = _relative_payload_path(
            entry.get("path"), field=f"model file entry {position} path"
        )
        key = relative.as_posix()
        if key in declared:
            raise ValueError(f"duplicate model manifest path: {key}")
        file_path = require_regular_file(
            model_path / relative, field=f"model payload {key}"
        )
        expected_bytes = require_positive_int(
            entry.get("bytes"), field=f"model payload {key} bytes"
        )
        if file_path.stat().st_size != expected_bytes:
            raise ValueError(f"model payload byte count mismatch: {key}")
        expected_file_hash = require_sha256(
            entry.get("sha256"), field=f"model payload {key} sha256"
        )
        if sha256_file(file_path) != expected_file_hash:
            raise ValueError(f"model payload sha256 mismatch: {key}")
        declared[key] = entry

    observed: set[str] = set()
    for candidate in model_path.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"model payload contains a symlink: {candidate}")
        if candidate.is_file():
            observed.add(candidate.relative_to(model_path).as_posix())
    if observed != set(declared):
        raise ValueError(
            "model manifest does not enumerate the exact payload file set: "
            f"missing={sorted(observed - set(declared))!r} "
            f"extra={sorted(set(declared) - observed)!r}"
        )
    if "config.json" not in observed:
        raise ValueError("merged model payload lacks config.json")
    if not any(name.endswith(".safetensors") for name in observed):
        raise ValueError("merged model payload lacks safetensors weights")
    return VerifiedModel(
        path=model_path,
        manifest_path=path,
        manifest_sha256=observed_digest,
        checkpoint_step=step,
        training_run_id=training_run_id,
        source_commits=normalized_source,
        file_count=len(observed),
        model_kind=model_kind,
        model_id=model_id,
        source_revision=source_revision,
    )


def load_eval_plan(
    *,
    run_id: str,
    run_dir: str | os.PathLike[str],
    resolved_config_path: str | os.PathLike[str],
    expected_resolved_config_sha256: str,
    schedule_path: str | os.PathLike[str],
    expected_schedule_sha256: str,
    schedule_manifest_path: str | os.PathLike[str],
    expected_schedule_manifest_sha256: str,
    route_registry_path: str | os.PathLike[str],
    expected_route_registry_sha256: str,
    agent_loop_config_path: str | os.PathLike[str],
    expected_agent_loop_config_sha256: str,
    model_manifest_path: str | os.PathLike[str],
    expected_model_manifest_sha256: str,
    training_run_id: str,
    training_outer_commit: str,
    training_inner_commit: str,
    training_verl_commit: str,
    evaluator_outer_commit: str,
    evaluator_inner_commit: str,
    evaluator_verl_commit: str,
    method_id: str = "agemem",
    model_kind: str = "merged_checkpoint",
    checkpoint_step: int = EXPECTED_CHECKPOINT_STEP,
    batch_size: int = EXPECTED_BATCH_SIZE,
    num_gpus: int = EXPECTED_NUM_GPUS,
    gpu_memory_utilization: float = 0.8,
) -> HeldoutEvalPlan:
    """Load and cross-check all immutable inputs before importing GPU code."""

    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters or length")
    method_id = str(method_id or "").strip().lower()
    model_kind = str(model_kind or "").strip().lower()
    if method_id not in SUPPORTED_METHOD_IDS:
        raise ValueError(f"unsupported held-out method_id {method_id!r}")
    if model_kind not in SUPPORTED_MODEL_KINDS:
        raise ValueError(f"unsupported held-out model_kind {model_kind!r}")
    expected_kind = "merged_checkpoint" if method_id == "agemem" else "frozen_hf"
    if model_kind != expected_kind:
        raise ValueError(
            f"held-out method {method_id!r} requires model_kind {expected_kind!r}"
        )
    expected_step = EXPECTED_CHECKPOINT_STEP if model_kind == "merged_checkpoint" else 0
    if checkpoint_step != expected_step:
        if model_kind == "merged_checkpoint":
            raise ValueError("native held-out evaluation is permitted only at update200")
        raise ValueError(
            f"held-out {model_kind} evaluation requires checkpoint_step={expected_step}"
        )
    if batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError(f"held-out batch_size must remain {EXPECTED_BATCH_SIZE}")
    if num_gpus != EXPECTED_NUM_GPUS:
        raise ValueError(f"held-out evaluation requires exactly {EXPECTED_NUM_GPUS} GPUs")
    if not isinstance(gpu_memory_utilization, (int, float)) or not 0 < float(
        gpu_memory_utilization
    ) < 1:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    destination = Path(run_dir)
    if not destination.is_absolute() or destination.is_symlink():
        raise ValueError("run_dir must be an absolute non-symlink path")

    resolved = _absolute_regular(resolved_config_path, field="resolved config")
    resolved_hash = require_sha256(
        expected_resolved_config_sha256,
        field="resolved config expected sha256",
    )
    if sha256_file(resolved) != resolved_hash:
        raise ValueError("resolved config sha256 mismatch")

    schedule_manifest = _absolute_regular(
        schedule_manifest_path, field="held-out schedule manifest"
    )
    schedule_manifest_hash = require_sha256(
        expected_schedule_manifest_sha256,
        field="held-out schedule manifest expected sha256",
    )
    if sha256_file(schedule_manifest) != schedule_manifest_hash:
        raise ValueError("held-out schedule manifest sha256 mismatch")
    schedule_metadata = _mapping(
        read_json(schedule_manifest), field="held-out schedule manifest"
    )
    if schedule_metadata.get("schema") != SCHEDULE_MANIFEST_SCHEMA:
        raise ValueError("held-out schedule manifest schema mismatch")
    route_counts, episode_count, split_authority, coding_admission = (
        _schedule_counts_and_coding_authority(schedule_metadata)
    )

    schedule = _absolute_regular(schedule_path, field="held-out schedule")
    schedule_hash = require_sha256(
        expected_schedule_sha256, field="held-out schedule expected sha256"
    )
    if schedule_metadata.get("schedule_sha256") != schedule_hash:
        raise ValueError("held-out schedule manifest identity/count drift")
    rows = inspect_heldout_schedule(
        schedule,
        expected_sha256=schedule_hash,
        expected_count=episode_count,
    )
    observed_counts = {
        route_id: sum(row["route_id"] == route_id for row in rows)
        for route_id in CANONICAL_ROUTES
    }
    if observed_counts != route_counts:
        raise ValueError(f"held-out route counts drifted: {observed_counts!r}")

    registry = _absolute_regular(route_registry_path, field="route registry")
    registry_hash = require_sha256(
        expected_route_registry_sha256,
        field="route registry expected sha256",
    )
    if sha256_file(registry) != registry_hash:
        raise ValueError("route registry sha256 mismatch")
    if schedule_metadata.get("route_registry_sha256") != registry_hash:
        raise ValueError("schedule and runtime route registry hashes differ")
    route_registry = load_route_registry(
        registry,
        expected_sha256=registry_hash,
        expected_route_ids=CANONICAL_ROUTES,
    )
    route_max_rounds = {
        route.route_id: route.max_rounds for route in route_registry.routes
    }
    route_attestations = {
        route.route_id: require_sha256(
            route.route_attestation_sha256,
            field=f"route {route.route_id} attestation",
        )
        for route in route_registry.routes
    }

    loop_config = _absolute_regular(
        agent_loop_config_path, field="agent-loop config"
    )
    loop_hash = require_sha256(
        expected_agent_loop_config_sha256,
        field="agent-loop config expected sha256",
    )
    if sha256_file(loop_config) != loop_hash:
        raise ValueError("agent-loop config sha256 mismatch")

    source_commits = {
        "outer": training_outer_commit,
        "inner": training_inner_commit,
        "verl": training_verl_commit,
    }
    model = verify_model_manifest(
        model_manifest_path,
        expected_manifest_sha256=expected_model_manifest_sha256,
        expected_checkpoint_step=checkpoint_step,
        expected_training_run_id=training_run_id,
        expected_source_commits=source_commits,
        model_kind=model_kind,
    )
    return HeldoutEvalPlan(
        run_id=run_id,
        run_dir=destination,
        resolved_config_path=resolved,
        resolved_config_sha256=resolved_hash,
        schedule_path=schedule,
        schedule_sha256=schedule_hash,
        schedule_manifest_path=schedule_manifest,
        schedule_manifest_sha256=schedule_manifest_hash,
        route_registry_path=registry,
        route_registry_sha256=registry_hash,
        route_counts=route_counts,
        episode_count=episode_count,
        complete_split_authority=split_authority,
        swesmith_formal_eval_authority=coding_admission,
        route_max_rounds=route_max_rounds,
        route_attestations=route_attestations,
        agent_loop_config_path=loop_config,
        agent_loop_config_sha256=loop_hash,
        model=model,
        evaluator_outer_commit=_commit(
            evaluator_outer_commit, field="evaluator outer commit"
        ),
        evaluator_inner_commit=_commit(
            evaluator_inner_commit, field="evaluator inner commit"
        ),
        evaluator_verl_commit=_commit(
            evaluator_verl_commit, field="evaluator veRL commit"
        ),
        method_id=method_id,
        model_kind=model_kind,
        batch_size=batch_size,
        num_gpus=num_gpus,
        gpu_memory_utilization=float(gpu_memory_utilization),
    )


def _nested_get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"resolved config is missing {dotted}")
        current = current[key]
    return current


def _nested_set(payload: dict[str, Any], dotted: str, value: Any) -> None:
    current = payload
    parts = dotted.split(".")
    for key in parts[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"resolved config is missing object {'.'.join(parts[:-1])}")
        current = child
    current[parts[-1]] = value


def derive_eval_config(plan: HeldoutEvalPlan) -> dict[str, Any]:
    """Derive a minimal-drift standalone inference config from the formal run."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - deployment image owns PyYAML
        raise RuntimeError("PyYAML is required to load the resolved config") from exc
    payload = yaml.safe_load(plan.resolved_config_path.read_text(encoding="utf-8"))
    config = deepcopy(_mapping(payload, field="resolved config"))
    required = {
        "actor_rollout_ref.rollout.name": "sglang",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "actor_rollout_ref.rollout.agent.default_agent_loop": AGENT_NAME,
        "data.custom_cls.name": "AMGTrajectoryDataset",
        "data.shuffle": False,
        "data.apply_chat_template_kwargs.enable_thinking": False,
        "distillation.enabled": False,
        "reward.reward_model.enable": False,
    }
    for field, expected in required.items():
        observed = _nested_get(config, field)
        if observed != expected:
            raise ValueError(
                f"formal resolved config drifted at {field}: "
                f"{observed!r} != {expected!r}"
            )

    overrides = {
        "actor_rollout_ref.model.path": str(plan.model.path),
        "actor_rollout_ref.model.tokenizer_path": str(plan.model.path),
        "actor_rollout_ref.model.use_shm": False,
        "actor_rollout_ref.rollout.nnodes": 1,
        "actor_rollout_ref.rollout.n_gpus_per_node": plan.num_gpus,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.data_parallel_size": 1,
        "actor_rollout_ref.rollout.pipeline_model_parallel_size": 1,
        "actor_rollout_ref.rollout.load_format": "auto",
        "actor_rollout_ref.rollout.skip_tokenizer_init": False,
        "actor_rollout_ref.rollout.gpu_memory_utilization": plan.gpu_memory_utilization,
        "actor_rollout_ref.rollout.full_determinism": True,
        "actor_rollout_ref.rollout.temperature": 0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.do_sample": False,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.agent.num_workers": plan.batch_size,
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(
            plan.agent_loop_config_path
        ),
        "actor_rollout_ref.rollout.trace.experiment_name": plan.run_id,
        "actor_rollout_ref.agentgym.route_registry_path": str(
            plan.route_registry_path
        ),
        "actor_rollout_ref.agentgym.route_registry_sha256": (
            plan.route_registry_sha256
        ),
        "data.train_files": str(plan.schedule_path),
        "data.val_files": str(plan.schedule_path),
        "data.train_max_samples": -1,
        "data.val_max_samples": -1,
        "data.train_batch_size": plan.batch_size,
        "data.gen_batch_size": plan.batch_size,
        "data.val_batch_size": plan.batch_size,
        "data.validation_shuffle": False,
        "data.dataloader_num_workers": 0,
        "data.agentgym.route_registry_path": str(plan.route_registry_path),
        "data.agentgym.route_registry_sha256": plan.route_registry_sha256,
        "trainer.experiment_name": plan.run_id,
        "trainer.validation_data_dir": str(plan.run_dir / "validation-data"),
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": plan.num_gpus,
    }
    for field, value in overrides.items():
        _nested_set(config, field, value)
    return config


def run_contract(plan: HeldoutEvalPlan, eval_config_sha256: str) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "run_id": plan.run_id,
        "method_id": plan.method_id,
        "model_kind": plan.model_kind,
        "checkpoint_step": plan.model.checkpoint_step,
        "training_run_id": plan.model.training_run_id,
        "training_source_commits": plan.model.source_commits,
        "evaluator_source_commits": {
            "outer": plan.evaluator_outer_commit,
            "inner": plan.evaluator_inner_commit,
            "verl": plan.evaluator_verl_commit,
        },
        "model": {
            "path": str(plan.model.path),
            "manifest_path": str(plan.model.manifest_path),
            "manifest_sha256": plan.model.manifest_sha256,
            "file_count": plan.model.file_count,
            "model_id": plan.model.model_id or None,
            "source_revision": plan.model.source_revision or None,
        },
        "schedule": {
            "path": str(plan.schedule_path),
            "sha256": plan.schedule_sha256,
            "manifest_path": str(plan.schedule_manifest_path),
            "manifest_sha256": plan.schedule_manifest_sha256,
            "episode_count": plan.episode_count,
            "per_route_rows": plan.route_counts,
            "complete_split_authority": plan.complete_split_authority,
            "swesmith_formal_eval_authority": plan.swesmith_formal_eval_authority,
        },
        "route_registry": {
            "path": str(plan.route_registry_path),
            "sha256": plan.route_registry_sha256,
            "max_rounds": plan.route_max_rounds,
            "route_attestations": plan.route_attestations,
        },
        "agent_loop_config": {
            "path": str(plan.agent_loop_config_path),
            "sha256": plan.agent_loop_config_sha256,
        },
        "formal_resolved_config": {
            "path": str(plan.resolved_config_path),
            "sha256": plan.resolved_config_sha256,
        },
        "derived_eval_config_sha256": require_sha256(
            eval_config_sha256, field="derived eval config sha256"
        ),
        "sampling": {
            "greedy": True,
            "temperature": 0,
            "top_p": 1.0,
            "top_k": -1,
            "rollout_n": 1,
            "batch_size": plan.batch_size,
            "padding_identity": "explicit_uid_and_eval_padding_marker_v1",
        },
        "resources": {
            "nnodes": 1,
            "gpus": plan.num_gpus,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": plan.gpu_memory_utilization,
        },
        "runner": {
            "dataset": "AMGTrajectoryDataset",
            "server_manager": "LLMServerManager.standalone",
            "agent_loop_manager": "AgentLoopManager",
            "policy_version_enforcement": (
                "every_action_row_min_max_equal_checkpoint_step"
            ),
            "generic_action_outcome_used": False,
        },
    }


def _runtime_environment_guard(plan: HeldoutEvalPlan) -> None:
    owner = os.environ.get("AGENTMEMORY_PROCESS_OWNER", "").strip()
    run_id = os.environ.get("AGENTMEMORY_RUN_ID", "").strip()
    if not owner or run_id != plan.run_id:
        raise RuntimeError(
            "held-out eval requires inherited AGENTMEMORY_PROCESS_OWNER and "
            "an exact AGENTMEMORY_RUN_ID"
        )
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        os.environ.pop(key, None)
        os.environ.pop(key.lower(), None)


def run_evaluation(plan: HeldoutEvalPlan) -> dict[str, Any]:
    """Run or strictly resume the held-out evaluation on one eight-GPU pod."""

    _runtime_environment_guard(plan)
    config_payload = derive_eval_config(plan)
    config_bytes = (
        json.dumps(
            config_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    config_sha256 = __import__("hashlib").sha256(config_bytes).hexdigest()
    contract = run_contract(plan, config_sha256)
    initialize_run_contract(plan.run_dir, contract)
    config_path = plan.run_dir / "eval-config.json"
    if config_path.exists() and config_path.read_bytes() != config_bytes:
        raise ValueError("existing derived eval config differs from requested run")
    if not config_path.exists():
        from .heldout_eval_contract import atomic_write_bytes

        atomic_write_bytes(config_path, config_bytes)

    schedule_rows = inspect_heldout_schedule(
        plan.schedule_path,
        expected_sha256=plan.schedule_sha256,
        expected_count=plan.episode_count,
    )
    resume = inspect_resume_state(plan.run_dir, schedule_rows=schedule_rows)
    if resume["next_schedule_position"] == len(schedule_rows):
        return finalize_run_metrics(
            plan.run_dir, expected_episode_count=plan.episode_count
        )

    # Delayed imports keep compose/verify/unit-test paths usable on the Mac,
    # which intentionally has no torch or Ray installation.
    import ray
    from omegaconf import OmegaConf

    from agentmemorygym_verl.dataset import AMGTrajectoryDataset
    from verl.experimental.agent_loop import AgentLoopManager
    from verl.protocol import DataProto
    from verl.utils import omega_conf_to_dataclass
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.workers.rollout.llm_server import LLMServerManager

    if ray.is_initialized():
        raise RuntimeError("refusing to attach held-out eval to an existing Ray runtime")
    config = OmegaConf.create(config_payload)
    runtime_env_vars = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "AGENTMEMORY_PROCESS_OWNER",
            "AGENTMEMORY_RUN_ID",
            "PYTHONPATH",
            "PATH",
            "LD_LIBRARY_PATH",
            "CUDA_HOME",
            "TOKENIZERS_PARALLELISM",
            "NCCL_DEBUG",
        }
    }
    ray.init(
        address="local",
        namespace=f"camg-heldout-{plan.run_id}",
        include_dashboard=False,
        runtime_env={"env_vars": runtime_env_vars},
    )
    try:
        visible_gpus = int(ray.cluster_resources().get("GPU", 0))
        if visible_gpus != plan.num_gpus:
            raise RuntimeError(
                f"held-out eval requires exactly {plan.num_gpus} Ray GPUs, "
                f"observed {visible_gpus}"
            )
        model_config = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        dataset = AMGTrajectoryDataset(
            data_files=str(plan.schedule_path),
            tokenizer=model_config.tokenizer,
            processor=model_config.processor,
            config=config.data,
            max_samples=-1,
        )
        if len(dataset) != plan.episode_count:
            raise RuntimeError(
                f"runtime dataset length drift: {len(dataset)} != {plan.episode_count}"
            )
        server_manager = LLMServerManager.create(
            config=config,
            worker_group=None,
            rollout_resource_pool=None,
        )
        server_handles = list(server_manager.server_handles)
        if len(server_handles) != plan.num_gpus:
            raise RuntimeError(
                f"standalone server replica count drift: "
                f"{len(server_handles)} != {plan.num_gpus}"
            )
        ray.get(
            [
                handle.set_global_steps.remote(plan.model.checkpoint_step)
                for handle in server_handles
            ]
        )
        loop_manager = AgentLoopManager.create(
            config=config,
            llm_client=server_manager.get_client(),
            reward_loop_worker_handles=None,
        )

        next_position = int(resume["next_schedule_position"])
        batch_index = int(resume["next_batch_index"])
        while next_position < len(schedule_rows):
            stop = min(next_position + plan.batch_size, len(schedule_rows))
            processed_rows = [dataset[index] for index in range(next_position, stop)]
            for offset, (processed, expected) in enumerate(
                zip(processed_rows, schedule_rows[next_position:stop])
            ):
                position = next_position + offset
                route_id = str(expected.get("route_id", ""))
                try:
                    attestation = plan.route_attestations[route_id]
                except KeyError as exc:
                    raise RuntimeError(
                        f"runtime dataset row {position} selects an unknown route"
                    ) from exc
                verify_runtime_dataset_row(
                    expected,
                    processed,
                    expected_route_attestation_sha256=attestation,
                    schedule_position=position,
                )
            padded_rows = pad_batch_rows(
                processed_rows,
                batch_index=batch_index,
                size_divisor=plan.batch_size,
                padding_index_base=PADDING_INDEX_BASE,
            )
            batch_dict = collate_fn(padded_rows)
            prompts = DataProto.from_single_dict(batch_dict)
            prompts.meta_info.update(
                {
                    "validate": True,
                    "global_steps": plan.model.checkpoint_step,
                    "temperature": 0,
                }
            )
            generated = loop_manager.generate_sequences(prompts)
            materialized = materialize_generated_batch(
                generated.non_tensor_batch,
                padded_rows,
                expected_global_step=plan.model.checkpoint_step,
                route_max_rounds=plan.route_max_rounds,
                method_id=plan.method_id,
            )
            receipt = commit_batch(
                plan.run_dir,
                batch_index=batch_index,
                schedule_start=next_position,
                schedule_stop=stop,
                materialized=materialized,
            )
            next_position = stop
            batch_index += 1
            atomic_write_json(
                plan.run_dir / "progress.json",
                {
                    "schema": "camg_heldout_eval_progress_v1",
                    "run_id": plan.run_id,
                    "checkpoint_step": plan.model.checkpoint_step,
                    "completed_batches": batch_index,
                    "completed_episodes": next_position,
                    "expected_episodes": plan.episode_count,
                    "latest_batch_metrics": receipt["batch_metrics"],
                    "updated_unix_seconds": time.time(),
                },
            )
        return finalize_run_metrics(
            plan.run_dir, expected_episode_count=plan.episode_count
        )
    finally:
        ray.shutdown()
