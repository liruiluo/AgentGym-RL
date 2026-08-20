"""Fail-closed launch contract for AMG on veRL's native fully-async PPO.

This module validates only AMG-specific composition and experiment accounting.
It deliberately does not implement a scheduler, queue, checkpoint manager,
weight synchronizer, or PPO trainer; those remain owned by upstream veRL.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _at(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"resolved config is missing required field {path!r}")
        value = value[component]
    return value


def _require_equal(config: Mapping[str, Any], path: str, expected: Any) -> None:
    observed = _at(config, path)
    if observed != expected:
        raise ValueError(
            f"resolved config {path} must be {expected!r}, got {observed!r}"
        )


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a positive integer, got bool")
    try:
        integer = int(value)
        exact = float(value) == float(integer)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive integer, got {value!r}") from exc
    if not exact or integer <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return integer


_PPO_MINI_BATCH_TARGET = 512

# Reuse the largest packed-token budgets already validated by the matched
# eight-way synchronous OpenMLE PPO lineage. The six-way learner keeps
# gradient checkpointing enabled and must re-prove these values on B200 before
# a fresh formal lineage is admitted.
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU = 131_072
CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU = 163_840
CRITIC_FORWARD_MAX_TOKEN_LEN_PER_GPU = 262_144


def resolve_ppo_mini_batch_size(trainer_gpus: Any) -> int:
    """Keep the 512-row target while satisfying veRL's native DP alignment."""

    dp_size = _positive_int(trainer_gpus, field="trainer GPU DP size")
    mini_batch_size = _PPO_MINI_BATCH_TARGET - (_PPO_MINI_BATCH_TARGET % dp_size)
    if mini_batch_size <= 0:
        raise ValueError(
            "trainer GPU DP size cannot exceed the PPO mini-batch target: "
            f"{dp_size} > {_PPO_MINI_BATCH_TARGET}"
        )
    return mini_batch_size


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def verify_resolved_config(
    config: Mapping[str, Any],
    *,
    mode: str,
    expected_budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the resolved Hydra config against a publication-derived budget."""

    if mode not in {"gate", "formal"}:
        raise ValueError(f"unsupported AMG launch mode {mode!r}")
    if not isinstance(config, Mapping):
        raise TypeError("resolved config must be a mapping")
    if not isinstance(expected_budget, Mapping):
        raise TypeError("expected_budget must be a publication-derived mapping")
    if expected_budget.get("mode") != mode:
        raise ValueError(
            "expected_budget mode does not match the requested launch mode"
        )
    required_budget_fields = (
        "role",
        "publication_cycles",
        "trigger_parameter_sync_step",
        "optimizer_updates",
        "samples_per_update",
        "episodes",
        "save_freq",
        "max_actor_ckpt_to_keep",
        "max_critic_ckpt_to_keep",
        "model_path",
    )
    missing_budget_fields = [
        field for field in required_budget_fields if field not in expected_budget
    ]
    if missing_budget_fields:
        raise ValueError(
            "expected_budget is missing: " + ", ".join(missing_budget_fields)
        )
    expected = dict(expected_budget)

    if (
        _at(config, "algorithm.adv_estimator") != "amg_action_axis_gae"
        or _at(config, "critic.enable") is not True
    ):
        raise ValueError(
            "AMG PPO requires registered action-axis GAE and an enabled critic"
        )

    _require_equal(config, "actor_rollout_ref.rollout.n", 1)
    _require_equal(config, "rollout.n", 1)
    _require_equal(config, "actor_rollout_ref.rollout.calculate_log_probs", True)
    _require_equal(config, "actor_rollout_ref.actor.use_rollout_log_probs", True)
    _require_equal(config, "algorithm.rollout_correction.bypass_mode", True)
    _require_equal(config, "algorithm.rollout_correction.loss_type", "ppo_clip")
    _require_equal(
        config, "actor_rollout_ref.actor.policy_loss.loss_mode", "bypass_mode"
    )

    _require_equal(config, "actor_rollout_ref.hybrid_engine", False)
    if _at(config, "async_training.use_dynamic_resource_scheduling") is not True:
        raise ValueError(
            "AMG full async requires dynamic Hybrid + Standalone scheduling"
        )
    _require_equal(config, "async_training.dynamic_schedule_policy", "default")
    _require_equal(config, "async_training.dynamic_schedule_enable_rebalance", True)
    _require_equal(config, "actor_rollout_ref.rollout.name", "vllm")
    _require_equal(config, "actor_rollout_ref.rollout.mode", "async")

    trainer_nodes = _positive_int(_at(config, "trainer.nnodes"), field="trainer.nnodes")
    trainer_gpus_per_node = _positive_int(
        _at(config, "trainer.n_gpus_per_node"), field="trainer.n_gpus_per_node"
    )
    rollout_nodes = _positive_int(_at(config, "rollout.nnodes"), field="rollout.nnodes")
    rollout_gpus_per_node = _positive_int(
        _at(config, "rollout.n_gpus_per_node"), field="rollout.n_gpus_per_node"
    )
    trainer_gpus = trainer_nodes * trainer_gpus_per_node
    standalone_rollout_gpus = rollout_nodes * rollout_gpus_per_node
    if (trainer_gpus, standalone_rollout_gpus) not in {(4, 4), (6, 2)}:
        raise ValueError(
            "AMG reviewed Hybrid + Standalone topology must be 4+4 or 6+2, "
            f"got {trainer_gpus}+{standalone_rollout_gpus}"
        )

    hybrid_memory = _finite_number(
        _at(config, "actor_rollout_ref.rollout.gpu_memory_utilization"),
        field="hybrid rollout gpu_memory_utilization",
    )
    standalone_memory = _finite_number(
        _at(config, "actor_rollout_ref.rollout.standalone_gpu_memory_utilization"),
        field="standalone rollout gpu_memory_utilization",
    )
    if not 0.0 < hybrid_memory < standalone_memory <= 0.95:
        raise ValueError(
            "standalone rollout memory utilization must exceed the hybrid value and remain <=0.95"
        )

    if (
        _at(config, "data.continuous_token.enable") is not True
        or _at(config, "data.continuous_token.model_family") != "qwen35"
    ):
        raise ValueError("AMG Qwen3.5 multi-action rollout requires Continuous Token")
    _require_equal(config, "data.apply_chat_template_kwargs.enable_thinking", False)
    # veRL fully async stamps ``single_turn_agent`` when multi_turn is false.
    # AMG therefore needs the native multi-turn switch even though lifecycle
    # semantics stay inside the task-neutral custom AgentLoop.
    _require_equal(config, "actor_rollout_ref.rollout.multi_turn.enable", True)
    _require_equal(
        config,
        "actor_rollout_ref.rollout.agent.default_agent_loop",
        "amg_task_neutral_async",
    )
    loop_config_path = str(
        _at(config, "actor_rollout_ref.rollout.agent.agent_loop_config_path")
    )
    if not loop_config_path.endswith("amg_task_neutral_agent_loop.yaml"):
        raise ValueError("AMG memory/compaction AgentLoop config path is not selected")
    _require_equal(config, "data.custom_cls.path", "pkg://agentmemorygym_verl.dataset")
    _require_equal(config, "data.custom_cls.name", "AMGTrajectoryDataset")

    actor_agentgym = _plain(_at(config, "actor_rollout_ref.agentgym"))
    data_agentgym = _plain(_at(config, "data.agentgym"))
    if actor_agentgym != data_agentgym:
        raise ValueError(
            "actor_rollout_ref.agentgym and data.agentgym configs must match"
        )
    for key, expected_value in {
        "task_name": "openmle_fast",
        "max_rounds": 30,
        "max_observation_tokens": 8192,
        "timeout": 240,
        "max_retries": 2,
    }.items():
        if actor_agentgym.get(key) != expected_value:
            raise ValueError(
                f"AMG agentgym.{key} must be {expected_value!r}, "
                f"got {actor_agentgym.get(key)!r}"
            )
    env_addr = str(actor_agentgym.get("env_addr", ""))
    if not env_addr.startswith(("http://", "https://")):
        raise ValueError("AMG agentgym.env_addr must be an HTTP endpoint")
    expected_endpoint_role = str(expected["role"])
    endpoint_sha_fields = (
        "expected_manifest_sha256",
        "expected_materializer_sha256",
        "expected_actions_sha256",
    )
    for key in endpoint_sha_fields:
        value = actor_agentgym.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"AMG agentgym.{key} must be a lowercase SHA-256")
    for key in (
        "expected_release_revision",
        "expected_outer_commit",
        "expected_inner_commit",
    ):
        value = actor_agentgym.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"AMG agentgym.{key} must be a full Git revision")
    runtime_digest = actor_agentgym.get("expected_executor_runtime_digest")
    if (
        not isinstance(runtime_digest, str)
        or not runtime_digest.startswith("sha256:")
        or len(runtime_digest) != 71
        or any(character not in "0123456789abcdef" for character in runtime_digest[7:])
    ):
        raise ValueError(
            "AMG agentgym.expected_executor_runtime_digest must use sha256:<hex>"
        )
    if actor_agentgym.get("expected_role") != expected_endpoint_role:
        raise ValueError(
            "AMG endpoint role must match the launch mode: "
            f"expected {expected_endpoint_role!r}, got "
            f"{actor_agentgym.get('expected_role')!r}"
        )
    if actor_agentgym.get("expected_max_observation_tokens") != 8192:
        raise ValueError(
            "AMG endpoint expected_max_observation_tokens must be exactly 8192"
        )

    actor_fused = _at(config, "actor_rollout_ref.model.use_fused_kernels")
    critic_fused = _at(config, "critic.model.use_fused_kernels")
    if not isinstance(actor_fused, bool):
        raise ValueError("actor use_fused_kernels must be boolean")
    if not isinstance(critic_fused, bool):
        raise ValueError("critic use_fused_kernels must be boolean")
    _require_equal(
        config,
        "actor_rollout_ref.model.fused_kernel_options.impl_backend",
        "torch",
    )
    _require_equal(
        config,
        "critic.model.fused_kernel_options.impl_backend",
        "torch",
    )

    ppo_mini_batch_size = resolve_ppo_mini_batch_size(trainer_gpus)
    for path, expected_value in {
        "data.train_batch_size": 0,
        "data.gen_batch_size": 1,
        "data.shuffle": False,
        "data.seed": 233,
        "data.max_prompt_length": 16384,
        "data.max_response_length": 2048,
        "data.return_raw_chat": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "critic.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.ppo_mini_batch_size": ppo_mini_batch_size,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 8,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": (
            ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU
        ),
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "critic.ppo_mini_batch_size": ppo_mini_batch_size,
        "critic.ppo_micro_batch_size_per_gpu": 8,
        "critic.ppo_max_token_len_per_gpu": CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU,
        "critic.forward_max_token_len_per_gpu": (
            CRITIC_FORWARD_MAX_TOKEN_LEN_PER_GPU
        ),
        "critic.ppo_epochs": 1,
        "critic.shuffle": False,
        "critic.use_dynamic_bsz": True,
        "critic.strategy": "fsdp2",
        "algorithm.gamma": 1.0,
        "algorithm.lam": 1.0,
        "algorithm.amg_advantage_normalization": "upstream_masked_whiten",
        "algorithm.use_kl_in_reward": False,
        "trainer.total_epochs": 1,
        "trainer.val_before_train": False,
        "trainer.test_freq": -1,
        "trainer.resume_mode": "disable",
        "trainer.resume_from_path": None,
        "async_training.use_trainer_do_validate": False,
        "async_training.partial_rollout": True,
    }.items():
        _require_equal(config, path, expected_value)
    rollout_data_dir = _at(config, "trainer.rollout_data_dir")
    if not isinstance(rollout_data_dir, str) or not rollout_data_dir.strip():
        raise ValueError(
            "AMG fully-async training requires trainer.rollout_data_dir for durable row evidence"
        )

    runtime_receipt_path = _at(config, "async_training.runtime_receipt_path")
    if not isinstance(runtime_receipt_path, str) or not runtime_receipt_path.strip():
        raise ValueError(
            "AMG fully-async training requires the native runtime_receipt_path"
        )
    _require_equal(
        config,
        "async_training.rollout_data_non_tensor_keys",
        ["step_record_json"],
    )
    _require_equal(config, "async_training.rollout_data_non_tensor_max_keys", 1)
    for path, expected_value in {
        "async_training.parameter_update_probe.enabled": True,
        "async_training.parameter_update_probe.max_parameters": 8,
        "async_training.parameter_update_probe.max_elements_per_parameter": 16,
        "async_training.parameter_update_probe.atol": 0.0,
        "async_training.parameter_update_probe.require_change": True,
    }.items():
        _require_equal(config, path, expected_value)

    actor_model = str(_at(config, "actor_rollout_ref.model.path"))
    critic_model = str(_at(config, "critic.model.path"))
    expected_model = str(expected["model_path"])
    if actor_model != critic_model or actor_model != expected_model:
        raise ValueError(
            "actor and critic must start from the publication-locked model: "
            f"{actor_model!r}, {critic_model!r}, expected {expected_model!r}"
        )

    actor_lr = _finite_number(
        _at(config, "actor_rollout_ref.actor.optim.lr"), field="actor lr"
    )
    critic_lr = _finite_number(_at(config, "critic.optim.lr"), field="critic lr")
    if not math.isclose(actor_lr, 1e-6, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            f"actor lr must match synchronous comparator 1e-6, got {actor_lr}"
        )
    if not math.isclose(critic_lr, 1e-5, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            f"critic lr must match synchronous comparator 1e-5, got {critic_lr}"
        )

    mini_batch = _positive_int(
        _at(config, "actor_rollout_ref.actor.ppo_mini_batch_size"),
        field="actor ppo_mini_batch_size",
    )
    require_batches = _finite_number(
        _at(config, "async_training.require_batches"),
        field="async_training.require_batches",
    )
    required_samples_float = mini_batch * require_batches
    if (
        required_samples_float != int(required_samples_float)
        or required_samples_float <= 0
    ):
        raise ValueError(
            "ppo_mini_batch_size * require_batches must be a positive integral sample count"
        )
    samples_per_update = int(required_samples_float)
    expected_samples_per_update = _positive_int(
        expected["samples_per_update"], field="expected samples_per_update"
    )
    if samples_per_update != expected_samples_per_update:
        raise ValueError(
            "AMG update sample count differs from the selected publication: "
            f"{samples_per_update} != {expected_samples_per_update}"
        )

    trigger = _positive_int(
        _at(config, "async_training.trigger_parameter_sync_step"),
        field="trigger_parameter_sync_step",
    )
    publication_cycles = _positive_int(
        _at(config, "trainer.total_training_steps"),
        field="trainer.total_training_steps",
    )
    optimizer_updates = publication_cycles * trigger
    episodes = optimizer_updates * samples_per_update
    if publication_cycles != expected["publication_cycles"]:
        raise ValueError(
            f"{mode} trainer.total_training_steps denotes publication cycles and must be "
            f"{expected['publication_cycles']}, got {publication_cycles}"
        )
    if trigger != expected["trigger_parameter_sync_step"]:
        raise ValueError(
            f"{mode} trigger_parameter_sync_step must be "
            f"{expected['trigger_parameter_sync_step']}, got {trigger}"
        )
    if (
        optimizer_updates != expected["optimizer_updates"]
        or episodes != expected["episodes"]
    ):
        raise ValueError(
            f"{mode} budget mismatch: updates={optimizer_updates}, episodes={episodes}"
        )
    rollout_horizon = _positive_int(
        _at(config, "rollout.total_rollout_steps"),
        field="rollout.total_rollout_steps",
    )
    if rollout_horizon != episodes:
        raise ValueError(
            f"rollout horizon must equal the exact episode budget {episodes}, got {rollout_horizon}"
        )
    _require_equal(
        config,
        "trainer.save_freq",
        _positive_int(expected["save_freq"], field="expected save_freq"),
    )
    _require_equal(
        config,
        "trainer.max_actor_ckpt_to_keep",
        _positive_int(
            expected["max_actor_ckpt_to_keep"],
            field="expected max_actor_ckpt_to_keep",
        ),
    )
    _require_equal(
        config,
        "trainer.max_critic_ckpt_to_keep",
        _positive_int(
            expected["max_critic_ckpt_to_keep"],
            field="expected max_critic_ckpt_to_keep",
        ),
    )

    return {
        "schema": "amg_verl_fully_async_budget_v2",
        "mode": mode,
        "role": expected_endpoint_role,
        "publication_cycles": publication_cycles,
        "trigger_parameter_sync_step": trigger,
        "optimizer_updates": optimizer_updates,
        "samples_per_update": samples_per_update,
        "episodes": episodes,
        "trainer_gpus": trainer_gpus,
        "standalone_rollout_gpus": standalone_rollout_gpus,
        "dynamic_hybrid_enabled": True,
        "gradient_checkpointing": {"actor": True, "critic": True},
        "token_budgets": {
            "actor_ppo_max_token_len_per_gpu": (
                ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU
            ),
            "critic_ppo_max_token_len_per_gpu": (
                CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU
            ),
            "critic_forward_max_token_len_per_gpu": (
                CRITIC_FORWARD_MAX_TOKEN_LEN_PER_GPU
            ),
        },
        "fused_kernels": {
            "actor": actor_fused,
            "critic": critic_fused,
            "impl_backend": "torch",
        },
        "rollout_n": 1,
        "adv_estimator": "amg_action_axis_gae",
        "advantage_normalization": "upstream_masked_whiten",
        "model_path": actor_model,
        "env_addr": env_addr,
        "save_freq": _positive_int(expected["save_freq"], field="expected save_freq"),
        "max_actor_ckpt_to_keep": _positive_int(
            expected["max_actor_ckpt_to_keep"],
            field="expected max_actor_ckpt_to_keep",
        ),
        "max_critic_ckpt_to_keep": _positive_int(
            expected["max_critic_ckpt_to_keep"],
            field="expected max_critic_ckpt_to_keep",
        ),
        "task_count": expected.get("task_count"),
        "source_family_count": expected.get("source_family_count"),
        "schedule_sha256": expected.get("schedule_sha256"),
        "manifest_sha256": expected.get("manifest_sha256"),
        "routing_sha256": expected.get("routing_sha256"),
    }


def inspect_schedule(
    path: str | os.PathLike[str],
    *,
    expected_count: int | None = None,
    expected_sha256: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Validate a frozen AMG JSONL schedule and return exact identity evidence."""

    schedule_path = Path(path)
    if not schedule_path.is_file():
        raise FileNotFoundError(f"AMG schedule not found: {schedule_path}")
    digest = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"AMG schedule sha256 mismatch: expected {expected_sha256}, got {digest}"
        )

    item_ids: set[str] = set()
    manifest_digest: str | None = None
    panel_id: str | None = None
    role: str | None = None
    count = 0
    with schedule_path.open("r", encoding="utf-8") as handle:
        for position, raw_line in enumerate(handle):
            if not raw_line.strip():
                raise ValueError(f"AMG schedule contains a blank line at {position}")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid AMG schedule JSON at line {position + 1}"
                ) from exc
            if not isinstance(row, Mapping):
                raise TypeError(f"AMG schedule row {position} is not an object")
            item_id = str(row.get("item_id", ""))
            if not item_id:
                raise ValueError(f"AMG schedule row {position} has no item_id")
            extra = row.get("extra_info")
            if not isinstance(extra, Mapping):
                raise TypeError(f"AMG schedule row {position} has no extra_info object")
            if extra.get("schedule_position") != position:
                raise ValueError(
                    f"AMG schedule_position drift at row {position}: "
                    f"{extra.get('schedule_position')!r}"
                )
            if extra.get("index") != row.get("data_idx"):
                raise ValueError(f"AMG schedule index/data_idx drift at row {position}")
            row_role = extra.get("role")
            if row_role not in {"gate_only", "train_pool"}:
                raise ValueError(
                    f"AMG schedule row {position} has unsupported role {row_role!r}"
                )
            if expected_role is not None and row_role != expected_role:
                raise ValueError(
                    f"AMG schedule row {position} is not {expected_role}: {row_role!r}"
                )
            row_manifest = str(extra.get("manifest_digest", ""))
            row_panel = str(extra.get("panel_id", ""))
            if len(row_manifest) != 64:
                raise ValueError(
                    f"AMG schedule row {position} has invalid manifest digest"
                )
            if manifest_digest is None:
                manifest_digest = row_manifest
                panel_id = row_panel
                role = str(row_role)
            elif (
                row_manifest != manifest_digest
                or row_panel != panel_id
                or row_role != role
            ):
                raise ValueError(f"AMG schedule provenance drift at row {position}")
            if item_id in item_ids:
                raise ValueError(f"AMG schedule has duplicate item_id {item_id!r}")
            item_ids.add(item_id)
            count += 1

    if count == 0:
        raise ValueError("AMG schedule is empty")
    if expected_count is not None and count != expected_count:
        raise ValueError(
            f"AMG schedule must contain {expected_count} rows, got {count}"
        )
    return {
        "schema": "amg_schedule_identity_v1",
        "path": str(schedule_path.resolve()),
        "sha256": digest,
        "count": count,
        "unique_item_ids": len(item_ids),
        "manifest_digest": manifest_digest,
        "panel_id": panel_id,
        "role": role,
        "first_schedule_position": 0,
        "last_schedule_position": count - 1,
    }
