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

from .routes import load_route_registry


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
# Reserve the final 2,048-token response inside Qwen3.5's 32K context.
# A valid 30-action WebShop trajectory can exceed the former 16K prompt cap.
PPO_MAX_PROMPT_TOKENS = 30720


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
    _require_equal(config, "actor_rollout_ref.rollout.name", "sglang")
    _require_equal(config, "actor_rollout_ref.rollout.mode", "async")
    engine_kwargs = _plain(_at(config, "actor_rollout_ref.rollout.engine_kwargs"))
    if not isinstance(engine_kwargs, Mapping):
        raise ValueError("rollout engine_kwargs must be a mapping")
    if engine_kwargs.get("vllm") not in (None, {}):
        raise ValueError("SGLang rollout must not retain vLLM engine kwargs")
    sglang_kwargs = engine_kwargs.get("sglang")
    if not isinstance(sglang_kwargs, Mapping):
        raise ValueError("SGLang rollout requires engine_kwargs.sglang")
    for key, expected_value in {
        "mamba_scheduler_strategy": "no_buffer",
        "disable_radix_cache": True,
        "cuda_graph_max_bs": 32,
        "max_running_requests": 32,
        "chunked_prefill_size": 16384,
        "max_prefill_tokens": 16384,
    }.items():
        observed = sglang_kwargs.get(key)
        if observed != expected_value:
            raise ValueError(
                f"SGLang rollout engine_kwargs.{key} must be "
                f"{expected_value!r}, got {observed!r}"
            )
    _require_equal(config, "actor_rollout_ref.rollout.max_num_seqs", 32)
    _require_equal(config, "actor_rollout_ref.rollout.enforce_eager", False)
    _require_equal(config, "actor_rollout_ref.rollout.free_cache_engine", True)

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
    topology = (trainer_gpus, standalone_rollout_gpus)
    if mode == "formal" and topology != (6, 2):
        raise ValueError(
            "AMG formal Hybrid + Standalone topology must be 6+2, "
            f"got {trainer_gpus}+{standalone_rollout_gpus}"
        )
    if mode == "gate" and topology not in {(4, 4), (6, 2)}:
        raise ValueError(
            "AMG gate Hybrid + Standalone topology must be 4+4 or 6+2, "
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

    # Latest veRL enables Continuous Token for every AgentLoop and infers the
    # Qwen3.5 builder from the root Hugging Face model_type. The removed legacy
    # data.continuous_token config must not be reintroduced as an AMG shim.
    data_config = _plain(_at(config, "data"))
    if not isinstance(data_config, Mapping):
        raise ValueError("resolved data config must be a mapping")
    if "continuous_token" in data_config:
        raise ValueError(
            "legacy data.continuous_token must be absent; latest veRL owns "
            "AgentLoop Continuous Token selection from model_type"
        )
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
    has_registry = any(
        key in actor_agentgym
        for key in ("route_registry_path", "route_registry_sha256")
    )
    route_ids: list[str] | None = None
    route_registry_sha256: str | None = None
    env_addr: str | None = None
    if has_registry:
        forbidden_global_fields = (
            "task_name",
            "env_addr",
            "max_rounds",
            "max_observation_tokens",
            "timeout",
            "max_retries",
            "expected_manifest_sha256",
            "expected_release_revision",
            "expected_outer_commit",
            "expected_inner_commit",
            "expected_role",
            "expected_executor_runtime_digest",
            "expected_materializer_sha256",
            "expected_actions_sha256",
            "expected_max_observation_tokens",
        )
        for key in forbidden_global_fields:
            if key in actor_agentgym:
                raise ValueError(
                    f"multi-environment config must not set global agentgym.{key}"
                )
        expected_route_ids = expected.get("route_ids")
        if isinstance(expected_route_ids, (str, bytes)) or not isinstance(
            expected_route_ids, Sequence
        ):
            raise ValueError(
                "multi-environment expected_budget.route_ids must be a sequence"
            )
        registry = load_route_registry(
            actor_agentgym.get("route_registry_path"),
            expected_sha256=str(actor_agentgym.get("route_registry_sha256", "")),
            expected_route_ids=expected_route_ids,
        )
        if not registry.route_ids:
            raise ValueError("AMG route-set config requires at least one route")
        route_ids = list(registry.route_ids)
        route_registry_sha256 = registry.sha256
        if route_registry_sha256 != expected.get("route_registry_sha256"):
            raise ValueError(
                "AMG route registry differs from the selected multitask budget"
            )
    else:
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
            or any(
                character not in "0123456789abcdef" for character in runtime_digest[7:]
            )
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
        "data.max_prompt_length": PPO_MAX_PROMPT_TOKENS,
        "data.max_response_length": 2048,
        "data.return_raw_chat": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "critic.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.ppo_mini_batch_size": ppo_mini_batch_size,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 8,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.use_prefix_grouper": False,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.fsdp_config.reshard_after_forward": True,
        "critic.fsdp.reshard_after_forward": True,
        "critic.ppo_mini_batch_size": ppo_mini_batch_size,
        "critic.ppo_micro_batch_size_per_gpu": 8,
        "critic.ppo_epochs": 1,
        "critic.shuffle": False,
        "critic.use_dynamic_bsz": True,
        "critic.loss_agg_mode": "token-mean",
        "critic.ppo_max_token_len_per_gpu": 65536,
        "critic.ppo_infer_max_token_len_per_gpu": 32768,
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

    # These fields belonged to the previous locally patched veRL runtime.  They
    # are silently ignored by current upstream and must stay absent so a green
    # Hydra compose cannot be mistaken for runtime evidence.  Latest veRL owns
    # rollout JSONL, gradient/off-policy metrics, staleness counters, and
    # checkpoint persistence directly.
    legacy_async_evidence_fields = (
        "runtime_receipt_path",
        "rollout_data_non_tensor_keys",
        "rollout_data_non_tensor_max_keys",
        "parameter_update_probe",
    )
    async_training = _plain(_at(config, "async_training"))
    if not isinstance(async_training, Mapping):
        raise ValueError("resolved async_training config must be a mapping")
    present_legacy_fields = sorted(
        field for field in legacy_async_evidence_fields if field in async_training
    )
    if present_legacy_fields:
        raise ValueError(
            "legacy no-op async evidence config must be absent on latest veRL: "
            + ", ".join(present_legacy_fields)
        )

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
        "role": str(expected["role"]),
        "publication_cycles": publication_cycles,
        "trigger_parameter_sync_step": trigger,
        "optimizer_updates": optimizer_updates,
        "samples_per_update": samples_per_update,
        "episodes": episodes,
        "trainer_gpus": trainer_gpus,
        "standalone_rollout_gpus": standalone_rollout_gpus,
        "dynamic_hybrid_enabled": True,
        "gradient_checkpointing": {"actor": True, "critic": True},
        "critic_train_token_budget": 65536,
        "critic_infer_token_budget": 32768,
        "rollout_backend": "sglang",
        "fsdp2_reshard_after_forward": {"actor": True, "critic": True},
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
        "route_ids": route_ids,
        "route_registry_sha256": route_registry_sha256,
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
    expected_route_ids: Sequence[str] | None = None,
    expected_route_registry_sha256: str | None = None,
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
    global_indices: set[int] = set()
    route_mode: bool | None = None
    normalized_expected_routes: tuple[str, ...] | None = None
    if expected_route_ids is not None:
        if isinstance(expected_route_ids, (str, bytes)):
            raise TypeError("expected_route_ids must be a sequence, not a string")
        normalized_expected_routes = tuple(str(value) for value in expected_route_ids)
        if not normalized_expected_routes or any(
            not route_id for route_id in normalized_expected_routes
        ):
            raise ValueError("expected_route_ids must contain non-empty route IDs")
        if len(set(normalized_expected_routes)) != len(normalized_expected_routes):
            raise ValueError("expected_route_ids contains duplicates")
    if expected_route_registry_sha256 is not None:
        if len(expected_route_registry_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in expected_route_registry_sha256
        ):
            raise ValueError(
                "expected_route_registry_sha256 must be a lowercase SHA-256"
            )
    route_order: list[str] = []
    per_route_counts: dict[str, int] = {}
    per_route_provenance: dict[str, dict[str, str | None]] = {}
    route_registry_sha256: str | None = None
    agent_name: str | None = None
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
            data_idx = row.get("data_idx")
            if (
                not isinstance(data_idx, int)
                or isinstance(data_idx, bool)
                or data_idx < 0
            ):
                raise ValueError(
                    f"AMG schedule row {position} has invalid data_idx {data_idx!r}"
                )
            row_route = row.get("route_id")
            extra_route = extra.get("route_id")
            if row_route is not None and extra_route is not None:
                if str(row_route) != str(extra_route):
                    raise ValueError(f"AMG schedule route_id drift at row {position}")
            has_route = row_route is not None or extra_route is not None
            if route_mode is None:
                route_mode = has_route
            elif route_mode != has_route:
                raise ValueError("AMG schedule mixes routed and legacy rows")
            if normalized_expected_routes is not None and not has_route:
                raise ValueError(
                    "AMG multi-environment schedule row is missing route_id"
                )
            if has_route:
                route_id = str(row_route if row_route is not None else extra_route)
                if route_id not in per_route_counts:
                    route_order.append(route_id)
                    per_route_counts[route_id] = 0
                per_route_counts[route_id] += 1
                if normalized_expected_routes is not None:
                    expected_route = normalized_expected_routes[
                        position % len(normalized_expected_routes)
                    ]
                    if route_id != expected_route:
                        raise ValueError(
                            "AMG schedule route order drift at row "
                            f"{position}: expected {expected_route!r}, got {route_id!r}"
                        )
                    observed_agent_name = row.get("agent_name")
                    if observed_agent_name != "amg_task_neutral_async":
                        raise ValueError(
                            "AMG multitask schedule must select the shared "
                            "amg_task_neutral_async AgentLoop"
                        )
                    if row.get("data_source") != route_id:
                        raise ValueError(
                            f"AMG schedule data_source drift at row {position}"
                        )
                    if agent_name is None:
                        agent_name = str(observed_agent_name)
                    elif agent_name != observed_agent_name:
                        raise ValueError(
                            f"AMG schedule agent_name drift at row {position}"
                        )

                    registry_digest = str(extra.get("route_registry_sha256", ""))
                    attestation_digest = str(extra.get("route_attestation_sha256", ""))
                    source_schedule_digest = str(
                        extra.get("source_schedule_sha256", "")
                    )
                    source_manifest_digest = str(
                        extra.get("source_manifest_digest", "")
                    )
                    for field, value in (
                        ("route_registry_sha256", registry_digest),
                        ("route_attestation_sha256", attestation_digest),
                        ("source_schedule_sha256", source_schedule_digest),
                        ("source_manifest_digest", source_manifest_digest),
                    ):
                        if len(value) != 64 or any(
                            character not in "0123456789abcdef" for character in value
                        ):
                            raise ValueError(
                                f"AMG schedule row {position} has invalid {field}"
                            )
                    if route_registry_sha256 is None:
                        route_registry_sha256 = registry_digest
                    elif route_registry_sha256 != registry_digest:
                        raise ValueError(
                            f"AMG schedule route registry drift at row {position}"
                        )
                    provenance = {
                        "route_attestation_sha256": attestation_digest,
                        "source_schedule_sha256": source_schedule_digest,
                        "source_manifest_digest": source_manifest_digest,
                        "source_panel_id": (
                            str(extra["source_panel_id"])
                            if extra.get("source_panel_id") is not None
                            else None
                        ),
                    }
                    previous = per_route_provenance.setdefault(route_id, provenance)
                    if previous != provenance:
                        raise ValueError(
                            f"AMG schedule route provenance drift for {route_id!r}"
                        )

            raw_global_index = extra.get("index")
            if (
                not isinstance(raw_global_index, int)
                or isinstance(raw_global_index, bool)
                or raw_global_index < 0
            ):
                raise ValueError(
                    f"AMG schedule row {position} has invalid global index "
                    f"{raw_global_index!r}"
                )
            if has_route:
                if row.get("index") != raw_global_index:
                    raise ValueError(
                        f"AMG schedule global index drift at row {position}"
                    )
                if raw_global_index != position:
                    raise ValueError(
                        f"AMG schedule global index must equal schedule_position "
                        f"at row {position}"
                    )
                if raw_global_index in global_indices:
                    raise ValueError(
                        f"AMG schedule has duplicate global index {raw_global_index}"
                    )
            elif raw_global_index != data_idx:
                raise ValueError(
                    f"AMG legacy schedule index/data_idx drift at row {position}"
                )
            global_indices.add(raw_global_index)
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
    if normalized_expected_routes is not None:
        if tuple(route_order) != normalized_expected_routes:
            raise ValueError(
                "AMG schedule route order mismatch: "
                f"{tuple(route_order)!r} != {normalized_expected_routes!r}"
            )
        if set(per_route_counts) != set(normalized_expected_routes):
            raise ValueError("AMG schedule does not contain every expected route")
    if (
        expected_route_registry_sha256 is not None
        and route_registry_sha256 != expected_route_registry_sha256
    ):
        raise ValueError(
            "AMG schedule route registry digest mismatch: "
            f"{route_registry_sha256!r} != {expected_route_registry_sha256!r}"
        )
    return {
        "schema": "amg_schedule_identity_v1",
        "path": str(schedule_path.resolve()),
        "sha256": digest,
        "count": count,
        "unique_item_ids": len(item_ids),
        "unique_global_indices": len(global_indices),
        "manifest_digest": manifest_digest,
        "panel_id": panel_id,
        "role": role,
        "route_order": route_order,
        "per_route_counts": per_route_counts,
        "route_registry_sha256": route_registry_sha256,
        "agent_name": agent_name,
        "per_route_provenance": per_route_provenance,
        "first_schedule_position": 0,
        "last_schedule_position": count - 1,
    }
