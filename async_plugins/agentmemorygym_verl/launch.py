"""Thin launcher for AMG on upstream veRL's native fully-async entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_contract import (
    PPO_MAX_PROMPT_TOKENS,
    inspect_schedule,
    resolve_ppo_mini_batch_size,
    verify_resolved_config,
)
from .finalizer import finalize_run
from .identity import (
    EXPECTED_VERL_COMMIT,
    LOCKED_MODEL_FILE_SHA256,
    TRL_WHEEL_RELATIVE_PATH,
    TRL_WHEEL_SHA256,
    reject_ambient_identity,
    validate_outer_change_paths,
    validate_training_runtime_lock,
    verify_hash_manifest,
)
from .orchestrator_lifecycle import process_identity_alive
from .routes import load_route_registry
from .resume_provenance import validate_resume_provenance_rebind

_ENDPOINT_SOURCE_LOCK_SCHEMA = "openmle_fast_launcher_source_lock_v1"
_MULTITASK_SOURCE_LOCK_SCHEMA = "amg_multitask_launcher_source_lock_v1"
_MULTITASK_SCHEDULE_CERTIFICATE_SCHEMA = "amg_multitask_schedule_certificate_v1"
_MULTITASK_ORCHESTRATOR_PREFLIGHT_SCHEMA = "amg_multitask_orchestrator_preflight_v1"
_MULTITASK_ROUTE_IDS = (
    "webshop",
    "swesmith",
    "literesearcher",
    "openmle_fast",
)
_ENDPOINT_ENV_FIELDS = {
    "expected_manifest_sha256": "OPENMLE_FAST_TASK_MANIFEST_SHA256",
    "expected_release_revision": "OPENMLE_FAST_RELEASE_REVISION",
    "expected_outer_commit": "OPENMLE_FAST_RUNTIME_OUTER_COMMIT",
    "expected_inner_commit": "OPENMLE_FAST_RUNTIME_INNER_COMMIT",
    "expected_role": "OPENMLE_FAST_MANIFEST_ROLE",
    "expected_executor_runtime_digest": "OPENMLE_FAST_EXECUTOR_RUNTIME_DIGEST",
    "expected_materializer_sha256": "OPENMLE_FAST_MATERIALIZER_SHA256",
    "expected_actions_sha256": "OPENMLE_FAST_ACTIONS_SHA256",
    "expected_max_observation_tokens": "OPENMLE_FAST_MAX_OBSERVATION_TOKENS",
}
_MAX_OBSERVATION_TOKENS = 8192
_UPSTREAM_ENTRYPOINT = "verl.experimental.fully_async_policy.fully_async_main"
_CUDA13_TOOLKIT_ROOT = Path("/dev/shm/cuda-13-b300-toolkit")
_EXPECTED_CUDA_VERSION = "13.0"
_ASYNC_TUNING = {
    "gate": {
        "trigger_parameter_sync_step": 1,
        "save_freq": 1,
        "max_actor_ckpt_to_keep": 1,
        "max_critic_ckpt_to_keep": 1,
    },
    "formal": {
        "trigger_parameter_sync_step": 1,
        "save_freq": 10,
        "max_actor_ckpt_to_keep": 1,
        "max_critic_ckpt_to_keep": 1,
    },
}


@dataclass(frozen=True)
class LaunchInputs:
    mode: str
    verl_root: Path
    outer_root: Path
    schedule: Path
    env_addr: str | None
    run_dir: Path
    experiment_name: str
    endpoint_source_lock: Path | None
    endpoint_contract_tool: Path | None
    publication_receipt: Path | None
    formal_schedule_certificate: Path | None
    trainer_gpus: int = 6
    standalone_rollout_gpus: int = 2
    actor_use_fused_kernels: bool = False
    critic_use_fused_kernels: bool = False
    route_registry: Path | None = None
    route_registry_sha256: str | None = None
    multitask_source_lock: Path | None = None
    multitask_schedule_certificate: Path | None = None
    multitask_orchestrator_preflight: Path | None = None
    learner_token_budget_profile: str = "default-65536-v1"
    actor_train_token_budget: int = 65_536
    critic_train_token_budget: int = 65_536
    resume_from_path: Path | None = None
    resume_prefix_run_dir: Path | None = None
    resume_start_update: int | None = None
    resume_target_update: int | None = None
    resume_sampler_samples_yielded: int | None = None
    resume_provenance_rebind: Path | None = None


def _resume_requested(inputs: LaunchInputs) -> bool:
    values = (
        inputs.resume_from_path,
        inputs.resume_prefix_run_dir,
        inputs.resume_start_update,
        inputs.resume_target_update,
        inputs.resume_sampler_samples_yielded,
    )
    present = tuple(value is not None for value in values)
    if any(present) and not all(present):
        raise ValueError(
            "resume_from_path, resume_prefix_run_dir, resume_start_update, "
            "resume_target_update, and resume_sampler_samples_yielded must be "
            "provided together"
        )
    if inputs.resume_provenance_rebind is not None and not all(present):
        raise ValueError("resume_provenance_rebind requires a complete resume request")
    return all(present)


def _string(value: str | Path) -> str:
    rendered = str(value)
    if not rendered or any(character in rendered for character in ("\n", "\r", "\0")):
        raise ValueError(f"unsafe empty or multiline Hydra value: {rendered!r}")
    return rendered


def build_overrides(
    inputs: LaunchInputs,
    *,
    effective_schedule: Path,
    endpoint_client_config: Mapping[str, str | int] | None,
    budget_contract: Mapping[str, Any],
    training_runtime: Mapping[str, Any],
) -> list[str]:
    """Build only Hydra overrides; upstream owns the composed base config."""

    if inputs.mode not in {"gate", "formal"}:
        raise ValueError(f"unsupported launch mode {inputs.mode!r}")
    token_budget_profiles = {
        "default-65536-v1": (65_536, 65_536),
        "multitask-131072-v1": (131_072, 131_072),
    }
    expected_token_budgets = token_budget_profiles.get(
        inputs.learner_token_budget_profile
    )
    if expected_token_budgets is None:
        raise ValueError(
            "unsupported learner token-budget profile: "
            f"{inputs.learner_token_budget_profile!r}"
        )
    actor_train_token_budget = _require_positive_int(
        inputs.actor_train_token_budget, field="actor train token budget"
    )
    critic_train_token_budget = _require_positive_int(
        inputs.critic_train_token_budget, field="critic train token budget"
    )
    if (actor_train_token_budget, critic_train_token_budget) != expected_token_budgets:
        raise ValueError(
            "learner token budgets do not match profile "
            f"{inputs.learner_token_budget_profile!r}: "
            f"{actor_train_token_budget}/{critic_train_token_budget} != "
            f"{expected_token_budgets[0]}/{expected_token_budgets[1]}"
        )
    if (inputs.trainer_gpus, inputs.standalone_rollout_gpus) not in {(4, 4), (6, 2)}:
        raise ValueError(
            "reviewed AMG Hybrid + Standalone topologies are 4+4 and 6+2, got "
            f"{inputs.trainer_gpus}+{inputs.standalone_rollout_gpus}"
        )
    tuning = _ASYNC_TUNING[inputs.mode]
    publication_cycles = _require_positive_int(
        budget_contract.get("publication_cycles"), field="budget publication_cycles"
    )
    trigger_parameter_sync_step = _require_positive_int(
        budget_contract.get("trigger_parameter_sync_step"),
        field="budget trigger_parameter_sync_step",
    )
    total_episodes = _require_positive_int(
        budget_contract.get("episodes"), field="budget episodes"
    )
    samples_per_update = _require_positive_int(
        budget_contract.get("samples_per_update"), field="budget samples_per_update"
    )
    save_freq = _require_positive_int(
        budget_contract.get("save_freq"), field="budget save_freq"
    )
    max_actor_ckpt_to_keep = _require_positive_int(
        budget_contract.get("max_actor_ckpt_to_keep"),
        field="budget max_actor_ckpt_to_keep",
    )
    max_critic_ckpt_to_keep = _require_positive_int(
        budget_contract.get("max_critic_ckpt_to_keep"),
        field="budget max_critic_ckpt_to_keep",
    )
    if trigger_parameter_sync_step != tuning["trigger_parameter_sync_step"]:
        raise ValueError(
            "publication budget does not match the reviewed async sync cadence"
        )
    if save_freq != tuning["save_freq"]:
        raise ValueError(
            "publication budget does not match the reviewed checkpoint cadence"
        )
    if (
        max_actor_ckpt_to_keep != tuning["max_actor_ckpt_to_keep"]
        or max_critic_ckpt_to_keep != tuning["max_critic_ckpt_to_keep"]
    ):
        raise ValueError(
            "publication budget does not match the reviewed checkpoint retention"
        )
    ppo_mini_batch_size = resolve_ppo_mini_batch_size(inputs.trainer_gpus)
    require_batches = samples_per_update / ppo_mini_batch_size
    if (
        require_batches <= 0
        or int(ppo_mini_batch_size * require_batches) != samples_per_update
    ):
        raise ValueError(
            "publication samples_per_update cannot be represented by require_batches"
        )
    model_path = _string(training_runtime["base_model"])
    schedule_path = _string(effective_schedule)
    run_dir = _string(inputs.run_dir)
    loop_config = _string(
        inputs.outer_root
        / "async_plugins"
        / "config"
        / "amg_task_neutral_agent_loop.yaml"
    )
    if (inputs.route_registry is None) != (inputs.route_registry_sha256 is None):
        raise ValueError(
            "route_registry and route_registry_sha256 must be provided together"
        )
    if inputs.route_registry is not None:
        if inputs.env_addr is not None:
            raise ValueError("multi-environment launch must not set a global env_addr")
        if endpoint_client_config:
            raise ValueError(
                "multi-environment launch must not set a global endpoint client config"
            )
        registry = load_route_registry(
            inputs.route_registry,
            expected_sha256=str(inputs.route_registry_sha256),
        )
        if len(registry.route_ids) != 4:
            raise ValueError(
                "AMG multitask launch requires exactly four registered routes"
            )
        agentgym: dict[str, Any] = {
            "route_registry_path": str(registry.source_path),
            "route_registry_sha256": registry.sha256,
            "route_registry_expected_ids": list(registry.route_ids),
        }
    else:
        if inputs.env_addr is None:
            raise ValueError("single-environment launch requires env_addr")
        if endpoint_client_config is None:
            raise ValueError(
                "single-environment launch requires endpoint client config"
            )
        env_addr = _string(inputs.env_addr.rstrip("/"))
        agentgym = {
            "task_name": "openmle_fast",
            "env_addr": env_addr,
            "max_rounds": 30,
            "max_observation_tokens": _MAX_OBSERVATION_TOKENS,
            "timeout": 240,
            "max_retries": 2,
            **endpoint_client_config,
        }
    overrides = [
        f"data.train_files={schedule_path}",
        f"data.val_files={schedule_path}",
        "data.train_batch_size=0",
        "data.gen_batch_size=1",
        "data.train_max_samples=-1",
        "data.val_max_samples=1",
        "data.dataloader_num_workers=0",
        "data.prompt_key=item_id",
        f"data.max_prompt_length={PPO_MAX_PROMPT_TOKENS}",
        "data.max_response_length=2048",
        "data.truncation=error",
        "data.return_raw_chat=True",
        "data.return_raw_input_ids=False",
        "data.shuffle=False",
        "data.seed=233",
        "data.custom_cls.path=pkg://agentmemorygym_verl.dataset",
        "data.custom_cls.name=AMGTrajectoryDataset",
        # Latest veRL owns Continuous Token unconditionally for AgentLoop and
        # selects Qwen3.5 from the root Hugging Face model_type. Do not restore
        # the removed legacy data.continuous_token config surface.
        # Reuse veRL/Transformers native Qwen3.5 template control. The frozen
        # synchronous baseline used a closed thinking block so each generation
        # is the bare three-tool action expected by the environment parser.
        "+data.apply_chat_template_kwargs.enable_thinking=False",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.model.use_fused_kernels={inputs.actor_use_fused_kernels}",
        "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch",
        # Keep veRL's native HF/FSDP gradient checkpointing enabled. The
        # synchronous comparator used the upstream default successfully;
        # disabling it made the six-way async learner retain full activations.
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"critic.model.path={model_path}",
        f"critic.model.tokenizer_path={model_path}",
        "critic.model.trust_remote_code=True",
        "critic.model.use_remove_padding=True",
        f"critic.model.use_fused_kernels={inputs.critic_use_fused_kernels}",
        "critic.model.fused_kernel_options.impl_backend=torch",
        "critic.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.actor.fsdp_config.strategy=fsdp2",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        # Start the latest-veRL migration from the upstream FSDP2 default.
        "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.shuffle=False",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        # Six-way FSDP leaves less activation headroom than the historical
        # eight-way synchronous trainer. Keep microbatch=8 but bound packed
        # training tokens; formal tuning may raise these after measured headroom.
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor_train_token_budget}",
        "actor_rollout_ref.actor.use_rollout_log_probs=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.weight_decay=0.01",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        "actor_rollout_ref.actor.clip_ratio=0.2",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.2",
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.kl_loss_coef=0.0",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        # Synthetic rows used only for native DP/mini-batch alignment are neutral
        # under token-mean aggregation. PrefixGrouper requires a separate grouped
        # padding proof and is deliberately not enabled in this baseline.
        "actor_rollout_ref.actor.use_prefix_grouper=False",
        "actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode",
        "critic.enable=True",
        "critic.strategy=fsdp2",
        "critic.fsdp.strategy=fsdp2",
        "critic.fsdp.param_offload=False",
        "critic.fsdp.optimizer_offload=False",
        # Start the latest-veRL migration from the upstream FSDP2 default. Any
        # no-reshard treatment must earn its place in a later isolated speed and
        # action-adoption comparison rather than leaking into this baseline.
        "critic.fsdp.reshard_after_forward=True",
        f"critic.ppo_mini_batch_size={ppo_mini_batch_size}",
        "critic.ppo_micro_batch_size_per_gpu=8",
        "critic.ppo_epochs=1",
        "critic.shuffle=False",
        "critic.use_dynamic_bsz=True",
        "critic.loss_agg_mode=token-mean",
        # Preserve r38's verified 65,536-token critic training budget. Latest
        # veRL splits critic inference packing into its own knob, which remains
        # at the upstream 32,768-token default; do not conflate the two.
        f"critic.ppo_max_token_len_per_gpu={critic_train_token_budget}",
        "+critic.ppo_infer_max_token_len_per_gpu=32768",
        "critic.forward_max_token_len_per_gpu=262144",
        "critic.optim.lr=1e-5",
        "critic.optim.weight_decay=0.01",
        "critic.optim.lr_warmup_steps=0",
        "critic.optim.lr_scheduler_type=constant",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.dtype=bfloat16",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
        "actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.8",
        "actor_rollout_ref.rollout.max_model_len=32768",
        "actor_rollout_ref.rollout.max_num_seqs=32",
        "actor_rollout_ref.rollout.enforce_eager=False",
        # Qwen3.5's GDN state is handled by SGLang's native no-buffer scheduler.
        # These are the upstream Qwen3.5 fully-async knobs; AMG does not add an
        # inference scheduler or a model-specific rollout implementation.
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy=no_buffer",
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True",
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=32",
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests=32",
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=16384",
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=16384",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.disable_log_stats=False",
        # Upstream fully-async prepare_single_generation_data() selects the
        # configured AgentLoop only when multi_turn is enabled; otherwise it
        # deliberately stamps every sample as single_turn_agent.
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=131072",
        "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024",
        "actor_rollout_ref.nccl_timeout=9600",
        "actor_rollout_ref.rollout.agent.num_workers=64",
        "actor_rollout_ref.rollout.agent.default_agent_loop=amg_task_neutral_async",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={loop_config}",
        "actor_rollout_ref.hybrid_engine=False",
        "algorithm.adv_estimator=amg_action_axis_gae",
        "++algorithm.amg_advantage_normalization=upstream_masked_whiten",
        "algorithm.gamma=1.0",
        "algorithm.lam=1.0",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.0",
        "algorithm.rollout_correction.bypass_mode=True",
        "algorithm.rollout_correction.loss_type=ppo_clip",
        "algorithm.rollout_correction.rollout_is=null",
        "algorithm.rollout_correction.rollout_rs=null",
        # Ray otherwise reserves roughly 30% of the node's currently available
        # memory in tmpfs. AMG batches use far less object-store space, while
        # the colocated retrieval index needs that headroom to stay resident.
        "++ray_kwargs.ray_init.object_store_memory=8589934592",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={inputs.trainer_gpus}",
        "trainer.device=cuda",
        "trainer.balance_batch=True",
        "trainer.critic_warmup=0",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={publication_cycles}",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        (
            "trainer.resume_mode=resume_path"
            if _resume_requested(inputs)
            else "trainer.resume_mode=disable"
        ),
        (
            f"trainer.resume_from_path={_string(inputs.resume_from_path)}"
            if _resume_requested(inputs)
            else "trainer.resume_from_path=null"
        ),
        f"trainer.save_freq={save_freq}",
        f"trainer.max_actor_ckpt_to_keep={max_actor_ckpt_to_keep}",
        f"trainer.max_critic_ckpt_to_keep={max_critic_ckpt_to_keep}",
        "trainer.logger=[console,file]",
        "trainer.project_name=agentmemorygym",
        f"trainer.experiment_name={_string(inputs.experiment_name)}",
        f"trainer.default_local_dir={run_dir}/checkpoints",
        f"trainer.rollout_data_dir={run_dir}/rollout_data",
        "trainer.validation_data_dir=null",
        "rollout.nnodes=1",
        f"rollout.n_gpus_per_node={inputs.standalone_rollout_gpus}",
        "rollout.n=1",
        f"rollout.total_rollout_steps={total_episodes}",
        "async_training.staleness_threshold=0.1",
        f"async_training.require_batches={require_batches:.17g}",
        f"async_training.trigger_parameter_sync_step={trigger_parameter_sync_step}",
        "async_training.partial_rollout=True",
        "async_training.use_trainer_do_validate=False",
        "async_training.use_dynamic_resource_scheduling=True",
        "async_training.dynamic_schedule_policy=default",
        "async_training.dynamic_schedule_deactivate_ratio=0.6",
        "async_training.dynamic_schedule_enable_rebalance=True",
        "async_training.concurrent_samples_per_replica=16",
        "+trainer.worker_env.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"hydra.run.dir={run_dir}/hydra",
        "hydra.output_subdir=.hydra",
        "hydra.job.chdir=False",
    ]
    for prefix in ("actor_rollout_ref", "data"):
        for key, value in agentgym.items():
            rendered = (
                json.dumps(value, ensure_ascii=True, separators=(",", ":"))
                if isinstance(value, (str, list, tuple, dict))
                else str(value)
            )
            overrides.append(f"++{prefix}.agentgym.{key}={rendered}")
    return overrides


def build_runtime_env(
    inputs: LaunchInputs,
    *,
    training_runtime: Mapping[str, Any],
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    python_entries = [
        inputs.outer_root / TRL_WHEEL_RELATIVE_PATH,
        inputs.outer_root / "async_plugins",
        inputs.verl_root,
        inputs.outer_root / "AgentGym" / "agentenv",
        inputs.outer_root / "AgentGym" / "agentenv-openmle-fast",
    ]
    closed_pythonpath = os.pathsep.join(str(entry) for entry in python_entries)
    inherited_pythonpath = env.get("PYTHONPATH")
    environment_to_check = dict(env)
    if inherited_pythonpath == closed_pythonpath:
        # The shell wrapper needs this exact closed path to import the launcher;
        # it is recomputed here rather than treated as caller-selected identity.
        environment_to_check.pop("PYTHONPATH")
    reject_ambient_identity(environment_to_check)
    env["PYTHONPATH"] = closed_pythonpath
    runtime_bin = str(Path(_string(training_runtime["python"])).parent)
    cuda_home = str(_CUDA13_TOOLKIT_ROOT)
    cuda_bin = str(_CUDA13_TOOLKIT_ROOT / "bin")
    inherited_path = env.get("PATH", "")
    path_entries = [
        entry
        for entry in inherited_path.split(os.pathsep)
        if entry and entry not in {cuda_bin, runtime_bin}
    ]
    env["PATH"] = os.pathsep.join([cuda_bin, runtime_bin, *path_entries])
    runtime_cuda_lib = str(
        Path(_string(training_runtime["site_packages"])) / "nvidia" / "cu13" / "lib"
    )
    cuda_library_entries = [
        str(_CUDA13_TOOLKIT_ROOT / "lib64"),
        "/usr/local/cuda/lib64/stubs",
        runtime_cuda_lib,
    ]
    inherited_ld_library_path = env.get("LD_LIBRARY_PATH", "")
    cuda_library_entries.extend(
        entry
        for entry in inherited_ld_library_path.split(os.pathsep)
        if entry and entry not in cuda_library_entries
    )
    env["CUDA_HOME"] = cuda_home
    env["CUDA_PATH"] = cuda_home
    env["LD_LIBRARY_PATH"] = os.pathsep.join(cuda_library_entries)
    env["VERL_USE_EXTERNAL_MODULES"] = "agentmemorygym_verl.action_gae"
    env["VERL_USE_EXTERNAL_PLUGINS"] = "none"
    env["VERL_FILE_LOGGER_PATH"] = str(inputs.run_dir / "metrics.jsonl")
    env.pop("VERL_FULLY_ASYNC_RUNTIME_RECEIPT_PATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HYDRA_FULL_ERROR"] = "1"
    env["RAY_DEDUP_LOGS"] = "0"
    # Keep Ray's memory-pressure fail-safe enabled while leaving enough headroom
    # for the colocated LiteResearcher retrieval stack. The default 0.95
    # threshold killed the asynchronous MessageQueue at 97.37% node memory.
    env["RAY_memory_usage_threshold"] = "0.98"
    return env


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"endpoint source lock {field} must be a lowercase SHA-256")
    return value


def _require_git_revision(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"endpoint source lock {field} must be a full Git revision")
    return value


def _require_runtime_digest(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError("endpoint executor runtime digest must use sha256:<hex>")
    _require_sha256(value[7:], field="exact_runtime.runtime_digest")
    return value


def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON mapping: {path}")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def _require_regular_file(path: Path | None, *, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is required")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is missing or not regular: {path}")
    return path


def _require_exact_multitask_outer_commit(
    *,
    launch_identity_schema: object,
    publication_outer_commit: str,
    observed_outer_commit: str,
) -> None:
    """Freeze the complete outer runtime for a multitask launch."""

    if (
        launch_identity_schema == "amg_multitask_source_identity_v1"
        and observed_outer_commit != publication_outer_commit
    ):
        raise RuntimeError(
            "multitask launch requires the exact outer commit from its source lock: "
            f"expected {publication_outer_commit}, got {observed_outer_commit}"
        )


def _preserve_legacy_runtime_preflight_fields(
    receipt: Mapping[str, Any], *, multitask: bool
) -> dict[str, Any]:
    """Keep the v5 single-route receipt surface while adding route receipts."""

    normalized = dict(receipt)
    if multitask:
        return normalized
    routes = normalized.get("routes")
    if (
        isinstance(routes, (str, bytes))
        or not isinstance(routes, Sequence)
        or len(routes) != 1
        or not isinstance(routes[0], Mapping)
    ):
        raise RuntimeError(
            "legacy runtime preflight must contain exactly one route receipt"
        )
    route = routes[0]
    for field in ("policy_framing_messages", "policy_framing_sha256"):
        value = route.get(field)
        if field in normalized and normalized[field] != value:
            raise RuntimeError(
                f"legacy runtime preflight {field} disagrees with route receipt"
            )
        normalized[field] = value
    return normalized


def _load_multitask_identity(
    inputs: LaunchInputs, *, schedule_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve one immutable four-environment source/runtime identity.

    The route registry and schedule certificate own environment composition.
    The source lock owns only repository, veRL, runtime, model, and selected-file
    identity.  This keeps the generic launcher independent of any one wrapper's
    publication format while retaining the stricter legacy OpenMLE validator.
    """

    if inputs.mode not in _ASYNC_TUNING:
        raise ValueError(f"unsupported launch mode {inputs.mode!r}")
    if inputs.route_registry is None or inputs.route_registry_sha256 is None:
        raise ValueError(
            "multitask launch requires route_registry and route_registry_sha256"
        )
    if inputs.env_addr is not None:
        raise ValueError("multitask launch must not set a global env_addr")
    endpoint_only = {
        "endpoint_source_lock": inputs.endpoint_source_lock,
        "endpoint_contract_tool": inputs.endpoint_contract_tool,
        "publication_receipt": inputs.publication_receipt,
        "formal_schedule_certificate": inputs.formal_schedule_certificate,
    }
    conflicts = sorted(
        name for name, value in endpoint_only.items() if value is not None
    )
    if conflicts:
        raise ValueError(
            "multitask launch must not set OpenMLE-only publication inputs: "
            + ", ".join(conflicts)
        )

    source_lock_path = _require_regular_file(
        inputs.multitask_source_lock, label="multitask source lock"
    )
    certificate_path = _require_regular_file(
        inputs.multitask_schedule_certificate,
        label="multitask schedule certificate",
    )
    registry = load_route_registry(
        inputs.route_registry,
        expected_sha256=inputs.route_registry_sha256,
        expected_route_ids=_MULTITASK_ROUTE_IDS,
    )
    source_lock = _load_json_mapping(source_lock_path, label="multitask source lock")
    certificate = _load_json_mapping(
        certificate_path, label="multitask schedule certificate"
    )
    if (
        source_lock.get("schema") != _MULTITASK_SOURCE_LOCK_SCHEMA
        or source_lock.get("status") != "pass"
    ):
        raise ValueError("multitask source lock is not a completed v1 lock")
    if certificate.get("schema") != _MULTITASK_SCHEDULE_CERTIFICATE_SCHEMA:
        raise ValueError("multitask schedule certificate schema drifted")

    source_lock_sha256 = _sha256(source_lock_path)
    certificate_sha256 = _sha256(certificate_path)
    runtime_source = source_lock.get("runtime_source")
    integration = source_lock.get("integration")
    raw_training_runtime = source_lock.get("training_runtime")
    if not all(
        isinstance(value, Mapping)
        for value in (runtime_source, integration, raw_training_runtime)
    ):
        raise ValueError("multitask source lock omitted runtime identity sections")
    assert isinstance(runtime_source, Mapping)
    assert isinstance(integration, Mapping)
    assert isinstance(raw_training_runtime, Mapping)

    publication_outer_commit = _require_git_revision(
        runtime_source.get("outer_commit"), field="runtime_source.outer_commit"
    )
    publication_inner_commit = _require_git_revision(
        runtime_source.get("inner_commit"), field="runtime_source.inner_commit"
    )
    locked_verl_commit = _require_git_revision(
        runtime_source.get("verl_commit"), field="runtime_source.verl_commit"
    )
    if locked_verl_commit != EXPECTED_VERL_COMMIT:
        raise ValueError(
            "multitask source lock selected an unreviewed veRL commit: "
            f"{locked_verl_commit} != {EXPECTED_VERL_COMMIT}"
        )
    selected_files = runtime_source.get("selected_files")
    if not isinstance(selected_files, Mapping) or not selected_files:
        raise ValueError("multitask source lock omitted selected runtime file hashes")
    for identity_path, digest in selected_files.items():
        if not isinstance(identity_path, str):
            raise TypeError("multitask selected file identity must be text")
        _require_sha256(digest, field=f"runtime_source.selected_files.{identity_path}")
    selected_outer_files, selected_inner_files = _partition_selected_file_hashes(
        selected_files
    )
    if not selected_outer_files or not selected_inner_files:
        raise ValueError(
            "multitask source lock must bind both outer and AgentGym runtime files"
        )
    training_runtime = validate_training_runtime_lock(raw_training_runtime)

    registry_binding = integration.get("route_registry")
    schedule_binding = integration.get("schedule_certificate")
    if not isinstance(registry_binding, Mapping) or not isinstance(
        schedule_binding, Mapping
    ):
        raise ValueError("multitask source lock omitted integration bindings")
    bound_route_ids = registry_binding.get("route_ids")
    if (
        isinstance(bound_route_ids, (str, bytes))
        or not isinstance(bound_route_ids, Sequence)
        or tuple(str(value) for value in bound_route_ids) != _MULTITASK_ROUTE_IDS
    ):
        raise ValueError("multitask source lock route order drifted")
    if (
        _require_sha256(
            registry_binding.get("sha256"),
            field="integration.route_registry.sha256",
        )
        != registry.sha256
    ):
        raise ValueError("multitask source lock route registry digest drifted")
    if (
        _require_sha256(
            schedule_binding.get("sha256"),
            field="integration.schedule_certificate.sha256",
        )
        != certificate_sha256
    ):
        raise ValueError("multitask source lock schedule certificate digest drifted")

    role = "gate_only" if inputs.mode == "gate" else "train_pool"
    if certificate.get("role") != role:
        raise ValueError(
            f"multitask schedule role mismatch: {certificate.get('role')!r} != {role!r}"
        )
    if certificate.get("agent_name") != "amg_task_neutral_async":
        raise ValueError("multitask schedule selected a non-shared AgentLoop")
    route_order = certificate.get("route_order")
    if (
        isinstance(route_order, (str, bytes))
        or not isinstance(route_order, Sequence)
        or tuple(str(value) for value in route_order) != _MULTITASK_ROUTE_IDS
    ):
        raise ValueError("multitask schedule certificate route order drifted")
    if certificate.get("route_registry_sha256") != registry.sha256:
        raise ValueError("multitask schedule certificate registry digest drifted")

    optimizer_updates = _require_positive_int(
        certificate.get("optimizer_updates"),
        field="multitask certificate optimizer_updates",
    )
    samples_per_update = _require_positive_int(
        certificate.get("samples_per_update"),
        field="multitask certificate samples_per_update",
    )
    scheduled_episode_count = _require_positive_int(
        certificate.get("row_count"), field="multitask certificate row_count"
    )
    expected_updates = 1 if inputs.mode == "gate" else 400
    if optimizer_updates != expected_updates or samples_per_update != 64:
        raise ValueError(
            "multitask budget must be gate1 or formal400 with 64 episodes/update: "
            f"updates={optimizer_updates}, samples_per_update={samples_per_update}"
        )
    if scheduled_episode_count != optimizer_updates * samples_per_update:
        raise ValueError(
            "multitask schedule rows do not conserve optimizer update budget"
        )
    rows_per_route = scheduled_episode_count // len(_MULTITASK_ROUTE_IDS)
    expected_per_route_rows = {
        route_id: rows_per_route for route_id in _MULTITASK_ROUTE_IDS
    }
    if certificate.get("per_route_rows") != expected_per_route_rows:
        raise ValueError("multitask certificate per-route episode budget drifted")

    certificate_schedule_sha256 = _require_sha256(
        certificate.get("schedule_sha256"),
        field="multitask certificate schedule_sha256",
    )
    if (
        _require_sha256(
            schedule_binding.get("schedule_sha256"),
            field="integration.schedule_certificate.schedule_sha256",
        )
        != certificate_schedule_sha256
    ):
        raise ValueError("multitask source lock schedule digest drifted")
    certificate_panel_id = certificate.get("panel_id")
    if not isinstance(certificate_panel_id, str) or not certificate_panel_id.strip():
        raise ValueError("multitask certificate panel_id must be non-empty text")
    expected_report = {
        "sha256": certificate_schedule_sha256,
        "count": scheduled_episode_count,
        "role": role,
        "panel_id": certificate_panel_id,
        "route_order": list(_MULTITASK_ROUTE_IDS),
        "per_route_counts": expected_per_route_rows,
        "route_registry_sha256": registry.sha256,
        "agent_name": "amg_task_neutral_async",
        "manifest_digest": _require_sha256(
            certificate.get("spec_sha256"),
            field="multitask certificate spec_sha256",
        ),
    }
    for field, expected in expected_report.items():
        if schedule_report.get(field) != expected:
            raise ValueError(
                f"multitask schedule {field} drifted: "
                f"{schedule_report.get(field)!r} != {expected!r}"
            )

    sources = certificate.get("sources")
    provenance = schedule_report.get("per_route_provenance")
    if not isinstance(sources, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("multitask schedule omitted per-route provenance")
    if set(sources) != set(_MULTITASK_ROUTE_IDS):
        raise ValueError("multitask certificate source routes drifted")
    for route in registry.routes:
        source = sources.get(route.route_id)
        route_provenance = provenance.get(route.route_id)
        if not isinstance(source, Mapping) or not isinstance(route_provenance, Mapping):
            raise ValueError(
                f"multitask route {route.route_id!r} omitted source provenance"
            )
        source_row_count = _require_positive_int(
            source.get("source_row_count"),
            field=(f"multitask certificate sources.{route.route_id}.source_row_count"),
        )
        allow_repetition = source.get("allow_repetition")
        if not isinstance(allow_repetition, bool):
            raise TypeError(
                "multitask certificate sources."
                f"{route.route_id}.allow_repetition must be boolean"
            )
        scheduled_route_rows = expected_per_route_rows[route.route_id]
        if scheduled_route_rows > source_row_count and not allow_repetition:
            raise ValueError(
                f"multitask route {route.route_id!r} source would exhaust at "
                f"{source_row_count}/{scheduled_route_rows} rows without explicit "
                "repetition"
            )
        source_schedule_sha256 = _require_sha256(
            source.get("schedule_sha256"),
            field=f"multitask certificate sources.{route.route_id}.schedule_sha256",
        )
        route_attestation_sha256 = _require_sha256(
            source.get("route_attestation_sha256"),
            field=(
                "multitask certificate sources."
                f"{route.route_id}.route_attestation_sha256"
            ),
        )
        if route_attestation_sha256 != route.route_attestation_sha256:
            raise ValueError(f"multitask route {route.route_id!r} attestation drifted")
        if route_provenance.get("source_schedule_sha256") != source_schedule_sha256:
            raise ValueError(
                f"multitask route {route.route_id!r} source schedule drifted"
            )
        if route_provenance.get("route_attestation_sha256") != route_attestation_sha256:
            raise ValueError(
                f"multitask route {route.route_id!r} schedule attestation drifted"
            )

    resume = _resume_requested(inputs)
    target_optimizer_updates = optimizer_updates
    target_episode_count = scheduled_episode_count
    resume_budget: dict[str, Any] | None = None
    if resume:
        if inputs.mode != "formal":
            raise ValueError("checkpoint resume is supported only for formal launches")
        assert inputs.resume_start_update is not None
        assert inputs.resume_target_update is not None
        assert inputs.resume_sampler_samples_yielded is not None
        resume_start_update = _require_positive_int(
            inputs.resume_start_update, field="resume start update"
        )
        target_optimizer_updates = _require_positive_int(
            inputs.resume_target_update, field="resume target update"
        )
        sampler_samples_yielded = _require_positive_int(
            inputs.resume_sampler_samples_yielded,
            field="resume sampler samples_yielded",
        )
        if not resume_start_update < target_optimizer_updates <= optimizer_updates:
            raise ValueError(
                "resume update range must satisfy 0 < start < target <= schedule capacity"
            )
        invocation_optimizer_updates = target_optimizer_updates - resume_start_update
        target_episode_count = target_optimizer_updates * samples_per_update
        invocation_episodes = invocation_optimizer_updates * samples_per_update
        remaining_schedule_capacity = scheduled_episode_count - sampler_samples_yielded
        if remaining_schedule_capacity < invocation_episodes:
            raise ValueError(
                "resume schedule has insufficient rows after the saved sampler offset: "
                f"{remaining_schedule_capacity} < {invocation_episodes}"
            )
        resume_budget = {
            "schema": "amg_verl_resume_budget_v1",
            "resume_from_path": str(inputs.resume_from_path),
            "resume_prefix_run_dir": str(inputs.resume_prefix_run_dir),
            "resume_start_update": resume_start_update,
            "target_optimizer_updates": target_optimizer_updates,
            "target_episodes": target_episode_count,
            "invocation_optimizer_updates": invocation_optimizer_updates,
            "invocation_episodes": invocation_episodes,
            "sampler_samples_yielded": sampler_samples_yielded,
            "schedule_capacity_optimizer_updates": optimizer_updates,
            "schedule_capacity_episodes": scheduled_episode_count,
        }

    tuning = _ASYNC_TUNING[inputs.mode]
    trigger_parameter_sync_step = tuning["trigger_parameter_sync_step"]
    if target_optimizer_updates % trigger_parameter_sync_step:
        raise ValueError(
            "multitask optimizer updates are not divisible by parameter-sync cadence"
        )
    publication_cycles = target_optimizer_updates // trigger_parameter_sync_step
    budget_contract = {
        "schema": "amg_verl_multitask_budget_contract_v1",
        "mode": inputs.mode,
        "role": role,
        "publication_cycles": publication_cycles,
        "trigger_parameter_sync_step": trigger_parameter_sync_step,
        "optimizer_updates": target_optimizer_updates,
        "samples_per_update": samples_per_update,
        "episodes": target_episode_count,
        "learner_token_budget_profile": inputs.learner_token_budget_profile,
        "actor_train_token_budget": inputs.actor_train_token_budget,
        "critic_train_token_budget": inputs.critic_train_token_budget,
        "save_freq": tuning["save_freq"],
        "max_actor_ckpt_to_keep": tuning["max_actor_ckpt_to_keep"],
        "max_critic_ckpt_to_keep": tuning["max_critic_ckpt_to_keep"],
        "model_path": training_runtime["base_model"],
        "route_ids": list(_MULTITASK_ROUTE_IDS),
        "route_registry_sha256": registry.sha256,
        "schedule_sha256": certificate_schedule_sha256,
        "manifest_sha256": expected_report["manifest_digest"],
        "routing_sha256": certificate_schedule_sha256,
        "schedule_capacity_optimizer_updates": optimizer_updates,
        "schedule_capacity_episodes": scheduled_episode_count,
        "resume": resume_budget,
    }
    return {
        "schema": "amg_multitask_source_identity_v1",
        "source_lock_path": str(source_lock_path),
        "source_lock_sha256": source_lock_sha256,
        "schedule_certificate_path": str(certificate_path),
        "schedule_certificate_sha256": certificate_sha256,
        "publication_outer_commit": publication_outer_commit,
        "publication_inner_commit": publication_inner_commit,
        "verl_commit": locked_verl_commit,
        "route_registry_path": str(registry.source_path),
        "route_registry_sha256": registry.sha256,
        "route_ids": list(registry.route_ids),
        "schedule_count": scheduled_episode_count,
        "schedule_sha256": certificate_schedule_sha256,
        "formal_schedule_contract": dict(certificate),
        "budget_contract": budget_contract,
        "client_config": None,
        "environment": {},
        "selected_files": dict(selected_files),
        "training_runtime": training_runtime,
    }


def _load_endpoint_identity(
    inputs: LaunchInputs, *, schedule_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve one immutable publication without dated task/count/hash literals."""

    paths = (
        (inputs.endpoint_source_lock, "endpoint source lock"),
        (inputs.endpoint_contract_tool, "endpoint contract tool"),
        (inputs.publication_receipt, "publication receipt"),
        (inputs.formal_schedule_certificate, "formal schedule certificate"),
    )
    for path, label in paths:
        _require_regular_file(path, label=label)

    assert inputs.endpoint_source_lock is not None
    assert inputs.endpoint_contract_tool is not None
    assert inputs.publication_receipt is not None
    assert inputs.formal_schedule_certificate is not None

    validated = subprocess.run(
        [
            sys.executable,
            str(inputs.endpoint_contract_tool),
            "validate-lock",
            "--source-lock",
            str(inputs.endpoint_source_lock),
            "--require-final-runtime",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if validated.returncode != 0:
        stderr_tail = "\n".join(validated.stderr.splitlines()[-20:])
        raise RuntimeError(
            "canonical OpenMLE source-lock validation failed: " + stderr_tail
        )
    lines = [line for line in validated.stdout.splitlines() if line.strip()]
    try:
        validation = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "canonical OpenMLE source-lock validator returned no JSON receipt"
        ) from exc
    if (
        not isinstance(validation, Mapping)
        or validation.get("status") != "pass"
        or validation.get("runtime_final") is not True
        or not str(validation.get("schema", "")).startswith(
            "openmle_fast_launcher_source_lock_validation_v"
        )
    ):
        raise RuntimeError(
            f"canonical OpenMLE source-lock validator did not pass: {validation!r}"
        )

    source_lock = _load_json_mapping(
        inputs.endpoint_source_lock, label="endpoint source lock"
    )
    publication = _load_json_mapping(
        inputs.publication_receipt, label="publication receipt"
    )
    schedule_certificate = _load_json_mapping(
        inputs.formal_schedule_certificate, label="formal schedule certificate"
    )
    if source_lock.get("schema") != _ENDPOINT_SOURCE_LOCK_SCHEMA:
        raise ValueError("endpoint source lock schema drifted")
    publication_schema = str(publication.get("schema", ""))
    if (
        publication.get("status") != "pass"
        or not publication_schema.startswith("openmle_")
        or "_publication_receipt_v" not in publication_schema
    ):
        raise ValueError("OpenMLE publication receipt is not a completed publication")
    gates = publication.get("gates")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or any(
            not isinstance(gate, Mapping) or gate.get("status") != "pass"
            for gate in gates.values()
        )
    ):
        raise ValueError("OpenMLE publication receipt has an incomplete gate")

    source_lock_sha256 = _sha256(inputs.endpoint_source_lock)
    schedule_certificate_sha256 = _sha256(inputs.formal_schedule_certificate)
    certified_formal_schedule_sha256 = _require_sha256(
        schedule_certificate.get("output_sha256"),
        field="formal schedule output_sha256",
    )
    artifacts = publication.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("OpenMLE publication receipt omitted artifact bindings")
    expected_artifacts = {
        "source-lock.json": source_lock_sha256,
        "formal100-schedule-certificate.json": schedule_certificate_sha256,
        "formal100-schedule-preview.jsonl": certified_formal_schedule_sha256,
    }
    for name, expected_sha256 in expected_artifacts.items():
        binding = artifacts.get(name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"OpenMLE publication receipt omitted {name!r}")
        observed = _require_sha256(
            binding.get("sha256"), field=f"publication.artifacts.{name}.sha256"
        )
        if observed != expected_sha256:
            raise ValueError(
                f"OpenMLE publication artifact {name!r} does not match the selected file"
            )

    if (
        schedule_certificate.get("status") != "pass"
        or schedule_certificate.get("role") != "train_pool"
        or not str(schedule_certificate.get("schema", "")).startswith(
            "openmle_fast_formal100_schedule_certificate_v"
        )
    ):
        raise ValueError("OpenMLE formal schedule certificate did not pass")
    if schedule_certificate.get("source_lock_sha256") != source_lock_sha256:
        raise ValueError("formal schedule certificate is not bound to the source lock")

    runtime_source = source_lock.get("runtime_source")
    exact_runtime = source_lock.get("exact_runtime")
    integration = source_lock.get("integration")
    launch_contracts = source_lock.get("launch_contracts")
    raw_training_runtime = source_lock.get("training_runtime")
    if not all(
        isinstance(value, Mapping)
        for value in (
            runtime_source,
            exact_runtime,
            integration,
            launch_contracts,
            raw_training_runtime,
        )
    ):
        raise ValueError("endpoint source lock omitted runtime identity sections")
    training_runtime = validate_training_runtime_lock(raw_training_runtime)
    publication_outer_commit = _require_git_revision(
        runtime_source.get("outer_commit"), field="runtime_source.outer_commit"
    )
    publication_inner_commit = _require_git_revision(
        runtime_source.get("inner_commit"), field="runtime_source.inner_commit"
    )

    role = "gate_only" if inputs.mode == "gate" else "train_pool"
    contract_name = "gate1" if inputs.mode == "gate" else "formal100"
    manifests = integration.get("manifests")
    routings = integration.get("routing")
    manifest = manifests.get(role) if isinstance(manifests, Mapping) else None
    routing = routings.get(role) if isinstance(routings, Mapping) else None
    launch_contract = launch_contracts.get(contract_name)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("role") != role
        or not isinstance(routing, Mapping)
        or routing.get("role") != role
        or not isinstance(launch_contract, Mapping)
    ):
        raise ValueError(f"endpoint source lock omitted the {role!r} launch contract")

    manifest_sha256 = _require_sha256(
        manifest.get("sha256"), field=f"integration.manifests.{role}.sha256"
    )
    routing_sha256 = _require_sha256(
        routing.get("sha256"), field=f"integration.routing.{role}.sha256"
    )
    task_count = _require_positive_int(
        manifest.get("task_count"), field=f"{role} task_count"
    )
    source_family_count = _require_positive_int(
        manifest.get("source_family_count"), field=f"{role} source_family_count"
    )
    optimizer_updates = _require_positive_int(
        launch_contract.get("optimizer_updates"),
        field=f"launch_contracts.{contract_name}.optimizer_updates",
    )
    samples_per_update = _require_positive_int(
        launch_contract.get("train_batch_size"),
        field=f"launch_contracts.{contract_name}.train_batch_size",
    )
    if source_family_count > task_count:
        raise ValueError(f"{role} source-family count exceeds task count")
    for key, expected in {
        "manifest_role": role,
        "routing_role": role,
        "task_count": task_count,
        "source_family_count": source_family_count,
    }.items():
        if launch_contract.get(key) != expected:
            raise ValueError(
                f"source-lock launch contract {contract_name}.{key} drifted: "
                f"{launch_contract.get(key)!r} != {expected!r}"
            )

    if inputs.mode == "gate":
        expected_schedule_count = _require_positive_int(
            routing.get("row_count"), field="gate routing row_count"
        )
        expected_schedule_sha256 = routing_sha256
    else:
        scheduled_episode_count = _require_positive_int(
            launch_contract.get("scheduled_episode_count"),
            field="formal launch scheduled_episode_count",
        )
        certificate_expectations = {
            "manifest_sha256": manifest_sha256,
            "source_routing_sha256": routing_sha256,
            "task_count": task_count,
            "source_family_count": source_family_count,
            "optimizer_updates": optimizer_updates,
            "scheduled_episode_count": scheduled_episode_count,
            "output_row_count": scheduled_episode_count,
            "train_batch_size": samples_per_update,
        }
        for key, expected in certificate_expectations.items():
            if schedule_certificate.get(key) != expected:
                raise ValueError(
                    f"formal schedule certificate {key} drifted: "
                    f"{schedule_certificate.get(key)!r} != {expected!r}"
                )
        expected_schedule_count = scheduled_episode_count
        expected_schedule_sha256 = certified_formal_schedule_sha256

    if expected_schedule_count != optimizer_updates * samples_per_update:
        raise ValueError(
            "publication budget is not conserved: schedule episodes != "
            "optimizer_updates * train_batch_size"
        )
    tuning = _ASYNC_TUNING[inputs.mode]
    trigger_parameter_sync_step = tuning["trigger_parameter_sync_step"]
    if optimizer_updates % trigger_parameter_sync_step:
        raise ValueError(
            "publication optimizer updates are not divisible by the reviewed "
            "parameter-sync cadence"
        )
    publication_cycles = optimizer_updates // trigger_parameter_sync_step

    if schedule_report.get("role") != role:
        raise ValueError(
            f"training schedule role mismatch: {schedule_report.get('role')!r} != {role!r}"
        )
    if schedule_report.get("manifest_digest") != manifest_sha256:
        raise ValueError(
            "training schedule manifest does not match the mode-specific endpoint lock"
        )
    if schedule_report.get("count") != expected_schedule_count:
        raise ValueError(
            "training schedule row count does not match the selected launch contract: "
            f"{schedule_report.get('count')} != {expected_schedule_count}"
        )
    if schedule_report.get("sha256") != expected_schedule_sha256:
        raise ValueError(
            "training schedule digest does not match the selected immutable publication"
        )

    derived_contract = publication.get("derived_contract")
    if not isinstance(derived_contract, Mapping):
        raise ValueError("OpenMLE publication receipt omitted its derived contract")
    publication_schedule = derived_contract.get("schedule")
    train_manifest = (
        manifests.get("train_pool") if isinstance(manifests, Mapping) else None
    )
    if not isinstance(publication_schedule, Mapping) or not isinstance(
        train_manifest, Mapping
    ):
        raise ValueError("OpenMLE publication receipt omitted schedule accounting")
    for key, expected in {
        "train_task_count": _require_positive_int(
            train_manifest.get("task_count"), field="train task_count"
        ),
        "train_source_family_count": _require_positive_int(
            train_manifest.get("source_family_count"),
            field="train source_family_count",
        ),
        "schedule_output_sha256": certified_formal_schedule_sha256,
    }.items():
        if derived_contract.get(key) != expected:
            raise ValueError(
                f"OpenMLE publication derived contract {key} drifted: "
                f"{derived_contract.get(key)!r} != {expected!r}"
            )
    for key in (
        "scheduled_episode_count",
        "optimizer_updates",
        "minimum_task_reuse",
        "maximum_task_reuse",
        "partial_repetition_task_count",
    ):
        if publication_schedule.get(key) != schedule_certificate.get(key):
            raise ValueError(f"OpenMLE publication/certificate schedule {key} drifted")

    selected_files = runtime_source.get("selected_files")
    if not isinstance(selected_files, Mapping) or not selected_files:
        raise ValueError("endpoint source lock omitted selected file hashes")
    client_config: dict[str, str | int] = {
        "expected_manifest_sha256": manifest_sha256,
        "expected_release_revision": _require_git_revision(
            runtime_source.get("openmle_tasks_revision"),
            field="runtime_source.openmle_tasks_revision",
        ),
        "expected_outer_commit": publication_outer_commit,
        "expected_inner_commit": publication_inner_commit,
        "expected_role": role,
        "expected_executor_runtime_digest": _require_runtime_digest(
            exact_runtime.get("runtime_digest")
        ),
        "expected_materializer_sha256": _require_sha256(
            selected_files.get(
                "inner:agentenv-openmle-fast/agentenv_openmle_fast/materializer.py"
            ),
            field="runtime_source.selected_files.materializer",
        ),
        "expected_actions_sha256": _require_sha256(
            selected_files.get(
                "inner:agentenv-openmle-fast/agentenv_openmle_fast/actions.py"
            ),
            field="runtime_source.selected_files.actions",
        ),
        "expected_max_observation_tokens": _MAX_OBSERVATION_TOKENS,
    }
    environment = {
        _ENDPOINT_ENV_FIELDS[key]: str(value) for key, value in client_config.items()
    }
    budget_contract = {
        "schema": "amg_verl_publication_budget_contract_v1",
        "mode": inputs.mode,
        "role": role,
        "publication_cycles": publication_cycles,
        "trigger_parameter_sync_step": trigger_parameter_sync_step,
        "optimizer_updates": optimizer_updates,
        "samples_per_update": samples_per_update,
        "episodes": expected_schedule_count,
        "learner_token_budget_profile": inputs.learner_token_budget_profile,
        "actor_train_token_budget": inputs.actor_train_token_budget,
        "critic_train_token_budget": inputs.critic_train_token_budget,
        "save_freq": tuning["save_freq"],
        "max_actor_ckpt_to_keep": tuning["max_actor_ckpt_to_keep"],
        "max_critic_ckpt_to_keep": tuning["max_critic_ckpt_to_keep"],
        "model_path": training_runtime["base_model"],
        "task_count": task_count,
        "source_family_count": source_family_count,
        "schedule_sha256": expected_schedule_sha256,
        "manifest_sha256": manifest_sha256,
        "routing_sha256": routing_sha256,
    }
    return {
        "schema": "amg_openmle_publication_identity_v3",
        "source_lock_path": str(inputs.endpoint_source_lock),
        "source_lock_sha256": source_lock_sha256,
        "contract_tool_path": str(inputs.endpoint_contract_tool),
        "contract_tool_sha256": _sha256(inputs.endpoint_contract_tool),
        "publication_receipt_path": str(inputs.publication_receipt),
        "publication_receipt_sha256": _sha256(inputs.publication_receipt),
        "schedule_certificate_path": str(inputs.formal_schedule_certificate),
        "schedule_certificate_sha256": schedule_certificate_sha256,
        "canonical_validation": dict(validation),
        "publication_outer_commit": publication_outer_commit,
        "publication_inner_commit": publication_inner_commit,
        "manifest_role": role,
        "manifest_sha256": manifest_sha256,
        "routing_sha256": routing_sha256,
        "task_count": task_count,
        "source_family_count": source_family_count,
        "schedule_count": expected_schedule_count,
        "schedule_sha256": expected_schedule_sha256,
        "launch_contract": dict(launch_contract),
        "formal_schedule_contract": dict(schedule_certificate),
        "budget_contract": budget_contract,
        "client_config": client_config,
        "environment": environment,
        "selected_files": dict(selected_files),
        "training_runtime": training_runtime,
    }


def _load_launch_identity(
    inputs: LaunchInputs, *, schedule_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Select the generic multitask or strict legacy OpenMLE identity path."""

    has_registry = (
        inputs.route_registry is not None or inputs.route_registry_sha256 is not None
    )
    has_multitask_lock = (
        inputs.multitask_source_lock is not None
        or inputs.multitask_schedule_certificate is not None
    )
    if has_registry or has_multitask_lock:
        return _load_multitask_identity(inputs, schedule_report=schedule_report)
    return _load_endpoint_identity(inputs, schedule_report=schedule_report)


def _preflight_regular_file(value: Any, *, label: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return _require_regular_file(path, label=label).resolve()


def _verify_preflight_file(
    value: Any,
    digest: Any,
    *,
    label: str,
) -> Path:
    path = _preflight_regular_file(value, label=label)
    expected = _require_sha256(digest, field=f"{label} sha256")
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} sha256 mismatch: expected {expected}, got {observed}"
        )
    return path


def _load_multitask_orchestrator_preflight(
    inputs: LaunchInputs,
    *,
    launch_identity: Mapping[str, Any],
    schedule_report: Mapping[str, Any],
    budget_contract: Mapping[str, Any],
    required: bool,
) -> dict[str, Any] | None:
    """Validate the live endpoint/holder handoff before a multitask trainer."""

    path = inputs.multitask_orchestrator_preflight
    is_multitask = inputs.route_registry is not None
    if not is_multitask:
        if path is not None:
            raise ValueError(
                "single-environment launch must not set a multitask "
                "orchestrator preflight"
            )
        return None
    if path is None:
        if required:
            raise ValueError(
                "full multitask launch requires --multitask-orchestrator-preflight"
            )
        return None

    path = _require_regular_file(path, label="multitask orchestrator preflight")
    expected_path = (inputs.run_dir / "orchestrator-preflight.json").resolve()
    if path.resolve() != expected_path:
        raise ValueError(
            "multitask orchestrator preflight must be the current run's "
            f"receipt: {expected_path}"
        )
    receipt = _load_json_mapping(path, label="multitask orchestrator preflight")
    if (
        receipt.get("schema") != _MULTITASK_ORCHESTRATOR_PREFLIGHT_SCHEMA
        or receipt.get("status") != "pass"
    ):
        raise ValueError(
            "multitask orchestrator preflight is not a completed v1 receipt"
        )

    assert inputs.route_registry is not None
    assert inputs.multitask_source_lock is not None
    assert inputs.multitask_schedule_certificate is not None
    expected_values = {
        "route_registry_path": str(inputs.route_registry.resolve()),
        "route_registry_sha256": inputs.route_registry_sha256,
        "route_order": list(_MULTITASK_ROUTE_IDS),
        "schedule_path": str(inputs.schedule.resolve()),
        "schedule_sha256": schedule_report.get("sha256"),
        "schedule_count": schedule_report.get("count"),
        "multitask_source_lock_path": str(inputs.multitask_source_lock.resolve()),
        "multitask_source_lock_sha256": launch_identity.get("source_lock_sha256"),
        "multitask_schedule_certificate_path": str(
            inputs.multitask_schedule_certificate.resolve()
        ),
        "multitask_schedule_certificate_sha256": launch_identity.get(
            "schedule_certificate_sha256"
        ),
        "budget": {
            "optimizer_updates": budget_contract.get("optimizer_updates"),
            "samples_per_update": budget_contract.get("samples_per_update"),
            "episodes": budget_contract.get("episodes"),
        },
    }
    for field, expected in expected_values.items():
        observed = receipt.get(field)
        if observed != expected:
            raise RuntimeError(
                f"multitask orchestrator preflight {field} mismatch: "
                f"{observed!r} != {expected!r}"
            )

    _verify_preflight_file(
        receipt.get("config_path"),
        receipt.get("config_sha256"),
        label="multitask orchestrator config",
    )
    _verify_preflight_file(
        receipt.get("endpoint_registry_path"),
        receipt.get("endpoint_registry_sha256"),
        label="multitask endpoint registry",
    )

    holder = receipt.get("holder_transaction")
    if not isinstance(holder, Mapping) or holder.get("status") != "acquired":
        raise RuntimeError(
            "multitask orchestrator preflight holder transaction is not acquired"
        )
    _verify_preflight_file(
        holder.get("lease_path"),
        holder.get("lease_sha256"),
        label="multitask holder lease",
    )
    holder_state_path = _preflight_regular_file(
        holder.get("state_path"), label="multitask holder transaction state"
    )
    if (
        holder_state_path
        != (inputs.run_dir / "holder-transaction" / "state.json").resolve()
    ):
        raise RuntimeError(
            "multitask holder transaction state is outside the current run"
        )
    holder_state = _load_json_mapping(
        holder_state_path, label="multitask holder transaction state"
    )
    if (
        holder_state.get("schema") != "amg_marker_transaction_v1"
        or holder_state.get("status") != "acquired"
        or holder_state.get("run_id") != inputs.experiment_name
    ):
        raise RuntimeError("multitask holder transaction is not actively acquired")
    holder_parent = holder_state.get("parent")
    if not isinstance(holder_parent, Mapping):
        raise RuntimeError("multitask holder transaction omitted parent identity")
    holder_parent_pid = holder_parent.get("pid")
    holder_parent_ticks = holder_parent.get("start_ticks")
    if (
        isinstance(holder_parent_pid, bool)
        or not isinstance(holder_parent_pid, int)
        or not isinstance(holder_parent_ticks, str)
        or not process_identity_alive(holder_parent_pid, holder_parent_ticks)
    ):
        raise RuntimeError(
            "multitask holder transaction parent PID/start-ticks is not alive"
        )
    watcher_pid = holder.get("watcher_pid")
    watcher_start_ticks = holder.get("watcher_start_ticks")
    if (
        isinstance(watcher_pid, bool)
        or not isinstance(watcher_pid, int)
        or not isinstance(watcher_start_ticks, str)
        or not process_identity_alive(watcher_pid, watcher_start_ticks)
    ):
        raise RuntimeError("multitask holder watcher PID/start-ticks is not alive")
    try:
        watcher_process_group = os.getpgid(watcher_pid)
    except ProcessLookupError as exc:
        raise RuntimeError("multitask holder watcher disappeared") from exc
    if watcher_process_group != watcher_pid:
        raise RuntimeError("multitask holder watcher process group drifted")

    registry = load_route_registry(
        inputs.route_registry,
        expected_sha256=str(inputs.route_registry_sha256),
        expected_route_ids=_MULTITASK_ROUTE_IDS,
    )
    endpoint_receipts = receipt.get("endpoints")
    if (
        isinstance(endpoint_receipts, (str, bytes))
        or not isinstance(endpoint_receipts, Sequence)
        or len(endpoint_receipts) != len(_MULTITASK_ROUTE_IDS)
    ):
        raise RuntimeError(
            "multitask orchestrator preflight must contain exactly four endpoints"
        )
    normalized_endpoints: list[dict[str, Any]] = []
    for index, (route, endpoint_receipt) in enumerate(
        zip(registry.routes, endpoint_receipts)
    ):
        if not isinstance(endpoint_receipt, Mapping):
            raise TypeError(
                f"multitask endpoint preflight entry {index} must be a mapping"
            )
        expected_endpoint = str(route.client_config["env_addr"])
        for field, expected in {
            "route_id": route.route_id,
            "route_attestation_sha256": route.route_attestation_sha256,
            "endpoint": expected_endpoint,
        }.items():
            observed = endpoint_receipt.get(field)
            if observed != expected:
                raise RuntimeError(
                    "multitask endpoint preflight "
                    f"{route.route_id} {field} mismatch: "
                    f"{observed!r} != {expected!r}"
                )
        for prefix in ("gate_receipt", "launcher", "metadata"):
            artifact_path = _verify_preflight_file(
                endpoint_receipt.get(f"{prefix}_path"),
                endpoint_receipt.get(f"{prefix}_sha256"),
                label=f"{route.route_id} endpoint {prefix}",
            )
            if prefix == "metadata":
                expected_metadata_path = (
                    inputs.run_dir / "endpoints" / route.route_id / "metadata.json"
                ).resolve()
                if artifact_path != expected_metadata_path:
                    raise RuntimeError(
                        f"multitask endpoint {route.route_id} metadata path "
                        "is outside the current run"
                    )
        pid = endpoint_receipt.get("pid")
        start_ticks = endpoint_receipt.get("start_ticks")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or not isinstance(start_ticks, str)
            or not process_identity_alive(pid, start_ticks)
        ):
            raise RuntimeError(
                f"multitask endpoint {route.route_id} PID/start-ticks is not alive"
            )
        try:
            process_group = os.getpgid(pid)
        except ProcessLookupError as exc:
            raise RuntimeError(
                f"multitask endpoint {route.route_id} process disappeared"
            ) from exc
        if process_group != pid:
            raise RuntimeError(
                f"multitask endpoint {route.route_id} process group drifted"
            )
        normalized_endpoints.append(dict(endpoint_receipt))

    normalized = dict(receipt)
    normalized["path"] = str(path.resolve())
    normalized["sha256"] = _sha256(path)
    normalized["endpoints"] = normalized_endpoints
    return normalized


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _partition_selected_file_hashes(
    selected_files: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Map source-lock identities to their repository-relative runtime paths."""

    outer_manifest: dict[str, str] = {}
    inner_manifest: dict[str, str] = {}
    for identity_path, digest in selected_files.items():
        if not isinstance(identity_path, str) or not isinstance(digest, str):
            raise TypeError("selected runtime file manifest is malformed")
        if identity_path.startswith("inner:"):
            inner_manifest[identity_path.removeprefix("inner:")] = digest
        elif identity_path.startswith("outer:AgentGym-RL/"):
            # ``outer_root`` is the checkout root. AgentGym-RL is a tracked
            # subdirectory in that repository, so preserve it in the relative
            # path instead of treating it as a display-only identity prefix.
            outer_manifest[identity_path.removeprefix("outer:")] = digest
        elif identity_path.startswith("outer:"):
            outer_manifest[identity_path.removeprefix("outer:")] = digest
        else:
            raise RuntimeError(
                f"unsupported selected runtime file identity: {identity_path!r}"
            )
    return outer_manifest, inner_manifest


def _verify_source(
    inputs: LaunchInputs,
    *,
    require_outer_clean: bool,
    launch_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not (inputs.verl_root / "verl" / "experimental" / "fully_async_policy").is_dir():
        raise FileNotFoundError(f"not a veRL source tree: {inputs.verl_root}")
    verl_commit = _git(inputs.verl_root, "rev-parse", "HEAD")
    if verl_commit != EXPECTED_VERL_COMMIT:
        raise RuntimeError(
            f"veRL commit mismatch: expected {EXPECTED_VERL_COMMIT}, got {verl_commit}"
        )
    locked_verl_commit = launch_identity.get("verl_commit")
    if locked_verl_commit is not None and locked_verl_commit != verl_commit:
        raise RuntimeError(
            "launch source lock veRL commit mismatch: "
            f"expected {locked_verl_commit}, got {verl_commit}"
        )
    verl_status = _git(inputs.verl_root, "status", "--porcelain")
    if verl_status:
        raise RuntimeError("veRL runtime tree must be clean after the reviewed commit")

    publication_outer_commit = _require_git_revision(
        launch_identity.get("publication_outer_commit"),
        field="launch_identity.publication_outer_commit",
    )
    publication_inner_commit = _require_git_revision(
        launch_identity.get("publication_inner_commit"),
        field="launch_identity.publication_inner_commit",
    )
    outer_commit = _git(inputs.outer_root, "rev-parse", "HEAD")
    _require_exact_multitask_outer_commit(
        launch_identity_schema=launch_identity.get("schema"),
        publication_outer_commit=publication_outer_commit,
        observed_outer_commit=outer_commit,
    )
    ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(inputs.outer_root),
                "merge-base",
                "--is-ancestor",
                publication_outer_commit,
                outer_commit,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    committed_paths = tuple(
        line
        for line in _git(
            inputs.outer_root,
            "diff",
            "--name-only",
            f"{publication_outer_commit}..{outer_commit}",
        ).splitlines()
        if line
    )
    dirty_paths = sorted(
        {
            line
            for arguments in (
                ("diff", "--name-only"),
                ("diff", "--cached", "--name-only"),
                ("ls-files", "--others", "--exclude-standard"),
            )
            for line in _git(inputs.outer_root, *arguments).splitlines()
            if line
        }
    )
    outer_changes = validate_outer_change_paths(
        locked_outer_commit=publication_outer_commit,
        ancestor_is_locked=ancestor,
        committed_paths=committed_paths,
        dirty_paths=dirty_paths,
        require_clean=require_outer_clean,
    )

    agentgym_root = inputs.outer_root / "AgentGym"
    if not agentgym_root.is_dir():
        raise FileNotFoundError(
            f"AgentGym submodule directory missing: {agentgym_root}"
        )
    expected_agentgym_commit = _git(inputs.outer_root, "rev-parse", "HEAD:AgentGym")
    try:
        agentgym_commit = _git(agentgym_root, "rev-parse", "HEAD")
        agentgym_status = _git(agentgym_root, "status", "--porcelain")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "AgentGym must be an initialized git submodule, not a copied source tree"
        ) from exc
    if expected_agentgym_commit != publication_inner_commit:
        raise RuntimeError(
            "AgentGym gitlink identity mismatch: "
            f"expected {publication_inner_commit}, got {expected_agentgym_commit}"
        )
    if agentgym_commit != publication_inner_commit:
        raise RuntimeError(
            "AgentGym submodule commit mismatch: "
            f"expected {publication_inner_commit}, got {agentgym_commit}"
        )
    if agentgym_status:
        raise RuntimeError("AgentGym runtime submodule must be clean")

    selected_files = launch_identity.get("selected_files")
    if not isinstance(selected_files, Mapping) or not selected_files:
        raise RuntimeError("selected launch identity omitted runtime file hashes")
    outer_manifest, inner_manifest = _partition_selected_file_hashes(selected_files)
    verified_outer_files = verify_hash_manifest(inputs.outer_root, outer_manifest)
    verified_inner_files = verify_hash_manifest(agentgym_root, inner_manifest)

    raw_training_runtime = launch_identity.get("training_runtime")
    training_runtime = validate_training_runtime_lock(raw_training_runtime)
    runtime_python = Path(training_runtime["python"])
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        raise FileNotFoundError(
            f"publication training Python is missing: {runtime_python}"
        )
    if Path(sys.executable).resolve() != runtime_python.resolve():
        raise RuntimeError(
            f"launcher Python identity mismatch: {sys.executable} != {runtime_python}"
        )
    site_packages = Path(training_runtime["site_packages"])
    if not site_packages.is_dir():
        raise FileNotFoundError(
            f"publication training site-packages is missing: {site_packages}"
        )
    bundle_sha256_file = Path(training_runtime["bundle_sha256_file"])
    if not bundle_sha256_file.is_file() or bundle_sha256_file.is_symlink():
        raise FileNotFoundError(
            f"publication runtime bundle sha256 file is missing: {bundle_sha256_file}"
        )
    bundle_tokens = bundle_sha256_file.read_text(encoding="utf-8").split()
    if not bundle_tokens or bundle_tokens[0] != training_runtime["bundle_sha256"]:
        raise RuntimeError("publication runtime bundle sha256 file content mismatch")
    trl_wheel = verify_hash_manifest(
        inputs.outer_root,
        {TRL_WHEEL_RELATIVE_PATH: TRL_WHEEL_SHA256},
    )
    model_path = Path(training_runtime["base_model"])
    model_files = verify_hash_manifest(model_path, LOCKED_MODEL_FILE_SHA256)

    return {
        "verl_commit": verl_commit,
        "verl_clean": not bool(verl_status),
        "publication_outer_commit": publication_outer_commit,
        "outer_commit": outer_commit,
        "outer_diff_paths": list(committed_paths),
        "outer_clean": not bool(dirty_paths),
        "outer_change_proof": outer_changes,
        "agentgym_commit": agentgym_commit,
        "agentgym_expected_commit": expected_agentgym_commit,
        "agentgym_clean": not bool(agentgym_status),
        "selected_outer_files_sha256": verified_outer_files,
        "selected_inner_files_sha256": verified_inner_files,
        "training_runtime": training_runtime,
        "trl_wheel_sha256": trl_wheel,
        "model_files_sha256": model_files,
    }


def _production_manifest(outer_root: Path) -> dict[str, str]:
    roots = [
        outer_root / "async_plugins" / "agentmemorygym_verl",
        outer_root / "async_plugins" / "config",
        outer_root / "async_plugins" / "scripts",
        outer_root / "async_plugins" / "vendor",
    ]
    manifest: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            manifest[str(path.relative_to(outer_root))] = _sha256(path)
    return manifest


def _load_yaml(text: str) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - veRL runtime owns PyYAML
        raise RuntimeError(
            "PyYAML is required to inspect Hydra's resolved config"
        ) from exc
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise TypeError("Hydra --cfg job did not produce a config mapping")
    return value


def _resolved_command(overrides: list[str], *, python: str) -> list[str]:
    return [
        python,
        "-m",
        _UPSTREAM_ENTRYPOINT,
        "--cfg",
        "job",
        "--resolve",
        *overrides,
    ]


def _training_command(overrides: list[str], *, python: str) -> list[str]:
    return [
        python,
        "-X",
        "faulthandler",
        "-m",
        _UPSTREAM_ENTRYPOINT,
        *overrides,
    ]


def _validate_accelerator_runtime(
    observed: Mapping[str, Any], *, training_runtime: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the launch to the publication GPU identity and CUDA 13 toolchain."""

    expected_gpu_count = _require_positive_int(
        training_runtime.get("gpu_count"), field="training runtime gpu_count"
    )
    expected_gpu_type = _string(training_runtime.get("gpu_type", ""))
    gpu_count = observed.get("gpu_count")
    gpu_names = observed.get("gpu_names")
    if gpu_count != expected_gpu_count:
        raise RuntimeError(
            f"runtime GPU count {gpu_count!r} does not match publication {expected_gpu_count}"
        )
    if (
        not isinstance(gpu_names, Sequence)
        or isinstance(gpu_names, (str, bytes))
        or len(gpu_names) != expected_gpu_count
        or any(
            not isinstance(name, str) or expected_gpu_type not in name
            for name in gpu_names
        )
    ):
        raise RuntimeError(
            f"runtime GPUs do not match publication type {expected_gpu_type}: {gpu_names!r}"
        )
    if observed.get("torch_cuda_available") is not True:
        raise RuntimeError("publication runtime reports CUDA unavailable")
    if observed.get("torch_cuda") != _EXPECTED_CUDA_VERSION:
        raise RuntimeError(
            "PyTorch CUDA build does not match the locked CUDA 13 runtime: "
            f"{observed.get('torch_cuda')!r}"
        )
    if observed.get("nvcc_release") != _EXPECTED_CUDA_VERSION:
        raise RuntimeError(
            "nvcc does not match the locked CUDA 13 runtime: "
            f"{observed.get('nvcc_release')!r}"
        )
    if observed.get("cuda_home") != str(_CUDA13_TOOLKIT_ROOT):
        raise RuntimeError(
            f"CUDA_HOME must be {_CUDA13_TOOLKIT_ROOT}, got {observed.get('cuda_home')!r}"
        )
    if observed.get("cudart_linker_ready") is not True:
        raise RuntimeError("CUDA 13 toolkit has no usable libcudart linker name")
    if observed.get("cccl_target_ready") is not True:
        raise RuntimeError("CUDA 13 JIT runtime has no usable CCCL nv/target header")
    normalized = dict(observed)
    normalized["gpu_names"] = list(gpu_names)
    return normalized


def _runtime_preflight(
    inputs: LaunchInputs,
    env: dict[str, str],
    *,
    model_path: Path,
    training_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"publication model directory missing: {model_path}")
    code = r"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import torch
import trl
from transformers import AutoConfig, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from agentmemorygym_verl.active_source_audit import audit_resolved_active_sources
# Match veRL's real entrypoint: external estimator registration must occur
# before the runtime probe asks the upstream registry for the AMG estimator.
from agentmemorygym_verl import action_gae as _amg_action_gae
from agentmemorygym_verl.agent_loop import AMGTaskNeutralAgentLoop
from agentmemorygym_verl.dataset import AMGTrajectoryDataset
from agentmemorygym_verl.env_client import create_env_client
from agentmemorygym_verl.routes import (
    canonical_policy_framing_sha256,
    load_route_registry,
)
from verl.trainer.ppo.core_algos import get_adv_estimator_fn
from verl.utils.tokenizer.continuous_token_wiring import (
    create_continuous_token_builder,
    infer_continuous_token_model_family,
)

fn = get_adv_estimator_fn("amg_action_axis_gae")
model_path = sys.argv[1]
registry_path = sys.argv[2]
registry_sha256 = sys.argv[3]
verl_root = Path(sys.argv[4])
outer_root = Path(sys.argv[5])
hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
continuous_token_family = infer_continuous_token_model_family(
    hf_model_type=getattr(hf_config, "model_type", None),
)
continuous_token_builder = create_continuous_token_builder(
    tokenizer,
    hf_model_type=getattr(hf_config, "model_type", None),
)
if str(continuous_token_family) != "qwen35":
    raise RuntimeError(
        "publication model must resolve to veRL native Qwen3.5 Continuous Token, "
        f"got {continuous_token_family!s}"
    )
route_receipts = []
if registry_path:
    registry = load_route_registry(
        registry_path,
        expected_sha256=registry_sha256,
    )
    for route in registry.routes:
        client = create_env_client(dict(route.client_config))
        try:
            framing = client.policy_framing()
        finally:
            client.close()
        if not isinstance(framing, (list, tuple)) or not framing:
            raise RuntimeError(
                f"AMG route {route.route_id!r} returned empty policy framing"
            )
        framing_sha256 = canonical_policy_framing_sha256(framing)
        if framing_sha256 != route.policy_framing_sha256:
            raise RuntimeError(
                f"AMG route {route.route_id!r} policy framing digest drifted"
            )
        route_receipts.append({
            "route_id": route.route_id,
            "task_name": str(route.client_config["task_name"]),
            "env_addr": str(route.client_config["env_addr"]),
            "policy_framing_messages": len(framing),
            "policy_framing_sha256": framing_sha256,
            "route_attestation_sha256": route.route_attestation_sha256,
        })
else:
    client_config = json.loads(os.environ["AMG_ENDPOINT_CLIENT_CONFIG_JSON"])
    client = create_env_client({
        "task_name": "openmle_fast",
        "env_addr": os.environ["AMG_ENV_ADDR"],
        "timeout": 240,
        "max_retries": 2,
        **client_config,
    })
    try:
        framing = client.policy_framing()
    finally:
        client.close()
    if not isinstance(framing, (list, tuple)) or not framing:
        raise RuntimeError("AMG endpoint returned empty policy framing")
    route_receipts.append({
        "route_id": "openmle_fast",
        "task_name": "openmle_fast",
        "env_addr": os.environ["AMG_ENV_ADDR"],
        "policy_framing_messages": len(framing),
        "policy_framing_sha256": canonical_policy_framing_sha256(framing),
        "route_attestation_sha256": None,
    })
ninja_path = shutil.which("ninja")
if ninja_path is None:
    raise RuntimeError("publication runtime PATH does not provide ninja")
nvcc_path = shutil.which("nvcc")
if nvcc_path is None:
    raise RuntimeError("publication runtime PATH does not provide nvcc")
nvcc_output = subprocess.check_output([nvcc_path, "--version"], text=True)
nvcc_match = re.search(r"release\s+(\d+\.\d+)", nvcc_output)
if nvcc_match is None:
    raise RuntimeError("cannot parse nvcc release")
nvidia_smi = shutil.which("nvidia-smi")
if nvidia_smi is None:
    raise RuntimeError("publication runtime PATH does not provide nvidia-smi")
gpu_names = [
    line.strip()
    for line in subprocess.check_output(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if line.strip()
]
active_source_ast_audit = audit_resolved_active_sources(
    verl_root=verl_root,
    outer_root=outer_root,
)
cuda_home = Path(os.environ.get("CUDA_HOME", ""))
cudart_linker = cuda_home / "lib64" / "libcudart.so"
cccl_include = Path(os.environ.get("CPATH", "").split(os.pathsep)[0])
cccl_target = cccl_include / "nv" / "target"
print(json.dumps({
    "adv_estimator": fn.__name__,
    "agent_loop": AMGTaskNeutralAgentLoop.__name__,
    "dataset": AMGTrajectoryDataset.__name__,
    "routes": route_receipts,
    "ninja_path": ninja_path,
    "cuda_home": os.environ.get("CUDA_HOME"),
    "cudart_linker_path": str(cudart_linker),
    "cudart_linker_ready": cudart_linker.is_file(),
    "cccl_target_path": str(cccl_target),
    "cccl_target_ready": cccl_target.is_file(),
    "nvcc_path": nvcc_path,
    "nvcc_release": nvcc_match.group(1),
    "torch_cuda": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
    "gpu_count": len(gpu_names),
    "gpu_names": gpu_names,
    "trl_version": trl.__version__,
    "value_head_class": AutoModelForCausalLMWithValueHead.__name__,
    "continuous_token_family": str(continuous_token_family),
    "continuous_token_builder": type(continuous_token_builder).__name__,
    "active_source_ast_audit": active_source_ast_audit,
}, sort_keys=True))
"""
    probe_env = dict(env)
    registry_path = ""
    registry_sha256 = ""
    if inputs.route_registry is not None:
        assert inputs.route_registry_sha256 is not None
        registry_path = str(inputs.route_registry.resolve())
        registry_sha256 = inputs.route_registry_sha256
        if (
            "AMG_ENDPOINT_CLIENT_CONFIG_JSON" in probe_env
            or "AMG_ENV_ADDR" in probe_env
        ):
            raise RuntimeError(
                "multitask runtime preflight inherited a global endpoint identity"
            )
    else:
        assert inputs.env_addr is not None
        probe_env["AMG_ENV_ADDR"] = inputs.env_addr
        if "AMG_ENDPOINT_CLIENT_CONFIG_JSON" not in probe_env:
            raise RuntimeError(
                "endpoint client identity was not exported for preflight"
            )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(model_path),
            registry_path,
            registry_sha256,
            str(inputs.verl_root),
            str(inputs.outer_root),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=probe_env,
        cwd=inputs.verl_root,
    )
    if completed.returncode != 0:
        stdout = completed.stdout.strip() or "<empty>"
        stderr = completed.stderr.strip() or "<empty>"
        raise RuntimeError(
            "AMG runtime preflight failed with "
            f"exit code {completed.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("AMG runtime preflight produced no receipt")
    try:
        receipt = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("AMG runtime preflight receipt is not JSON") from exc
    if not isinstance(receipt, Mapping):
        raise RuntimeError("AMG runtime preflight receipt is not an object")
    active_source_audit = receipt.get("active_source_ast_audit")
    if (
        not isinstance(active_source_audit, Mapping)
        or active_source_audit.get("schema") != "amg_runtime_active_source_ast_audit_v1"
        or active_source_audit.get("status") != "pass"
    ):
        raise RuntimeError(
            "AMG runtime preflight omitted a passing active-source AST audit"
        )
    active_roots = active_source_audit.get("roots")
    expected_roots = {
        "verl": str(inputs.verl_root.resolve()),
        "plugin": str((inputs.outer_root / "async_plugins").resolve()),
        "agentgym": str((inputs.outer_root / "AgentGym" / "agentenv").resolve()),
    }
    if not isinstance(active_roots, Mapping) or dict(active_roots) != expected_roots:
        raise RuntimeError(
            "AMG runtime active-source AST audit resolved unexpected roots"
        )
    receipt = _preserve_legacy_runtime_preflight_fields(
        receipt, multitask=inputs.route_registry is not None
    )
    return _validate_accelerator_runtime(receipt, training_runtime=training_runtime)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_resume_checkpoint(
    inputs: LaunchInputs,
    *,
    training_runtime: Mapping[str, Any],
    schedule_report: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Bind one native veRL checkpoint without mutating its dataloader state."""

    if not _resume_requested(inputs):
        return None
    assert inputs.resume_from_path is not None
    assert inputs.resume_prefix_run_dir is not None
    assert inputs.resume_start_update is not None
    assert inputs.resume_target_update is not None
    assert inputs.resume_sampler_samples_yielded is not None

    checkpoint = inputs.resume_from_path.resolve()
    prefix_run = inputs.resume_prefix_run_dir.resolve()
    start = int(inputs.resume_start_update)
    target = int(inputs.resume_target_update)
    expected_checkpoint = prefix_run / "checkpoints" / f"global_step_{start}"
    if checkpoint != expected_checkpoint.resolve():
        raise ValueError(
            f"resume checkpoint must be {expected_checkpoint}, got {checkpoint}"
        )
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise FileNotFoundError(f"resume checkpoint is missing or symlinked: {checkpoint}")
    if prefix_run == inputs.run_dir.resolve():
        raise ValueError("resume prefix run and successor run must be distinct")

    data_path = checkpoint / "data.pt"
    if data_path.is_symlink() or not data_path.is_file() or data_path.stat().st_size <= 0:
        raise FileNotFoundError(f"resume dataloader state is missing: {data_path}")
    world_size = int(inputs.trainer_gpus)
    missing: list[str] = []
    for role in ("actor", "critic"):
        for kind in ("model", "optim", "extra_state"):
            for rank in range(world_size):
                path = checkpoint / role / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
                if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                    missing.append(str(path.relative_to(checkpoint)))
    if missing:
        raise FileNotFoundError(
            "resume checkpoint is incomplete: " + ", ".join(missing)
        )

    runtime_python = str(training_runtime["python"])
    probe = subprocess.run(
        [
            runtime_python,
            "-B",
            "-c",
            (
                "import json,sys,torch; "
                "x=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
                "print(json.dumps({k:x.get(k) for k in "
                "('_sampler_iter_yielded','_num_yielded','_iterator_finished')}))"
            ),
            str(data_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "cannot read native resume dataloader state: "
            + "\n".join(probe.stderr.splitlines()[-20:])
        )
    try:
        sampler_state = json.loads(probe.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("resume dataloader probe returned invalid JSON") from exc
    expected_yielded = int(inputs.resume_sampler_samples_yielded)
    if (
        sampler_state.get("_sampler_iter_yielded") != expected_yielded
        or sampler_state.get("_num_yielded") != expected_yielded
        or sampler_state.get("_iterator_finished") is not False
    ):
        raise RuntimeError(
            "resume dataloader state differs from the declared sampler offset: "
            f"{sampler_state!r}"
        )

    prefix_launch = prefix_run / "launch-receipt.json"
    prefix_metrics = prefix_run / "metrics.jsonl"
    prefix_rollout_dir = prefix_run / "rollout_data"
    for path, label in (
        (prefix_launch, "prefix launch receipt"),
        (prefix_metrics, "prefix FileLogger"),
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{label} is missing or symlinked: {path}")
    prefix_receipt = _load_json_mapping(prefix_launch, label="prefix launch receipt")
    prefix_schedule = prefix_receipt.get("schedule")
    prefix_inputs = prefix_receipt.get("inputs")
    if not isinstance(prefix_schedule, Mapping) or not isinstance(
        prefix_inputs, Mapping
    ):
        raise RuntimeError("resume prefix launch identity is incomplete")
    prefix_schedule_sha256 = prefix_schedule.get("sha256")
    prefix_route_registry_sha256 = prefix_inputs.get("route_registry_sha256")
    successor_schedule_sha256 = schedule_report.get("sha256")
    schedule_changed = prefix_schedule_sha256 != successor_schedule_sha256
    registry_changed = prefix_route_registry_sha256 != inputs.route_registry_sha256
    provenance_rebind = None
    if schedule_changed or registry_changed:
        if inputs.resume_provenance_rebind is None:
            raise RuntimeError(
                "resume prefix provenance differs without an explicit rebind receipt"
            )
        prefix_schedule_path = Path(str(prefix_schedule.get("path", "")))
        prefix_registry_path = Path(str(prefix_inputs.get("route_registry", "")))
        if inputs.route_registry is None:
            raise RuntimeError("resume provenance rebind requires a route registry")
        provenance_rebind = validate_resume_provenance_rebind(
            inputs.resume_provenance_rebind,
            prefix_schedule_path=prefix_schedule_path,
            successor_schedule_path=inputs.schedule,
            prefix_route_registry_path=prefix_registry_path,
            successor_route_registry_path=inputs.route_registry,
            prefix_schedule_sha256=str(prefix_schedule_sha256),
            successor_schedule_sha256=str(successor_schedule_sha256),
            prefix_route_registry_sha256=str(prefix_route_registry_sha256),
            successor_route_registry_sha256=str(inputs.route_registry_sha256),
        )
    elif inputs.resume_provenance_rebind is not None:
        raise RuntimeError(
            "resume provenance rebind was provided but prefix provenance is unchanged"
        )

    metrics_steps: set[int] = set()
    for line in prefix_metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = row.get("step")
        if isinstance(step, int) and not isinstance(step, bool) and step > 0:
            metrics_steps.add(step)
    if not set(range(1, start + 1)).issubset(metrics_steps):
        raise RuntimeError("resume prefix FileLogger does not cover every prefix update")

    rollout_files: dict[str, str] = {}
    for step in range(1, start + 1):
        path = prefix_rollout_dir / f"{step}.jsonl"
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"resume prefix rollout is missing: {path}")
        rollout_files[str(step)] = _sha256(path)

    return {
        "schema": "amg_verl_resume_contract_v1",
        "prefix_run_dir": str(prefix_run),
        "prefix_launch_receipt": {
            "path": str(prefix_launch),
            "sha256": _sha256(prefix_launch),
        },
        "prefix_file_logger": {
            "path": str(prefix_metrics),
            "sha256": _sha256(prefix_metrics),
        },
        "prefix_rollout_data": {
            "path": str(prefix_rollout_dir),
            "files": rollout_files,
        },
        "resume_from_path": str(checkpoint),
        "resume_checkpoint_step": start,
        "resume_checkpoint_data": {
            "path": str(data_path),
            "sha256": _sha256(data_path),
        },
        "sampler_samples_yielded": expected_yielded,
        "schedule_capacity_episodes": int(schedule_report["count"]),
        "target_optimizer_updates": target,
        "target_episodes": target * 64,
        "invocation_optimizer_updates": target - start,
        "invocation_episodes": (target - start) * 64,
        "provenance_rebind": provenance_rebind,
    }


def prepare_launch(
    inputs: LaunchInputs,
    *,
    resolve_only: bool,
    skip_endpoint_preflight: bool,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Resolve and verify the exact upstream config before returning its command."""

    if skip_endpoint_preflight and not resolve_only:
        raise ValueError(
            "endpoint preflight may only be skipped for resolve-only checks"
        )
    inputs.run_dir.mkdir(parents=True, exist_ok=True)
    if inputs.route_registry is not None:
        if inputs.route_registry_sha256 is None:
            raise ValueError(
                "route_registry and route_registry_sha256 must be provided together"
            )
        registry = load_route_registry(
            inputs.route_registry,
            expected_sha256=inputs.route_registry_sha256,
            expected_route_ids=_MULTITASK_ROUTE_IDS,
        )
        schedule_report = inspect_schedule(
            inputs.schedule,
            expected_route_ids=registry.route_ids,
            expected_route_registry_sha256=registry.sha256,
        )
    else:
        schedule_report = inspect_schedule(inputs.schedule)
    launch_identity = _load_launch_identity(inputs, schedule_report=schedule_report)
    budget_contract = launch_identity.get("budget_contract")
    training_runtime = launch_identity.get("training_runtime")
    if not isinstance(budget_contract, Mapping):
        raise RuntimeError("publication identity omitted its async budget contract")
    if not isinstance(training_runtime, Mapping):
        raise RuntimeError("publication identity omitted its training runtime")
    resume_contract = _validate_resume_checkpoint(
        inputs,
        training_runtime=training_runtime,
        schedule_report=schedule_report,
    )
    orchestrator_preflight = _load_multitask_orchestrator_preflight(
        inputs,
        launch_identity=launch_identity,
        schedule_report=schedule_report,
        budget_contract=budget_contract,
        required=not resolve_only,
    )
    model_path = Path(str(training_runtime["base_model"]))
    runtime_python = str(training_runtime["python"])

    source_report_runtime = _verify_source(
        inputs,
        require_outer_clean=not resolve_only,
        launch_identity=launch_identity,
    )
    env = build_runtime_env(inputs, training_runtime=training_runtime)
    identity_environment = launch_identity.get("environment")
    if not isinstance(identity_environment, Mapping):
        raise RuntimeError("launch identity environment must be a mapping")
    env.update({str(key): str(value) for key, value in identity_environment.items()})
    endpoint_client_config = launch_identity.get("client_config")
    if endpoint_client_config is not None:
        if not isinstance(endpoint_client_config, Mapping):
            raise RuntimeError("launch endpoint client config must be a mapping")
        env["AMG_ENDPOINT_CLIENT_CONFIG_JSON"] = json.dumps(
            endpoint_client_config,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    overrides = build_overrides(
        inputs,
        effective_schedule=inputs.schedule,
        endpoint_client_config=endpoint_client_config,
        budget_contract=budget_contract,
        training_runtime=training_runtime,
    )
    resolved = subprocess.run(
        _resolved_command(overrides, python=runtime_python),
        check=False,
        text=True,
        capture_output=True,
        env=env,
        cwd=inputs.verl_root,
    )
    resolved_path = inputs.run_dir / "resolved-config.yaml"
    resolved_stderr_path = inputs.run_dir / "resolved-config.stderr.log"
    resolved_path.write_text(resolved.stdout, encoding="utf-8")
    resolved_stderr_path.write_text(resolved.stderr, encoding="utf-8")
    if resolved.returncode != 0:
        stderr_tail = "\n".join(resolved.stderr.splitlines()[-40:])
        raise RuntimeError(
            f"Hydra config resolution failed with exit {resolved.returncode}; "
            f"stderr saved to {resolved_stderr_path}:\n{stderr_tail}"
        )
    resolved_config = _load_yaml(resolved.stdout)
    budget = verify_resolved_config(
        resolved_config,
        mode=inputs.mode,
        expected_budget=budget_contract,
    )

    train_files = resolved_config["data"]["train_files"]
    if isinstance(train_files, str):
        resolved_train_files = [train_files]
    else:
        resolved_train_files = [str(value) for value in train_files]
    if resolved_train_files != [str(inputs.schedule)]:
        raise RuntimeError(
            f"Hydra resolved unexpected train_files: {resolved_train_files!r}"
        )

    runtime = None
    if not skip_endpoint_preflight:
        runtime = _runtime_preflight(
            inputs,
            env,
            model_path=model_path,
            training_runtime=training_runtime,
        )
    training_command = _training_command(overrides, python=runtime_python)
    receipt = {
        "schema": (
            "amg_verl_fully_async_multitask_launch_receipt_v1"
            if inputs.route_registry is not None
            else "amg_verl_fully_async_launch_receipt_v5"
        ),
        "entrypoint": _UPSTREAM_ENTRYPOINT,
        "inputs": {
            "mode": inputs.mode,
            "experiment_name": inputs.experiment_name,
            "model_path": str(model_path),
            "env_addr": inputs.env_addr,
            "route_registry": (
                str(inputs.route_registry)
                if inputs.route_registry is not None
                else None
            ),
            "route_registry_sha256": inputs.route_registry_sha256,
            "multitask_orchestrator_preflight": (
                str(inputs.multitask_orchestrator_preflight)
                if inputs.multitask_orchestrator_preflight is not None
                else None
            ),
            "run_dir": str(inputs.run_dir),
            "trainer_gpus": inputs.trainer_gpus,
            "standalone_rollout_gpus": inputs.standalone_rollout_gpus,
            "actor_use_fused_kernels": inputs.actor_use_fused_kernels,
            "critic_use_fused_kernels": inputs.critic_use_fused_kernels,
            "resume_from_path": (
                str(inputs.resume_from_path)
                if inputs.resume_from_path is not None
                else None
            ),
            "resume_prefix_run_dir": (
                str(inputs.resume_prefix_run_dir)
                if inputs.resume_prefix_run_dir is not None
                else None
            ),
            "resume_start_update": inputs.resume_start_update,
            "resume_target_update": inputs.resume_target_update,
            "resume_sampler_samples_yielded": inputs.resume_sampler_samples_yielded,
            "resume_provenance_rebind": (
                str(inputs.resume_provenance_rebind)
                if inputs.resume_provenance_rebind is not None
                else None
            ),
        },
        "source": source_report_runtime,
        "plugin_manifest": _production_manifest(inputs.outer_root),
        "schedule": schedule_report,
        "launch_identity": launch_identity,
        "endpoint_publication": (
            launch_identity
            if launch_identity.get("schema") == "amg_openmle_publication_identity_v3"
            else None
        ),
        "budget_contract": dict(budget_contract),
        "resume_contract": resume_contract,
        "budget": budget,
        "resolved_config": {
            "path": str(resolved_path),
            "sha256": _sha256(resolved_path),
        },
        "runtime_preflight": runtime,
        "multitask_orchestrator_preflight": orchestrator_preflight,
        "runtime_artifacts": {
            "file_logger": str(inputs.run_dir / "metrics.jsonl"),
            "rollout_data": str(inputs.run_dir / "rollout_data"),
            "hydra_config": str(inputs.run_dir / "hydra" / ".hydra" / "config.yaml"),
            "checkpoints": str(inputs.run_dir / "checkpoints"),
            "finalization": str(inputs.run_dir / "finalization.json"),
            **(
                {"trainer_log": str(inputs.run_dir / "trainer.log")}
                if inputs.route_registry is not None
                else {}
            ),
        },
        "training_command": training_command,
        "validation_enabled": False,
    }
    _atomic_json(inputs.run_dir / "launch-receipt.json", receipt)
    return training_command, env, receipt


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    outer_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Launch AMG through veRL's native fully-async PPO entrypoint"
    )
    parser.add_argument("--mode", choices=("gate", "formal"), required=True)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--outer-root", type=Path, default=outer_default)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--env-addr")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--endpoint-source-lock", type=Path)
    parser.add_argument("--endpoint-contract-tool", type=Path)
    parser.add_argument("--publication-receipt", type=Path)
    parser.add_argument("--formal-schedule-certificate", type=Path)
    parser.add_argument("--route-registry", type=Path)
    parser.add_argument("--route-registry-sha256")
    parser.add_argument("--multitask-source-lock", type=Path)
    parser.add_argument("--multitask-schedule-certificate", type=Path)
    parser.add_argument("--multitask-orchestrator-preflight", type=Path)
    parser.add_argument("--trainer-gpus", type=int, default=6)
    parser.add_argument("--standalone-rollout-gpus", type=int, default=2)
    parser.add_argument(
        "--learner-token-budget-profile", default="default-65536-v1"
    )
    parser.add_argument("--actor-train-token-budget", type=int, default=65_536)
    parser.add_argument("--critic-train-token-budget", type=int, default=65_536)
    parser.add_argument("--actor-use-fused-kernels", action="store_true")
    parser.add_argument("--critic-use-fused-kernels", action="store_true")
    parser.add_argument("--resume-from-path", type=Path)
    parser.add_argument("--resume-prefix-run-dir", type=Path)
    parser.add_argument("--resume-start-update", type=int)
    parser.add_argument("--resume-target-update", type=int)
    parser.add_argument("--resume-sampler-samples-yielded", type=int)
    parser.add_argument("--resume-provenance-rebind", type=Path)
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument(
        "--skip-runtime-preflight",
        "--skip-endpoint-preflight",
        dest="skip_runtime_preflight",
        action="store_true",
    )
    return parser.parse_args(argv)


def _resolve_cli_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"{label} is missing or symlinked: {path}")
    return path.resolve()


def _resolve_cli_regular_file(path: Path | None, *, label: str) -> Path | None:
    if path is None:
        return None
    return _require_regular_file(path, label=label).resolve()


def _resolve_cli_output_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise FileNotFoundError(f"{label} must not be a symlink: {path}")
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inputs = LaunchInputs(
        mode=args.mode,
        verl_root=_resolve_cli_directory(args.verl_root, label="veRL source root"),
        outer_root=_resolve_cli_directory(args.outer_root, label="outer source root"),
        schedule=_resolve_cli_regular_file(args.schedule, label="schedule"),
        env_addr=args.env_addr,
        run_dir=_resolve_cli_output_directory(args.run_dir, label="run directory"),
        experiment_name=args.experiment_name,
        endpoint_source_lock=_resolve_cli_regular_file(
            args.endpoint_source_lock, label="endpoint source lock"
        ),
        endpoint_contract_tool=_resolve_cli_regular_file(
            args.endpoint_contract_tool, label="endpoint contract tool"
        ),
        publication_receipt=_resolve_cli_regular_file(
            args.publication_receipt, label="publication receipt"
        ),
        formal_schedule_certificate=_resolve_cli_regular_file(
            args.formal_schedule_certificate,
            label="formal schedule certificate",
        ),
        trainer_gpus=args.trainer_gpus,
        standalone_rollout_gpus=args.standalone_rollout_gpus,
        learner_token_budget_profile=args.learner_token_budget_profile,
        actor_train_token_budget=args.actor_train_token_budget,
        critic_train_token_budget=args.critic_train_token_budget,
        actor_use_fused_kernels=args.actor_use_fused_kernels,
        critic_use_fused_kernels=args.critic_use_fused_kernels,
        resume_from_path=(
            _resolve_cli_directory(args.resume_from_path, label="resume checkpoint")
            if args.resume_from_path is not None
            else None
        ),
        resume_prefix_run_dir=(
            _resolve_cli_directory(args.resume_prefix_run_dir, label="resume prefix run")
            if args.resume_prefix_run_dir is not None
            else None
        ),
        resume_start_update=args.resume_start_update,
        resume_target_update=args.resume_target_update,
        resume_sampler_samples_yielded=args.resume_sampler_samples_yielded,
        resume_provenance_rebind=_resolve_cli_regular_file(
            args.resume_provenance_rebind,
            label="resume provenance rebind",
        ),
        route_registry=_resolve_cli_regular_file(
            args.route_registry, label="route registry"
        ),
        route_registry_sha256=args.route_registry_sha256,
        multitask_source_lock=_resolve_cli_regular_file(
            args.multitask_source_lock, label="multitask source lock"
        ),
        multitask_schedule_certificate=_resolve_cli_regular_file(
            args.multitask_schedule_certificate,
            label="multitask schedule certificate",
        ),
        multitask_orchestrator_preflight=_resolve_cli_regular_file(
            args.multitask_orchestrator_preflight,
            label="multitask orchestrator preflight",
        ),
    )
    command, env, receipt = prepare_launch(
        inputs,
        resolve_only=args.resolve_only,
        skip_endpoint_preflight=args.skip_runtime_preflight,
    )
    if args.resolve_only:
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    trainer_exit_code = 125
    try:
        completed = subprocess.run(
            command,
            check=False,
            env=env,
            cwd=inputs.verl_root,
        )
        trainer_exit_code = int(completed.returncode)
        if trainer_exit_code < 0:
            trainer_exit_code = 128 + abs(trainer_exit_code)
    except KeyboardInterrupt:
        trainer_exit_code = 130
    except Exception as exc:
        print(f"failed to execute native veRL trainer: {exc}", file=sys.stderr)
    verdict = finalize_run(inputs.run_dir, trainer_exit_code=trainer_exit_code)
    if trainer_exit_code != 0:
        return trainer_exit_code
    return 0 if verdict.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
