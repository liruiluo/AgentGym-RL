"""Thin launcher for AMG on upstream veRL's native fully-async entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_contract import (
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

_ENDPOINT_SOURCE_LOCK_SCHEMA = "openmle_fast_launcher_source_lock_v1"
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
    env_addr: str
    run_dir: Path
    experiment_name: str
    endpoint_source_lock: Path
    endpoint_contract_tool: Path
    publication_receipt: Path
    formal_schedule_certificate: Path
    trainer_gpus: int = 4
    standalone_rollout_gpus: int = 4
    actor_use_fused_kernels: bool = False
    critic_use_fused_kernels: bool = False


def _string(value: str | Path) -> str:
    rendered = str(value)
    if not rendered or any(character in rendered for character in ("\n", "\r", "\0")):
        raise ValueError(f"unsafe empty or multiline Hydra value: {rendered!r}")
    return rendered


def build_overrides(
    inputs: LaunchInputs,
    *,
    effective_schedule: Path,
    endpoint_client_config: Mapping[str, str | int],
    budget_contract: Mapping[str, Any],
    training_runtime: Mapping[str, Any],
) -> list[str]:
    """Build only Hydra overrides; upstream owns the composed base config."""

    if inputs.mode not in {"gate", "formal"}:
        raise ValueError(f"unsupported launch mode {inputs.mode!r}")
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
        "data.max_prompt_length=16384",
        "data.max_response_length=2048",
        "data.truncation=error",
        "data.return_raw_chat=True",
        "data.return_raw_input_ids=False",
        "data.shuffle=False",
        "data.seed=233",
        "data.custom_cls.path=pkg://agentmemorygym_verl.dataset",
        "data.custom_cls.name=AMGTrajectoryDataset",
        "data.continuous_token.enable=True",
        "data.continuous_token.model_family=qwen35",
        # Reuse veRL/Transformers native Qwen3.5 template control. The frozen
        # synchronous baseline used a closed thinking block so each generation
        # is the bare three-tool action expected by the environment parser.
        "+data.apply_chat_template_kwargs.enable_thinking=False",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.use_fused_kernels="
        f"{inputs.actor_use_fused_kernels}",
        "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch",
        # Keep veRL's native HF/FSDP gradient checkpointing enabled. The
        # synchronous comparator used the upstream default successfully;
        # disabling it made the four-way async critic retain full activations.
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
        # Keep the accepted actor baseline unchanged. The candidate changes only
        # critic FSDP2 resharding below.
        "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.shuffle=False",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        # Four-way FSDP leaves less activation headroom than the historical
        # eight-way synchronous trainer. Keep microbatch=8 but bound packed
        # training tokens; formal tuning may raise these after measured headroom.
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536",
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
        "actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode",
        "critic.enable=True",
        "critic.strategy=fsdp2",
        "critic.fsdp.strategy=fsdp2",
        "critic.fsdp.param_offload=False",
        "critic.fsdp.optimizer_offload=False",
        # Upstream FSDP2 can retain unsharded critic parameters between forward
        # and backward. B200 memory headroom is traded for fewer parameter
        # all-gathers; PPO data, losses, and optimizer semantics stay unchanged.
        "critic.fsdp.reshard_after_forward=False",
        f"critic.ppo_mini_batch_size={ppo_mini_batch_size}",
        "critic.ppo_micro_batch_size_per_gpu=8",
        "critic.ppo_epochs=1",
        "critic.shuffle=False",
        "critic.use_dynamic_bsz=True",
        # G64 r6/r8/r9/r11 showed that mechanically lowering this target while
        # gradient checkpointing was disabled did not control the activation
        # peak. Retain the conservative 32,768 target for the first gate with
        # upstream checkpointing restored; tune only from measured headroom.
        "critic.ppo_max_token_len_per_gpu=32768",
        "critic.forward_max_token_len_per_gpu=262144",
        "critic.optim.lr=1e-5",
        "critic.optim.weight_decay=0.01",
        "critic.optim.lr_warmup_steps=0",
        "critic.optim.lr_scheduler_type=constant",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.dtype=bfloat16",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
        "actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.8",
        "actor_rollout_ref.rollout.max_model_len=32768",
        "actor_rollout_ref.rollout.max_num_batched_tokens=131072",
        "actor_rollout_ref.rollout.max_num_seqs=32",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "+actor_rollout_ref.rollout.enable_sleep_mode=True",
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
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={inputs.trainer_gpus}",
        "trainer.device=cuda",
        "trainer.balance_batch=True",
        "trainer.critic_warmup=0",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={publication_cycles}",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.resume_mode=disable",
        "trainer.resume_from_path=null",
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
        f"async_training.runtime_receipt_path={run_dir}/native-runtime-receipt.json",
        "async_training.rollout_data_non_tensor_keys=[step_record_json]",
        "async_training.rollout_data_non_tensor_max_keys=1",
        "async_training.parameter_update_probe.enabled=True",
        "async_training.parameter_update_probe.max_parameters=8",
        "async_training.parameter_update_probe.max_elements_per_parameter=16",
        "async_training.parameter_update_probe.atol=0.0",
        "async_training.parameter_update_probe.require_change=True",
        f"hydra.run.dir={run_dir}/hydra",
        "hydra.output_subdir=.hydra",
        "hydra.job.chdir=False",
    ]
    for prefix in ("actor_rollout_ref", "data"):
        for key, value in agentgym.items():
            rendered = (
                json.dumps(value, ensure_ascii=True)
                if isinstance(value, str)
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
    inherited_path = env.get("PATH", "")
    path_entries = [
        entry
        for entry in inherited_path.split(os.pathsep)
        if entry and entry != runtime_bin
    ]
    env["PATH"] = os.pathsep.join([runtime_bin, *path_entries])
    env["VERL_USE_EXTERNAL_MODULES"] = "agentmemorygym_verl.action_gae"
    env["VERL_USE_EXTERNAL_PLUGINS"] = "none"
    env["VERL_FILE_LOGGER_PATH"] = str(inputs.run_dir / "metrics.jsonl")
    env["VERL_FULLY_ASYNC_RUNTIME_RECEIPT_PATH"] = str(
        inputs.run_dir / "native-runtime-receipt.json"
    )
    env["VLLM_USE_V1"] = "1"
    # vLLM otherwise emits platform-detection INFO to stdout before Hydra YAML.
    env["VLLM_LOGGING_LEVEL"] = "ERROR"
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HYDRA_FULL_ERROR"] = "1"
    env["RAY_DEDUP_LOGS"] = "0"
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
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{label} is missing or not regular: {path}")

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
    """Map publication identities to their repository-relative runtime paths."""

    outer_manifest: dict[str, str] = {}
    inner_manifest: dict[str, str] = {}
    for identity_path, digest in selected_files.items():
        if not isinstance(identity_path, str) or not isinstance(digest, str):
            raise TypeError("OpenMLE selected file manifest is malformed")
        if identity_path.startswith("inner:"):
            inner_manifest[identity_path.removeprefix("inner:")] = digest
        elif identity_path.startswith("outer:AgentGym-RL/"):
            # ``outer_root`` is the checkout root. AgentGym-RL is a tracked
            # subdirectory in that repository, so preserve it in the relative
            # path instead of treating it as a display-only identity prefix.
            outer_manifest[identity_path.removeprefix("outer:")] = digest
        else:
            raise RuntimeError(
                f"unsupported OpenMLE selected file identity: {identity_path!r}"
            )
    return outer_manifest, inner_manifest


def _verify_source(
    inputs: LaunchInputs,
    *,
    require_outer_clean: bool,
    endpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not (inputs.verl_root / "verl" / "experimental" / "fully_async_policy").is_dir():
        raise FileNotFoundError(f"not a veRL source tree: {inputs.verl_root}")
    verl_commit = _git(inputs.verl_root, "rev-parse", "HEAD")
    if verl_commit != EXPECTED_VERL_COMMIT:
        raise RuntimeError(
            f"veRL commit mismatch: expected {EXPECTED_VERL_COMMIT}, got {verl_commit}"
        )
    verl_status = _git(inputs.verl_root, "status", "--porcelain")
    if verl_status:
        raise RuntimeError("veRL runtime tree must be clean after the reviewed commit")

    publication_outer_commit = _require_git_revision(
        endpoint_identity.get("publication_outer_commit"),
        field="endpoint_identity.publication_outer_commit",
    )
    publication_inner_commit = _require_git_revision(
        endpoint_identity.get("publication_inner_commit"),
        field="endpoint_identity.publication_inner_commit",
    )
    outer_commit = _git(inputs.outer_root, "rev-parse", "HEAD")
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

    selected_files = endpoint_identity.get("selected_files")
    if not isinstance(selected_files, Mapping) or not selected_files:
        raise RuntimeError("selected OpenMLE publication omitted selected file hashes")
    outer_manifest, inner_manifest = _partition_selected_file_hashes(selected_files)
    verified_outer_files = verify_hash_manifest(inputs.outer_root, outer_manifest)
    verified_inner_files = verify_hash_manifest(agentgym_root, inner_manifest)

    raw_training_runtime = endpoint_identity.get("training_runtime")
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


def _runtime_preflight(
    inputs: LaunchInputs, env: dict[str, str], *, model_path: Path
) -> dict[str, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"publication model directory missing: {model_path}")
    code = r"""
import json
import shutil
import trl
from trl import AutoModelForCausalLMWithValueHead
from agentmemorygym_verl.agent_loop import AMGTaskNeutralAgentLoop
from agentmemorygym_verl.dataset import AMGTrajectoryDataset
from agentmemorygym_verl.env_client import create_env_client
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

fn = get_adv_estimator_fn("amg_action_axis_gae")
client_config = json.loads(__import__("os").environ["AMG_ENDPOINT_CLIENT_CONFIG_JSON"])
client = create_env_client({
    "task_name": "openmle_fast",
    "env_addr": __import__("os").environ["AMG_ENV_ADDR"],
    "timeout": 240,
    "max_retries": 2,
    **client_config,
})
try:
    framing = client.policy_framing()
finally:
    client.close()
if not isinstance(framing, list) or not framing:
    raise RuntimeError("AMG endpoint returned empty policy framing")
ninja_path = shutil.which("ninja")
if ninja_path is None:
    raise RuntimeError("publication runtime PATH does not provide ninja")
print(json.dumps({
    "adv_estimator": fn.__name__,
    "agent_loop": AMGTaskNeutralAgentLoop.__name__,
    "dataset": AMGTrajectoryDataset.__name__,
    "policy_framing_messages": len(framing),
    "ninja_path": ninja_path,
    "trl_version": trl.__version__,
    "value_head_class": AutoModelForCausalLMWithValueHead.__name__,
}, sort_keys=True))
"""
    probe_env = dict(env)
    probe_env["AMG_ENV_ADDR"] = inputs.env_addr
    if "AMG_ENDPOINT_CLIENT_CONFIG_JSON" not in probe_env:
        raise RuntimeError("endpoint client identity was not exported for preflight")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=probe_env,
        cwd=inputs.verl_root,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("AMG runtime preflight produced no receipt")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("AMG runtime preflight receipt is not JSON") from exc


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
    schedule_report = inspect_schedule(inputs.schedule)
    endpoint_identity = _load_endpoint_identity(inputs, schedule_report=schedule_report)
    budget_contract = endpoint_identity.get("budget_contract")
    training_runtime = endpoint_identity.get("training_runtime")
    if not isinstance(budget_contract, Mapping):
        raise RuntimeError("publication identity omitted its async budget contract")
    if not isinstance(training_runtime, Mapping):
        raise RuntimeError("publication identity omitted its training runtime")
    model_path = Path(str(training_runtime["base_model"]))
    runtime_python = str(training_runtime["python"])

    source_report_runtime = _verify_source(
        inputs,
        require_outer_clean=not resolve_only,
        endpoint_identity=endpoint_identity,
    )
    env = build_runtime_env(inputs, training_runtime=training_runtime)
    env.update(endpoint_identity["environment"])
    env["AMG_ENDPOINT_CLIENT_CONFIG_JSON"] = json.dumps(
        endpoint_identity["client_config"],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    overrides = build_overrides(
        inputs,
        effective_schedule=inputs.schedule,
        endpoint_client_config=endpoint_identity["client_config"],
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
        runtime = _runtime_preflight(inputs, env, model_path=model_path)
    training_command = _training_command(overrides, python=runtime_python)
    receipt = {
        "schema": "amg_verl_fully_async_launch_receipt_v4",
        "entrypoint": _UPSTREAM_ENTRYPOINT,
        "inputs": {
            "mode": inputs.mode,
            "experiment_name": inputs.experiment_name,
            "model_path": str(model_path),
            "env_addr": inputs.env_addr,
            "run_dir": str(inputs.run_dir),
            "trainer_gpus": inputs.trainer_gpus,
            "standalone_rollout_gpus": inputs.standalone_rollout_gpus,
            "actor_use_fused_kernels": inputs.actor_use_fused_kernels,
            "critic_use_fused_kernels": inputs.critic_use_fused_kernels,
        },
        "source": source_report_runtime,
        "plugin_manifest": _production_manifest(inputs.outer_root),
        "schedule": schedule_report,
        "endpoint_publication": endpoint_identity,
        "budget_contract": dict(budget_contract),
        "budget": budget,
        "resolved_config": {
            "path": str(resolved_path),
            "sha256": _sha256(resolved_path),
        },
        "runtime_preflight": runtime,
        "runtime_artifacts": {
            "native_receipt": str(inputs.run_dir / "native-runtime-receipt.json"),
            "file_logger": str(inputs.run_dir / "metrics.jsonl"),
            "rollout_data": str(inputs.run_dir / "rollout_data"),
            "hydra_config": str(inputs.run_dir / "hydra" / ".hydra" / "config.yaml"),
            "checkpoints": str(inputs.run_dir / "checkpoints"),
            "finalization": str(inputs.run_dir / "finalization.json"),
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
    parser.add_argument("--env-addr", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--endpoint-source-lock", type=Path, required=True)
    parser.add_argument("--endpoint-contract-tool", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument("--formal-schedule-certificate", type=Path, required=True)
    parser.add_argument("--trainer-gpus", type=int, default=4)
    parser.add_argument("--standalone-rollout-gpus", type=int, default=4)
    parser.add_argument("--actor-use-fused-kernels", action="store_true")
    parser.add_argument("--critic-use-fused-kernels", action="store_true")
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--skip-endpoint-preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inputs = LaunchInputs(
        mode=args.mode,
        verl_root=args.verl_root.resolve(),
        outer_root=args.outer_root.resolve(),
        schedule=args.schedule.resolve(),
        env_addr=args.env_addr,
        run_dir=args.run_dir.resolve(),
        experiment_name=args.experiment_name,
        endpoint_source_lock=args.endpoint_source_lock.resolve(),
        endpoint_contract_tool=args.endpoint_contract_tool.resolve(),
        publication_receipt=args.publication_receipt.resolve(),
        formal_schedule_certificate=args.formal_schedule_certificate.resolve(),
        trainer_gpus=args.trainer_gpus,
        standalone_rollout_gpus=args.standalone_rollout_gpus,
        actor_use_fused_kernels=args.actor_use_fused_kernels,
        critic_use_fused_kernels=args.critic_use_fused_kernels,
    )
    command, env, receipt = prepare_launch(
        inputs,
        resolve_only=args.resolve_only,
        skip_endpoint_preflight=args.skip_endpoint_preflight,
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
