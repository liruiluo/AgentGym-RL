from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from agentmemorygym_verl.config_contract import inspect_schedule, verify_resolved_config
from agentmemorygym_verl.identity import (
    EXPECTED_VERL_COMMIT,
    LOCKED_MODEL_FILE_SHA256,
    validate_training_runtime_lock,
)

RICH_V8_FIXTURES = next(
    (
        candidate
        for candidate in (
            Path("/tmp/openmle-v8-launch-fixtures-20260818"),
            Path("/private/tmp/openmle-v8-launch-fixtures"),
        )
        if (candidate / "source-lock.json").is_file()
    ),
    Path("/tmp/openmle-v8-launch-fixtures-20260818"),
)

FINAL_STATISTICS_VERL_COMMIT = "5a4ef518fa2552816d31ac28241df6f583eadd0a"
MULTITASK_ROUTES = ("webshop", "swesmith", "literesearcher", "openmle_fast")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publication_fixture_identity() -> tuple[dict, dict, dict, dict[str, str]]:
    required = {
        "source_lock": RICH_V8_FIXTURES / "source-lock.json",
        "publication_receipt": RICH_V8_FIXTURES / "publication-receipt.json",
        "formal_schedule_certificate": (
            RICH_V8_FIXTURES / "formal100-schedule-certificate.json"
        ),
        "endpoint_contract_tool": RICH_V8_FIXTURES / "launcher_contract.py",
        "gate_schedule": RICH_V8_FIXTURES / "g64-gate-single-pass.jsonl",
        "formal_schedule": RICH_V8_FIXTURES / "formal100-schedule.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "publication test fixture is incomplete: " + ", ".join(missing)
        )
    source_lock = json.loads(required["source_lock"].read_text(encoding="utf-8"))
    publication = json.loads(
        required["publication_receipt"].read_text(encoding="utf-8")
    )
    certificate = json.loads(
        required["formal_schedule_certificate"].read_text(encoding="utf-8")
    )
    digests = {name: sha256(path) for name, path in required.items()}
    return source_lock, publication, certificate, digests


SOURCE_LOCK, PUBLICATION_RECEIPT, FORMAL_SCHEDULE_CERTIFICATE, FIXTURE_SHA256 = (
    _publication_fixture_identity()
)
PUBLICATION_OUTER_COMMIT = SOURCE_LOCK["runtime_source"]["outer_commit"]
PUBLICATION_INNER_COMMIT = SOURCE_LOCK["runtime_source"]["inner_commit"]
PUBLICATION_TRAINING_RUNTIME = validate_training_runtime_lock(
    SOURCE_LOCK["training_runtime"]
)


def read_schedule(mode: str) -> tuple[Path, list[dict]]:
    filename = (
        "g64-gate-single-pass.jsonl" if mode == "gate" else "formal100-schedule.jsonl"
    )
    path = RICH_V8_FIXTURES / filename
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return path, rows


def endpoint_identity(mode: str) -> dict:
    source_lock = json.loads(
        (RICH_V8_FIXTURES / "source-lock.json").read_text(encoding="utf-8")
    )
    role = "gate_only" if mode == "gate" else "train_pool"
    runtime_source = source_lock["runtime_source"]
    selected = runtime_source["selected_files"]
    return {
        "expected_manifest_sha256": source_lock["integration"]["manifests"][role][
            "sha256"
        ],
        "expected_release_revision": runtime_source["openmle_tasks_revision"],
        "expected_outer_commit": runtime_source["outer_commit"],
        "expected_inner_commit": runtime_source["inner_commit"],
        "expected_role": role,
        "expected_executor_runtime_digest": source_lock["exact_runtime"][
            "runtime_digest"
        ],
        "expected_materializer_sha256": selected[
            "inner:agentenv-openmle-fast/agentenv_openmle_fast/materializer.py"
        ],
        "expected_actions_sha256": selected[
            "inner:agentenv-openmle-fast/agentenv_openmle_fast/actions.py"
        ],
        "expected_max_observation_tokens": 8192,
    }


def resolved_config(mode: str, run_dir: Path, schedule: Path) -> dict:
    formal = mode == "formal"
    trainer_gpus = 6 if formal else 4
    rollout_gpus = 2 if formal else 4
    ppo_mini_batch_size = 510 if formal else 512
    identity = endpoint_identity(mode)
    agentgym = {
        "task_name": "openmle_fast",
        "env_addr": "http://127.0.0.1:65525",
        "max_rounds": 30,
        "max_observation_tokens": 8192,
        "timeout": 240,
        "max_retries": 2,
        **identity,
    }
    return {
        "actor_rollout_ref": {
            "hybrid_engine": False,
            "agentgym": dict(agentgym),
            "model": {
                "path": PUBLICATION_TRAINING_RUNTIME["base_model"],
                "enable_gradient_checkpointing": True,
                "use_fused_kernels": False,
                "fused_kernel_options": {"impl_backend": "torch"},
            },
            "actor": {
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": 8,
                "ppo_epochs": 1,
                "shuffle": False,
                "use_dynamic_bsz": True,
                "loss_agg_mode": "token-mean",
                "use_prefix_grouper": False,
                "use_rollout_log_probs": True,
                "strategy": "fsdp2",
                "fsdp_config": {
                    "strategy": "fsdp2",
                    "param_offload": False,
                    "optimizer_offload": False,
                    "reshard_after_forward": True,
                },
                "optim": {"lr": 1e-6},
                "policy_loss": {"loss_mode": "bypass_mode"},
            },
            "rollout": {
                "n": 1,
                "name": "sglang",
                "mode": "async",
                "calculate_log_probs": True,
                "gpu_memory_utilization": 0.35,
                "standalone_gpu_memory_utilization": 0.8,
                "max_num_seqs": 32,
                "enforce_eager": False,
                "free_cache_engine": True,
                "engine_kwargs": {
                    "sglang": {
                        "mamba_scheduler_strategy": "no_buffer",
                        "disable_radix_cache": True,
                        "cuda_graph_max_bs": 32,
                        "max_running_requests": 32,
                        "chunked_prefill_size": 16384,
                        "max_prefill_tokens": 16384,
                    }
                },
                "multi_turn": {"enable": True},
                "agent": {
                    "default_agent_loop": "amg_task_neutral_async",
                    "agent_loop_config_path": "/plugin/amg_task_neutral_agent_loop.yaml",
                },
            },
        },
        "critic": {
            "enable": True,
            "strategy": "fsdp2",
            "ppo_mini_batch_size": ppo_mini_batch_size,
            "ppo_micro_batch_size_per_gpu": 8,
            "ppo_epochs": 1,
            "shuffle": False,
            "use_dynamic_bsz": True,
            "loss_agg_mode": "token-mean",
            "fsdp": {
                "strategy": "fsdp2",
                "param_offload": False,
                "optimizer_offload": False,
                "reshard_after_forward": True,
            },
            "optim": {"lr": 1e-5},
            "model": {
                "path": PUBLICATION_TRAINING_RUNTIME["base_model"],
                "enable_gradient_checkpointing": True,
                "use_fused_kernels": False,
                "fused_kernel_options": {"impl_backend": "torch"},
            },
        },
        "algorithm": {
            "adv_estimator": "amg_action_axis_gae",
            "amg_advantage_normalization": "upstream_masked_whiten",
            "gamma": 1.0,
            "lam": 1.0,
            "use_kl_in_reward": False,
            "rollout_correction": {"bypass_mode": True, "loss_type": "ppo_clip"},
        },
        "data": {
            "train_files": [str(schedule)],
            "val_files": [str(schedule)],
            "train_batch_size": 0,
            "gen_batch_size": 1,
            "shuffle": False,
            "seed": 233,
            "max_prompt_length": 16384,
            "max_response_length": 2048,
            "return_raw_chat": True,
            "custom_cls": {
                "path": "pkg://agentmemorygym_verl.dataset",
                "name": "AMGTrajectoryDataset",
            },
            "apply_chat_template_kwargs": {"enable_thinking": False},
            "agentgym": dict(agentgym),
        },
        "async_training": {
            "staleness_threshold": 0.1,
            "trigger_parameter_sync_step": 1,
            "require_batches": 64 / ppo_mini_batch_size,
            "partial_rollout": True,
            "use_trainer_do_validate": False,
            "use_dynamic_resource_scheduling": True,
            "dynamic_schedule_policy": "default",
            "dynamic_schedule_deactivate_ratio": 0.6,
            "dynamic_schedule_enable_rebalance": True,
            "concurrent_samples_per_replica": 16,
        },
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": trainer_gpus,
            "total_training_steps": 100 if formal else 1,
            "total_epochs": 1,
            "val_before_train": False,
            "test_freq": -1,
            "resume_mode": "disable",
            "resume_from_path": None,
            "save_freq": 10 if formal else 1,
            "max_actor_ckpt_to_keep": 1,
            "max_critic_ckpt_to_keep": 1,
            "logger": ["console", "file"],
            "default_local_dir": str(run_dir / "checkpoints"),
            "rollout_data_dir": str(run_dir / "rollout_data"),
            "validation_data_dir": None,
        },
        "rollout": {
            "nnodes": 1,
            "n_gpus_per_node": rollout_gpus,
            "n": 1,
            "total_rollout_steps": 6400 if formal else 64,
        },
    }


def _execution_info(
    *,
    kind="shell_command",
    changed_paths=(),
    stdout="ok",
    attempts=0,
    completed=0,
) -> dict:
    return {
        "action_kind": kind,
        "action_status": "completed",
        "counter_delta": {
            "execution_action_count": int(attempts > 0),
            "execution_attempt_count": attempts,
            "execution_completed_count": completed,
        },
        "execution": {
            "action_kind": kind,
            "status": "completed",
            "exit_code": 0 if kind == "shell_command" else None,
            "stdout": stdout,
            "changed_paths": list(changed_paths),
            "execution_action_delta": int(attempts > 0),
            "execution_attempt_delta": attempts,
            "execution_completed_delta": completed,
        },
    }


def _action_rows(
    item_id: str, data_idx: int, trajectory_uid: str, version: int, chain: bool
) -> list[dict]:
    if chain:
        actions = [
            (
                'shell_command {"command":"mkdir -p .agent_memory && printf '
                "'%s\\\\n' 'objective: improve validation' "
                "'measured_validation_or_failure: validation_mae=1.0' "
                "'conclusion: update the model' 'code_path: train.py' "
                "'next_action: edit train.py before rerunning' > "
                '.agent_memory/OPENMLE_CONTINUATION.md","workdir":".",'
                '"timeout_ms":20000}',
                _execution_info(
                    changed_paths=(".agent_memory/OPENMLE_CONTINUATION.md",)
                ),
            ),
            (
                'shell_command {"command":"cat '
                '.agent_memory/OPENMLE_CONTINUATION.md","workdir":".",'
                '"timeout_ms":20000}',
                _execution_info(
                    stdout=(
                        "objective: improve validation\n"
                        "measured_validation_or_failure: validation_mae=1.0\n"
                        "conclusion: update the model\n"
                        "code_path: train.py\n"
                        "next_action: edit train.py before rerunning\n"
                    )
                ),
            ),
            (
                "apply_patch\n*** Begin Patch\n*** Update File: train.py\n"
                "@@\n-print(1)\n+print(2)\n*** End Patch",
                _execution_info(kind="apply_patch", changed_paths=("train.py",)),
            ),
            (
                'shell_command {"command":"python train.py","workdir":".",'
                '"timeout_ms":20000}',
                _execution_info(stdout="validation_mae=0.8", attempts=1, completed=1),
            ),
        ]
    else:
        actions = [("submit", {"action_kind": "submit", "action_status": "completed"})]
    rows = []
    for order, (action, env_info) in enumerate(actions):
        wrapper_evidence = {}
        context_transition = {}
        control_request = None
        if chain and order == 0:
            wrapper_evidence = {
                "event": "context_compaction",
                "continuation_path": ".agent_memory/OPENMLE_CONTINUATION.md",
                "continuation_persisted": True,
                "preserved_policy_output": True,
                "preserved_native_observation": True,
                "native_action_kind": env_info["action_kind"],
                "native_action_status": env_info["action_status"],
            }
            context_transition = {"operation": "replace_messages", "messages": []}
            control_request = "Persist continuation state before compaction."
        elif chain and order == 1:
            wrapper_evidence = {
                "memory_event": "read",
                "document_read_observed": True,
            }
        elif chain and order == 2:
            wrapper_evidence = {"memory_event": "modify"}
        elif chain and order == 3:
            wrapper_evidence = {"memory_event": "execute", "outcome": "success"}
        rows.append(
            {
                "schema": "amg_task_neutral_action_row_v1",
                "item_id": item_id,
                "data_idx": data_idx,
                "trajectory_uid": trajectory_uid,
                "trajectory_row_uid": f"{trajectory_uid}-row-{order}",
                "trajectory_row_order": order,
                "trajectory_terminal": order == len(actions) - 1,
                "rollout_done_flag": order == len(actions) - 1,
                "immediate_reward": 1.0 if order == len(actions) - 1 else 0.0,
                "trajectory_return": 1.0,
                "task_round": order + 1,
                "action": action,
                "action_submission": {"raw_policy_output": action},
                "env_info_after": env_info,
                "context_transition": context_transition,
                "wrapper_evidence": wrapper_evidence,
                "control_request": control_request,
                "outcome": "success" if order == len(actions) - 1 else "continue",
                "prompt_token_count": 2,
                "prompt_token_sha256": "a" * 64,
                "response_token_count": 1,
                "response_token_sha256": "b" * 64,
                "min_global_steps": version,
                "max_global_steps": version,
            }
        )
    return rows


def _checkpoint(run_dir: Path, step: int, *, world_size: int) -> None:
    root = run_dir / "checkpoints"
    target = root / f"global_step_{step}"
    for role in ("actor", "critic"):
        role_dir = target / role
        role_dir.mkdir(parents=True)
        for rank in range(world_size):
            for kind in ("model", "optim", "extra_state"):
                (
                    role_dir / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
                ).write_bytes(f"{role}:{kind}:{rank}".encode())
        hf = role_dir / "huggingface"
        hf.mkdir()
        (hf / "config.json").write_text("{}\n", encoding="utf-8")
    (target / "data.pt").write_bytes(b"native-rollouter-dataloader-state")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")


def build_valid_run(run_dir: Path, mode: str = "gate") -> dict:
    formal = mode == "formal"
    run_dir.mkdir(parents=True, exist_ok=True)
    schedule_path, schedule_rows = read_schedule(mode)
    role = "gate_only" if mode == "gate" else "train_pool"
    episodes = len(schedule_rows)
    collections = 1 if mode == "gate" else 100
    publication_cycles = 1 if mode == "gate" else 100
    collections_per_publication = 1

    config = resolved_config(mode, run_dir, schedule_path)
    ppo_mini_batch_size = config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"]
    resolved_path = run_dir / "resolved-config.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    hydra_dir = run_dir / "hydra" / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )

    rollout_dir = run_dir / "rollout_data"
    rollout_dir.mkdir()
    real_rows: list[dict] = []
    version_pairs: Counter[str] = Counter()
    collection_real_rows: list[int] = []
    padding_rows = 0
    for collection in range(collections):
        jsonl_lines: list[str] = []
        collection_records: list[dict] = []
        start = collection * 64
        for position, schedule_row in enumerate(
            schedule_rows[start : start + 64], start=start
        ):
            version = min(
                publication_cycles - 1,
                collection // collections_per_publication,
            )
            item_id = schedule_row["item_id"]
            trajectory_uid = f"trajectory-{item_id}"
            rows = _action_rows(
                item_id,
                schedule_row["data_idx"],
                trajectory_uid,
                version,
                chain=position % 64 == 0,
            )
            for row in rows:
                real_rows.append(row)
                collection_records.append(row)
                version_pairs[f"{version}:{version}"] += 1
                jsonl_lines.append(
                    json.dumps(
                        {
                            "input": "policy prompt",
                            "output": row["action"],
                            "gts": None,
                            "score": row["immediate_reward"],
                            "step": collection + 1,
                            "step_record_json": json.dumps(row, sort_keys=True),
                            "is_padding": False,
                        },
                        sort_keys=True,
                    )
                )
        (rollout_dir / f"{collection + 1}.jsonl").write_text(
            "\n".join(jsonl_lines) + "\n", encoding="utf-8"
        )
        collection_real_rows.append(len(collection_records))
        padding_rows += (-len(collection_records)) % ppo_mini_batch_size

    metrics_path = run_dir / "metrics.jsonl"
    metric_lines = []
    for publication in range(publication_cycles):
        metric_lines.append(
            json.dumps(
                {
                    "step": publication + 1,
                    "data": {
                        "actor/grad_norm": 1.0,
                        "critic/grad_norm": 1.0,
                        "fully_async/count/current_param_version": publication,
                        "fully_async/count/stale_trajectory_processed": 0,
                        "fully_async/count/total_generated_samples": min(
                            episodes, (publication + 1) * 64
                        ),
                        "fully_async/count/dropped_stale_samples": 0,
                        "fully_async/monitor/queue/mq_queue_size": 0,
                        "fully_async/static/required_samples": 64,
                        "rollout_corr/kl": 0.01,
                        "rollout_corr/k3_kl": 0.001,
                        "rollout_corr/log_ppl_abs_diff": 0.01,
                    },
                },
                sort_keys=True,
            )
        )
    metrics_path.write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    checkpoint_step = publication_cycles
    _checkpoint(run_dir, checkpoint_step, world_size=6 if formal else 4)
    manifest = SOURCE_LOCK["integration"]["manifests"][role]
    routing = SOURCE_LOCK["integration"]["routing"][role]
    budget_contract = {
        "schema": "amg_verl_publication_budget_contract_v1",
        "mode": mode,
        "role": role,
        "publication_cycles": publication_cycles,
        "trigger_parameter_sync_step": collections_per_publication,
        "optimizer_updates": collections,
        "samples_per_update": 64,
        "episodes": episodes,
        "save_freq": 1 if mode == "gate" else 10,
        "max_actor_ckpt_to_keep": 1,
        "max_critic_ckpt_to_keep": 1,
        "model_path": PUBLICATION_TRAINING_RUNTIME["base_model"],
        "task_count": manifest["task_count"],
        "source_family_count": manifest["source_family_count"],
        "schedule_sha256": sha256(schedule_path),
        "manifest_sha256": manifest["sha256"],
        "routing_sha256": routing["sha256"],
    }
    budget = verify_resolved_config(config, mode=mode, expected_budget=budget_contract)
    publication_identity = {
        "schema": "amg_openmle_publication_identity_v3",
        "source_lock_path": str(RICH_V8_FIXTURES / "source-lock.json"),
        "source_lock_sha256": FIXTURE_SHA256["source_lock"],
        "contract_tool_path": str(RICH_V8_FIXTURES / "launcher_contract.py"),
        "contract_tool_sha256": FIXTURE_SHA256["endpoint_contract_tool"],
        "publication_receipt_path": str(RICH_V8_FIXTURES / "publication-receipt.json"),
        "publication_receipt_sha256": FIXTURE_SHA256["publication_receipt"],
        "schedule_certificate_path": str(
            RICH_V8_FIXTURES / "formal100-schedule-certificate.json"
        ),
        "schedule_certificate_sha256": FIXTURE_SHA256["formal_schedule_certificate"],
        "canonical_validation": {"status": "valid"},
        "publication_outer_commit": PUBLICATION_OUTER_COMMIT,
        "publication_inner_commit": PUBLICATION_INNER_COMMIT,
        "manifest_role": role,
        "manifest_sha256": manifest["sha256"],
        "routing_sha256": routing["sha256"],
        "task_count": manifest["task_count"],
        "source_family_count": manifest["source_family_count"],
        "schedule_count": episodes,
        "schedule_sha256": sha256(schedule_path),
        "launch_contract": dict(
            SOURCE_LOCK["launch_contracts"]["gate1" if mode == "gate" else "formal100"]
        ),
        "formal_schedule_contract": dict(FORMAL_SCHEDULE_CERTIFICATE),
        "budget_contract": dict(budget_contract),
        "client_config": endpoint_identity(mode),
        "environment": {},
        "selected_files": dict(SOURCE_LOCK["runtime_source"]["selected_files"]),
        "training_runtime": dict(PUBLICATION_TRAINING_RUNTIME),
    }
    launch_receipt = {
        "schema": "amg_verl_fully_async_launch_receipt_v5",
        "entrypoint": "verl.experimental.fully_async_policy.fully_async_main",
        "inputs": {
            "mode": mode,
            "experiment_name": f"rich-v8-{mode}",
            "model_path": PUBLICATION_TRAINING_RUNTIME["base_model"],
            "env_addr": "http://127.0.0.1:65525",
            "run_dir": str(run_dir),
        },
        "source": {
            "verl_commit": EXPECTED_VERL_COMMIT,
            "publication_outer_commit": PUBLICATION_OUTER_COMMIT,
            "outer_commit": PUBLICATION_OUTER_COMMIT,
            "outer_diff_paths": [],
            "agentgym_commit": PUBLICATION_INNER_COMMIT,
            "agentgym_expected_commit": PUBLICATION_INNER_COMMIT,
            "training_runtime": dict(PUBLICATION_TRAINING_RUNTIME),
            "model_files_sha256": LOCKED_MODEL_FILE_SHA256,
        },
        "schedule": {
            "path": str(schedule_path),
            "sha256": sha256(schedule_path),
            "count": episodes,
            "unique_item_ids": episodes,
            "manifest_digest": manifest["sha256"],
            "role": role,
        },
        "endpoint_publication": publication_identity,
        "budget_contract": budget_contract,
        "budget": budget,
        "resolved_config": {
            "path": str(resolved_path),
            "sha256": sha256(resolved_path),
        },
        "runtime_artifacts": {
            "file_logger": str(metrics_path),
            "rollout_data": str(rollout_dir),
            "hydra_config": str(hydra_dir / "config.yaml"),
            "checkpoints": str(run_dir / "checkpoints"),
            "finalization": str(run_dir / "finalization.json"),
        },
        "validation_enabled": False,
    }
    launch_path = run_dir / "launch-receipt.json"
    launch_path.write_text(
        json.dumps(launch_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "run_dir": run_dir,
        "mode": mode,
        "schedule_rows": schedule_rows,
        "real_rows": real_rows,
        "padding_rows": padding_rows,
        "metrics_path": metrics_path,
        "resolved_path": resolved_path,
        "hydra_path": hydra_dir / "config.yaml",
        "rollout_dir": rollout_dir,
        "checkpoint_root": run_dir / "checkpoints",
        "launch_path": launch_path,
    }


def build_valid_multitask_run(
    run_dir: Path,
    *,
    updates: int = 8,
    route_counts_by_update: list[dict[str, int]] | None = None,
) -> dict:
    """Build a compact four-route receipt and exact owner telemetry fixture."""

    run_dir.mkdir(parents=True, exist_ok=True)
    identity_dir = run_dir / "identity"
    identity_dir.mkdir()
    route_ids = MULTITASK_ROUTES
    samples_per_update = 64
    episodes = updates * samples_per_update
    role = "train_pool"
    if route_counts_by_update is None:
        route_counts_by_update = [
            {route_id: samples_per_update // len(route_ids) for route_id in route_ids}
            for _ in range(updates)
        ]
    if len(route_counts_by_update) != updates:
        raise ValueError("route count fixture must cover every update")
    if any(
        sum(counts.values()) != samples_per_update for counts in route_counts_by_update
    ):
        raise ValueError("each fixture update must contain exactly 64 episodes")
    if any(set(counts) != set(route_ids) for counts in route_counts_by_update):
        raise ValueError("fixture update route set drifted")
    expected_per_route = episodes // len(route_ids)
    if any(
        sum(counts[route_id] for counts in route_counts_by_update) != expected_per_route
        for route_id in route_ids
    ):
        raise ValueError("fixture route counts must conserve the frozen schedule")

    registry_payload = {
        "schema": "amg_route_registry_v1",
        "agent_name": "amg_task_neutral_async",
        "routes": [
            {
                "route_id": route_id,
                "max_rounds": 30,
                "max_observation_tokens": 8192,
                "policy_framing_sha256": str(index + 5) * 64,
                "route_attestation_sha256": str(index + 1) * 64,
                "client": {
                    "task_name": route_id,
                    "env_addr": f"http://127.0.0.1:{65101 + index}",
                    "timeout": 240,
                    "max_retries": 2,
                },
            }
            for index, route_id in enumerate(route_ids)
        ],
    }
    registry_path = identity_dir / "route-registry.json"
    registry_path.write_text(
        json.dumps(registry_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    registry_sha256 = sha256(registry_path)
    manifest_digest = "a" * 64
    schedule_path = identity_dir / "multitask-schedule.jsonl"
    schedule_rows: list[dict] = []
    route_schedule_rows: dict[str, list[dict]] = {
        route_id: [] for route_id in route_ids
    }
    route_local_indices = Counter()
    for position in range(episodes):
        route_id = route_ids[position % len(route_ids)]
        route_index = route_local_indices[route_id]
        route_local_indices[route_id] += 1
        route_offset = route_ids.index(route_id)
        row = {
            "index": position,
            "data_idx": route_index,
            "route_id": route_id,
            "data_source": route_id,
            "agent_name": "amg_task_neutral_async",
            "item_id": f"{route_id}:item-{route_index}",
            "extra_info": {
                "index": position,
                "route_id": route_id,
                "manifest_digest": manifest_digest,
                "panel_id": "multitask-fixture",
                "role": role,
                "schedule_position": position,
                "route_registry_sha256": registry_sha256,
                "route_attestation_sha256": str(route_offset + 1) * 64,
                "source_schedule_sha256": str(route_offset + 5) * 64,
                "source_manifest_digest": "9abc"[route_offset] * 64,
                "source_panel_id": f"source-{route_offset}",
            },
        }
        schedule_rows.append(row)
        route_schedule_rows[route_id].append(row)
    schedule_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in schedule_rows) + "\n",
        encoding="utf-8",
    )
    schedule_sha256 = sha256(schedule_path)
    schedule_report = inspect_schedule(
        schedule_path,
        expected_count=episodes,
        expected_sha256=schedule_sha256,
        expected_role=role,
        expected_route_ids=route_ids,
        expected_route_registry_sha256=registry_sha256,
    )

    certificate = {
        "schema": "amg_multitask_schedule_certificate_v1",
        "spec_sha256": manifest_digest,
        "schedule_sha256": schedule_sha256,
        "route_registry_sha256": registry_sha256,
        "role": role,
        "panel_id": "multitask-fixture",
        "agent_name": "amg_task_neutral_async",
        "optimizer_updates": updates,
        "samples_per_update": samples_per_update,
        "row_count": episodes,
        "route_order": list(route_ids),
        "per_route_rows": {route_id: expected_per_route for route_id in route_ids},
        "sources": {
            route_id: {
                "schedule_sha256": str(index + 5) * 64,
                "route_attestation_sha256": str(index + 1) * 64,
                "source_row_count": expected_per_route,
                "allow_repetition": False,
            }
            for index, route_id in enumerate(route_ids)
        },
    }
    certificate_path = identity_dir / "schedule-certificate.json"
    certificate_path.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_lock = {
        "schema": "amg_multitask_launcher_source_lock_v1",
        "status": "pass",
        "runtime_source": {
            "outer_commit": "c" * 40,
            "inner_commit": "d" * 40,
            "verl_commit": FINAL_STATISTICS_VERL_COMMIT,
            "selected_files": {
                "outer:async_plugins/agentmemorygym_verl/finalizer.py": "1" * 64,
                "inner:agentenv/agentenv/envs/task.py": "2" * 64,
            },
        },
        "training_runtime": dict(PUBLICATION_TRAINING_RUNTIME),
        "integration": {
            "route_registry": {
                "sha256": registry_sha256,
                "route_ids": list(route_ids),
            },
            "schedule_certificate": {
                "sha256": sha256(certificate_path),
                "schedule_sha256": schedule_sha256,
            },
        },
    }
    source_lock_path = identity_dir / "source-lock.json"
    source_lock_path.write_text(
        json.dumps(source_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = resolved_config("formal", run_dir, schedule_path)
    route_config = {
        "route_registry_path": str(registry_path),
        "route_registry_sha256": registry_sha256,
        "route_registry_expected_ids": list(route_ids),
    }
    config["actor_rollout_ref"]["agentgym"] = dict(route_config)
    config["data"]["agentgym"] = dict(route_config)
    config["trainer"]["total_training_steps"] = updates
    config["rollout"]["total_rollout_steps"] = episodes
    resolved_path = run_dir / "resolved-config.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    hydra_dir = run_dir / "hydra" / ".hydra"
    hydra_dir.mkdir(parents=True)
    hydra_path = hydra_dir / "config.yaml"
    hydra_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    budget_contract = {
        "schema": "amg_verl_multitask_budget_contract_v1",
        "mode": "formal",
        "role": role,
        "publication_cycles": updates,
        "trigger_parameter_sync_step": 1,
        "optimizer_updates": updates,
        "samples_per_update": samples_per_update,
        "episodes": episodes,
        "save_freq": 10,
        "max_actor_ckpt_to_keep": 1,
        "max_critic_ckpt_to_keep": 1,
        "model_path": PUBLICATION_TRAINING_RUNTIME["base_model"],
        "route_ids": list(route_ids),
        "route_registry_sha256": registry_sha256,
        "schedule_sha256": schedule_sha256,
        "manifest_sha256": manifest_digest,
        "routing_sha256": schedule_sha256,
    }
    budget = verify_resolved_config(
        config, mode="formal", expected_budget=budget_contract
    )

    rollout_dir = run_dir / "rollout_data"
    rollout_dir.mkdir()
    route_cursors = Counter()
    cumulative_episodes = Counter()
    cumulative_actions = Counter()
    cumulative_tokens = Counter()
    all_real_rows: list[dict] = []
    metric_lines: list[str] = []
    padding_rows = 0
    for update, route_counts in enumerate(route_counts_by_update, start=1):
        documents: list[dict] = []
        update_episodes = Counter()
        update_actions = Counter()
        update_tokens = Counter()
        for route_id in route_ids:
            for local_position in range(route_counts[route_id]):
                schedule_row = route_schedule_rows[route_id][route_cursors[route_id]]
                route_cursors[route_id] += 1
                uid = f"trajectory-{schedule_row['item_id']}"
                rows = _action_rows(
                    schedule_row["item_id"],
                    schedule_row["data_idx"],
                    uid,
                    update - 1,
                    chain=local_position == 0,
                )
                for record in rows:
                    record["route_id"] = route_id
                    record["data_source"] = route_id
                    all_real_rows.append(record)
                    update_actions[route_id] += 1
                    update_tokens[route_id] += record["response_token_count"]
                    documents.append(
                        {
                            "input": "policy prompt",
                            "output": record["action"],
                            "gts": None,
                            "score": record["immediate_reward"],
                            "step": update,
                            "step_record_json": json.dumps(record, sort_keys=True),
                            "is_padding": False,
                        }
                    )
                update_episodes[route_id] += 1
        (rollout_dir / f"{update}.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in documents) + "\n",
            encoding="utf-8",
        )
        padding_rows += (-len(documents)) % config["actor_rollout_ref"]["actor"][
            "ppo_mini_batch_size"
        ]
        cumulative_episodes.update(update_episodes)
        cumulative_actions.update(update_actions)
        cumulative_tokens.update(update_tokens)
        data = {
            "actor/grad_norm": 1.0,
            "critic/grad_norm": 1.0,
            "fully_async/count/current_param_version": update - 1,
            "fully_async/count/stale_trajectory_processed": 0,
            "fully_async/count/total_generated_samples": update * samples_per_update,
            "fully_async/count/dropped_stale_samples": 0,
            "fully_async/monitor/queue/mq_queue_size": 0,
            "fully_async/static/required_samples": samples_per_update,
            "rollout_corr/kl": 0.01,
            "rollout_corr/k3_kl": 0.001,
            "rollout_corr/log_ppl_abs_diff": 0.01,
        }
        for measure, current, cumulative in (
            ("episodes", update_episodes, cumulative_episodes),
            ("action_rows", update_actions, cumulative_actions),
            ("policy_response_tokens", update_tokens, cumulative_tokens),
        ):
            data[f"fully_async/sum/optimizer_consumed_{measure}"] = sum(
                current.values()
            )
            data[f"fully_async/count/optimizer_consumed_{measure}"] = sum(
                cumulative.values()
            )
            for route_id in route_ids:
                data[
                    f"fully_async/sum/optimizer_consumed_{measure}/data_source/{route_id}"
                ] = current[route_id]
                data[
                    f"fully_async/count/optimizer_consumed_{measure}/data_source/{route_id}"
                ] = cumulative[route_id]
        metric_lines.append(json.dumps({"step": update, "data": data}, sort_keys=True))
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    trainer_log = run_dir / "train.log"
    max_required_samples = int(samples_per_update * (0.1 + 1.0))
    final_statistics = {
        "schema": "verl_fully_async_final_statistics_v1",
        "queue": {
            "queue_size": 0,
            "total_produced": episodes,
            "total_consumed": episodes,
            "dropped_samples": 0,
            "total_cleared": 0,
            "max_queue_size": max_required_samples,
            "enqueued_by_data_source": dict(cumulative_episodes),
            "consumed_by_data_source": dict(cumulative_episodes),
            "evicted_by_data_source": {},
            "cleared_by_data_source": {},
            "resident_by_data_source": {},
        },
        "rollouter": {
            "monitor/active_tasks_size": 0,
            "monitor/queue/pending_queue_size": 0,
            "monitor/queue/mq_queue_size": 0,
            "count/total_generated_samples": episodes,
            "count/rollout_dispatched_samples": episodes,
            "count/rollout_inflight_samples": 0,
            "count/rollout_completed_samples": episodes,
            "count/rollout_failed_samples": 0,
            "count/rollout_cancelled_samples": 0,
            "count/queue_enqueued_samples": episodes,
            "count/queue_dequeued_samples": episodes,
            "count/queue_overflow_evictions": 0,
            "count/queue_cleared_samples": 0,
            "count/queue_resident_samples": 0,
            "count/staleness_samples": 0,
            "count/dropped_stale_samples": 0,
            "static/max_required_samples": max_required_samples,
            "static/required_samples": samples_per_update,
            "static/staleness_threshold": 0.1,
            "static/max_queue_size": max_required_samples,
            "static/max_concurrent_samples": 32,
        },
        "trainer": {
            "optimizer_consumed_episodes": episodes,
            "optimizer_consumed_action_rows": sum(cumulative_actions.values()),
            "optimizer_consumed_policy_response_tokens": sum(
                cumulative_tokens.values()
            ),
            "optimizer_consumed_episodes_by_data_source": dict(cumulative_episodes),
            "optimizer_consumed_action_rows_by_data_source": dict(cumulative_actions),
            "optimizer_consumed_policy_response_tokens_by_data_source": dict(
                cumulative_tokens
            ),
            "stale_action_rows": 0,
            "stale_action_rows_by_data_source": {},
            "current_param_version": updates,
        },
        "queue_cleanup": {"status": "completed"},
    }
    for event in (
        "rollout_dispatched",
        "rollout_completed",
        "queue_enqueued",
        "queue_dequeued",
    ):
        for route_id in route_ids:
            final_statistics["rollouter"][f"count/{event}/data_source/{route_id}"] = (
                cumulative_episodes[route_id]
            )
    for event in (
        "rollout_inflight",
        "rollout_failed",
        "rollout_cancelled",
    ):
        for route_id in route_ids:
            final_statistics["rollouter"][f"count/{event}/data_source/{route_id}"] = 0
    trainer_log.write_text(
        "runtime output\n[FullyAsyncTaskRunner][FinalStatistics] "
        + json.dumps(final_statistics, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    _checkpoint(run_dir, updates, world_size=6)
    launch_identity = {
        "schema": "amg_multitask_source_identity_v1",
        "source_lock_path": str(source_lock_path),
        "source_lock_sha256": sha256(source_lock_path),
        "schedule_certificate_path": str(certificate_path),
        "schedule_certificate_sha256": sha256(certificate_path),
        "publication_outer_commit": "c" * 40,
        "publication_inner_commit": "d" * 40,
        "verl_commit": FINAL_STATISTICS_VERL_COMMIT,
        "route_registry_path": str(registry_path),
        "route_registry_sha256": registry_sha256,
        "route_ids": list(route_ids),
        "schedule_count": episodes,
        "schedule_sha256": schedule_sha256,
        "formal_schedule_contract": certificate,
        "budget_contract": budget_contract,
        "client_config": None,
        "environment": {},
        "selected_files": dict(source_lock["runtime_source"]["selected_files"]),
        "training_runtime": dict(PUBLICATION_TRAINING_RUNTIME),
    }
    launch_receipt = {
        "schema": "amg_verl_fully_async_multitask_launch_receipt_v1",
        "entrypoint": "verl.experimental.fully_async_policy.fully_async_main",
        "inputs": {
            "mode": "formal",
            "experiment_name": "multitask-fixture",
            "model_path": PUBLICATION_TRAINING_RUNTIME["base_model"],
            "env_addr": None,
            "route_registry": str(registry_path),
            "route_registry_sha256": registry_sha256,
            "run_dir": str(run_dir),
            "trainer_gpus": 6,
            "standalone_rollout_gpus": 2,
        },
        "source": {
            "verl_commit": FINAL_STATISTICS_VERL_COMMIT,
            "verl_clean": True,
            "publication_outer_commit": "c" * 40,
            "outer_commit": "c" * 40,
            "outer_diff_paths": [],
            "outer_clean": True,
            "agentgym_commit": "d" * 40,
            "agentgym_expected_commit": "d" * 40,
            "agentgym_clean": True,
            "training_runtime": dict(PUBLICATION_TRAINING_RUNTIME),
            "model_files_sha256": LOCKED_MODEL_FILE_SHA256,
        },
        "schedule": schedule_report,
        "launch_identity": launch_identity,
        "endpoint_publication": None,
        "budget_contract": budget_contract,
        "budget": budget,
        "resolved_config": {
            "path": str(resolved_path),
            "sha256": sha256(resolved_path),
        },
        "runtime_artifacts": {
            "file_logger": str(metrics_path),
            "rollout_data": str(rollout_dir),
            "hydra_config": str(hydra_path),
            "checkpoints": str(run_dir / "checkpoints"),
            "finalization": str(run_dir / "finalization.json"),
            "trainer_log": str(trainer_log),
        },
        "validation_enabled": False,
    }
    launch_path = run_dir / "launch-receipt.json"
    launch_path.write_text(
        json.dumps(launch_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "run_dir": run_dir,
        "route_ids": route_ids,
        "schedule_rows": schedule_rows,
        "real_rows": all_real_rows,
        "padding_rows": padding_rows,
        "metrics_path": metrics_path,
        "resolved_path": resolved_path,
        "hydra_path": hydra_path,
        "rollout_dir": rollout_dir,
        "checkpoint_root": run_dir / "checkpoints",
        "launch_path": launch_path,
        "trainer_log": trainer_log,
    }


def mutate_final_statistics(path: Path, mutate) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = "[FullyAsyncTaskRunner][FinalStatistics] "
    matches = [index for index, line in enumerate(lines) if marker in line]
    if len(matches) != 1:
        raise ValueError("fixture trainer log does not contain one statistics row")
    index = matches[0]
    prefix, payload = lines[index].split(marker, 1)
    value = json.loads(payload)
    mutate(value)
    lines[index] = prefix + marker + json.dumps(value, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mutate_json(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
