from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from agentmemorygym_verl.config_contract import verify_resolved_config
from agentmemorygym_verl.identity import (
    EXPECTED_VERL_COMMIT,
    LOCKED_MODEL_FILE_SHA256,
    validate_training_runtime_lock,
)

RICH_V8_FIXTURES = Path("/tmp/openmle-v8-launch-fixtures-20260818")


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
                "ppo_mini_batch_size": 512,
                "ppo_micro_batch_size_per_gpu": 8,
                "ppo_epochs": 1,
                "shuffle": False,
                "use_dynamic_bsz": True,
                "use_rollout_log_probs": True,
                "strategy": "fsdp2",
                "fsdp_config": {
                    "strategy": "fsdp2",
                    "param_offload": False,
                    "optimizer_offload": False,
                    "reshard_after_forward": False,
                },
                "optim": {"lr": 1e-6},
                "policy_loss": {"loss_mode": "bypass_mode"},
            },
            "rollout": {
                "n": 1,
                "name": "vllm",
                "mode": "async",
                "calculate_log_probs": True,
                "gpu_memory_utilization": 0.35,
                "standalone_gpu_memory_utilization": 0.8,
                "engine_kwargs": {"vllm": {"gdn_prefill_backend": "triton"}},
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
            "ppo_mini_batch_size": 512,
            "ppo_micro_batch_size_per_gpu": 8,
            "ppo_epochs": 1,
            "shuffle": False,
            "use_dynamic_bsz": True,
            "fsdp": {
                "strategy": "fsdp2",
                "param_offload": False,
                "optimizer_offload": False,
                "reshard_after_forward": False,
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
            "continuous_token": {"enable": True, "model_family": "qwen35"},
            "apply_chat_template_kwargs": {"enable_thinking": False},
            "agentgym": dict(agentgym),
        },
        "async_training": {
            "staleness_threshold": 0.1,
            "trigger_parameter_sync_step": 1,
            "require_batches": 0.125,
            "partial_rollout": True,
            "use_trainer_do_validate": False,
            "use_dynamic_resource_scheduling": True,
            "dynamic_schedule_policy": "default",
            "dynamic_schedule_deactivate_ratio": 0.6,
            "dynamic_schedule_enable_rebalance": True,
            "concurrent_samples_per_replica": 16,
            "runtime_receipt_path": str(run_dir / "native-runtime-receipt.json"),
            "rollout_data_non_tensor_keys": ["step_record_json"],
            "rollout_data_non_tensor_max_keys": 1,
            "parameter_update_probe": {
                "enabled": True,
                "max_parameters": 8,
                "max_elements_per_parameter": 16,
                "atol": 0.0,
                "require_change": True,
            },
        },
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": 4,
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
            "n_gpus_per_node": 4,
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
    item_id: str, trajectory_uid: str, version: int, chain: bool
) -> list[dict]:
    if chain:
        actions = [
            (
                "shell_command {\"command\":\"mkdir -p .agent_memory && printf "
                "'%s\\\\n' 'objective: improve validation' "
                "'measured_validation_or_failure: validation_mae=1.0' "
                "'conclusion: update the model' 'code_path: train.py' "
                "'next_action: edit train.py before rerunning' > "
                ".agent_memory/OPENMLE_CONTINUATION.md\",\"workdir\":\".\","
                "\"timeout_ms\":20000}",
                _execution_info(
                    changed_paths=(".agent_memory/OPENMLE_CONTINUATION.md",)
                ),
            ),
            (
                "shell_command {\"command\":\"cat "
                ".agent_memory/OPENMLE_CONTINUATION.md\",\"workdir\":\".\","
                "\"timeout_ms\":20000}",
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
                "shell_command {\"command\":\"python train.py\",\"workdir\":\".\","
                "\"timeout_ms\":20000}",
                _execution_info(
                    stdout="validation_mae=0.8", attempts=1, completed=1
                ),
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
        rows.append(
            {
                "schema": "amg_task_neutral_action_row_v1",
                "item_id": item_id,
                "data_idx": 0,
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


def _checkpoint(run_dir: Path, step: int) -> None:
    root = run_dir / "checkpoints"
    target = root / f"global_step_{step}"
    for role in ("actor", "critic"):
        role_dir = target / role
        role_dir.mkdir(parents=True)
        for rank in range(4):
            for kind in ("model", "optim", "extra_state"):
                (role_dir / f"{kind}_world_size_4_rank_{rank}.pt").write_bytes(
                    f"{role}:{kind}:{rank}".encode()
                )
        hf = role_dir / "huggingface"
        hf.mkdir()
        (hf / "config.json").write_text("{}\n", encoding="utf-8")
    (target / "data.pt").write_bytes(b"native-rollouter-dataloader-state")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")


def build_valid_run(run_dir: Path, mode: str = "gate") -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    schedule_path, schedule_rows = read_schedule(mode)
    role = "gate_only" if mode == "gate" else "train_pool"
    episodes = len(schedule_rows)
    collections = 1 if mode == "gate" else 100
    publication_cycles = 1 if mode == "gate" else 100
    collections_per_publication = 1

    config = resolved_config(mode, run_dir, schedule_path)
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
    collection_real_tokens: list[int] = []
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
                item_id, trajectory_uid, version, chain=position % 64 == 0
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
        collection_real_tokens.append(
            sum(row["response_token_count"] for row in collection_records)
        )
        padding_rows += (-len(collection_records)) % 512

    metrics_path = run_dir / "metrics.jsonl"
    metric_lines = []
    for publication in range(publication_cycles):
        collection_start = publication * collections_per_publication
        collection_stop = collection_start + collections_per_publication
        real_token_count = sum(collection_real_tokens[collection_start:collection_stop])
        metric_lines.append(
            json.dumps(
                {
                    "step": publication + 1,
                    "data": {
                        "actor/grad_norm": 1.0,
                        "critic/grad_norm": 1.0,
                        "fully_async/count/current_param_version": publication,
                        "fully_async/count/stale_trajectory_processed": 0,
                        "fully_async/count/terminal_underfill_samples": 0,
                        "rollout_corr/bypass_real_token_count": real_token_count,
                        "rollout_corr/bypass_max_abs_diff": 0.0,
                        "parameter_update_probe/actor/changed": True,
                        "parameter_update_probe/critic/changed": True,
                    },
                },
                sort_keys=True,
            )
        )
    metrics_path.write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    checkpoint_step = publication_cycles
    _checkpoint(run_dir, checkpoint_step)
    probe = {
        "changed": True,
        "changed_elements": 1,
        "sampled_elements": 16,
        "sampled_parameters": 8,
        "worker_count": 4,
        "max_abs_diff": 0.0001,
        "atol": 0.0,
    }
    trainer_statistics = {
        "global_steps": collections + 1,
        "current_param_version": publication_cycles,
        "total_train_steps": publication_cycles,
        "local_trigger_step": 1,
        "processed_samples": episodes,
        "stale_trajectory_processed": 0,
        "terminal_underfill_events": 0,
        "terminal_underfill_samples": 0,
        "pending_rollout_dump_writes": 0,
        "latest_bypass_log_prob_evidence": {
            "rollout_corr/bypass_real_token_count": collection_real_tokens[-1],
            "rollout_corr/bypass_max_abs_diff": 0.0,
        },
        "latest_parameter_update_probe": {
            "actor": dict(probe),
            "critic": dict(probe),
        },
    }
    max_required_samples = int(64 * 1.1 * collections_per_publication)
    rollouter_statistics = {
        "monitor/active_tasks_size": 0,
        "monitor/queue/pending_queue_size": 0,
        "monitor/queue/mq_queue_size": 0,
        "count/total_generated_samples": episodes,
        "count/staleness_samples": 0,
        "count/dropped_stale_samples": 0,
        "static/max_required_samples": max_required_samples,
        "static/required_samples": 64,
        "static/staleness_threshold": 0.1,
        "static/max_queue_size": max_required_samples,
        "static/max_concurrent_samples": 64,
    }
    queue_statistics = {
        "queue_size": 0,
        "total_produced": episodes,
        "total_consumed": episodes,
        "dropped_samples": 0,
        "max_queue_size": max_required_samples,
        "closed": True,
        "real_enqueued": episodes,
        "real_consumed": episodes,
        "real_evicted": 0,
        "real_cleared": 0,
        "real_resident": 0,
        "control_signals_enqueued": 0,
        "last_dequeue_residence_s": 0.01,
    }

    def snapshot() -> dict:
        return {
            "timestamp": "2026-08-18T00:00:01+00:00",
            "trainer": {
                "timestamp": "2026-08-18T00:00:01+00:00",
                "available": True,
                "statistics": json.loads(json.dumps(trainer_statistics)),
            },
            "rollouter": {
                "timestamp": "2026-08-18T00:00:01+00:00",
                "available": True,
                "statistics": dict(rollouter_statistics),
            },
            "queue": {
                "timestamp": "2026-08-18T00:00:01+00:00",
                "available": True,
                "statistics": dict(queue_statistics),
            },
        }

    runtime_receipt = {
        "schema_version": 1,
        "outcome": "success",
        "status": "completed",
        "exception": None,
        "timestamps": {
            "run_started_at": "2026-08-18T00:00:00+00:00",
            "finalization_started_at": "2026-08-18T00:00:01+00:00",
            "finalized_at": "2026-08-18T00:00:02+00:00",
        },
        "snapshots": {"before_clear": snapshot(), "after_clear": snapshot()},
        "queue_conservation": {
            "before_clear": True,
            "after_clear": True,
            "clear_delta_matches_resident": True,
        },
        "finalization_errors": [],
        "trainer_step": publication_cycles,
    }
    runtime_path = run_dir / "native-runtime-receipt.json"
    runtime_path.write_text(
        json.dumps(
            {"step": publication_cycles, "data": runtime_receipt},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

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
        "schema": "amg_verl_fully_async_launch_receipt_v4",
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
            "native_receipt": str(runtime_path),
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
        "runtime_path": runtime_path,
        "metrics_path": metrics_path,
        "resolved_path": resolved_path,
        "hydra_path": hydra_dir / "config.yaml",
        "rollout_dir": rollout_dir,
        "checkpoint_root": run_dir / "checkpoints",
        "launch_path": launch_path,
    }


def mutate_json(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
