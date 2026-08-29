from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentmemorygym_verl.config_contract import (
    inspect_schedule,
    verify_resolved_config,
)
from agentmemorygym_verl.routes import canonical_policy_framing_sha256


def _endpoint_identity(role: str) -> dict:
    return {
        "expected_manifest_sha256": "a" * 64,
        "expected_release_revision": "b" * 40,
        "expected_outer_commit": "c" * 40,
        "expected_inner_commit": "d" * 40,
        "expected_role": role,
        "expected_executor_runtime_digest": "sha256:" + "e" * 64,
        "expected_materializer_sha256": "f" * 64,
        "expected_actions_sha256": "1" * 64,
        "expected_max_observation_tokens": 8192,
    }


def _budget(mode: str) -> dict:
    formal = mode == "formal"
    return {
        "schema": "amg_verl_publication_budget_contract_v1",
        "mode": mode,
        "role": "train_pool" if formal else "gate_only",
        "publication_cycles": 100 if formal else 1,
        "trigger_parameter_sync_step": 1,
        "optimizer_updates": 100 if formal else 1,
        "samples_per_update": 64,
        "episodes": 6400 if formal else 64,
        "save_freq": 10 if formal else 1,
        "max_actor_ckpt_to_keep": 1,
        "max_critic_ckpt_to_keep": 1,
        "model_path": "/models/Qwen3.5-4B",
        "task_count": 762 if formal else 64,
        "source_family_count": 664 if formal else 64,
        "schedule_sha256": "2" * 64,
        "manifest_sha256": "a" * 64,
        "routing_sha256": "3" * 64,
    }


def _verify(config: dict, *, mode: str) -> dict:
    return verify_resolved_config(config, mode=mode, expected_budget=_budget(mode))


def _config(*, mode: str = "formal") -> dict:
    formal = mode == "formal"
    endpoint_identity = _endpoint_identity("train_pool" if formal else "gate_only")
    trainer_gpus = 6 if formal else 4
    rollout_gpus = 2 if formal else 4
    ppo_mini_batch_size = 510 if formal else 512
    return {
        "actor_rollout_ref": {
            "hybrid_engine": False,
            "agentgym": {
                "task_name": "openmle_fast",
                "env_addr": "http://127.0.0.1:65525",
                "max_rounds": 30,
                "max_observation_tokens": 8192,
                "timeout": 240,
                "max_retries": 2,
                **endpoint_identity,
            },
            "model": {
                "path": "/models/Qwen3.5-4B",
                "enable_gradient_checkpointing": True,
                "use_fused_kernels": False,
                "fused_kernel_options": {"impl_backend": "torch"},
            },
            "actor": {
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": 8,
                "ppo_max_token_len_per_gpu": 65536,
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
            "ppo_max_token_len_per_gpu": 65536,
            "ppo_infer_max_token_len_per_gpu": 32768,
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
                "path": "/models/Qwen3.5-4B",
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
            "rollout_correction": {
                "bypass_mode": True,
                "loss_type": "ppo_clip",
            },
        },
        "data": {
            "train_files": ["/run/schedule.jsonl"],
            "val_files": ["/run/schedule.jsonl"],
            "train_batch_size": 0,
            "gen_batch_size": 1,
            "shuffle": False,
            "seed": 233,
            "max_prompt_length": 30720,
            "max_response_length": 2048,
            "return_raw_chat": True,
            "custom_cls": {
                "path": "pkg://agentmemorygym_verl.dataset",
                "name": "AMGTrajectoryDataset",
            },
            "apply_chat_template_kwargs": {"enable_thinking": False},
            "agentgym": {
                "task_name": "openmle_fast",
                "env_addr": "http://127.0.0.1:65525",
                "max_rounds": 30,
                "max_observation_tokens": 8192,
                "timeout": 240,
                "max_retries": 2,
                **endpoint_identity,
            },
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
            "rollout_data_dir": "/run/rollout_data",
        },
        "rollout": {
            "nnodes": 1,
            "n_gpus_per_node": rollout_gpus,
            "n": 1,
            "total_rollout_steps": 6400 if formal else 64,
        },
    }


def _write_schedule(
    path: Path,
    count: int = 6400,
    *,
    role: str = "train_pool",
    task_count: int = 762,
) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for position in range(count):
            row = {
                "data_idx": position % task_count,
                "extra_info": {
                    "index": position % task_count,
                    "manifest_digest": "a" * 64,
                    "panel_id": f"current-publication-{role}",
                    "role": role,
                    "schedule_position": position,
                    "schedule_repetition": position // task_count,
                },
                "item_id": f"openmlefast_formal_{position:06d}",
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestAMGFullyAsyncConfigContract(unittest.TestCase):
    def test_formal100_budget_and_topology(self):
        report = _verify(_config(mode="formal"), mode="formal")
        self.assertEqual(report["optimizer_updates"], 100)
        self.assertEqual(report["publication_cycles"], 100)
        self.assertEqual(report["episodes"], 6400)
        self.assertEqual(report["samples_per_update"], 64)
        self.assertEqual(report["trainer_gpus"], 6)
        self.assertEqual(report["standalone_rollout_gpus"], 2)
        self.assertEqual(report["actor_train_token_budget"], 65536)
        self.assertEqual(report["critic_train_token_budget"], 65536)
        self.assertEqual(report["critic_infer_token_budget"], 32768)
        self.assertEqual(
            report["gradient_checkpointing"], {"actor": True, "critic": True}
        )
        self.assertEqual(
            report["fsdp2_reshard_after_forward"],
            {"actor": True, "critic": True},
        )

    def test_formal_rejects_every_topology_except_six_plus_two(self):
        config = _config(mode="formal")
        config["trainer"]["n_gpus_per_node"] = 4
        config["rollout"]["n_gpus_per_node"] = 4
        config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 512
        config["critic"]["ppo_mini_batch_size"] = 512
        config["async_training"]["require_batches"] = 64 / 512
        with self.assertRaisesRegex(ValueError, r"formal.*6\+2"):
            _verify(config, mode="formal")

    def test_accepts_explicit_paired_learner_training_budget(self):
        config = _config(mode="formal")
        config["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] = 131_072
        config["critic"]["ppo_max_token_len_per_gpu"] = 131_072
        report = verify_resolved_config(
            config,
            mode="formal",
            expected_budget=_budget("formal"),
            expected_actor_train_token_budget=131_072,
            expected_critic_train_token_budget=131_072,
        )
        self.assertEqual(report["actor_train_token_budget"], 131_072)
        self.assertEqual(report["critic_train_token_budget"], 131_072)

    def test_rejects_actor_training_budget_drift(self):
        config = _config(mode="formal")
        config["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] = 32_768
        with self.assertRaisesRegex(
            ValueError, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"
        ):
            _verify(config, mode="formal")

    def test_rejects_critic_training_budget_drift(self):
        config = _config(mode="formal")
        config["critic"]["ppo_max_token_len_per_gpu"] = 32768
        with self.assertRaisesRegex(ValueError, "critic.ppo_max_token_len_per_gpu"):
            _verify(config, mode="formal")

    def test_rejects_critic_inference_budget_drift(self):
        config = _config(mode="formal")
        config["critic"]["ppo_infer_max_token_len_per_gpu"] = 65536
        with self.assertRaisesRegex(
            ValueError, "critic.ppo_infer_max_token_len_per_gpu"
        ):
            _verify(config, mode="formal")

    def test_actor_only_fused_six_plus_two_is_resolved_and_reported(self):
        config = _config(mode="gate")
        config["trainer"]["n_gpus_per_node"] = 6
        config["rollout"]["n_gpus_per_node"] = 2
        config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 510
        config["critic"]["ppo_mini_batch_size"] = 510
        config["async_training"]["require_batches"] = 64 / 510
        config["actor_rollout_ref"]["model"]["use_fused_kernels"] = True

        report = _verify(config, mode="gate")

        self.assertEqual(report["trainer_gpus"], 6)
        self.assertEqual(report["standalone_rollout_gpus"], 2)
        self.assertEqual(
            report["fused_kernels"],
            {"actor": True, "critic": False, "impl_backend": "torch"},
        )

    def test_rejects_non_boolean_fused_kernel_selection(self):
        config = _config()
        config["critic"]["model"]["use_fused_kernels"] = "false"
        with self.assertRaisesRegex(ValueError, "critic use_fused_kernels"):
            _verify(config, mode="formal")

    def test_one_update_budget(self):
        report = _verify(_config(mode="gate"), mode="gate")
        self.assertEqual(report["optimizer_updates"], 1)
        self.assertEqual(report["episodes"], 64)

    def test_rejects_grpo_or_missing_critic(self):
        config = _config()
        config["algorithm"]["adv_estimator"] = "grpo"
        config["critic"]["enable"] = False
        with self.assertRaisesRegex(ValueError, "action-axis GAE"):
            _verify(config, mode="formal")

    def test_rejects_unwhitened_action_axis_advantages(self):
        config = _config()
        config["algorithm"]["amg_advantage_normalization"] = "none"
        with self.assertRaisesRegex(ValueError, "amg_advantage_normalization"):
            _verify(config, mode="formal")

    def test_rejects_half_async_or_validation(self):
        config = _config()
        config["async_training"]["use_dynamic_resource_scheduling"] = False
        config["trainer"]["val_before_train"] = True
        with self.assertRaisesRegex(ValueError, "dynamic Hybrid.*Standalone"):
            _verify(config, mode="formal")

    def test_rejects_wrong_publication_math(self):
        config = _config()
        config["trainer"]["total_training_steps"] = 25
        with self.assertRaisesRegex(ValueError, "publication cycles"):
            _verify(config, mode="formal")

    def test_rejects_checkpoint_retention_drift(self):
        config = _config()
        config["trainer"]["max_actor_ckpt_to_keep"] = 3
        with self.assertRaisesRegex(ValueError, "max_actor_ckpt_to_keep"):
            _verify(config, mode="formal")

    def test_rejects_legacy_continuous_token_config(self):
        config = _config()
        config["data"]["continuous_token"] = {
            "enable": True,
            "model_family": "qwen35",
        }
        with self.assertRaisesRegex(ValueError, "legacy data.continuous_token"):
            _verify(config, mode="formal")

    def test_rejects_retired_noop_evidence_config(self):
        for field, value in (
            ("runtime_receipt_path", "/run/native-runtime-receipt.json"),
            ("rollout_data_non_tensor_keys", ["step_record_json"]),
            ("rollout_data_non_tensor_max_keys", 1),
            ("parameter_update_probe", {"enabled": True}),
        ):
            with self.subTest(field=field):
                config = _config(mode="gate")
                config["async_training"][field] = value
                with self.assertRaisesRegex(ValueError, "legacy no-op async evidence"):
                    _verify(config, mode="gate")

    def test_rejects_non_amg_agent_loop(self):
        config = _config()
        config["actor_rollout_ref"]["rollout"]["agent"]["default_agent_loop"] = (
            "single_turn_agent"
        )
        with self.assertRaisesRegex(ValueError, "default_agent_loop"):
            _verify(config, mode="formal")

    def test_rejects_padding_unsafe_loss_or_prefix_grouping(self):
        mutations = (
            (("actor_rollout_ref", "actor"), "loss_agg_mode", "seq-mean-token-mean"),
            (("actor_rollout_ref", "actor"), "use_prefix_grouper", True),
            (("critic",), "loss_agg_mode", "seq-mean-token-mean"),
        )
        for path, key, wrong in mutations:
            with self.subTest(path=path, key=key):
                config = _config()
                target = config
                for component in path:
                    target = target[component]
                target[key] = wrong
                with self.assertRaisesRegex(ValueError, key):
                    _verify(config, mode="formal")

    def test_rejects_fsdp2_reshard_drift_from_upstream_default(self):
        for path, wrong in (
            (("actor_rollout_ref", "actor", "fsdp_config"), False),
            (("critic", "fsdp"), False),
        ):
            with self.subTest(path=path):
                config = _config()
                target = config
                for key in path:
                    target = target[key]
                target["reshard_after_forward"] = wrong
                with self.assertRaisesRegex(ValueError, "reshard_after_forward"):
                    _verify(config, mode="formal")

    def test_rejects_disabling_upstream_gradient_checkpointing(self):
        for role in ("actor_rollout_ref", "critic"):
            with self.subTest(role=role):
                config = _config()
                config[role]["model"]["enable_gradient_checkpointing"] = False
                with self.assertRaisesRegex(
                    ValueError, "enable_gradient_checkpointing"
                ):
                    _verify(config, mode="formal")

    def test_rejects_native_qwen_thinking_that_breaks_bare_action_contract(self):
        config = _config()
        config["data"]["apply_chat_template_kwargs"]["enable_thinking"] = True
        with self.assertRaisesRegex(ValueError, "enable_thinking"):
            _verify(config, mode="formal")

    def test_rejects_vllm_engine_residue_after_sglang_cutover(self):
        config = _config(mode="formal")
        config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"] = {
            "gdn_prefill_backend": "triton"
        }
        with self.assertRaisesRegex(ValueError, "must not retain vLLM"):
            _verify(config, mode="formal")

    def test_rejects_buffered_mamba_scheduler_for_qwen35(self):
        config = _config(mode="formal")
        config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["sglang"][
            "mamba_scheduler_strategy"
        ] = "extra_buffer"
        with self.assertRaisesRegex(ValueError, "mamba_scheduler_strategy"):
            _verify(config, mode="formal")

    def test_accepts_formal400_route_registry_without_global_route_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_ids = ("webshop", "swesmith", "literesearcher", "openmle_fast")
            payload = {
                "schema": "amg_route_registry_v1",
                "agent_name": "amg_task_neutral_async",
                "routes": [
                    {
                        "route_id": route_id,
                        "max_rounds": 30,
                        "max_observation_tokens": 8192,
                        "policy_framing_sha256": canonical_policy_framing_sha256(
                            [{"role": "system", "content": route_id}]
                        ),
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
            registry = root / "routes.json"
            registry.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            registry_sha256 = hashlib.sha256(registry.read_bytes()).hexdigest()
            agentgym = {
                "route_registry_path": str(registry),
                "route_registry_sha256": registry_sha256,
                "route_registry_expected_ids": list(route_ids),
            }
            config = _config(mode="formal")
            config["actor_rollout_ref"]["agentgym"] = copy.deepcopy(agentgym)
            config["data"]["agentgym"] = copy.deepcopy(agentgym)
            config["trainer"]["total_training_steps"] = 400
            config["rollout"]["total_rollout_steps"] = 25_600
            budget = _budget("formal")
            budget.update(
                publication_cycles=400,
                optimizer_updates=400,
                episodes=25_600,
                route_registry_sha256=registry_sha256,
                route_ids=list(route_ids),
            )

            report = verify_resolved_config(
                config, mode="formal", expected_budget=budget
            )

            self.assertEqual(report["optimizer_updates"], 400)
            self.assertEqual(report["episodes"], 25_600)
            self.assertEqual(report["route_ids"], list(route_ids))
            self.assertEqual(report["route_registry_sha256"], registry_sha256)
            self.assertIsNone(report["env_addr"])

    def test_accepts_formal100_single_route_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_ids = ("literesearcher",)
            payload = {
                "schema": "amg_route_registry_v1",
                "agent_name": "amg_task_neutral_async",
                "routes": [
                    {
                        "route_id": "literesearcher",
                        "max_rounds": 30,
                        "max_observation_tokens": 8192,
                        "policy_framing_sha256": canonical_policy_framing_sha256(
                            [{"role": "system", "content": "literesearcher"}]
                        ),
                        "route_attestation_sha256": "1" * 64,
                        "client": {
                            "task_name": "literesearcher",
                            "env_addr": "http://127.0.0.1:65122",
                            "timeout": 240,
                            "max_retries": 2,
                        },
                    }
                ],
            }
            registry = root / "routes.json"
            registry.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            registry_sha256 = hashlib.sha256(registry.read_bytes()).hexdigest()
            agentgym = {
                "route_registry_path": str(registry),
                "route_registry_sha256": registry_sha256,
                "route_registry_expected_ids": list(route_ids),
            }
            config = _config(mode="formal")
            config["actor_rollout_ref"]["agentgym"] = copy.deepcopy(agentgym)
            config["data"]["agentgym"] = copy.deepcopy(agentgym)
            budget = _budget("formal")
            budget.update(
                route_registry_sha256=registry_sha256,
                route_ids=list(route_ids),
            )

            report = verify_resolved_config(
                config, mode="formal", expected_budget=budget
            )

            self.assertEqual(report["optimizer_updates"], 100)
            self.assertEqual(report["episodes"], 6400)
            self.assertEqual(report["route_ids"], ["literesearcher"])
            self.assertEqual(report["route_registry_sha256"], registry_sha256)

    def test_multitask_registry_rejects_legacy_global_route_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_ids = ("webshop", "swesmith", "literesearcher", "openmle_fast")
            payload = {
                "schema": "amg_route_registry_v1",
                "agent_name": "amg_task_neutral_async",
                "routes": [
                    {
                        "route_id": route_id,
                        "max_rounds": 30,
                        "max_observation_tokens": 8192,
                        "policy_framing_sha256": "a" * 64,
                        "route_attestation_sha256": str(index + 1) * 64,
                        "client": {
                            "task_name": route_id,
                            "env_addr": f"http://127.0.0.1:{65101 + index}",
                        },
                    }
                    for index, route_id in enumerate(route_ids)
                ],
            }
            registry = root / "routes.json"
            registry.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            digest = hashlib.sha256(registry.read_bytes()).hexdigest()
            agentgym = {
                "route_registry_path": str(registry),
                "route_registry_sha256": digest,
                "route_registry_expected_ids": list(route_ids),
                "env_addr": "http://127.0.0.1:65525",
            }
            config = _config(mode="formal")
            config["actor_rollout_ref"]["agentgym"] = copy.deepcopy(agentgym)
            config["data"]["agentgym"] = copy.deepcopy(agentgym)
            config["trainer"]["total_training_steps"] = 400
            config["rollout"]["total_rollout_steps"] = 25_600
            budget = _budget("formal")
            budget.update(
                publication_cycles=400,
                optimizer_updates=400,
                episodes=25_600,
                route_registry_sha256=digest,
                route_ids=list(route_ids),
            )
            with self.assertRaisesRegex(ValueError, "global agentgym.env_addr"):
                verify_resolved_config(config, mode="formal", expected_budget=budget)

    def test_rejects_multi_turn_disabled_before_upstream_stamps_single_turn(self):
        config = _config()
        config["actor_rollout_ref"]["rollout"]["multi_turn"]["enable"] = False
        with self.assertRaisesRegex(ValueError, "multi_turn.enable"):
            _verify(config, mode="formal")

    def test_rejects_actor_data_agentgym_drift(self):
        config = _config()
        config["data"]["agentgym"]["max_rounds"] = 31
        with self.assertRaisesRegex(ValueError, "agentgym configs must match"):
            _verify(config, mode="formal")

    def test_budget_is_derived_not_fixed_to_one_publication(self):
        config = _config(mode="formal")
        config["trainer"]["total_training_steps"] = 8
        config["rollout"]["total_rollout_steps"] = 512
        budget = _budget("formal")
        budget.update(
            publication_cycles=8,
            optimizer_updates=8,
            episodes=512,
            task_count=70,
            source_family_count=65,
            schedule_sha256="4" * 64,
        )
        report = verify_resolved_config(config, mode="formal", expected_budget=budget)
        self.assertEqual(report["optimizer_updates"], 8)
        self.assertEqual(report["episodes"], 512)
        self.assertEqual(report["task_count"], 70)


class TestAMGScheduleContract(unittest.TestCase):
    def test_formal_and_gate_are_distinct_frozen_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "formal.jsonl"
            source_sha = _write_schedule(source)
            full = inspect_schedule(
                source,
                expected_count=6400,
                expected_sha256=source_sha,
                expected_role="train_pool",
            )
            self.assertEqual(full["count"], 6400)
            self.assertEqual(full["manifest_digest"], "a" * 64)
            self.assertEqual(full["unique_item_ids"], 6400)

            gate = Path(directory) / "gate.jsonl"
            gate_sha = _write_schedule(gate, count=64, role="gate_only", task_count=64)
            inspected = inspect_schedule(
                gate,
                expected_count=64,
                expected_sha256=gate_sha,
                expected_role="gate_only",
            )
            self.assertEqual(inspected["last_schedule_position"], 63)
            self.assertEqual(inspected["role"], "gate_only")

            with self.assertRaisesRegex(ValueError, "is not gate_only"):
                inspect_schedule(source, expected_role="gate_only")

    def test_rejects_duplicate_item(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            _write_schedule(path, count=3)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["item_id"] = rows[0]["item_id"]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate item_id"):
                inspect_schedule(path, expected_count=3)

    def test_multienvironment_schedule_uses_unique_global_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multienv.jsonl"
            rows = []
            for position, route_id in enumerate(("webshop", "swesmith")):
                rows.append(
                    {
                        "index": position,
                        "data_idx": 7,
                        "route_id": route_id,
                        "extra_info": {
                            "index": position,
                            "route_id": route_id,
                            "manifest_digest": "a" * 64,
                            "panel_id": "multienv-train",
                            "role": "train_pool",
                            "schedule_position": position,
                        },
                        "item_id": f"{route_id}:task-7:occurrence-0",
                    }
                )
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = inspect_schedule(path, expected_count=2)
            self.assertEqual(report["unique_global_indices"], 2)

            rows[1]["extra_info"]["index"] = 0
            rows[1]["index"] = 0
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "global index"):
                inspect_schedule(path, expected_count=2)

    def test_multienvironment_schedule_reports_route_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multienv.jsonl"
            route_ids = ("webshop", "swesmith", "literesearcher", "openmle_fast")
            registry_sha256 = "f" * 64
            rows = []
            for position in range(8):
                route_id = route_ids[position % 4]
                rows.append(
                    {
                        "index": position,
                        "data_idx": position // 4,
                        "route_id": route_id,
                        "data_source": route_id,
                        "agent_name": "amg_task_neutral_async",
                        "extra_info": {
                            "index": position,
                            "route_id": route_id,
                            "manifest_digest": "a" * 64,
                            "panel_id": "multienv-train",
                            "role": "train_pool",
                            "schedule_position": position,
                            "route_registry_sha256": registry_sha256,
                            "route_attestation_sha256": str(
                                route_ids.index(route_id) + 1
                            )
                            * 64,
                            "source_schedule_sha256": str(route_ids.index(route_id) + 5)
                            * 64,
                            "source_manifest_digest": "9abc"[route_ids.index(route_id)]
                            * 64,
                            "source_panel_id": f"source-{route_id}",
                        },
                        "item_id": f"{route_id}:task-{position}",
                    }
                )
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = inspect_schedule(
                path,
                expected_count=8,
                expected_route_ids=route_ids,
                expected_route_registry_sha256=registry_sha256,
            )

            self.assertEqual(report["route_order"], list(route_ids))
            self.assertEqual(
                report["per_route_counts"], {route_id: 2 for route_id in route_ids}
            )
            self.assertEqual(report["route_registry_sha256"], registry_sha256)
            self.assertEqual(report["agent_name"], "amg_task_neutral_async")
            self.assertEqual(
                report["per_route_provenance"]["webshop"],
                {
                    "route_attestation_sha256": "1" * 64,
                    "source_schedule_sha256": "5" * 64,
                    "source_manifest_digest": "9" * 64,
                    "source_panel_id": "source-webshop",
                },
            )

            rows[5]["route_id"] = "webshop"
            rows[5]["extra_info"]["route_id"] = "webshop"
            rows[5]["data_source"] = "webshop"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "route order"):
                inspect_schedule(path, expected_route_ids=route_ids)

    def test_rejects_schedule_position_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "position.jsonl"
            _write_schedule(path, count=3)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[2]["extra_info"]["schedule_position"] = 99
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "schedule_position"):
                inspect_schedule(path, expected_count=3)

    def test_verification_does_not_mutate_config(self):
        config = _config()
        before = copy.deepcopy(config)
        _verify(config, mode="formal")
        self.assertEqual(config, before)


if __name__ == "__main__":
    unittest.main()
