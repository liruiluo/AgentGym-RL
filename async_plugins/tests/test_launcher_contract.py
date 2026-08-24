from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
import zipfile
from pathlib import Path
from unittest import mock

from agentmemorygym_verl.config_contract import inspect_schedule
from agentmemorygym_verl.identity import (
    EXPECTED_VERL_COMMIT,
    TRL_WHEEL_RELATIVE_PATH,
    TRL_WHEEL_SHA256,
)
from agentmemorygym_verl.launch import (
    LaunchInputs,
    _load_endpoint_identity,
    _load_launch_identity,
    _load_multitask_identity,
    _load_multitask_orchestrator_preflight,
    _parse_args,
    _partition_selected_file_hashes,
    _preserve_legacy_runtime_preflight_fields,
    _require_exact_multitask_outer_commit,
    _validate_accelerator_runtime,
    build_overrides,
    build_runtime_env,
    main as launch_main,
)
from agentmemorygym_verl.routes import canonical_policy_framing_sha256

FIXTURES = Path("/tmp/openmle-v8-launch-fixtures-20260818")


class TestAMGFullyAsyncLauncherContract(unittest.TestCase):
    def setUp(self) -> None:
        required = (
            "source-lock.json",
            "publication-receipt.json",
            "formal100-schedule-certificate.json",
            "launcher_contract.py",
            "g64-gate-single-pass.jsonl",
            "formal100-schedule.jsonl",
        )
        for filename in required:
            self.assertTrue((FIXTURES / filename).is_file(), filename)
        self.source_lock = json.loads(
            (FIXTURES / "source-lock.json").read_text(encoding="utf-8")
        )

    def _inputs(self, root: Path, mode: str = "formal") -> LaunchInputs:
        schedule = (
            FIXTURES / "formal100-schedule.jsonl"
            if mode == "formal"
            else FIXTURES / "g64-gate-single-pass.jsonl"
        )
        return LaunchInputs(
            mode=mode,
            verl_root=root / "verl",
            outer_root=root / "AgentGym-RL",
            schedule=schedule,
            env_addr="http://127.0.0.1:65525",
            run_dir=root / "run",
            experiment_name=f"current-publication-{mode}",
            endpoint_source_lock=FIXTURES / "source-lock.json",
            endpoint_contract_tool=FIXTURES / "launcher_contract.py",
            publication_receipt=FIXTURES / "publication-receipt.json",
            formal_schedule_certificate=FIXTURES
            / "formal100-schedule-certificate.json",
        )

    @staticmethod
    def _values(overrides: list[str]) -> dict[str, str]:
        return {
            key.lstrip("+"): value
            for key, value in (item.split("=", 1) for item in overrides)
        }

    def _identity(self, root: Path, mode: str) -> tuple[LaunchInputs, dict]:
        inputs = self._inputs(root, mode=mode)
        role = "gate_only" if mode == "gate" else "train_pool"
        schedule = inspect_schedule(inputs.schedule, expected_role=role)
        return inputs, _load_endpoint_identity(inputs, schedule_report=schedule)

    def test_formal_overrides_reuse_upstream_fully_async_ppo_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, identity = self._identity(Path(directory), "formal")
            budget = identity["budget_contract"]
            values = self._values(
                build_overrides(
                    inputs,
                    effective_schedule=inputs.schedule,
                    endpoint_client_config=identity["client_config"],
                    budget_contract=budget,
                    training_runtime=identity["training_runtime"],
                )
            )

            self.assertEqual(values["algorithm.adv_estimator"], "amg_action_axis_gae")
            self.assertEqual(
                values["algorithm.amg_advantage_normalization"],
                "upstream_masked_whiten",
            )
            self.assertEqual(
                values["algorithm.rollout_correction.loss_type"], "ppo_clip"
            )
            self.assertEqual(values["data.max_prompt_length"], "30720")
            self.assertEqual(values["actor_rollout_ref.rollout.n"], "1")
            self.assertEqual(
                values["actor_rollout_ref.rollout.name"],
                "sglang",
            )
            self.assertEqual(
                values[
                    "actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy"
                ],
                "no_buffer",
            )
            self.assertEqual(
                values[
                    "actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache"
                ],
                "True",
            )
            self.assertFalse(
                any("engine_kwargs.vllm" in key for key in values),
                values,
            )
            self.assertEqual(values["critic.enable"], "True")
            self.assertEqual(values["trainer.n_gpus_per_node"], "6")
            self.assertEqual(values["rollout.n_gpus_per_node"], "2")
            self.assertEqual(
                values["actor_rollout_ref.actor.loss_agg_mode"], "token-mean"
            )
            self.assertEqual(
                values["actor_rollout_ref.actor.use_prefix_grouper"], "False"
            )
            self.assertEqual(values["critic.loss_agg_mode"], "token-mean")
            self.assertEqual(
                values["ray_kwargs.ray_init.object_store_memory"], "8589934592"
            )
            self.assertEqual(
                values["trainer.total_training_steps"],
                str(budget["publication_cycles"]),
            )
            self.assertEqual(
                values["rollout.total_rollout_steps"], str(budget["episodes"])
            )
            self.assertEqual(
                values["async_training.trigger_parameter_sync_step"],
                str(budget["trigger_parameter_sync_step"]),
            )
            self.assertEqual(
                float(values["async_training.require_batches"]),
                budget["samples_per_update"] / 510,
            )
            self.assertEqual(values["trainer.val_before_train"], "False")
            self.assertEqual(values["trainer.test_freq"], "-1")
            self.assertEqual(values["trainer.resume_mode"], "disable")
            self.assertEqual(values["trainer.max_actor_ckpt_to_keep"], "1")
            self.assertEqual(values["trainer.max_critic_ckpt_to_keep"], "1")
            self.assertEqual(values["trainer.logger"], "[console,file]")
            self.assertEqual(values["actor_rollout_ref.hybrid_engine"], "False")
            self.assertEqual(
                values["data.apply_chat_template_kwargs.enable_thinking"], "False"
            )
            self.assertFalse(
                any(key.startswith("data.continuous_token") for key in values),
                values,
            )
            self.assertFalse(
                any(
                    key.startswith(
                        (
                            "async_training.runtime_receipt_path",
                            "async_training.rollout_data_non_tensor",
                            "async_training.parameter_update_probe",
                        )
                    )
                    for key in values
                ),
                values,
            )
            self.assertEqual(
                values["actor_rollout_ref.model.enable_gradient_checkpointing"],
                "True",
            )
            self.assertEqual(
                values["critic.model.enable_gradient_checkpointing"], "True"
            )
            self.assertEqual(
                values["actor_rollout_ref.actor.fsdp_config.reshard_after_forward"],
                "True",
            )
            self.assertEqual(values["critic.fsdp.reshard_after_forward"], "True")
            self.assertEqual(
                values["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"],
                "65536",
            )
            self.assertEqual(values["critic.ppo_max_token_len_per_gpu"], "65536")
            self.assertEqual(values["critic.ppo_infer_max_token_len_per_gpu"], "32768")
            self.assertEqual(
                values["actor_rollout_ref.rollout.multi_turn.enable"], "True"
            )
            self.assertEqual(
                values["actor_rollout_ref.rollout.agent.default_agent_loop"],
                "amg_task_neutral_async",
            )
            self.assertEqual(
                json.loads(values["data.agentgym.expected_role"]), "train_pool"
            )

    def test_actor_only_fused_six_plus_two_uses_upstream_native_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, identity = self._identity(Path(directory), "gate")
            inputs = replace(
                inputs,
                trainer_gpus=6,
                standalone_rollout_gpus=2,
                actor_use_fused_kernels=True,
                critic_use_fused_kernels=False,
            )
            values = self._values(
                build_overrides(
                    inputs,
                    effective_schedule=inputs.schedule,
                    endpoint_client_config=identity["client_config"],
                    budget_contract=identity["budget_contract"],
                    training_runtime=identity["training_runtime"],
                )
            )
            self.assertEqual(values["trainer.n_gpus_per_node"], "6")
            self.assertEqual(values["rollout.n_gpus_per_node"], "2")
            self.assertEqual(
                values["actor_rollout_ref.actor.ppo_mini_batch_size"], "510"
            )
            self.assertEqual(values["critic.ppo_mini_batch_size"], "510")
            self.assertAlmostEqual(
                float(values["async_training.require_batches"]),
                identity["budget_contract"]["samples_per_update"] / 510,
                places=11,
            )
            self.assertEqual(
                int(values["actor_rollout_ref.actor.ppo_mini_batch_size"])
                * float(values["async_training.require_batches"]),
                identity["budget_contract"]["samples_per_update"],
            )
            self.assertEqual(
                int(values["actor_rollout_ref.actor.ppo_mini_batch_size"]) % 6,
                0,
            )
            self.assertEqual(
                values["actor_rollout_ref.model.use_fused_kernels"], "True"
            )
            self.assertEqual(values["critic.model.use_fused_kernels"], "False")
            self.assertEqual(
                values["actor_rollout_ref.model.fused_kernel_options.impl_backend"],
                "torch",
            )
            self.assertEqual(
                values["critic.model.fused_kernel_options.impl_backend"], "torch"
            )

    def test_actor_and_critic_keep_upstream_fsdp2_reshard_default(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, identity = self._identity(Path(directory), "formal")
            values = self._values(
                build_overrides(
                    inputs,
                    effective_schedule=inputs.schedule,
                    endpoint_client_config=identity["client_config"],
                    budget_contract=identity["budget_contract"],
                    training_runtime=identity["training_runtime"],
                )
            )

            self.assertEqual(
                values["actor_rollout_ref.actor.fsdp_config.reshard_after_forward"],
                "True",
            )
            self.assertEqual(values["critic.fsdp.reshard_after_forward"], "True")

    def test_gate_role_and_budget_are_derived_from_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, identity = self._identity(Path(directory), "gate")
            budget = identity["budget_contract"]
            values = self._values(
                build_overrides(
                    inputs,
                    effective_schedule=inputs.schedule,
                    endpoint_client_config=identity["client_config"],
                    budget_contract=budget,
                    training_runtime=identity["training_runtime"],
                )
            )
            manifest = self.source_lock["integration"]["manifests"]["gate_only"]
            self.assertEqual(identity["task_count"], manifest["task_count"])
            self.assertEqual(
                identity["source_family_count"], manifest["source_family_count"]
            )
            self.assertEqual(identity["schedule_count"], budget["episodes"])
            self.assertEqual(identity["client_config"]["expected_role"], "gate_only")
            self.assertEqual(values["trainer.total_training_steps"], "1")
            self.assertEqual(values["async_training.trigger_parameter_sync_step"], "1")
            self.assertEqual(
                values["rollout.total_rollout_steps"], str(budget["episodes"])
            )

    def test_formal_identity_matches_selected_publication_without_dated_literals(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, identity = self._identity(Path(directory), "formal")
            manifest = self.source_lock["integration"]["manifests"]["train_pool"]
            routing = self.source_lock["integration"]["routing"]["train_pool"]
            self.assertEqual(identity["task_count"], manifest["task_count"])
            self.assertEqual(
                identity["source_family_count"], manifest["source_family_count"]
            )
            self.assertEqual(identity["manifest_sha256"], manifest["sha256"])
            self.assertEqual(identity["routing_sha256"], routing["sha256"])
            self.assertEqual(
                identity["training_runtime"], self.source_lock["training_runtime"]
            )
            self.assertEqual(
                identity["schedule_sha256"], inspect_schedule(inputs.schedule)["sha256"]
            )

    def test_publication_identity_rejects_role_or_schedule_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._inputs(Path(directory), mode="gate")
            formal = inspect_schedule(
                FIXTURES / "formal100-schedule.jsonl", expected_role="train_pool"
            )
            with self.assertRaisesRegex(ValueError, "schedule role mismatch"):
                _load_endpoint_identity(inputs, schedule_report=formal)

            gate = inspect_schedule(inputs.schedule, expected_role="gate_only")
            gate["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "schedule digest"):
                _load_endpoint_identity(inputs, schedule_report=gate)

    def test_selected_file_identities_map_to_real_checkout_roots(self):
        selected = self.source_lock["runtime_source"]["selected_files"]
        outer_manifest, inner_manifest = _partition_selected_file_hashes(selected)

        self.assertIn(
            "AgentGym-RL/scripts/agentmemory/analyze_openmle_local_iteration.py",
            outer_manifest,
        )
        self.assertNotIn(
            "scripts/agentmemory/analyze_openmle_local_iteration.py",
            outer_manifest,
        )
        checkout = Path(__file__).resolve().parents[2]
        for relative in outer_manifest:
            self.assertTrue((checkout / relative).is_file(), relative)
        for relative in inner_manifest:
            self.assertTrue((checkout / "AgentGym" / relative).is_file(), relative)

    def test_locked_trl_wheel_is_exact_upstream_verl_extra(self):
        checkout = Path(__file__).resolve().parents[2]
        wheel = checkout / TRL_WHEEL_RELATIVE_PATH
        self.assertTrue(wheel.is_file())
        self.assertFalse(wheel.is_symlink())
        self.assertEqual(
            hashlib.sha256(wheel.read_bytes()).hexdigest(), TRL_WHEEL_SHA256
        )
        with zipfile.ZipFile(wheel) as archive:
            metadata = archive.read("trl-0.9.6.dist-info/METADATA").decode("utf-8")
        self.assertIn("Name: trl\n", metadata)
        self.assertIn("Version: 0.9.6\n", metadata)

    def test_runtime_env_is_closed_and_pins_native_artifact_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._inputs(Path(directory), mode="gate")
            env = build_runtime_env(
                inputs,
                training_runtime=self.source_lock["training_runtime"],
                base_env={"PATH": "/usr/bin"},
            )
            self.assertEqual(
                env["PYTHONPATH"].split(":"),
                [
                    str(
                        inputs.outer_root
                        / "async_plugins"
                        / "vendor"
                        / "trl-0.9.6-py3-none-any.whl"
                    ),
                    str(inputs.outer_root / "async_plugins"),
                    str(inputs.verl_root),
                    str(inputs.outer_root / "AgentGym" / "agentenv"),
                    str(inputs.outer_root / "AgentGym" / "agentenv-openmle-fast"),
                ],
            )
            runtime_bin = str(
                Path(self.source_lock["training_runtime"]["python"]).parent
            )
            self.assertEqual(
                env["PATH"].split(":"),
                ["/dev/shm/cuda-13-b300-toolkit/bin", runtime_bin, "/usr/bin"],
            )
            self.assertEqual(env["CUDA_HOME"], "/dev/shm/cuda-13-b300-toolkit")
            self.assertEqual(env["CUDA_PATH"], "/dev/shm/cuda-13-b300-toolkit")
            self.assertEqual(
                env["LD_LIBRARY_PATH"].split(":"),
                [
                    "/dev/shm/cuda-13-b300-toolkit/lib64",
                    "/usr/local/cuda/lib64/stubs",
                    str(
                        Path(self.source_lock["training_runtime"]["site_packages"])
                        / "nvidia"
                        / "cu13"
                        / "lib"
                    ),
                ],
            )
            self.assertEqual(
                env["VERL_USE_EXTERNAL_MODULES"], "agentmemorygym_verl.action_gae"
            )
            self.assertEqual(
                env["VERL_FILE_LOGGER_PATH"], str(inputs.run_dir / "metrics.jsonl")
            )
            self.assertNotIn("VERL_FULLY_ASYNC_RUNTIME_RECEIPT_PATH", env)
            self.assertEqual(env["RAY_memory_usage_threshold"], "0.98")
            self.assertNotIn("VLLM_USE_V1", env)
            self.assertNotIn("VLLM_LOGGING_LEVEL", env)
            with self.assertRaisesRegex(RuntimeError, "PYTHONPATH"):
                build_runtime_env(
                    inputs,
                    training_runtime=self.source_lock["training_runtime"],
                    base_env={"PYTHONPATH": "/caller"},
                )

    def test_runtime_env_pins_cuda13_toolchain(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._inputs(Path(directory), mode="gate")
            runtime = dict(self.source_lock["training_runtime"])
            env = build_runtime_env(
                inputs,
                training_runtime=runtime,
                base_env={"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/caller/lib"},
            )
            cuda_home = "/dev/shm/cuda-13-b300-toolkit"
            self.assertEqual(env["CUDA_HOME"], cuda_home)
            self.assertEqual(env["CUDA_PATH"], cuda_home)
            self.assertEqual(
                env["PATH"].split(":"),
                [cuda_home + "/bin", str(Path(runtime["python"]).parent), "/usr/bin"],
            )
            self.assertEqual(
                env["LD_LIBRARY_PATH"].split(":"),
                [
                    cuda_home + "/lib64",
                    "/usr/local/cuda/lib64/stubs",
                    runtime["site_packages"] + "/nvidia/cu13/lib",
                    "/caller/lib",
                ],
            )

    def test_accelerator_runtime_must_match_locked_b300_and_cuda13(self):
        training_runtime = dict(self.source_lock["training_runtime"], gpu_type="B300")
        observed = {
            "cuda_home": "/dev/shm/cuda-13-b300-toolkit",
            "cudart_linker_ready": True,
            "cccl_target_ready": True,
            "nvcc_path": "/dev/shm/cuda-13-b300-toolkit/bin/nvcc",
            "nvcc_release": "13.0",
            "torch_cuda": "13.0",
            "torch_cuda_available": True,
            "gpu_count": 8,
            "gpu_names": ["NVIDIA B300 SXM6 AC"] * 8,
        }
        self.assertEqual(
            _validate_accelerator_runtime(observed, training_runtime=training_runtime)[
                "gpu_count"
            ],
            8,
        )
        for field, wrong in (
            ("nvcc_release", "12.8"),
            ("torch_cuda", "12.8"),
            ("gpu_count", 7),
            ("gpu_names", ["NVIDIA B200"] * 8),
        ):
            mutated = dict(observed)
            mutated[field] = wrong
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                _validate_accelerator_runtime(
                    mutated, training_runtime=training_runtime
                )

    def test_cli_has_no_commit_model_or_budget_identity_override(self):
        common = [
            "--mode",
            "gate",
            "--verl-root",
            "/verl",
            "--schedule",
            str(FIXTURES / "g64-gate-single-pass.jsonl"),
            "--env-addr",
            "http://127.0.0.1:65525",
            "--run-dir",
            "/run",
            "--experiment-name",
            "gate",
            "--endpoint-source-lock",
            str(FIXTURES / "source-lock.json"),
            "--endpoint-contract-tool",
            str(FIXTURES / "launcher_contract.py"),
            "--publication-receipt",
            str(FIXTURES / "publication-receipt.json"),
            "--formal-schedule-certificate",
            str(FIXTURES / "formal100-schedule-certificate.json"),
        ]
        parsed = _parse_args(common)
        for name in ("expected_verl_commit", "model_path", "episodes", "task_count"):
            self.assertFalse(hasattr(parsed, name))
        actor_only = _parse_args(common + ["--actor-use-fused-kernels"])
        self.assertTrue(actor_only.actor_use_fused_kernels)
        self.assertFalse(actor_only.critic_use_fused_kernels)
        critic_only = _parse_args(common + ["--critic-use-fused-kernels"])
        self.assertFalse(critic_only.actor_use_fused_kernels)
        self.assertTrue(critic_only.critic_use_fused_kernels)
        for forbidden in (
            ["--expected-verl-commit", "0" * 40],
            ["--model-path", "/tmp/model"],
            ["--episodes", "64"],
            ["--use-fused-kernels"],
        ):
            with self.subTest(forbidden=forbidden), self.assertRaises(SystemExit):
                _parse_args(common + forbidden)

    def test_shell_launcher_selects_python_from_publication(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "launch_amg_fully_async.sh"
        )
        text = script.read_text(encoding="utf-8")
        self.assertIn(".training_runtime.python", text)
        self.assertIn("--endpoint-source-lock", text)
        self.assertIn("--multitask-source-lock", text)
        self.assertIn("PYTHONPATH is an identity conflict", text)
        self.assertIn("trl-0.9.6-py3-none-any.whl", text)
        self.assertIn("libcudart.so.13", text)
        self.assertIn("flashinfer/data/cccl/libcudacxx/include", text)
        self.assertIn('export CPATH="${CUDA13_CCCL_INCLUDE}', text)
        self.assertNotIn("/dev/shm/qwen35-runtime", text)
        self.assertNotIn("${PYTHONPATH:+", text)
        self.assertIn(EXPECTED_VERL_COMMIT, EXPECTED_VERL_COMMIT)

    def test_orchestrator_cutover_covers_sglang_and_drops_vllm_env(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "orchestrate_openmle_fully_async.sh"
        )
        text = script.read_text(encoding="utf-8")
        self.assertIn("SGLang::", text)
        self.assertIn("sglang\\.(launch_server|serve)", text)
        self.assertIn("formal Hybrid + Standalone topology must be 6+2", text)
        self.assertIn("/dev/shm/cuda-13-b300-toolkit", text)
        self.assertIn("foreign Ray/inference-engine residue", text)
        self.assertNotIn("export VLLM_", text)
        self.assertNotIn("PROCESS_OWNER=amg-verl-v090", text)


class TestAMGMultitaskLauncherContract(unittest.TestCase):
    ROUTES = ("webshop", "swesmith", "literesearcher", "openmle_fast")

    @staticmethod
    def _values(overrides: list[str]) -> dict[str, str]:
        return {
            key.lstrip("+"): value
            for key, value in (item.split("=", 1) for item in overrides)
        }

    def _registry(self, root: Path) -> tuple[Path, str]:
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
                for index, route_id in enumerate(self.ROUTES)
            ],
        }
        path = root / "routes.json"
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def _identity_fixture(
        self, root: Path, *, mode: str = "formal"
    ) -> tuple[LaunchInputs, dict, dict]:
        registry, registry_sha256 = self._registry(root)
        updates = 400 if mode == "formal" else 1
        role = "train_pool" if mode == "formal" else "gate_only"
        row_count = updates * 64
        rows_per_route = row_count // len(self.ROUTES)
        schedule_sha256 = "e" * 64
        spec_sha256 = "d" * 64
        sources = {
            route_id: {
                "schedule_sha256": str(index + 5) * 64,
                "route_attestation_sha256": str(index + 1) * 64,
                "source_row_count": rows_per_route,
                "allow_repetition": True,
            }
            for index, route_id in enumerate(self.ROUTES)
        }
        certificate = {
            "schema": "amg_multitask_schedule_certificate_v1",
            "spec_sha256": spec_sha256,
            "schedule_sha256": schedule_sha256,
            "route_registry_sha256": registry_sha256,
            "role": role,
            "panel_id": "multitask-panel",
            "agent_name": "amg_task_neutral_async",
            "optimizer_updates": updates,
            "samples_per_update": 64,
            "row_count": row_count,
            "route_order": list(self.ROUTES),
            "per_route_rows": {route_id: rows_per_route for route_id in self.ROUTES},
            "sources": sources,
        }
        certificate_path = root / "multitask-schedule-certificate.json"
        certificate_path.write_text(
            json.dumps(certificate, sort_keys=True) + "\n", encoding="utf-8"
        )
        certificate_sha256 = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
        source_lock = {
            "schema": "amg_multitask_launcher_source_lock_v1",
            "status": "pass",
            "runtime_source": {
                "outer_commit": "a" * 40,
                "inner_commit": "b" * 40,
                "verl_commit": EXPECTED_VERL_COMMIT,
                "selected_files": {
                    "outer:async_plugins/agentmemorygym_verl/routes.py": "1" * 64,
                    "inner:agentenv/agentenv/envs/swesmith.py": "2" * 64,
                },
            },
            "training_runtime": {
                "base_model": "/models/Qwen3.5-4B",
                "python": "/runtime/bin/python3.12",
                "site_packages": "/runtime/lib/python3.12/site-packages",
                "bundle_sha256": "3" * 64,
                "bundle_sha256_file": "/runtime/runtime.sha256",
                "gpu_count": 8,
                "gpu_type": "B300",
            },
            "integration": {
                "route_registry": {
                    "sha256": registry_sha256,
                    "route_ids": list(self.ROUTES),
                },
                "schedule_certificate": {
                    "sha256": certificate_sha256,
                    "schedule_sha256": schedule_sha256,
                },
            },
        }
        source_lock_path = root / "multitask-source-lock.json"
        source_lock_path.write_text(
            json.dumps(source_lock, sort_keys=True) + "\n", encoding="utf-8"
        )
        inputs = LaunchInputs(
            mode=mode,
            verl_root=root / "verl",
            outer_root=root / "outer",
            schedule=root / "multitask.jsonl",
            env_addr=None,
            run_dir=root / "run",
            experiment_name=f"multitask-{mode}",
            endpoint_source_lock=None,
            endpoint_contract_tool=None,
            publication_receipt=None,
            formal_schedule_certificate=None,
            route_registry=registry,
            route_registry_sha256=registry_sha256,
            multitask_source_lock=source_lock_path,
            multitask_schedule_certificate=certificate_path,
        )
        schedule_report = {
            "sha256": schedule_sha256,
            "count": row_count,
            "role": role,
            "panel_id": "multitask-panel",
            "route_order": list(self.ROUTES),
            "per_route_counts": {route_id: rows_per_route for route_id in self.ROUTES},
            "route_registry_sha256": registry_sha256,
            "agent_name": "amg_task_neutral_async",
            "manifest_digest": spec_sha256,
            "per_route_provenance": {
                route_id: {
                    "route_attestation_sha256": source["route_attestation_sha256"],
                    "source_schedule_sha256": source["schedule_sha256"],
                    "source_manifest_digest": str(index + 9) * 64,
                    "source_panel_id": f"source-{route_id}",
                }
                for index, (route_id, source) in enumerate(sources.items())
            },
        }
        return inputs, schedule_report, source_lock

    def test_multitask_overrides_propagate_registry_without_global_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, registry_sha256 = self._registry(root)
            inputs = LaunchInputs(
                mode="formal",
                verl_root=root / "verl",
                outer_root=root / "outer",
                schedule=root / "formal400.jsonl",
                env_addr=None,
                run_dir=root / "run",
                experiment_name="multitask400",
                endpoint_source_lock=None,
                endpoint_contract_tool=None,
                publication_receipt=None,
                formal_schedule_certificate=None,
                route_registry=registry,
                route_registry_sha256=registry_sha256,
            )
            budget = {
                "publication_cycles": 400,
                "trigger_parameter_sync_step": 1,
                "optimizer_updates": 400,
                "samples_per_update": 64,
                "episodes": 25_600,
                "save_freq": 10,
                "max_actor_ckpt_to_keep": 1,
                "max_critic_ckpt_to_keep": 1,
            }
            values = self._values(
                build_overrides(
                    inputs,
                    effective_schedule=inputs.schedule,
                    endpoint_client_config=None,
                    budget_contract=budget,
                    training_runtime={"base_model": "/models/Qwen3.5-4B"},
                )
            )

            for prefix in ("actor_rollout_ref", "data"):
                self.assertEqual(
                    values[f"{prefix}.agentgym.route_registry_path"],
                    json.dumps(str(registry.resolve())),
                )
                self.assertEqual(
                    values[f"{prefix}.agentgym.route_registry_sha256"],
                    json.dumps(registry_sha256),
                )
                self.assertEqual(
                    json.loads(
                        values[f"{prefix}.agentgym.route_registry_expected_ids"]
                    ),
                    list(self.ROUTES),
                )
                for forbidden in (
                    "task_name",
                    "env_addr",
                    "max_rounds",
                    "max_observation_tokens",
                ):
                    self.assertNotIn(f"{prefix}.agentgym.{forbidden}", values)
            self.assertEqual(values["trainer.total_training_steps"], "400")
            self.assertEqual(values["rollout.total_rollout_steps"], "25600")
            self.assertEqual(
                values["actor_rollout_ref.rollout.agent.default_agent_loop"],
                "amg_task_neutral_async",
            )
            self.assertEqual(values["data.shuffle"], "False")

    def test_multitask_identity_binds_formal400_sources_runtime_and_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, schedule_report, source_lock = self._identity_fixture(
                Path(directory)
            )

            identity = _load_multitask_identity(inputs, schedule_report=schedule_report)

            self.assertEqual(identity["schema"], "amg_multitask_source_identity_v1")
            self.assertEqual(identity["route_ids"], list(self.ROUTES))
            self.assertIsNone(identity["client_config"])
            self.assertEqual(identity["environment"], {})
            self.assertEqual(
                identity["selected_files"],
                source_lock["runtime_source"]["selected_files"],
            )
            budget = identity["budget_contract"]
            self.assertEqual(budget["optimizer_updates"], 400)
            self.assertEqual(budget["samples_per_update"], 64)
            self.assertEqual(budget["episodes"], 25_600)
            self.assertEqual(budget["publication_cycles"], 400)
            self.assertEqual(budget["route_ids"], list(self.ROUTES))
            self.assertEqual(
                _load_launch_identity(inputs, schedule_report=schedule_report),
                identity,
            )

    def _rewrite_multitask_certificate(
        self, inputs: LaunchInputs, source_lock: dict, certificate: dict
    ) -> None:
        assert inputs.multitask_schedule_certificate is not None
        assert inputs.multitask_source_lock is not None
        inputs.multitask_schedule_certificate.write_text(
            json.dumps(certificate, sort_keys=True) + "\n", encoding="utf-8"
        )
        source_lock["integration"]["schedule_certificate"]["sha256"] = hashlib.sha256(
            inputs.multitask_schedule_certificate.read_bytes()
        ).hexdigest()
        inputs.multitask_source_lock.write_text(
            json.dumps(source_lock, sort_keys=True) + "\n", encoding="utf-8"
        )

    def test_multitask_identity_validates_panel_and_source_repetition_contract(self):
        mutations = (
            (
                "panel",
                lambda certificate: certificate.__setitem__("panel_id", "wrong-panel"),
                "panel_id drifted",
            ),
            (
                "source row count",
                lambda certificate: certificate["sources"]["webshop"].__setitem__(
                    "source_row_count", 0
                ),
                "source_row_count",
            ),
            (
                "repetition type",
                lambda certificate: certificate["sources"]["webshop"].__setitem__(
                    "allow_repetition", "true"
                ),
                "allow_repetition must be boolean",
            ),
            (
                "repetition permission",
                lambda certificate: certificate["sources"]["webshop"].update(
                    source_row_count=1, allow_repetition=False
                ),
                "would exhaust",
            ),
        )
        for label, mutate, error in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                inputs, schedule_report, source_lock = self._identity_fixture(
                    Path(directory)
                )
                assert inputs.multitask_schedule_certificate is not None
                certificate = json.loads(
                    inputs.multitask_schedule_certificate.read_text(encoding="utf-8")
                )
                mutate(certificate)
                self._rewrite_multitask_certificate(inputs, source_lock, certificate)
                with self.assertRaisesRegex((TypeError, ValueError), error):
                    _load_multitask_identity(inputs, schedule_report=schedule_report)

    def test_multitask_outer_commit_must_equal_locked_commit(self):
        exact = "a" * 40
        _require_exact_multitask_outer_commit(
            launch_identity_schema="amg_multitask_source_identity_v1",
            publication_outer_commit=exact,
            observed_outer_commit=exact,
        )
        with self.assertRaisesRegex(RuntimeError, "exact outer commit"):
            _require_exact_multitask_outer_commit(
                launch_identity_schema="amg_multitask_source_identity_v1",
                publication_outer_commit=exact,
                observed_outer_commit="b" * 40,
            )
        _require_exact_multitask_outer_commit(
            launch_identity_schema="amg_openmle_publication_identity_v3",
            publication_outer_commit=exact,
            observed_outer_commit="b" * 40,
        )

    def test_legacy_runtime_receipt_preserves_policy_framing_fields(self):
        route = {
            "route_id": "openmle_fast",
            "policy_framing_messages": 2,
            "policy_framing_sha256": "f" * 64,
        }
        normalized = _preserve_legacy_runtime_preflight_fields(
            {"routes": [route]}, multitask=False
        )
        self.assertEqual(normalized["policy_framing_messages"], 2)
        self.assertEqual(normalized["policy_framing_sha256"], "f" * 64)
        multitask = _preserve_legacy_runtime_preflight_fields(
            {"routes": [route]}, multitask=True
        )
        self.assertNotIn("policy_framing_messages", multitask)

    def test_multitask_identity_rejects_schedule_and_registry_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, schedule_report, _ = self._identity_fixture(Path(directory))
            mutated_report = dict(schedule_report, sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "schedule sha256 drifted"):
                _load_multitask_identity(inputs, schedule_report=mutated_report)

            source_lock = json.loads(inputs.multitask_source_lock.read_text())
            source_lock["integration"]["route_registry"]["sha256"] = "0" * 64
            inputs.multitask_source_lock.write_text(
                json.dumps(source_lock, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "registry digest drifted"):
                _load_multitask_identity(inputs, schedule_report=schedule_report)

    def test_multitask_cli_does_not_require_openmle_global_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, _, _ = self._identity_fixture(Path(directory))
            parsed = _parse_args(
                [
                    "--mode",
                    "formal",
                    "--verl-root",
                    str(inputs.verl_root),
                    "--schedule",
                    str(inputs.schedule),
                    "--run-dir",
                    str(inputs.run_dir),
                    "--experiment-name",
                    inputs.experiment_name,
                    "--route-registry",
                    str(inputs.route_registry),
                    "--route-registry-sha256",
                    str(inputs.route_registry_sha256),
                    "--multitask-source-lock",
                    str(inputs.multitask_source_lock),
                    "--multitask-schedule-certificate",
                    str(inputs.multitask_schedule_certificate),
                    "--resolve-only",
                    "--skip-runtime-preflight",
                ]
            )
            self.assertIsNone(parsed.env_addr)
            self.assertIsNone(parsed.endpoint_source_lock)
            self.assertIsNone(parsed.multitask_orchestrator_preflight)
            self.assertTrue(parsed.skip_runtime_preflight)

    def test_generic_cli_rejects_raw_symlink_paths_before_resolution(self):
        file_options = (
            "schedule",
            "endpoint-source-lock",
            "endpoint-contract-tool",
            "publication-receipt",
            "formal-schedule-certificate",
            "route-registry",
            "multitask-source-lock",
            "multitask-schedule-certificate",
            "multitask-orchestrator-preflight",
        )
        directory_options = ("verl-root", "outer-root")
        for option in (*file_options, *directory_options):
            with (
                self.subTest(option=option),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                files = {}
                for name in file_options:
                    path = root / f"{name}.json"
                    path.write_text("{}\n", encoding="utf-8")
                    files[name] = path
                directories = {}
                for name in directory_options:
                    path = root / name
                    path.mkdir()
                    directories[name] = path

                target = (
                    files[option] if option in file_options else directories[option]
                )
                symlink = root / f"{option}.link"
                symlink.symlink_to(target, target_is_directory=target.is_dir())
                if option in file_options:
                    files[option] = symlink
                else:
                    directories[option] = symlink

                argv = [
                    "--mode",
                    "formal",
                    "--verl-root",
                    str(directories["verl-root"]),
                    "--outer-root",
                    str(directories["outer-root"]),
                    "--schedule",
                    str(files["schedule"]),
                    "--run-dir",
                    str(root / "run"),
                    "--experiment-name",
                    "raw-symlink-negative",
                    "--endpoint-source-lock",
                    str(files["endpoint-source-lock"]),
                    "--endpoint-contract-tool",
                    str(files["endpoint-contract-tool"]),
                    "--publication-receipt",
                    str(files["publication-receipt"]),
                    "--formal-schedule-certificate",
                    str(files["formal-schedule-certificate"]),
                    "--route-registry",
                    str(files["route-registry"]),
                    "--route-registry-sha256",
                    "1" * 64,
                    "--multitask-source-lock",
                    str(files["multitask-source-lock"]),
                    "--multitask-schedule-certificate",
                    str(files["multitask-schedule-certificate"]),
                    "--multitask-orchestrator-preflight",
                    str(files["multitask-orchestrator-preflight"]),
                    "--resolve-only",
                    "--skip-runtime-preflight",
                ]
                with (
                    mock.patch(
                        "agentmemorygym_verl.launch.prepare_launch",
                        side_effect=AssertionError("raw path validation was bypassed"),
                    ) as prepare,
                    self.assertRaisesRegex(FileNotFoundError, "symlink|regular"),
                ):
                    launch_main(argv)
                prepare.assert_not_called()

    def test_full_multitask_launch_requires_live_orchestrator_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, schedule_report, _ = self._identity_fixture(root)
            identity = _load_multitask_identity(inputs, schedule_report=schedule_report)
            budget = identity["budget_contract"]
            with self.assertRaisesRegex(ValueError, "full multitask launch requires"):
                _load_multitask_orchestrator_preflight(
                    inputs,
                    launch_identity=identity,
                    schedule_report=schedule_report,
                    budget_contract=budget,
                    required=True,
                )

            inputs.run_dir.mkdir()
            files = {}
            for name in ("config", "endpoint-registry", "holder-lease"):
                path = root / f"{name}.json"
                path.write_text("{}\n", encoding="utf-8")
                files[name] = path
            files["holder-state"] = inputs.run_dir / "holder-transaction" / "state.json"
            files["holder-state"].parent.mkdir(parents=True)
            files["holder-state"].write_text(
                json.dumps(
                    {
                        "schema": "amg_marker_transaction_v1",
                        "status": "acquired",
                        "run_id": inputs.experiment_name,
                        "parent": {"pid": 123, "start_ticks": "456"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            endpoint_entries = []
            registry_payload = json.loads(inputs.route_registry.read_text())
            for route in registry_payload["routes"]:
                route_id = route["route_id"]
                entry = {
                    "route_id": route_id,
                    "route_attestation_sha256": route["route_attestation_sha256"],
                    "endpoint": route["client"]["env_addr"],
                    "pid": 1000 + len(endpoint_entries),
                    "start_ticks": str(2000 + len(endpoint_entries)),
                }
                for prefix in ("gate_receipt", "launcher", "metadata"):
                    artifact = (
                        inputs.run_dir / "endpoints" / route_id / "metadata.json"
                        if prefix == "metadata"
                        else root / f"{route_id}-{prefix}.json"
                    )
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_text("{}\n", encoding="utf-8")
                    entry[f"{prefix}_path"] = str(artifact)
                    entry[f"{prefix}_sha256"] = hashlib.sha256(
                        artifact.read_bytes()
                    ).hexdigest()
                endpoint_entries.append(entry)
            receipt = {
                "schema": "amg_multitask_orchestrator_preflight_v1",
                "status": "pass",
                "config_path": str(files["config"]),
                "config_sha256": hashlib.sha256(
                    files["config"].read_bytes()
                ).hexdigest(),
                "endpoint_registry_path": str(files["endpoint-registry"]),
                "endpoint_registry_sha256": hashlib.sha256(
                    files["endpoint-registry"].read_bytes()
                ).hexdigest(),
                "route_registry_path": str(inputs.route_registry.resolve()),
                "route_registry_sha256": inputs.route_registry_sha256,
                "route_order": list(self.ROUTES),
                "schedule_path": str(inputs.schedule.resolve()),
                "schedule_sha256": schedule_report["sha256"],
                "schedule_count": schedule_report["count"],
                "multitask_source_lock_path": str(
                    inputs.multitask_source_lock.resolve()
                ),
                "multitask_source_lock_sha256": identity["source_lock_sha256"],
                "multitask_schedule_certificate_path": str(
                    inputs.multitask_schedule_certificate.resolve()
                ),
                "multitask_schedule_certificate_sha256": identity[
                    "schedule_certificate_sha256"
                ],
                "budget": {
                    "optimizer_updates": 400,
                    "samples_per_update": 64,
                    "episodes": 25_600,
                },
                "holder_transaction": {
                    "status": "acquired",
                    "lease_path": str(files["holder-lease"]),
                    "lease_sha256": hashlib.sha256(
                        files["holder-lease"].read_bytes()
                    ).hexdigest(),
                    "state_path": str(files["holder-state"]),
                    "watcher_pid": 900,
                    "watcher_start_ticks": "901",
                },
                "endpoints": endpoint_entries,
            }
            preflight = inputs.run_dir / "orchestrator-preflight.json"
            preflight.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            inputs = replace(inputs, multitask_orchestrator_preflight=preflight)
            with (
                mock.patch(
                    "agentmemorygym_verl.launch.process_identity_alive",
                    return_value=True,
                ),
                mock.patch(
                    "agentmemorygym_verl.launch.os.getpgid", side_effect=lambda pid: pid
                ),
            ):
                validated = _load_multitask_orchestrator_preflight(
                    inputs,
                    launch_identity=identity,
                    schedule_report=schedule_report,
                    budget_contract=budget,
                    required=True,
                )
            self.assertEqual(
                [entry["route_id"] for entry in validated["endpoints"]],
                list(self.ROUTES),
            )

            with (
                mock.patch(
                    "agentmemorygym_verl.launch.process_identity_alive",
                    side_effect=lambda pid, _ticks: pid != 900,
                ),
                self.assertRaisesRegex(RuntimeError, "holder watcher"),
            ):
                _load_multitask_orchestrator_preflight(
                    inputs,
                    launch_identity=identity,
                    schedule_report=schedule_report,
                    budget_contract=budget,
                    required=True,
                )

            receipt["endpoints"][0]["endpoint"] = "http://127.0.0.1:1"
            preflight.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            with (
                mock.patch(
                    "agentmemorygym_verl.launch.process_identity_alive",
                    return_value=True,
                ),
                mock.patch(
                    "agentmemorygym_verl.launch.os.getpgid", side_effect=lambda pid: pid
                ),
                self.assertRaisesRegex(RuntimeError, "endpoint mismatch"),
            ):
                _load_multitask_orchestrator_preflight(
                    inputs,
                    launch_identity=identity,
                    schedule_report=schedule_report,
                    budget_contract=budget,
                    required=True,
                )


if __name__ == "__main__":
    unittest.main()
