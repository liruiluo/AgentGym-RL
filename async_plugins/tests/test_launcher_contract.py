from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
import zipfile
from pathlib import Path

from agentmemorygym_verl.config_contract import inspect_schedule
from agentmemorygym_verl.identity import (
    EXPECTED_VERL_COMMIT,
    TRL_WHEEL_RELATIVE_PATH,
    TRL_WHEEL_SHA256,
)
from agentmemorygym_verl.launch import (
    LaunchInputs,
    _load_endpoint_identity,
    _parse_args,
    _partition_selected_file_hashes,
    _validate_accelerator_runtime,
    build_overrides,
    build_runtime_env,
)

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
            self.assertEqual(values["actor_rollout_ref.actor.loss_agg_mode"], "token-mean")
            self.assertEqual(values["actor_rollout_ref.actor.use_prefix_grouper"], "False")
            self.assertEqual(values["critic.loss_agg_mode"], "token-mean")
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
                values[
                    "actor_rollout_ref.actor.fsdp_config.reshard_after_forward"
                ],
                "True",
            )
            self.assertEqual(
                values["critic.fsdp.reshard_after_forward"], "True"
            )
            self.assertEqual(
                values["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"],
                "65536",
            )
            self.assertEqual(
                values["critic.ppo_max_token_len_per_gpu"], "32768"
            )

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

    def test_native_dynamic_token_budgets_are_configurable(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, identity = self._identity(Path(directory), "formal")
            inputs = replace(
                inputs,
                actor_ppo_max_tokens_per_gpu=131072,
                critic_ppo_max_tokens_per_gpu=65536,
            )
            budget = {
                **identity["budget_contract"],
                "actor_ppo_max_tokens_per_gpu": 131072,
                "critic_ppo_max_tokens_per_gpu": 65536,
            }
            values = self._values(
                build_overrides(
                    inputs,
                    effective_schedule=inputs.schedule,
                    endpoint_client_config=identity["client_config"],
                    budget_contract=budget,
                    training_runtime=identity["training_runtime"],
                )
            )
            self.assertEqual(
                values["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"],
                "131072",
            )
            self.assertEqual(
                values["critic.ppo_max_token_len_per_gpu"], "65536"
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
                values[
                    "actor_rollout_ref.model.fused_kernel_options.impl_backend"
                ],
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
                values[
                    "actor_rollout_ref.actor.fsdp_config.reshard_after_forward"
                ],
                "True",
            )
            self.assertEqual(
                values["critic.fsdp.reshard_after_forward"], "True"
            )

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
        self.assertEqual(hashlib.sha256(wheel.read_bytes()).hexdigest(), TRL_WHEEL_SHA256)
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
            "nvcc_path": "/dev/shm/cuda-13-b300-toolkit/bin/nvcc",
            "nvcc_release": "13.0",
            "torch_cuda": "13.0",
            "torch_cuda_available": True,
            "gpu_count": 8,
            "gpu_names": ["NVIDIA B300 SXM6 AC"] * 8,
        }
        self.assertEqual(
            _validate_accelerator_runtime(
                observed, training_runtime=training_runtime
            )["gpu_count"],
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
        self.assertEqual(parsed.actor_ppo_max_tokens_per_gpu, 65536)
        self.assertEqual(parsed.critic_ppo_max_tokens_per_gpu, 32768)
        tuned = _parse_args(
            common
            + [
                "--actor-ppo-max-tokens-per-gpu",
                "131072",
                "--critic-ppo-max-tokens-per-gpu",
                "65536",
            ]
        )
        self.assertEqual(tuned.actor_ppo_max_tokens_per_gpu, 131072)
        self.assertEqual(tuned.critic_ppo_max_tokens_per_gpu, 65536)
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
        self.assertIn("PYTHONPATH is an identity conflict", text)
        self.assertIn("trl-0.9.6-py3-none-any.whl", text)
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


if __name__ == "__main__":
    unittest.main()
