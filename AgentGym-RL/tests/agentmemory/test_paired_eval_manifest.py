from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from test_paired_eval_support import (
    ManualClock,
    SHA_A,
    SHA_B,
    SHA_C,
    SHA_D,
    make_fake_runtime,
)

from paired_eval.controller import DependencyLightPolicyTurnController
from paired_eval.evidence import AppendSafeJsonlWriter, PrivateEvidenceStore
from paired_eval.manifest import execute_manifest, expand_manifest
from paired_eval.runner import PairedRunner
from paired_eval.verifier import PairVerificationError, verify_pair_completeness


def manifest_payload() -> dict:
    return {
        "schema": "amg.paired_eval.manifest",
        "schema_version": "1.0.0",
        "run_id": "six-case-run",
        "arms": ["native", "amg_memory"],
        "common": {
            "model": {
                "model_id": "test-model",
                "revision": "test-model-revision",
                "tokenizer_sha256": SHA_A,
            },
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 64,
                "stop": [],
            },
            "budgets": {
                "max_policy_turns": 3,
                "max_total_tokens": 4096,
                "max_tool_calls": 3,
                "max_wall_seconds": 60.0,
                "max_prompt_tokens": 2048,
                "max_model_tokens": 2112,
                "max_observation_tokens": 256,
                "action_observation_envelope_tokens": 16,
            },
            "compaction": {
                "policy": "policy_authored_task_neutral_v1",
                "trigger": "wrapper_token_pressure_v1",
                "summary_max_tokens": 256,
                "config_sha256": SHA_B,
            },
            "source": {
                "outer_commit": "d5892e63de0f8ad2ebdcedf09be46d3bca4117d1",
                "inner_commit": "017ebd2fbc0ab8e53a0ba743f79b50d6e46d1a42",
                "adapter_sha256": SHA_C,
                "runner_sha256": SHA_D,
            },
            "runtime": {
                "image_digest": "sha256:" + SHA_A,
                "runtime_sha256": SHA_B,
                "compute_class": "cpu-test",
            },
            "grader": {
                "name": "fake_official_grader",
                "revision": "grader-revision-1",
                "config_sha256": SHA_C,
            },
        },
        "tasks": [
            {
                "benchmark": "gaia_text",
                "protocol": "gaia-text@frozen",
                "task_id": "gaia-001",
                "task_index": 0,
                "seed": 7,
                "native_tools": ["search", "browse", "answer"],
                "artifact_type": "answer",
            },
            {
                "benchmark": "swebench_verified",
                "protocol": "swebench-verified@frozen",
                "task_id": "swe-001",
                "task_index": 1,
                "seed": 7,
                "native_tools": ["shell_command", "apply_patch"],
                "artifact_type": "patch",
            },
            {
                "benchmark": "mlebench_lite",
                "protocol": "mlebench-lite@frozen",
                "task_id": "mle-001",
                "task_index": 2,
                "seed": 7,
                "native_tools": ["shell_command", "apply_patch", "submit"],
                "artifact_type": "submission",
            },
        ],
    }


class ManifestExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.store = PrivateEvidenceStore(root / "evidence")
        self.writer = AppendSafeJsonlWriter(root / "results.jsonl")
        self.runner = PairedRunner(
            controller=DependencyLightPolicyTurnController(),
            evidence_store=self.store,
            clock=ManualClock(),
        )

    def test_all_six_cases_call_the_same_real_run_task(self) -> None:
        configs = expand_manifest(manifest_payload())
        self.assertEqual(len(configs), 6)
        original = PairedRunner.run_task
        calls = []

        def runtime_factory(config):
            return make_fake_runtime(config, self.store)

        def real_spy(instance, config, adapter, model):
            calls.append((instance, config.task.benchmark, config.capability.arm.value))
            return original(instance, config, adapter, model)

        PairedRunner.run_task = real_spy
        try:
            rows = execute_manifest(
                manifest_payload(),
                runner=self.runner,
                runtime_factory=runtime_factory,
                writer=self.writer,
            )
        finally:
            PairedRunner.run_task = original

        self.assertEqual(len(rows), 6)
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(instance is self.runner for instance, _, _ in calls))
        self.assertEqual(
            {(benchmark, arm) for _, benchmark, arm in calls},
            {
                ("gaia_text", "native"),
                ("gaia_text", "amg_memory"),
                ("swebench_verified", "native"),
                ("swebench_verified", "amg_memory"),
                ("mlebench_lite", "native"),
                ("mlebench_lite", "amg_memory"),
            },
        )
        report = verify_pair_completeness(rows)
        self.assertEqual(report["pair_count"], 3)
        with self.assertRaises(RuntimeError):
            execute_manifest(
                manifest_payload(),
                runner=self.runner,
                runtime_factory=lambda config: make_fake_runtime(
                    config, self.store
                ),
                writer=self.writer,
            )

    def test_runtime_failure_does_not_emit_an_orphan_pair(self) -> None:
        calls = 0

        def failing_factory(config):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic runtime binding failure")
            return make_fake_runtime(config, self.store)

        with self.assertRaises(RuntimeError):
            execute_manifest(
                manifest_payload(),
                runner=self.runner,
                runtime_factory=failing_factory,
                writer=self.writer,
            )
        self.assertFalse(self.writer.path.exists())

    def test_missing_duplicate_and_treatment_drift_fail_closed(self) -> None:
        rows = execute_manifest(
            manifest_payload(),
            runner=self.runner,
            runtime_factory=lambda config: make_fake_runtime(config, self.store),
            writer=self.writer,
        )

        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(rows[:-1])
        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(rows + [deepcopy(rows[0])])

        drifted = deepcopy(rows)
        drifted[1]["config"]["decoding"]["top_p"] = 0.9
        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(drifted)

        prompt_drift = deepcopy(rows)
        prompt_drift[1]["prompt"]["treatment_excluded_sha256"] = "f" * 64
        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(prompt_drift)

        namespace_leak = deepcopy(rows)
        namespace_leak[1]["namespace"] = deepcopy(namespace_leak[0]["namespace"])
        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(namespace_leak)

        root_reuse = deepcopy(rows)
        reused_root_id = root_reuse[0]["lifecycle"]["declared_roots"][0][
            "root_id"
        ]
        root_reuse[1]["lifecycle"]["declared_roots"][0][
            "root_id"
        ] = reused_root_id
        root_reuse[1]["lifecycle"]["closed_roots"][0][
            "root_id"
        ] = reused_root_id
        root_reuse[1]["turns"][0]["root_id"] = reused_root_id
        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(root_reuse)

    def test_manifest_rejects_duplicate_or_incomplete_arm_declarations(self) -> None:
        missing = manifest_payload()
        missing["arms"] = ["native"]
        with self.assertRaises(ValueError):
            expand_manifest(missing)

        duplicate = manifest_payload()
        duplicate["arms"] = ["native", "native", "amg_memory"]
        with self.assertRaises(ValueError):
            expand_manifest(duplicate)


if __name__ == "__main__":
    unittest.main()
