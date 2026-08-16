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
from paired_eval.contracts import EXTERNAL_MEMORY_CAPABILITY_SURFACES
from paired_eval.evidence import AppendSafeJsonlWriter, PrivateEvidenceStore
from paired_eval.manifest import execute_manifest, expand_manifest
from paired_eval.runner import PairedRunner
from paired_eval.serialization import sha256_json
from paired_eval.verifier import PairVerificationError, verify_pair_completeness


def manifest_payload() -> dict:
    return {
        "schema": "amg.paired_eval.manifest",
        "schema_version": "2.0.0",
        "run_id": "nine-cell-run",
        "arms": ["native", "amg_compaction_only", "amg_memory"],
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
                "summary_instruction_sha256": SHA_C,
                "context_pressure_policy_sha256": SHA_D,
                "context_transition_schema": (
                    "agentmemory_task_neutral_context_transition_v1"
                ),
                "action_accounting": "global_policy_action_budget_v1",
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

    def test_exact_nine_cells_call_the_same_real_run_task(self) -> None:
        configs = expand_manifest(manifest_payload())
        self.assertEqual(len(configs), 9)
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

        self.assertEqual(len(rows), 9)
        self.assertEqual(len(calls), 9)
        self.assertTrue(all(instance is self.runner for instance, _, _ in calls))
        self.assertEqual(
            {(benchmark, arm) for _, benchmark, arm in calls},
            {
                ("gaia_text", "native"),
                ("gaia_text", "amg_compaction_only"),
                ("gaia_text", "amg_memory"),
                ("swebench_verified", "native"),
                ("swebench_verified", "amg_compaction_only"),
                ("swebench_verified", "amg_memory"),
                ("mlebench_lite", "native"),
                ("mlebench_lite", "amg_compaction_only"),
                ("mlebench_lite", "amg_memory"),
            },
        )
        report = verify_pair_completeness(rows)
        self.assertEqual(report["pair_count"], 3)
        self.assertEqual(report["triad_count"], 3)
        self.assertEqual(report["cell_count"], 9)
        self.assertEqual(
            report["capability_lattice"],
            {
                "native": "00",
                "amg_compaction_only": "10",
                "amg_memory": "11",
            },
        )
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

    def test_compaction_only_leakage_fails_closed(self) -> None:
        rows = execute_manifest(
            manifest_payload(),
            runner=self.runner,
            runtime_factory=lambda config: make_fake_runtime(config, self.store),
            writer=self.writer,
        )
        compaction_index = next(
            index
            for index, row in enumerate(rows)
            if row["benchmark"] == "gaia_text"
            and row["arm"] == "amg_compaction_only"
        )

        for surface in EXTERNAL_MEMORY_CAPABILITY_SURFACES:
            leaked = deepcopy(rows)
            leaked[compaction_index]["config"]["capability"][
                "external_memory_surfaces"
            ] = [surface]
            leaked[compaction_index]["treatment"][
                "external_memory_surfaces"
            ] = [surface]
            with self.subTest(surface=surface), self.assertRaises(
                PairVerificationError
            ):
                verify_pair_completeness(leaked)

        prompt_leak = deepcopy(rows)
        prompt_leak[compaction_index]["config"]["capability"][
            "prompt_declaration"
        ] += " unexpected_external_memory_declaration"
        prompt_leak[compaction_index]["treatment"]["prompt_declaration"] = (
            prompt_leak[compaction_index]["config"]["capability"][
                "prompt_declaration"
            ]
        )
        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(prompt_leak)

        tool_leak = deepcopy(rows)
        tool_leak[compaction_index]["config"]["capability"]["tools"] = [
            "external_memory_write"
        ]
        tool_leak[compaction_index]["treatment"]["tools"] = [
            "external_memory_write"
        ]
        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(tool_leak)

        root_leak = deepcopy(rows)
        leaked_row = root_leak[compaction_index]
        external_root = {
            "capability_id": "external_memory",
            "root_kind": "external_memory",
            "root_id": sha256_json(
                {
                    "namespace": leaked_row["namespace"],
                    "capability_id": "external_memory",
                    "root_kind": "external_memory",
                }
            ),
            "namespace_sha256": leaked_row["namespace_sha256"],
        }
        leaked_row["lifecycle"]["declared_roots"].append(external_root)
        leaked_row["lifecycle"]["closed_roots"].append(external_root)
        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(root_leak)

        receipt_leak = deepcopy(rows)
        leaked_turn = receipt_leak[compaction_index]["turns"][0]
        leaked_turn["capability_id"] = "external_memory"
        leaked_turn["root_kind"] = "external_memory"
        leaked_turn["root_id"] = external_root["root_id"]
        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(receipt_leak)

    def test_individually_valid_non_capability_drift_fails_closed(self) -> None:
        rows = execute_manifest(
            manifest_payload(),
            runner=self.runner,
            runtime_factory=lambda config: make_fake_runtime(config, self.store),
            writer=self.writer,
        )
        compaction_index = next(
            index
            for index, row in enumerate(rows)
            if row["benchmark"] == "gaia_text"
            and row["arm"] == "amg_compaction_only"
        )

        def refresh_config_digests(row):
            row["full_config_sha256"] = sha256_json(row["config"])
            excluded = deepcopy(row["config"])
            del excluded["capability"]
            row["treatment_excluded_config_sha256"] = sha256_json(excluded)

        drifted = deepcopy(rows)
        row = drifted[compaction_index]
        row["config"]["budgets"]["max_policy_turns"] += 1
        row["budgets"]["max_policy_turns"] += 1
        refresh_config_digests(row)
        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(drifted)

        native_tool_drift = deepcopy(rows)
        row = native_tool_drift[compaction_index]
        row["config"]["task"]["native_tools"].append("leaked_tool")
        refresh_config_digests(row)
        with self.assertRaises(PairVerificationError):
            verify_pair_completeness(native_tool_drift)

    def test_manifest_rejects_duplicate_or_incomplete_arm_declarations(self) -> None:
        missing = manifest_payload()
        missing["arms"] = ["native", "amg_compaction_only"]
        with self.assertRaises(ValueError):
            expand_manifest(missing)

        duplicate = manifest_payload()
        duplicate["arms"] = [
            "native",
            "amg_compaction_only",
            "amg_compaction_only",
        ]
        with self.assertRaises(ValueError):
            expand_manifest(duplicate)

        fourth = manifest_payload()
        fourth["arms"] = [
            "native",
            "amg_compaction_only",
            "amg_memory",
            "memory_only",
        ]
        with self.assertRaises(ValueError):
            expand_manifest(fourth)


if __name__ == "__main__":
    unittest.main()
