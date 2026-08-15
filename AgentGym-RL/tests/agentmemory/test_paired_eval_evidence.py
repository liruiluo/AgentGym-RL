from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from test_paired_eval_support import (
    Arm,
    ManualClock,
    make_config,
    make_fake_runtime,
    with_arm,
)

from paired_eval.controller import DependencyLightPolicyTurnController
from paired_eval.evidence import AppendSafeJsonlWriter, PrivateEvidenceStore
from paired_eval.runner import PairedRunner
from paired_eval.verifier import (
    PairVerificationError,
    ResultValidationError,
    build_public_summary,
    validate_result_row,
)


class EvidenceSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.store = PrivateEvidenceStore(self.root / "evidence")
        self.runner = PairedRunner(
            controller=DependencyLightPolicyTurnController(),
            evidence_store=self.store,
            clock=ManualClock(),
        )

    def result_row(self, config=None):
        config = make_config() if config is None else config
        bindings = make_fake_runtime(config, self.store)
        return self.runner.run_task(config, bindings.adapter, bindings.model)

    def test_append_safe_jsonl_uses_private_mode_and_complete_lines(self) -> None:
        row = self.result_row()
        path = self.root / "private" / "results.jsonl"
        writer = AppendSafeJsonlWriter(path, validator=validate_result_row)
        writer.append(row)
        writer.append(row)

        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0]), row)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_writer_rejects_a_preexisting_partial_line(self) -> None:
        row = self.result_row()
        path = self.root / "partial.jsonl"
        path.write_text('{"partial":true}', encoding="utf-8")
        writer = AppendSafeJsonlWriter(path, validator=validate_result_row)
        with self.assertRaises(RuntimeError):
            writer.append(row)

    def test_raw_gated_content_is_rejected_and_never_public(self) -> None:
        row = self.result_row()
        unsafe = deepcopy(row)
        unsafe["request_messages"] = [
            {"role": "user", "content": "GATED PRIVATE CONTENT"}
        ]
        with self.assertRaises(ResultValidationError):
            validate_result_row(unsafe)

        forbidden_compaction = deepcopy(row)
        forbidden_compaction["turns"][0][
            "context_operation"
        ] = "replace_messages"
        forbidden_compaction["compaction"]["receipt_count"] = 1
        with self.assertRaises(ResultValidationError):
            validate_result_row(forbidden_compaction)

        compaction_row = self.result_row(
            with_arm(make_config(), Arm.AMG_COMPACTION_ONLY)
        )
        memory_row = self.result_row(with_arm(make_config(), Arm.AMG_MEMORY))
        for result, score in zip(
            (row, compaction_row, memory_row),
            (0.25, 0.5, 0.9),
        ):
            result["scorer"]["public_metrics"] = {
                "score": score,
                "passed": score >= 0.5,
            }
        summary = build_public_summary([row, compaction_row, memory_row])
        serialized = json.dumps(summary, sort_keys=True)
        self.assertNotIn("evidence://", serialized)
        self.assertNotIn("protected_ref", serialized)
        self.assertNotIn("private_grader_detail", serialized)
        self.assertNotIn("messages", serialized)
        self.assertNotIn("interaction", serialized.lower())
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["triad_count"], 1)
        triad = summary["triads"][0]
        self.assertEqual(
            triad["raw_arm_metrics"],
            {
                "native": {"score": 0.25, "passed": False},
                "amg_compaction_only": {"score": 0.5, "passed": True},
                "amg_memory": {"score": 0.9, "passed": True},
            },
        )
        self.assertEqual(
            triad["contrasts"],
            {
                "compaction_effect": {"score": 0.25},
                "external_memory_incremental_effect": {
                    "score": 0.4,
                },
                "full_amg_effect": {"score": 0.65},
            },
        )

    def test_public_summary_rejects_absolute_path_labels(self) -> None:
        protected_path = "/protected/private/gold-answer.json"
        base_config = make_config()
        identity_configs = {
            "run_id": replace(base_config, run_id=protected_path),
            "benchmark": replace(
                base_config,
                task=replace(base_config.task, benchmark=protected_path),
            ),
            "protocol": replace(
                base_config,
                task=replace(base_config.task, protocol=protected_path),
            ),
            "task_id": replace(
                base_config,
                task=replace(base_config.task, task_id=protected_path),
            ),
        }
        for field, config in identity_configs.items():
            with self.subTest(identity=field):
                rows = [
                    self.result_row(with_arm(config, arm))
                    for arm in (
                        Arm.NATIVE,
                        Arm.AMG_COMPACTION_ONLY,
                        Arm.AMG_MEMORY,
                    )
                ]
                with self.assertRaises(PairVerificationError):
                    build_public_summary(rows)

        metric_rows = [
            self.result_row(with_arm(base_config, arm))
            for arm in (
                Arm.NATIVE,
                Arm.AMG_COMPACTION_ONLY,
                Arm.AMG_MEMORY,
            )
        ]
        for row in metric_rows:
            row["scorer"]["public_metrics"] = {protected_path: 1.0}
        with self.assertRaises(PairVerificationError):
            build_public_summary(metric_rows)


if __name__ == "__main__":
    unittest.main()
