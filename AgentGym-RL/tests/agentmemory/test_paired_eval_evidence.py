from __future__ import annotations

from copy import deepcopy
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

        memory_row = self.result_row(with_arm(make_config(), Arm.AMG_MEMORY))
        summary = build_public_summary([row, memory_row])
        serialized = json.dumps(summary, sort_keys=True)
        self.assertNotIn("evidence://", serialized)
        self.assertNotIn("protected_ref", serialized)
        self.assertNotIn("private_grader_detail", serialized)
        self.assertNotIn("messages", serialized)


if __name__ == "__main__":
    unittest.main()
