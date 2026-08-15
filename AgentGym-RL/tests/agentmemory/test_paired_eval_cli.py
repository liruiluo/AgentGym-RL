from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_paired_eval_manifest import manifest_payload
from test_paired_eval_support import ManualClock, ROOT, make_fake_runtime

from paired_eval.controller import DependencyLightPolicyTurnController
from paired_eval.evidence import AppendSafeJsonlWriter, PrivateEvidenceStore
from paired_eval.manifest import execute_manifest
from paired_eval.runner import PairedRunner
from paired_eval.verifier import validate_result_row


class PairedEvalCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(manifest_payload()), encoding="utf-8"
        )
        package_path = str(ROOT / "scripts" / "agentmemory")
        self.environment = dict(os.environ)
        existing = self.environment.get("PYTHONPATH")
        self.environment["PYTHONPATH"] = (
            package_path if not existing else package_path + os.pathsep + existing
        )

    def run_cli(self, *arguments: str) -> dict:
        completed = subprocess.run(
            [sys.executable, "-m", "paired_eval", *arguments],
            cwd=ROOT,
            env=self.environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = completed.stdout.splitlines()
        self.assertTrue(lines)
        return json.loads(lines[-1])

    def test_expand_verify_and_public_summary_commands(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "paired_eval",
                "expand",
                "--manifest",
                str(self.manifest_path),
            ],
            cwd=ROOT,
            env=self.environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        expanded = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(expanded), 9)

        store = PrivateEvidenceStore(self.root / "evidence")
        results = self.root / "results.jsonl"
        runner = PairedRunner(
            controller=DependencyLightPolicyTurnController(),
            evidence_store=store,
            clock=ManualClock(),
        )
        execute_manifest(
            manifest_payload(),
            runner=runner,
            runtime_factory=lambda config: make_fake_runtime(config, store),
            writer=AppendSafeJsonlWriter(
                results,
                validator=validate_result_row,
            ),
        )

        report = self.run_cli("verify", "--results", str(results))
        self.assertEqual(report["pair_count"], 3)
        self.assertEqual(report["triad_count"], 3)
        self.assertEqual(report["cell_count"], 9)
        summary = self.run_cli("public-summary", "--results", str(results))
        self.assertEqual(summary["row_count"], 9)
        self.assertEqual(summary["triad_count"], 3)
        self.assertEqual(len(summary["triads"]), 3)
        self.assertTrue(
            all(
                set(triad["raw_arm_metrics"])
                == {"native", "amg_compaction_only", "amg_memory"}
                for triad in summary["triads"]
            )
        )
        serialized = json.dumps(summary, sort_keys=True)
        self.assertNotIn("evidence://", serialized)
        self.assertNotIn("protected_ref", serialized)


if __name__ == "__main__":
    unittest.main()
