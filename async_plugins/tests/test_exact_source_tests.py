from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts/run_exact_source_tests.py"
SPEC = importlib.util.spec_from_file_location("run_exact_source_tests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXACT_SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXACT_SOURCE)


class TestExactSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.output = self.root / "output"
        self.repo.mkdir()
        self.output.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(
            ["git", "config", "user.email", "fixture@example.com"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Fixture"],
            cwd=self.repo,
            check=True,
        )
        self.source = self.repo / "selected.py"
        self.source.write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "selected.py"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "fixture"],
            cwd=self.repo,
            check=True,
        )
        self.head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True
        ).strip()
        self.manifest = self.output / "selected-files.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema": "exact_source_selected_files_v1",
                    "repo": "outer",
                    "files": {
                        "selected.py": hashlib.sha256(
                            self.source.read_bytes()
                        ).hexdigest()
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.interpreter = Path(sys.executable).resolve(strict=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _command(
        self,
        *command: str,
        log: Path | None = None,
        receipt: Path | None = None,
        selected_manifest: Path | None = None,
    ) -> list[str]:
        return [
            str(self.interpreter),
            str(SCRIPT),
            "--repo",
            f"outer={self.repo}",
            "--expected-head",
            f"outer={self.head}",
            "--selected-manifest",
            f"outer={selected_manifest or self.manifest}",
            "--interpreter",
            str(self.interpreter),
            "--interpreter-sha256",
            hashlib.sha256(self.interpreter.read_bytes()).hexdigest(),
            "--cwd",
            str(self.repo),
            "--log",
            str(log or self.output / "suite.log"),
            "--receipt",
            str(receipt or self.output / "receipt.json"),
            "--env",
            f"PYTHONPATH={self.repo}",
            "--",
            *command,
        ]

    def test_pass_receipt_binds_pre_and_post_source(self) -> None:
        completed = subprocess.run(
            self._command("-c", "print('EXACT_SOURCE_PASS')"),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(
            (self.output / "receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["schema"], "exact_source_test_receipt_v1")
        self.assertEqual(receipt["status"], "pass")
        self.assertEqual(receipt["command"][0], str(self.interpreter))
        self.assertEqual(
            receipt["interpreter"]["pre"], receipt["interpreter"]["post"]
        )
        self.assertEqual(
            receipt["interpreter"]["pre"]["sha256"],
            receipt["interpreter"]["expected_sha256"],
        )
        self.assertEqual(receipt["repositories"]["outer"]["pre"]["head"], self.head)
        self.assertEqual(
            receipt["repositories"]["outer"]["pre"],
            receipt["repositories"]["outer"]["post"],
        )
        self.assertEqual(receipt["exit_code"], 0)
        self.assertEqual(receipt["final_post_output_audit"]["status"], "pass")
        self.assertEqual(receipt["publication"]["state"], "final")
        log = (self.output / "suite.log").read_text(encoding="utf-8")
        self.assertIn("EXACT_SOURCE_TEST_PRE ", log)
        self.assertIn("EXACT_SOURCE_PASS", log)
        self.assertIn("EXACT_SOURCE_TEST_POST ", log)
        self.assertIn("EXACT_SOURCE_TEST_FINAL ", log)

    def test_symlinked_output_parent_into_repository_fails_before_command(self) -> None:
        generated = self.repo / "generated"
        generated.mkdir()
        output_link = self.root / "out-link"
        output_link.symlink_to(generated, target_is_directory=True)
        sentinel = self.output / "must-not-exist"
        completed = subprocess.run(
            self._command(
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
                log=output_link / "suite.log",
            ),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("inside repository outer", completed.stderr)
        self.assertFalse(sentinel.exists())
        self.assertFalse((generated / "suite.log").exists())
        self.assertFalse((self.output / "receipt.json").exists())
        self.assertEqual(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.repo,
                text=True,
            ),
            "",
        )

    def test_output_cannot_alias_or_clobber_selected_manifest(self) -> None:
        manifest_before = self.manifest.read_bytes()
        sentinel = self.output / "must-not-exist"
        completed = subprocess.run(
            self._command(
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
                log=self.manifest,
            ),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("selected manifest", completed.stderr)
        self.assertFalse(sentinel.exists())
        self.assertEqual(self.manifest.read_bytes(), manifest_before)
        self.assertFalse((self.output / "receipt.json").exists())

    def test_output_hardlink_to_selected_manifest_fails_before_command(self) -> None:
        aliased_log = self.output / "manifest-hardlink.log"
        os.link(self.manifest, aliased_log)
        manifest_before = self.manifest.read_bytes()
        sentinel = self.output / "must-not-exist"
        completed = subprocess.run(
            self._command(
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
                log=aliased_log,
            ),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("same existing file", completed.stderr)
        self.assertFalse(sentinel.exists())
        self.assertEqual(self.manifest.read_bytes(), manifest_before)
        self.assertEqual(aliased_log.read_bytes(), manifest_before)
        self.assertFalse((self.output / "receipt.json").exists())

    def test_log_and_receipt_hardlink_alias_fails_before_command(self) -> None:
        log = self.output / "aliased.log"
        receipt_path = self.output / "aliased-receipt.json"
        log.write_text("preserve\n", encoding="utf-8")
        os.link(log, receipt_path)
        before = log.read_bytes()
        sentinel = self.output / "must-not-exist"
        completed = subprocess.run(
            self._command(
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
                log=log,
                receipt=receipt_path,
            ),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("same existing file", completed.stderr)
        self.assertFalse(sentinel.exists())
        self.assertEqual(log.read_bytes(), before)
        self.assertEqual(receipt_path.read_bytes(), before)

    def test_source_mutation_during_output_publication_fails_final_audit(self) -> None:
        original_atomic_write = EXACT_SOURCE._atomic_write
        calls = 0

        def write_then_mutate(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
            nonlocal calls
            original_atomic_write(path, payload, mode=mode)
            if calls == 0:
                self.source.write_text("VALUE = 2\n", encoding="utf-8")
            calls += 1

        arguments = self._command("-c", "print('OUTPUT_RACE')")[2:]
        with mock.patch.object(
            EXACT_SOURCE, "_atomic_write", side_effect=write_then_mutate
        ):
            status = EXACT_SOURCE.main(arguments)

        self.assertEqual(status, 2)
        receipt = json.loads(
            (self.output / "receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["status"], "fail")
        self.assertEqual(receipt["final_post_output_audit"]["status"], "fail")
        self.assertTrue(
            any(
                "final post-output" in violation
                for violation in receipt["violations"]
            )
        )

    def test_final_audit_crash_never_leaves_pass_receipt(self) -> None:
        arguments = self._command("-c", "print('OUTPUT_RACE')")[2:]
        with (
            mock.patch.object(
                EXACT_SOURCE,
                "_final_post_output_audit",
                side_effect=RuntimeError("injected final audit crash"),
            ),
            self.assertRaisesRegex(RuntimeError, "injected final audit crash"),
        ):
            EXACT_SOURCE.main(arguments)

        receipt = json.loads(
            (self.output / "receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["status"], "fail")
        self.assertEqual(
            receipt["publication"]["state"],
            "pending_final_post_output_audit",
        )
        self.assertIsNone(receipt["final_post_output_audit"])
        log = (self.output / "suite.log").read_text(encoding="utf-8")
        self.assertNotIn("EXACT_SOURCE_TEST_FINAL ", log)

    def test_source_mutation_fails_and_records_pre_post_drift(self) -> None:
        mutation = (
            "from pathlib import Path; "
            f"Path({str(self.source)!r}).write_text('VALUE = 2\\n')"
        )
        completed = subprocess.run(
            self._command("-c", mutation),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        receipt = json.loads(
            (self.output / "receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["status"], "fail")
        self.assertEqual(receipt["exit_code"], 0)
        self.assertTrue(
            any("post-run" in violation for violation in receipt["violations"])
        )
        pre = receipt["repositories"]["outer"]["pre"]
        post = receipt["repositories"]["outer"]["post"]
        self.assertNotEqual(pre["selected_files"], post["selected_files"])

    def test_dirty_precondition_prevents_command_execution(self) -> None:
        sentinel = self.output / "must-not-exist"
        (self.repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        completed = subprocess.run(
            self._command(
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
            ),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(sentinel.exists())
        receipt = json.loads(
            (self.output / "receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["status"], "fail")
        self.assertIsNone(receipt["exit_code"])
        self.assertTrue(
            any("pre-run git status is dirty" in item for item in receipt["violations"])
        )

    def test_pre_hash_mismatch_prevents_command_execution(self) -> None:
        sentinel = self.output / "must-not-exist"
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["files"]["selected.py"] = "0" * 64
        self.manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            self._command(
                "-c",
                f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
            ),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(sentinel.exists())
        receipt = json.loads(
            (self.output / "receipt.json").read_text(encoding="utf-8")
        )
        self.assertTrue(
            any(
                "pre-run selected-file hash mismatch" in item
                for item in receipt["violations"]
            )
        )

    def test_nonzero_test_command_is_a_failed_receipt(self) -> None:
        completed = subprocess.run(
            self._command("-c", "raise SystemExit(7)"),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        receipt = json.loads(
            (self.output / "receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["status"], "fail")
        self.assertEqual(receipt["exit_code"], 7)
        self.assertIn("test command exited 7", receipt["violations"])


if __name__ == "__main__":
    unittest.main()
