from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from swebench_triad_eval.atomic import ImmutableConflictError, canonical_json_bytes
from swebench_triad_eval.official_grader import (
    DOCKER_SOCKET,
    GraderConfigurationError,
    GraderContractError,
    OfficialGradeRequest,
    OfficialGraderConfig,
    RetryableGraderError,
    grade_attempt_directory,
    grader_command,
    run_official_grader,
    verify_pinned_import,
)
from swebench_triad_eval.state import (
    CellKey,
    CellStateStore,
    ManifestCell,
    OwnerIdentity,
    sha256_json,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
INSTANCE_ID = "owner__repo-1"


def prediction(
    model_patch: str = "diff --git a/value.py b/value.py\n",
) -> dict[str, str]:
    return {
        "instance_id": INSTANCE_ID,
        "model_name_or_path": "qwen35-4b-native",
        "model_patch": model_patch,
    }


def grade_request(model_patch: str = "diff --git a/value.py b/value.py\n"):
    prediction_row = prediction(model_patch)
    prediction_sha256 = sha256_json(prediction_row)
    handoff = {
        "prediction_sha256": prediction_sha256,
        "official_resolved": None,
        "grader_revision": "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
    }
    accepted = {
        "schema": "swebench_triad_accepted_cell_v1",
        "cell": {"task_index": 0, "arm": "native"},
        "instance_id": INSTANCE_ID,
        "manifest_cell_sha256": SHA_A,
        "attempt_generation": 7,
        "endpoint_sha256": SHA_B,
        "prediction_sha256": prediction_sha256,
        "handoff_sha256": sha256_json(handoff),
    }
    return OfficialGradeRequest(
        task_index=0,
        arm="native",
        generation=7,
        grader_attempt=1,
        prediction=prediction_row,
        accepted_cell=accepted,
        queued_handoff=handoff,
    )


def aggregate_report(instance_id: str, outcome: str) -> dict[str, object]:
    ids = {
        "completed": [],
        "resolved": [],
        "unresolved": [],
        "empty_patch": [],
        "error": [],
    }
    if outcome == "resolved":
        ids["completed"] = [instance_id]
        ids["resolved"] = [instance_id]
    elif outcome == "unresolved":
        ids["completed"] = [instance_id]
        ids["unresolved"] = [instance_id]
    elif outcome == "empty_patch":
        ids["empty_patch"] = [instance_id]
    elif outcome == "error":
        ids["error"] = [instance_id]
    else:
        raise ValueError(outcome)
    return {
        "total_instances": 1,
        "submitted_instances": 1,
        "completed_instances": len(ids["completed"]),
        "resolved_instances": len(ids["resolved"]),
        "unresolved_instances": len(ids["unresolved"]),
        "empty_patch_instances": len(ids["empty_patch"]),
        "error_instances": len(ids["error"]),
        "completed_ids": ids["completed"],
        "incomplete_ids": [],
        "empty_patch_ids": ids["empty_patch"],
        "submitted_ids": [instance_id],
        "resolved_ids": ids["resolved"],
        "unresolved_ids": ids["unresolved"],
        "error_ids": ids["error"],
        "schema_version": 2,
    }


class FakeHarness:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def __call__(self, command, **kwargs):
        command = list(command)
        cwd = Path(kwargs["cwd"])
        environment = dict(kwargs["env"])
        self.calls.append((command, cwd, environment))
        run_id = command[command.index("--run_id") + 1]
        instance_id = command[command.index("--instance_ids") + 1]
        prediction_path = Path(command[command.index("--predictions_path") + 1])
        prediction_row = json.loads(prediction_path.read_text().strip())
        model = prediction_row["model_name_or_path"].replace("/", "__")
        aggregate_path = cwd / f"{model}.{run_id}.json"
        instance_root = (
            cwd / "logs" / "run_evaluation" / run_id / model / instance_id
        )

        if self.mode == "missing":
            return subprocess.CompletedProcess(command, 0, b"", b"")

        report_outcome = self.mode
        if self.mode in {"patch_apply_failure", "test_timeout", "infrastructure"}:
            report_outcome = "error"
        if self.mode in {"non_boolean", "duplicate", "stale"}:
            report_outcome = "resolved"
        aggregate_path.write_bytes(
            canonical_json_bytes(aggregate_report(instance_id, report_outcome))
        )

        if self.mode in {"resolved", "unresolved", "non_boolean", "duplicate"}:
            instance_root.mkdir(parents=True)
            resolved: object = self.mode == "resolved"
            if self.mode == "non_boolean":
                resolved = 1
            (instance_root / "report.json").write_bytes(
                canonical_json_bytes({instance_id: {"resolved": resolved}})
            )
        elif self.mode == "patch_apply_failure":
            instance_root.mkdir(parents=True)
            (instance_root / "run_instance.log").write_text(
                ">>>>> Patch Apply Failed:\nerror: corrupt patch\n",
                encoding="utf-8",
            )
        elif self.mode == "test_timeout":
            instance_root.mkdir(parents=True)
            (instance_root / "run_instance.log").write_text(
                "Test timed out after 1800 seconds.\n",
                encoding="utf-8",
            )
            (instance_root / "test_output.txt").write_text(
                "Timeout error: 1800 seconds exceeded.\n",
                encoding="utf-8",
            )
        elif self.mode == "infrastructure":
            instance_root.mkdir(parents=True)
            (instance_root / "run_instance.log").write_text(
                "docker.errors.APIError: daemon unavailable\n",
                encoding="utf-8",
            )

        if self.mode == "duplicate":
            duplicate_root = instance_root.parent / "wrong-instance"
            duplicate_root.mkdir(parents=True)
            (duplicate_root / "report.json").write_bytes(
                canonical_json_bytes({instance_id: {"resolved": True}})
            )
        if self.mode == "stale":
            old = time.time() - 30
            os.utime(aggregate_path, (old, old))
        return subprocess.CompletedProcess(command, 0, b"stdout", b"stderr")


class OfficialGraderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.harness_root = self.root / "SWE-bench"
        self.harness_root.mkdir()
        module_path = (
            self.harness_root
            / "swebench"
            / "harness"
            / "run_evaluation.py"
        )
        module_path.parent.mkdir(parents=True)
        module_path.write_text("# fake pinned module\n", encoding="utf-8")
        self.dataset_path = self.root / "verified.jsonl"
        self.dataset_path.write_text(
            json.dumps({"instance_id": INSTANCE_ID}) + "\n",
            encoding="utf-8",
        )
        self.config = OfficialGraderConfig(
            python_executable=Path(sys.executable),
            harness_root=self.harness_root,
            dataset_path=self.dataset_path,
            output_root=self.root / "grades",
        )

    def run_fake(self, mode: str, request=None):
        fake = FakeHarness(mode)
        with patch(
            "swebench_triad_eval.official_grader.verify_grader_environment",
            return_value={
                "harness_commit": (
                    "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
                ),
                "harness_tree": "f178530b37202c549b1b2b3300db2da90da648db",
                "dataset_sha256": (
                    "392529c5e79ca273bf0b073be35169beb68c604a26d9aef5514912fc584fa6cb"
                ),
                "docker_socket": str(DOCKER_SOCKET),
            },
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.run",
            side_effect=fake,
        ):
            result = run_official_grader(
                self.config,
                request or grade_request(),
            )
        return result, fake

    def test_invocation_is_single_instance_pinned_and_prediction_is_canonical(
        self,
    ) -> None:
        outcome, fake = self.run_fake("resolved")

        self.assertEqual(
            outcome,
            {
                "instance_id": INSTANCE_ID,
                "arm": "native",
                "resolved": True,
                "failure_class": None,
                "report_sha256": outcome["report_sha256"],
            },
        )
        self.assertEqual(len(fake.calls), 1)
        command, cwd, environment = fake.calls[0]
        self.assertEqual(
            command[:3],
            [
                str(Path(sys.executable)),
                "-m",
                "swebench.harness.run_evaluation",
            ],
        )
        self.assertEqual(
            command[command.index("--dataset_name") + 1],
            str(self.dataset_path.resolve()),
        )
        self.assertEqual(command[command.index("--split") + 1], "test")
        self.assertEqual(command[command.index("--instance_ids") + 1], INSTANCE_ID)
        self.assertEqual(command[command.index("--max_workers") + 1], "1")
        self.assertEqual(command[command.index("--timeout") + 1], "1800")
        self.assertEqual(command[command.index("--namespace") + 1], "swebench")
        self.assertEqual(command[command.index("--force_rebuild") + 1], "false")
        self.assertEqual(command[command.index("--cache_level") + 1], "instance")
        self.assertEqual(command[command.index("--clean") + 1], "false")
        self.assertEqual(environment["PYTHONPATH"], str(self.harness_root.resolve()))
        self.assertEqual(environment["DOCKER_HOST"], f"unix://{DOCKER_SOCKET}")
        prediction_path = Path(command[command.index("--predictions_path") + 1])
        self.assertEqual(
            prediction_path.read_text(encoding="utf-8"),
            json.dumps(
                prediction(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
        )
        report_path = cwd / "official-aggregate-report.json"
        self.assertTrue(report_path.is_file())
        self.assertEqual(
            outcome["report_sha256"],
            hashlib.sha256(report_path.read_bytes()).hexdigest(),
        )

    def test_request_rejects_schema_and_queued_outcome_misuse(self) -> None:
        good = grade_request()
        malformed_prediction = dict(good.prediction)
        malformed_prediction["extra"] = "field"
        with self.assertRaises(ValueError):
            OfficialGradeRequest(
                task_index=good.task_index,
                arm=good.arm,
                generation=good.generation,
                grader_attempt=good.grader_attempt,
                prediction=malformed_prediction,
                accepted_cell=good.accepted_cell,
                queued_handoff=good.queued_handoff,
            )
        resolved_handoff = dict(good.queued_handoff)
        resolved_handoff["official_resolved"] = False
        with self.assertRaises(ValueError):
            OfficialGradeRequest(
                task_index=good.task_index,
                arm=good.arm,
                generation=good.generation,
                grader_attempt=good.grader_attempt,
                prediction=good.prediction,
                accepted_cell=good.accepted_cell,
                queued_handoff=resolved_handoff,
            )

    def test_config_rejects_wrong_docker_socket_and_relative_roots(self) -> None:
        with self.assertRaises(ValueError):
            OfficialGraderConfig(
                python_executable=Path(sys.executable),
                harness_root=self.harness_root,
                dataset_path=self.dataset_path,
                output_root=self.root / "grades",
                docker_socket=self.root / "docker.sock",
            )
        with self.assertRaises(ValueError):
            OfficialGraderConfig(
                python_executable=Path(sys.executable),
                harness_root=Path("relative-harness"),
                dataset_path=self.dataset_path,
                output_root=self.root / "grades",
            )

    def test_non_pinned_import_is_rejected(self) -> None:
        completed = subprocess.CompletedProcess(
            ["python", "-c", "probe"],
            0,
            b"/tmp/site-packages/swebench/harness/run_evaluation.py\n",
            b"",
        )
        with patch(
            "swebench_triad_eval.official_grader.subprocess.run",
            return_value=completed,
        ), self.assertRaises(GraderConfigurationError):
            verify_pinned_import(self.config)

    def test_pinned_import_preserves_venv_python_invocation_path(self) -> None:
        venv_python = self.root / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.symlink_to(Path(sys.executable).resolve())
        config = OfficialGraderConfig(
            python_executable=venv_python,
            harness_root=self.harness_root,
            dataset_path=self.dataset_path,
            output_root=self.root / "venv-import-probe",
        )
        completed = subprocess.CompletedProcess(
            [str(venv_python), "-c", "probe"],
            0,
            (
                str(
                    (
                        self.harness_root
                        / "swebench"
                        / "harness"
                        / "run_evaluation.py"
                    ).resolve()
                )
                + "\n"
            ).encode(),
            b"",
        )
        with patch(
            "swebench_triad_eval.official_grader.subprocess.run",
            return_value=completed,
        ) as run:
            verify_pinned_import(config)
        self.assertEqual(run.call_args.args[0][0], str(venv_python))
        command = grader_command(
            config,
            grade_request(),
            prediction_path=self.root / "prediction.jsonl",
        )
        self.assertEqual(command[0], str(venv_python))

    def test_pinned_import_failure_surfaces_bounded_child_stderr(self) -> None:
        completed = subprocess.CompletedProcess(
            ["python", "-c", "probe"],
            1,
            b"",
            b"ModuleNotFoundError: No module named 'bs4'\n" + b"x" * 8_192,
        )
        with patch(
            "swebench_triad_eval.official_grader.subprocess.run",
            return_value=completed,
        ), self.assertRaises(GraderConfigurationError) as raised:
            verify_pinned_import(self.config)
        message = str(raised.exception)
        self.assertIn("exit code 1", message)
        self.assertIn("ModuleNotFoundError: No module named 'bs4'", message)
        self.assertIn("[stderr truncated]", message)
        self.assertLess(len(message), 5_000)

    def test_empty_patch_patch_failure_and_timeout_are_terminal_false(self) -> None:
        empty, _ = self.run_fake("empty_patch", grade_request(""))
        self.assertEqual(
            (empty["resolved"], empty["failure_class"]),
            (False, "empty_patch"),
        )

        patch_failure, _ = self.run_fake(
            "patch_apply_failure",
            grade_request(),
        )
        self.assertEqual(
            (patch_failure["resolved"], patch_failure["failure_class"]),
            (False, "patch_apply_failure"),
        )

        timeout_request = grade_request()
        timeout_request = OfficialGradeRequest(
            **{
                **timeout_request.to_payload(),
                "grader_attempt": 2,
            }
        )
        timed_out, _ = self.run_fake("test_timeout", timeout_request)
        self.assertEqual(
            (timed_out["resolved"], timed_out["failure_class"]),
            (False, "test_timeout"),
        )

    def test_generic_harness_failure_remains_retryable(self) -> None:
        with self.assertRaises(RetryableGraderError) as raised:
            self.run_fake("infrastructure")
        self.assertEqual(
            raised.exception.failure_class,
            "harness_infrastructure_failure",
        )

    def test_process_timeout_is_retryable_and_same_attempt_reentry_is_stable(
        self,
    ) -> None:
        request = grade_request()
        request = OfficialGradeRequest(
            **{
                **request.to_payload(),
                "grader_attempt": 40,
            }
        )
        timeout = subprocess.TimeoutExpired(
            cmd=["python", "-m", "swebench.harness.run_evaluation"],
            timeout=2_100,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )
        environment_receipt = {
            "harness_commit": "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
            "harness_tree": "f178530b37202c549b1b2b3300db2da90da648db",
            "dataset_sha256": (
                "392529c5e79ca273bf0b073be35169beb68c604a26d9aef5514912fc584fa6cb"
            ),
            "docker_socket": str(DOCKER_SOCKET),
        }
        with patch(
            "swebench_triad_eval.official_grader.verify_grader_environment",
            return_value=environment_receipt,
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.run",
            side_effect=timeout,
        ), self.assertRaises(RetryableGraderError) as first:
            run_official_grader(self.config, request)
        self.assertEqual(first.exception.failure_class, "grader_process_timeout")

        with patch(
            "swebench_triad_eval.official_grader.verify_grader_environment",
            side_effect=AssertionError(
                "durable process receipt must not need live grader"
            ),
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.run"
        ) as runner, self.assertRaises(RetryableGraderError) as second:
            run_official_grader(self.config, request)
        runner.assert_not_called()
        self.assertEqual(second.exception.failure_class, "grader_process_timeout")

    def test_outcome_crash_window_rebuilds_terminal_receipt_without_rerun(
        self,
    ) -> None:
        request = grade_request()
        outcome, _ = self.run_fake("resolved", request)
        attempt_root = grade_attempt_directory(self.config, request)
        result_path = attempt_root / "grade-result.json"
        result_path.unlink()
        with patch(
            "swebench_triad_eval.official_grader.verify_grader_environment",
            side_effect=AssertionError("durable outcome must not need live grader"),
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.run"
        ) as runner:
            recovered = run_official_grader(self.config, request)
        runner.assert_not_called()
        self.assertEqual(recovered, outcome)
        self.assertTrue(result_path.is_file())

    def test_missing_stale_duplicate_and_non_boolean_reports_are_rejected(self) -> None:
        cases = (
            ("missing", RetryableGraderError),
            ("stale", GraderContractError),
            ("duplicate", GraderContractError),
            ("non_boolean", GraderContractError),
        )
        for index, (mode, error_type) in enumerate(cases, start=1):
            with self.subTest(mode=mode):
                request = grade_request()
                request = OfficialGradeRequest(
                    **{
                        **request.to_payload(),
                        "grader_attempt": 10 + index,
                    }
                )
                with self.assertRaises(error_type):
                    self.run_fake(mode, request)

    def test_attempt_path_binds_cell_generation_attempt_and_prediction(self) -> None:
        first = grade_request()
        second = grade_request("different patch\n")
        first_path = grade_attempt_directory(self.config, first)
        second_path = grade_attempt_directory(self.config, second)
        self.assertNotEqual(first_path, second_path)
        self.assertIn("0000-native", str(first_path))
        self.assertIn("generation-00000007", str(first_path))
        self.assertIn("attempt-000001", str(first_path))

    def test_state_rejects_duplicate_outcome_and_denominator_drift(self) -> None:
        cells = (
            ManifestCell(CellKey(0, "native"), INSTANCE_ID, SHA_A),
            ManifestCell(CellKey(0, "amg_compaction_only"), INSTANCE_ID, SHA_B),
            ManifestCell(CellKey(0, "amg_memory"), INSTANCE_ID, SHA_C),
        )
        owner = OwnerIdentity("host", "boot", 101, 1001)
        store = CellStateStore(
            self.root / "state",
            manifest=cells,
            owner=owner,
            owner_is_alive=lambda candidate: True,
            endpoint_validator=lambda row: None,
        )
        for cell in cells:
            accepted = grade_request().accepted_cell
            accepted = {
                **accepted,
                "cell": cell.key.to_payload(),
                "instance_id": cell.instance_id,
                "manifest_cell_sha256": cell.manifest_cell_sha256,
            }
            store.accepted_path(cell.key).parent.mkdir(parents=True, exist_ok=True)
            store.accepted_path(cell.key).write_bytes(canonical_json_bytes(accepted))
        store.record_official_outcome(
            cells[0].key,
            {
                "instance_id": INSTANCE_ID,
                "arm": "native",
                "resolved": False,
                "failure_class": None,
                "report_sha256": SHA_D,
            },
        )
        with self.assertRaises(ImmutableConflictError):
            store.record_official_outcome(
                cells[0].key,
                {
                    "instance_id": INSTANCE_ID,
                    "arm": "native",
                    "resolved": True,
                    "failure_class": None,
                    "report_sha256": SHA_D,
                },
            )
        with self.assertRaises(ValueError):
            store.official_summary()


if __name__ == "__main__":
    unittest.main()
