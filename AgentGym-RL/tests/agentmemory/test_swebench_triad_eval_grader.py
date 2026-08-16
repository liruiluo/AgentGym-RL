from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from swebench_triad_eval.atomic import ImmutableConflictError, canonical_json_bytes
from swebench_triad_eval.official_grader import (
    DOCKER_SOCKET,
    GRADER_OWNER_ENV,
    GraderBusyError,
    GraderConfigurationError,
    GraderContractError,
    OfficialGradeRequest,
    OfficialGraderConfig,
    RetryableGraderError,
    expected_raw_paths,
    grade_attempt_directory,
    grader_command,
    grader_process_is_alive,
    owned_grader_group_members,
    request_binding,
    run_official_grader,
    terminate_owned_grader_group,
    terminate_unreceipted_grader_process_group,
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


class FakePopen:
    def __init__(self, harness: FakeHarness, command, **kwargs) -> None:
        self.harness = harness
        self.command = list(command)
        self.kwargs = kwargs
        self.pid = 424242
        self.returncode = None

    def communicate(self, timeout=None):
        del timeout
        completed = self.harness(self.command, **self.kwargs)
        self.returncode = completed.returncode
        return completed.stdout, completed.stderr


class TimeoutPopen:
    def __init__(self, command, **kwargs) -> None:
        del kwargs
        self.command = list(command)
        self.pid = 434343
        self.returncode = None
        self.calls = 0

    def communicate(self, timeout=None):
        self.calls += 1
        if timeout is not None:
            raise subprocess.TimeoutExpired(
                cmd=self.command,
                timeout=timeout,
                output=b"partial stdout",
                stderr=b"partial stderr",
            )
        self.returncode = -9
        return b"partial stdout", b"partial stderr"


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
            command_ledger_path=self.root / "command-exit-ledger.jsonl",
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
            "swebench_triad_eval.official_grader.subprocess.Popen",
            side_effect=lambda command, **kwargs: FakePopen(
                fake, command, **kwargs
            ),
        ), patch(
            "swebench_triad_eval.official_grader.process_start_ticks",
            return_value=123456,
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
        ledger = [
            json.loads(line)
            for line in self.config.command_ledger_path.read_text().splitlines()
        ]
        self.assertEqual([row["event"] for row in ledger], ["start", "exit"])
        self.assertEqual(ledger[0]["command"], command)
        self.assertEqual(ledger[1]["process_result"]["returncode"], 0)

    def test_live_incomplete_grader_fences_retry_and_next_attempt(self) -> None:
        request = grade_request()
        attempt_root = grade_attempt_directory(self.config, request)
        attempt_root.mkdir(parents=True)
        (attempt_root / "request.json").write_bytes(
            canonical_json_bytes(
                {
                    **{
                        "schema": "swebench_triad_grader_binding_v1",
                        "task_index": 0,
                        "arm": "native",
                        "generation": 7,
                        "grader_attempt": 1,
                        "instance_id": INSTANCE_ID,
                        "prediction_sha256": sha256_json(request.prediction),
                        "harness_commit": "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
                        "harness_tree": "f178530b37202c549b1b2b3300db2da90da648db",
                        "dataset_sha256": (
                            "392529c5e79ca273bf0b073be35169beb"
                            "68c604a26d9aef5514912fc584fa6cb"
                        ),
                        "namespace": "swebench",
                        "timeout_seconds": 1800,
                    },
                    "accepted_cell": dict(request.accepted_cell),
                    "queued_handoff": dict(request.queued_handoff),
                }
            )
        )
        prediction_path = attempt_root / "prediction.jsonl"
        prediction_path.write_text(
            json.dumps(request.prediction, separators=(",", ":")) + "\n"
        )
        command = grader_command(
            self.config, request, prediction_path=prediction_path.resolve()
        )
        (attempt_root / "started.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema": "swebench_triad_grader_started_v2",
                    "started_at_ns": time.time_ns(),
                    "pid": 999,
                    "start_ticks": 888,
                    "command_sha256": sha256_json(command),
                }
            )
        )
        with patch(
            "swebench_triad_eval.official_grader.started_process",
            return_value=(999, 888, True),
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.Popen"
        ) as runner, self.assertRaises(GraderBusyError):
            run_official_grader(self.config, request)
        runner.assert_not_called()

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
            "swebench_triad_eval.official_grader.subprocess.Popen",
            side_effect=TimeoutPopen,
        ), patch(
            "swebench_triad_eval.official_grader.process_start_ticks",
            return_value=123456,
        ), patch(
            "swebench_triad_eval.official_grader.os.killpg"
        ), self.assertRaises(RetryableGraderError) as first:
            run_official_grader(self.config, request)
        self.assertEqual(first.exception.failure_class, "grader_process_timeout")

        with patch(
            "swebench_triad_eval.official_grader.verify_grader_environment",
            side_effect=AssertionError(
                "durable process receipt must not need live grader"
            ),
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.Popen"
        ) as runner, self.assertRaises(RetryableGraderError) as second:
            run_official_grader(self.config, request)
        runner.assert_not_called()
        self.assertEqual(second.exception.failure_class, "grader_process_timeout")

    def test_spawn_failure_reentry_preserves_one_exit_terminal(self) -> None:
        request = OfficialGradeRequest(
            **{**grade_request().to_payload(), "grader_attempt": 45}
        )
        environment_receipt = {
            "harness_commit": "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
            "harness_tree": "f178530b37202c549b1b2b3300db2da90da648db",
            "dataset_sha256": (
                "392529c5e79ca273bf0b073be35169beb"
                "68c604a26d9aef5514912fc584fa6cb"
            ),
            "docker_socket": str(DOCKER_SOCKET),
        }
        with patch(
            "swebench_triad_eval.official_grader.verify_grader_environment",
            return_value=environment_receipt,
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.Popen",
            side_effect=OSError("simulated spawn failure"),
        ), self.assertRaises(RetryableGraderError) as first:
            run_official_grader(self.config, request)
        self.assertEqual(first.exception.failure_class, "grader_spawn_failure")

        with patch(
            "swebench_triad_eval.official_grader.verify_grader_environment",
            side_effect=AssertionError(
                "durable spawn receipt must not rerun environment preflight"
            ),
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.Popen"
        ) as runner, self.assertRaises(RetryableGraderError) as second:
            run_official_grader(self.config, request)
        runner.assert_not_called()
        self.assertEqual(second.exception.failure_class, "grader_spawn_failure")
        ledger = [
            json.loads(line)
            for line in self.config.command_ledger_path.read_text().splitlines()
        ]
        binding = sha256_json(request_binding(request))
        self.assertEqual(
            [(row["event_id"], row["event"]) for row in ledger],
            [(binding + ":start", "start"), (binding + ":exit", "exit")],
        )

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
            "swebench_triad_eval.official_grader.subprocess.Popen"
        ) as runner:
            recovered = run_official_grader(self.config, request)
        runner.assert_not_called()
        self.assertEqual(recovered, outcome)
        self.assertTrue(result_path.is_file())

    def test_pre_receipt_output_is_abandoned_and_retried_in_a_new_attempt(self):
        request = grade_request()
        attempt_root = grade_attempt_directory(self.config, request)
        attempt_root.mkdir(parents=True)
        aggregate_path, _, _, _ = expected_raw_paths(attempt_root, request)
        aggregate_path.write_bytes(
            canonical_json_bytes(aggregate_report(INSTANCE_ID, "resolved"))
        )
        with patch(
            "swebench_triad_eval.official_grader.find_matching_grader_process",
            return_value=None,
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.Popen"
        ) as runner, self.assertRaises(RetryableGraderError) as raised:
            run_official_grader(self.config, request)
        runner.assert_not_called()
        self.assertEqual(
            raised.exception.failure_class, "grader_pre_receipt_attempt_abandoned"
        )
        abandoned = json.loads((attempt_root / "abandoned.json").read_text())
        self.assertEqual(abandoned["reason"], "pre_receipt_output_without_process")
        ledger = [
            json.loads(line)
            for line in self.config.command_ledger_path.read_text().splitlines()
        ]
        binding = sha256_json(request_binding(request))
        self.assertEqual(
            [(row["event_id"], row["event"]) for row in ledger],
            [
                (binding + ":start", "start"),
                (binding + ":abandoned", "abandoned"),
            ],
        )

    def test_durable_abandoned_receipt_is_authoritative_after_ledger_crash(self):
        request = OfficialGradeRequest(
            **{**grade_request().to_payload(), "grader_attempt": 46}
        )
        attempt_root = grade_attempt_directory(self.config, request)
        attempt_root.mkdir(parents=True)
        prediction_path = attempt_root / "prediction.jsonl"
        command = grader_command(
            self.config, request, prediction_path=prediction_path.resolve()
        )
        binding_sha256 = sha256_json(request_binding(request))
        (attempt_root / "launching.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema": "swebench_triad_grader_launching_v1",
                    "binding_sha256": binding_sha256,
                    "started_at_ns": time.time_ns(),
                    "command_sha256": sha256_json(command),
                }
            )
        )
        module = __import__(
            "swebench_triad_eval.official_grader",
            fromlist=["append_command_ledger"],
        )
        real_append = module.append_command_ledger
        failed = False

        def fail_after_abandoned_receipt(config, event):
            nonlocal failed
            if event["event"] == "abandoned" and not failed:
                failed = True
                raise OSError("simulated crash before terminal ledger append")
            return real_append(config, event)

        with patch(
            "swebench_triad_eval.official_grader.find_matching_grader_process",
            return_value=None,
        ), patch(
            "swebench_triad_eval.official_grader.owned_grader_process_groups",
            return_value={999: {999}},
        ), patch(
            "swebench_triad_eval.official_grader.terminate_owned_grader_group",
            return_value={"members_before": [999], "residue": []},
        ), patch(
            "swebench_triad_eval.official_grader.append_command_ledger",
            side_effect=fail_after_abandoned_receipt,
        ), self.assertRaisesRegex(OSError, "terminal ledger"):
            run_official_grader(self.config, request)

        abandoned_path = attempt_root / "abandoned.json"
        abandoned = json.loads(abandoned_path.read_text())
        self.assertEqual(abandoned["reason"], "pre_receipt_group_without_leader")
        original_bytes = abandoned_path.read_bytes()
        with patch(
            "swebench_triad_eval.official_grader.find_matching_grader_process",
            return_value=None,
        ), patch(
            "swebench_triad_eval.official_grader.owned_grader_process_groups",
            return_value={},
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.Popen"
        ) as runner, self.assertRaises(RetryableGraderError) as recovered:
            run_official_grader(self.config, request)
        runner.assert_not_called()
        self.assertEqual(
            recovered.exception.failure_class,
            "grader_pre_receipt_attempt_abandoned",
        )
        self.assertEqual(abandoned_path.read_bytes(), original_bytes)
        ledger = [
            json.loads(line)
            for line in self.config.command_ledger_path.read_text().splitlines()
        ]
        self.assertEqual(
            [(row["event_id"], row["event"]) for row in ledger],
            [
                (binding_sha256 + ":start", "start"),
                (binding_sha256 + ":abandoned", "abandoned"),
            ],
        )

    def test_launching_crash_window_is_abandoned_without_rewriting_receipt(self):
        request = OfficialGradeRequest(
            **{**grade_request().to_payload(), "grader_attempt": 42}
        )
        attempt_root = grade_attempt_directory(self.config, request)
        attempt_root.mkdir(parents=True)
        prediction_path = attempt_root / "prediction.jsonl"
        command = grader_command(
            self.config, request, prediction_path=prediction_path.resolve()
        )
        binding_sha256 = sha256_json(request_binding(request))
        launching = {
            "schema": "swebench_triad_grader_launching_v1",
            "binding_sha256": binding_sha256,
            "started_at_ns": time.time_ns(),
            "command_sha256": sha256_json(command),
        }
        (attempt_root / "launching.json").write_bytes(
            canonical_json_bytes(launching)
        )

        with patch(
            "swebench_triad_eval.official_grader.find_matching_grader_process",
            return_value=None,
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.Popen"
        ) as runner, self.assertRaises(RetryableGraderError) as raised:
            run_official_grader(self.config, request)

        runner.assert_not_called()
        self.assertEqual(
            raised.exception.failure_class,
            "grader_pre_receipt_attempt_abandoned",
        )
        self.assertEqual(
            json.loads((attempt_root / "launching.json").read_text()),
            launching,
        )
        self.assertEqual(
            json.loads((attempt_root / "abandoned.json").read_text())["reason"],
            "launching_without_process",
        )

    def test_adopted_pre_receipt_process_keeps_original_outer_timeout(self):
        request = OfficialGradeRequest(
            **{**grade_request().to_payload(), "grader_attempt": 43}
        )
        attempt_root = grade_attempt_directory(self.config, request)
        attempt_root.mkdir(parents=True)
        prediction_path = attempt_root / "prediction.jsonl"
        command = grader_command(
            self.config, request, prediction_path=prediction_path.resolve()
        )
        binding_sha256 = sha256_json(request_binding(request))
        original_started_at = time.time_ns() - int(
            (self.config.timeout_seconds + 301) * 1_000_000_000
        )
        (attempt_root / "launching.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema": "swebench_triad_grader_launching_v1",
                    "binding_sha256": binding_sha256,
                    "started_at_ns": original_started_at,
                    "command_sha256": sha256_json(command),
                }
            )
        )

        with patch(
            "swebench_triad_eval.official_grader.find_matching_grader_process",
            return_value=(999, 888),
        ), patch(
            "swebench_triad_eval.official_grader.os.getpgid", return_value=999
        ), patch(
            "swebench_triad_eval.official_grader.process_environment_value",
            return_value=binding_sha256,
            create=True,
        ), patch(
            "swebench_triad_eval.official_grader.owned_grader_group_members",
            return_value={999},
        ), patch(
            "swebench_triad_eval.official_grader.terminate_owned_grader_group",
            return_value={"members_before": [999], "residue": []},
            create=True,
        ) as terminate, patch(
            "swebench_triad_eval.official_grader.subprocess.Popen"
        ) as runner, self.assertRaises(RetryableGraderError) as raised:
            run_official_grader(self.config, request)

        runner.assert_not_called()
        self.assertEqual(raised.exception.failure_class, "grader_process_timeout")
        terminate.assert_called_once()
        started = json.loads((attempt_root / "started.json").read_text())
        self.assertEqual(started["started_at_ns"], original_started_at)

    def test_dead_started_leader_cleans_surviving_owned_group_before_retry(self):
        request = OfficialGradeRequest(
            **{**grade_request().to_payload(), "grader_attempt": 44}
        )
        attempt_root = grade_attempt_directory(self.config, request)
        attempt_root.mkdir(parents=True)
        prediction_path = attempt_root / "prediction.jsonl"
        prediction_path.write_bytes(
            json.dumps(request.prediction, separators=(",", ":")).encode()
            + b"\n"
        )
        command = grader_command(
            self.config, request, prediction_path=prediction_path.resolve()
        )
        binding_sha256 = sha256_json(request_binding(request))
        (attempt_root / "started.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema": "swebench_triad_grader_started_v4",
                    "started_at_ns": time.time_ns(),
                    "pid": 999,
                    "pgid": 999,
                    "start_ticks": 888,
                    "command_sha256": sha256_json(command),
                    "owner_binding_sha256": binding_sha256,
                }
            )
        )

        with patch(
            "swebench_triad_eval.official_grader.started_process",
            return_value=(999, 888, False),
        ), patch(
            "swebench_triad_eval.official_grader.terminate_owned_grader_group",
            return_value={"members_before": [1000], "residue": []},
            create=True,
        ) as terminate, patch(
            "swebench_triad_eval.official_grader.subprocess.Popen"
        ) as runner, self.assertRaises(RetryableGraderError) as raised:
            run_official_grader(self.config, request)

        runner.assert_not_called()
        self.assertEqual(raised.exception.failure_class, "grader_attempt_incomplete")
        terminate.assert_called_once()

    def test_exact_owned_grader_group_is_killed_and_recensused(self):
        binding_sha256 = "9" * 64
        with patch(
            "swebench_triad_eval.official_grader.owned_grader_group_members",
            side_effect=({999, 1000}, set()),
        ), patch(
            "swebench_triad_eval.official_grader.os.killpg"
        ) as killpg:
            receipt = terminate_owned_grader_group(
                999,
                binding_sha256,
                timeout_seconds=0.0,
            )
        killpg.assert_called_once_with(999, signal.SIGKILL)
        self.assertEqual(receipt["members_before"], [999, 1000])
        self.assertEqual(receipt["residue"], [])

    def test_unreceipted_cleanup_reaps_leader_before_final_group_census(self):
        binding_sha256 = "6" * 64

        class ReapAwareProcess:
            pid = 999

            def __init__(self):
                self.reaped = False

            def communicate(self):
                self.reaped = True
                return b"", b""

        process = ReapAwareProcess()

        def require_reaped(_pgid, _binding):
            if not process.reaped:
                raise AssertionError("final census preceded leader reap")
            return {"members_before": [], "residue": []}

        with patch(
            "swebench_triad_eval.official_grader.os.getpgid", return_value=999
        ), patch(
            "swebench_triad_eval.official_grader.owned_grader_group_members",
            return_value={999},
        ), patch(
            "swebench_triad_eval.official_grader.os.killpg"
        ) as killpg, patch(
            "swebench_triad_eval.official_grader.terminate_owned_grader_group",
            side_effect=require_reaped,
        ):
            terminate_unreceipted_grader_process_group(process, binding_sha256)
        killpg.assert_called_once_with(999, signal.SIGKILL)

    @unittest.skipUnless(Path("/proc").is_dir(), "requires Linux /proc")
    def test_unreceipted_cleanup_reaps_a_real_owned_subprocess(self):
        binding_sha256 = "5" * 64
        environment = dict(os.environ)
        environment[GRADER_OWNER_ENV] = binding_sha256
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.addCleanup(lambda: process.poll() is None and process.kill())
        terminate_unreceipted_grader_process_group(process, binding_sha256)
        self.assertIsNotNone(process.poll())

    def test_grader_group_with_foreign_member_is_never_signalled(self):
        binding_sha256 = "8" * 64
        with patch(
            "swebench_triad_eval.official_grader.grader_process_group_members",
            return_value={999, 1000},
        ), patch(
            "swebench_triad_eval.official_grader.process_state",
            return_value="S",
        ), patch(
            "swebench_triad_eval.official_grader.process_environment_value",
            side_effect=(binding_sha256, "7" * 64),
        ):
            with self.assertRaisesRegex(GraderContractError, "foreign members"):
                owned_grader_group_members(999, binding_sha256)

    def test_zombie_grader_leader_and_descendants_are_not_live_residue(self):
        binding_sha256 = "4" * 64
        command = ["python", "-m", "swebench.harness.run_evaluation"]
        with patch(
            "swebench_triad_eval.official_grader.process_start_ticks",
            return_value=123,
        ), patch(
            "swebench_triad_eval.official_grader.process_arguments",
            return_value=command,
        ), patch(
            "swebench_triad_eval.official_grader.process_state",
            return_value="Z",
            create=True,
        ):
            self.assertFalse(grader_process_is_alive(999, 123, command))

        with patch(
            "swebench_triad_eval.official_grader.grader_process_group_members",
            return_value={999, 1000},
        ), patch(
            "swebench_triad_eval.official_grader.process_state",
            return_value="Z",
            create=True,
        ), patch(
            "swebench_triad_eval.official_grader.process_environment_value"
        ) as environment:
            self.assertEqual(
                owned_grader_group_members(999, binding_sha256),
                set(),
            )
        environment.assert_not_called()

    def test_post_spawn_receipt_failure_terminates_and_reaps_exact_group(self):
        request = OfficialGradeRequest(
            **{**grade_request().to_payload(), "grader_attempt": 41}
        )
        process = FakePopen(FakeHarness("resolved"), ["placeholder"])
        real_write = __import__(
            "swebench_triad_eval.official_grader", fromlist=["write_immutable_json"]
        ).write_immutable_json

        def fail_started(path, payload):
            if Path(path).name == "started.json":
                raise OSError("simulated receipt crash")
            return real_write(path, payload)

        with patch(
            "swebench_triad_eval.official_grader.verify_grader_environment",
            return_value={
                "harness_commit": "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
                "harness_tree": "f178530b37202c549b1b2b3300db2da90da648db",
                "dataset_sha256": (
                    "392529c5e79ca273bf0b073be35169beb"
                    "68c604a26d9aef5514912fc584fa6cb"
                ),
                "docker_socket": str(DOCKER_SOCKET),
            },
        ), patch(
            "swebench_triad_eval.official_grader.subprocess.Popen",
            return_value=process,
        ), patch(
            "swebench_triad_eval.official_grader.process_start_ticks",
            return_value=123456,
        ), patch(
            "swebench_triad_eval.official_grader.write_immutable_json",
            side_effect=fail_started,
        ), patch(
            "swebench_triad_eval.official_grader.terminate_grader_process_group"
        ) as terminate, self.assertRaises(RetryableGraderError):
            run_official_grader(self.config, request)
        terminate.assert_called_once_with(
            process,
            expected_start_ticks=123456,
            binding_sha256=sha256_json(request_binding(request)),
        )

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
            token = store.acquire(cell.key)
            endpoint = {
                "instance_id": cell.instance_id,
                "arm": cell.key.arm,
                "comparable": True,
                "failure": {"class": None},
                "final_artifact": {"sha256": SHA_D},
                "scorer": {"public_metrics": {"official_resolved": None}},
                "lifecycle": {
                    "close_receipt_ref": "evidence://close/" + SHA_D,
                },
            }
            prediction_row = prediction()
            handoff = {
                "prediction_sha256": sha256_json(prediction_row),
                "official_resolved": None,
                "grader_revision": (
                    "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
                ),
            }
            store.record_endpoint(token, endpoint)
            store.record_prediction(token, prediction_row)
            store.record_handoff(token, handoff)
            store.accept_current_attempt(token)
        grade_token = store.acquire_grade(cells[0].key)
        store.record_official_outcome(
            grade_token,
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
                grade_token,
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
