"""Pinned single-instance SWE-bench v4.1.0 grading."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
from typing import Any, Mapping

from paired_eval.serialization import canonical_json_bytes

from . import ARMS
from .atomic import (
    ensure_private_directory,
    read_json,
    write_immutable_bytes,
    write_immutable_json,
)
from .identity import (
    HARNESS_COMMIT,
    HARNESS_TREE,
    PRODUCTION_DATASET_PINS,
    require_directory,
    require_regular_file,
    sha256_file,
)
from .state import require_sha256, sha256_json


DOCKER_SOCKET = Path(
    "/root/.local/state/amg-external-eval-container-runtime-v1/docker.sock"
)
PREDICTION_FIELDS = (
    "instance_id",
    "model_name_or_path",
    "model_patch",
)
ACCEPTED_CELL_FIELDS = {
    "schema",
    "cell",
    "instance_id",
    "manifest_cell_sha256",
    "attempt_generation",
    "endpoint_sha256",
    "prediction_sha256",
    "handoff_sha256",
}
QUEUED_HANDOFF_FIELDS = {
    "prediction_sha256",
    "official_resolved",
    "grader_revision",
}
OUTCOME_FIELDS = {
    "instance_id",
    "arm",
    "resolved",
    "failure_class",
    "report_sha256",
}
AGGREGATE_REPORT_FIELDS = {
    "total_instances",
    "submitted_instances",
    "completed_instances",
    "resolved_instances",
    "unresolved_instances",
    "empty_patch_instances",
    "error_instances",
    "completed_ids",
    "incomplete_ids",
    "empty_patch_ids",
    "submitted_ids",
    "resolved_ids",
    "unresolved_ids",
    "error_ids",
    "schema_version",
}
REPORT_FRESHNESS_TOLERANCE_NS = 2_000_000_000
PROCESS_TIMEOUT_GRACE_SECONDS = 300
GRADER_OWNER_ENV = "AMG_SWEBENCH_GRADER_BINDING_SHA256"
PROCESS_GROUP_SETTLE_SECONDS = 30.0
MAX_DIAGNOSTIC_LOG_BYTES = 16 * 1024 * 1024
MAX_IMPORT_PROBE_STDERR_BYTES = 4 * 1024


class GraderConfigurationError(RuntimeError):
    """The pinned grader environment is not the requested environment."""


class GraderContractError(RuntimeError):
    """The harness emitted stale, ambiguous, or malformed evidence."""


class RetryableGraderError(RuntimeError):
    """The endpoint is accepted, but official grading needs a fresh attempt."""

    def __init__(
        self,
        message: str,
        *,
        failure_class: str,
        attempt_directory: Path,
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.attempt_directory = attempt_directory


class GraderBusyError(RuntimeError):
    """A previously launched, identity-bound official grader is still live."""



def require_absolute_path(path: Path | str, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return value


@dataclass(frozen=True)
class OfficialGraderConfig:
    python_executable: Path
    harness_root: Path
    dataset_path: Path
    output_root: Path
    docker_socket: Path = DOCKER_SOCKET
    timeout_seconds: int = 1_800
    namespace: str = "swebench"
    command_ledger_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "python_executable",
            "harness_root",
            "dataset_path",
            "output_root",
            "docker_socket",
        ):
            object.__setattr__(
                self,
                name,
                require_absolute_path(getattr(self, name), name),
            )
        if self.docker_socket != DOCKER_SOCKET:
            raise ValueError("official grading requires the isolated Docker socket")
        if self.timeout_seconds != 1_800:
            raise ValueError("official SWE-bench timeout is pinned to 1800 seconds")
        if self.namespace != "swebench":
            raise ValueError("official SWE-bench namespace drifted")
        if self.command_ledger_path is not None:
            object.__setattr__(
                self,
                "command_ledger_path",
                require_absolute_path(
                    self.command_ledger_path, "command_ledger_path"
                ),
            )


@dataclass(frozen=True)
class OfficialGradeRequest:
    task_index: int
    arm: str
    generation: int
    grader_attempt: int
    prediction: Mapping[str, Any]
    accepted_cell: Mapping[str, Any]
    queued_handoff: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("task_index", "generation", "grader_attempt"):
            value = getattr(self, name)
            minimum = 0 if name == "task_index" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} is invalid")
        if self.arm not in ARMS:
            raise ValueError("official grade arm is unsupported")
        validate_prediction(self.prediction)
        prediction_sha256 = sha256_json(self.prediction)

        accepted = self.accepted_cell
        if not isinstance(accepted, Mapping) or set(accepted) != ACCEPTED_CELL_FIELDS:
            raise ValueError("accepted cell fields are not canonical")
        if accepted["schema"] != "swebench_triad_accepted_cell_v1":
            raise ValueError("accepted cell schema drifted")
        if accepted["cell"] != {
            "task_index": self.task_index,
            "arm": self.arm,
        }:
            raise ValueError("accepted cell identity drifted")
        if accepted["instance_id"] != self.prediction["instance_id"]:
            raise ValueError("accepted instance identity drifted")
        if accepted["attempt_generation"] != self.generation:
            raise ValueError("accepted generation drifted")
        for name in (
            "manifest_cell_sha256",
            "endpoint_sha256",
            "prediction_sha256",
            "handoff_sha256",
        ):
            require_sha256(accepted[name], f"accepted {name}")
        if accepted["prediction_sha256"] != prediction_sha256:
            raise ValueError("accepted prediction digest drifted")

        handoff = self.queued_handoff
        if not isinstance(handoff, Mapping) or set(handoff) != QUEUED_HANDOFF_FIELDS:
            raise ValueError("queued grader handoff fields are not canonical")
        if handoff["official_resolved"] is not None:
            raise ValueError("queued handoff cannot be used as an official outcome")
        if handoff["grader_revision"] != HARNESS_COMMIT:
            raise ValueError("queued grader revision drifted")
        require_sha256(handoff["prediction_sha256"], "queued prediction")
        if handoff["prediction_sha256"] != prediction_sha256:
            raise ValueError("queued prediction digest drifted")
        if accepted["handoff_sha256"] != sha256_json(handoff):
            raise ValueError("accepted grader handoff digest drifted")

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_index": self.task_index,
            "arm": self.arm,
            "generation": self.generation,
            "grader_attempt": self.grader_attempt,
            "prediction": dict(self.prediction),
            "accepted_cell": dict(self.accepted_cell),
            "queued_handoff": dict(self.queued_handoff),
        }


def validate_prediction(value: Any) -> None:
    if not isinstance(value, Mapping) or tuple(value) != PREDICTION_FIELDS:
        raise ValueError("official prediction schema or field order drifted")
    if any(not isinstance(value[name], str) for name in PREDICTION_FIELDS):
        raise ValueError("official prediction fields must be text")
    if not value["instance_id"] or not value["model_name_or_path"]:
        raise ValueError("official prediction identity must be nonempty")
    model_name = value["model_name_or_path"]
    if any(part in {"", ".", ".."} for part in model_name.split("/")):
        raise ValueError("official prediction model label is unsafe")


def grader_environment(
    config: OfficialGraderConfig,
    *,
    owner_binding_sha256: str | None = None,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(config.harness_root.resolve(strict=True)),
            "PYTHONNOUSERSITE": "1",
            "DOCKER_HOST": f"unix://{config.docker_socket}",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if owner_binding_sha256 is not None:
        require_sha256(owner_binding_sha256, "grader owner binding")
        environment[GRADER_OWNER_ENV] = owner_binding_sha256
    return environment


def run_checked(command: list[str], *, timeout: int = 60) -> bytes:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        stderr = bytes(completed.stderr or b"").decode("utf-8", errors="replace")
        raise GraderConfigurationError(
            f"pinned grader identity command failed: {stderr.strip()}"
        )
    return bytes(completed.stdout or b"")


def verify_harness_checkout(harness_root: Path) -> dict[str, str]:
    root = require_directory(harness_root, "SWE-bench harness")
    commit = run_checked(["git", "-C", str(root), "rev-parse", "HEAD"]).decode(
        "ascii"
    ).strip()
    tree = run_checked(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"]
    ).decode("ascii").strip()
    status = run_checked(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    ).decode("utf-8", errors="strict")
    if commit != HARNESS_COMMIT or tree != HARNESS_TREE:
        raise GraderConfigurationError("SWE-bench harness commit or tree drifted")
    if status:
        raise GraderConfigurationError("SWE-bench harness checkout is not clean")
    module_path = root / "swebench" / "harness" / "run_evaluation.py"
    require_regular_file(module_path, "SWE-bench run_evaluation module")
    return {"harness_commit": commit, "harness_tree": tree}


def verify_dataset(dataset_path: Path) -> dict[str, Any]:
    path = require_regular_file(dataset_path, "SWE-bench Verified dataset")
    digest = sha256_file(path)
    if digest != PRODUCTION_DATASET_PINS.jsonl_sha256:
        raise GraderConfigurationError("SWE-bench Verified dataset digest drifted")
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GraderConfigurationError(
            "SWE-bench Verified dataset is invalid"
        ) from error
    if len(rows) != PRODUCTION_DATASET_PINS.row_count:
        raise GraderConfigurationError("SWE-bench Verified dataset row count drifted")
    return {"dataset_sha256": digest, "dataset_rows": len(rows)}


def verify_docker_socket(socket_path: Path) -> dict[str, str]:
    if socket_path != DOCKER_SOCKET:
        raise GraderConfigurationError("isolated Docker socket path drifted")
    try:
        info = socket_path.lstat()
    except OSError as error:
        raise GraderConfigurationError(
            "isolated Docker socket is unavailable"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISSOCK(info.st_mode):
        raise GraderConfigurationError("isolated Docker endpoint is not a socket")
    return {"docker_socket": str(socket_path)}


def verify_python_executable(path: Path) -> Path:
    invocation_path = Path(path)
    if not invocation_path.is_absolute():
        raise GraderConfigurationError(
            "official grader Python invocation path must be absolute"
        )
    try:
        resolved = invocation_path.resolve(strict=True)
    except OSError as error:
        raise GraderConfigurationError(
            "official grader Python is unavailable"
        ) from error
    executable = require_regular_file(resolved, "official grader Python")
    if not os.access(invocation_path, os.X_OK):
        raise GraderConfigurationError("official grader Python is not executable")
    return invocation_path


def verify_pinned_import(config: OfficialGraderConfig) -> dict[str, str]:
    executable = verify_python_executable(config.python_executable)
    root = require_directory(config.harness_root, "SWE-bench harness")
    output_root = ensure_private_directory(config.output_root)
    probe = (
        "from pathlib import Path; "
        "import swebench.harness.run_evaluation as module; "
        "print(Path(module.__file__).resolve())"
    )
    completed = subprocess.run(
        [str(executable), "-c", probe],
        cwd=output_root,
        env=grader_environment(config),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        stderr = bytes(completed.stderr or b"")
        bounded_stderr = stderr[:MAX_IMPORT_PROBE_STDERR_BYTES].decode(
            "utf-8",
            errors="replace",
        ).strip()
        if len(stderr) > MAX_IMPORT_PROBE_STDERR_BYTES:
            bounded_stderr += " [stderr truncated]"
        if not bounded_stderr:
            bounded_stderr = "<empty stderr>"
        raise GraderConfigurationError(
            "pinned SWE-bench import probe failed with exit code "
            f"{completed.returncode}: {bounded_stderr}"
        )
    try:
        imported = Path(bytes(completed.stdout or b"").decode("utf-8").strip())
    except UnicodeError as error:
        raise GraderConfigurationError("pinned import path is not UTF-8") from error
    expected = (root / "swebench" / "harness" / "run_evaluation.py").resolve(
        strict=True
    )
    if imported != expected:
        raise GraderConfigurationError("Python imported a non-pinned SWE-bench harness")
    return {"run_evaluation_module": str(imported)}


def verify_grader_environment(config: OfficialGraderConfig) -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    receipt.update(verify_harness_checkout(config.harness_root))
    receipt.update(verify_dataset(config.dataset_path))
    receipt.update(verify_docker_socket(config.docker_socket))
    receipt.update(verify_pinned_import(config))
    receipt["python_executable"] = str(
        verify_python_executable(config.python_executable)
    )
    return receipt


def verify_instance_in_dataset(dataset_path: Path, instance_id: str) -> None:
    matches = 0
    try:
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise GraderConfigurationError("dataset row is not an object")
            if row.get("instance_id") == instance_id:
                matches += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GraderConfigurationError("cannot verify grader task identity") from error
    if matches != 1:
        raise GraderConfigurationError(
            "official grader instance is missing or duplicated in the dataset"
        )


def request_binding(request: OfficialGradeRequest) -> dict[str, Any]:
    request.validate()
    return {
        "schema": "swebench_triad_grader_binding_v1",
        "task_index": request.task_index,
        "arm": request.arm,
        "generation": request.generation,
        "grader_attempt": request.grader_attempt,
        "instance_id": request.prediction["instance_id"],
        "prediction_sha256": sha256_json(request.prediction),
        "harness_commit": HARNESS_COMMIT,
        "harness_tree": HARNESS_TREE,
        "dataset_sha256": PRODUCTION_DATASET_PINS.jsonl_sha256,
        "namespace": "swebench",
        "timeout_seconds": 1_800,
    }


def grade_attempt_directory(
    config: OfficialGraderConfig,
    request: OfficialGradeRequest,
) -> Path:
    binding_sha256 = sha256_json(request_binding(request))
    return (
        config.output_root
        / f"{request.task_index:04d}-{request.arm}"
        / f"generation-{request.generation:08d}"
        / f"attempt-{request.grader_attempt:06d}-{binding_sha256}"
    )


def grader_run_id(request: OfficialGradeRequest) -> str:
    binding_sha256 = sha256_json(request_binding(request))
    return (
        f"amg-sbv-{request.task_index:04d}-{request.arm}"
        f"-g{request.generation:08d}-a{request.grader_attempt:06d}"
        f"-{binding_sha256[:16]}"
    )


def prediction_jsonl_bytes(prediction: Mapping[str, Any]) -> bytes:
    validate_prediction(prediction)
    ordered = {name: prediction[name] for name in PREDICTION_FIELDS}
    return (
        json.dumps(
            ordered,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def grader_command(
    config: OfficialGraderConfig,
    request: OfficialGradeRequest,
    *,
    prediction_path: Path,
) -> list[str]:
    return [
        str(verify_python_executable(config.python_executable)),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(config.dataset_path.resolve(strict=True)),
        "--split",
        "test",
        "--instance_ids",
        request.prediction["instance_id"],
        "--predictions_path",
        str(prediction_path),
        "--max_workers",
        "1",
        "--timeout",
        "1800",
        "--force_rebuild",
        "false",
        "--cache_level",
        "instance",
        "--clean",
        "false",
        "--namespace",
        "swebench",
        "--run_id",
        grader_run_id(request),
    ]


def safe_environment_receipt(config: OfficialGraderConfig) -> dict[str, str]:
    return {
        "PYTHONPATH": str(config.harness_root.resolve(strict=True)),
        "PYTHONNOUSERSITE": "1",
        "DOCKER_HOST": f"unix://{config.docker_socket}",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def path_is_fresh(path: Path, started_at_ns: int) -> bool:
    info = path.lstat()
    return info.st_mtime_ns + REPORT_FRESHNESS_TOLERANCE_NS >= started_at_ns


def require_real_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as error:
        raise GraderContractError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise GraderContractError(f"{label} is not a real regular file")
    return path


def load_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GraderContractError(f"{label} is malformed") from error


def validate_aggregate_report(
    report: Any,
    *,
    instance_id: str,
) -> str:
    if not isinstance(report, Mapping) or set(report) != AGGREGATE_REPORT_FIELDS:
        raise GraderContractError("official aggregate report fields drifted")
    if type(report["schema_version"]) is not int or report["schema_version"] != 2:
        raise GraderContractError("official aggregate report schema drifted")
    for name in (
        "total_instances",
        "submitted_instances",
        "completed_instances",
        "resolved_instances",
        "unresolved_instances",
        "empty_patch_instances",
        "error_instances",
    ):
        if type(report[name]) is not int or report[name] < 0:
            raise GraderContractError("official aggregate count is invalid")
    list_fields = (
        "completed_ids",
        "incomplete_ids",
        "empty_patch_ids",
        "submitted_ids",
        "resolved_ids",
        "unresolved_ids",
        "error_ids",
    )
    for name in list_fields:
        values = report[name]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or values != sorted(values)
            or len(values) != len(set(values))
        ):
            raise GraderContractError("official aggregate ID ledger is invalid")
        if set(values) - {instance_id}:
            raise GraderContractError("official aggregate contains a foreign instance")
    if (
        report["total_instances"] != 1
        or report["submitted_instances"] != 1
        or report["submitted_ids"] != [instance_id]
        or report["incomplete_ids"]
    ):
        raise GraderContractError("official aggregate denominator drifted")
    count_to_ids = {
        "completed_instances": "completed_ids",
        "resolved_instances": "resolved_ids",
        "unresolved_instances": "unresolved_ids",
        "empty_patch_instances": "empty_patch_ids",
        "error_instances": "error_ids",
    }
    if any(report[count] != len(report[ids]) for count, ids in count_to_ids.items()):
        raise GraderContractError("official aggregate counts disagree with IDs")
    if report["completed_ids"] != sorted(
        report["resolved_ids"] + report["unresolved_ids"]
    ):
        raise GraderContractError("official completed ledger drifted")
    categories = {
        name
        for name, field in (
            ("resolved", "resolved_ids"),
            ("unresolved", "unresolved_ids"),
            ("empty_patch", "empty_patch_ids"),
            ("error", "error_ids"),
        )
        if report[field] == [instance_id]
    }
    if len(categories) != 1:
        raise GraderContractError("official aggregate outcome is ambiguous")
    return categories.pop()


def validate_instance_report(
    payload: bytes,
    *,
    instance_id: str,
    expected_resolved: bool,
) -> None:
    report = load_json_bytes(payload, "official instance report")
    if not isinstance(report, Mapping) or set(report) != {instance_id}:
        raise GraderContractError("official instance report identity drifted")
    details = report[instance_id]
    if not isinstance(details, Mapping) or type(details.get("resolved")) is not bool:
        raise GraderContractError("official instance outcome is not boolean")
    if details["resolved"] is not expected_resolved:
        raise GraderContractError("aggregate and instance reports disagree")


def read_bounded_text(path: Path) -> str:
    path = require_real_file(path, "official grader diagnostic log")
    size = path.stat().st_size
    if size > MAX_DIAGNOSTIC_LOG_BYTES:
        raise GraderContractError("official grader diagnostic log is oversized")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise GraderContractError(
            "cannot read official grader diagnostic log"
        ) from error


def write_retryable_receipt(
    attempt_root: Path,
    failure_class: str,
) -> None:
    write_immutable_json(
        attempt_root / "grade-result.json",
        {
            "schema": "swebench_triad_grade_result_v1",
            "status": "retryable",
            "failure_class": failure_class,
        },
    )


def write_terminal_receipt(
    attempt_root: Path,
    outcome: Mapping[str, Any],
) -> None:
    write_immutable_json(
        attempt_root / "grade-result.json",
        {
            "schema": "swebench_triad_grade_result_v1",
            "status": "terminal",
            "failure_class": outcome["failure_class"],
            "resolved": outcome["resolved"],
            "report_sha256": outcome["report_sha256"],
        },
    )


def raise_retryable(
    attempt_root: Path,
    failure_class: str,
    message: str,
) -> None:
    write_retryable_receipt(attempt_root, failure_class)
    raise RetryableGraderError(
        message,
        failure_class=failure_class,
        attempt_directory=attempt_root,
    )


def expected_raw_paths(
    attempt_root: Path,
    request: OfficialGradeRequest,
) -> tuple[Path, Path, Path, Path]:
    run_id = grader_run_id(request)
    model = request.prediction["model_name_or_path"].replace("/", "__")
    instance_root = (
        attempt_root
        / "logs"
        / "run_evaluation"
        / run_id
        / model
        / request.prediction["instance_id"]
    )
    aggregate = attempt_root / f"{model}.{run_id}.json"
    return (
        aggregate,
        instance_root / "report.json",
        instance_root / "run_instance.log",
        instance_root / "test_output.txt",
    )


def finish_grade_attempt(
    attempt_root: Path,
    request: OfficialGradeRequest,
    *,
    command: list[str] | None = None,
) -> dict[str, Any]:
    process_result = read_json(attempt_root / "process-result.json")
    if not isinstance(process_result, Mapping):
        raise GraderContractError("grader process receipt is invalid")
    process_status = process_result.get("status")
    retryable_process_failures = {
        "process_timeout": "grader_process_timeout",
        "spawn_error": "grader_spawn_failure",
        "post_spawn_error": "grader_post_spawn_failure",
    }
    if process_status in retryable_process_failures:
        failure_class = retryable_process_failures[process_status]
        raise_retryable(
            attempt_root,
            failure_class,
            "official grader process did not complete",
        )
    if process_status != "completed":
        raise GraderContractError("grader process status is invalid")
    if process_result.get("returncode") != 0:
        raise_retryable(
            attempt_root,
            "grader_process_failure",
            "official grader process exited nonzero",
        )
    started = read_json(attempt_root / "started.json")
    if command is not None:
        started_process(attempt_root, command)
    started_at_ns = (
        started.get("started_at_ns")
        if isinstance(started, Mapping)
        else None
    )
    if type(started_at_ns) is not int or started_at_ns <= 0:
        raise GraderContractError("grader start receipt is invalid")

    aggregate_path, instance_report_path, instance_log_path, test_output_path = (
        expected_raw_paths(attempt_root, request)
    )
    run_id = grader_run_id(request)
    aggregate_candidates = list(attempt_root.glob(f"*.{run_id}.json"))
    if not aggregate_candidates:
        raise_retryable(
            attempt_root,
            "missing_aggregate_report",
            "official grader did not emit an aggregate report",
        )
    if aggregate_candidates != [aggregate_path]:
        raise GraderContractError("official grader emitted duplicate aggregate reports")
    aggregate_path = require_real_file(aggregate_path, "official aggregate report")
    if not path_is_fresh(aggregate_path, started_at_ns):
        raise GraderContractError("official aggregate report is stale")
    aggregate_bytes = aggregate_path.read_bytes()
    write_immutable_bytes(
        attempt_root / "official-aggregate-report.json",
        aggregate_bytes,
    )
    aggregate = load_json_bytes(aggregate_bytes, "official aggregate report")
    category = validate_aggregate_report(
        aggregate,
        instance_id=request.prediction["instance_id"],
    )

    run_root = attempt_root / "logs" / "run_evaluation" / run_id
    report_candidates = list(run_root.rglob("report.json")) if run_root.exists() else []
    outcome: dict[str, Any]
    if category in {"resolved", "unresolved"}:
        if report_candidates != [instance_report_path]:
            raise GraderContractError("official grader report is absent or duplicated")
        report_path = require_real_file(
            instance_report_path,
            "official instance report",
        )
        if not path_is_fresh(report_path, started_at_ns):
            raise GraderContractError("official instance report is stale")
        report_bytes = report_path.read_bytes()
        expected_resolved = category == "resolved"
        validate_instance_report(
            report_bytes,
            instance_id=request.prediction["instance_id"],
            expected_resolved=expected_resolved,
        )
        write_immutable_bytes(
            attempt_root / "official-instance-report.json",
            report_bytes,
        )
        outcome = {
            "instance_id": request.prediction["instance_id"],
            "arm": request.arm,
            "resolved": expected_resolved,
            "failure_class": None,
            "report_sha256": hashlib.sha256(aggregate_bytes).hexdigest(),
        }
    elif category == "empty_patch":
        if request.prediction["model_patch"] != "":
            raise GraderContractError("harness reported an unexpected empty patch")
        if report_candidates:
            raise GraderContractError("empty patch unexpectedly emitted a report")
        outcome = {
            "instance_id": request.prediction["instance_id"],
            "arm": request.arm,
            "resolved": False,
            "failure_class": "empty_patch",
            "report_sha256": hashlib.sha256(aggregate_bytes).hexdigest(),
        }
    else:
        if report_candidates:
            raise GraderContractError("failed harness run emitted an ambiguous report")
        diagnostic = ""
        if instance_log_path.exists():
            diagnostic += read_bounded_text(instance_log_path)
        if test_output_path.exists():
            diagnostic += "\n" + read_bounded_text(test_output_path)
        patch_failed = ">>>>> Patch Apply Failed" in diagnostic
        test_timed_out = (
            "Test timed out after " in diagnostic
            or "Timeout error: " in diagnostic
        )
        if patch_failed and test_timed_out:
            raise GraderContractError("official grader failure class is ambiguous")
        if patch_failed:
            failure_class = "patch_apply_failure"
        elif test_timed_out:
            failure_class = "test_timeout"
        else:
            raise_retryable(
                attempt_root,
                "harness_infrastructure_failure",
                "official harness failed without a terminal task classification",
            )
        outcome = {
            "instance_id": request.prediction["instance_id"],
            "arm": request.arm,
            "resolved": False,
            "failure_class": failure_class,
            "report_sha256": hashlib.sha256(aggregate_bytes).hexdigest(),
        }

    validate_outcome(outcome, request)
    write_immutable_json(attempt_root / "official-outcome.json", outcome)
    write_terminal_receipt(attempt_root, outcome)
    return outcome


def validate_outcome(
    outcome: Any,
    request: OfficialGradeRequest,
) -> dict[str, Any]:
    if not isinstance(outcome, Mapping) or set(outcome) != OUTCOME_FIELDS:
        raise GraderContractError("official outcome fields are not canonical")
    if (
        outcome["instance_id"] != request.prediction["instance_id"]
        or outcome["arm"] != request.arm
    ):
        raise GraderContractError("official outcome identity drifted")
    if type(outcome["resolved"]) is not bool:
        raise GraderContractError("official outcome is not boolean")
    failure_class = outcome["failure_class"]
    if failure_class is not None and (
        not isinstance(failure_class, str) or not failure_class
    ):
        raise GraderContractError("official failure class is invalid")
    require_sha256(outcome["report_sha256"], "official report")
    return dict(outcome)


def bytes_or_empty(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise GraderContractError("grader process output has an unsupported type")


def process_stat_fields(pid: int) -> list[str]:
    if type(pid) is not int or pid <= 0:
        raise ValueError("grader process PID is invalid")
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError as error:
        raise ProcessLookupError(pid) from error
    close = stat_line.rfind(")")
    fields = stat_line[close + 2 :].split() if close >= 0 else []
    if (
        len(fields) <= 19
        or len(fields[0]) != 1
        or not fields[19].isdigit()
    ):
        raise GraderContractError("grader process stat is malformed")
    return fields


def process_state(pid: int) -> str:
    return process_stat_fields(pid)[0]


def process_start_ticks(pid: int) -> int:
    fields = process_stat_fields(pid)
    return int(fields[19])


def process_arguments(pid: int) -> list[str]:
    try:
        payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as error:
        raise ProcessLookupError(pid) from error
    values = [
        value.decode("utf-8", errors="surrogateescape")
        for value in payload.split(b"\0")
        if value
    ]
    if not values:
        raise ProcessLookupError(pid)
    return values


def process_environment_value(pid: int, name: str) -> str | None:
    if not name or "=" in name or "\0" in name:
        raise ValueError("process environment name is invalid")
    try:
        payload = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as error:
        raise ProcessLookupError(pid) from error
    prefix = name.encode("utf-8") + b"="
    matches = [
        value[len(prefix) :]
        for value in payload.split(b"\0")
        if value.startswith(prefix)
    ]
    if len(matches) > 1:
        raise GraderContractError("grader owner environment is duplicated")
    if not matches:
        return None
    try:
        return matches[0].decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise GraderContractError("grader owner environment is not UTF-8") from error


def grader_process_group_members(pgid: int) -> set[int]:
    if type(pgid) is not int or pgid <= 0:
        raise ValueError("grader process-group ID is invalid")
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return set()
    members: set[int] = set()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if process_state(pid) == "Z":
                continue
            current_pgid = os.getpgid(pid)
        except (OSError, ProcessLookupError):
            continue
        if current_pgid == pgid:
            members.add(pid)
    return members


def owned_grader_group_members(pgid: int, binding_sha256: str) -> set[int]:
    require_sha256(binding_sha256, "grader owner binding")
    members = grader_process_group_members(pgid)
    owned: set[int] = set()
    foreign: list[int] = []
    for pid in sorted(members):
        try:
            if process_state(pid) == "Z":
                continue
            marker = process_environment_value(pid, GRADER_OWNER_ENV)
        except ProcessLookupError:
            continue
        if marker == binding_sha256:
            owned.add(pid)
        else:
            foreign.append(pid)
    if foreign:
        raise GraderContractError(
            "refusing to signal a grader group with foreign members: "
            + ",".join(str(pid) for pid in foreign)
        )
    return owned


def owned_grader_process_groups(binding_sha256: str) -> dict[int, set[int]]:
    require_sha256(binding_sha256, "grader owner binding")
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return {}
    groups: dict[int, set[int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if process_state(pid) == "Z":
                continue
            if process_environment_value(pid, GRADER_OWNER_ENV) != binding_sha256:
                continue
            pgid = os.getpgid(pid)
        except (OSError, ProcessLookupError):
            continue
        groups.setdefault(pgid, set()).add(pid)
    return groups


def terminate_owned_grader_group(
    pgid: int,
    binding_sha256: str,
    *,
    timeout_seconds: float = PROCESS_GROUP_SETTLE_SECONDS,
) -> dict[str, Any]:
    if timeout_seconds < 0:
        raise ValueError("grader group timeout cannot be negative")
    before = owned_grader_group_members(pgid, binding_sha256)
    if before:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_seconds
    residue = owned_grader_group_members(pgid, binding_sha256)
    while residue and time.monotonic() < deadline:
        time.sleep(0.05)
        residue = owned_grader_group_members(pgid, binding_sha256)
    if residue:
        raise GraderContractError(
            "owned grader process group did not become empty: "
            + ",".join(str(pid) for pid in sorted(residue))
        )
    return {
        "pgid": pgid,
        "members_before": sorted(before),
        "residue": [],
    }


def grader_process_is_alive(
    pid: int,
    start_ticks: int,
    command: list[str],
) -> bool:
    try:
        return (
            process_state(pid) != "Z"
            and process_start_ticks(pid) == start_ticks
            and process_arguments(pid) == command
        )
    except ProcessLookupError:
        return False


def find_matching_grader_process(command: list[str]) -> tuple[int, int] | None:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    matches = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if process_state(pid) != "Z" and process_arguments(pid) == command:
                matches.append((pid, process_start_ticks(pid)))
        except (OSError, ProcessLookupError):
            continue
    if len(matches) > 1:
        raise GraderContractError("multiple identity-matching grader processes are live")
    return matches[0] if matches else None


def append_command_ledger(
    config: OfficialGraderConfig,
    event: Mapping[str, Any],
) -> None:
    path = config.command_ledger_path
    if path is None:
        return
    if not isinstance(event, Mapping) or not isinstance(event.get("event_id"), str):
        raise GraderContractError("command-ledger event is invalid")
    ensure_private_directory(path.parent)
    payload = canonical_json_bytes(dict(event)) + b"\n"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "r+b", closefd=False) as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            existing = stream.read().splitlines()
            for line in existing:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise GraderContractError(
                        "command-exit ledger is malformed"
                    ) from error
                if row.get("event_id") == event["event_id"]:
                    if canonical_json_bytes(row) + b"\n" != payload:
                        raise GraderContractError(
                            "command-ledger event identity conflicted"
                        )
                    return
            stream.seek(0, os.SEEK_END)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def read_started_receipt(
    attempt_root: Path,
    command: list[str],
) -> dict[str, Any]:
    started = read_json(attempt_root / "started.json")
    if not isinstance(started, Mapping):
        raise GraderContractError("grader start receipt is invalid")
    schema = started.get("schema")
    v2_fields = {
        "schema",
        "started_at_ns",
        "pid",
        "start_ticks",
        "command_sha256",
    }
    v3_fields = v2_fields | {"pgid"}
    v4_fields = v3_fields | {"owner_binding_sha256"}
    if (schema == "swebench_triad_grader_started_v2" and set(started) != v2_fields) or (
        schema == "swebench_triad_grader_started_v3" and set(started) != v3_fields
    ) or (
        schema == "swebench_triad_grader_started_v4" and set(started) != v4_fields
    ):
        raise GraderContractError("grader start receipt is invalid")
    if schema not in {
        "swebench_triad_grader_started_v2",
        "swebench_triad_grader_started_v3",
        "swebench_triad_grader_started_v4",
    }:
        raise GraderContractError("grader start receipt schema drifted")
    pid = started.get("pid")
    ticks = started.get("start_ticks")
    if type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks <= 0:
        raise GraderContractError("grader process identity is invalid")
    if schema in {
        "swebench_triad_grader_started_v3",
        "swebench_triad_grader_started_v4",
    } and started.get("pgid") != pid:
        raise GraderContractError("grader process-group identity drifted")
    if schema == "swebench_triad_grader_started_v4":
        require_sha256(started.get("owner_binding_sha256"), "grader owner binding")
    if started.get("command_sha256") != sha256_json(command):
        raise GraderContractError("grader process command digest drifted")
    return dict(started)


def started_process(
    attempt_root: Path,
    command: list[str],
) -> tuple[int, int, bool]:
    started = read_started_receipt(attempt_root, command)
    pid = started["pid"]
    ticks = started["start_ticks"]
    alive = grader_process_is_alive(pid, ticks, command)
    if alive:
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            alive = False
        else:
            if pgid != pid:
                raise GraderContractError("live grader escaped its exact process group")
    return pid, ticks, alive


def read_launching_receipt(
    attempt_root: Path,
    command: list[str],
    binding_sha256: str,
) -> dict[str, Any] | None:
    path = attempt_root / "launching.json"
    if not path.exists():
        return None
    value = read_json(path)
    fields = {
        "schema",
        "binding_sha256",
        "started_at_ns",
        "command_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise GraderContractError("grader launching receipt is invalid")
    if value.get("schema") != "swebench_triad_grader_launching_v1":
        raise GraderContractError("grader launching receipt schema drifted")
    if value.get("binding_sha256") != binding_sha256:
        raise GraderContractError("grader launching binding drifted")
    started_at_ns = value.get("started_at_ns")
    if type(started_at_ns) is not int or started_at_ns <= 0:
        raise GraderContractError("grader launching time is invalid")
    if value.get("command_sha256") != sha256_json(command):
        raise GraderContractError("grader launching command drifted")
    return dict(value)


def terminate_grader_process_group(
    process: subprocess.Popen[Any],
    *,
    expected_start_ticks: int,
    binding_sha256: str,
) -> None:
    """Kill and reap only the new session created for this exact Popen."""

    pid = getattr(process, "pid", None)
    if type(pid) is not int or pid <= 0:
        raise GraderContractError("spawned grader PID is invalid")
    try:
        current_ticks = process_start_ticks(pid)
    except ProcessLookupError:
        current_ticks = None
    if current_ticks is not None and current_ticks != expected_start_ticks:
        raise GraderContractError("refusing to signal a reused grader PID")
    terminate_unreceipted_grader_process_group(process, binding_sha256)


def terminate_unreceipted_grader_process_group(
    process: subprocess.Popen[Any],
    binding_sha256: str,
) -> None:
    pid = getattr(process, "pid", None)
    if type(pid) is not int or pid <= 0:
        raise GraderContractError("spawned grader PID is invalid")
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        pgid = None
    if pgid is not None and pgid != pid:
        raise GraderContractError("spawned grader did not own a new session")
    members = owned_grader_group_members(pid, binding_sha256)
    if members:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.communicate()
    except (OSError, ValueError):
        try:
            process.wait(timeout=30)
        except (AttributeError, OSError, subprocess.TimeoutExpired):
            pass
    terminate_owned_grader_group(pid, binding_sha256)


def abandon_pre_receipt_attempt(
    config: OfficialGraderConfig,
    request: OfficialGradeRequest,
    attempt_root: Path,
    ledger_base: Mapping[str, Any],
    command: list[str],
    *,
    reason: str,
) -> None:
    del request
    binding_sha256 = str(ledger_base["binding_sha256"])
    append_start_ledger(config, attempt_root, command, ledger_base)
    write_immutable_json(
        attempt_root / "abandoned.json",
        {
            "schema": "swebench_triad_grader_abandoned_v1",
            "binding_sha256": binding_sha256,
            "reason": reason,
        },
    )
    append_command_ledger(
        config,
        {
            **dict(ledger_base),
            "event_id": binding_sha256 + ":abandoned",
            "event": "abandoned",
            "reason": reason,
        },
    )
    raise_retryable(
        attempt_root,
        "grader_pre_receipt_attempt_abandoned",
        "grader pre-receipt attempt was durably abandoned; use a new attempt",
    )


def recover_abandoned_attempt(
    config: OfficialGraderConfig,
    request: OfficialGradeRequest,
    attempt_root: Path,
    ledger_base: Mapping[str, Any],
    command: list[str],
) -> None:
    value = read_json(attempt_root / "abandoned.json")
    fields = {"schema", "binding_sha256", "reason"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise GraderContractError("grader abandoned receipt is invalid")
    if value.get("schema") != "swebench_triad_grader_abandoned_v1":
        raise GraderContractError("grader abandoned receipt schema drifted")
    binding_sha256 = str(ledger_base["binding_sha256"])
    if value.get("binding_sha256") != binding_sha256:
        raise GraderContractError("grader abandoned binding drifted")
    reason = value.get("reason")
    pre_receipt_reasons = {
        "pre_receipt_group_without_leader",
        "pre_receipt_output_without_process",
        "launching_without_process",
    }
    if reason not in pre_receipt_reasons | {"dead_without_process_result"}:
        raise GraderContractError("grader abandoned reason drifted")
    event = {
        **dict(ledger_base),
        "event_id": binding_sha256 + ":abandoned",
        "event": "abandoned",
        "reason": reason,
    }
    started_path = attempt_root / "started.json"
    if reason == "dead_without_process_result":
        if not started_path.exists():
            raise GraderContractError("dead grader abandonment lacks start receipt")
        started = read_started_receipt(attempt_root, command)
        event.update(pid=started["pid"], start_ticks=started["start_ticks"])
        failure_class = "grader_attempt_incomplete"
        message = "grader attempt has no process completion receipt"
    else:
        if started_path.exists():
            raise GraderContractError("pre-receipt abandonment has a start receipt")
        failure_class = "grader_pre_receipt_attempt_abandoned"
        message = "grader pre-receipt attempt was durably abandoned; use a new attempt"
    append_start_ledger(config, attempt_root, command, ledger_base)
    append_command_ledger(config, event)
    raise_retryable(attempt_root, failure_class, message)


def append_start_ledger(
    config: OfficialGraderConfig,
    attempt_root: Path,
    command: list[str],
    ledger_base: Mapping[str, Any],
) -> None:
    binding_sha256 = str(ledger_base["binding_sha256"])
    append_command_ledger(
        config,
        {
            **dict(ledger_base),
            "event_id": binding_sha256 + ":start",
            "event": "start",
            "command": command,
            "cwd": str(attempt_root),
            "environment": safe_environment_receipt(config),
        },
    )


def persist_process_timeout(
    config: OfficialGraderConfig,
    attempt_root: Path,
    process_result_path: Path,
    ledger_base: Mapping[str, Any],
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> None:
    stdout = bytes_or_empty(stdout)
    stderr = bytes_or_empty(stderr)
    write_immutable_bytes(attempt_root / "stdout.log", stdout)
    write_immutable_bytes(attempt_root / "stderr.log", stderr)
    process_result = {
        "schema": "swebench_triad_grader_process_v1",
        "status": "process_timeout",
        "returncode": None,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }
    write_immutable_json(process_result_path, process_result)
    binding_sha256 = str(ledger_base["binding_sha256"])
    append_command_ledger(
        config,
        {
            **dict(ledger_base),
            "event_id": binding_sha256 + ":exit",
            "event": "exit",
            "process_result": process_result,
        },
    )
    raise_retryable(
        attempt_root,
        "grader_process_timeout",
        "official grader process exceeded its outer timeout",
    )


def run_official_grader(
    config: OfficialGraderConfig,
    request: OfficialGradeRequest,
) -> dict[str, Any]:
    request.validate()
    attempt_root = ensure_private_directory(grade_attempt_directory(config, request))
    binding = request_binding(request)
    request_payload = {
        **binding,
        "accepted_cell": dict(request.accepted_cell),
        "queued_handoff": dict(request.queued_handoff),
    }
    write_immutable_json(attempt_root / "request.json", request_payload)
    prediction_path = write_immutable_bytes(
        attempt_root / "prediction.jsonl",
        prediction_jsonl_bytes(request.prediction),
    )

    outcome_path = attempt_root / "official-outcome.json"
    if outcome_path.exists():
        outcome = validate_outcome(read_json(outcome_path), request)
        write_terminal_receipt(attempt_root, outcome)
        return outcome

    started_path = attempt_root / "started.json"
    process_result_path = attempt_root / "process-result.json"
    command = grader_command(
        config,
        request,
        prediction_path=prediction_path,
    )
    binding_sha256 = sha256_json(binding)
    ledger_base = {
        "schema": "swebench_triad_command_exit_event_v1",
        "binding_sha256": binding_sha256,
        "task_index": request.task_index,
        "arm": request.arm,
        "generation": request.generation,
        "grader_attempt": request.grader_attempt,
        "prediction_sha256": binding["prediction_sha256"],
    }
    abandoned_path = attempt_root / "abandoned.json"
    if abandoned_path.exists() and process_result_path.exists():
        raise GraderContractError("grader attempt has multiple terminal receipts")
    if abandoned_path.exists():
        recover_abandoned_attempt(
            config,
            request,
            attempt_root,
            ledger_base,
            command,
        )
    if process_result_path.exists() and not started_path.exists():
        append_start_ledger(config, attempt_root, command, ledger_base)
        result = read_json(process_result_path)
        append_command_ledger(
            config,
            {
                **ledger_base,
                "event_id": binding_sha256 + ":exit",
                "event": "exit",
                "process_result": result,
            },
        )
        return finish_grade_attempt(attempt_root, request, command=command)
    launching = read_launching_receipt(attempt_root, command, binding_sha256)
    if started_path.exists():
        started = read_started_receipt(attempt_root, command)
        owner_binding = started.get("owner_binding_sha256")
        if owner_binding is not None and owner_binding != binding_sha256:
            raise GraderContractError("grader start owner binding drifted")
        pgid = int(started.get("pgid", started["pid"]))
        if process_result_path.exists():
            terminate_owned_grader_group(pgid, binding_sha256)
            result = read_json(process_result_path)
            append_command_ledger(
                config,
                {
                    **ledger_base,
                    "event_id": binding_sha256 + ":exit",
                    "event": "exit",
                    "process_result": result,
                },
            )
            return finish_grade_attempt(
                attempt_root, request, command=command
            )
        pid, ticks, alive = started_process(attempt_root, command)
        if owner_binding is not None:
            group_members = owned_grader_group_members(pgid, binding_sha256)
            if alive and pid not in group_members:
                raise GraderContractError(
                    "live grader leader is absent from its owned process group"
                )
        if alive:
            started_at_ns = started["started_at_ns"]
            deadline_ns = started_at_ns + int(
                (config.timeout_seconds + PROCESS_TIMEOUT_GRACE_SECONDS)
                * 1_000_000_000
            )
            if time.time_ns() >= deadline_ns:
                terminate_owned_grader_group(pgid, binding_sha256)
                persist_process_timeout(
                    config,
                    attempt_root,
                    process_result_path,
                    ledger_base,
                )
            raise GraderBusyError(
                f"official grader is still live: pid={pid} start_ticks={ticks}"
            )
        terminate_owned_grader_group(pgid, binding_sha256)
        write_immutable_json(
            attempt_root / "abandoned.json",
            {
                "schema": "swebench_triad_grader_abandoned_v1",
                "binding_sha256": binding_sha256,
                "reason": "dead_without_process_result",
            },
        )
        append_command_ledger(
            config,
            {
                **ledger_base,
                "event_id": binding_sha256 + ":abandoned",
                "event": "abandoned",
                "pid": pid,
                "start_ticks": ticks,
                "reason": "dead_without_process_result",
            },
        )
        raise_retryable(
            attempt_root,
            "grader_attempt_incomplete",
            "grader attempt has no process completion receipt",
        )

    aggregate_path, _, _, _ = expected_raw_paths(attempt_root, request)
    run_root = attempt_root / "logs" / "run_evaluation" / grader_run_id(request)
    matching = find_matching_grader_process(command)
    if matching is not None:
        try:
            matching_pgid = os.getpgid(matching[0])
        except ProcessLookupError:
            matching = None
        else:
            if matching_pgid != matching[0]:
                raise GraderContractError(
                    "pre-receipt grader escaped its exact process group"
                )
    if matching is not None:
        if process_environment_value(matching[0], GRADER_OWNER_ENV) != binding_sha256:
            raise GraderContractError("pre-receipt grader owner binding drifted")
        if matching[0] not in owned_grader_group_members(
            matching[0], binding_sha256
        ):
            raise GraderContractError(
                "pre-receipt grader leader is absent from its owned process group"
            )
        if launching is None:
            raise GraderContractError("live pre-receipt grader lacks launch time")
        append_start_ledger(config, attempt_root, command, ledger_base)
        write_immutable_json(
            started_path,
            {
                "schema": "swebench_triad_grader_started_v4",
                "started_at_ns": launching["started_at_ns"],
                "pid": matching[0],
                "pgid": matching[0],
                "start_ticks": matching[1],
                "command_sha256": sha256_json(command),
                "owner_binding_sha256": binding_sha256,
            },
        )
        deadline_ns = launching["started_at_ns"] + int(
            (config.timeout_seconds + PROCESS_TIMEOUT_GRACE_SECONDS)
            * 1_000_000_000
        )
        if time.time_ns() >= deadline_ns:
            terminate_owned_grader_group(matching[0], binding_sha256)
            persist_process_timeout(
                config,
                attempt_root,
                process_result_path,
                ledger_base,
            )
        raise GraderBusyError(
            "official grader is live in the pre-receipt crash window: "
            f"pid={matching[0]} start_ticks={matching[1]}"
        )

    owned_groups = owned_grader_process_groups(binding_sha256)
    if len(owned_groups) > 1:
        raise GraderContractError("pre-receipt grader spans multiple process groups")
    if owned_groups:
        pgid = next(iter(owned_groups))
        terminate_owned_grader_group(pgid, binding_sha256)
        abandon_pre_receipt_attempt(
            config,
            request,
            attempt_root,
            ledger_base,
            command,
            reason="pre_receipt_group_without_leader",
        )
    if launching is not None:
        abandon_pre_receipt_attempt(
            config,
            request,
            attempt_root,
            ledger_base,
            command,
            reason=(
                "pre_receipt_output_without_process"
                if aggregate_path.exists() or run_root.exists()
                else "launching_without_process"
            ),
        )
    if aggregate_path.exists() or run_root.exists():
        abandon_pre_receipt_attempt(
            config,
            request,
            attempt_root,
            ledger_base,
            command,
            reason="pre_receipt_output_without_process",
        )

    environment_receipt = verify_grader_environment(config)
    verify_instance_in_dataset(config.dataset_path, request.prediction["instance_id"])
    write_immutable_json(
        attempt_root / "invocation.json",
        {
            "schema": "swebench_triad_official_invocation_v1",
            "binding_sha256": binding_sha256,
            "command": command,
            "cwd": str(attempt_root),
            "environment": safe_environment_receipt(config),
            "verified_environment": environment_receipt,
        },
    )

    started_at_ns = time.time_ns()
    append_start_ledger(config, attempt_root, command, ledger_base)
    environment = grader_environment(
        config,
        owner_binding_sha256=binding_sha256,
    )
    process: subprocess.Popen[Any] | None = None
    ticks: int | None = None
    try:
        write_immutable_json(
            attempt_root / "launching.json",
            {
                "schema": "swebench_triad_grader_launching_v1",
                "binding_sha256": binding_sha256,
                "started_at_ns": started_at_ns,
                "command_sha256": sha256_json(command),
            },
        )
        process = subprocess.Popen(
            command,
            cwd=attempt_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        ticks = process_start_ticks(process.pid)
        write_immutable_json(
            started_path,
            {
                "schema": "swebench_triad_grader_started_v4",
                "started_at_ns": started_at_ns,
                "pid": process.pid,
                "pgid": process.pid,
                "start_ticks": ticks,
                "command_sha256": sha256_json(command),
                "owner_binding_sha256": binding_sha256,
            },
        )
        stdout, stderr = process.communicate(
            timeout=config.timeout_seconds + PROCESS_TIMEOUT_GRACE_SECONDS
        )
    except subprocess.TimeoutExpired as error:
        if process is None or ticks is None:
            raise GraderContractError("grader timeout preceded process identity") from error
        terminate_grader_process_group(
            process,
            expected_start_ticks=ticks,
            binding_sha256=binding_sha256,
        )
        final_stdout = getattr(error, "stdout", None)
        final_stderr = getattr(error, "stderr", None)
        stdout = bytes_or_empty(final_stdout or error.stdout)
        stderr = bytes_or_empty(final_stderr or error.stderr)
        persist_process_timeout(
            config,
            attempt_root,
            process_result_path,
            ledger_base,
            stdout=stdout,
            stderr=stderr,
        )
    except OSError as error:
        post_spawn = process is not None
        if process is not None:
            if ticks is None:
                terminate_unreceipted_grader_process_group(
                    process, binding_sha256
                )
            else:
                terminate_grader_process_group(
                    process,
                    expected_start_ticks=ticks,
                    binding_sha256=binding_sha256,
                )
        process_result = {
            "schema": "swebench_triad_grader_process_v1",
            "status": "post_spawn_error" if post_spawn else "spawn_error",
            "returncode": None,
            "error_class": type(error).__name__,
        }
        write_immutable_json(process_result_path, process_result)
        append_command_ledger(
            config,
            {
                **ledger_base,
                "event_id": binding_sha256 + ":exit",
                "event": "exit",
                "process_result": process_result,
            },
        )
        raise_retryable(
            attempt_root,
            "grader_post_spawn_failure" if post_spawn else "grader_spawn_failure",
            "official grader process could not start or persist its identity",
        )
    except BaseException:
        if process is not None:
            if ticks is None:
                terminate_unreceipted_grader_process_group(
                    process, binding_sha256
                )
            else:
                terminate_grader_process_group(
                    process,
                    expected_start_ticks=ticks,
                    binding_sha256=binding_sha256,
                )
        raise

    terminate_owned_grader_group(process.pid, binding_sha256)
    stdout = bytes_or_empty(stdout)
    stderr = bytes_or_empty(stderr)
    write_immutable_bytes(attempt_root / "stdout.log", stdout)
    write_immutable_bytes(attempt_root / "stderr.log", stderr)
    process_result = {
        "schema": "swebench_triad_grader_process_v1",
        "status": "completed",
        "returncode": process.returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }
    write_immutable_json(process_result_path, process_result)
    append_command_ledger(
        config,
        {
            **ledger_base,
            "event_id": binding_sha256 + ":exit",
            "event": "exit",
            "process_result": process_result,
        },
    )
    return finish_grade_attempt(attempt_root, request, command=command)


__all__ = [
    "DOCKER_SOCKET",
    "GraderConfigurationError",
    "GraderContractError",
    "OfficialGradeRequest",
    "OfficialGraderConfig",
    "RetryableGraderError",
    "grade_attempt_directory",
    "expected_raw_paths",
    "grader_run_id",
    "run_official_grader",
    "terminate_grader_process_group",
    "verify_grader_environment",
    "verify_pinned_import",
]
