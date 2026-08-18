"""Production-only binding for the formal SWE-bench triad lifecycle.

The reusable paired runner remains benchmark neutral.  This module parses the
sealed deployment description and supplies the host-specific operations used
by :mod:`swebench_triad_eval.cli`.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from paired_eval.contracts import RunConfig
from paired_eval.controller import AgentGymPolicyTurnController
from paired_eval.evidence import PrivateEvidenceStore
from paired_eval.manifest import expand_manifest
from paired_eval.runner import PairedRunner
from paired_eval.serialization import canonical_json_bytes

from . import ARMS
from .atomic import atomic_write_json, ensure_private_directory, read_json
from .identity import (
    HARNESS_COMMIT,
    HARNESS_TREE,
    INNER_COMMIT,
    OUTER_COMMIT,
    PRODUCTION_DATASET_PINS,
    sha256_file,
    validate_frozen_common,
    verify_dataset,
    verify_image_index,
    verify_model_files,
)
from .oci import (
    CachedOciStore,
    DockerCli,
    OciImageBinding,
    attest_rootfs,
    build_docker_archive,
    ensure_repository_mirror,
    materialize_rootfs,
    require_task_eviction_ready,
)
from .official_grader import (
    OfficialGradeRequest,
    OfficialGraderConfig,
    RetryableGraderError,
    expected_raw_paths,
    find_matching_grader_process,
    grade_attempt_directory,
    grader_command,
    grader_run_id,
    run_official_grader,
)
from .resource_guard import (
    CGROUP_RELATIVE_PREFIX,
    CgroupV1CellEnvelope,
    CgroupV1Limits,
    MountNamespaceCgroupV1Backend,
    cgroup_structure_lock,
)
from .runtime_factory import (
    SwebenchRuntimeEndpoint,
    make_swebench_runtime_factory,
)
from .shared_pool_contract import (
    SHARED_MODEL_POOL_ASSIGNMENT,
    SHARED_MODEL_POOL_CLEANUP,
    SHARED_MODEL_POOL_LISTENER_ADDRESS,
    SHARED_MODEL_POOL_LISTENER_FAMILY,
    SHARED_MODEL_POOL_LISTENER_SOURCE,
    validate_shared_model_pool_snapshot,
)
from .state import CellKey, OwnerIdentity, sha256_json
from .state import RuntimeLaneToken


RUN_CONFIG_SCHEMA = "amg_swebench_triad_run_config_v1"
SHARED_POOL_RUN_CONFIG_SCHEMA = "amg_swebench_triad_run_config_shared_pool_v3"
OWNER_LABEL = "amg-swebench-triad-eval-0816"
CONTAINER_NAME_PREFIX = "amg-sbv-triad-"
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"\A[0-9a-f]{40}\Z")
HOLDER_RETENTION_FLOOR_PERCENT = 5.0
HOLDER_TARGET_PERCENT = 10.0

TOP_LEVEL_FIELDS = {
    "schema",
    "run_root",
    "manifest_path",
    "manifest_sha256",
    "evidence_root",
    "source",
    "assets",
    "pod",
    "docker",
    "task4_receipt",
    "serving",
    "runtime",
    "grader",
}
SOURCE_FIELDS = {
    "root",
    "integration_commit",
    "deployment_commit",
    "inner_commit",
}
ASSET_FIELDS = {
    "dataset_manifest",
    "dataset_jsonl",
    "image_index",
    "image_digests",
    "image_manifests",
    "blob_root",
    "blob_certificate",
    "blob_certificate_sha256",
    "blob_revalidation_receipt",
    "blob_revalidation_sha256",
    "exact_identity_receipt",
    "harness_root",
    "model_root",
    "rg_binary",
    "rg_sha256",
}
POD_FIELDS = {"job", "pod", "hostname", "boot_id", "gpu_uuid"}
DOCKER_FIELDS = {
    "socket",
    "executable",
    "pid_file",
    "readiness_receipt",
    "readiness_receipt_sha256",
    "daemon_id",
    "pid",
    "start_ticks",
}
RECEIPT_FIELDS = {"path", "sha256"}
SERVING_FIELDS = {
    "base_url",
    "model_id",
    "pid_file",
    "pid",
    "start_ticks",
    "receipt_path",
    "receipt_sha256",
}
RUNTIME_FIELDS = {
    "pod_local_root",
    "mirrors_root",
    "server_port",
    "container_python",
    "model_timeout_seconds",
    "environment_timeout_seconds",
    "memory_bytes",
    "max_processes",
    "workspace_bytes",
    "workspace_inodes",
    "external_memory_bytes",
    "external_memory_inodes",
}
GRADER_FIELDS = {"python_executable", "output_root", "max_attempts"}
SHARED_RUNTIME_FIELDS = (RUNTIME_FIELDS - {"server_port"}) | {
    "task_slots_per_replica",
    "server_ports",
}
SHARED_GRADER_FIELDS = GRADER_FIELDS | {
    "global_max_concurrency",
    "semaphore_root",
}
SHARED_MODEL_POOL_FIELDS = {
    "owner",
    "readiness_path",
    "readiness_sha256",
    "marker_lease_path",
    "marker_lease_sha256",
    "replica_index",
    "replica_count",
    "gpu_index",
    "gpu_uuid",
    "model_id",
    "model_revision",
    "model_port",
    "proxy_port",
    "assignment_algorithm",
    "cleanup_policy",
}
@dataclass(frozen=True)
class ProductionTaskStage:
    task_index: int
    instance_id: str
    repo: str
    base_commit: str
    binding: OciImageBinding
    rootfs_cache: Path
    archive_path: Path
    mirror_path: Path
    task_root: Path
    slot_index: int
    server_port: int
    lane_generation: int
    lane_fencing_token: str

    def __post_init__(self) -> None:
        if type(self.task_index) is not int or not 0 <= self.task_index < 500:
            raise ValueError("stage task index is invalid")
        text_value(self.instance_id, "stage instance ID")
        text_value(self.repo, "stage repository")
        commit_value(self.base_commit, "stage base commit")
        if not isinstance(self.binding, OciImageBinding):
            raise TypeError("stage OCI binding has the wrong type")
        for name in ("rootfs_cache", "archive_path", "mirror_path", "task_root"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"stage {name} must be an absolute path")
        if type(self.slot_index) is not int or self.slot_index < 0:
            raise ValueError("stage slot index is invalid")
        if type(self.server_port) is not int or not 1 <= self.server_port <= 65535:
            raise ValueError("stage server port is invalid")
        if type(self.lane_generation) is not int or self.lane_generation <= 0:
            raise ValueError("stage lane generation is invalid")
        if (
            not isinstance(self.lane_fencing_token, str)
            or SHA256_RE.fullmatch(self.lane_fencing_token) is None
        ):
            raise ValueError("stage lane fencing token is invalid")


def exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields drifted")


def object_value(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def text_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be normalized nonempty text")
    return value


def sha256_value(value: Any, label: str) -> str:
    result = text_value(value, label)
    if SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return result


def commit_value(value: Any, label: str) -> str:
    result = text_value(value, label)
    if COMMIT_RE.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase commit")
    return result


def path_value(value: Any, label: str) -> Path:
    path = Path(text_value(value, label))
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path


def integer_value(value: Any, label: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds its maximum")
    return value


def nonnegative_integer_value(
    value: Any, label: str, *, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds its maximum")
    return value


def positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def timed_phase(
    rows: list[dict[str, Any]],
    phase: str,
    operation: Callable[[], Any],
) -> Any:
    started_wall_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    status = "PASS"
    try:
        return operation()
    except BaseException:
        status = "FAIL"
        raise
    finally:
        ended_wall_ns = time.time_ns()
        ended_monotonic_ns = time.monotonic_ns()
        rows.append(
            {
                "phase": phase,
                "status": status,
                "started_wall_ns": started_wall_ns,
                "ended_wall_ns": ended_wall_ns,
                "started_monotonic_ns": started_monotonic_ns,
                "ended_monotonic_ns": ended_monotonic_ns,
                "duration_ns": max(
                    0, ended_monotonic_ns - started_monotonic_ns
                ),
            }
        )


def read_canonical_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonically serialized")
    return value


def read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def command_output(
    arguments: Sequence[str],
    *,
    label: str,
    timeout: int = 60,
) -> str:
    try:
        completed = subprocess.run(
            list(arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"{label} could not complete") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(f"{label} failed: {detail}")
    return completed.stdout.strip()


def git_output(root: Path, *arguments: str) -> str:
    return command_output(
        ["git", "-C", str(root), *arguments],
        label="git " + " ".join(arguments),
    )


def http_json(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> Mapping[str, Any]:
    data = None if payload is None else canonical_json_bytes(payload)
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib_request.Request(url, data=data, headers=headers)
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
    except (OSError, urllib_error.URLError) as error:
        raise RuntimeError(f"HTTP probe failed: {url}") from error
    if len(raw) > 16 * 1024 * 1024:
        raise RuntimeError("HTTP probe response exceeded its bound")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"HTTP probe returned invalid JSON: {url}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"HTTP probe returned a non-object: {url}")
    return value


def process_command_argv(pid: int, label: str) -> list[str]:
    try:
        payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as error:
        raise RuntimeError(f"{label} command is unavailable") from error
    fields = payload.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if not fields or any(not field for field in fields):
        raise RuntimeError(f"{label} command is empty or malformed")
    try:
        return [field.decode("utf-8") for field in fields]
    except UnicodeError as error:
        raise RuntimeError(f"{label} command is not UTF-8") from error


def command_argv_sha256(arguments: Sequence[str]) -> str:
    if not arguments or any(not isinstance(value, str) or not value for value in arguments):
        raise ValueError("process command arguments are invalid")
    return hashlib.sha256(b"\0".join(value.encode("utf-8") for value in arguments)).hexdigest()


def require_process_identity(pid: int, start_ticks: int, label: str) -> str:
    if linux_process_start_ticks(pid) != start_ticks:
        raise RuntimeError(f"{label} PID start ticks drifted")
    return " ".join(process_command_argv(pid, label))


def require_recorded_process_identity(
    value: Mapping[str, Any], label: str
) -> tuple[int, int, list[str]]:
    pid = integer_value(value.get("pid"), f"{label} PID")
    start_ticks = integer_value(value.get("start_ticks"), f"{label} start ticks")
    if linux_process_start_ticks(pid) != start_ticks:
        raise RuntimeError(f"{label} PID start ticks drifted")
    live = process_command_argv(pid, label)
    recorded = value.get("command")
    if (
        not isinstance(recorded, list)
        or any(not isinstance(item, str) or not item for item in recorded)
        or value.get("command_sha256") != command_argv_sha256(recorded)
    ):
        raise RuntimeError(f"{label} recorded command identity drifted")
    target = supervised_target_command(recorded, label)
    if live not in (recorded, target):
        raise RuntimeError(f"{label} exact command identity drifted")
    return pid, start_ticks, live


def command_option(arguments: Sequence[str], name: str, label: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise RuntimeError(f"{label} option {name} drifted")
    return arguments[positions[0] + 1]


def supervised_target_command(
    arguments: Sequence[str], label: str
) -> list[str]:
    if "--command-json" not in arguments:
        return list(arguments)
    try:
        nested = json.loads(command_option(arguments, "--command-json", label))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} nested command is invalid JSON") from error
    if (
        not isinstance(nested, list)
        or any(not isinstance(item, str) or not item for item in nested)
    ):
        raise RuntimeError(f"{label} nested command is invalid")
    return nested


def exact_token_proxy_target(arguments: Sequence[str]) -> list[str]:
    target = supervised_target_command(arguments, "proxy")
    scripts = [
        value for value in target if Path(value).name == "gaia_vllm_token_proxy.py"
    ]
    if len(scripts) != 1:
        raise RuntimeError("exact-token proxy script identity drifted")
    return target


def validate_exact_token_proxy_config(
    target: Sequence[str],
    *,
    model_port: int,
    proxy_port: int,
    model_id: str,
    model_revision: str,
    proxy_source_sha256: str,
) -> Mapping[str, Any]:
    config_path = Path(command_option(target, "--config", "exact-token proxy"))
    if not config_path.is_absolute():
        raise RuntimeError("exact-token proxy config path is not absolute")
    config = read_json_object(config_path, "exact-token proxy config")
    expected_upstream = f"http://127.0.0.1:{model_port}"
    if (
        config.get("schema") != "gaia_vllm_exact_token_proxy_config_v1"
        or config.get("listen_host") != "127.0.0.1"
        or config.get("listen_port") != proxy_port
        or config.get("upstream_base_url") != expected_upstream
        or config.get("upstream_model_id") != model_id
        or config.get("upstream_model_revision") != model_revision
        or config.get("proxy_source_sha256") != proxy_source_sha256
    ):
        raise RuntimeError("exact-token proxy route config drifted")
    source_paths = [
        Path(value)
        for value in target
        if Path(value).name == "gaia_vllm_token_proxy.py"
    ]
    if len(source_paths) != 1 or sha256_file(source_paths[0]) != proxy_source_sha256:
        raise RuntimeError("exact-token proxy source identity drifted")
    return {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "proxy_source_sha256": proxy_source_sha256,
        "runtime_sha256": sha256_value(
            config.get("runtime_sha256"), "proxy runtime SHA-256"
        ),
        "tokenizer_sha256": sha256_value(
            config.get("tokenizer_sha256"), "proxy tokenizer SHA-256"
        ),
        "upstream_base_url": expected_upstream,
        "upstream_base_url_sha256": hashlib.sha256(
            expected_upstream.encode("utf-8")
        ).hexdigest(),
    }


def shared_model_pool_snapshot_receipt(
    shared: Mapping[str, Any],
    *,
    readiness_sha256: str,
    marker_lease_sha256: str,
    selected: Mapping[str, Any],
    selected_live: Mapping[str, Any],
    assigned_gpu_process_pids: Sequence[int],
    live_replica_count: int,
) -> dict[str, Any]:
    """Build the one exact pool snapshot embedded at every durable boundary."""

    server = object_value(selected.get("server"), "selected model server")
    proxy = object_value(selected.get("proxy"), "selected token proxy")
    snapshot = {
        "status": "PASS",
        "owner": shared["owner"],
        "readiness_sha256": readiness_sha256,
        "marker_lease_sha256": marker_lease_sha256,
        "replica_index": shared["replica_index"],
        "replica_count": shared["replica_count"],
        "gpu_index": shared["gpu_index"],
        "gpu_uuid": shared["gpu_uuid"],
        "model_id": shared["model_id"],
        "model_revision": shared["model_revision"],
        "model_port": shared["model_port"],
        "proxy_port": shared["proxy_port"],
        "server_pid": server["pid"],
        "server_start_ticks": server["start_ticks"],
        "proxy_pid": proxy["pid"],
        "proxy_start_ticks": proxy["start_ticks"],
        "server_target_pids": selected_live["server_target_pids"],
        "server_listener_pids": selected_live["server_listener_pids"],
        "server_listener_census": selected_live["server_listener_census"],
        "proxy_target_pids": selected_live["proxy_target_pids"],
        "proxy_listener_pids": selected_live["proxy_listener_pids"],
        "proxy_listener_census": selected_live["proxy_listener_census"],
        "proxy_route": selected_live["proxy_route"],
        "assigned_gpu_process_pids": list(assigned_gpu_process_pids),
        "all_replicas_alive": live_replica_count == shared["replica_count"],
        "all_endpoints_healthy": True,
        "assignment_algorithm": SHARED_MODEL_POOL_ASSIGNMENT,
        "cleanup_policy": SHARED_MODEL_POOL_CLEANUP,
    }
    return dict(validate_shared_model_pool_snapshot(snapshot))


def mount_filesystem_type(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        rows = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError("cannot read Linux mountinfo") from error
    matches: list[tuple[int, str]] = []
    for row in rows:
        before, separator, after = row.partition(" - ")
        if not separator:
            raise RuntimeError("mountinfo row is malformed")
        fields = before.split()
        suffix = after.split()
        if len(fields) < 5 or len(suffix) < 1:
            raise RuntimeError("mountinfo row is incomplete")
        mountpoint = fields[4].replace("\\040", " ").replace("\\011", "\t")
        candidate = Path(mountpoint)
        try:
            resolved.relative_to(candidate)
        except ValueError:
            continue
        matches.append((len(candidate.parts), suffix[0]))
    if not matches:
        raise RuntimeError("active rootfs filesystem is unidentifiable")
    return max(matches)[1]


@dataclass(frozen=True)
class ProductionRunConfig:
    """Strict, immutable description of the one authorized deployment."""

    path: Path
    payload: Mapping[str, Any]
    manifest: Mapping[str, Any]
    configs: tuple[RunConfig, ...]

    @classmethod
    def load(cls, path: Path | str) -> "ProductionRunConfig":
        config_path = Path(path)
        if not config_path.is_absolute():
            raise ValueError("run config path must be absolute")
        payload = read_canonical_object(config_path, "run config")
        cls.validate_payload(payload)

        manifest_path = path_value(payload["manifest_path"], "manifest path")
        try:
            manifest_bytes = manifest_path.read_bytes()
        except OSError as error:
            raise RuntimeError("cannot read the frozen manifest") from error
        if hashlib.sha256(manifest_bytes).hexdigest() != payload["manifest_sha256"]:
            raise ValueError("manifest SHA-256 drifted")
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("frozen manifest is invalid JSON") from error
        if not isinstance(manifest, Mapping):
            raise TypeError("frozen manifest must be an object")
        if canonical_json_bytes(manifest) != manifest_bytes:
            raise ValueError("frozen manifest is not canonical")
        common = object_value(manifest.get("common"), "manifest common")
        validate_frozen_common(common)
        configs = tuple(expand_manifest(manifest))
        cls.validate_configs(configs)
        return cls(
            path=config_path,
            payload=dict(payload),
            manifest=dict(manifest),
            configs=configs,
        )

    @staticmethod
    def validate_payload(payload: Mapping[str, Any]) -> None:
        schema = payload.get("schema")
        if schema == RUN_CONFIG_SCHEMA:
            exact_fields(payload, TOP_LEVEL_FIELDS, "run config")
        elif schema == SHARED_POOL_RUN_CONFIG_SCHEMA:
            exact_fields(
                payload, TOP_LEVEL_FIELDS | {"shared_model_pool"}, "run config"
            )
        else:
            raise ValueError("run config schema drifted")
        for name in ("run_root", "manifest_path", "evidence_root"):
            path_value(payload[name], name)
        sha256_value(payload["manifest_sha256"], "manifest SHA-256")

        source = object_value(payload["source"], "source config")
        exact_fields(source, SOURCE_FIELDS, "source config")
        path_value(source["root"], "source root")
        integration = commit_value(
            source["integration_commit"], "integration commit"
        )
        deployment = commit_value(
            source["deployment_commit"], "deployment commit"
        )
        inner = commit_value(source["inner_commit"], "inner commit")
        if integration != OUTER_COMMIT:
            raise ValueError("integration commit drifted")
        if inner != INNER_COMMIT:
            raise ValueError("inner commit drifted")
        if deployment == integration:
            raise ValueError("deployment commit did not include guarded runtime")

        assets = object_value(payload["assets"], "asset config")
        exact_fields(assets, ASSET_FIELDS, "asset config")
        for name in ASSET_FIELDS - {
            "blob_certificate_sha256",
            "blob_revalidation_sha256",
            "rg_sha256",
        }:
            path_value(assets[name], f"asset {name}")
        for name in (
            "blob_certificate_sha256",
            "blob_revalidation_sha256",
            "rg_sha256",
        ):
            sha256_value(assets[name], f"asset {name}")

        pod = object_value(payload["pod"], "pod config")
        exact_fields(pod, POD_FIELDS, "pod config")
        for name in POD_FIELDS:
            text_value(pod[name], f"pod {name}")

        docker = object_value(payload["docker"], "Docker config")
        exact_fields(docker, DOCKER_FIELDS, "Docker config")
        for name in ("socket", "executable", "pid_file", "readiness_receipt"):
            path_value(docker[name], f"Docker {name}")
        sha256_value(
            docker["readiness_receipt_sha256"], "Docker readiness receipt"
        )
        text_value(docker["daemon_id"], "Docker daemon ID")
        integer_value(docker["pid"], "Docker PID")
        integer_value(docker["start_ticks"], "Docker start ticks")

        task4 = object_value(payload["task4_receipt"], "Task-4 receipt")
        exact_fields(task4, RECEIPT_FIELDS, "Task-4 receipt")
        path_value(task4["path"], "Task-4 receipt path")
        sha256_value(task4["sha256"], "Task-4 receipt SHA-256")

        serving = object_value(payload["serving"], "serving config")
        exact_fields(serving, SERVING_FIELDS, "serving config")
        base_url = text_value(serving["base_url"], "serving base URL")
        if not base_url.startswith("http://127.0.0.1:"):
            raise ValueError("serving endpoint must be loopback HTTP")
        text_value(serving["model_id"], "served model ID")
        for name in ("pid_file", "receipt_path"):
            path_value(serving[name], f"serving {name}")
        integer_value(serving["pid"], "serving PID")
        integer_value(serving["start_ticks"], "serving start ticks")
        sha256_value(serving["receipt_sha256"], "serving receipt SHA-256")

        if schema == SHARED_POOL_RUN_CONFIG_SCHEMA:
            shared = object_value(
                payload["shared_model_pool"], "shared model pool config"
            )
            exact_fields(
                shared, SHARED_MODEL_POOL_FIELDS, "shared model pool config"
            )
            text_value(shared["owner"], "shared model pool owner")
            for name in ("readiness_path", "marker_lease_path"):
                path_value(shared[name], f"shared model pool {name}")
            for name in ("readiness_sha256", "marker_lease_sha256"):
                sha256_value(shared[name], f"shared model pool {name}")
            replica_index = nonnegative_integer_value(
                shared["replica_index"],
                "shared model pool replica index",
                maximum=7,
            )
            replica_count = integer_value(
                shared["replica_count"],
                "shared model pool replica count",
                maximum=8,
            )
            gpu_index = nonnegative_integer_value(
                shared["gpu_index"], "shared model pool GPU index", maximum=7
            )
            if replica_count != 8 or replica_index != gpu_index:
                raise ValueError("shared model pool replica lattice drifted")
            text_value(shared["gpu_uuid"], "shared model pool GPU UUID")
            text_value(shared["model_id"], "shared model pool model ID")
            commit_value(
                shared["model_revision"], "shared model pool model revision"
            )
            integer_value(
                shared["model_port"],
                "shared model pool model port",
                maximum=65535,
            )
            integer_value(
                shared["proxy_port"],
                "shared model pool proxy port",
                maximum=65535,
            )
            if shared["assignment_algorithm"] != SHARED_MODEL_POOL_ASSIGNMENT:
                raise ValueError("shared model pool assignment drifted")
            if shared["cleanup_policy"] != SHARED_MODEL_POOL_CLEANUP:
                raise ValueError("shared model pool cleanup policy drifted")
            if pod["gpu_uuid"] != shared["gpu_uuid"]:
                raise ValueError("pod and shared-pool GPU UUIDs differ")
            if serving["model_id"] != shared["model_id"]:
                raise ValueError("serving and shared-pool model IDs differ")
            if base_url != f"http://127.0.0.1:{shared['proxy_port']}/v1":
                raise ValueError(
                    "serving URL is not the assigned exact-token proxy"
                )

        runtime = object_value(payload["runtime"], "runtime config")
        exact_fields(
            runtime,
            SHARED_RUNTIME_FIELDS
            if schema == SHARED_POOL_RUN_CONFIG_SCHEMA
            else RUNTIME_FIELDS,
            "runtime config",
        )
        path_value(runtime["pod_local_root"], "pod-local root")
        path_value(runtime["mirrors_root"], "mirror root")
        if schema == SHARED_POOL_RUN_CONFIG_SCHEMA:
            slots = integer_value(
                runtime["task_slots_per_replica"],
                "task slots per replica",
                maximum=2,
            )
            ports = runtime["server_ports"]
            if (
                slots != 2
                or not isinstance(ports, list)
                or len(ports) != slots
                or len(set(ports)) != slots
            ):
                raise ValueError("shared runtime requires exactly two unique slots")
            for slot_index, port in enumerate(ports):
                integer_value(
                    port,
                    f"server port for slot {slot_index}",
                    maximum=65535,
                )
        else:
            integer_value(runtime["server_port"], "server port", maximum=65535)
        text_value(runtime["container_python"], "container Python")
        positive_number(runtime["model_timeout_seconds"], "model timeout")
        integer_value(runtime["environment_timeout_seconds"], "environment timeout")
        for name in (
            "memory_bytes",
            "max_processes",
            "workspace_bytes",
            "workspace_inodes",
            "external_memory_bytes",
            "external_memory_inodes",
        ):
            integer_value(runtime[name], f"runtime {name}")
        if runtime["memory_bytes"] % 4096:
            raise ValueError("runtime memory limit must be page aligned")

        grader = object_value(payload["grader"], "grader config")
        exact_fields(
            grader,
            SHARED_GRADER_FIELDS
            if schema == SHARED_POOL_RUN_CONFIG_SCHEMA
            else GRADER_FIELDS,
            "grader config",
        )
        path_value(grader["python_executable"], "grader Python")
        path_value(grader["output_root"], "grader output root")
        integer_value(grader["max_attempts"], "grader attempts")
        if schema == SHARED_POOL_RUN_CONFIG_SCHEMA:
            path_value(grader["semaphore_root"], "grader semaphore root")
            if (
                integer_value(
                    grader["global_max_concurrency"],
                    "global grader concurrency",
                    maximum=8,
                )
                != 8
            ):
                raise ValueError("shared-pool grader concurrency must equal eight")

    @staticmethod
    def validate_configs(configs: Sequence[RunConfig]) -> None:
        if len(configs) != 1500:
            raise ValueError("formal manifest must contain exactly 1,500 cells")
        for task_index in range(500):
            triad = configs[task_index * 3 : task_index * 3 + 3]
            if [row.task.task_index for row in triad] != [task_index] * 3:
                raise ValueError("formal manifest task order drifted")
            if tuple(row.capability.arm.value for row in triad) != ARMS:
                raise ValueError("formal manifest arm lattice drifted")
            if len({row.task.task_id for row in triad}) != 1:
                raise ValueError("formal manifest triad identity drifted")
            if len({row.treatment_excluded_config_sha256 for row in triad}) != 1:
                raise ValueError("formal manifest treatment exclusion drifted")

    def section(self, name: str) -> Mapping[str, Any]:
        return object_value(self.payload[name], f"{name} config")

    @property
    def shared_model_pool(self) -> Mapping[str, Any] | None:
        if self.payload["schema"] != SHARED_POOL_RUN_CONFIG_SCHEMA:
            return None
        return object_value(
            self.payload["shared_model_pool"], "shared model pool config"
        )

    @property
    def run_root(self) -> Path:
        return path_value(self.payload["run_root"], "run root")

    @property
    def evidence_root(self) -> Path:
        return path_value(self.payload["evidence_root"], "evidence root")

    @property
    def task_slots_per_replica(self) -> int:
        if self.shared_model_pool is None:
            return 1
        return int(self.section("runtime")["task_slots_per_replica"])

    @property
    def server_ports(self) -> tuple[int, ...]:
        runtime = self.section("runtime")
        if self.shared_model_pool is None:
            return (int(runtime["server_port"]),)
        return tuple(runtime["server_ports"])

    def server_port(self, slot_index: int) -> int:
        if type(slot_index) is not int or not 0 <= slot_index < len(self.server_ports):
            raise ValueError("runtime slot is outside the configured port lattice")
        return self.server_ports[slot_index]

    @property
    def preflight_expectations(self) -> dict[str, Any]:
        source = self.section("source")
        assets = self.section("assets")
        pod = self.section("pod")
        docker = self.section("docker")
        task4 = self.section("task4_receipt")
        serving = self.section("serving")
        runtime = self.section("runtime")
        return {
            "deployment_commit": source["deployment_commit"],
            "inner_commit": source["inner_commit"],
            "job": pod["job"],
            "pod": pod["pod"],
            "hostname": pod["hostname"],
            "boot_id": pod["boot_id"],
            "gpu_uuid": pod["gpu_uuid"],
            "gpu_count": (
                self.shared_model_pool["replica_count"]
                if self.shared_model_pool is not None
                else 1
            ),
            "shared_model_pool": (
                {
                    "owner": self.shared_model_pool["owner"],
                    "readiness_sha256": self.shared_model_pool[
                        "readiness_sha256"
                    ],
                    "marker_lease_sha256": self.shared_model_pool[
                        "marker_lease_sha256"
                    ],
                    "replica_index": self.shared_model_pool["replica_index"],
                    "replica_count": self.shared_model_pool["replica_count"],
                    "gpu_index": self.shared_model_pool["gpu_index"],
                    "gpu_uuid": self.shared_model_pool["gpu_uuid"],
                    "model_revision": self.shared_model_pool["model_revision"],
                    "model_port": self.shared_model_pool["model_port"],
                    "proxy_port": self.shared_model_pool["proxy_port"],
                }
                if self.shared_model_pool is not None
                else None
            ),
            "docker_daemon_id": docker["daemon_id"],
            "docker_pid": docker["pid"],
            "docker_start_ticks": docker["start_ticks"],
            "model_pid": serving["pid"],
            "model_start_ticks": serving["start_ticks"],
            "model_id": serving["model_id"],
            "blob_certificate_sha256": assets["blob_certificate_sha256"],
            "blob_revalidation_sha256": assets["blob_revalidation_sha256"],
            "docker_receipt_sha256": docker["readiness_receipt_sha256"],
            "task4_receipt_sha256": task4["sha256"],
            "rootfs_prefix": str(path_value(runtime["pod_local_root"], "pod root")),
        }


def linux_process_start_ticks(pid: int) -> int:
    """Return Linux ``/proc/<pid>/stat`` start ticks without parsing comm."""

    integer_value(pid, "process PID")
    try:
        line = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError as error:
        raise RuntimeError(f"process is unavailable: {pid}") from error
    closing = line.rfind(")")
    fields = line[closing + 2 :].split()
    if closing <= 0 or len(fields) < 20 or not fields[19].isdigit():
        raise RuntimeError(f"process stat is malformed: {pid}")
    return int(fields[19])


def current_owner_identity() -> OwnerIdentity:
    return OwnerIdentity(
        host_id=socket.gethostname(),
        boot_id=Path("/proc/sys/kernel/random/boot_id")
        .read_text(encoding="ascii")
        .strip(),
        pid=os.getpid(),
        pid_start_ticks=linux_process_start_ticks(os.getpid()),
    )


def owner_is_alive(owner: OwnerIdentity) -> bool:
    if not isinstance(owner, OwnerIdentity):
        raise TypeError("claim owner must be OwnerIdentity")
    try:
        host = socket.gethostname()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except OSError:
        return True
    if owner.host_id != host:
        # A remote process cannot be disproved from this host.  Production
        # recovery uses DriverLeaseRegistry heartbeat expiry for that proof.
        return True
    if owner.boot_id != boot:
        return False
    try:
        return linux_process_start_ticks(owner.pid) == owner.pid_start_ticks
    except RuntimeError:
        return False


def accepted_rows_for_eviction(
    task_index: int,
    instance_id: str,
    accepted_cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Adapt canonical accepted records to the OCI eviction contract."""

    if type(task_index) is not int or task_index < 0:
        raise ValueError("task index is invalid")
    text_value(instance_id, "instance ID")
    result: list[dict[str, str]] = []
    for expected_arm, accepted in zip(ARMS, accepted_cells):
        if not isinstance(accepted, Mapping):
            raise ValueError("accepted cell is not an object")
        cell = accepted.get("cell")
        if (
            accepted.get("schema") != "swebench_triad_accepted_cell_v1"
            or accepted.get("instance_id") != instance_id
            or not isinstance(cell, Mapping)
            or cell.get("task_index") != task_index
            or cell.get("arm") != expected_arm
        ):
            raise ValueError("accepted cell identity drifted")
        result.append(
            {
                "instance_id": instance_id,
                "status": "accepted",
                "arm": expected_arm,
            }
        )
    if len(result) != len(ARMS):
        raise ValueError("accepted cell triad is incomplete")
    return result


def summarize_task4_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_sha256: str,
) -> dict[str, Any]:
    """Reduce the immutable live-probe evidence to preflight booleans."""

    sha256_value(receipt_sha256, "Task-4 receipt SHA-256")
    if not isinstance(receipt, Mapping):
        raise TypeError("Task-4 receipt must be an object")
    if receipt.get("schema") != "amg_swebench_task4_live_negative_probes_v1":
        raise ValueError("Task-4 probe schema drifted")
    if receipt.get("status") != "PASS":
        raise ValueError("Task-4 probe did not pass")
    if receipt.get("network_downloads") != 0:
        raise ValueError("Task-4 probe performed a network download")

    memory = object_value(receipt.get("memory_probe"), "memory probe")
    memory_teardown = object_value(memory.get("teardown"), "memory teardown")
    if not isinstance(memory_teardown.get("memory_failcnt"), int) or (
        memory_teardown["memory_failcnt"] <= 0
    ):
        raise ValueError("memory exhaustion was not blocked")

    pids = object_value(receipt.get("pids_probe"), "pids probe")
    pids_teardown = object_value(pids.get("teardown"), "pids teardown")
    if not isinstance(pids_teardown.get("pids_max_events"), int) or (
        pids_teardown["pids_max_events"] <= 0
    ):
        raise ValueError("fork exhaustion was not blocked")

    quota_values = {}
    for name, label in (
        ("byte_quota_probe", "byte quota"),
        ("inode_quota_probe", "inode quota"),
    ):
        probe = object_value(receipt.get(name), label)
        outcome = object_value(probe.get("outcome"), f"{label} outcome")
        if outcome.get("errno") != errno.ENOSPC:
            raise ValueError(f"{label} was not blocked")
        quota_values[name] = True

    mutation = object_value(
        receipt.get("rootfs_mutation_probe"), "rootfs mutation probe"
    )
    if mutation.get("detected") is not True:
        raise ValueError("rootfs mutation was not detected")
    cgroups = object_value(receipt.get("cgroup_residue"), "cgroup residue")
    mounts = object_value(receipt.get("tmpfs_residue"), "tmpfs residue")
    if cgroups.get("absent") is not True:
        raise ValueError("Task-4 cgroup residue survived")
    if mounts.get("absent") is not True:
        raise ValueError("Task-4 tmpfs residue survived")
    docker = object_value(receipt.get("docker_after"), "Docker residue")
    for name in ("containers", "images", "volumes"):
        value = object_value(docker.get(name), f"Docker {name}")
        if value.get("count") != 0:
            raise ValueError(f"Task-4 Docker {name} residue survived")
    return {
        "receipt_sha256": receipt_sha256,
        "schema": receipt["schema"],
        "status": "PASS",
        "network_downloads": 0,
        "memory_exhaustion_blocked": True,
        "fork_exhaustion_blocked": True,
        "byte_quota_blocked": quota_values["byte_quota_probe"],
        "inode_quota_blocked": quota_values["inode_quota_probe"],
        "rootfs_mutation_detected": True,
        "cgroup_residue_absent": True,
        "tmpfs_residue_absent": True,
        "docker_residue_absent": True,
    }


class ProductionRuntime(Protocol):
    def preflight(self) -> Mapping[str, Any]: ...

    def stage_task(self, task_index: int, *, slot: RuntimeLaneToken) -> Any: ...

    def reconcile_cell(self, config: RunConfig, **kwargs: Any) -> Mapping[str, Any]: ...

    def reconcile_grade(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def reconcile_startup(
        self,
        *,
        task_indices: Sequence[int],
        allow_foreign_loaded_images: bool = False,
        slots: Sequence[RuntimeLaneToken],
    ) -> Mapping[str, Any]: ...

    def reconcile_unbound_loaded_images(self) -> Mapping[str, Any]: ...

    def run_cell(
        self,
        config: RunConfig,
        stage: Any,
        *,
        generation: int,
        slot: RuntimeLaneToken,
    ) -> Mapping[str, Any]: ...

    def grade(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def audit_residue(
        self, task_index: int, *, slot: RuntimeLaneToken
    ) -> Mapping[str, Any]: ...

    def evict_task(
        self, task_index: int, stage: Any, *, slot: RuntimeLaneToken
    ) -> Mapping[str, Any]: ...

    def cleanup(self, *, slots: Sequence[RuntimeLaneToken]) -> Mapping[str, Any]: ...

    def final_audit(self) -> Mapping[str, Any]: ...


class LinuxProductionRuntime:
    """Concrete, fail-closed lifecycle on the single assigned Linux pod."""

    def __init__(
        self,
        config: ProductionRunConfig,
        configs: Sequence[RunConfig],
    ) -> None:
        if not isinstance(config, ProductionRunConfig):
            raise TypeError("Linux runtime requires ProductionRunConfig")
        if tuple(configs) != config.configs:
            raise ValueError("Linux runtime received another manifest")
        self.config = config
        self.configs = tuple(configs)
        self.by_task = {
            task_index: self.configs[task_index * 3 : task_index * 3 + 3]
            for task_index in range(500)
        }
        if config.shared_model_pool is None:
            self.task_slots = {task_index: 0 for task_index in self.by_task}
        else:
            shared = config.shared_model_pool
            assigned = [
                task_index
                for task_index, triad in self.by_task.items()
                if int.from_bytes(
                    hashlib.sha256(
                        triad[0].task.task_id.encode("utf-8")
                    ).digest()[:8],
                    "big",
                )
                % shared["replica_count"]
                == shared["replica_index"]
            ]
            self.task_slots = {
                task_index: position % config.task_slots_per_replica
                for position, task_index in enumerate(assigned)
            }
        self._dataset_rows: tuple[Mapping[str, Any], ...] | None = None
        self._testspec_resolver: Any = None
        self._oci_store: CachedOciStore | None = None
        self._docker_cli: DockerCli | None = None

    def require_slot(
        self,
        slot: RuntimeLaneToken,
        *,
        task_index: int | None,
        allow_global: bool = False,
    ) -> RuntimeLaneToken:
        if not isinstance(slot, RuntimeLaneToken):
            raise TypeError("production operation requires a runtime lane token")
        if slot.task_index != task_index and not (
            allow_global and slot.task_index is None and task_index is not None
        ):
            raise ValueError("runtime lane task binding drifted")
        if slot.server_port != self.config.server_port(slot.slot_index):
            raise ValueError("runtime lane port binding drifted")
        if task_index is not None:
            expected_slot = self.task_slots.get(task_index)
            if expected_slot is None:
                raise ValueError("task is assigned to another shared-model replica")
            if slot.slot_index != expected_slot:
                raise ValueError("task was routed to the wrong deterministic slot")
        return slot

    def section(self, name: str) -> Mapping[str, Any]:
        return self.config.section(name)

    def docker(self) -> DockerCli:
        if self._docker_cli is None:
            docker = self.section("docker")
            self._docker_cli = DockerCli(
                socket_path=path_value(docker["socket"], "Docker socket"),
                executable=str(
                    path_value(docker["executable"], "Docker executable")
                ),
            )
        return self._docker_cli

    def oci_store(self) -> CachedOciStore:
        if self._oci_store is None:
            assets = self.section("assets")
            self._oci_store = CachedOciStore(
                index_path=path_value(assets["image_index"], "image index"),
                manifest_root=path_value(
                    assets["image_manifests"], "image manifests"
                ),
                blob_root=path_value(assets["blob_root"], "blob root"),
            )
        return self._oci_store

    def dataset_rows(self) -> tuple[Mapping[str, Any], ...]:
        if self._dataset_rows is None:
            path = path_value(
                self.section("assets")["dataset_jsonl"], "dataset JSONL"
            )
            try:
                lines = path.read_bytes().splitlines()
                values = tuple(json.loads(line) for line in lines)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RuntimeError("cannot read production dataset rows") from error
            if len(values) != 500 or any(
                not isinstance(value, Mapping) for value in values
            ):
                raise RuntimeError("production dataset row count or shape drifted")
            self._dataset_rows = values
        return self._dataset_rows

    def testspec_resolver(self):
        if self._testspec_resolver is None:
            from agentenv_swebench_verified.testspec import (  # type: ignore
                OfficialTestSpecResolver,
            )

            self._testspec_resolver = OfficialTestSpecResolver(
                source_root=path_value(
                    self.section("assets")["harness_root"], "harness root"
                )
            )
        return self._testspec_resolver

    def preflight(self) -> Mapping[str, Any]:
        runtime = self.section("runtime")
        pod_local_root = ensure_private_directory(
            path_value(runtime["pod_local_root"], "pod-local root")
        )
        filesystem_type = mount_filesystem_type(pod_local_root)
        if filesystem_type in {"nfs", "nfs4"}:
            raise RuntimeError("active per-task rootfs is on shared NFS")
        shared_pool = (
            self.shared_model_pool_snapshot()
            if self.config.shared_model_pool is not None
            else None
        )
        snapshot = {
            "source": self.source_snapshot(),
            "dataset": self.dataset_snapshot(),
            "image_index": self.image_index_snapshot(),
            "model": self.model_snapshot(),
            "blob_cache": self.blob_snapshot(),
            "pod": self.pod_snapshot(),
            "docker": self.docker_snapshot(),
            "task4_negative_probes": self.task4_snapshot(),
            "model_process": self.model_process_snapshot(shared_pool),
            "vllm": self.vllm_snapshot(),
            "swe_metadata": self.swe_metadata_snapshot(),
            "residue": self.global_residue_snapshot(),
            "rootfs": {
                "path": str(pod_local_root),
                "pod_local": True,
            },
        }
        if shared_pool is not None:
            snapshot["shared_model_pool"] = shared_pool
        return snapshot

    def source_snapshot(self) -> dict[str, Any]:
        source = self.section("source")
        root = path_value(source["root"], "source root")
        inner = root / "AgentGym"
        deployment = git_output(root, "rev-parse", "HEAD")
        inner_commit = git_output(inner, "rev-parse", "HEAD")
        protected_paths = (
            "AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/"
            "vllm_rollout.py",
            "AgentGym-RL/scripts/agentmemory/paired_eval",
            "AgentGym",
        )
        protected_diff = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--quiet",
                OUTER_COMMIT,
                deployment,
                "--",
                *protected_paths,
            ],
            check=False,
        )
        if protected_diff.returncode not in {0, 1}:
            raise RuntimeError("protected-source diff audit failed")
        return {
            "deployment_commit": deployment,
            "inner_commit": inner_commit,
            "deployment_clean": not bool(
                git_output(root, "status", "--porcelain=v1", "-uall")
            ),
            "inner_clean": not bool(
                git_output(inner, "status", "--porcelain=v1", "-uall")
            ),
            "protected_diff_zero": protected_diff.returncode == 0,
        }

    def dataset_snapshot(self) -> dict[str, Any]:
        assets = self.section("assets")
        identity = verify_dataset(
            path_value(assets["dataset_jsonl"], "dataset JSONL")
        )
        return {
            "rows": identity["rows"],
            "jsonl_sha256": identity["jsonl_sha256"],
            "id_ledger_sha256": identity["id_ledger_sha256"],
        }

    def image_index_snapshot(self) -> dict[str, Any]:
        assets = self.section("assets")
        identity = verify_image_index(
            path_value(assets["image_index"], "image index")
        )
        return {
            "rows": identity["rows"],
            "index_sha256": identity["index_sha256"],
            "tag_ledger_sha256": identity["tag_ledger_sha256"],
            "digest_tsv_sha256": identity["digest_tsv_sha256"],
        }

    def model_snapshot(self) -> dict[str, Any]:
        identity = verify_model_files(
            path_value(self.section("assets")["model_root"], "model root")
        )
        return {
            "file_count": identity["file_count"],
            "file_ledger_sha256": identity["file_ledger_sha256"],
        }

    def blob_snapshot(self) -> dict[str, Any]:
        assets = self.section("assets")
        certificate_path = path_value(
            assets["blob_certificate"], "blob certificate"
        )
        revalidation_path = path_value(
            assets["blob_revalidation_receipt"], "blob revalidation"
        )
        certificate_sha = sha256_file(certificate_path)
        revalidation_sha = sha256_file(revalidation_path)
        if certificate_sha != assets["blob_certificate_sha256"]:
            raise RuntimeError("blob certificate SHA-256 drifted")
        if revalidation_sha != assets["blob_revalidation_sha256"]:
            raise RuntimeError("blob revalidation SHA-256 drifted")
        certificate = read_json_object(certificate_path, "blob certificate")
        revalidation = read_json_object(revalidation_path, "blob revalidation")
        if certificate.get("descriptor_count") != 1158:
            raise RuntimeError("blob descriptor count drifted")
        blob_root = path_value(assets["blob_root"], "blob root")
        blob_files = []
        for path in blob_root.iterdir():
            info = path.lstat()
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("blob cache contains a non-regular entry")
            blob_files.append((path, info.st_size))
        expected = {
            "certificate_sha256": certificate_sha,
            "revalidation_sha256": revalidation_sha,
            "descriptor_count": 1158,
            "file_count": 1158,
            "total_bytes": 117637519356,
            "downloaded_count": 0,
            "verified_bad_count": 0,
        }
        actual = {
            "certificate_sha256": certificate_sha,
            "revalidation_sha256": revalidation_sha,
            "descriptor_count": certificate.get("descriptor_count"),
            "file_count": len(blob_files),
            "total_bytes": sum(size for _path, size in blob_files),
            "downloaded_count": revalidation.get("downloaded_count"),
            "verified_bad_count": revalidation.get("verified_bad_count"),
        }
        if (
            revalidation.get("descriptor_count") != 1158
            or revalidation.get("total_bytes") != 117637519356
        ):
            raise RuntimeError("blob revalidation denominator drifted")
        if actual != expected:
            raise RuntimeError("blob revalidation contents drifted")
        return actual

    def pod_snapshot(self) -> dict[str, Any]:
        pod = self.section("pod")
        hostname = socket.gethostname()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        gpu_values = [
            value.strip()
            for value in command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid",
                    "--format=csv,noheader,nounits",
                ],
                label="GPU identity probe",
            ).splitlines()
            if value.strip()
        ]
        if hostname != pod["hostname"] or boot_id != pod["boot_id"]:
            raise RuntimeError("pod hostname or boot identity drifted")
        shared = self.config.shared_model_pool
        expected_count = shared["replica_count"] if shared is not None else 1
        if len(gpu_values) != expected_count:
            raise RuntimeError("assigned GPU count drifted")
        if shared is None:
            if gpu_values != [pod["gpu_uuid"]]:
                raise RuntimeError("assigned GPU identity drifted")
            selected_uuid = gpu_values[0]
        else:
            gpu_index = shared["gpu_index"]
            if gpu_values[gpu_index] != pod["gpu_uuid"]:
                raise RuntimeError("shared-pool replica GPU identity drifted")
            if len(set(gpu_values)) != len(gpu_values):
                raise RuntimeError("shared-pool GPU UUIDs are not unique")
            selected_uuid = gpu_values[gpu_index]
        return {
            "job": pod["job"],
            "pod": pod["pod"],
            "hostname": hostname,
            "boot_id": boot_id,
            "gpu_uuid": selected_uuid,
            "gpu_count": len(gpu_values),
        }

    def docker_snapshot(self) -> dict[str, Any]:
        docker_config = self.section("docker")
        receipt_path = path_value(
            docker_config["readiness_receipt"], "Docker readiness receipt"
        )
        receipt_sha = sha256_file(receipt_path)
        if receipt_sha != docker_config["readiness_receipt_sha256"]:
            raise RuntimeError("Docker readiness receipt drifted")
        command = require_process_identity(
            docker_config["pid"],
            docker_config["start_ticks"],
            "Docker daemon",
        )
        if "amg-external-eval-container-runtime-v1" not in command:
            raise RuntimeError("Docker daemon command identity drifted")
        docker = self.docker()
        info_result = docker.run("info", "--format", "{{json .}}")
        if info_result.returncode != 0:
            raise RuntimeError("Docker info probe failed")
        try:
            info = json.loads(docker.output_text(info_result.stdout))
        except json.JSONDecodeError as error:
            raise RuntimeError("Docker info returned invalid JSON") from error
        if not isinstance(info, Mapping):
            raise RuntimeError("Docker info is not an object")
        version_result = docker.run(
            "version", "--format", "{{.Server.APIVersion}}"
        )
        if version_result.returncode != 0:
            raise RuntimeError("Docker API version probe failed")
        volumes_result = docker.run("volume", "ls", "--quiet")
        if volumes_result.returncode != 0:
            raise RuntimeError("Docker volume census failed")
        volumes = [
            row
            for row in docker.output_text(volumes_result.stdout).splitlines()
            if row
        ]
        return {
            "receipt_sha256": receipt_sha,
            "daemon_id": info.get("ID"),
            "pid": docker_config["pid"],
            "start_ticks": docker_config["start_ticks"],
            "version": info.get("ServerVersion"),
            "api_version": docker.output_text(version_result.stdout).strip(),
            "cgroup_version": str(info.get("CgroupVersion")),
            "cgroup_driver": info.get("CgroupDriver"),
            "storage_driver": info.get("Driver"),
            "containers": info.get("Containers"),
            "images": info.get("Images"),
            "volumes": len(volumes),
        }

    def task4_snapshot(self) -> dict[str, Any]:
        task4 = self.section("task4_receipt")
        path = path_value(task4["path"], "Task-4 receipt")
        digest = sha256_file(path)
        if digest != task4["sha256"]:
            raise RuntimeError("Task-4 receipt SHA-256 drifted")
        return summarize_task4_receipt(
            read_json_object(path, "Task-4 receipt"),
            receipt_sha256=digest,
        )

    @staticmethod
    def tcp_listener_census(port: int) -> dict[str, Any]:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("TCP listener port is invalid")
        listeners: list[dict[str, Any]] = []
        tables = (
            (
                Path(SHARED_MODEL_POOL_LISTENER_SOURCE),
                SHARED_MODEL_POOL_LISTENER_FAMILY,
                socket.AF_INET,
            ),
            (Path("/proc/net/tcp6"), "ipv6", socket.AF_INET6),
        )
        for table, family, address_family in tables:
            try:
                rows = table.read_text(encoding="ascii").splitlines()
            except OSError as error:
                raise RuntimeError("TCP listener census is unavailable") from error
            if not rows or rows[0].split()[:2] != ["sl", "local_address"]:
                raise RuntimeError("TCP listener census header is malformed")
            for row in rows[1:]:
                if not row.strip():
                    continue
                fields = row.split()
                if len(fields) < 10:
                    raise RuntimeError("TCP listener census row is malformed")
                local = fields[1]
                try:
                    raw_address, separator, raw_port = local.partition(":")
                    if not separator or ":" in raw_port:
                        raise ValueError("invalid local endpoint")
                    packed = bytes.fromhex(raw_address)
                    if family == SHARED_MODEL_POOL_LISTENER_FAMILY:
                        if len(packed) != 4:
                            raise ValueError("invalid IPv4 address")
                        packed = packed[::-1]
                    else:
                        if len(packed) != 16:
                            raise ValueError("invalid IPv6 address")
                        packed = b"".join(
                            packed[index : index + 4][::-1]
                            for index in range(0, 16, 4)
                        )
                    address = socket.inet_ntop(address_family, packed)
                    local_port = int(raw_port, 16)
                    inode = int(fields[9])
                except (OSError, ValueError) as error:
                    raise RuntimeError("TCP listener census row is invalid") from error
                if fields[3] == "0A" and local_port == port:
                    listeners.append(
                        {
                            "source": str(table),
                            "family": family,
                            "address": address,
                            "port": local_port,
                            "inode": inode,
                        }
                    )
        if not listeners:
            raise RuntimeError("expected TCP listener is absent")
        if len(listeners) != 1:
            raise RuntimeError("expected TCP listener census is ambiguous")
        listener = listeners[0]
        if (
            listener["source"] != SHARED_MODEL_POOL_LISTENER_SOURCE
            or listener["family"] != SHARED_MODEL_POOL_LISTENER_FAMILY
            or listener["address"] != SHARED_MODEL_POOL_LISTENER_ADDRESS
            or listener["port"] != port
            or type(listener["inode"]) is not int
            or listener["inode"] <= 0
        ):
            raise RuntimeError("expected TCP listener is not exact IPv4 loopback")
        return listener

    @staticmethod
    def listener_inode_owners(listener_inodes: set[int]) -> dict[int, set[int]]:
        if (
            not listener_inodes
            or any(type(inode) is not int or inode <= 0 for inode in listener_inodes)
        ):
            raise RuntimeError("listener inode census is invalid")
        owners = {inode: set() for inode in listener_inodes}
        try:
            processes = list(Path("/proc").iterdir())
        except OSError as error:
            raise RuntimeError("listener owner census is unavailable") from error
        for process in processes:
            if not process.name.isdigit():
                continue
            pid = int(process.name)
            try:
                descriptors = list((process / "fd").iterdir())
            except OSError as error:
                if error.errno in (errno.ENOENT, errno.ESRCH):
                    continue
                raise RuntimeError("listener owner census is incomplete") from error
            for descriptor in descriptors:
                try:
                    target = os.readlink(descriptor)
                except OSError as error:
                    if error.errno in (errno.ENOENT, errno.ESRCH):
                        continue
                    raise RuntimeError("listener owner census is incomplete") from error
                match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
                if match is None:
                    continue
                inode = int(match.group(1))
                if inode in owners:
                    owners[inode].add(pid)
        return owners

    def target_process_pids(
        self, root_pid: int, target: Sequence[str], label: str
    ) -> tuple[set[int], set[int]]:
        tree = self.process_descendants(root_pid) | {root_pid}
        matches: set[int] = set()
        for pid in tree:
            try:
                arguments = process_command_argv(pid, f"{label} process")
            except RuntimeError:
                if pid == root_pid:
                    raise
                continue
            if arguments == list(target):
                matches.add(pid)
        if not matches:
            raise RuntimeError(f"{label} target process is absent")
        return matches, tree

    def listener_census(
        self, port: int, candidate_pids: set[int], label: str
    ) -> dict[str, Any]:
        if not candidate_pids or any(
            type(pid) is not int or pid <= 0 for pid in candidate_pids
        ):
            raise RuntimeError(f"{label} process tree is invalid")
        listener = self.tcp_listener_census(port)
        listener_inodes = {listener["inode"]}
        owners_by_inode = self.listener_inode_owners(listener_inodes)
        if (
            not isinstance(owners_by_inode, Mapping)
            or set(owners_by_inode) != listener_inodes
            or any(
                not isinstance(owners, set)
                or not owners
                or any(type(pid) is not int or pid <= 0 for pid in owners)
                for owners in owners_by_inode.values()
            )
        ):
            raise RuntimeError(f"{label} listener owner census is incomplete")
        owners = set().union(*owners_by_inode.values())
        if not owners.issubset(candidate_pids):
            raise RuntimeError(
                f"{label} listener escaped its process tree through a foreign owner"
            )
        return {**listener, "owner_pids": sorted(owners)}

    def listener_pids(
        self, port: int, candidate_pids: set[int], label: str
    ) -> list[int]:
        return self.listener_census(port, candidate_pids, label)["owner_pids"]

    @staticmethod
    def gpu_compute_bindings() -> dict[int, str]:
        output = command_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            label="GPU process binding probe",
        )
        bindings: dict[int, str] = {}
        for line in output.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or not fields[1].isdigit():
                raise RuntimeError("GPU process binding row is malformed")
            pid = int(fields[1])
            previous = bindings.setdefault(pid, fields[0])
            if previous != fields[0]:
                raise RuntimeError("GPU process appeared on multiple GPUs")
        return bindings

    def preflight_shared_model_pool_snapshot(self) -> Mapping[str, Any]:
        snapshot_path = self.config.run_root / "control" / "preflight-snapshot.json"
        receipt_path = self.config.run_root / "control" / "preflight-PASS.json"
        try:
            snapshot = read_json(snapshot_path)
            receipt = read_json(receipt_path)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                "shared model pool preflight reference is unavailable"
            ) from error
        receipt_fields = {
            "schema",
            "status",
            "snapshot_sha256",
            "deployment_commit",
            "inner_commit",
            "boot_id",
            "gpu_uuid",
            "docker_daemon_id",
            "model_id",
        }
        if (
            not isinstance(snapshot, Mapping)
            or not isinstance(receipt, Mapping)
            or set(receipt) != receipt_fields
            or receipt.get("schema") != "swebench_triad_preflight_pass_v1"
            or receipt.get("status") != "PASS"
            or receipt.get("snapshot_sha256") != sha256_json(snapshot)
        ):
            raise RuntimeError("shared model pool preflight reference drifted")
        return validate_shared_model_pool_snapshot(
            snapshot.get("shared_model_pool"),
            "preflight shared model pool snapshot",
        )

    def shared_model_pool_snapshot(
        self, *, require_preflight_binding: bool = False
    ) -> dict[str, Any]:
        if type(require_preflight_binding) is not bool:
            raise TypeError("preflight listener binding policy must be boolean")
        shared = self.config.shared_model_pool
        if shared is None:
            raise RuntimeError("shared model pool was not configured")
        readiness_path = path_value(
            shared["readiness_path"], "shared model pool readiness"
        )
        marker_path = path_value(
            shared["marker_lease_path"], "shared model pool marker lease"
        )
        readiness_sha = sha256_file(readiness_path)
        marker_sha = sha256_file(marker_path)
        if readiness_sha != shared["readiness_sha256"]:
            raise RuntimeError("shared model pool readiness SHA-256 drifted")
        if marker_sha != shared["marker_lease_sha256"]:
            raise RuntimeError("shared model pool marker lease SHA-256 drifted")
        readiness = read_json_object(readiness_path, "shared model pool readiness")
        marker = read_json_object(marker_path, "shared model pool marker lease")
        assignment = object_value(
            readiness.get("assignment"), "shared model pool assignment"
        )
        expected = {
            "schema": "amg_g_qwen35_dp8_pool_readiness_v1",
            "status": "PASS",
            "owner": shared["owner"],
            "boot_id": self.section("pod")["boot_id"],
            "model_id": shared["model_id"],
            "model_revision": shared["model_revision"],
            "replica_count": shared["replica_count"],
        }
        for name, value in expected.items():
            if readiness.get(name) != value:
                raise RuntimeError(f"shared model pool {name} drifted")
        if (
            assignment.get("algorithm") != SHARED_MODEL_POOL_ASSIGNMENT
            or assignment.get("paired_arms_same_replica") is not True
            or assignment.get("arms") != ["00", "10", "11"]
        ):
            raise RuntimeError("shared model pool assignment contract drifted")
        if readiness.get("marker_lease_sha256") != marker_sha:
            raise RuntimeError("shared model pool marker binding drifted")

        parent = object_value(marker.get("parent"), "marker lease parent")
        parent_pid = integer_value(parent.get("pid"), "marker parent PID")
        parent_ticks = integer_value(
            parent.get("start_ticks"), "marker parent start ticks"
        )
        require_process_identity(parent_pid, parent_ticks, "marker lease parent")
        markers = marker.get("markers")
        if not isinstance(markers, list) or len(markers) != 2:
            raise RuntimeError("shared model pool marker lattice drifted")
        expected_marker_paths = {
            "/tmp/crg-holder-yield",
            "/tmp/agentmemory-formal-cpu-active",
        }
        observed_marker_paths = set()
        for row in markers:
            value = object_value(row, "marker lease row")
            current_path = path_value(value.get("path"), "marker lease path")
            observed_marker_paths.add(str(current_path))
            info = current_path.lstat()
            if (
                current_path.is_symlink()
                or not stat.S_ISREG(info.st_mode)
                or info.st_dev != value.get("device")
                or info.st_ino != value.get("inode")
                or sha256_file(current_path) != value.get("sha256")
            ):
                raise RuntimeError("shared model pool marker identity drifted")
        if observed_marker_paths != expected_marker_paths:
            raise RuntimeError("shared model pool marker paths drifted")

        replicas = readiness.get("replicas")
        if not isinstance(replicas, list) or len(replicas) != shared["replica_count"]:
            raise RuntimeError("shared model pool replica records drifted")
        seen_gpu_uuids: set[str] = set()
        seen_ports: set[int] = set()
        selected: Mapping[str, Any] | None = None
        selected_live: Mapping[str, Any] | None = None
        live_processes: list[dict[str, Any]] = []
        proxy_source_sha = sha256_value(
            readiness.get("proxy_source_sha256"), "pool proxy source SHA-256"
        )
        for expected_replica, raw_replica in enumerate(replicas):
            replica = object_value(raw_replica, "shared model pool replica")
            if replica.get("replica") != expected_replica:
                raise RuntimeError("shared model pool replica order drifted")
            gpu_uuid = text_value(replica.get("gpu_uuid"), "replica GPU UUID")
            model_port = integer_value(replica.get("model_port"), "replica model port")
            proxy_port = integer_value(replica.get("proxy_port"), "replica proxy port")
            if (
                gpu_uuid in seen_gpu_uuids
                or model_port in seen_ports
                or proxy_port in seen_ports
                or model_port == proxy_port
            ):
                raise RuntimeError("shared model pool replica identities are not unique")
            seen_gpu_uuids.add(gpu_uuid)
            seen_ports.update((model_port, proxy_port))
            server = object_value(replica.get("server"), "replica model server")
            proxy = object_value(replica.get("proxy"), "replica token proxy")
            server_pid, server_ticks, server_argv = require_recorded_process_identity(
                server, "replica model server"
            )
            proxy_pid, proxy_ticks, proxy_argv = require_recorded_process_identity(
                proxy, "replica token proxy"
            )
            server_target = supervised_target_command(server_argv, "model server")
            if (
                command_option(server_target, "--host", "model server")
                != "127.0.0.1"
                or command_option(server_target, "--port", "model server")
                != str(model_port)
                or command_option(
                    server_target, "--served-model-name", "model server"
                )
                != shared["model_id"]
            ):
                raise RuntimeError("shared model pool server route drifted")
            server_target_pids, server_tree = self.target_process_pids(
                server_pid, server_target, "model server"
            )
            server_listener_census = self.listener_census(
                model_port, server_target_pids, "model server"
            )
            server_listener_pids = server_listener_census["owner_pids"]

            proxy_target = exact_token_proxy_target(proxy_argv)
            proxy_route = validate_exact_token_proxy_config(
                proxy_target,
                model_port=model_port,
                proxy_port=proxy_port,
                model_id=shared["model_id"],
                model_revision=shared["model_revision"],
                proxy_source_sha256=proxy_source_sha,
            )
            proxy_target_pids, proxy_tree = self.target_process_pids(
                proxy_pid, proxy_target, "exact-token proxy"
            )
            proxy_listener_census = self.listener_census(
                proxy_port, proxy_target_pids, "exact-token proxy"
            )
            proxy_listener_pids = proxy_listener_census["owner_pids"]

            registry = http_json(f"http://127.0.0.1:{model_port}/v1/models")
            data = registry.get("data")
            if (
                not isinstance(data, list)
                or len(data) != 1
                or not isinstance(data[0], Mapping)
                or data[0].get("id") != shared["model_id"]
            ):
                raise RuntimeError("shared model pool registry drifted")
            proxy_health = http_json(f"http://127.0.0.1:{proxy_port}/health")
            if (
                proxy_health.get("schema")
                != "gaia_vllm_exact_token_proxy_identity_v1"
                or proxy_health.get("upstream_model_id") != shared["model_id"]
                or proxy_health.get("upstream_model_revision")
                != shared["model_revision"]
                or proxy_health.get("upstream_base_url_sha256")
                != proxy_route["upstream_base_url_sha256"]
                or proxy_health.get("proxy_source_sha256") != proxy_source_sha
                or proxy_health.get("runtime_sha256")
                != proxy_route["runtime_sha256"]
                or proxy_health.get("tokenizer_sha256")
                != proxy_route["tokenizer_sha256"]
                or proxy_health.get("return_token_ids_forced") is not True
                or proxy_health.get("routes")
                != ["/tokenize", "/v1/tokenize", "/v1/chat/completions"]
            ):
                raise RuntimeError("shared model pool proxy health drifted")
            live = {
                "replica": expected_replica,
                "server_pid": server_pid,
                "server_start_ticks": server_ticks,
                "server_target_pids": sorted(server_target_pids),
                "server_listener_pids": server_listener_pids,
                "server_listener_census": server_listener_census,
                "proxy_pid": proxy_pid,
                "proxy_start_ticks": proxy_ticks,
                "proxy_target_pids": sorted(proxy_target_pids),
                "proxy_listener_pids": proxy_listener_pids,
                "proxy_listener_census": proxy_listener_census,
                "proxy_route": dict(proxy_route),
            }
            live_processes.append(live)
            if expected_replica == shared["replica_index"]:
                selected = replica
                selected_live = live
        if selected is None or selected_live is None:
            raise RuntimeError("assigned shared model pool replica is absent")
        for name in ("gpu_index", "gpu_uuid", "model_port", "proxy_port"):
            if selected.get(name) != shared[name]:
                raise RuntimeError(f"assigned shared model pool {name} drifted")
        if (
            selected["server"].get("pid") != self.section("serving")["pid"]
            or selected["server"].get("start_ticks")
            != self.section("serving")["start_ticks"]
        ):
            raise RuntimeError("serving process is not the assigned pool replica")

        gpu_bindings = self.gpu_compute_bindings()
        server_pid = selected["server"]["pid"]
        server_tree = self.process_descendants(server_pid) | {server_pid}
        assigned_gpu_pids = sorted(server_tree & set(gpu_bindings))
        if not assigned_gpu_pids or any(
            gpu_bindings[pid] != shared["gpu_uuid"] for pid in assigned_gpu_pids
        ):
            raise RuntimeError("assigned model process tree is not GPU-bound")
        snapshot = shared_model_pool_snapshot_receipt(
            shared,
            readiness_sha256=readiness_sha,
            marker_lease_sha256=marker_sha,
            selected=selected,
            selected_live=selected_live,
            assigned_gpu_process_pids=assigned_gpu_pids,
            live_replica_count=len(live_processes),
        )
        if require_preflight_binding:
            snapshot = dict(
                validate_shared_model_pool_snapshot(
                    snapshot,
                    "live shared model pool snapshot",
                    listener_reference=self.preflight_shared_model_pool_snapshot(),
                )
            )
        return snapshot

    def model_process_snapshot(
        self, shared_pool: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        serving = self.section("serving")
        receipt_path = path_value(serving["receipt_path"], "serving receipt")
        if sha256_file(receipt_path) != serving["receipt_sha256"]:
            raise RuntimeError("serving receipt SHA-256 drifted")
        command = require_process_identity(
            serving["pid"], serving["start_ticks"], "model server"
        )
        model_root = str(
            path_value(self.section("assets")["model_root"], "model root")
        )
        matches = model_root in command and str(serving["model_id"]) in command
        if not matches:
            raise RuntimeError("model server command identity drifted")
        if shared_pool is not None and (
            shared_pool.get("server_pid") != serving["pid"]
            or shared_pool.get("server_start_ticks") != serving["start_ticks"]
        ):
            raise RuntimeError("shared-pool model process binding drifted")
        return {
            "pid": serving["pid"],
            "start_ticks": serving["start_ticks"],
            "alive": True,
            "command_matches": True,
        }

    def vllm_snapshot(self) -> dict[str, Any]:
        serving = self.section("serving")
        base_url = text_value(serving["base_url"], "serving base URL").rstrip("/")
        root_url = base_url[:-3] if base_url.endswith("/v1") else base_url
        shared = self.config.shared_model_pool
        registry_url = base_url
        if shared is not None:
            # The exact-token proxy deliberately exposes only tokenization and
            # generation.  Its frozen upstream vLLM endpoint remains the
            # authoritative model registry.
            registry_url = f"http://127.0.0.1:{shared['model_port']}/v1"
        models = http_json(registry_url + "/models")
        data = models.get("data")
        if not isinstance(data, list) or len(data) != 1:
            raise RuntimeError("vLLM model registry shape drifted")
        model = object_value(data[0], "vLLM model")
        model_id = model.get("id")
        if model_id != serving["model_id"]:
            raise RuntimeError("vLLM served model identity drifted")
        messages = [{"role": "user", "content": "Reply with exactly: AMG_OK"}]
        tokenize_payload = {
            "model": model_id,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        tokenized = http_json(root_url + "/tokenize", payload=tokenize_payload)
        prompt_ids = tokenized.get("tokens")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise RuntimeError("vLLM tokenize IDs are missing")
        chat_payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 16,
            "seed": 0,
            "n": 1,
            "stream": False,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        first = http_json(base_url + "/chat/completions", payload=chat_payload)
        second = http_json(base_url + "/chat/completions", payload=chat_payload)

        def response(value: Mapping[str, Any]) -> tuple[list[int], list[int], str]:
            choices = value.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise RuntimeError("vLLM deterministic probe choices drifted")
            choice = object_value(choices[0], "vLLM choice")
            message = object_value(choice.get("message"), "vLLM message")
            returned_prompt = value.get("prompt_token_ids")
            response_ids = choice.get("token_ids")
            text = message.get("content")
            for name, values in (
                ("prompt", returned_prompt),
                ("response", response_ids),
            ):
                if (
                    not isinstance(values, list)
                    or not values
                    or any(type(item) is not int or item < 0 for item in values)
                ):
                    raise RuntimeError(f"vLLM {name} token IDs are malformed")
            if returned_prompt != prompt_ids or not isinstance(text, str):
                raise RuntimeError("vLLM prompt IDs or response text drifted")
            return returned_prompt, response_ids, text

        first_prompt, first_response, first_text = response(first)
        second_prompt, second_response, second_text = response(second)
        return {
            "model_id": model_id,
            "prompt_token_ids": first_prompt,
            "response_token_ids": first_response,
            "repeat_prompt_token_ids": second_prompt,
            "repeat_response_token_ids": second_response,
            "repeat_text_equal": first_text == second_text,
        }

    def swe_metadata_snapshot(self) -> dict[str, Any]:
        from agentenv_swebench_verified.dataset import VerifiedDataset  # type: ignore
        from agentenv_swebench_verified.environment import (  # type: ignore
            VerifiedEpisodeManager,
        )
        from agentenv_swebench_verified.exporter import (  # type: ignore
            PredictionStore,
            SolutionPatchExporter,
        )
        from agentenv_swebench_verified.images import (  # type: ignore
            VerifiedImageManifest,
        )
        from agentenv_swebench_verified.protocol import (  # type: ignore
            EVALUATION_MAX_POLICY_TURNS,
        )
        from agentenv_swebench_verified.sandbox import (  # type: ignore
            SANDBOX_CONTRACT,
        )
        from agentenv_swebench_verified.testspec import (  # type: ignore
            OfficialTestSpecResolver,
            TESTSPEC_BINDING_CONTRACT,
        )
        from agentenv_swebench_verified.workspace import (  # type: ignore
            VerifiedWorkspaceMaterializer,
        )

        assets = self.section("assets")
        runtime = self.section("runtime")
        scratch = ensure_private_directory(
            path_value(runtime["pod_local_root"], "pod-local root")
            / "preflight-metadata"
        )
        episodes = ensure_private_directory(scratch / "episodes")
        predictions = ensure_private_directory(scratch / "predictions")
        try:
            dataset = VerifiedDataset(
                path_value(assets["dataset_manifest"], "dataset manifest")
            )
            images_path = path_value(assets["image_digests"], "image digests")
            images = VerifiedImageManifest(
                images_path,
                expected_manifest_sha256=sha256_file(images_path),
            )
            manager = VerifiedEpisodeManager(
                dataset=dataset,
                materializer=VerifiedWorkspaceMaterializer(
                    mirrors_root=path_value(
                        runtime["mirrors_root"], "mirrors root"
                    ),
                    episodes_root=episodes,
                ),
                testspec_resolver=OfficialTestSpecResolver(
                    source_root=path_value(assets["harness_root"], "harness root")
                ),
                sandbox_factory=lambda *_args: (_ for _ in ()).throw(
                    RuntimeError("metadata probe cannot create a sandbox")
                ),
                exporter=SolutionPatchExporter(),
                prediction_store=PredictionStore(
                    predictions, instance_ids=dataset.instance_ids
                ),
                max_native_actions=EVALUATION_MAX_POLICY_TURNS,
                max_observation_bytes=6144,
                runtime_metadata={
                    "testspec_contract": TESTSPEC_BINDING_CONTRACT,
                    "sandbox_contract": SANDBOX_CONTRACT,
                    "image_manifest": images.public_metadata(),
                    "max_observation_tokens": 8192,
                },
            )
            metadata = manager.metadata()
        finally:
            shutil.rmtree(scratch)
        return {
            "schema": metadata.get("schema"),
            "task_count": metadata.get("task_count"),
            "full_benchmark_task_count": metadata.get(
                "full_benchmark_task_count"
            ),
            "supported_arms": metadata.get("supported_arms"),
            "active_slot_count": metadata.get("active_slot_count"),
            "active_workspace_count": metadata.get("active_workspace_count"),
            "official_grading_inside_adapter": metadata.get(
                "official_grading_inside_adapter"
            ),
            "evaluation_max_policy_turns": metadata.get(
                "evaluation_max_policy_turns"
            ),
            "max_native_actions": metadata.get("max_native_actions"),
            "max_observation_tokens": metadata.get("max_observation_tokens"),
        }

    def owned_container_ids(
        self,
        task_index: int | None = None,
        arm: str | None = None,
        slot_index: int | None = None,
    ) -> list[str]:
        arguments = [
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            f"label=amg.owner={OWNER_LABEL}",
        ]
        if task_index is not None:
            arguments.extend(
                ["--filter", f"label=amg.task_index={task_index:04d}"]
            )
        if arm is not None:
            if arm not in ARMS:
                raise ValueError("owned-container arm is invalid")
            arguments.extend(["--filter", f"label=amg.arm={arm}"])
        if slot_index is not None:
            if (
                type(slot_index) is not int
                or not 0 <= slot_index < self.config.task_slots_per_replica
            ):
                raise ValueError("owned-container slot is invalid")
            arguments.extend(
                ["--filter", f"label=amg.slot_index={slot_index}"]
            )
        result = self.docker().run(*arguments)
        if result.returncode != 0:
            raise RuntimeError("owned Docker container census failed")
        return sorted(
            row
            for row in self.docker().output_text(result.stdout).splitlines()
            if row
        )

    def container_record(self, container_id: str) -> Mapping[str, Any]:
        if not isinstance(container_id, str) or not container_id:
            raise ValueError("container ID is invalid")
        inspected = self.docker().run(
            "container", "inspect", "--format", "{{json .}}", container_id
        )
        raw = self.require_docker_result(inspected, "container identity inspection")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError("container identity is invalid JSON") from error
        if not isinstance(value, Mapping):
            raise RuntimeError("container identity is not an object")
        return value

    def _cgroup_paths_unlocked(self, task_index: int | None = None) -> list[str]:
        docker = self.section("docker")
        controller_values: list[set[str]] = []
        for controller in ("memory", "pids"):
            parent = (
                Path(f"/proc/{docker['pid']}/root/sys/fs/cgroup")
                / controller
                / CGROUP_RELATIVE_PREFIX
            )
            if not parent.exists():
                controller_values.append(set())
                continue
            if not parent.is_dir():
                raise RuntimeError("owned cgroup parent is not a directory")
            values = {path.name for path in parent.iterdir() if path.is_dir()}
            if any(
                re.fullmatch(
                    r"[0-9]{4}-(?:native|amg_compaction_only|amg_memory)",
                    value,
                )
                is None
                for value in values
            ):
                raise RuntimeError("owned cgroup parent contains an unknown path")
            controller_values.append(values)
        if controller_values[0] != controller_values[1]:
            raise RuntimeError("memory/pids cgroup residue disagreed")
        values = controller_values[0]
        if task_index is not None:
            prefix = f"{task_index:04d}-"
            values = {value for value in values if value.startswith(prefix)}
        return sorted(values)

    def cgroup_paths(self, task_index: int | None = None) -> list[str]:
        with cgroup_structure_lock():
            return self._cgroup_paths_unlocked(task_index)

    def task_root_path(self, task_index: int) -> Path:
        if type(task_index) is not int or task_index not in self.by_task:
            raise ValueError("task index is outside the production manifest")
        runtime = self.section("runtime")
        return (
            path_value(runtime["pod_local_root"], "pod-local root")
            / "tasks"
            / f"task-{task_index:04d}"
        )

    @staticmethod
    def decode_mountinfo_path(value: str) -> str:
        return (
            value.replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\")
        )

    def mount_records_under(self, root: Path) -> list[dict[str, Any]]:
        if not root.is_absolute():
            raise ValueError("mount census root must be absolute")
        resolved = root.resolve(strict=False)
        records: list[dict[str, Any]] = []
        seen_namespaces: set[str] = set()
        for proc in sorted(
            (path for path in Path("/proc").iterdir() if path.name.isdigit()),
            key=lambda path: int(path.name),
        ):
            try:
                namespace = os.readlink(proc / "ns/mnt")
            except FileNotFoundError:
                continue
            except OSError as error:
                raise RuntimeError("cannot inspect a process mount namespace") from error
            if namespace in seen_namespaces:
                continue
            try:
                rows = (proc / "mountinfo").read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                continue
            except OSError as error:
                raise RuntimeError("cannot read a process mount table") from error
            seen_namespaces.add(namespace)
            for row in rows:
                before, separator, after = row.partition(" - ")
                fields = before.split()
                suffix = after.split()
                if not separator or len(fields) < 6 or len(suffix) < 2:
                    raise RuntimeError("mountinfo row is malformed")
                mountpoint = Path(self.decode_mountinfo_path(fields[4]))
                try:
                    mountpoint.relative_to(resolved)
                except ValueError:
                    continue
                records.append(
                    {
                        "namespace": namespace,
                        "representative_pid": int(proc.name),
                        "mount_id": int(fields[0]),
                        "mount_point": str(mountpoint),
                        "fs_type": suffix[0],
                        "source": self.decode_mountinfo_path(suffix[1]),
                    }
                )
        return sorted(
            records,
            key=lambda row: (
                row["namespace"],
                row["mount_point"],
                row["mount_id"],
            ),
        )

    def cgroup_process_ids(self, task_index: int | None = None) -> list[int]:
        with cgroup_structure_lock():
            docker = self.section("docker")
            values_by_controller: list[set[int]] = []
            selected = set(self._cgroup_paths_unlocked(task_index))
            for controller in ("memory", "pids"):
                parent = (
                    Path(f"/proc/{docker['pid']}/root/sys/fs/cgroup")
                    / controller
                    / CGROUP_RELATIVE_PREFIX
                )
                values: set[int] = set()
                for name in selected:
                    root = parent / name
                    if not root.is_dir():
                        raise RuntimeError(
                            "owned cgroup disappeared during process census"
                        )
                    for path in sorted(root.rglob("cgroup.procs")):
                        try:
                            lines = path.read_text(encoding="ascii").splitlines()
                        except OSError as error:
                            raise RuntimeError(
                                "cannot read owned cgroup processes"
                            ) from error
                        for line in lines:
                            if not line.isdigit() or int(line) <= 0:
                                raise RuntimeError(
                                    "owned cgroup contains an invalid PID"
                                )
                            values.add(int(line))
                values_by_controller.append(values)
        if values_by_controller[0] != values_by_controller[1]:
            raise RuntimeError("memory/pids cgroup process census disagreed")
        return sorted(values_by_controller[0])

    def certified_image_identities(self) -> dict[str, str]:
        identities: dict[str, str] = {}
        for row in self.oci_store().rows():
            image = text_value(row.get("image"), "certified image alias")
            config = object_value(row.get("config"), "certified image config")
            digest = text_value(config.get("digest"), "certified image ID")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
                raise RuntimeError("certified image ID drifted")
            identities[image] = digest
        if len(identities) != 500:
            raise RuntimeError("certified image identity denominator drifted")
        return identities

    def loaded_task_image_identities(
        self, task_index: int | None = None
    ) -> list[tuple[str, str]]:
        if task_index is None:
            expected = self.certified_image_identities()
        else:
            identity = self.task_image_identity(task_index)
            expected = {} if identity is None else {identity[0]: identity[1]}
        result = self.docker().run(
            "image",
            "ls",
            "--all",
            "--no-trunc",
            "--format",
            "{{.Repository}}:{{.Tag}}\t{{.ID}}",
        )
        output = self.require_docker_result(result, "Docker image census")
        observed: dict[str, str] = {}
        observed_ids: set[str] = set()
        for line in output.splitlines():
            alias, separator, image_id = line.partition("\t")
            if not separator or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
                raise RuntimeError("Docker image census row is malformed")
            observed[alias] = image_id
            observed_ids.add(image_id)
        for alias, image_id in expected.items():
            if alias in observed and observed[alias] != image_id:
                raise RuntimeError("certified Docker image alias drifted")
        by_id: dict[str, list[str]] = {}
        for alias, image_id in expected.items():
            by_id.setdefault(image_id, []).append(alias)
        return sorted(
            (sorted(aliases)[0], image_id)
            for image_id, aliases in by_id.items()
            if image_id in observed_ids
        )

    def loaded_task_images(self, task_index: int | None = None) -> list[str]:
        return [
            image
            for image, _ in self.loaded_task_image_identities(task_index)
        ]

    def reconcile_unbound_loaded_images(self) -> Mapping[str, Any]:
        """Evict certified images left unbound after every shard reconciles."""

        loaded = self.loaded_task_image_identities()
        evicted = [self.evict_image(image, image_id) for image, image_id in loaded]
        remaining = self.loaded_task_image_identities()
        if remaining:
            raise RuntimeError("unbound certified Docker images survived reconciliation")
        return {
            "schema": "swebench_triad_unbound_image_reconciliation_v1",
            "status": "PASS",
            "evicted_images": evicted,
            "remaining_images": 0,
        }

    def task_root_indices(self) -> list[int]:
        tasks_root = (
            path_value(
                self.section("runtime")["pod_local_root"], "pod-local root"
            )
            / "tasks"
        )
        if not tasks_root.exists():
            return []
        if tasks_root.is_symlink() or not tasks_root.is_dir():
            raise RuntimeError("task-root parent changed type")
        indices = []
        for path in sorted(tasks_root.iterdir()):
            match = re.fullmatch(r"task-([0-9]{4})", path.name)
            if match is None or path.is_symlink() or not path.is_dir():
                raise RuntimeError("task-root parent contains an unknown path")
            index = int(match.group(1))
            if index not in self.by_task:
                raise RuntimeError("task-root index is outside the manifest")
            indices.append(index)
        return indices

    def verify_task_rootfs(self, task_index: int) -> Mapping[str, Any] | None:
        receipt = self.stage_receipt(task_index)
        if receipt is None:
            return None
        rootfs_cache = path_value(
            receipt.get("rootfs_cache"), "staged rootfs cache"
        )
        if not rootfs_cache.exists():
            return None
        current = attest_rootfs(rootfs_cache)
        if current != receipt.get("rootfs_attestation"):
            raise RuntimeError("staged task rootfs mutation was detected")
        return current

    def remove_inactive_task_root(self, task_index: int) -> bool:
        task_root = self.task_root_path(task_index)
        if not task_root.exists():
            return False
        info = task_root.lstat()
        if task_root.is_symlink() or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("owned task root changed type")
        self.verify_task_rootfs(task_index)
        mounts = self.mount_records_under(task_root)
        if mounts:
            raise RuntimeError("owned task root still contains a live mount")
        if self.owned_container_ids(task_index) or self.cgroup_process_ids(task_index):
            raise RuntimeError("owned task root still has a live runtime")
        shutil.rmtree(task_root)
        return True

    def reconcile_cell(
        self,
        config: RunConfig,
        *,
        generation: int,
        before_preflight: bool,
        slot: RuntimeLaneToken,
    ) -> Mapping[str, Any]:
        if config not in self.configs:
            raise ValueError("cell recovery config is outside the manifest")
        if type(generation) is not int or generation <= 0:
            raise ValueError("cell recovery generation is invalid")
        if type(before_preflight) is not bool:
            raise TypeError("cell recovery phase flag must be boolean")
        slot = self.require_slot(
            slot,
            task_index=config.task.task_index,
            allow_global=before_preflight,
        )
        removed = []
        for container_id in self.owned_container_ids(
            config.task.task_index,
            config.capability.arm.value,
            slot.slot_index,
        ):
            record = self.container_record(container_id)
            container_config = object_value(record.get("Config"), "container config")
            labels = container_config.get("Labels")
            if not isinstance(labels, Mapping) or labels.get("amg.owner") != OWNER_LABEL:
                raise RuntimeError("cell recovery found a non-owned container")
            expected = {
                "amg.task_index": f"{config.task.task_index:04d}",
                "amg.arm": config.capability.arm.value,
                "amg.slot_index": str(slot.slot_index),
                "amg.server_port": str(slot.server_port),
            }
            if any(labels.get(name) != value for name, value in expected.items()):
                raise RuntimeError("cell recovery container identity drifted")
            old_generation = labels.get("amg.generation")
            old_lane_generation = labels.get("amg.lane_generation")
            if (
                not isinstance(old_generation, str)
                or not old_generation.isdigit()
                or int(old_generation) > generation
                or not isinstance(old_lane_generation, str)
                or not old_lane_generation.isdigit()
                or int(old_lane_generation) > slot.generation
            ):
                raise RuntimeError("cell recovery container generation drifted")
            removed.append(self.remove_owned_container_id(container_id))

        cell_name = f"{config.task.task_index:04d}-{config.capability.arm.value}"
        removed_cgroup = None
        if cell_name in self.cgroup_paths(config.task.task_index):
            removed_cgroup = self.cgroup_backend().remove(
                f"{CGROUP_RELATIVE_PREFIX}/{cell_name}"
            )
            if removed_cgroup.get("removed") is not True:
                raise RuntimeError("stale cell cgroup was not removed")
        mounts = self.mount_records_under(self.task_root_path(config.task.task_index))
        if mounts:
            raise RuntimeError("stale cell mount namespace survived reconciliation")
        if self.cgroup_process_ids(config.task.task_index):
            raise RuntimeError("stale cell process survived reconciliation")
        return {
            "schema": "swebench_triad_cell_reconciliation_v1",
            "task_index": config.task.task_index,
            "arm": config.capability.arm.value,
            "new_generation": generation,
            "before_preflight": before_preflight,
            "removed_containers": removed,
            "removed_cgroup": removed_cgroup,
            "mounts_after": 0,
            "processes_after": 0,
            "slot_index": slot.slot_index,
            "server_port": slot.server_port,
            "lane_generation": slot.generation,
        }

    def global_residue_snapshot(self) -> dict[str, int]:
        runtime = self.section("runtime")
        pod_root = path_value(runtime["pod_local_root"], "pod-local root")
        tasks_root = pod_root / "tasks"
        scratch_paths = []
        if tasks_root.is_dir():
            scratch_paths = [path for path in tasks_root.iterdir()]
        containers = self.owned_container_ids()
        cgroups = self.cgroup_paths()
        processes = self.cgroup_process_ids()
        mounts = self.mount_records_under(pod_root)
        tmpfs_mounts = [row for row in mounts if row["fs_type"] == "tmpfs"]
        images = self.loaded_task_images()
        return {
            "active_owned_processes": len(processes),
            "active_cgroups": len(cgroups),
            "active_tmpfs_mounts": len(tmpfs_mounts),
            "active_mounts": len(mounts),
            "active_scratch_paths": len(scratch_paths),
            "loaded_task_images": len(images),
            "owned_containers": len(containers),
        }

    def stage_task(
        self,
        task_index: int,
        *,
        slot: RuntimeLaneToken,
    ) -> ProductionTaskStage:
        if type(task_index) is not int or task_index not in self.by_task:
            raise ValueError("task index is outside the production manifest")
        slot = self.require_slot(slot, task_index=task_index)
        configs = self.by_task[task_index]
        instance_id = configs[0].task.task_id
        if any(config.task.task_id != instance_id for config in configs):
            raise RuntimeError("production triad task identity drifted")
        row = self.dataset_rows()[task_index]
        if row.get("instance_id") != instance_id:
            raise RuntimeError("dataset task order drifted from the manifest")
        repo = text_value(row.get("repo"), "dataset repository")
        base_commit = commit_value(row.get("base_commit"), "dataset base commit")
        testspec = self.testspec_resolver().resolve(row)
        if (
            testspec.instance_id != instance_id
            or testspec.repo != repo
            or testspec.base_commit != base_commit
        ):
            raise RuntimeError("pinned TestSpec task identity drifted")
        binding = self.oci_store().resolve(testspec.instance_image_key)

        runtime = self.section("runtime")
        task_root = ensure_private_directory(
            path_value(runtime["pod_local_root"], "pod-local root")
            / "tasks"
            / f"task-{task_index:04d}"
        )
        phase_timings: list[dict[str, Any]] = []
        rootfs_cache = timed_phase(
            phase_timings,
            "rootfs_materialization",
            lambda: materialize_rootfs(binding, task_root / "oci-rootfs"),
        )
        archive_path = task_root / "image.tar"

        def prepare_archive() -> None:
            if archive_path.exists():
                archive_receipt = read_json_object(
                    task_root / "image.tar.receipt.json",
                    "Docker archive receipt",
                )
                if (
                    archive_receipt.get("image") != binding.image
                    or archive_receipt.get("manifest_digest")
                    != binding.manifest_digest
                    or archive_receipt.get("config_digest")
                    != binding.config_digest
                    or archive_receipt.get("archive_size")
                    != archive_path.stat().st_size
                    or archive_receipt.get("archive_sha256")
                    != sha256_file(archive_path)
                ):
                    raise RuntimeError("resumed Docker archive identity drifted")
            else:
                build_docker_archive(binding, archive_path)

        timed_phase(phase_timings, "docker_archive", prepare_archive)
        docker_load = timed_phase(
            phase_timings,
            "docker_load",
            lambda: self.docker().ensure_loaded(binding, archive_path),
        )
        mirror_path = timed_phase(
            phase_timings,
            "repository_mirror",
            lambda: ensure_repository_mirror(
                rootfs_cache / "rootfs",
                path_value(runtime["mirrors_root"], "mirrors root"),
                repo=repo,
                base_commit=base_commit,
            ),
        )
        stage = ProductionTaskStage(
            task_index=task_index,
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            binding=binding,
            rootfs_cache=rootfs_cache,
            archive_path=archive_path,
            mirror_path=mirror_path,
            task_root=task_root,
            slot_index=slot.slot_index,
            server_port=slot.server_port,
            lane_generation=slot.generation,
            lane_fencing_token=slot.fencing_token,
        )
        atomic_write_json(
            self.config.run_root
            / "control"
            / "stages"
            / f"task-{task_index:04d}.json",
            {
                "schema": "swebench_triad_task_stage_v1",
                "task_index": task_index,
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": base_commit,
                "binding": binding.receipt(),
                "rootfs_cache": str(rootfs_cache),
                "rootfs_attestation": attest_rootfs(rootfs_cache),
                "archive_path": str(archive_path),
                "archive_sha256": sha256_file(archive_path),
                "mirror_path": str(mirror_path),
                "docker_load": docker_load,
                "network_downloads": 0,
                "slot_index": slot.slot_index,
                "server_port": slot.server_port,
                "lane_generation": slot.generation,
                "lane_fencing_token_sha256": hashlib.sha256(
                    slot.fencing_token.encode("ascii")
                ).hexdigest(),
                "phase_timings": phase_timings,
            },
        )
        return stage

    @staticmethod
    def require_stage(
        config: RunConfig,
        stage: Any,
        slot: RuntimeLaneToken,
    ) -> ProductionTaskStage:
        if not isinstance(stage, ProductionTaskStage):
            raise TypeError("production cell requires a typed task stage")
        if (
            stage.task_index != config.task.task_index
            or stage.instance_id != config.task.task_id
            or stage.slot_index != slot.slot_index
            or stage.server_port != slot.server_port
            or stage.lane_generation != slot.generation
            or stage.lane_fencing_token != slot.fencing_token
        ):
            raise ValueError("production cell stage identity drifted")
        return stage

    def cgroup_backend(self) -> MountNamespaceCgroupV1Backend:
        docker = self.section("docker")
        return MountNamespaceCgroupV1Backend(
            namespace_pid=docker["pid"],
            python_executable=os.environ.get("SWEBENCH_TRIAD_HOST_PYTHON")
            or shutil.which("python3")
            or "python3",
            timeout_seconds=45,
        )

    @staticmethod
    def require_docker_result(
        result: subprocess.CompletedProcess,
        label: str,
    ) -> str:
        if not isinstance(result, subprocess.CompletedProcess):
            raise RuntimeError(f"{label} returned an invalid process result")
        stdout = DockerCli.output_text(result.stdout)
        if result.returncode != 0:
            stderr = DockerCli.output_text(result.stderr)
            raise RuntimeError(f"{label} failed: {(stderr or stdout)[-2000:]}")
        return stdout.strip()

    def container_name(self, config: RunConfig, generation: int) -> str:
        if type(generation) is not int or generation <= 0:
            raise ValueError("cell generation is invalid")
        return (
            f"{CONTAINER_NAME_PREFIX}{config.task.task_index:04d}-"
            f"{config.capability.arm.value}-g{generation:08d}"
        )

    @staticmethod
    def python_path(source_root: Path) -> str:
        paths = (
            source_root / "AgentGym-RL/scripts/agentmemory",
            source_root / "AgentGym/agentenv",
            source_root / "AgentGym/agentenv-swebench-verified",
            source_root / "AgentGym/agentenv-swesmith",
            source_root / "AgentGym/agentenv-agentmemory",
        )
        if any(not path.is_dir() for path in paths):
            raise RuntimeError("production Python source paths are incomplete")
        return ":".join(str(path) for path in paths)

    @staticmethod
    def mount_arguments(path: Path, *, readonly: bool) -> list[str]:
        resolved = path.resolve(strict=True)
        value = f"type=bind,src={resolved},dst={resolved}"
        if readonly:
            value += ",readonly"
        return ["--mount", value]

    def server_container_arguments(
        self,
        config: RunConfig,
        stage: ProductionTaskStage,
        *,
        generation: int,
        envelope: CgroupV1CellEnvelope,
        attempt_root: Path,
        container_name: str,
        slot: RuntimeLaneToken,
    ) -> list[str]:
        assets = self.section("assets")
        source = self.section("source")
        runtime = self.section("runtime")
        source_root = path_value(source["root"], "source root")
        episodes = ensure_private_directory(attempt_root / "episodes")
        predictions = ensure_private_directory(attempt_root / "predictions")
        leases = ensure_private_directory(attempt_root / "uid-leases")
        image_digests = path_value(assets["image_digests"], "image digests")
        environment = {
            "PYTHONPATH": self.python_path(source_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "SWEBENCH_VERIFIED_DATASET_MANIFEST": str(
                path_value(assets["dataset_manifest"], "dataset manifest")
            ),
            "SWEBENCH_VERIFIED_HARNESS_ROOT": str(
                path_value(assets["harness_root"], "harness root")
            ),
            "SWEBENCH_VERIFIED_IMAGE_DIGESTS": str(image_digests),
            "SWEBENCH_VERIFIED_IMAGE_DIGESTS_SHA256": sha256_file(
                image_digests
            ),
            "SWEBENCH_VERIFIED_MIRRORS_ROOT": str(
                path_value(runtime["mirrors_root"], "mirrors root")
            ),
            "SWEBENCH_VERIFIED_EPISODES_ROOT": str(episodes),
            "SWEBENCH_VERIFIED_PREDICTIONS_ROOT": str(predictions),
            "SWEBENCH_VERIFIED_OCI_CACHE_ROOT": str(stage.rootfs_cache.parent),
            "SWEBENCH_VERIFIED_RG_BINARY": str(
                path_value(assets["rg_binary"], "ripgrep binary")
            ),
            "SWEBENCH_VERIFIED_RG_SHA256": assets["rg_sha256"],
            "SWEBENCH_VERIFIED_UID_LEASE_ROOT": str(leases),
            "SWEBENCH_VERIFIED_HOST": "127.0.0.1",
            "SWEBENCH_VERIFIED_PORT": str(slot.server_port),
            "SWEBENCH_VERIFIED_WORKSPACE_BYTES": str(
                runtime["workspace_bytes"]
            ),
            "SWEBENCH_VERIFIED_WORKSPACE_INODES": str(
                runtime["workspace_inodes"]
            ),
            "SWEBENCH_VERIFIED_MAX_PROCESSES": str(runtime["max_processes"]),
            "SWEBENCH_VERIFIED_MAX_OBSERVATION_TOKENS": "8192",
            "SWEBENCH_TRIAD_ROOTFS_CACHE": str(stage.rootfs_cache),
            "SWEBENCH_TRIAD_WORKSPACE_BYTES": str(runtime["workspace_bytes"]),
            "SWEBENCH_TRIAD_WORKSPACE_INODES": str(runtime["workspace_inodes"]),
            "SWEBENCH_TRIAD_EXTERNAL_MEMORY_BYTES": str(
                runtime["external_memory_bytes"]
            ),
            "SWEBENCH_TRIAD_EXTERNAL_MEMORY_INODES": str(
                runtime["external_memory_inodes"]
            ),
        }
        arguments = [
            "container",
            "run",
            "--detach",
            "--name",
            container_name,
            "--label",
            f"amg.owner={OWNER_LABEL}",
            "--label",
            f"amg.task_index={config.task.task_index:04d}",
            "--label",
            f"amg.arm={config.capability.arm.value}",
            "--label",
            f"amg.generation={generation:08d}",
            "--label",
            f"amg.slot_index={slot.slot_index}",
            "--label",
            f"amg.server_port={slot.server_port}",
            "--label",
            f"amg.lane_generation={slot.generation:08d}",
            "--network",
            "host",
            "--privileged",
            "--runtime",
            "amg-runc",
            *envelope.docker_resource_arguments(),
        ]
        readonly_paths = {
            source_root,
            path_value(assets["dataset_manifest"], "dataset manifest").parent,
            path_value(assets["dataset_jsonl"], "dataset JSONL").parent,
            image_digests.parent,
            path_value(assets["harness_root"], "harness root"),
            path_value(runtime["mirrors_root"], "mirrors root"),
            path_value(assets["rg_binary"], "ripgrep binary").parent,
            stage.rootfs_cache,
        }
        for path in sorted(readonly_paths, key=lambda item: str(item)):
            arguments.extend(self.mount_arguments(path, readonly=True))
        arguments.extend(self.mount_arguments(attempt_root, readonly=False))
        for name, value in sorted(environment.items()):
            arguments.extend(["--env", f"{name}={value}"])
        arguments.extend(
            [
                "--entrypoint",
                text_value(runtime["container_python"], "container Python"),
                stage.binding.image,
                "-m",
                "swebench_triad_eval.server_runtime",
            ]
        )
        return arguments

    def container_pid(self, container_name: str) -> int:
        result = self.docker().run(
            "container",
            "inspect",
            "--format",
            "{{.State.Pid}}",
            container_name,
        )
        value = self.require_docker_result(result, "container PID inspection")
        if not value.isdigit() or int(value) <= 0:
            raise RuntimeError("server container has no live init PID")
        return int(value)

    def wait_server(self, base_url: str, container_name: str) -> Mapping[str, Any]:
        deadline = time.monotonic() + 120.0
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            state = self.docker().run(
                "container",
                "inspect",
                "--format",
                "{{.State.Running}}",
                container_name,
            )
            running = self.require_docker_result(state, "container state inspection")
            if running != "true":
                raise RuntimeError("SWE server container exited before readiness")
            try:
                health = http_json(base_url + "/", timeout=2.0)
                if health.get("status") == "ok":
                    return http_json(base_url + "/metadata", timeout=10.0)
            except RuntimeError as error:
                last_error = error
            time.sleep(0.25)
        raise RuntimeError("SWE server readiness timed out") from last_error

    def container_logs(self, container_name: str) -> dict[str, str]:
        result = self.docker().run("container", "logs", container_name)
        return {
            "stdout": self.docker().output_text(result.stdout)[-4 * 1024 * 1024 :],
            "stderr": self.docker().output_text(result.stderr)[-4 * 1024 * 1024 :],
            "returncode": str(result.returncode),
        }

    def stop_remove_container(self, container_name: str) -> dict[str, Any]:
        stopped = self.docker().run(
            "container", "stop", "--time", "30", container_name
        )
        stop_text = self.docker().output_text(stopped.stderr).lower()
        if stopped.returncode != 0 and "no such container" not in stop_text:
            raise RuntimeError("owned server container could not stop")
        removed = self.docker().run(
            "container", "rm", "--force", container_name
        )
        remove_text = self.docker().output_text(removed.stderr).lower()
        if removed.returncode != 0 and "no such container" not in remove_text:
            raise RuntimeError("owned server container could not be removed")
        return {"stopped": True, "removed": True, "name": container_name}

    def run_cell(
        self,
        config: RunConfig,
        stage: Any,
        *,
        generation: int,
        slot: RuntimeLaneToken,
    ) -> Mapping[str, Any]:
        slot = self.require_slot(slot, task_index=config.task.task_index)
        typed_stage = self.require_stage(config, stage, slot)
        if type(generation) is not int or generation <= 0:
            raise ValueError("cell generation is invalid")
        if self.owned_container_ids(
            config.task.task_index, slot_index=slot.slot_index
        ):
            raise RuntimeError("owned task container residue forbids a new cell")
        if self.cgroup_paths(config.task.task_index):
            raise RuntimeError("owned task cgroup residue forbids a new cell")

        runtime = self.section("runtime")
        cell_name = f"{config.task.task_index:04d}-{config.capability.arm.value}"
        container_name = self.container_name(config, generation)
        attempt_root = typed_stage.task_root / "attempts" / (
            f"{config.capability.arm.value}-g{generation:08d}"
        )
        if attempt_root.exists():
            raise RuntimeError("attempt-local runtime directory already exists")
        attempt_root = ensure_private_directory(attempt_root)
        receipt_path = (
            self.config.run_root
            / "control"
            / "cells"
            / cell_name
            / f"generation-{generation:08d}.json"
        )
        envelope = CgroupV1CellEnvelope(
            cell_name=cell_name,
            limits=CgroupV1Limits(
                memory_bytes=runtime["memory_bytes"],
                max_processes=runtime["max_processes"],
            ),
            backend=self.cgroup_backend(),
        )
        base_url = f"http://127.0.0.1:{slot.server_port}"
        run_id = (
            f"amg-sbv-{config.task.task_index:04d}-"
            f"{config.capability.arm.value}-g{generation:08d}"
        )
        run_capability = secrets.token_urlsafe(48)
        phase_timings: list[dict[str, Any]] = []
        record: dict[str, Any] = {
            "schema": "swebench_triad_cell_runtime_v1",
            "status": "FAIL",
            "task_index": config.task.task_index,
            "instance_id": config.task.task_id,
            "arm": config.capability.arm.value,
            "generation": generation,
            "container_name": container_name,
            "run_id": run_id,
            "run_capability_sha256": hashlib.sha256(
                run_capability.encode("ascii")
            ).hexdigest(),
            "slot_index": slot.slot_index,
            "server_port": slot.server_port,
            "lane_generation": slot.generation,
            "lane_fencing_token_sha256": hashlib.sha256(
                slot.fencing_token.encode("ascii")
            ).hexdigest(),
            "phase_timings": phase_timings,
        }
        if self.config.shared_model_pool is not None:
            shared = self.config.shared_model_pool
            expected_replica = int.from_bytes(
                hashlib.sha256(config.task.task_id.encode("utf-8")).digest()[:8],
                "big",
            ) % shared["replica_count"]
            if expected_replica != shared["replica_index"]:
                raise RuntimeError(
                    "task was routed to the wrong shared-model replica"
                )
            live_pool = validate_shared_model_pool_snapshot(
                self.shared_model_pool_snapshot(require_preflight_binding=True),
                "cell runtime shared model pool snapshot",
            )
            if live_pool.get("replica_index") != expected_replica:
                raise RuntimeError("live shared-model replica binding drifted")
            record["shared_model_pool"] = dict(live_pool)
        prepared = False
        container_created = False
        try:
            record["rootfs_before"] = timed_phase(
                phase_timings,
                "rootfs_attestation_before",
                lambda: attest_rootfs(typed_stage.rootfs_cache),
            )
            record["cgroup_prepare"] = timed_phase(
                phase_timings, "cgroup_prepare", envelope.prepare
            )
            prepared = True
            arguments = self.server_container_arguments(
                config,
                typed_stage,
                generation=generation,
                envelope=envelope,
                attempt_root=attempt_root,
                container_name=container_name,
                slot=slot,
            )
            launched = timed_phase(
                phase_timings,
                "environment_container_launch",
                lambda: self.docker().run(*arguments),
            )
            container_id = self.require_docker_result(
                launched, "SWE server container launch"
            )
            if len(container_id) < 12:
                raise RuntimeError("Docker returned an invalid server container ID")
            container_created = True
            record["container_id"] = container_id
            init_pid = self.container_pid(container_name)
            record["cgroup_descendants_before"] = envelope.verify_descendants(
                container_init_pid=init_pid
            )
            metadata_before = timed_phase(
                phase_timings,
                "environment_server_readiness",
                lambda: self.wait_server(base_url, container_name),
            )
            if (
                metadata_before.get("active_slot_count") != 0
                or metadata_before.get("active_workspace_count") != 0
            ):
                raise RuntimeError("new SWE server started with active policy state")
            record["metadata_before"] = metadata_before

            image_digests = path_value(
                self.section("assets")["image_digests"], "image digests"
            )

            def endpoint_resolver(value: RunConfig) -> SwebenchRuntimeEndpoint:
                if value.full_config_sha256 != config.full_config_sha256:
                    raise RuntimeError("runtime factory requested another cell")
                return SwebenchRuntimeEndpoint(
                    env_server_base=base_url,
                    private_run_id=run_id,
                    run_capability=run_capability,
                    image_manifest_sha256=sha256_file(image_digests),
                    task_index=config.task.task_index,
                    arm=config.capability.arm.value,
                    generation=generation,
                )

            evidence = PrivateEvidenceStore(self.config.evidence_root)
            factory = make_swebench_runtime_factory(
                evidence_store=evidence,
                endpoint_resolver=endpoint_resolver,
                model_base_url=text_value(
                    self.section("serving")["base_url"], "serving base URL"
                ),
                model_timeout_seconds=positive_number(
                    runtime["model_timeout_seconds"], "model timeout"
                ),
                environment_timeout_seconds=integer_value(
                    runtime["environment_timeout_seconds"],
                    "environment timeout",
                ),
            )
            bindings = factory(config)
            runner = PairedRunner(
                controller=AgentGymPolicyTurnController.from_agentenv(),
                evidence_store=evidence,
            )
            endpoint = timed_phase(
                phase_timings,
                "policy_and_model_execution",
                lambda: runner.run_task(
                    config,
                    bindings.adapter,
                    bindings.model,
                ),
            )
            transport_events = getattr(bindings.model.transport, "events", None)
            if not isinstance(transport_events, list) or any(
                not isinstance(event, Mapping) for event in transport_events
            ):
                raise RuntimeError("exact-token transport timing receipt is missing")
            record["model_transport_events"] = [
                dict(event) for event in transport_events
            ]
            metadata_after = timed_phase(
                phase_timings,
                "environment_finalization",
                lambda: http_json(base_url + "/metadata", timeout=10.0),
            )
            if (
                metadata_after.get("active_slot_count") != 0
                or metadata_after.get("active_workspace_count") != 0
            ):
                raise RuntimeError("SWE policy state survived adapter close")
            record["metadata_after"] = metadata_after
            record["cgroup_descendants_after"] = envelope.verify_descendants(
                container_init_pid=init_pid
            )
            record["rootfs_after"] = attest_rootfs(typed_stage.rootfs_cache)
            if record["rootfs_after"] != record["rootfs_before"]:
                raise RuntimeError("active task rootfs mutated during the cell")
            record["container_logs"] = self.container_logs(container_name)
            record["container_cleanup"] = timed_phase(
                phase_timings,
                "environment_container_cleanup",
                lambda: self.stop_remove_container(container_name),
            )
            container_created = False
            record["cgroup_teardown"] = timed_phase(
                phase_timings, "cgroup_teardown", envelope.teardown
            )
            prepared = False
            record["status"] = "PASS"
            atomic_write_json(receipt_path, record)
            return endpoint
        except BaseException as error:
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            cleanup_errors: list[str] = []
            if container_created:
                try:
                    record["failure_container_logs"] = self.container_logs(
                        container_name
                    )
                except BaseException as cleanup_error:
                    cleanup_errors.append(repr(cleanup_error))
                try:
                    record["failure_container_cleanup"] = (
                        self.stop_remove_container(container_name)
                    )
                    container_created = False
                except BaseException as cleanup_error:
                    cleanup_errors.append(repr(cleanup_error))
            if prepared and not container_created:
                try:
                    record["failure_cgroup_teardown"] = envelope.teardown()
                    prepared = False
                except BaseException as cleanup_error:
                    cleanup_errors.append(repr(cleanup_error))
            record["cleanup_errors"] = cleanup_errors
            atomic_write_json(receipt_path, record)
            if cleanup_errors:
                raise RuntimeError(
                    "cell execution failed and owned cleanup was incomplete"
                ) from error
            raise

    def official_grader_config(self) -> OfficialGraderConfig:
        grader = self.section("grader")
        assets = self.section("assets")
        return OfficialGraderConfig(
            python_executable=path_value(
                grader["python_executable"], "grader Python"
            ),
            harness_root=path_value(assets["harness_root"], "harness root"),
            dataset_path=path_value(assets["dataset_jsonl"], "dataset JSONL"),
            output_root=path_value(grader["output_root"], "grader output root"),
            docker_socket=path_value(
                self.section("docker")["socket"], "Docker socket"
            ),
            command_ledger_path=self.config.run_root
            / "full"
            / "command-exit-ledger.jsonl",
            semaphore_root=(
                path_value(grader["semaphore_root"], "grader semaphore root")
                if self.config.shared_model_pool is not None
                else self.config.run_root / "state" / "grader-semaphore"
            ),
            max_concurrent_graders=(
                grader["global_max_concurrency"]
                if self.config.shared_model_pool is not None
                else 8
            ),
        )

    @staticmethod
    def official_request(
        *,
        key: CellKey,
        accepted: Mapping[str, Any],
        prediction: Mapping[str, Any],
        handoff: Mapping[str, Any],
        grader_attempt: int,
    ) -> OfficialGradeRequest:
        generation = accepted.get("attempt_generation")
        if type(generation) is not int or generation <= 0:
            raise ValueError("accepted cell generation is invalid")
        return OfficialGradeRequest(
            task_index=key.task_index,
            arm=key.arm,
            generation=generation,
            grader_attempt=grader_attempt,
            prediction=prediction,
            accepted_cell=accepted,
            queued_handoff=handoff,
        )

    def grader_container_ids(self, request: OfficialGradeRequest) -> list[str]:
        expected_name = (
            f"sweb.eval.{request.prediction['instance_id'].lower()}."
            f"{grader_run_id(request)}"
        )
        result = self.docker().run(
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            f"name={expected_name}",
        )
        if result.returncode != 0:
            raise RuntimeError("official-grader container census failed")
        matches = []
        for container_id in self.docker().output_text(result.stdout).splitlines():
            if not container_id:
                continue
            record = self.container_record(container_id)
            name = str(record.get("Name", "")).lstrip("/")
            if name == expected_name:
                matches.append(container_id)
        return sorted(matches)

    def remove_grader_container(
        self,
        container_id: str,
        request: OfficialGradeRequest,
    ) -> Mapping[str, Any]:
        expected_name = (
            f"sweb.eval.{request.prediction['instance_id'].lower()}."
            f"{grader_run_id(request)}"
        )
        record = self.container_record(container_id)
        name = str(record.get("Name", "")).lstrip("/")
        if name != expected_name:
            raise RuntimeError("refusing to remove a non-bound grader container")
        removed = self.docker().run("container", "rm", "--force", container_id)
        self.require_docker_result(removed, "official-grader container removal")
        return {"container_id": container_id, "name": name, "removed": True}

    def reconcile_grade(
        self,
        *,
        key: CellKey,
        accepted: Mapping[str, Any],
        prediction: Mapping[str, Any],
        handoff: Mapping[str, Any],
        slot: RuntimeLaneToken,
    ) -> Mapping[str, Any]:
        slot = self.require_slot(
            slot, task_index=key.task_index, allow_global=True
        )
        grader_config = self.official_grader_config()
        max_attempts = self.section("grader")["max_attempts"]
        inspected = []
        removed = []
        for attempt in range(1, max_attempts + 1):
            request = self.official_request(
                key=key,
                accepted=accepted,
                prediction=prediction,
                handoff=handoff,
                grader_attempt=attempt,
            )
            attempt_root = grade_attempt_directory(grader_config, request)
            if not attempt_root.exists():
                continue
            prediction_path = attempt_root / "prediction.jsonl"
            command = grader_command(
                grader_config, request, prediction_path=prediction_path
            )
            started_path = attempt_root / "started.json"
            launching_path = attempt_root / "launching.json"
            process_result_path = attempt_root / "process-result.json"
            process_receipt: dict[str, Any] | None = None
            aggregate_path, _, _, _ = expected_raw_paths(attempt_root, request)
            run_root = (
                attempt_root
                / "logs"
                / "run_evaluation"
                / grader_run_id(request)
            )
            matching = find_matching_grader_process(command)
            if any(
                (
                    started_path.exists(),
                    launching_path.exists(),
                    process_result_path.exists(),
                    aggregate_path.exists(),
                    run_root.exists(),
                    matching is not None,
                )
            ):
                try:
                    recovered = run_official_grader(grader_config, request)
                except RetryableGraderError as error:
                    process_receipt = {
                        "status": "retryable",
                        "failure_class": error.failure_class,
                        "process_result": process_result_path.exists(),
                    }
                else:
                    process_receipt = {
                        "status": "official_outcome_recovered",
                        "resolved": recovered["resolved"],
                        "process_result": process_result_path.exists(),
                    }
            for container_id in self.grader_container_ids(request):
                removed.append(self.remove_grader_container(container_id, request))
            inspected.append(
                {
                    "attempt": attempt,
                    "run_id": grader_run_id(request),
                    "process": process_receipt,
                }
            )
        return {
            "schema": "swebench_triad_grader_reconciliation_v1",
            "cell": key.to_payload(),
            "inspected_attempts": inspected,
            "removed_containers": removed,
            "slot_index": slot.slot_index,
            "server_port": slot.server_port,
            "lane_generation": slot.generation,
        }

    def reconcile_startup(
        self,
        *,
        task_indices: Sequence[int],
        allow_foreign_loaded_images: bool = False,
        slots: Sequence[RuntimeLaneToken],
    ) -> Mapping[str, Any]:
        if type(allow_foreign_loaded_images) is not bool:
            raise TypeError("foreign loaded-image policy must be boolean")
        slot_tokens = tuple(slots)
        if (
            len(slot_tokens) != self.config.task_slots_per_replica
            or tuple(token.slot_index for token in slot_tokens)
            != tuple(range(self.config.task_slots_per_replica))
        ):
            raise ValueError("startup reconciliation slot lattice drifted")
        for token in slot_tokens:
            self.require_slot(token, task_index=None)
        selected = tuple(task_indices)
        if (
            not selected
            or tuple(sorted(set(selected))) != selected
            or any(task_index not in self.by_task for task_index in selected)
        ):
            raise ValueError("startup reconciliation task shard is invalid")
        reconciled_graders = []
        evicted_images = []
        removed_task_roots = []
        selected_set = set(selected)
        staged = set(self.staged_task_indices())
        roots = set(self.task_root_indices())

        for container_id in self.owned_container_ids():
            record = self.container_record(container_id)
            container_config = object_value(record.get("Config"), "container config")
            labels = object_value(container_config.get("Labels"), "container labels")
            raw_index = labels.get("amg.task_index")
            if not isinstance(raw_index, str) or not raw_index.isdigit():
                raise RuntimeError("owned container task identity is invalid")
            if int(raw_index) in selected_set:
                raise RuntimeError("leased task containers require cell reconciliation")

        cgroups = self.cgroup_paths()
        selected_cgroups = [
            name for name in cgroups if int(name.split("-", 1)[0]) in selected_set
        ]
        if selected_cgroups or any(
            pid > 0 for pid in self.cgroup_process_ids() if selected_cgroups
        ):
            raise RuntimeError("leased task cgroups require cell reconciliation")

        pod_root = path_value(
            self.section("runtime")["pod_local_root"], "pod-local root"
        )
        for mount in self.mount_records_under(pod_root):
            mount_path = Path(text_value(mount.get("mount_point"), "mount point"))
            try:
                relative = mount_path.relative_to(pod_root / "tasks")
            except ValueError:
                raise RuntimeError("owned mount escaped the task-root lattice")
            if not relative.parts:
                raise RuntimeError("owned task parent itself is mounted")
            match = re.fullmatch(r"task-([0-9]{4})", relative.parts[0])
            if match is None:
                raise RuntimeError("owned mount task identity is invalid")
            if int(match.group(1)) in selected_set:
                raise RuntimeError("leased task mounts require cell reconciliation")

        loaded = set(self.loaded_task_image_identities())
        loaded_by_id = {image_id: (image, image_id) for image, image_id in loaded}
        if len(loaded_by_id) != len(loaded):
            raise RuntimeError("loaded task image ID has multiple census identities")
        retired = set(self.retired_task_indices())
        loaded_by_task: dict[int, tuple[str, str]] = {}
        loaded_claims: dict[str, list[tuple[int, tuple[str, str]]]] = {}
        for task_index in staged - retired:
            identity = self.task_image_identity(task_index)
            if identity is not None and identity[1] in loaded_by_id:
                loaded_claims.setdefault(identity[1], []).append(
                    (task_index, identity)
                )
        for claims in loaded_claims.values():
            if len(claims) != 1:
                raise RuntimeError(
                    "loaded task image maps to multiple active task leases"
                )
            task_index, identity = claims[0]
            loaded_by_task[task_index] = identity
        bound_image_ids = {identity[1] for identity in loaded_by_task.values()}
        foreign_loaded_images = {
            loaded_by_id[image_id]
            for image_id in set(loaded_by_id) - bound_image_ids
        }
        if foreign_loaded_images and not allow_foreign_loaded_images:
            raise RuntimeError("loaded task image has no durable task-stage binding")

        active_tasks = roots | set(loaded_by_task)
        for task_index in sorted(active_tasks & selected_set):
            identity = loaded_by_task.get(task_index)
            if identity is not None:
                evicted_images.append(self.evict_image(*identity))
            if self.remove_inactive_task_root(task_index):
                removed_task_roots.append(task_index)
        return {
            "schema": "swebench_triad_startup_reconciliation_v1",
            "task_indices": list(selected),
            "reconciled_graders": reconciled_graders,
            "evicted_images": evicted_images,
            "removed_task_roots": removed_task_roots,
            "foreign_staged_tasks": sorted(active_tasks - selected_set),
            "foreign_loaded_images": [
                {"image": image, "image_id": image_id}
                for image, image_id in sorted(foreign_loaded_images)
            ],
            "residue": self.global_residue_snapshot(),
            "slots": [
                {
                    "slot_index": token.slot_index,
                    "server_port": token.server_port,
                    "lane_generation": token.generation,
                }
                for token in slot_tokens
            ],
        }

    def grade(
        self,
        *,
        key: CellKey,
        accepted: Mapping[str, Any],
        prediction: Mapping[str, Any],
        handoff: Mapping[str, Any],
        slot: RuntimeLaneToken,
    ) -> Mapping[str, Any]:
        if not isinstance(key, CellKey):
            raise TypeError("official grading requires a CellKey")
        slot = self.require_slot(slot, task_index=key.task_index)
        grader = self.section("grader")
        grader_config = self.official_grader_config()
        last_error: RetryableGraderError | None = None
        for attempt in range(1, grader["max_attempts"] + 1):
            request = self.official_request(
                key=key,
                accepted=accepted,
                prediction=prediction,
                handoff=handoff,
                grader_attempt=attempt,
            )
            try:
                return run_official_grader(grader_config, request)
            except RetryableGraderError as error:
                last_error = error
                self.reconcile_grade(
                    key=key,
                    accepted=accepted,
                    prediction=prediction,
                    handoff=handoff,
                    slot=slot,
                )
        assert last_error is not None
        raise last_error

    def stage_receipt_path(self, task_index: int) -> Path:
        if type(task_index) is not int or task_index not in self.by_task:
            raise ValueError("task index is outside the production manifest")
        return (
            self.config.run_root
            / "control"
            / "stages"
            / f"task-{task_index:04d}.json"
        )

    def stage_receipt(self, task_index: int) -> Mapping[str, Any] | None:
        path = self.stage_receipt_path(task_index)
        if not path.exists():
            return None
        value = read_json(path)
        if (
            not isinstance(value, Mapping)
            or value.get("schema") != "swebench_triad_task_stage_v1"
            or value.get("task_index") != task_index
        ):
            raise RuntimeError("task stage receipt is invalid")
        return value

    def eviction_receipt_path(self, task_index: int) -> Path:
        if type(task_index) is not int or task_index not in self.by_task:
            raise ValueError("task index is outside the production manifest")
        return (
            self.config.run_root
            / "control"
            / "evictions"
            / f"task-{task_index:04d}.json"
        )

    def eviction_receipt(self, task_index: int) -> Mapping[str, Any] | None:
        path = self.eviction_receipt_path(task_index)
        if not path.exists():
            return None
        value = read_json(path)
        fields = {
            "schema",
            "task_index",
            "instance_id",
            "readiness",
            "image",
            "task_root_removed",
            "certified_blobs_retained",
            "repository_mirror_retained",
            "slot_index",
            "server_port",
            "lane_generation",
        }
        expected_instance = self.by_task[task_index][0].task.task_id
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value.get("schema") != "swebench_triad_task_eviction_v1"
            or value.get("task_index") != task_index
            or value.get("instance_id") != expected_instance
            or not isinstance(value.get("readiness"), Mapping)
            or not isinstance(value.get("image"), Mapping)
            or type(value.get("task_root_removed")) is not bool
            or value.get("certified_blobs_retained") is not True
            or value.get("repository_mirror_retained") is not True
            or value.get("slot_index") != self.task_slots.get(task_index)
            or value.get("server_port")
            != self.config.server_port(value.get("slot_index"))
            or type(value.get("lane_generation")) is not int
            or value["lane_generation"] <= 0
        ):
            raise RuntimeError("task eviction receipt is invalid")
        return value

    def task_image_identity(self, task_index: int) -> tuple[str, str] | None:
        receipt = self.stage_receipt(task_index)
        if receipt is None:
            return None
        binding = object_value(receipt.get("binding"), "staged OCI binding")
        image = text_value(binding.get("image"), "staged image")
        config_digest = text_value(
            binding.get("config_digest"), "staged config digest"
        )
        if re.fullmatch(r"sha256:[0-9a-f]{64}", config_digest) is None:
            raise RuntimeError("staged Docker config digest drifted")
        return image, config_digest

    def image_container_ids(self, image: str) -> list[str]:
        result = self.docker().run(
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            f"ancestor={image}",
        )
        if result.returncode != 0:
            raise RuntimeError("task-image container census failed")
        return sorted(
            row
            for row in self.docker().output_text(result.stdout).splitlines()
            if row
        )

    def audit_residue(
        self,
        task_index: int,
        *,
        slot: RuntimeLaneToken,
    ) -> Mapping[str, Any]:
        if type(task_index) is not int or task_index not in self.by_task:
            raise ValueError("task index is outside the production manifest")
        slot = self.require_slot(slot, task_index=task_index)
        server_containers = self.owned_container_ids(
            task_index, slot_index=slot.slot_index
        )
        image = self.task_image_identity(task_index)
        image_containers = [] if image is None else self.image_container_ids(image[0])
        containers = sorted(set(server_containers) | set(image_containers))
        cgroups = self.cgroup_paths(task_index)
        processes = self.cgroup_process_ids(task_index)
        mounts = self.mount_records_under(self.task_root_path(task_index))
        tmpfs_mounts = [row for row in mounts if row["fs_type"] == "tmpfs"]
        rootfs = self.verify_task_rootfs(task_index)
        return {
            "task_index": task_index,
            "active_slots": 0 if not processes and not server_containers else 1,
            "active_workspaces": 0 if not mounts and not processes else 1,
            "containers": len(containers),
            "processes": len(processes),
            "cgroups": len(cgroups),
            "tmpfs_mounts": len(tmpfs_mounts),
            "mounts": len(mounts),
            "rootfs_attested": rootfs is not None,
            "slot_index": slot.slot_index,
            "server_port": slot.server_port,
            "lane_generation": slot.generation,
        }

    def task_state_rows(
        self, task_index: int
    ) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
        accepted = []
        outcomes = []
        for arm in ARMS:
            slug = f"{task_index:04d}-{arm}"
            accepted_value = read_json(
                self.config.run_root / "state" / "accepted" / f"{slug}.json"
            )
            outcome_value = read_json(
                self.config.run_root / "state" / "outcomes" / f"{slug}.json"
            )
            if not isinstance(accepted_value, Mapping) or not isinstance(
                outcome_value, Mapping
            ):
                raise RuntimeError("task state row is invalid")
            accepted.append(accepted_value)
            outcomes.append(outcome_value)
        return accepted, outcomes

    def evict_image(self, image: str, config_digest: str) -> dict[str, Any]:
        certified = self.certified_image_identities()
        aliases = sorted(
            alias for alias, digest in certified.items() if digest == config_digest
        )
        if image not in aliases or not aliases:
            raise RuntimeError("refusing to evict an uncertified task image")
        inspected_alias = self.docker().inspect(image)
        if inspected_alias.returncode == 0:
            alias_id = self.docker().output_text(inspected_alias.stdout).strip()
            if alias_id != config_digest:
                raise RuntimeError("refusing to evict a mismatched task image")
        elif not self.docker().is_missing_image(inspected_alias):
            raise RuntimeError("Docker image inspection failed during eviction")

        inspected_id = self.docker().inspect(config_digest)
        if self.docker().is_missing_image(inspected_id):
            if inspected_alias.returncode == 0:
                raise RuntimeError("task image alias survived without its image ID")
            return {
                "schema": "swebench_verified_docker_eviction_v1",
                "status": "already_absent",
                "image": image,
                "image_id": config_digest,
            }
        image_id = self.require_docker_result(
            inspected_id, "task image ID inspection"
        )
        if image_id != config_digest:
            raise RuntimeError("task image ID drifted before eviction")
        if self.image_container_ids(config_digest):
            raise RuntimeError("task image ID still has containers")

        removed = self.docker().run("image", "rm", "--force", config_digest)
        self.require_docker_result(removed, "task image ID eviction")
        if not self.docker().is_missing_image(self.docker().inspect(config_digest)):
            raise RuntimeError("task image ID survived eviction")
        for alias in aliases:
            if not self.docker().is_missing_image(self.docker().inspect(alias)):
                raise RuntimeError("certified task image alias survived eviction")
        return {
            "schema": "swebench_verified_docker_eviction_v1",
            "status": "evicted",
            "image": image,
            "image_id": config_digest,
            "certified_aliases_removed": aliases,
        }

    def evict_task(
        self,
        task_index: int,
        stage: Any,
        *,
        slot: RuntimeLaneToken,
    ) -> Mapping[str, Any]:
        slot = self.require_slot(slot, task_index=task_index)
        if stage is not None:
            if not isinstance(stage, ProductionTaskStage):
                raise TypeError("task eviction stage has the wrong type")
            if stage.task_index != task_index:
                raise ValueError("task eviction stage identity drifted")
        instance_id = self.by_task[task_index][0].task.task_id
        accepted, outcomes = self.task_state_rows(task_index)
        adapted = accepted_rows_for_eviction(task_index, instance_id, accepted)
        readiness = require_task_eviction_ready(
            instance_id,
            adapted,
            outcomes,
        )
        residue = self.audit_residue(task_index, slot=slot)
        if any(
            residue[name] != 0
            for name in (
                "active_slots",
                "active_workspaces",
                "containers",
                "processes",
                "cgroups",
                "tmpfs_mounts",
                "mounts",
            )
        ):
            raise RuntimeError("task runtime residue forbids eviction")
        identity = self.task_image_identity(task_index)
        if identity is None:
            image_eviction = {"status": "never_staged"}
        else:
            image_eviction = self.evict_image(*identity)
        task_root_removed = self.remove_inactive_task_root(task_index)
        receipt = {
            "schema": "swebench_triad_task_eviction_v1",
            "task_index": task_index,
            "instance_id": instance_id,
            "readiness": readiness,
            "image": image_eviction,
            "task_root_removed": task_root_removed,
            "certified_blobs_retained": True,
            "repository_mirror_retained": True,
            "slot_index": slot.slot_index,
            "server_port": slot.server_port,
            "lane_generation": slot.generation,
        }
        atomic_write_json(
            self.config.run_root
            / "control"
            / "evictions"
            / f"task-{task_index:04d}.json",
            receipt,
        )
        return receipt

    def remove_owned_container_id(self, container_id: str) -> dict[str, Any]:
        value = self.container_record(container_id)
        name = str(value.get("Name", "")).lstrip("/")
        config = object_value(value.get("Config"), "container config")
        labels = config.get("Labels")
        labels = labels if isinstance(labels, Mapping) else {}
        match = re.fullmatch(
            re.escape(CONTAINER_NAME_PREFIX)
            + r"([0-9]{4})-(native|amg_compaction_only|amg_memory)-g([0-9]{8})",
            name,
        )
        expected_amg_labels = {
            "amg.owner": OWNER_LABEL,
            "amg.task_index": match.group(1) if match is not None else None,
            "amg.arm": match.group(2) if match is not None else None,
            "amg.generation": match.group(3) if match is not None else None,
        }
        if match is None or any(
            labels.get(name) != expected
            for name, expected in expected_amg_labels.items()
        ):
            raise RuntimeError("refusing to remove an unowned task container")
        slot_index = labels.get("amg.slot_index")
        server_port = labels.get("amg.server_port")
        lane_generation = labels.get("amg.lane_generation")
        if (
            not isinstance(slot_index, str)
            or not slot_index.isdigit()
            or int(slot_index) >= self.config.task_slots_per_replica
            or server_port != str(self.config.server_port(int(slot_index)))
            or not isinstance(lane_generation, str)
            or not lane_generation.isdigit()
            or int(lane_generation) <= 0
        ):
            raise RuntimeError("refusing to remove a slot-unbound task container")
        unexpected_amg_labels = {
            str(name)
            for name in labels
            if str(name).startswith("amg.")
        } - {
            *expected_amg_labels,
            "amg.slot_index",
            "amg.server_port",
            "amg.lane_generation",
        }
        if unexpected_amg_labels:
            raise RuntimeError("refusing to remove a task container with unknown ownership labels")
        removed = self.docker().run(
            "container", "rm", "--force", container_id
        )
        self.require_docker_result(removed, "owned container removal")
        return {"container_id": container_id, "name": name, "removed": True}

    def staged_task_indices(self) -> list[int]:
        root = self.config.run_root / "control" / "stages"
        if not root.exists():
            return []
        indices = []
        for path in sorted(root.glob("task-*.json")):
            match = re.fullmatch(r"task-([0-9]{4})\.json", path.name)
            if match is None:
                raise RuntimeError("stage receipt directory contains an unknown file")
            index = int(match.group(1))
            self.stage_receipt(index)
            indices.append(index)
        return indices

    def retired_task_indices(self) -> list[int]:
        root = self.config.run_root / "control" / "evictions"
        if not root.exists():
            return []
        indices = []
        for path in sorted(root.glob("task-*.json")):
            match = re.fullmatch(r"task-([0-9]{4})\.json", path.name)
            if match is None:
                raise RuntimeError(
                    "eviction receipt directory contains an unknown file"
                )
            index = int(match.group(1))
            self.eviction_receipt(index)
            indices.append(index)
        return indices

    @staticmethod
    def process_command_line(pid: int) -> str:
        try:
            payload = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError as error:
            raise RuntimeError("process command line is unavailable") from error
        command = payload.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if not command:
            raise RuntimeError("process command line is empty")
        return command

    @staticmethod
    def process_group_id(pid: int) -> int:
        try:
            value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError("process group is unavailable") from error
        close = value.rfind(")")
        fields = value[close + 2 :].split() if close >= 0 else []
        if len(fields) < 3 or not fields[2].isdigit():
            raise RuntimeError("process group stat is malformed")
        return int(fields[2])

    @staticmethod
    def process_state(pid: int) -> str:
        try:
            value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError("process state is unavailable") from error
        close = value.rfind(")")
        fields = value[close + 2 :].split() if close >= 0 else []
        if not fields or len(fields[0]) != 1:
            raise RuntimeError("process state stat is malformed")
        return fields[0]

    def direct_process_children(self, parent_pid: int) -> set[int]:
        children: set[int] = set()
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            pid = int(path.name)
            try:
                if (
                    self.process_state(pid) != "Z"
                    and self.process_parent_pid(pid) == parent_pid
                ):
                    children.add(pid)
            except RuntimeError:
                continue
        return children

    def process_descendants(self, parent_pid: int) -> set[int]:
        result: set[int] = set()
        frontier = [parent_pid]
        while frontier:
            parent = frontier.pop()
            for child in self.direct_process_children(parent):
                if child in result or child == parent_pid:
                    continue
                result.add(child)
                frontier.append(child)
        return result

    def process_group_members(self, pgid: int) -> set[int]:
        members: set[int] = set()
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            pid = int(path.name)
            try:
                if (
                    self.process_state(pid) != "Z"
                    and self.process_group_id(pid) == pgid
                ):
                    members.add(pid)
            except RuntimeError:
                continue
        return members

    @staticmethod
    def gpu_compute_pids() -> set[int]:
        return {
            int(line.strip())
            for line in command_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                label="GPU process census",
            ).splitlines()
            if line.strip().isdigit()
        }

    def endpoint_is_alive(self, base_url: str) -> bool:
        url = text_value(base_url, "model endpoint").rstrip("/") + "/models"
        try:
            with urllib_request.urlopen(url, timeout=2.0) as response:
                return 200 <= response.status < 500
        except (OSError, urllib_error.URLError):
            return False

    def matching_model_processes(self) -> set[int]:
        serving = self.section("serving")
        model_root = str(
            path_value(self.section("assets")["model_root"], "model root")
        )
        model_id = str(serving["model_id"])
        matches = set()
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            pid = int(path.name)
            try:
                if self.process_state(pid) == "Z":
                    continue
                command = self.process_command_line(pid)
            except RuntimeError:
                continue
            if model_root in command or model_id in command:
                matches.add(pid)
        return matches

    def model_process_tree_snapshot(self) -> dict[str, Any]:
        serving = self.section("serving")
        pid = serving["pid"]
        if self.process_state(pid) == "Z":
            raise RuntimeError("model server process is a zombie")
        if linux_process_start_ticks(pid) != serving["start_ticks"]:
            raise RuntimeError("model server PID identity drifted")
        try:
            pgid = os.getpgid(pid)
        except OSError as error:
            raise RuntimeError("model server process group is unavailable") from error
        if pgid != pid:
            raise RuntimeError("model server is not the leader of its owned process group")
        group = self.process_group_members(pgid)
        descendants = self.process_descendants(pid)
        if pid not in group or not descendants.issubset(group):
            raise RuntimeError("model descendants escaped the owned process group")
        matching = self.matching_model_processes()
        if not matching.issubset(group):
            raise RuntimeError("model process escaped the owned process group")
        processes = {
            str(member): linux_process_start_ticks(member) for member in sorted(group)
        }
        gpu_pids = self.gpu_compute_pids() & group
        if not gpu_pids:
            raise RuntimeError("model process group has no GPU-resident worker")
        return {
            "pgid": pgid,
            "processes": processes,
            "gpu_pids": sorted(gpu_pids),
        }

    def model_shutdown_residue(
        self,
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        pgid = snapshot.get("pgid")
        if type(pgid) is not int or pgid <= 0:
            raise RuntimeError("model process-group snapshot is invalid")
        processes = object_value(snapshot.get("processes"), "model process snapshot")
        for raw_pid, raw_ticks in processes.items():
            if not isinstance(raw_pid, str) or not raw_pid.isdigit():
                raise RuntimeError("model process snapshot PID is invalid")
            if type(raw_ticks) is not int or raw_ticks <= 0:
                raise RuntimeError("model process snapshot start time is invalid")
        captured_gpu = snapshot.get("gpu_pids")
        if not isinstance(captured_gpu, list) or any(
            type(pid) is not int or pid <= 0 for pid in captured_gpu
        ):
            raise RuntimeError("model GPU process snapshot is invalid")
        live_group = self.process_group_members(pgid)
        live_gpu = sorted(live_group & self.gpu_compute_pids())
        live_matching = sorted(self.matching_model_processes())
        endpoint_alive = self.endpoint_is_alive(
            text_value(self.section("serving")["base_url"], "serving base URL")
        )
        return {
            "live_processes": sorted(live_group),
            "live_gpu_pids": live_gpu,
            "live_matching_processes": live_matching,
            "endpoint_alive": endpoint_alive,
        }

    @staticmethod
    def require_model_shutdown(residue: Mapping[str, Any]) -> None:
        if (
            residue.get("live_processes")
            or residue.get("live_gpu_pids")
            or residue.get("live_matching_processes")
            or residue.get("endpoint_alive") is not False
        ):
            raise RuntimeError("owned model process tree did not fully stop")

    def stop_model_process(
        self,
        *,
        timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        if timeout_seconds < 0:
            raise ValueError("model shutdown timeout cannot be negative")
        serving = self.section("serving")
        pid = serving["pid"]
        try:
            if self.process_state(pid) == "Z":
                raise RuntimeError("model server leader is a zombie")
            current_ticks = linux_process_start_ticks(pid)
        except RuntimeError:
            snapshot = {
                "pgid": pid,
                "processes": {},
                "gpu_pids": [],
            }
            residue = self.model_shutdown_residue(snapshot)
            if not any(
                (
                    residue["live_processes"],
                    residue["live_gpu_pids"],
                    residue["live_matching_processes"],
                    residue["endpoint_alive"],
                )
            ):
                return {
                    "pid": pid,
                    "status": "already_stopped",
                    "residue": residue,
                }
        else:
            if current_ticks != serving["start_ticks"]:
                raise RuntimeError("refusing to signal a reused model PID")
            command = require_process_identity(
                pid, serving["start_ticks"], "model server"
            )
            model_root = str(
                path_value(self.section("assets")["model_root"], "model root")
            )
            if model_root not in command or str(serving["model_id"]) not in command:
                raise RuntimeError("refusing to signal a mismatched model process")
            snapshot = self.model_process_tree_snapshot()
        pgid = snapshot["pgid"]
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout_seconds
        residue = self.model_shutdown_residue(snapshot)
        while any(
            (
                residue["live_processes"],
                residue["live_gpu_pids"],
                residue["live_matching_processes"],
                residue["endpoint_alive"],
            )
        ) and time.monotonic() < deadline:
            time.sleep(0.25)
            residue = self.model_shutdown_residue(snapshot)
        escalated = False
        if any(
            (
                residue["live_processes"],
                residue["live_gpu_pids"],
                residue["live_matching_processes"],
                residue["endpoint_alive"],
            )
        ):
            escalated = True
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            kill_deadline = time.monotonic() + min(10.0, timeout_seconds)
            residue = self.model_shutdown_residue(snapshot)
            while any(
                (
                    residue["live_processes"],
                    residue["live_gpu_pids"],
                    residue["live_matching_processes"],
                    residue["endpoint_alive"],
                )
            ) and time.monotonic() < kill_deadline:
                time.sleep(0.25)
                residue = self.model_shutdown_residue(snapshot)
        self.require_model_shutdown(residue)
        return {
            "pid": pid,
            "pgid": pgid,
            "status": "stopped",
            "signal": "SIGKILL" if escalated else "SIGTERM",
            "processes_before": snapshot["processes"],
            "gpu_pids_before": snapshot["gpu_pids"],
            "residue": residue,
        }

    @staticmethod
    def process_parent_pid(pid: int) -> int:
        try:
            value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError("holder process is unavailable") from error
        close = value.rfind(")")
        fields = value[close + 2 :].split() if close >= 0 else []
        if len(fields) < 2 or not fields[1].isdigit():
            raise RuntimeError("holder process stat is malformed")
        return int(fields[1])

    @staticmethod
    def process_cpu_ticks(pid: int) -> int:
        try:
            value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError("holder worker is unavailable") from error
        close = value.rfind(")")
        fields = value[close + 2 :].split() if close >= 0 else []
        if len(fields) <= 12 or not fields[11].isdigit() or not fields[12].isdigit():
            raise RuntimeError("holder worker stat is malformed")
        return int(fields[11]) + int(fields[12])

    @staticmethod
    def gpu_utilization_sample() -> dict[int, int]:
        rows = command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            label="holder GPU utilization sample",
        ).splitlines()
        result: dict[int, int] = {}
        for row in rows:
            fields = [field.strip() for field in row.split(",")]
            if (
                len(fields) != 2
                or not fields[0].isdigit()
                or not fields[1].isdigit()
            ):
                raise RuntimeError("holder GPU utilization row is malformed")
            index = int(fields[0])
            utilization = int(fields[1])
            if index in result or not 0 <= utilization <= 100:
                raise RuntimeError("holder GPU utilization sample is invalid")
            result[index] = utilization
        if not result:
            raise RuntimeError("holder GPU utilization sample is empty")
        return result

    @staticmethod
    def cpu_capacity_count() -> int:
        affinity = getattr(os, "sched_getaffinity", None)
        count = len(affinity(0)) if affinity is not None else (os.cpu_count() or 0)
        if count <= 0:
            raise RuntimeError("holder CPU capacity is unavailable")
        return count

    @staticmethod
    def require_holder_retention(
        cpu_utilization: Sequence[float],
        gpu_samples: Sequence[Mapping[int, int]],
        expected_gpu_indices: set[int],
    ) -> None:
        if not cpu_utilization or (
            min(cpu_utilization) < HOLDER_RETENTION_FLOOR_PERCENT
            or sum(cpu_utilization) / len(cpu_utilization)
            < HOLDER_TARGET_PERCENT
        ):
            raise RuntimeError("holder CPU utilization is below the retention floor")
        required_samples = max(1, (len(gpu_samples) + 1) // 2)
        for index in expected_gpu_indices:
            values = [sample[index] for sample in gpu_samples]
            if (
                sum(
                    value >= HOLDER_RETENTION_FLOOR_PERCENT
                    for value in values
                )
                < required_samples
                or sum(values) / len(values) < HOLDER_TARGET_PERCENT
            ):
                raise RuntimeError(
                    "holder GPU utilization is below the retention floor"
                )

    def holder_snapshot(
        self,
        *,
        auto_state_path: Path = Path("/tmp/crg-holder.state"),
        fallback_state_path: Path = Path(
            "/tmp/non-yield-gpu-cpu-fallback-holder.state.json"
        ),
        sample_count: int = 3,
        sample_gap: float = 1.0,
    ) -> dict[str, Any]:
        if type(sample_count) is not int or sample_count <= 0:
            raise ValueError("holder sample count must be positive")
        if sample_gap < 0:
            raise ValueError("holder sample gap cannot be negative")
        auto_path = Path(auto_state_path)
        fallback_path = Path(fallback_state_path)
        try:
            auto_info = auto_path.lstat()
        except OSError as error:
            raise RuntimeError("auto-yield holder state is unavailable") from error
        if auto_path.is_symlink() or not stat.S_ISREG(auto_info.st_mode):
            raise RuntimeError("auto-yield holder state is not a regular file")
        if abs(time.time() - auto_info.st_mtime) > 90:
            raise RuntimeError("auto-yield holder state is stale")
        try:
            auto_text = auto_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise RuntimeError("auto-yield holder state is unavailable") from error
        auto_match = re.fullmatch(
            r"([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) "
            r"mode=hold pid=([0-9]+) gpu=([0-9]+) cpu=([0-9]+) work=([^ ]+)",
            auto_text,
        )
        if auto_match is None:
            raise RuntimeError("auto-yield holder did not return to HOLD")
        try:
            auto_timestamp = time.mktime(
                time.strptime(auto_match.group(1), "%Y-%m-%d %H:%M:%S")
            )
        except (OverflowError, ValueError) as error:
            raise RuntimeError("auto-yield holder timestamp is invalid") from error
        if abs(time.time() - auto_timestamp) > 90:
            raise RuntimeError("auto-yield holder state is stale")
        auto_pid = int(auto_match.group(2))
        auto_gpu = int(auto_match.group(3))
        auto_cpu = int(auto_match.group(4))
        if auto_gpu != 1:
            raise RuntimeError("auto-yield holder GPU cardinality drifted")
        if auto_cpu <= 0:
            raise RuntimeError("auto-yield holder CPU cardinality drifted")
        auto_ticks = linux_process_start_ticks(auto_pid)
        auto_command = require_process_identity(
            auto_pid, auto_ticks, "auto-yield holder"
        )
        if not any(
            name in auto_command
            for name in (
                "_heavy_holder.py",
                "gpu_cpu_auto_yield_holder.py",
                "platform_gpu_only_auto_yield_holder.py",
            )
        ):
            raise RuntimeError("auto-yield holder process identity drifted")

        fallback = read_json_object(fallback_path, "fallback-holder state")
        exact_fields(
            fallback,
            {
                "timestamp",
                "parent_pid",
                "cpu_workers",
                "gpu_workers",
                "cpu_duty",
                "gpu_duty",
                "mode",
            },
            "fallback-holder state",
        )
        if fallback.get("mode") != "hold":
            raise RuntimeError("fallback holder is not in HOLD")
        timestamp = fallback.get("timestamp")
        if type(timestamp) is not int or abs(time.time() - timestamp) > 120:
            raise RuntimeError("fallback holder state is stale")
        parent = fallback.get("parent_pid")
        cpu_workers = fallback.get("cpu_workers")
        gpu_workers = fallback.get("gpu_workers")
        cpu_duty = fallback.get("cpu_duty")
        gpu_duty = fallback.get("gpu_duty")
        for duty, label in ((cpu_duty, "CPU"), (gpu_duty, "GPU")):
            if (
                isinstance(duty, bool)
                or not isinstance(duty, (int, float))
                or not 0.0 <= float(duty) <= 1.0
                or float(duty) * 100.0 < HOLDER_TARGET_PERCENT
            ):
                raise RuntimeError(
                    f"fallback holder {label} duty is below the retention floor"
                )
        if (
            type(parent) is not int
            or parent <= 0
            or not isinstance(cpu_workers, Mapping)
            or not cpu_workers
            or not isinstance(gpu_workers, Mapping)
            or set(gpu_workers) != {"0"}
        ):
            raise RuntimeError("fallback holder worker lattice drifted")
        parent_ticks = linux_process_start_ticks(parent)
        parent_command = require_process_identity(
            parent, parent_ticks, "fallback holder"
        )
        if "non_yield_gpu_cpu_fallback_holder.py" not in parent_command:
            raise RuntimeError("fallback holder process identity drifted")
        if set(cpu_workers) != {str(index) for index in range(len(cpu_workers))}:
            raise RuntimeError("fallback CPU holder worker lattice drifted")
        worker_pids = [*cpu_workers.values(), *gpu_workers.values()]
        if any(type(pid) is not int or pid <= 0 for pid in worker_pids):
            raise RuntimeError("fallback holder worker PID is invalid")
        if len(set(worker_pids)) != len(worker_pids):
            raise RuntimeError("fallback holder workers are not unique")

        auto_children = self.direct_process_children(auto_pid)
        fallback_children = self.direct_process_children(parent)
        if auto_children & fallback_children or auto_pid == parent:
            raise RuntimeError("holder process layers overlap")
        declared_fallback = set(worker_pids)
        if not declared_fallback.issubset(fallback_children):
            raise RuntimeError("fallback holder worker ancestry drifted")

        def resource_trackers(children: set[int]) -> set[int]:
            result = set()
            for pid in children:
                command = self.process_command_line(pid)
                if "multiprocessing.resource_tracker" in command:
                    result.add(pid)
                elif not (
                    "multiprocessing.spawn" in command
                    or "--multiprocessing-fork" in command
                ):
                    raise RuntimeError("holder worker command identity drifted")
            return result

        auto_trackers = resource_trackers(auto_children)
        fallback_trackers = resource_trackers(fallback_children)
        if fallback_children - declared_fallback - fallback_trackers:
            raise RuntimeError("fallback holder has an unknown child process")

        gpu_pids = self.gpu_compute_pids()
        fallback_gpu_pids = set(gpu_workers.values())
        fallback_cpu_pids = set(cpu_workers.values())
        if gpu_pids & fallback_children != fallback_gpu_pids:
            raise RuntimeError("fallback GPU holder worker ownership drifted")
        if fallback_cpu_pids & gpu_pids:
            raise RuntimeError("fallback CPU worker unexpectedly owns a GPU")
        auto_gpu_pids = (auto_children - auto_trackers) & gpu_pids
        auto_cpu_pids = auto_children - auto_trackers - auto_gpu_pids
        if len(auto_gpu_pids) != auto_gpu or len(auto_cpu_pids) != auto_cpu:
            raise RuntimeError("auto-yield holder worker lattice drifted")
        expected_gpu_pids = auto_gpu_pids | fallback_gpu_pids
        unknown_gpu_pids = gpu_pids - expected_gpu_pids
        if unknown_gpu_pids:
            raise RuntimeError(
                "unknown GPU holder process survived restoration: "
                + ",".join(str(pid) for pid in sorted(unknown_gpu_pids))
            )

        worker_ticks = {}
        all_workers = auto_gpu_pids | auto_cpu_pids | declared_fallback
        for pid in sorted(all_workers):
            worker_ticks[str(pid)] = linux_process_start_ticks(pid)
            expected_parent = auto_pid if pid in auto_children else parent
            if self.process_parent_pid(pid) != expected_parent:
                raise RuntimeError("holder worker ancestry drifted")

        cpu_pids = sorted(auto_cpu_pids | fallback_cpu_pids)
        cpu_samples = [
            {str(pid): self.process_cpu_ticks(pid) for pid in cpu_pids}
        ]
        cpu_sample_times = [time.monotonic()]
        gpu_samples = []
        for index in range(sample_count):
            gpu_samples.append(self.gpu_utilization_sample())
            if index + 1 < sample_count:
                time.sleep(sample_gap)
                cpu_samples.append(
                    {str(pid): self.process_cpu_ticks(pid) for pid in cpu_pids}
                )
                cpu_sample_times.append(time.monotonic())
        for before, after in zip(cpu_samples, cpu_samples[1:]):
            if any(after[pid] <= before[pid] for pid in before):
                raise RuntimeError("holder CPU worker did not sustain forward work")
        if len(cpu_samples) < 2:
            raise RuntimeError("holder CPU retention floor needs multiple samples")
        clock_ticks = os.sysconf("SC_CLK_TCK")
        if type(clock_ticks) is not int or clock_ticks <= 0:
            raise RuntimeError("holder CPU clock frequency is invalid")
        cpu_capacity = self.cpu_capacity_count()
        cpu_utilization = []
        for before, after, before_time, after_time in zip(
            cpu_samples,
            cpu_samples[1:],
            cpu_sample_times,
            cpu_sample_times[1:],
        ):
            elapsed = after_time - before_time
            if elapsed <= 0:
                raise RuntimeError("holder CPU sample clock did not advance")
            tick_delta = sum(after[pid] - before[pid] for pid in before)
            utilization = (
                100.0 * tick_delta / (clock_ticks * elapsed * cpu_capacity)
            )
            cpu_utilization.append(utilization)
        expected_gpu_indices = set(range(auto_gpu))
        if any(set(sample) != expected_gpu_indices for sample in gpu_samples):
            raise RuntimeError("holder GPU utilization index lattice drifted")
        self.require_holder_retention(
            cpu_utilization,
            gpu_samples,
            expected_gpu_indices,
        )
        return {
            "schema": "swebench_triad_holder_snapshot_v2",
            "auto": {
                "pid": auto_pid,
                "start_ticks": auto_ticks,
                "gpu_workers": auto_gpu,
                "cpu_workers": auto_cpu,
                "gpu_worker_pids": sorted(auto_gpu_pids),
                "cpu_worker_pids": sorted(auto_cpu_pids),
                "resource_tracker_pids": sorted(auto_trackers),
                "mode": "hold",
                "state_mtime_ns": auto_info.st_mtime_ns,
            },
            "fallback": {
                "parent_pid": parent,
                "parent_start_ticks": parent_ticks,
                "gpu_workers": dict(gpu_workers),
                "cpu_workers": dict(cpu_workers),
                "worker_start_ticks": worker_ticks,
                "resource_tracker_pids": sorted(fallback_trackers),
                "mode": "hold",
            },
            "gpu_process_pids": sorted(gpu_pids),
            "cpu_tick_samples": cpu_samples,
            "cpu_sample_times": cpu_sample_times,
            "cpu_capacity": cpu_capacity,
            "cpu_utilization_percent": cpu_utilization,
            "gpu_utilization_samples": gpu_samples,
            "retention_floor_percent": HOLDER_RETENTION_FLOOR_PERCENT,
            "target_percent": HOLDER_TARGET_PERCENT,
        }

    def restore_holders(self) -> dict[str, Any]:
        removed_markers = []
        for marker in (
            Path("/tmp/crg-holder-yield"),
            Path("/tmp/agentmemory-formal-cpu-active"),
        ):
            if not marker.exists():
                continue
            info = marker.lstat()
            if marker.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
                raise RuntimeError("holder marker identity is unsafe")
            value = marker.read_text(encoding="utf-8", errors="strict")
            if OWNER_LABEL not in value:
                raise RuntimeError("refusing to remove another owner's holder marker")
            marker.unlink()
            removed_markers.append(str(marker))
        deadline = time.monotonic() + 90.0
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                snapshot = self.holder_snapshot()
                return {
                    "schema": "swebench_triad_holder_restore_v1",
                    "status": "PASS",
                    "removed_markers": removed_markers,
                    "snapshot": snapshot,
                }
            except (OSError, RuntimeError, ValueError) as error:
                last_error = error
                time.sleep(1.0)
        raise RuntimeError("GPU+CPU holders did not restore") from last_error

    def command_ledger_audit(self) -> dict[str, Any]:
        ledger_path = self.config.run_root / "full" / "command-exit-ledger.jsonl"
        if not ledger_path.is_file() or ledger_path.is_symlink():
            raise RuntimeError("command-exit ledger is unavailable")
        events = []
        event_ids = set()
        for line in ledger_path.read_bytes().splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError("command-exit ledger is malformed") from error
            if not isinstance(value, Mapping) or canonical_json_bytes(value) != line:
                raise RuntimeError("command-exit ledger row is noncanonical")
            event_id = value.get("event_id")
            if not isinstance(event_id, str) or event_id in event_ids:
                raise RuntimeError("command-exit ledger event identity drifted")
            event_ids.add(event_id)
            events.append(value)

        expected: dict[tuple[int, str, int], tuple[str, str]] = {}
        accepted_root = self.config.run_root / "state" / "accepted"
        for accepted_path in accepted_root.glob("*.json"):
            accepted = read_json_object(accepted_path, "accepted cell")
            cell = object_value(accepted.get("cell"), "accepted cell identity")
            task_index = cell.get("task_index")
            arm = cell.get("arm")
            generation = accepted.get("attempt_generation")
            if (
                type(task_index) is not int
                or task_index not in self.by_task
                or arm not in ARMS
                or type(generation) is not int
                or generation <= 0
            ):
                raise RuntimeError("accepted command-ledger identity is invalid")
            key = (task_index, arm, generation)
            if key in expected:
                raise RuntimeError("accepted command-ledger identity is duplicated")
            expected[key] = (
                text_value(accepted.get("instance_id"), "accepted instance ID"),
                sha256_value(
                    accepted.get("prediction_sha256"), "accepted prediction"
                ),
            )

        common_fields = {
            "schema",
            "binding_sha256",
            "task_index",
            "arm",
            "generation",
            "grader_attempt",
            "prediction_sha256",
            "event_id",
            "event",
        }
        grouped: dict[str, dict[str, tuple[int, Mapping[str, Any]]]] = {}
        for position, row in enumerate(events):
            event = row.get("event")
            event_fields = {
                "start": {"command", "cwd", "environment"},
                "exit": {"process_result"},
                "abandoned": {"reason"},
            }
            if event not in event_fields:
                raise RuntimeError("command-exit ledger event kind is invalid")
            allowed = common_fields | event_fields[event]
            if event == "abandoned" and {"pid", "start_ticks"}.issubset(row):
                allowed |= {"pid", "start_ticks"}
            if set(row) != allowed:
                raise RuntimeError("command-exit ledger event fields drifted")
            if row.get("schema") != "swebench_triad_command_exit_event_v1":
                raise RuntimeError("command-exit ledger schema drifted")
            binding = sha256_value(
                row.get("binding_sha256"), "command binding SHA-256"
            )
            prediction_sha256 = sha256_value(
                row.get("prediction_sha256"), "command prediction SHA-256"
            )
            task_index = row.get("task_index")
            arm = row.get("arm")
            generation = row.get("generation")
            grader_attempt = row.get("grader_attempt")
            if (
                type(task_index) is not int
                or arm not in ARMS
                or type(generation) is not int
                or generation <= 0
                or type(grader_attempt) is not int
                or grader_attempt <= 0
            ):
                raise RuntimeError("command-exit ledger cell binding is invalid")
            accepted_key = (task_index, arm, generation)
            accepted_binding = expected.get(accepted_key)
            if accepted_binding is None or accepted_binding[1] != prediction_sha256:
                raise RuntimeError("command-exit ledger accepted-cell binding drifted")
            expected_binding = sha256_json(
                {
                    "schema": "swebench_triad_grader_binding_v1",
                    "task_index": task_index,
                    "arm": arm,
                    "generation": generation,
                    "grader_attempt": grader_attempt,
                    "instance_id": accepted_binding[0],
                    "prediction_sha256": prediction_sha256,
                    "harness_commit": HARNESS_COMMIT,
                    "harness_tree": HARNESS_TREE,
                    "dataset_sha256": PRODUCTION_DATASET_PINS.jsonl_sha256,
                    "namespace": "swebench",
                    "timeout_seconds": 1_800,
                }
            )
            if binding != expected_binding:
                raise RuntimeError("command-exit ledger grader binding digest drifted")
            if row["event_id"] != binding + ":" + event:
                raise RuntimeError("command-exit ledger event ID drifted")
            by_event = grouped.setdefault(binding, {})
            if event in by_event:
                raise RuntimeError("command-exit ledger binding event is duplicated")
            by_event[event] = (position, row)

            if event == "start":
                command = row["command"]
                environment = row["environment"]
                if (
                    not isinstance(command, list)
                    or not command
                    or any(not isinstance(value, str) or not value for value in command)
                    or not isinstance(row["cwd"], str)
                    or not Path(row["cwd"]).is_absolute()
                    or not isinstance(environment, Mapping)
                    or any(
                        not isinstance(name, str) or not isinstance(value, str)
                        for name, value in environment.items()
                    )
                ):
                    raise RuntimeError("command-exit ledger start event is invalid")
                grader_config = self.section("grader")
                assets = self.section("assets")
                docker = self.section("docker")
                attempt_root = (
                    path_value(grader_config["output_root"], "grader output root")
                    / f"{task_index:04d}-{arm}"
                    / f"generation-{generation:08d}"
                    / f"attempt-{grader_attempt:06d}-{binding}"
                )
                run_id = (
                    f"amg-sbv-{task_index:04d}-{arm}-g{generation:08d}"
                    f"-a{grader_attempt:06d}-{binding[:16]}"
                )
                expected_command = [
                    str(path_value(grader_config["python_executable"], "grader Python")),
                    "-m",
                    "swebench.harness.run_evaluation",
                    "--dataset_name",
                    str(path_value(assets["dataset_jsonl"], "dataset JSONL")),
                    "--split",
                    "test",
                    "--instance_ids",
                    accepted_binding[0],
                    "--predictions_path",
                    str(attempt_root / "prediction.jsonl"),
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
                    run_id,
                ]
                expected_environment = {
                    "PYTHONPATH": str(
                        path_value(assets["harness_root"], "harness root")
                    ),
                    "PYTHONNOUSERSITE": "1",
                    "DOCKER_HOST": "unix://"
                    + str(path_value(docker["socket"], "Docker socket")),
                    "HF_DATASETS_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
                if (
                    command != expected_command
                    or row["cwd"] != str(attempt_root)
                    or dict(environment) != expected_environment
                ):
                    raise RuntimeError("command-exit ledger canonical invocation drifted")
            elif event == "exit":
                result = object_value(row["process_result"], "command process result")
                status = result.get("status")
                if status == "completed":
                    exact_fields(
                        result,
                        {
                            "schema",
                            "status",
                            "returncode",
                            "stdout_sha256",
                            "stderr_sha256",
                        },
                        "completed command process result",
                    )
                    if type(result.get("returncode")) is not int:
                        raise RuntimeError("completed command return code is invalid")
                    sha256_value(result.get("stdout_sha256"), "command stdout")
                    sha256_value(result.get("stderr_sha256"), "command stderr")
                elif status == "process_timeout":
                    exact_fields(
                        result,
                        {
                            "schema",
                            "status",
                            "returncode",
                            "stdout_sha256",
                            "stderr_sha256",
                        },
                        "timed-out command process result",
                    )
                    if result.get("returncode") is not None:
                        raise RuntimeError("timed-out command return code is invalid")
                    sha256_value(result.get("stdout_sha256"), "command stdout")
                    sha256_value(result.get("stderr_sha256"), "command stderr")
                elif status in {"spawn_error", "post_spawn_error"}:
                    exact_fields(
                        result,
                        {"schema", "status", "returncode", "error_class"},
                        "failed command process result",
                    )
                    if result.get("returncode") is not None or not isinstance(
                        result.get("error_class"), str
                    ):
                        raise RuntimeError("failed command process result is invalid")
                else:
                    raise RuntimeError("command process result status is invalid")
                if result.get("schema") != "swebench_triad_grader_process_v1":
                    raise RuntimeError("command process result schema drifted")
            else:
                if not isinstance(row.get("reason"), str) or not row["reason"]:
                    raise RuntimeError("abandoned command reason is invalid")
                if "pid" in row and (
                    type(row["pid"]) is not int
                    or row["pid"] <= 0
                    or type(row["start_ticks"]) is not int
                    or row["start_ticks"] <= 0
                ):
                    raise RuntimeError("abandoned command process identity is invalid")

        covered: set[tuple[int, str, int]] = set()
        for by_event in grouped.values():
            start = by_event.get("start")
            terminals = [
                by_event[name] for name in ("exit", "abandoned") if name in by_event
            ]
            if start is None or len(terminals) != 1 or start[0] >= terminals[0][0]:
                raise RuntimeError("command-exit ledger start/terminal pairing drifted")
            terminal = terminals[0][1]
            common = (
                "task_index",
                "arm",
                "generation",
                "grader_attempt",
                "prediction_sha256",
            )
            if any(start[1].get(name) != terminal.get(name) for name in common):
                raise RuntimeError("command-exit ledger pair binding drifted")
            if terminal["event"] == "exit":
                result = terminal["process_result"]
                if result["status"] == "completed" and result["returncode"] == 0:
                    covered.add(
                        (
                            terminal["task_index"],
                            terminal["arm"],
                            terminal["generation"],
                        )
                    )
        if not set(expected).issubset(covered):
            raise RuntimeError("command-exit ledger does not cover every accepted cell")
        return {
            "schema": "swebench_triad_command_ledger_audit_v1",
            "status": "PASS",
            "events": len(events),
            "covered_cells": len(expected),
            "sha256": sha256_file(ledger_path),
        }

    def final_audit(self) -> Mapping[str, Any]:
        residue = self.global_residue_snapshot()
        if any(residue.values()):
            raise RuntimeError("final owned runtime residue is nonzero")
        shared_pool = None
        if self.config.shared_model_pool is not None:
            shared_pool = dict(
                validate_shared_model_pool_snapshot(
                    self.shared_model_pool_snapshot(require_preflight_binding=True),
                    "final audit shared model pool snapshot",
                )
            )
        pod = self.pod_snapshot()
        allocation_retained = (
            pod["job"] == self.section("pod")["job"]
            and pod["boot_id"] == self.section("pod")["boot_id"]
        )
        if not allocation_retained:
            raise RuntimeError("final allocation identity drifted")
        return {
            "schema": "swebench_triad_final_runtime_audit_v1",
            "status": "PASS",
            "residue": residue,
            "command_ledger": self.command_ledger_audit(),
            "allocation_retained": allocation_retained,
            "shared_model_pool": shared_pool,
        }

    def cleanup(self, *, slots: Sequence[RuntimeLaneToken]) -> Mapping[str, Any]:
        if self.config.shared_model_pool is not None:
            raise RuntimeError(
                "shared-model pool cleanup requires the eight-replica coordinator"
            )
        removed_containers = []
        container_ids = set(self.owned_container_ids())
        for container_id in sorted(container_ids):
            removed_containers.append(
                self.remove_owned_container_id(container_id)
            )

        backend = self.cgroup_backend()
        removed_cgroups = []
        for cell_name in self.cgroup_paths():
            relative_path = f"{CGROUP_RELATIVE_PREFIX}/{cell_name}"
            receipt = backend.remove(relative_path)
            if receipt.get("removed") is not True:
                raise RuntimeError("owned cgroup cleanup did not remove its path")
            removed_cgroups.append(cell_name)

        startup_reconciliation = self.reconcile_startup(
            task_indices=tuple(sorted(self.by_task)),
            slots=slots,
        )
        model = self.stop_model_process()
        holders = self.restore_holders()
        pod = self.pod_snapshot()
        residue = self.global_residue_snapshot()
        owned_residue = sum(residue.values())
        receipt = {
            "schema": "swebench_triad_owned_cleanup_v1",
            "owned_residue": owned_residue,
            "allocation_retained": (
                pod["job"] == self.section("pod")["job"]
                and pod["boot_id"] == self.section("pod")["boot_id"]
            ),
            "removed_containers": removed_containers,
            "removed_cgroups": removed_cgroups,
            "startup_reconciliation": startup_reconciliation,
            "model_process": model,
            "holders": holders,
            "holders_restored": True,
            "external_pool_retained": False,
            "residue": residue,
            "docker_daemon_retained": True,
            "certified_blobs_retained": True,
            "repository_mirrors_retained": True,
        }
        if owned_residue != 0 or receipt["allocation_retained"] is not True:
            raise RuntimeError("owned cleanup or allocation-retention proof failed")
        return receipt


class ProductionLifecycleOperations:
    """Host-specific operations are implemented below in bounded primitives."""

    def __init__(
        self,
        config: ProductionRunConfig,
        configs: Sequence[RunConfig],
        *,
        runtime: ProductionRuntime | None = None,
    ) -> None:
        if not isinstance(config, ProductionRunConfig):
            raise TypeError("production operations require ProductionRunConfig")
        if tuple(configs) != config.configs:
            raise ValueError("production operations received another manifest")
        self.config = config
        self.configs = tuple(configs)
        self.runtime = runtime or LinuxProductionRuntime(config, self.configs)

    def preflight(self) -> Mapping[str, Any]:
        return self.runtime.preflight()

    def stage_task(self, task_index: int, *, slot: RuntimeLaneToken) -> Any:
        return self.runtime.stage_task(task_index, slot=slot)

    def reconcile_cell(
        self,
        config: RunConfig,
        *,
        generation: int,
        before_preflight: bool,
        slot: RuntimeLaneToken,
    ) -> Mapping[str, Any]:
        return self.runtime.reconcile_cell(
            config,
            generation=generation,
            before_preflight=before_preflight,
            slot=slot,
        )

    def reconcile_grade(self, **kwargs: Any) -> Mapping[str, Any]:
        return self.runtime.reconcile_grade(**kwargs)

    def reconcile_startup(
        self,
        *,
        task_indices: Sequence[int],
        allow_foreign_loaded_images: bool = False,
        slots: Sequence[RuntimeLaneToken],
    ) -> Mapping[str, Any]:
        if allow_foreign_loaded_images:
            return self.runtime.reconcile_startup(
                task_indices=task_indices,
                allow_foreign_loaded_images=True,
                slots=slots,
            )
        return self.runtime.reconcile_startup(
            task_indices=task_indices, slots=slots
        )

    def reconcile_unbound_loaded_images(self) -> Mapping[str, Any]:
        return self.runtime.reconcile_unbound_loaded_images()

    def run_cell(
        self,
        config: RunConfig,
        stage: Any,
        *,
        generation: int,
        slot: RuntimeLaneToken,
    ) -> Mapping[str, Any]:
        return self.runtime.run_cell(
            config, stage, generation=generation, slot=slot
        )

    def grade(self, **kwargs: Any) -> Mapping[str, Any]:
        return self.runtime.grade(**kwargs)

    def audit_residue(
        self, task_index: int, *, slot: RuntimeLaneToken
    ) -> Mapping[str, Any]:
        return self.runtime.audit_residue(task_index, slot=slot)

    def evict_task(
        self,
        task_index: int,
        stage: Any,
        *,
        slot: RuntimeLaneToken,
    ) -> Mapping[str, Any]:
        return self.runtime.evict_task(task_index, stage, slot=slot)

    def cleanup(self, *, slots: Sequence[RuntimeLaneToken]) -> Mapping[str, Any]:
        return self.runtime.cleanup(slots=slots)

    def final_audit(self) -> Mapping[str, Any]:
        return self.runtime.final_audit()

    def timing_identity(self) -> Mapping[str, Any]:
        source = self.config.section("source")
        pod = self.config.section("pod")
        shared = self.config.shared_model_pool
        return {
            "deployment_commit": source["deployment_commit"],
            "inner_commit": source["inner_commit"],
            "source_identity_sha256": sha256_json(source),
            "run_config_sha256": sha256_file(self.config.path),
            "manifest_sha256": self.config.payload["manifest_sha256"],
            "replica_index": (
                shared["replica_index"] if shared is not None else 0
            ),
            "gpu_uuid": pod["gpu_uuid"],
        }


OperationsFactory = Callable[
    [ProductionRunConfig, Sequence[RunConfig]], ProductionLifecycleOperations
]


__all__ = [
    "LinuxProductionRuntime",
    "ProductionLifecycleOperations",
    "ProductionRunConfig",
    "ProductionTaskStage",
    "accepted_rows_for_eviction",
    "current_owner_identity",
    "linux_process_start_ticks",
    "owner_is_alive",
    "shared_model_pool_snapshot_receipt",
    "summarize_task4_receipt",
]
