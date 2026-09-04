#!/usr/bin/env python3
"""Own the private grader and public FastAPI process for one exact run."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, TextIO
from urllib.request import urlopen

PROCESS_ROLES = ("private-grader", "public-environment")
PRIVATE_SOCKET_ROOT = Path("/run/openmle-fast")
LIMIT_ENVIRONMENT = {
    "max_policy_actions": "OPENMLE_FAST_MAX_POLICY_TURNS",
    "cpu_vcpus": "OPENMLE_FAST_CPU_VCPUS",
    "memory_bytes": "OPENMLE_FAST_MEMORY_BYTES",
    "swap_bytes": "OPENMLE_FAST_SWAP_BYTES",
    "workspace_bytes": "OPENMLE_FAST_WORKSPACE_BYTES",
    "tmp_bytes": "OPENMLE_FAST_TMP_BYTES",
    "max_processes": "OPENMLE_FAST_MAX_PROCESSES",
    "max_open_files": "OPENMLE_FAST_MAX_OPEN_FILES",
    "max_files": "OPENMLE_FAST_MAX_FILES",
    "max_file_bytes": "OPENMLE_FAST_MAX_FILE_BYTES",
    "max_submission_bytes": "OPENMLE_FAST_MAX_SUBMISSION_BYTES",
    "shell_wall_ms": "OPENMLE_FAST_SHELL_WALL_MS",
    "managed_runtime_per_action_ms": "OPENMLE_FAST_MANAGED_ACTION_MS",
    "managed_runtime_per_episode_ms": "OPENMLE_FAST_MANAGED_EPISODE_MS",
    "episode_wall_ms": "OPENMLE_FAST_EPISODE_WALL_MS",
    "grader_cpu_vcpus": "OPENMLE_FAST_GRADER_CPU_VCPUS",
    "grader_memory_bytes": "OPENMLE_FAST_GRADER_MEMORY_BYTES",
    "grader_max_processes": "OPENMLE_FAST_GRADER_MAX_PROCESSES",
    "grader_worker_wall_ms": "OPENMLE_FAST_GRADER_WORKER_WALL_MS",
    "grader_total_wall_ms": "OPENMLE_FAST_GRADER_TOTAL_WALL_MS",
    "grader_max_concurrent_requests": "OPENMLE_FAST_GRADER_MAX_CONCURRENT_REQUESTS",
    "grader_input_bytes": "OPENMLE_FAST_GRADER_INPUT_BYTES",
    "raw_output_bytes": "OPENMLE_FAST_RAW_OUTPUT_BYTES",
    "observation_bytes": "OPENMLE_FAST_OBSERVATION_BYTES",
    "observation_head_bytes": "OPENMLE_FAST_OBSERVATION_HEAD_BYTES",
    "observation_tail_bytes": "OPENMLE_FAST_OBSERVATION_TAIL_BYTES",
}


def load_contract_module(path: Path):
    spec = importlib.util.spec_from_file_location("openmle_launcher_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load launcher contract module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_parent_death_signal() -> None:
    if sys.platform != "linux":
        raise RuntimeError("endpoint parent-death guard requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def sanitized_environment() -> Dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("OPENMLE_FAST_") and "PRIVATE" not in key
    }


def write_credential(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    credential = secrets.token_hex(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (credential + "\n").encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return credential


def write_forbidden_canaries(
    path: Path,
    credential: str,
    private_environment: Mapping[str, str],
) -> None:
    canaries = [
        credential,
        private_environment["OPENMLE_FAST_PRIVATE_TASK_MANIFEST"],
        private_environment["OPENMLE_FAST_PRIVATE_PACKAGE_ROOT"],
        private_environment["OPENMLE_FAST_PRIVATE_RUNNER"],
        private_environment["OPENMLE_FAST_GRADER_CREDENTIAL"],
    ]
    if len(set(canaries)) != len(canaries) or any(not value for value in canaries):
        raise RuntimeError("private canaries are empty or duplicated")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = (json.dumps(canaries, sort_keys=True) + "\n").encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def common_environment(document: Mapping[str, Any], pythonpath: str) -> Dict[str, str]:
    environment = sanitized_environment()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": pythonpath,
        }
    )
    for field, variable in LIMIT_ENVIRONMENT.items():
        environment[variable] = str(document["resource_limits"][field])
    return environment


def run_identity_digest(owner: str, run_id: str) -> str:
    return hashlib.sha256(f"{owner}\0{run_id}".encode("utf-8")).hexdigest()


def private_socket_path(owner: str, run_id: str) -> Path:
    path = PRIVATE_SOCKET_ROOT / run_identity_digest(owner, run_id) / "grader.sock"
    if len(os.fsencode(path)) > 107:
        raise RuntimeError("private grader socket exceeds Linux AF_UNIX path limit")
    return path


def require_private_directory(path: Path, label: str) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise RuntimeError(f"{label} is not an exact owner-only directory: {path}")


def prepare_private_socket_path(path: Path) -> None:
    if path.parent.parent != PRIVATE_SOCKET_ROOT or path.name != "grader.sock":
        raise RuntimeError("private grader socket escaped the managed runtime root")
    PRIVATE_SOCKET_ROOT.mkdir(mode=0o700, parents=False, exist_ok=True)
    require_private_directory(PRIVATE_SOCKET_ROOT, "private socket root")
    path.parent.mkdir(mode=0o700, parents=False, exist_ok=False)
    path.parent.chmod(0o700)
    require_private_directory(path.parent, "private socket run directory")


def cleanup_private_socket_path(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISSOCK(metadata.st_mode):
            raise RuntimeError(f"refusing to unlink non-socket grader endpoint: {path}")
        path.unlink()
    try:
        path.parent.rmdir()
    except FileNotFoundError:
        pass


def read_attested_public_workspace_parent(public_runner: Path) -> Path:
    completed = subprocess.run(
        [str(public_runner), "metadata"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    metadata = json.loads(completed.stdout)
    value = metadata.get("workspace_parent")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise RuntimeError("public runner metadata lacks an absolute workspace parent")
    parent = Path(value)
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("attested public workspace parent is not a real directory")
    return parent.resolve()


def public_episodes_path(parent: Path, owner: str, run_id: str) -> Path:
    return parent / f"service-{run_identity_digest(owner, run_id)}"


def prepare_public_episodes_path(path: Path, parent: Path) -> None:
    parent = parent.resolve()
    if path.parent.resolve() != parent or not path.name.startswith("service-"):
        raise RuntimeError("public episodes root escaped the attested workspace parent")
    path.mkdir(mode=0o700, parents=False, exist_ok=False)
    path.chmod(0o700)
    require_private_directory(path, "public episodes run directory")


def cleanup_public_episodes_path(path: Path) -> None:
    try:
        path.rmdir()
    except FileNotFoundError:
        pass

def build_service_environments(
    document: Mapping[str, Any],
    contract_name: str,
    run_dir: Path,
    outer_root: Path,
    inner_root: Path,
    port: int,
    owner: str,
    run_id: str,
) -> tuple[Dict[str, str], Dict[str, str]]:
    integration = document["integration"]
    exact = document["exact_runtime"]
    source = document["runtime_source"]
    role = document["launch_contracts"][contract_name]["manifest_role"]
    manifest = integration["manifests"][role]
    runtime_root = Path(integration["pod_root"])
    exact_root = Path(exact["install_root"])
    public_runner = exact_root / exact["public_runner"]["relpath"]
    workspace_parent = read_attested_public_workspace_parent(public_runner)
    service_root = run_dir / "endpoints"
    private_root = service_root / "private"
    public_root = service_root / "public"
    socket_path = private_socket_path(owner, run_id)
    credential_path = private_root / "credential"
    pythonpath_parts = (
        inner_root / "agentenv-openmle-fast",
        inner_root / "agentenv",
        outer_root / "AgentGym-RL",
    )
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = ":".join(str(path) for path in pythonpath_parts)
    if inherited_pythonpath:
        pythonpath = f"{pythonpath}:{inherited_pythonpath}"
    base = common_environment(document, pythonpath)
    identity = {
        "OPENMLE_FAST_PROCESS_OWNER": owner,
        "OPENMLE_FAST_RUN_ID": run_id,
    }

    private_environment = dict(base)
    private_environment.update(identity)
    private_environment.update(
        {
            "OPENMLE_FAST_PRIVATE_TASK_MANIFEST": str(
                runtime_root / integration["private_manifest"]["relpath"]
            ),
            "OPENMLE_FAST_PRIVATE_TASK_MANIFEST_SHA256": integration[
                "private_manifest"
            ]["sha256"],
            "OPENMLE_FAST_PRIVATE_PACKAGE_ROOT": str(
                Path(integration["source_root"]) / "private-grader"
            ),
            "OPENMLE_FAST_PRIVATE_ARCHIVE_ROOT": integration["source_root"],
            "OPENMLE_FAST_RELEASE_REVISION": source["openmle_tasks_revision"],
            "OPENMLE_FAST_PRIVATE_RUNTIME_DIGEST": exact["runtime_digest"],
            "OPENMLE_FAST_PRIVATE_RUNNER": str(
                exact_root / exact["private_runner"]["relpath"]
            ),
            "OPENMLE_FAST_PRIVATE_RUNNER_SHA256": exact["private_runner"][
                "expected_sha256"
            ],
            "OPENMLE_FAST_RUNTIME_ARTIFACT_LOCK_SHA256": exact["artifact_lock"][
                "expected_sha256"
            ],
            "OPENMLE_FAST_GRADER_ENDPOINT": str(socket_path),
            "OPENMLE_FAST_GRADER_CREDENTIAL": str(credential_path),
            "OPENMLE_FAST_PRIVATE_AUDIT_ROOT": str(private_root / "audit"),
            "OPENMLE_FAST_PRIVATE_CPU_VCPUS": str(
                document["resource_limits"]["grader_cpu_vcpus"]
            ),
            "OPENMLE_FAST_PRIVATE_MEMORY_BYTES": str(
                document["resource_limits"]["grader_memory_bytes"]
            ),
            "OPENMLE_FAST_PRIVATE_MAX_PROCESSES": str(
                document["resource_limits"]["grader_max_processes"]
            ),
            "OPENMLE_FAST_PRIVATE_WORKER_WALL_MS": str(
                document["resource_limits"]["grader_worker_wall_ms"]
            ),
            "OPENMLE_FAST_PRIVATE_TOTAL_WALL_MS": str(
                document["resource_limits"]["grader_total_wall_ms"]
            ),
            "OPENMLE_FAST_PRIVATE_MAX_CONCURRENT_REQUESTS": str(
                document["resource_limits"]["grader_max_concurrent_requests"]
            ),
        }
    )

    public_environment = dict(base)
    public_environment.update(identity)
    public_environment.update(
        {
            "OPENMLE_FAST_TASK_MANIFEST": str(runtime_root / manifest["relpath"]),
            "OPENMLE_FAST_TASK_MANIFEST_SHA256": manifest["sha256"],
            "OPENMLE_FAST_PACKAGE_ROOT": integration["source_root"],
            "OPENMLE_FAST_ARCHIVE_ROOT": integration["source_root"],
            "OPENMLE_FAST_EPISODES_ROOT": str(
                public_episodes_path(workspace_parent, owner, run_id)
            ),
            "OPENMLE_FAST_RELEASE_REVISION": source["openmle_tasks_revision"],
            "OPENMLE_FAST_MANIFEST_ROLE": role,
            "OPENMLE_FAST_MATERIALIZER_SHA256": source["selected_files"][
                "inner:agentenv-openmle-fast/agentenv_openmle_fast/materializer.py"
            ],
            "OPENMLE_FAST_ACTIONS_SHA256": source["selected_files"][
                "inner:agentenv-openmle-fast/agentenv_openmle_fast/actions.py"
            ],
            "OPENMLE_FAST_EXECUTOR_RUNNER": str(
                exact_root / exact["public_runner"]["relpath"]
            ),
            "OPENMLE_FAST_EXECUTOR_RUNNER_SHA256": exact["public_runner"][
                "expected_sha256"
            ],
            "OPENMLE_FAST_EXECUTOR_RUNTIME_DIGEST": exact["runtime_digest"],
            "OPENMLE_FAST_RUNTIME_ARTIFACT_LOCK_SHA256": exact["artifact_lock"][
                "expected_sha256"
            ],
            "OPENMLE_FAST_GRADER_CLIENT_TIMEOUT_SECONDS": "15",
            "OPENMLE_FAST_GRADER_TIMEOUT_MARGIN_SECONDS": "2",
            "OPENMLE_FAST_CLIENT_TIMEOUT_SECONDS": "200",
            "OPENMLE_FAST_CLIENT_TIMEOUT_MARGIN_SECONDS": "5",
            "OPENMLE_FAST_GRADER_ENDPOINT": str(socket_path),
            "OPENMLE_FAST_GRADER_CREDENTIAL": str(credential_path),
            "OPENMLE_FAST_AUDIT_ROOT": str(public_root / "audit"),
            "OPENMLE_FAST_RUNTIME_OUTER_COMMIT": source["outer_commit"],
            "OPENMLE_FAST_RUNTIME_INNER_COMMIT": source["inner_commit"],
            "OPENMLE_FAST_MAX_OBSERVATION_TOKENS": "8192",
            "OPENMLE_FAST_HOST": "127.0.0.1",
            "OPENMLE_FAST_PORT": str(port),
            "OPENMLE_FAST_LOG_LEVEL": "info",
        }
    )
    if any("PRIVATE" in key for key in public_environment):
        raise RuntimeError("public service environment contains a private-root variable")
    return private_environment, public_environment


def process_record(
    role: str,
    process: subprocess.Popen,
    module,
    owner: str,
    run_id: str,
) -> Dict[str, Any]:
    start_ticks = module.process_start_ticks(process.pid)
    if start_ticks is None:
        raise RuntimeError(f"{role} exited before process identity was captured")
    return {
        "role": role,
        "pid": process.pid,
        "start_ticks": start_ticks,
        "process_owner": owner,
        "run_id": run_id,
    }


def spawn_service(
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
) -> tuple[subprocess.Popen, TextIO]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=dict(environment),
        start_new_session=True,
        preexec_fn=set_parent_death_signal,
        text=True,
    )
    return process, log_handle


def start_private_grader(
    command: Sequence[str],
    environment: Mapping[str, str],
    run_dir: Path,
    module,
    owner: str,
    run_id: str,
) -> tuple[subprocess.Popen, TextIO, Dict[str, Any]]:
    socket_path = Path(environment["OPENMLE_FAST_GRADER_ENDPOINT"])
    prepare_private_socket_path(socket_path)
    process, log_handle = spawn_service(
        command, environment, run_dir / "endpoints/private-grader.log"
    )
    try:
        record = process_record("private-grader", process, module, owner, run_id)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("private grader exited before its socket was ready")
            try:
                mode = socket_path.stat().st_mode
            except FileNotFoundError:
                time.sleep(0.2)
                continue
            if stat.S_ISSOCK(mode) and stat.S_IMODE(mode) == 0o600:
                return process, log_handle, record
            raise RuntimeError(
                "private grader endpoint is not the required mode-0600 socket"
            )
        raise TimeoutError("private grader did not become ready")
    except Exception:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        log_handle.close()
        raise


def start_public_endpoint(
    command: Sequence[str],
    environment: Mapping[str, str],
    run_dir: Path,
    module,
    owner: str,
    run_id: str,
    expected_role: str,
    expected_manifest_sha256: str,
    expected_task_count: int,
) -> tuple[subprocess.Popen, TextIO, Dict[str, Any], Dict[str, Any]]:
    episodes_root = Path(environment["OPENMLE_FAST_EPISODES_ROOT"])
    workspace_parent = read_attested_public_workspace_parent(
        Path(environment["OPENMLE_FAST_EXECUTOR_RUNNER"])
    )
    prepare_public_episodes_path(episodes_root, workspace_parent)
    process, log_handle = spawn_service(
        command, environment, run_dir / "endpoints/public-environment.log"
    )
    try:
        record = process_record("public-environment", process, module, owner, run_id)
        url = f"http://127.0.0.1:{environment['OPENMLE_FAST_PORT']}/metadata"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("public endpoint exited before metadata was ready")
            try:
                with urlopen(url, timeout=2) as response:
                    metadata = json.loads(response.read())
            except (OSError, TimeoutError, json.JSONDecodeError):
                time.sleep(0.2)
                continue
            expected = {
                "schema": "openmle_fast_public_metadata_v1",
                "role": expected_role,
                "task_count": expected_task_count,
                "task_manifest_sha256": expected_manifest_sha256,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise RuntimeError(
                        f"public metadata {key} drifted: "
                        f"{metadata.get(key)!r} != {value!r}"
                    )
            coverage = metadata.get("executor_coverage")
            if (
                not isinstance(coverage, Mapping)
                or coverage.get("formal_eligible") is not True
            ):
                raise RuntimeError(
                    "public executor coverage is not formal eligible: "
                    f"{coverage!r}"
                )
            return process, log_handle, record, metadata
        raise TimeoutError("public endpoint did not become ready")
    except Exception:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        log_handle.close()
        raise


def stop_exact_process(
    process: subprocess.Popen,
    record: Mapping[str, Any],
    module,
) -> tuple[int, bool]:
    termination_requested = False
    if module.exact_process_alive(record["pid"], record["start_ticks"]):
        os.killpg(process.pid, signal.SIGTERM)
        termination_requested = True
    try:
        return process.wait(timeout=30), termination_requested
    except subprocess.TimeoutExpired:
        if module.exact_process_alive(record["pid"], record["start_ticks"]):
            os.killpg(process.pid, signal.SIGKILL)
        return process.wait(timeout=10), termination_requested


def managed_exit_is_clean(exit_code: int, termination_requested: bool) -> bool:
    return exit_code == 0 or (
        termination_requested and exit_code == -int(signal.SIGTERM)
    )
