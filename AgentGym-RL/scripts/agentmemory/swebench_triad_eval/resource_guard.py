"""Deployment-owned cgroup, tmpfs, and immutable-rootfs guards.

The published SWE environment remains task neutral.  This module supplies the
outer lifecycle controls that are specific to the formal external evaluation:
one aggregate cgroup-v1 envelope per server container, hard tmpfs quotas for
episode state, and full-tree rootfs re-attestation before every policy command.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Protocol, Union


class ResourceGuardError(RuntimeError):
    """Raised when a formal resource boundary cannot be proven."""


CGROUP_RUNTIME_PARENT = "amg-external-eval-container-runtime-v1"
CGROUP_EVALUATION_PARENT = "swebench-triad-v1"
CGROUP_RELATIVE_PREFIX = (
    f"{CGROUP_RUNTIME_PARENT}/{CGROUP_EVALUATION_PARENT}"
)
CGROUP_STRUCTURE_LOCK_PATH = Path(
    "/tmp/amg-swebench-triad-cgroup-v1-structure.lock"
)
_CELL_NAME_RE = re.compile(r"\A[0-9]{4}-(?:native|amg_compaction_only|amg_memory)\Z")
_PAGE_BYTES = 4096


@contextmanager
def cgroup_structure_lock():
    """Serialize cgroup-v1 two-controller structure changes and censuses."""

    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(CGROUP_STRUCTURE_LOCK_PATH, flags, 0o600)
    except OSError as error:
        raise ResourceGuardError("cannot open the cgroup structure lock") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ResourceGuardError("cgroup structure lock is not a regular file")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _integer_counter(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourceGuardError(f"{label} is not a non-negative integer")
    return value


def _pid_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ResourceGuardError(f"{label} is not a list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ResourceGuardError(f"{label} contains an invalid PID")
        result.append(item)
    if result != sorted(set(result)):
        raise ResourceGuardError(f"{label} is not sorted and unique")
    return result


@dataclass(frozen=True)
class CgroupV1Limits:
    memory_bytes: int
    max_processes: int

    def __post_init__(self) -> None:
        memory = _positive_integer(self.memory_bytes, "cgroup memory limit")
        _positive_integer(self.max_processes, "cgroup process limit")
        if memory % _PAGE_BYTES:
            raise ValueError("cgroup memory limit must be page aligned")


class CgroupV1Backend(Protocol):
    def prepare(
        self,
        relative_path: str,
        limits: CgroupV1Limits,
    ) -> Mapping[str, Any]: ...

    def snapshot(self, relative_path: str) -> Mapping[str, Any]: ...

    def remove(self, relative_path: str) -> Mapping[str, Any]: ...


class CgroupV1CellEnvelope:
    """Stateful aggregate memory+pids envelope for one server container."""

    def __init__(
        self,
        *,
        cell_name: str,
        limits: CgroupV1Limits,
        backend: CgroupV1Backend,
    ) -> None:
        if not isinstance(cell_name, str) or _CELL_NAME_RE.fullmatch(cell_name) is None:
            raise ValueError("cgroup cell name is not canonical")
        if not isinstance(limits, CgroupV1Limits):
            raise TypeError("cgroup limits must be CgroupV1Limits")
        for method in ("prepare", "snapshot", "remove"):
            if not callable(getattr(backend, method, None)):
                raise TypeError("cgroup backend is incomplete")
        self.cell_name = cell_name
        self.limits = limits
        self.backend = backend
        self.relative_path = f"{CGROUP_RELATIVE_PREFIX}/{cell_name}"
        self.docker_parent = f"/{self.relative_path}"
        self._state = "new"
        self._prepare_receipt: Mapping[str, Any] | None = None
        self._observed_memory_peak = 0
        self._observed_memory_failcnt = 0
        self._observed_pids_peak = 0
        self._observed_pids_max_events = 0

    def prepare(self) -> Mapping[str, Any]:
        if self._state != "new":
            raise ResourceGuardError("cgroup envelope was already prepared")
        receipt = self.backend.prepare(self.relative_path, self.limits)
        self._validate_prepare_receipt(receipt)
        self._prepare_receipt = dict(receipt)
        self._state = "prepared"
        return dict(receipt)

    def _validate_prepare_receipt(self, receipt: Mapping[str, Any]) -> None:
        if not isinstance(receipt, Mapping):
            raise ResourceGuardError("cgroup prepare receipt is not an object")
        if receipt.get("schema") != "swebench_cgroup_v1_prepare_v1":
            raise ResourceGuardError("cgroup prepare schema drifted")
        if receipt.get("relative_path") != self.relative_path:
            raise ResourceGuardError("cgroup prepare path drifted")
        if receipt.get("limits_applied_before_tasks") is not True:
            raise ResourceGuardError("cgroup limits were not proven before attachment")
        memory = receipt.get("memory")
        pids = receipt.get("pids")
        if not isinstance(memory, Mapping) or not isinstance(pids, Mapping):
            raise ResourceGuardError("cgroup controller receipt is incomplete")
        expected_memory = {
            "limit_in_bytes": self.limits.memory_bytes,
            "memsw_limit_in_bytes": self.limits.memory_bytes,
            "swappiness": 0,
            "use_hierarchy": 1,
            "tasks": [],
        }
        expected_pids = {"max": self.limits.max_processes, "tasks": []}
        if dict(memory) != expected_memory or dict(pids) != expected_pids:
            raise ResourceGuardError("cgroup limits or initial task lists drifted")

    def docker_resource_arguments(self) -> tuple[str, ...]:
        if self._state not in {"prepared", "verified"}:
            raise ResourceGuardError(
                "Docker resources are unavailable before cgroup preparation"
            )
        return (
            f"--cgroup-parent={self.docker_parent}",
            f"--memory={self.limits.memory_bytes}",
            f"--memory-swap={self.limits.memory_bytes}",
            "--memory-swappiness=0",
            f"--pids-limit={self.limits.max_processes}",
        )

    def verify_descendants(self, *, container_init_pid: int) -> dict[str, Any]:
        if self._state not in {"prepared", "verified"}:
            raise ResourceGuardError("cgroup is not prepared")
        init_pid = _positive_integer(container_init_pid, "container init PID")
        snapshot = self._observe_snapshot()
        memory_tasks = snapshot["memory_tasks"]
        pids_tasks = snapshot["pids_tasks"]
        memory_procs = snapshot["memory_procs"]
        pids_procs = snapshot["pids_procs"]
        if memory_tasks != pids_tasks or memory_procs != pids_procs:
            raise ResourceGuardError("memory and pids descendant task sets disagree")
        if init_pid not in memory_procs:
            raise ResourceGuardError("container init PID is absent from the cell cgroup")
        memberships = snapshot["memberships"]
        descendant_pids = sorted(set(memory_procs) | set(pids_procs))
        if set(memberships) != set(descendant_pids):
            raise ResourceGuardError("descendant membership evidence is incomplete")
        for pid in descendant_pids:
            values = memberships[pid]
            for controller in ("memory", "pids"):
                path = values.get(controller)
                if not isinstance(path, str) or not _cgroup_path_is_within(
                    path, self.docker_parent
                ):
                    raise ResourceGuardError(
                        f"PID {pid} escaped the {controller} cell cgroup"
                    )
        self._state = "verified"
        return {
            "schema": "swebench_cgroup_v1_descendant_verification_v1",
            "relative_path": self.relative_path,
            "container_init_pid": init_pid,
            "descendant_pids": descendant_pids,
            "memory_pids_task_sets_equal": True,
            "all_descendants_inherited": True,
        }

    def observe(self) -> dict[str, Any]:
        """Capture live counters before Docker removes its child cgroup."""
        if self._state not in {"prepared", "verified"}:
            raise ResourceGuardError("cgroup is not prepared")
        snapshot = self._observe_snapshot()
        return {
            "schema": "swebench_cgroup_v1_observation_v1",
            "relative_path": self.relative_path,
            "memory_peak_bytes": self._observed_memory_peak,
            "memory_failcnt": self._observed_memory_failcnt,
            "pids_current": snapshot["pids_current"],
            "pids_peak": self._observed_pids_peak,
            "pids_max_events": self._observed_pids_max_events,
            "memory_tasks": snapshot["memory_tasks"],
            "memory_procs": snapshot["memory_procs"],
            "pids_tasks": snapshot["pids_tasks"],
            "pids_procs": snapshot["pids_procs"],
        }

    def teardown(self) -> dict[str, Any]:
        if self._state not in {"prepared", "verified"}:
            raise ResourceGuardError("cgroup envelope is not open")
        snapshot = self._observe_snapshot()
        if any(
            (
                snapshot["memory_tasks"],
                snapshot["memory_procs"],
                snapshot["pids_tasks"],
                snapshot["pids_procs"],
            )
        ):
            raise ResourceGuardError(
                "refusing to remove a nonempty memory or pids cgroup"
            )
        removed = self.backend.remove(self.relative_path)
        if (
            not isinstance(removed, Mapping)
            or removed.get("schema") != "swebench_cgroup_v1_remove_v1"
            or removed.get("relative_path") != self.relative_path
            or removed.get("memory_tasks_empty") is not True
            or removed.get("pids_tasks_empty") is not True
            or removed.get("removed") is not True
        ):
            raise ResourceGuardError("cgroup removal receipt drifted")
        self._state = "closed"
        return {
            "schema": "swebench_cgroup_v1_teardown_v1",
            "relative_path": self.relative_path,
            "memory_peak_bytes": self._observed_memory_peak,
            "memory_failcnt": self._observed_memory_failcnt,
            "pids_peak": self._observed_pids_peak,
            "pids_max_events": self._observed_pids_max_events,
            "memory_tasks_empty": True,
            "pids_tasks_empty": True,
            "removed": True,
        }

    def _observe_snapshot(self) -> dict[str, Any]:
        snapshot = self._validated_snapshot(self.backend.snapshot(self.relative_path))
        self._observed_memory_peak = max(
            self._observed_memory_peak, snapshot["memory_peak_bytes"]
        )
        self._observed_memory_failcnt = max(
            self._observed_memory_failcnt, snapshot["memory_failcnt"]
        )
        self._observed_pids_peak = max(
            self._observed_pids_peak, snapshot["pids_current"]
        )
        self._observed_pids_max_events = max(
            self._observed_pids_max_events, snapshot["pids_max_events"]
        )
        return snapshot

    def _validated_snapshot(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping) or value.get("schema") != (
            "swebench_cgroup_v1_snapshot_v1"
        ):
            raise ResourceGuardError("cgroup snapshot schema drifted")
        memory = value.get("memory")
        pids = value.get("pids")
        memberships_raw = value.get("memberships")
        if not isinstance(memory, Mapping) or not isinstance(pids, Mapping):
            raise ResourceGuardError("cgroup snapshot controller data is missing")
        if not isinstance(memberships_raw, Mapping):
            raise ResourceGuardError("cgroup snapshot memberships are missing")
        if memory.get("limit_in_bytes") != self.limits.memory_bytes:
            raise ResourceGuardError("memory cgroup limit drifted")
        if memory.get("memsw_limit_in_bytes") != self.limits.memory_bytes:
            raise ResourceGuardError("memory+swap cgroup limit drifted")
        if pids.get("max") != self.limits.max_processes:
            raise ResourceGuardError("pids cgroup limit drifted")
        memory_tasks = _pid_list(memory.get("tasks"), "memory tasks")
        memory_procs = _pid_list(memory.get("cgroup_procs"), "memory processes")
        pids_tasks = _pid_list(pids.get("tasks"), "pids tasks")
        pids_procs = _pid_list(pids.get("cgroup_procs"), "pids processes")
        memberships: dict[int, Mapping[str, Any]] = {}
        for raw_pid, payload in memberships_raw.items():
            if not isinstance(raw_pid, str) or not raw_pid.isdigit():
                raise ResourceGuardError("cgroup membership PID is invalid")
            pid = int(raw_pid)
            if pid <= 0 or not isinstance(payload, Mapping):
                raise ResourceGuardError("cgroup membership payload is invalid")
            memberships[pid] = payload
        return {
            "memory_tasks": memory_tasks,
            "memory_procs": memory_procs,
            "pids_tasks": pids_tasks,
            "pids_procs": pids_procs,
            "memberships": memberships,
            "memory_peak_bytes": _integer_counter(
                memory.get("max_usage_in_bytes"), "memory peak"
            ),
            "memory_failcnt": _integer_counter(memory.get("failcnt"), "memory failcnt"),
            "pids_current": _integer_counter(pids.get("current"), "pids current"),
            "pids_max_events": _integer_counter(
                pids.get("max_events"), "pids max events"
            ),
        }


def _cgroup_path_is_within(path: str, parent: str) -> bool:
    def canonical_parts(value: str) -> tuple[str, ...] | None:
        if not isinstance(value, str) or not value.startswith("/"):
            return None
        stripped = value.strip("/")
        if not stripped:
            return ()
        parts = tuple(stripped.split("/"))
        if any(not part or part in {".", ".."} for part in parts):
            return None
        return parts

    actual = canonical_parts(path)
    expected = canonical_parts(parent)
    if actual is None or not expected or len(actual) < len(expected):
        return False
    # /proc/<pid>/cgroup reports the host-global Kubernetes prefix, while the
    # private daemon mount namespace exposes the same owned subtree as its root.
    # Match the complete owned path-segment sequence at any ancestor depth.
    width = len(expected)
    return any(
        actual[offset : offset + width] == expected
        for offset in range(len(actual) - width + 1)
    )


class MountNamespaceCgroupV1Backend:
    """Run cgroup operations inside the isolated Docker daemon mount namespace."""

    def __init__(
        self,
        *,
        namespace_pid: int,
        python_executable: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.namespace_pid = _positive_integer(namespace_pid, "namespace PID")
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = _positive_integer(
            timeout_seconds, "cgroup helper timeout"
        )

    def prepare(
        self,
        relative_path: str,
        limits: CgroupV1Limits,
    ) -> Mapping[str, Any]:
        return self._request(
            {
                "action": "prepare",
                "relative_path": relative_path,
                "memory_bytes": limits.memory_bytes,
                "max_processes": limits.max_processes,
            }
        )

    def snapshot(self, relative_path: str) -> Mapping[str, Any]:
        return self._request({"action": "snapshot", "relative_path": relative_path})

    def remove(self, relative_path: str) -> Mapping[str, Any]:
        return self._request({"action": "remove", "relative_path": relative_path})

    def _request(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if not Path(f"/proc/{self.namespace_pid}/ns/mnt").exists():
            raise ResourceGuardError("Docker daemon mount namespace is unavailable")
        with cgroup_structure_lock():
            try:
                completed = subprocess.run(
                    [
                        self.python_executable,
                        "-m",
                        "swebench_triad_eval.resource_guard",
                        "_cgroup_helper",
                        str(self.namespace_pid),
                    ],
                    input=json.dumps(value, sort_keys=True),
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ResourceGuardError(
                    "cgroup namespace helper could not run"
                ) from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise ResourceGuardError(f"cgroup namespace helper failed: {detail}")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ResourceGuardError("cgroup helper emitted invalid JSON") from error
        if not isinstance(result, Mapping):
            raise ResourceGuardError("cgroup helper result is not an object")
        return result


@dataclass(frozen=True)
class QuotaMountSpec:
    byte_limit: int
    inode_limit: int
    purpose: str

    def __post_init__(self) -> None:
        byte_limit = _positive_integer(self.byte_limit, "tmpfs byte limit")
        _positive_integer(self.inode_limit, "tmpfs inode limit")
        if byte_limit % _PAGE_BYTES:
            raise ValueError("tmpfs byte limit must be page aligned")
        if self.purpose not in {"workspace", "external-memory"}:
            raise ValueError("tmpfs purpose is unsupported")


class TmpfsMountBackend(Protocol):
    def mount_tmpfs(
        self,
        target: Path,
        spec: QuotaMountSpec,
        *,
        owner_uid: int,
        owner_gid: int,
    ) -> Mapping[str, Any]: ...

    def unmount_tmpfs(self, target: Path, mount_id: int) -> Mapping[str, Any]: ...


class LinuxTmpfsMountBackend:
    """Create and identify exact tmpfs mounts in the current mount namespace."""

    def __init__(
        self,
        *,
        mount_executable: str | None = None,
        umount_executable: str | None = None,
    ) -> None:
        self.mount_executable = mount_executable or shutil.which("mount") or "mount"
        self.umount_executable = umount_executable or shutil.which("umount") or "umount"

    def mount_tmpfs(
        self,
        target: Path,
        spec: QuotaMountSpec,
        *,
        owner_uid: int,
        owner_gid: int,
    ) -> Mapping[str, Any]:
        target = _require_real_directory(target, "tmpfs mountpoint")
        uid = _non_negative_integer(owner_uid, "tmpfs owner UID")
        gid = _non_negative_integer(owner_gid, "tmpfs owner GID")
        if self._mount_record(target) is not None:
            raise ResourceGuardError("tmpfs target is already a mountpoint")
        options = (
            f"rw,nosuid,nodev,size={spec.byte_limit},nr_inodes={spec.inode_limit},"
            f"mode=0700,uid={uid},gid={gid}"
        )
        try:
            result = subprocess.run(
                [
                    self.mount_executable,
                    "-t",
                    "tmpfs",
                    "-o",
                    options,
                    f"swebench-{spec.purpose}",
                    str(target),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise ResourceGuardError("tmpfs mount executable failed") from error
        if result.returncode != 0:
            raise ResourceGuardError(
                "tmpfs mount failed: " + (result.stderr or "unknown error").strip()
            )
        try:
            record = self._mount_record(target)
            if record is None or record["fs_type"] != "tmpfs":
                raise ResourceGuardError("tmpfs mount identity is unavailable")
            filesystem = os.statvfs(target)
            total_bytes = filesystem.f_blocks * filesystem.f_frsize
            total_inodes = filesystem.f_files
            return {
                "schema": "swebench_tmpfs_mount_v1",
                **record,
                "total_bytes": total_bytes,
                "total_inodes": total_inodes,
                "owner_uid": os.stat(target, follow_symlinks=False).st_uid,
                "owner_gid": os.stat(target, follow_symlinks=False).st_gid,
            }
        except BaseException:
            subprocess.run(
                [self.umount_executable, str(target)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            raise

    def unmount_tmpfs(self, target: Path, mount_id: int) -> Mapping[str, Any]:
        target = _require_real_directory(target, "tmpfs mountpoint")
        expected_id = _positive_integer(mount_id, "tmpfs mount ID")
        record = self._mount_record(target)
        if record is None or record.get("mount_id") != expected_id:
            raise ResourceGuardError("refusing to unmount an unowned mount")
        if record.get("fs_type") != "tmpfs":
            raise ResourceGuardError("refusing to unmount a non-tmpfs filesystem")
        try:
            result = subprocess.run(
                [self.umount_executable, str(target)],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise ResourceGuardError("tmpfs unmount executable failed") from error
        if result.returncode != 0:
            raise ResourceGuardError(
                "tmpfs unmount failed: " + (result.stderr or "unknown error").strip()
            )
        if self._mount_record(target) is not None:
            raise ResourceGuardError("tmpfs mount survived unmount")
        return {"mount_id": expected_id, "unmounted": True}

    @staticmethod
    def _mount_record(target: Path) -> dict[str, Any] | None:
        target_text = str(target.resolve(strict=True))
        try:
            lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ResourceGuardError("cannot read mountinfo") from error
        matches: list[dict[str, Any]] = []
        for line in lines:
            before, separator, after = line.partition(" - ")
            if not separator:
                raise ResourceGuardError("mountinfo row is malformed")
            fields = before.split()
            suffix = after.split()
            if len(fields) < 6 or len(suffix) < 3:
                raise ResourceGuardError("mountinfo row is incomplete")
            mountpoint = _decode_mountinfo_path(fields[4])
            if mountpoint != target_text:
                continue
            options = sorted(
                set(fields[5].split(",")) | set(suffix[2].split(","))
            )
            matches.append(
                {
                    "mount_id": int(fields[0]),
                    "mount_point": mountpoint,
                    "fs_type": suffix[0],
                    "source": suffix[1],
                    "options": options,
                }
            )
        if len(matches) > 1:
            raise ResourceGuardError("tmpfs target has stacked mounts")
        return matches[0] if matches else None


class TmpfsQuotaMounts:
    """Own only the two tmpfs mounts created for one environment server."""

    def __init__(self, backend: TmpfsMountBackend) -> None:
        for method in ("mount_tmpfs", "unmount_tmpfs"):
            if not callable(getattr(backend, method, None)):
                raise TypeError("tmpfs backend is incomplete")
        self.backend = backend
        self._mounts: list[tuple[Path, int, Mapping[str, Any]]] = []
        self._closed = False

    def mount_preserving(
        self,
        target: Path | str,
        spec: QuotaMountSpec,
        *,
        owner_uid: int,
        owner_gid: int,
    ) -> dict[str, Any]:
        if self._closed:
            raise ResourceGuardError("tmpfs mount owner is closed")
        if not isinstance(spec, QuotaMountSpec):
            raise TypeError("tmpfs quota must be QuotaMountSpec")
        root = _require_real_directory(Path(target), f"{spec.purpose} root")
        if any(existing == root for existing, _, _ in self._mounts):
            raise ResourceGuardError("tmpfs target is already owned")
        uid = _non_negative_integer(owner_uid, "tmpfs owner UID")
        gid = _non_negative_integer(owner_gid, "tmpfs owner GID")
        original = root.lstat()
        staging = root.parent / (
            f".{root.name}.pre-tmpfs-{os.getpid()}-{secrets.token_hex(8)}"
        )
        os.rename(root, staging)
        root.mkdir(mode=stat.S_IMODE(original.st_mode))
        mounted: Mapping[str, Any] | None = None
        try:
            mounted = self.backend.mount_tmpfs(
                root,
                spec,
                owner_uid=uid,
                owner_gid=gid,
            )
            receipt = self._validate_mount_receipt(
                mounted,
                root,
                spec,
                owner_uid=uid,
                owner_gid=gid,
            )
            _copy_directory_contents(staging, root)
            _chown_tree(root, uid, gid)
            shutil.rmtree(staging)
            self._mounts.append((root, receipt["mount_id"], receipt))
            return receipt
        except BaseException:
            cleanup_error: BaseException | None = None
            if mounted is not None:
                mount_id = mounted.get("mount_id")
                if isinstance(mount_id, int) and not isinstance(mount_id, bool):
                    try:
                        self.backend.unmount_tmpfs(root, mount_id)
                    except BaseException as error:
                        cleanup_error = error
            if cleanup_error is not None:
                raise ResourceGuardError(
                    "tmpfs setup failed and the owned mount could not be removed"
                ) from cleanup_error
            _remove_real_directory(root)
            if staging.exists():
                os.rename(staging, root)
            raise

    @staticmethod
    def _validate_mount_receipt(
        value: Mapping[str, Any],
        target: Path,
        spec: QuotaMountSpec,
        *,
        owner_uid: int,
        owner_gid: int,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping) or value.get("schema") != (
            "swebench_tmpfs_mount_v1"
        ):
            raise ResourceGuardError("tmpfs mount receipt schema drifted")
        mount_id = value.get("mount_id")
        if isinstance(mount_id, bool) or not isinstance(mount_id, int) or mount_id <= 0:
            raise ResourceGuardError("tmpfs mount ID is invalid")
        if value.get("mount_point") != str(target) or value.get("fs_type") != "tmpfs":
            raise ResourceGuardError("tmpfs mount identity drifted")
        if (
            value.get("total_bytes") != spec.byte_limit
            or value.get("total_inodes") != spec.inode_limit
        ):
            raise ResourceGuardError("tmpfs byte or inode quota drifted")
        if value.get("owner_uid") != owner_uid or value.get("owner_gid") != owner_gid:
            raise ResourceGuardError("tmpfs ownership drifted")
        options = value.get("options")
        if not isinstance(options, list) or not {"nosuid", "nodev"}.issubset(options):
            raise ResourceGuardError("tmpfs safety mount options drifted")
        return dict(value)

    def unmount(self, target: Path | str) -> Mapping[str, Any]:
        root = Path(target).resolve(strict=True)
        for index in range(len(self._mounts) - 1, -1, -1):
            existing, mount_id, _ = self._mounts[index]
            if existing != root:
                continue
            receipt = self.backend.unmount_tmpfs(existing, mount_id)
            self._mounts.pop(index)
            return receipt
        raise ResourceGuardError("tmpfs target is not owned by this lifecycle")

    def close(self) -> dict[str, Any]:
        if self._closed:
            return {
                "schema": "swebench_tmpfs_teardown_v1",
                "unmounted_count": 0,
                "already_closed": True,
            }
        unmounted = 0
        while self._mounts:
            target, mount_id, _ = self._mounts[-1]
            self.backend.unmount_tmpfs(target, mount_id)
            self._mounts.pop()
            unmounted += 1
        self._closed = True
        return {
            "schema": "swebench_tmpfs_teardown_v1",
            "unmounted_count": unmounted,
            "already_closed": False,
        }


RootfsAttestor = Callable[[Union[Path, str]], Mapping[str, Any]]


class RootfsMutationGuard:
    """Require the deployment's expensive full-tree attestation to stay fixed."""

    def __init__(
        self,
        cache_directory: Path | str,
        *,
        attestor: RootfsAttestor | None = None,
    ) -> None:
        self.cache_directory = Path(cache_directory)
        if attestor is None:
            from .oci import attest_rootfs

            attestor = attest_rootfs
        if not callable(attestor):
            raise TypeError("rootfs attestor must be callable")
        self.attestor = attestor
        self.baseline = self._read_attestation()

    def _read_attestation(self) -> dict[str, Any]:
        try:
            value = self.attestor(self.cache_directory)
        except BaseException as error:
            raise ResourceGuardError("full rootfs attestation failed") from error
        if not isinstance(value, Mapping) or value.get("schema") != (
            "swebench_verified_rootfs_attestation_v1"
        ):
            raise ResourceGuardError("full rootfs attestation schema drifted")
        if value.get("status") != "pass":
            raise ResourceGuardError("full rootfs attestation did not pass")
        required_text = ("image", "manifest_digest", "config_digest", "tree_sha256")
        if any(not isinstance(value.get(name), str) or not value[name] for name in required_text):
            raise ResourceGuardError("full rootfs attestation identity is incomplete")
        _positive_integer(value.get("path_count"), "rootfs path count")
        return dict(value)

    def attest(self) -> dict[str, Any]:
        current = self._read_attestation()
        if current != self.baseline:
            raise ResourceGuardError("immutable rootfs mutated after baseline attestation")
        return current


class GuardedEpisodeSandboxMixin:
    """Deployment-only mixin for the published Verified sandbox class."""

    def configure_deployment_guards(
        self,
        *,
        quota_mounts: TmpfsQuotaMounts,
        workspace_quota: QuotaMountSpec,
        external_memory_quota: QuotaMountSpec,
        rootfs_guard: RootfsMutationGuard,
    ) -> None:
        if hasattr(self, "_deployment_quota_mounts"):
            raise ResourceGuardError("sandbox deployment guards were already configured")
        if not isinstance(quota_mounts, TmpfsQuotaMounts):
            raise TypeError("sandbox quota mounts have the wrong type")
        if not isinstance(workspace_quota, QuotaMountSpec):
            raise TypeError("workspace quota has the wrong type")
        if workspace_quota.purpose != "workspace":
            raise ValueError("workspace quota purpose drifted")
        if not isinstance(external_memory_quota, QuotaMountSpec):
            raise TypeError("external-memory quota has the wrong type")
        if external_memory_quota.purpose != "external-memory":
            raise ValueError("external-memory quota purpose drifted")
        if not isinstance(rootfs_guard, RootfsMutationGuard):
            raise TypeError("rootfs guard has the wrong type")
        self._deployment_quota_mounts = quota_mounts
        self._deployment_workspace_quota = workspace_quota
        self._deployment_external_memory_quota = external_memory_quota
        self._deployment_rootfs_guard = rootfs_guard

    def _deployment_guards(
        self,
    ) -> tuple[TmpfsQuotaMounts, QuotaMountSpec, QuotaMountSpec, RootfsMutationGuard]:
        try:
            return (
                self._deployment_quota_mounts,
                self._deployment_workspace_quota,
                self._deployment_external_memory_quota,
                self._deployment_rootfs_guard,
            )
        except AttributeError as error:
            raise ResourceGuardError("sandbox deployment guards are not configured") from error

    def attach_workspace(self, workspace_root: Path | str):
        mounts, workspace_quota, _, rootfs_guard = self._deployment_guards()
        root = Path(workspace_root)
        # The expensive full-tree check is cell-scoped.  The published sandbox
        # still performs its cheap key-file fingerprint check before every
        # action, while the extracted rootfs is bound read-only to the policy.
        rootfs_guard.attest()
        mounts.mount_preserving(
            root,
            workspace_quota,
            owner_uid=self.model_uid,
            owner_gid=self.model_gid,
        )
        try:
            return super().attach_workspace(root)
        except BaseException:
            mounts.unmount(root)
            raise

    def attach_external_memory(self, memory_root: Path | str):
        mounts, _, memory_quota, _ = self._deployment_guards()
        root = Path(memory_root)
        mounts.mount_preserving(
            root,
            memory_quota,
            owner_uid=self.model_uid,
            owner_gid=self.model_gid,
        )
        try:
            return super().attach_external_memory(root)
        except BaseException:
            mounts.unmount(root)
            raise

    def _run_namespace(self, workspace_root: Path, **kwargs: Any):
        self._deployment_guards()
        return super()._run_namespace(workspace_root, **kwargs)

    def close(self) -> None:
        mounts, _, _, _ = self._deployment_guards()
        primary: BaseException | None = None
        try:
            super().close()
        except BaseException as error:
            primary = error
        try:
            mounts.close()
        except BaseException as cleanup_error:
            if primary is not None:
                raise ResourceGuardError(
                    "sandbox and tmpfs cleanup both failed"
                ) from cleanup_error
            raise
        if primary is not None:
            raise primary


def _non_negative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _require_real_directory(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as error:
        raise ResourceGuardError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ResourceGuardError(f"{label} must be a real directory")
    return path.resolve(strict=True)


def _copy_directory_contents(source: Path, destination: Path) -> None:
    try:
        for child in source.iterdir():
            target = destination / child.name
            if child.is_symlink():
                os.symlink(os.readlink(child), target)
            elif child.is_dir():
                shutil.copytree(
                    child,
                    target,
                    symlinks=True,
                    copy_function=shutil.copy2,
                )
            elif child.is_file():
                shutil.copy2(child, target, follow_symlinks=False)
            else:
                raise ResourceGuardError("quota source contains a special file")
    except (OSError, shutil.Error) as error:
        if isinstance(error, ResourceGuardError):
            raise
        raise ResourceGuardError("cannot populate quota filesystem") from error


def _chown_tree(root: Path, uid: int, gid: int) -> None:
    try:
        os.chown(root, uid, gid, follow_symlinks=False)
        for current, directory_names, file_names in os.walk(root, followlinks=False):
            current_path = Path(current)
            for name in [*directory_names, *file_names]:
                os.chown(current_path / name, uid, gid, follow_symlinks=False)
    except OSError as error:
        raise ResourceGuardError("cannot assign quota tree ownership") from error


def _remove_real_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ResourceGuardError("refusing to remove a non-directory mountpoint")
    shutil.rmtree(path)


def _decode_mountinfo_path(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 8))

    return re.sub(r"\\([0-7]{3})", replace, value)


def _enter_mount_namespace(pid: int) -> None:
    namespace = f"/proc/{pid}/ns/mnt"
    descriptor = os.open(namespace, os.O_RDONLY)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.setns(descriptor, 0x00020000)
        if result != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), namespace)
    finally:
        os.close(descriptor)


def _validate_helper_relative_path(value: Any) -> str:
    if not isinstance(value, str):
        raise ResourceGuardError("cgroup relative path must be text")
    parts = value.split("/")
    if (
        len(parts) != 3
        or parts[0] != CGROUP_RUNTIME_PARENT
        or parts[1] != CGROUP_EVALUATION_PARENT
        or _CELL_NAME_RE.fullmatch(parts[2]) is None
    ):
        raise ResourceGuardError("cgroup relative path is outside the owned scope")
    return value


def _cgroup_directory(controller: str, relative_path: str) -> Path:
    if controller not in {"memory", "pids"}:
        raise ResourceGuardError("unsupported cgroup controller")
    root = _require_real_directory(Path("/sys/fs/cgroup") / controller, "cgroup root")
    path = root
    for part in relative_path.split("/"):
        path = path / part
    return path


def _cgroup_read(path: Path) -> str:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ResourceGuardError("cgroup control file is a symlink")
        return path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise ResourceGuardError(f"cannot read cgroup control: {path.name}") from error


def _cgroup_write(path: Path, value: int) -> None:
    flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            payload = str(value).encode("ascii")
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError("short cgroup write")
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ResourceGuardError(f"cannot write cgroup control: {path.name}") from error


def _read_pid_file(path: Path) -> list[int]:
    text = _cgroup_read(path)
    if not text:
        return []
    values: list[int] = []
    for line in text.splitlines():
        try:
            pid = int(line)
        except ValueError as error:
            raise ResourceGuardError("cgroup task list is malformed") from error
        if pid <= 0:
            raise ResourceGuardError("cgroup task list contains an invalid PID")
        values.append(pid)
    return sorted(set(values))


def _cgroup_directory_chain(path: Path, root: Path) -> list[Path]:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError as error:
        raise ResourceGuardError("owned cgroup path escaped its controller") from error
    if not relative_parts:
        raise ResourceGuardError("owned cgroup path is empty")
    resolved_root = _require_real_directory(root, "cgroup controller root")
    chain: list[Path] = []
    current = resolved_root
    for part in relative_parts:
        current = current / part
        chain.append(current)
    return chain


def _require_cgroup_directory_chain(path: Path, root: Path) -> None:
    for candidate in _cgroup_directory_chain(path, root):
        resolved = _require_real_directory(candidate, "owned cgroup component")
        if resolved != candidate:
            raise ResourceGuardError(
                "owned cgroup component must be a real directory"
            )


def _ensure_cgroup_directory(
    path: Path,
    *,
    root: Path,
    created: list[Path],
) -> None:
    for candidate in _cgroup_directory_chain(path, root):
        try:
            candidate.lstat()
        except FileNotFoundError:
            try:
                candidate.mkdir(mode=0o755)
            except OSError as error:
                raise ResourceGuardError("cannot create owned cgroup") from error
            created.append(candidate)
        except OSError as error:
            raise ResourceGuardError("cannot inspect owned cgroup") from error
        _require_real_directory(candidate, "owned cgroup component")
    _require_cgroup_directory_chain(path, root)


def _helper_prepare(request: Mapping[str, Any]) -> dict[str, Any]:
    relative = _validate_helper_relative_path(request.get("relative_path"))
    limits = CgroupV1Limits(
        memory_bytes=request.get("memory_bytes"),
        max_processes=request.get("max_processes"),
    )
    memory = _cgroup_directory("memory", relative)
    pids = _cgroup_directory("pids", relative)
    created: list[Path] = []
    try:
        _ensure_cgroup_directory(
            memory,
            root=memory.parents[2],
            created=created,
        )
        _ensure_cgroup_directory(
            pids,
            root=pids.parents[2],
            created=created,
        )
        _require_cgroup_directory_chain(memory, memory.parents[2])
        _require_cgroup_directory_chain(pids, pids.parents[2])
        if _read_pid_file(memory / "tasks") or _read_pid_file(pids / "tasks"):
            raise ResourceGuardError("owned cgroup is not empty before limits")
        _cgroup_write(memory / "memory.use_hierarchy", 1)
        _cgroup_write(memory / "memory.limit_in_bytes", limits.memory_bytes)
        _cgroup_write(memory / "memory.memsw.limit_in_bytes", limits.memory_bytes)
        _cgroup_write(memory / "memory.swappiness", 0)
        _cgroup_write(pids / "pids.max", limits.max_processes)
        receipt = {
            "schema": "swebench_cgroup_v1_prepare_v1",
            "relative_path": relative,
            "limits_applied_before_tasks": True,
            "memory": {
                "limit_in_bytes": int(_cgroup_read(memory / "memory.limit_in_bytes")),
                "memsw_limit_in_bytes": int(
                    _cgroup_read(memory / "memory.memsw.limit_in_bytes")
                ),
                "swappiness": int(_cgroup_read(memory / "memory.swappiness")),
                "use_hierarchy": int(
                    _cgroup_read(memory / "memory.use_hierarchy")
                ),
                "tasks": _read_pid_file(memory / "tasks"),
            },
            "pids": {
                "max": int(_cgroup_read(pids / "pids.max")),
                "tasks": _read_pid_file(pids / "tasks"),
            },
        }
        return receipt
    except BaseException:
        for path in reversed(created):
            try:
                path.rmdir()
            except OSError:
                pass
        raise


def _descendant_cgroup_directories(root: Path) -> list[Path]:
    directories: list[Path] = []
    try:
        for current, directory_names, _ in os.walk(root, followlinks=False):
            current_path = Path(current)
            retained: list[str] = []
            for name in directory_names:
                child = current_path / name
                if child.is_symlink():
                    raise ResourceGuardError("cgroup tree contains a symlink")
                retained.append(name)
            directory_names[:] = retained
            directories.append(current_path)
    except OSError as error:
        raise ResourceGuardError("cannot enumerate cgroup descendants") from error
    return directories


def _descendant_pid_lists(root: Path) -> tuple[list[int], list[int]]:
    tasks: set[int] = set()
    processes: set[int] = set()
    for current_path in _descendant_cgroup_directories(root):
        tasks.update(_read_pid_file(current_path / "tasks"))
        processes.update(_read_pid_file(current_path / "cgroup.procs"))
    return sorted(tasks), sorted(processes)


def _sum_descendant_counter(root: Path, filename: str) -> int:
    return sum(
        int(_cgroup_read(directory / filename))
        for directory in _descendant_cgroup_directories(root)
    )


def _max_descendant_counter(root: Path, filename: str) -> int:
    return max(
        int(_cgroup_read(directory / filename))
        for directory in _descendant_cgroup_directories(root)
    )


def _pid_memberships(pid: int) -> dict[str, str]:
    try:
        rows = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii").splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return {}
    except OSError as error:
        raise ResourceGuardError("cannot read descendant cgroup membership") from error
    memberships: dict[str, str] = {}
    for row in rows:
        fields = row.split(":", 2)
        if len(fields) != 3:
            raise ResourceGuardError("descendant cgroup membership is malformed")
        for controller in fields[1].split(","):
            if controller in {"memory", "pids"}:
                memberships[controller] = fields[2]
    return memberships


def _read_pids_events(path: Path) -> int:
    values: dict[str, int] = {}
    for row in _cgroup_read(path).splitlines():
        fields = row.split()
        if len(fields) != 2:
            raise ResourceGuardError("pids event counter is malformed")
        try:
            values[fields[0]] = int(fields[1])
        except ValueError as error:
            raise ResourceGuardError("pids event counter is malformed") from error
    return values.get("max", 0)


def _helper_snapshot(request: Mapping[str, Any]) -> dict[str, Any]:
    relative = _validate_helper_relative_path(request.get("relative_path"))
    memory = _require_real_directory(
        _cgroup_directory("memory", relative), "memory cell cgroup"
    )
    pids = _require_real_directory(
        _cgroup_directory("pids", relative), "pids cell cgroup"
    )
    memory_tasks, memory_procs = _descendant_pid_lists(memory)
    pids_tasks, pids_procs = _descendant_pid_lists(pids)
    processes = sorted(set(memory_procs) | set(pids_procs))
    return {
        "schema": "swebench_cgroup_v1_snapshot_v1",
        "relative_path": relative,
        "memory": {
            "tasks": memory_tasks,
            "cgroup_procs": memory_procs,
            "limit_in_bytes": int(_cgroup_read(memory / "memory.limit_in_bytes")),
            "memsw_limit_in_bytes": int(
                _cgroup_read(memory / "memory.memsw.limit_in_bytes")
            ),
            "max_usage_in_bytes": _max_descendant_counter(
                memory, "memory.max_usage_in_bytes"
            ),
            "failcnt": _sum_descendant_counter(memory, "memory.failcnt"),
        },
        "pids": {
            "tasks": pids_tasks,
            "cgroup_procs": pids_procs,
            "max": int(_cgroup_read(pids / "pids.max")),
            "current": _max_descendant_counter(pids, "pids.current"),
            "max_events": sum(
                _read_pids_events(directory / "pids.events")
                for directory in _descendant_cgroup_directories(pids)
            ),
        },
        "memberships": {str(pid): _pid_memberships(pid) for pid in processes},
    }


def _remove_empty_cgroup_tree(root: Path) -> None:
    directories: list[Path] = []
    for current, directory_names, _ in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.append(current_path)
        for name in directory_names:
            if (current_path / name).is_symlink():
                raise ResourceGuardError("cgroup tree contains a symlink")
    for directory in reversed(directories):
        if _read_pid_file(directory / "tasks") or _read_pid_file(
            directory / "cgroup.procs"
        ):
            raise ResourceGuardError("cgroup became nonempty during removal")
        try:
            directory.rmdir()
        except OSError as error:
            raise ResourceGuardError("cannot remove empty owned cgroup") from error


def _helper_remove(request: Mapping[str, Any]) -> dict[str, Any]:
    relative = _validate_helper_relative_path(request.get("relative_path"))
    snapshot = _helper_snapshot(request)
    for controller in ("memory", "pids"):
        details = snapshot[controller]
        if details["tasks"] or details["cgroup_procs"]:
            raise ResourceGuardError("refusing to remove a nonempty cgroup")
    for controller in ("pids", "memory"):
        cell = _cgroup_directory(controller, relative)
        _remove_empty_cgroup_tree(cell)
        evaluation_parent = cell.parent
        try:
            evaluation_parent.rmdir()
        except OSError:
            pass
    return {
        "schema": "swebench_cgroup_v1_remove_v1",
        "relative_path": relative,
        "memory_tasks_empty": True,
        "pids_tasks_empty": True,
        "removed": True,
    }


def _cgroup_helper_main(namespace_pid: int) -> int:
    try:
        raw = sys.stdin.read()
        request = json.loads(raw)
        if not isinstance(request, Mapping):
            raise ResourceGuardError("cgroup helper request is not an object")
        _enter_mount_namespace(namespace_pid)
        action = request.get("action")
        if action == "prepare":
            result = _helper_prepare(request)
        elif action == "snapshot":
            result = _helper_snapshot(request)
        elif action == "remove":
            result = _helper_remove(request)
        else:
            raise ResourceGuardError("cgroup helper action is unsupported")
        sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
        return 0
    except BaseException as error:
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 1


def _main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "_cgroup_helper":
        try:
            pid = int(argv[1])
        except ValueError:
            return 2
        return _cgroup_helper_main(pid)
    return 2


__all__ = [
    "CgroupV1CellEnvelope",
    "CgroupV1Limits",
    "GuardedEpisodeSandboxMixin",
    "LinuxTmpfsMountBackend",
    "MountNamespaceCgroupV1Backend",
    "QuotaMountSpec",
    "ResourceGuardError",
    "RootfsMutationGuard",
    "TmpfsQuotaMounts",
]


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
