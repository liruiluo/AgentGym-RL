"""Crash-safe supervision for the non-yield GPU/CPU fallback holder.

The platform fallback predates formal-run yield markers.  This module keeps a
single watchdog alive, pauses it transactionally for a formal run, and only
resumes it after every durably published run-owned process has drained.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class FallbackError(RuntimeError):
    """A fallback holder transaction could not be proved safe."""


_ZOMBIE_STATES = {"Z", "X", "x"}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_PAUSE_SCHEMA = "amg_fallback_pause_v2"
_RELEASE_REQUEST_SCHEMA = "amg_fallback_release_request_v1"
_PENDING_RESUME_SCHEMA = "amg_fallback_pending_resume_v1"
_STATE_SCHEMA = "amg_fallback_supervisor_v2"
_PROCESS_IDENTITY_SCHEMA = "amg_fallback_managed_process_identity_v2"
_PIDFD_SEND_SIGNAL_SYSCALL = 424
_PIDFD_OPEN_SYSCALL = 434
_BOOTSTRAP_DRAIN_GRACE_SECONDS = 15.0
_INNER_HOLDER_MARKERS = {
    "cpu": Path("/tmp/agentmemory-formal-cpu-active"),
    "gpu": Path("/tmp/crg-holder-yield"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, expected_sha256: str | None = None) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise FallbackError(f"required file is missing: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise FallbackError(f"required path is not a regular file: {path}")
    if expected_sha256 is not None:
        observed = _sha256(path)
        if observed != expected_sha256:
            raise FallbackError(
                f"sha256 mismatch for {path}: expected={expected_sha256} "
                f"observed={observed}"
            )
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temp)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    _regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FallbackError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise FallbackError(f"JSON object required: {path}")
    return value


def _bound_json_file(
    path: Path, *, maximum_bytes: int = 1 << 20
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read JSON and its file identity from one descriptor.

    The process-identity files authorize later cleanup.  Reading the payload,
    metadata, and digest through separate path opens leaves a replacement
    window in which those three facts can describe different files.
    """

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FallbackError(f"cannot open bound JSON file {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FallbackError(f"bound JSON path is not a regular file: {path}")
        if before.st_size > maximum_bytes:
            raise FallbackError(f"bound JSON file is too large: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise FallbackError(f"bound JSON file is too large: {path}")
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_ctime_ns", "st_size")
        ) or len(raw) != after.st_size:
            raise FallbackError(f"bound JSON file changed while reading: {path}")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FallbackError(f"invalid JSON file {path}: {error}") from error
        if not isinstance(payload, dict):
            raise FallbackError(f"JSON object required: {path}")
        try:
            named = path.lstat()
        except OSError as error:
            raise FallbackError(
                f"bound JSON path disappeared while reading: {path}"
            ) from error
        if path.is_symlink() or any(
            getattr(named, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_ctime_ns", "st_size")
        ):
            raise FallbackError(f"bound JSON path was replaced while reading: {path}")
        binding = {
            "path": str(path),
            "device": after.st_dev,
            "inode": after.st_ino,
            "ctime_ns": after.st_ctime_ns,
            "size": after.st_size,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "payload": payload,
        }
        return payload, binding
    finally:
        os.close(descriptor)


def _proc_identity(pid: int) -> dict[str, Any] | None:
    try:
        fields = (
            Path(f"/proc/{pid}/stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .split()
        )
    except (FileNotFoundError, IndexError, OSError):
        return None
    return {
        "pid": pid,
        "state": fields[0],
        "ppid": int(fields[1]),
        "pgrp": int(fields[2]),
        "start_ticks": fields[19],
    }


def _identity_alive(identity: Mapping[str, Any] | None) -> bool:
    if not identity:
        return False
    try:
        observed = _proc_identity(int(identity["pid"]))
        return bool(
            observed
            and observed["state"] not in _ZOMBIE_STATES
            and observed["start_ticks"] == str(identity["start_ticks"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _capture_identity(pid: int, timeout_seconds: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        identity = _proc_identity(pid)
        if identity and identity["state"] not in _ZOMBIE_STATES:
            return identity
        time.sleep(0.01)
    raise FallbackError(f"could not capture live process identity for pid={pid}")


def _pidfd_open_exact(pid: int) -> int:
    """Open a pidfd or fail closed; never fall back to signalling a raw PID."""

    if not sys.platform.startswith("linux"):
        raise FallbackError("exact process signalling requires Linux pidfds")
    pidfd_open = getattr(os, "pidfd_open", None)
    if callable(pidfd_open):
        try:
            return int(pidfd_open(pid, 0))
        except ProcessLookupError:
            raise
        except PermissionError as error:
            raise FallbackError(
                f"pidfd_open permission denied for pid {pid}"
            ) from error
        except OSError as error:
            raise FallbackError(f"pidfd_open failed for pid {pid}: {error}") from error
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    descriptor = int(syscall(_PIDFD_OPEN_SYSCALL, pid, 0))
    if descriptor >= 0:
        return descriptor
    error_number = ctypes.get_errno()
    if error_number == errno.ESRCH:
        raise ProcessLookupError(pid)
    raise FallbackError(
        "pidfd_open syscall failed: "
        f"pid={pid} errno={error_number} {os.strerror(error_number)}"
    )


def _pidfd_send_signal_exact(descriptor: int, signum: int, *, pid: int) -> None:
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if callable(pidfd_send_signal):
        try:
            pidfd_send_signal(descriptor, signum, None, 0)
        except ProcessLookupError:
            raise
        except PermissionError as error:
            raise FallbackError(
                f"pidfd_send_signal permission denied for pid {pid}"
            ) from error
        except OSError as error:
            raise FallbackError(
                f"pidfd_send_signal failed for pid {pid}: {error}"
            ) from error
        return
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    result = int(
        syscall(
            _PIDFD_SEND_SIGNAL_SYSCALL,
            descriptor,
            signum,
            ctypes.c_void_p(0),
            0,
        )
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.ESRCH:
        raise ProcessLookupError(pid)
    raise FallbackError(
        "pidfd_send_signal syscall failed: "
        f"pid={pid} errno={error_number} {os.strerror(error_number)}"
    )


def _signal_identity(identity: Mapping[str, Any], signum: int) -> bool:
    if not _identity_alive(identity):
        return False
    pid = int(identity["pid"])
    descriptor = -1
    try:
        descriptor = _pidfd_open_exact(pid)
        # Opening a pidfd closes the PID-reuse window, while the second
        # start-ticks check proves that the pidfd names the recorded process.
        if not _identity_alive(identity):
            return False
        _pidfd_send_signal_exact(descriptor, signum, pid=pid)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise FallbackError(
            f"pidfd signalling permission denied for pid {pid}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return True


def _cmdline(pid: int) -> str:
    """Read a command line without hiding permission or decoding failures."""

    return " ".join(_process_argv(pid))


def _process_argv(pid: int) -> tuple[str, ...]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return ()
    except OSError as error:
        if error.errno in {errno.ENOENT, errno.ESRCH}:
            return ()
        raise FallbackError(
            f"cannot audit process command line /proc/{pid}/cmdline: {error}"
        ) from error
    try:
        return tuple(
            item.decode("utf-8") for item in raw.split(b"\0") if item
        )
    except UnicodeDecodeError as error:
        raise FallbackError(
            f"cannot decode process command line /proc/{pid}/cmdline"
        ) from error


def _active_group_identities(process_group: int) -> tuple[dict[str, Any], ...]:
    identities: list[dict[str, Any]] = []
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        identity = _proc_identity(int(candidate.name))
        if (
            identity
            and identity["state"] not in _ZOMBIE_STATES
            and identity["pgrp"] == process_group
        ):
            identities.append(identity)
    return tuple(sorted(identities, key=lambda item: int(item["pid"])))


def _wait_dead(identity: Mapping[str, Any], timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _identity_alive(identity):
            return True
        time.sleep(0.05)
    return not _identity_alive(identity)


def _marker_observation(path: Path) -> dict[str, Any]:
    try:
        payload, binding = _bound_json_file(path, maximum_bytes=1 << 16)
    except FileNotFoundError:
        return {"exists": False}
    except FallbackError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            return {"exists": False}
        raise
    if payload.get("schema") != _PAUSE_SCHEMA or not payload.get("token"):
        raise FallbackError(f"invalid pause marker contract: {path}")
    return {
        "exists": True,
        "path": binding["path"],
        "device": binding["device"],
        "inode": binding["inode"],
        "ctime_ns": binding["ctime_ns"],
        "size": binding["size"],
        "sha256": binding["sha256"],
        "payload": payload,
    }


def _create_pause_marker(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return _marker_observation(path)


def _create_release_request(
    path: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish one release request without replacing a prior generation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    _payload, binding = _bound_json_file(path, maximum_bytes=1 << 20)
    return binding


def _remove_bound_json(path: Path, expected: Mapping[str, Any], *, token: str) -> None:
    current_payload, current = _bound_json_file(path, maximum_bytes=1 << 20)
    if current != dict(expected) or str(current_payload.get("token")) != token:
        raise FallbackError(f"refusing to remove replaced JSON authority: {path}")
    quarantine = _release_request_quarantine(path, token)
    if quarantine.exists() or quarantine.is_symlink():
        raise FallbackError(f"release-request quarantine already exists: {quarantine}")
    _rename_noreplace(path, quarantine)
    moved_payload, moved = _bound_json_file(quarantine, maximum_bytes=1 << 20)
    if (
        str(moved_payload.get("token", "")) != token
        or not _renamed_binding_matches(moved, expected)
    ):
        if not path.exists():
            _rename_noreplace(quarantine, path)
        raise FallbackError("release request changed during consumption")
    quarantine.unlink()
    _fsync_directory(path.parent)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise FallbackError("renameat2 is required for pause-marker CAS")
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(source))


def _release_pause_marker(
    path: Path,
    *,
    token: str,
    expected: Mapping[str, Any],
) -> bool:
    current = _marker_observation(path)
    if not current.get("exists"):
        return False
    if current["payload"].get("token") != token or _marker_binding_view(
        current
    ) != dict(expected):
        raise FallbackError("refusing to release a foreign/replaced pause marker")
    quarantine = _pause_marker_quarantine(path, token)
    try:
        quarantine.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FallbackError(
            f"refusing to replace an existing pause-marker quarantine: {quarantine}"
        )
    _rename_noreplace(path, quarantine)
    moved = _marker_observation(quarantine)
    if (
        not moved.get("exists")
        or moved["payload"].get("token") != token
        or not _renamed_binding_matches(_marker_binding_view(moved), expected)
    ):
        if not path.exists():
            _rename_noreplace(quarantine, path)
        raise FallbackError("pause marker changed during release")
    quarantine.unlink()
    _fsync_directory(path.parent)
    return True


def _pause_marker_quarantine(path: Path, token: str) -> Path:
    return path.with_name(f".{path.name}.released.{token}")


def _release_request_quarantine(path: Path, token: str) -> Path:
    return path.with_name(f".{path.name}.consumed.{token.replace('/', '_')}")


def _renamed_binding_matches(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Compare a bound file after its path-only quarantine rename."""

    # A rename is allowed to change ctime and necessarily changes the reported
    # pathname.  Device/inode preserve the object identity; size, digest, and
    # (for release requests) payload preserve the authenticated bytes.
    keys = {"device", "inode", "size", "sha256"}
    if "payload" in expected:
        keys.add("payload")
    return all(observed.get(key) == expected.get(key) for key in keys)


def _acquire_lock(path: Path, *, nonblocking: bool = True) -> int:
    if path.is_symlink():
        raise FallbackError(f"lock path must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    operation = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
    try:
        fcntl.flock(descriptor, operation)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _holder_identity(
    pid_file: Path,
    state_file: Path,
    *,
    watchdog_identity: Mapping[str, Any] | None = None,
    holder_path: Path | None = None,
) -> dict[str, Any] | None:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        state = _load_json(state_file)
    except (FileNotFoundError, OSError, ValueError, FallbackError):
        return None
    identity = _proc_identity(pid)
    if not identity or identity["state"] in _ZOMBIE_STATES:
        return None
    if int(state.get("parent_pid", -1)) != pid or state.get("mode") != "hold":
        return None
    gpu_workers = state.get("gpu_workers", {})
    cpu_workers = state.get("cpu_workers", {})
    if (
        not isinstance(gpu_workers, dict)
        or not isinstance(cpu_workers, dict)
        or len(gpu_workers) != 8
        or len(cpu_workers) != 20
    ):
        return None
    if watchdog_identity is not None:
        if not _identity_alive(watchdog_identity):
            return None
        # The holder is a grandchild below process_bootstrap, but it remains
        # in the bootstrap leader's authenticated process group.
        if identity["pgrp"] != int(watchdog_identity["pid"]):
            return None
    if holder_path is not None and str(holder_path) not in _cmdline(pid):
        return None
    worker_identities: dict[str, tuple[dict[str, Any], ...]] = {}
    all_worker_pids: set[int] = set()
    for kind, raw_workers in (("gpu", gpu_workers), ("cpu", cpu_workers)):
        observed_workers: list[dict[str, Any]] = []
        for raw_pid in raw_workers.values():
            try:
                worker_pid = int(raw_pid)
            except (TypeError, ValueError):
                return None
            if worker_pid <= 0 or worker_pid in all_worker_pids:
                return None
            all_worker_pids.add(worker_pid)
            worker = _proc_identity(worker_pid)
            if (
                not worker
                or worker["state"] in _ZOMBIE_STATES
                or worker["pgrp"] != identity["pgrp"]
            ):
                return None
            observed_workers.append(worker)
        worker_identities[kind] = tuple(
            sorted(observed_workers, key=lambda item: int(item["pid"]))
        )
    identity["gpu_workers"] = len(gpu_workers)
    identity["cpu_workers"] = len(cpu_workers)
    identity["gpu_worker_identities"] = worker_identities["gpu"]
    identity["cpu_worker_identities"] = worker_identities["cpu"]
    return identity


def _published_identities(root: Path) -> tuple[dict[str, Any], ...]:
    if not root.exists():
        return ()
    if root.is_symlink() or not root.is_dir():
        raise FallbackError(f"run identity root is unsafe: {root}")
    identities: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*process-identity.json")):
        payload = _load_json(path)
        try:
            identity = {
                "pid": int(payload["pid"]),
                "start_ticks": str(payload["start_ticks"]),
                "path": str(path),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise FallbackError(f"invalid process identity: {path}") from error
        identities.append(identity)
    return tuple(identities)


def _proc_identity_for_inventory(pid: int) -> dict[str, Any] | None:
    path = Path(f"/proc/{pid}/stat")
    try:
        fields = path.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno in {errno.ENOENT, errno.ESRCH}:
            return None
        raise FallbackError(f"cannot audit process identity {path}: {error}") from error
    except IndexError as error:
        raise FallbackError(f"invalid process stat during inventory: {path}") from error
    return {
        "pid": pid,
        "state": fields[0],
        "ppid": int(fields[1]),
        "pgrp": int(fields[2]),
        "start_ticks": fields[19],
    }


def _proc_environment_for_inventory(pid: int) -> tuple[bytes, ...] | None:
    path = Path(f"/proc/{pid}/environ")
    try:
        return tuple(path.read_bytes().split(b"\0"))
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno in {errno.ENOENT, errno.ESRCH}:
            return None
        raise FallbackError(
            f"cannot audit process environment {path}: {error}"
        ) from error


def _run_owned_processes(run_id: str) -> tuple[dict[str, Any], ...]:
    needles = {
        f"AMG_MULTITASK_RUN_ID={run_id}".encode(),
        f"AGENTMEMORY_RUN_ID={run_id}".encode(),
    }
    found: list[dict[str, Any]] = []
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        pid = int(candidate.name)
        identity = _proc_identity_for_inventory(pid)
        if not identity or identity["state"] in _ZOMBIE_STATES:
            continue
        environment = _proc_environment_for_inventory(pid)
        if environment is None:
            continue
        if needles.intersection(environment):
            found.append(
                {
                    "pid": pid,
                    "start_ticks": identity["start_ticks"],
                    "cmdline": _cmdline(pid),
                }
            )
    return tuple(sorted(found, key=lambda item: int(item["pid"])))


def _process_cpu_ticks(identity: Mapping[str, Any]) -> int | None:
    pid = int(identity["pid"])
    try:
        fields = (
            Path(f"/proc/{pid}/stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .split()
        )
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None
    if fields[0] in _ZOMBIE_STATES or fields[19] != str(identity["start_ticks"]):
        return None
    return int(fields[11]) + int(fields[12])


def _identity_key(identity: Mapping[str, Any]) -> tuple[int, str]:
    return int(identity["pid"]), str(identity["start_ticks"])


def _stable_identity_fields(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pid": int(identity["pid"]),
        "start_ticks": str(identity["start_ticks"]),
        "pgrp": int(identity["pgrp"]),
    }


def _holder_lease_snapshot(holder: Mapping[str, Any]) -> dict[str, Any]:
    """Return only immutable identities from a live holder observation."""

    return {
        "parent": _stable_identity_fields(holder),
        "gpu_workers": sorted(
            (
                _stable_identity_fields(identity)
                for identity in holder.get("gpu_worker_identities", ())
            ),
            key=lambda item: (item["pid"], item["start_ticks"]),
        ),
        "cpu_workers": sorted(
            (
                _stable_identity_fields(identity)
                for identity in holder.get("cpu_worker_identities", ())
            ),
            key=lambda item: (item["pid"], item["start_ticks"]),
        ),
    }


def _holder_lease_matches(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> bool:
    try:
        return _holder_lease_snapshot(expected) == _holder_lease_snapshot(observed)
    except (KeyError, TypeError, ValueError):
        return False


def _descendant_identities(root: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if not _identity_alive(root):
        raise FallbackError("holder root identity is not alive")
    discovered: dict[int, dict[str, Any]] = {}
    frontier = {int(root["pid"])}
    while frontier:
        parent_ids = set(frontier)
        frontier.clear()
        for candidate in Path("/proc").iterdir():
            if not candidate.name.isdigit():
                continue
            identity = _proc_identity_for_inventory(int(candidate.name))
            if (
                identity
                and identity["state"] not in _ZOMBIE_STATES
                and int(identity["ppid"]) in parent_ids
                and int(identity["pid"]) not in discovered
            ):
                discovered[int(identity["pid"])] = identity
                frontier.add(int(identity["pid"]))
    return tuple(sorted(discovered.values(), key=lambda item: int(item["pid"])))


def _read_small_regular_text(path: Path, *, maximum_bytes: int = 65536) -> str:
    _regular_file(path)
    metadata = path.stat()
    if metadata.st_size > maximum_bytes:
        raise FallbackError(f"runtime state file is too large: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise FallbackError(f"cannot read runtime state file {path}: {error}") from error


def _auto_gpu_holder_inventory(
    state_path: Path, *, command_fragment: str
) -> dict[str, Any]:
    raw = _read_small_regular_text(state_path, maximum_bytes=4096).strip()
    match = re.fullmatch(
        r"[^\n]*\bmode=hold\s+pid=(\d+)\s+gpu=(\d+)\s+cpu=(\d+)\s+work=([^\s]+)",
        raw,
    )
    if match is None or int(match.group(2)) != 8:
        raise FallbackError(f"auto GPU holder is not in the expected hold state: {raw!r}")
    identity = _proc_identity_for_inventory(int(match.group(1)))
    if not identity or identity["state"] in _ZOMBIE_STATES:
        raise FallbackError("auto GPU holder parent identity is not alive")
    command = _cmdline(int(identity["pid"]))
    if command_fragment not in command:
        raise FallbackError("auto GPU holder command identity mismatch")
    descendants = _descendant_identities(identity)
    return {
        "state_path": str(state_path),
        "state_sha256": _sha256(state_path),
        "parent": identity,
        "command": command,
        "descendants": descendants,
    }


def _auto_cpu_holder_inventory(
    state_path: Path, *, command_fragment: str
) -> dict[str, Any]:
    state = _load_json(state_path)
    if state.get("state") != "holding":
        raise FallbackError("auto CPU holder is not in the expected holding state")
    try:
        parent_pid = int(state["parent_pid"])
        worker_pids = tuple(int(value) for value in state["worker_pids"])
        expected_workers = int(state["worker_count_requested"])
    except (KeyError, TypeError, ValueError) as error:
        raise FallbackError("invalid auto CPU holder state") from error
    if len(worker_pids) != expected_workers or len(set(worker_pids)) != expected_workers:
        raise FallbackError("auto CPU holder worker inventory mismatch")
    parent = _proc_identity_for_inventory(parent_pid)
    if not parent or parent["state"] in _ZOMBIE_STATES:
        raise FallbackError("auto CPU holder parent identity is not alive")
    command = _cmdline(parent_pid)
    if command_fragment not in command:
        raise FallbackError("auto CPU holder command identity mismatch")
    descendants = {int(item["pid"]): item for item in _descendant_identities(parent)}
    if any(pid not in descendants for pid in worker_pids):
        raise FallbackError("auto CPU holder has missing/non-descendant workers")
    return {
        "state_path": str(state_path),
        "state_sha256": _sha256(state_path),
        "parent": parent,
        "command": command,
        "worker_count": expected_workers,
        "workers": tuple(descendants[pid] for pid in sorted(worker_pids)),
    }


def _nvidia_inventory(
    nvidia_smi: Path, *, expected_sha256: str
) -> dict[str, Any]:
    _regular_file(nvidia_smi, expected_sha256)
    devices = subprocess.run(
        [
            str(nvidia_smi),
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if devices.returncode != 0:
        raise FallbackError(
            f"nvidia-smi GPU inventory failed rc={devices.returncode}: {devices.stderr}"
        )
    device_rows: list[dict[str, Any]] = []
    for raw in devices.stdout.splitlines():
        if not raw.strip():
            continue
        fields = [field.strip() for field in raw.split(",", 2)]
        if len(fields) != 3:
            raise FallbackError(f"invalid nvidia-smi GPU inventory row: {raw!r}")
        device_rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2]})
    if len(device_rows) != 8 or sorted(row["index"] for row in device_rows) != list(range(8)):
        raise FallbackError(f"expected exactly eight GPUs, observed {device_rows!r}")
    applications = subprocess.run(
        [
            str(nvidia_smi),
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if applications.returncode != 0:
        raise FallbackError(
            "nvidia-smi compute inventory failed "
            f"rc={applications.returncode}: {applications.stderr}"
        )
    application_rows: list[dict[str, Any]] = []
    for raw in applications.stdout.splitlines():
        if not raw.strip():
            continue
        fields = [field.strip() for field in raw.split(",", 3)]
        if len(fields) != 4:
            raise FallbackError(f"invalid nvidia-smi compute row: {raw!r}")
        try:
            pid = int(fields[1])
            used_memory = int(fields[3])
        except ValueError as error:
            raise FallbackError(f"invalid nvidia-smi compute row: {raw!r}") from error
        identity = _proc_identity_for_inventory(pid)
        application_rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": pid,
                "process_name": fields[2],
                "used_gpu_memory_mib": used_memory,
                "identity": identity,
                "cmdline": _cmdline(pid) if identity else "",
            }
        )
    return {"devices": tuple(device_rows), "applications": tuple(application_rows)}


_RELEVANT_PROCESS_TOKEN_PATTERN = re.compile(
    r"FullyAsyncTrainer|FullyAsyncRollouter|multitask_orchestrator|raylet|"
    r"sglang|vllm|MemoryArena|swesmith|literesearcher|openmle_fast",
    re.IGNORECASE,
)


def _looks_like_relevant_workload(argv: Sequence[str]) -> bool:
    """Classify executable/module tokens, never arbitrary command text.

    A substring search over the flattened command line is unsafe here.  The
    Python runtime directory itself can contain ``sglang`` and a read-only
    ``bash -c`` audit can mention environment names in data arguments.  Those
    processes are not workloads.  Run-owned descendants are still caught by
    their mandatory owner tags; this fallback only recognizes an actual
    executable, ``python -m`` module, or executed script.
    """

    if not argv:
        return False
    executable = Path(argv[0]).name
    if _RELEVANT_PROCESS_TOKEN_PATTERN.search(executable):
        return True
    if executable in {"bash", "sh", "zsh", "dash", "ksh"}:
        if len(argv) < 2 or argv[1] in {"-c", "-lc", "-ic"}:
            return False
        return bool(_RELEVANT_PROCESS_TOKEN_PATTERN.search(argv[1]))
    if not executable.startswith(("python", "pypy")):
        return False
    for index, token in enumerate(argv[1:], start=1):
        if token == "-m" and index + 1 < len(argv):
            return bool(_RELEVANT_PROCESS_TOKEN_PATTERN.search(argv[index + 1]))
        if token in {"-c", "-"}:
            return False
        if token.endswith((".py", ".pyz")):
            return bool(_RELEVANT_PROCESS_TOKEN_PATTERN.search(token))
        if token.startswith("-"):
            continue
        # The first positional Python argument is the executed script.  Do
        # not scan later data arguments, which may merely name an environment.
        return False
    return False


def _relevant_processes(
    *, allowed_identities: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    allowed = {_identity_key(identity) for identity in allowed_identities}
    found: list[dict[str, Any]] = []
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        pid = int(candidate.name)
        identity = _proc_identity_for_inventory(pid)
        if not identity or identity["state"] in _ZOMBIE_STATES:
            continue
        if _identity_key(identity) in allowed:
            continue
        argv = _process_argv(pid)
        command = " ".join(argv)
        environment = _proc_environment_for_inventory(pid)
        if environment is None:
            continue
        owner_tags = tuple(
            sorted(
                value.decode("utf-8", errors="replace").split("=", 1)[0]
                for value in environment
                if value.startswith((b"AMG_MULTITASK_RUN_ID=", b"AGENTMEMORY_RUN_ID="))
            )
        )
        if owner_tags or _looks_like_relevant_workload(argv):
            found.append(
                {
                    **identity,
                    "cmdline": command,
                    "owner_tags": owner_tags,
                }
            )
    return tuple(sorted(found, key=lambda item: int(item["pid"])))


def _pod_preflight_inventory(
    *,
    nvidia_smi: Path,
    nvidia_smi_sha256: str,
    auto_gpu_holder_state: Path,
    auto_gpu_holder_command_fragment: str,
    auto_cpu_holder_state: Path,
    auto_cpu_holder_command_fragment: str,
    allowed_identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gpu_holder = _auto_gpu_holder_inventory(
        auto_gpu_holder_state,
        command_fragment=auto_gpu_holder_command_fragment,
    )
    cpu_holder = _auto_cpu_holder_inventory(
        auto_cpu_holder_state,
        command_fragment=auto_cpu_holder_command_fragment,
    )
    hardware = _nvidia_inventory(
        nvidia_smi, expected_sha256=nvidia_smi_sha256
    )
    allowed_gpu_identities = {
        _identity_key(gpu_holder["parent"]),
        *(_identity_key(item) for item in gpu_holder["descendants"]),
    }
    transient_gpu_pids = tuple(
        int(item["pid"])
        for item in hardware["applications"]
        if item["identity"] is None
    )
    foreign_gpu = tuple(
        item
        for item in hardware["applications"]
        if item["identity"] is not None
        and _identity_key(item["identity"]) not in allowed_gpu_identities
    )
    allowed_process_identities = (
        *(item for item in allowed_identities if _identity_alive(item)),
        gpu_holder["parent"],
        *gpu_holder["descendants"],
        cpu_holder["parent"],
        *cpu_holder["workers"],
    )
    relevant = _relevant_processes(
        allowed_identities=allowed_process_identities
    )
    safe = not transient_gpu_pids and not foreign_gpu and not relevant
    return {
        "schema": "amg_pod_preflight_inventory_v1",
        "status": "pass" if safe else "fail",
        "safe": safe,
        "auto_gpu_holder": gpu_holder,
        "auto_cpu_holder": cpu_holder,
        "hardware": hardware,
        "transient_gpu_pids": transient_gpu_pids,
        "foreign_gpu_applications": foreign_gpu,
        "foreign_relevant_processes": relevant,
        "updated_unix": time.time(),
    }


def _holder_resource_attestation(
    holder: Mapping[str, Any],
    *,
    nvidia_smi: Path,
    nvidia_smi_sha256: str,
    cpu_sample_seconds: float = 0.25,
    gpu_sample_count: int = 5,
    gpu_sample_seconds: float = 0.2,
) -> dict[str, Any]:
    gpu_workers = tuple(holder.get("gpu_worker_identities", ()))
    cpu_workers = tuple(holder.get("cpu_worker_identities", ()))
    if len(gpu_workers) != 8 or len(cpu_workers) != 20:
        raise FallbackError("fallback holder worker identity inventory is incomplete")
    before = {_identity_key(item): _process_cpu_ticks(item) for item in cpu_workers}
    time.sleep(cpu_sample_seconds)
    after = {_identity_key(item): _process_cpu_ticks(item) for item in cpu_workers}
    progressed = tuple(
        key
        for key, first in before.items()
        if first is not None and after.get(key) is not None and int(after[key]) > int(first)
    )
    if len(progressed) != len(cpu_workers):
        raise FallbackError(
            "fallback CPU holder workers did not all show execution progress: "
            f"progressed={len(progressed)} expected={len(cpu_workers)}"
        )
    hardware = _nvidia_inventory(
        nvidia_smi, expected_sha256=nvidia_smi_sha256
    )
    expected_gpu_identities = {_identity_key(item) for item in gpu_workers}
    observed_rows = tuple(
        item
        for item in hardware["applications"]
        if item["identity"] is not None
        and _identity_key(item["identity"]) in expected_gpu_identities
    )
    observed_gpu_identities = {
        _identity_key(item["identity"]) for item in observed_rows
    }
    observed_gpu_uuids = {str(item["gpu_uuid"]) for item in observed_rows}
    if (
        observed_gpu_identities != expected_gpu_identities
        or len(observed_rows) != len(gpu_workers)
        or len(observed_gpu_uuids) != 8
        or any(int(item["used_gpu_memory_mib"]) <= 0 for item in observed_rows)
    ):
        raise FallbackError(
            "fallback GPU holder workers are not resident one-per-device: "
            f"expected={sorted(expected_gpu_identities)} rows={observed_rows!r}"
        )
    utilization_by_uuid: dict[str, list[int]] = {
        str(row["uuid"]): [] for row in hardware["devices"]
    }
    for sample_index in range(gpu_sample_count):
        sample = subprocess.run(
            [
                str(nvidia_smi),
                "--query-gpu=uuid,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if sample.returncode != 0:
            raise FallbackError(
                "nvidia-smi utilization sample failed "
                f"rc={sample.returncode}: {sample.stderr}"
            )
        seen: set[str] = set()
        for raw in sample.stdout.splitlines():
            if not raw.strip():
                continue
            fields = [field.strip() for field in raw.split(",", 1)]
            if len(fields) != 2 or fields[0] not in utilization_by_uuid:
                raise FallbackError(
                    f"invalid nvidia-smi utilization row: {raw!r}"
                )
            try:
                utilization = int(fields[1])
            except ValueError as error:
                raise FallbackError(
                    f"invalid nvidia-smi utilization row: {raw!r}"
                ) from error
            utilization_by_uuid[fields[0]].append(utilization)
            seen.add(fields[0])
        if seen != set(utilization_by_uuid):
            raise FallbackError(
                "nvidia-smi utilization sample did not cover every GPU"
            )
        if sample_index + 1 < gpu_sample_count:
            time.sleep(gpu_sample_seconds)
    utilization_max = {
        uuid: max(samples) if samples else 0
        for uuid, samples in utilization_by_uuid.items()
    }
    if any(value < 5 for value in utilization_max.values()):
        raise FallbackError(
            "fallback GPU holder did not show >=5% compute activity on every GPU: "
            f"{utilization_max!r}"
        )
    return {
        "schema": "amg_fallback_holder_activity_v1",
        "status": "pass",
        "cpu_progressed_workers": len(progressed),
        "cpu_worker_count": len(cpu_workers),
        "gpu_worker_count": len(observed_gpu_identities),
        "gpu_uuid_count": len(observed_gpu_uuids),
        "gpu_rows": observed_rows,
        "gpu_utilization_samples": utilization_by_uuid,
        "gpu_utilization_max": utilization_max,
        "updated_unix": time.time(),
    }


def _listening_ports(ports: Sequence[int]) -> tuple[int, ...]:
    wanted = set(int(port) for port in ports)
    found: set[int] = set()
    for raw in ("/proc/net/tcp", "/proc/net/tcp6"):
        path = Path(raw)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":
                continue
            port = int(fields[1].rsplit(":", 1)[1], 16)
            if port in wanted:
                found.add(port)
    return tuple(sorted(found))


def _owned_mounts(run_id: str, run_dir: Path) -> tuple[str, ...]:
    path = Path("/proc/self/mountinfo")
    if not path.is_file():
        return ()
    needles = (run_id, str(run_dir), f".swesmith-sandbox-root-{run_id}")
    return tuple(
        line
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if any(needle and needle in line for needle in needles)
    )


def _read_marker_value(path: Path) -> str | None:
    if path.is_symlink():
        raise FallbackError(f"holder marker must not be a symlink: {path}")
    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise FallbackError(f"cannot open holder marker {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FallbackError(f"holder marker must be a regular file: {path}")
        raw = os.read(descriptor, 4097)
        if len(raw) > 4096:
            raise FallbackError(f"holder marker exceeds 4096 bytes: {path}")
        return raw.decode("utf-8").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise FallbackError(f"cannot read holder marker {path}: {error}") from error
    finally:
        os.close(descriptor)


def _inner_holder_restore_report(run_dir: Path) -> dict[str, Any]:
    transaction_dir = run_dir / "holder-transaction"
    if not transaction_dir.exists():
        return {"present": False, "safe": True}
    if transaction_dir.is_symlink() or not transaction_dir.is_dir():
        raise FallbackError(
            f"inner holder transaction path is unsafe: {transaction_dir}"
        )
    state_path = transaction_dir / "state.json"
    watcher_receipt_path = transaction_dir / "watcher-exit.json"
    state = _load_json(state_path)
    receipt = _load_json(watcher_receipt_path)
    if state.get("schema") != "amg_marker_transaction_v1":
        raise FallbackError("inner holder transaction schema mismatch")
    if state.get("status") != "restored":
        raise FallbackError(
            "inner holder transaction is not restored: "
            f"{state.get('status')!r}"
        )
    if (
        receipt.get("schema") != "amg_marker_watcher_exit_v1"
        or receipt.get("status") != "pass"
        or receipt.get("state_status") != "restored"
    ):
        raise FallbackError(
            "inner holder watcher did not attest restored state: "
            f"{receipt!r}"
        )
    markers = state.get("markers")
    if not isinstance(markers, list) or len(markers) != len(_INNER_HOLDER_MARKERS):
        raise FallbackError("inner holder transaction marker inventory mismatch")
    observations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for marker in markers:
        if not isinstance(marker, dict):
            raise FallbackError("inner holder marker must be a JSON object")
        name = str(marker.get("name", ""))
        expected_path = _INNER_HOLDER_MARKERS.get(name)
        if expected_path is None or name in seen:
            raise FallbackError(f"unexpected inner holder marker name: {name!r}")
        seen.add(name)
        observed_path = Path(str(marker.get("path", "")))
        if observed_path != expected_path:
            raise FallbackError(
                f"inner holder marker path mismatch for {name}: {observed_path}"
            )
        if marker.get("restored") is not True or marker.get("restore_target_set") is not True:
            raise FallbackError(f"inner holder marker is not restored: {name}")
        expected_value = marker.get("restore_target")
        if expected_value is not None and not isinstance(expected_value, str):
            raise FallbackError(f"invalid restore target for inner marker {name}")
        observed_value = _read_marker_value(expected_path)
        if observed_value != expected_value:
            raise FallbackError(
                f"inner holder marker restore mismatch for {name}: "
                f"expected={expected_value!r} observed={observed_value!r}"
            )
        observations.append(
            {
                "name": name,
                "path": str(expected_path),
                "expected_value": expected_value,
                "observed_value": observed_value,
            }
        )
    if seen != set(_INNER_HOLDER_MARKERS):
        raise FallbackError("inner holder transaction is missing required markers")
    return {
        "present": True,
        "safe": True,
        "state_status": state["status"],
        "watcher_status": receipt["status"],
        "watcher_mode": receipt.get("mode"),
        "markers": observations,
    }


def _drain_report(marker: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(marker.get("run_id", ""))
    run_dir = Path(str(marker.get("run_dir", "/nonexistent")))
    protected = tuple(marker.get("protected_identities", ()))
    alive_protected = tuple(item for item in protected if _identity_alive(item))
    orchestrator: tuple[dict[str, Any], ...] = ()
    inventory_errors: list[str] = []
    orchestrator_path_raw = marker.get("orchestrator_identity")
    orchestrator_binding = marker.get("orchestrator_identity_binding")
    if orchestrator_path_raw and isinstance(orchestrator_binding, dict):
        orchestrator_path = Path(str(orchestrator_path_raw))
        try:
            current_binding = _process_identity_file_binding(orchestrator_path)
            if current_binding != orchestrator_binding:
                raise FallbackError(
                    f"bound orchestrator identity was replaced: {orchestrator_path}"
                )
            payload = orchestrator_binding["payload"]
            identity = {
                "pid": int(payload["pid"]),
                "start_ticks": str(payload["start_ticks"]),
                "path": str(orchestrator_path),
            }
            if _identity_alive(identity):
                orchestrator = (identity,)
        except Exception as error:
            inventory_errors.append(
                f"orchestrator lease: {type(error).__name__}: {error}"
            )
    elif orchestrator_path_raw:
        inventory_errors.append("orchestrator identity is not durably bound")
    published = _published_identities(run_dir)
    alive_published = tuple(item for item in published if _identity_alive(item))
    try:
        owner_processes = _run_owned_processes(run_id) if run_id else ()
    except Exception as error:
        owner_processes = ()
        inventory_errors.append(
            f"process inventory: {type(error).__name__}: {error}"
        )
    ports = _listening_ports(tuple(int(value) for value in marker.get("ports", ())))
    mounts = _owned_mounts(run_id, run_dir) if run_id else ()
    try:
        inner_holder_restore = _inner_holder_restore_report(run_dir)
    except Exception as error:
        inner_holder_restore = {
            "present": (run_dir / "holder-transaction").exists(),
            "safe": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "safe": not (
            alive_protected
            or orchestrator
            or alive_published
            or owner_processes
            or ports
            or mounts
            or inventory_errors
            or not inner_holder_restore["safe"]
        ),
        "alive_protected": alive_protected,
        "alive_orchestrator": orchestrator,
        "alive_published": alive_published,
        "owned_processes": owner_processes,
        "listening_ports": ports,
        "owned_mounts": mounts,
        "inventory_errors": inventory_errors,
        "inner_holder_restore": inner_holder_restore,
    }


def _state_payload(mode: str, **extra: Any) -> dict[str, Any]:
    identity = _proc_identity(os.getpid())
    return {
        "schema": _STATE_SCHEMA,
        "mode": mode,
        "supervisor": identity,
        "updated_unix": time.time(),
        **extra,
    }


def _same_process_lease(
    expected: Mapping[str, Any] | None,
    observed: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        return False
    try:
        return _stable_identity_fields(expected) == _stable_identity_fields(observed)
    except (KeyError, TypeError, ValueError):
        return False


def _supervisor_generation_matches(
    state: Mapping[str, Any],
    *,
    expected_supervisor: Mapping[str, Any],
    expected_generation: int,
) -> bool:
    try:
        generation = int(state.get("generation", -1))
    except (TypeError, ValueError):
        return False
    return generation == expected_generation and _same_process_lease(
        expected_supervisor, state.get("supervisor")
    )


def _marker_binding_view(observation: Mapping[str, Any]) -> dict[str, Any]:
    if observation.get("exists") is not True:
        raise FallbackError("pause marker is not present")
    return {
        key: observation[key]
        for key in ("path", "device", "inode", "ctime_ns", "size", "sha256")
    }


def _pending_resume_bindings(
    pending_resume: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    if pending_resume.get("schema") != _PENDING_RESUME_SCHEMA:
        raise FallbackError("pending resume schema mismatch")
    token = str(pending_resume.get("token", ""))
    marker_binding = pending_resume.get("marker_binding")
    request_binding = pending_resume.get("request_binding")
    if not token or not isinstance(marker_binding, dict):
        raise FallbackError("pending resume is missing its marker binding")
    if request_binding is not None and not isinstance(request_binding, dict):
        raise FallbackError("pending resume release-request binding is invalid")
    marker_binding = dict(marker_binding)
    if set(marker_binding) != {
        "path",
        "device",
        "inode",
        "ctime_ns",
        "size",
        "sha256",
    }:
        raise FallbackError("pending resume pause-marker binding is invalid")
    if str(marker_binding["path"]) == "" or not re.fullmatch(
        r"[0-9a-f]{64}", str(marker_binding["sha256"])
    ):
        raise FallbackError("pending resume pause-marker binding is invalid")
    try:
        if any(
            int(marker_binding[key]) < 0
            for key in ("device", "inode", "ctime_ns", "size")
        ):
            raise ValueError
    except (TypeError, ValueError) as error:
        raise FallbackError(
            "pending resume pause-marker binding is invalid"
        ) from error
    if request_binding is not None:
        request_binding = dict(request_binding)
        if set(request_binding) != {
            "path",
            "device",
            "inode",
            "ctime_ns",
            "size",
            "sha256",
            "payload",
        } or not isinstance(request_binding.get("payload"), dict):
            raise FallbackError(
                "pending resume release-request binding is invalid"
            )
        if (
            str(request_binding["path"]) == ""
            or str(request_binding["payload"].get("token", "")) != token
            or not re.fullmatch(r"[0-9a-f]{64}", str(request_binding["sha256"]))
        ):
            raise FallbackError(
                "pending resume release-request binding is invalid"
            )
        try:
            if any(
                int(request_binding[key]) < 0
                for key in ("device", "inode", "ctime_ns", "size")
            ):
                raise ValueError
        except (TypeError, ValueError) as error:
            raise FallbackError(
                "pending resume release-request binding is invalid"
            ) from error
    return token, marker_binding, request_binding


def _finish_renamed_marker_release(
    *, quarantine: Path, expected: Mapping[str, Any], token: str
) -> None:
    observed = _marker_observation(quarantine)
    if (
        not observed.get("exists")
        or str(observed["payload"].get("token", "")) != token
        or not _renamed_binding_matches(_marker_binding_view(observed), expected)
    ):
        raise FallbackError(
            "refusing to finish a foreign/replaced pause-marker quarantine"
        )
    quarantine.unlink()
    _fsync_directory(quarantine.parent)


def _finish_renamed_request_consumption(
    *, quarantine: Path, expected: Mapping[str, Any], token: str
) -> None:
    payload, observed = _bound_json_file(quarantine, maximum_bytes=1 << 20)
    if (
        str(payload.get("token", "")) != token
        or not _renamed_binding_matches(observed, expected)
    ):
        raise FallbackError(
            "refusing to finish a foreign/replaced release-request quarantine"
        )
    quarantine.unlink()
    _fsync_directory(quarantine.parent)


def _assert_pending_resume_publishable(
    *,
    pause_path: Path,
    release_request_path: Path,
    pending_resume: Mapping[str, Any],
) -> None:
    """Prove the release transaction is complete before publishing holding."""

    token, _marker_binding, request_binding = _pending_resume_bindings(
        pending_resume
    )
    marker = _marker_observation(pause_path)
    request_present = release_request_path.exists() or release_request_path.is_symlink()
    marker_quarantine = _pause_marker_quarantine(pause_path, token)
    request_quarantine = _release_request_quarantine(release_request_path, token)
    marker_quarantine_present = (
        marker_quarantine.exists() or marker_quarantine.is_symlink()
    )
    request_quarantine_present = (
        request_binding is not None
        and (request_quarantine.exists() or request_quarantine.is_symlink())
    )
    if (
        marker.get("exists")
        or request_present
        or marker_quarantine_present
        or request_quarantine_present
    ):
        raise FallbackError(
            "pending resume release transaction is not complete: "
            f"marker_exists={bool(marker.get('exists'))} "
            f"request_exists={request_present} "
            f"marker_quarantine_exists={marker_quarantine_present} "
            f"request_quarantine_exists={request_quarantine_present}"
        )


def _complete_pending_resume_release(
    *,
    pause_path: Path,
    release_request_path: Path,
    pending_resume: Mapping[str, Any],
) -> None:
    """Idempotently finish one already-authorized marker release.

    Authorization, including the exact marker and optional request bindings,
    is persisted before either filesystem authority is removed.  A replacement
    supervisor therefore resumes this transaction rather than discarding it or
    accepting a stale release request.
    """

    token, marker_binding, request_binding = _pending_resume_bindings(
        pending_resume
    )
    marker_quarantine = _pause_marker_quarantine(pause_path, token)
    request_quarantine = _release_request_quarantine(release_request_path, token)
    marker = _marker_observation(pause_path)
    if marker.get("exists"):
        if _marker_binding_view(marker) != marker_binding:
            raise FallbackError(
                "pending resume pause marker was replaced before release"
            )
        _release_pause_marker(
            pause_path,
            token=token,
            expected=marker_binding,
        )
    elif marker_quarantine.exists() or marker_quarantine.is_symlink():
        _finish_renamed_marker_release(
            quarantine=marker_quarantine,
            expected=marker_binding,
            token=token,
        )

    request_present = release_request_path.exists() or release_request_path.is_symlink()
    if request_present:
        if request_binding is None:
            raise FallbackError(
                "release request exists without a pending-resume binding"
            )
        _remove_bound_json(
            release_request_path,
            request_binding,
            token=token,
        )
    elif request_binding is not None and (
        request_quarantine.exists() or request_quarantine.is_symlink()
    ):
        _finish_renamed_request_consumption(
            quarantine=request_quarantine,
            expected=request_binding,
            token=token,
        )

    _assert_pending_resume_publishable(
        pause_path=pause_path,
        release_request_path=release_request_path,
        pending_resume=pending_resume,
    )


def _load_release_request(
    path: Path,
    *,
    marker: Mapping[str, Any],
    expected_contract: Mapping[str, str],
    expected_supervisor: Mapping[str, Any],
    expected_generation: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, binding = _bound_json_file(path, maximum_bytes=1 << 20)
    marker_payload = marker["payload"]
    if payload.get("schema") != _RELEASE_REQUEST_SCHEMA:
        raise FallbackError("pause-release request schema mismatch")
    if payload.get("token") != marker_payload.get("token"):
        raise FallbackError("pause-release request token mismatch")
    if payload.get("immutable_contract") != dict(expected_contract):
        raise FallbackError("pause-release request immutable contract mismatch")
    if payload.get("marker_binding") != _marker_binding_view(marker):
        raise FallbackError("pause-release request marker binding mismatch")
    if not _same_process_lease(payload.get("owner"), marker_payload.get("owner")):
        raise FallbackError("pause-release request owner mismatch")
    if not _supervisor_generation_matches(
        {
            "supervisor": payload.get("expected_supervisor"),
            "generation": payload.get("expected_generation"),
        },
        expected_supervisor=expected_supervisor,
        expected_generation=expected_generation,
    ):
        raise FallbackError("pause-release request supervisor generation mismatch")
    return payload, binding


def _publish_release_request(
    path: Path,
    *,
    marker: Mapping[str, Any],
    paused_state: Mapping[str, Any],
    owner: Mapping[str, Any],
    expected_contract: Mapping[str, str],
    mode: str,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise FallbackError(f"pause-release request already exists: {path}")
    try:
        generation = int(paused_state["generation"])
        supervisor = paused_state["supervisor"]
    except (KeyError, TypeError, ValueError) as error:
        raise FallbackError("paused supervisor state has no generation lease") from error
    if not _identity_alive(supervisor):
        raise FallbackError("paused supervisor generation is not alive")
    payload = {
        "schema": _RELEASE_REQUEST_SCHEMA,
        "token": marker["payload"]["token"],
        "mode": mode,
        "owner": dict(owner),
        "expected_supervisor": dict(supervisor),
        "expected_generation": generation,
        "marker_binding": _marker_binding_view(marker),
        "immutable_contract": dict(expected_contract),
        "created_unix": time.time(),
    }
    return _create_release_request(path, payload)


def _stop_managed_watchdog(
    process: subprocess.Popen[bytes],
    identity: Mapping[str, Any] | None,
    *,
    timeout_seconds: float,
) -> None:
    # ``process`` is the exact, authenticated process_bootstrap child.  That
    # bootstrap is a subreaper and does not exit until all of its descendants
    # (including setsid children) have drained.  Never signal a numeric PGID
    # after the leader may have been reaped: it could already belong to an
    # unrelated workload.
    if _identity_alive(identity):
        _signal_identity(identity or {}, signal.SIGTERM)
    elif process.poll() is None:
        # ``process`` is the exact Popen object created by this supervisor.
        # Its PID cannot refer to a foreign process until it has been reaped.
        process.terminate()
    try:
        # The bootstrap starts its SIGKILL phase only after
        # ``timeout_seconds``.  Give that subreaper a separate grace window to
        # reap daemonized/setsid descendants; killing the anchor at the same
        # deadline would lose the only complete descendant inventory.
        process.wait(timeout=timeout_seconds + _BOOTSTRAP_DRAIN_GRACE_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise FallbackError(
            "authenticated fallback watchdog did not drain its descendants"
        ) from error


def _clear_drained_watchdog_artifacts(
    *,
    identity_path: Path,
    expected_lease: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    pid_file: Path,
    holder_state_file: Path,
    holder_path: Path,
) -> None:
    """Remove one generation's files only after exact drain is proved.

    These files are the durable recovery authority for a platform restart.
    Never erase them merely because the current supervisor is exiting: a
    failed cleanup must leave enough evidence for the next generation to
    authenticate and finish draining the old holder.
    """

    current_lease = _process_identity_file_binding(identity_path)
    if current_lease != dict(expected_lease):
        raise FallbackError(
            "refusing to clear a replaced fallback watchdog identity"
        )
    if _identity_alive(expected_identity):
        raise FallbackError(
            "refusing to clear a live fallback watchdog identity"
        )
    holder = _holder_identity(
        pid_file,
        holder_state_file,
        holder_path=holder_path,
    )
    if holder is not None:
        raise FallbackError("refusing to clear a live fallback holder identity")
    if pid_file.exists():
        try:
            stale_pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as error:
            raise FallbackError(
                "fallback pid file is not safely attributable to a drained holder"
            ) from error
        observed = _proc_identity(stale_pid)
        if observed and observed["state"] not in _ZOMBIE_STATES:
            raise FallbackError(
                "fallback pid file still names a live unauthenticated process"
            )
    if holder_state_file.exists():
        stale_state = _load_json(holder_state_file)
        try:
            stale_parent = int(stale_state["parent_pid"])
        except (KeyError, TypeError, ValueError) as error:
            raise FallbackError(
                "fallback state file is not safely attributable to a drained holder"
            ) from error
        observed = _proc_identity(stale_parent)
        if observed and observed["state"] not in _ZOMBIE_STATES:
            raise FallbackError(
                "fallback state file still names a live unauthenticated process"
            )
    identity_path.unlink()
    with contextlib.suppress(FileNotFoundError):
        pid_file.unlink()
    with contextlib.suppress(FileNotFoundError):
        holder_state_file.unlink()


def _validate_process_identity_payload(
    payload: Mapping[str, Any],
    *,
    path: Path,
    expected_name: str | None = None,
    expected_bootstrap: Path | None = None,
    expected_command: Sequence[str] | None = None,
    expected_contract: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if payload.get("schema") != _PROCESS_IDENTITY_SCHEMA:
        raise FallbackError(f"process identity schema mismatch: {path}")
    try:
        identity = {
            "pid": int(payload["pid"]),
            "start_ticks": str(payload["start_ticks"]),
            "pgrp": int(payload.get("process_group", payload["pid"])),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise FallbackError(f"invalid process identity: {path}") from error
    if identity["pid"] <= 0 or identity["pgrp"] <= 0 or not identity["start_ticks"]:
        raise FallbackError(f"invalid process identity: {path}")
    if identity["pgrp"] != identity["pid"]:
        raise FallbackError(f"process identity is not a group leader: {path}")
    if expected_name is not None and payload.get("name") != expected_name:
        raise FallbackError(f"process identity name mismatch: {path}")
    if expected_bootstrap is not None and payload.get("bootstrap") != str(
        expected_bootstrap
    ):
        raise FallbackError(f"process identity bootstrap mismatch: {path}")
    if expected_command is not None and payload.get("command") != list(
        expected_command
    ):
        raise FallbackError(f"process identity command mismatch: {path}")
    if expected_contract is not None and payload.get("immutable_contract") != dict(
        expected_contract
    ):
        raise FallbackError(f"process identity immutable contract mismatch: {path}")
    if _identity_alive(identity) and expected_bootstrap is not None:
        try:
            argv = tuple(
                part.decode("utf-8")
                for part in Path(f"/proc/{identity['pid']}/cmdline")
                .read_bytes()
                .split(b"\0")
                if part
            )
        except (OSError, UnicodeDecodeError) as error:
            raise FallbackError(
                f"cannot authenticate process identity command line: {path}"
            ) from error
        command = tuple(expected_command or ())
        if len(argv) < 2 or argv[1] != str(expected_bootstrap) or (
            command and argv[-len(command) :] != command
        ):
            raise FallbackError(f"live process identity command line mismatch: {path}")
    return identity


def _load_process_identity(
    path: Path,
    *,
    expected_name: str | None = None,
    expected_bootstrap: Path | None = None,
    expected_command: Sequence[str] | None = None,
    expected_contract: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    payload, _binding = _bound_json_file(path)
    return _validate_process_identity_payload(
        payload,
        path=path,
        expected_name=expected_name,
        expected_bootstrap=expected_bootstrap,
        expected_command=expected_command,
        expected_contract=expected_contract,
    )


def _process_identity_file_binding(path: Path) -> dict[str, Any]:
    """Bind a durable process lease to both bytes and inode identity."""

    payload, binding = _bound_json_file(path)
    if payload.get("schema") != _PROCESS_IDENTITY_SCHEMA:
        raise FallbackError(f"process identity schema mismatch: {path}")
    return binding


def _load_bound_previous_watchdog(
    *,
    state_path: Path,
    identity_path: Path,
    expected_bootstrap: Path,
    expected_command: Sequence[str],
    expected_contract: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate a previous generation without trusting a replaceable path."""

    previous_state = _load_json(state_path)
    if previous_state.get("schema") != _STATE_SCHEMA:
        raise FallbackError("previous fallback supervisor state schema mismatch")
    if previous_state.get("immutable_contract") != dict(expected_contract):
        raise FallbackError("previous fallback immutable contract mismatch")
    expected_binding = previous_state.get("watchdog_lease")
    if not isinstance(expected_binding, dict):
        raise FallbackError("previous fallback watchdog lease is not bound")
    current_binding = _process_identity_file_binding(identity_path)
    if current_binding != expected_binding:
        raise FallbackError("persisted fallback watchdog identity was replaced")
    identity = _validate_process_identity_payload(
        current_binding["payload"],
        path=identity_path,
        expected_name="fallback-holder-watchdog",
        expected_bootstrap=expected_bootstrap,
        expected_command=expected_command,
        expected_contract=expected_contract,
    )
    saved_identity = previous_state.get("watchdog")
    if not isinstance(saved_identity, dict) or any(
        str(saved_identity.get(key)) != str(identity.get(key))
        for key in ("pid", "start_ticks", "pgrp")
    ):
        raise FallbackError("previous fallback watchdog state/lease mismatch")
    return identity, current_binding, previous_state


def _drain_orphaned_watchdog_group(
    identity: Mapping[str, Any],
    *,
    authenticated_members: Sequence[Mapping[str, Any]] = (),
    timeout_seconds: float,
) -> None:
    """Drain a previous generation using exact leases only.

    The authenticated bootstrap is a subreaper and owns descendant cleanup.
    Persisted holder/worker identities are a fallback for an already-dead
    bootstrap.  Numeric process-group membership is evidence only; it is never
    signal authority because PGIDs can be reused.
    """

    process_group = int(identity.get("pgrp", identity["pid"]))
    authorized: dict[tuple[int, str], Mapping[str, Any]] = {}
    live_anchors: list[Mapping[str, Any]] = []
    for anchor in (identity, *authenticated_members):
        authorized[_identity_key(anchor)] = anchor
        if _identity_alive(anchor):
            observed = _proc_identity(int(anchor["pid"]))
            if observed is None or int(observed["pgrp"]) != process_group:
                raise FallbackError(
                    "authenticated orphan-drain identity changed process group"
                )
            live_anchors.append(anchor)
        for key in ("gpu_worker_identities", "cpu_worker_identities"):
            for worker in anchor.get(key, ()):
                authorized[_identity_key(worker)] = worker
    if live_anchors:
        # While an exact member of this session is still alive, every current
        # member of its process group belongs to that generation.  Snapshot
        # exact PID/start-ticks leases, then revalidate the anchor before any
        # signal so a dying leader cannot turn a reused numeric PGID into
        # authority over a foreign process.
        group_snapshot = _active_group_identities(process_group)
        if not any(
            _identity_alive(anchor)
            and (observed := _proc_identity(int(anchor["pid"]))) is not None
            and int(observed["pgrp"]) == process_group
            for anchor in live_anchors
        ):
            raise FallbackError(
                "authenticated orphan-drain anchor died during group snapshot"
            )
        for member in group_snapshot:
            authorized[_identity_key(member)] = member
    for member in authorized.values():
        if _identity_alive(member):
            _signal_identity(member, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not any(_identity_alive(member) for member in authorized.values()):
            break
        time.sleep(0.05)
    for member in authorized.values():
        if _identity_alive(member):
            _signal_identity(member, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not any(_identity_alive(member) for member in authorized.values()):
            break
        time.sleep(0.05)
    alive = tuple(member for member in authorized.values() if _identity_alive(member))
    if alive:
        raise FallbackError(
            "orphaned fallback watchdog identities did not drain: " f"{alive!r}"
        )
    remaining = _active_group_identities(process_group)
    if remaining:
        raise FallbackError(
            "refusing to act on a stale numeric process group with "
            "unauthenticated members: "
            f"pgrp={process_group} members={remaining!r}"
        )


def supervise(args: argparse.Namespace) -> int:
    if not Path("/proc/self/stat").is_file():
        raise FallbackError("fallback supervisor requires Linux /proc")
    _regular_file(Path(args.python), args.python_sha256)
    original_path = _regular_file(Path(args.original), args.original_sha256)
    holder_path = _regular_file(Path(args.holder), args.holder_sha256)
    wrapper_path = _regular_file(
        Path(args.watchdog_wrapper), args.watchdog_wrapper_sha256
    )
    supervisor_path = _regular_file(
        Path(args.supervisor_script), args.supervisor_script_sha256
    )
    bootstrap_path = _regular_file(Path(args.bootstrap), args.bootstrap_sha256)
    _regular_file(Path(args.nvidia_smi), args.nvidia_smi_sha256)
    if supervisor_path.resolve() != Path(__file__).resolve():
        raise FallbackError(
            "running fallback supervisor does not match --supervisor-script"
        )
    lock_fd = _acquire_lock(Path(args.supervisor_lock))
    state_path = Path(args.supervisor_state)
    pause_path = Path(args.pause_marker)
    release_request_path = Path(args.release_request_file)
    pid_file = Path(args.pid_file)
    holder_state_file = Path(args.holder_state_file)
    log_path = Path(args.watchdog_log)
    watchdog_identity_path = Path(args.watchdog_identity_file)
    expected_contract = _expected_contract(args)
    self_identity = _capture_identity(os.getpid())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    child: subprocess.Popen[bytes] | None = None
    child_identity: dict[str, Any] | None = None
    child_lease: dict[str, Any] | None = None
    managed_generation = False
    generation = 0
    marker_observation: dict[str, Any] | None = None
    pending_resume: dict[str, Any] | None = None

    def write(mode: str, **extra: Any) -> None:
        if child_lease is not None:
            observed_lease = _process_identity_file_binding(watchdog_identity_path)
            if observed_lease != child_lease:
                raise FallbackError(
                    "managed fallback watchdog identity changed after publication"
                )
        holder_identity = _holder_identity(
            pid_file,
            holder_state_file,
            watchdog_identity=child_identity,
            holder_path=holder_path,
        )
        _atomic_json(
            state_path,
            _state_payload(
                mode,
                generation=generation,
                watchdog=child_identity,
                watchdog_lease=child_lease,
                holder=holder_identity,
                holder_files_clear=not pid_file.exists() and not holder_state_file.exists(),
                immutable_contract=expected_contract,
                pending_resume=pending_resume,
                **extra,
            ),
        )

    try:
        if state_path.exists():
            prior_state = _load_json(state_path)
            if (
                prior_state.get("schema") == _STATE_SCHEMA
                and prior_state.get("immutable_contract") == expected_contract
            ):
                try:
                    generation = int(prior_state.get("generation", 0))
                except (TypeError, ValueError) as error:
                    raise FallbackError(
                        "previous fallback generation is invalid"
                    ) from error
                candidate_pending = prior_state.get("pending_resume")
                if candidate_pending is not None:
                    if not isinstance(candidate_pending, dict):
                        raise FallbackError(
                            "previous fallback pending-resume state is invalid"
                        )
                    try:
                        _pending_resume_bindings(candidate_pending)
                        pending_generation = int(
                            candidate_pending.get("target_generation", -1)
                        )
                    except (FallbackError, TypeError, ValueError) as error:
                        raise FallbackError(
                            "previous fallback pending-resume state is invalid"
                        ) from error
                    if (
                        candidate_pending.get("immutable_contract")
                        != expected_contract
                        or pending_generation not in {generation, generation + 1}
                    ):
                        raise FallbackError(
                            "previous fallback pending-resume state is invalid"
                        )
                    pending_resume = dict(candidate_pending)
        if watchdog_identity_path.exists():
            previous_command = ("bash", str(original_path))
            previous_identity, child_lease, previous_state = (
                _load_bound_previous_watchdog(
                state_path=state_path,
                identity_path=watchdog_identity_path,
                expected_bootstrap=bootstrap_path,
                expected_command=previous_command,
                expected_contract=expected_contract,
                )
            )
            try:
                generation = int(previous_state.get("generation", generation))
            except (TypeError, ValueError) as error:
                raise FallbackError(
                    "previous fallback generation is invalid"
                ) from error
            previous_holder = _holder_identity(
                pid_file,
                holder_state_file,
                watchdog_identity=previous_identity,
                holder_path=holder_path,
            )
            saved_holder = previous_state.get("holder")
            if previous_holder is not None:
                if not isinstance(saved_holder, dict) or not _holder_lease_matches(
                    saved_holder, previous_holder
                ):
                    raise FallbackError(
                        "current fallback holder does not match its persisted lease"
                    )
            elif isinstance(saved_holder, dict) and _identity_alive(saved_holder):
                raise FallbackError(
                    "persisted fallback holder lease is alive but current holder "
                    "files do not authenticate it"
                )
            _drain_orphaned_watchdog_group(
                previous_identity,
                authenticated_members=(
                    (previous_holder,) if previous_holder is not None else ()
                ),
                timeout_seconds=args.stop_timeout_seconds,
            )
            _clear_drained_watchdog_artifacts(
                identity_path=watchdog_identity_path,
                expected_lease=child_lease,
                expected_identity=previous_identity,
                pid_file=pid_file,
                holder_state_file=holder_state_file,
                holder_path=holder_path,
            )
            child_lease = None
            if pending_resume is not None:
                pending_resume["target_generation"] = generation + 1
        stale_deadline = time.monotonic() + args.stop_timeout_seconds
        while (
            pid_file.exists() or holder_state_file.exists()
        ) and time.monotonic() < stale_deadline:
            time.sleep(0.05)
        if pid_file.exists() or holder_state_file.exists():
            raise FallbackError(
                "fallback pid/state files remained without an authenticated watchdog"
            )
        write("starting")
        while not stop_requested:
            try:
                marker_observation = _marker_observation(pause_path)
            except Exception as error:
                if child is not None:
                    _stop_managed_watchdog(
                        child, child_identity, timeout_seconds=args.stop_timeout_seconds
                    )
                    child = None
                    child_identity = None
                    child_lease = None
                write("pause_marker_error", error=f"{type(error).__name__}: {error}")
                time.sleep(args.poll_seconds)
                continue

            if pending_resume is not None:
                try:
                    _complete_pending_resume_release(
                        pause_path=pause_path,
                        release_request_path=release_request_path,
                        pending_resume=pending_resume,
                    )
                    marker_observation = {"exists": False}
                    write("resume_release_committed")
                except Exception as error:
                    if child is not None:
                        stopped_identity = child_identity or {}
                        stopped_lease = child_lease
                        _stop_managed_watchdog(
                            child,
                            stopped_identity,
                            timeout_seconds=args.stop_timeout_seconds,
                        )
                        if managed_generation:
                            if stopped_lease is None:
                                raise FallbackError(
                                    "managed fallback generation has no durable lease"
                                )
                            _clear_drained_watchdog_artifacts(
                                identity_path=watchdog_identity_path,
                                expected_lease=stopped_lease,
                                expected_identity=stopped_identity,
                                pid_file=pid_file,
                                holder_state_file=holder_state_file,
                                holder_path=holder_path,
                            )
                        child = None
                        child_identity = None
                        child_lease = None
                        managed_generation = False
                    write(
                        "resume_release_error",
                        error=f"{type(error).__name__}: {error}",
                    )
                    time.sleep(args.poll_seconds)
                    continue

            if marker_observation.get("exists"):
                if child is not None:
                    stopped_identity = child_identity or {}
                    stopped_lease = child_lease
                    _stop_managed_watchdog(
                        child, stopped_identity, timeout_seconds=args.stop_timeout_seconds
                    )
                    if managed_generation:
                        if stopped_lease is None:
                            raise FallbackError(
                                "managed fallback generation has no durable lease"
                            )
                        _clear_drained_watchdog_artifacts(
                            identity_path=watchdog_identity_path,
                            expected_lease=stopped_lease,
                            expected_identity=stopped_identity,
                            pid_file=pid_file,
                            holder_state_file=holder_state_file,
                            holder_path=holder_path,
                        )
                    child = None
                    child_identity = None
                    child_lease = None
                    managed_generation = False
                payload = marker_observation["payload"]
                marker_contract = payload.get("immutable_contract")
                if marker_contract != expected_contract:
                    write(
                        "pause_contract_error",
                        pause_token=payload["token"],
                        error="pause marker immutable contract mismatch",
                    )
                    time.sleep(args.poll_seconds)
                    continue
                if payload.get("release_request_file") != str(
                    release_request_path
                ):
                    write(
                        "pause_contract_error",
                        pause_token=payload["token"],
                        error="pause marker release-request path mismatch",
                    )
                    time.sleep(args.poll_seconds)
                    continue
                owner_alive = _identity_alive(payload.get("owner"))
                drain = None
                preflight = None
                release_mode: str | None = None
                request_binding: dict[str, Any] | None = None
                if owner_alive and release_request_path.exists():
                    try:
                        request, request_binding = _load_release_request(
                            release_request_path,
                            marker=marker_observation,
                            expected_contract=expected_contract,
                            expected_supervisor=self_identity,
                            expected_generation=generation,
                        )
                        drain = _drain_report(payload)
                        if not drain["safe"]:
                            raise FallbackError(
                                f"release requested before formal drain: {drain!r}"
                            )
                        preflight = _run_recovery_preflight(
                            expected_contract,
                            allowed_identities=(self_identity, payload["owner"]),
                        )
                        release_mode = str(request.get("mode", "owner_release"))
                    except Exception as error:
                        write(
                            "pause_release_request_error",
                            pause_token=payload["token"],
                            pause_owner_alive=True,
                            error=f"{type(error).__name__}: {error}",
                        )
                        time.sleep(args.poll_seconds)
                        continue
                elif not owner_alive:
                    try:
                        drain = _drain_report(payload)
                        if not drain["safe"]:
                            raise FallbackError(
                                f"dead formal owner still has live residue: {drain!r}"
                            )
                        preflight = _run_recovery_preflight(
                            expected_contract,
                            allowed_identities=(self_identity,),
                        )
                        release_mode = "owner_death_drained"
                        if release_request_path.exists():
                            _request, request_binding = _load_release_request(
                                release_request_path,
                                marker=marker_observation,
                                expected_contract=expected_contract,
                                expected_supervisor=self_identity,
                                expected_generation=generation,
                            )
                    except Exception as error:
                        write(
                            "owner_death_recovery_blocked",
                            pause_token=payload["token"],
                            pause_owner_alive=False,
                            recovery_drain=drain,
                            error=f"{type(error).__name__}: {error}",
                        )
                        time.sleep(args.poll_seconds)
                        continue
                if release_mode is not None:
                    pending_resume = {
                        "schema": _PENDING_RESUME_SCHEMA,
                        "mode": release_mode,
                        "token": str(payload["token"]),
                        "target_generation": generation + 1,
                        "recovery_receipt": payload.get("recovery_receipt"),
                        "drain": drain,
                        "preflight": preflight,
                        "source_supervisor": self_identity,
                        "source_generation": generation,
                        "marker_binding": _marker_binding_view(marker_observation),
                        "request_binding": request_binding,
                        "immutable_contract": expected_contract,
                        "created_unix": time.time(),
                    }
                    write(
                        "resume_authorized",
                        pause_token=payload["token"],
                        pause_owner_alive=owner_alive,
                    )
                    continue
                write(
                    "paused",
                    pause_token=payload["token"],
                    pause_owner_alive=owner_alive,
                    recovery_drain=drain,
                )
                time.sleep(args.poll_seconds)
                continue

            if release_request_path.exists() and pending_resume is None:
                write(
                    "orphan_release_request_error",
                    error="release request exists without an owned pause marker",
                )
                time.sleep(args.poll_seconds)
                continue

            if child is None:
                # Revalidate executable bytes on every generation.  A runtime
                # file change must stop recovery rather than silently start a
                # different holder implementation.
                _regular_file(original_path, args.original_sha256)
                _regular_file(holder_path, args.holder_sha256)
                _regular_file(wrapper_path, args.watchdog_wrapper_sha256)
                _regular_file(supervisor_path, args.supervisor_script_sha256)
                _regular_file(bootstrap_path, args.bootstrap_sha256)
                environment = dict(os.environ)
                environment.update(
                    {
                        "PYTHON": args.python,
                        "HOLDER": args.holder,
                        "CPU_WORKERS": "20",
                        "CPU_DUTY": "0.9",
                        "GPU_COUNT": "8",
                        "GPU_DUTY": "0.18",
                        "GPU_HOLD_MB": "512",
                        "GPU_MATRIX_SIZE": "4096",
                        "GPU_BURST_MATMULS": "8",
                        "PID_FILE": str(pid_file),
                        "STATE_FILE": str(holder_state_file),
                    }
                )
                log_handle = log_path.open("ab", buffering=0)
                try:
                    pending_generation = generation + 1

                    def bind_before_ack(
                        pending_identity: Mapping[str, Any],
                        pending_lease: Mapping[str, Any],
                    ) -> None:
                        nonlocal child_lease
                        child_lease = dict(pending_lease)
                        _atomic_json(
                            state_path,
                            _state_payload(
                                "watchdog_blocked",
                                generation=pending_generation,
                                watchdog=dict(pending_identity),
                                watchdog_lease=child_lease,
                                holder=None,
                                holder_files_clear=(
                                    not pid_file.exists()
                                    and not holder_state_file.exists()
                                ),
                                immutable_contract=expected_contract,
                                pending_resume=pending_resume,
                            ),
                        )

                    child, child_identity = _spawn_blocked(
                        name="fallback-holder-watchdog",
                        python=args.python,
                        bootstrap=bootstrap_path,
                        parent_identity=_capture_identity(os.getpid()),
                        command=["bash", str(original_path)],
                        cwd=original_path.parent,
                        environment=environment,
                        output=log_handle,
                        identity_path=watchdog_identity_path,
                        cleanup_timeout_seconds=args.stop_timeout_seconds,
                        immutable_contract=expected_contract,
                        before_ack=bind_before_ack,
                    )
                finally:
                    log_handle.close()
                generation = pending_generation
                managed_generation = True
                write("starting_holder")
            elif child.poll() is not None:
                # A SIGKILLed bootstrap cannot run its own subreaper cleanup.
                # Drain only the exact holder/worker leases published by that
                # generation; never signal a potentially reused numeric PGID.
                write("watchdog_exit_cleanup")
                unexpected_holder = _holder_identity(
                    pid_file,
                    holder_state_file,
                    holder_path=holder_path,
                )
                _drain_orphaned_watchdog_group(
                    child_identity or {},
                    authenticated_members=(
                        (unexpected_holder,) if unexpected_holder is not None else ()
                    ),
                    timeout_seconds=args.stop_timeout_seconds,
                )
                if managed_generation:
                    if child_lease is None:
                        raise FallbackError(
                            "managed fallback generation has no durable lease"
                        )
                    _clear_drained_watchdog_artifacts(
                        identity_path=watchdog_identity_path,
                        expected_lease=child_lease,
                        expected_identity=child_identity or {},
                        pid_file=pid_file,
                        holder_state_file=holder_state_file,
                        holder_path=holder_path,
                    )
                child = None
                child_identity = None
                child_lease = None
                managed_generation = False
                write("watchdog_exited")
                time.sleep(args.restart_delay_seconds)
                continue
            holder_identity = _holder_identity(
                pid_file,
                holder_state_file,
                watchdog_identity=child_identity,
                holder_path=holder_path,
            )
            if holder_identity is not None:
                if pending_resume is not None:
                    try:
                        holder_activity = _holder_resource_attestation(
                            holder_identity,
                            nvidia_smi=Path(args.nvidia_smi),
                            nvidia_smi_sha256=args.nvidia_smi_sha256,
                        )
                        write(
                            "holder_restored_pending_receipt",
                            holder_activity=holder_activity,
                        )
                        _assert_pending_resume_publishable(
                            pause_path=pause_path,
                            release_request_path=release_request_path,
                            pending_resume=pending_resume,
                        )
                        receipt_raw = pending_resume.get("recovery_receipt")
                        if receipt_raw:
                            _atomic_json(
                                Path(str(receipt_raw)),
                                {
                                    "schema": "amg_fallback_recovery_v2",
                                    "status": "pass",
                                    "mode": pending_resume["mode"],
                                    "token": pending_resume["token"],
                                    "immutable_contract": expected_contract,
                                    "drain": pending_resume["drain"],
                                    "preflight": pending_resume["preflight"],
                                    "restored_supervisor": self_identity,
                                    "restored_generation": generation,
                                    "restored_holder": _holder_lease_snapshot(
                                        holder_identity
                                    ),
                                    "holder_activity": holder_activity,
                                    "updated_unix": time.time(),
                                },
                            )
                        pending_resume = None
                        write("holding", holder_activity=holder_activity)
                    except Exception as error:
                        write(
                            "holder_restore_pending",
                            error=f"{type(error).__name__}: {error}",
                        )
                else:
                    write("holding")
            else:
                write("starting_holder")
            time.sleep(args.poll_seconds)
    finally:
        cleanup_error: Exception | None = None
        if child is not None:
            stopped_identity = child_identity or {}
            stopped_lease = child_lease
            try:
                _stop_managed_watchdog(
                    child,
                    stopped_identity,
                    timeout_seconds=args.stop_timeout_seconds,
                )
                if managed_generation:
                    if stopped_lease is None:
                        raise FallbackError(
                            "managed fallback generation has no durable lease"
                        )
                    _clear_drained_watchdog_artifacts(
                        identity_path=watchdog_identity_path,
                        expected_lease=stopped_lease,
                        expected_identity=stopped_identity,
                        pid_file=pid_file,
                        holder_state_file=holder_state_file,
                        holder_path=holder_path,
                    )
            except Exception as error:
                # Preserve the last state plus identity/pid/holder files.  A
                # platform restart can then authenticate and finish draining
                # this exact generation instead of starting a duplicate.
                cleanup_error = error
            else:
                child = None
                child_identity = None
                child_lease = None
                managed_generation = False
        durable_generation_remains = (
            watchdog_identity_path.exists()
            or pid_file.exists()
            or holder_state_file.exists()
        )
        if cleanup_error is None and not durable_generation_remains:
            write("stopped", stop_requested=stop_requested)
        os.close(lock_fd)
        if cleanup_error is not None:
            raise FallbackError(
                "fallback supervisor stopped without proving generation drain; "
                "durable recovery evidence was preserved"
            ) from cleanup_error
    return 0


def _wait_supervisor_state(
    state_path: Path,
    *,
    modes: set[str],
    timeout_seconds: float,
    pause_token: str | None = None,
    require_holder_files_clear: bool = False,
    require_live_holder: bool = False,
    expected_contract: Mapping[str, str] | None = None,
    expected_supervisor: Mapping[str, Any] | None = None,
    expected_generation: int | None = None,
    require_resume_complete: bool = False,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            last = _load_json(state_path)
        except FallbackError:
            time.sleep(0.05)
            continue
        if last.get("schema") != _STATE_SCHEMA or not _identity_alive(
            last.get("supervisor")
        ):
            time.sleep(0.05)
            continue
        if last.get("mode") not in modes:
            time.sleep(0.05)
            continue
        if pause_token is not None and last.get("pause_token") != pause_token:
            time.sleep(0.05)
            continue
        if require_holder_files_clear and last.get("holder_files_clear") is not True:
            time.sleep(0.05)
            continue
        if expected_contract is not None:
            observed_contract = last.get("immutable_contract")
            if not isinstance(observed_contract, dict) or any(
                observed_contract.get(key) != value
                for key, value in expected_contract.items()
            ):
                time.sleep(0.05)
                continue
        if expected_generation is not None:
            try:
                if int(last.get("generation", -1)) != expected_generation:
                    time.sleep(0.05)
                    continue
            except (TypeError, ValueError):
                time.sleep(0.05)
                continue
        if expected_supervisor is not None and not _same_process_lease(
            expected_supervisor, last.get("supervisor")
        ):
            time.sleep(0.05)
            continue
        if require_resume_complete and last.get("pending_resume") is not None:
            time.sleep(0.05)
            continue
        if require_live_holder:
            watchdog = last.get("watchdog")
            holder = last.get("holder")
            if not _identity_alive(watchdog) or not _identity_alive(holder):
                time.sleep(0.05)
                continue
            try:
                if int(holder["pgrp"]) != int(watchdog["pid"]):
                    time.sleep(0.05)
                    continue
            except (KeyError, TypeError, ValueError):
                time.sleep(0.05)
                continue
        return last
    raise FallbackError(
        f"fallback supervisor did not reach modes={sorted(modes)}; last={last!r}"
    )


def _parent_identity_for(child_identity: Mapping[str, Any]) -> dict[str, Any]:
    observed = _proc_identity(int(child_identity["pid"]))
    if not observed or observed["start_ticks"] != str(child_identity["start_ticks"]):
        raise FallbackError("fallback child identity changed before parent capture")
    return _capture_identity(int(observed["ppid"]))


def _expected_contract(args: argparse.Namespace) -> dict[str, str]:
    return {
        "python": str(Path(args.python)),
        "python_sha256": str(args.python_sha256),
        "original": str(Path(args.original)),
        "original_sha256": str(args.original_sha256),
        "holder": str(Path(args.holder)),
        "holder_sha256": str(args.holder_sha256),
        "watchdog_wrapper": str(Path(args.watchdog_wrapper)),
        "watchdog_wrapper_sha256": str(args.watchdog_wrapper_sha256),
        "supervisor_script": str(Path(args.supervisor_script)),
        "supervisor_script_sha256": str(args.supervisor_script_sha256),
        "bootstrap": str(Path(args.bootstrap)),
        "bootstrap_sha256": str(args.bootstrap_sha256),
        "watchdog_identity_file": str(Path(args.watchdog_identity_file)),
        "pid_file": str(Path(args.pid_file)),
        "holder_state_file": str(Path(args.holder_state_file)),
        "pause_marker": str(Path(args.pause_marker)),
        "release_request_file": str(Path(args.release_request_file)),
        "supervisor_state": str(Path(args.supervisor_state)),
        "supervisor_lock": str(Path(args.supervisor_lock)),
        "transaction_lock": str(Path(args.transaction_lock)),
        "watchdog_log": str(Path(args.watchdog_log)),
        "nvidia_smi": str(Path(args.nvidia_smi)),
        "nvidia_smi_sha256": str(args.nvidia_smi_sha256),
        "auto_gpu_holder_state": str(Path(args.auto_gpu_holder_state)),
        "auto_gpu_holder_command_fragment": str(
            args.auto_gpu_holder_command_fragment
        ),
        "auto_cpu_holder_state": str(Path(args.auto_cpu_holder_state)),
        "auto_cpu_holder_command_fragment": str(
            args.auto_cpu_holder_command_fragment
        ),
        "poll_seconds": str(float(args.poll_seconds)),
        "restart_delay_seconds": str(float(args.restart_delay_seconds)),
        "stop_timeout_seconds": str(float(args.stop_timeout_seconds)),
    }


def _preflight_contract(args: argparse.Namespace) -> dict[str, str]:
    return {
        "nvidia_smi": str(Path(args.nvidia_smi)),
        "nvidia_smi_sha256": str(args.nvidia_smi_sha256),
        "auto_gpu_holder_state": str(Path(args.auto_gpu_holder_state)),
        "auto_gpu_holder_command_fragment": str(
            args.auto_gpu_holder_command_fragment
        ),
        "auto_cpu_holder_state": str(Path(args.auto_cpu_holder_state)),
        "auto_cpu_holder_command_fragment": str(
            args.auto_cpu_holder_command_fragment
        ),
    }


def _run_recovery_preflight(
    contract: Mapping[str, Any],
    *,
    allowed_identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "nvidia_smi",
        "nvidia_smi_sha256",
        "auto_gpu_holder_state",
        "auto_gpu_holder_command_fragment",
        "auto_cpu_holder_state",
        "auto_cpu_holder_command_fragment",
    }
    if not required.issubset(contract) or any(
        not str(contract[key]) for key in required
    ):
        raise FallbackError("owner-death recovery preflight contract is incomplete")
    report = _pod_preflight_inventory(
        nvidia_smi=Path(str(contract["nvidia_smi"])),
        nvidia_smi_sha256=str(contract["nvidia_smi_sha256"]),
        auto_gpu_holder_state=Path(str(contract["auto_gpu_holder_state"])),
        auto_gpu_holder_command_fragment=str(
            contract["auto_gpu_holder_command_fragment"]
        ),
        auto_cpu_holder_state=Path(str(contract["auto_cpu_holder_state"])),
        auto_cpu_holder_command_fragment=str(
            contract["auto_cpu_holder_command_fragment"]
        ),
        allowed_identities=allowed_identities,
    )
    if not report["safe"]:
        raise FallbackError(
            "owner-death recovery found an unowned formal/GPU workload: "
            f"{report!r}"
        )
    return report


def migrate(args: argparse.Namespace) -> int:
    lock_fd = _acquire_lock(Path(args.transaction_lock))
    pause_path = Path(args.pause_marker)
    state_path = Path(args.supervisor_state)
    pid_file = Path(args.pid_file)
    holder_state_file = Path(args.holder_state_file)
    receipt_path = Path(args.receipt)
    release_request_path = Path(args.release_request_file)
    expected_contract = _expected_contract(args)
    _regular_file(Path(args.python), args.python_sha256)
    _regular_file(Path(args.original), args.original_sha256)
    _regular_file(Path(args.holder), args.holder_sha256)
    _regular_file(Path(args.watchdog_wrapper), args.watchdog_wrapper_sha256)
    supervisor_path = _regular_file(
        Path(args.supervisor_script), args.supervisor_script_sha256
    )
    _regular_file(Path(args.bootstrap), args.bootstrap_sha256)
    _regular_file(Path(args.nvidia_smi), args.nvidia_smi_sha256)
    if supervisor_path.resolve() != Path(__file__).resolve():
        raise FallbackError(
            "running fallback supervisor does not match --supervisor-script"
        )
    if pause_path.exists():
        raise FallbackError(f"pause marker already exists: {pause_path}")
    if release_request_path.exists() or release_request_path.is_symlink():
        raise FallbackError(
            f"pause-release request already exists: {release_request_path}"
        )
    old_holder = _holder_identity(pid_file, holder_state_file)
    if old_holder is None:
        raise FallbackError("current fallback holder identity is not healthy")
    old_watchdog = _parent_identity_for(old_holder)
    old_holder_cmd = _cmdline(int(old_holder["pid"]))
    old_watchdog_cmd = _cmdline(int(old_watchdog["pid"]))
    if args.holder not in old_holder_cmd or args.original not in old_watchdog_cmd:
        raise FallbackError("current fallback command identity does not match migration")
    owner = _capture_identity(os.getpid())
    token = f"fallback-migration:{int(time.time())}:{os.getpid()}"
    marker_payload = {
        "schema": _PAUSE_SCHEMA,
        "token": token,
        "owner": owner,
        "run_id": "",
        "run_dir": str(receipt_path.parent / "no-run"),
        "ports": [],
        "protected_identities": [old_watchdog, old_holder],
        "recovery_receipt": str(receipt_path.with_name("fallback-migration-recovery.json")),
        "release_request_file": str(release_request_path),
        "immutable_contract": expected_contract,
    }
    marker_observation = _create_pause_marker(pause_path, marker_payload)
    # ``dist_train.py`` is the durable external watchdog.  Once the exact old
    # fallback exits it reloads the newly installed watchdog wrapper, which in
    # turn execs this supervisor.  Starting another detached copy here would
    # race that platform-owned restart and can create two holders.
    _signal_identity(old_watchdog, signal.SIGTERM)
    if not _wait_dead(old_watchdog, args.stop_timeout_seconds):
        raise FallbackError("old fallback watchdog did not exit")
    if not _wait_dead(old_holder, args.stop_timeout_seconds):
        raise FallbackError("old fallback child did not exit")
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and (pid_file.exists() or holder_state_file.exists()):
        time.sleep(0.1)
    if pid_file.exists() or holder_state_file.exists():
        raise FallbackError("old fallback pid/state files did not clear")
    paused = _wait_supervisor_state(
        state_path,
        modes={"paused"},
        timeout_seconds=args.supervisor_restart_timeout_seconds,
        pause_token=token,
        require_holder_files_clear=True,
        expected_contract=expected_contract,
    )
    supervisor_identity = paused["supervisor"]
    if (
        int(supervisor_identity["pid"]) == int(old_watchdog["pid"])
        and str(supervisor_identity["start_ticks"])
        == str(old_watchdog["start_ticks"])
    ):
        raise FallbackError("platform watchdog did not start a new supervisor")
    _publish_release_request(
        release_request_path,
        marker=marker_observation,
        paused_state=paused,
        owner=owner,
        expected_contract=expected_contract,
        mode="migration",
    )
    restored = _wait_supervisor_state(
        state_path,
        modes={"holding"},
        timeout_seconds=60,
        require_live_holder=True,
        expected_contract=expected_contract,
        expected_generation=int(paused["generation"]) + 1,
        require_resume_complete=True,
    )
    _atomic_json(
        receipt_path,
        {
            "schema": "amg_fallback_migration_v2",
            "status": "pass",
            "token": token,
            "old_watchdog": old_watchdog,
            "old_holder": old_holder,
            "new_supervisor": supervisor_identity,
            "new_holder": restored.get("holder"),
            "original_sha256": args.original_sha256,
            "holder_sha256": args.holder_sha256,
            "watchdog_wrapper_sha256": args.watchdog_wrapper_sha256,
            "supervisor_script_sha256": args.supervisor_script_sha256,
            "bootstrap_sha256": args.bootstrap_sha256,
            "immutable_contract": expected_contract,
            "updated_unix": time.time(),
        },
    )
    os.close(lock_fd)
    return 0


def _spawn_blocked(
    *,
    name: str,
    python: str,
    bootstrap: Path,
    parent_identity: Mapping[str, Any],
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    output: Any,
    identity_path: Path,
    cleanup_timeout_seconds: float,
    immutable_contract: Mapping[str, str],
    before_ack: Any | None = None,
) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    read_fd, write_fd = os.pipe()
    process: subprocess.Popen[bytes] | None = None
    identity: dict[str, Any] | None = None
    watched_signals = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched_signals)
    try:
        process = subprocess.Popen(
            [
                python,
                str(bootstrap),
                "--ack-fd",
                str(read_fd),
                "--parent-pid",
                str(parent_identity["pid"]),
                "--parent-start-ticks",
                str(parent_identity["start_ticks"]),
                "--cleanup-timeout-seconds",
                str(cleanup_timeout_seconds),
                "--",
                *command,
            ],
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(read_fd,),
        )
        os.close(read_fd)
        read_fd = -1
        identity = _capture_identity(process.pid)
        identity_payload = {
                "schema": _PROCESS_IDENTITY_SCHEMA,
                "name": name,
                "pid": identity["pid"],
                "start_ticks": identity["start_ticks"],
                "process_group": identity["pid"],
                "bootstrap": str(bootstrap),
                "command": list(command),
                "immutable_contract": dict(immutable_contract),
            }
        _atomic_json(identity_path, identity_payload)
        identity_binding = _process_identity_file_binding(identity_path)
        if before_ack is not None:
            before_ack(identity, identity_binding)
        if os.write(write_fd, b"1") != 1:
            raise FallbackError("failed to acknowledge outer process bootstrap")
        os.close(write_fd)
        write_fd = -1
        return process, identity
    except Exception:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        if process is not None:
            if identity is not None:
                _stop_managed_watchdog(
                    process,
                    identity,
                    timeout_seconds=cleanup_timeout_seconds,
                )
            else:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=cleanup_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _wait_drain(
    marker_payload: Mapping[str, Any], *, timeout_seconds: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = _drain_report(marker_payload)
        if last["safe"]:
            return last
        time.sleep(0.2)
    raise FallbackError(f"run-owned workload did not drain: {last!r}")


def formal(args: argparse.Namespace) -> int:
    lock_fd = _acquire_lock(Path(args.transaction_lock))
    _regular_file(Path(args.python), args.python_sha256)
    _regular_file(Path(args.original), args.original_sha256)
    _regular_file(Path(args.holder), args.holder_sha256)
    _regular_file(Path(args.watchdog_wrapper), args.watchdog_wrapper_sha256)
    supervisor_path = _regular_file(
        Path(args.supervisor_script), args.supervisor_script_sha256
    )
    _regular_file(Path(args.bootstrap), args.bootstrap_sha256)
    _regular_file(Path(args.nvidia_smi), args.nvidia_smi_sha256)
    if supervisor_path.resolve() != Path(__file__).resolve():
        raise FallbackError(
            "running fallback supervisor does not match --supervisor-script"
        )
    attempt_dir = Path(args.attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    owner_identity = _capture_identity(os.getpid())
    supervisor_state_path = Path(args.supervisor_state)
    expected_contract = _expected_contract(args)
    initial = _wait_supervisor_state(
        supervisor_state_path,
        modes={"holding"},
        timeout_seconds=15,
        require_live_holder=True,
        expected_contract=expected_contract,
    )
    initial_holder_activity = _holder_resource_attestation(
        initial["holder"],
        nvidia_smi=Path(args.nvidia_smi),
        nvidia_smi_sha256=args.nvidia_smi_sha256,
    )
    pause_path = Path(args.pause_marker)
    release_request_path = Path(args.release_request_file)
    if pause_path.exists() or pause_path.is_symlink():
        raise FallbackError(f"pause marker already exists: {pause_path}")
    if release_request_path.exists() or release_request_path.is_symlink():
        raise FallbackError(
            f"pause-release request already exists: {release_request_path}"
        )
    token = f"{args.run_id}:{int(time.time())}:{os.getpid()}"
    run_dir = Path(args.run_dir)
    marker_payload = {
        "schema": _PAUSE_SCHEMA,
        "token": token,
        "owner": owner_identity,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "ports": [int(value) for value in args.port],
        "protected_identities": [],
        "orchestrator_identity": str(attempt_dir / "orchestrator-process-identity.json"),
        "recovery_receipt": str(attempt_dir / "fallback-recovery.json"),
        "release_request_file": str(release_request_path),
        "formal_contract": {
            "run_id": args.run_id,
            "run_dir": str(run_dir),
            "attempt_dir": str(attempt_dir),
            "cwd": str(Path(args.cwd)),
            "console": str(Path(args.console)),
            "ports": [int(value) for value in args.port],
            "command": list(args.command),
            "cleanup_timeout_seconds": float(args.cleanup_timeout_seconds),
        },
        "immutable_contract": expected_contract,
    }
    marker_observation: dict[str, Any] | None = None
    process: subprocess.Popen[bytes] | None = None
    process_identity: dict[str, Any] | None = None
    return_code = 125
    errors: list[str] = []
    preflight_inventory: dict[str, Any] | None = None
    restored_holder_activity: dict[str, Any] | None = None
    signal_requested: int | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal signal_requested
        if signal_requested is None:
            signal_requested = signum

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        environment = dict(os.environ)
        environment["AMG_MULTITASK_RUN_ID"] = args.run_id
        environment["AGENTMEMORY_RUN_ID"] = args.run_id
        environment.pop("PYTHONPATH", None)
        with Path(args.console).open("ab", buffering=0) as output:
            def bind_formal_before_ack(
                pending_identity: Mapping[str, Any],
                pending_binding: Mapping[str, Any],
            ) -> None:
                nonlocal marker_observation, preflight_inventory
                marker_payload["orchestrator_identity_binding"] = dict(
                    pending_binding
                )
                marker_observation = _create_pause_marker(
                    pause_path, marker_payload
                )
                _wait_supervisor_state(
                    supervisor_state_path,
                    modes={"paused"},
                    timeout_seconds=60,
                    pause_token=token,
                    require_holder_files_clear=True,
                    expected_contract=expected_contract,
                )
                paused_state = _load_json(supervisor_state_path)
                preflight_inventory = _pod_preflight_inventory(
                    nvidia_smi=Path(args.nvidia_smi),
                    nvidia_smi_sha256=args.nvidia_smi_sha256,
                    auto_gpu_holder_state=Path(args.auto_gpu_holder_state),
                    auto_gpu_holder_command_fragment=(
                        args.auto_gpu_holder_command_fragment
                    ),
                    auto_cpu_holder_state=Path(args.auto_cpu_holder_state),
                    auto_cpu_holder_command_fragment=(
                        args.auto_cpu_holder_command_fragment
                    ),
                    allowed_identities=(
                        owner_identity,
                        paused_state["supervisor"],
                        pending_identity,
                    ),
                )
                _atomic_json(
                    attempt_dir / "pod-preflight-inventory.json",
                    preflight_inventory,
                )
                if not preflight_inventory["safe"]:
                    raise FallbackError(
                        "current Pod contains an unowned formal/GPU workload: "
                        f"{preflight_inventory!r}"
                    )

            process, process_identity = _spawn_blocked(
                name="outer-multitask-orchestrator",
                python=args.python,
                bootstrap=Path(args.bootstrap),
                parent_identity=owner_identity,
                command=args.command,
                cwd=Path(args.cwd),
                environment=environment,
                output=output,
                identity_path=Path(marker_payload["orchestrator_identity"]),
                cleanup_timeout_seconds=args.cleanup_timeout_seconds,
                immutable_contract=expected_contract,
                before_ack=bind_formal_before_ack,
            )
            while process.poll() is None:
                if signal_requested is not None and process_identity is not None:
                    _signal_identity(process_identity, signal_requested)
                    signal_requested = None
                state = _load_json(supervisor_state_path)
                if not _identity_alive(state.get("supervisor")):
                    errors.append("fallback supervisor died while formal was active")
                    if process_identity is not None:
                        _signal_identity(process_identity, signal.SIGTERM)
                    break
                if any(
                    state.get("immutable_contract", {}).get(key) != value
                    for key, value in expected_contract.items()
                ):
                    errors.append("fallback supervisor immutable contract changed")
                    if process_identity is not None:
                        _signal_identity(process_identity, signal.SIGTERM)
                    break
                if state.get("mode") != "paused" or state.get("pause_token") != token:
                    errors.append("fallback supervisor left the owned paused state")
                    if process_identity is not None:
                        _signal_identity(process_identity, signal.SIGTERM)
                    break
                time.sleep(0.5)
            try:
                return_code = process.wait(
                    timeout=(
                        args.cleanup_timeout_seconds
                        + _BOOTSTRAP_DRAIN_GRACE_SECONDS
                    )
                )
            except subprocess.TimeoutExpired:
                # Keep the subreaper anchor alive.  It is the only process
                # with a complete ancestry inventory for setsid descendants;
                # killing it here could make a later drain look empty.
                return_code = 125
                errors.append(
                    "outer orchestrator bootstrap exceeded cleanup timeout plus "
                    "subreaper drain grace"
                )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        if process is not None and process.poll() is None and process_identity is not None:
            _signal_identity(process_identity, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(
                    timeout=(
                        args.cleanup_timeout_seconds
                        + _BOOTSTRAP_DRAIN_GRACE_SECONDS
                    )
                )
    drain: dict[str, Any] | None = None
    restored: dict[str, Any] | None = None
    try:
        if marker_observation is None:
            raise FallbackError("formal pause marker was never installed")
        drain = _wait_drain(marker_payload, timeout_seconds=args.cleanup_timeout_seconds)
        state = _load_json(supervisor_state_path)
        if not _identity_alive(state.get("supervisor")):
            raise FallbackError("fallback supervisor is dead at pause release")
        if (
            state.get("mode") != "paused"
            or state.get("pause_token") != token
            or state.get("immutable_contract") != expected_contract
        ):
            raise FallbackError(
                "fallback supervisor no longer owns the expected paused contract"
            )
        _publish_release_request(
            release_request_path,
            marker=marker_observation,
            paused_state=state,
            owner=owner_identity,
            expected_contract=expected_contract,
            mode="formal_owner_release",
        )
        restored = _wait_supervisor_state(
            supervisor_state_path,
            modes={"holding"},
            timeout_seconds=60,
            require_live_holder=True,
            expected_contract=expected_contract,
            expected_generation=int(state["generation"]) + 1,
            require_resume_complete=True,
        )
        restored_holder_activity = _holder_resource_attestation(
            restored["holder"],
            nvidia_smi=Path(args.nvidia_smi),
            nvidia_smi_sha256=args.nvidia_smi_sha256,
        )
    except Exception as error:
        errors.append(f"fallback restore: {type(error).__name__}: {error}")
    receipt = {
        "schema": "amg_formal_owner_v2",
        "status": "pass" if return_code == 0 and not errors else "fail",
        "run_id": args.run_id,
        "owner": owner_identity,
        "initial_fallback": initial,
        "initial_holder_activity": initial_holder_activity,
        "pod_preflight_inventory": preflight_inventory,
        "orchestrator": process_identity,
        "orchestrator_exit_code": return_code,
        "drain": drain,
        "restored_fallback": restored,
        "restored_holder_activity": restored_holder_activity,
        "pause_marker_exists": pause_path.exists(),
        "immutable_contract": expected_contract,
        "errors": errors,
        "updated_unix": time.time(),
    }
    _atomic_json(attempt_dir / "formal-owner-receipt.json", receipt)
    os.close(lock_fd)
    if errors:
        return 125
    return return_code


def _add_common_supervisor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--original-sha256", required=True)
    parser.add_argument("--holder", required=True)
    parser.add_argument("--holder-sha256", required=True)
    parser.add_argument("--watchdog-wrapper", required=True)
    parser.add_argument("--watchdog-wrapper-sha256", required=True)
    parser.add_argument("--supervisor-script", required=True)
    parser.add_argument("--supervisor-script-sha256", required=True)
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--watchdog-identity-file", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--holder-state-file", required=True)
    parser.add_argument("--pause-marker", required=True)
    parser.add_argument("--release-request-file", required=True)
    parser.add_argument("--supervisor-state", required=True)
    parser.add_argument("--supervisor-lock", required=True)
    parser.add_argument("--transaction-lock", required=True)
    parser.add_argument("--watchdog-log", required=True)
    parser.add_argument("--nvidia-smi", required=True)
    parser.add_argument("--nvidia-smi-sha256", required=True)
    parser.add_argument("--auto-gpu-holder-state", required=True)
    parser.add_argument("--auto-gpu-holder-command-fragment", required=True)
    parser.add_argument("--auto-cpu-holder-state", required=True)
    parser.add_argument("--auto-cpu-holder-command-fragment", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    parser.add_argument("--restart-delay-seconds", type=float, default=5)
    parser.add_argument("--stop-timeout-seconds", type=float, default=120)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    supervise_parser = subparsers.add_parser("supervise")
    _add_common_supervisor_args(supervise_parser)
    supervise_parser.set_defaults(handler=supervise)

    migrate_parser = subparsers.add_parser("migrate")
    _add_common_supervisor_args(migrate_parser)
    migrate_parser.add_argument("--receipt", required=True)
    migrate_parser.add_argument(
        "--supervisor-restart-timeout-seconds", type=float, default=45
    )
    migrate_parser.set_defaults(handler=migrate)

    formal_parser = subparsers.add_parser("formal")
    _add_common_supervisor_args(formal_parser)
    formal_parser.add_argument("--attempt-dir", required=True)
    formal_parser.add_argument("--run-id", required=True)
    formal_parser.add_argument("--run-dir", required=True)
    formal_parser.add_argument("--cwd", required=True)
    formal_parser.add_argument("--console", required=True)
    formal_parser.add_argument("--port", action="append", default=[])
    formal_parser.add_argument("--cleanup-timeout-seconds", type=float, default=300)
    formal_parser.add_argument("command", nargs=argparse.REMAINDER)
    formal_parser.set_defaults(handler=formal)
    args = parser.parse_args(argv)
    if getattr(args, "command", None) and args.command[0] == "--":
        args.command = args.command[1:]
    if getattr(args, "subcommand", None) == "formal" and not args.command:
        parser.error("formal requires a command after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return int(args.handler(args))
    except BlockingIOError:
        print("fallback/formal transaction lock is already held", file=sys.stderr)
        return 73
    except Exception as error:
        print(
            f"fallback supervisor failed closed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
