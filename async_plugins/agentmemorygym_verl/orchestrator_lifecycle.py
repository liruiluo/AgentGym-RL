"""Crash-safe operational helpers for AMG fully-asynchronous launchers.

This module deliberately owns only machine-local orchestration concerns that
veRL does not own: holder-marker transactions, bounded GPU telemetry,
filesystem-capacity admission, sealed-publication selection, and atomic
publication of run evidence.  It does not implement scheduling, queues,
staleness control, weight synchronization, rollout, or PPO.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NoReturn


class LifecycleError(RuntimeError):
    """Fail-closed lifecycle contract violation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def _atomic_write_text(path: Path, text: str, mode: int = 0o600) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def _atomic_write_json(path: Path, value: Any, mode: int = 0o600) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, payload, mode=mode)


def _load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise LifecycleError(
            f"required regular JSON file is missing or symlinked: {path}"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LifecycleError(f"invalid JSON file {path}: {error}") from error


def process_start_ticks(pid: int) -> str | None:
    """Return Linux /proc start ticks, or ``None`` for a dead/reused process."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw.rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X", "x"}:
            return None
        return fields[19]
    except (FileNotFoundError, IndexError, OSError):
        return None


def process_identity_alive(pid: int, start_ticks: str) -> bool:
    return (
        pid > 0 and bool(start_ticks) and process_start_ticks(pid) == str(start_ticks)
    )


def _read_marker(path: Path) -> str | None:
    if path.is_symlink():
        raise LifecycleError(f"marker must not be a symlink: {path}")
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
        raise LifecycleError(f"cannot open marker {path}: {error}") from error
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise LifecycleError(f"marker must be a regular file: {path}")
        raw = os.read(descriptor, 4097)
        if len(raw) > 4096:
            raise LifecycleError(f"marker exceeds 4096 bytes: {path}")
        try:
            value = raw.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise LifecycleError(f"marker must be valid UTF-8: {path}") from error
    except OSError as error:
        raise LifecycleError(f"cannot read marker {path}: {error}") from error
    finally:
        os.close(descriptor)
    if not value:
        raise LifecycleError(f"marker must not be empty: {path}")
    return value


def _write_marker(path: Path, value: str) -> None:
    if not value or "\n" in value:
        raise LifecycleError(f"invalid marker value for {path}: {value!r}")
    _atomic_write_text(path, value + "\n", mode=0o600)


def _create_marker_exclusive(path: Path, value: str) -> None:
    if not value or "\n" in value:
        raise LifecycleError(f"invalid marker value for {path}: {value!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.claim.", dir=path.parent
    )
    temp = Path(raw_temp)
    try:
        os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            stream.write((value + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            try:
                os.link(temp, path, follow_symlinks=False)
            except TypeError:  # pragma: no cover - Python without follow_symlinks
                os.link(temp, path)
        except FileExistsError as error:
            raise LifecycleError(
                f"marker appeared during exclusive claim: {path}"
            ) from error
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()
        _fsync_directory(path.parent)


def _restore_quarantined_marker(backup: Path, path: Path) -> None:
    try:
        os.link(backup, path)
    except FileExistsError as error:
        raise LifecycleError(
            f"foreign marker appeared while restoring quarantined marker: {path}"
        ) from error
    backup.unlink()
    _fsync_directory(path.parent)


def _same_inode(left: Path, right: Path) -> bool:
    try:
        left_stat = os.lstat(left)
        right_stat = os.lstat(right)
    except OSError:
        return False
    return (
        stat.S_ISREG(left_stat.st_mode)
        and stat.S_ISREG(right_stat.st_mode)
        and left_stat.st_dev == right_stat.st_dev
        and left_stat.st_ino == right_stat.st_ino
    )


def _quarantine_marker_noreplace(path: Path, backup: Path) -> None:
    """Move a marker to quarantine without replacing an existing backup.

    The hard-link is the no-replace linearization point.  The source name stays
    present until both names are proven to reference the same inode, so a
    concurrent exclusive claimant cannot enter between quarantine preparation
    and source unlink.  A crash while both links exist is resumed by
    :func:`_cas_marker`.
    """

    try:
        try:
            os.link(path, backup, follow_symlinks=False)
        except TypeError:  # pragma: no cover - Python without follow_symlinks
            os.link(path, backup)
    except FileExistsError:
        raise
    except FileNotFoundError:
        raise
    except OSError as error:
        raise LifecycleError(
            f"cannot quarantine marker without replacement {path}: {error}"
        ) from error
    if not _same_inode(path, backup):
        raise LifecycleError(
            f"marker quarantine inode verification failed: {path} -> {backup}"
        )
    path.unlink()
    _fsync_directory(path.parent)


def _marker_transition_backup(path: Path, transition_id: str) -> Path:
    token = hashlib.sha256(transition_id.encode("utf-8")).hexdigest()[:24]
    return path.with_name(f".{path.name}.{token}.transition")


def _cas_marker(
    path: Path,
    expected: str | None,
    replacement: str | None,
    *,
    transition_id: str = "default",
) -> None:
    """Transition a marker without ever overwriting a concurrent writer.

    POSIX regular files do not expose compare-and-swap.  For a present marker,
    move the current inode to a deterministic quarantine path, validate it,
    then create the replacement with O_EXCL.  A concurrent non-cooperating
    writer can make the transition fail, but its claim is never overwritten.
    The quarantine path also makes a crash between rename and create
    recoverable and idempotent.
    """

    backup = _marker_transition_backup(path, transition_id)
    if backup.is_symlink():
        raise LifecycleError(f"marker transition backup is a symlink: {backup}")

    if backup.exists():
        observed_backup = _read_marker(backup)
        if observed_backup != expected:
            if _read_marker(path) is None:
                _restore_quarantined_marker(backup, path)
            raise LifecycleError(
                f"marker transition backup mismatch for {path}: "
                f"expected={expected!r}, backup={observed_backup!r}"
            )
        current = _read_marker(path)
        if current == expected and _same_inode(path, backup):
            path.unlink()
            _fsync_directory(path.parent)
            current = None
        if current is None:
            if replacement is not None:
                _create_marker_exclusive(path, replacement)
        elif current != replacement:
            raise LifecycleError(
                f"foreign marker appeared during transition for {path}: {current!r}"
            )
        backup.unlink()
        _fsync_directory(path.parent)
        if _read_marker(path) != replacement:
            raise LifecycleError(
                f"marker transition recovery verification failed: {path}"
            )
        return

    current = _read_marker(path)
    if current == replacement and expected != replacement:
        return
    if current != expected:
        raise LifecycleError(
            f"marker CAS mismatch for {path}: expected={expected!r}, "
            f"current={current!r}"
        )
    if expected is None:
        if replacement is not None:
            _create_marker_exclusive(path, replacement)
    else:
        try:
            _quarantine_marker_noreplace(path, backup)
        except FileExistsError as error:
            raise LifecycleError(
                f"marker transition backup appeared before quarantine: {backup}"
            ) from error
        except FileNotFoundError as error:
            raise LifecycleError(
                f"marker disappeared before quarantine: {path}"
            ) from error
        _fsync_directory(path.parent)
        observed_backup = _read_marker(backup)
        if observed_backup != expected:
            try:
                _restore_quarantined_marker(backup, path)
            finally:
                raise LifecycleError(
                    f"marker changed before quarantine for {path}: "
                    f"expected={expected!r}, observed={observed_backup!r}"
                )
        try:
            if replacement is not None:
                _create_marker_exclusive(path, replacement)
        except Exception:
            # Never replace a concurrently created foreign marker.  Retain the
            # quarantine file as exact forensic/recovery evidence.
            if _read_marker(path) is None:
                _restore_quarantined_marker(backup, path)
            raise
        backup.unlink()
        _fsync_directory(path.parent)
    observed = _read_marker(path)
    if observed != replacement:
        raise LifecycleError(
            f"marker CAS verification failed for {path}: "
            f"wanted={replacement!r}, observed={observed!r}"
        )


def _acquire_lock_descriptor(path: Path, operation: int) -> int:
    if path.is_symlink():
        raise LifecycleError(f"lock path must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise LifecycleError(f"cannot open lifecycle lock {path}: {error}") from error
    try:
        os.set_inheritable(descriptor, False)
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, operation)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _release_lock_descriptor(descriptor: int) -> None:
    with contextlib.suppress(OSError):
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


@contextlib.contextmanager
def _file_lock(path: Path, operation: int) -> Iterator[None]:
    descriptor = _acquire_lock_descriptor(path, operation)
    try:
        yield
    finally:
        _release_lock_descriptor(descriptor)


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    with _file_lock(path, fcntl.LOCK_EX):
        yield


@contextlib.contextmanager
def _shared_lock(path: Path) -> Iterator[None]:
    with _file_lock(path, fcntl.LOCK_SH):
        yield


def _marker_record(
    name: str,
    path: Path,
    original_value: str | None,
    original_pid: int,
    original_start_ticks: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "original_value": original_value,
        "original_identity": {
            "pid": original_pid,
            "start_ticks": str(original_start_ticks),
        },
        "acquire_started": False,
        "acquired": False,
        "restored": False,
        "restore_started": False,
        "restore_target_set": False,
        "restore_target": None,
    }


def prepare_marker_transaction(
    *,
    state_path: Path,
    lock_path: Path,
    run_id: str,
    parent_pid: int,
    parent_start_ticks: str,
    markers: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    with _exclusive_lock(lock_path):
        if state_path.exists() or state_path.is_symlink():
            raise LifecycleError(f"refusing to reuse marker state: {state_path}")
        if not process_identity_alive(parent_pid, parent_start_ticks):
            raise LifecycleError(
                "launcher parent identity is not alive during marker prepare"
            )
        normalized: list[dict[str, Any]] = []
        for raw in markers:
            marker = dict(raw)
            path = Path(marker["path"])
            current = _read_marker(path)
            if current != marker["original_value"]:
                raise LifecycleError(
                    f"marker changed before prepare for {path}: "
                    f"expected={marker['original_value']!r}, current={current!r}"
                )
            identity = marker["original_identity"]
            if marker["original_value"] is not None and not process_identity_alive(
                int(identity["pid"]), str(identity["start_ticks"])
            ):
                raise LifecycleError(
                    f"original {marker['name']} marker owner identity is not alive"
                )
            normalized.append(marker)
        state = {
            "schema": "amg_marker_transaction_v1",
            "run_id": run_id,
            "status": "prepared",
            "parent": {"pid": parent_pid, "start_ticks": str(parent_start_ticks)},
            "lock_path": str(lock_path),
            "markers": normalized,
            "updated_unix": time.time(),
        }
        _atomic_write_json(state_path, state)
        return state


def _load_marker_state(state_path: Path) -> dict[str, Any]:
    state = _load_json(state_path)
    if state.get("schema") != "amg_marker_transaction_v1":
        raise LifecycleError(f"unsupported marker state schema in {state_path}")
    if not isinstance(state.get("markers"), list) or not state["markers"]:
        raise LifecycleError(f"marker state has no markers: {state_path}")
    return state


def _save_marker_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_unix"] = time.time()
    _atomic_write_json(state_path, state)


def _restore_target(marker: dict[str, Any]) -> str | None:
    original = marker["original_value"]
    if original is None:
        return None
    identity = marker["original_identity"]
    if process_identity_alive(int(identity["pid"]), str(identity["start_ticks"])):
        return str(original)
    return None


def _restore_markers_locked(state_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    state["status"] = "restoring"
    _save_marker_state(state_path, state)
    try:
        for marker in state["markers"]:
            path = Path(marker["path"])
            acquire_transition_id = f"{state['run_id']}:{marker['name']}:acquire"
            acquire_backup = _marker_transition_backup(path, acquire_transition_id)
            current = _read_marker(path)
            if (
                marker.get("acquire_started", False)
                and not marker["acquired"]
                and (
                    acquire_backup.exists()
                    or acquire_backup.is_symlink()
                    or current == state["run_id"]
                )
            ):
                _cas_marker(
                    path,
                    marker["original_value"],
                    state["run_id"],
                    transition_id=acquire_transition_id,
                )
                marker["acquired"] = True
                _save_marker_state(state_path, state)
            if not marker["restore_target_set"]:
                marker["restore_target"] = _restore_target(marker)
                marker["restore_target_set"] = True
                _save_marker_state(state_path, state)
            target = marker["restore_target"]
            current = _read_marker(path)
            if marker["restored"]:
                if current != target:
                    raise LifecycleError(
                        f"already-restored marker drifted for {path}: "
                        f"expected={target!r}, current={current!r}"
                    )
                continue
            # A SIGKILL can land after the filesystem CAS commits but before
            # the state receipt records ``restored=True``.  Treat an exact
            # target value as committed and make the retry idempotent.
            if current == target:
                commit_was_intended = marker.get("restore_started", False)
                never_mutated = (
                    not marker["acquired"] and current == marker["original_value"]
                )
                if not commit_was_intended and not never_mutated:
                    raise LifecycleError(
                        "owned marker disappeared or changed before restore CAS: "
                        f"{path}"
                    )
                marker["restored"] = True
                _save_marker_state(state_path, state)
                continue
            changed_by_transaction = marker["acquired"] or (
                marker.get("acquire_started", False) and current == state["run_id"]
            )
            if changed_by_transaction:
                marker["restore_started"] = True
                _save_marker_state(state_path, state)
                _cas_marker(
                    path,
                    state["run_id"],
                    target,
                    transition_id=f"{state['run_id']}:{marker['name']}:restore",
                )
                marker["acquired"] = True
            else:
                if current != marker["original_value"]:
                    raise LifecycleError(
                        f"unacquired marker drifted for {path}: "
                        f"expected={marker['original_value']!r}, current={current!r}"
                    )
                if current != target:
                    marker["restore_started"] = True
                    _save_marker_state(state_path, state)
                    _cas_marker(
                        path,
                        current,
                        target,
                        transition_id=f"{state['run_id']}:{marker['name']}:restore",
                    )
            marker["restored"] = True
            _save_marker_state(state_path, state)
        for marker in state["markers"]:
            observed = _read_marker(Path(marker["path"]))
            if observed != marker["restore_target"]:
                raise LifecycleError(
                    f"marker restoration verification failed for {marker['path']}: "
                    f"expected={marker['restore_target']!r}, observed={observed!r}"
                )
        state["status"] = "restored"
        state.pop("last_error", None)
        _save_marker_state(state_path, state)
        return state
    except Exception as error:
        state["status"] = "restore_failed"
        state["last_error"] = f"{type(error).__name__}: {error}"
        _save_marker_state(state_path, state)
        raise


def acquire_marker_transaction(state_path: Path, lock_path: Path) -> dict[str, Any]:
    with _exclusive_lock(lock_path):
        state = _load_marker_state(state_path)
        if state["status"] != "prepared":
            raise LifecycleError(
                f"marker transaction is not prepared: {state['status']}"
            )
        parent = state["parent"]
        if not process_identity_alive(int(parent["pid"]), str(parent["start_ticks"])):
            raise LifecycleError(
                "launcher parent identity died before marker acquisition"
            )
        state["status"] = "acquiring"
        _save_marker_state(state_path, state)
        try:
            for marker in state["markers"]:
                marker["acquire_started"] = True
                _save_marker_state(state_path, state)
                _cas_marker(
                    Path(marker["path"]),
                    marker["original_value"],
                    state["run_id"],
                    transition_id=f"{state['run_id']}:{marker['name']}:acquire",
                )
                marker["acquired"] = True
                _save_marker_state(state_path, state)
            state["status"] = "acquired"
            _save_marker_state(state_path, state)
            return state
        except Exception as error:
            state["status"] = "acquisition_failed"
            state["last_error"] = f"{type(error).__name__}: {error}"
            _save_marker_state(state_path, state)
            try:
                _restore_markers_locked(state_path, state)
                state["status"] = "acquisition_rolled_back"
                _save_marker_state(state_path, state)
            except Exception as rollback_error:
                state["status"] = "acquisition_rollback_failed"
                state["rollback_error"] = (
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
                _save_marker_state(state_path, state)
            raise LifecycleError(
                f"marker acquisition failed; rollback_status={state['status']}: {error}"
            ) from error


def restore_marker_transaction(state_path: Path, lock_path: Path) -> dict[str, Any]:
    with _exclusive_lock(lock_path):
        state = _load_marker_state(state_path)
        if state["status"] == "restored":
            return state
        return _restore_markers_locked(state_path, state)


def _marker_state_status(state_path: Path, lock_path: Path) -> str:
    with _exclusive_lock(lock_path):
        return str(_load_marker_state(state_path)["status"])


def watch_marker_transaction(
    *,
    state_path: Path,
    lock_path: Path,
    parent_pid: int,
    parent_start_ticks: str,
    ready_path: Path,
    receipt_path: Path,
    poll_seconds: float,
    restore_timeout_seconds: float,
) -> int:
    state = _load_marker_state(state_path)
    if state["parent"] != {
        "pid": parent_pid,
        "start_ticks": str(parent_start_ticks),
    }:
        raise LifecycleError("marker watcher parent identity does not match state")
    watcher_ticks = process_start_ticks(os.getpid())
    _atomic_write_json(
        ready_path,
        {
            "schema": "amg_marker_watcher_start_v1",
            "status": "ready",
            "pid": os.getpid(),
            "start_ticks": watcher_ticks,
            "parent_pid": parent_pid,
            "parent_start_ticks": str(parent_start_ticks),
            "state_path": str(state_path),
        },
    )
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    mode = ""
    error_text: str | None = None
    deadline: float | None = None
    while True:
        status = _marker_state_status(state_path, lock_path)
        if status in {"restored", "acquisition_rolled_back"}:
            mode = "explicit_restore"
            break
        if not process_identity_alive(parent_pid, parent_start_ticks):
            mode = "parent_death_restore"
            if deadline is None:
                deadline = time.monotonic() + restore_timeout_seconds
            try:
                restore_marker_transaction(state_path, lock_path)
                error_text = None
                break
            except Exception as error:  # retry a transient partial restore
                error_text = f"{type(error).__name__}: {error}"
                if time.monotonic() >= deadline:
                    _atomic_write_json(
                        receipt_path,
                        {
                            "schema": "amg_marker_watcher_exit_v1",
                            "status": "fail",
                            "mode": mode,
                            "error": error_text,
                            "pid": os.getpid(),
                        },
                    )
                    return 1
        elif stop_requested:
            mode = "signal_before_restore"
            error_text = "watcher was signalled while launcher still owned markers"
            _atomic_write_json(
                receipt_path,
                {
                    "schema": "amg_marker_watcher_exit_v1",
                    "status": "fail",
                    "mode": mode,
                    "error": error_text,
                    "pid": os.getpid(),
                },
            )
            return 1
        time.sleep(poll_seconds)
    _atomic_write_json(
        receipt_path,
        {
            "schema": "amg_marker_watcher_exit_v1",
            "status": "pass",
            "mode": mode,
            "error": error_text,
            "pid": os.getpid(),
            "state_status": _marker_state_status(state_path, lock_path),
        },
    )
    return 0


def _active_process_group_pids(process_group_id: int) -> list[int]:
    proc_root = Path("/proc")
    if proc_root.is_dir():
        active: list[int] = []
        for candidate in proc_root.iterdir():
            if not candidate.name.isdigit():
                continue
            try:
                fields = (
                    (candidate / "stat")
                    .read_text(encoding="utf-8")
                    .rsplit(")", 1)[1]
                    .split()
                )
            except (FileNotFoundError, IndexError, OSError):
                continue
            state = fields[0]
            if state not in {"Z", "X", "x"} and int(fields[2]) == process_group_id:
                active.append(int(candidate.name))
        return sorted(active)
    try:  # pragma: no cover - production monitor runs on Linux
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return []
    except PermissionError:
        return [process_group_id]
    return [process_group_id]


def _kill_process_group(process_group_id: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGKILL)


def _wait_process_group_empty(
    process_group_id: int, timeout_seconds: float
) -> list[int]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        active = _active_process_group_pids(process_group_id)
        if not active or time.monotonic() >= deadline:
            return active
        time.sleep(0.02)


def _run_gpu_sample(
    command: Sequence[str], command_timeout_seconds: float
) -> tuple[subprocess.CompletedProcess[str], bool, list[int]]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    stdout = ""
    stderr = ""
    try:
        stdout, stderr = process.communicate(timeout=command_timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(process.pid)
        try:
            stdout, stderr = process.communicate(
                timeout=max(1.0, command_timeout_seconds)
            )
        except subprocess.TimeoutExpired:
            _kill_process_group(process.pid)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1.0)
            stderr = (
                "sampler process group did not close its output pipes after SIGKILL\n"
            )
    active = _active_process_group_pids(process.pid)
    if active:
        _kill_process_group(process.pid)
        active = _wait_process_group_empty(process.pid, timeout_seconds=1.0)
    result = subprocess.CompletedProcess(
        args=list(command),
        returncode=process.returncode
        if process.returncode is not None
        else -signal.SIGKILL,
        stdout=stdout or "",
        stderr=stderr or "",
    )
    return result, timed_out, active


def run_gpu_monitor(
    *,
    parent_pid: int,
    parent_start_ticks: str,
    output_path: Path,
    stderr_path: Path,
    ready_path: Path,
    receipt_path: Path,
    nvidia_smi: str,
    interval_seconds: float,
    command_timeout_seconds: float,
) -> int:
    if not process_identity_alive(parent_pid, parent_start_ticks):
        raise LifecycleError("GPU monitor parent identity is not alive")
    stopped = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    _atomic_write_json(
        ready_path,
        {
            "schema": "amg_gpu_monitor_start_v1",
            "status": "ready",
            "pid": os.getpid(),
            "start_ticks": process_start_ticks(os.getpid()),
            "parent_pid": parent_pid,
            "parent_start_ticks": str(parent_start_ticks),
            "command_timeout_seconds": command_timeout_seconds,
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    samples = 0
    timeouts = 0
    errors = 0
    sampler_cleanup_uncertain = 0
    process_groups_started = 0
    process_groups_cleaned = 0
    active_sampler_process_groups = 0
    command = [
        nvidia_smi,
        "--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    with (
        output_path.open("a", encoding="utf-8", buffering=1) as output,
        stderr_path.open("a", encoding="utf-8", buffering=1) as errors_stream,
    ):
        while not stopped and process_identity_alive(parent_pid, parent_start_ticks):
            started = time.monotonic()
            try:
                process_groups_started += 1
                result, timed_out, active_group_pids = _run_gpu_sample(
                    command, command_timeout_seconds
                )
                if active_group_pids:
                    active_sampler_process_groups += 1
                    errors += 1
                    errors_stream.write(
                        f"{time.time():.6f} nvidia-smi process-group cleanup failed "
                        f"active_pids={active_group_pids}\n"
                    )
                    break
                process_groups_cleaned += 1
                if result.stdout:
                    output.write(result.stdout)
                    if not result.stdout.endswith("\n"):
                        output.write("\n")
                if result.stderr:
                    errors_stream.write(result.stderr)
                    if not result.stderr.endswith("\n"):
                        errors_stream.write("\n")
                if timed_out:
                    timeouts += 1
                    errors_stream.write(
                        f"{time.time():.6f} nvidia-smi timeout="
                        f"{command_timeout_seconds}\n"
                    )
                elif result.returncode:
                    errors += 1
                    errors_stream.write(
                        f"{time.time():.6f} nvidia-smi returncode={result.returncode}\n"
                    )
                else:
                    samples += 1
            except OSError as error:
                errors += 1
                sampler_cleanup_uncertain += 1
                errors_stream.write(
                    f"{time.time():.6f} nvidia-smi sampler state unknown={error}\n"
                )
            elapsed = time.monotonic() - started
            remaining = max(0.0, interval_seconds - elapsed)
            deadline = time.monotonic() + remaining
            while not stopped and time.monotonic() < deadline:
                if not process_identity_alive(parent_pid, parent_start_ticks):
                    break
                time.sleep(min(0.1, deadline - time.monotonic()))
    status = (
        "pass"
        if active_sampler_process_groups == 0 and sampler_cleanup_uncertain == 0
        else "fail"
    )
    _atomic_write_json(
        receipt_path,
        {
            "schema": "amg_gpu_monitor_exit_v1",
            "status": status,
            "mode": "signal" if stopped else "parent_death",
            "samples": samples,
            "timeouts": timeouts,
            "errors": errors,
            "sampler_cleanup_uncertain": sampler_cleanup_uncertain,
            "sampler_process_groups_started": process_groups_started,
            "sampler_process_groups_cleaned": process_groups_cleaned,
            "active_sampler_process_groups": active_sampler_process_groups,
            "pid": os.getpid(),
        },
    )
    return 0 if status == "pass" else 1


def _probe_writable_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise LifecycleError(f"capacity path must be a real directory: {path}")
    fd, raw = tempfile.mkstemp(prefix=".amg-capacity-probe.", dir=path)
    probe = Path(raw)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(b"capacity-probe\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        with contextlib.suppress(FileNotFoundError):
            probe.unlink()


def _decode_mountinfo_path(raw: str) -> str:
    return (
        raw.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _filesystem_device(path: Path) -> int:
    return os.stat(path).st_dev


def _filesystem_identity(path: Path) -> dict[str, str] | None:
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return None
    resolved = path.resolve()
    candidates: list[dict[str, str]] = []
    for raw in mountinfo.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        try:
            separator = fields.index("-")
            mountpoint = Path(_decode_mountinfo_path(fields[4]))
            filesystem_type = fields[separator + 1]
            source = _decode_mountinfo_path(fields[separator + 2])
        except (IndexError, ValueError):
            continue
        if resolved == mountpoint or mountpoint in resolved.parents:
            candidates.append(
                {
                    "mountpoint": str(mountpoint),
                    "filesystem_type": filesystem_type,
                    "source": source,
                }
            )
    if not candidates:
        raise LifecycleError(f"no mountinfo entry contains capacity path: {path}")
    return max(candidates, key=lambda item: len(item["mountpoint"]))


def capacity_admission(
    *,
    volatile_path: Path,
    persistent_path: Path,
    checkpoint_bytes: int,
    volatile_checkpoint_copies: int,
    persistent_checkpoint_copies: int,
    volatile_margin_bytes: int,
    persistent_margin_bytes: int,
    output_path: Path,
    require_distinct_filesystems: bool = False,
    expected_persistent_filesystem_types: Sequence[str] = (),
) -> dict[str, Any]:
    for number, label in (
        (checkpoint_bytes, "checkpoint_bytes"),
        (volatile_checkpoint_copies, "volatile_checkpoint_copies"),
        (persistent_checkpoint_copies, "persistent_checkpoint_copies"),
        (volatile_margin_bytes, "volatile_margin_bytes"),
        (persistent_margin_bytes, "persistent_margin_bytes"),
    ):
        if number < 0:
            raise LifecycleError(f"{label} must be non-negative")
    _probe_writable_directory(volatile_path)
    _probe_writable_directory(persistent_path)
    volatile_usage = shutil.disk_usage(volatile_path)
    persistent_usage = shutil.disk_usage(persistent_path)
    volatile_device = _filesystem_device(volatile_path)
    persistent_device = _filesystem_device(persistent_path)
    shared_filesystem = volatile_device == persistent_device
    persistent_filesystem = _filesystem_identity(persistent_path)
    volatile_required = (
        checkpoint_bytes * volatile_checkpoint_copies + volatile_margin_bytes
    )
    persistent_required = (
        checkpoint_bytes * persistent_checkpoint_copies + persistent_margin_bytes
    )
    report = {
        "schema": "amg_persistence_capacity_admission_v1",
        "status": "pass",
        "checkpoint_bytes": checkpoint_bytes,
        "shared_filesystem": shared_filesystem,
        "volatile_device": volatile_device,
        "persistent_device": persistent_device,
        "persistent_filesystem": persistent_filesystem,
        "required_persistent_filesystem_types": list(
            expected_persistent_filesystem_types
        ),
        "require_distinct_filesystems": require_distinct_filesystems,
        "volatile": {
            "path": str(volatile_path),
            "checkpoint_copies": volatile_checkpoint_copies,
            "margin_bytes": volatile_margin_bytes,
            "required_bytes": volatile_required,
            "total_bytes": volatile_usage.total,
            "used_bytes": volatile_usage.used,
            "free_bytes": volatile_usage.free,
        },
        "persistent": {
            "path": str(persistent_path),
            "checkpoint_copies": persistent_checkpoint_copies,
            "margin_bytes": persistent_margin_bytes,
            "required_bytes": persistent_required,
            "total_bytes": persistent_usage.total,
            "used_bytes": persistent_usage.used,
            "free_bytes": persistent_usage.free,
        },
    }
    failures = []
    if require_distinct_filesystems and shared_filesystem:
        failures.append("volatile and persistent paths resolve to the same filesystem")
    if expected_persistent_filesystem_types:
        observed_type = (persistent_filesystem or {}).get("filesystem_type")
        if observed_type not in set(expected_persistent_filesystem_types):
            failures.append(
                f"persistent filesystem type={observed_type!r} is not one of "
                f"{sorted(set(expected_persistent_filesystem_types))}"
            )
    if shared_filesystem:
        combined_required = volatile_required + persistent_required
        report["shared_filesystem_required_bytes"] = combined_required
        shared_free = min(volatile_usage.free, persistent_usage.free)
        report["shared_filesystem_free_bytes"] = shared_free
        if shared_free < combined_required:
            failures.append(
                f"shared filesystem free={shared_free} required={combined_required}"
            )
    else:
        if volatile_usage.free < volatile_required:
            failures.append(
                f"volatile free={volatile_usage.free} required={volatile_required}"
            )
        if persistent_usage.free < persistent_required:
            failures.append(
                f"persistent free={persistent_usage.free} "
                f"required={persistent_required}"
            )
    if failures:
        report["status"] = "fail"
        report["failures"] = failures
    _atomic_write_json(output_path, report)
    if failures:
        raise LifecycleError("capacity admission failed: " + "; ".join(failures))
    return report


def _publication_version(path: Path) -> int | None:
    match = re.fullmatch(r"openmle-fast-rich-v([0-9]+)-publication", path.parent.name)
    return int(match.group(1)) if match else None


def _assert_no_symlink_components(path: Path, anchor: Path, label: str) -> None:
    """Reject parent traversal and symlinks from a trusted lexical anchor."""

    if ".." in Path(os.fspath(path)).parts:
        raise LifecycleError(f"{label} path contains parent traversal: {path}")
    absolute = Path(os.path.abspath(os.fspath(path)))
    absolute_anchor = Path(os.path.abspath(os.fspath(anchor)))
    if absolute_anchor != absolute and absolute_anchor not in absolute.parents:
        raise LifecycleError(
            f"{label} path {absolute} is outside trusted anchor {absolute_anchor}"
        )
    components = [absolute]
    while components[-1] != absolute_anchor:
        components.append(components[-1].parent)
    for component in reversed(components):
        try:
            mode = os.lstat(component).st_mode
        except OSError as error:
            raise LifecycleError(
                f"cannot attest {label} path component {component}: {error}"
            ) from error
        if stat.S_ISLNK(mode):
            raise LifecycleError(
                f"{label} path contains symlink component: {component}"
            )


def select_latest_publication(
    *,
    registry_root: Path,
    receipt_glob: str,
    fixture_receipt: Path,
    fixture_lock: Path,
    fixture_certificate: Path,
    output_path: Path,
) -> dict[str, Any]:
    registry_root = Path(os.path.abspath(os.fspath(registry_root)))
    _assert_no_symlink_components(
        registry_root, registry_root, "publication registry root"
    )
    if ".." in Path(receipt_glob).parts:
        raise LifecycleError(
            f"publication receipt glob contains parent traversal: {receipt_glob}"
        )
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for raw in glob.glob(receipt_glob):
        path = Path(raw)
        version = _publication_version(path)
        if version is None:
            continue
        _assert_no_symlink_components(path, registry_root, "publication receipt")
        if not path.is_file():
            raise LifecycleError(f"publication receipt is not a regular file: {path}")
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if receipt.get("status") == "pass":
            candidates.append((version, path, receipt))
    if not candidates:
        raise LifecycleError("no sealed pass publication is available")
    version = max(item[0] for item in candidates)
    latest = [item for item in candidates if item[0] == version]
    if len(latest) != 1:
        raise LifecycleError(
            f"ambiguous latest sealed publication v{version}: "
            f"{[str(item[1]) for item in latest]}"
        )
    _, receipt_path, _receipt = latest[0]
    publication_root = receipt_path.parent
    checks = [
        (receipt_path, fixture_receipt, "publication receipt"),
        (publication_root / "artifacts/source-lock.json", fixture_lock, "source lock"),
        (
            publication_root / "artifacts/formal100-schedule-certificate.json",
            fixture_certificate,
            "formal schedule certificate",
        ),
    ]
    fixture_anchor = Path(
        os.path.commonpath([fixture_receipt, fixture_lock, fixture_certificate])
    )
    for selected, fixture, label in checks:
        _assert_no_symlink_components(selected, publication_root, f"latest {label}")
        _assert_no_symlink_components(fixture, fixture_anchor, f"staged {label}")
        if not selected.is_file():
            raise LifecycleError(f"latest {label} is missing: {selected}")
        if not fixture.is_file():
            raise LifecycleError(f"staged {label} is missing: {fixture}")
        if _sha256(selected) != _sha256(fixture):
            raise LifecycleError(
                f"staged fixture is stale versus latest v{version} {label}"
            )
    lock = _load_json(fixture_lock)
    certificate = _load_json(fixture_certificate)
    selection = {
        "schema": "amg_latest_openmle_publication_selection_v1",
        "status": "pass",
        "version": version,
        "registry_root": str(registry_root),
        "publication_root": str(publication_root),
        "publication_receipt_sha256": _sha256(receipt_path),
        "source_lock_sha256": _sha256(fixture_lock),
        "schedule_certificate_sha256": _sha256(fixture_certificate),
        "task_count": certificate["task_count"],
        "source_family_count": certificate["source_family_count"],
        "episodes": certificate["scheduled_episode_count"],
        "optimizer_updates": certificate["optimizer_updates"],
        "manifest_sha256": certificate["manifest_sha256"],
        "schedule_sha256": certificate["output_sha256"],
        "pod_root": lock["integration"]["pod_root"],
    }
    _atomic_write_json(output_path, selection)
    return selection


def assert_publication_selection_unchanged(
    first_path: Path, second_path: Path, receipt_path: Path
) -> dict[str, Any]:
    first = _load_json(first_path)
    second = _load_json(second_path)
    if first != second:
        report = {
            "schema": "amg_publication_launch_race_check_v1",
            "status": "fail",
            "first": first,
            "second": second,
        }
        _atomic_write_json(receipt_path, report)
        raise LifecycleError("latest sealed publication changed before trainer launch")
    report = {
        "schema": "amg_publication_launch_race_check_v1",
        "status": "pass",
        "selection": first,
    }
    _atomic_write_json(receipt_path, report)
    return report


def _find_symlinks(root: Path) -> list[str]:
    if root.is_symlink():
        return [str(root)]
    found: list[str] = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*dirs, *files]:
            candidate = base / name
            if candidate.is_symlink():
                found.append(str(candidate.relative_to(root)))
    return sorted(found)


def _validate_source_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise LifecycleError(
            f"source root is missing, symlinked, or not a directory: {root}"
        )
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in dirs:
            path = base / name
            if path.is_symlink() or not path.is_dir():
                raise LifecycleError(
                    f"source tree contains a non-directory entry: {path}"
                )
        for name in names:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise LifecycleError(f"source tree contains a non-regular file: {path}")


def _regular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in dirs:
            if (base / name).is_symlink():
                raise LifecycleError(f"staged directory symlink: {base / name}")
        for name in names:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise LifecycleError(f"staged non-regular file: {path}")
            files.append(path)
    return sorted(files, key=lambda path: str(path.relative_to(root)))


def _hash_manifest(root: Path, files: Sequence[Path]) -> list[tuple[str, str]]:
    return [(_sha256(path), str(path.relative_to(root))) for path in files]


def _manifest_text(rows: Sequence[tuple[str, str]]) -> str:
    return "".join(f"{digest}  {relative}\n" for digest, relative in rows)


def _run_rsync(arguments: Sequence[str]) -> None:
    command = ["rsync", "-a", *arguments]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode:
        raise LifecycleError(
            f"rsync failed rc={result.returncode}: {result.stderr.strip()}"
        )


def _checkpoint_paths(run_dir: Path, step: int) -> tuple[Path, Path]:
    tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    checkpoint = run_dir / f"checkpoints/global_step_{step}"
    if tracker.is_symlink() or not tracker.is_file():
        raise LifecycleError(f"checkpoint tracker is missing or symlinked: {tracker}")
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise LifecycleError(f"final checkpoint is missing or symlinked: {checkpoint}")
    if tracker.read_text(encoding="utf-8").strip() != str(step):
        raise LifecycleError("checkpoint tracker does not match requested final step")
    return tracker, checkpoint


def _audit_and_discard_gate_checkpoints(run_dir: Path) -> dict[str, Any]:
    checkpoint_root = run_dir / "checkpoints"
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise LifecycleError("gate checkpoint root is missing or symlinked")
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if tracker.is_symlink() or not tracker.is_file():
        raise LifecycleError("gate checkpoint tracker is missing or symlinked")
    raw_step = tracker.read_text(encoding="utf-8").strip()
    if not raw_step.isdigit():
        raise LifecycleError(f"invalid gate checkpoint step: {raw_step!r}")
    step = int(raw_step)
    _tracker, checkpoint = _checkpoint_paths(run_dir, step)
    files = _regular_files(checkpoint)
    rows = _hash_manifest(run_dir, [*files, tracker])
    manifest_path = run_dir / "gate-checkpoint-before-delete.sha256"
    _atomic_write_text(manifest_path, _manifest_text(rows), mode=0o600)
    size_bytes = sum(path.stat().st_size for path in files) + tracker.stat().st_size
    receipt = {
        "schema": "amg_gate_checkpoint_discard_v1",
        "status": "pass",
        "checkpoint_step": step,
        "file_count": len(rows),
        "size_bytes": size_bytes,
        "manifest_sha256": _sha256(manifest_path),
    }
    shutil.rmtree(checkpoint_root)
    if checkpoint_root.exists() or checkpoint_root.is_symlink():
        raise LifecycleError("gate checkpoint deletion did not complete")
    receipt["deleted"] = True
    _atomic_write_json(run_dir / "gate-checkpoint-deletion.json", receipt)
    return receipt


def _validate_launcher_exit(run_dir: Path, run_id: str) -> None:
    path = run_dir / "launcher-exit.env"
    if path.is_symlink() or not path.is_file():
        raise LifecycleError(f"launcher exit receipt is missing or symlinked: {path}")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if not separator or not key or key in values:
            raise LifecycleError(f"invalid launcher exit receipt line: {raw!r}")
        values[key] = value
    required = {
        "trainer_exit_code": "0",
        "cleanup_status": "pass",
        "publication_status": "ready_for_atomic_publication",
        "run_id": run_id,
    }
    mismatches = {
        key: {"expected": expected, "observed": values.get(key)}
        for key, expected in required.items()
        if values.get(key) != expected
    }
    if mismatches:
        raise LifecycleError(
            f"launcher exit receipt is not terminally clean: {mismatches}"
        )


def atomic_publish_run(
    *,
    run_dir: Path,
    persist_root: Path,
    run_id: str,
    mode: str,
    checkpoint_step: int | None,
    discard_gate_checkpoints: bool,
    terminal_exit: bool = False,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise LifecycleError(f"unsafe run id: {run_id!r}")
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise LifecycleError(f"run directory is missing or symlinked: {run_dir}")
    if persist_root.is_symlink() or not persist_root.is_dir():
        raise LifecycleError(f"persistent root is missing or symlinked: {persist_root}")
    _validate_launcher_exit(run_dir, run_id)
    _validate_source_tree(run_dir)
    if discard_gate_checkpoints:
        if mode != "gate" or checkpoint_step is not None:
            raise LifecycleError(
                "checkpoint discard is valid only for gate publication"
            )
        _audit_and_discard_gate_checkpoints(run_dir)
    final_path = persist_root / run_id
    if final_path.exists() or final_path.is_symlink():
        raise LifecycleError(f"persistent destination already exists: {final_path}")
    stage_path = persist_root / f".{run_id}.publish.tmp.{os.getpid()}"
    if stage_path.exists() or stage_path.is_symlink():
        raise LifecycleError(f"publication stage already exists: {stage_path}")
    stage_path.mkdir(mode=0o700)
    try:
        _run_rsync(["--exclude=/checkpoints/", f"{run_dir}/", f"{stage_path}/"])
        checkpoint_rows: list[tuple[str, str]] = []
        if checkpoint_step is not None:
            tracker, checkpoint = _checkpoint_paths(run_dir, checkpoint_step)
            source_files = _regular_files(checkpoint)
            source_rows = _hash_manifest(run_dir, [*source_files, tracker])
            source_manifest = run_dir / "final-checkpoint-source.sha256"
            _atomic_write_text(source_manifest, _manifest_text(source_rows), mode=0o600)
            # The source manifest was created after the first non-checkpoint copy.
            shutil.copy2(source_manifest, stage_path / source_manifest.name)
            (stage_path / "checkpoints").mkdir(mode=0o700)
            _run_rsync([str(checkpoint), f"{stage_path}/checkpoints/"])
            shutil.copy2(
                tracker,
                stage_path / "checkpoints/latest_checkpointed_iteration.txt",
            )
            staged_checkpoint = (
                stage_path / f"checkpoints/global_step_{checkpoint_step}"
            )
            staged_tracker = (
                stage_path / "checkpoints/latest_checkpointed_iteration.txt"
            )
            target_files = _regular_files(staged_checkpoint)
            target_rows = _hash_manifest(stage_path, [*target_files, staged_tracker])
            if source_rows != target_rows:
                raise LifecycleError(
                    "final checkpoint source/target SHA256 manifests differ"
                )
            checkpoint_rows = target_rows
            _atomic_write_text(
                stage_path / "final-checkpoint-target.sha256",
                _manifest_text(target_rows),
                mode=0o600,
            )
        staged_symlinks = _find_symlinks(stage_path)
        if staged_symlinks:
            raise LifecycleError(
                f"staged tree contains symlinks: {staged_symlinks[:20]}"
            )
        terminal_receipt = {
            "schema": "amg_terminal_publisher_v1",
            "status": "complete_when_public",
            "run_id": run_id,
            "linearization_point": "atomic_directory_rename",
            "post_rename_work": "none",
            "process_transition": "os._exit(0)_immediately_after_rename",
            "launcher_exit_sha256": _sha256(run_dir / "launcher-exit.env"),
        }
        if terminal_exit:
            _atomic_write_json(stage_path / "TERMINAL-PUBLISHER.json", terminal_receipt)
        metadata = {
            "schema": "amg_atomic_run_publication_v1",
            "status": "complete",
            "run_id": run_id,
            "mode": mode,
            "checkpoint_step": checkpoint_step,
            "checkpoint_file_count": len(checkpoint_rows),
            "terminal_publisher": terminal_exit,
            "published_unix": time.time(),
        }
        if terminal_exit:
            metadata["terminal_publisher_sha256"] = _sha256(
                stage_path / "TERMINAL-PUBLISHER.json"
            )
        _atomic_write_json(stage_path / "PUBLICATION-COMPLETE.json", metadata)
        manifest_path = stage_path / "TREE-SHA256SUMS"
        files = [path for path in _regular_files(stage_path) if path != manifest_path]
        rows_by_relative = {relative: digest for digest, relative in checkpoint_rows}
        tree_rows: list[tuple[str, str]] = []
        for path in files:
            relative = str(path.relative_to(stage_path))
            digest = rows_by_relative.get(relative) or _sha256(path)
            tree_rows.append((digest, relative))
        _atomic_write_text(manifest_path, _manifest_text(tree_rows), mode=0o600)
        # Verify all small files and rely on the already repeated source/target
        # checkpoint pass for the large checkpoint subtree.
        for digest, relative in tree_rows:
            if relative.startswith("checkpoints/global_step_"):
                continue
            observed = _sha256(stage_path / relative)
            if observed != digest:
                raise LifecycleError(f"staged tree hash mismatch: {relative}")
        if _find_symlinks(stage_path):
            raise LifecycleError("staged tree gained a symlink before publication")
        report = {
            **metadata,
            "persistent_path": str(final_path),
            "tree_manifest_sha256": _sha256(manifest_path),
        }
        _fsync_directory(stage_path)
        lock_path = persist_root / ".amg-atomic-publication.lock"
        with _exclusive_lock(lock_path):
            if final_path.exists() or final_path.is_symlink():
                raise LifecycleError(
                    f"persistent destination appeared during publication: {final_path}"
                )
            os.rename(stage_path, final_path)
            if terminal_exit:
                os._exit(0)
            _fsync_directory(persist_root)
        if not (final_path / "PUBLICATION-COMPLETE.json").is_file():
            raise LifecycleError("atomically published tree lacks completion receipt")
        if not (final_path / "TREE-SHA256SUMS").is_file():
            raise LifecycleError("atomically published tree lacks SHA256 manifest")
        return report
    except Exception:
        # Keep an incomplete hidden stage for forensic inspection.  It never
        # acquires the public final path and therefore cannot look complete.
        raise


def terminal_atomic_publish_run(
    *,
    run_dir: Path,
    persist_root: Path,
    run_id: str,
    mode: str,
    checkpoint_step: int | None,
    discard_gate_checkpoints: bool,
) -> NoReturn:
    atomic_publish_run(
        run_dir=run_dir,
        persist_root=persist_root,
        run_id=run_id,
        mode=mode,
        checkpoint_step=checkpoint_step,
        discard_gate_checkpoints=discard_gate_checkpoints,
        terminal_exit=True,
    )
    raise LifecycleError("terminal publisher returned after public rename")


def _parse_original(value: str) -> str | None:
    return value if value else None


def _cmd_process_identity_alive(args: argparse.Namespace) -> None:
    raise SystemExit(0 if process_identity_alive(args.pid, args.start_ticks) else 1)


def _cmd_marker_read(args: argparse.Namespace) -> None:
    value = _read_marker(Path(args.path))
    if value is not None:
        print(value)


def _cmd_marker_prepare(args: argparse.Namespace) -> None:
    markers = [
        _marker_record(
            "cpu",
            Path(args.cpu_path),
            _parse_original(args.cpu_original_value),
            args.cpu_original_pid,
            args.cpu_original_start_ticks,
        ),
        _marker_record(
            "gpu",
            Path(args.gpu_path),
            _parse_original(args.gpu_original_value),
            args.gpu_original_pid,
            args.gpu_original_start_ticks,
        ),
    ]
    state = prepare_marker_transaction(
        state_path=Path(args.state),
        lock_path=Path(args.lock),
        run_id=args.run_id,
        parent_pid=args.parent_pid,
        parent_start_ticks=args.parent_start_ticks,
        markers=markers,
    )
    print(json.dumps(state, sort_keys=True))


def _cmd_marker_acquire(args: argparse.Namespace) -> None:
    state = acquire_marker_transaction(Path(args.state), Path(args.lock))
    print(json.dumps(state, sort_keys=True))


def _cmd_marker_restore(args: argparse.Namespace) -> None:
    state = restore_marker_transaction(Path(args.state), Path(args.lock))
    print(json.dumps(state, sort_keys=True))


def _cmd_marker_status(args: argparse.Namespace) -> None:
    state = _load_marker_state(Path(args.state))
    print(json.dumps(state, sort_keys=True))
    if args.require and state["status"] != args.require:
        raise LifecycleError(
            f"marker state status={state['status']!r}, required={args.require!r}"
        )


def _cmd_marker_watch(args: argparse.Namespace) -> None:
    rc = watch_marker_transaction(
        state_path=Path(args.state),
        lock_path=Path(args.lock),
        parent_pid=args.parent_pid,
        parent_start_ticks=args.parent_start_ticks,
        ready_path=Path(args.ready),
        receipt_path=Path(args.receipt),
        poll_seconds=args.poll_seconds,
        restore_timeout_seconds=args.restore_timeout_seconds,
    )
    raise SystemExit(rc)


def _cmd_gpu_monitor(args: argparse.Namespace) -> None:
    rc = run_gpu_monitor(
        parent_pid=args.parent_pid,
        parent_start_ticks=args.parent_start_ticks,
        output_path=Path(args.output),
        stderr_path=Path(args.stderr),
        ready_path=Path(args.ready),
        receipt_path=Path(args.receipt),
        nvidia_smi=args.nvidia_smi,
        interval_seconds=args.interval_seconds,
        command_timeout_seconds=args.command_timeout_seconds,
    )
    raise SystemExit(rc)


def _cmd_capacity(args: argparse.Namespace) -> None:
    report = capacity_admission(
        volatile_path=Path(args.volatile_path),
        persistent_path=Path(args.persistent_path),
        checkpoint_bytes=args.checkpoint_bytes,
        volatile_checkpoint_copies=args.volatile_checkpoint_copies,
        persistent_checkpoint_copies=args.persistent_checkpoint_copies,
        volatile_margin_bytes=args.volatile_margin_bytes,
        persistent_margin_bytes=args.persistent_margin_bytes,
        output_path=Path(args.output),
        require_distinct_filesystems=args.require_distinct_filesystems,
        expected_persistent_filesystem_types=args.expected_persistent_fs_type,
    )
    print(json.dumps(report, sort_keys=True))


def _cmd_select_latest(args: argparse.Namespace) -> None:
    selection = select_latest_publication(
        registry_root=Path(args.registry_root),
        receipt_glob=args.receipt_glob,
        fixture_receipt=Path(args.fixture_receipt),
        fixture_lock=Path(args.fixture_lock),
        fixture_certificate=Path(args.fixture_certificate),
        output_path=Path(args.output),
    )
    print(json.dumps(selection, sort_keys=True))


def _cmd_assert_selection(args: argparse.Namespace) -> None:
    report = assert_publication_selection_unchanged(
        Path(args.first), Path(args.second), Path(args.output)
    )
    print(json.dumps(report, sort_keys=True))


def _cmd_exec_after_publication_check(args: argparse.Namespace) -> None:
    if not args.exec_command:
        raise LifecycleError("exec-after-publication-check requires a command")
    environment = os.environ.copy()
    for key in args.unset_env:
        environment.pop(key, None)
    command = list(args.exec_command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise LifecycleError("exec-after-publication-check requires a command")
    registry_lock = Path(args.registry_lock)
    with _shared_lock(registry_lock):
        select_latest_publication(
            registry_root=Path(args.registry_root),
            receipt_glob=args.receipt_glob,
            fixture_receipt=Path(args.fixture_receipt),
            fixture_lock=Path(args.fixture_lock),
            fixture_certificate=Path(args.fixture_certificate),
            output_path=Path(args.selection_output),
        )
        report = assert_publication_selection_unchanged(
            Path(args.first), Path(args.selection_output), Path(args.check_output)
        )
        report.update(
            {
                "registry_lock": str(registry_lock),
                "registry_lock_mode": "shared_until_exec",
                "linearization_point": ("final_selection_under_shared_registry_lock"),
                "publication_after_linearization": "belongs_to_next_lineage",
            }
        )
        _atomic_write_json(Path(args.check_output), report)
        # The lock descriptor is explicitly CLOEXEC.  A cooperating publication
        # writer cannot seal a newer version between the final selection and
        # this exec boundary; any version sealed after it belongs to the next
        # lineage by contract.
        os.execvpe(command[0], command, environment)


def _cmd_atomic_publish(args: argparse.Namespace) -> None:
    report = atomic_publish_run(
        run_dir=Path(args.run_dir),
        persist_root=Path(args.persist_root),
        run_id=args.run_id,
        mode=args.mode,
        checkpoint_step=args.checkpoint_step,
        discard_gate_checkpoints=args.discard_gate_checkpoints,
    )
    print(json.dumps(report, sort_keys=True))


def _cmd_terminal_publish(args: argparse.Namespace) -> None:
    terminal_atomic_publish_run(
        run_dir=Path(args.run_dir),
        persist_root=Path(args.persist_root),
        run_id=args.run_id,
        mode=args.mode,
        checkpoint_step=args.checkpoint_step,
        discard_gate_checkpoints=args.discard_gate_checkpoints,
    )


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    alive = subparsers.add_parser("process-identity-alive")
    alive.add_argument("--pid", required=True, type=int)
    alive.add_argument("--start-ticks", required=True)
    alive.set_defaults(handler=_cmd_process_identity_alive)

    marker_read = subparsers.add_parser("marker-read")
    marker_read.add_argument("--path", required=True)
    marker_read.set_defaults(handler=_cmd_marker_read)

    prepare = subparsers.add_parser("marker-prepare")
    prepare.add_argument("--state", required=True)
    prepare.add_argument("--lock", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--parent-pid", required=True, type=int)
    prepare.add_argument("--parent-start-ticks", required=True)
    for name in ("cpu", "gpu"):
        prepare.add_argument(f"--{name}-path", required=True)
        prepare.add_argument(f"--{name}-original-value", default="")
        prepare.add_argument(f"--{name}-original-pid", type=int, default=0)
        prepare.add_argument(f"--{name}-original-start-ticks", default="")
    prepare.set_defaults(handler=_cmd_marker_prepare)

    for command, handler in (
        ("marker-acquire", _cmd_marker_acquire),
        ("marker-restore", _cmd_marker_restore),
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--state", required=True)
        child.add_argument("--lock", required=True)
        child.set_defaults(handler=handler)

    status = subparsers.add_parser("marker-status")
    status.add_argument("--state", required=True)
    status.add_argument("--require")
    status.set_defaults(handler=_cmd_marker_status)

    watch = subparsers.add_parser("marker-watch")
    watch.add_argument("--state", required=True)
    watch.add_argument("--lock", required=True)
    watch.add_argument("--parent-pid", required=True, type=int)
    watch.add_argument("--parent-start-ticks", required=True)
    watch.add_argument("--ready", required=True)
    watch.add_argument("--receipt", required=True)
    watch.add_argument("--poll-seconds", type=_positive_float, default=0.5)
    watch.add_argument("--restore-timeout-seconds", type=_positive_float, default=300)
    watch.set_defaults(handler=_cmd_marker_watch)

    monitor = subparsers.add_parser("gpu-monitor")
    monitor.add_argument("--parent-pid", required=True, type=int)
    monitor.add_argument("--parent-start-ticks", required=True)
    monitor.add_argument("--output", required=True)
    monitor.add_argument("--stderr", required=True)
    monitor.add_argument("--ready", required=True)
    monitor.add_argument("--receipt", required=True)
    monitor.add_argument("--nvidia-smi", default="nvidia-smi")
    monitor.add_argument("--interval-seconds", type=_positive_float, default=2)
    monitor.add_argument("--command-timeout-seconds", type=_positive_float, default=5)
    monitor.set_defaults(handler=_cmd_gpu_monitor)

    capacity = subparsers.add_parser("capacity-check")
    capacity.add_argument("--volatile-path", required=True)
    capacity.add_argument("--persistent-path", required=True)
    capacity.add_argument("--checkpoint-bytes", type=int, required=True)
    capacity.add_argument("--volatile-checkpoint-copies", type=int, required=True)
    capacity.add_argument("--persistent-checkpoint-copies", type=int, required=True)
    capacity.add_argument("--volatile-margin-bytes", type=int, required=True)
    capacity.add_argument("--persistent-margin-bytes", type=int, required=True)
    capacity.add_argument("--output", required=True)
    capacity.add_argument("--require-distinct-filesystems", action="store_true")
    capacity.add_argument("--expected-persistent-fs-type", action="append", default=[])
    capacity.set_defaults(handler=_cmd_capacity)

    select = subparsers.add_parser("select-latest-publication")
    select.add_argument("--registry-root", required=True)
    select.add_argument("--receipt-glob", required=True)
    select.add_argument("--fixture-receipt", required=True)
    select.add_argument("--fixture-lock", required=True)
    select.add_argument("--fixture-certificate", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(handler=_cmd_select_latest)

    unchanged = subparsers.add_parser("assert-publication-unchanged")
    unchanged.add_argument("--first", required=True)
    unchanged.add_argument("--second", required=True)
    unchanged.add_argument("--output", required=True)
    unchanged.set_defaults(handler=_cmd_assert_selection)

    checked_exec = subparsers.add_parser("exec-after-publication-check")
    checked_exec.add_argument("--first", required=True)
    checked_exec.add_argument("--registry-root", required=True)
    checked_exec.add_argument("--receipt-glob", required=True)
    checked_exec.add_argument("--fixture-receipt", required=True)
    checked_exec.add_argument("--fixture-lock", required=True)
    checked_exec.add_argument("--fixture-certificate", required=True)
    checked_exec.add_argument("--selection-output", required=True)
    checked_exec.add_argument("--check-output", required=True)
    checked_exec.add_argument("--registry-lock", required=True)
    checked_exec.add_argument("--unset-env", action="append", default=[])
    checked_exec.add_argument("exec_command", nargs=argparse.REMAINDER)
    checked_exec.set_defaults(handler=_cmd_exec_after_publication_check)

    publish = subparsers.add_parser("atomic-publish")
    publish.add_argument("--run-dir", required=True)
    publish.add_argument("--persist-root", required=True)
    publish.add_argument("--run-id", required=True)
    publish.add_argument("--mode", choices=("gate", "formal"), required=True)
    publish.add_argument("--checkpoint-step", type=int)
    publish.add_argument("--discard-gate-checkpoints", action="store_true")
    publish.set_defaults(handler=_cmd_atomic_publish)

    terminal_publish = subparsers.add_parser("terminal-publish")
    terminal_publish.add_argument("--run-dir", required=True)
    terminal_publish.add_argument("--persist-root", required=True)
    terminal_publish.add_argument("--run-id", required=True)
    terminal_publish.add_argument("--mode", choices=("gate", "formal"), required=True)
    terminal_publish.add_argument("--checkpoint-step", type=int)
    terminal_publish.add_argument("--discard-gate-checkpoints", action="store_true")
    terminal_publish.set_defaults(handler=_cmd_terminal_publish)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except LifecycleError as error:
        print(f"lifecycle contract failed: {error}", file=sys.stderr)
        return 78
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
