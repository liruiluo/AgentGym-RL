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
import ctypes
import errno
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
from collections.abc import Iterator, Mapping, Sequence
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


def _read_marker_with_metadata(
    path: Path,
) -> tuple[str | None, os.stat_result | None]:
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
        return None, None
    except OSError as error:
        raise LifecycleError(f"cannot open marker {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
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
    return value, metadata


def _read_marker(path: Path) -> str | None:
    value, _metadata = _read_marker_with_metadata(path)
    return value


def _marker_file_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISDIR(mode):
        return "directory"
    return "other"


def _add_marker_metadata(observed: dict[str, Any], metadata: os.stat_result) -> None:
    observed.update(
        {
            "exists": True,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
            "file_type": _marker_file_type(metadata.st_mode),
        }
    )


def _marker_observation(path: Path) -> dict[str, Any]:
    """Atomically read a regular marker's value plus opened-inode metadata."""

    observed: dict[str, Any] = {"path": str(path), "exists": False, "value": None}
    try:
        value, metadata = _read_marker_with_metadata(path)
    except LifecycleError as error:
        observed["error"] = f"{type(error).__name__}: {error}"
        try:
            diagnostic_metadata = os.lstat(path)
        except OSError:
            return observed
        _add_marker_metadata(observed, diagnostic_metadata)
        return observed
    observed["value"] = value
    if metadata is not None:
        _add_marker_metadata(observed, metadata)
    return observed


def _owned_marker_identity(observation: dict[str, Any]) -> dict[str, int]:
    required = ("device", "inode", "ctime_ns")
    if observation.get("value") is None or not all(
        isinstance(observation.get(key), int) for key in required
    ):
        raise LifecycleError(
            f"cannot record owned marker identity: {observation.get('path')}"
        )
    return {key: int(observation[key]) for key in required}


def _marker_identity_matches(
    observation: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
    *,
    include_ctime: bool,
) -> bool:
    if expected is None:
        return True
    keys = ("device", "inode", "ctime_ns") if include_ctime else ("device", "inode")
    return all(
        isinstance(expected.get(key), int)
        and observation.get(key) == expected[key]
        for key in keys
    )


def _write_marker(path: Path, value: str) -> None:
    if not value or "\n" in value:
        raise LifecycleError(f"invalid marker value for {path}: {value!r}")
    _atomic_write_text(path, value + "\n", mode=0o600)


def _marker_transition_token(transition_id: str) -> str:
    return hashlib.sha256(transition_id.encode("utf-8")).hexdigest()[:24]


def _marker_transition_backup(path: Path, transition_id: str) -> Path:
    token = _marker_transition_token(transition_id)
    return path.with_name(f".{path.name}.{token}.transition")


def _marker_transition_claim(path: Path, transition_id: str) -> Path:
    token = _marker_transition_token(transition_id)
    return path.with_name(f".{path.name}.{token}.claim")


def _inode_identity(observation: Mapping[str, Any]) -> dict[str, int]:
    required = ("device", "inode")
    if not all(isinstance(observation.get(key), int) for key in required):
        raise LifecycleError(
            f"cannot record marker inode identity: {observation.get('path')}"
        )
    return {key: int(observation[key]) for key in required}


def _create_marker_claim(
    path: Path,
    value: str,
    *,
    transition_id: str,
) -> dict[str, int]:
    """Create a fully-written retained claim without touching ``path``.

    The deterministic claim pathname must be absent.  A random staging hardlink
    is deliberately retained as another inode pin: cleanup by pathname would
    reintroduce the exact check/unlink race this transaction is designed to
    avoid.  These run-scoped files are tiny forensic evidence.
    """

    if not value or "\n" in value:
        raise LifecycleError(f"invalid marker value for {path}: {value!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    claim = _marker_transition_claim(path, transition_id)
    claim_observation = _marker_observation(claim)
    if claim_observation.get("exists") or claim_observation.get("error"):
        raise LifecycleError(
            f"marker transition claim already exists or is invalid: "
            f"{claim_observation!r}"
        )

    token = _marker_transition_token(transition_id)
    descriptor, raw_stage = tempfile.mkstemp(
        prefix=f".{path.name}.{token}.claim-stage.", dir=path.parent
    )
    stage = Path(raw_stage)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write((value + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            try:
                os.link(stage, claim, follow_symlinks=False)
            except TypeError:  # pragma: no cover - old Python compatibility
                os.link(stage, claim)
        except FileExistsError as error:
            raise LifecycleError(
                f"marker transition claim appeared during creation: {claim}"
            ) from error
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        # Retain ``stage``.  Removing a pathname after an identity check would
        # permit a non-cooperating writer to swap in a foreign inode first.

    claim_observation = _marker_observation(claim)
    if (
        claim_observation.get("value") != value
        or not _same_inode(stage, claim)
        or claim_observation.get("error")
    ):
        raise LifecycleError(
            f"marker transition claim changed during creation: "
            f"{claim_observation!r}"
        )
    return _inode_identity(claim_observation)


def _verify_marker_claim(
    path: Path,
    value: str,
    *,
    transition_id: str,
    expected_identity: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    claim = _marker_transition_claim(path, transition_id)
    observation = _marker_observation(claim)
    if (
        observation.get("value") != value
        or observation.get("error")
        or not _marker_identity_matches(
            observation, expected_identity, include_ctime=False
        )
    ):
        raise LifecycleError(
            f"marker transition claim mismatch for {path}: "
            f"wanted={value!r}/{dict(expected_identity)!r}, "
            f"observed={observation!r}"
        )
    return claim, observation


def _install_marker_claim(
    path: Path,
    value: str,
    *,
    transition_id: str,
    claim_identity: Mapping[str, Any],
) -> dict[str, int]:
    """Install the exact pre-recorded claim inode at an absent marker path."""

    claim, _claim_observation = _verify_marker_claim(
        path,
        value,
        transition_id=transition_id,
        expected_identity=claim_identity,
    )
    current = _marker_observation(path)
    if current.get("exists") or current.get("error"):
        if (
            current.get("value") == value
            and not current.get("error")
            and _marker_identity_matches(
                current, claim_identity, include_ctime=False
            )
        ):
            return _owned_marker_identity(current)
        raise LifecycleError(
            f"foreign marker appeared during exact claim install: {current!r}"
        )
    try:
        try:
            os.link(claim, path, follow_symlinks=False)
        except TypeError:  # pragma: no cover - old Python compatibility
            os.link(claim, path)
    except FileExistsError as error:
        current = _marker_observation(path)
        if not (
            current.get("value") == value
            and not current.get("error")
            and _marker_identity_matches(
                current, claim_identity, include_ctime=False
            )
        ):
            raise LifecycleError(
                f"foreign marker won exact claim install: {current!r}"
            ) from error
    _fsync_directory(path.parent)
    installed = _marker_observation(path)
    # Re-read both names after the link.  A writer may replace either pathname
    # between the precheck and link; never accept a same-value foreign inode.
    _verify_marker_claim(
        path,
        value,
        transition_id=transition_id,
        expected_identity=claim_identity,
    )
    if (
        installed.get("value") != value
        or installed.get("error")
        or not _marker_identity_matches(
            installed, claim_identity, include_ctime=False
        )
    ):
        raise LifecycleError(
            f"exact marker claim identity changed during install: {installed!r}"
        )
    return _owned_marker_identity(installed)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically move ``source`` only when ``destination`` is absent.

    A marker writer that ignores our advisory lock can replace the pathname at
    any time.  Linux ``renameat2(RENAME_NOREPLACE)`` and Darwin
    ``renamex_np(RENAME_EXCL)`` make pathname removal and quarantine install one
    kernel operation.  The caller validates the inode the kernel actually moved
    and never performs a later pathname unlink.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source)
    destination_raw = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        if function is None:
            raise LifecycleError(
                "renameat2(RENAME_NOREPLACE) is required for marker safety"
            )
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(-100, source_raw, -100, destination_raw, 1)
    elif sys.platform == "darwin":
        function = getattr(libc, "renamex_np", None)
        if function is None:  # pragma: no cover - all supported macOS has it
            raise LifecycleError("renamex_np(RENAME_EXCL) is unavailable")
        function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        function.restype = ctypes.c_int
        result = function(source_raw, destination_raw, 0x00000004)
    else:  # pragma: no cover - production and development are Linux/Darwin
        raise LifecycleError(
            f"no atomic no-replace rename primitive on {sys.platform}"
        )
    if result == 0:
        _fsync_directory(source.parent)
        if destination.parent != source.parent:
            _fsync_directory(destination.parent)
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    if error_number == errno.ENOENT:
        raise FileNotFoundError(error_number, os.strerror(error_number), source)
    raise OSError(
        error_number,
        f"atomic no-replace rename failed: {source} -> {destination}: "
        f"{os.strerror(error_number)}",
    )


def _restore_quarantined_marker(
    backup: Path,
    path: Path,
    *,
    moved_identity: Mapping[str, Any] | None = None,
) -> None:
    try:
        _rename_noreplace(backup, path)
    except FileExistsError as error:
        raise LifecycleError(
            f"foreign marker appeared while restoring quarantined marker: {path}"
        ) from error
    if moved_identity is not None:
        restored = _marker_observation(path)
        if not _marker_identity_matches(
            restored, moved_identity, include_ctime=False
        ):
            raise LifecycleError(
                f"quarantined marker identity changed during restore: {restored!r}"
            )


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
    """Atomically retain the current marker inode at a run-scoped path."""

    try:
        _rename_noreplace(path, backup)
    except (FileExistsError, FileNotFoundError):
        raise
    except OSError as error:
        raise LifecycleError(
            f"cannot quarantine marker without replacement {path}: {error}"
        ) from error


def _transition_backup_observation(
    path: Path,
    *,
    transition_id: str,
    expected: str,
    expected_identity: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    backup = _marker_transition_backup(path, transition_id)
    observation = _marker_observation(backup)
    if (
        observation.get("value") != expected
        or observation.get("error")
        or not _marker_identity_matches(
            observation, expected_identity, include_ctime=False
        )
    ):
        raise LifecycleError(
            f"marker transition backup mismatch for {path}: "
            f"expected={expected!r}/{dict(expected_identity)!r}, "
            f"backup={observation!r}"
        )
    return backup, observation


def _cas_marker(
    path: Path,
    expected: str | None,
    replacement: str | None,
    *,
    transition_id: str = "default",
    expected_identity: Mapping[str, Any] | None = None,
    replacement_claim_identity: Mapping[str, Any] | None = None,
) -> dict[str, int] | None:
    """Apply or recover an inode-authenticated marker transition.

    A replacement claim is created and its inode persisted by the transaction
    state *before* this function is called.  Source markers are retained at a
    deterministic quarantine pathname.  Neither pathname is unlinked, so a
    non-cooperating writer cannot swap in a foreign inode that cleanup deletes.
    The retained source plus retained claim make the operation idempotent across
    a crash after either quarantine or commit.
    """

    if expected is not None and expected_identity is None:
        raise LifecycleError(
            f"present marker transition requires expected inode identity: {path}"
        )
    if replacement is not None and replacement_claim_identity is None:
        raise LifecycleError(
            f"marker replacement requires a pre-recorded claim identity: {path}"
        )
    if replacement is not None:
        _verify_marker_claim(
            path,
            replacement,
            transition_id=transition_id,
            expected_identity=replacement_claim_identity or {},
        )

    backup = _marker_transition_backup(path, transition_id)
    backup_observation = _marker_observation(backup)
    if backup_observation.get("error"):
        raise LifecycleError(
            f"invalid marker transition backup for {path}: {backup_observation!r}"
        )
    backup_exists = bool(backup_observation.get("exists"))
    current_observation = _marker_observation(path)
    if current_observation.get("error"):
        raise LifecycleError(
            f"invalid current marker during CAS for {path}: {current_observation!r}"
        )

    replacement_already_installed = replacement is not None and (
        current_observation.get("value") == replacement
        and _marker_identity_matches(
            current_observation,
            replacement_claim_identity,
            include_ctime=False,
        )
    )
    deletion_already_installed = replacement is None and not current_observation.get(
        "exists", False
    )

    # Idempotent crash recovery after commit.  A present source must still have
    # the exact quarantined source inode; an originally absent source has no
    # backup and is authenticated solely by the retained claim inode.
    if replacement_already_installed or deletion_already_installed:
        if expected is None:
            if backup_exists:
                raise LifecycleError(
                    f"unexpected backup for absent-source transition: {backup}"
                )
        else:
            _transition_backup_observation(
                path,
                transition_id=transition_id,
                expected=expected,
                expected_identity=expected_identity or {},
            )
        if replacement is None:
            return None
        _verify_marker_claim(
            path,
            replacement,
            transition_id=transition_id,
            expected_identity=replacement_claim_identity or {},
        )
        return _owned_marker_identity(current_observation)

    if backup_exists:
        if expected is None:
            raise LifecycleError(
                f"unexpected transition backup for absent source marker: {backup}"
            )
        _transition_backup_observation(
            path,
            transition_id=transition_id,
            expected=expected,
            expected_identity=expected_identity or {},
        )
        if current_observation.get("exists", False):
            raise LifecycleError(
                f"marker transition recovery is ambiguous for {path}; "
                f"preserving current={current_observation!r} and backup={backup}"
            )
    else:
        current = current_observation.get("value")
        if current != expected or (
            expected is not None
            and not _marker_identity_matches(
                current_observation, expected_identity, include_ctime=True
            )
        ):
            raise LifecycleError(
                f"marker CAS mismatch for {path}: "
                f"expected={expected!r}/{expected_identity!r}, "
                f"current={current_observation!r}"
            )
        if expected is not None:
            moved_identity = _owned_marker_identity(current_observation)
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
            moved_observation = _marker_observation(backup)
            if (
                moved_observation.get("value") != expected
                or moved_observation.get("error")
                or not _marker_identity_matches(
                    moved_observation, moved_identity, include_ctime=False
                )
            ):
                try:
                    if moved_observation.get("exists", False) and not _marker_observation(
                        path
                    ).get("exists", False):
                        _restore_quarantined_marker(
                            backup,
                            path,
                            moved_identity=_inode_identity(moved_observation),
                        )
                finally:
                    raise LifecycleError(
                        f"marker changed before quarantine for {path}: "
                        f"expected={expected!r}/{moved_identity!r}, "
                        f"observed={moved_observation!r}"
                    )
            # Persistently retaining this inode is the commit witness.
            _transition_backup_observation(
                path,
                transition_id=transition_id,
                expected=expected,
                expected_identity=moved_identity,
            )

    if replacement is not None:
        installed_identity = _install_marker_claim(
            path,
            replacement,
            transition_id=transition_id,
            claim_identity=replacement_claim_identity or {},
        )
    else:
        installed_identity = None

    final_observation = _marker_observation(path)
    if replacement is None:
        if final_observation.get("exists") or final_observation.get("error"):
            raise LifecycleError(
                f"marker deletion transition did not remain absent: "
                f"{final_observation!r}"
            )
    elif (
        final_observation.get("value") != replacement
        or final_observation.get("error")
        or not _marker_identity_matches(
            final_observation, replacement_claim_identity, include_ctime=False
        )
    ):
        raise LifecycleError(
            f"marker CAS verification failed for {path}: "
            f"wanted={replacement!r}/{replacement_claim_identity!r}, "
            f"observed={final_observation!r}"
        )

    if expected is not None:
        _transition_backup_observation(
            path,
            transition_id=transition_id,
            expected=expected,
            expected_identity=expected_identity or {},
        )
    if replacement is not None:
        _verify_marker_claim(
            path,
            replacement,
            transition_id=transition_id,
            expected_identity=replacement_claim_identity or {},
        )
    return installed_identity


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
            observation = _marker_observation(path)
            if observation.get("error"):
                raise LifecycleError(
                    f"cannot inspect marker during prepare for {path}: "
                    f"{observation['error']}"
                )
            current = observation.get("value")
            if current != marker["original_value"]:
                raise LifecycleError(
                    f"marker changed before prepare for {path}: "
                    f"expected={marker['original_value']!r}, current={current!r}"
                )
            marker["original_file_identity"] = (
                _owned_marker_identity(observation) if current is not None else None
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


def _ensure_marker_claim_locked(
    state_path: Path,
    state: dict[str, Any],
    marker: dict[str, Any],
    *,
    phase: str,
    value: str | None,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    identity_key = f"{phase}_claim_identity"
    path_key = f"{phase}_claim_path"
    transition_id = f"{state['run_id']}:{marker['name']}:{phase}"
    identity = marker.get(identity_key)
    if identity is None:
        identity = _create_marker_claim(
            Path(marker["path"]), value, transition_id=transition_id
        )
        marker[identity_key] = identity
        marker[path_key] = str(
            _marker_transition_claim(Path(marker["path"]), transition_id)
        )
        # This save is intentionally before any mutation of the canonical
        # marker.  A post-commit crash can then authenticate the exact claim.
        _save_marker_state(state_path, state)
    _verify_marker_claim(
        Path(marker["path"]),
        value,
        transition_id=transition_id,
        expected_identity=identity,
    )
    return identity


def _restore_one_marker_locked(
    state_path: Path,
    state: dict[str, Any],
    marker: dict[str, Any],
) -> None:
    path = Path(marker["path"])
    acquire_transition_id = f"{state['run_id']}:{marker['name']}:acquire"
    current_observation = _marker_observation(path)
    if current_observation.get("error"):
        raise LifecycleError(
            f"cannot inspect marker during recovery: {current_observation!r}"
        )

    # Recover an acquisition that committed after ``acquire_started`` was saved
    # but before ``owned_identity``/``acquired`` reached the state file.
    if marker.get("acquire_started", False) and not marker["acquired"]:
        backup_observation = _marker_observation(
            _marker_transition_backup(path, acquire_transition_id)
        )
        if backup_observation.get("error"):
            raise LifecycleError(
                f"invalid acquisition backup during recovery: "
                f"{backup_observation!r}"
            )
        claim_identity = marker.get("acquire_claim_identity")
        current_is_exact_claim = bool(
            claim_identity
            and current_observation.get("value") == state["run_id"]
            and _marker_identity_matches(
                current_observation, claim_identity, include_ctime=False
            )
        )
        mutation_evidence = bool(
            current_is_exact_claim or backup_observation.get("exists", False)
        )
        if mutation_evidence:
            if claim_identity is None:
                raise LifecycleError(
                    f"acquisition changed marker without a recorded claim: {path}"
                )
            installed_identity = _cas_marker(
                path,
                marker["original_value"],
                state["run_id"],
                transition_id=acquire_transition_id,
                expected_identity=marker.get("original_file_identity"),
                replacement_claim_identity=claim_identity,
            )
            if installed_identity is None:
                raise LifecycleError(
                    f"acquisition recovery omitted marker identity: {path}"
                )
            marker["owned_identity"] = installed_identity
            marker["acquired"] = True
            _save_marker_state(state_path, state)
            current_observation = _marker_observation(path)
        else:
            original = marker["original_value"]
            if current_observation.get("value") != original or (
                original is not None
                and not _marker_identity_matches(
                    current_observation,
                    marker.get("original_file_identity"),
                    include_ctime=True,
                )
            ):
                raise LifecycleError(
                    f"acquisition state is ambiguous for {path}: "
                    f"observed={current_observation!r}"
                )

    if not marker["restore_target_set"]:
        marker["restore_target"] = _restore_target(marker)
        marker["restore_target_set"] = True
        _save_marker_state(state_path, state)
    target = marker["restore_target"]
    current_observation = _marker_observation(path)
    if current_observation.get("error"):
        raise LifecycleError(
            f"cannot inspect marker before restore: {current_observation!r}"
        )

    if marker["restored"]:
        if current_observation.get("value") != target or (
            target is not None
            and not _marker_identity_matches(
                current_observation,
                marker.get("restored_identity"),
                include_ctime=True,
            )
        ):
            raise LifecycleError(
                f"already-restored marker drifted for {path}: "
                f"target={target!r}, observed={current_observation!r}"
            )
        return

    if marker["acquired"]:
        expected = state["run_id"]
        expected_identity = marker.get("owned_identity")
    else:
        expected = marker["original_value"]
        expected_identity = marker.get("original_file_identity")
        if current_observation.get("value") != expected or (
            expected is not None
            and not _marker_identity_matches(
                current_observation, expected_identity, include_ctime=True
            )
        ):
            raise LifecycleError(
                f"unacquired marker drifted for {path}: "
                f"expected={expected!r}/{expected_identity!r}, "
                f"observed={current_observation!r}"
            )
        if target == expected:
            marker["restored_identity"] = expected_identity
            marker["restored"] = True
            marker.pop("restore_error", None)
            _save_marker_state(state_path, state)
            return

    restore_claim_identity = _ensure_marker_claim_locked(
        state_path,
        state,
        marker,
        phase="restore",
        value=target,
    )
    if not marker.get("restore_started", False):
        marker["restore_started"] = True
        _save_marker_state(state_path, state)
    restored_identity = _cas_marker(
        path,
        expected,
        target,
        transition_id=f"{state['run_id']}:{marker['name']}:restore",
        expected_identity=expected_identity,
        replacement_claim_identity=restore_claim_identity,
    )
    marker["restored_identity"] = restored_identity
    marker["restored"] = True
    marker.pop("restore_error", None)
    _save_marker_state(state_path, state)


def _restore_markers_locked(state_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    state["status"] = "restoring"
    _save_marker_state(state_path, state)
    errors: list[str] = []
    for marker in state["markers"]:
        try:
            _restore_one_marker_locked(state_path, state, marker)
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            marker["restore_error"] = error_text
            errors.append(f"{marker['name']}: {error_text}")
            _save_marker_state(state_path, state)

    for marker in state["markers"]:
        try:
            observed = _read_marker(Path(marker["path"]))
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            if marker.get("restore_error") != error_text:
                errors.append(f"{marker['name']} verification: {error_text}")
            continue
        if observed != marker["restore_target"]:
            error_text = (
                f"marker restoration verification failed for {marker['path']}: "
                f"expected={marker['restore_target']!r}, observed={observed!r}"
            )
            if marker.get("restore_error") != f"LifecycleError: {error_text}":
                errors.append(f"{marker['name']} verification: {error_text}")

    if errors:
        state["status"] = "restore_failed"
        state["last_error"] = "; ".join(errors)
        _save_marker_state(state_path, state)
        raise LifecycleError(state["last_error"])

    state["status"] = "restored"
    state.pop("last_error", None)
    _save_marker_state(state_path, state)
    return state


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
                claim_identity = _ensure_marker_claim_locked(
                    state_path,
                    state,
                    marker,
                    phase="acquire",
                    value=state["run_id"],
                )
                marker["acquire_started"] = True
                _save_marker_state(state_path, state)
                installed_identity = _cas_marker(
                    Path(marker["path"]),
                    marker["original_value"],
                    state["run_id"],
                    transition_id=f"{state['run_id']}:{marker['name']}:acquire",
                    expected_identity=marker.get("original_file_identity"),
                    replacement_claim_identity=claim_identity,
                )
                observation = _marker_observation(Path(marker["path"]))
                if observation.get("value") != state["run_id"] or not (
                    installed_identity
                    and _marker_identity_matches(
                        observation, installed_identity, include_ctime=True
                    )
                ):
                    raise LifecycleError(
                        f"owned marker verification failed after acquisition: "
                        f"{marker['path']}"
                    )
                marker["owned_identity"] = installed_identity
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
    with _shared_lock(lock_path):
        return str(_load_marker_state(state_path)["status"])


def _owned_marker_drifts(state: dict[str, Any]) -> list[dict[str, Any]]:
    if state["status"] != "acquired":
        return []
    drifts: list[dict[str, Any]] = []
    for marker in state["markers"]:
        if not marker.get("acquired", False) or marker.get("restored", False):
            continue
        observation = _marker_observation(Path(marker["path"]))
        expected_identity = marker.get("owned_identity") or {}
        identity_mismatch = any(
            observation.get(key) != value for key, value in expected_identity.items()
        )
        value_mismatch = observation.get("value") != state["run_id"]
        if value_mismatch or identity_mismatch or observation.get("error"):
            drifts.append(
                {
                    "name": marker["name"],
                    "path": marker["path"],
                    "expected_value": state["run_id"],
                    "expected_identity": expected_identity,
                    "value_mismatch": value_mismatch,
                    "identity_mismatch": identity_mismatch,
                    "observation": observation,
                }
            )
    return drifts


def _record_marker_ownership_loss(
    state_path: Path,
    lock_path: Path,
) -> dict[str, Any] | None:
    with _exclusive_lock(lock_path):
        state = _load_marker_state(state_path)
        if state["status"] == "ownership_lost":
            loss = state.get("ownership_loss")
            return loss if isinstance(loss, dict) else None
        drifts = _owned_marker_drifts(state)
        if not drifts:
            return None
        loss = {
            "schema": "amg_marker_ownership_loss_v1",
            "status": "fail",
            "run_id": state["run_id"],
            "detected_unix": time.time(),
            "parent": state["parent"],
            "markers": drifts,
        }
        state["status"] = "ownership_lost"
        state["ownership_loss"] = loss
        state["last_error"] = "marker ownership changed while launcher was active"
        _save_marker_state(state_path, state)
        return loss


def _signal_process_identity(
    pid: int,
    start_ticks: str,
    signum: int,
) -> bool:
    """Signal only the Linux process instance named by PID plus start ticks."""

    if not sys.platform.startswith("linux"):
        raise LifecycleError("exact process signalling requires Linux pidfds")
    descriptor = -1
    try:
        pidfd_open = getattr(os, "pidfd_open", None)
        if callable(pidfd_open):
            descriptor = pidfd_open(pid, 0)
        else:
            libc = ctypes.CDLL(None, use_errno=True)
            syscall = libc.syscall
            syscall.restype = ctypes.c_long
            descriptor = int(syscall(434, pid, 0))
            if descriptor < 0:
                error_number = ctypes.get_errno()
                if error_number == errno.ESRCH:
                    return False
                raise LifecycleError(
                    "pidfd_open syscall failed: "
                    f"pid={pid} errno={error_number} {os.strerror(error_number)}"
                )
        if not process_identity_alive(pid, start_ticks):
            return False
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if callable(pidfd_send_signal):
            pidfd_send_signal(descriptor, signum, None, 0)
        else:
            libc = ctypes.CDLL(None, use_errno=True)
            syscall = libc.syscall
            syscall.restype = ctypes.c_long
            result = int(
                syscall(424, descriptor, signum, ctypes.c_void_p(0), 0)
            )
            if result != 0:
                error_number = ctypes.get_errno()
                if error_number == errno.ESRCH:
                    return False
                raise LifecycleError(
                    "pidfd_send_signal syscall failed: "
                    f"pid={pid} errno={error_number} {os.strerror(error_number)}"
                )
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise LifecycleError(f"pidfd signalling permission denied for pid {pid}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return True


def _write_marker_watcher_receipt(
    receipt_path: Path,
    *,
    status: str,
    mode: str,
    error: str | None,
    state_status: str,
    ownership_loss: dict[str, Any] | None = None,
    termination_signal_sent: bool | None = None,
    phase: str = "final",
) -> None:
    payload: dict[str, Any] = {
        "schema": "amg_marker_watcher_exit_v1",
        "status": status,
        "mode": mode,
        "phase": phase,
        "error": error,
        "pid": os.getpid(),
        "state_status": state_status,
    }
    if ownership_loss is not None:
        payload["ownership_loss"] = ownership_loss
    if termination_signal_sent is not None:
        payload["termination_signal_sent"] = termination_signal_sent
    _atomic_write_json(receipt_path, payload)


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
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    _atomic_write_json(
        ready_path,
        {
            "schema": "amg_marker_watcher_start_v1",
            "status": "ready",
            "pid": os.getpid(),
            "start_ticks": watcher_ticks,
            "signal_handlers_installed": True,
            "parent_pid": parent_pid,
            "parent_start_ticks": str(parent_start_ticks),
            "state_path": str(state_path),
        },
    )
    mode = ""
    error_text: str | None = None
    deadline: float | None = None
    ownership_loss: dict[str, Any] | None = None
    termination_signal_sent: bool | None = None
    while True:
        if ownership_loss is None:
            ownership_loss = _record_marker_ownership_loss(state_path, lock_path)
            if ownership_loss is not None:
                mode = "marker_ownership_lost"
                termination_signal_sent = _signal_process_identity(
                    parent_pid, str(parent_start_ticks), signal.SIGTERM
                )
                deadline = time.monotonic() + restore_timeout_seconds
                _write_marker_watcher_receipt(
                    receipt_path,
                    status="fail",
                    mode=mode,
                    phase="detected",
                    error="marker ownership changed while launcher was active",
                    state_status=_marker_state_status(state_path, lock_path),
                    ownership_loss=ownership_loss,
                    termination_signal_sent=termination_signal_sent,
                )

        status = _marker_state_status(state_path, lock_path)
        parent_alive = process_identity_alive(parent_pid, parent_start_ticks)
        if ownership_loss is not None:
            if parent_alive:
                if deadline is not None and time.monotonic() >= deadline:
                    error_text = "launcher did not exit after marker ownership loss"
                    _write_marker_watcher_receipt(
                        receipt_path,
                        status="fail",
                        mode=mode,
                        error=error_text,
                        state_status=status,
                        ownership_loss=ownership_loss,
                        termination_signal_sent=termination_signal_sent,
                    )
                    return 1
                time.sleep(poll_seconds)
                continue
            try:
                restore_marker_transaction(state_path, lock_path)
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
            status = _marker_state_status(state_path, lock_path)
            _write_marker_watcher_receipt(
                receipt_path,
                status="fail",
                mode=mode,
                error=error_text or "marker ownership was lost during the run",
                state_status=status,
                ownership_loss=ownership_loss,
                termination_signal_sent=termination_signal_sent,
            )
            return 1

        if status in {"restored", "acquisition_rolled_back"}:
            mode = "explicit_restore"
            break
        if not parent_alive:
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
                    _write_marker_watcher_receipt(
                        receipt_path,
                        status="fail",
                        mode=mode,
                        error=error_text,
                        state_status=_marker_state_status(state_path, lock_path),
                    )
                    return 1
        elif stop_requested:
            mode = "signal_before_restore"
            error_text = "watcher was signalled while launcher still owned markers"
            _write_marker_watcher_receipt(
                receipt_path,
                status="fail",
                mode=mode,
                error=error_text,
                state_status=status,
            )
            return 1
        time.sleep(poll_seconds)
    _write_marker_watcher_receipt(
        receipt_path,
        status="pass",
        mode=mode,
        error=error_text,
        state_status=_marker_state_status(state_path, lock_path),
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
                sleep_seconds = min(
                    0.1, max(0.0, deadline - time.monotonic())
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
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


def _read_cgroup_byte_counter(path: Path, *, label: str) -> int:
    if not path.is_file():
        raise LifecycleError(f"{label} must be a readable cgroup file: {path}")
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise LifecycleError(f"cannot read {label} from {path}: {exc}") from exc
    if value < 0:
        raise LifecycleError(f"{label} must be non-negative, got {value}")
    return value


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
    memory_cgroup_usage_path: Path | None = None,
    memory_cgroup_limit_path: Path | None = None,
    memory_cgroup_checkpoint_copies: int = 0,
    memory_cgroup_margin_bytes: int = 0,
    require_distinct_filesystems: bool = False,
    expected_persistent_filesystem_types: Sequence[str] = (),
) -> dict[str, Any]:
    for number, label in (
        (checkpoint_bytes, "checkpoint_bytes"),
        (volatile_checkpoint_copies, "volatile_checkpoint_copies"),
        (persistent_checkpoint_copies, "persistent_checkpoint_copies"),
        (volatile_margin_bytes, "volatile_margin_bytes"),
        (persistent_margin_bytes, "persistent_margin_bytes"),
        (memory_cgroup_checkpoint_copies, "memory_cgroup_checkpoint_copies"),
        (memory_cgroup_margin_bytes, "memory_cgroup_margin_bytes"),
    ):
        if number < 0:
            raise LifecycleError(f"{label} must be non-negative")
    if (memory_cgroup_usage_path is None) != (memory_cgroup_limit_path is None):
        raise LifecycleError(
            "memory cgroup usage and limit paths must be provided together"
        )
    if memory_cgroup_usage_path is None and (
        memory_cgroup_checkpoint_copies or memory_cgroup_margin_bytes
    ):
        raise LifecycleError(
            "memory cgroup capacity was requested without usage and limit paths"
        )
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
        "schema": "amg_persistence_capacity_admission_v2",
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
    if memory_cgroup_usage_path is not None:
        memory_usage = _read_cgroup_byte_counter(
            memory_cgroup_usage_path, label="memory cgroup usage"
        )
        memory_limit = _read_cgroup_byte_counter(
            memory_cgroup_limit_path, label="memory cgroup limit"
        )
        if memory_limit <= 0:
            raise LifecycleError(
                f"memory cgroup limit must be positive, got {memory_limit}"
            )
        memory_headroom = max(0, memory_limit - memory_usage)
        memory_required = (
            checkpoint_bytes * memory_cgroup_checkpoint_copies
            + memory_cgroup_margin_bytes
        )
        report["memory_cgroup"] = {
            "usage_path": str(memory_cgroup_usage_path),
            "limit_path": str(memory_cgroup_limit_path),
            "usage_bytes": memory_usage,
            "limit_bytes": memory_limit,
            "headroom_bytes": memory_headroom,
            "checkpoint_copies": memory_cgroup_checkpoint_copies,
            "margin_bytes": memory_cgroup_margin_bytes,
            "required_headroom_bytes": memory_required,
        }
        if memory_usage > memory_limit:
            failures.append(
                f"memory cgroup usage={memory_usage} exceeds limit={memory_limit}"
            )
        elif memory_headroom < memory_required:
            failures.append(
                f"memory cgroup headroom={memory_headroom} required={memory_required}"
            )
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


def _seal_tree_read_only(root: Path) -> None:
    _validate_source_tree(root)
    files = _regular_files(root)
    directories = [root]
    for directory, names, _files in os.walk(root, followlinks=False):
        base = Path(directory)
        directories.extend(base / name for name in names)
    for path in files:
        mode = stat.S_IMODE(os.lstat(path).st_mode)
        path.chmod((mode & ~0o222) | stat.S_IRUSR)
    for path in sorted(
        (item for item in directories if item != root),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        mode = stat.S_IMODE(os.lstat(path).st_mode)
        path.chmod((mode & ~0o222) | stat.S_IRUSR | stat.S_IXUSR)
    # Keep the stage root owner-writable because macOS rejects renaming a
    # non-writable source directory even when its parent is writable.  All
    # payload files and child directories remain sealed read-only.
    mode = stat.S_IMODE(os.lstat(root).st_mode)
    root.chmod((mode & ~0o022) | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def _tree_metadata_snapshot(root: Path) -> dict[str, tuple[int, ...]]:
    _validate_source_tree(root)
    paths = [root, *_regular_files(root)]
    for directory, names, _files in os.walk(root, followlinks=False):
        base = Path(directory)
        paths.extend(base / name for name in names)
    snapshot: dict[str, tuple[int, ...]] = {}
    for path in paths:
        metadata = os.lstat(path)
        relative = "." if path == root else str(path.relative_to(root))
        snapshot[relative] = (
            stat.S_IFMT(metadata.st_mode),
            stat.S_IMODE(metadata.st_mode),
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
    return snapshot


def _verify_staged_tree(
    root: Path,
    rows: Sequence[tuple[str, str]],
    manifest_path: Path,
    manifest_sha256: str,
) -> None:
    files = _regular_files(root)
    expected = {relative for _digest, relative in rows} | {
        str(manifest_path.relative_to(root))
    }
    actual = {str(path.relative_to(root)) for path in files}
    if actual != expected:
        raise LifecycleError(
            "staged tree manifest is not exhaustive: "
            f"missing={sorted(actual - expected)} extra={sorted(expected - actual)}"
        )
    if _sha256(manifest_path) != manifest_sha256:
        raise LifecycleError("staged tree SHA256 manifest changed")
    for digest, relative in rows:
        if relative.startswith("checkpoints/global_step_"):
            continue
        observed = _sha256(root / relative)
        if observed != digest:
            raise LifecycleError(f"staged tree hash mismatch: {relative}")


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


def _parse_launcher_exit(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        key, separator, value = raw.partition("=")
        if not separator or not key or key in values:
            raise LifecycleError(f"invalid launcher exit receipt line: {raw!r}")
        values[key] = value
    return values


def _safe_recovery_artifact(run_dir: Path, relative: str) -> Path:
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise LifecycleError(f"unsafe recovery artifact path: {relative!r}")
    path = run_dir / candidate
    if path.is_symlink() or not path.is_file():
        raise LifecycleError(
            f"recovery artifact is missing, non-regular, or symlinked: {path}"
        )
    return path


def _validate_recovery_manifest(run_dir: Path, manifest_path: Path) -> None:
    recovery_root = run_dir / "recovery"
    expected: dict[str, str] = {}
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = raw.partition("  ")
        if (
            not separator
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or relative in expected
        ):
            raise LifecycleError(f"invalid recovery manifest row: {raw!r}")
        path = _safe_recovery_artifact(recovery_root, relative)
        observed = _sha256(path)
        if observed != digest:
            raise LifecycleError(
                f"recovery manifest hash mismatch for {relative}: "
                f"expected={digest} observed={observed}"
            )
        expected[relative] = digest
    excluded = {"RECOVERY-SHA256SUMS", "RECOVERY-COMMIT.json"}
    actual = {
        str(path.relative_to(recovery_root))
        for path in _regular_files(recovery_root)
        if str(path.relative_to(recovery_root)) not in excluded
    }
    if set(expected) != actual:
        raise LifecycleError(
            "recovery manifest is not exhaustive: "
            f"missing={sorted(actual - set(expected))} "
            f"extra={sorted(set(expected) - actual)}"
        )


def _validate_recovery_publication_state(
    run_dir: Path, run_id: str, values: Mapping[str, str]
) -> None:
    mode = values.get("recovery_mode")
    commit_digest = values.get("recovery_commit_sha256")
    if mode != "post_run_evidence_preserving":
        raise LifecycleError(f"unsupported recovery publication mode: {mode!r}")
    if not commit_digest or not re.fullmatch(r"[0-9a-f]{64}", commit_digest):
        raise LifecycleError("invalid or missing recovery_commit_sha256")

    commit_path = run_dir / "recovery/RECOVERY-COMMIT.json"
    commit = _load_json(commit_path)
    if _sha256(commit_path) != commit_digest:
        raise LifecycleError("recovery commit hash does not match launcher receipt")
    required_commit = {
        "schema": "amg_recovery_publication_commit_v1",
        "status": "ready_for_atomic_publication",
        "run_id": run_id,
    }
    mismatches = {
        key: {"expected": expected, "observed": commit.get(key)}
        for key, expected in required_commit.items()
        if commit.get(key) != expected
    }
    if mismatches:
        raise LifecycleError(f"invalid recovery commit: {mismatches}")

    launcher_contract = commit.get("launcher_contract")
    if not isinstance(launcher_contract, dict) or not launcher_contract:
        raise LifecycleError("recovery commit launcher_contract is missing or invalid")
    required_contract = {
        "trainer_exit_code": "0",
        "cleanup_status": "pass",
        "publication_status": "ready_for_atomic_publication",
        "run_id": run_id,
        "recovery_mode": "post_run_evidence_preserving",
    }
    for key, expected in required_contract.items():
        if launcher_contract.get(key) != expected:
            raise LifecycleError(
                f"recovery launcher contract mismatch for {key}: "
                f"expected={expected!r} observed={launcher_contract.get(key)!r}"
            )
    for key, expected in launcher_contract.items():
        if not isinstance(key, str) or not isinstance(expected, str):
            raise LifecycleError("recovery launcher contract must contain strings")
        if values.get(key) != expected:
            raise LifecycleError(
                f"launcher is not bound to recovery commit for {key}: "
                f"expected={expected!r} observed={values.get(key)!r}"
            )

    required_artifacts = {
        "recovery_receipt": "recovery/RECOVERY-RECEIPT.json",
        "post_recovery_state": "recovery/POST-RECOVERY-STATE.json",
        "recovery_manifest": "recovery/RECOVERY-SHA256SUMS",
        "finalization": "finalization.json",
        "trainer_exit_code": "trainer-exit-code",
        "persistent_evidence_path": "persistent-evidence-path",
    }
    artifacts = commit.get("artifacts")
    if not isinstance(artifacts, dict):
        raise LifecycleError("recovery commit artifacts are missing or invalid")
    artifact_paths: dict[str, Path] = {}
    for name, expected_relative in required_artifacts.items():
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise LifecycleError(f"missing recovery artifact binding: {name}")
        relative = record.get("path")
        digest = record.get("sha256")
        if relative != expected_relative:
            raise LifecycleError(
                f"unexpected recovery artifact path for {name}: {relative!r}"
            )
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise LifecycleError(f"invalid recovery artifact hash for {name}")
        path = _safe_recovery_artifact(run_dir, relative)
        observed = _sha256(path)
        if observed != digest:
            raise LifecycleError(
                f"recovery artifact hash mismatch for {name}: "
                f"expected={digest} observed={observed}"
            )
        artifact_paths[name] = path

    receipt_digest = artifacts["recovery_receipt"]["sha256"]
    if values.get("recovery_receipt_sha256") != receipt_digest:
        raise LifecycleError("launcher recovery receipt hash is not commit-bound")
    receipt = _load_json(artifact_paths["recovery_receipt"])
    if receipt.get("status") != "pass" or receipt.get("run_id") != run_id:
        raise LifecycleError("recovery receipt is not a passing receipt for this run")
    post_state = _load_json(artifact_paths["post_recovery_state"])
    if (
        post_state.get("status") != "ready_for_atomic_publication"
        or post_state.get("run_id") != run_id
        or post_state.get("recovery_receipt_sha256") != receipt_digest
    ):
        raise LifecycleError("post-recovery state is not bound to the recovery receipt")
    _validate_recovery_manifest(run_dir, artifact_paths["recovery_manifest"])


def _validate_launcher_exit_values(
    run_dir: Path, run_id: str, values: Mapping[str, str]
) -> None:
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
    recovery_keys = {key for key in values if key.startswith("recovery_")}
    if recovery_keys:
        _validate_recovery_publication_state(run_dir, run_id, values)


def _validate_launcher_exit(run_dir: Path, run_id: str) -> dict[str, str]:
    path = run_dir / "launcher-exit.env"
    if path.is_symlink() or not path.is_file():
        raise LifecycleError(f"launcher exit receipt is missing or symlinked: {path}")
    values = _parse_launcher_exit(path.read_text(encoding="utf-8"))
    _validate_launcher_exit_values(run_dir, run_id, values)
    return values


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
        _validate_source_tree(stage_path)
        launcher_values = _validate_launcher_exit(stage_path, run_id)
        terminal_receipt = {
            "schema": "amg_terminal_publisher_v1",
            "status": "complete_when_public",
            "run_id": run_id,
            "linearization_point": "atomic_directory_rename",
            "post_rename_work": "none",
            "process_transition": "os._exit(0)_immediately_after_rename",
            "launcher_exit_sha256": _sha256(stage_path / "launcher-exit.env"),
        }
        if "recovery_commit_sha256" in launcher_values:
            terminal_receipt["recovery_commit_sha256"] = launcher_values[
                "recovery_commit_sha256"
            ]
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
        if "recovery_commit_sha256" in launcher_values:
            metadata["recovery_commit_sha256"] = launcher_values[
                "recovery_commit_sha256"
            ]
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
        manifest_sha256 = _sha256(manifest_path)
        _verify_staged_tree(
            stage_path, tree_rows, manifest_path, manifest_sha256
        )
        _seal_tree_read_only(stage_path)
        sealed_snapshot = _tree_metadata_snapshot(stage_path)
        report = {
            **metadata,
            "persistent_path": str(final_path),
            "tree_manifest_sha256": manifest_sha256,
        }
        _fsync_directory(stage_path)
        lock_path = persist_root / ".amg-atomic-publication.lock"
        with _exclusive_lock(lock_path):
            if final_path.exists() or final_path.is_symlink():
                raise LifecycleError(
                    f"persistent destination appeared during publication: {final_path}"
                )
            _validate_source_tree(stage_path)
            final_launcher_values = _validate_launcher_exit(stage_path, run_id)
            if final_launcher_values != launcher_values:
                raise LifecycleError(
                    "staged launcher exit receipt changed during publication"
                )
            if terminal_receipt["launcher_exit_sha256"] != _sha256(
                stage_path / "launcher-exit.env"
            ):
                raise LifecycleError(
                    "terminal publisher receipt is not bound to staged launcher"
                )
            _verify_staged_tree(
                stage_path, tree_rows, manifest_path, manifest_sha256
            )
            if _tree_metadata_snapshot(stage_path) != sealed_snapshot:
                raise LifecycleError(
                    "sealed staged tree changed before atomic publication"
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
        memory_cgroup_usage_path=(
            Path(args.memory_cgroup_usage_path)
            if args.memory_cgroup_usage_path
            else None
        ),
        memory_cgroup_limit_path=(
            Path(args.memory_cgroup_limit_path)
            if args.memory_cgroup_limit_path
            else None
        ),
        memory_cgroup_checkpoint_copies=args.memory_cgroup_checkpoint_copies,
        memory_cgroup_margin_bytes=args.memory_cgroup_margin_bytes,
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
    capacity.add_argument("--memory-cgroup-usage-path")
    capacity.add_argument("--memory-cgroup-limit-path")
    capacity.add_argument("--memory-cgroup-checkpoint-copies", type=int, default=0)
    capacity.add_argument("--memory-cgroup-margin-bytes", type=int, default=0)
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
