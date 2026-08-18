"""Crash-safe private JSON primitives for the deployment WAL."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Iterator


class ImmutableConflictError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def ensure_private_directory(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = directory.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"private state path is not a real directory: {directory}")
    if stat.S_IMODE(info.st_mode) != 0o700:
        os.chmod(directory, 0o700, follow_symlinks=False)
    return directory.resolve(strict=True)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while persisting private state")
        view = view[written:]


def temporary_path(parent: Path, stem: str) -> Path:
    return parent / f".{stem}.tmp-{os.getpid()}-{secrets.token_hex(8)}"


def write_temp(path: Path, payload: bytes) -> Path:
    temp = temporary_path(path.parent, path.name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temp, flags, 0o600)
    try:
        write_all(descriptor, payload)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temp.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    return temp


def read_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"{label} is not a real regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path | str, payload: bytes) -> Path:
    target = Path(path)
    parent = ensure_private_directory(target.parent)
    target = parent / target.name
    try:
        existing = target.lstat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
            raise RuntimeError(
                f"state target is not a real regular file: {target}"
            )
    temp = write_temp(target, payload)
    try:
        os.replace(temp, target)
        os.chmod(target, 0o600, follow_symlinks=False)
        fsync_directory(parent)
    finally:
        temp.unlink(missing_ok=True)
    return target


def atomic_write_json(path: Path | str, value: Any) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(value))


def write_immutable_bytes(path: Path | str, payload: bytes) -> Path:
    target = Path(path)
    parent = ensure_private_directory(target.parent)
    target = parent / target.name
    temp = write_temp(target, payload)
    try:
        try:
            os.link(temp, target, follow_symlinks=False)
        except FileExistsError:
            try:
                info = target.lstat()
            except OSError as error:
                raise ImmutableConflictError(
                    f"cannot verify immutable state: {target}"
                ) from error
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ImmutableConflictError(
                    f"immutable state is not a real regular file: {target}"
                )
            try:
                existing = read_regular_bytes(target, "immutable state")
            except RuntimeError as error:
                raise ImmutableConflictError(
                    f"cannot verify immutable state: {target}"
                ) from error
            if existing != payload:
                raise ImmutableConflictError(
                    f"immutable state already exists with different bytes: {target}"
                )
        else:
            os.chmod(target, 0o600, follow_symlinks=False)
            fsync_directory(parent)
    finally:
        temp.unlink(missing_ok=True)
    return target


def write_immutable_json(path: Path | str, value: Any) -> Path:
    return write_immutable_bytes(path, canonical_json_bytes(value))


def read_json(path: Path | str) -> Any:
    target = Path(path)
    info = target.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"state JSON is not a real regular file: {target}")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"state JSON is invalid: {target}") from error


@contextmanager
def exclusive_lock(path: Path | str) -> Iterator[None]:
    target = Path(path)
    parent = ensure_private_directory(target.parent)
    target = parent / target.name
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


__all__ = [
    "ImmutableConflictError",
    "atomic_write_bytes",
    "atomic_write_json",
    "canonical_json_bytes",
    "ensure_private_directory",
    "exclusive_lock",
    "read_json",
    "write_immutable_bytes",
    "write_immutable_json",
]
