"""Private digest-addressed evidence storage and append-safe JSONL output."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional

from .contracts import EvidenceReference
from .serialization import canonical_json_bytes, sha256_bytes


CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise RuntimeError(f"private output directory must not be a symlink: {path}")
    os.chmod(path, 0o700)


def write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise RuntimeError("evidence write made no progress")
        remaining = remaining[written:]


class PrivateEvidenceStore:
    """Store content-addressed private blobs without exposing host paths in rows."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        private_directory(self.root)

    def put_json(self, category: str, payload: Any) -> EvidenceReference:
        return self.put_bytes(
            category,
            canonical_json_bytes(payload),
            media_type="application/json",
            suffix=".json",
        )

    def put_bytes(
        self,
        category: str,
        payload: bytes,
        *,
        media_type: str = "application/octet-stream",
        suffix: str = ".bin",
    ) -> EvidenceReference:
        if CATEGORY_PATTERN.fullmatch(category) is None:
            raise ValueError("evidence category must be a safe lowercase identifier")
        if not isinstance(payload, bytes):
            raise TypeError("evidence payload must be bytes")
        digest = sha256_bytes(payload)
        category_dir = self.root / category
        private_directory(category_dir)
        target = category_dir / f"{digest}{suffix}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(target, flags, 0o600)
        except FileExistsError:
            if target.is_symlink() or target.read_bytes() != payload:
                raise RuntimeError("content-addressed evidence collision")
        else:
            try:
                write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.chmod(target, 0o600)
        return EvidenceReference(
            protected_ref=f"evidence://{category}/{digest}",
            sha256=digest,
            byte_count=len(payload),
            media_type=media_type,
        )


class AppendSafeJsonlWriter:
    """Append exactly one fsynced canonical JSON line under an advisory lock."""

    def __init__(
        self,
        path: Path,
        *,
        validator: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self.path = Path(path)
        self.validator = validator

    def append(self, row: Mapping[str, Any]) -> None:
        self.append_many((row,))

    def append_many(self, rows, *, require_empty: bool = False) -> None:
        selected = list(rows)
        if not selected:
            raise ValueError("at least one JSONL row is required")
        if self.validator is not None:
            for row in selected:
                self.validator(row)
        private_directory(self.path.parent)
        if self.path.is_symlink():
            raise RuntimeError("JSONL output must not be a symlink")
        encoded = b"".join(
            canonical_json_bytes(row) + b"\n" for row in selected
        )
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            size = os.lseek(descriptor, 0, os.SEEK_END)
            if size and require_empty:
                raise RuntimeError("refusing to append a duplicate result batch")
            if size:
                if os.pread(descriptor, 1, size - 1) != b"\n":
                    raise RuntimeError("refusing to append after a partial JSONL line")
            write_all(descriptor, encoded)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o600)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
