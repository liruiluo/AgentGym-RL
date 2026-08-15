"""Canonical JSON and digest helpers for paired evaluation evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with one stable, UTF-8 canonical representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def token_ids_sha256(token_ids: Any) -> str:
    return sha256_json([int(token_id) for token_id in token_ids])
