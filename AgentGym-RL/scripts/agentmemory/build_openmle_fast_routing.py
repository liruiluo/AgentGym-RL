#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


MANIFEST_SCHEMA = "openmle_fast_public_manifest_v1"
CERTIFICATE_SCHEMA = "openmle_fast_routing_certificate_v1"
ITEM_ID_SCHEME = "openmle_fast_opaque_schedule_v1"
EXPECTED_OPENMLE_TASKS_REVISION = "f56e4b31252a9b81d95fea100098cd49b7290398"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def require_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def validate_manifest(document: Any) -> list[dict[str, Any]]:
    if not isinstance(document, dict):
        raise ValueError("manifest must be a JSON object")
    if document.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    require_nonempty_string(document.get("panel_id"), "panel_id")
    require_nonempty_string(document.get("role"), "role")
    revision = require_nonempty_string(
        document.get("openmle_tasks_revision"),
        "openmle_tasks_revision",
    )
    if REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("openmle_tasks_revision must be a lowercase Git SHA")
    if revision != EXPECTED_OPENMLE_TASKS_REVISION:
        raise ValueError("openmle_tasks_revision does not match the approved release")
    task_count = require_integer(document.get("task_count"), "task_count")
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    max_policy_actions = require_integer(
        document.get("max_policy_actions"),
        "max_policy_actions",
    )
    if max_policy_actions != 30:
        raise ValueError("max_policy_actions must be 30")
    for field in ("task_id_list_sha256", "compact_panel_sha256"):
        validate_sha256(document.get(field), field)
    records = document.get("records")
    if not isinstance(records, list) or len(records) != task_count:
        raise ValueError("records must be a list matching task_count")

    task_ids: set[str] = set()
    source_families: set[str] = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{position}] must be an object")
        data_idx = require_integer(
            record.get("data_idx"),
            f"records[{position}].data_idx",
        )
        if data_idx != position:
            raise ValueError(
                "manifest data_idx values must be contiguous and match record order"
            )
        task_id = require_nonempty_string(
            record.get("task_id"),
            f"records[{position}].task_id",
        )
        source_family = require_nonempty_string(
            record.get("source_family"),
            f"records[{position}].source_family",
        )
        if task_id in task_ids:
            raise ValueError(f"duplicate task_id: {task_id!r}")
        if source_family in source_families:
            raise ValueError(f"duplicate source_family: {source_family!r}")
        task_ids.add(task_id)
        source_families.add(source_family)
    return records


def validate_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def build_routing_rows(
    document: dict[str, Any],
    manifest_sha256: str,
    *,
    repetitions: int = 1,
) -> list[dict[str, Any]]:
    records = validate_manifest(document)
    manifest_sha256 = validate_sha256(manifest_sha256, "manifest_sha256")
    repetitions = require_integer(repetitions, "repetitions")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    rows = []
    for repetition in range(repetitions):
        for record in records:
            data_idx = record["data_idx"]
            schedule_position = len(rows)
            opaque_digest = sha256_bytes(
                (
                    f"{ITEM_ID_SCHEME}\0{manifest_sha256}\0"
                    f"{schedule_position}\0{data_idx}"
                ).encode("ascii")
            )
            rows.append(
                {
                    "item_id": (
                        f"openmlefast_{schedule_position:06d}_{opaque_digest[:20]}"
                    ),
                    "data_idx": data_idx,
                    "extra_info": {
                        "index": data_idx,
                        "manifest_digest": manifest_sha256,
                        "panel_id": document["panel_id"],
                        "schedule_position": schedule_position,
                        "schedule_repetition": repetition,
                    },
                }
            )
    return rows


def routing_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_routing_artifacts(
    manifest_path: Path,
    routing_path: Path,
    certificate_path: Path,
    *,
    repetitions: int = 1,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    routing_path = Path(routing_path)
    certificate_path = Path(certificate_path)
    resolved_paths = {
        manifest_path.resolve(),
        routing_path.resolve(),
        certificate_path.resolve(),
    }
    if len(resolved_paths) != 3:
        raise ValueError("manifest, routing, and certificate paths must be distinct")

    manifest_raw = manifest_path.read_bytes()
    document = json.loads(manifest_raw.decode("utf-8"))
    manifest_sha256 = sha256_bytes(manifest_raw)
    rows = build_routing_rows(
        document,
        manifest_sha256,
        repetitions=repetitions,
    )
    encoded_routing = routing_bytes(rows)
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "item_id_scheme": ITEM_ID_SCHEME,
        "compact_panel_sha256": document["compact_panel_sha256"],
        "manifest_schema": document["schema"],
        "manifest_sha256": manifest_sha256,
        "max_policy_actions": document["max_policy_actions"],
        "openmle_tasks_revision": document["openmle_tasks_revision"],
        "panel_id": document["panel_id"],
        "repetitions": repetitions,
        "role": document["role"],
        "routing_row_count": len(rows),
        "routing_sha256": sha256_bytes(encoded_routing),
        "schedule_policy": "manifest_order_repeated",
        "task_count": document["task_count"],
        "task_id_list_sha256": document["task_id_list_sha256"],
        "unique_data_idx_count": len({row["data_idx"] for row in rows}),
        "unique_task_id_count": len(
            {record["task_id"] for record in document["records"]}
        ),
    }
    atomic_write(routing_path, encoded_routing)
    atomic_write(certificate_path, canonical_json_bytes(certificate))
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic OpenMLE-fast PPO routing artifacts."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    certificate = write_routing_artifacts(
        args.manifest,
        args.routing,
        args.certificate,
        repetitions=args.repetitions,
    )
    print(json.dumps(certificate, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
