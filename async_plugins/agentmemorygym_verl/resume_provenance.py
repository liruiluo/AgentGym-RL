"""Fail-closed validation for provenance-only checkpoint resume rebinds.

The sampled task stream is part of checkpoint state.  A resume may therefore
bind a repaired runtime publication only when every schedule row is identical
after removing a deliberately tiny set of provenance fields.  This module
keeps that exception separate from the normal exact-hash resume path.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


_DECLARATION_SCHEMA = "amg_resume_provenance_rebind_v1"
_VALIDATION_SCHEMA = "amg_resume_provenance_rebind_validation_v1"
_ROUTE_IDS = ("webshop", "swesmith", "literesearcher", "openmle_fast")
_SCHEDULE_FIELDS_ALL = (
    "extra_info.manifest_digest",
    "extra_info.route_registry_sha256",
)
_SCHEDULE_FIELDS_OPENMLE = (
    "extra_info.source_manifest_digest",
    "extra_info.source_schedule_sha256",
)
_REGISTRY_FIELDS_OPENMLE = ("client.expected_manifest_sha256",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return dict(value)


def _binding(
    declaration: Mapping[str, Any],
    side: str,
    kind: str,
    actual_path: Path,
    actual_sha256: str,
) -> dict[str, str]:
    side_value = declaration.get(side)
    if not isinstance(side_value, Mapping):
        raise ValueError(f"declaration {side} binding is missing")
    value = side_value.get(kind)
    if not isinstance(value, Mapping):
        raise ValueError(f"declaration {side}.{kind} binding is missing")
    declared_path = Path(str(value.get("path", ""))).resolve()
    declared_sha256 = str(value.get("sha256", ""))
    if declared_path != actual_path or declared_sha256 != actual_sha256:
        raise ValueError(
            f"declaration {side}.{kind} does not bind the exact launch artifact"
        )
    return {"path": str(actual_path), "sha256": actual_sha256}


def _leaf_differences(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: set[str] = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(_leaf_differences(left[key], right[key], path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {prefix}
        differences: set[str] = set()
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            differences.update(
                _leaf_differences(left_value, right_value, f"{prefix}[{index}]")
            )
        return differences
    return set() if left == right else {prefix}


def _canonical_sha256(values: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _load_schedule(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"schedule contains a blank row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"schedule row {line_number} is invalid JSON") from exc
        if not isinstance(row, Mapping):
            raise TypeError(f"schedule row {line_number} is not an object")
        rows.append(dict(row))
    if not rows:
        raise ValueError("schedule is empty")
    return rows


def _strip_schedule_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(row))
    extra = value.get("extra_info")
    if not isinstance(extra, dict):
        raise ValueError("schedule row has no extra_info object")
    for field in (
        "manifest_digest",
        "route_registry_sha256",
        "source_manifest_digest",
        "source_schedule_sha256",
    ):
        extra.pop(field, None)
    return value


def _registry_by_route(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    routes = document.get("routes")
    if not isinstance(routes, list):
        raise ValueError("route registry has no routes list")
    result: dict[str, dict[str, Any]] = {}
    for raw in routes:
        if not isinstance(raw, Mapping):
            raise TypeError("route registry entry is not an object")
        route = deepcopy(dict(raw))
        route_id = str(route.get("route_id", ""))
        if route_id in result:
            raise ValueError(f"duplicate route {route_id!r}")
        result[route_id] = route
    if tuple(result) != _ROUTE_IDS:
        raise ValueError(f"route order drifted: {tuple(result)!r}")
    return result


def validate_resume_provenance_rebind(
    declaration_path: Path,
    *,
    prefix_schedule_path: Path,
    successor_schedule_path: Path,
    prefix_route_registry_path: Path,
    successor_route_registry_path: Path,
    prefix_schedule_sha256: str,
    successor_schedule_sha256: str,
    prefix_route_registry_sha256: str,
    successor_route_registry_sha256: str,
) -> dict[str, Any]:
    """Validate the only supported resume rebind: OpenMLE grader provenance."""

    declaration_path = _regular(declaration_path, label="provenance declaration")
    prefix_schedule_path = _regular(prefix_schedule_path, label="prefix schedule")
    successor_schedule_path = _regular(
        successor_schedule_path, label="successor schedule"
    )
    prefix_route_registry_path = _regular(
        prefix_route_registry_path, label="prefix route registry"
    )
    successor_route_registry_path = _regular(
        successor_route_registry_path, label="successor route registry"
    )
    declaration = _load_mapping(declaration_path, label="provenance declaration")
    if declaration.get("schema") != _DECLARATION_SCHEMA:
        raise ValueError("resume provenance declaration schema mismatch")
    if declaration.get("status") != "approved":
        raise ValueError("resume provenance declaration is not approved")
    if declaration.get("reason") != "openmle_fast_private_metric_infrastructure_fix":
        raise ValueError("resume provenance declaration reason is unsupported")
    if declaration.get("allowed_schedule_changes") != {
        "all_routes": list(_SCHEDULE_FIELDS_ALL),
        "openmle_fast": list(_SCHEDULE_FIELDS_OPENMLE),
    }:
        raise ValueError("resume provenance schedule whitelist drifted")
    if declaration.get("allowed_route_registry_changes") != {
        "openmle_fast": list(_REGISTRY_FIELDS_OPENMLE)
    }:
        raise ValueError("resume provenance registry whitelist drifted")

    actual_digests = {
        "prefix_schedule": _sha256(prefix_schedule_path),
        "successor_schedule": _sha256(successor_schedule_path),
        "prefix_registry": _sha256(prefix_route_registry_path),
        "successor_registry": _sha256(successor_route_registry_path),
    }
    expected_digests = {
        "prefix_schedule": prefix_schedule_sha256,
        "successor_schedule": successor_schedule_sha256,
        "prefix_registry": prefix_route_registry_sha256,
        "successor_registry": successor_route_registry_sha256,
    }
    if actual_digests != expected_digests:
        raise ValueError("resume provenance artifact digest does not match launch identity")
    prefix_binding = {
        "schedule": _binding(
            declaration,
            "prefix",
            "schedule",
            prefix_schedule_path,
            prefix_schedule_sha256,
        ),
        "route_registry": _binding(
            declaration,
            "prefix",
            "route_registry",
            prefix_route_registry_path,
            prefix_route_registry_sha256,
        ),
    }
    successor_binding = {
        "schedule": _binding(
            declaration,
            "successor",
            "schedule",
            successor_schedule_path,
            successor_schedule_sha256,
        ),
        "route_registry": _binding(
            declaration,
            "successor",
            "route_registry",
            successor_route_registry_path,
            successor_route_registry_sha256,
        ),
    }

    prefix_rows = _load_schedule(prefix_schedule_path)
    successor_rows = _load_schedule(successor_schedule_path)
    if len(prefix_rows) != len(successor_rows):
        raise ValueError("resume provenance schedules have different row counts")
    expected_row_count = declaration.get("row_count")
    if expected_row_count != len(prefix_rows) or isinstance(expected_row_count, bool):
        raise ValueError("resume provenance declaration row count drifted")

    field_counts: Counter[str] = Counter()
    identities: list[Mapping[str, Any]] = []
    route_counts: Counter[str] = Counter()
    for position, (prefix_row, successor_row) in enumerate(
        zip(prefix_rows, successor_rows)
    ):
        prefix_extra = prefix_row.get("extra_info")
        successor_extra = successor_row.get("extra_info")
        if not isinstance(prefix_extra, Mapping) or not isinstance(
            successor_extra, Mapping
        ):
            raise ValueError(f"schedule row {position} has no extra_info object")
        route_id = str(prefix_extra.get("route_id", prefix_row.get("data_source", "")))
        successor_route_id = str(
            successor_extra.get("route_id", successor_row.get("data_source", ""))
        )
        if route_id != successor_route_id or route_id not in _ROUTE_IDS:
            raise ValueError(f"schedule route identity drifted at row {position}")
        route_counts[route_id] += 1
        allowed = set(_SCHEDULE_FIELDS_ALL)
        if route_id == "openmle_fast":
            allowed.update(_SCHEDULE_FIELDS_OPENMLE)
        differences = _leaf_differences(prefix_row, successor_row)
        if not differences or not differences.issubset(allowed):
            raise ValueError(
                f"schedule row {position} changed outside provenance whitelist: "
                f"{sorted(differences)!r}"
            )
        for field in differences:
            field_counts[field] += 1
        prefix_identity = _strip_schedule_provenance(prefix_row)
        successor_identity = _strip_schedule_provenance(successor_row)
        if prefix_identity != successor_identity:
            raise ValueError(f"sampled task identity drifted at row {position}")
        identities.append(prefix_identity)

    required_counts = {
        "extra_info.manifest_digest": len(prefix_rows),
        "extra_info.route_registry_sha256": len(prefix_rows),
        "extra_info.source_manifest_digest": route_counts["openmle_fast"],
        "extra_info.source_schedule_sha256": route_counts["openmle_fast"],
    }
    if dict(field_counts) != required_counts:
        raise ValueError(
            "resume provenance schedule changes are not the exact required set: "
            f"{dict(field_counts)!r}"
        )
    identity_sha256 = _canonical_sha256(identities)
    if declaration.get("sample_identity_sha256") != identity_sha256:
        raise ValueError("resume provenance sample identity digest drifted")

    prefix_registry = _registry_by_route(
        _load_mapping(prefix_route_registry_path, label="prefix route registry")
    )
    successor_registry = _registry_by_route(
        _load_mapping(successor_route_registry_path, label="successor route registry")
    )
    registry_differences: dict[str, list[str]] = {}
    for route_id in _ROUTE_IDS:
        differences = _leaf_differences(
            prefix_registry[route_id], successor_registry[route_id]
        )
        expected = (
            set(_REGISTRY_FIELDS_OPENMLE) if route_id == "openmle_fast" else set()
        )
        if differences != expected:
            raise ValueError(
                f"route registry {route_id!r} changed outside the exact whitelist: "
                f"{sorted(differences)!r}"
            )
        registry_differences[route_id] = sorted(differences)

    old_manifest = prefix_registry["openmle_fast"]["client"].get(
        "expected_manifest_sha256"
    )
    new_manifest = successor_registry["openmle_fast"]["client"].get(
        "expected_manifest_sha256"
    )
    if old_manifest == new_manifest:
        raise ValueError("OpenMLE manifest binding did not change")
    for value in (old_manifest, new_manifest):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("OpenMLE manifest binding is not a SHA-256")

    return {
        "schema": _VALIDATION_SCHEMA,
        "status": "pass",
        "reason": declaration["reason"],
        "declaration": {
            "path": str(declaration_path),
            "sha256": _sha256(declaration_path),
        },
        "prefix": prefix_binding,
        "successor": successor_binding,
        "row_count": len(prefix_rows),
        "route_counts": dict(sorted(route_counts.items())),
        "schedule_changed_field_counts": dict(sorted(field_counts.items())),
        "route_registry_changed_fields": registry_differences,
        "sample_identity_sha256": identity_sha256,
        "openmle_fast_manifest": {"prefix": old_manifest, "successor": new_manifest},
    }
