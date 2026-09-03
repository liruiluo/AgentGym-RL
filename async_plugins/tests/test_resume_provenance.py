from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from agentmemorygym_verl.resume_provenance import (
    validate_resume_provenance_rebind,
)


ROUTES = ("webshop", "swesmith", "literesearcher", "openmle_fast")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def _write_jsonl(path: Path, values: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )
    return _sha256(path)


def _sample_identity_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        value = deepcopy(row)
        extra = value["extra_info"]
        for field in (
            "manifest_digest",
            "route_registry_sha256",
            "source_manifest_digest",
            "source_schedule_sha256",
        ):
            extra.pop(field, None)
        digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _fixture(root: Path) -> dict:
    old_manifest = "1" * 64
    new_manifest = "2" * 64
    prefix_registry = {
        "routes": [
            {
                "route_id": route_id,
                "client": {
                    "expected_manifest_sha256": (
                        old_manifest if route_id == "openmle_fast" else "3" * 64
                    )
                },
            }
            for route_id in ROUTES
        ]
    }
    successor_registry = deepcopy(prefix_registry)
    successor_registry["routes"][-1]["client"][
        "expected_manifest_sha256"
    ] = new_manifest
    prefix_registry_path = root / "prefix-registry.json"
    successor_registry_path = root / "successor-registry.json"
    prefix_registry_sha256 = _write_json(prefix_registry_path, prefix_registry)
    successor_registry_sha256 = _write_json(
        successor_registry_path, successor_registry
    )

    prefix_rows: list[dict] = []
    successor_rows: list[dict] = []
    for position, route_id in enumerate(ROUTES):
        common = {
            "data_source": route_id,
            "item_id": f"item-{position}",
            "data_idx": position,
            "extra_info": {
                "route_id": route_id,
                "manifest_digest": old_manifest,
                "route_registry_sha256": prefix_registry_sha256,
            },
        }
        prefix = deepcopy(common)
        successor = deepcopy(common)
        successor["extra_info"].update(
            manifest_digest=new_manifest,
            route_registry_sha256=successor_registry_sha256,
        )
        if route_id == "openmle_fast":
            prefix["extra_info"].update(
                source_manifest_digest=old_manifest,
                source_schedule_sha256="4" * 64,
            )
            successor["extra_info"].update(
                source_manifest_digest=new_manifest,
                source_schedule_sha256="5" * 64,
            )
        prefix_rows.append(prefix)
        successor_rows.append(successor)
    prefix_schedule_path = root / "prefix-schedule.jsonl"
    successor_schedule_path = root / "successor-schedule.jsonl"
    prefix_schedule_sha256 = _write_jsonl(prefix_schedule_path, prefix_rows)
    successor_schedule_sha256 = _write_jsonl(
        successor_schedule_path, successor_rows
    )
    declaration_path = root / "resume-provenance.json"
    declaration = {
        "schema": "amg_resume_provenance_rebind_v1",
        "status": "approved",
        "reason": "openmle_fast_private_metric_infrastructure_fix",
        "allowed_schedule_changes": {
            "all_routes": [
                "extra_info.manifest_digest",
                "extra_info.route_registry_sha256",
            ],
            "openmle_fast": [
                "extra_info.source_manifest_digest",
                "extra_info.source_schedule_sha256",
            ],
        },
        "allowed_route_registry_changes": {
            "openmle_fast": ["client.expected_manifest_sha256"]
        },
        "row_count": len(prefix_rows),
        "sample_identity_sha256": _sample_identity_sha256(prefix_rows),
        "prefix": {
            "schedule": {
                "path": str(prefix_schedule_path),
                "sha256": prefix_schedule_sha256,
            },
            "route_registry": {
                "path": str(prefix_registry_path),
                "sha256": prefix_registry_sha256,
            },
        },
        "successor": {
            "schedule": {
                "path": str(successor_schedule_path),
                "sha256": successor_schedule_sha256,
            },
            "route_registry": {
                "path": str(successor_registry_path),
                "sha256": successor_registry_sha256,
            },
        },
    }
    _write_json(declaration_path, declaration)
    return {
        "declaration_path": declaration_path,
        "declaration": declaration,
        "prefix_rows": prefix_rows,
        "successor_rows": successor_rows,
        "prefix_schedule_path": prefix_schedule_path,
        "successor_schedule_path": successor_schedule_path,
        "prefix_registry_path": prefix_registry_path,
        "successor_registry_path": successor_registry_path,
        "prefix_schedule_sha256": prefix_schedule_sha256,
        "successor_schedule_sha256": successor_schedule_sha256,
        "prefix_registry_sha256": prefix_registry_sha256,
        "successor_registry_sha256": successor_registry_sha256,
    }


def _validate(fixture: dict) -> dict:
    return validate_resume_provenance_rebind(
        fixture["declaration_path"],
        prefix_schedule_path=fixture["prefix_schedule_path"],
        successor_schedule_path=fixture["successor_schedule_path"],
        prefix_route_registry_path=fixture["prefix_registry_path"],
        successor_route_registry_path=fixture["successor_registry_path"],
        prefix_schedule_sha256=fixture["prefix_schedule_sha256"],
        successor_schedule_sha256=fixture["successor_schedule_sha256"],
        prefix_route_registry_sha256=fixture["prefix_registry_sha256"],
        successor_route_registry_sha256=fixture["successor_registry_sha256"],
    )


class TestResumeProvenanceRebind(unittest.TestCase):
    def test_exact_provenance_only_rebind_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            receipt = _validate(fixture)

            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(receipt["row_count"], 4)
            self.assertEqual(receipt["route_counts"], {route: 1 for route in ROUTES})
            self.assertEqual(
                receipt["sample_identity_sha256"],
                fixture["declaration"]["sample_identity_sha256"],
            )

    def test_task_identity_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            rows = deepcopy(fixture["successor_rows"])
            rows[0]["item_id"] = "different-item"
            digest = _write_jsonl(fixture["successor_schedule_path"], rows)
            fixture["successor_schedule_sha256"] = digest
            declaration = deepcopy(fixture["declaration"])
            declaration["successor"]["schedule"]["sha256"] = digest
            _write_json(fixture["declaration_path"], declaration)

            with self.assertRaisesRegex(ValueError, "outside provenance whitelist"):
                _validate(fixture)

    def test_non_openmle_registry_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            registry = json.loads(
                fixture["successor_registry_path"].read_text(encoding="utf-8")
            )
            registry["routes"][0]["client"]["unexpected"] = True
            digest = _write_json(fixture["successor_registry_path"], registry)
            fixture["successor_registry_sha256"] = digest
            declaration = deepcopy(fixture["declaration"])
            declaration["successor"]["route_registry"]["sha256"] = digest
            _write_json(fixture["declaration_path"], declaration)

            with self.assertRaisesRegex(ValueError, "outside the exact whitelist"):
                _validate(fixture)

    def test_declaration_cannot_expand_the_whitelist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            declaration = deepcopy(fixture["declaration"])
            declaration["allowed_schedule_changes"]["all_routes"].append(
                "item_id"
            )
            _write_json(fixture["declaration_path"], declaration)

            with self.assertRaisesRegex(ValueError, "whitelist drifted"):
                _validate(fixture)


if __name__ == "__main__":
    unittest.main()
