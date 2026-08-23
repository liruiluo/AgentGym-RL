from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentmemorygym_verl.config_contract import inspect_schedule
from agentmemorygym_verl.multitask_manifest import compose_multitask_manifest


ROUTES = ("webshop", "swesmith", "literesearcher", "openmle_fast")


def _write_source(
    root: Path,
    route_id: str,
    *,
    count: int,
    role: str = "train_pool",
) -> tuple[Path, str]:
    path = root / f"{route_id}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for position in range(count):
            row = {
                "item_id": f"source-{route_id}-{position}",
                "data_idx": position % 2,
                "data_source": f"legacy-{route_id}",
                "source_family": f"family-{route_id}-{position % 2}",
                "extra_info": {
                    "index": position % 2,
                    "schedule_position": position,
                    "role": role,
                    "manifest_digest": hashlib.sha256(route_id.encode()).hexdigest(),
                    "panel_id": f"source-{route_id}",
                },
            }
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _write_spec(
    root: Path,
    *,
    source_count: int,
    optimizer_updates: int,
    samples_per_update: int,
    allow_repetition: bool,
) -> tuple[Path, str]:
    routes = []
    for index, route_id in enumerate(ROUTES):
        source, source_sha256 = _write_source(root, route_id, count=source_count)
        routes.append(
            {
                "route_id": route_id,
                "schedule": source.name,
                "schedule_sha256": source_sha256,
                "route_attestation_sha256": str(index + 1) * 64,
                "role": "train_pool",
                "allow_repetition": allow_repetition,
            }
        )
    payload = {
        "schema": "amg_multitask_manifest_spec_v1",
        "agent_name": "amg_task_neutral_async",
        "panel_id": "amg-four-env-formal400",
        "role": "train_pool",
        "route_registry_sha256": "a" * 64,
        "optimizer_updates": optimizer_updates,
        "samples_per_update": samples_per_update,
        "routes": routes,
    }
    path = root / "multitask-spec.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


class TestMultitaskManifest(unittest.TestCase):
    def test_formal400_is_balanced_unique_and_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, spec_sha256 = _write_spec(
                root,
                source_count=1,
                optimizer_updates=400,
                samples_per_update=64,
                allow_repetition=True,
            )
            first = root / "formal400-a.jsonl"
            second = root / "formal400-b.jsonl"

            first_report = compose_multitask_manifest(
                spec,
                expected_spec_sha256=spec_sha256,
                output_path=first,
            )
            second_report = compose_multitask_manifest(
                spec,
                expected_spec_sha256=spec_sha256,
                output_path=second,
            )

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_report, second_report)
            self.assertEqual(first_report["row_count"], 25_600)
            self.assertEqual(first_report["route_order"], list(ROUTES))
            self.assertEqual(
                first_report["per_route_rows"], {route: 6_400 for route in ROUTES}
            )
            inspected = inspect_schedule(
                first,
                expected_count=25_600,
                expected_sha256=first_report["schedule_sha256"],
                expected_role="train_pool",
            )
            self.assertEqual(inspected["unique_global_indices"], 25_600)

            rows = [json.loads(line) for line in first.read_text().splitlines()]
            self.assertEqual(len({row["index"] for row in rows}), 25_600)
            self.assertEqual(len({row["item_id"] for row in rows}), 25_600)
            for position, row in enumerate(rows):
                expected_route = ROUTES[position % len(ROUTES)]
                self.assertEqual(row["index"], position)
                self.assertEqual(row["extra_info"]["index"], position)
                self.assertEqual(row["extra_info"]["schedule_position"], position)
                self.assertEqual(row["route_id"], expected_route)
                self.assertEqual(row["data_source"], expected_route)
                self.assertEqual(row["agent_name"], "amg_task_neutral_async")
                self.assertEqual(row["data_idx"], 0)
                self.assertEqual(row["source_family"], f"family-{expected_route}-0")
                self.assertEqual(
                    row["extra_info"]["source_schedule_sha256"],
                    first_report["sources"][expected_route]["schedule_sha256"],
                )
                self.assertEqual(
                    row["extra_info"]["route_attestation_sha256"],
                    first_report["sources"][expected_route][
                        "route_attestation_sha256"
                    ],
                )

    def test_small_fixture_interleaves_one_row_per_route_without_repetition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, digest = _write_spec(
                root,
                source_count=3,
                optimizer_updates=3,
                samples_per_update=4,
                allow_repetition=False,
            )
            output = root / "small.jsonl"

            report = compose_multitask_manifest(
                spec,
                expected_spec_sha256=digest,
                output_path=output,
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(report["row_count"], 12)
            for block in range(3):
                self.assertEqual(
                    [row["route_id"] for row in rows[block * 4 : block * 4 + 4]],
                    list(ROUTES),
                )
                self.assertEqual(
                    [row["data_idx"] for row in rows[block * 4 : block * 4 + 4]],
                    [block % 2] * 4,
                )

    def test_source_identity_allows_shuffled_and_repeated_task_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _ = _write_spec(
                root,
                source_count=3,
                optimizer_updates=3,
                samples_per_update=4,
                allow_repetition=False,
            )
            for route_id in ROUTES:
                source = root / f"{route_id}.jsonl"
                rows = [json.loads(line) for line in source.read_text().splitlines()]
                for position, row in enumerate(rows):
                    task_index = (position * 2 + 1) % 3
                    row["data_idx"] = task_index
                    row["extra_info"]["index"] = task_index
                    row["extra_info"]["schedule_repetition"] = position + 7
                source.write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8",
                )
            payload = json.loads(spec.read_text())
            for route in payload["routes"]:
                route["schedule_sha256"] = hashlib.sha256(
                    (root / f"{route['route_id']}.jsonl").read_bytes()
                ).hexdigest()
            spec.write_text(json.dumps(payload, sort_keys=True) + "\n")
            spec_sha256 = hashlib.sha256(spec.read_bytes()).hexdigest()

            output = root / "output.jsonl"
            compose_multitask_manifest(
                spec,
                expected_spec_sha256=spec_sha256,
                output_path=output,
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(rows[0]["data_idx"], 1)
            self.assertEqual(rows[0]["extra_info"]["source_index"], 1)
            self.assertEqual(
                rows[0]["extra_info"]["source_schedule_repetition_declared"], 7
            )
            self.assertEqual(rows[4]["data_idx"], 0)
            self.assertEqual(rows[4]["extra_info"]["source_index"], 0)
            self.assertEqual(
                rows[4]["extra_info"]["source_schedule_repetition_declared"], 8
            )

    def test_rejects_source_exhaustion_without_explicit_repetition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, digest = _write_spec(
                root,
                source_count=1,
                optimizer_updates=2,
                samples_per_update=4,
                allow_repetition=False,
            )
            with self.assertRaisesRegex(ValueError, "exhaust"):
                compose_multitask_manifest(
                    spec,
                    expected_spec_sha256=digest,
                    output_path=root / "output.jsonl",
                )

    def test_rejects_source_identity_role_and_global_index_drift(self):
        mutations = (
            (
                "duplicate item_id",
                lambda rows: rows[1].__setitem__("item_id", rows[0]["item_id"]),
            ),
            (
                "role",
                lambda rows: rows[1]["extra_info"].__setitem__("role", "gate_only"),
            ),
            (
                "index/data_idx drift",
                lambda rows: rows[1]["extra_info"].__setitem__("index", 99),
            ),
        )
        for expected, mutate in mutations:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec, _ = _write_spec(
                    root,
                    source_count=2,
                    optimizer_updates=2,
                    samples_per_update=4,
                    allow_repetition=False,
                )
                source = root / "webshop.jsonl"
                rows = [json.loads(line) for line in source.read_text().splitlines()]
                mutate(rows)
                source.write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8",
                )
                payload = json.loads(spec.read_text())
                payload["routes"][0]["schedule_sha256"] = hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
                spec.write_text(json.dumps(payload, sort_keys=True) + "\n")
                spec_sha256 = hashlib.sha256(spec.read_bytes()).hexdigest()

                with self.assertRaisesRegex(ValueError, expected):
                    compose_multitask_manifest(
                        spec,
                        expected_spec_sha256=spec_sha256,
                        output_path=root / "output.jsonl",
                    )

    def test_rejects_missing_attestation_and_wrong_source_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _ = _write_spec(
                root,
                source_count=1,
                optimizer_updates=1,
                samples_per_update=4,
                allow_repetition=False,
            )
            payload = json.loads(spec.read_text())
            del payload["routes"][0]["route_attestation_sha256"]
            spec.write_text(json.dumps(payload, sort_keys=True) + "\n")
            digest = hashlib.sha256(spec.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "route_attestation_sha256"):
                compose_multitask_manifest(
                    spec,
                    expected_spec_sha256=digest,
                    output_path=root / "output.jsonl",
                )

            payload["routes"][0]["route_attestation_sha256"] = "1" * 64
            payload["routes"][0]["schedule_sha256"] = "0" * 64
            spec.write_text(json.dumps(payload, sort_keys=True) + "\n")
            digest = hashlib.sha256(spec.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "schedule sha256 mismatch"):
                compose_multitask_manifest(
                    spec,
                    expected_spec_sha256=digest,
                    output_path=root / "output.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
