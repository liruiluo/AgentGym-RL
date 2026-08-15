#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "agentmemory" / "build_openmle_fast_routing.py"
PROCEDURAL_INDEX = ROOT / "verl" / "utils" / "agent_dataset" / "procedural_index.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_openmle_fast_routing",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load OpenMLE-fast routing builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_procedural_index_module():
    spec = importlib.util.spec_from_file_location(
        "openmle_fast_procedural_index_for_test",
        PROCEDURAL_INDEX,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load procedural index helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def manifest(
    *, role: str = "gate_only", panel_id: str = "openmle-fast-test-v1"
) -> dict:
    return {
        "schema": "openmle_fast_public_manifest_v1",
        "panel_id": panel_id,
        "role": role,
        "openmle_tasks_revision": "f56e4b31252a9b81d95fea100098cd49b7290398",
        "task_count": 2,
        "task_id_list_sha256": "4" * 64,
        "compact_panel_sha256": "5" * 64,
        "max_policy_actions": 30,
        "records": [
            {
                "data_idx": 0,
                "task_id": "alpha@1",
                "source_family": "KAGGLE_DATASET:alpha",
                "role": role,
                "reward_eligible": True,
                "baseline_score": 0.5,
                "ideal_score": 1.0,
                "higher_is_better": True,
            },
            {
                "data_idx": 1,
                "task_id": "beta@1",
                "source_family": "KAGGLE_DATASET:beta",
                "role": role,
                "reward_eligible": True,
                "baseline_score": 1.0,
                "ideal_score": 0.0,
                "higher_is_better": False,
            },
        ],
    }


class OpenMLEFastRoutingTests(unittest.TestCase):
    def test_train_pool_covers_every_task_before_repetition(self) -> None:
        module = load_module()
        document = manifest(role="train_pool")
        # Full-pool manifests may contain multiple task variants from one source
        # family; the split builder, not this single-manifest router, owns the
        # cross-partition family assignment proof.
        document["records"][1]["source_family"] = "KAGGLE_DATASET:alpha"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            first_routing = root / "first.jsonl"
            first_certificate = root / "first-certificate.json"
            second_routing = root / "second.jsonl"
            second_certificate = root / "second-certificate.json"

            first = module.write_routing_artifacts(
                manifest_path,
                first_routing,
                first_certificate,
                repetitions=2,
            )
            second = module.write_routing_artifacts(
                manifest_path,
                second_routing,
                second_certificate,
                repetitions=2,
            )

            self.assertEqual(first_routing.read_bytes(), second_routing.read_bytes())
            self.assertEqual(
                first_certificate.read_bytes(),
                second_certificate.read_bytes(),
            )
            self.assertEqual(first, second)
            rows = [
                json.loads(line)
                for line in first_routing.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["data_idx"] for row in rows], [0, 1, 0, 1])
            self.assertEqual(
                [row["extra_info"]["index"] for row in rows],
                [0, 1, 0, 1],
            )
            self.assertEqual(
                [row["extra_info"]["schedule_position"] for row in rows],
                [0, 1, 2, 3],
            )
            self.assertEqual(
                {row["extra_info"]["role"] for row in rows},
                {"train_pool"},
            )
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            self.assertEqual(
                {row["extra_info"]["manifest_digest"] for row in rows},
                {manifest_digest},
            )
            self.assertEqual(len({row["item_id"] for row in rows}), 4)
            self.assertTrue(
                all(
                    task_id not in row["item_id"]
                    for task_id in ("alpha", "beta")
                    for row in rows
                )
            )
            self.assertEqual(first["routing_row_count"], 4)
            self.assertEqual(first["unique_task_id_count"], 2)
            self.assertEqual(first["unique_source_family_count"], 1)
            self.assertEqual(first["task_id_list_sha256"], "4" * 64)
            self.assertEqual(first["compact_panel_sha256"], "5" * 64)
            self.assertEqual(
                first["schedule_policy"],
                "full_pool_pass_before_repetition",
            )
            self.assertEqual(
                first["routing_sha256"],
                hashlib.sha256(first_routing.read_bytes()).hexdigest(),
            )

            procedural_index = load_procedural_index_module()
            batch = {"data_idx": rows[3]["data_idx"]}
            self.assertTrue(procedural_index.promote_data_idx_for_rollout(batch))
            handler = SimpleNamespace(data_idx=batch["rollout_data_indices"])
            self.assertEqual(
                procedural_index.resolve_rollout_reset_index(handler),
                1,
            )

    def test_gate_only_and_heldout_are_single_pass(self) -> None:
        module = load_module()
        for role in ("gate_only", "heldout"):
            document = manifest(role=role)
            rows = module.build_routing_rows(document, "a" * 64, repetitions=1)
            self.assertEqual([row["data_idx"] for row in rows], [0, 1])
            with self.assertRaisesRegex(ValueError, "single pass"):
                module.build_routing_rows(document, "a" * 64, repetitions=2)

    def test_frozen_g64_can_only_be_gate_only(self) -> None:
        module = load_module()
        document = manifest(
            role="train_pool",
            panel_id="openmle-fast-g64-v1",
        )
        with self.assertRaisesRegex(ValueError, "G64 panel must remain gate_only"):
            module.validate_manifest(document)

    def test_rejects_legacy_or_unknown_roles_and_record_role_drift(self) -> None:
        module = load_module()
        for role in ("mechanism_gate", "formal", ""):
            document = manifest(role=role)
            with self.subTest(role=role), self.assertRaises(ValueError):
                module.validate_manifest(document)

        document = manifest(role="train_pool")
        document["records"][1]["role"] = "heldout"
        with self.assertRaisesRegex(ValueError, "must match manifest role"):
            module.validate_manifest(document)

    def test_rejects_non_reward_bearing_routing_rows(self) -> None:
        module = load_module()
        for mutation, message in (
            ({"reward_eligible": False}, "not eligible"),
            ({"reward_eligible": None}, "not eligible"),
            ({"baseline_score": 1.0, "ideal_score": 1.0}, "normalization gap"),
            ({"higher_is_better": "yes"}, "normalization fields"),
        ):
            document = manifest(role="train_pool")
            document["records"][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ValueError, message
            ):
                module.validate_manifest(document)

    def test_gate_only_rejects_duplicate_source_family(self) -> None:
        module = load_module()
        document = manifest()
        document["records"][1]["source_family"] = document["records"][0][
            "source_family"
        ]
        with self.assertRaisesRegex(ValueError, "duplicate gate-only source_family"):
            module.validate_manifest(document)

    def test_rejects_reordered_or_non_integer_manifest_indices(self) -> None:
        module = load_module()
        bad_documents = []

        reordered = manifest()
        reordered["records"] = list(reversed(reordered["records"]))
        bad_documents.append(reordered)

        boolean_index = manifest()
        boolean_index["records"][0]["data_idx"] = False
        bad_documents.append(boolean_index)

        duplicate_task = manifest()
        duplicate_task["records"][1]["task_id"] = "alpha@1"
        bad_documents.append(duplicate_task)

        for document in bad_documents:
            with self.subTest(document=document), self.assertRaises(ValueError):
                module.validate_manifest(document)

    def test_rejects_invalid_repetition_count(self) -> None:
        module = load_module()
        document = manifest(role="train_pool")
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.build_routing_rows(
                    document,
                    "a" * 64,
                    repetitions=value,
                )


if __name__ == "__main__":
    unittest.main()
