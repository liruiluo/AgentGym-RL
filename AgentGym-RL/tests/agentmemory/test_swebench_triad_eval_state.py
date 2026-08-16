from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from swebench_triad_eval.atomic import (
    ImmutableConflictError,
    atomic_write_json,
    canonical_json_bytes,
    ensure_private_directory,
    write_immutable_json,
)
from swebench_triad_eval.state import (
    AlreadyAcceptedError,
    CellKey,
    CellStateStore,
    ClaimBusyError,
    FenceViolationError,
    ManifestCell,
    OwnerIdentity,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def endpoint_row(instance_id: str, arm: str) -> dict[str, object]:
    return {
        "marker": "valid-endpoint",
        "instance_id": instance_id,
        "arm": arm,
        "comparable": True,
        "failure": {"class": None},
        "termination": {"reason": "horizon"},
        "final_artifact": {"sha256": SHA_A},
        "scorer": {"public_metrics": {"official_resolved": None}},
        "lifecycle": {"close_receipt_ref": "evidence://sha256/" + SHA_B},
    }


def prediction(instance_id: str) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "model_name_or_path": "Qwen3.5-4B",
        "model_patch": "",
    }


def endpoint_validator(row: object) -> None:
    if not isinstance(row, dict) or row.get("marker") != "valid-endpoint":
        raise ValueError("invalid endpoint row")


class AtomicStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name) / "state"
        self.owner_a = OwnerIdentity("host", "boot", 101, 1001)
        self.owner_b = OwnerIdentity("host", "boot", 202, 2002)
        self.live = {self.owner_a, self.owner_b}
        self.cells = (
            ManifestCell(CellKey(0, "native"), "task-0", SHA_A),
            ManifestCell(CellKey(0, "amg_compaction_only"), "task-0", SHA_B),
            ManifestCell(CellKey(0, "amg_memory"), "task-0", SHA_C),
        )

    def make_store(self, owner: OwnerIdentity) -> CellStateStore:
        return CellStateStore(
            self.root,
            manifest=self.cells,
            owner=owner,
            owner_is_alive=lambda candidate: candidate in self.live,
            endpoint_validator=endpoint_validator,
        )

    def test_atomic_json_is_private_and_immutable(self) -> None:
        directory = ensure_private_directory(self.root)
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        path = directory / "receipt.json"
        atomic_write_json(path, {"value": 1})
        self.assertEqual(json.loads(path.read_text()), {"value": 1})
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        write_immutable_json(path, {"value": 1})
        with self.assertRaises(ImmutableConflictError):
            write_immutable_json(path, {"value": 2})
        self.assertEqual(path.read_bytes(), canonical_json_bytes({"value": 1}))

    def test_live_claim_is_busy_dead_claim_is_fenced_and_reclaimed(self) -> None:
        store_a = self.make_store(self.owner_a)
        token_a = store_a.acquire(self.cells[0].key)
        self.assertEqual(token_a.generation, 1)

        store_b = self.make_store(self.owner_b)
        with self.assertRaises(ClaimBusyError):
            store_b.acquire(self.cells[0].key)

        self.live.remove(self.owner_a)
        token_b = store_b.acquire(self.cells[0].key)
        self.assertEqual(token_b.generation, 2)
        with self.assertRaises(FenceViolationError):
            store_a.record_prediction(token_a, prediction("task-0"))
        store_b.record_prediction(token_b, prediction("task-0"))

    def test_partial_attempt_retries_but_complete_attempt_is_reconciled(self) -> None:
        store_a = self.make_store(self.owner_a)
        token_a = store_a.acquire(self.cells[0].key)
        store_a.record_prediction(token_a, prediction("task-0"))
        self.live.remove(self.owner_a)

        store_b = self.make_store(self.owner_b)
        token_b = store_b.acquire(self.cells[0].key)
        self.assertIsNone(store_b.reconcile_complete_attempt(token_b))
        store_b.record_endpoint(token_b, endpoint_row("task-0", "native"))
        store_b.record_prediction(token_b, prediction("task-0"))
        store_b.record_handoff(
            token_b,
            {
                "prediction_sha256": store_b.prediction_sha256(token_b),
                "official_resolved": None,
                "grader_revision": "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
            },
        )

        self.live.remove(self.owner_b)
        owner_c = OwnerIdentity("host", "boot", 303, 3003)
        self.live.add(owner_c)
        store_c = self.make_store(owner_c)
        token_c = store_c.acquire(self.cells[0].key)
        accepted = store_c.reconcile_complete_attempt(token_c)
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted["attempt_generation"], 2)
        with self.assertRaises(AlreadyAcceptedError):
            store_c.acquire(self.cells[0].key)

    def accept(self, store: CellStateStore, cell: ManifestCell) -> dict[str, object]:
        token = store.acquire(cell.key)
        store.record_endpoint(
            token,
            endpoint_row(cell.instance_id, cell.key.arm),
        )
        store.record_prediction(token, prediction(cell.instance_id))
        store.record_handoff(
            token,
            {
                "prediction_sha256": store.prediction_sha256(token),
                "official_resolved": None,
                "grader_revision": "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
            },
        )
        return store.accept_current_attempt(token)

    def test_acceptance_requires_all_durable_boundaries_and_is_idempotent(self) -> None:
        store = self.make_store(self.owner_a)
        token = store.acquire(self.cells[0].key)
        store.record_endpoint(token, endpoint_row("task-0", "native"))
        with self.assertRaises(ValueError):
            store.accept_current_attempt(token)
        store.record_prediction(token, prediction("task-0"))
        with self.assertRaises(ValueError):
            store.accept_current_attempt(token)
        store.record_handoff(
            token,
            {
                "prediction_sha256": store.prediction_sha256(token),
                "official_resolved": None,
                "grader_revision": "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
            },
        )
        accepted = store.accept_current_attempt(token)
        self.assertEqual(
            store.accept_current_attempt(token),
            accepted,
        )

        changed = endpoint_row("task-0", "native")
        changed["final_artifact"] = {"sha256": SHA_D}
        with self.assertRaises(ImmutableConflictError):
            store.record_endpoint(token, changed)

    def test_results_and_outcomes_require_exact_manifest_join(self) -> None:
        store = self.make_store(self.owner_a)
        for cell in self.cells:
            self.accept(store, cell)
        rows = store.assemble_results()
        self.assertEqual([row["arm"] for row in rows], [cell.key.arm for cell in self.cells])

        for index, cell in enumerate(self.cells):
            store.record_official_outcome(
                cell.key,
                {
                    "instance_id": cell.instance_id,
                    "arm": cell.key.arm,
                    "resolved": index == 2,
                    "failure_class": None,
                    "report_sha256": SHA_D,
                },
            )
        summary = store.official_summary()
        self.assertEqual(summary["denominator_per_arm"], 1)
        self.assertEqual(summary["scores"]["native"], 0.0)
        self.assertEqual(summary["scores"]["amg_memory"], 1.0)
        self.assertEqual(summary["contrasts"]["amg_memory-native"], 1.0)

        with self.assertRaises(ImmutableConflictError):
            store.record_official_outcome(
                self.cells[0].key,
                {
                    "instance_id": "task-0",
                    "arm": "native",
                    "resolved": True,
                    "failure_class": None,
                    "report_sha256": SHA_D,
                },
            )

    def test_invalid_manifest_and_non_boolean_outcomes_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            CellStateStore(
                self.root,
                manifest=(self.cells[0], self.cells[0]),
                owner=self.owner_a,
                owner_is_alive=lambda candidate: False,
                endpoint_validator=endpoint_validator,
            )
        store = self.make_store(self.owner_a)
        self.accept(store, self.cells[0])
        with self.assertRaises(ValueError):
            store.record_official_outcome(
                self.cells[0].key,
                {
                    "instance_id": "task-0",
                    "arm": "native",
                    "resolved": 1,
                    "failure_class": None,
                    "report_sha256": SHA_D,
                },
            )


if __name__ == "__main__":
    unittest.main()
