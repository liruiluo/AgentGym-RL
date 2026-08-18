from __future__ import annotations

import json
from pathlib import Path
import stat
import tempfile
import threading
import unittest

from paired_eval.serialization import sha256_json as paired_sha256_json
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
    DriverLeaseRegistry,
    FenceViolationError,
    ManifestCell,
    OwnerIdentity,
    sha256_json,
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

    def test_logical_json_digest_matches_paired_evidence_encoding(self) -> None:
        value = {"task_id": "task-0", "model_patch": ""}
        self.assertEqual(sha256_json(value), paired_sha256_json(value))

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

    def test_accepted_record_is_bound_to_current_manifest_and_all_artifacts(
        self,
    ) -> None:
        store = self.make_store(self.owner_a)
        for cell in self.cells:
            self.accept(store, cell)
        key = self.cells[0].key
        path = store.accepted_path(key)
        original = json.loads(path.read_text())
        mutations = {
            "schema": {**original, "schema": "stale-schema"},
            "cell": {
                **original,
                "cell": {"task_index": 0, "arm": "amg_memory"},
            },
            "instance": {**original, "instance_id": "other-task"},
            "manifest": {**original, "manifest_cell_sha256": SHA_D},
            "generation": {**original, "attempt_generation": 0},
            "endpoint": {**original, "endpoint_sha256": SHA_D},
            "prediction": {**original, "prediction_sha256": SHA_D},
            "handoff": {**original, "handoff_sha256": SHA_D},
            "fields": {**original, "unexpected": True},
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                atomic_write_json(path, mutation)
                with self.assertRaises(ValueError):
                    store.assemble_results()
                atomic_write_json(path, original)

        stale_cells = (
            ManifestCell(key, "task-0", SHA_D),
            *self.cells[1:],
        )
        stale_store = CellStateStore(
            self.root,
            manifest=stale_cells,
            owner=self.owner_a,
            owner_is_alive=lambda candidate: candidate in self.live,
            endpoint_validator=endpoint_validator,
        )
        with self.assertRaises(ValueError):
            stale_store.assemble_results()

    def test_endpoint_rejects_conflicting_dual_task_identities(self) -> None:
        store = self.make_store(self.owner_a)
        token = store.acquire(self.cells[0].key)
        row = endpoint_row("task-0", "native")
        row["task_id"] = "different-task"
        with self.assertRaisesRegex(ValueError, "cell identity drifted"):
            store.record_endpoint(token, row)

    def test_results_and_outcomes_require_exact_manifest_join(self) -> None:
        store = self.make_store(self.owner_a)
        for cell in self.cells:
            self.accept(store, cell)
        rows = store.assemble_results()
        self.assertEqual([row["arm"] for row in rows], [cell.key.arm for cell in self.cells])

        grade_tokens = {}
        for index, cell in enumerate(self.cells):
            grade_tokens[cell.key] = store.acquire_grade(cell.key)
            store.record_official_outcome(
                grade_tokens[cell.key],
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
                grade_tokens[self.cells[0].key],
                {
                    "instance_id": "task-0",
                    "arm": "native",
                    "resolved": True,
                    "failure_class": None,
                    "report_sha256": SHA_D,
                },
            )

    def test_official_outcome_is_bound_to_validated_accepted_attempt(self) -> None:
        store = self.make_store(self.owner_a)
        for cell in self.cells:
            self.accept(store, cell)
            token = store.acquire_grade(cell.key)
            store.record_official_outcome(
                token,
                {
                    "instance_id": cell.instance_id,
                    "arm": cell.key.arm,
                    "resolved": False,
                    "failure_class": None,
                    "report_sha256": SHA_D,
                },
            )

        key = self.cells[0].key
        path = store.outcome_path(key)
        original = json.loads(path.read_text())
        mutations = {
            "schema": {**original, "schema": "stale-schema"},
            "prediction": {**original, "prediction_sha256": SHA_A},
            "generation": {
                **original,
                "attempt_generation": original["attempt_generation"] + 1,
            },
            "report": {**original, "report_sha256": "not-a-digest"},
            "fields": {**original, "unexpected": True},
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                atomic_write_json(path, mutation)
                with self.assertRaises(ValueError):
                    store.official_summary()
                with self.assertRaises(ValueError):
                    store.acquire_grade(key)
                atomic_write_json(path, original)

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
        grade_token = store.acquire_grade(self.cells[0].key)
        with self.assertRaises(ValueError):
            store.record_official_outcome(
                grade_token,
                {
                    "instance_id": "task-0",
                    "arm": "native",
                    "resolved": 1,
                    "failure_class": None,
                    "report_sha256": SHA_D,
                },
            )

    def test_official_grading_is_live_owner_fenced(self) -> None:
        store_a = self.make_store(self.owner_a)
        self.accept(store_a, self.cells[0])
        token_a = store_a.acquire_grade(self.cells[0].key)
        self.live.remove(self.owner_a)
        store_b = self.make_store(self.owner_b)
        token_b = store_b.acquire_grade(self.cells[0].key)
        self.assertEqual(token_b.generation, token_a.generation + 1)
        outcome = {
            "instance_id": "task-0",
            "arm": "native",
            "resolved": False,
            "failure_class": None,
            "report_sha256": SHA_D,
        }
        with self.assertRaises(FenceViolationError):
            store_a.record_official_outcome(token_a, outcome)
        store_b.record_official_outcome(token_b, outcome)


class CrossHostLeaseRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "leases"
        self.now = 1_000_000_000
        self.owner_a = OwnerIdentity("host-a", "boot-a", 101, 1001)
        self.owner_b = OwnerIdentity("host-b", "boot-b", 202, 2002)

    def registry(self, owner, tasks):
        return DriverLeaseRegistry(
            self.root,
            owner=owner,
            assigned_task_indices=tasks,
            now_ns=lambda: self.now,
            ttl_ns=60_000_000_000,
            local_owner_is_alive=lambda _owner: True,
            slot_ports=(18100, 18101),
        )

    def test_disjoint_shards_share_distinct_slots_but_not_one_slot(self) -> None:
        first = self.registry(self.owner_a, (0, 2, 4))
        second = self.registry(self.owner_b, (1, 3, 5))
        first.acquire()
        second.acquire()
        self.assertTrue(second.owner_is_alive(self.owner_a))
        self.assertEqual(first.assigned_task_indices, (0, 2, 4))
        self.assertEqual(second.assigned_task_indices, (1, 3, 5))

        first_token = first.acquire_lane(task_index=0, slot_index=0)
        with self.assertRaises(ClaimBusyError):
            second.acquire_lane(task_index=1, slot_index=0)
        second_token = second.acquire_lane(task_index=1, slot_index=1)
        self.assertNotEqual(first_token.server_port, second_token.server_port)
        first.release_lane(first_token)
        takeover = second.acquire_lane(task_index=1, slot_index=0)
        self.assertEqual(takeover.slot_index, 0)

    def test_stale_slot_fence_cannot_release_successor(self) -> None:
        first = self.registry(self.owner_a, (0,))
        first.acquire()
        token = first.acquire_lane(task_index=0, slot_index=0)
        first.release_lane(token)
        successor = first.acquire_lane(task_index=0, slot_index=0)
        self.assertGreater(successor.generation, token.generation)
        with self.assertRaises(FenceViolationError):
            first.release_lane(token)
        first.assert_lane(successor)

    def test_dead_slot_owner_is_taken_over_with_a_new_generation(self) -> None:
        first = self.registry(self.owner_a, (0,))
        second = self.registry(self.owner_b, (1,))
        first.acquire()
        second.acquire()
        original = first.acquire_lane(task_index=0, slot_index=0)
        first._close_process_liveness_lock()
        successor = second.acquire_lane(task_index=1, slot_index=0)
        self.assertGreater(successor.generation, original.generation)
        self.assertNotEqual(successor.fencing_token, original.fencing_token)
        with self.assertRaises(FenceViolationError):
            first.assert_lane(original)

    def test_overlapping_two_driver_race_has_exactly_one_winner(self) -> None:
        first = self.registry(self.owner_a, (0,))
        second = self.registry(self.owner_b, (0,))
        barrier = threading.Barrier(2)
        results = []
        result_lock = threading.Lock()

        def race(registry):
            barrier.wait()
            try:
                registry.acquire()
                value = "acquired"
            except ClaimBusyError:
                value = "busy"
            with result_lock:
                results.append(value)

        threads = [
            threading.Thread(target=race, args=(registry,))
            for registry in (first, second)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(sorted(results), ["acquired", "busy"])

    def test_cross_host_lock_must_release_before_takeover(self) -> None:
        first = self.registry(self.owner_a, (0,))
        first.acquire()
        second = self.registry(self.owner_b, (0,))
        self.assertTrue(second.owner_is_alive(self.owner_a))
        self.now += 61_000_000_000
        self.assertTrue(second.owner_is_alive(self.owner_a))
        first._close_process_liveness_lock()
        self.assertFalse(second.owner_is_alive(self.owner_a))
        second.acquire()
        self.assertTrue(second.owner_is_alive(self.owner_b))


if __name__ == "__main__":
    unittest.main()
