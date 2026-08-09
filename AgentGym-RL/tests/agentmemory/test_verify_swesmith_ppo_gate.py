#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "agentmemory" / "verify_swesmith_ppo_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_swesmith_ppo_gate", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load SWE-smith PPO verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SwesmithPpoGateRowEvidenceTests(unittest.TestCase):
    def test_accepts_formal_step_records_and_binds_all_indices(self) -> None:
        module = load_module()
        rows = [
            {
                "schema_version": module.STEP_SCHEMA,
                "parent_index": index,
                "item_id": f"swesmith_{index}",
            }
            for index in range(8)
        ]
        self.assertEqual(
            module.verify_row_evidence(
                {
                    "row_evidence": {
                        "schema": "agentmemory_formal_step_records_v1",
                        "task_name": "swesmith",
                        "rows": rows,
                    },
                    "formal_step_records": rows,
                },
                set(range(8)),
            ),
            {
                "schema": "agentmemory_formal_step_records_v1",
                "dataset_indices": list(range(8)),
                "row_count": 8,
            },
        )

    def test_accepts_generic_dataset_rows(self) -> None:
        module = load_module()
        self.assertEqual(
            module.verify_row_evidence(
                {
                    "row_evidence": {
                        "schema": "generic_task_dataset_rows_v1",
                        "task_name": "swesmith",
                        "index_field": "index",
                        "dataset_indices": list(range(8)),
                    }
                },
                set(range(8)),
            )["dataset_indices"],
            list(range(8)),
        )

    def test_rejects_incomplete_index_coverage(self) -> None:
        module = load_module()
        with self.assertRaises(AssertionError):
            module.verify_row_evidence(
                {
                    "row_evidence": {
                        "schema": "generic_task_dataset_rows_v1",
                        "task_name": "swesmith",
                        "index_field": "index",
                        "dataset_indices": list(range(7)),
                    }
                },
                set(range(8)),
            )


class SwesmithPpoGateAuditSelectionTests(unittest.TestCase):
    @staticmethod
    def _audit(*, audit_id: str, index: int, slot: int, started_at: str) -> dict:
        return {
            "schema": "agentmemory_swesmith_private_episode_audit_v1",
            "audit_id": audit_id,
            "data_idx": index,
            "slot_id": slot,
            "started_at": started_at,
            "close_reason": "client_close",
            "done": True,
            "reward": 1.0 if index == 0 else 0.0,
            "grade": {"resolution_status": "RESOLVED_YES" if index == 0 else "RESOLVED_NO"},
            "step_count": 3,
        }

    def test_excludes_stale_preflight_audits_from_reused_endpoint(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            current_probe_ids = []
            for index in range(8):
                stale_id = f"{index + 1:032x}"
                probe_id = f"{index + 101:032x}"
                trainer_id = f"{index + 201:032x}"
                current_probe_ids.append(probe_id)
                fixtures = (
                    self._audit(
                        audit_id=stale_id,
                        index=index,
                        slot=index,
                        started_at="2026-08-09T00:00:00Z",
                    ),
                    self._audit(
                        audit_id=probe_id,
                        index=index,
                        slot=index,
                        started_at="2026-08-09T00:30:00Z",
                    ),
                    self._audit(
                        audit_id=trainer_id,
                        index=index,
                        slot=index,
                        started_at="2026-08-09T01:00:01Z",
                    ),
                )
                for payload in fixtures:
                    (root / f"episode-{payload['audit_id']}.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )

            result = module.verify_audits(
                root,
                {"audit_ids": current_probe_ids},
                set(range(8)),
                module.parse_time("2026-08-09T01:00:00Z", "test.started_at"),
            )

        self.assertEqual(result["audit_count"], 8)
        self.assertEqual(result["dataset_indices"], list(range(8)))
        self.assertEqual(result["stale_audit_count"], 8)
        self.assertEqual(result["selection"], "run-start-time-minus-current-probe")

    def test_rejects_naive_run_timestamp(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(AssertionError, "must include a timezone"):
            module.parse_time("2026-08-09T01:00:00", "test.started_at")

    def test_accepts_repeated_task_indices_for_batch64(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for parent_index in range(64):
                task_index = parent_index % 8
                audit_id = f"{parent_index + 1:032x}"
                payload = self._audit(
                    audit_id=audit_id,
                    index=task_index,
                    slot=parent_index,
                    started_at="2026-08-09T01:00:01Z",
                )
                (root / f"episode-{audit_id}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            result = module.verify_audits(
                root,
                {"audit_ids": []},
                set(range(8)),
                module.parse_time("2026-08-09T01:00:00Z", "test.started_at"),
                expected_audit_count=64,
                expected_data_idx_counts=module.Counter({index: 8 for index in range(8)}),
            )

        self.assertEqual(result["audit_count"], 64)
        self.assertEqual(result["data_idx_counts"], {str(index): 8 for index in range(8)})


if __name__ == "__main__":
    unittest.main()
