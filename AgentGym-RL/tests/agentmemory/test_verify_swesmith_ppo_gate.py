#!/usr/bin/env python3

import importlib.util
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
