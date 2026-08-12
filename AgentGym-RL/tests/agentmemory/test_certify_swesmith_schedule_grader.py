from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.agentmemory.certify_swesmith_schedule_grader import (
    load_schedule_indices,
    result_passes,
)


class CertifySwesmithScheduleGraderTest(unittest.TestCase):
    def write_routing(self, rows: list[dict[str, object]]) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", delete=False)
        with temporary:
            for row in rows:
                temporary.write(json.dumps(row) + "\n")
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return Path(temporary.name)

    def test_loads_exact_schedule_prefix(self) -> None:
        path = self.write_routing(
            [
                {
                    "item_id": f"swesmith_{position}",
                    "data_idx": data_idx,
                    "extra_info": {
                        "index": data_idx,
                        "schedule_position": position,
                    },
                }
                for position, data_idx in enumerate((4, 1, 7, 2, 9, 3))
            ]
        )
        self.assertEqual(
            load_schedule_indices(path, train_batch_size=2, gate_steps=2),
            [4, 1, 7, 2],
        )

    def test_rejects_duplicate_schedule_prefix(self) -> None:
        path = self.write_routing(
            [
                {
                    "item_id": f"swesmith_{position}",
                    "data_idx": data_idx,
                    "extra_info": {
                        "index": data_idx,
                        "schedule_position": position,
                    },
                }
                for position, data_idx in enumerate((4, 1, 4, 2))
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_schedule_indices(path, train_batch_size=2, gate_steps=2)

    def test_result_contract_separates_positive_and_negative_arms(self) -> None:
        self.assertTrue(
            result_passes(
                {
                    "arm": "gold",
                    "reward": 1.0,
                    "episode_success": True,
                    "resolution_status": "RESOLVED_FULL",
                    "grader_error": None,
                    "errors": [],
                }
            )
        )
        for arm in ("wrong", "tamper"):
            self.assertTrue(
                result_passes(
                    {
                        "arm": arm,
                        "reward": 0.0,
                        "episode_success": False,
                        "resolution_status": "UNRESOLVED",
                        "grader_error": None,
                        "errors": [],
                    }
                )
            )
        self.assertFalse(
            result_passes(
                {
                    "arm": "wrong",
                    "reward": 1.0,
                    "episode_success": True,
                    "resolution_status": "RESOLVED_FULL",
                    "grader_error": None,
                    "errors": [],
                }
            )
        )
        self.assertFalse(
            result_passes(
                {
                    "arm": "gold",
                    "reward": None,
                    "episode_success": False,
                    "resolution_status": None,
                    "grader_error": None,
                    "errors": ["reset failed"],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
