from __future__ import annotations

import unittest
from copy import deepcopy
from types import SimpleNamespace

from agentmemorygym_verl.dataset import AMGTrajectoryDataset


class TestAMGTrajectoryDataset(unittest.TestCase):
    def _dataset(self, rows):
        dataset = object.__new__(AMGTrajectoryDataset)
        dataset.dataframe = deepcopy(rows)
        dataset._policy_framing = [
            {"role": "system", "content": "Use ordinary shell and filesystem actions."}
        ]
        dataset.config = SimpleNamespace(
            agentgym=SimpleNamespace(task_name="openmle_fast")
        )
        return dataset

    def test_preserves_frozen_schedule_identity_and_order_without_mutating_source(self):
        rows = [
            {"item_id": "task-b", "data_idx": 9, "extra_info": {"index": 9}},
            {"item_id": "task-a", "data_idx": 3, "extra_info": {"index": 3}},
        ]
        dataset = self._dataset(rows)
        first = dataset[0]
        second = dataset[1]

        self.assertEqual(
            (first["item_id"], first["data_idx"], first["index"]), ("task-b", 9, 9)
        )
        self.assertEqual(
            (second["item_id"], second["data_idx"], second["index"]), ("task-a", 3, 3)
        )
        self.assertEqual(dataset.dataframe, rows)
        first["raw_prompt"][0]["content"] = "mutated"
        self.assertNotEqual(first["raw_prompt"], dataset._policy_framing)

    def test_rejects_nonintegral_data_idx(self):
        dataset = self._dataset([{"item_id": "task", "data_idx": 1.5}])
        with self.assertRaisesRegex(ValueError, "data_idx must be an integer"):
            dataset[0]

    def test_rejects_schedule_index_drift(self):
        dataset = self._dataset(
            [{"item_id": "task", "data_idx": 7, "extra_info": {"index": 8}}]
        )
        with self.assertRaisesRegex(ValueError, "schedule index differs"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
