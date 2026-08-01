from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "verl/utils/agentgym/rollout_context.py"
SPEC = importlib.util.spec_from_file_location(
    "agentmemory_formal_validation_reuse_for_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeDataProto:
    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = {}

    def __len__(self):
        return len(self.batch["response_mask"])


class FormalValidationReuseTest(unittest.TestCase):
    update_keys = (
        "old_log_probs",
        "values",
        "token_level_scores",
        "token_level_rewards",
        "advantages",
        "returns",
    )

    def _base_data(self):
        row_count = 3
        response_mask = torch.tensor(
            [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]],
            dtype=torch.long,
        )
        batch = {
            MODULE.AGENTMEMORY_TRAJECTORY_RETURN: torch.tensor([1.0, 0.0, 1.0]),
            MODULE.AGENTMEMORY_IMMEDIATE_REWARD: torch.tensor([0.0, 1.0, 0.0]),
            MODULE.AGENTMEMORY_TRAJECTORY_ROW_ORDER: torch.tensor([0, 1, 0]),
            MODULE.AGENTMEMORY_TRAJECTORY_TERMINAL: torch.tensor(
                [False, True, False]
            ),
            MODULE.AGENTMEMORY_GENERATION_PROMPT_LENGTH: torch.tensor([2, 2, 2]),
            MODULE.AGENTMEMORY_PACKED_PROMPT_LENGTH: torch.tensor([2, 2, 2]),
            MODULE.AGENTMEMORY_GENERATION_RESPONSE_LENGTH: torch.tensor([2, 3, 2]),
            MODULE.AGENTMEMORY_PACKED_RESPONSE_LENGTH: torch.tensor([2, 3, 2]),
            MODULE.AGENTMEMORY_SUFFIX_CREDIT_APPLIED: torch.zeros(
                row_count, dtype=torch.bool
            ),
            MODULE.AGENTMEMORY_SUFFIX_RETURN: torch.tensor([1.0, 1.0, 0.0]),
            "prompts": torch.tensor([[1, 2], [3, 4], [5, 6]]),
            "attention_mask": torch.ones(row_count, 6, dtype=torch.long),
            "responses": torch.tensor(
                [[7, 8, 0, 0], [9, 10, 11, 0], [12, 13, 0, 0]]
            ),
            "response_mask": response_mask,
            "scores": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0] * 4]
            ),
            "task_rounds": torch.tensor([1, 2, 1]),
            "ppo_valid_sample_mask": torch.tensor([True, True, False]),
        }
        non_tensor_batch = {
            MODULE.AGENTMEMORY_PARENT_GROUP_UID: np.array(["g", "g", "g"]),
            MODULE.AGENTMEMORY_EXACT_STATE_UID: np.array(["s0", "s1", "s2"]),
            MODULE.AGENTMEMORY_REPLICA_INDEX: np.array([0, 0, 1], dtype=object),
            MODULE.AGENTMEMORY_TRAJECTORY_UID: np.array(["t0", "t0", "t1"]),
            MODULE.AGENTMEMORY_TRAJECTORY_ROW_UID: np.array(["r0", "r1", "r2"]),
            MODULE.AGENTMEMORY_ACTION_TEXT: np.array(["a0", "a1", "a2"]),
            MODULE.AGENTMEMORY_GENERATION_PROMPT_DIGEST: np.array(["p0", "p1", "p2"]),
            MODULE.AGENTMEMORY_PACKED_PROMPT_DIGEST: np.array(["p0", "p1", "p2"]),
            MODULE.AGENTMEMORY_GENERATION_RESPONSE_DIGEST: np.array(["q0", "q1", "q2"]),
            MODULE.AGENTMEMORY_PACKED_RESPONSE_DIGEST: np.array(["q0", "q1", "q2"]),
            MODULE.AGENTMEMORY_STEP_RECORD_JSON: np.array(["{}", "{}", "{}"]),
            "rollout_parent_indices": np.array([0, 0, 0], dtype=object),
            "rollout_uid": np.array(["s0", "s1", "s2"]),
            "rollout_done_flags": np.array([False, True, False], dtype=object),
        }
        return FakeDataProto(batch, non_tensor_batch)

    def _validated_update_data(self):
        data = self._base_data()
        groups = {"g": [{"trajectory_uid": "t0", "row_indices": [0, 1]}]}
        receipt = MODULE.capture_formal_validation_receipt(data, grouped=groups)
        data.batch["response_mask"] = data.batch["response_mask"] * data.batch[
            "ppo_valid_sample_mask"
        ].unsqueeze(-1)
        for key in self.update_keys:
            data.batch[key] = torch.zeros(3, 4)
        return data, receipt

    def test_incremental_validator_accepts_new_update_tensors(self):
        data, receipt = self._validated_update_data()
        summary = MODULE.validate_formal_ppo_update_tensors(
            data,
            receipt=receipt,
            required_tensor_keys=self.update_keys,
        )
        self.assertEqual(summary, {"tensor_count": 6, "checked_rows": 2})

    def test_missing_update_tensor_fails_closed(self):
        data, receipt = self._validated_update_data()
        del data.batch["returns"]
        with self.assertRaisesRegex(RuntimeError, "missing"):
            MODULE.validate_formal_ppo_update_tensors(
                data,
                receipt=receipt,
                required_tensor_keys=self.update_keys,
            )

    def test_nonfinite_update_tensor_fails_closed(self):
        data, receipt = self._validated_update_data()
        data.batch["advantages"][0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            MODULE.validate_formal_ppo_update_tensors(
                data,
                receipt=receipt,
                required_tensor_keys=self.update_keys,
            )

    def test_misaligned_update_tensor_fails_closed(self):
        data, receipt = self._validated_update_data()
        data.batch["returns"] = torch.zeros(3, 3)
        with self.assertRaisesRegex(RuntimeError, "shape differs"):
            MODULE.validate_formal_ppo_update_tensors(
                data,
                receipt=receipt,
                required_tensor_keys=self.update_keys,
            )

    def test_changed_row_identity_fails_closed(self):
        data, receipt = self._validated_update_data()
        data.non_tensor_batch[MODULE.AGENTMEMORY_TRAJECTORY_ROW_UID][0] = "other"
        with self.assertRaisesRegex(RuntimeError, "changed after validation"):
            MODULE.validate_formal_ppo_update_tensors(
                data,
                receipt=receipt,
                required_tensor_keys=self.update_keys,
            )

    def test_changed_immutable_tensor_fails_closed(self):
        data, receipt = self._validated_update_data()
        data.batch["responses"][0, 0] = 99
        with self.assertRaisesRegex(RuntimeError, "tensor responses changed"):
            MODULE.validate_formal_ppo_update_tensors(
                data,
                receipt=receipt,
                required_tensor_keys=self.update_keys,
            )

    def test_changed_valid_response_mask_fails_closed(self):
        data, receipt = self._validated_update_data()
        data.batch["response_mask"][0] = torch.tensor([0, 1, 1, 0])
        with self.assertRaisesRegex(RuntimeError, "response mask changed"):
            MODULE.validate_formal_ppo_update_tensors(
                data,
                receipt=receipt,
                required_tensor_keys=self.update_keys,
            )


if __name__ == "__main__":
    unittest.main()
