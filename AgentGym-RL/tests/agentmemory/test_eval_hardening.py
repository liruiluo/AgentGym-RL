import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from verl import DataProto


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl/agent_trainer/main_generation.py"
)
_SPEC = importlib.util.spec_from_file_location("main_generation_for_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class EvalHardeningTest(unittest.TestCase):
    def test_formal_eval_scrubs_train_only_rollout_flags(self):
        env = {
            "AGENTMEMORY_ACTION_SEQUENCE_ENUMERATION_ROLLOUT": "1",
            "AGENTMEMORY_LATEST_OBS_SUFFIX_CREDIT": "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            removed = _MODULE._scrub_training_rollout_flags()
            self.assertEqual(sorted(removed), sorted(env))
            for name in env:
                self.assertNotIn(name, os.environ)

        with mock.patch.dict(
            os.environ,
            {
                **env,
                "AGENTMEMORY_ALLOW_TRAIN_ROLLOUT_FLAGS_IN_EVAL": "1",
            },
            clear=True,
        ):
            self.assertEqual(_MODULE._scrub_training_rollout_flags(), [])
            for name in env:
                self.assertIn(name, os.environ)

    def test_tiny_eval_batch_pads_to_dp_size(self):
        data = DataProto.from_dict(
            tensors={"row": torch.tensor([[7]])},
            non_tensors={"item_id": np.array(["only"], dtype=object)},
        )
        padded, dummy_count = _MODULE._pad_dataproto_for_dp(data, dp_size=8)
        self.assertEqual(len(padded), 8)
        self.assertEqual(dummy_count, 7)
        torch.testing.assert_close(padded.batch["row"], torch.full((8, 1), 7))

    def test_episode_aggregation_rejects_misaligned_rows(self):
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[1.0], [2.0]])},
            non_tensors={
                "rollout_parent_indices": np.array([0, 0], dtype=object),
                "rollout_done_flags": np.array([True, True], dtype=object),
            },
        )
        output.non_tensor_batch["rollout_parent_indices"] = np.array(
            [0], dtype=object
        )
        with self.assertRaisesRegex(ValueError, "misaligned"):
            _MODULE._aggregate_episode_scores(output, real_batch_size=1)


if __name__ == "__main__":
    unittest.main()
