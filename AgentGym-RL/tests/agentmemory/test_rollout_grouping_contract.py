import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.agent_trainer.ppo.ray_trainer import _agentmemory_dump_ppo_batch_debug
from verl.utils.agentgym.rollout_context import validate_state_aware_rollout_uids
from verl.workers.rollout.agent_vllm_rollout.agentmemory_grouping import (
    expand_excluded_rollout_parent_groups,
    trainable_rollout_row_positions,
)


class RolloutGroupingContractTest(unittest.TestCase):
    def _rollout_output(self, rollout_uids=None):
        non_tensors = {
            "rollout_parent_indices": np.array([0, 0], dtype=object),
        }
        if rollout_uids is not None:
            non_tensors["rollout_uid"] = np.array(rollout_uids, dtype=object)
        return DataProto.from_dict(
            tensors={"responses": torch.zeros(2, 1)},
            non_tensors=non_tensors,
        )

    def test_action_level_rollout_requires_state_aware_uid(self):
        with self.assertRaisesRegex(RuntimeError, "missing rollout_uid"):
            validate_state_aware_rollout_uids(self._rollout_output())
        with self.assertRaisesRegex(RuntimeError, "not prompt-state-aware"):
            validate_state_aware_rollout_uids(
                self._rollout_output(["source-uuid", "source-uuid"])
            )
        validate_state_aware_rollout_uids(
            self._rollout_output(
                ["0:turn1:statev1:abc", "0:turn1:statev1:abc"]
            )
        )

    def test_debug_uid_groups_exclude_padding(self):
        batch = DataProto.from_dict(
            tensors={
                "response_mask": torch.tensor([[1, 1], [1, 0], [0, 0]]),
                "scores": torch.zeros(3, 2),
                core_algos.PPO_VALID_SAMPLE_MASK: torch.tensor([True, True, False]),
            },
            non_tensors={
                "uid": np.array(["state-a", "state-b", "state-a"], dtype=object),
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(
                trainer=SimpleNamespace(
                    default_local_dir=str(Path(tmpdir) / "checkpoints")
                )
            )
            with mock.patch.dict(os.environ, {"AGENTMEMORY_BATCH_DEBUG": "1"}):
                _agentmemory_dump_ppo_batch_debug(
                    batch=batch,
                    config=config,
                    global_steps=1,
                    stage="post_adv",
                )
            debug_path = Path(tmpdir) / "diagnostics/ppo_batch_step1_post_adv.json"
            summary = json.loads(debug_path.read_text())

        self.assertEqual(summary["batch_size"], 3)
        self.assertEqual(summary["valid_rows"], 2)
        self.assertEqual(summary["padding_rows"], 1)
        self.assertEqual(summary["uid_group_sizes"], [1, 1])
        self.assertFalse(summary["rows"][2]["ppo_valid_sample"])

    def test_formal_capture_and_infra_exclusion_are_both_wired(self):
        rollout_path = (
            Path(__file__).resolve().parents[2]
            / "verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py"
        )
        source = rollout_path.read_text()
        function_start = source.index("def generate_task_neutral_policy")
        function_end = source.index("@torch.no_grad()", function_start)
        function_source = source[function_start:function_end]

        for required in (
            "env_info_before",
            "env_info_after",
            "trajectory_row_uid",
            "trajectory_row_order",
            "trajectory_terminal",
            "flat_step_refs",
            "excluded_rollout_indices",
            "flat_rollout_indices",
            "expand_excluded_rollout_parent_groups",
            "trainable_rollout_row_positions",
            "sample_excluded",
        ):
            self.assertIn(required, function_source)
        self.assertLess(
            function_source.index(
                'getattr(env_clients[index], "sample_excluded"'
            ),
            function_source.index("step_record = build_task_neutral_step_record"),
        )
        pack_start = source.index("def pack_rollout_handlers")
        pack_end = source.index("def _task_neutral_prompt_from_messages", pack_start)
        pack_source = source[pack_start:pack_end]
        self.assertIn("AGENTMEMORY_STEP_RECORD_JSON", pack_source)
        self.assertIn("validate_formal_runtime_evidence_rows", pack_source)
        self.assertIn("TASK_NEUTRAL_POLICY_STEP_SCHEMA", pack_source)
        self.assertNotIn("FORMAL_WEBSHOP_SCHEMA_V2", pack_source)
        self.assertNotIn("FORMAL_DOMAIN_SCHEMA_V3", pack_source)

    def test_infra_failure_removes_the_complete_parent_group(self):
        expanded = expand_excluded_rollout_parent_groups(
            rollout_parent_indices=[0, 0, 1, 1],
            excluded_rollout_indices=[1],
        )
        self.assertEqual(expanded, {0, 1})
        keep = trainable_rollout_row_positions(
            flat_rollout_indices=[0, 2, 1, 3],
            excluded_rollout_indices=expanded,
        )
        self.assertEqual(keep, [1, 3])
        with self.assertRaisesRegex(ValueError, "outside"):
            expand_excluded_rollout_parent_groups([0, 0], [2])


if __name__ == "__main__":
    unittest.main()
