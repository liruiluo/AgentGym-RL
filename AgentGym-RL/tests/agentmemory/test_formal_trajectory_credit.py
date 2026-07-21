from __future__ import annotations

import importlib.util
import importlib
import ast
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


_ROOT = Path(__file__).resolve().parents[2]
_GROUPING_PATH = (
    _ROOT / "verl/workers/rollout/agent_vllm_rollout/agentmemory_grouping.py"
)
_GROUPING_SPEC = importlib.util.spec_from_file_location(
    "agentmemory_formal_grouping_for_test", _GROUPING_PATH
)
assert _GROUPING_SPEC is not None and _GROUPING_SPEC.loader is not None
grouping = importlib.util.module_from_spec(_GROUPING_SPEC)
_GROUPING_SPEC.loader.exec_module(grouping)
sys.modules.setdefault(
    "verl.workers.rollout.agent_vllm_rollout.agentmemory_grouping", grouping
)

# core_algos only needs this module for helpers outside the focused function.
torch_functional_stub = types.ModuleType("verl.utils.torch_functional")
torch_functional_stub.masked_whiten = lambda values, mask: values
torch_functional_stub.masked_mean = lambda values, mask: values[mask.bool()].mean()
verl_utils = importlib.import_module("verl.utils")
sys.modules.setdefault("verl.utils.torch_functional", torch_functional_stub)
setattr(verl_utils, "torch_functional", torch_functional_stub)
_CORE_PATH = _ROOT / "verl/agent_trainer/ppo/core_algos.py"
_CORE_SPEC = importlib.util.spec_from_file_location(
    "agentmemory_formal_core_algos_for_test", _CORE_PATH
)
assert _CORE_SPEC is not None and _CORE_SPEC.loader is not None
core_algos = importlib.util.module_from_spec(_CORE_SPEC)
_CORE_SPEC.loader.exec_module(core_algos)

_ROLLOUT_CONTEXT_PATH = _ROOT / "verl/utils/agentgym/rollout_context.py"
_ROLLOUT_CONTEXT_SPEC = importlib.util.spec_from_file_location(
    "agentmemory_formal_rollout_context_for_test", _ROLLOUT_CONTEXT_PATH
)
assert (
    _ROLLOUT_CONTEXT_SPEC is not None
    and _ROLLOUT_CONTEXT_SPEC.loader is not None
)
rollout_context = importlib.util.module_from_spec(_ROLLOUT_CONTEXT_SPEC)
_ROLLOUT_CONTEXT_SPEC.loader.exec_module(rollout_context)


class FakeDataProto:
    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = {}

    def __len__(self):
        for value in self.batch.values():
            return len(value)
        for value in self.non_tensor_batch.values():
            return len(value)
        return 0


def load_ray_trainer_functions():
    trainer_path = _ROOT / "verl/agent_trainer/ppo/ray_trainer.py"
    tree = ast.parse(trainer_path.read_text())
    wanted = {
        "_agentmemory_env_flag",
        "_safe_float",
        "_get_ppo_valid_sample_mask",
        "compute_advantage",
        "_agentmemory_dump_ppo_batch_debug",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    namespace = {
        "AGENTMEMORY_EXACT_STATE_UID": grouping.AGENTMEMORY_EXACT_STATE_UID,
        "AGENTMEMORY_IMMEDIATE_REWARD": grouping.AGENTMEMORY_IMMEDIATE_REWARD,
        "AGENTMEMORY_PARENT_GROUP_UID": grouping.AGENTMEMORY_PARENT_GROUP_UID,
        "AGENTMEMORY_REPLICA_INDEX": grouping.AGENTMEMORY_REPLICA_INDEX,
        "AGENTMEMORY_TRAJECTORY_RETURN": grouping.AGENTMEMORY_TRAJECTORY_RETURN,
        "AGENTMEMORY_TRAJECTORY_ROW_ORDER": (
            grouping.AGENTMEMORY_TRAJECTORY_ROW_ORDER
        ),
        "AGENTMEMORY_TRAJECTORY_ROW_UID": (
            grouping.AGENTMEMORY_TRAJECTORY_ROW_UID
        ),
        "AGENTMEMORY_TRAJECTORY_TERMINAL": (
            grouping.AGENTMEMORY_TRAJECTORY_TERMINAL
        ),
        "AGENTMEMORY_TRAJECTORY_UID": grouping.AGENTMEMORY_TRAJECTORY_UID,
        "DataProto": FakeDataProto,
        "core_algos": core_algos,
        "json": json,
        "np": np,
        "os": os,
        "requires_formal_trajectory_metadata": (
            rollout_context.requires_formal_trajectory_metadata
        ),
        "torch": torch,
        "validate_formal_trajectory_metadata": (
            rollout_context.validate_formal_trajectory_metadata
        ),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), trainer_path, "exec"),
        namespace,
    )
    return namespace


ray_trainer_functions = load_ray_trainer_functions()


def build_rows(turn_counts=(3, 5, 4, 6), returns=(1.0, 3.0, 0.0, 2.0)):
    parent_group_uids = []
    exact_state_uids = []
    replica_indices = []
    trajectory_uids = []
    trajectory_returns = []
    immediate_rewards = []
    trajectory_row_uids = []
    trajectory_row_orders = []
    trajectory_terminals = []
    parent_indices = []
    rollout_uids = []
    for replica_index, (turn_count, trajectory_return) in enumerate(
        zip(turn_counts, returns)
    ):
        parent_group_uid = grouping.build_parent_group_uid(7)
        trajectory_uid = grouping.build_trajectory_uid(
            parent_group_uid, replica_index
        )
        for turn in range(1, turn_count + 1):
            row_order = turn - 1
            exact_uid = f"7:turn{turn}:statev1:r{replica_index}-s{turn}"
            parent_group_uids.append(parent_group_uid)
            exact_state_uids.append(exact_uid)
            rollout_uids.append(exact_uid)
            replica_indices.append(replica_index)
            trajectory_uids.append(trajectory_uid)
            trajectory_returns.append(trajectory_return)
            immediate_rewards.append(
                trajectory_return if turn == turn_count else 0.0
            )
            trajectory_row_uids.append(
                grouping.build_row_uid(trajectory_uid, row_order)
            )
            trajectory_row_orders.append(row_order)
            trajectory_terminals.append(turn == turn_count)
            parent_indices.append(7)
    return {
        "parent_group_uids": parent_group_uids,
        "exact_state_uids": exact_state_uids,
        "replica_indices": replica_indices,
        "trajectory_uids": trajectory_uids,
        "trajectory_returns": trajectory_returns,
        "immediate_rewards": immediate_rewards,
        "trajectory_row_uids": trajectory_row_uids,
        "trajectory_row_orders": trajectory_row_orders,
        "trajectory_terminals": trajectory_terminals,
        "parent_indices": parent_indices,
        "rollout_uids": rollout_uids,
        "valid_mask": [True] * len(parent_group_uids),
    }


class FormalTrajectoryGroupingTest(unittest.TestCase):
    def _data_proto(self, rows):
        return FakeDataProto(
            batch={
                grouping.AGENTMEMORY_TRAJECTORY_RETURN: torch.tensor(
                    rows["trajectory_returns"]
                ),
                grouping.AGENTMEMORY_IMMEDIATE_REWARD: torch.tensor(
                    rows["immediate_rewards"]
                ),
                grouping.AGENTMEMORY_TRAJECTORY_ROW_ORDER: torch.tensor(
                    rows["trajectory_row_orders"], dtype=torch.long
                ),
                grouping.AGENTMEMORY_TRAJECTORY_TERMINAL: torch.tensor(
                    rows["trajectory_terminals"], dtype=torch.bool
                ),
            },
            non_tensor_batch={
                grouping.AGENTMEMORY_PARENT_GROUP_UID: np.array(
                    rows["parent_group_uids"], dtype=object
                ),
                grouping.AGENTMEMORY_EXACT_STATE_UID: np.array(
                    rows["exact_state_uids"], dtype=object
                ),
                grouping.AGENTMEMORY_REPLICA_INDEX: np.array(
                    rows["replica_indices"], dtype=object
                ),
                grouping.AGENTMEMORY_TRAJECTORY_UID: np.array(
                    rows["trajectory_uids"], dtype=object
                ),
                grouping.AGENTMEMORY_TRAJECTORY_ROW_UID: np.array(
                    rows["trajectory_row_uids"], dtype=object
                ),
                "rollout_parent_indices": np.array(
                    rows["parent_indices"], dtype=object
                ),
                "rollout_uid": np.array(rows["rollout_uids"], dtype=object),
            },
        )

    def test_four_different_length_trajectories_are_normalized_once_each(self):
        rows = build_rows()
        groups = grouping.validate_formal_trajectory_rows(
            **rows, expected_replicas=4
        )
        self.assertEqual(list(groups), [grouping.build_parent_group_uid(7)])
        self.assertEqual(
            [trajectory["row_count"] for trajectory in groups[next(iter(groups))]],
            [3, 5, 4, 6],
        )

        response_mask = torch.ones(len(rows["trajectory_returns"]), 2)
        advantages, _ = core_algos.compute_grpo_trajectory_outcome_advantage(
            trajectory_returns=torch.tensor(rows["trajectory_returns"]),
            eos_mask=response_mask,
            parent_group_uids=np.array(rows["parent_group_uids"], dtype=object),
            trajectory_uids=np.array(rows["trajectory_uids"], dtype=object),
            replica_indices=np.array(rows["replica_indices"], dtype=object),
            expected_replicas=4,
        )

        expected_returns = torch.tensor([1.0, 3.0, 0.0, 2.0])
        expected = (expected_returns - expected_returns.mean()) / (
            expected_returns.std() + 1e-6
        )
        offset = 0
        for replica_index, turn_count in enumerate((3, 5, 4, 6)):
            torch.testing.assert_close(
                advantages[offset : offset + turn_count],
                expected[replica_index].expand(turn_count, 2),
            )
            offset += turn_count

    def test_divergent_later_exact_states_still_get_parent_group_credit(self):
        rows = build_rows(returns=(0.0, 1.0, 2.0, 3.0))
        self.assertEqual(
            len(set(rows["exact_state_uids"][4:])),
            len(rows["exact_state_uids"][4:]),
        )
        response_mask = torch.ones(len(rows["trajectory_returns"]), 1)
        advantages, _ = core_algos.compute_grpo_trajectory_outcome_advantage(
            trajectory_returns=torch.tensor(rows["trajectory_returns"]),
            eos_mask=response_mask,
            parent_group_uids=np.array(rows["parent_group_uids"], dtype=object),
            trajectory_uids=np.array(rows["trajectory_uids"], dtype=object),
            replica_indices=np.array(rows["replica_indices"], dtype=object),
            expected_replicas=4,
        )
        self.assertTrue(torch.all(advantages[:3] < 0))
        self.assertTrue(torch.all(advantages[-6:] > 0))

    def test_padding_does_not_enter_group_statistics_or_receive_advantage(self):
        rows = build_rows()
        for key in (
            "parent_group_uids",
            "exact_state_uids",
            "replica_indices",
            "trajectory_uids",
            "trajectory_returns",
            "immediate_rewards",
            "trajectory_row_uids",
            "trajectory_row_orders",
            "trajectory_terminals",
            "parent_indices",
            "rollout_uids",
        ):
            rows[key].append(rows[key][0])
        rows["valid_mask"].append(False)
        grouping.validate_formal_trajectory_rows(**rows, expected_replicas=4)

        sample_mask = torch.tensor(rows["valid_mask"])
        response_mask = torch.ones(len(sample_mask), 2)
        advantages, _ = core_algos.compute_grpo_trajectory_outcome_advantage(
            trajectory_returns=torch.tensor(rows["trajectory_returns"]),
            eos_mask=response_mask,
            parent_group_uids=np.array(rows["parent_group_uids"], dtype=object),
            trajectory_uids=np.array(rows["trajectory_uids"], dtype=object),
            replica_indices=np.array(rows["replica_indices"], dtype=object),
            sample_mask=sample_mask,
            expected_replicas=4,
        )
        torch.testing.assert_close(advantages[-1], torch.zeros(2))

    def test_truncated_trajectory_keeps_observed_return_without_refill(self):
        rows = build_rows(turn_counts=(3, 5, 4, 2), returns=(1.0, 3.0, 0.0, -0.5))
        groups = grouping.validate_formal_trajectory_rows(
            **rows, expected_replicas=4
        )
        last = groups[grouping.build_parent_group_uid(7)][-1]
        self.assertEqual(last["row_count"], 2)
        self.assertEqual(last["trajectory_return"], -0.5)

    def test_missing_replica_fails_closed(self):
        rows = build_rows(turn_counts=(3, 5, 4), returns=(1.0, 3.0, 0.0))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            grouping.validate_formal_trajectory_rows(
                **rows, expected_replicas=4
            )

    def test_conflicting_return_for_same_trajectory_fails_closed(self):
        rows = build_rows()
        rows["trajectory_returns"][1] = 99.0
        with self.assertRaisesRegex(ValueError, "trajectory return"):
            grouping.validate_formal_trajectory_rows(
                **rows, expected_replicas=4
            )

    def test_metadata_length_mismatch_fails_closed(self):
        rows = build_rows()
        rows["trajectory_uids"].pop()
        with self.assertRaisesRegex(ValueError, "length"):
            grouping.validate_formal_trajectory_rows(
                **rows, expected_replicas=4
            )

    def test_partial_or_missing_dataproto_metadata_fails_closed(self):
        rows = build_rows()
        data = self._data_proto(rows)
        del data.batch[grouping.AGENTMEMORY_IMMEDIATE_REWARD]
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            rollout_context.validate_formal_trajectory_metadata(
                data, expected_replicas=4, require=True
            )

        missing = FakeDataProto(
            batch={"responses": torch.zeros(2, 1)},
            non_tensor_batch={
                "rollout_parent_indices": np.array([0, 0], dtype=object)
            },
        )
        with self.assertRaisesRegex(RuntimeError, "missing trajectory metadata"):
            rollout_context.validate_formal_trajectory_metadata(
                missing, expected_replicas=2, require=True
            )

    def test_complete_dataproto_metadata_validates_after_padding(self):
        rows = build_rows()
        data = self._data_proto(rows)
        data.batch["ppo_valid_sample_mask"] = torch.ones(
            len(rows["valid_mask"]), dtype=torch.bool
        )
        groups = rollout_context.validate_formal_trajectory_metadata(
            data, expected_replicas=4, require=True
        )
        self.assertEqual(len(groups), 1)

    def test_trainer_uses_raw_trajectory_return_and_debugs_broadcast_credit(self):
        rows = build_rows()
        data = self._data_proto(rows)
        row_count = len(rows["valid_mask"])
        data.batch.update(
            {
                "response_mask": torch.ones(row_count, 2),
                "token_level_rewards": torch.zeros(row_count, 2),
                "scores": torch.zeros(row_count, 2),
                "old_log_probs": torch.zeros(row_count, 2),
                core_algos.PPO_VALID_SAMPLE_MASK: torch.ones(
                    row_count, dtype=torch.bool
                ),
            }
        )
        data.non_tensor_batch["uid"] = np.array(
            rows["rollout_uids"], dtype=object
        )
        compute_advantage = ray_trainer_functions["compute_advantage"]
        compute_advantage(data, adv_estimator="grpo", num_repeat=4)

        self.assertTrue(torch.all(data.batch["advantages"][:3] < 0))
        self.assertTrue(torch.all(data.batch["advantages"][3:8] > 0))
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(
                trainer=SimpleNamespace(
                    default_local_dir=str(Path(tmpdir) / "checkpoints")
                )
            )
            with mock.patch.dict(os.environ, {"AGENTMEMORY_BATCH_DEBUG": "1"}):
                ray_trainer_functions["_agentmemory_dump_ppo_batch_debug"](
                    batch=data,
                    config=config,
                    global_steps=1,
                    stage="post_adv",
                )
            debug_path = Path(tmpdir) / "diagnostics/ppo_batch_step1_post_adv.json"
            payload = json.loads(debug_path.read_text())

        self.assertEqual(len(payload["agentmemory_parent_groups"]), 1)
        parent = payload["agentmemory_parent_groups"][0]
        self.assertEqual(parent["unique_replicas"], 4)
        self.assertEqual(len(parent["trajectories"]), 4)
        for trajectory in parent["trajectories"]:
            self.assertEqual(
                trajectory["action_row_token_mean_advantage_min"],
                trajectory["action_row_token_mean_advantage_max"],
            )
        first_row = payload["rows"][0]
        for key in (
            grouping.AGENTMEMORY_PARENT_GROUP_UID,
            grouping.AGENTMEMORY_EXACT_STATE_UID,
            grouping.AGENTMEMORY_REPLICA_INDEX,
            grouping.AGENTMEMORY_TRAJECTORY_UID,
            grouping.AGENTMEMORY_TRAJECTORY_RETURN,
            grouping.AGENTMEMORY_IMMEDIATE_REWARD,
            grouping.AGENTMEMORY_TRAJECTORY_ROW_UID,
            grouping.AGENTMEMORY_TRAJECTORY_ROW_ORDER,
            grouping.AGENTMEMORY_TRAJECTORY_TERMINAL,
        ):
            self.assertIn(key, first_row)

    def test_parent_group_collision_across_source_parents_fails_closed(self):
        rows = build_rows()
        rows["parent_indices"][-1] = 8
        with self.assertRaisesRegex(ValueError, "source parent"):
            grouping.validate_formal_trajectory_rows(
                **rows, expected_replicas=4
            )

    def test_ppo_suffix_target_preserves_immediate_reward_evidence(self):
        immediate_rewards = [0.0, 0.0, 1.0]
        suffix_scores = grouping.compute_suffix_credit_scores(
            immediate_rewards, [1, 2, 3]
        )
        self.assertEqual(suffix_scores, [1.0, 1.0, 1.0])

        rows = build_rows(turn_counts=(3,), returns=(1.0,))
        groups = grouping.validate_formal_trajectory_rows(
            **rows, expected_replicas=1
        )
        trajectory = groups[grouping.build_parent_group_uid(7)][0]
        self.assertEqual(trajectory["immediate_rewards"], immediate_rewards)


if __name__ == "__main__":
    unittest.main()
