import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.protocol import pad_dataproto_to_divisor
from verl.utils.agentgym.rollout_context import align_batch_to_rollout
from verl.workers.ppo_token_normalization import (
    mask_padding_rows,
    scale_token_mean_loss,
    valid_response_token_count,
)

_GROUPING_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl/workers/rollout/agent_vllm_rollout/agentmemory_grouping.py"
)
_GROUPING_SPEC = importlib.util.spec_from_file_location(
    "agentmemory_grouping_for_test", _GROUPING_PATH
)
assert _GROUPING_SPEC is not None and _GROUPING_SPEC.loader is not None
_GROUPING_MODULE = importlib.util.module_from_spec(_GROUPING_SPEC)
_GROUPING_SPEC.loader.exec_module(_GROUPING_MODULE)
resolve_rollout_parent_index = _GROUPING_MODULE.resolve_rollout_parent_index


class GrpoAlignmentTest(unittest.TestCase):
    def test_global_source_indices_survive_dp_chunk_and_align(self):
        source = DataProto.from_dict(
            tensors={"source_row": torch.arange(4).unsqueeze(-1)},
            non_tensors={"item_id": np.array(["a", "b", "c", "d"], dtype=object)},
        )
        source.non_tensor_batch["rollout_source_parent_indices"] = np.arange(
            len(source), dtype=object
        )

        rank_batches = source.chunk(chunks=2)
        np.testing.assert_array_equal(
            rank_batches[0].non_tensor_batch["rollout_source_parent_indices"],
            np.array([0, 1]),
        )
        np.testing.assert_array_equal(
            rank_batches[1].non_tensor_batch["rollout_source_parent_indices"],
            np.array([2, 3]),
        )

        # Simulate n=2 generation on each rank. Workers return the carried
        # global values, not rank-local 0/1 parent positions.
        returned_parent_indices = np.concatenate(
            [
                np.array(
                    [
                        resolve_rollout_parent_index(
                            local_index,
                            source_parent_indices=rank_batch.non_tensor_batch[
                                "rollout_source_parent_indices"
                            ],
                        )
                        for local_index in range(len(rank_batch))
                        for _ in range(2)
                    ],
                    dtype=object,
                )
                for rank_batch in rank_batches
            ]
        )
        rollout_output = DataProto.from_dict(
            tensors={"responses": torch.zeros(len(returned_parent_indices), 1)},
            non_tensors={"rollout_parent_indices": returned_parent_indices},
        )

        aligned = align_batch_to_rollout(source, rollout_output, repeat_times=2)
        torch.testing.assert_close(
            aligned.batch["source_row"].squeeze(-1),
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        )
        np.testing.assert_array_equal(
            aligned.non_tensor_batch["item_id"],
            np.array(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object),
        )

    def test_singleton_group_has_zero_advantage(self):
        rewards = torch.tensor([[3.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        response_mask = torch.ones_like(rewards)
        group_ids = np.array(["singleton", "pair", "pair"], dtype=object)

        advantages, _ = core_algos.compute_grpo_outcome_advantage(
            rewards, response_mask, group_ids
        )

        torch.testing.assert_close(advantages[0], torch.zeros_like(advantages[0]))
        self.assertLess(advantages[1, 0].item(), 0.0)
        self.assertGreater(advantages[2, 0].item(), 0.0)

    def test_padding_does_not_change_grpo_groups(self):
        rewards = torch.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
        response_mask = torch.ones_like(rewards)
        group_ids = np.array(["pair", "pair", "singleton"], dtype=object)
        expected, _ = core_algos.compute_grpo_outcome_advantage(
            rewards, response_mask, group_ids
        )

        data = DataProto.from_dict(
            tensors={
                "rewards": rewards,
                "response_mask": response_mask,
                core_algos.PPO_VALID_SAMPLE_MASK: torch.ones(3, dtype=torch.bool),
            },
            non_tensors={"uid": group_ids},
        )
        padded, pad_size = pad_dataproto_to_divisor(data, size_divisor=5)
        self.assertEqual(pad_size, 2)
        padded.batch[core_algos.PPO_VALID_SAMPLE_MASK][-pad_size:] = False
        actual, _ = core_algos.compute_grpo_outcome_advantage(
            padded.batch["rewards"],
            padded.batch["response_mask"],
            padded.non_tensor_batch["uid"],
            sample_mask=padded.batch[core_algos.PPO_VALID_SAMPLE_MASK],
        )

        torch.testing.assert_close(actual[:3], expected)
        torch.testing.assert_close(actual[3:], torch.zeros_like(actual[3:]))

    def test_padding_preserves_global_token_mean_gradient(self):
        features = torch.tensor(
            [[0.1, -0.2, 0.3], [0.2, -0.1, 0.4], [-0.3, 0.2, 0.1]]
        )
        advantages = torch.tensor(
            [[1.0, 0.5, -0.5], [-1.0, 0.2, 0.7], [0.3, -0.4, 1.0]]
        )
        response_mask = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        )

        reference_parameter = torch.tensor(0.25, requires_grad=True)
        reference_loss, _, _ = core_algos.compute_policy_loss(
            torch.zeros_like(features),
            reference_parameter * features,
            advantages,
            response_mask,
            cliprange=0.2,
        )
        reference_loss.backward()

        padded_features = torch.cat([features, features[:1]], dim=0)
        padded_advantages = torch.cat([advantages, advantages[:1]], dim=0)
        padded_response_mask = torch.cat([response_mask, response_mask[:1]], dim=0)
        valid_samples = torch.tensor([True, True, True, False])
        masked_response = mask_padding_rows(padded_response_mask, valid_samples)
        global_tokens = valid_response_token_count(masked_response)

        actual_parameter = torch.tensor(0.25, requires_grad=True)
        rank_losses = []
        for rank_indices in ([0, 1], [2, 3]):
            rank_loss = actual_parameter * 0.0
            for row_index in rank_indices:
                row = slice(row_index, row_index + 1)
                local_loss, _, _ = core_algos.compute_policy_loss(
                    torch.zeros_like(padded_features[row]),
                    actual_parameter * padded_features[row],
                    padded_advantages[row],
                    masked_response[row],
                    cliprange=0.2,
                )
                rank_loss = rank_loss + scale_token_mean_loss(
                    local_loss,
                    valid_response_token_count(masked_response[row]),
                    global_tokens,
                ) * 2.0
            rank_losses.append(rank_loss)
        simulated_fsdp_loss = sum(rank_losses) / 2.0
        simulated_fsdp_loss.backward()

        torch.testing.assert_close(simulated_fsdp_loss, reference_loss.detach())
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad)

    def test_actor_update_policy_declares_process_groups(self):
        actor_path = (
            Path(__file__).resolve().parents[2]
            / "verl/workers/agent_actor/dp_actor.py"
        )
        module = ast.parse(actor_path.read_text())
        update_policy = next(
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "update_policy"
        )
        assigned_names = {
            target.id
            for node in ast.walk(update_policy)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

        self.assertIn("loss_group", assigned_names)
        self.assertIn("metric_group", assigned_names)


if __name__ == "__main__":
    unittest.main()
