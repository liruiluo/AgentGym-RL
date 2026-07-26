import math
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.agent_trainer.ppo import core_algos
from verl.utils import torch_functional as verl_F
from verl.workers.ppo_token_normalization import (
    GLOBAL_TOKEN_MEAN,
    PER_ACTION_TOKEN_MEAN,
    distributed_sum,
    mask_padding_rows,
    per_action_token_mean,
    scale_action_mean_loss,
    valid_response_action_count,
    validate_policy_loss_aggregation,
)


ROOT = Path(__file__).resolve().parents[2]


def _ppo_elements(parameter, features, advantages, cliprange=0.2):
    return core_algos.compute_policy_loss_elements(
        old_log_prob=torch.zeros_like(features),
        log_prob=parameter * features,
        advantages=advantages,
        cliprange=cliprange,
    )[0]


def _distributed_action_mean_worker(rank, world_size, rendezvous_path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        features = torch.tensor(
            [
                [0.10, -0.20, 0.30],
                [0.40, 0.00, 0.00],
                [-0.70, 0.80, 0.00],
                [0.20, -0.10, 0.00],
            ]
        )
        advantages = torch.tensor(
            [
                [1.00, 0.50, -0.50],
                [-1.00, 0.00, 0.00],
                [0.30, -0.40, 0.00],
                [0.70, -0.20, 0.00],
            ]
        )
        response_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        valid_samples = torch.tensor([True, True, False, True])
        real_mask = mask_padding_rows(response_mask, valid_samples)

        reference_parameter = torch.tensor(0.25, requires_grad=True)
        reference_loss = per_action_token_mean(
            _ppo_elements(reference_parameter, features, advantages),
            real_mask,
        )
        reference_loss.backward()

        local_indices = [0, 1] if rank == 0 else [2, 3]
        global_actions = distributed_sum(
            valid_response_action_count(real_mask[local_indices])
        )
        actual_parameter = torch.tensor(0.25, requires_grad=True)
        for row_index in local_indices:
            row = slice(row_index, row_index + 1)
            row_loss = per_action_token_mean(
                _ppo_elements(
                    actual_parameter,
                    features[row],
                    advantages[row],
                ),
                real_mask[row],
            )
            scaled = scale_action_mean_loss(
                row_loss,
                valid_response_action_count(real_mask[row]),
                global_actions,
            )
            scaled.backward()

        dist.all_reduce(actual_parameter.grad)
        actual_parameter.grad /= world_size
        torch.testing.assert_close(
            actual_parameter.grad,
            reference_parameter.grad,
        )
        if not torch.isfinite(actual_parameter.grad):
            raise AssertionError("distributed action-mean gradient is not finite")
    finally:
        dist.destroy_process_group()


class ActionNormalizedTokenPPOTest(unittest.TestCase):
    def test_equal_length_actions_match_token_mean_loss_and_gradient(self):
        features = torch.tensor(
            [[0.10, -0.20], [0.30, 0.40], [-0.50, 0.20]]
        )
        advantages = torch.tensor(
            [[1.00, 0.50], [-0.20, 0.70], [0.30, -0.40]]
        )
        mask = torch.ones_like(features)

        token_parameter = torch.tensor(0.15, requires_grad=True)
        token_loss = verl_F.masked_mean(
            _ppo_elements(token_parameter, features, advantages), mask
        )
        token_loss.backward()

        action_parameter = torch.tensor(0.15, requires_grad=True)
        action_loss = per_action_token_mean(
            _ppo_elements(action_parameter, features, advantages), mask
        )
        action_loss.backward()

        torch.testing.assert_close(action_loss, token_loss.detach())
        torch.testing.assert_close(action_parameter.grad, token_parameter.grad)

    def test_unequal_length_actions_use_different_declared_weights(self):
        values = torch.tensor(
            [[1.0, 3.0, 5.0], [9.0, 99.0, 99.0]],
            requires_grad=True,
        )
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])

        token_mean = verl_F.masked_mean(values, mask)
        action_mean = per_action_token_mean(values, mask)

        torch.testing.assert_close(token_mean, torch.tensor(4.5))
        torch.testing.assert_close(action_mean, torch.tensor(6.0))
        token_gradient = torch.autograd.grad(
            token_mean, values, retain_graph=True
        )[0]
        action_gradient = torch.autograd.grad(action_mean, values)[0]
        torch.testing.assert_close(
            token_gradient,
            torch.tensor([[0.25, 0.25, 0.25], [0.25, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            action_gradient,
            torch.tensor(
                [[1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], [0.5, 0.0, 0.0]]
            ),
        )

    def test_padding_invalid_samples_and_empty_rows_do_not_enter_denominator(self):
        values = torch.tensor(
            [[2.0, 4.0], [100.0, 200.0], [7.0, 9.0], [5.0, 6.0]],
            requires_grad=True,
        )
        response_mask = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
        )
        valid_samples = torch.tensor([True, False, True, True])
        mask = mask_padding_rows(response_mask, valid_samples)

        self.assertEqual(valid_response_action_count(mask).item(), 2.0)
        loss = per_action_token_mean(values, mask)
        torch.testing.assert_close(loss, torch.tensor(4.0))
        loss.backward()
        torch.testing.assert_close(
            values.grad,
            torch.tensor(
                [[0.25, 0.25], [0.0, 0.0], [0.0, 0.0], [0.5, 0.0]]
            ),
        )

    def test_ppo_ratio_clip_and_advantage_sign_are_unchanged(self):
        old_log_prob = torch.zeros(1, 4)
        log_prob = torch.tensor(
            [[math.log(1.5), math.log(0.5), 0.0, float("nan")]]
        )
        advantages = torch.tensor([[1.0, -1.0, 0.25, 99.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        losses, clipped, token_kl = core_algos.compute_policy_loss_elements(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            cliprange=0.2,
        )
        self.assertLess(losses[0, 0].item(), 0.0)
        self.assertGreater(losses[0, 1].item(), 0.0)
        torch.testing.assert_close(losses[0, :3], torch.tensor([-1.2, 0.8, -0.25]))
        torch.testing.assert_close(clipped[0, :3], torch.tensor([1.0, 1.0, 0.0]))
        self.assertTrue(torch.isfinite(losses.masked_select(mask.bool())).all())
        self.assertTrue(torch.isfinite(token_kl.masked_select(mask.bool())).all())

        reduced = core_algos.compute_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=mask,
            cliprange=0.2,
        )
        torch.testing.assert_close(
            reduced[0], verl_F.masked_mean(losses, mask)
        )
        torch.testing.assert_close(
            reduced[1], verl_F.masked_mean(clipped, mask)
        )
        torch.testing.assert_close(
            reduced[2], verl_F.masked_mean(token_kl, mask)
        )

    def test_default_mode_remains_global_token_mean(self):
        self.assertEqual(
            validate_policy_loss_aggregation(None), GLOBAL_TOKEN_MEAN
        )
        self.assertEqual(
            validate_policy_loss_aggregation(GLOBAL_TOKEN_MEAN),
            GLOBAL_TOKEN_MEAN,
        )
        self.assertEqual(
            validate_policy_loss_aggregation(PER_ACTION_TOKEN_MEAN),
            PER_ACTION_TOKEN_MEAN,
        )
        with self.assertRaisesRegex(ValueError, "loss_aggregation"):
            validate_policy_loss_aggregation("rank_mean")

        config = (
            ROOT / "verl/agent_trainer/config/ppo_trainer.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("loss_aggregation: global_token_mean", config)

    def test_multi_rank_action_mean_matches_single_process_gradient(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is unavailable")
        with tempfile.TemporaryDirectory() as temporary_directory:
            rendezvous_path = os.path.join(
                temporary_directory, "gloo-rendezvous"
            )
            mp.spawn(
                _distributed_action_mean_worker,
                args=(2, rendezvous_path),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
