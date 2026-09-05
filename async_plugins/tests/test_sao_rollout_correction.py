from __future__ import annotations

import math
import types
import unittest

import torch
from verl.trainer.ppo.core_algos import compute_policy_loss_bypass_mode
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights


class TestSAODirectDoubleSidedIS(unittest.TestCase):
    def test_double_sided_token_weights_use_inclusive_bounds(self):
        ratios = torch.tensor([[0.199, 0.2, 1.0, 4.0, 4.001]], dtype=torch.float64)
        weights, metrics = compute_rollout_correction_weights(
            log_ratio=ratios.log(),
            response_mask=torch.ones_like(ratios),
            rollout_is="token",
            rollout_is_threshold="0.2_4.0",
            rollout_is_batch_normalize=False,
        )
        torch.testing.assert_close(
            weights,
            torch.tensor([[0.0, 0.2, 1.0, 4.0, 0.0]], dtype=torch.float64),
        )
        self.assertAlmostEqual(metrics["rollout_is_oob_ratio"], 0.4)

    def test_out_of_bounds_tokens_have_zero_gradient_for_both_advantage_signs(self):
        ratios = torch.tensor([[0.1, 0.5, 2.0, 5.0]], dtype=torch.float64)
        rollout_log_prob = torch.full_like(ratios, -10.0)
        config = types.SimpleNamespace(
            policy_loss={
                "rollout_correction": {
                    "loss_type": "reinforce",
                    "rollout_is": "token",
                    "rollout_is_threshold": "0.2_4.0",
                    "rollout_is_batch_normalize": False,
                    "rollout_rs": None,
                    "rollout_rs_threshold": None,
                }
            },
            global_batch_info={},
        )

        nonzero_masks = []
        for sign in (1.0, -1.0):
            current_log_prob = (rollout_log_prob + ratios.log()).clone().requires_grad_()
            loss, _ = compute_policy_loss_bypass_mode(
                old_log_prob=rollout_log_prob,
                log_prob=current_log_prob,
                advantages=torch.full_like(ratios, sign),
                response_mask=torch.ones_like(ratios),
                loss_agg_mode="token-mean",
                config=config,
            )
            loss.backward()
            nonzero_masks.append(current_log_prob.grad.ne(0))

        expected = torch.tensor([[False, True, True, False]])
        self.assertTrue(torch.equal(nonzero_masks[0], expected))
        self.assertTrue(torch.equal(nonzero_masks[1], expected))


if __name__ == "__main__":
    unittest.main()
