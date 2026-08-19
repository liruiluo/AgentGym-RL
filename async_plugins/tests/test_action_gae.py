from __future__ import annotations

import unittest

import numpy as np
import torch
from agentmemorygym_verl.action_gae import compute_amg_action_gae
from verl.trainer.ppo.core_algos import get_adv_estimator_fn


class TestAMGActionGAE(unittest.TestCase):
    def test_registers_the_canonical_action_axis_estimator_name(self):
        self.assertIs(
            get_adv_estimator_fn("amg_action_axis_gae"),
            compute_amg_action_gae,
        )

    def _fixture(self):
        rewards = torch.zeros((4, 3), dtype=torch.float32)
        rewards[0, 1] = 1.0
        rewards[1, 0] = 2.0
        rewards[2, 0] = 3.0
        values = torch.tensor(
            [
                [0.5, 99.0, 101.0],
                [0.25, 77.0, 0.0],
                [1.0, 55.0, 0.0],
                [123.0, 456.0, 789.0],
            ],
            dtype=torch.float32,
        )
        response_mask = torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=torch.long,
        )
        batch = {
            "token_level_rewards": rewards,
            "values": values,
            "response_mask": response_mask,
            "rollout_log_probs": torch.tensor(
                [
                    [-0.1, -0.2, -0.3],
                    [-0.4, -0.5, 0.0],
                    [-0.6, 0.0, 0.0],
                    [float("nan"), float("nan"), float("nan")],
                ],
                dtype=torch.float32,
            ),
        }
        batch["old_log_probs"] = batch["rollout_log_probs"]
        non_tensor_batch = {
            "trajectory_uid": np.array(["a", "a", "b", "pad"], dtype=object),
            "trajectory_row_uid": np.array(
                ["a-0", "a-1", "b-0", "pad-0"], dtype=object
            ),
            "trajectory_row_order": np.array([0, 1, 0, 0], dtype=object),
            "trajectory_terminal": np.array([False, True, True, True], dtype=object),
            "rollout_done_flag": np.array([False, True, True, True], dtype=object),
            "immediate_reward": np.array([1.0, 2.0, 3.0, 0.0], dtype=object),
            "is_padding": np.array([False, False, False, True], dtype=object),
        }
        config = {
            "gamma": 0.9,
            "lam": 0.8,
            "amg_reward_tolerance": 1e-6,
            "amg_advantage_normalization": "none",
        }
        return batch, non_tensor_batch, config

    def test_uses_causal_first_response_value_and_broadcasts_over_action_tokens(self):
        batch, non_tensor_batch, config = self._fixture()
        advantages, returns = compute_amg_action_gae(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            config=config,
        )

        # Trajectory a: A1 = 2 - .25 = 1.75;
        # A0 = 1 + .9*.25 - .5 + .9*.8*1.75 = 1.985.
        expected_advantages = torch.tensor(
            [
                [1.985, 1.985, 1.985],
                [1.75, 1.75, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        expected_returns = torch.tensor(
            [
                [2.485, 2.485, 2.485],
                [2.0, 2.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(advantages, expected_advantages)
        torch.testing.assert_close(returns, expected_returns)

    def test_upstream_masked_whitening_zero_centers_only_real_policy_tokens(self):
        batch, non_tensor_batch, config = self._fixture()
        config["amg_advantage_normalization"] = "upstream_masked_whiten"
        # Exercise the synthetic-row exclusion directly: the padding row looks
        # like a finite sampled response before ``is_padding`` removes it from
        # the whitening population.
        batch["response_mask"][3] = torch.tensor([1, 1, 0], dtype=torch.long)
        batch["rollout_log_probs"][3] = torch.tensor(
            [-0.7, -0.8, 0.0], dtype=torch.float32
        )
        batch["old_log_probs"] = batch["rollout_log_probs"].clone()

        advantages, returns = compute_amg_action_gae(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            config=config,
        )

        self.assertEqual(int(batch["response_mask"][3].sum().item()), 2)
        real_policy_mask = batch["response_mask"].to(dtype=torch.bool).clone()
        real_policy_mask[3] = False
        selected = advantages[real_policy_mask]
        self.assertAlmostEqual(float(selected.mean().item()), 0.0, places=6)
        self.assertAlmostEqual(
            float(selected.var(unbiased=True).item()), 1.0, delta=1e-5
        )
        self.assertEqual(float(advantages[3].abs().sum().item()), 0.0)
        self.assertEqual(float(advantages[1, 2].item()), 0.0)
        self.assertEqual(float(advantages[2, 1:].abs().sum().item()), 0.0)
        self.assertEqual(float(advantages[0].var(unbiased=False).item()), 0.0)
        self.assertEqual(float(advantages[1, :2].var(unbiased=False).item()), 0.0)

        expected_returns = torch.tensor(
            [
                [2.485, 2.485, 2.485],
                [2.0, 2.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(returns, expected_returns)

    def test_rejects_reward_packing_that_does_not_conserve_action_reward(self):
        batch, non_tensor_batch, config = self._fixture()
        batch["token_level_rewards"][0, 1] = 0.75
        with self.assertRaisesRegex(ValueError, "packed token reward differs"):
            compute_amg_action_gae(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                config=config,
            )

    def test_rejects_incomplete_or_duplicated_action_order(self):
        batch, non_tensor_batch, config = self._fixture()
        non_tensor_batch["trajectory_row_order"][1] = 2
        with self.assertRaisesRegex(
            ValueError, "row order is incomplete or duplicated"
        ):
            compute_amg_action_gae(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                config=config,
            )

    def test_rejects_recomputed_old_logprob_on_any_policy_token(self):
        batch, non_tensor_batch, config = self._fixture()
        batch["old_log_probs"] = batch["rollout_log_probs"].clone()
        batch["old_log_probs"][1, 1] += 1e-7
        with self.assertRaisesRegex(ValueError, "exactly the rollout behavior"):
            compute_amg_action_gae(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                config=config,
            )

    def test_requires_both_behavior_and_old_logprob_tensors(self):
        batch, non_tensor_batch, config = self._fixture()
        del batch["old_log_probs"]
        with self.assertRaisesRegex(ValueError, "old_log_probs"):
            compute_amg_action_gae(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                config=config,
            )


if __name__ == "__main__":
    unittest.main()
