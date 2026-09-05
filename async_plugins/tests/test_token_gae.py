from __future__ import annotations

import unittest

import numpy as np
import torch
from agentmemorygym_verl.token_gae import (
    compute_amg_sao_token_gae,
    length_adaptive_policy_lambda,
)
from verl.trainer.ppo.core_algos import get_adv_estimator_fn


class TestAMGSAOTokenGAE(unittest.TestCase):
    def _two_action_fixture(self):
        rewards = torch.zeros((3, 3), dtype=torch.float32)
        rewards[0, 1] = 0.25
        rewards[1, 1] = 1.0
        values = torch.tensor(
            [
                [0.1, 0.4, 99.0],
                [0.8, 0.2, 99.0],
                [float("nan"), float("nan"), float("nan")],
            ],
            dtype=torch.float32,
        )
        response_mask = torch.tensor(
            [
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
            ],
            dtype=torch.long,
        )
        rollout_log_probs = torch.tensor(
            [
                [-1.0, -1.1, 0.0],
                [-1.2, -1.3, 0.0],
                [float("nan"), float("nan"), float("nan")],
            ],
            dtype=torch.float32,
        )
        batch = {
            "token_level_rewards": rewards,
            "values": values,
            "response_mask": response_mask,
            "rollout_log_probs": rollout_log_probs,
            "old_log_probs": rollout_log_probs.clone(),
        }
        non_tensor_batch = {
            "trajectory_uid": np.array(["episode-a", "episode-a", "pad"], dtype=object),
            "trajectory_row_uid": np.array(["a-0", "a-1", "pad-0"], dtype=object),
            "trajectory_row_order": np.array([0, 1, 0], dtype=object),
            "trajectory_terminal": np.array([False, True, True], dtype=object),
            "rollout_done_flag": np.array([False, True, True], dtype=object),
            "immediate_reward": np.array([0.25, 1.0, 0.0], dtype=object),
            "outcome": np.array(["continue", "success", "padding"], dtype=object),
            "declared_max_rounds": np.array([30, 30, 0], dtype=object),
            "termination_kind": np.array(
                ["in_progress", "environment_done", "padding"], dtype=object
            ),
            "horizon_finalizer_receipt": np.array(
                [
                    "not_applicable:nonterminal",
                    "not_invoked:environment_done",
                    "padding",
                ],
                dtype=object,
            ),
            "is_padding": np.array([False, False, True], dtype=object),
            # A context replacement and a policy publication between actions
            # must not split or reject the policy-token credit chain.
            "context_transition": np.array(
                [
                    {"operation": "replace_messages"},
                    {"operation": "append_observation"},
                    None,
                ],
                dtype=object,
            ),
            "min_global_steps": np.array([7, 9, -1], dtype=object),
            "max_global_steps": np.array([7, 9, -1], dtype=object),
        }
        config = {
            "gamma": 1.0,
            "amg_policy_lambda_mode": "length_adaptive",
            "amg_policy_lambda_scale": 1.5,
            "amg_critic_lambda": 1.0,
            "amg_reward_tolerance": 1e-6,
            "amg_advantage_normalization": "none",
        }
        return batch, non_tensor_batch, config

    def test_registers_token_estimator(self):
        self.assertIs(
            get_adv_estimator_fn("amg_sao_token_gae"),
            compute_amg_sao_token_gae,
        )

    def test_length_adaptive_lambda_uses_complete_episode_policy_tokens(self):
        self.assertAlmostEqual(length_adaptive_policy_lambda(1, scale=1.5), 1.0 / 3.0)
        self.assertAlmostEqual(length_adaptive_policy_lambda(4, scale=1.5), 5.0 / 6.0)
        with self.assertRaisesRegex(ValueError, "token_count"):
            length_adaptive_policy_lambda(0, scale=1.5)

    def test_token_gae_bridges_action_and_context_boundaries(self):
        batch, non_tensor_batch, config = self._two_action_fixture()
        advantages, returns = compute_amg_sao_token_gae(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            config=config,
        )

        # Flattened episode values are [.1, .4, .8, .2], rewards are
        # [0, .25, 0, 1], and lambda_policy=1-1/(1.5*4)=5/6.
        expected_advantages = torch.tensor(
            [
                [0.88796294, 0.70555556, 0.0],
                [0.06666667, 0.8, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        # lambda_critic=1 yields undiscounted cumulative future reward for
        # gamma=1, including reward across the action/context boundary.
        expected_returns = torch.tensor(
            [
                [1.25, 1.25, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(advantages, expected_advantages)
        torch.testing.assert_close(returns, expected_returns)

    def test_single_action_is_still_token_level(self):
        rewards = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        values = torch.zeros_like(rewards)
        rollout_log_probs = torch.full_like(rewards, -1.0)
        batch = {
            "token_level_rewards": rewards,
            "values": values,
            "response_mask": torch.ones_like(rewards, dtype=torch.long),
            "rollout_log_probs": rollout_log_probs,
            "old_log_probs": rollout_log_probs.clone(),
        }
        metadata = {
            "trajectory_uid": np.array(["single"], dtype=object),
            "trajectory_row_uid": np.array(["single-0"], dtype=object),
            "trajectory_row_order": np.array([0], dtype=object),
            "trajectory_terminal": np.array([True], dtype=object),
            "rollout_done_flag": np.array([True], dtype=object),
            "immediate_reward": np.array([1.0], dtype=object),
            "outcome": np.array(["success"], dtype=object),
            "declared_max_rounds": np.array([30], dtype=object),
            "termination_kind": np.array(["environment_done"], dtype=object),
            "horizon_finalizer_receipt": np.array(
                ["not_invoked:environment_done"], dtype=object
            ),
            "is_padding": np.array([False], dtype=object),
            "min_global_steps": np.array([4], dtype=object),
            "max_global_steps": np.array([4], dtype=object),
        }
        config = {
            "gamma": 1.0,
            "amg_policy_lambda_mode": "length_adaptive",
            "amg_policy_lambda_scale": 1.5,
            "amg_critic_lambda": 1.0,
            "amg_advantage_normalization": "none",
        }
        advantages, returns = compute_amg_sao_token_gae(
            batch=batch,
            non_tensor_batch=metadata,
            config=config,
        )
        torch.testing.assert_close(
            advantages,
            torch.tensor([[49.0 / 81.0, 7.0 / 9.0, 1.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(returns, torch.ones_like(returns))

    def test_accepts_attested_complete_max_rounds_horizon(self):
        batch, metadata, config = self._two_action_fixture()
        metadata["rollout_done_flag"][1] = False
        metadata["outcome"][1] = "max_rounds"
        metadata["declared_max_rounds"][:2] = 2
        metadata["termination_kind"][1] = "max_rounds"
        metadata["horizon_finalizer_receipt"][1] = (
            "no_terminal_transition:returned_none"
        )

        advantages, returns = compute_amg_sao_token_gae(
            batch=batch,
            non_tensor_batch=metadata,
            config=config,
        )

        self.assertEqual(tuple(advantages.shape), (3, 3))
        self.assertEqual(tuple(returns.shape), (3, 3))

    def test_rejects_non_done_continue_as_terminal(self):
        batch, metadata, config = self._two_action_fixture()
        metadata["rollout_done_flag"][1] = False
        metadata["termination_kind"][1] = "max_rounds"
        metadata["horizon_finalizer_receipt"][1] = "no_terminal_transition:no_hook"
        with self.assertRaisesRegex(ValueError, "exact max_rounds horizon attestation"):
            compute_amg_sao_token_gae(
                batch=batch,
                non_tensor_batch=metadata,
                config=config,
            )

    def test_rejects_truncated_max_rounds_claim(self):
        batch, metadata, config = self._two_action_fixture()
        metadata["rollout_done_flag"][1] = False
        metadata["outcome"][1] = "max_rounds"
        metadata["termination_kind"][1] = "max_rounds"
        metadata["horizon_finalizer_receipt"][1] = "no_terminal_transition:no_hook"
        with self.assertRaisesRegex(ValueError, "exact max_rounds horizon attestation"):
            compute_amg_sao_token_gae(
                batch=batch,
                non_tensor_batch=metadata,
                config=config,
            )

    def test_rejects_max_rounds_without_finalizer_receipt(self):
        batch, metadata, config = self._two_action_fixture()
        metadata["rollout_done_flag"][1] = False
        metadata["outcome"][1] = "max_rounds"
        metadata["declared_max_rounds"][:2] = 2
        metadata["termination_kind"][1] = "max_rounds"
        metadata["horizon_finalizer_receipt"][1] = ""
        with self.assertRaisesRegex(TypeError, "horizon_finalizer_receipt"):
            compute_amg_sao_token_gae(
                batch=batch,
                non_tensor_batch=metadata,
                config=config,
            )

    def test_whitens_once_over_all_real_policy_tokens(self):
        batch, non_tensor_batch, config = self._two_action_fixture()
        config["amg_advantage_normalization"] = "upstream_masked_whiten"
        advantages, returns = compute_amg_sao_token_gae(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            config=config,
        )
        real_mask = batch["response_mask"].bool()
        real_mask[2] = False
        selected = advantages[real_mask]
        self.assertAlmostEqual(float(selected.mean().item()), 0.0, places=6)
        self.assertAlmostEqual(float(selected.var(unbiased=True).item()), 1.0, places=5)
        self.assertEqual(float(advantages[2].abs().sum().item()), 0.0)
        torch.testing.assert_close(
            returns,
            torch.tensor(
                [[1.25, 1.25, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
        )

    def test_rejects_reward_moved_off_the_last_action_token(self):
        batch, non_tensor_batch, config = self._two_action_fixture()
        batch["token_level_rewards"][0, 0] = 0.25
        batch["token_level_rewards"][0, 1] = 0.0
        with self.assertRaisesRegex(ValueError, "final valid policy token"):
            compute_amg_sao_token_gae(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                config=config,
            )

    def test_rejects_reward_that_is_not_conserved(self):
        batch, non_tensor_batch, config = self._two_action_fixture()
        batch["token_level_rewards"][1, 1] = 0.75
        with self.assertRaisesRegex(ValueError, "immediate action reward"):
            compute_amg_sao_token_gae(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                config=config,
            )

    def test_rejects_recomputed_old_logprob(self):
        batch, non_tensor_batch, config = self._two_action_fixture()
        batch["old_log_probs"][1, 0] += 1e-7
        with self.assertRaisesRegex(ValueError, "rollout behavior"):
            compute_amg_sao_token_gae(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                config=config,
            )

    def test_rejects_incomplete_action_order(self):
        batch, non_tensor_batch, config = self._two_action_fixture()
        non_tensor_batch["trajectory_row_order"][1] = 2
        with self.assertRaisesRegex(ValueError, "row order"):
            compute_amg_sao_token_gae(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
                config=config,
            )

    def test_uid_fields_reject_non_string_values_before_grouping(self):
        invalid_values = (None, np.nan, 1, 1.0, "")
        for field in ("trajectory_uid", "trajectory_row_uid"):
            for invalid in invalid_values:
                with self.subTest(field=field, value=repr(invalid)):
                    batch, metadata, config = self._two_action_fixture()
                    metadata[field][0] = invalid
                    with self.assertRaisesRegex(TypeError, field):
                        compute_amg_sao_token_gae(
                            batch=batch,
                            non_tensor_batch=metadata,
                            config=config,
                        )

    def test_uid_fields_reject_numeric_string_collision_instead_of_coercing(self):
        for field in ("trajectory_uid", "trajectory_row_uid"):
            with self.subTest(field=field):
                batch, metadata, config = self._two_action_fixture()
                metadata[field][0] = 1
                metadata[field][1] = "1"
                with self.assertRaisesRegex(TypeError, field):
                    compute_amg_sao_token_gae(
                        batch=batch,
                        non_tensor_batch=metadata,
                        config=config,
                    )

    def test_uid_fields_accept_numpy_string_scalars(self):
        batch, metadata, config = self._two_action_fixture()
        metadata["trajectory_uid"][0] = np.array("traj-a", dtype=np.str_)
        metadata["trajectory_uid"][1] = np.array("traj-a", dtype=np.str_)
        metadata["trajectory_row_uid"][0] = np.array("traj-a-0", dtype=np.str_)
        metadata["trajectory_row_uid"][1] = np.array("traj-a-1", dtype=np.str_)

        advantages, returns = compute_amg_sao_token_gae(
            batch=batch,
            non_tensor_batch=metadata,
            config=config,
        )

        self.assertEqual(tuple(advantages.shape), tuple(batch["values"].shape))
        self.assertEqual(tuple(returns.shape), tuple(batch["values"].shape))

    def test_rejects_early_horizon_finalized_terminal(self):
        batch, metadata, config = self._two_action_fixture()
        metadata["termination_kind"][1] = "horizon_finalized"
        metadata["horizon_finalizer_receipt"][1] = "terminal_transition_applied"

        with self.assertRaisesRegex(ValueError, "complete declared horizon"):
            compute_amg_sao_token_gae(
                batch=batch,
                non_tensor_batch=metadata,
                config=config,
            )

    def test_accepts_horizon_finalized_terminal_at_declared_horizon(self):
        batch, metadata, config = self._two_action_fixture()
        metadata["declared_max_rounds"][:2] = 2
        metadata["termination_kind"][1] = "horizon_finalized"
        metadata["horizon_finalizer_receipt"][1] = "terminal_transition_applied"

        advantages, returns = compute_amg_sao_token_gae(
            batch=batch,
            non_tensor_batch=metadata,
            config=config,
        )

        self.assertEqual(tuple(advantages.shape), tuple(batch["values"].shape))
        self.assertEqual(tuple(returns.shape), tuple(batch["values"].shape))

    def test_requires_and_validates_rollout_policy_version_span(self):
        batch, non_tensor_batch, config = self._two_action_fixture()
        missing = dict(non_tensor_batch)
        missing.pop("min_global_steps")
        with self.assertRaisesRegex(ValueError, "min_global_steps"):
            compute_amg_sao_token_gae(
                batch=batch,
                non_tensor_batch=missing,
                config=config,
            )

        reversed_span = dict(non_tensor_batch)
        reversed_span["min_global_steps"] = np.array([7, 10, -1], dtype=object)
        reversed_span["max_global_steps"] = np.array([7, 9, -1], dtype=object)
        with self.assertRaisesRegex(ValueError, "min_global_steps <= max_global_steps"):
            compute_amg_sao_token_gae(
                batch=batch,
                non_tensor_batch=reversed_span,
                config=config,
            )


if __name__ == "__main__":
    unittest.main()
