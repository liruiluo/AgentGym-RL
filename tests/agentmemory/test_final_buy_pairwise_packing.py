from __future__ import annotations

import unittest

from verl.workers.rollout.agent_vllm_rollout.pairwise_final_buy import (
    summarize_pairwise_logprob_rows,
    validate_pairwise_packing,
    validate_visible_final_buy_pair,
)


def action(parent: int, variant: str) -> str:
    product = f"generic_{parent:02d}_{variant}"
    return (
        f'BUY {{"product_id":"{product}","memory_ids":["C0"],'
        f'"why":"C0 evidence supports visible candidate {product}."}}'
    )


def uid(parent: int, digest_offset: int = 0) -> str:
    return f"{parent}:turn4:statev1:{parent + digest_offset + 1:064x}"


def packed_rows():
    parents = []
    rounds = []
    uids = []
    actions = []
    rewards = []
    errors = []
    for parent in range(16):
        for variant, reward in (("a", 2.0), ("b", -0.01)):
            parents.append(parent)
            rounds.append(4)
            uids.append(uid(parent))
            actions.append(action(parent, variant))
            rewards.append(reward)
            errors.append(None)
    return parents, rounds, uids, actions, rewards, errors


class PairwisePackingTests(unittest.TestCase):
    def test_visible_menu_pair_is_schema_only(self):
        parsed = validate_visible_final_buy_pair([action(0, "left"), action(0, "right")])
        self.assertEqual({row["product_id"] for row in parsed}, {"generic_00_left", "generic_00_right"})

    def test_visible_menu_pair_rejects_bad_memory_or_evidence(self):
        bad_memory = action(0, "a").replace('["C0"]', '["wrong"]')
        with self.assertRaisesRegex(ValueError, "memory_ids"):
            validate_visible_final_buy_pair([bad_memory, action(0, "b")])
        bad_evidence = action(0, "a").replace("generic_00_a.", "another product.")
        with self.assertRaisesRegex(ValueError, "evidence"):
            validate_visible_final_buy_pair([bad_evidence, action(0, "b")])

    def test_exact_terminal_pair_packing_passes(self):
        values = packed_rows()
        summary = validate_pairwise_packing(
            parent_indices=values[0],
            task_rounds=values[1],
            uids=values[2],
            actions=values[3],
            rewards=values[4],
            errors=values[5],
            expected_parent_count=16,
        )
        self.assertEqual(summary, {"rows": 32, "exact_state_groups": 16, "parents": 16})

    def test_packing_rejects_cross_state_or_replay_error(self):
        values = [list(value) for value in packed_rows()]
        values[2][1] = uid(0, digest_offset=100)
        with self.assertRaisesRegex(ValueError, "one exact-state group per parent|multiple prompt-state"):
            validate_pairwise_packing(
                parent_indices=values[0], task_rounds=values[1], uids=values[2],
                actions=values[3], rewards=values[4], errors=values[5],
                expected_parent_count=16,
            )
        values = [list(value) for value in packed_rows()]
        values[5][0] = "environment replay failed"
        with self.assertRaisesRegex(ValueError, "replay row"):
            validate_pairwise_packing(
                parent_indices=values[0], task_rounds=values[1], uids=values[2],
                actions=values[3], rewards=values[4], errors=values[5],
                expected_parent_count=16,
            )

    def test_after_update_margin_readback_is_grouped_by_exact_uid(self):
        rows = []
        for parent in range(16):
            rows.extend(
                [
                    {
                        "uid": uid(parent), "parent_index": parent,
                        "action": action(parent, "a"), "reward": 2.0,
                        "ppo_valid_sample": True, "response_tokens": 10,
                        "before_seq_logp": -4.0, "after_seq_logp": -3.0,
                    },
                    {
                        "uid": uid(parent), "parent_index": parent,
                        "action": action(parent, "b"), "reward": -0.01,
                        "ppo_valid_sample": True, "response_tokens": 10,
                        "before_seq_logp": -4.0, "after_seq_logp": -5.0,
                    },
                ]
            )
        summary = summarize_pairwise_logprob_rows(rows, expected_group_count=16)
        self.assertEqual(summary["nondecreasing_margin_states"], 16)
        self.assertAlmostEqual(summary["mean_margin_delta"], 2.0)
        self.assertGreater(summary["max_abs_seq_logp_delta"], 0)

    def test_readback_rejects_padding_or_cross_state_pairs(self):
        rows = [
            {
                "uid": uid(0), "parent_index": 0, "action": action(0, "a"),
                "reward": 2.0, "ppo_valid_sample": False, "response_tokens": 10,
                "before_seq_logp": -4.0, "after_seq_logp": -3.0,
            },
            {
                "uid": uid(0), "parent_index": 0, "action": action(0, "b"),
                "reward": -0.01, "ppo_valid_sample": True, "response_tokens": 10,
                "before_seq_logp": -4.0, "after_seq_logp": -5.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "actor-valid"):
            summarize_pairwise_logprob_rows(rows, expected_group_count=1)


if __name__ == "__main__":
    unittest.main()
