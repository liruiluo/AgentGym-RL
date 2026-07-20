from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl"
    / "workers"
    / "rollout"
    / "agent_vllm_rollout"
    / "formal_buy_transition.py"
)
SPEC = importlib.util.spec_from_file_location("formal_buy_transition_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FormalBuyTransitionError = MODULE.FormalBuyTransitionError
validate_formal_buy_transition = MODULE.validate_formal_buy_transition


def buy_record(*, correct: bool, advanced: bool, terminal: bool) -> dict:
    return {
        "op": "BUY",
        "step": 3,
        "committed": True,
        "purchase_correct": correct,
        "session_advanced": advanced,
        "terminal": terminal,
    }


class FormalBuyTransitionTests(unittest.TestCase):
    def test_correct_intermediate_buy_advances(self) -> None:
        evidence = validate_formal_buy_transition(
            tool_ops=[buy_record(correct=True, advanced=True, terminal=False)],
            env_step=3,
            subtask_index_before=1,
            subtask_index_after=2,
            done=False,
        )
        self.assertTrue(evidence["accepted_purchase"])

    def test_incorrect_buy_commits_and_terminates_without_advancing(self) -> None:
        evidence = validate_formal_buy_transition(
            tool_ops=[buy_record(correct=False, advanced=False, terminal=True)],
            env_step=3,
            subtask_index_before=2,
            subtask_index_after=2,
            done=True,
        )
        self.assertTrue(evidence["committed_purchase"])
        self.assertFalse(evidence["purchase_correct"])

    def test_incorrect_buy_cannot_retry(self) -> None:
        with self.assertRaisesRegex(FormalBuyTransitionError, "did not terminate"):
            validate_formal_buy_transition(
                tool_ops=[buy_record(correct=False, advanced=False, terminal=False)],
                env_step=3,
                subtask_index_before=2,
                subtask_index_after=2,
                done=False,
            )

    def test_non_buy_action_cannot_advance(self) -> None:
        with self.assertRaisesRegex(FormalBuyTransitionError, "without a committed BUY"):
            validate_formal_buy_transition(
                tool_ops=[{"op": "ADD", "step": 3}],
                env_step=3,
                subtask_index_before=0,
                subtask_index_after=1,
                done=False,
            )


if __name__ == "__main__":
    unittest.main()
