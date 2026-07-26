from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl"
    / "utils"
    / "agentgym"
    / "formal_training_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("formal_training_metrics_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
summarize_formal_training_rows = MODULE.summarize_formal_training_rows


def record(
    action: str,
    *,
    before: int,
    after: int,
    components=(),
    memory_ops=(),
    accepted=False,
    committed=False,
    advanced=False,
    done=False,
    execution="executed",
):
    op = action.split(None, 1)[0]
    return {
        "action": action,
        "action_execution": (
            None
            if execution is None
            else {
                "status": execution,
                "executed_action_op": op,
            }
        ),
        "subtask_index_before": before,
        "subtask_index_after": after,
        "buy_accepted": accepted,
        "buy_committed": committed,
        "session_advanced": advanced,
        "done": done,
        "outcome": "success" if done and accepted else "running",
        "env_info_after": {
            "reward_components": [
                {"name": name, "value": value} for name, value in components
            ],
            "memory_ops": list(memory_ops),
        },
    }


def row(uid, order, reward, suffix, total, advantage, step, *, terminal=False):
    return {
        "trajectory_uid": uid,
        "row_order": order,
        "terminal": terminal,
        "immediate_reward": reward,
        "suffix_return": suffix,
        "trajectory_return": total,
        "advantage_token_mean": advantage,
        "record": step,
    }


class FormalTrainingMetricsTests(unittest.TestCase):
    def test_native_records_use_reward_ledger_for_invalid_count(self) -> None:
        valid_native = record(
            "search[widget]",
            before=0,
            after=0,
            components=(("search_transition", 0.0),),
            execution=None,
        )
        invalid_native = record(
            "CLICK[BUY NOW]",
            before=0,
            after=0,
            components=(("invalid_action", -0.01),),
            execution=None,
        )
        summary = summarize_formal_training_rows(
            [
                row("valid", 0, 0.0, 0.0, 0.0, 0.0, valid_native, terminal=True),
                row(
                    "invalid",
                    0,
                    -0.01,
                    -0.01,
                    -0.01,
                    -0.1,
                    invalid_native,
                    terminal=True,
                ),
            ]
        )
        self.assertEqual(summary["invalid_action_count"], 1.0)

    def test_native_memory_ops_drive_chain_without_execution_metadata(self) -> None:
        rows = [
            row(
                "native",
                0,
                0.0,
                2.0,
                2.0,
                0.1,
                record(
                    'ADD {"key":"source","value":"ma_a"}',
                    before=0,
                    after=0,
                    memory_ops=({"op": "ADD"},),
                    execution=None,
                ),
            ),
            row(
                "native",
                1,
                1.0,
                2.0,
                2.0,
                1.0,
                record(
                    'BUY {"product_id":"B01"}',
                    before=0,
                    after=1,
                    accepted=True,
                    committed=True,
                    advanced=True,
                    execution=None,
                ),
            ),
            row(
                "native",
                2,
                0.0,
                1.0,
                2.0,
                0.2,
                record(
                    'RETRIEVE {"query":"source","top_k":3}',
                    before=1,
                    after=1,
                    components=(
                        ("memory_retrieve_first_relevant_before_dependent_buy", 0.0),
                    ),
                    memory_ops=({"op": "RETRIEVE", "retrieved_count": 1},),
                    execution=None,
                ),
            ),
            row(
                "native",
                3,
                1.0,
                1.0,
                2.0,
                1.2,
                record(
                    'BUY {"product_id":"B02"}',
                    before=1,
                    after=2,
                    accepted=True,
                    committed=True,
                    advanced=True,
                    done=True,
                    execution=None,
                ),
                terminal=True,
            ),
        ]
        summary = summarize_formal_training_rows(rows)
        self.assertEqual(summary["invalid_action_count"], 0.0)
        self.assertEqual(summary["nonempty_retrieve_count"], 1.0)
        self.assertEqual(summary["functional_memory_chain_count"], 1.0)

    def test_action_text_alone_does_not_create_memory_write_position(self) -> None:
        rows = [
            row(
                "no_write",
                0,
                0.0,
                1.0,
                1.0,
                0.0,
                record(
                    'ADD {"key":"source","value":"ma_a"}',
                    before=0,
                    after=0,
                    execution=None,
                ),
            ),
            row(
                "no_write",
                1,
                0.0,
                1.0,
                1.0,
                0.0,
                record(
                    'RETRIEVE {"query":"source","top_k":3}',
                    before=1,
                    after=1,
                    components=(
                        ("memory_retrieve_first_relevant_before_dependent_buy", 0.0),
                    ),
                    memory_ops=({"op": "RETRIEVE", "retrieved_count": 1},),
                    execution=None,
                ),
            ),
            row(
                "no_write",
                2,
                1.0,
                1.0,
                1.0,
                1.0,
                record(
                    'BUY {"product_id":"B02"}',
                    before=1,
                    after=2,
                    accepted=True,
                    committed=True,
                    advanced=True,
                    done=True,
                    execution=None,
                ),
                terminal=True,
            ),
        ]
        summary = summarize_formal_training_rows(rows)
        self.assertEqual(summary["functional_memory_chain_count"], 0.0)

    def test_separates_reward_axes_and_detects_functional_chain(self) -> None:
        rows = [
            row(
                "t0",
                0,
                0.05,
                2.10,
                2.10,
                1.0,
                record(
                    'ADD {"key":"source","value":"ma_a"}',
                    before=0,
                    after=0,
                    components=(("memory_add_first_visible_product_reference", 0.05),),
                    memory_ops=({"op": "ADD"},),
                ),
            ),
            row(
                "t0",
                1,
                1.0,
                2.05,
                2.10,
                1.0,
                record(
                    'BUY {"product_id":"B01"}',
                    before=0,
                    after=1,
                    accepted=True,
                    committed=True,
                    advanced=True,
                ),
            ),
            row(
                "t0",
                2,
                0.05,
                1.05,
                2.10,
                0.8,
                record(
                    'RETRIEVE {"query":"source","top_k":3}',
                    before=1,
                    after=1,
                    components=(("memory_retrieve_first_relevant_before_dependent_buy", 0.05),),
                    memory_ops=(({"op": "RETRIEVE", "retrieved_count": 1}),),
                ),
            ),
            row(
                "t0",
                3,
                1.0,
                1.0,
                2.10,
                1.2,
                record(
                    'BUY {"product_id":"B02"}',
                    before=1,
                    after=2,
                    accepted=True,
                    committed=True,
                    advanced=True,
                    done=True,
                ),
                terminal=True,
            ),
        ]
        summary = summarize_formal_training_rows(rows)
        self.assertEqual(summary["trajectory_count"], 1.0)
        self.assertEqual(summary["functional_memory_chain_count"], 1.0)
        self.assertEqual(summary["progress_ge_2_count"], 1.0)
        self.assertEqual(summary["nonempty_retrieve_count"], 1.0)
        self.assertEqual(summary["relevant_retrieve_count"], 1.0)
        self.assertAlmostEqual(summary["trajectory_return_mean"], 2.10)
        self.assertAlmostEqual(summary["immediate_reward_per_action_mean"], 0.525)
        self.assertAlmostEqual(summary["suffix_return_per_action_mean"], 1.55)

    def test_first_valid_later_session_retrieve_can_be_empty(self) -> None:
        step = record(
            'RETRIEVE {"query":"missing","top_k":3}',
            before=1,
            after=1,
            components=(("memory_retrieve_first_valid_later_session", 0.1),),
            memory_ops=(
                {
                    "op": "RETRIEVE",
                    "retrieved_count": 0,
                    "retrieved_memory_ids": [],
                },
            ),
        )
        summary = summarize_formal_training_rows(
            [row("empty", 0, 0.1, 0.1, 0.1, 0.2, step, terminal=True)]
        )
        self.assertEqual(summary["first_valid_later_session_retrieve_count"], 1.0)
        self.assertEqual(
            summary["empty_first_valid_later_session_retrieve_count"], 1.0
        )
        self.assertEqual(summary["nonempty_retrieve_count"], 0.0)
        self.assertEqual(summary["relevant_retrieve_count"], 0.0)
        self.assertEqual(summary["source_linked_retrieve_count"], 0.0)
        self.assertEqual(summary["functional_memory_chain_count"], 0.0)

    def test_memory_ids_prove_source_link_and_functional_chain(self) -> None:
        rows = [
            row(
                "strict",
                0,
                0.01,
                2.11,
                2.11,
                0.2,
                record(
                    'ADD {"key":"source","value":"ma_a"}',
                    before=0,
                    after=0,
                    components=(("memory_add_first_valid_this_session", 0.01),),
                    memory_ops=(
                        {"op": "ADD", "memory_id": "mem_0000"},
                    ),
                ),
            ),
            row(
                "strict",
                1,
                1.0,
                2.10,
                2.11,
                1.0,
                record(
                    'BUY {"product_id":"B01"}',
                    before=0,
                    after=1,
                    accepted=True,
                    committed=True,
                    advanced=True,
                ),
            ),
            row(
                "strict",
                2,
                0.1,
                1.10,
                2.11,
                0.4,
                record(
                    'RETRIEVE {"query":"source","top_k":3}',
                    before=1,
                    after=1,
                    components=(
                        ("memory_retrieve_first_valid_later_session", 0.1),
                    ),
                    memory_ops=(
                        {
                            "op": "RETRIEVE",
                            "retrieved_count": 1,
                            "retrieved_memory_ids": ["mem_0000"],
                        },
                    ),
                ),
            ),
            row(
                "strict",
                3,
                1.0,
                1.0,
                2.11,
                1.1,
                record(
                    'BUY {"product_id":"B02"}',
                    before=1,
                    after=2,
                    accepted=True,
                    committed=True,
                    advanced=True,
                    done=True,
                ),
                terminal=True,
            ),
        ]
        summary = summarize_formal_training_rows(rows)
        self.assertEqual(summary["first_valid_add_count"], 1.0)
        self.assertEqual(summary["first_valid_later_session_retrieve_count"], 1.0)
        self.assertEqual(
            summary["empty_first_valid_later_session_retrieve_count"], 0.0
        )
        self.assertEqual(summary["nonempty_retrieve_count"], 1.0)
        self.assertEqual(summary["relevant_retrieve_count"], 0.0)
        self.assertEqual(
            summary["source_memory_write_before_correct_buy_count"], 1.0
        )
        self.assertEqual(summary["source_linked_retrieve_count"], 1.0)
        self.assertEqual(summary["functional_memory_chain_count"], 1.0)

    def test_wrong_retrieved_memory_id_is_not_source_linked(self) -> None:
        rows = [
            row(
                "wrong_id",
                0,
                0.0,
                2.0,
                2.0,
                0.1,
                record(
                    'UPDATE {"memory_id":"mem_source","value":"ma_a"}',
                    before=0,
                    after=0,
                    memory_ops=(
                        {"op": "UPDATE", "memory_id": "mem_source"},
                    ),
                ),
            ),
            row(
                "wrong_id",
                1,
                1.0,
                2.0,
                2.0,
                1.0,
                record(
                    'BUY {"product_id":"B01"}',
                    before=0,
                    after=1,
                    accepted=True,
                    committed=True,
                    advanced=True,
                ),
            ),
            row(
                "wrong_id",
                2,
                0.0,
                1.0,
                2.0,
                0.2,
                record(
                    'RETRIEVE {"query":"other","top_k":3}',
                    before=1,
                    after=1,
                    memory_ops=(
                        {
                            "op": "RETRIEVE",
                            "retrieved_count": 1,
                            "retrieved_memory_ids": ["mem_other"],
                        },
                    ),
                ),
            ),
            row(
                "wrong_id",
                3,
                1.0,
                1.0,
                2.0,
                1.0,
                record(
                    'BUY {"product_id":"B02"}',
                    before=1,
                    after=2,
                    accepted=True,
                    committed=True,
                    advanced=True,
                    done=True,
                ),
                terminal=True,
            ),
        ]
        summary = summarize_formal_training_rows(rows)
        self.assertEqual(summary["nonempty_retrieve_count"], 1.0)
        self.assertEqual(
            summary["source_memory_write_before_correct_buy_count"], 1.0
        )
        self.assertEqual(summary["source_linked_retrieve_count"], 0.0)
        self.assertEqual(summary["functional_memory_chain_count"], 0.0)

    def test_retrieve_before_source_buy_cannot_form_strict_chain(self) -> None:
        rows = [
            row(
                "wrong_order",
                0,
                0.0,
                2.0,
                2.0,
                0.1,
                record(
                    'ADD {"key":"source","value":"ma_a"}',
                    before=0,
                    after=0,
                    memory_ops=(
                        {"op": "ADD", "memory_id": "mem_0000"},
                    ),
                ),
            ),
            row(
                "wrong_order",
                1,
                0.0,
                2.0,
                2.0,
                0.2,
                record(
                    'RETRIEVE {"query":"source","top_k":3}',
                    before=1,
                    after=1,
                    memory_ops=(
                        {
                            "op": "RETRIEVE",
                            "retrieved_count": 1,
                            "retrieved_memory_ids": ["mem_0000"],
                        },
                    ),
                ),
            ),
            row(
                "wrong_order",
                2,
                1.0,
                2.0,
                2.0,
                1.0,
                record(
                    'BUY {"product_id":"B01"}',
                    before=0,
                    after=1,
                    accepted=True,
                    committed=True,
                    advanced=True,
                ),
            ),
            row(
                "wrong_order",
                3,
                1.0,
                1.0,
                2.0,
                1.0,
                record(
                    'BUY {"product_id":"B02"}',
                    before=1,
                    after=2,
                    accepted=True,
                    committed=True,
                    advanced=True,
                    done=True,
                ),
                terminal=True,
            ),
        ]
        summary = summarize_formal_training_rows(rows)
        self.assertEqual(
            summary["source_memory_write_before_correct_buy_count"], 1.0
        )
        self.assertEqual(summary["source_linked_retrieve_count"], 0.0)
        self.assertEqual(summary["functional_memory_chain_count"], 0.0)

    def test_reports_positive_timeout_credit_separately(self) -> None:
        timeout = record(
            'RETRIEVE {"query":"x","top_k":3}',
            before=0,
            after=0,
            components=(("max_round_timeout_failure", -0.05),),
        )
        summary = summarize_formal_training_rows(
            [row("t0", 0, -0.05, -0.05, -0.05, 0.3, timeout, terminal=True)]
        )
        self.assertEqual(summary["timeout_trajectory_count"], 1.0)
        self.assertEqual(summary["timeout_positive_advantage_rate"], 1.0)
        self.assertEqual(summary["correct_buy_positive_advantage_rate"], 0.0)

    def test_additional_nonempty_retrieve_with_relevant_flag_counts(self) -> None:
        step = record(
            'RETRIEVE {"query":"source","top_k":3}',
            before=1,
            after=1,
            components=(("environment_base_reward", 0.0),),
            memory_ops=(({"op": "RETRIEVE", "retrieved_count": 1}),),
        )
        step["env_info_after"]["reward_components"].append(
            {
                "name": "memory_retrieve_additional_nonempty_dependent_context",
                "value": 0.0,
                "relevant": True,
            }
        )
        summary = summarize_formal_training_rows(
            [row("t0", 0, 0.0, 0.0, 0.0, 0.1, step, terminal=True)]
        )
        self.assertEqual(summary["nonempty_retrieve_count"], 1.0)
        self.assertEqual(summary["relevant_retrieve_count"], 1.0)

    def test_fails_closed_on_reward_or_terminal_mismatch(self) -> None:
        terminal = record('ANSWER {"text":"x"}', before=0, after=0)
        with self.assertRaisesRegex(ValueError, "reward sum mismatch"):
            summarize_formal_training_rows(
                [row("t0", 0, 0.0, 0.0, 1.0, 0.0, terminal, terminal=True)]
            )
        with self.assertRaisesRegex(ValueError, "terminal placement"):
            summarize_formal_training_rows(
                [row("t0", 0, 0.0, 0.0, 0.0, 0.0, terminal, terminal=False)]
            )
        with self.assertRaisesRegex(ValueError, "suffix return mismatch"):
            summarize_formal_training_rows(
                [row("t0", 0, 0.0, 1.0, 0.0, 0.0, terminal, terminal=True)]
            )


if __name__ == "__main__":
    unittest.main()
