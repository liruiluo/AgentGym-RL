from __future__ import annotations

import copy
import hashlib
import unittest

from agentmemorygym_verl.compactionrl_evidence import (
    summarize_compactionrl_step_records,
)


def _native_row(*, uid: str, route: str, order: int, terminal: bool = False):
    return {
        "schema": "amg_task_neutral_action_row_v1",
        "trajectory_uid": uid,
        "trajectory_row_order": order,
        "route_id": route,
        "data_source": route,
        "action": '{"shell_command":"true"}',
        "response_token_count": 5,
        "immediate_reward": 1.0 if terminal else 0.0,
        "trajectory_terminal": terminal,
        "outcome": "success" if terminal else "running",
        "context_transition": {
            "schema": "agentmemory_task_neutral_context_transition_v1",
            "operation": "append_observation",
        },
    }


def _summary_row(*, uid: str, route: str, order: int, terminal: bool = False):
    action = "keep the exact constraints and continue from the verified state"
    row = {
        "schema": "amg_task_neutral_action_row_v1",
        "trajectory_uid": uid,
        "trajectory_row_order": order,
        "route_id": route,
        "data_source": route,
        "action": action,
        "response_token_count": 12,
        "immediate_reward": 0.0,
        "trajectory_terminal": terminal,
        "outcome": "horizon" if terminal else "running",
        "context_transition": {
            "schema": "agentmemory_task_neutral_context_transition_v1",
            "operation": "replace_messages",
        },
        "action_submission": {
            "raw_policy_output": action,
            "parser_status": "compactionrl_summary_not_dispatched",
        },
        "wrapper_evidence": {
            "schema": "agentmemory_compactionrl_receipt_v1",
            "event": "context_compaction",
            "context_memory_mode": "compactionrl",
            "context_replaced": True,
            "summary_valid": True,
            "summary_sent_to_native_environment": False,
            "native_environment_call_count": 0,
            "summary_specific_reward": False,
            "summary_sha256": hashlib.sha256(action.encode()).hexdigest(),
            "summary_byte_count": len(action.encode()),
            "summary_failure_reason": None,
            "retry_pending": False,
        },
    }
    if terminal:
        row["immediate_reward"] = -0.25
        row["horizon_finalization"] = {"applied": True}
    return row


class CompactionRLEvidenceTest(unittest.TestCase):
    def test_summarizes_valid_direct_summary_chains(self):
        records = [
            _native_row(uid="shop-1", route="webshop", order=0),
            _summary_row(uid="shop-1", route="webshop", order=1),
            _native_row(uid="shop-1", route="webshop", order=2, terminal=True),
            _summary_row(uid="code-1", route="swesmith", order=0, terminal=True),
        ]

        summary = summarize_compactionrl_step_records(
            records, expected_routes=("webshop", "swesmith")
        )

        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["totals"]["rows"], 4)
        self.assertEqual(summary["totals"]["trajectories"], 2)
        self.assertEqual(summary["totals"]["valid_compactions"], 2)
        self.assertEqual(summary["totals"]["summary_response_tokens"], 24)
        self.assertEqual(
            summary["totals"]["compactions_with_successor_policy_row"], 1
        )
        self.assertEqual(summary["totals"]["terminal_compaction_rows"], 1)
        self.assertEqual(summary["totals"]["horizon_reward_overlay_rows"], 1)
        self.assertEqual(summary["routes"]["webshop"]["valid_compactions"], 1)
        self.assertEqual(summary["routes"]["swesmith"]["valid_compactions"], 1)
        self.assertEqual(summary["routes_without_valid_compaction"], [])

    def test_route_without_compaction_is_descriptive_not_failure(self):
        summary = summarize_compactionrl_step_records(
            [_native_row(uid="code-1", route="swesmith", order=0, terminal=True)],
            expected_routes=("swesmith",),
        )
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["routes_without_valid_compaction"], ["swesmith"])

    def test_recoverable_oversize_summary_is_behavior_not_infrastructure_failure(self):
        row = _summary_row(uid="research-1", route="openmle_fast", order=0)
        row["context_transition"]["operation"] = "preserve"
        row["wrapper_evidence"].update(
            {
                "summary_valid": False,
                "context_replaced": False,
                "summary_failure_reason": "summary_too_large",
                "retry_pending": True,
                "summary_max_bytes": 8,
                "pre_context_message_count": 9,
                "post_context_message_count": 9,
                "pre_context_sha256": "a" * 64,
                "post_context_sha256": "a" * 64,
                "retained_recent_steps": 0,
            }
        )

        summary = summarize_compactionrl_step_records(
            [row], expected_routes=("openmle_fast",)
        )

        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["totals"]["valid_compactions"], 0)
        self.assertEqual(summary["totals"]["invalid_compactions"], 1)
        self.assertEqual(summary["invalid_summary_reasons"], {"summary_too_large": 1})

    def test_fails_closed_on_summary_contract_drift(self):
        base = _summary_row(uid="shop-1", route="webshop", order=0)
        mutations = []
        sent = copy.deepcopy(base)
        sent["wrapper_evidence"]["summary_sent_to_native_environment"] = True
        mutations.append(sent)
        shaped = copy.deepcopy(base)
        shaped["wrapper_evidence"]["summary_specific_reward"] = True
        mutations.append(shaped)
        wrong_hash = copy.deepcopy(base)
        wrong_hash["wrapper_evidence"]["summary_sha256"] = "0" * 64
        mutations.append(wrong_hash)
        wrong_transition = copy.deepcopy(base)
        wrong_transition["context_transition"]["operation"] = "append_observation"
        mutations.append(wrong_transition)
        native_call = copy.deepcopy(base)
        native_call["wrapper_evidence"]["native_environment_call_count"] = 1
        mutations.append(native_call)

        for record in mutations:
            with self.subTest(record=record):
                summary = summarize_compactionrl_step_records([record])
                self.assertEqual(summary["status"], "FAIL")
                self.assertGreater(summary["violation_count"], 0)

    def test_nonzero_summary_reward_requires_terminal_horizon_overlay(self):
        row = _summary_row(uid="shop-1", route="webshop", order=0)
        row["immediate_reward"] = -0.25
        summary = summarize_compactionrl_step_records([row])
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(
            any("horizon overlay" in item for item in summary["violations"])
        )

    def test_fails_on_row_order_route_and_expected_route_drift(self):
        first = _native_row(uid="shop-1", route="webshop", order=0)
        second = _native_row(uid="shop-1", route="webshop", order=2, terminal=True)
        second["data_source"] = "swesmith"
        summary = summarize_compactionrl_step_records(
            [first, second], expected_routes=("webshop", "swesmith")
        )
        self.assertEqual(summary["status"], "FAIL")
        self.assertEqual(summary["missing_routes"], ["swesmith"])
        self.assertTrue(any("row order" in item for item in summary["violations"]))
        self.assertTrue(any("route/data_source" in item for item in summary["violations"]))


if __name__ == "__main__":
    unittest.main()
