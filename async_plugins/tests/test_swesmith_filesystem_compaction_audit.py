#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from agentmemorygym_verl import swesmith_filesystem_compaction_audit as audit


CHECKPOINT_PATH = ".agent_memory/CONTINUATION.md"
CHECKPOINT_SHA = "c" * 64


def receipt(*, changed: bool, completed: bool = True, sha: str = CHECKPOINT_SHA):
    return {
        "schema": "agentmemory_filesystem_checkpoint_receipt_v1",
        "action_completed": completed,
        "action_kind": "shell_command",
        "changed": changed,
        "exists": True,
        "regular_file": True,
        "path": CHECKPOINT_PATH,
        "sha256": sha,
        "size_bytes": 321,
    }


def row(
    order, action, *, event="native_action", checkpoint=None, replace=False, retry=False
):
    info = {
        "action_kind": "shell_command",
        "actor_credit": {
            "schema": "task_neutral_actor_credit_v1",
            "basis": "shell_executed",
            "positive_eligible": True,
        },
    }
    if checkpoint is not None:
        info["filesystem_checkpoint"] = dict(checkpoint)
    evidence = {
        "event": event,
        "actor_credit": dict(info["actor_credit"]),
    }
    if event == "context_compaction":
        evidence.update(
            checkpoint_failure_reason=None,
            checkpoint_receipt=dict(checkpoint),
            checkpoint_max_bytes=8192,
            context_replaced=replace,
            continuation_path=CHECKPOINT_PATH,
            continuation_persisted=replace,
            native_observation_preserved_in_ledger=True,
            replacement_contains_native_observation=False,
            replacement_contains_policy_output=False,
            retry_pending=retry,
            retry_context_restored=retry,
            sampled_policy_output_preserved_in_ledger=True,
        )
    return {
        "trajectory_uid": "u",
        "trajectory_row_order": order,
        "item_id": "i",
        "data_source": "swesmith",
        "action": action,
        "action_submission": {"raw_policy_output": action},
        "wrapper_evidence": evidence,
        "context_transition": {
            "schema": "agentmemory_task_neutral_context_transition_v1",
            "operation": (
                "replace_messages"
                if replace
                else "retry_control"
                if retry
                else "append_observation"
            ),
            "messages": [],
        },
        "control_request": "write checkpoint" if event == "context_compaction" else None,
        "env_info_after": info,
        "response_token_count": 10,
        "rollout_done_flag": False,
        "trajectory_terminal": False,
        "trajectory_return": -0.01,
        "outcome": "continue",
    }


class AuditTest(unittest.TestCase):
    def emit(self, path, records):
        directory = path / "rollout_data"
        directory.mkdir(parents=True)
        with (directory / "1.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps({"step_record_json": json.dumps(record)}) + "\n")

    def valid_chain(self):
        failed = row(
            3,
            'shell_command {"command":"true","workdir":"."}',
            event="context_compaction",
            checkpoint=receipt(changed=False, completed=False),
            retry=True,
        )
        write = row(
            4,
            'shell_command {"command":"printf state > .agent_memory/CONTINUATION.md","workdir":"."}',
            event="context_compaction",
            checkpoint=receipt(changed=True),
            replace=True,
        )
        read = row(
            5,
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md","workdir":"."}',
            checkpoint=receipt(changed=False),
        )
        task = row(
            6,
            'shell_command {"command":"sed -n 1,80p src/module.py","workdir":"."}',
            checkpoint=receipt(changed=False),
        )
        task["trajectory_terminal"] = True
        task["trajectory_return"] = 1.0
        task["outcome"] = "success"
        return [failed, write, read, task]

    def test_q_style_receipts_count_attempt_opportunity_read_and_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.emit(path, self.valid_chain())
            result = audit.analyze(path)
            self.assertEqual(result["schema"], "amg_swesmith_filesystem_compaction_audit_v2")
            self.assertEqual(result["compaction_action_attempt_count"], 2)
            self.assertEqual(result["compaction_opportunity_count"], 1)
            self.assertEqual(result["successful_replacement_count"], 1)
            self.assertEqual(result["strict_write_compaction_read_chain_count"], 1)
            self.assertEqual(result["behavioral_continuation_chain_count"], 1)
            self.assertEqual(result["strict_chain_task_success_count"], 1)
            self.assertEqual(result["invalid_replacement_count"], 0)
            self.assertEqual(result["bounded_retry_restore_count"], 1)
            self.assertEqual(result["invalid_retry_transition_count"], 0)

    def test_failed_checkpoint_must_restore_the_precontrol_context(self):
        with tempfile.TemporaryDirectory() as directory:
            failed = row(
                3,
                'shell_command {"command":"true","workdir":"."}',
                event="context_compaction",
                checkpoint=receipt(changed=False, completed=False),
            )
            path = Path(directory)
            self.emit(path, [failed])
            result = audit.analyze(path)
            self.assertEqual(result["bounded_retry_restore_count"], 0)
            self.assertEqual(result["invalid_retry_transition_count"], 1)

    def test_q_style_chain_fails_closed(self):
        cases = (
            "write_not_changed",
            "read_hash_mismatch",
            "path_mentioned_but_not_read",
            "replacement_contains_policy_output",
            "no_later_task_action",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                records = self.valid_chain()
                if case == "write_not_changed":
                    records[1]["wrapper_evidence"]["checkpoint_receipt"]["changed"] = False
                    records[1]["env_info_after"]["filesystem_checkpoint"]["changed"] = False
                elif case == "read_hash_mismatch":
                    records[2]["env_info_after"]["filesystem_checkpoint"]["sha256"] = "d" * 64
                elif case == "path_mentioned_but_not_read":
                    action = 'shell_command {"command":"printf .agent_memory/CONTINUATION.md","workdir":"."}'
                    records[2]["action"] = action
                    records[2]["action_submission"]["raw_policy_output"] = action
                elif case == "replacement_contains_policy_output":
                    records[1]["wrapper_evidence"]["replacement_contains_policy_output"] = True
                else:
                    records[3]["wrapper_evidence"]["actor_credit"]["positive_eligible"] = False
                    records[3]["env_info_after"]["filesystem_checkpoint"]["action_completed"] = False
                path = Path(directory)
                self.emit(path, records)
                result = audit.analyze(path)
                if case == "no_later_task_action":
                    self.assertEqual(result["strict_write_compaction_read_chain_count"], 1)
                    self.assertEqual(result["behavioral_continuation_chain_count"], 0)
                else:
                    self.assertEqual(result["strict_write_compaction_read_chain_count"], 0)
                    self.assertEqual(result["behavioral_continuation_chain_count"], 0)

    def test_replace_without_valid_receipt_is_reported_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = row(
                3,
                'shell_command {"command":"true","workdir":"."}',
                event="context_compaction",
                checkpoint=receipt(changed=False),
                replace=True,
            )
            path = Path(directory)
            self.emit(path, [bad])
            result = audit.analyze(path)
            self.assertEqual(result["successful_replacement_count"], 0)
            self.assertEqual(result["invalid_replacement_count"], 1)
            self.assertEqual(result["unresolved_compaction_opportunity_count"], 1)


if __name__ == "__main__":
    unittest.main()
