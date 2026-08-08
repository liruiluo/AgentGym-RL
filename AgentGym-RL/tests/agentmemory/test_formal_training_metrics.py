from __future__ import annotations

import hashlib
import importlib.util
import json
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


def workspace_snapshot(files=(), directories=()):
    manifest = [
        {"path": path, "sha256": sha256, "bytes": size}
        for path, sha256, size in sorted(files)
    ]
    directory_set = set(directories)
    for item in manifest:
        parts = item["path"].split("/")[:-1]
        directory_set.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
    directory_manifest = sorted(directory_set)
    encoded = json.dumps(
        {"directories": directory_manifest, "files": manifest},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "agentmemory_workspace_snapshot_v2",
        "file_count": len(manifest),
        "directory_count": len(directory_manifest),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "directories": directory_manifest,
        "files": manifest,
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def workspace_diff(before, after):
    before_files = {item["path"]: item for item in before["files"]}
    after_files = {item["path"]: item for item in after["files"]}
    before_paths = set(before_files)
    after_paths = set(after_files)
    return {
        "added": [after_files[path] for path in sorted(after_paths - before_paths)],
        "modified": [
            {"before": before_files[path], "after": after_files[path]}
            for path in sorted(before_paths & after_paths)
            if before_files[path]["sha256"] != after_files[path]["sha256"]
        ],
        "deleted": [before_files[path] for path in sorted(before_paths - after_paths)],
        "directories_added": sorted(
            set(after["directories"]) - set(before["directories"])
        ),
        "directories_deleted": sorted(
            set(before["directories"]) - set(after["directories"])
        ),
    }


def workspace_event(op, *, event_id, phase_index, before, after):
    event = {
        "event_id": event_id,
        "op": op,
        "status": "executed",
        "phase_index": phase_index,
        "workspace_tree_sha256_before": before["tree_sha256"],
        "workspace_tree_sha256_after": after["tree_sha256"],
        "workspace_diff": workspace_diff(before, after),
    }
    if op == "SHELL_COMMAND":
        event.update({"exit_code": 0, "timed_out": False})
    elif op == "APPLY_PATCH":
        event["transactional"] = True
    else:
        raise ValueError(f"unsupported workspace event op: {op}")
    return event


def workspace_info(
    *,
    snapshot,
    audit_count,
    tool_ops=(),
    workspace_event=None,
    intervention="enabled",
):
    enabled = intervention == "enabled"
    workspace_ops = [] if workspace_event is None else [workspace_event]
    return {
        "reward_components": [],
        "tool_ops": [*workspace_ops, *tool_ops],
        "workspace_ops": workspace_ops,
        "memory_ops": [],
        "workspace_surface": "codex_workspace_v2",
        "workspace_tool_contract": "codex_shell_command_apply_patch_v1",
        "workspace_tool_ops": ["SHELL_COMMAND", "APPLY_PATCH"],
        "workspace_intervention": intervention,
        "workspace_shell_enabled": enabled,
        "workspace_apply_patch_enabled": enabled,
        "workspace_snapshot": snapshot,
        "workspace_audit_event_count": audit_count,
        "workspace_latest_event": workspace_event,
    }


def attach_workspace(
    step,
    *,
    before_snapshot,
    after_snapshot,
    before_audit_count,
    after_audit_count,
    workspace_event=None,
    other_tool_ops=(),
):
    step["env_info_before"] = workspace_info(
        snapshot=before_snapshot,
        audit_count=before_audit_count,
    )
    after_info = workspace_info(
        snapshot=after_snapshot,
        audit_count=after_audit_count,
        tool_ops=other_tool_ops,
        workspace_event=workspace_event,
    )
    after_info["reward_components"] = step["env_info_after"]["reward_components"]
    if workspace_event is not None:
        op = str(workspace_event["op"]).upper()
        expected_name = f"{op.lower()}_transition"
        if not any(
            component.get("name") == expected_name
            for component in after_info["reward_components"]
        ):
            after_info["reward_components"].append(
                {"name": expected_name, "op": op, "value": 0.0}
            )
    step["env_info_after"] = after_info
    return step


def task_neutralize(step, *, before, after, event, session_advanced=None):
    for key in (
        "subtask_index_before",
        "subtask_index_after",
        "buy_accepted",
        "buy_committed",
        "session_advanced",
    ):
        step.pop(key, None)
    step["schema_version"] = "task_neutral_policy_step_v1"
    step["env_info_before"]["current_subtask_index"] = before
    step["env_info_after"]["current_subtask_index"] = after
    step["wrapper_evidence"] = {"event": event}
    if session_advanced is not None:
        step["wrapper_evidence"]["session_advanced"] = session_advanced
    return step


class FormalTrainingMetricsTests(unittest.TestCase):
    def test_task_neutral_rows_bind_phases_and_do_not_recount_handoff_buy(self) -> None:
        empty = workspace_snapshot()
        shell = workspace_event(
            "SHELL_COMMAND",
            event_id=0,
            phase_index=1,
            before=empty,
            after=empty,
        )
        native_buy = task_neutralize(
            attach_workspace(
                record(
                    "click[Buy Now]",
                    before=0,
                    after=1,
                    components=(("buy_committed_correct", 1.0),),
                ),
                before_snapshot=empty,
                after_snapshot=empty,
                before_audit_count=0,
                after_audit_count=0,
                other_tool_ops=({"op": "BUY"},),
            ),
            before=0,
            after=1,
            event="native_action",
            session_advanced=True,
        )
        handoff = task_neutralize(
            attach_workspace(
                record(
                    "no locator available",
                    before=1,
                    after=1,
                    components=(("buy_committed_correct", 1.0),),
                ),
                before_snapshot=empty,
                after_snapshot=empty,
                before_audit_count=0,
                after_audit_count=0,
            ),
            before=1,
            after=1,
            event="webshop_session_handoff",
        )
        later_shell = task_neutralize(
            attach_workspace(
                record(
                    'shell_command {"command":"rg -n pattern ."}',
                    before=1,
                    after=1,
                    components=(("shell_command_transition", 0.0),),
                ),
                before_snapshot=empty,
                after_snapshot=empty,
                before_audit_count=0,
                after_audit_count=1,
                workspace_event=shell,
            ),
            before=1,
            after=1,
            event="native_action",
            session_advanced=False,
        )
        summary = summarize_formal_training_rows(
            [
                row("neutral", 0, 1.0, 1.0, 1.0, 0.5, native_buy),
                row("neutral", 1, 0.0, 0.0, 1.0, 0.0, handoff),
                row(
                    "neutral",
                    2,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    later_shell,
                    terminal=True,
                ),
            ]
        )
        self.assertEqual(summary["correct_buy_count"], 1.0)
        self.assertEqual(summary["session_advance_count"], 1.0)
        self.assertEqual(summary["progress_ge_1_count"], 1.0)
        self.assertEqual(summary["workspace_shell_command_count"], 1.0)

    def test_task_neutral_workspace_phase_mismatch_fails_closed(self) -> None:
        empty = workspace_snapshot()
        event = workspace_event(
            "SHELL_COMMAND",
            event_id=0,
            phase_index=0,
            before=empty,
            after=empty,
        )
        step = task_neutralize(
            attach_workspace(
                record(
                    'shell_command {"command":"rg -n pattern ."}',
                    before=1,
                    after=1,
                    components=(("shell_command_transition", 0.0),),
                ),
                before_snapshot=empty,
                after_snapshot=empty,
                before_audit_count=0,
                after_audit_count=1,
                workspace_event=event,
            ),
            before=1,
            after=1,
            event="native_action",
            session_advanced=False,
        )
        with self.assertRaisesRegex(ValueError, "different session"):
            summarize_formal_training_rows(
                [row("mismatch", 0, 0.0, 0.0, 0.0, 0.0, step, terminal=True)]
            )

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

    def test_workspace_candidate_chain_requires_persisted_write_and_later_shell(self) -> None:
        path = ".agent_memory/MEMORY.md"
        content = b"selected finish: black"
        content_sha256 = hashlib.sha256(content).hexdigest()
        empty = workspace_snapshot()
        written = workspace_snapshot(((path, content_sha256, len(content)),))
        write = workspace_event(
            "APPLY_PATCH",
            event_id=0,
            phase_index=0,
            before=empty,
            after=written,
        )
        shell = workspace_event(
            "SHELL_COMMAND",
            event_id=1,
            phase_index=1,
            before=written,
            after=written,
        )
        rows = [
            row(
                "filesystem",
                0,
                0.0,
                2.0,
                2.0,
                0.1,
                attach_workspace(
                    record(
                        "apply_patch\n*** Begin Patch\n*** Add File: .agent_memory/MEMORY.md\n+selected finish: black\n*** End Patch",
                        before=0,
                        after=0,
                    ),
                    before_snapshot=empty,
                    after_snapshot=written,
                    before_audit_count=0,
                    after_audit_count=1,
                    workspace_event=write,
                ),
            ),
            row(
                "filesystem",
                1,
                1.0,
                2.0,
                2.0,
                1.0,
                attach_workspace(
                    record(
                        'BUY {"product_id":"B01"}',
                        before=0,
                        after=1,
                        accepted=True,
                        committed=True,
                        advanced=True,
                    ),
                    before_snapshot=written,
                    after_snapshot=written,
                    before_audit_count=1,
                    after_audit_count=1,
                    other_tool_ops=({"op": "BUY"},),
                ),
            ),
            row(
                "filesystem",
                2,
                0.0,
                1.0,
                2.0,
                0.2,
                attach_workspace(
                    record(
                        'shell_command {"command":"rg -n \'selected finish\' .agent_memory/MEMORY.md","workdir":".","timeout_ms":10000}',
                        before=1,
                        after=1,
                    ),
                    before_snapshot=written,
                    after_snapshot=written,
                    before_audit_count=1,
                    after_audit_count=2,
                    workspace_event=shell,
                ),
            ),
            row(
                "filesystem",
                3,
                1.0,
                1.0,
                2.0,
                1.2,
                attach_workspace(
                    record(
                        'BUY {"product_id":"B02"}',
                        before=1,
                        after=2,
                        accepted=True,
                        committed=True,
                        advanced=True,
                        done=True,
                    ),
                    before_snapshot=written,
                    after_snapshot=written,
                    before_audit_count=2,
                    after_audit_count=2,
                    other_tool_ops=({"op": "BUY"},),
                ),
                terminal=True,
            ),
        ]
        summary = summarize_formal_training_rows(rows)
        self.assertEqual(summary["filesystem_trajectory_count"], 1.0)
        self.assertEqual(summary["workspace_action_count"], 2.0)
        self.assertEqual(summary["workspace_apply_patch_count"], 1.0)
        self.assertEqual(summary["workspace_shell_command_count"], 1.0)
        self.assertEqual(
            summary["source_workspace_write_before_correct_buy_count"], 1.0
        )
        self.assertEqual(
            summary["later_session_shell_after_source_write_count"], 1.0
        )
        self.assertEqual(
            summary["workspace_cross_session_success_candidate_count"], 1.0
        )
        self.assertEqual(summary["functional_memory_chain_count"], 0.0)
        self.assertEqual(summary["workspace_snapshot_record_count"], 4.0)
        self.assertEqual(summary["workspace_tree_change_count"], 1.0)
        self.assertEqual(summary["workspace_final_file_count_mean"], 1.0)
        self.assertEqual(summary["workspace_final_total_bytes_mean"], len(content))
        self.assertEqual(summary["workspace_final_audit_event_count_mean"], 2.0)

    def test_workspace_overwritten_source_version_is_not_a_candidate_chain(self) -> None:
        path = "notes.md"
        original = b"black"
        revised = b"gray"
        original_sha = hashlib.sha256(original).hexdigest()
        revised_sha = hashlib.sha256(revised).hexdigest()
        empty = workspace_snapshot()
        original_snapshot = workspace_snapshot(
            ((path, original_sha, len(original)),)
        )
        revised_snapshot = workspace_snapshot(((path, revised_sha, len(revised)),))
        write = workspace_event(
            "APPLY_PATCH",
            event_id=0,
            phase_index=0,
            before=empty,
            after=original_snapshot,
        )
        edit = workspace_event(
            "APPLY_PATCH",
            event_id=1,
            phase_index=1,
            before=original_snapshot,
            after=revised_snapshot,
        )
        shell = workspace_event(
            "SHELL_COMMAND",
            event_id=2,
            phase_index=1,
            before=revised_snapshot,
            after=revised_snapshot,
        )
        steps = [
            attach_workspace(
                record(
                    "apply_patch\n*** Begin Patch\n*** Add File: notes.md\n+black\n*** End Patch",
                    before=0,
                    after=0,
                ),
                before_snapshot=empty,
                after_snapshot=original_snapshot,
                before_audit_count=0,
                after_audit_count=1,
                workspace_event=write,
            ),
            attach_workspace(
                record(
                    'BUY {"product_id":"B01"}', before=0, after=1,
                    accepted=True, committed=True, advanced=True,
                ),
                before_snapshot=original_snapshot,
                after_snapshot=original_snapshot,
                before_audit_count=1,
                after_audit_count=1,
                other_tool_ops=({"op": "BUY"},),
            ),
            attach_workspace(
                record(
                    "apply_patch\n*** Begin Patch\n*** Update File: notes.md\n@@\n-black\n+gray\n*** End Patch",
                    before=1,
                    after=1,
                ),
                before_snapshot=original_snapshot,
                after_snapshot=revised_snapshot,
                before_audit_count=1,
                after_audit_count=2,
                workspace_event=edit,
            ),
            attach_workspace(
                record(
                    'shell_command {"command":"cat notes.md","workdir":".","timeout_ms":10000}',
                    before=1,
                    after=1,
                ),
                before_snapshot=revised_snapshot,
                after_snapshot=revised_snapshot,
                before_audit_count=2,
                after_audit_count=3,
                workspace_event=shell,
            ),
            attach_workspace(
                record(
                    'BUY {"product_id":"B02"}', before=1, after=2,
                    accepted=True, committed=True, advanced=True, done=True,
                ),
                before_snapshot=revised_snapshot,
                after_snapshot=revised_snapshot,
                before_audit_count=3,
                after_audit_count=3,
                other_tool_ops=({"op": "BUY"},),
            ),
        ]
        rewards = (0.0, 1.0, 0.0, 0.0, 1.0)
        suffixes = (2.0, 2.0, 1.0, 1.0, 1.0)
        rows = [
            row(
                "same-session",
                index,
                rewards[index],
                suffixes[index],
                2.0,
                0.1,
                step,
                terminal=index == len(steps) - 1,
            )
            for index, step in enumerate(steps)
        ]
        summary = summarize_formal_training_rows(rows)
        self.assertEqual(summary["source_workspace_write_before_correct_buy_count"], 2.0)
        self.assertEqual(summary["later_session_shell_after_source_write_count"], 0.0)
        self.assertEqual(summary["workspace_cross_session_success_candidate_count"], 0.0)
        self.assertEqual(summary["functional_memory_chain_count"], 0.0)

    def test_workspace_action_text_alone_is_not_workspace_evidence(self) -> None:
        empty = workspace_snapshot()
        step = attach_workspace(
            record(
                "apply_patch\n*** Begin Patch\n*** Add File: notes.md\n+black\n*** End Patch",
                before=0,
                after=0,
            ),
            before_snapshot=empty,
            after_snapshot=empty,
            before_audit_count=0,
            after_audit_count=0,
        )
        summary = summarize_formal_training_rows(
            [row("text-only", 0, 0.0, 0.0, 0.0, 0.0, step, terminal=True)]
        )
        self.assertEqual(summary["workspace_action_count"], 0.0)
        self.assertEqual(summary["source_workspace_write_before_correct_buy_count"], 0.0)
        self.assertEqual(summary["workspace_cross_session_success_candidate_count"], 0.0)

    def test_workspace_evidence_fails_closed_on_reward_or_audit_tampering(self) -> None:
        path = "notes.md"
        content = b"black"
        content_sha = hashlib.sha256(content).hexdigest()
        empty = workspace_snapshot()
        written = workspace_snapshot(((path, content_sha, len(content)),))
        write = workspace_event(
            "APPLY_PATCH",
            event_id=0,
            phase_index=0,
            before=empty,
            after=written,
        )
        with self.assertRaisesRegex(ValueError, "non-zero task reward"):
            summarize_formal_training_rows(
                [
                    row(
                        "reward",
                        0,
                        0.1,
                        0.1,
                        0.1,
                        0.0,
                        attach_workspace(
                            record(
                                "apply_patch\n*** Begin Patch\n*** Add File: notes.md\n+black\n*** End Patch",
                                before=0,
                                after=0,
                                components=(("apply_patch_transition", 0.1),),
                            ),
                            before_snapshot=empty,
                            after_snapshot=written,
                            before_audit_count=0,
                            after_audit_count=1,
                            workspace_event=write,
                        ),
                        terminal=True,
                    )
                ]
            )
        with self.assertRaisesRegex(ValueError, "audit count"):
            summarize_formal_training_rows(
                [
                    row(
                        "audit",
                        0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        attach_workspace(
                            record(
                                "apply_patch\n*** Begin Patch\n*** Add File: notes.md\n+black\n*** End Patch",
                                before=0,
                                after=0,
                            ),
                            before_snapshot=empty,
                            after_snapshot=written,
                            before_audit_count=0,
                            after_audit_count=2,
                            workspace_event=write,
                        ),
                        terminal=True,
                    )
                ]
            )

    def test_workspace_zero_reward_allows_separate_max_round_penalty(self) -> None:
        empty = workspace_snapshot()
        event = workspace_event(
            "SHELL_COMMAND",
            event_id=0,
            phase_index=0,
            before=empty,
            after=empty,
        )
        step = attach_workspace(
            record(
                'shell_command {"command":"rg --hidden -n pattern ."}',
                before=0,
                after=0,
                components=(
                    ("shell_command_transition", 0.0),
                    ("max_round_timeout_failure", -0.01),
                ),
            ),
            before_snapshot=empty,
            after_snapshot=empty,
            before_audit_count=0,
            after_audit_count=1,
            workspace_event=event,
        )
        summary = summarize_formal_training_rows(
            [row("timeout", 0, -0.01, -0.01, -0.01, 0.0, step, terminal=True)]
        )
        self.assertEqual(summary["workspace_action_count"], 1.0)
        self.assertEqual(summary["workspace_shell_command_count"], 1.0)
        self.assertEqual(summary["timeout_trajectory_count"], 1.0)

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
