#!/usr/bin/env python3
"""Verify a real LiteResearcher PPO update on the frozen 64-task substrate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


STEP_SCHEMA = "task_neutral_policy_step_v1"
TRANSITION_SCHEMA = "agentmemory_task_neutral_context_transition_v1"
NATIVE_EVENT = "native_action"
COMPACTION_EVENT = "context_compaction"

PARENT_GROUP_UID = "agentmemory_parent_group_uid"
EXACT_STATE_UID = "agentmemory_exact_state_uid"
REPLICA_INDEX = "agentmemory_replica_index"
TRAJECTORY_UID = "agentmemory_trajectory_uid"
TRAJECTORY_ROW_UID = "agentmemory_trajectory_row_uid"
TRAJECTORY_ROW_ORDER = "agentmemory_trajectory_row_order"
TRAJECTORY_TERMINAL = "agentmemory_trajectory_terminal"
TRAJECTORY_RETURN = "agentmemory_trajectory_return"
IMMEDIATE_REWARD = "agentmemory_immediate_reward"
SUFFIX_RETURN = "agentmemory_suffix_return"
SUFFIX_CREDIT_APPLIED = "agentmemory_suffix_credit_applied"
GENERATION_PROMPT_LENGTH = "agentmemory_generation_prompt_length"
GENERATION_PROMPT_DIGEST = "agentmemory_generation_prompt_digest"
PACKED_PROMPT_LENGTH = "agentmemory_packed_prompt_length"
PACKED_PROMPT_DIGEST = "agentmemory_packed_prompt_digest"
ACTION_TEXT = "agentmemory_action_text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoint-probe", type=Path, required=True)
    parser.add_argument("--metadata-after", type=Path, required=True)
    parser.add_argument("--global-step", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--task-count", type=int, default=64)
    parser.add_argument(
        "--endpoint-probe-indices",
        default="0,1,13,14,30,31,47,63",
    )
    parser.add_argument(
        "--require-compaction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected an object in {path}")
    return value


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"{label} is not finite: {value!r}")
    return result


def assert_close(actual: Any, expected: Any, label: str) -> None:
    if not math.isclose(
        finite(actual, f"{label}.actual"),
        finite(expected, f"{label}.expected"),
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise AssertionError(
            f"{label} differs: actual={actual!r} expected={expected!r}"
        )


def parse_probe_indices(raw: str) -> list[int]:
    try:
        values = [int(value) for value in raw.split(",") if value != ""]
    except ValueError as exc:
        raise ValueError("probe indices must be comma-separated integers") from exc
    if len(values) != 8:
        raise ValueError("the endpoint isolation probe requires exactly 8 indices")
    if any(value < 0 for value in values):
        raise ValueError("probe indices must be nonnegative")
    if len(set(values)) != len(values):
        raise ValueError("probe indices must be distinct")
    return values


def verify_endpoint_probe(
    payload: Mapping[str, Any],
    *,
    expected_indices: list[int],
    task_count: int,
) -> dict[str, Any]:
    assert payload["status"] == "pass"
    actual = [int(value) for value in payload["indices"]]
    assert actual == expected_indices
    assert all(0 <= value < task_count for value in actual)
    source_indices = [int(value) for value in payload["source_data_indices"]]
    assert len(source_indices) == 8
    assert len(set(source_indices)) == 8
    assert len(payload["slot_ids"]) == 8
    assert len(set(int(value) for value in payload["slot_ids"])) == 8
    assert int(payload["metadata_after"]["active_environment_count"]) == 0
    assert int(payload["metadata_after"]["active_workspace_count"]) == 0
    return {
        "indices": actual,
        "source_data_indices": source_indices,
        "manifest_sha256": str(payload["manifest_sha256"]),
    }


def verify_response_cap_truncation(
    record: Mapping[str, Any], *, parent_index: int
) -> dict[str, Any]:
    response_count = int(record["response_token_count"])
    response_cap = int(record["max_response_tokens"])
    assert record["truncated"] is True
    assert record["finish_reason"] == "length"
    assert str(record["finish_reason_source"]).endswith(":backend")
    assert record.get("generation_stop_reason") is None
    assert record.get("stop_reason") is None
    assert response_count == response_cap > 0
    assert int(record["generation_response_length"]) == response_count
    assert int(record["packed_response_length"]) == response_count
    assert record["generation_token_ids_are_exact"] is True
    assert record["backend_token_ids_are_exact"] is True
    return {
        "parent_index": parent_index,
        "task_round": int(record["task_round"]),
        "response_token_count": response_count,
        "max_response_tokens": response_cap,
    }


def classify_action(record: Mapping[str, Any]) -> str:
    evidence = record["wrapper_evidence"]
    event = str(evidence["event"])
    transition = record["context_transition"]
    assert transition["schema"] == TRANSITION_SCHEMA
    submission = record["action_submission"]

    if event == COMPACTION_EVENT:
        assert transition["operation"] == "replace_messages"
        assert transition["messages"]
        assert submission["submitted_action"] is None
        assert submission["parser_status"] == "policy_context_compaction"
        assert float(record["immediate_reward"]) == 0.0
        assert record["done"] is False
        return "compaction"

    assert event == NATIVE_EVENT
    assert transition["operation"] == "append_observation"
    assert submission["raw_policy_output"] == record["action"]
    server_evidence = evidence.get("server_wrapper_evidence", {})
    if "tool" in submission:
        kind = str(submission["tool"])
        assert kind in {"search", "visit"}
        assert int(server_evidence["native_environment_call_count"]) == 1
        return kind
    if submission.get("kind") == "workspace":
        op = str(submission["op"]).upper()
        assert op in {"SHELL_COMMAND", "APPLY_PATCH"}
        assert str(server_evidence["workspace_op"]).upper() == op
        assert float(server_evidence["workspace_reward"]) == 0.0
        assert int(server_evidence["native_environment_call_count"]) == 0
        return op.lower()
    if submission.get("kind") == "answer":
        assert server_evidence["terminal_answer_only"] is True
        assert bool(server_evidence["answer_correct"]) == (
            float(record["immediate_reward"]) == 1.0
        )
        return "answer"
    assert server_evidence.get("invalid_action") is True
    return "invalid_action"


def row_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(record["parent_index"]),
        str(record["parent_group_uid"]),
        int(record["replica_index"]),
        str(record["trajectory_uid"]),
        str(record["trajectory_row_uid"]),
        int(record["trajectory_row_order"]),
        bool(record["trajectory_terminal"]),
        int(record["task_round"]),
        str(record["action"]),
        tuple(int(value) for value in record["response_token_ids"]),
        int(record["generation_prompt_length"]),
        str(record["generation_prompt_digest"]),
        int(record["packed_prompt_length"]),
        str(record["packed_prompt_digest"]),
        finite(record["immediate_reward"], "row_identity.immediate_reward"),
        finite(record["trajectory_return"], "row_identity.trajectory_return"),
    )


def verify_row_binding(
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    parent_index: int,
) -> None:
    """Bind one PPO diagnostic row to its canonical sampled policy record."""

    assert record["schema_version"] == STEP_SCHEMA
    assert int(record["parent_index"]) == parent_index
    assert int(record["task_round"]) == int(row["task_round"])
    assert int(record["trajectory_row_order"]) + 1 == int(record["task_round"])
    assert record["item_id"] == f"literesearcher_{parent_index}"

    exact_fields = (
        ("parent_group_uid", PARENT_GROUP_UID, str),
        ("exact_state_uid", EXACT_STATE_UID, str),
        ("replica_index", REPLICA_INDEX, int),
        ("trajectory_uid", TRAJECTORY_UID, str),
        ("trajectory_row_uid", TRAJECTORY_ROW_UID, str),
        ("trajectory_row_order", TRAJECTORY_ROW_ORDER, int),
        ("trajectory_terminal", TRAJECTORY_TERMINAL, bool),
        ("action", ACTION_TEXT, str),
        ("generation_prompt_length", GENERATION_PROMPT_LENGTH, int),
        ("generation_prompt_digest", GENERATION_PROMPT_DIGEST, str),
        ("packed_prompt_length", PACKED_PROMPT_LENGTH, int),
        ("packed_prompt_digest", PACKED_PROMPT_DIGEST, str),
        ("suffix_credit_applied", SUFFIX_CREDIT_APPLIED, bool),
    )
    for record_key, row_key, normalize in exact_fields:
        assert normalize(record[record_key]) == normalize(row[row_key]), (
            f"{record_key} differs between the PPO row and formal record"
        )

    assert int(record["generation_prompt_length"]) > 0
    assert int(record["packed_prompt_length"]) > 0
    assert record["generation_token_ids_are_exact"] is True
    assert record["backend_token_ids_are_exact"] is True
    assert record["response_token_ids"]
    response_count = len(record["response_token_ids"])
    assert int(record["response_token_count"]) == response_count
    assert int(record["generation_response_length"]) == response_count
    assert int(record["packed_response_length"]) == response_count
    assert int(row["response_mask_sum"]) == response_count

    assert_close(row[IMMEDIATE_REWARD], record["immediate_reward"], IMMEDIATE_REWARD)
    assert_close(row["score_sum"], record["immediate_reward"], "score_sum")
    assert_close(record["score"], record["immediate_reward"], "record.score")
    assert_close(row[TRAJECTORY_RETURN], record["trajectory_return"], TRAJECTORY_RETURN)
    assert_close(row[SUFFIX_RETURN], record["suffix_return"], SUFFIX_RETURN)
    finite(row["old_logprob_mean"], "old_logprob_mean")


def verify_trajectory_records(
    parent_index: int,
    records: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Require one ordered, internally consistent trajectory per dataset row."""

    ordered = sorted(records, key=lambda value: int(value["trajectory_row_order"]))
    row_count = len(ordered)
    assert row_count > 0
    assert [int(value["trajectory_row_order"]) for value in ordered] == list(
        range(row_count)
    )
    assert [int(value["task_round"]) for value in ordered] == list(
        range(1, row_count + 1)
    )
    assert len({str(value["trajectory_row_uid"]) for value in ordered}) == row_count
    assert len({str(value["parent_group_uid"]) for value in ordered}) == 1
    assert len({str(value["trajectory_uid"]) for value in ordered}) == 1
    assert len({int(value["replica_index"]) for value in ordered}) == 1
    assert sum(bool(value["trajectory_terminal"]) for value in ordered) == 1
    assert ordered[-1]["trajectory_terminal"] is True
    assert all(
        int(value["env_info_after"]["data_idx"]) == parent_index
        for value in ordered
    )

    returns = [finite(value["trajectory_return"], "trajectory_return") for value in ordered]
    assert all(math.isclose(value, returns[0], abs_tol=1e-6) for value in returns)
    reward_sum = sum(
        finite(value["immediate_reward"], "immediate_reward") for value in ordered
    )
    assert math.isclose(reward_sum, returns[0], rel_tol=1e-6, abs_tol=1e-6)
    return ordered


def verify_readback(
    run_dir: Path,
    *,
    global_step: int,
    expected_indices: set[int],
    expected_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    path = run_dir / "diagnostics" / f"formal_update_readback_step{global_step}.json"
    payload = load_json(path)
    assert int(payload["global_step"]) == global_step
    assert payload["role"] == "same_batch_post_optimizer_readback"
    evidence = payload["row_evidence"]
    assert evidence["schema"] == "agentmemory_formal_step_records_v1"
    assert evidence["task_name"] == "literesearcher"
    rows = evidence["rows"]
    assert rows
    assert payload["formal_step_records"] == rows
    assert {int(row["parent_index"]) for row in rows} == expected_indices
    actual_identities = [row_identity(row) for row in rows]
    expected_identities = [row_identity(row) for row in expected_records]
    assert len(set(actual_identities)) == len(actual_identities)
    assert actual_identities == expected_identities

    result: dict[str, Any] = {
        "row_evidence_schema": evidence["schema"],
        "row_count": len(rows),
    }
    for role in ("actor", "critic"):
        role_payload = payload[role]
        max_abs_delta = finite(
            role_payload["summary"]["max_abs_delta"],
            f"{role}.max_abs_delta",
        )
        delta_l2 = finite(
            role_payload["parameter_delta_l2"],
            f"{role}.parameter_delta_l2",
        )
        changed = int(
            role_payload["parameter_probe"]["parameter_probe_changed_count"]
        )
        assert max_abs_delta > 0.0
        assert delta_l2 > 0.0
        assert changed > 0
        result[role] = {
            "max_abs_delta": max_abs_delta,
            "parameter_delta_l2": delta_l2,
            "parameter_probe_changed_count": changed,
        }
    return result


def verify_metadata_after(payload: Mapping[str, Any]) -> dict[str, Any]:
    assert payload["domain_id"] == "literesearcher"
    assert int(payload["active_environment_count"]) == 0
    assert int(payload["active_workspace_count"]) == 0
    return {
        "domain_id": payload["domain_id"],
        "active_environment_count": 0,
        "active_workspace_count": 0,
        "service_fingerprint_sha256": payload["service"]["fingerprint_sha256"],
    }


def main() -> None:
    args = parse_args()
    assert args.global_step > 0
    assert args.train_batch_size == 64
    assert args.task_count == 64
    expected_indices = set(range(args.task_count))

    endpoint_probe = verify_endpoint_probe(
        load_json(args.endpoint_probe),
        expected_indices=parse_probe_indices(args.endpoint_probe_indices),
        task_count=args.task_count,
    )
    batch_path = (
        args.run_dir
        / "diagnostics"
        / f"ppo_batch_step{args.global_step}_post_adv.json"
    )
    payload = load_json(batch_path)
    assert int(payload["global_step"]) == args.global_step
    assert payload["stage"] == "post_adv"
    assert payload["prompt_attestation_passed"] is True
    assert int(payload["valid_rows"]) > 0

    rows_by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    valid_records: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    nonzero_advantage_rows = 0
    nonzero_return_rows = 0
    terminal_successes: set[int] = set()
    truncated_rows: list[dict[str, Any]] = []

    for row in payload["rows"]:
        if not row["ppo_valid_sample"]:
            continue
        parent_index = int(row["parent_index"])
        assert parent_index in expected_indices
        record = row["formal_step_record"]
        verify_row_binding(row, record, parent_index=parent_index)
        nonzero_advantage_rows += int(row.get("adv_nonzero", 0) > 0)
        nonzero_return_rows += int(row.get("return_nonzero", 0) > 0)
        if record["truncated"]:
            truncated_rows.append(
                verify_response_cap_truncation(record, parent_index=parent_index)
            )

        env_info = record["env_info_after"]
        assert env_info["domain_id"] == "literesearcher"
        assert int(env_info["data_idx"]) == parent_index
        assert str(env_info["task_id"]).startswith("stage1:")
        action_kind = classify_action(record)
        action_counts[action_kind] += 1
        if (
            record["trajectory_terminal"]
            and record["outcome"] == "success"
            and float(record["trajectory_return"]) > 0.0
        ):
            terminal_successes.add(parent_index)
        rows_by_parent[parent_index].append(record)
        valid_records.append(record)

    assert set(rows_by_parent) == expected_indices
    assert nonzero_advantage_rows > 0
    assert nonzero_return_rows > 0
    assert terminal_successes
    for parent_index, records in rows_by_parent.items():
        rows_by_parent[parent_index] = list(
            verify_trajectory_records(parent_index, records)
        )

    for action_kind in ("search", "visit", "answer"):
        assert action_counts[action_kind] > 0
    assert action_counts["shell_command"] + action_counts["apply_patch"] > 0
    if args.require_compaction:
        assert action_counts["compaction"] > 0

    metadata_after = verify_metadata_after(load_json(args.metadata_after))
    readback = verify_readback(
        args.run_dir,
        global_step=args.global_step,
        expected_indices=expected_indices,
        expected_records=valid_records,
    )
    evidence = {
        "schema": "agentmemory_literesearcher_ppo_gate_attestation_v1",
        "status": "pass",
        "global_step": args.global_step,
        "train_batch_size": args.train_batch_size,
        "task_count": args.task_count,
        "dataset_indices": sorted(expected_indices),
        "valid_rows": len(valid_records),
        "parent_row_counts": {
            str(parent): len(records)
            for parent, records in sorted(rows_by_parent.items())
        },
        "action_kind_counts": dict(action_counts),
        "terminal_success_count": len(terminal_successes),
        "terminal_success_parent_indices": sorted(terminal_successes),
        "nonzero_advantage_rows": nonzero_advantage_rows,
        "nonzero_return_rows": nonzero_return_rows,
        "compaction_required": args.require_compaction,
        "truncated_rows": truncated_rows,
        "endpoint_probe": endpoint_probe,
        "readback": readback,
        "metadata_after": metadata_after,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
