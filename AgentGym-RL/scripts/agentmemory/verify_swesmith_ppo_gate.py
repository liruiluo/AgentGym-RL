#!/usr/bin/env python3
"""Verify a declared eight-rank SWE-smith PPO endpoint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any


STEP_SCHEMA = "task_neutral_policy_step_v1"
AUDIT_SCHEMA = "agentmemory_swesmith_private_episode_audit_v1"
ENDPOINT_PROBE_SCHEMA = "agentmemory_swesmith_resident_endpoint_probe_v2"
NATIVE_EVENT = "native_action"
COMPACTION_EVENT = "context_compaction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--endpoint-probe", type=Path, required=True)
    parser.add_argument("--metadata-after", type=Path, required=True)
    parser.add_argument("--global-step", type=int, default=1)
    parser.add_argument("--first-global-step", type=int, default=1)
    parser.add_argument(
        "--parent-run-dir",
        type=Path,
        help=(
            "Optional verified parent run for a resumed segment. When the "
            "current segment contains only zero-return batches, its earlier "
            "post-advantage diagnostics may supply the return-signal evidence."
        ),
    )
    parser.add_argument(
        "--allow-zero-resolved",
        action="store_true",
        help=(
            "Allow the current resumed batch/segment to have zero resolved "
            "audits, provided parent-run return evidence is supplied."
        ),
    )
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--task-count", type=int, default=8)
    parser.add_argument(
        "--routing-file",
        type=Path,
        help=(
            "Optional ordered JSONL routing schedule. When set, PPO parent indices "
            "remain local to each batch while item_id and endpoint data_idx are "
            "verified against the declared global schedule."
        ),
    )
    parser.add_argument(
        "--endpoint-probe-indices",
        default="0,1,2,3,4,5,6,7",
    )
    parser.add_argument(
        "--require-compaction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help=(
            "Verify a completed PPO prefix while the trainer may already be "
            "processing the next disjoint full-pool batch."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time(value: str, label: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise AssertionError(f"{label} is not an ISO-8601 timestamp: {value!r}") from exc
    if result.tzinfo is None:
        raise AssertionError(f"{label} must include a timezone: {value!r}")
    return result


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"{label} is not finite: {value!r}")
    return result


def optimizer_update_count(first_global_step: int, global_step: int) -> int:
    assert first_global_step > 0
    assert global_step >= first_global_step
    return global_step - first_global_step + 1


def load_routing_contract(
    path: Path,
    *,
    train_batch_size: int,
    first_global_step: int,
    global_step: int,
) -> dict[str, Any]:
    """Bind a no-shuffle PPO segment to its ordered endpoint routing rows."""
    assert train_batch_size > 0
    optimizer_update_count(first_global_step, global_step)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"routing row {line_number} is not valid JSON"
            ) from exc
        assert isinstance(row, dict)
        position = len(rows)
        assert row["item_id"] == f"swesmith_{position}"
        data_idx = row["data_idx"]
        assert isinstance(data_idx, int) and not isinstance(data_idx, bool)
        assert data_idx >= 0
        extra_info = row["extra_info"]
        assert isinstance(extra_info, dict)
        assert int(extra_info["index"]) == data_idx
        assert int(extra_info["schedule_position"]) == position
        rows.append(row)

    required_rows = global_step * train_batch_size
    assert len(rows) >= required_rows
    segment_start = (first_global_step - 1) * train_batch_size
    segment_rows = rows[segment_start:required_rows]
    current_rows = rows[required_rows - train_batch_size : required_rows]
    assert len(current_rows) == train_batch_size
    assert len(segment_rows) == (
        optimizer_update_count(first_global_step, global_step) * train_batch_size
    )

    current_item_ids = {
        parent_index: str(row["item_id"])
        for parent_index, row in enumerate(current_rows)
    }
    segment_data_idx_counts = Counter(int(row["data_idx"]) for row in segment_rows)
    return {
        "routing_row_count": len(rows),
        "training_data_indices": {int(row["data_idx"]) for row in rows},
        "segment_schedule_positions": [
            int(row["extra_info"]["schedule_position"]) for row in segment_rows
        ],
        "current_schedule_positions": [
            int(row["extra_info"]["schedule_position"]) for row in current_rows
        ],
        "current_item_ids": current_item_ids,
        "current_data_indices": [int(row["data_idx"]) for row in current_rows],
        "segment_data_idx_counts": segment_data_idx_counts,
    }


def count_nonzero_return_rows(payload: dict[str, Any]) -> int:
    return sum(
        int(row.get("return_nonzero", 0) > 0)
        for row in payload["rows"]
        if row["ppo_valid_sample"]
    )


def verify_segment_return_signal(
    run_dir: Path,
    *,
    first_global_step: int,
    global_step: int,
    endpoint_payload: dict[str, Any],
    endpoint_nonzero_return_rows: int,
    parent_run_dir: Path | None = None,
) -> dict[str, Any]:
    """Find real PPO return evidence anywhere in the declared update segment."""
    assert int(endpoint_payload["global_step"]) == global_step
    assert endpoint_payload["stage"] == "post_adv"
    assert endpoint_nonzero_return_rows == count_nonzero_return_rows(endpoint_payload)
    if endpoint_nonzero_return_rows > 0:
        return {
            "source_global_step": global_step,
            "nonzero_return_rows": endpoint_nonzero_return_rows,
            "endpoint_has_return_signal": True,
        }

    for step in range(first_global_step, global_step):
        path = run_dir / "diagnostics" / f"ppo_batch_step{step}_post_adv.json"
        payload = load_json(path)
        assert int(payload["global_step"]) == step
        assert payload["stage"] == "post_adv"
        nonzero_return_rows = count_nonzero_return_rows(payload)
        if nonzero_return_rows > 0:
            return {
                "source_global_step": step,
                "nonzero_return_rows": nonzero_return_rows,
                "endpoint_has_return_signal": False,
            }

    if parent_run_dir is not None:
        parent_candidates = sorted(
            parent_run_dir.glob("diagnostics/ppo_batch_step*_post_adv.json"),
            key=lambda path: int(path.name.split("step", 1)[1].split("_", 1)[0]),
        )
        for path in parent_candidates:
            payload = load_json(path)
            assert payload["stage"] == "post_adv"
            nonzero_return_rows = count_nonzero_return_rows(payload)
            if nonzero_return_rows > 0:
                return {
                    "source_global_step": int(payload["global_step"]),
                    "nonzero_return_rows": nonzero_return_rows,
                    "endpoint_has_return_signal": False,
                    "source": "verified_parent_run",
                    "parent_run_dir": str(parent_run_dir),
                }

    raise AssertionError(
        "declared optimizer-update segment has no nonzero PPO return rows"
    )


def parse_endpoint_probe_indices(raw: str) -> list[int]:
    try:
        indices = [int(value) for value in raw.split(",") if value != ""]
    except ValueError as exc:
        raise ValueError("endpoint probe indices must be comma-separated integers") from exc
    if len(indices) != 8:
        raise ValueError("the endpoint isolation probe requires exactly 8 indices")
    if any(index < 0 for index in indices):
        raise ValueError("endpoint probe indices must be nonnegative")
    if len(set(indices)) != len(indices):
        raise ValueError("endpoint probe indices must be distinct")
    return indices


def parse_endpoint_probe_slots(endpoint_probe: dict[str, Any]) -> list[int]:
    slots = [int(value) for value in endpoint_probe["slot_ids"]]
    assert len(slots) == 8
    assert all(slot >= 0 for slot in slots)
    assert len(set(slots)) == len(slots)
    return slots


def verify_endpoint_submission_contract_probe(
    endpoint_probe: dict[str, Any], slots: list[int]
) -> dict[str, Any]:
    """Require live evidence for submit, rejection, and horizon semantics."""
    assert endpoint_probe["schema"] == ENDPOINT_PROBE_SCHEMA
    probe = endpoint_probe["submission_contract_probe"]
    assert isinstance(probe, dict)
    horizon_slot = int(probe["horizon_slot_id"])
    valid_slot = int(probe["valid_submission_slot_id"])
    assert horizon_slot in slots
    assert valid_slot in slots
    assert horizon_slot != valid_slot
    assert probe["horizon_grade"] is None
    assert probe["non_first_line_done"] is False
    assert probe["nonzero_exit_done"] is False
    assert probe["valid_submission_grade_present"] is True
    assert float(probe["valid_submission_reward"]) in {0.0, 1.0}
    return {
        "horizon_slot_id": horizon_slot,
        "valid_submission_slot_id": valid_slot,
        "valid_submission_reward": float(probe["valid_submission_reward"]),
    }


def verify_endpoint_probe_indices(
    endpoint_probe: dict[str, Any],
    expected_probe_indices: list[int],
    expected_training_indices: set[int],
) -> None:
    actual = [int(value) for value in endpoint_probe["indices"]]
    assert actual == expected_probe_indices
    assert set(actual).issubset(expected_training_indices)


def verify_event_coverage(
    event_counts: Counter[str],
    action_kind_counts: Counter[str],
    *,
    require_compaction: bool,
) -> None:
    assert event_counts[NATIVE_EVENT] > 0
    if require_compaction:
        assert event_counts[COMPACTION_EVENT] > 0
    assert (
        action_kind_counts["shell_command"] + action_kind_counts["apply_patch"]
        > 0
    )


def verify_response_cap_truncation(
    record: dict[str, Any], *, parent_index: int
) -> dict[str, Any]:
    """Accept only an exact backend length stop as a trainable negative row."""
    response_token_count = int(record["response_token_count"])
    max_response_tokens = int(record["max_response_tokens"])
    assert record["truncated"] is True
    assert record["finish_reason"] == "length"
    assert str(record["finish_reason_source"]).endswith(":backend")
    assert record.get("generation_stop_reason") is None
    assert record.get("stop_reason") is None
    assert max_response_tokens > 0
    assert response_token_count == max_response_tokens
    assert int(record["generation_response_length"]) == response_token_count
    assert int(record["packed_response_length"]) == response_token_count
    assert record["generation_token_ids_are_exact"] is True
    assert record["backend_token_ids_are_exact"] is True
    assert record["done"] is False
    assert record["trajectory_terminal"] is False
    assert record["outcome"] == "continue"
    assert float(record["immediate_reward"]) == 0.0
    return {
        "kind": "exact_backend_response_cap",
        "parent_index": parent_index,
        "task_round": int(record["task_round"]),
        "response_token_count": response_token_count,
        "max_response_tokens": max_response_tokens,
        "finish_reason": record["finish_reason"],
        "finish_reason_source": record["finish_reason_source"],
    }


def verify_wrapper_transition(
    record: dict[str, Any], *, previous_native_step: int | None
) -> dict[str, Any]:
    evidence = record["wrapper_evidence"]
    event = str(evidence["event"])
    assert event in {NATIVE_EVENT, COMPACTION_EVENT}
    native_before = int(record["env_info_before"]["step"])
    native_after = int(record["env_info_after"]["step"])
    if previous_native_step is not None:
        assert native_before == previous_native_step

    transition = record["context_transition"]
    assert transition["schema"] == "agentmemory_task_neutral_context_transition_v1"
    action_kind: str | None = None
    if event == COMPACTION_EVENT:
        assert native_after == native_before
        assert transition["operation"] == "replace_messages"
        assert transition["messages"]
        assert record["action_submission"]["submitted_action"] is None
        assert record["action_submission"]["parser_status"] == (
            "policy_context_compaction"
        )
    else:
        assert native_after == native_before + 1
        assert transition["operation"] == "append_observation"
        action_kind = str(record["env_info_after"].get("action_kind", ""))
        assert action_kind in {
            "shell_command",
            "apply_patch",
            "final",
            "parser_error",
            "policy_turn_horizon",
        }
        trajectory_terminal = bool(record["trajectory_terminal"])
        assert bool(record["done"]) == trajectory_terminal
        immediate_reward = float(record["immediate_reward"])
        if not trajectory_terminal:
            assert record["outcome"] == "continue"
            assert immediate_reward == 0.0
        else:
            assert immediate_reward in {0.0, 1.0}
            assert record["outcome"] == (
                "success" if immediate_reward == 1.0 else "terminal_failure"
            )
            horizon_finalization = record.get("horizon_finalization")
            if horizon_finalization is None:
                # Only an explicit sentinel submission is graded by env.step.
                assert bool(record["env_info_after"]["episode_success"]) == (
                    immediate_reward == 1.0
                )
                if immediate_reward == 1.0:
                    assert record["env_info_after"]["terminal"] is True
            else:
                # At the policy-turn boundary the last sampled action is kept
                # as the trainable terminal row, while the wrapper records a
                # no-submission horizon failure without invoking the grader.
                assert isinstance(horizon_finalization, dict)
                assert immediate_reward == 0.0
                assert record["outcome"] == "terminal_failure"
                assert float(horizon_finalization["reward"]) == immediate_reward
                assert horizon_finalization["done"] is True
                horizon_info = horizon_finalization["info"]
                assert horizon_info["action_submission"]["control_action"] == "horizon"
                assert horizon_info["wrapper_evidence"]["event"] == "horizon_finalization"
                assert int(horizon_info["native_step_before"]) == native_after
                assert int(horizon_info["native_step_after"]) == native_after
                assert int(horizon_info["policy_step_after"]) == int(
                    record["task_round"]
                )
                terminal_info = horizon_info["env_info"]
                assert int(terminal_info["step"]) == native_after
                assert terminal_info["terminal"] is True
                assert terminal_info["action_kind"] == "policy_turn_horizon"
                assert bool(terminal_info["episode_success"]) is False
                assert record["env_info_after"]["terminal"] is False
                assert bool(record["env_info_after"]["episode_success"]) is False

    return {
        "event": event,
        "workspace_continuity_id": int(evidence["workspace_continuity_id"]),
        "native_step_after": native_after,
        "action_kind": action_kind,
    }


def verify_row_evidence(
    payload: dict[str, Any],
    expected_indices: set[int],
    expected_item_ids: dict[int, str] | None = None,
) -> dict[str, Any]:
    evidence = payload["row_evidence"]
    assert evidence["task_name"] == "swesmith"
    schema = evidence["schema"]
    if schema == "agentmemory_formal_step_records_v1":
        rows = evidence["rows"]
        assert rows
        assert payload["formal_step_records"] == rows
        indices = [int(row["parent_index"]) for row in rows]
        assert set(indices) == expected_indices
        for row, index in zip(rows, indices):
            assert row["schema_version"] == STEP_SCHEMA
            expected_item_id = (
                f"swesmith_{index}"
                if expected_item_ids is None
                else expected_item_ids[index]
            )
            assert row["item_id"] == expected_item_id
        return {
            "schema": schema,
            "dataset_indices": sorted(set(indices)),
            "row_count": len(rows),
        }
    if schema == "generic_task_dataset_rows_v1":
        assert evidence["index_field"] in {
            "rollout_data_indices",
            "data_idx",
            "index",
        }
        indices = [int(value) for value in evidence["dataset_indices"]]
        assert set(indices) == expected_indices
        return {
            "schema": schema,
            "dataset_indices": sorted(set(indices)),
            "row_count": len(indices),
        }
    raise AssertionError(f"unexpected row evidence schema: {schema!r}")


def verify_readback(
    run_dir: Path,
    global_step: int,
    expected_indices: set[int],
    expected_item_ids: dict[int, str] | None = None,
) -> dict[str, Any]:
    path = run_dir / "diagnostics" / f"formal_update_readback_step{global_step}.json"
    payload = load_json(path)
    assert int(payload["global_step"]) == global_step
    assert payload["role"] == "same_batch_post_optimizer_readback"
    result: dict[str, Any] = {
        "row_evidence": verify_row_evidence(
            payload,
            expected_indices,
            expected_item_ids,
        )
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
        assert max_abs_delta > 0
        assert delta_l2 > 0
        assert changed > 0
        result[role] = {
            "max_abs_delta": max_abs_delta,
            "parameter_delta_l2": delta_l2,
            "parameter_probe_changed_count": changed,
        }
    return result


def verify_audits(
    audit_root: Path,
    endpoint_probe: dict[str, Any],
    expected_indices: set[int],
    run_started_at: datetime,
    *,
    expected_audit_count: int | None = None,
    expected_data_idx_counts: Counter[int] | None = None,
    expected_slot_counts: Counter[int] | None = None,
    expected_slot_cardinality: int | None = None,
    allowed_future_indices: set[int] | None = None,
) -> dict[str, Any]:
    probe_audit_ids = set(str(value) for value in endpoint_probe["audit_ids"])
    trainer_audits: list[dict[str, Any]] = []
    ungraded_terminal_rejections: list[dict[str, Any]] = []
    ungraded_horizon_exhaustions: list[dict[str, Any]] = []
    stale_audit_count = 0
    future_distinct_audit_count = 0
    for path in sorted(audit_root.glob("episode-*.json")):
        payload = load_json(path)
        if str(payload["audit_id"]) in probe_audit_ids:
            continue
        audit_started_at = parse_time(
            str(payload.get("started_at", "")), f"{path.name}.started_at"
        )
        if audit_started_at < run_started_at:
            stale_audit_count += 1
            continue
        if int(payload["data_idx"]) not in expected_indices:
            if (
                allowed_future_indices is not None
                and int(payload["data_idx"]) in allowed_future_indices
            ):
                future_distinct_audit_count += 1
                continue
            raise AssertionError(
                f"unexpected in-run audit data_idx: {payload['data_idx']!r}"
            )
        assert payload["schema"] == AUDIT_SCHEMA
        assert payload["close_reason"] == "client_close"
        assert payload["done"] is True
        assert float(payload["reward"]) in {0.0, 1.0}
        assert int(payload["step_count"]) > 0
        if payload["grade"] is None:
            assert float(payload["reward"]) == 0.0
            last_event = payload["evidence"][-1]
            if last_event["event"] == "horizon_exhaustion":
                action_kind = str(last_event["action"]["kind"])
                assert action_kind in {"horizon", "policy_turn_horizon"}
                assert last_event["termination_reason"] in {
                    "max_steps",
                    "policy_turn_horizon",
                }
                assert last_event["terminal_grade"] == {
                    "reward": 0.0,
                    "resolved": False,
                    "grader_error": None,
                    "graded": False,
                }
                ungraded_horizon_exhaustions.append({
                    "audit_id": str(payload["audit_id"]),
                    "data_idx": int(payload["data_idx"]),
                    "slot_id": int(payload["slot_id"]),
                    "step_count": int(payload["step_count"]),
                    "action_kind": action_kind,
                    "termination_reason": last_event["termination_reason"],
                })
            else:
                assert last_event["event"] == "policy_step"
                actor_credit = last_event["actor_credit"]
                assert actor_credit == {
                    "schema": "task_neutral_actor_credit_v1",
                    "positive_eligible": False,
                    "basis": "executor_rejected",
                }
                action_kind = str(last_event["action"]["kind"])
                assert action_kind in {"shell_command", "apply_patch"}
                assert str(last_event["observation_after"]).startswith(
                    f"{action_kind} failed:"
                )
                ungraded_terminal_rejections.append({
                    "audit_id": str(payload["audit_id"]),
                    "data_idx": int(payload["data_idx"]),
                    "slot_id": int(payload["slot_id"]),
                    "step_count": int(payload["step_count"]),
                    "actor_credit_basis": actor_credit["basis"],
                    "action_kind": action_kind,
                })
        trainer_audits.append(payload)
    if expected_audit_count is None:
        expected_audit_count = len(expected_indices)
    assert len(trainer_audits) == expected_audit_count
    indices = [int(value["data_idx"]) for value in trainer_audits]
    assert set(indices) == expected_indices
    observed_data_idx_counts = Counter(indices)
    if expected_data_idx_counts is None:
        assert len(indices) == len(set(indices))
    else:
        assert observed_data_idx_counts == expected_data_idx_counts
    observed_slot_counts = Counter(
        int(value["slot_id"]) for value in trainer_audits
    )
    if expected_slot_counts is None:
        if expected_slot_cardinality is None:
            expected_slot_cardinality = len(expected_indices)
        assert len(observed_slot_counts) == expected_slot_cardinality
        assert expected_audit_count % expected_slot_cardinality == 0
        expected_slot_frequency = expected_audit_count // expected_slot_cardinality
        assert set(observed_slot_counts.values()) == {expected_slot_frequency}
    else:
        assert observed_slot_counts == expected_slot_counts
    return {
        "audit_count": len(trainer_audits),
        "dataset_indices": sorted(set(indices)),
        "data_idx_counts": {
            str(index): count
            for index, count in sorted(observed_data_idx_counts.items())
        },
        "slot_counts": {
            str(slot): count
            for slot, count in sorted(observed_slot_counts.items())
        },
        "resolved_count": sum(float(value["reward"]) == 1.0 for value in trainer_audits),
        "graded_audit_count": sum(value["grade"] is not None for value in trainer_audits),
        "ungraded_terminal_rejections": ungraded_terminal_rejections,
        "ungraded_horizon_exhaustions": ungraded_horizon_exhaustions,
        "audit_ids": sorted(str(value["audit_id"]) for value in trainer_audits),
        "selection": "run-start-time-minus-current-probe",
        "run_started_at": run_started_at.isoformat(),
        "stale_audit_count": stale_audit_count,
        "future_distinct_audit_count": future_distinct_audit_count,
    }


def main() -> None:
    args = parse_args()
    assert args.global_step > 0
    segment_update_count = optimizer_update_count(
        args.first_global_step,
        args.global_step,
    )
    assert args.train_batch_size > 0
    assert args.task_count > 0
    expected_parent_indices = set(range(args.train_batch_size))
    routing_contract = None
    allowed_future_indices: set[int] | None = None
    endpoint_training_indices: set[int] | None = None
    expected_current_item_ids = {
        index: f"swesmith_{index}" for index in expected_parent_indices
    }
    if args.routing_file is None:
        assert args.train_batch_size % args.task_count == 0
        expected_data_indices = set(range(args.task_count))
        expected_data_idx_counts = Counter({
            index: segment_update_count * args.train_batch_size // args.task_count
            for index in expected_data_indices
        })
        endpoint_training_indices = expected_data_indices
    else:
        routing_contract = load_routing_contract(
            args.routing_file,
            train_batch_size=args.train_batch_size,
            first_global_step=args.first_global_step,
            global_step=args.global_step,
        )
        assert args.task_count == routing_contract["routing_row_count"]
        expected_data_idx_counts = routing_contract["segment_data_idx_counts"]
        expected_data_indices = set(expected_data_idx_counts)
        endpoint_training_indices = routing_contract["training_data_indices"]
        expected_current_item_ids = routing_contract["current_item_ids"]
        if args.online:
            future_rows = []
            for line in args.routing_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    future_rows.append(json.loads(line))
            future_start = args.global_step * args.train_batch_size
            allowed_future_indices = {
                int(row["data_idx"]) for row in future_rows[future_start:]
            }
    assert endpoint_training_indices is not None
    endpoint_probe = load_json(args.endpoint_probe)
    assert endpoint_probe["status"] == "pass"
    endpoint_probe_slots = parse_endpoint_probe_slots(endpoint_probe)
    endpoint_submission_probe = verify_endpoint_submission_contract_probe(
        endpoint_probe, endpoint_probe_slots
    )
    verify_endpoint_probe_indices(
        endpoint_probe,
        parse_endpoint_probe_indices(args.endpoint_probe_indices),
        endpoint_training_indices,
    )
    # The resident manager allocates a fresh service-local slot for every
    # episode and never reuses a closed slot.  Slot IDs may have gaps because
    # the endpoint probe and other internal reservations use the same counter.
    # Verify one unique slot per selected audit instead of assuming that the
    # train batch owns a fixed reusable slot range.
    expected_slot_counts = None
    expected_audit_count = segment_update_count * args.train_batch_size

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
    event_counts: Counter[str] = Counter()
    action_kind_counts: Counter[str] = Counter()
    nonzero_advantage_rows = 0
    nonzero_return_rows = 0
    truncated_rows = []

    for row in payload["rows"]:
        if not row["ppo_valid_sample"]:
            continue
        parent_index = int(row["parent_index"])
        assert parent_index in expected_parent_indices
        record = row["formal_step_record"]
        assert record["schema_version"] == STEP_SCHEMA
        assert int(record["parent_index"]) == parent_index
        assert int(record["task_round"]) == int(row["task_round"])
        assert record["item_id"] == expected_current_item_ids[parent_index]
        assert record["action"] == row["agentmemory_action_text"]
        assert record["action_submission"]["raw_policy_output"] == record["action"]
        assert record["generation_token_ids_are_exact"] is True
        assert record["backend_token_ids_are_exact"] is True
        assert record["response_token_ids"]
        assert int(record["response_token_count"]) == len(record["response_token_ids"])
        assert int(row["response_mask_sum"]) == len(record["response_token_ids"])
        assert record["generation_prompt_digest"] == row[
            "agentmemory_generation_prompt_digest"
        ]
        assert record["packed_prompt_digest"] == row[
            "agentmemory_packed_prompt_digest"
        ]
        assert abs(
            finite(row["score_sum"], "score_sum")
            - finite(record["immediate_reward"], "immediate_reward")
        ) < 1e-6
        finite(row["old_logprob_mean"], "old_logprob_mean")
        nonzero_advantage_rows += int(row.get("adv_nonzero", 0) > 0)
        nonzero_return_rows += int(row.get("return_nonzero", 0) > 0)
        if record["truncated"]:
            truncated_rows.append(
                verify_response_cap_truncation(
                    record,
                    parent_index=parent_index,
                )
            )
        rows_by_parent[parent_index].append(record)

    assert set(rows_by_parent) == expected_parent_indices
    assert nonzero_advantage_rows > 0
    segment_return_signal = verify_segment_return_signal(
        args.run_dir,
        first_global_step=args.first_global_step,
        global_step=args.global_step,
        endpoint_payload=payload,
        endpoint_nonzero_return_rows=nonzero_return_rows,
        parent_run_dir=args.parent_run_dir,
    )

    workspace_ids: dict[int, int] = {}
    for parent_index, records in sorted(rows_by_parent.items()):
        records.sort(key=lambda value: int(value["task_round"]))
        assert [int(value["task_round"]) for value in records] == list(
            range(1, len(records) + 1)
        )
        previous_native_step: int | None = None
        for record in records:
            transition = verify_wrapper_transition(
                record,
                previous_native_step=previous_native_step,
            )
            event_counts[transition["event"]] += 1
            workspace_id = int(transition["workspace_continuity_id"])
            workspace_ids.setdefault(parent_index, workspace_id)
            assert workspace_ids[parent_index] == workspace_id
            previous_native_step = int(transition["native_step_after"])
            if transition["action_kind"] is not None:
                action_kind_counts[str(transition["action_kind"])] += 1

    assert len(set(workspace_ids.values())) == args.train_batch_size
    verify_event_coverage(
        event_counts,
        action_kind_counts,
        require_compaction=args.require_compaction,
    )

    metadata_after = load_json(args.metadata_after)
    for key in (
        "active_slot_count",
        "active_environment_count",
        "active_workspace_count",
    ):
        value = int(metadata_after[key])
        assert value >= 0
        if not args.online:
            assert value == 0

    readback = verify_readback(
        args.run_dir,
        args.global_step,
        expected_parent_indices,
        expected_current_item_ids,
    )
    run_started_at = parse_time(
        (args.run_dir / "started_at").read_text(encoding="utf-8"),
        "run.started_at",
    )
    audits = verify_audits(
        args.audit_root,
        endpoint_probe,
        expected_data_indices,
        run_started_at,
        expected_audit_count=expected_audit_count,
        expected_data_idx_counts=expected_data_idx_counts,
        expected_slot_counts=expected_slot_counts,
        expected_slot_cardinality=expected_audit_count,
        allowed_future_indices=allowed_future_indices,
    )
    if not args.allow_zero_resolved:
        assert int(audits["resolved_count"]) > 0
    else:
        assert args.parent_run_dir is not None
        assert segment_return_signal.get("source") == "verified_parent_run" or int(
            audits["resolved_count"]
        ) > 0
    evidence = {
        "schema": "agentmemory_swesmith_ppo_gate_attestation_v1",
        "status": "pass",
        "global_step": args.global_step,
        "first_global_step": args.first_global_step,
        "optimizer_update_count": segment_update_count,
        "cumulative_optimizer_update_count": args.global_step,
        "train_batch_size": args.train_batch_size,
        "task_count": args.task_count,
        "sampling_contract": (
            "ordered_routing_without_shuffle_v1"
            if routing_contract is not None
            else "fixed_task_panel_v1"
        ),
        "routing": (
            None
            if routing_contract is None
            else {
                "path": str(args.routing_file),
                "routing_row_count": routing_contract["routing_row_count"],
                "segment_schedule_position_first": routing_contract[
                    "segment_schedule_positions"
                ][0],
                "segment_schedule_position_last": routing_contract[
                    "segment_schedule_positions"
                ][-1],
                "current_schedule_positions": routing_contract[
                    "current_schedule_positions"
                ],
                "current_data_indices": routing_contract[
                    "current_data_indices"
                ],
            }
        ),
        "valid_rows": int(payload["valid_rows"]),
        "parent_row_counts": {
            str(parent): len(records) for parent, records in sorted(rows_by_parent.items())
        },
        "event_counts": dict(event_counts),
        "action_kind_counts": dict(action_kind_counts),
        "compaction_required": args.require_compaction,
        "online_prefix_verification": args.online,
        "allow_zero_resolved": args.allow_zero_resolved,
        "workspace_continuity_ids": workspace_ids,
        "truncation_contract": "exact_backend_response_cap_is_trainable_negative_v1",
        "truncated_rows": truncated_rows,
        "endpoint_nonzero_return_rows": nonzero_return_rows,
        "segment_return_signal": segment_return_signal,
        "endpoint_submission_contract_probe": endpoint_submission_probe,
        "readback": readback,
        "private_audits": audits,
        "metadata_after": metadata_after,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
