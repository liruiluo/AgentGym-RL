"""Audit direct-summary CompactionRL evidence in task-neutral action rows."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

COMPACTIONRL_EVIDENCE_SUMMARY_SCHEMA = "camg_compactionrl_evidence_summary_v1"
COMPACTIONRL_RECEIPT_SCHEMA = "agentmemory_compactionrl_receipt_v1"
TASK_NEUTRAL_ACTION_ROW_SCHEMA = "amg_task_neutral_action_row_v1"
TASK_NEUTRAL_CONTEXT_SCHEMA = "agentmemory_task_neutral_context_transition_v1"
ALLOWED_INVALID_SUMMARY_REASONS = frozenset(
    {"empty_summary", "summary_too_large"}
)


def summarize_compactionrl_step_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_routes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a fail-closed aggregate of CompactionRL summary transitions.

    A route may legitimately finish every held-out episode before the context
    threshold, so zero summaries is reported rather than treated as a protocol
    failure.  Any row that claims to be a CompactionRL summary must satisfy the
    complete no-native-dispatch, no-shaping, task-neutral replacement contract.
    """

    route_rows: Counter[str] = Counter()
    route_trajectories: dict[str, set[str]] = defaultdict(set)
    route_compactions: Counter[str] = Counter()
    route_invalid_compactions: Counter[str] = Counter()
    route_invalid_summary_tokens: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    route_summary_tokens: Counter[str] = Counter()
    route_successors: Counter[str] = Counter()
    route_terminal_compactions: Counter[str] = Counter()
    route_horizon_overlays: Counter[str] = Counter()
    terminal_outcomes: Counter[str] = Counter()
    trajectories: dict[str, dict[str, Any]] = {}
    trajectories_with_compaction: set[str] = set()
    violations: list[str] = []
    violation_count = 0

    def violate(message: str) -> None:
        nonlocal violation_count
        violation_count += 1
        if len(violations) < 200:
            violations.append(message)

    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            violate(f"record[{record_index}] is not a mapping")
            continue
        if record.get("schema") != TASK_NEUTRAL_ACTION_ROW_SCHEMA:
            violate(f"record[{record_index}] has an invalid action-row schema")

        trajectory_uid = record.get("trajectory_uid")
        if not isinstance(trajectory_uid, str) or not trajectory_uid:
            violate(f"record[{record_index}] has no trajectory_uid")
            trajectory_uid = f"<missing:{record_index}>"
        batch_identity = record.get(
            "_eval_batch_index", record.get("_rollout_update", "direct")
        )
        trajectory_key = f"{batch_identity}:{trajectory_uid}"

        route = record.get("route_id")
        data_source = record.get("data_source")
        if not isinstance(route, str) or not route:
            violate(f"record[{record_index}] has no route_id")
            route = "<missing>"
        if data_source != route:
            violate(
                f"trajectory {trajectory_uid} route/data_source disagree: "
                f"{route!r} != {data_source!r}"
            )
        route_rows[route] += 1
        route_trajectories[route].add(trajectory_key)

        state = trajectories.setdefault(
            trajectory_key,
            {"route": route, "last_row_order": -1, "rows": []},
        )
        if state["route"] != route:
            violate(
                f"trajectory {trajectory_uid} changed route "
                f"{state['route']!r}->{route!r}"
            )
        row_order = record.get("trajectory_row_order")
        if type(row_order) is not int or row_order < 0:
            violate(f"trajectory {trajectory_uid} has invalid row order {row_order!r}")
        elif row_order != state["last_row_order"] + 1:
            violate(
                f"trajectory {trajectory_uid} row order jumped "
                f"{state['last_row_order']}->{row_order}"
            )
        else:
            state["last_row_order"] = row_order
        state["rows"].append(record)

        transition = record.get("context_transition")
        if not isinstance(transition, Mapping):
            violate(
                f"trajectory {trajectory_uid} row {row_order}: missing context transition"
            )
            transition_operation = None
        else:
            if transition.get("schema") != TASK_NEUTRAL_CONTEXT_SCHEMA:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    "invalid context-transition schema"
                )
            transition_operation = transition.get("operation")
            if transition_operation not in {
                "append_observation",
                "preserve",
                "replace_messages",
            }:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    f"invalid context operation {transition_operation!r}"
                )

        evidence = record.get("wrapper_evidence")
        is_compaction = (
            isinstance(evidence, Mapping)
            and evidence.get("schema") == COMPACTIONRL_RECEIPT_SCHEMA
        )
        if is_compaction:
            trajectories_with_compaction.add(trajectory_key)
            if evidence.get("event") != "context_compaction":
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    "invalid compaction event"
                )
            if evidence.get("context_memory_mode") != "compactionrl":
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    "invalid context_memory_mode"
                )
            if evidence.get("summary_sent_to_native_environment") is not False:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    "summary was sent to the native environment"
                )
            if evidence.get("native_environment_call_count") != 0:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    "summary caused a native environment call"
                )
            if evidence.get("summary_specific_reward") is not False:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    "summary-specific reward shaping is present"
                )

            action = record.get("action")
            if not isinstance(action, str):
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    "summary action is not text"
                )
            else:
                encoded_action = action.encode("utf-8")
                expected_hash = hashlib.sha256(encoded_action).hexdigest()
                if evidence.get("summary_sha256") != expected_hash:
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "summary hash mismatch"
                    )
                if evidence.get("summary_byte_count") != len(encoded_action):
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "summary byte count mismatch"
                    )
            submission = record.get("action_submission")
            if not isinstance(submission, Mapping):
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    "missing action submission receipt"
                )
            else:
                if submission.get("raw_policy_output") != action:
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "summary raw policy output mismatch"
                    )
                if (
                    submission.get("parser_status")
                    != "compactionrl_summary_not_dispatched"
                ):
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "summary parser status drift"
                    )

            response_tokens = record.get("response_token_count")
            if type(response_tokens) is not int or response_tokens < 0:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    f"invalid response_token_count {response_tokens!r}"
                )
                response_tokens = 0

            summary_valid = evidence.get("summary_valid")
            if summary_valid is True:
                route_compactions[route] += 1
                route_summary_tokens[route] += response_tokens
                if transition_operation != "replace_messages":
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "valid summary lacks replace_messages transition"
                    )
                if evidence.get("context_replaced") is not True:
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "valid summary did not replace context"
                    )
                if evidence.get("summary_failure_reason") is not None:
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "valid summary has a failure reason"
                    )
                if evidence.get("retry_pending") is not False:
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "valid summary left retry pending"
                    )
            elif summary_valid is False:
                route_invalid_compactions[route] += 1
                route_invalid_summary_tokens[route] += response_tokens
                reason = evidence.get("summary_failure_reason")
                if reason not in ALLOWED_INVALID_SUMMARY_REASONS:
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        f"unknown invalid-summary reason {reason!r}"
                    )
                else:
                    invalid_reasons[str(reason)] += 1
                if transition_operation != "preserve":
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "invalid summary changed context"
                    )
                if evidence.get("context_replaced") is not False:
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "invalid summary claims context replacement"
                    )
                if evidence.get("retry_pending") is not True:
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "invalid summary did not request retry"
                    )
                if evidence.get("pre_context_message_count") != evidence.get(
                    "post_context_message_count"
                ):
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "invalid summary changed message count"
                    )
                if evidence.get("pre_context_sha256") != evidence.get(
                    "post_context_sha256"
                ):
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "invalid summary changed context bytes"
                    )
                if evidence.get("retained_recent_steps") != 0:
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "invalid summary retained a context tail"
                    )
                if isinstance(action, str) and reason == "empty_summary" and action.strip():
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "empty-summary receipt contains non-whitespace text"
                    )
                if isinstance(action, str) and reason == "summary_too_large":
                    byte_count = evidence.get("summary_byte_count")
                    max_bytes = evidence.get("summary_max_bytes")
                    if (
                        type(byte_count) is not int
                        or type(max_bytes) is not int
                        or byte_count <= max_bytes
                    ):
                        violate(
                            f"trajectory {trajectory_uid} row {row_order}: "
                            "oversize receipt is within byte cap"
                        )
            else:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    "summary_valid is not boolean"
                )

            terminal = record.get("trajectory_terminal") is True
            if terminal and summary_valid is True:
                route_terminal_compactions[route] += 1
            reward = record.get("immediate_reward", 0.0)
            try:
                nonzero_reward = float(reward) != 0.0
            except (TypeError, ValueError):
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: "
                    f"invalid immediate reward {reward!r}"
                )
                nonzero_reward = False
            if nonzero_reward:
                if not terminal or not isinstance(
                    record.get("horizon_finalization"), Mapping
                ):
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: "
                        "nonzero summary reward lacks terminal horizon overlay"
                    )
                else:
                    route_horizon_overlays[route] += 1
        elif transition_operation == "replace_messages":
            violate(
                f"trajectory {trajectory_uid} row {row_order}: "
                "replace_messages lacks CompactionRL receipt"
            )

        if record.get("trajectory_terminal") is True:
            terminal_outcomes[str(record.get("outcome"))] += 1

    for trajectory_key, state in trajectories.items():
        rows = state["rows"]
        for index, record in enumerate(rows):
            evidence = record.get("wrapper_evidence")
            if not (
                isinstance(evidence, Mapping)
                and evidence.get("schema") == COMPACTIONRL_RECEIPT_SCHEMA
            ):
                continue
            route = state["route"]
            if evidence.get("summary_valid") is True and index + 1 < len(rows):
                route_successors[route] += 1

    expected = sorted(set(str(route) for route in expected_routes))
    missing_routes = sorted(set(expected) - set(route_rows))
    if not trajectories:
        violate("no step records were provided")
    if missing_routes:
        violate("missing expected routes: " + ", ".join(missing_routes))

    route_names = sorted(set(route_rows) | set(expected))
    routes = {
        route: {
            "rows": route_rows[route],
            "trajectories": len(route_trajectories[route]),
            "valid_compactions": route_compactions[route],
            "invalid_compactions": route_invalid_compactions[route],
            "summary_response_tokens": route_summary_tokens[route],
            "invalid_summary_response_tokens": route_invalid_summary_tokens[route],
            "compactions_with_successor_policy_row": route_successors[route],
            "terminal_compaction_rows": route_terminal_compactions[route],
            "horizon_reward_overlay_rows": route_horizon_overlays[route],
        }
        for route in route_names
    }
    return {
        "schema": COMPACTIONRL_EVIDENCE_SUMMARY_SCHEMA,
        "status": "PASS" if violation_count == 0 else "FAIL",
        "expected_routes": expected,
        "missing_routes": missing_routes,
        "routes_without_valid_compaction": sorted(
            route for route in expected if route_compactions[route] == 0
        ),
        "totals": {
            "rows": sum(route_rows.values()),
            "trajectories": len(trajectories),
            "valid_compactions": sum(route_compactions.values()),
            "invalid_compactions": sum(route_invalid_compactions.values()),
            "summary_response_tokens": sum(route_summary_tokens.values()),
            "invalid_summary_response_tokens": sum(
                route_invalid_summary_tokens.values()
            ),
            "trajectories_with_compaction": len(trajectories_with_compaction),
            "compactions_with_successor_policy_row": sum(route_successors.values()),
            "terminal_compaction_rows": sum(route_terminal_compactions.values()),
            "horizon_reward_overlay_rows": sum(route_horizon_overlays.values()),
        },
        "routes": routes,
        "terminal_outcomes": dict(sorted(terminal_outcomes.items())),
        "invalid_summary_reasons": dict(sorted(invalid_reasons.items())),
        "violation_count": violation_count,
        "violations": violations,
    }


__all__ = [
    "COMPACTIONRL_EVIDENCE_SUMMARY_SCHEMA",
    "COMPACTIONRL_RECEIPT_SCHEMA",
    "summarize_compactionrl_step_records",
]
