"""Method-aware evidence validation for CAMG held-out baselines.

The held-out runner is shared by trained CAMG policies and frozen baselines.
Method-specific state remains in environment-client adapters; this module only
audits the task-neutral action-row receipts emitted by those adapters.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .agemem_evidence import summarize_agemem_step_records
from .action_budget import normalize_action_budget_receipt


METHOD_EVIDENCE_SCHEMA = "camg_heldout_method_evidence_summary_v1"
SUPPORTED_METHOD_IDS = ("agemem", "qwen35_4b", "mem0", "letta_code")

_CONTEXT_SCHEMA = "agentmemory_task_neutral_context_transition_v1"
_CONTEXT_OPERATIONS = {"append_observation", "preserve", "replace_messages"}
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MEM0_SOURCE_REVISION = "71fba8d46436f88569d600f81a55208c38ad30b5"
_MEM0_VERSION = "2.0.19"
_LETTA_SOURCE_REVISION = "787b856f9db9f5030dc2976618e1d1f909f61612"
_LETTA_OPERATIONS = {
    "str_replace",
    "insert",
    "delete",
    "rename",
    "update_description",
    "create",
    "apply_patch",
}


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _generic_summary(
    records: Iterable[Mapping[str, Any]], *, method_id: str
) -> dict[str, Any]:
    rows = list(records)
    violations: list[str] = []
    operation_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    trajectories: set[str] = set()
    hidden_calls = 0
    hidden_input_tokens = 0
    hidden_output_tokens = 0
    hidden_latency_ms = 0
    method_rows = 0
    memory_rows = 0
    native_rows = 0
    commits = 0
    combined_steps = 0
    auxiliary_steps = 0
    budget_state: dict[str, tuple[int, int, bool]] = {}

    def violate(message: str) -> None:
        if len(violations) < 200:
            violations.append(message)

    for index, record in enumerate(rows):
        if not isinstance(record, Mapping):
            violate(f"record[{index}] is not a mapping")
            continue
        uid = record.get("trajectory_uid")
        if not isinstance(uid, str) or not uid:
            violate(f"record[{index}] has no trajectory_uid")
            uid = f"<missing:{index}>"
        trajectories.add(uid)
        raw_budget = record.get("action_budget")
        require_budget = method_id in {"mem0", "letta_code"}
        if raw_budget is not None or require_budget:
            if not isinstance(raw_budget, Mapping):
                violate(
                    f"trajectory {uid} row {index}: action-budget receipt is missing"
                )
                budget = None
            else:
                raw_maximum = raw_budget.get("maximum_steps")
                maximum = (
                    raw_maximum
                    if type(raw_maximum) is int and raw_maximum > 0
                    else 1
                )
                expected_before, expected_maximum, prior_terminated = budget_state.get(
                    uid, (0, maximum, False)
                )
                if prior_terminated:
                    violate(
                        f"trajectory {uid} row {index}: action follows a budget termination"
                    )
                try:
                    budget = normalize_action_budget_receipt(
                        raw_budget,
                        maximum_steps=expected_maximum,
                        consumed_steps_before=expected_before,
                        allow_implicit_policy_action=False,
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    violate(f"trajectory {uid} row {index}: {exc}")
                    budget = None
            if budget is not None:
                row_cost = int(budget["policy_action_steps"]) + int(
                    budget["auxiliary_steps"]
                )
                combined_steps += row_cost
                auxiliary_steps += int(budget["auxiliary_steps"])
                budget_state[uid] = (
                    int(budget["consumed_steps_after"]),
                    int(budget["maximum_steps"]),
                    bool(budget["terminate_after_action"]),
                )
        transition = record.get("context_transition")
        operation = transition.get("operation") if isinstance(transition, Mapping) else None
        if (
            not isinstance(transition, Mapping)
            or transition.get("schema") != _CONTEXT_SCHEMA
            or operation not in _CONTEXT_OPERATIONS
        ):
            violate(f"trajectory {uid} row {index}: invalid context transition")
        else:
            context_counts[str(operation)] += 1

        wrapper = record.get("wrapper_evidence")
        wrapper = wrapper if isinstance(wrapper, Mapping) else {}
        adapter_keys = {
            key
            for key in ("agemem_adapter", "mem0_adapter", "letta_code_adapter")
            if key in wrapper
        }

        if method_id == "qwen35_4b":
            if adapter_keys:
                violate(
                    f"trajectory {uid} row {index}: frozen Qwen row carries "
                    f"method adapter evidence {sorted(adapter_keys)!r}"
                )
            native_rows += 1
            continue

        expected_key = f"{method_id}_adapter" if method_id == "mem0" else "letta_code_adapter"
        foreign_keys = adapter_keys - {expected_key}
        if foreign_keys:
            violate(
                f"trajectory {uid} row {index}: carries foreign adapter evidence "
                f"{sorted(foreign_keys)!r}"
            )
        adapter = wrapper.get(expected_key)
        if not isinstance(adapter, Mapping):
            violate(f"trajectory {uid} row {index}: missing {method_id} adapter evidence")
            continue
        method_rows += 1
        if adapter.get("episode_private") is not True:
            violate(f"trajectory {uid} row {index}: method state is not episode-private")
        calls = adapter.get("hidden_model_calls", 0)
        input_tokens = adapter.get("hidden_input_tokens", 0)
        output_tokens = adapter.get("hidden_output_tokens", 0)
        latency_ms = adapter.get("hidden_latency_ms", 0)
        for name, value in (
            ("hidden_model_calls", calls),
            ("hidden_input_tokens", input_tokens),
            ("hidden_output_tokens", output_tokens),
            ("hidden_latency_ms", latency_ms),
        ):
            if not _nonnegative_int(value):
                violate(f"trajectory {uid} row {index}: invalid {name}")
        if all(_nonnegative_int(value) for value in (calls, input_tokens, output_tokens, latency_ms)):
            hidden_calls += calls
            hidden_input_tokens += input_tokens
            hidden_output_tokens += output_tokens
            hidden_latency_ms += latency_ms

        event = adapter.get("event")
        if event == "native_action_passthrough":
            native_rows += 1
        elif event == "memory_tool_action" and method_id == "letta_code":
            memory_rows += 1
            op = str(adapter.get("operation") or "<parse>")
            operation_counts[op] += 1
            if adapter.get("accepted") is True:
                if op not in _LETTA_OPERATIONS:
                    violate(f"trajectory {uid} row {index}: unsupported Letta operation {op!r}")
                reason = adapter.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    violate(f"trajectory {uid} row {index}: accepted Letta write lacks reason")
                commit = adapter.get("commit_sha")
                if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
                    violate(f"trajectory {uid} row {index}: accepted Letta write lacks git commit")
                else:
                    commits += 1
            elif not isinstance(adapter.get("error_code"), str):
                violate(f"trajectory {uid} row {index}: rejected Letta action lacks error_code")
        elif event == "memory_filesystem_read" and method_id == "letta_code":
            memory_rows += 1
            operation_counts["read"] += 1
            if adapter.get("operation") != "read" or adapter.get("accepted") is not True:
                violate(f"trajectory {uid} row {index}: invalid Letta filesystem read")
            read_path = adapter.get("read_path")
            read_bytes = adapter.get("read_bytes")
            if not isinstance(read_path, str) or not read_path.endswith(".md"):
                violate(f"trajectory {uid} row {index}: invalid Letta read_path")
            if not _nonnegative_int(read_bytes):
                violate(f"trajectory {uid} row {index}: invalid Letta read_bytes")
        else:
            violate(f"trajectory {uid} row {index}: invalid {method_id} event {event!r}")

        if method_id == "mem0":
            if adapter.get("schema") != "camg_mem0_adapter_v1":
                violate(f"trajectory {uid} row {index}: invalid Mem0 adapter schema")
            if adapter.get("official_pipeline") is not True:
                violate(f"trajectory {uid} row {index}: Mem0 official pipeline not attested")
            if adapter.get("source_revision") != _MEM0_SOURCE_REVISION:
                violate(f"trajectory {uid} row {index}: Mem0 source revision drift")
            if adapter.get("version") != _MEM0_VERSION:
                violate(f"trajectory {uid} row {index}: Mem0 version drift")
            boundary_requested = adapter.get("boundary_requested")
            boundary = adapter.get("boundary_pipeline")
            if not isinstance(boundary_requested, bool):
                violate(
                    f"trajectory {uid} row {index}: "
                    "invalid Mem0 boundary request flag"
                )
            if not isinstance(boundary, bool):
                violate(f"trajectory {uid} row {index}: invalid Mem0 boundary flag")
            if boundary is True and boundary_requested is not True:
                violate(
                    f"trajectory {uid} row {index}: "
                    "Mem0 pipeline ran without a boundary request"
                )
            raw_operation_counts = adapter.get("operation_counts")
            if not isinstance(raw_operation_counts, Mapping):
                violate(f"trajectory {uid} row {index}: invalid Mem0 operation counts")
                raw_operation_counts = {}
            for op, count in raw_operation_counts.items():
                if not _nonnegative_int(count):
                    violate(f"trajectory {uid} row {index}: invalid Mem0 operation count")
                else:
                    operation_counts[str(op)] += count
            if boundary is True:
                if raw_operation_counts.get("add") != 1 or raw_operation_counts.get("search") != 1:
                    violate(
                        f"trajectory {uid} row {index}: Mem0 boundary lacks one add/search"
                    )
                if not _nonnegative_int(calls) or calls < 1:
                    violate(
                        f"trajectory {uid} row {index}: Mem0 boundary lacks hidden LLM evidence"
                    )
                if budget is not None and (
                    budget["auxiliary_steps"] != 2
                    or budget["required_auxiliary_steps"] != 2
                    or budget["atomic_operation_blocked"] is not False
                ):
                    violate(
                        f"trajectory {uid} row {index}: Mem0 add/search budget charge drifted"
                    )
            elif boundary is False and (raw_operation_counts or calls != 0):
                violate(
                    f"trajectory {uid} row {index}: non-boundary Mem0 row has hidden work"
                )
            elif boundary_requested is True:
                if budget is not None and (
                    budget["auxiliary_steps"] != 0
                    or budget["required_auxiliary_steps"] != 2
                    or budget["atomic_operation_blocked"] is not True
                    or budget["terminate_after_action"] is not True
                ):
                    violate(
                        f"trajectory {uid} row {index}: blocked Mem0 boundary budget drifted"
                    )
            elif budget is not None and (
                budget["auxiliary_steps"] != 0
                or budget["required_auxiliary_steps"] != 0
            ):
                violate(
                    f"trajectory {uid} row {index}: non-boundary Mem0 budget is nonzero"
                )
        else:
            if adapter.get("schema") != "camg_letta_code_adapter_v1":
                violate(f"trajectory {uid} row {index}: invalid Letta adapter schema")
            if adapter.get("git_backed") is not True:
                violate(f"trajectory {uid} row {index}: Letta MemFS is not git-backed")
            if adapter.get("source_revision") != _LETTA_SOURCE_REVISION:
                violate(f"trajectory {uid} row {index}: Letta source revision drift")
            if budget is not None and (
                budget["auxiliary_steps"] != 0
                or budget["required_auxiliary_steps"] != 0
            ):
                violate(
                    f"trajectory {uid} row {index}: Letta policy action was double-counted"
                )

    if not rows:
        violate("no step records were provided")
    if method_id != "qwen35_4b" and method_rows != len(rows):
        violate(f"method evidence coverage is {method_rows}/{len(rows)}")
    if method_id == "letta_code" and hidden_calls:
        violate("Letta Code must not use hidden model calls")

    return {
        "schema": METHOD_EVIDENCE_SCHEMA,
        "method_id": method_id,
        "status": "PASS" if not violations else "FAIL",
        "totals": {
            "rows": len(rows),
            "trajectories": len(trajectories),
            "method_evidence_rows": method_rows,
            "native_action_rows": native_rows,
            "memory_tool_rows": memory_rows,
            "git_commits": commits,
            "hidden_model_calls": hidden_calls,
            "hidden_input_tokens": hidden_input_tokens,
            "hidden_output_tokens": hidden_output_tokens,
            "hidden_latency_ms": hidden_latency_ms,
            "combined_steps": combined_steps,
            "auxiliary_steps": auxiliary_steps,
        },
        "operation_counts": dict(sorted(operation_counts.items())),
        "context_operation_counts": dict(sorted(context_counts.items())),
        "violations": violations,
    }


def summarize_method_step_records(
    records: Iterable[Mapping[str, Any]], *, method_id: str
) -> dict[str, Any]:
    """Validate one method's action-row ledger and return a common summary."""

    normalized = str(method_id or "").strip().lower()
    if normalized not in SUPPORTED_METHOD_IDS:
        raise ValueError(f"unsupported held-out method_id {method_id!r}")
    rows = list(records)
    if normalized == "agemem":
        legacy = summarize_agemem_step_records(rows)
        foreign: list[str] = []
        for index, row in enumerate(rows):
            wrapper = row.get("wrapper_evidence") if isinstance(row, Mapping) else None
            if not isinstance(wrapper, Mapping):
                continue
            keys = sorted(
                key
                for key in ("mem0_adapter", "letta_code_adapter")
                if key in wrapper
            )
            if keys:
                foreign.append(
                    f"record[{index}] carries foreign adapter evidence {keys!r}"
                )
        violations = [*legacy["violations"], *foreign]
        return {
            "schema": METHOD_EVIDENCE_SCHEMA,
            "method_id": normalized,
            "status": "PASS" if not violations else "FAIL",
            "totals": legacy["totals"],
            "operation_counts": legacy["operation_counts"],
            "context_operation_counts": legacy["context_operation_counts"],
            "violations": violations,
            "legacy_agemem_summary": legacy,
        }
    return _generic_summary(rows, method_id=normalized)


__all__ = [
    "METHOD_EVIDENCE_SCHEMA",
    "SUPPORTED_METHOD_IDS",
    "summarize_method_step_records",
]
