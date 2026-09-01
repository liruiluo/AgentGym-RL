"""Audit AgeMem-style adapter evidence in AMG rollout JSONL ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

AGEMEM_EVIDENCE_SUMMARY_SCHEMA = "camg_agemem_style_evidence_summary_v1"
AGEMEM_ADAPTER_SCHEMA = "camg_agemem_style_adapter_v1"
_TASK_NEUTRAL_CONTEXT_SCHEMA = "agentmemory_task_neutral_context_transition_v1"
_MEMORY_OPERATIONS = (
    "Add_memory",
    "Update_memory",
    "Delete_memory",
    "Retrieve_memory",
    "Summary_context",
    "Filter_context",
)


def summarize_agemem_step_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_routes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a fail-closed aggregate over task-neutral action-row records."""

    total_rows = 0
    adapter_rows = 0
    memory_rows = 0
    native_rows = 0
    hidden_model_calls = 0
    operation_counts: Counter[str] = Counter()
    accepted_operation_counts: Counter[str] = Counter()
    rejected_operation_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    context_operation_counts: Counter[str] = Counter()
    terminal_outcomes: Counter[str] = Counter()
    route_rows: Counter[str] = Counter()
    route_memory_rows: Counter[str] = Counter()
    route_trajectories: dict[str, set[str]] = defaultdict(set)
    trajectories: dict[str, dict[str, Any]] = {}
    cross_context_retrievals = 0
    cross_context_retrieval_trajectories: set[str] = set()
    trajectories_with_memory_actions: set[str] = set()
    violations: list[str] = []
    violation_count = 0

    def violate(message: str) -> None:
        nonlocal violation_count
        violation_count += 1
        if len(violations) < 200:
            violations.append(message)

    for record_index, record in enumerate(records):
        total_rows += 1
        if not isinstance(record, Mapping):
            violate(f"record[{record_index}] is not a mapping")
            continue
        trajectory_uid = record.get("trajectory_uid")
        if not isinstance(trajectory_uid, str) or not trajectory_uid:
            violate(f"record[{record_index}] has no trajectory_uid")
            trajectory_uid = f"<missing:{record_index}>"
        rollout_update = record.get("_rollout_update", "direct")
        trajectory_key = f"{rollout_update}:{trajectory_uid}"
        route = record.get("route_id", record.get("data_source"))
        if not isinstance(route, str) or not route:
            violate(f"record[{record_index}] has no route_id/data_source")
            route = "<missing>"
        route_rows[route] += 1
        route_trajectories[route].add(trajectory_key)

        state = trajectories.setdefault(
            trajectory_key,
            {
                "route": route,
                "last_row_order": -1,
                "memory_action_index": 0,
                "memory_size": 0,
                "context_epoch": 0,
                "memory_birth_epoch": {},
                "operations": set(),
            },
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

        wrapper_evidence = record.get("wrapper_evidence")
        adapter = (
            wrapper_evidence.get("agemem_adapter")
            if isinstance(wrapper_evidence, Mapping)
            else None
        )
        if not isinstance(adapter, Mapping):
            violate(f"trajectory {trajectory_uid} row {row_order}: missing adapter evidence")
            continue
        adapter_rows += 1
        if adapter.get("schema") != AGEMEM_ADAPTER_SCHEMA:
            violate(
                f"trajectory {trajectory_uid} row {row_order}: bad adapter schema "
                f"{adapter.get('schema')!r}"
            )
        if adapter.get("episode_private") is not True:
            violate(
                f"trajectory {trajectory_uid} row {row_order}: store is not episode-private"
            )
        calls = adapter.get("hidden_model_calls")
        if type(calls) is not int or calls < 0:
            violate(
                f"trajectory {trajectory_uid} row {row_order}: invalid hidden_model_calls"
            )
        else:
            hidden_model_calls += calls
            if calls:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: hidden model call observed"
                )

        transition = record.get("context_transition")
        context_operation = (
            transition.get("operation") if isinstance(transition, Mapping) else None
        )
        if (
            not isinstance(transition, Mapping)
            or transition.get("schema") != _TASK_NEUTRAL_CONTEXT_SCHEMA
        ):
            violate(
                f"trajectory {trajectory_uid} row {row_order}: invalid context transition"
            )
        elif context_operation not in {
            "append_observation",
            "preserve",
            "replace_messages",
        }:
            violate(
                f"trajectory {trajectory_uid} row {row_order}: invalid context operation "
                f"{context_operation!r}"
            )
        else:
            context_operation_counts[str(context_operation)] += 1
        if adapter.get("context_operation") not in {None, context_operation}:
            violate(
                f"trajectory {trajectory_uid} row {row_order}: adapter/row context "
                "operations disagree"
            )

        event = adapter.get("event")
        if event == "memory_tool_action":
            memory_rows += 1
            route_memory_rows[route] += 1
            trajectories_with_memory_actions.add(trajectory_key)
            action_index = adapter.get("memory_action_index")
            expected_index = state["memory_action_index"] + 1
            if type(action_index) is not int or action_index != expected_index:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: memory action index "
                    f"{action_index!r}, expected {expected_index}"
                )
            else:
                state["memory_action_index"] = action_index
            before = adapter.get("memory_size_before")
            after = adapter.get("memory_size_after")
            if type(before) is not int or type(after) is not int:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: invalid memory size"
                )
            else:
                if before != state["memory_size"]:
                    violate(
                        f"trajectory {trajectory_uid} row {row_order}: memory size before "
                        f"{before}, expected {state['memory_size']}"
                    )
                accepted = adapter.get("accepted") is True
                operation = adapter.get("operation")
                operation_key = str(operation) if operation is not None else "<parse>"
                operation_counts[operation_key] += 1
                if accepted:
                    accepted_operation_counts[operation_key] += 1
                    if operation not in _MEMORY_OPERATIONS:
                        violate(
                            f"trajectory {trajectory_uid} row {row_order}: accepted unknown "
                            f"operation {operation!r}"
                        )
                    expected_after = before
                    if operation == "Add_memory":
                        expected_after += 1
                    elif operation == "Delete_memory":
                        expected_after -= 1
                    if after != expected_after:
                        violate(
                            f"trajectory {trajectory_uid} row {row_order}: {operation} "
                            f"changed size {before}->{after}, expected {expected_after}"
                        )
                    memory_id = adapter.get("memory_id")
                    if operation in {"Add_memory", "Update_memory", "Delete_memory"}:
                        if not isinstance(memory_id, str) or not memory_id:
                            violate(
                                f"trajectory {trajectory_uid} row {row_order}: {operation} "
                                "omits memory_id"
                            )
                    if operation == "Add_memory" and isinstance(memory_id, str):
                        state["memory_birth_epoch"][memory_id] = state["context_epoch"]
                    if operation == "Retrieve_memory":
                        retrieved = adapter.get("retrieved_memory_ids")
                        count = adapter.get("retrieved_memory_count")
                        if not isinstance(retrieved, list) or any(
                            not isinstance(item, str) for item in retrieved
                        ):
                            violate(
                                f"trajectory {trajectory_uid} row {row_order}: invalid "
                                "retrieved_memory_ids"
                            )
                        elif type(count) is not int or count != len(retrieved):
                            violate(
                                f"trajectory {trajectory_uid} row {row_order}: retrieval "
                                "count disagrees with ids"
                            )
                        else:
                            for memory_id in retrieved:
                                birth_epoch = state["memory_birth_epoch"].get(memory_id)
                                if birth_epoch is None:
                                    violate(
                                        f"trajectory {trajectory_uid} row {row_order}: "
                                        f"retrieved unknown memory {memory_id}"
                                    )
                                elif state["context_epoch"] > birth_epoch:
                                    cross_context_retrievals += 1
                                    cross_context_retrieval_trajectories.add(
                                        trajectory_key
                                    )
                else:
                    rejected_operation_counts[operation_key] += 1
                    if after != before:
                        violate(
                            f"trajectory {trajectory_uid} row {row_order}: rejected action "
                            f"changed size {before}->{after}"
                        )
                    error_code = adapter.get("error_code")
                    if not isinstance(error_code, str) or not error_code:
                        violate(
                            f"trajectory {trajectory_uid} row {row_order}: rejected action "
                            "omits error_code"
                        )
                    else:
                        error_counts[error_code] += 1
                state["memory_size"] = after
                state["operations"].add(operation_key)
        elif event == "native_action_passthrough":
            native_rows += 1
            action_count = adapter.get("memory_action_count")
            after = adapter.get("memory_size_after")
            if type(action_count) is not int or action_count != state["memory_action_index"]:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: native row memory action "
                    f"count {action_count!r}, expected {state['memory_action_index']}"
                )
            if type(after) is not int or after != state["memory_size"]:
                violate(
                    f"trajectory {trajectory_uid} row {row_order}: native row memory size "
                    f"{after!r}, expected {state['memory_size']}"
                )
        else:
            violate(
                f"trajectory {trajectory_uid} row {row_order}: unknown adapter event {event!r}"
            )

        if context_operation == "replace_messages":
            state["context_epoch"] += 1
        if record.get("trajectory_terminal") is True:
            terminal_outcomes[str(record.get("outcome"))] += 1

    expected = sorted(set(str(route) for route in expected_routes))
    missing_routes = sorted(set(expected) - set(route_rows))
    if total_rows == 0:
        violate("no step records were provided")
    if adapter_rows != total_rows:
        violate(f"adapter evidence coverage is {adapter_rows}/{total_rows}")
    if missing_routes:
        violate("missing expected routes: " + ", ".join(missing_routes))

    route_summary: dict[str, Any] = {}
    for route in sorted(route_rows):
        route_summary[route] = {
            "rows": route_rows[route],
            "trajectories": len(route_trajectories[route]),
            "memory_tool_rows": route_memory_rows[route],
        }
    return {
        "schema": AGEMEM_EVIDENCE_SUMMARY_SCHEMA,
        "status": "PASS" if violation_count == 0 else "FAIL",
        "expected_routes": expected,
        "missing_routes": missing_routes,
        "totals": {
            "rows": total_rows,
            "trajectories": len(trajectories),
            "adapter_evidence_rows": adapter_rows,
            "native_action_rows": native_rows,
            "memory_tool_rows": memory_rows,
            "trajectories_with_memory_actions": len(
                trajectories_with_memory_actions
            ),
            "hidden_model_calls": hidden_model_calls,
            "cross_context_retrievals": cross_context_retrievals,
            "trajectories_with_cross_context_retrieval": len(
                cross_context_retrieval_trajectories
            ),
        },
        "routes": route_summary,
        "operation_counts": dict(sorted(operation_counts.items())),
        "accepted_operation_counts": dict(
            sorted(accepted_operation_counts.items())
        ),
        "rejected_operation_counts": dict(
            sorted(rejected_operation_counts.items())
        ),
        "error_counts": dict(sorted(error_counts.items())),
        "context_operation_counts": dict(sorted(context_operation_counts.items())),
        "terminal_outcomes": dict(sorted(terminal_outcomes.items())),
        "violation_count": violation_count,
        "violations": violations,
        "violations_truncated": violation_count > len(violations),
    }


def _input_files(paths: Sequence[Path], through_update: int | None) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(candidate for candidate in path.glob("*.jsonl") if candidate.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)

    def sort_key(path: Path) -> tuple[int, int | str]:
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.name)

    selected: list[Path] = []
    for path in sorted(set(candidate.resolve() for candidate in files), key=sort_key):
        if through_update is not None:
            try:
                update = int(path.stem)
            except ValueError as exc:
                raise ValueError(
                    f"--through-update requires numeric rollout filenames: {path}"
                ) from exc
            if update > through_update:
                continue
        selected.append(path)
    return selected


def _iter_step_records(
    files: Sequence[Path], manifest: list[dict[str, Any]]
) -> Iterable[dict[str, Any]]:
    """Stream large formal ledgers while filling a byte-level input manifest."""

    for path in files:
        digest = hashlib.sha256()
        file_records: list[dict[str, Any]] = []
        entry: dict[str, Any] = {
            "path": str(path),
            "sha256": None,
            "jsonl_rows": 0,
            "step_records": 0,
        }
        manifest.append(entry)
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                entry["jsonl_rows"] += 1
                try:
                    envelope = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                raw_record = envelope.get("step_record_json")
                if isinstance(raw_record, str):
                    try:
                        record = json.loads(raw_record)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path}:{line_number}: invalid step_record_json"
                        ) from exc
                elif isinstance(raw_record, Mapping):
                    record = dict(raw_record)
                else:
                    raise ValueError(
                        f"{path}:{line_number}: step_record_json is missing"
                    )
                if not isinstance(record, Mapping):
                    raise ValueError(
                        f"{path}:{line_number}: decoded step record is not an object"
                    )
                normalized_record = dict(record)
                envelope_update = envelope.get("step")
                try:
                    filename_update: int | str = int(path.stem)
                except ValueError:
                    filename_update = path.name
                if type(envelope_update) is int:
                    if isinstance(filename_update, int) and envelope_update != filename_update:
                        raise ValueError(
                            f"{path}:{line_number}: envelope step {envelope_update} "
                            f"does not match filename update {filename_update}"
                        )
                    rollout_update: int | str = envelope_update
                else:
                    rollout_update = filename_update
                normalized_record["_rollout_update"] = rollout_update
                entry["step_records"] += 1
                file_records.append(normalized_record)
        entry["sha256"] = digest.hexdigest()
        # Fully asynchronous writers emit rows in completion order.  Restore
        # the policy order independently inside each optimizer-update file.
        file_records.sort(
            key=lambda record: (
                str(record.get("trajectory_uid", "")),
                int(record.get("trajectory_row_order", -1)),
            )
        )
        yield from file_records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--through-update", type=int)
    parser.add_argument("--expect-route", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.through_update is not None and args.through_update <= 0:
        parser.error("--through-update must be positive")
    files = _input_files(args.input, args.through_update)
    manifest: list[dict[str, Any]] = []
    summary = summarize_agemem_step_records(
        _iter_step_records(files, manifest),
        expected_routes=args.expect_route,
    )
    summary["input_manifest"] = manifest
    summary["through_update"] = args.through_update
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
