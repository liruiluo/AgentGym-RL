from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import re
from typing import Any


WORKSPACE_TOOL_OPS = frozenset({"SHELL_COMMAND", "APPLY_PATCH"})
LEGACY_FILE_TOOL_OPS = frozenset({"READ", "WRITE", "EDIT", "GREP", "GLOB"})
WORKSPACE_SURFACE = "codex_workspace_v2"
WORKSPACE_TOOL_CONTRACT = "codex_shell_command_apply_patch_v1"
WORKSPACE_SNAPSHOT_SCHEMA = "agentmemory_workspace_snapshot_v2"
TASK_NEUTRAL_POLICY_STEP_SCHEMA = "task_neutral_policy_step_v1"
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_EMPTY_WORKSPACE_MANIFEST = {"directories": [], "files": []}
_EMPTY_WORKSPACE_TREE_SHA256 = hashlib.sha256(
    json.dumps(
        _EMPTY_WORKSPACE_MANIFEST,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _component_names(record: dict[str, Any]) -> set[str]:
    info = record.get("env_info_after")
    if not isinstance(info, dict):
        return set()
    components = info.get("reward_components")
    if not isinstance(components, list):
        return set()
    return {
        str(component["name"])
        for component in components
        if isinstance(component, dict) and component.get("name")
    }


def _has_authoritative_invalid_action(
    record: dict[str, Any], *, component_names: set[str]
) -> bool:
    """Read invalid-action status from the environment receipt.

    LiteResearcher keeps parser outcomes in ``status`` and
    ``wrapper_evidence``.  Older native receipts only exposed the reward
    component, while task-neutral rows may wrap the server evidence one level
    deeper.  These are alternate encodings of one action, so combine them with
    ``or`` rather than incrementing once per source.
    """

    info = record.get("env_info_after")
    if not isinstance(info, dict):
        return "invalid_action" in component_names

    if str(info.get("status", "")).lower() == "invalid_action":
        return True

    info_evidence = info.get("wrapper_evidence")
    if isinstance(info_evidence, dict) and info_evidence.get("invalid_action") is True:
        return True

    row_evidence = record.get("wrapper_evidence")
    if isinstance(row_evidence, dict):
        if row_evidence.get("invalid_action") is True:
            return True
        server_evidence = row_evidence.get("server_wrapper_evidence")
        if isinstance(server_evidence, dict) and server_evidence.get(
            "invalid_action"
        ) is True:
            return True

    return "invalid_action" in component_names


def _current_memory_ops(record: dict[str, Any]) -> list[dict[str, Any]]:
    info = record.get("env_info_after")
    if not isinstance(info, dict):
        return []
    memory_ops = info.get("memory_ops")
    if not isinstance(memory_ops, list):
        return []
    return [item for item in memory_ops if isinstance(item, dict)]


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"Filesystem evidence has invalid {name}={value!r}.")
    return value


def _workspace_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Filesystem evidence lacks a workspace snapshot object.")
    if value.get("schema") != WORKSPACE_SNAPSHOT_SCHEMA:
        raise ValueError("Filesystem evidence has an unknown workspace snapshot schema.")
    directories = value.get("directories")
    files = value.get("files")
    if not isinstance(directories, list) or any(
        not isinstance(item, str) or not item for item in directories
    ):
        raise ValueError("Filesystem workspace snapshot directories must be paths.")
    if directories != sorted(directories) or len(set(directories)) != len(directories):
        raise ValueError(
            "Filesystem snapshot directories are not unique deterministic paths."
        )
    if not isinstance(files, list):
        raise ValueError("Filesystem workspace snapshot files must be a list.")
    normalized_files: list[dict[str, Any]] = []
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ValueError(f"Filesystem snapshot file {index} is not an object.")
        path = item.get("path")
        size = item.get("bytes")
        if not isinstance(path, str) or not path:
            raise ValueError(f"Filesystem snapshot file {index} has no path.")
        if type(size) is not int or size < 0:
            raise ValueError(f"Filesystem snapshot file {index} has invalid bytes.")
        normalized_files.append(
            {
                "path": path,
                "sha256": _require_sha256(
                    item.get("sha256"), name=f"snapshot file {index} sha256"
                ),
                "bytes": size,
            }
        )
    if normalized_files != sorted(normalized_files, key=lambda item: item["path"]):
        raise ValueError("Filesystem snapshot files are not in deterministic path order.")
    if len({item["path"] for item in normalized_files}) != len(normalized_files):
        raise ValueError("Filesystem snapshot contains duplicate paths.")
    file_count = value.get("file_count")
    directory_count = value.get("directory_count")
    total_bytes = value.get("total_bytes")
    if type(file_count) is not int or file_count != len(normalized_files):
        raise ValueError("Filesystem snapshot file_count disagrees with its manifest.")
    if type(total_bytes) is not int or total_bytes != sum(
        item["bytes"] for item in normalized_files
    ):
        raise ValueError("Filesystem snapshot total_bytes disagrees with its manifest.")
    if type(directory_count) is not int or directory_count != len(directories):
        raise ValueError(
            "Filesystem snapshot directory_count disagrees with its manifest."
        )
    manifest = json.dumps(
        {"directories": directories, "files": normalized_files},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_tree_sha256 = hashlib.sha256(manifest).hexdigest()
    tree_sha256 = _require_sha256(
        value.get("tree_sha256"), name="workspace tree_sha256"
    )
    if tree_sha256 != expected_tree_sha256:
        raise ValueError("Filesystem workspace tree hash disagrees with its manifest.")
    return {
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "directories": list(directories),
        "files": normalized_files,
        "tree_sha256": tree_sha256,
    }


def _current_workspace_ops(record: dict[str, Any]) -> list[dict[str, Any]]:
    info = record.get("env_info_after")
    if not isinstance(info, dict):
        return []
    workspace_ops = info.get("workspace_ops")
    if not isinstance(workspace_ops, list) or any(
        not isinstance(item, dict) for item in workspace_ops
    ):
        raise ValueError("Filesystem workspace_ops ledger must be a list of objects.")
    if len(workspace_ops) > 1:
        raise ValueError("Filesystem step contains more than one workspace operation.")
    tool_ops = info.get("tool_ops")
    if not isinstance(tool_ops, list):
        raise ValueError("Filesystem evidence lacks the authoritative tool_ops ledger.")
    stale_ops = [
        item
        for item in tool_ops
        if isinstance(item, dict)
        and str(item.get("op", "")).upper() in LEGACY_FILE_TOOL_OPS
    ]
    if stale_ops:
        raise ValueError("Filesystem-v2 evidence contains a legacy five-tool operation.")
    expected = [
        item
        for item in tool_ops
        if isinstance(item, dict)
        and str(item.get("op", "")).upper() in WORKSPACE_TOOL_OPS
    ]
    if workspace_ops != expected:
        raise ValueError("Filesystem workspace_ops disagrees with authoritative tool_ops.")
    for event in workspace_ops:
        op = str(event.get("op", "")).upper()
        if op not in WORKSPACE_TOOL_OPS or event.get("status") != "executed":
            raise ValueError("Filesystem workspace operation is malformed or not executed.")
    return workspace_ops


def _validate_zero_workspace_reward_component(
    record: dict[str, Any], *, op: str
) -> None:
    info = record.get("env_info_after")
    if not isinstance(info, dict):
        raise ValueError("Filesystem evidence lacks post-step environment info.")
    components = info.get("reward_components")
    if not isinstance(components, list) or any(
        not isinstance(component, dict) for component in components
    ):
        raise ValueError("Filesystem reward_components ledger must be a list of objects.")
    expected_name = f"{op.lower()}_transition"
    matching = [
        component
        for component in components
        if component.get("name") == expected_name
    ]
    if len(matching) != 1:
        raise ValueError(
            "Filesystem workspace action lacks exactly one matching reward component."
        )
    component = matching[0]
    if component.get("name") != expected_name:
        raise ValueError("Filesystem workspace reward component has the wrong name.")
    component_op = component.get("op")
    if component_op is not None and str(component_op).upper() != op:
        raise ValueError("Filesystem workspace reward component has the wrong operation.")
    value = component.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Filesystem workspace reward component is not numeric.")
    if not math.isfinite(float(value)) or not math.isclose(
        float(value), 0.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "Filesystem workspace action received non-zero task reward."
        )


def _workspace_diff(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
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


def _workspace_diff_has_changes(diff: dict[str, Any]) -> bool:
    return any(diff[field] for field in diff)


def _workspace_written_versions(diff: dict[str, Any]) -> set[tuple[str, str]]:
    versions = {
        (item["path"], item["sha256"])
        for item in diff["added"]
    }
    versions.update(
        (item["after"]["path"], item["after"]["sha256"])
        for item in diff["modified"]
    )
    return versions


def _validate_workspace_info(info: Any) -> tuple[dict[str, Any], int, str]:
    if not isinstance(info, dict):
        raise ValueError("Filesystem evidence lacks a workspace info object.")
    if info.get("workspace_surface") != WORKSPACE_SURFACE:
        raise ValueError("Filesystem trajectory mixes workspace surface contracts.")
    if info.get("workspace_tool_contract") != WORKSPACE_TOOL_CONTRACT:
        raise ValueError("Filesystem evidence has the wrong workspace tool contract.")
    if info.get("workspace_tool_ops") != ["SHELL_COMMAND", "APPLY_PATCH"]:
        raise ValueError("Filesystem evidence has the wrong workspace tool operation set.")
    if info.get("memory_ops") != []:
        raise ValueError("Filesystem-v2 evidence contains a legacy memory operation.")
    if "file_ops" in info:
        raise ValueError("Filesystem-v2 evidence contains the retired file_ops ledger.")
    intervention = info.get("workspace_intervention")
    if intervention not in {"enabled", "no_workspace"}:
        raise ValueError("Filesystem evidence has an invalid workspace intervention.")
    expected_enabled = intervention == "enabled"
    if info.get("workspace_shell_enabled") is not expected_enabled:
        raise ValueError("Filesystem shell availability disagrees with its intervention.")
    if info.get("workspace_apply_patch_enabled") is not expected_enabled:
        raise ValueError("Filesystem patch availability disagrees with its intervention.")
    snapshot = _workspace_snapshot(info.get("workspace_snapshot"))
    if not expected_enabled and snapshot["tree_sha256"] != _EMPTY_WORKSPACE_TREE_SHA256:
        raise ValueError("no_workspace intervention exposes a non-empty workspace.")
    audit_count = info.get("workspace_audit_event_count")
    if type(audit_count) is not int or audit_count < 0:
        raise ValueError("Filesystem evidence has an invalid workspace audit count.")
    return snapshot, audit_count, intervention


def _memory_ids(memory_op: dict[str, Any]) -> set[str]:
    memory_ids: set[str] = set()
    memory_id = memory_op.get("memory_id")
    if isinstance(memory_id, str) and memory_id:
        memory_ids.add(memory_id)
    retrieved_memory_ids = memory_op.get("retrieved_memory_ids")
    if isinstance(retrieved_memory_ids, list):
        memory_ids.update(
            item for item in retrieved_memory_ids if isinstance(item, str) and item
        )
    return memory_ids


def _is_relevant_retrieve(record: dict[str, Any]) -> bool:
    info = record.get("env_info_after")
    if not isinstance(info, dict):
        return False
    components = info.get("reward_components")
    if not isinstance(components, list):
        return False
    for component in components:
        if not isinstance(component, dict):
            continue
        name = component.get("name")
        if name == "memory_retrieve_first_relevant_before_dependent_buy":
            return True
        if (
            name == "memory_retrieve_additional_nonempty_dependent_context"
            and component.get("relevant") is True
        ):
            return True
    return False


def _record_phase_bounds(record: dict[str, Any]) -> tuple[int, int]:
    """Read session bounds from either the legacy row or opaque env receipt."""

    before_info = record.get("env_info_before")
    after_info = record.get("env_info_after")
    env_before = (
        before_info.get("current_subtask_index")
        if isinstance(before_info, dict)
        else None
    )
    env_after = (
        after_info.get("current_subtask_index")
        if isinstance(after_info, dict)
        else None
    )
    row_before = record.get("subtask_index_before")
    row_after = record.get("subtask_index_after")

    if (row_before is None) != (row_after is None):
        raise ValueError("Formal row exposes only one legacy subtask boundary.")
    if row_before is None:
        if env_before is None and env_after is None:
            return 0, 0
        if env_before is None or env_after is None:
            raise ValueError("Task-neutral row exposes only one environment phase boundary.")
        before, after = env_before, env_after
    else:
        before, after = row_before, row_after
        if env_before is not None and env_before != before:
            raise ValueError("Legacy and environment pre-step phase boundaries disagree.")
        if env_after is not None and env_after != after:
            raise ValueError("Legacy and environment post-step phase boundaries disagree.")

    if (
        type(before) is not int
        or type(after) is not int
        or before < 0
        or after < 0
    ):
        raise ValueError("Formal row has an invalid environment phase boundary.")
    return before, after


def _record_purchase_state(
    record: dict[str, Any],
    *,
    component_names: set[str],
    phase_before: int,
    phase_after: int,
) -> tuple[bool, bool, bool]:
    """Return accepted, committed, and advanced without replaying stale receipts."""

    if record.get("schema_version") != TASK_NEUTRAL_POLICY_STEP_SCHEMA:
        return (
            bool(record.get("buy_accepted")),
            bool(record.get("buy_committed")),
            bool(record.get("session_advanced")),
        )

    wrapper_evidence = record.get("wrapper_evidence")
    if not isinstance(wrapper_evidence, dict):
        raise ValueError("Task-neutral row lacks wrapper evidence.")
    if wrapper_evidence.get("event") != "native_action":
        return False, False, False

    declared_advanced = wrapper_evidence.get("session_advanced")
    inferred_advanced = phase_after > phase_before
    if declared_advanced is not None:
        if type(declared_advanced) is not bool:
            raise ValueError("Task-neutral session advance evidence is not boolean.")
        if declared_advanced != inferred_advanced:
            raise ValueError("Task-neutral session advance evidence disagrees with phases.")

    correct = "buy_committed_correct" in component_names
    wrong = "buy_committed_incorrect" in component_names
    if correct and wrong:
        raise ValueError("Task-neutral row contains conflicting BUY outcomes.")
    if correct != inferred_advanced:
        raise ValueError("Task-neutral correct BUY evidence disagrees with session advance.")
    return correct, correct or wrong, inferred_advanced


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_formal_training_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize action rows without conflating rewards, suffixes, or episodes."""

    if not rows:
        raise ValueError("Formal training metrics require at least one action row.")
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"Formal metric row {index} is not an object.")
        trajectory_uid = row.get("trajectory_uid")
        if not isinstance(trajectory_uid, str) or not trajectory_uid:
            raise ValueError(f"Formal metric row {index} has no trajectory_uid.")
        for field in (
            "row_order",
            "immediate_reward",
            "suffix_return",
            "trajectory_return",
            "advantage_token_mean",
        ):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Formal metric row {index} has invalid {field}={value!r}.")
            if field != "row_order" and not math.isfinite(float(value)):
                raise ValueError(f"Formal metric row {index} has non-finite {field}.")
        if not isinstance(row.get("record"), dict):
            raise TypeError(f"Formal metric row {index} has no decoded step record.")
        trajectories[trajectory_uid].append(row)

    immediate_rewards = [float(row["immediate_reward"]) for row in rows]
    suffix_returns = [float(row["suffix_return"]) for row in rows]
    trajectory_returns: list[float] = []
    counts = defaultdict(int)
    terminal_advantages: dict[str, list[float]] = defaultdict(list)
    workspace_final_file_counts: list[float] = []
    workspace_final_total_bytes: list[float] = []
    workspace_final_audit_counts: list[float] = []

    for trajectory_uid, trajectory_rows in trajectories.items():
        ordered = sorted(trajectory_rows, key=lambda row: int(row["row_order"]))
        row_orders = [int(row["row_order"]) for row in ordered]
        if row_orders != list(range(len(ordered))):
            raise ValueError(
                f"Trajectory {trajectory_uid!r} has non-contiguous row order {row_orders}."
            )
        declared_returns = {float(row["trajectory_return"]) for row in ordered}
        if len(declared_returns) != 1:
            raise ValueError(f"Trajectory {trajectory_uid!r} has conflicting returns.")
        trajectory_return = declared_returns.pop()
        if not math.isclose(
            sum(float(row["immediate_reward"]) for row in ordered),
            trajectory_return,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(f"Trajectory {trajectory_uid!r} reward sum mismatch.")
        running_suffix = 0.0
        for row in reversed(ordered):
            running_suffix += float(row["immediate_reward"])
            if not math.isclose(
                float(row["suffix_return"]),
                running_suffix,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"Trajectory {trajectory_uid!r} suffix return mismatch at "
                    f"row {row['row_order']}."
                )
        terminal_rows = [row for row in ordered if bool(row.get("terminal"))]
        if len(terminal_rows) != 1 or terminal_rows[0] is not ordered[-1]:
            raise ValueError(f"Trajectory {trajectory_uid!r} terminal placement is invalid.")
        trajectory_returns.append(trajectory_return)

        workspace_enabled = any(
            isinstance(row["record"].get(key), dict)
            and row["record"][key].get("workspace_surface") == WORKSPACE_SURFACE
            for row in ordered
            for key in ("env_info_before", "env_info_after")
        )
        if workspace_enabled:
            counts["filesystem_trajectory_count"] += 1
            counts["workspace_trajectory_count"] += 1

        write_positions: list[int] = []
        relevant_retrieve_positions: list[int] = []
        dependent_buy_positions: list[int] = []
        memory_write_events: list[tuple[set[str], int, int]] = []
        memory_retrieve_events: list[tuple[set[str], int, int]] = []
        workspace_write_events: list[tuple[set[tuple[str, str]], int, int]] = []
        shell_events: list[
            tuple[int, int, set[tuple[str, str]]]
        ] = []
        correct_buy_events: list[tuple[int, int]] = []
        has_memory_id = False
        max_progress = 0
        terminal_success = False
        previous_workspace_snapshot: dict[str, Any] | None = None
        previous_workspace_audit_count: int | None = None
        workspace_intervention: str | None = None
        final_workspace_snapshot: dict[str, Any] | None = None
        for position, row in enumerate(ordered):
            record = row["record"]
            component_names = _component_names(record)
            memory_ops = _current_memory_ops(record)
            session_index, next_session_index = _record_phase_bounds(record)
            max_progress = max(max_progress, next_session_index)
            buy_accepted, buy_committed, session_advanced = _record_purchase_state(
                record,
                component_names=component_names,
                phase_before=session_index,
                phase_after=next_session_index,
            )
            workspace_ops = (
                _current_workspace_ops(record) if workspace_enabled else []
            )
            if workspace_enabled:
                before_info = record.get("env_info_before")
                after_info = record.get("env_info_after")
                before_snapshot, before_audit_count, before_intervention = (
                    _validate_workspace_info(before_info)
                )
                after_snapshot, after_audit_count, after_intervention = (
                    _validate_workspace_info(after_info)
                )
                if before_intervention != after_intervention:
                    raise ValueError(
                        "Filesystem intervention changed within one environment step."
                    )
                if workspace_intervention is None:
                    workspace_intervention = before_intervention
                    counts[f"workspace_{before_intervention}_trajectory_count"] += 1
                elif workspace_intervention != before_intervention:
                    raise ValueError(
                        "Filesystem intervention changed within one trajectory."
                    )
                if previous_workspace_snapshot is not None:
                    if before_snapshot != previous_workspace_snapshot:
                        raise ValueError(
                            "Filesystem pre-step snapshot breaks trajectory continuity."
                        )
                    if before_audit_count != previous_workspace_audit_count:
                        raise ValueError(
                            "Filesystem pre-step audit count breaks trajectory continuity."
                        )
                final_workspace_snapshot = after_snapshot
                counts["workspace_snapshot_record_count"] += 1
                if after_snapshot["file_count"] > 0:
                    counts["workspace_nonempty_snapshot_record_count"] += 1
                latest_event = after_info.get("workspace_latest_event")
                expected_diff = _workspace_diff(before_snapshot, after_snapshot)
                if workspace_ops:
                    event = workspace_ops[0]
                    op = str(event.get("op", "")).upper()
                    event_id = event.get("event_id")
                    phase_index = event.get("phase_index")
                    if type(event_id) is not int or event_id != before_audit_count:
                        raise ValueError(
                            "Filesystem audit event id is not contiguous within the episode."
                        )
                    if type(phase_index) is not int or phase_index != session_index:
                        raise ValueError(
                            "Filesystem audit event is bound to a different session."
                        )
                    before_tree = _require_sha256(
                        event.get("workspace_tree_sha256_before"),
                        name="workspace event before-tree sha256",
                    )
                    after_tree = _require_sha256(
                        event.get("workspace_tree_sha256_after"),
                        name="workspace event after-tree sha256",
                    )
                    if before_tree != before_snapshot["tree_sha256"]:
                        raise ValueError(
                            "Filesystem workspace event before-tree hash breaks continuity."
                        )
                    if after_tree != after_snapshot["tree_sha256"]:
                        raise ValueError(
                            "Filesystem workspace event after-tree hash disagrees with snapshot."
                        )
                    if after_audit_count != before_audit_count + 1:
                        raise ValueError(
                            "Filesystem audit count did not advance exactly once."
                        )
                    if latest_event != event:
                        raise ValueError(
                            "Filesystem latest-event evidence disagrees with workspace_ops."
                        )
                    if event.get("workspace_diff") != expected_diff:
                        raise ValueError(
                            "Filesystem workspace_diff disagrees with before/after snapshots."
                        )
                    _validate_zero_workspace_reward_component(record, op=op)
                    counts["workspace_action_count"] += 1
                    counts[f"workspace_{op.lower()}_count"] += 1
                    if _workspace_diff_has_changes(expected_diff):
                        counts["workspace_mutating_action_count"] += 1
                        counts["workspace_tree_change_count"] += 1
                    written_versions = _workspace_written_versions(expected_diff)
                    if written_versions:
                        counts["workspace_content_write_action_count"] += 1
                        workspace_write_events.append(
                            (written_versions, position, session_index)
                        )
                    if expected_diff["deleted"]:
                        counts["workspace_delete_action_count"] += 1
                    snapshot_versions = {
                        (item["path"], item["sha256"])
                        for item in after_snapshot["files"]
                    }
                    if op == "SHELL_COMMAND":
                        shell_events.append(
                            (position, session_index, snapshot_versions)
                        )
                        exit_code = event.get("exit_code")
                        if type(exit_code) is not int:
                            raise ValueError(
                                "Filesystem shell event has no integer exit code."
                            )
                        if exit_code != 0:
                            counts["workspace_shell_nonzero_exit_count"] += 1
                        if event.get("timed_out") is True:
                            counts["workspace_shell_timeout_count"] += 1
                    elif event.get("transactional") is not True:
                        raise ValueError(
                            "Filesystem apply_patch event is not transactional."
                        )
                else:
                    if after_audit_count != before_audit_count:
                        raise ValueError(
                            "Filesystem audit count changed without a file operation."
                        )
                    if after_snapshot != before_snapshot:
                        raise ValueError(
                            "Filesystem tree changed without a file operation."
                        )
                    if latest_event is not None:
                        raise ValueError(
                            "Filesystem non-file step exposes a current latest event."
                        )
                previous_workspace_snapshot = after_snapshot
                previous_workspace_audit_count = after_audit_count
            # Native records may omit action_execution. The environment receipt
            # is authoritative; legacy reward components remain compatible.
            if _has_authoritative_invalid_action(
                record, component_names=component_names
            ):
                counts["invalid_action_count"] += 1
            write_ops = [
                memory_op
                for memory_op in memory_ops
                if str(memory_op.get("op", "")).upper() in {"ADD", "UPDATE"}
            ]
            if write_ops:
                write_positions.append(position)
            for memory_op in memory_ops:
                memory_ids = _memory_ids(memory_op)
                has_memory_id = has_memory_id or bool(memory_ids)
                op = str(memory_op.get("op", "")).upper()
                if op in {"ADD", "UPDATE"} and memory_ids:
                    memory_write_events.append((memory_ids, position, session_index))
                elif (
                    op == "RETRIEVE"
                    and int(memory_op.get("retrieved_count", 0)) > 0
                    and memory_ids
                ):
                    memory_retrieve_events.append(
                        (memory_ids, position, session_index)
                    )
            if "memory_add_first_visible_product_reference" in component_names:
                counts["source_memory_add_progress_count"] += 1
            if "memory_add_first_valid_this_session" in component_names:
                counts["first_valid_add_count"] += 1

            nonempty_retrieve = any(
                str(memory_op.get("op", "")).upper() == "RETRIEVE"
                and int(memory_op.get("retrieved_count", 0)) > 0
                for memory_op in memory_ops
            )
            if nonempty_retrieve:
                counts["nonempty_retrieve_count"] += 1
            first_valid_later_session_retrieve = (
                "memory_retrieve_first_valid_later_session" in component_names
            )
            if first_valid_later_session_retrieve:
                counts["first_valid_later_session_retrieve_count"] += 1
                if not nonempty_retrieve:
                    counts["empty_first_valid_later_session_retrieve_count"] += 1
            if _is_relevant_retrieve(record):
                relevant_retrieve_positions.append(position)
                counts["relevant_retrieve_count"] += 1

            if buy_accepted:
                counts["correct_buy_count"] += 1
                correct_buy_events.append((position, session_index))
                if session_index >= 1:
                    dependent_buy_positions.append(position)
                terminal_advantages["correct_buy"].append(
                    float(row["advantage_token_mean"])
                )
            elif buy_committed:
                counts["wrong_buy_count"] += 1
                terminal_advantages["wrong_buy"].append(
                    float(row["advantage_token_mean"])
                )
            if session_advanced:
                counts["session_advance_count"] += 1
            if "max_round_timeout_failure" in component_names:
                counts["timeout_trajectory_count"] += 1
                terminal_advantages["timeout"].append(
                    float(row["advantage_token_mean"])
                )
            terminal_success = terminal_success or bool(
                record.get("outcome") == "success" and record.get("done")
            )

        if max_progress >= 1:
            counts["progress_ge_1_count"] += 1
        if max_progress >= 2:
            counts["progress_ge_2_count"] += 1
        if terminal_success:
            counts["terminal_success_count"] += 1
        source_memory_writes: list[tuple[set[str], int, int, int]] = []
        for memory_ids, write_position, write_session in memory_write_events:
            source_buy_position = next(
                (
                    buy_position
                    for buy_position, buy_session in correct_buy_events
                    if buy_session == write_session and buy_position > write_position
                ),
                None,
            )
            if source_buy_position is not None:
                source_memory_writes.append(
                    (memory_ids, write_position, write_session, source_buy_position)
                )
        counts["source_memory_write_before_correct_buy_count"] += len(
            source_memory_writes
        )

        source_workspace_writes: list[
            tuple[set[tuple[str, str]], int, int, int]
        ] = []
        for versions, write_position, write_session in workspace_write_events:
            source_buy_position = next(
                (
                    buy_position
                    for buy_position, buy_session in correct_buy_events
                    if buy_session == write_session and buy_position > write_position
                ),
                None,
            )
            if source_buy_position is not None:
                source_workspace_writes.append(
                    (versions, write_position, write_session, source_buy_position)
                )
        counts["source_workspace_write_before_correct_buy_count"] += len(
            source_workspace_writes
        )

        strict_functional_chain = False
        for retrieved_ids, retrieve_position, retrieve_session in memory_retrieve_events:
            source_linked = any(
                source_ids.intersection(retrieved_ids)
                and source_session < retrieve_session
                and source_buy_position < retrieve_position
                for (
                    source_ids,
                    _write_position,
                    source_session,
                    source_buy_position,
                ) in source_memory_writes
            )
            if not source_linked:
                continue
            counts["source_linked_retrieve_count"] += 1
            if any(
                buy_session == retrieve_session and buy_position > retrieve_position
                for buy_position, buy_session in correct_buy_events
            ):
                strict_functional_chain = True

        legacy_functional_chain = (
            not has_memory_id
            and any(
                write < retrieve < buy
                for write in write_positions
                for retrieve in relevant_retrieve_positions
                for buy in dependent_buy_positions
            )
        )
        workspace_success_candidate = False
        for shell_position, shell_session, visible_versions in shell_events:
            follows_persisted_source_write = any(
                source_versions.intersection(visible_versions)
                and source_session < shell_session
                and source_buy_position < shell_position
                for (
                    source_versions,
                    _write_position,
                    source_session,
                    source_buy_position,
                ) in source_workspace_writes
            )
            if not follows_persisted_source_write:
                continue
            counts["later_session_shell_after_source_write_count"] += 1
            if any(
                buy_session == shell_session and buy_position > shell_position
                for buy_position, buy_session in correct_buy_events
            ):
                workspace_success_candidate = True
        if workspace_success_candidate:
            counts["workspace_cross_session_success_candidate_count"] += 1
        if strict_functional_chain or legacy_functional_chain:
            counts["functional_memory_chain_count"] += 1
        if workspace_enabled:
            if final_workspace_snapshot is None:
                raise ValueError("Filesystem trajectory has no final workspace snapshot.")
            workspace_final_file_counts.append(
                float(final_workspace_snapshot["file_count"])
            )
            workspace_final_total_bytes.append(
                float(final_workspace_snapshot["total_bytes"])
            )
            workspace_final_audit_counts.append(
                float(previous_workspace_audit_count or 0)
            )

    result = {
        "trajectory_count": float(len(trajectories)),
        "action_row_count": float(len(rows)),
        "trajectory_return_mean": _mean(trajectory_returns),
        "immediate_reward_per_action_mean": _mean(immediate_rewards),
        "suffix_return_per_action_mean": _mean(suffix_returns),
    }
    for name in (
        "correct_buy_count",
        "wrong_buy_count",
        "timeout_trajectory_count",
        "session_advance_count",
        "source_memory_add_progress_count",
        "first_valid_add_count",
        "first_valid_later_session_retrieve_count",
        "empty_first_valid_later_session_retrieve_count",
        "nonempty_retrieve_count",
        "relevant_retrieve_count",
        "source_memory_write_before_correct_buy_count",
        "source_linked_retrieve_count",
        "workspace_action_count",
        "workspace_shell_command_count",
        "workspace_apply_patch_count",
        "workspace_mutating_action_count",
        "workspace_content_write_action_count",
        "workspace_delete_action_count",
        "workspace_shell_nonzero_exit_count",
        "workspace_shell_timeout_count",
        "source_workspace_write_before_correct_buy_count",
        "later_session_shell_after_source_write_count",
        "workspace_cross_session_success_candidate_count",
        "filesystem_trajectory_count",
        "workspace_trajectory_count",
        "workspace_enabled_trajectory_count",
        "workspace_no_workspace_trajectory_count",
        "workspace_snapshot_record_count",
        "workspace_nonempty_snapshot_record_count",
        "workspace_tree_change_count",
        "functional_memory_chain_count",
        "progress_ge_1_count",
        "progress_ge_2_count",
        "terminal_success_count",
        "invalid_action_count",
    ):
        result[name] = float(counts[name])
    result["workspace_final_file_count_mean"] = _mean(
        workspace_final_file_counts
    )
    result["workspace_final_total_bytes_mean"] = _mean(
        workspace_final_total_bytes
    )
    result["workspace_final_audit_event_count_mean"] = _mean(
        workspace_final_audit_counts
    )
    for kind in ("correct_buy", "wrong_buy", "timeout"):
        values = terminal_advantages[kind]
        positive = sum(value > 0.0 for value in values)
        result[f"{kind}_positive_advantage_count"] = float(positive)
        result[f"{kind}_positive_advantage_rate"] = (
            positive / len(values) if values else 0.0
        )
    return result
