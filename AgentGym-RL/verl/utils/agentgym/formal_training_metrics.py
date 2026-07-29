from __future__ import annotations

from collections import defaultdict
import math
from typing import Any


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


def _current_memory_ops(record: dict[str, Any]) -> list[dict[str, Any]]:
    info = record.get("env_info_after")
    if not isinstance(info, dict):
        return []
    memory_ops = info.get("memory_ops")
    if not isinstance(memory_ops, list):
        return []
    return [item for item in memory_ops if isinstance(item, dict)]


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

        write_positions: list[int] = []
        relevant_retrieve_positions: list[int] = []
        dependent_buy_positions: list[int] = []
        memory_write_events: list[tuple[set[str], int, int]] = []
        memory_retrieve_events: list[tuple[set[str], int, int]] = []
        correct_buy_events: list[tuple[int, int]] = []
        has_memory_id = False
        max_progress = 0
        terminal_success = False
        for position, row in enumerate(ordered):
            record = row["record"]
            component_names = _component_names(record)
            memory_ops = _current_memory_ops(record)
            session_index = int(record.get("subtask_index_before", 0))
            max_progress = max(
                max_progress,
                int(record.get("subtask_index_after", record.get("next_session_index", 0))),
            )
            # Native records may omit action_execution. The environment ledger is
            # the authoritative parser outcome for the current action.
            if "invalid_action" in component_names:
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

            if bool(record.get("buy_accepted")):
                counts["correct_buy_count"] += 1
                correct_buy_events.append((position, session_index))
                if int(record.get("subtask_index_before", 0)) >= 1:
                    dependent_buy_positions.append(position)
                terminal_advantages["correct_buy"].append(
                    float(row["advantage_token_mean"])
                )
            elif bool(record.get("buy_committed")):
                counts["wrong_buy_count"] += 1
                terminal_advantages["wrong_buy"].append(
                    float(row["advantage_token_mean"])
                )
            if bool(record.get("session_advanced")):
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
        if strict_functional_chain or legacy_functional_chain:
            counts["functional_memory_chain_count"] += 1

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
        "functional_memory_chain_count",
        "progress_ge_1_count",
        "progress_ge_2_count",
        "terminal_success_count",
        "invalid_action_count",
    ):
        result[name] = float(counts[name])
    for kind in ("correct_buy", "wrong_buy", "timeout"):
        values = terminal_advantages[kind]
        positive = sum(value > 0.0 for value in values)
        result[f"{kind}_positive_advantage_count"] = float(positive)
        result[f"{kind}_positive_advantage_rate"] = (
            positive / len(values) if values else 0.0
        )
    return result
