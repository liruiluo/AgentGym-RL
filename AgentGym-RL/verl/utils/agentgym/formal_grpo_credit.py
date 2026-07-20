"""Fail-closed whole-trajectory GRPO credit for action-row rollouts.

This module is deliberately free of torch/numpy dependencies so the metadata
contract can be tested before a distributed training process is armed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence


REQUIRED_ROW_FIELDS = frozenset(
    {
        "parent_uid",
        "trajectory_uid",
        "row_uid",
        "row_order",
        "terminal",
        "immediate_reward",
    }
)


@dataclass(frozen=True)
class CreditRow:
    """One validated policy-action row, aligned to the input row order."""

    input_index: int
    parent_uid: str
    trajectory_uid: str
    row_uid: str
    row_order: int
    terminal: bool
    immediate_reward: float
    suffix_return: float
    trajectory_return: float
    advantage: float


@dataclass(frozen=True)
class TrajectoryCredit:
    """Credit assigned once to one complete continuation."""

    parent_uid: str
    trajectory_uid: str
    row_count: int
    trajectory_return: float
    advantage: float


@dataclass(frozen=True)
class ParentCreditGroup:
    """A GRPO comparison group rooted at one initial task/prompt."""

    parent_uid: str
    trajectory_uids: tuple[str, ...]
    mean_return: float
    sample_std: float


@dataclass(frozen=True)
class FormalGrpoCredit:
    """Validated row, trajectory, and parent-group credit."""

    rows: tuple[CreditRow, ...]
    trajectories: tuple[TrajectoryCredit, ...]
    groups: tuple[ParentCreditGroup, ...]


def build_row_uid(trajectory_uid: str, row_order: int) -> str:
    """Bind a stable row identity to a trajectory and environment-time index."""

    payload = json.dumps(
        [str(trajectory_uid), int(row_order)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "agentmemory:trajrowv1:" + hashlib.sha256(payload).hexdigest()


def _require_nonempty_text(value: Any, *, name: str, row_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text at row {row_index}.")
    return value


def _require_nonnegative_int(value: Any, *, name: str, row_index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer at row {row_index}.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer at row {row_index}.")
    return normalized


def _require_finite_real(value: Any, *, name: str, row_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real at row {row_index}.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real at row {row_index}.")
    return normalized


def _close(left: float, right: float, *, tolerance: float) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def compute_formal_grpo_credit(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_group_size: int,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    declared_return_field: str | None = "declared_trajectory_return",
    allow_singleton_group: bool = False,
) -> FormalGrpoCredit:
    """Validate complete trajectories, normalize returns, and broadcast credit.

    Rows may arrive in any batch order. ``row_order`` is environment time within
    one trajectory, while ``row_uid`` detects an independently shuffled metadata
    column. A finite-horizon continuation is complete only when its orders are
    contiguous from zero and exactly its final row is marked ``terminal``.

    ``declared_return_field`` is audit evidence only. The training target is
    always recomputed from ordered immediate rewards.
    """

    if isinstance(expected_group_size, bool) or not isinstance(
        expected_group_size, Integral
    ):
        raise ValueError("expected_group_size must be an integer.")
    expected_group_size = int(expected_group_size)
    minimum_group_size = 1 if allow_singleton_group else 2
    if expected_group_size < minimum_group_size:
        raise ValueError(
            "Formal GRPO requires expected_group_size >= "
            f"{minimum_group_size}."
        )
    gamma = _require_finite_real(gamma, name="gamma", row_index=-1)
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
    epsilon = _require_finite_real(epsilon, name="epsilon", row_index=-1)
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("Formal GRPO requires at least one action row.")

    validated_rows: list[dict[str, Any]] = []
    seen_row_uids: set[str] = set()
    seen_trajectory_orders: set[tuple[str, int]] = set()
    trajectory_owner: dict[str, str] = {}
    declared_presence: list[bool] = []

    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Formal GRPO row {row_index} must be a mapping.")
        missing = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing:
            raise ValueError(
                f"Formal GRPO row {row_index} is missing metadata: {missing}"
            )
        parent_uid = _require_nonempty_text(
            row["parent_uid"], name="parent_uid", row_index=row_index
        )
        trajectory_uid = _require_nonempty_text(
            row["trajectory_uid"], name="trajectory_uid", row_index=row_index
        )
        row_uid = _require_nonempty_text(
            row["row_uid"], name="row_uid", row_index=row_index
        )
        row_order = _require_nonnegative_int(
            row["row_order"], name="row_order", row_index=row_index
        )
        terminal = row["terminal"]
        if not isinstance(terminal, bool):
            raise ValueError(f"terminal must be boolean at row {row_index}.")
        immediate_reward = _require_finite_real(
            row["immediate_reward"],
            name="immediate_reward",
            row_index=row_index,
        )

        expected_row_uid = build_row_uid(trajectory_uid, row_order)
        if row_uid != expected_row_uid:
            raise ValueError(
                "Formal GRPO row drift detected: "
                f"row={row_index} uid={row_uid!r} expected={expected_row_uid!r}."
            )
        if row_uid in seen_row_uids:
            raise ValueError(f"Duplicate formal GRPO row_uid at row {row_index}.")
        seen_row_uids.add(row_uid)
        trajectory_order = (trajectory_uid, row_order)
        if trajectory_order in seen_trajectory_orders:
            raise ValueError(
                "Duplicate row_order within one trajectory: "
                f"trajectory={trajectory_uid!r} order={row_order}."
            )
        seen_trajectory_orders.add(trajectory_order)

        previous_parent = trajectory_owner.setdefault(trajectory_uid, parent_uid)
        if previous_parent != parent_uid:
            raise ValueError(
                "Trajectory crosses parent groups: "
                f"trajectory={trajectory_uid!r} "
                f"parents={previous_parent!r},{parent_uid!r}."
            )

        declared_present = bool(
            declared_return_field is not None and declared_return_field in row
        )
        declared_presence.append(declared_present)
        declared_return = None
        if declared_present:
            declared_return = _require_finite_real(
                row[declared_return_field],
                name=str(declared_return_field),
                row_index=row_index,
            )
        validated_rows.append(
            {
                "input_index": row_index,
                "parent_uid": parent_uid,
                "trajectory_uid": trajectory_uid,
                "row_uid": row_uid,
                "row_order": row_order,
                "terminal": terminal,
                "immediate_reward": immediate_reward,
                "declared_return": declared_return,
            }
        )

    if any(declared_presence) and not all(declared_presence):
        raise ValueError(
            "Declared trajectory return metadata must be present on every row or none."
        )

    rows_by_trajectory: dict[str, list[dict[str, Any]]] = {}
    for row in validated_rows:
        rows_by_trajectory.setdefault(row["trajectory_uid"], []).append(row)

    trajectory_data: dict[str, dict[str, Any]] = {}
    trajectories_by_parent: dict[str, list[str]] = {}
    for trajectory_uid, trajectory_rows in rows_by_trajectory.items():
        trajectory_rows.sort(key=lambda row: row["row_order"])
        actual_orders = [row["row_order"] for row in trajectory_rows]
        expected_orders = list(range(len(trajectory_rows)))
        if actual_orders != expected_orders:
            raise ValueError(
                "Formal GRPO trajectory row_order is not contiguous from zero: "
                f"trajectory={trajectory_uid!r} "
                f"expected={expected_orders} actual={actual_orders}."
            )
        terminal_orders = [
            row["row_order"] for row in trajectory_rows if row["terminal"]
        ]
        expected_terminal = len(trajectory_rows) - 1
        if terminal_orders != [expected_terminal]:
            raise ValueError(
                "Formal GRPO trajectory must have exactly one terminal final row: "
                f"trajectory={trajectory_uid!r} "
                f"expected={[expected_terminal]} actual={terminal_orders}."
            )
        trajectory_return = math.fsum(
            (gamma ** row["row_order"]) * row["immediate_reward"]
            for row in trajectory_rows
        )
        running_suffix = 0.0
        for row in reversed(trajectory_rows):
            running_suffix = row["immediate_reward"] + gamma * running_suffix
            row["suffix_return"] = running_suffix
        if declared_return_field is not None and all(declared_presence):
            for row in trajectory_rows:
                if not _close(
                    row["declared_return"],
                    trajectory_return,
                    tolerance=epsilon,
                ):
                    raise ValueError(
                        "Declared trajectory return disagrees with ordered immediate "
                        f"rewards: trajectory={trajectory_uid!r} "
                        f"declared={row['declared_return']} "
                        f"computed={trajectory_return}."
                    )
        parent_uid = trajectory_rows[0]["parent_uid"]
        trajectory_data[trajectory_uid] = {
            "parent_uid": parent_uid,
            "rows": trajectory_rows,
            "return": trajectory_return,
        }
        trajectories_by_parent.setdefault(parent_uid, []).append(trajectory_uid)

    for parent_uid, trajectory_uids in trajectories_by_parent.items():
        if len(trajectory_uids) != expected_group_size:
            raise ValueError(
                "Formal GRPO parent group is incomplete: "
                f"parent={parent_uid!r} expected={expected_group_size} "
                f"actual={len(trajectory_uids)}."
            )

    advantage_by_trajectory: dict[str, float] = {}
    group_summaries: list[ParentCreditGroup] = []
    for parent_uid in sorted(trajectories_by_parent):
        trajectory_uids = sorted(trajectories_by_parent[parent_uid])
        returns = [trajectory_data[uid]["return"] for uid in trajectory_uids]
        mean_return = math.fsum(returns) / len(returns)
        if len(returns) == 1:
            sample_std = 0.0
        else:
            sample_variance = math.fsum(
                (value - mean_return) ** 2 for value in returns
            ) / (len(returns) - 1)
            sample_std = math.sqrt(max(sample_variance, 0.0))
        for trajectory_uid, trajectory_return in zip(trajectory_uids, returns):
            if sample_std == 0.0:
                advantage = 0.0
            else:
                advantage = (trajectory_return - mean_return) / (
                    sample_std + epsilon
                )
            advantage_by_trajectory[trajectory_uid] = advantage
        group_summaries.append(
            ParentCreditGroup(
                parent_uid=parent_uid,
                trajectory_uids=tuple(trajectory_uids),
                mean_return=mean_return,
                sample_std=sample_std,
            )
        )

    output_rows: list[CreditRow] = []
    for row in sorted(validated_rows, key=lambda item: item["input_index"]):
        trajectory_uid = row["trajectory_uid"]
        output_rows.append(
            CreditRow(
                input_index=row["input_index"],
                parent_uid=row["parent_uid"],
                trajectory_uid=trajectory_uid,
                row_uid=row["row_uid"],
                row_order=row["row_order"],
                terminal=row["terminal"],
                immediate_reward=row["immediate_reward"],
                suffix_return=row["suffix_return"],
                trajectory_return=trajectory_data[trajectory_uid]["return"],
                advantage=advantage_by_trajectory[trajectory_uid],
            )
        )

    trajectory_summaries = [
        TrajectoryCredit(
            parent_uid=data["parent_uid"],
            trajectory_uid=trajectory_uid,
            row_count=len(data["rows"]),
            trajectory_return=data["return"],
            advantage=advantage_by_trajectory[trajectory_uid],
        )
        for trajectory_uid, data in sorted(trajectory_data.items())
    ]
    return FormalGrpoCredit(
        rows=tuple(output_rows),
        trajectories=tuple(trajectory_summaries),
        groups=tuple(group_summaries),
    )


def broadcast_action_token_advantages(
    row_advantages: Sequence[Real],
    response_masks: Sequence[Sequence[Any]],
) -> list[list[float]]:
    """Broadcast one trajectory advantage to every policy token in each row."""

    if len(row_advantages) != len(response_masks):
        raise ValueError(
            "row_advantages and response_masks must have the same row count."
        )
    token_advantages: list[list[float]] = []
    for row_index, (row_advantage, response_mask) in enumerate(
        zip(row_advantages, response_masks)
    ):
        advantage = _require_finite_real(
            row_advantage, name="row advantage", row_index=row_index
        )
        if not isinstance(response_mask, Sequence) or isinstance(
            response_mask, (str, bytes)
        ):
            raise ValueError(f"response mask must be a sequence at row {row_index}.")
        normalized_mask: list[bool] = []
        for token_index, value in enumerate(response_mask):
            if isinstance(value, bool):
                normalized_mask.append(value)
            elif isinstance(value, Integral) and int(value) in (0, 1):
                normalized_mask.append(bool(value))
            else:
                raise ValueError(
                    "response mask values must be boolean/0/1: "
                    f"row={row_index} token={token_index} value={value!r}."
                )
        if not any(normalized_mask):
            raise ValueError(f"Policy action row {row_index} has no response tokens.")
        token_advantages.append(
            [advantage if visible else 0.0 for visible in normalized_mask]
        )
    return token_advantages
