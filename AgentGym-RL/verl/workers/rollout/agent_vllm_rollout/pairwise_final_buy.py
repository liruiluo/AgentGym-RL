"""Pure validation and readback helpers for final-BUY pairwise PPO rows."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


UID_RE = re.compile(
    r"^(?P<parent>-?\d+):turn(?P<turn>\d+):statev1:(?P<digest>[0-9a-f]{64})$"
)


def parse_visible_final_buy(action: str) -> dict[str, Any]:
    """Validate one concrete memory-backed BUY without inferring correctness."""

    content = str(action).strip()
    if not content.startswith("BUY "):
        raise ValueError(f"pairwise action is not BUY: {content!r}")
    try:
        payload = json.loads(content[4:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"pairwise BUY is not valid JSON: {content!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("pairwise BUY payload must be an object")
    product_id = str(payload.get("product_id", ""))
    if not product_id:
        raise ValueError("pairwise BUY has no concrete product_id")
    if payload.get("memory_ids") != ["C0"]:
        raise ValueError(
            f"pairwise BUY must cite exactly memory_ids=[C0]: {payload.get('memory_ids')!r}"
        )
    why = str(payload.get("why", ""))
    if product_id not in why:
        raise ValueError("pairwise BUY evidence must name its visible product_id")
    return payload


def validate_visible_final_buy_pair(actions: Sequence[str]) -> list[dict[str, Any]]:
    if len(actions) != 2:
        raise ValueError(f"final-BUY pair must contain exactly two actions: {len(actions)}")
    parsed = [parse_visible_final_buy(action) for action in actions]
    product_ids = {str(payload["product_id"]) for payload in parsed}
    if len(product_ids) != 2 or len({str(action).strip() for action in actions}) != 2:
        raise ValueError("final-BUY pair must contain two distinct visible actions")
    return parsed


def validate_pairwise_packing(
    *,
    parent_indices: Sequence[int],
    task_rounds: Sequence[int],
    uids: Sequence[str],
    actions: Sequence[str],
    rewards: Sequence[float],
    errors: Sequence[str | None],
    expected_parent_count: int,
    tolerance: float = 1e-6,
) -> dict[str, int]:
    lengths = {
        len(parent_indices),
        len(task_rounds),
        len(uids),
        len(actions),
        len(rewards),
        len(errors),
    }
    if len(lengths) != 1:
        raise ValueError(f"pairwise packing arrays have different lengths: {sorted(lengths)}")
    expected_rows = int(expected_parent_count) * 2
    if not lengths or next(iter(lengths)) != expected_rows:
        raise ValueError(
            f"pairwise packing must contain {expected_rows} rows: {next(iter(lengths), 0)}"
        )

    groups: dict[str, list[int]] = defaultdict(list)
    parent_uids: dict[int, set[str]] = defaultdict(set)
    for index, (parent, task_round, uid, error) in enumerate(
        zip(parent_indices, task_rounds, uids, errors)
    ):
        if error is not None:
            raise ValueError(f"pairwise replay row {index} failed: {error}")
        match = UID_RE.fullmatch(str(uid))
        if match is None:
            raise ValueError(f"pairwise row {index} has invalid exact-state UID: {uid!r}")
        if int(task_round) != 4 or int(match.group("turn")) != 4:
            raise ValueError(f"pairwise row {index} is not a turn-4 action")
        if int(parent) != int(match.group("parent")):
            raise ValueError(f"pairwise row {index} parent/UID mismatch")
        groups[str(uid)].append(index)
        parent_uids[int(parent)].add(str(uid))

    if len(groups) != expected_parent_count or len(parent_uids) != expected_parent_count:
        raise ValueError(
            "pairwise packing did not produce one exact-state group per parent: "
            f"groups={len(groups)} parents={len(parent_uids)} expected={expected_parent_count}"
        )
    if any(len(values) != 1 for values in parent_uids.values()):
        raise ValueError("a pairwise parent was packed from multiple prompt-state UIDs")

    for uid, indices in groups.items():
        if len(indices) != 2:
            raise ValueError(f"exact-state group {uid} does not contain two rows")
        validate_visible_final_buy_pair([actions[index] for index in indices])
        group_rewards = sorted(float(rewards[index]) for index in indices)
        if any(not math.isfinite(value) for value in group_rewards):
            raise ValueError(f"exact-state group {uid} has non-finite rewards")
        if (
            abs(group_rewards[0] - (-0.01)) > tolerance
            or abs(group_rewards[1] - 2.0) > tolerance
        ):
            raise ValueError(
                f"exact-state group {uid} rewards are not environment pair [-0.01, 2.0]: "
                f"{group_rewards}"
            )
    return {
        "rows": expected_rows,
        "exact_state_groups": len(groups),
        "parents": len(parent_uids),
    }


def summarize_pairwise_logprob_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_group_count: int,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        uid = str(row.get("uid", ""))
        if UID_RE.fullmatch(uid) is None:
            raise ValueError(f"readback row {index} has invalid UID: {uid!r}")
        if row.get("ppo_valid_sample") is not True:
            raise ValueError(f"readback row {index} is not actor-valid")
        parse_visible_final_buy(str(row.get("action", "")))
        for key in ("reward", "before_seq_logp", "after_seq_logp"):
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"readback row {index} has non-finite {key}: {value}")
        if int(row.get("response_tokens", 0)) <= 0:
            raise ValueError(f"readback row {index} has no response tokens")
        groups[uid].append(dict(row))

    if len(groups) != expected_group_count:
        raise ValueError(
            f"readback expected {expected_group_count} exact-state groups: {len(groups)}"
        )
    summaries = []
    for uid, members in sorted(groups.items()):
        if len(members) != 2:
            raise ValueError(f"readback group {uid} does not contain two rows")
        validate_visible_final_buy_pair([str(row["action"]) for row in members])
        ordered = sorted(members, key=lambda row: float(row["reward"]))
        wrong, correct = ordered
        if (
            abs(float(wrong["reward"]) - (-0.01)) > tolerance
            or abs(float(correct["reward"]) - 2.0) > tolerance
        ):
            raise ValueError(f"readback group {uid} has unexpected reward pair")
        before_margin = float(correct["before_seq_logp"]) - float(wrong["before_seq_logp"])
        after_margin = float(correct["after_seq_logp"]) - float(wrong["after_seq_logp"])
        summaries.append(
            {
                "uid": uid,
                "parent_index": int(correct["parent_index"]),
                "correct_action": str(correct["action"]),
                "wrong_action": str(wrong["action"]),
                "before_margin": before_margin,
                "after_margin": after_margin,
                "margin_delta": after_margin - before_margin,
            }
        )

    before_mean = sum(row["before_margin"] for row in summaries) / len(summaries)
    after_mean = sum(row["after_margin"] for row in summaries) / len(summaries)
    nondecreasing = sum(
        row["after_margin"] + tolerance >= row["before_margin"] for row in summaries
    )
    max_abs_logp_delta = max(
        abs(float(row["after_seq_logp"]) - float(row["before_seq_logp"]))
        for row in rows
    )
    return {
        "rows": len(rows),
        "exact_state_groups": len(groups),
        "before_mean_margin": before_mean,
        "after_mean_margin": after_mean,
        "mean_margin_delta": after_mean - before_mean,
        "nondecreasing_margin_states": nondecreasing,
        "max_abs_seq_logp_delta": max_abs_logp_delta,
        "groups": summaries,
    }
