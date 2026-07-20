"""Pure helpers for AgentMemory rollout alignment, grouping, and credit."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from verl.utils.agentgym.rollout_context import (
    AGENTMEMORY_EXACT_STATE_UID,
    AGENTMEMORY_IMMEDIATE_REWARD,
    AGENTMEMORY_PARENT_GROUP_UID,
    AGENTMEMORY_REPLICA_INDEX,
    AGENTMEMORY_TRAJECTORY_RETURN,
    AGENTMEMORY_TRAJECTORY_ROW_ORDER,
    AGENTMEMORY_TRAJECTORY_ROW_UID,
    AGENTMEMORY_TRAJECTORY_TERMINAL,
    AGENTMEMORY_TRAJECTORY_UID,
    build_parent_group_uid,
    build_row_uid,
    build_trajectory_uid,
    validate_formal_trajectory_rows,
)


def resolve_rollout_parent_index(
    local_index: int,
    *,
    source_parent_indices: Sequence[Any] | None = None,
    eval_parent_indices: Sequence[Any] | None = None,
) -> int:
    """Resolve a worker-local row to its driver-global source row."""

    indices = eval_parent_indices
    if indices is None:
        indices = source_parent_indices
    if indices is None:
        return int(local_index)
    try:
        return int(indices[local_index])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid rollout parent mapping: "
            f"local_index={local_index} mapping_size={len(indices)}"
        ) from exc


def prompt_state_digest(prompt_token_ids: Sequence[int]) -> str:
    """Hash the exact token sequence that conditions the policy action."""

    digest = hashlib.sha256()
    for token_id in prompt_token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def build_state_aware_rollout_uid(
    parent_index: int,
    task_round: int,
    prompt_token_ids: Sequence[int],
) -> str:
    """Group only actions conditioned on the same source and prompt state."""

    return (
        f"{int(parent_index)}:turn{int(task_round)}:"
        f"statev1:{prompt_state_digest(prompt_token_ids)}"
    )


def compute_suffix_credit_scores(
    scores: Sequence[float],
    task_rounds: Sequence[int],
) -> list[float]:
    """Compute the undiscounted return-to-go for each sampled action row."""

    if len(scores) != len(task_rounds):
        raise ValueError(
            "scores and task_rounds must have the same length: "
            f"scores={len(scores)} rounds={len(task_rounds)}"
        )
    suffix_scores = [0.0] * len(scores)
    running_score = 0.0
    for index in range(len(scores) - 1, -1, -1):
        running_score += float(scores[index])
        suffix_scores[index] = running_score
    return suffix_scores


def expand_excluded_rollout_parent_groups(
    rollout_parent_indices: Sequence[Any],
    excluded_rollout_indices: Sequence[int],
) -> set[int]:
    """Exclude every replica of a parent with one infrastructure failure."""

    row_count = len(rollout_parent_indices)
    excluded = {int(index) for index in excluded_rollout_indices}
    if any(index < 0 or index >= row_count for index in excluded):
        raise ValueError(
            "excluded rollout index is outside the rollout batch: "
            f"rows={row_count} excluded={sorted(excluded)}"
        )
    excluded_parents = {
        int(rollout_parent_indices[index]) for index in excluded
    }
    return {
        index
        for index, parent_index in enumerate(rollout_parent_indices)
        if int(parent_index) in excluded_parents
    }


def trainable_rollout_row_positions(
    flat_rollout_indices: Sequence[int],
    excluded_rollout_indices: Sequence[int],
) -> list[int]:
    """Return aligned flat-row positions that survive rollout exclusion."""

    excluded = {int(index) for index in excluded_rollout_indices}
    return [
        position
        for position, rollout_index in enumerate(flat_rollout_indices)
        if int(rollout_index) not in excluded
    ]
