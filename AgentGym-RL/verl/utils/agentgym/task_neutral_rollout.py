"""Task-neutral policy-row helpers for the shared AgentGym rollout.

Environment wrappers own lifecycle decisions.  This module only normalizes
their receipt and preserves the exact sampled generation metadata needed by
the PPO readback validator.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from verl.utils.agentgym.rollout_context import TASK_NEUTRAL_POLICY_STEP_SCHEMA


_CONTEXT_TRANSITION_SCHEMA = "agentmemory_task_neutral_context_transition_v1"


def mapping_copy(value: Any) -> dict[str, Any]:
    """Return a detached object for opaque wrapper evidence."""

    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def env_info_from_payload(value: Any) -> dict[str, Any]:
    """Extract the wrapper's opaque environment-info object without semantics."""

    payload = value if isinstance(value, Mapping) else {}
    for key in ("env_info", "info"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return mapping_copy(candidate)
    return {}


def receipt_parts(step_output: Any, action: str) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Read a task-neutral receipt, supplying a legacy append fallback."""

    info = getattr(step_output, "info", {})
    info = info if isinstance(info, Mapping) else {}
    env_info = env_info_from_payload(info)
    action_submission = mapping_copy(info.get("action_submission"))
    if not action_submission:
        action_submission = {"raw_policy_output": str(action)}
    transition = info.get("context_transition")
    if not isinstance(transition, Mapping):
        transition = {
            "schema": _CONTEXT_TRANSITION_SCHEMA,
            "operation": "append_observation",
            "messages": [],
        }
    else:
        transition = mapping_copy(transition)
    wrapper_evidence = mapping_copy(info.get("wrapper_evidence"))
    return env_info, action_submission, transition, wrapper_evidence


def outcome_from_receipt(
    *,
    done: bool,
    reward: float,
    env_info_after: Mapping[str, Any],
    wrapper_evidence: Mapping[str, Any],
) -> str:
    """Choose an auditable label without changing the reward contract."""

    if not done:
        return "continue"
    explicit = wrapper_evidence.get("outcome")
    if explicit in {"success", "terminal_failure", "environment_error"}:
        return str(explicit)
    if env_info_after.get("episode_success") is True:
        return "success"
    if env_info_after.get("resolved") is True:
        return "success"
    if float(reward) > 0:
        return "success"
    return "terminal_failure"


def build_task_neutral_step_record(
    *,
    item_id: str,
    parent_index: int,
    parent_group_uid: str,
    replica_index: int,
    trajectory_uid: str,
    exact_state_uid: str,
    task_round: int,
    prompt_token_ids: list[int],
    content: str,
    response_token_ids: list[int],
    score: float,
    done: bool,
    generation_record: Mapping[str, Any],
    env_result: str,
    env_info_before: Mapping[str, Any],
    env_info_after: Mapping[str, Any],
    action_submission: Mapping[str, Any],
    context_transition: Mapping[str, Any],
    wrapper_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one lifecycle-neutral row before PPO packing adds exact digests."""

    return {
        "schema_version": TASK_NEUTRAL_POLICY_STEP_SCHEMA,
        "item_id": str(item_id),
        "parent_index": int(parent_index),
        "parent_group_uid": str(parent_group_uid),
        "replica_index": int(replica_index),
        "trajectory_uid": str(trajectory_uid),
        "exact_state_uid": str(exact_state_uid),
        "task_round": int(task_round),
        "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
        "content": str(content),
        "action": str(content),
        "score": float(score),
        "immediate_reward": float(score),
        "done": bool(done),
        "response_token_ids": [int(token_id) for token_id in response_token_ids],
        "response_token_count": int(generation_record["response_token_count"]),
        "max_response_tokens": int(generation_record["max_response_tokens"]),
        "finish_reason": str(generation_record["finish_reason"]),
        "finish_reason_source": str(generation_record["finish_reason_source"]),
        "stop_reason": generation_record.get("stop_reason"),
        "generation_backend_source": str(generation_record["backend_source"]),
        "generation_stop_reason": generation_record.get("stop_reason"),
        "generation_eos_token_ids": list(
            generation_record["configured_eos_token_ids"]
        ),
        "tokenizer_primary_eos_token_id": generation_record[
            "primary_eos_token_id"
        ],
        "tokenizer_pad_token_id": generation_record["tokenizer_pad_token_id"],
        "generation_token_ids_are_exact": bool(
            generation_record["token_ids_are_exact"]
        ),
        "backend_token_ids_are_exact": bool(
            generation_record["backend_token_ids_are_exact"]
        ),
        "truncated": bool(generation_record["truncated"]),
        "env_result": str(env_result),
        "env_info_before": mapping_copy(env_info_before),
        "env_info_after": mapping_copy(env_info_after),
        "action_submission": mapping_copy(action_submission),
        "context_transition": mapping_copy(context_transition),
        "wrapper_evidence": mapping_copy(wrapper_evidence),
        "outcome": outcome_from_receipt(
            done=bool(done),
            reward=float(score),
            env_info_after=env_info_after,
            wrapper_evidence=wrapper_evidence,
        ),
        # These fields are completed from the trajectory after the horizon.
        "trajectory_row_order": 0,
        "trajectory_row_uid": "",
        "trajectory_terminal": False,
        "suffix_return": float(score),
        "suffix_credit_applied": False,
        "trajectory_return": float(score),
    }
