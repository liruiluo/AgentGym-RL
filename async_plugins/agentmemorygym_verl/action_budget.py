"""Validation for task-neutral combined policy and auxiliary step budgets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentenv.controller.types import (
    TASK_NEUTRAL_ACTION_BUDGET_SCHEMA,
    PolicyActionBudget,
    build_task_neutral_action_budget_receipt,
)


_FIELDS = {
    "schema",
    "maximum_steps",
    "consumed_steps_before",
    "policy_action_steps",
    "auxiliary_steps",
    "required_auxiliary_steps",
    "consumed_steps_after",
    "remaining_steps_after",
    "atomic_operation_blocked",
    "terminate_after_action",
}


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(
            f"action-budget receipt {field} must be an integer >= {minimum}"
        )
    return value


def normalize_action_budget_receipt(
    value: Mapping[str, Any] | None,
    *,
    maximum_steps: int,
    consumed_steps_before: int,
    allow_implicit_policy_action: bool,
) -> dict[str, Any]:
    """Return a canonical receipt, rejecting gaps or arithmetic drift."""

    budget = PolicyActionBudget(
        maximum_steps=maximum_steps,
        consumed_steps=consumed_steps_before,
    )
    if value is None:
        if not allow_implicit_policy_action:
            raise RuntimeError("action-budget receipt is missing")
        return build_task_neutral_action_budget_receipt(budget)
    if not isinstance(value, Mapping):
        raise RuntimeError("action-budget receipt must be a mapping")
    observed_fields = set(value)
    if observed_fields != _FIELDS:
        missing = sorted(_FIELDS - observed_fields)
        extra = sorted(observed_fields - _FIELDS)
        raise RuntimeError(
            "action-budget receipt fields drifted: "
            f"missing={missing!r} extra={extra!r}"
        )
    if value.get("schema") != TASK_NEUTRAL_ACTION_BUDGET_SCHEMA:
        raise RuntimeError("action-budget receipt schema drifted")
    maximum = _integer(value.get("maximum_steps"), field="maximum_steps", minimum=1)
    before = _integer(
        value.get("consumed_steps_before"), field="consumed_steps_before"
    )
    policy = _integer(value.get("policy_action_steps"), field="policy_action_steps")
    auxiliary = _integer(value.get("auxiliary_steps"), field="auxiliary_steps")
    required = _integer(
        value.get("required_auxiliary_steps"),
        field="required_auxiliary_steps",
    )
    after = _integer(value.get("consumed_steps_after"), field="consumed_steps_after")
    remaining = _integer(
        value.get("remaining_steps_after"), field="remaining_steps_after"
    )
    blocked = value.get("atomic_operation_blocked")
    terminate = value.get("terminate_after_action")
    if not isinstance(blocked, bool) or not isinstance(terminate, bool):
        raise RuntimeError("action-budget receipt flags must be boolean")
    if maximum != maximum_steps or before != consumed_steps_before:
        raise RuntimeError("action-budget receipt does not match runner-owned state")
    if policy != 1:
        raise RuntimeError("action-budget receipt must charge one policy action step")
    if after != before + policy + auxiliary:
        raise RuntimeError("action-budget receipt consumed-step arithmetic drifted")
    if after > maximum or remaining != maximum - after:
        raise RuntimeError(
            "action-budget receipt exceeds or misstates the global budget"
        )
    if blocked:
        if auxiliary != 0 or required <= 0 or remaining >= required:
            raise RuntimeError(
                "action-budget receipt has an invalid blocked atomic group"
            )
        if terminate is not True:
            raise RuntimeError(
                "blocked atomic work must terminate after the policy action"
            )
    else:
        if required != auxiliary:
            raise RuntimeError("completed auxiliary work was not charged atomically")
        if terminate is not False:
            raise RuntimeError("unblocked action-budget receipt cannot force termination")
    return {field: value[field] for field in sorted(_FIELDS)}


__all__ = ["normalize_action_budget_receipt"]
