"""Validate formal AgentMemoryGym BUY transitions without model dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class FormalBuyTransitionError(ValueError):
    """Raised when environment state contradicts the formal BUY contract."""


def validate_formal_buy_transition(
    *,
    tool_ops: Sequence[Any],
    env_step: int,
    subtask_index_before: int,
    subtask_index_after: int,
    done: bool,
) -> dict[str, Any]:
    """Validate one formal step and return normalized BUY evidence.

    Formal semantics are fail-fast: a correct BUY advances exactly one
    subtask, while an incorrect BUY commits the attempted purchase and ends
    the whole trajectory without advancing. Non-BUY actions cannot advance.
    """

    before = int(subtask_index_before)
    after = int(subtask_index_after)
    if after < before or after > before + 1:
        raise FormalBuyTransitionError(
            f"invalid subtask transition: before={before} after={after}"
        )
    session_advanced = after > before
    buy_ops = [
        tool_op
        for tool_op in tool_ops
        if isinstance(tool_op, dict)
        and str(tool_op.get("op", "")).upper() == "BUY"
        and int(tool_op.get("step", -1)) == int(env_step)
    ]
    if len(buy_ops) > 1:
        raise FormalBuyTransitionError(
            f"multiple BUY records for env step {env_step}: {len(buy_ops)}"
        )
    if not buy_ops:
        if session_advanced:
            raise FormalBuyTransitionError(
                "session advanced without a committed BUY record: "
                f"before={before} after={after}"
            )
        return {
            "committed_purchase": False,
            "purchase_correct": None,
            "accepted_purchase": False,
            "session_advanced": False,
            "buy_record": None,
        }

    buy_record = buy_ops[0]
    required = {"committed", "purchase_correct", "session_advanced", "terminal"}
    missing = sorted(required - set(buy_record))
    if missing:
        raise FormalBuyTransitionError(
            f"committed BUY record missing fields: {missing}"
        )
    if buy_record["committed"] is not True:
        raise FormalBuyTransitionError("formal BUY record is not committed")
    for field in ("purchase_correct", "session_advanced", "terminal"):
        if not isinstance(buy_record[field], bool):
            raise FormalBuyTransitionError(
                f"formal BUY field {field!r} must be bool"
            )

    purchase_correct = buy_record["purchase_correct"]
    if buy_record["session_advanced"] != session_advanced:
        raise FormalBuyTransitionError(
            "BUY record/session index disagree about advancement: "
            f"record={buy_record['session_advanced']} observed={session_advanced}"
        )
    if buy_record["terminal"] != bool(done):
        raise FormalBuyTransitionError(
            "BUY record/environment disagree about terminal state: "
            f"record={buy_record['terminal']} observed={bool(done)}"
        )
    if purchase_correct and not session_advanced:
        raise FormalBuyTransitionError("correct BUY did not advance the session")
    if not purchase_correct:
        if session_advanced:
            raise FormalBuyTransitionError("incorrect BUY advanced the session")
        if not done:
            raise FormalBuyTransitionError(
                "incorrect BUY did not terminate the formal trajectory"
            )

    return {
        "committed_purchase": True,
        "purchase_correct": purchase_correct,
        "accepted_purchase": purchase_correct,
        "session_advanced": session_advanced,
        "buy_record": buy_record,
    }
