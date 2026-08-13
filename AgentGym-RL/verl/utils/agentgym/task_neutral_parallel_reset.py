"""Ordered concurrent reset for independent task-neutral rollout clients."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Sequence


PolicyMessage = dict[str, str]


@dataclass(frozen=True)
class ResetItemTiming:
    index: int
    reset_index: int
    started_offset_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class ParallelResetResult:
    policy_messages: tuple[tuple[PolicyMessage, ...], ...]
    excluded_indices: frozenset[int]
    item_timings: tuple[ResetItemTiming, ...]
    wall_seconds: float


@dataclass(frozen=True)
class _ResetOutcome:
    index: int
    reset_index: int
    messages: tuple[PolicyMessage, ...]
    excluded: bool
    error: Exception | None
    started_offset_seconds: float
    elapsed_seconds: float


def reset_task_neutral_policy_contexts(
    rollout_handlers: Sequence[Any],
    env_clients: Sequence[Any],
    *,
    resolve_reset_index: Callable[[Any], int],
    bind_initial_policy_context: Callable[
        [Any, Sequence[Mapping[str, str]]], list[PolicyMessage]
    ],
    max_workers: int | None = None,
) -> ParallelResetResult:
    """Reset independent clients concurrently while preserving batch order.

    Workers own only their matching ``(handler, client)`` pair. Results and
    failures are consumed in input-index order, matching the old serial loop's
    deterministic row/error semantics.
    """

    if len(rollout_handlers) != len(env_clients):
        raise ValueError(
            "task-neutral rollout client count must match handler count: "
            f"handlers={len(rollout_handlers)} clients={len(env_clients)}"
        )
    if not rollout_handlers:
        return ParallelResetResult(
            policy_messages=(),
            excluded_indices=frozenset(),
            item_timings=(),
            wall_seconds=0.0,
        )
    if max_workers is None:
        max_workers = len(rollout_handlers)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError("task-neutral reset max_workers must be an integer")
    if max_workers <= 0:
        raise ValueError("task-neutral reset max_workers must be positive")
    max_workers = min(max_workers, len(rollout_handlers))

    batch_started = time.perf_counter()

    def reset_one(index: int) -> _ResetOutcome:
        handler = rollout_handlers[index]
        client = env_clients[index]
        started = time.perf_counter()
        reset_index = -1
        try:
            reset_index = int(resolve_reset_index(handler))
            client.reset(reset_index)
            initial_messages = [
                message.to_dict() for message in handler.messages
            ]
            initial_messages.append(
                {"role": "user", "content": str(client.observe())}
            )
            messages = bind_initial_policy_context(client, initial_messages)
            return _ResetOutcome(
                index=index,
                reset_index=reset_index,
                messages=tuple(dict(message) for message in messages),
                excluded=False,
                error=None,
                started_offset_seconds=started - batch_started,
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as exc:
            return _ResetOutcome(
                index=index,
                reset_index=reset_index,
                messages=(),
                excluded=bool(getattr(client, "sample_excluded", False)),
                error=exc,
                started_offset_seconds=started - batch_started,
                elapsed_seconds=time.perf_counter() - started,
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        outcomes = tuple(executor.map(reset_one, range(len(rollout_handlers))))

    policy_messages: list[tuple[PolicyMessage, ...]] = [
        () for _ in rollout_handlers
    ]
    excluded_indices: set[int] = set()
    item_timings: list[ResetItemTiming] = []
    for outcome in outcomes:
        item_timings.append(
            ResetItemTiming(
                index=outcome.index,
                reset_index=outcome.reset_index,
                started_offset_seconds=outcome.started_offset_seconds,
                elapsed_seconds=outcome.elapsed_seconds,
            )
        )
        if outcome.error is not None:
            if outcome.excluded:
                excluded_indices.add(outcome.index)
                rollout_handlers[outcome.index].done = True
                continue
            handler = rollout_handlers[outcome.index]
            raise RuntimeError(
                "task-neutral environment reset failed: "
                f"row={outcome.index} item_id={handler.item_id} "
                f"error={type(outcome.error).__name__}: {outcome.error}"
            ) from outcome.error
        policy_messages[outcome.index] = outcome.messages

    return ParallelResetResult(
        policy_messages=tuple(policy_messages),
        excluded_indices=frozenset(excluded_indices),
        item_timings=tuple(item_timings),
        wall_seconds=time.perf_counter() - batch_started,
    )
