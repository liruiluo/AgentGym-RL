"""One-pass, read-only evidence snapshots for a live fully-async run."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .finalizer import (
    _LEGACY_RECEIPT_SCHEMA,
    _MULTITASK_RECEIPT_SCHEMA,
    _at,
    _atomic_json,
    _Audit,
    _emitted_memory_events,
    _has_complete_memory_chain,
    _load_json,
    _nonnegative_integral,
    _path_overlaps_inputs,
    _path_within,
    _receipt_protected_paths,
    _rolling_episode_shares,
)

SNAPSHOT_UPDATES = frozenset({1, 5, 20, 40, 80})


def _validate_snapshot_update(update: int) -> None:
    if update not in SNAPSHOT_UPDATES:
        raise ValueError(
            f"snapshot update must be one of {sorted(SNAPSHOT_UPDATES)}, got {update}"
        )


def _complete_jsonl_rows(path: Path, label: str) -> list[Mapping[str, Any]]:
    """Read complete newline-terminated rows from one instantaneous file view."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    complete = payload.splitlines(keepends=True)
    if complete and not complete[-1].endswith((b"\n", b"\r")):
        complete.pop()
    rows: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(complete, start=1):
        if not raw.strip():
            raise ValueError(f"blank row in {label} at {path}:{line_number}")
        try:
            value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid complete row in {label} at {path}:{line_number}"
            ) from exc
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} row is not an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} has no complete rows: {path}")
    return rows


def _metric_value(items: Sequence[tuple[str, Any]], key: str, *, label: str) -> int:
    values = [value for item_key, value in items if item_key == key]
    parsed = _nonnegative_integral(values[0]) if len(values) == 1 else None
    if parsed is None:
        raise ValueError(f"{label} has no unique integral {key}")
    return parsed


def _metric_routes(
    items: Sequence[tuple[str, Any]],
    prefix: str,
    route_ids: Sequence[str],
    *,
    label: str,
) -> dict[str, int]:
    observed: dict[str, int] = {}
    for key, value in items:
        if not key.startswith(prefix):
            continue
        route_id = key.removeprefix(prefix)
        if route_id in observed:
            raise ValueError(f"{label} repeats route {route_id!r}")
        parsed = _nonnegative_integral(value)
        if route_id not in route_ids or parsed is None:
            raise ValueError(f"{label} has an invalid route/count")
        observed[route_id] = parsed
    return {route_id: observed.get(route_id, 0) for route_id in route_ids}


def _episode_summary(
    documents: Sequence[Mapping[str, Any]],
    *,
    update: int,
    route_ids: Sequence[str],
    samples_per_update: int,
    schedule_routes: Mapping[tuple[str, int], str],
    prior_trajectory_uids: set[str],
    prior_schedule_instances: set[tuple[str, int]],
) -> dict[str, Any]:
    episodes: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    action_rows: Counter[str] = Counter()
    response_tokens: Counter[str] = Counter()
    for document in documents:
        if document.get("step") != update:
            raise ValueError(f"rollout update {update} contains another update label")
        if document.get("is_padding") is not False:
            raise ValueError(
                f"rollout update {update} contains synthetic or unlabelled padding"
            )
        raw_record = document.get("step_record_json")
        if not isinstance(raw_record, str):
            raise ValueError(f"rollout update {update} omitted step_record_json")
        try:
            record = json.loads(raw_record)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"rollout update {update} has invalid step_record_json"
            ) from exc
        if not isinstance(record, Mapping):
            raise TypeError(f"rollout update {update} step record is not an object")
        route_id = record.get("route_id")
        if (
            not isinstance(route_id, str)
            or route_id not in route_ids
            or record.get("data_source") != route_id
        ):
            raise ValueError(f"rollout update {update} has an invalid route label")
        action = record.get("action")
        if not (
            isinstance(action, str)
            and action
            and _at(record, "action_submission.raw_policy_output") == action
            and document.get("output") == action
        ):
            raise ValueError(
                f"rollout update {update} action is not bound to policy output"
            )
        token_count = record.get("response_token_count")
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
        ):
            raise ValueError(f"rollout update {update} has invalid response tokens")
        uid = record.get("trajectory_uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"rollout update {update} omitted trajectory_uid")
        if uid in prior_trajectory_uids:
            raise ValueError(
                f"rollout update {update} repeats an earlier trajectory_uid"
            )
        item_id = record.get("item_id")
        data_idx = record.get("data_idx")
        if (
            not isinstance(item_id, str)
            or not item_id
            or not isinstance(data_idx, int)
            or isinstance(data_idx, bool)
            or data_idx < 0
            or schedule_routes.get((item_id, data_idx)) != route_id
        ):
            raise ValueError(
                f"rollout update {update} route differs from the frozen schedule"
            )
        episodes.setdefault(uid, []).append((record, document))
        action_rows[route_id] += 1
        response_tokens[route_id] += token_count

    episode_counts: Counter[str] = Counter()
    rewards: Counter[str] = Counter()
    successes: Counter[str] = Counter()
    event_counts: dict[str, Counter[str]] = {
        route_id: Counter() for route_id in route_ids
    }
    chains: Counter[str] = Counter()
    current_schedule_instances: set[tuple[str, int]] = set()
    for uid, episode in episodes.items():
        ordered = sorted(
            episode, key=lambda pair: pair[0].get("trajectory_row_order", -1)
        )
        orders = [record.get("trajectory_row_order") for record, _ in ordered]
        if orders != list(range(len(ordered))):
            raise ValueError(
                f"rollout update {update} trajectory {uid!r} has non-contiguous rows"
            )
        routes = {record.get("route_id") for record, _ in ordered}
        if len(routes) != 1:
            raise ValueError(
                f"rollout update {update} trajectory {uid!r} changed routes"
            )
        route_id = str(next(iter(routes)))
        instances = {
            (record.get("item_id"), record.get("data_idx"))
            for record, _document in ordered
        }
        if len(instances) != 1:
            raise ValueError(
                f"rollout update {update} trajectory {uid!r} changed schedule identity"
            )
        instance = next(iter(instances))
        if (
            instance in prior_schedule_instances
            or instance in current_schedule_instances
        ):
            raise ValueError(f"rollout update {update} repeats a schedule instance")
        current_schedule_instances.add(instance)
        terminals = [
            index
            for index, (record, _document) in enumerate(ordered)
            if record.get("trajectory_terminal") is True
        ]
        if (
            terminals != [len(ordered) - 1]
            or ordered[-1][0].get("rollout_done_flag") is not True
        ):
            raise ValueError(
                f"rollout update {update} trajectory {uid!r} is not complete"
            )
        if any(
            record.get("trajectory_terminal") is not False
            or record.get("rollout_done_flag") is not False
            for record, _document in ordered[:-1]
        ):
            raise ValueError(f"rollout update {update} trajectory {uid!r} ended early")
        terminal = ordered[-1][0]
        reward = terminal.get("trajectory_return")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError(
                f"rollout update {update} trajectory {uid!r} omitted reward"
            )
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError(
                f"rollout update {update} trajectory {uid!r} reward is not finite"
            )
        episode_counts[route_id] += 1
        rewards[route_id] += reward
        successes[route_id] += int(terminal.get("outcome") == "success")
        for record, _document in ordered:
            event_counts[route_id].update(_emitted_memory_events(record))
        chains[route_id] += int(_has_complete_memory_chain(ordered))

    if len(episodes) != samples_per_update:
        raise ValueError(
            f"rollout update {update} consumed {len(episodes)} real episodes; "
            f"expected {samples_per_update}"
        )
    prior_trajectory_uids.update(episodes)
    prior_schedule_instances.update(current_schedule_instances)
    return {
        "episodes": {route_id: episode_counts[route_id] for route_id in route_ids},
        "action_rows": {route_id: action_rows[route_id] for route_id in route_ids},
        "policy_response_tokens": {
            route_id: response_tokens[route_id] for route_id in route_ids
        },
        "rewards": {route_id: float(rewards[route_id]) for route_id in route_ids},
        "successes": {route_id: successes[route_id] for route_id in route_ids},
        "memory_events": {
            route_id: {
                event: event_counts[route_id][event]
                for event in sorted(_emitted_memory_events_for_summary())
            }
            for route_id in route_ids
        },
        "memory_chains": {route_id: chains[route_id] for route_id in route_ids},
    }


def _emitted_memory_events_for_summary() -> tuple[str, ...]:
    return ("write", "compaction", "read", "reuse", "modify", "execute")


def _verify_file_logger_prefix(
    path: Path,
    *,
    through_update: int,
    route_ids: Sequence[str],
    updates: Sequence[Mapping[str, Any]],
) -> None:
    rows = _complete_jsonl_rows(path, "FileLogger JSONL")
    by_step: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        step = row.get("step")
        data = row.get("data")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or not isinstance(data, Mapping)
        ):
            raise ValueError("FileLogger row has no integer step/data mapping")
        if 1 <= step <= through_update:
            by_step.setdefault(step, []).append(data)
    if set(by_step) != set(range(1, through_update + 1)):
        raise ValueError("FileLogger prefix does not cover every requested update")

    cumulative = {
        "episodes": Counter(),
        "action_rows": Counter(),
        "policy_response_tokens": Counter(),
    }
    for update, expected in enumerate(updates, start=1):
        items = [
            (str(key), value) for data in by_step[update] for key, value in data.items()
        ]
        for measure in cumulative:
            route_values = _metric_routes(
                items,
                f"fully_async/sum/optimizer_consumed_{measure}/data_source/",
                route_ids,
                label=f"FileLogger update {update} {measure}",
            )
            if route_values != expected[measure]:
                raise ValueError(
                    f"FileLogger update {update} {measure} route totals mismatch"
                )
            if _metric_value(
                items,
                f"fully_async/sum/optimizer_consumed_{measure}",
                label=f"FileLogger update {update}",
            ) != sum(route_values.values()):
                raise ValueError(
                    f"FileLogger update {update} {measure} global total mismatch"
                )
            cumulative[measure].update(route_values)
            observed_cumulative = _metric_routes(
                items,
                f"fully_async/count/optimizer_consumed_{measure}/data_source/",
                route_ids,
                label=f"FileLogger update {update} cumulative {measure}",
            )
            expected_cumulative = {
                route_id: cumulative[measure][route_id] for route_id in route_ids
            }
            if observed_cumulative != expected_cumulative:
                raise ValueError(
                    f"FileLogger update {update} cumulative {measure} route totals mismatch"
                )
            if _metric_value(
                items,
                f"fully_async/count/optimizer_consumed_{measure}",
                label=f"FileLogger update {update}",
            ) != sum(expected_cumulative.values()):
                raise ValueError(
                    f"FileLogger update {update} cumulative {measure} global total mismatch"
                )


def _observe_run(
    directory: Path, update: int, launch: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit one prefix against one immutable in-memory receipt view."""

    schema = launch.get("schema")
    if schema != _MULTITASK_RECEIPT_SCHEMA:
        if schema == _LEGACY_RECEIPT_SCHEMA:
            raise ValueError("online composition snapshots require a multitask receipt")
        raise ValueError(f"unsupported launch receipt schema: {schema!r}")
    launch_audit = _Audit(directory, trainer_exit_code=0, require_trainer_log=False)
    launch_audit.audit_launch(launch)
    launch_audit.audit_config()
    if launch_audit.errors:
        raise ValueError(
            "launch identity/config audit failed: " + "; ".join(launch_audit.errors)
        )
    if launch_audit.launch is None or launch_audit.expected is None:
        raise ValueError("launch identity/config audit produced no bound receipt")
    launch = launch_audit.launch
    routes = launch_audit.route_ids
    budget = launch_audit.expected
    horizon = _nonnegative_integral(budget.get("optimizer_updates"))
    samples_per_update = _nonnegative_integral(budget.get("samples_per_update"))
    if horizon is None or horizon < update or not samples_per_update:
        raise ValueError("requested snapshot is outside the receipt optimizer budget")
    runtime_paths = launch_audit.runtime_paths
    rollout_dir = runtime_paths["rollout_data"]
    rollout_paths = [rollout_dir / f"{step}.jsonl" for step in range(1, update + 1)]
    if any(not path.is_file() or path.is_symlink() for path in rollout_paths):
        raise ValueError("rollout prefix does not cover every requested update")

    updates = []
    seen_trajectory_uids: set[str] = set()
    seen_schedule_instances: set[tuple[str, int]] = set()
    for step, path in enumerate(rollout_paths, start=1):
        updates.append(
            _episode_summary(
                _complete_jsonl_rows(path, f"rollout update {step}"),
                update=step,
                route_ids=routes,
                samples_per_update=samples_per_update,
                schedule_routes=launch_audit.schedule_routes,
                prior_trajectory_uids=seen_trajectory_uids,
                prior_schedule_instances=seen_schedule_instances,
            )
        )
    _verify_file_logger_prefix(
        runtime_paths["file_logger"],
        through_update=update,
        route_ids=routes,
        updates=updates,
    )

    prefix_routes: dict[str, Any] = {}
    for route_id in routes:
        episodes = sum(item["episodes"][route_id] for item in updates)
        reward = sum(item["rewards"][route_id] for item in updates)
        successes = sum(item["successes"][route_id] for item in updates)
        events = {
            event: sum(item["memory_events"][route_id][event] for item in updates)
            for event in _emitted_memory_events_for_summary()
        }
        prefix_routes[route_id] = {
            "optimizer_consumed_episodes": episodes,
            "optimizer_consumed_action_rows": sum(
                item["action_rows"][route_id] for item in updates
            ),
            "optimizer_consumed_policy_response_tokens": sum(
                item["policy_response_tokens"][route_id] for item in updates
            ),
            "reward_sum": reward,
            "reward_mean": reward / episodes if episodes else None,
            "native_successes": successes,
            "native_success_rate": successes / episodes if episodes else None,
            "document_writes": events["write"],
            "compactions": events["compaction"],
            "document_reads": events["read"],
            "memory_reuses_or_modifications": events["reuse"] + events["modify"],
            "executions": events["execute"],
            "complete_memory_chains": sum(
                item["memory_chains"][route_id] for item in updates
            ),
        }
    rolling = _rolling_episode_shares([item["episodes"] for item in updates], routes)
    descriptive = update < 8
    return {
        "schema": "amg_verl_fully_async_online_snapshot_v1",
        "status": "descriptive" if descriptive else rolling["status"],
        "snapshot_update": update,
        "launch_receipt_schema": schema,
        "source": {
            "outer_commit": _at(launch, "source.outer_commit"),
            "inner_commit": _at(launch, "source.agentgym_commit"),
            "verl_commit": _at(launch, "source.verl_commit"),
        },
        "schedule": {
            "sha256": _at(launch, "schedule.sha256"),
            "route_registry_sha256": _at(launch, "schedule.route_registry_sha256"),
        },
        "optimizer_budget": {
            "optimizer_updates": horizon,
            "samples_per_update": samples_per_update,
        },
        "routes": prefix_routes,
        "latest_update": updates[-1],
        "rolling_8_episode_share": rolling,
        "errors": [],
    }


def observe_run(run_dir: str | os.PathLike[str], update: int) -> dict[str, Any]:
    """Read one completed optimizer prefix without changing any runtime owner."""

    _validate_snapshot_update(update)
    directory = Path(run_dir).resolve()
    launch = _load_json(directory / "launch-receipt.json", "launch receipt")
    return _observe_run(directory, update, launch)


def write_snapshot(
    run_dir: str | os.PathLike[str],
    update: int,
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Atomically publish one snapshot; all runtime artifacts remain untouched."""

    _validate_snapshot_update(update)
    directory = Path(run_dir).resolve()
    output = _path_within(directory, str(Path(output_path).resolve()))
    if output is None:
        raise ValueError("snapshot output must be an absolute path inside run_dir")
    launch = _load_json(directory / "launch-receipt.json", "launch receipt")
    protected_files, protected_directories = _receipt_protected_paths(
        launch, directory, include_finalization=True
    )
    if _path_overlaps_inputs(output, protected_files, protected_directories):
        raise ValueError("snapshot output overlaps a receipt-bound input artifact")
    try:
        snapshot = _observe_run(directory, update, launch)
    except Exception as exc:
        snapshot = {
            "schema": "amg_verl_fully_async_online_snapshot_v1",
            "status": "fail",
            "snapshot_update": update,
            "errors": [str(exc)],
        }
    current_launch = _load_json(directory / "launch-receipt.json", "launch receipt")
    if current_launch != launch:
        raise ValueError("launch receipt changed during observation")
    protected_files, protected_directories = _receipt_protected_paths(
        current_launch, directory, include_finalization=True
    )
    if _path_overlaps_inputs(output, protected_files, protected_directories):
        raise ValueError("snapshot output overlaps a receipt-bound input artifact")
    _atomic_json(output, snapshot)
    return snapshot


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write one live AMG evidence snapshot")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--update", type=int, choices=sorted(SNAPSHOT_UPDATES), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    snapshot = write_snapshot(args.run_dir, args.update, args.output)
    return 0 if snapshot.get("status") in {"pass", "descriptive"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SNAPSHOT_UPDATES", "observe_run", "write_snapshot"]
