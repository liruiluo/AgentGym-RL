import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "agentmemory_rollout_timing_v2"
_REQUEST_TIMESTAMP_FIELDS = (
    "arrival_time",
    "queued_ts",
    "scheduled_ts",
    "first_token_ts",
    "last_token_ts",
)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def validate_request_metrics_config(
    *,
    timing_required: bool,
    official_vllm: bool,
    disable_log_stats: bool,
) -> None:
    if not timing_required:
        return
    if not official_vllm:
        raise RuntimeError(
            "Required AgentMemory rollout timing needs official vLLM request metrics."
        )
    if disable_log_stats:
        raise RuntimeError(
            "AGENTMEMORY_ROLLOUT_TIMING_REQUIRED=1 requires "
            "actor_rollout_ref.rollout.disable_log_stats=false."
        )


def request_output_timing_record(request_output: Any) -> dict[str, Any]:
    """Extract stable, JSON-safe timing fields from an official vLLM output."""

    metrics = getattr(request_output, "metrics", None)
    if metrics is None:
        return {
            "available": False,
            "request_id": str(getattr(request_output, "request_id", "")),
            "reason": "missing_metrics",
        }

    timestamps = {
        field: _finite_float(getattr(metrics, field, None))
        for field in _REQUEST_TIMESTAMP_FIELDS
    }
    scheduled = timestamps["scheduled_ts"]
    queued = timestamps["queued_ts"]
    first_token = timestamps["first_token_ts"]
    last_token = timestamps["last_token_ts"]
    first_token_latency = _finite_float(
        getattr(metrics, "first_token_latency", None)
    )
    corrupted = bool(getattr(metrics, "is_corrupted", False))
    # vLLM 0.24 records arrival_time with time.time(), while the engine-core
    # timestamps use time.monotonic(). Only compare or subtract timestamps
    # from the same engine-core clock.
    ordered = (
        queued is not None
        and scheduled is not None
        and first_token is not None
        and last_token is not None
        and 0.0 < queued <= scheduled <= first_token <= last_token
    )

    record: dict[str, Any] = {
        "available": bool(ordered and not corrupted),
        "request_id": str(getattr(request_output, "request_id", "")),
        "num_generation_tokens": int(
            getattr(metrics, "num_generation_tokens", 0)
        ),
        "is_corrupted": corrupted,
        "first_token_latency_seconds": first_token_latency,
        **timestamps,
    }
    if ordered:
        record.update(
            {
                "queue_seconds": scheduled - queued,
                "prefill_seconds": first_token - scheduled,
                "decode_seconds": last_token - first_token,
                "inference_seconds": last_token - scheduled,
                "engine_core_seconds": last_token - queued,
            }
        )
    else:
        record["reason"] = "missing_or_unordered_timestamps"
    return record


def write_rollout_timing_sidecar(
    root: str | os.PathLike[str],
    *,
    global_step: int,
    rank: int,
    payload: Mapping[str, Any],
) -> Path:
    step_dir = Path(root) / f"step{int(global_step)}"
    step_dir.mkdir(parents=True, exist_ok=True)
    target = step_dir / f"{int(rank)}.json"
    temporary = step_dir / f".{int(rank)}.json.tmp-{os.getpid()}"
    document = {
        "schema_version": SCHEMA_VERSION,
        "global_step": int(global_step),
        "rank": int(rank),
        **dict(payload),
    }
    temporary.write_text(
        json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, target)
    return target


def analyze_rollout_timing_documents(
    documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute conservative and optimistic bounds from synchronous traces."""

    if not documents:
        raise ValueError("no rollout timing documents")

    rank_summaries = []
    for document in documents:
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported rollout timing schema: {document.get('schema_version')!r}"
            )
        per_trajectory: dict[int, float] = {}
        generation_wall = 0.0
        sleep_wall = 0.0
        environment_serial_wall = 0.0
        environment_parallel_wall = 0.0
        request_count = 0
        unavailable_request_count = 0
        for round_record in document.get("rounds", []):
            generation_wall += float(round_record["generation_wall_seconds"])
            sleep_wall += float(round_record["sleep_wall_seconds"])
            environment_steps = round_record.get("environment_steps", [])
            environment_durations = [
                float(step["wall_seconds"]) for step in environment_steps
            ]
            environment_serial_wall += sum(environment_durations)
            environment_parallel_wall += max(environment_durations, default=0.0)
            environment_by_index = {
                int(step["rollout_index"]): float(step["wall_seconds"])
                for step in environment_steps
            }
            for request in round_record.get("requests", []):
                request_count += 1
                rollout_index = int(request["rollout_index"])
                timing = request["vllm_timing"]
                if not timing.get("available"):
                    unavailable_request_count += 1
                    continue
                request_and_environment_seconds = float(
                    timing["engine_core_seconds"]
                ) + environment_by_index.get(rollout_index, 0.0)
                per_trajectory[rollout_index] = per_trajectory.get(
                    rollout_index, 0.0
                ) + request_and_environment_seconds

        if unavailable_request_count:
            dependency_bound = None
        else:
            dependency_bound = max(per_trajectory.values(), default=0.0)
        observed_core = generation_wall + sleep_wall + environment_serial_wall
        no_sleep_core = observed_core - sleep_wall
        no_sleep_parallel_env_core = (
            generation_wall + environment_parallel_wall
        )
        rank_summaries.append(
            {
                "rank": int(document["rank"]),
                "request_count": request_count,
                "unavailable_request_count": unavailable_request_count,
                "generation_wall_seconds": generation_wall,
                "sleep_wall_seconds": sleep_wall,
                "environment_serial_wall_seconds": environment_serial_wall,
                "environment_parallel_wall_seconds": environment_parallel_wall,
                "observed_synchronous_core_seconds": observed_core,
                "no_sleep_core_seconds": no_sleep_core,
                "no_sleep_parallel_env_core_seconds": no_sleep_parallel_env_core,
                "optimistic_dependency_bound_seconds": dependency_bound,
                "rollout_rounds_wall_seconds": float(
                    document["rollout_rounds_wall_seconds"]
                ),
            }
        )

    def global_max(field: str) -> float | None:
        values = [row[field] for row in rank_summaries]
        if any(value is None for value in values):
            return None
        return max(float(value) for value in values)

    actual = global_max("rollout_rounds_wall_seconds")
    optimistic = global_max("optimistic_dependency_bound_seconds")
    return {
        "schema_version": "agentmemory_rollout_critical_path_summary_v2",
        "rank_count": len(rank_summaries),
        "rank_summaries": rank_summaries,
        "global": {
            "rollout_rounds_wall_seconds": actual,
            "observed_synchronous_core_seconds": global_max(
                "observed_synchronous_core_seconds"
            ),
            "no_sleep_core_seconds": global_max("no_sleep_core_seconds"),
            "no_sleep_parallel_env_core_seconds": global_max(
                "no_sleep_parallel_env_core_seconds"
            ),
            "optimistic_dependency_bound_seconds": optimistic,
            "optimistic_dependency_speedup": (
                actual / optimistic
                if actual is not None and optimistic not in (None, 0.0)
                else None
            ),
        },
    }
