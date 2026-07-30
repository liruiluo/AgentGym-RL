import json
from types import SimpleNamespace

import pytest

from verl.workers.rollout.agent_vllm_rollout.rollout_timing import (
    SCHEMA_VERSION,
    analyze_rollout_timing_documents,
    request_output_timing_record,
    write_rollout_timing_sidecar,
)


def _request_output(*, request_id="7", corrupted=False):
    metrics = SimpleNamespace(
        num_generation_tokens=12,
        arrival_time=10.0,
        queued_ts=10.1,
        scheduled_ts=10.2,
        first_token_ts=10.5,
        last_token_ts=12.0,
        first_token_latency=0.5,
        is_corrupted=corrupted,
    )
    return SimpleNamespace(request_id=request_id, metrics=metrics)


def test_request_output_timing_record_derives_durations():
    record = request_output_timing_record(_request_output())

    assert record["available"] is True
    assert record["request_id"] == "7"
    assert record["queue_seconds"] == pytest.approx(0.2)
    assert record["time_to_first_token_seconds"] == pytest.approx(0.5)
    assert record["decode_seconds"] == pytest.approx(1.5)
    assert record["request_seconds"] == pytest.approx(2.0)


def test_request_output_timing_record_rejects_corrupt_or_missing_metrics():
    assert request_output_timing_record(SimpleNamespace(request_id="x")) == {
        "available": False,
        "request_id": "x",
        "reason": "missing_metrics",
    }
    record = request_output_timing_record(_request_output(corrupted=True))
    assert record["available"] is False
    assert record["is_corrupted"] is True


def test_write_rollout_timing_sidecar_is_atomic_and_versioned(tmp_path):
    target = write_rollout_timing_sidecar(
        tmp_path,
        global_step=3,
        rank=2,
        payload={"rounds": [], "rollout_rounds_wall_seconds": 1.0},
    )

    assert target == tmp_path / "step3" / "2.json"
    document = json.loads(target.read_text())
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["global_step"] == 3
    assert document["rank"] == 2
    assert list((tmp_path / "step3").glob(".*.tmp-*")) == []


def test_analyze_rollout_timing_documents_separates_known_savings():
    document = {
        "schema_version": SCHEMA_VERSION,
        "global_step": 2,
        "rank": 0,
        "rollout_rounds_wall_seconds": 16.0,
        "rounds": [
            {
                "generation_wall_seconds": 8.0,
                "sleep_wall_seconds": 1.0,
                "environment_steps": [
                    {"rollout_index": 0, "wall_seconds": 0.5},
                    {"rollout_index": 1, "wall_seconds": 0.25},
                ],
                "requests": [
                    {
                        "rollout_index": 0,
                        "vllm_timing": {"available": True, "request_seconds": 8.0},
                    },
                    {
                        "rollout_index": 1,
                        "vllm_timing": {"available": True, "request_seconds": 1.0},
                    },
                ],
            },
            {
                "generation_wall_seconds": 6.0,
                "sleep_wall_seconds": 1.0,
                "environment_steps": [
                    {"rollout_index": 1, "wall_seconds": 0.25},
                ],
                "requests": [
                    {
                        "rollout_index": 1,
                        "vllm_timing": {"available": True, "request_seconds": 6.0},
                    }
                ],
            },
        ],
    }

    summary = analyze_rollout_timing_documents([document])
    rank = summary["rank_summaries"][0]
    assert rank["observed_synchronous_core_seconds"] == pytest.approx(16.0)
    assert rank["no_sleep_core_seconds"] == pytest.approx(14.0)
    assert rank["environment_parallel_wall_seconds"] == pytest.approx(0.75)
    assert rank["optimistic_dependency_bound_seconds"] == pytest.approx(8.5)
    assert summary["global"]["optimistic_dependency_speedup"] == pytest.approx(
        16.0 / 8.5
    )
