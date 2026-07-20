#!/usr/bin/env python3
"""CPU-only smoke for bounded actor-to-vLLM sync evidence."""

import os
import tempfile
from pathlib import Path

from verl.workers.sharding_manager.vllm_sync_evidence import (
    append_and_readback_event,
    bounded_tensor_fingerprint,
    build_sync_event,
    read_last_event,
    validate_sync_event,
)


class FakeTensor:
    dtype = "fake.float32"

    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def detach(self):
        return self

    def numel(self):
        return len(self.values)

    def reshape(self, _):
        return self

    def __getitem__(self, positions):
        return FakeTensor([self.values[position] for position in positions])

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


def make_weights(delta=0.0):
    weights = []
    for tensor_index in range(80):
        values = [float(tensor_index * 4096 + value_index) for value_index in range(1200)]
        if tensor_index == 40:
            values[600] += delta
        weights.append((f"layer.{tensor_index:03d}.weight", FakeTensor(values)))
    return weights


def target_result(sync_id, source_fingerprint, target_fingerprint):
    return {
        "sync_id": sync_id,
        "model": "FakeVllmModel",
        "loaded_count": 80,
        "source_fingerprint_sha256": source_fingerprint["sha256"],
        "loaded_source_fingerprint_sha256": source_fingerprint["sha256"],
        "target_after": target_fingerprint,
    }


def main():
    source1 = bounded_tensor_fingerprint(make_weights())
    source1_repeat = bounded_tensor_fingerprint(make_weights())
    source2 = bounded_tensor_fingerprint(make_weights(delta=1.0))
    assert source1 == source1_repeat
    assert source1["sampled_tensor_count"] == 64
    assert source1["sampled_value_count"] == 64 * 1024
    assert source2["sha256"] != source1["sha256"]

    target1 = bounded_tensor_fingerprint(make_weights(delta=10.0))
    target2 = bounded_tensor_fingerprint(make_weights(delta=11.0))
    sync1 = "rank0:pid1:seq1:step1"
    event1 = build_sync_event(
        rank=0,
        pid=1,
        global_steps=1,
        sync_sequence=1,
        sync_id=sync1,
        source_before=source1,
        apply_model_results=[[target_result(sync1, source1, target1)]],
    )
    validate_sync_event(event1)

    sync2_same = "rank0:pid1:seq2:step2-same"
    event2_same = build_sync_event(
        rank=0,
        pid=1,
        global_steps=2,
        sync_sequence=2,
        sync_id=sync2_same,
        source_before=source1,
        apply_model_results=[target_result(sync2_same, source1, target1)],
        previous_event=event1,
    )
    try:
        validate_sync_event(event2_same, previous_event=event1, require_change=True)
    except RuntimeError as error:
        assert "source fingerprint did not change" in str(error)
    else:
        raise AssertionError("unchanged post-update weights must fail closed")

    sync2_target_same = "rank0:pid1:seq2:step2-target-same"
    event2_target_same = build_sync_event(
        rank=0,
        pid=1,
        global_steps=2,
        sync_sequence=2,
        sync_id=sync2_target_same,
        source_before=source2,
        apply_model_results=[target_result(sync2_target_same, source2, target1)],
        previous_event=event1,
    )
    try:
        validate_sync_event(event2_target_same, previous_event=event1, require_change=True)
    except RuntimeError as error:
        assert "target fingerprint did not change" in str(error)
    else:
        raise AssertionError("unchanged vLLM target weights must fail closed")

    sync2 = "rank0:pid1:seq2:step2"
    event2 = build_sync_event(
        rank=0,
        pid=1,
        global_steps=2,
        sync_sequence=2,
        sync_id=sync2,
        source_before=source2,
        apply_model_results=[target_result(sync2, source2, target2)],
        previous_event=event1,
    )
    validate_sync_event(event2, previous_event=event1, require_change=True)
    assert event2["source_changed_from_previous"] is True
    assert event2["target_changed_from_previous"] is True
    assert event2["previous_sync_id"] == sync1

    with tempfile.TemporaryDirectory() as temp_dir:
        rank_path = os.path.join(temp_dir, "vllm_sync_rank0.jsonl")
        append_and_readback_event(rank_path, event1)
        append_and_readback_event(rank_path, event2)
        assert read_last_event(rank_path)["sync_id"] == sync2
        assert len(Path(rank_path).read_text(encoding="utf-8").splitlines()) == 2
        assert not Path(temp_dir, "vllm_sync_rank1.jsonl").exists()

    repo_root = Path(__file__).resolve().parents[2]
    worker_source = (repo_root / "verl/workers/agent_fsdp_workers.py").read_text(encoding="utf-8")
    set_index = worker_source.index("set_sync_context(prompts.meta_info)")
    enter_index = worker_source.index("with self.rollout_sharding_manager:", set_index)
    validate_index = worker_source.index("validate_sync_before_generation()", enter_index)
    generate_index = worker_source.index("self.rollout.generate_sequences(prompts=prompts)", validate_index)
    assert set_index < enter_index < validate_index < generate_index
    print("vllm sync evidence smoke: PASS")


if __name__ == "__main__":
    main()
