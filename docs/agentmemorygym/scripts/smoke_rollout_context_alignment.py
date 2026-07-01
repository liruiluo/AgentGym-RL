from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class FakeBatch(dict):
    def __getitem__(self, item):
        if torch.is_tensor(item):
            return FakeBatch({key: value[item] for key, value in self.items()})
        return super().__getitem__(item)


class FakeProto:
    def __init__(self, batch, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __len__(self):
        first = next(iter(self.batch.values()))
        return first.shape[0]

    def repeat(self, repeat_times=2, interleave=True):
        assert interleave
        return FakeProto(
            batch=FakeBatch({key: value.repeat_interleave(repeat_times, dim=0) for key, value in self.batch.items()}),
            non_tensor_batch={key: np.repeat(value, repeat_times, axis=0) for key, value in self.non_tensor_batch.items()},
            meta_info=self.meta_info,
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "AgentGym-RL"))

    from verl.utils.agentgym.context_policy import assert_rollout_context_supported, rollout_context_policy
    from verl.utils.agentgym.rollout_context import align_batch_to_rollout

    assert rollout_context_policy({"task_name": "agentmemory"}) == "latest_observation_only"
    assert_rollout_context_supported({"task_name": "agentmemory"})

    source = FakeProto(
        batch=FakeBatch(
            {
                "input_ids": torch.tensor([[10, 11], [20, 21]], dtype=torch.long),
                "attention_mask": torch.ones((2, 2), dtype=torch.long),
            }
        ),
        non_tensor_batch={
            "item_id": np.array(["agentmemory_0", "agentmemory_1"], dtype=object),
            "uid": np.array(["uid-a", "uid-b"], dtype=object),
        },
    )
    rollout = FakeProto(
        batch=FakeBatch(
            {
                "responses": torch.tensor([[1], [2], [3]], dtype=torch.long),
                "response_mask": torch.ones((3, 1), dtype=torch.long),
            }
        ),
        non_tensor_batch={"rollout_parent_indices": np.array([0, 0, 1], dtype=np.int64)},
    )
    aligned = align_batch_to_rollout(source, rollout, repeat_times=1)
    assert len(aligned) == 3, len(aligned)
    assert aligned.batch["input_ids"].tolist() == [[10, 11], [10, 11], [20, 21]]
    assert aligned.non_tensor_batch["item_id"].tolist() == ["agentmemory_0", "agentmemory_0", "agentmemory_1"]
    assert aligned.non_tensor_batch["uid"].tolist() == ["uid-a", "uid-a", "uid-b"]

    rollout_n2 = FakeProto(
        batch=FakeBatch({"responses": torch.ones((4, 1), dtype=torch.long)}),
        non_tensor_batch={"rollout_parent_indices": np.array([0, 0, 0, 1], dtype=np.int64)},
    )
    aligned_n2 = align_batch_to_rollout(source, rollout_n2, repeat_times=2)
    assert aligned_n2.non_tensor_batch["item_id"].tolist() == [
        "agentmemory_0",
        "agentmemory_0",
        "agentmemory_0",
        "agentmemory_1",
    ]

    repeated = align_batch_to_rollout(
        source,
        FakeProto(batch=FakeBatch({key: value.clone() for key, value in source.batch.items()})),
        repeat_times=2,
    )
    assert len(repeated) == 4, len(repeated)
    assert repeated.batch["input_ids"].tolist() == [[10, 11], [10, 11], [20, 21], [20, 21]]
    print("AGENTMEMORY_ROLLOUT_CONTEXT_ALIGNMENT_SMOKE_OK")


if __name__ == "__main__":
    main()
