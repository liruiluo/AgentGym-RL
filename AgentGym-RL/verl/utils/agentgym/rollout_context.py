from __future__ import annotations

from typing import Any

import numpy as np
import torch


def align_batch_to_rollout(batch: Any, rollout_output: Any, repeat_times: int) -> Any:
    if rollout_output.non_tensor_batch and "rollout_parent_indices" in rollout_output.non_tensor_batch:
        parent_indices = rollout_output.non_tensor_batch["rollout_parent_indices"].astype(np.int64)
        assert parent_indices.ndim == 1, f"rollout_parent_indices must be 1-D, got {parent_indices.shape}"
        assert len(parent_indices) == len(rollout_output), (
            f"rollout_parent_indices length {len(parent_indices)} does not match "
            f"rollout batch size {len(rollout_output)}."
        )
        assert len(parent_indices) > 0, "rollout_parent_indices must not be empty."
        assert parent_indices.min() >= 0 and parent_indices.max() < len(batch), (
            f"rollout_parent_indices out of range for source batch size {len(batch)}: {parent_indices.tolist()}"
        )
        torch_indices = torch.as_tensor(parent_indices, dtype=torch.long)
        aligned_non_tensor = {key: value[parent_indices] for key, value in batch.non_tensor_batch.items()}
        return batch.__class__(batch=batch.batch[torch_indices], non_tensor_batch=aligned_non_tensor, meta_info=batch.meta_info)
    return batch.repeat(repeat_times=repeat_times, interleave=True)
