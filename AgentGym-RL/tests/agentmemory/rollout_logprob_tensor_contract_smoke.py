#!/usr/bin/env python3
"""Exercise rollout-logprob alignment through the real PPO batch transforms."""

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.agent_trainer.ppo.ray_trainer import _validate_packed_rollout_logprobs
from verl.protocol import pad_dataproto_to_divisor
from verl.utils.agentgym.rollout_context import align_batch_to_rollout
from verl.utils.agentgym.rollout_logprob_reuse import ROLLOUT_LOGPROB_BATCH_KEY


def main():
    source = DataProto(
        batch=TensorDict(
            {"source_marker": torch.tensor([[100], [200]], dtype=torch.long)},
            batch_size=2,
        ),
        non_tensor_batch={"source_name": np.array(["a", "b"], dtype=object)},
    )
    responses = torch.tensor(
        [[11, 12, 0], [21, 0, 0], [31, 32, 33]], dtype=torch.long
    )
    response_mask = torch.tensor(
        [[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.bool
    )
    rollout_log_probs = torch.tensor(
        [[-0.11, -0.12, 0.0], [-0.21, 0.0, 0.0], [-0.31, -0.32, -0.33]],
        dtype=torch.float32,
    )
    rollout = DataProto(
        batch=TensorDict(
            {
                "responses": responses,
                "response_mask": response_mask,
                ROLLOUT_LOGPROB_BATCH_KEY: rollout_log_probs,
                "attention_mask": response_mask.to(torch.long),
            },
            batch_size=3,
        ),
        non_tensor_batch={
            "rollout_parent_indices": np.array([1, 0, 1], dtype=object),
        },
    )

    batch = align_batch_to_rollout(source, rollout, repeat_times=1)
    batch = batch.union(rollout)
    batch.batch[core_algos.PPO_VALID_SAMPLE_MASK] = torch.ones(
        len(batch), dtype=torch.bool
    )
    batch, pad_size = pad_dataproto_to_divisor(batch, 4)
    assert pad_size == 1
    batch.batch[core_algos.PPO_VALID_SAMPLE_MASK][-pad_size:] = False
    batch.reorder(torch.tensor([2, 0, 3, 1], dtype=torch.long))

    packed, valid_tokens, valid_samples = _validate_packed_rollout_logprobs(batch)
    assert valid_samples.tolist() == [True, True, False, True]
    assert int(valid_tokens.sum().item()) == 6
    for row in range(len(batch)):
        source_marker = int(batch.batch["source_marker"][row, 0].item())
        expected_source = {11: 200, 21: 100, 31: 200}[
            int(batch.batch["responses"][row, 0].item())
        ]
        assert source_marker == expected_source
        for token_id, log_prob, is_valid in zip(
            batch.batch["responses"][row].tolist(),
            packed[row].tolist(),
            batch.batch["response_mask"][row].tolist(),
        ):
            expected = -float(token_id) / 100.0 if is_valid else 0.0
            assert abs(float(log_prob) - expected) < 1e-6

    print("rollout logprob tensor contract smoke: PASS")


if __name__ == "__main__":
    main()
