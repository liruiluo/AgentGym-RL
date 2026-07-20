"""Run with torchrun to verify PPO token normalization across two ranks."""

import os

import torch
import torch.distributed as dist

from verl.agent_trainer.ppo import core_algos
from verl.workers.ppo_token_normalization import (
    TokenWeightedMetricAccumulator,
    distributed_sum,
    mask_padding_rows,
    reduce_worker_metrics,
    scale_token_mean_loss,
    select_response_values,
    valid_response_token_count,
)


def _global_fixture():
    features = torch.tensor(
        [[0.1, -0.2, 0.3], [0.2, -0.1, 0.4], [-0.3, 0.2, 0.1], [0.1, -0.2, 0.3]]
    )
    advantages = torch.tensor(
        [[1.0, 0.5, -0.5], [-1.0, 0.2, 0.7], [0.3, -0.4, 1.0], [1.0, 0.5, -0.5]]
    )
    response_mask = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    )
    valid_samples = torch.tensor([True, True, False, True])
    return features, advantages, response_mask, valid_samples


def _assert_policy_gradient(rank, micro_splits):
    features, advantages, response_mask, valid_samples = _global_fixture()
    real_mask = mask_padding_rows(response_mask, valid_samples)

    reference_parameter = torch.tensor(0.25, requires_grad=True)
    reference_loss, _, _ = core_algos.compute_policy_loss(
        torch.zeros_like(features[[0, 1, 3]]),
        reference_parameter * features[[0, 1, 3]],
        advantages[[0, 1, 3]],
        response_mask[[0, 1, 3]],
        cliprange=0.2,
    )
    reference_loss.backward()

    local_indices = [0, 1] if rank == 0 else [2, 3]
    local_mask = real_mask[local_indices]
    global_tokens = distributed_sum(valid_response_token_count(local_mask))
    parameter = torch.tensor(0.25, requires_grad=True)
    for split in micro_splits:
        indices = [local_indices[index] for index in split]
        local_loss, _, _ = core_algos.compute_policy_loss(
            torch.zeros_like(features[indices]),
            parameter * features[indices],
            advantages[indices],
            real_mask[indices],
            cliprange=0.2,
        )
        loss = scale_token_mean_loss(
            local_loss,
            valid_response_token_count(real_mask[indices]),
            global_tokens,
        )
        loss.backward()

    dist.all_reduce(parameter.grad)
    parameter.grad /= dist.get_world_size()
    torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    if rank == 1:
        assert torch.isfinite(parameter.grad)


def _assert_value_gradient(rank, micro_splits):
    features, _, response_mask, valid_samples = _global_fixture()
    real_mask = mask_padding_rows(response_mask, valid_samples)
    returns = torch.tensor(
        [[0.5, 0.2, -0.1], [0.1, 0.0, 0.0], [-0.3, 0.4, 0.0], [0.5, 0.2, -0.1]]
    )

    reference_parameter = torch.tensor(0.25, requires_grad=True)
    reference_loss, _ = core_algos.compute_value_loss(
        reference_parameter * features[[0, 1, 3]],
        returns[[0, 1, 3]],
        torch.zeros_like(features[[0, 1, 3]]),
        response_mask[[0, 1, 3]],
        cliprange_value=10.0,
    )
    reference_loss.backward()

    local_indices = [0, 1] if rank == 0 else [2, 3]
    local_mask = real_mask[local_indices]
    global_tokens = distributed_sum(valid_response_token_count(local_mask))
    parameter = torch.tensor(0.25, requires_grad=True)
    for split in micro_splits:
        indices = [local_indices[index] for index in split]
        local_loss, _ = core_algos.compute_value_loss(
            parameter * features[indices],
            returns[indices],
            torch.zeros_like(features[indices]),
            real_mask[indices],
            cliprange_value=10.0,
        )
        loss = scale_token_mean_loss(
            local_loss,
            valid_response_token_count(real_mask[indices]),
            global_tokens,
        )
        loss.backward()

    dist.all_reduce(parameter.grad)
    parameter.grad /= dist.get_world_size()
    torch.testing.assert_close(parameter.grad, reference_parameter.grad)


def _assert_metrics(rank):
    token_count = torch.tensor(4.0 if rank == 0 else 2.0)
    metric_value = 1.0 if rank == 0 else 9.0
    accumulator = TokenWeightedMetricAccumulator()
    accumulator.add({"loss": metric_value}, token_count)
    reduced = accumulator.reduce()
    assert abs(reduced["loss"][0] - (22.0 / 6.0)) < 1e-8

    step_metrics = reduce_worker_metrics({"grad_norm": [metric_value]})
    assert abs(step_metrics["grad_norm"][0] - 5.0) < 1e-8


def _assert_all_padding_rank(rank):
    local_features = torch.tensor([[0.2, -0.1]])
    local_mask = torch.tensor([[1.0, 1.0]]) if rank == 0 else torch.zeros(1, 2)
    global_tokens = distributed_sum(valid_response_token_count(local_mask))
    parameter = torch.tensor(0.5, requires_grad=True)
    local_loss = (parameter * local_features).sum() / local_mask.sum().clamp_min(1.0)
    loss = scale_token_mean_loss(
        local_loss,
        valid_response_token_count(local_mask),
        global_tokens,
    )
    loss.backward()
    local_grad = parameter.grad.detach().clone()
    if rank == 1:
        torch.testing.assert_close(local_grad, torch.zeros_like(local_grad))
    dist.all_reduce(parameter.grad)
    parameter.grad /= dist.get_world_size()
    torch.testing.assert_close(parameter.grad, torch.tensor(0.05))


def _assert_response_value_alignment():
    full_sequence_values = torch.arange(18, dtype=torch.float32).reshape(2, 9)
    response_mask = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
    )
    aligned = select_response_values(full_sequence_values, response_mask)
    torch.testing.assert_close(
        aligned,
        torch.tensor([[6.0, 7.0, 0.0], [15.0, 0.0, 0.0]]),
    )

    try:
        select_response_values(torch.zeros(2, 2), response_mask)
    except ValueError as exc:
        assert "shorter than response_mask" in str(exc)
    else:
        raise AssertionError("short critic sequences must fail closed")


def main():
    dist.init_process_group("gloo")
    rank = int(os.environ["RANK"])
    for micro_splits in ([[0, 1]], [[0], [1]]):
        _assert_policy_gradient(rank, micro_splits)
        _assert_value_gradient(rank, micro_splits)
    _assert_metrics(rank)
    _assert_all_padding_rank(rank)
    _assert_response_value_alignment()
    dist.barrier()
    if rank == 0:
        print("AGENTMEMORY_DISTRIBUTED_TOKEN_NORMALIZATION_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
