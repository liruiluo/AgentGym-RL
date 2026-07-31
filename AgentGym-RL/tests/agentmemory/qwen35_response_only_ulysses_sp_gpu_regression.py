#!/usr/bin/env python3
"""Two-GPU equivalence gate for response-only logits under Ulysses SP.

Run with:
    torchrun --nproc_per_node=2 \
        tests/agentmemory/qwen35_response_only_ulysses_sp_gpu_regression.py
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl.utils.ulysses import set_ulysses_sequence_parallel_group
from verl.workers.response_only_logits import (
    build_response_projection_plan,
    merge_sequence_parallel_response_outputs,
    scatter_response_outputs,
    shard_response_projection_plan,
)


def _fixture(device: torch.device):
    input_ids = torch.tensor(
        [
            [0, 0, 10, 11, 12, 21, 22, 0],
            [30, 31, 32, 33, 34, 41, 42, 43],
        ],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
        device=device,
    )
    responses = torch.tensor(
        [[21, 22, 0], [41, 42, 43]],
        dtype=torch.long,
        device=device,
    )
    response_mask = torch.tensor(
        [[1, 1, 0], [1, 1, 1]],
        dtype=torch.long,
        device=device,
    )
    unpadded_indices = attention_mask.flatten().nonzero().flatten()
    return input_ids, attention_mask, responses, response_mask, unpadded_indices


def _logprob_entropy(logits: torch.Tensor, labels: torch.Tensor):
    log_probs = logits.log_softmax(dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    probabilities = logits.softmax(dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - (
        probabilities * logits
    ).sum(dim=-1)
    return log_probs, entropy


def _loss(
    log_probs: torch.Tensor,
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    advantages = torch.tensor(
        [[0.5, -0.25, 0.0], [1.0, -0.75, 0.125]],
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    mask = response_mask.to(dtype=log_probs.dtype)
    return -(
        (log_probs * advantages + 0.01 * entropy) * mask
    ).sum() / mask.sum()


def _max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def main() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("this regression must run with exactly two ranks")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    set_ulysses_sequence_parallel_group(dist.group.WORLD)

    input_ids, attention_mask, responses, response_mask, indices = _fixture(device)
    plan = build_response_projection_plan(
        unpadded_indices=indices,
        input_ids=input_ids,
        attention_mask=attention_mask,
        responses=responses,
        response_mask=response_mask,
    )
    padding_size = (-plan.packed_token_count) % world_size
    local_plan = shard_response_projection_plan(
        plan,
        sequence_parallel_size=world_size,
        sequence_parallel_rank=rank,
        padding_size=padding_size,
    )

    torch.manual_seed(20260801)
    hidden_initial = torch.randn(
        plan.packed_token_count,
        8,
        dtype=torch.float64,
        device=device,
    )
    weight_initial = torch.randn(64, 8, dtype=torch.float64, device=device)

    reference_hidden = hidden_initial.detach().clone().requires_grad_(True)
    reference_weight = weight_initial.detach().clone().requires_grad_(True)
    reference_logits = (
        reference_hidden[plan.packed_predecessor_positions]
        @ reference_weight.t()
    ) / 0.9
    reference_selected_log_probs, reference_selected_entropy = _logprob_entropy(
        reference_logits,
        plan.labels,
    )
    reference_log_probs = scatter_response_outputs(
        reference_selected_log_probs,
        plan.response_mask,
    )
    reference_entropy = scatter_response_outputs(
        reference_selected_entropy,
        plan.response_mask,
    )
    reference_loss = _loss(
        reference_log_probs,
        reference_entropy,
        response_mask,
    )
    reference_loss.backward()

    padded_hidden = F.pad(hidden_initial, (0, 0, 0, padding_size))
    shard_size = padded_hidden.shape[0] // world_size
    shard_start = rank * shard_size
    local_hidden = (
        padded_hidden[shard_start : shard_start + shard_size]
        .detach()
        .clone()
        .requires_grad_(True)
    )
    sequence_parallel_weight = (
        weight_initial.detach().clone().requires_grad_(True)
    )
    local_logits = (
        local_hidden[local_plan.local_predecessor_positions]
        @ sequence_parallel_weight.t()
    ) / 0.9
    local_selected_log_probs, local_selected_entropy = _logprob_entropy(
        local_logits,
        local_plan.labels,
    )
    local_log_probs = scatter_response_outputs(
        local_selected_log_probs,
        local_plan.response_mask,
    )
    local_entropy = scatter_response_outputs(
        local_selected_entropy,
        local_plan.response_mask,
    )
    sequence_parallel_log_probs = merge_sequence_parallel_response_outputs(
        local_log_probs
    )
    sequence_parallel_entropy = merge_sequence_parallel_response_outputs(
        local_entropy
    )
    sequence_parallel_loss = _loss(
        sequence_parallel_log_probs,
        sequence_parallel_entropy,
        response_mask,
    )
    sequence_parallel_loss.backward()

    averaged_weight_gradient = sequence_parallel_weight.grad.detach().clone()
    dist.all_reduce(averaged_weight_gradient, op=dist.ReduceOp.SUM)
    averaged_weight_gradient.div_(world_size)

    normalized_local_hidden_gradient = local_hidden.grad.detach() / world_size
    gathered_hidden_gradients = [
        torch.empty_like(normalized_local_hidden_gradient)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered_hidden_gradients, normalized_local_hidden_gradient)
    gathered_hidden_gradient = torch.cat(gathered_hidden_gradients, dim=0)[
        : plan.packed_token_count
    ]

    merge_probe = torch.tensor(
        [float(rank + 1)],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    merge_sequence_parallel_response_outputs(merge_probe).sum().backward()

    deltas = {
        "logprob_max_abs_delta": _max_abs_delta(
            reference_log_probs,
            sequence_parallel_log_probs,
        ),
        "entropy_max_abs_delta": _max_abs_delta(
            reference_entropy,
            sequence_parallel_entropy,
        ),
        "loss_abs_delta": _max_abs_delta(
            reference_loss.reshape(1),
            sequence_parallel_loss.reshape(1),
        ),
        "averaged_weight_gradient_max_abs_delta": _max_abs_delta(
            reference_weight.grad,
            averaged_weight_gradient,
        ),
        "normalized_hidden_gradient_max_abs_delta": _max_abs_delta(
            reference_hidden.grad,
            gathered_hidden_gradient,
        ),
        "merge_backward_scale_abs_delta": abs(
            float(merge_probe.grad.item()) - world_size
        ),
    }
    tolerance = 1e-10
    failures = [
        f"{name}={value} exceeds {tolerance}"
        for name, value in deltas.items()
        if value > tolerance
    ]
    if local_plan.global_selected_response_tokens != int(response_mask.sum().item()):
        failures.append("global selected response-token count is inconsistent")

    if rank == 0:
        result = {
            "status": "fail" if failures else "pass",
            "world_size": world_size,
            "packed_tokens": plan.packed_token_count,
            "padding_size": padding_size,
            "response_tokens": int(response_mask.sum().item()),
            **deltas,
            "failures": failures,
        }
        print(json.dumps(result, sort_keys=True))
    dist.barrier()
    set_ulysses_sequence_parallel_group(None)
    dist.destroy_process_group()
    if failures:
        raise AssertionError("; ".join(failures))


if __name__ == "__main__":
    main()
