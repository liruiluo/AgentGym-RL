#!/usr/bin/env python3
"""Two-GPU Qwen3.5 full-attention equivalence gate for Ulysses SP2.

The packed fixture crosses the SP shard boundary inside its second sample, so
rank-one outputs must attend to tokens owned by rank zero. Run with:

    torchrun --nproc_per_node=2 \
        tests/agentmemory/qwen35_full_attention_ulysses_sp_gpu_regression.py
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

from verl.models.transformers.qwen3_5 import (
    apply_qwen3_5_packed_forward_patch,
    qwen3_5_ulysses_flash_attention_patch_installed,
)
from verl.utils.ulysses import set_ulysses_sequence_parallel_group


HIDDEN_SIZE = 512
HEAD_DIM = 128
SEQUENCE_LENGTHS = (6, 10)


def _make_layer(device: torch.device) -> Qwen3_5DecoderLayer:
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["full_attention"],
        attention_dropout=0.0,
        dtype=torch.bfloat16,
    )
    config._attn_implementation = "flash_attention_2"
    torch.manual_seed(20260801)
    layer = Qwen3_5DecoderLayer(config, 0).to(
        device=device,
        dtype=torch.bfloat16,
    )
    layer.eval()
    return layer


def _broadcast_parameters(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=0)


def _packed_position_ids(device: torch.device) -> torch.Tensor:
    return torch.cat(
        [torch.arange(length, device=device) for length in SEQUENCE_LENGTHS]
    ).unsqueeze(0)


def _position_embeddings(
    position_ids: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_frequency = 1.0 / (
        10000
        ** (
            torch.arange(0, HEAD_DIM, 2, device=position_ids.device).float()
            / HEAD_DIM
        )
    )
    frequencies = position_ids.float().unsqueeze(-1) * inverse_frequency
    embeddings = torch.cat((frequencies, frequencies), dim=-1)
    return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def _all_gather_sequence(local_tensor: torch.Tensor) -> torch.Tensor:
    gathered = [
        torch.empty_like(local_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered, local_tensor.contiguous())
    return torch.cat(gathered, dim=1)


def _error_ratio(reference: torch.Tensor, actual: torch.Tensor) -> float:
    error = (
        (reference.detach().float() - actual.detach().float())
        .flatten()
        .square()
        .mean()
        .sqrt()
        .item()
    )
    scale = (
        reference.detach().float().flatten().square().mean().sqrt().item()
    )
    return error / (scale + 1e-8)


def _parameter_gradients(
    layer: torch.nn.Module,
    *,
    all_reduce: bool,
) -> dict[str, torch.Tensor]:
    gradients = {}
    for name, parameter in layer.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"missing parameter gradient: {name}")
        gradient = parameter.grad.detach().float().clone()
        if all_reduce:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradients[name] = gradient
    return gradients


def _flatten_gradients(
    gradients: dict[str, torch.Tensor],
) -> torch.Tensor:
    return torch.cat([gradient.flatten() for gradient in gradients.values()])


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

    apply_qwen3_5_packed_forward_patch()
    if not qwen3_5_ulysses_flash_attention_patch_installed():
        raise RuntimeError("Qwen3.5 Ulysses FlashAttention patch is missing")
    layer = _make_layer(device)
    _broadcast_parameters(layer)

    total_tokens = sum(SEQUENCE_LENGTHS)
    local_tokens = total_tokens // world_size
    position_ids = _packed_position_ids(device)
    position_embeddings = _position_embeddings(position_ids, torch.bfloat16)
    torch.manual_seed(20260802)
    full_hidden = torch.randn(
        1,
        total_tokens,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(20260803)
    loss_weights = torch.randn_like(full_hidden)

    set_ulysses_sequence_parallel_group(None)
    reference_hidden = full_hidden.detach().clone().requires_grad_(True)
    reference_output = layer(
        reference_hidden,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        attention_mask=None,
    )

    perturbed_hidden = full_hidden.detach().clone()
    perturbed_hidden[:, SEQUENCE_LENGTHS[0], :].add_(0.5)
    with torch.no_grad():
        perturbed_output = layer(
            perturbed_hidden,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=None,
        )
    cross_shard_dependency = float(
        (
            perturbed_output[:, local_tokens:, :]
            - reference_output.detach()[:, local_tokens:, :]
        )
        .abs()
        .max()
        .item()
    )

    reference_loss = (
        reference_output.float() * loss_weights.float()
    ).sum()
    reference_loss.backward()
    reference_input_gradient = reference_hidden.grad.detach().clone()
    reference_parameter_gradients = _parameter_gradients(
        layer,
        all_reduce=False,
    )
    layer.zero_grad(set_to_none=True)

    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    start = rank * local_tokens
    end = start + local_tokens
    local_hidden = (
        full_hidden[:, start:end, :].detach().clone().requires_grad_(True)
    )
    local_output = layer(
        local_hidden,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        attention_mask=None,
    )
    local_loss = (
        local_output.float() * loss_weights[:, start:end, :].float()
    ).sum()
    local_loss.backward()

    sequence_parallel_output = _all_gather_sequence(local_output.detach())
    sequence_parallel_input_gradient = _all_gather_sequence(
        local_hidden.grad.detach()
    )
    sequence_parallel_parameter_gradients = _parameter_gradients(
        layer,
        all_reduce=True,
    )

    reference_flat_gradient = _flatten_gradients(
        reference_parameter_gradients
    )
    sequence_parallel_flat_gradient = _flatten_gradients(
        sequence_parallel_parameter_gradients
    )
    parameter_max_abs_delta = max(
        float(
            (
                sequence_parallel_parameter_gradients[name]
                - reference_parameter_gradients[name]
            )
            .abs()
            .max()
            .item()
        )
        for name in reference_parameter_gradients
    )
    metrics = {
        "output_error_ratio": _error_ratio(
            reference_output,
            sequence_parallel_output,
        ),
        "output_max_abs_delta": float(
            (sequence_parallel_output - reference_output.detach())
            .abs()
            .max()
            .item()
        ),
        "input_gradient_error_ratio": _error_ratio(
            reference_input_gradient,
            sequence_parallel_input_gradient,
        ),
        "input_gradient_max_abs_delta": float(
            (
                sequence_parallel_input_gradient
                - reference_input_gradient
            )
            .abs()
            .max()
            .item()
        ),
        "parameter_gradient_error_ratio": _error_ratio(
            reference_flat_gradient,
            sequence_parallel_flat_gradient,
        ),
        "parameter_gradient_max_abs_delta": parameter_max_abs_delta,
        "cross_shard_dependency_max_abs": cross_shard_dependency,
    }

    failures = []
    if metrics["cross_shard_dependency_max_abs"] <= 1e-4:
        failures.append("fixture has no measurable cross-shard dependency")
    for label in (
        "output_error_ratio",
        "input_gradient_error_ratio",
        "parameter_gradient_error_ratio",
    ):
        if metrics[label] >= 2e-2:
            failures.append(f"{label}={metrics[label]} exceeds 0.02")
    for label in (
        "output_max_abs_delta",
        "input_gradient_max_abs_delta",
        "parameter_gradient_max_abs_delta",
    ):
        if metrics[label] > 0.125:
            failures.append(f"{label}={metrics[label]} exceeds 0.125")

    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "fail" if failures else "pass",
                    "world_size": world_size,
                    "sequence_lengths": SEQUENCE_LENGTHS,
                    **metrics,
                    "failures": failures,
                },
                sort_keys=True,
            )
        )
    dist.barrier()
    set_ulysses_sequence_parallel_group(None)
    dist.destroy_process_group()
    if failures:
        raise AssertionError("; ".join(failures))


if __name__ == "__main__":
    main()
