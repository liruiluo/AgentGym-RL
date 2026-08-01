#!/usr/bin/env python3
"""Qwen3.5 packed linear-attention regression under Ulysses SP.

Adapted from VERL commit 6a6242f3's distributed regression. Run with:
    torchrun --nproc_per_node=2 \
        tests/agentmemory/qwen35_linear_attention_ulysses_sp_gpu_regression.py
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
)

from verl.models.transformers.qwen3_5 import apply_qwen3_5_packed_forward_patch
from verl.utils.ulysses import set_ulysses_sequence_parallel_group


PROBE_HIDDEN_SIZE = 512
PROBE_HEAD_DIM = 128


def _error_ratio(reference: torch.Tensor, actual: torch.Tensor) -> float:
    error = (
        (reference.detach() - actual.detach())
        .flatten()
        .float()
        .square()
        .mean()
        .sqrt()
        .item()
    )
    scale = (
        reference.detach().flatten().float().square().mean().sqrt().item()
    )
    return error / (scale + 1e-8)


def _make_layer(device: torch.device) -> Qwen3_5DecoderLayer:
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=PROBE_HIDDEN_SIZE,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=PROBE_HEAD_DIM,
        linear_key_head_dim=PROBE_HEAD_DIM,
        linear_value_head_dim=PROBE_HEAD_DIM,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
        dtype=torch.bfloat16,
    )
    torch.manual_seed(1234)
    layer = Qwen3_5DecoderLayer(config, 0).to(
        device=device,
        dtype=torch.bfloat16,
    )
    layer.eval()
    return layer


def _broadcast_parameters(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)


def _all_gather_sequence(local_tensor: torch.Tensor) -> torch.Tensor:
    gathered = [
        torch.empty_like(local_tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered, local_tensor.contiguous())
    return torch.cat(gathered, dim=1)


def _cases_for_world_size(world_size: int) -> list[tuple[str, list[int]]]:
    if world_size == 2:
        return [
            ("boundary_aligned", [8, 8]),
            ("sequence_cut", [6, 10]),
            ("single_sequence", [16]),
            ("many_short_sequences", [3, 4, 5, 4]),
        ]
    if world_size == 4:
        return [
            ("boundary_aligned", [8, 8, 8, 8]),
            ("sequence_cut", [20, 12]),
            ("single_sequence", [32]),
            ("many_short_sequences", [5, 7, 9, 11]),
        ]
    if world_size == 8:
        return [
            ("boundary_aligned", [128] * 8),
            ("sequence_cut", [700, 324]),
            ("single_sequence", [1024]),
            ("many_short_sequences", [100, 150, 200, 250, 124, 100, 100]),
        ]
    raise RuntimeError("this regression supports world sizes 2, 4, and 8")


def _run_case(
    *,
    case_name: str,
    lengths: list[int],
    layer: Qwen3_5DecoderLayer,
    device: torch.device,
) -> dict[str, object]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    total_tokens = sum(lengths)
    if total_tokens % world_size != 0:
        raise RuntimeError(
            f"{case_name}: total tokens {total_tokens} must divide "
            f"world size {world_size}."
        )

    cu_seqlens = torch.tensor(
        [0] + torch.cumsum(torch.tensor(lengths), 0).tolist(),
        device=device,
        dtype=torch.long,
    )
    cu_seqlens_cpu = cu_seqlens.cpu()
    position_embeddings = (
        torch.empty(0, device=device),
        torch.empty(0, device=device),
    )
    torch.manual_seed(5678 + total_tokens + len(lengths))
    full_hidden = torch.randn(
        1,
        total_tokens,
        PROBE_HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )

    set_ulysses_sequence_parallel_group(None)
    reference_hidden = full_hidden.detach().clone().requires_grad_(True)
    reference_output = layer(
        reference_hidden,
        position_embeddings=position_embeddings,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    reference_output.sum().backward()
    reference_gradient = reference_hidden.grad.detach()
    layer.zero_grad(set_to_none=True)

    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    local_sequence_length = total_tokens // world_size
    start = rank * local_sequence_length
    local_hidden = (
        full_hidden[:, start : start + local_sequence_length]
        .detach()
        .clone()
        .requires_grad_(True)
    )
    local_output = layer(
        local_hidden,
        position_embeddings=position_embeddings,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    local_output.sum().backward()

    sequence_parallel_output = _all_gather_sequence(local_output.detach())
    sequence_parallel_gradient = _all_gather_sequence(
        local_hidden.grad.detach()
    )
    output_error_ratio = _error_ratio(
        reference_output.detach(),
        sequence_parallel_output,
    )
    gradient_error_ratio = _error_ratio(
        reference_gradient,
        sequence_parallel_gradient,
    )
    output_max_abs_delta = float(
        (sequence_parallel_output - reference_output.detach())
        .abs()
        .max()
        .item()
    )
    gradient_max_abs_delta = float(
        (sequence_parallel_gradient - reference_gradient).abs().max().item()
    )

    torch.testing.assert_close(
        sequence_parallel_output,
        reference_output.detach(),
        atol=2e-2,
        rtol=2e-2,
    )
    return {
        "case": case_name,
        "output_max_abs_delta": output_max_abs_delta,
        "gradient_max_abs_delta": gradient_max_abs_delta,
        "output_error_ratio": output_error_ratio,
        "gradient_error_ratio": gradient_error_ratio,
    }


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (2, 4, 8):
        raise RuntimeError("this regression requires 2, 4, or 8 ranks")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    apply_qwen3_5_packed_forward_patch()

    cases = _cases_for_world_size(world_size)
    results = []
    failures = []
    for use_causal_conv1d in (True, False):
        layer = _make_layer(device)
        if not use_causal_conv1d:
            for module in layer.modules():
                if isinstance(module, Qwen3_5GatedDeltaNet):
                    module.causal_conv1d_fn = None
        _broadcast_parameters(layer)
        backend = (
            "causal_conv1d_fn" if use_causal_conv1d else "torch_conv1d_fallback"
        )
        for case_name, lengths in cases:
            result = _run_case(
                case_name=case_name,
                lengths=lengths,
                layer=layer,
                device=device,
            )
            result["backend"] = backend
            results.append(result)
            if result["gradient_error_ratio"] >= 2e-3:
                failures.append(
                    f"{backend}/{case_name}: gradient error ratio "
                    f"{result['gradient_error_ratio']} >= 0.002"
                )
            layer.zero_grad(set_to_none=True)

    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "fail" if failures else "pass",
                    "world_size": dist.get_world_size(),
                    "upstream_verl_commit": "6a6242f3",
                    "cases": results,
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
