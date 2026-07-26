#!/usr/bin/env python3
"""GPU regression for Qwen3.5 packed sequence boundaries.

The positive comparison runs each sample independently and then as one packed
sequence. The negative control deliberately omits packed boundaries so this
test also proves that the chosen inputs can detect cross-sample state leakage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from verl.models.transformers.qwen3_5 import (
    apply_qwen3_5_packed_forward_patch,
)


@dataclass
class ProbeResult:
    name: str
    output_error_ratio: float
    input_gradient_error_ratio: float
    negative_control_output_error_ratio: float
    negative_control_input_gradient_error_ratio: float
    reference_output_rms: float
    packed_output_rms: float
    reference_input_gradient_rms: float
    packed_input_gradient_rms: float
    input_gradient_error_rms: float


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


def _rms(tensor: torch.Tensor) -> float:
    return tensor.detach().flatten().float().square().mean().sqrt().item()


def _make_model(layer_types: list[str], device: torch.device):
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=layer_types,
        attention_dropout=0.0,
        dtype=torch.bfloat16,
    )
    config._attn_implementation = "flash_attention_2"
    torch.manual_seed(1234)
    model = Qwen3_5TextModel(config).to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def _run_model(
    model,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    output_cotangent: torch.Tensor,
    loss_denominator: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = inputs_embeds.detach().clone().requires_grad_(True)
    kwargs = {}
    if cu_seqlens is not None:
        kwargs = {
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_cpu": cu_seqlens.cpu(),
        }
    output = model(
        inputs_embeds=inputs,
        attention_mask=None,
        position_ids=position_ids,
        use_cache=False,
        **kwargs,
    ).last_hidden_state
    ((output.float() * output_cotangent.float()).sum() / loss_denominator).backward()
    gradient = inputs.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    return output.detach(), gradient


def _run_probe(
    name: str,
    layer_types: list[str],
    lengths: list[int],
    device: torch.device,
) -> ProbeResult:
    model = _make_model(layer_types, device)
    total_tokens = sum(lengths)
    torch.manual_seed(5678 + len(layer_types))
    packed_inputs = torch.randn(
        1,
        total_tokens,
        model.config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    boundaries = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(),
        device=device,
        dtype=torch.long,
    )
    packed_positions = torch.cat(
        [torch.arange(length, device=device) for length in lengths]
    ).unsqueeze(0)
    torch.manual_seed(9012 + len(layer_types))
    output_cotangent = torch.randn(
        1,
        total_tokens,
        model.config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )

    reference_outputs = []
    reference_gradients = []
    loss_denominator = total_tokens * model.config.hidden_size
    start = 0
    for length in lengths:
        end = start + length
        sample_output, sample_gradient = _run_model(
            model,
            packed_inputs[:, start:end],
            torch.arange(length, device=device).unsqueeze(0),
            torch.tensor([0, length], device=device, dtype=torch.long),
            output_cotangent[:, start:end],
            loss_denominator,
        )
        reference_outputs.append(sample_output)
        reference_gradients.append(sample_gradient)
        start = end

    reference_output = torch.cat(reference_outputs, dim=1)
    reference_gradient = torch.cat(reference_gradients, dim=1)
    packed_output, packed_gradient = _run_model(
        model,
        packed_inputs,
        packed_positions,
        boundaries,
        output_cotangent,
        loss_denominator,
    )
    negative_output, negative_gradient = _run_model(
        model,
        packed_inputs,
        torch.arange(total_tokens, device=device).unsqueeze(0),
        None,
        output_cotangent,
        loss_denominator,
    )

    output_error = _error_ratio(reference_output, packed_output)
    gradient_error = _error_ratio(reference_gradient, packed_gradient)
    negative_error = _error_ratio(reference_output, negative_output)
    negative_gradient_error = _error_ratio(
        reference_gradient,
        negative_gradient,
    )
    return ProbeResult(
        name=name,
        output_error_ratio=output_error,
        input_gradient_error_ratio=gradient_error,
        negative_control_output_error_ratio=negative_error,
        negative_control_input_gradient_error_ratio=negative_gradient_error,
        reference_output_rms=_rms(reference_output),
        packed_output_rms=_rms(packed_output),
        reference_input_gradient_rms=_rms(reference_gradient),
        packed_input_gradient_rms=_rms(packed_gradient),
        input_gradient_error_rms=_rms(reference_gradient - packed_gradient),
    )


def _validation_failures(results: list[ProbeResult]) -> list[str]:
    failures = []
    for result in results:
        if result.output_error_ratio >= 3e-3:
            failures.append(
                f"{result.name}: packed output error ratio "
                f"{result.output_error_ratio}"
            )
        if result.input_gradient_error_ratio >= 5e-3:
            failures.append(
                f"{result.name}: packed input-gradient error ratio "
                f"{result.input_gradient_error_ratio}"
            )
        if result.negative_control_output_error_ratio <= max(
            result.output_error_ratio * 5,
            5e-4,
        ):
            failures.append(
                f"{result.name}: negative control is not sensitive: "
                f"{result.negative_control_output_error_ratio}"
            )
        if result.negative_control_input_gradient_error_ratio <= max(
            result.input_gradient_error_ratio * 5,
            5e-4,
        ):
            failures.append(
                f"{result.name}: gradient negative control is not sensitive: "
                f"{result.negative_control_input_gradient_error_ratio}"
            )
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    changed = apply_qwen3_5_packed_forward_patch()
    if not changed:
        raise RuntimeError("Qwen3.5 patch was already installed before the gate")

    probes = (
        ("gdn", ["linear_attention"]),
        ("full_attention", ["full_attention"]),
        ("hybrid", ["linear_attention", "full_attention"]),
    )
    results = [
        _run_probe(name, layer_types, [5, 7, 4], device)
        for name, layer_types in probes
    ]
    failures = _validation_failures(results)
    print(
        json.dumps(
            {
                "status": "fail" if failures else "pass",
                "device": str(device),
                "gradient_probe": "fixed_random_output_cotangent",
                "results": [result.__dict__ for result in results],
                "failures": failures,
            },
            sort_keys=True,
        )
    )
    if failures:
        raise AssertionError("; ".join(failures))


if __name__ == "__main__":
    main()
