#!/usr/bin/env python3
"""GPU equivalence gate for response-only Qwen3.5 fused PPO kernels."""

from __future__ import annotations

import argparse
import copy
import json

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen3_5 import apply_qwen3_5_packed_forward_patch
from verl.workers.agent_actor.dp_actor import DataParallelPPOActor


class _ActorConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _make_model(device: torch.device) -> Qwen3_5ForCausalLM:
    config = Qwen3_5TextConfig(
        vocab_size=257,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        attention_dropout=0.0,
        tie_word_embeddings=False,
        dtype=torch.bfloat16,
    )
    config._attn_implementation = "flash_attention_2"
    torch.manual_seed(1234)
    return Qwen3_5ForCausalLM(config).to(
        device=device,
        dtype=torch.bfloat16,
    )


def _actor(
    model: Qwen3_5ForCausalLM,
    *,
    fused: bool,
    backend: str,
) -> DataParallelPPOActor:
    actor = DataParallelPPOActor(
        config=_ActorConfig(
            use_remove_padding=True,
            ulysses_sequence_parallel_size=1,
            use_response_only_logits=True,
            use_response_fused_kernels=fused,
            response_fused_kernel_backend=backend,
        ),
        actor_module=model,
    )
    actor.compute_entropy_from_logits = verl_F.entropy_from_logits
    return actor


def _micro_batch(device: torch.device) -> dict[str, torch.Tensor]:
    prompt_width = 8
    response_width = 4
    input_ids = torch.zeros(3, prompt_width + response_width, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    responses = torch.zeros(3, response_width, dtype=torch.long)
    response_mask = torch.zeros_like(responses)
    rows = (
        ([10, 11, 12, 13, 14], [21, 22, 23]),
        ([30, 31, 32, 33, 34, 35, 36, 37], [41, 42, 43, 44]),
        ([50, 51, 52], [61, 62]),
    )
    for row, (prompt, response) in enumerate(rows):
        prompt_start = prompt_width - len(prompt)
        input_ids[row, prompt_start:prompt_width] = torch.tensor(prompt)
        attention_mask[row, prompt_start:prompt_width] = 1
        responses[row, : len(response)] = torch.tensor(response)
        response_mask[row, : len(response)] = 1
        input_ids[row, prompt_width : prompt_width + len(response)] = torch.tensor(
            response
        )
        attention_mask[row, prompt_width : prompt_width + len(response)] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "position_ids": position_ids.to(device),
        "responses": responses.to(device),
        "response_mask": response_mask.to(device),
    }


def _gradient_delta(reference, actual) -> tuple[float, float]:
    max_abs_delta = 0.0
    delta_square_sum = 0.0
    reference_square_sum = 0.0
    for (reference_name, reference_parameter), (actual_name, actual_parameter) in zip(
        reference.named_parameters(),
        actual.named_parameters(),
        strict=True,
    ):
        if reference_name != actual_name:
            raise RuntimeError(
                f"parameter mismatch: {reference_name} != {actual_name}"
            )
        reference_gradient = reference_parameter.grad
        actual_gradient = actual_parameter.grad
        if (reference_gradient is None) != (actual_gradient is None):
            raise RuntimeError(f"gradient presence mismatch for {reference_name}")
        if reference_gradient is None:
            continue
        delta = reference_gradient.float() - actual_gradient.float()
        max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        delta_square_sum += float(delta.square().sum().item())
        reference_square_sum += float(
            reference_gradient.float().square().sum().item()
        )
    relative_l2_delta = (delta_square_sum / max(reference_square_sum, 1e-30)) ** 0.5
    return max_abs_delta, relative_l2_delta


def _run_pass(actor, model, micro_batch, temperature: float):
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(micro_batch["input_ids"].device)
    entropy, log_probs = actor._forward_micro_batch(
        micro_batch,
        temperature=temperature,
    )
    response_mask = micro_batch["response_mask"].to(torch.float32)
    loss = -(
        (log_probs.float() + 0.01 * entropy.float()) * response_mask
    ).sum() / response_mask.sum()
    loss.backward()
    torch.cuda.synchronize()
    return {
        "entropy": entropy.detach(),
        "log_probs": log_probs.detach(),
        "loss": float(loss.detach().item()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(
            micro_batch["input_ids"].device
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backend",
        choices=("torch", "triton"),
        default="triton",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    apply_qwen3_5_packed_forward_patch()

    baseline_model = _make_model(device)
    fused_model = copy.deepcopy(baseline_model)
    micro_batch = _micro_batch(device)
    baseline = _run_pass(
        _actor(baseline_model, fused=False, backend=args.backend),
        baseline_model,
        micro_batch,
        temperature=0.9,
    )
    fused = _run_pass(
        _actor(fused_model, fused=True, backend=args.backend),
        fused_model,
        micro_batch,
        temperature=0.9,
    )

    valid = micro_batch["response_mask"].bool()
    logprob_delta = float(
        (baseline["log_probs"][valid].float() - fused["log_probs"][valid].float())
        .abs()
        .max()
        .item()
    )
    entropy_delta = float(
        (baseline["entropy"][valid].float() - fused["entropy"][valid].float())
        .abs()
        .max()
        .item()
    )
    gradient_abs_delta, gradient_relative_l2_delta = _gradient_delta(
        baseline_model,
        fused_model,
    )
    failures = []
    if logprob_delta > 2e-2:
        failures.append(f"logprob delta {logprob_delta} > 0.02")
    if entropy_delta > 5e-2:
        failures.append(f"entropy delta {entropy_delta} > 0.05")
    if gradient_relative_l2_delta > 5e-2:
        failures.append(
            "gradient relative L2 delta "
            f"{gradient_relative_l2_delta} > 0.05"
        )
    result = {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "backend": args.backend,
        "device": str(device),
        "packed_tokens": int(micro_batch["attention_mask"].sum().item()),
        "selected_response_tokens": int(valid.sum().item()),
        "baseline_loss": baseline["loss"],
        "fused_loss": fused["loss"],
        "loss_abs_delta": abs(baseline["loss"] - fused["loss"]),
        "logprob_max_abs_delta": logprob_delta,
        "entropy_max_abs_delta": entropy_delta,
        "gradient_max_abs_delta": gradient_abs_delta,
        "gradient_relative_l2_delta": gradient_relative_l2_delta,
        "baseline_peak_allocated_bytes": baseline["peak_allocated_bytes"],
        "fused_peak_allocated_bytes": fused["peak_allocated_bytes"],
    }
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise AssertionError("; ".join(failures))


if __name__ == "__main__":
    main()
