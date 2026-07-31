#!/usr/bin/env python3
"""Run the historical 80,929-token OOM shape through response-only Qwen3.5."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import time

import torch
from transformers import AutoModelForCausalLM

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen3_5 import apply_qwen3_5_packed_forward_patch
from verl.workers.agent_actor.dp_actor import DataParallelPPOActor


class _ActorConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _micro_batch(
    device: torch.device,
    *,
    repeat: int,
) -> dict[str, torch.Tensor]:
    if repeat <= 0:
        raise ValueError(f"repeat must be positive, got {repeat}.")
    valid_lengths = [18_000, 16_000, 17_000, 29_929] * repeat
    response_lengths = [400, 400, 400, 628] * repeat
    sequence_length = 32_000
    response_width = 2_048
    prompt_width = sequence_length - response_width
    batch_rows = len(valid_lengths)
    input_ids = torch.zeros(batch_rows, sequence_length, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    responses = torch.zeros(batch_rows, response_width, dtype=torch.long)
    response_mask = torch.zeros_like(responses)

    for row, (valid_length, response_length) in enumerate(
        zip(valid_lengths, response_lengths, strict=True)
    ):
        prompt_length = valid_length - response_length
        prompt_start = prompt_width - prompt_length
        prompt_tokens = torch.arange(prompt_length, dtype=torch.long) % 1024 + 100
        response_tokens = torch.arange(response_length, dtype=torch.long) % 1024 + 2100
        input_ids[row, prompt_start:prompt_width] = prompt_tokens
        attention_mask[row, prompt_start:prompt_width] = 1
        responses[row, :response_length] = response_tokens
        response_mask[row, :response_length] = 1
        input_ids[row, prompt_width : prompt_width + response_length] = response_tokens
        attention_mask[row, prompt_width : prompt_width + response_length] = 1

    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "position_ids": position_ids.to(device),
        "responses": responses.to(device),
        "response_mask": response_mask.to(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the historical four-row OOM block for larger micro-batch stress.",
    )
    parser.add_argument(
        "--fused",
        action="store_true",
        help="Use the response-only fused PPO head instead of selected logits.",
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "triton"),
        default="triton",
    )
    parser.add_argument(
        "--tensor-output",
        help="Optional path for selected response log-probability and entropy tensors.",
    )
    parser.add_argument(
        "--skip-entropy",
        action="store_true",
        help="Skip entropy in the model forward, as compute_log_prob does.",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.001,
        help="Entropy coefficient used by the synthetic backward loss.",
    )
    parser.add_argument(
        "--attach-zero-entropy",
        action="store_true",
        help="Keep 0 * entropy in the graph to reproduce the legacy zero-coefficient path.",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Measure a no-grad log-probability forward without backward.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    apply_qwen3_5_packed_forward_patch()

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    load_seconds = time.perf_counter() - started

    actor = DataParallelPPOActor(
        config=_ActorConfig(
            use_remove_padding=True,
            ulysses_sequence_parallel_size=1,
            use_response_only_logits=True,
            use_response_fused_kernels=args.fused,
            response_fused_kernel_backend=args.backend,
        ),
        actor_module=model,
    )
    actor.compute_entropy_from_logits = verl_F.entropy_from_logits
    micro_batch = _micro_batch(device, repeat=args.repeat)
    packed_tokens = int(micro_batch["attention_mask"].sum().item())
    response_tokens = int(micro_batch["response_mask"].sum().item())
    expected_packed_tokens = 80_929 * args.repeat
    expected_response_tokens = 1_828 * args.repeat
    if (
        packed_tokens != expected_packed_tokens
        or response_tokens != expected_response_tokens
    ):
        raise RuntimeError(
            "historical shape mismatch: "
            f"packed={packed_tokens}/{expected_packed_tokens} "
            f"response={response_tokens}/{expected_response_tokens}"
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    forward_started = time.perf_counter()
    grad_context = torch.no_grad() if args.forward_only else nullcontext()
    with grad_context:
        entropy, log_probs = actor._forward_micro_batch(
            micro_batch,
            temperature=1.0,
            calculate_entropy=not args.skip_entropy,
        )
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    if not torch.isfinite(log_probs[micro_batch["response_mask"].bool()]).all():
        raise RuntimeError("response-only log probabilities are non-finite")
    if args.skip_entropy:
        if entropy is not None:
            raise RuntimeError("skip-entropy forward unexpectedly returned entropy")
    elif entropy is None or not torch.isfinite(
        entropy[micro_batch["response_mask"].bool()]
    ).all():
        raise RuntimeError("response-only entropy is missing or non-finite")

    loss_terms = log_probs
    if entropy is not None and (
        args.entropy_coeff != 0 or args.attach_zero_entropy
    ):
        loss_terms = loss_terms + args.entropy_coeff * entropy
    loss = -(loss_terms * micro_batch["response_mask"]).sum() / response_tokens
    backward_seconds = 0.0
    if not args.forward_only:
        torch.cuda.synchronize(device)
        backward_started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
    finite_gradient_parameters = 0
    nonzero_gradient_parameters = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("response-only backward produced a non-finite gradient")
        finite_gradient_parameters += 1
        if parameter.grad.detach().abs().sum().item() > 0:
            nonzero_gradient_parameters += 1
    if not args.forward_only and (
        finite_gradient_parameters == 0 or nonzero_gradient_parameters == 0
    ):
        raise RuntimeError("response-only backward produced no parameter gradients")

    if args.tensor_output:
        valid_response_mask = micro_batch["response_mask"].bool()
        torch.save(
            {
                "log_probs": log_probs[valid_response_mask].detach().float().cpu(),
                "entropy": (
                    None
                    if entropy is None
                    else entropy[valid_response_mask].detach().float().cpu()
                ),
                "loss": loss.detach().float().cpu(),
                "packed_tokens": packed_tokens,
                "response_tokens": response_tokens,
                "fused": args.fused,
                "backend": args.backend if args.fused else None,
            },
            args.tensor_output,
        )

    result = {
        "status": "pass",
        "device": str(device),
        "model": args.model,
        "batch_rows": int(micro_batch["input_ids"].shape[0]),
        "historical_block_repeat": args.repeat,
        "response_fused_kernels": args.fused,
        "response_fused_kernel_backend": args.backend if args.fused else None,
        "calculate_entropy": not args.skip_entropy,
        "entropy_coeff": args.entropy_coeff,
        "attach_zero_entropy": args.attach_zero_entropy,
        "forward_only": args.forward_only,
        "packed_tokens": packed_tokens,
        "selected_response_tokens": response_tokens,
        "projection_reduction_ratio": packed_tokens / response_tokens,
        "full_bf16_logits_gib": packed_tokens
        * model.config.vocab_size
        * 2
        / 1024**3,
        "selected_bf16_logits_gib": response_tokens
        * model.config.vocab_size
        * 2
        / 1024**3,
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "loss": float(loss.detach().item()),
        "finite_gradient_parameters": finite_gradient_parameters,
        "nonzero_gradient_parameters": nonzero_gradient_parameters,
        "tensor_output": args.tensor_output,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
