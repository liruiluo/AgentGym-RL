#!/usr/bin/env python3
"""Benchmark zero-entropy PPO paths after loading Qwen3.5 only once."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoModelForCausalLM

import verl.utils.torch_functional as verl_F
from qwen35_response_only_oom_shape_gpu_regression import _ActorConfig, _micro_batch
from verl.models.transformers.qwen3_5 import apply_qwen3_5_packed_forward_patch
from verl.workers.agent_actor.dp_actor import DataParallelPPOActor


def _max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    apply_qwen3_5_packed_forward_patch()

    load_started = time.perf_counter()
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
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    actor = DataParallelPPOActor(
        config=_ActorConfig(
            use_remove_padding=True,
            ulysses_sequence_parallel_size=1,
            use_response_only_logits=True,
            use_response_fused_kernels=True,
            response_fused_kernel_backend="torch",
        ),
        actor_module=model,
    )
    actor.compute_entropy_from_logits = verl_F.entropy_from_logits
    micro_batch = _micro_batch(device, repeat=args.repeat)
    response_mask = micro_batch["response_mask"]
    response_tokens = int(response_mask.sum().item())
    packed_tokens = int(micro_batch["attention_mask"].sum().item())

    def clear_graph_state() -> None:
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    def gradient_samples() -> dict[str, torch.Tensor]:
        samples = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            flat = parameter.grad.detach().flatten()
            count = min(64, flat.numel())
            indices = torch.arange(count, device=flat.device, dtype=torch.long)
            if count > 1:
                indices = indices * (flat.numel() - 1) // (count - 1)
            samples[name] = flat.index_select(0, indices).float().cpu()
            if len(samples) == 12:
                break
        if not samples:
            raise RuntimeError("actor backward produced no gradient samples")
        return samples

    def run_actor(attach_zero_entropy: bool) -> dict:
        clear_graph_state()
        torch.cuda.reset_peak_memory_stats(device)
        forward_started = time.perf_counter()
        entropy, log_probs = actor._forward_micro_batch(
            micro_batch,
            temperature=1.0,
            calculate_entropy=True,
        )
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_started
        if entropy is None:
            raise RuntimeError("actor benchmark requires entropy readback")

        loss_terms = log_probs + 0.0 * entropy if attach_zero_entropy else log_probs
        loss = -(loss_terms * response_mask).sum() / response_tokens
        backward_started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        result = {
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "total_seconds": forward_seconds + backward_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "loss": float(loss.detach().item()),
            "log_probs": log_probs.detach().float().cpu(),
            "entropy": entropy.detach().float().cpu(),
            "gradient_samples": gradient_samples(),
        }
        del entropy, log_probs, loss_terms, loss
        clear_graph_state()
        return result

    def run_oldlog(calculate_entropy: bool) -> dict:
        clear_graph_state()
        torch.cuda.reset_peak_memory_stats(device)
        forward_started = time.perf_counter()
        with torch.no_grad():
            entropy, log_probs = actor._forward_micro_batch(
                micro_batch,
                temperature=1.0,
                calculate_entropy=calculate_entropy,
            )
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_started
        if calculate_entropy != (entropy is not None):
            raise RuntimeError("old-logprob entropy contract mismatch")
        result = {
            "forward_seconds": forward_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "log_probs": log_probs.detach().float().cpu(),
            "entropy": None if entropy is None else entropy.detach().float().cpu(),
        }
        del entropy, log_probs
        clear_graph_state()
        return result

    # Warm both Transformer backward and logprob-only head paths before timing.
    run_actor(attach_zero_entropy=True)
    run_oldlog(calculate_entropy=False)

    actor_legacy = [run_actor(True)]
    actor_skip = [run_actor(False), run_actor(False)]
    actor_legacy.append(run_actor(True))
    oldlog_with_entropy = [run_oldlog(True)]
    oldlog_skip_entropy = [run_oldlog(False), run_oldlog(False)]
    oldlog_with_entropy.append(run_oldlog(True))

    actor_logprob_delta = _max_abs_delta(
        actor_legacy[0]["log_probs"], actor_skip[0]["log_probs"]
    )
    actor_entropy_delta = _max_abs_delta(
        actor_legacy[0]["entropy"], actor_skip[0]["entropy"]
    )
    oldlog_delta = _max_abs_delta(
        oldlog_with_entropy[0]["log_probs"], oldlog_skip_entropy[0]["log_probs"]
    )
    gradient_delta = 0.0
    if actor_legacy[0]["gradient_samples"].keys() != actor_skip[0]["gradient_samples"].keys():
        raise RuntimeError("actor gradient sample keys changed")
    for name in actor_legacy[0]["gradient_samples"]:
        gradient_delta = max(
            gradient_delta,
            _max_abs_delta(
                actor_legacy[0]["gradient_samples"][name],
                actor_skip[0]["gradient_samples"][name],
            ),
        )

    actor_legacy_seconds = statistics.median(
        result["total_seconds"] for result in actor_legacy
    )
    actor_skip_seconds = statistics.median(
        result["total_seconds"] for result in actor_skip
    )
    oldlog_with_entropy_seconds = statistics.median(
        result["forward_seconds"] for result in oldlog_with_entropy
    )
    oldlog_skip_entropy_seconds = statistics.median(
        result["forward_seconds"] for result in oldlog_skip_entropy
    )
    if actor_logprob_delta != 0 or actor_entropy_delta != 0 or oldlog_delta != 0:
        raise RuntimeError("zero-entropy optimization changed forward values")
    if gradient_delta != 0:
        raise RuntimeError(
            f"zero-entropy optimization changed sampled gradients: {gradient_delta}"
        )

    summary = {
        "status": "pass",
        "device": str(device),
        "model": args.model,
        "load_seconds": load_seconds,
        "batch_rows": int(micro_batch["input_ids"].shape[0]),
        "packed_tokens": packed_tokens,
        "selected_response_tokens": response_tokens,
        "actor_legacy_zero_seconds": actor_legacy_seconds,
        "actor_skip_zero_seconds": actor_skip_seconds,
        "actor_speedup_fraction": 1 - actor_skip_seconds / actor_legacy_seconds,
        "oldlog_with_entropy_seconds": oldlog_with_entropy_seconds,
        "oldlog_skip_entropy_seconds": oldlog_skip_entropy_seconds,
        "oldlog_speedup_fraction": 1 - oldlog_skip_entropy_seconds / oldlog_with_entropy_seconds,
        "actor_logprob_max_abs_delta": actor_logprob_delta,
        "actor_entropy_max_abs_delta": actor_entropy_delta,
        "actor_gradient_sample_max_abs_delta": gradient_delta,
        "oldlog_max_abs_delta": oldlog_delta,
        "actor_legacy_peak_reserved_gib": max(
            result["peak_reserved_gib"] for result in actor_legacy
        ),
        "actor_skip_peak_reserved_gib": max(
            result["peak_reserved_gib"] for result in actor_skip
        ),
        "oldlog_with_entropy_peak_reserved_gib": max(
            result["peak_reserved_gib"] for result in oldlog_with_entropy
        ),
        "oldlog_skip_entropy_peak_reserved_gib": max(
            result["peak_reserved_gib"] for result in oldlog_skip_entropy
        ),
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
