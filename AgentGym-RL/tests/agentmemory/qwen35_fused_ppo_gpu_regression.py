"""GPU equivalence gates for the upstream VERL fused PPO path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from verl.utils.experimental.torch_functional import FusedLinearForPPO


def _baseline(hidden, weight, labels, temperature):
    logits = (hidden @ weight.t()) / temperature
    logits_fp32 = logits.float()
    log_probs = logits_fp32.log_softmax(dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    probs = logits_fp32.softmax(dim=-1)
    entropy = torch.logsumexp(logits_fp32, dim=-1) - (probs * logits_fp32).sum(
        dim=-1
    )
    return log_probs, entropy.to(logits.dtype)


def run(device: str) -> dict[str, float | bool | str]:
    torch.manual_seed(20260726)
    target = torch.device(device)
    hidden_shape = (2, 19, 64)
    vocab_size = 256
    temperature = 0.73

    hidden_baseline = torch.randn(
        hidden_shape, device=target, dtype=torch.float32, requires_grad=True
    )
    weight_baseline = torch.randn(
        vocab_size,
        hidden_shape[-1],
        device=target,
        dtype=torch.float32,
        requires_grad=True,
    )
    hidden_fused = hidden_baseline.detach().clone().requires_grad_(True)
    weight_fused = weight_baseline.detach().clone().requires_grad_(True)
    labels = torch.randint(
        0, vocab_size, hidden_shape[:-1], device=target, dtype=torch.long
    )
    logprob_grad = torch.randn(hidden_shape[:-1], device=target)
    entropy_grad = torch.randn(hidden_shape[:-1], device=target)

    expected_log_probs, expected_entropy = _baseline(
        hidden_baseline, weight_baseline, labels, temperature
    )
    actual_log_probs, actual_entropy = FusedLinearForPPO(chunk_size=7)(
        hidden_fused, weight_fused, labels, temperature
    )

    expected_loss = (
        expected_log_probs * logprob_grad + expected_entropy * entropy_grad
    ).sum()
    actual_loss = (
        actual_log_probs * logprob_grad + actual_entropy * entropy_grad
    ).sum()
    expected_loss.backward()
    actual_loss.backward()

    torch.testing.assert_close(actual_log_probs, expected_log_probs, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_entropy, expected_entropy, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(hidden_fused.grad, hidden_baseline.grad, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(weight_fused.grad, weight_baseline.grad, atol=3e-5, rtol=3e-5)
    return {
        "passed": True,
        "device": str(target),
        "logprob_max_abs_delta": float(
            (actual_log_probs - expected_log_probs).abs().max().item()
        ),
        "entropy_max_abs_delta": float(
            (actual_entropy - expected_entropy).abs().max().item()
        ),
        "hidden_grad_max_abs_delta": float(
            (hidden_fused.grad - hidden_baseline.grad).abs().max().item()
        ),
        "weight_grad_max_abs_delta": float(
            (weight_fused.grad - weight_baseline.grad).abs().max().item()
        ),
    }


def _model_baseline(logits, labels, temperature):
    logits_fp32 = (logits / temperature).float()
    log_probs = logits_fp32.log_softmax(dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    probs = logits_fp32.softmax(dim=-1)
    entropy = torch.logsumexp(logits_fp32, dim=-1) - (
        probs * logits_fp32
    ).sum(dim=-1)
    return log_probs, entropy.to(logits.dtype)


def _gradient_snapshot(model) -> dict[str, dict[str, object]]:
    selected = {}
    required_parameter_groups = (
        ("model.layers.0.input_layernorm.weight",),
        ("model.layers.0.linear_attn.in_proj_qkv.weight",),
        ("model.layers.3.self_attn.q_proj.weight",),
        ("model.norm.weight",),
        # Qwen3.5 ties the LM head to input embeddings. named_parameters()
        # de-duplicates that storage and may expose only the embedding name.
        ("lm_head.weight", "model.embed_tokens.weight"),
    )
    preferred_suffixes = tuple(
        suffix for group in required_parameter_groups for suffix in group
    )
    for name, parameter in model.named_parameters():
        if not name.endswith(preferred_suffixes):
            continue
        grad = parameter.grad
        if grad is None:
            raise RuntimeError(f"Missing gradient for {name}.")
        flat = grad.detach().float().flatten()
        if not torch.isfinite(flat).all():
            raise RuntimeError(f"Non-finite gradient for {name}.")
        sample_count = min(32, flat.numel())
        if sample_count == 1:
            indices = torch.zeros(1, device=flat.device, dtype=torch.long)
        else:
            indices = (
                torch.arange(sample_count, device=flat.device, dtype=torch.long)
                * (flat.numel() - 1)
                // (sample_count - 1)
            )
        selected[name] = {
            "norm": float(torch.linalg.vector_norm(flat).item()),
            "samples": flat.index_select(0, indices).cpu(),
        }
    missing_groups = [
        group
        for group in required_parameter_groups
        if not any(
            name.endswith(suffix)
            for name in selected
            for suffix in group
        )
    ]
    if missing_groups:
        raise RuntimeError(
            "Could not find all expected Qwen3.5 GDN, full-attention, "
            "transformer, and LM-head parameters; missing "
            f"{missing_groups}, found {sorted(selected)}."
        )
    for name, values in selected.items():
        if values["norm"] == 0.0:
            raise RuntimeError(f"Zero gradient for {name}.")
    return selected


def _run_model_pass(
    *,
    model,
    input_ids,
    position_ids,
    labels,
    cu_seqlens,
    temperature,
    logprob_grad,
    entropy_grad,
    fused,
):
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(input_ids.device)
    torch.cuda.synchronize(input_ids.device)
    started = time.perf_counter()
    common_kwargs = {
        "input_ids": input_ids,
        "attention_mask": None,
        "position_ids": position_ids,
        "use_cache": False,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_cpu": cu_seqlens.detach().cpu(),
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if fused:
            output = model(
                **common_kwargs,
                _verl_fused_ppo=True,
                shift_labels=labels,
                temperature=float(temperature),
            )
            log_probs = output.log_probs
            entropy = output.entropy
        else:
            output = model(**common_kwargs)
            log_probs, entropy = _model_baseline(
                output.logits, labels, temperature
            )
        loss = (
            log_probs * logprob_grad + entropy * entropy_grad
        ).sum()
    loss.backward()
    torch.cuda.synchronize(input_ids.device)
    elapsed = time.perf_counter() - started
    peak_bytes = torch.cuda.max_memory_allocated(input_ids.device)
    return {
        "log_probs": log_probs.detach().float().cpu(),
        "entropy": entropy.detach().float().cpu(),
        "gradients": _gradient_snapshot(model),
        "seconds": elapsed,
        "peak_allocated_bytes": int(peak_bytes),
    }


def run_qwen35_model(model_path: str, device: str) -> dict[str, object]:
    """Exercise the actual text-only Qwen3.5 actor and packed GDN kernels."""

    from transformers import AutoConfig, AutoModelForCausalLM

    from verl.models.transformers.qwen3_5 import (
        apply_qwen3_5_packed_forward_patch,
    )
    from verl.workers.qwen35_runtime import resolve_qwen3_5_text_config

    target = torch.device(device)
    checkpoint = Path(model_path).expanduser().resolve()
    checkpoint_config = AutoConfig.from_pretrained(checkpoint)
    text_config = resolve_qwen3_5_text_config(checkpoint_config)
    apply_qwen3_5_packed_forward_patch()
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        config=text_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).to(target)
    model.train()
    model.config.use_cache = False

    torch.manual_seed(20260726)
    lengths = (7, 11)
    total_tokens = sum(lengths)
    vocab_size = int(text_config.vocab_size)
    input_ids = torch.randint(
        128,
        min(vocab_size, 32768),
        (1, total_tokens),
        device=target,
        dtype=torch.long,
    )
    position_ids = torch.cat(
        [torch.arange(length, device=target) for length in lengths]
    ).unsqueeze(0)
    cu_seqlens = torch.tensor(
        [0, lengths[0], total_tokens],
        device=target,
        dtype=torch.long,
    )
    labels = torch.roll(input_ids, shifts=-1, dims=-1)
    logprob_grad = torch.randn(
        labels.shape, device=target, dtype=torch.float32
    )
    entropy_grad = torch.randn(
        labels.shape, device=target, dtype=torch.float32
    ) * 0.01
    # Packed PPO never consumes the final next-token prediction of each
    # independently padded sample. Keep the backward gate aligned with that
    # contract instead of assigning a cross-sample rolled label at a boundary.
    sample_terminal_indices = (lengths[0] - 1, total_tokens - 1)
    logprob_grad[:, sample_terminal_indices] = 0
    entropy_grad[:, sample_terminal_indices] = 0
    temperature = 0.73

    baseline = _run_model_pass(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        cu_seqlens=cu_seqlens,
        temperature=temperature,
        logprob_grad=logprob_grad,
        entropy_grad=entropy_grad,
        fused=False,
    )
    fused = _run_model_pass(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        cu_seqlens=cu_seqlens,
        temperature=temperature,
        logprob_grad=logprob_grad,
        entropy_grad=entropy_grad,
        fused=True,
    )

    torch.testing.assert_close(
        fused["log_probs"], baseline["log_probs"], atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        fused["entropy"], baseline["entropy"], atol=2e-3, rtol=2e-3
    )
    gradient_deltas = {}
    for name, expected in baseline["gradients"].items():
        actual = fused["gradients"][name]
        torch.testing.assert_close(
            actual["samples"],
            expected["samples"],
            atol=2e-2,
            rtol=2e-2,
        )
        norm_relative_delta = abs(actual["norm"] - expected["norm"]) / max(
            expected["norm"], 1e-12
        )
        if norm_relative_delta > 2e-2:
            raise AssertionError(
                f"Gradient norm mismatch for {name}: "
                f"baseline={expected['norm']} fused={actual['norm']} "
                f"relative_delta={norm_relative_delta}"
            )
        gradient_deltas[name] = {
            "sample_max_abs_delta": float(
                (actual["samples"] - expected["samples"]).abs().max().item()
            ),
            "baseline_norm": expected["norm"],
            "fused_norm": actual["norm"],
            "norm_relative_delta": norm_relative_delta,
        }
    return {
        "passed": True,
        "model_path": str(checkpoint),
        "model_type": text_config.model_type,
        "tokens": total_tokens,
        "logprob_max_abs_delta": float(
            (fused["log_probs"] - baseline["log_probs"]).abs().max().item()
        ),
        "entropy_max_abs_delta": float(
            (fused["entropy"] - baseline["entropy"]).abs().max().item()
        ),
        "gradient_deltas": gradient_deltas,
        "baseline_seconds": baseline["seconds"],
        "fused_seconds": fused["seconds"],
        "baseline_peak_allocated_bytes": baseline["peak_allocated_bytes"],
        "fused_peak_allocated_bytes": fused["peak_allocated_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-path")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal fused PPO regression.")
    result = {"linear_head": run(args.device)}
    if args.model_path:
        result["qwen35_model"] = run_qwen35_model(
            args.model_path, args.device
        )
    payload = json.dumps(result, sort_keys=True)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(payload + "\n", encoding="utf-8")
        temporary_path.replace(output_path)
    print(payload)


if __name__ == "__main__":
    main()
