#!/usr/bin/env python3
"""GPU equivalence gate for Qwen3.5 response-only PPO logits."""

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
        vocab_size=128,
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
    return Qwen3_5ForCausalLM(config).to(device=device, dtype=torch.bfloat16)


def _actor(model, response_only: bool) -> DataParallelPPOActor:
    config = _ActorConfig(
        use_remove_padding=True,
        ulysses_sequence_parallel_size=1,
        use_response_only_logits=response_only,
    )
    actor = DataParallelPPOActor(config=config, actor_module=model)
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
        input_ids[row, prompt_width : prompt_width + len(response)] = torch.tensor(response)
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


def _max_gradient_delta(reference, actual) -> tuple[float, float]:
    max_abs_delta = 0.0
    max_relative_delta = 0.0
    for (reference_name, reference_parameter), (actual_name, actual_parameter) in zip(
        reference.named_parameters(), actual.named_parameters(), strict=True
    ):
        if reference_name != actual_name:
            raise RuntimeError(f"parameter mismatch: {reference_name} != {actual_name}")
        reference_gradient = reference_parameter.grad
        actual_gradient = actual_parameter.grad
        if (reference_gradient is None) != (actual_gradient is None):
            raise RuntimeError(f"gradient presence mismatch for {reference_name}")
        if reference_gradient is None:
            continue
        delta = (reference_gradient.float() - actual_gradient.float()).abs()
        max_abs_delta = max(max_abs_delta, float(delta.max().item()))
        scale = float(reference_gradient.float().abs().max().item())
        max_relative_delta = max(
            max_relative_delta,
            float(delta.max().item()) / (scale + 1e-8),
        )
    return max_abs_delta, max_relative_delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    apply_qwen3_5_packed_forward_patch()

    full_model = _make_model(device)
    sparse_model = copy.deepcopy(full_model)
    full_actor = _actor(full_model, response_only=False)
    sparse_actor = _actor(sparse_model, response_only=True)
    micro_batch = _micro_batch(device)
    response_mask = micro_batch["response_mask"].to(torch.bfloat16)

    full_entropy, full_log_probs = full_actor._forward_micro_batch(
        micro_batch, temperature=0.9
    )
    full_loss = -(
        (full_log_probs + 0.01 * full_entropy) * response_mask
    ).sum()
    full_loss.backward()

    sparse_entropy, sparse_log_probs = sparse_actor._forward_micro_batch(
        micro_batch, temperature=0.9
    )
    sparse_loss = -(
        (sparse_log_probs + 0.01 * sparse_entropy) * response_mask
    ).sum()
    sparse_loss.backward()

    valid = response_mask.bool()
    logprob_delta = float(
        (full_log_probs[valid].float() - sparse_log_probs[valid].float())
        .abs()
        .max()
        .item()
    )
    entropy_delta = float(
        (full_entropy[valid].float() - sparse_entropy[valid].float())
        .abs()
        .max()
        .item()
    )
    gradient_abs_delta, gradient_relative_delta = _max_gradient_delta(
        full_model, sparse_model
    )
    prompt_gradient_nonzero = any(
        parameter.grad is not None
        and parameter.grad.detach().float().abs().sum().item() > 0
        for name, parameter in sparse_model.named_parameters()
        if "embed_tokens" in name
    )
    result = {
        "device": str(device),
        "valid_response_tokens": int(valid.sum().item()),
        "packed_tokens": int(micro_batch["attention_mask"].sum().item()),
        "logprob_max_abs_delta": logprob_delta,
        "entropy_max_abs_delta": entropy_delta,
        "gradient_max_abs_delta": gradient_abs_delta,
        "gradient_max_relative_delta": gradient_relative_delta,
        "prompt_processing_gradient_nonzero": prompt_gradient_nonzero,
    }
    failures = []
    if logprob_delta > 5e-2:
        failures.append(f"logprob delta {logprob_delta} > 0.05")
    if entropy_delta > 5e-2:
        failures.append(f"entropy delta {entropy_delta} > 0.05")
    if gradient_relative_delta > 5e-2:
        failures.append(f"gradient relative delta {gradient_relative_delta} > 0.05")
    if not prompt_gradient_nonzero:
        failures.append("response loss did not reach prompt token embeddings")
    result["status"] = "fail" if failures else "pass"
    result["failures"] = failures
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise AssertionError("; ".join(failures))


if __name__ == "__main__":
    main()
