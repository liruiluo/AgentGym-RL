"""Runtime contract for native Qwen3.5 PPO training on NVIDIA GPUs."""

from __future__ import annotations

import importlib
import inspect
from importlib import metadata

import torch

from verl.models.transformers.qwen3_5 import is_qwen3_5_model_type


def resolve_qwen3_5_text_config(checkpoint_config):
    """Return the text config used by the causal actor, or the input config."""

    if getattr(checkpoint_config, "model_type", None) != "qwen3_5":
        return checkpoint_config
    text_config = getattr(checkpoint_config, "text_config", None)
    if getattr(text_config, "model_type", None) != "qwen3_5_text":
        raise ValueError("Qwen3.5 checkpoint config has no valid text_config.")
    return text_config


def model_type_from_module(module) -> str:
    """Read the HF model type through common FSDP/module wrappers."""

    seen = set()
    current = module
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        config = getattr(current, "config", None)
        model_type = getattr(config, "model_type", None)
        if isinstance(model_type, str):
            return model_type
        current = getattr(current, "module", None)
    return ""


def qwen3_5_packed_forward_kwargs(module, cu_seqlens, sequence_parallel_size: int):
    """Build the boundary ledger consumed by Qwen3.5's GDN layers."""

    if not is_qwen3_5_model_type(model_type_from_module(module)):
        return {}
    if sequence_parallel_size != 1:
        raise NotImplementedError(
            "AgentGym-RL Qwen3.5 currently validates remove-padding with "
            "Ulysses sequence parallel size 1. Use four-way FSDP data "
            "parallelism; gate SP>1 separately before enabling it."
        )
    cu_seqlens = cu_seqlens.to(dtype=torch.long)
    return {
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_cpu": cu_seqlens.detach().cpu(),
    }


def validate_qwen3_5_training_runtime(
    *,
    model_type: str,
    use_remove_padding: bool,
    sequence_parallel_size: int,
) -> dict[str, object] | None:
    """Fail before rollout when the native Qwen3.5 training path is unsafe."""

    if not is_qwen3_5_model_type(model_type):
        return None
    if not use_remove_padding:
        raise RuntimeError(
            "Qwen3.5 formal PPO requires use_remove_padding=true; the padded "
            "fallback is too slow for a scalable B200 run."
        )
    if sequence_parallel_size != 1:
        raise NotImplementedError(
            "Qwen3.5 SP>1 is not enabled in AgentGym-RL until its context-"
            "parallel path passes a separate B200 gate."
        )

    versions = {"torch": torch.__version__}
    for distribution, module_name in (
        ("transformers", "transformers"),
        ("flash-attn", "flash_attn"),
        ("flash-linear-attention", "fla"),
        ("causal-conv1d", "causal_conv1d"),
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"Qwen3.5 native training requires importable {module_name}."
            ) from exc
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "unknown"

    from causal_conv1d import causal_conv1d_fn
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.integrations.flash_attention import (
        _flash_attention_forward,
    )

    required_parameters = (
        (chunk_gated_delta_rule, "cu_seqlens", "FLA gated-delta rule"),
        (chunk_gated_delta_rule, "cu_seqlens_cpu", "FLA gated-delta rule"),
        (causal_conv1d_fn, "seq_idx", "causal-conv1d"),
        (
            _flash_attention_forward,
            "position_ids",
            "Transformers FlashAttention",
        ),
    )
    for function, parameter, label in required_parameters:
        if parameter not in inspect.signature(function).parameters:
            raise RuntimeError(
                f"Qwen3.5 packed training requires {label} to accept "
                f"{parameter}."
            )
    versions["packed_gdn_cu_seqlens"] = True
    versions["packed_conv_seq_idx"] = True
    versions["packed_flash_position_ids"] = True

    capability = torch.cuda.get_device_capability()
    versions["cuda_capability"] = f"{capability[0]}.{capability[1]}"
    if capability[0] >= 10:
        wy_fast = importlib.import_module("fla.ops.gated_delta_rule.wy_fast")
        warps = getattr(wy_fast, "PREPARE_WY_REPR_BWD_NUM_WARPS", None)
        stages = getattr(wy_fast, "PREPARE_WY_REPR_BWD_NUM_STAGES", None)
        if warps != [2] or stages != [4]:
            raise RuntimeError(
                "Blackwell Qwen3.5 training requires the FLA #1000 gated-"
                "delta backward restriction: num_warps=[2], num_stages=[4]. "
                f"Observed warps={warps!r}, stages={stages!r}."
            )
        versions["blackwell_gdn_backward_gate"] = "restricted_2x4"
    return versions
