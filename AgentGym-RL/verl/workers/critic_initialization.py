"""Fail-closed initialization for the formal PPO scalar value head."""

from __future__ import annotations

from typing import Sequence

import torch


def zero_initialize_scalar_value_head(critic_module: torch.nn.Module) -> dict:
    """Zero the single direct classification head used as the PPO value head.

    Qwen token-classification critics expose this layer as ``score``. The
    ``classifier`` name is accepted for equivalent Hugging Face architectures,
    but ambiguous or non-scalar heads fail closed.
    """

    num_labels = getattr(getattr(critic_module, "config", None), "num_labels", None)
    if num_labels != 1:
        raise RuntimeError(
            f"Formal PPO critic must declare num_labels=1, got {num_labels!r}."
        )

    candidates = []
    for name in ("score", "classifier"):
        head = getattr(critic_module, name, None)
        if head is not None:
            candidates.append((name, head))
    if len(candidates) != 1:
        raise RuntimeError(
            "Formal PPO critic must expose exactly one direct score/classifier "
            f"head, found {[name for name, _ in candidates]}."
        )

    head_name, head = candidates[0]
    if not isinstance(head, torch.nn.Linear) or head.out_features != 1:
        raise RuntimeError(
            "Formal PPO value head must be torch.nn.Linear with out_features=1, "
            f"got {type(head).__name__} at {head_name!r}."
        )
    parameters = list(head.parameters(recurse=True))
    if not parameters:
        raise RuntimeError("Formal PPO value head has no parameters to initialize.")

    meta_states = {bool(parameter.is_meta) for parameter in parameters}
    if len(meta_states) != 1:
        raise RuntimeError("Formal PPO value head mixes meta and materialized parameters.")
    if True in meta_states:
        return {
            "head_name": head_name,
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "status": "meta_deferred_to_rank0_fsdp_sync",
            "all_parameters_zero": None,
        }

    with torch.no_grad():
        for parameter in parameters:
            parameter.zero_()
    nonzero_count = sum(
        int(torch.count_nonzero(parameter.detach()).item()) for parameter in parameters
    )
    if nonzero_count != 0:
        raise RuntimeError(
            f"Formal PPO value head zero initialization left {nonzero_count} nonzero values."
        )
    return {
        "head_name": head_name,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "status": "zero_initialized",
        "all_parameters_zero": True,
    }


def initialize_critic_value_head(
    critic_module: torch.nn.Module,
    *,
    missing_keys: Sequence[str],
    policy: str,
) -> dict | None:
    """Zero a fresh scalar head while preserving a loaded pretrained head."""

    policy = str(policy).strip().lower()
    if policy == "preserve":
        return None
    if policy != "zero_if_missing":
        raise RuntimeError(
            "critic.model.value_head_init must be 'preserve' or "
            f"'zero_if_missing', got {policy!r}."
        )

    candidates = [
        name
        for name in ("score", "classifier")
        if getattr(critic_module, name, None) is not None
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Formal PPO critic must expose exactly one direct score/classifier "
            f"head, found {candidates}."
        )
    head_name = candidates[0]
    head = getattr(critic_module, head_name)
    expected_head_keys = {
        f"{head_name}.{parameter_name}"
        for parameter_name, _ in head.named_parameters(recurse=True)
    }
    missing_head_keys = expected_head_keys.intersection(set(missing_keys))
    if not missing_head_keys:
        return {
            "head_name": head_name,
            "parameter_count": sum(
                parameter.numel() for parameter in head.parameters(recurse=True)
            ),
            "status": "pretrained_head_preserved",
            "all_parameters_zero": None,
        }
    if missing_head_keys != expected_head_keys:
        raise RuntimeError(
            "Formal PPO critic value head was only partially loaded: "
            f"expected={sorted(expected_head_keys)} missing={sorted(missing_head_keys)}."
        )
    return zero_initialize_scalar_value_head(critic_module)
