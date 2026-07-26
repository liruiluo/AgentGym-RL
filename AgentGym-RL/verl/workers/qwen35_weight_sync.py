"""Qwen3.5 actor-to-vLLM weight-name and coverage contracts."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable

from verl.models.transformers.qwen3_5 import is_qwen3_5_model_type


def resolve_vllm_init_load_format(
    *,
    model_type: str,
    configured_init_load_format,
) -> str | None:
    """Keep vLLM construction separate from per-rollout actor synchronization."""

    if is_qwen3_5_model_type(model_type):
        if configured_init_load_format is None:
            raise ValueError(
                "Qwen3.5 native vLLM requires rollout.vllm_init_load_format=dummy."
            )
        resolved = str(configured_init_load_format).lower()
        if resolved != "dummy":
            raise ValueError(
                "Qwen3.5 native vLLM requires vllm_init_load_format=dummy; "
                "current actor weights are synchronized separately as HF weights."
            )
        return resolved
    if configured_init_load_format is None:
        return None
    return str(configured_init_load_format).lower()


def clean_fsdp_weight_name(name: str) -> str:
    for prefix in ("_fsdp_wrapped_module.", "module."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def map_actor_weight_name_for_vllm(name: str, *, model_type: str) -> str:
    """Restore the multimodal checkpoint namespace expected by vLLM."""

    cleaned = clean_fsdp_weight_name(name)
    if not is_qwen3_5_model_type(model_type):
        return cleaned
    if cleaned.startswith("model."):
        return "model.language_model." + cleaned[len("model.") :]
    if cleaned.startswith("lm_head."):
        return cleaned
    raise ValueError(f"Unexpected Qwen3.5 actor weight name: {cleaned}")


def validate_qwen35_mapped_source_names(
    names: Iterable[str],
    *,
    expected_names: Iterable[str] | None,
) -> dict[str, object]:
    rows = list(names)
    counts = Counter(rows)
    expected_rows = list(expected_names or ())
    expected = Counter(expected_rows)
    duplicates = sorted(name for name, count in counts.items() if count != 1)
    unexpected = sorted(
        name
        for name in counts
        if not (
            name.startswith("model.language_model.")
            or name.startswith("lm_head.")
        )
    )
    missing = sorted((expected - counts).elements())
    extra = sorted((counts - expected).elements())
    if not rows or not expected_rows or duplicates or unexpected or counts != expected:
        raise RuntimeError(
            "Qwen3.5 actor source mapping mismatch: "
            f"source_count={len(rows)} expected_count={len(expected_rows)} "
            f"duplicates={duplicates[:8]} unexpected={unexpected[:8]} "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    return {
        "mapped_source_parameter_count": len(rows),
        "mapped_source_parameter_names_sha256": _hash_names(counts),
    }


def validate_qwen35_vllm_load_coverage(
    *,
    loaded_names: Iterable[str],
    target_parameter_names: Iterable[str],
) -> dict[str, object]:
    """Require all and only vLLM's text-policy parameters to be loaded."""

    loaded = set(loaded_names)
    expected = {
        name for name in target_parameter_names if name.startswith("language_model.")
    }
    missing = sorted(expected - loaded)
    unexpected = sorted(loaded - expected)
    if not expected or not loaded or missing or unexpected:
        raise RuntimeError(
            "Qwen3.5 actor-to-vLLM weight coverage mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]} "
            f"expected_count={len(expected)} loaded_count={len(loaded)}"
        )
    return {
        "expected_text_parameter_count": len(expected),
        "loaded_text_parameter_count": len(loaded),
        "loaded_text_parameter_names_sha256": _hash_names(loaded),
    }


def _hash_names(names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
