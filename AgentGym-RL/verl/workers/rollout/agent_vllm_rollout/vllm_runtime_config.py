"""Runtime configuration helpers for the official vLLM backend."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def resolve_official_vllm_compilation_config(
    *,
    enforce_eager: bool,
    configured: Any = None,
    cudagraph_capture_sizes: Sequence[int] | None = None,
) -> int | dict[str, Any] | None:
    """Build the vLLM compilation config used by current VERL releases."""
    if isinstance(configured, str):
        try:
            configured = json.loads(configured)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "official vLLM compilation_config must be valid JSON"
            ) from exc

    if configured is None:
        compilation_config: int | dict[str, Any] = {}
    elif isinstance(configured, bool):
        raise TypeError(
            "official vLLM compilation_config must be a mapping or integer"
        )
    elif isinstance(configured, int):
        if cudagraph_capture_sizes:
            raise ValueError(
                "cudagraph_capture_sizes cannot be merged into an integer "
                "compilation_config"
            )
        return configured
    elif isinstance(configured, Mapping):
        compilation_config = dict(configured)
    else:
        raise TypeError(
            "official vLLM compilation_config must be a mapping or integer"
        )

    if not enforce_eager:
        compilation_config.setdefault(
            "cudagraph_mode", "FULL_AND_PIECEWISE"
        )
        if cudagraph_capture_sizes:
            compilation_config["cudagraph_capture_sizes"] = [
                int(size) for size in cudagraph_capture_sizes
            ]

    return compilation_config or None
