"""Runtime configuration helpers for the official vLLM backend."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TRAINING_TRITON_CACHE_ENV = "VERL_TRAINING_TRITON_CACHE_DIR"


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


def restore_training_triton_cache_after_vllm(
    *, triton_module: Any = None
) -> dict[str, str] | None:
    """Restore a stable Triton cache after vLLM rewrites its cache paths."""
    requested = os.environ.get(TRAINING_TRITON_CACHE_ENV)
    if not requested:
        return None

    requested_path = Path(requested).expanduser()
    if not requested_path.is_absolute():
        raise ValueError(
            f"{TRAINING_TRITON_CACHE_ENV} must be an absolute path, "
            f"got {requested!r}"
        )
    requested_path.mkdir(parents=True, exist_ok=True)
    if not requested_path.is_dir():
        raise RuntimeError(
            f"{TRAINING_TRITON_CACHE_ENV} is not a directory: "
            f"{requested_path}"
        )

    stable_dir = str(requested_path.resolve())
    os.environ[TRAINING_TRITON_CACHE_ENV] = stable_dir
    os.environ["TRITON_CACHE_DIR"] = stable_dir
    os.environ["FLA_CACHE_RESULTS"] = "1"

    if triton_module is None:
        import triton as triton_module

    try:
        runtime_dir = str(
            Path(os.fspath(triton_module.knobs.cache.dir))
            .expanduser()
            .resolve()
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "Unable to attest Triton runtime cache directory"
        ) from exc
    if runtime_dir != stable_dir:
        raise RuntimeError(
            "TRITON_CACHE_DIR did not take effect in the Triton runtime: "
            f"requested={stable_dir!r} runtime={runtime_dir!r}"
        )

    return {
        "requested_dir": stable_dir,
        "runtime_dir": runtime_dir,
        "fla_cache_results": os.environ["FLA_CACHE_RESULTS"],
        "triton_version": str(getattr(triton_module, "__version__", "unknown")),
    }
