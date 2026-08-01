"""Fail-closed transport selection for official vLLM HF weight sync."""

from __future__ import annotations

import os


TRANSPORT_ENV = "AGENTMEMORY_VLLM_HF_SYNC_TRANSPORT"
_VALID_TRANSPORTS = {"file", "direct_inproc"}
_EXPECTED_DIRECT_TYPES = {
    "engine_client": "vllm.v1.engine.core_client.InprocClient",
    "engine_core": "vllm.v1.engine.core.EngineCore",
    "model_executor": "vllm.v1.executor.uniproc_executor.UniProcExecutor",
}


def resolve_hf_sync_transport(environ=None):
    environ = os.environ if environ is None else environ
    value = str(environ.get(TRANSPORT_ENV, "file")).strip().lower()
    if value not in _VALID_TRANSPORTS:
        allowed = ", ".join(sorted(_VALID_TRANSPORTS))
        raise ValueError(f"{TRANSPORT_ENV} must be one of {allowed}, got {value!r}")
    return value


def _qualified_type(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def require_direct_inproc_runtime(llm_engine, *, infer_tp_size):
    """Prove apply_model executes the captured closure in this process."""
    if int(infer_tp_size) != 1:
        raise RuntimeError(
            "direct_inproc HF sync requires rollout tensor_model_parallel_size=1, "
            f"got {infer_tp_size}"
        )

    engine_client = getattr(llm_engine, "engine_core", None)
    engine_core = getattr(engine_client, "engine_core", None)
    model_executor = getattr(engine_core, "model_executor", None)
    actual = {
        "engine_client": _qualified_type(engine_client),
        "engine_core": _qualified_type(engine_core),
        "model_executor": _qualified_type(model_executor),
    }
    mismatches = {
        key: {"expected": expected, "actual": actual[key]}
        for key, expected in _EXPECTED_DIRECT_TYPES.items()
        if actual[key] != expected
    }
    if mismatches:
        raise RuntimeError(
            "direct_inproc HF sync requires the non-serializing official vLLM "
            f"InprocClient/EngineCore/UniProcExecutor path, got {mismatches}"
        )
    return actual
