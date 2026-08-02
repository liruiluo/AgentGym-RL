"""Fail-closed contracts for reusing vLLM sampled-token log probabilities."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


ROLLOUT_LOGPROB_MODE_ENV = "AGENTMEMORY_ROLLOUT_LOGPROB_MODE"
ROLLOUT_LOGPROB_MODE_OFF = "off"
ROLLOUT_LOGPROB_MODE_COMPARE = "compare"
ROLLOUT_LOGPROB_MODE_BYPASS = "bypass"
ROLLOUT_LOGPROB_BATCH_KEY = "rollout_log_probs"
SAMPLED_LOGPROBS_RECORD_KEY = "_agentmemory_sampled_token_logprobs"

_EVIDENCE_PROMPT_TOKEN_IDS = "prompt_token_ids"
_EVIDENCE_RESPONSE_TOKEN_IDS = "response_token_ids"
_EVIDENCE_LOG_PROBS = "log_probs"

_VALID_MODES = {
    ROLLOUT_LOGPROB_MODE_OFF,
    ROLLOUT_LOGPROB_MODE_COMPARE,
    ROLLOUT_LOGPROB_MODE_BYPASS,
}


def resolve_rollout_logprob_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    mode = str(source.get(ROLLOUT_LOGPROB_MODE_ENV, ROLLOUT_LOGPROB_MODE_OFF))
    mode = mode.strip().lower()
    if mode not in _VALID_MODES:
        valid = ", ".join(sorted(_VALID_MODES))
        raise ValueError(
            f"{ROLLOUT_LOGPROB_MODE_ENV} must be one of {valid}; got {mode!r}."
        )
    return mode


def rollout_logprob_reuse_enabled(mode: str) -> bool:
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown rollout-logprob mode: {mode!r}.")
    return mode != ROLLOUT_LOGPROB_MODE_OFF


def validate_rollout_logprob_training_scope(
    *,
    task_name: str,
    adv_estimator: str,
    mode: str,
) -> bool:
    """Validate the opt-in trainer scope and return whether this is AMG."""

    agentmemory_task = str(task_name).strip().lower() == "agentmemory"
    if not rollout_logprob_reuse_enabled(mode):
        return agentmemory_task
    if not agentmemory_task:
        raise ValueError("Rollout-logprob reuse is scoped to AgentMemoryGym.")
    if str(adv_estimator).strip().lower() != "gae":
        raise ValueError("Rollout-logprob reuse currently requires PPO/GAE.")
    return True


def _sampling_value(sampling_params: Any, name: str, default: Any) -> Any:
    if isinstance(sampling_params, Mapping):
        return sampling_params.get(name, default)
    return getattr(sampling_params, name, default)


def _require_numeric_equal(name: str, value: Any, expected: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(
            f"Rollout-logprob reuse requires numeric {name}={expected}; "
            f"got {value!r}."
        )
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Rollout-logprob reuse requires {name}={expected}; got {value!r}."
        )


def _is_unset_or_empty_collection(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, Mapping):
        return len(value) == 0
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) == 0
    return False


def validate_rollout_logprob_sampling_contract(
    sampling_params: Any,
    mode: str,
) -> dict[str, Any]:
    """Require vLLM logprobs to represent the same raw policy used by PPO."""

    if not rollout_logprob_reuse_enabled(mode):
        return {"mode": mode}

    logprobs = _sampling_value(sampling_params, "logprobs", None)
    if isinstance(logprobs, bool) or not isinstance(logprobs, int) or logprobs < 1:
        raise RuntimeError(
            "Rollout-logprob reuse requires SamplingParams.logprobs >= 1."
        )
    n = _sampling_value(sampling_params, "n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n != 1:
        raise RuntimeError("Rollout-logprob reuse requires SamplingParams.n=1.")

    _require_numeric_equal(
        "temperature", _sampling_value(sampling_params, "temperature", 1.0), 1.0
    )
    _require_numeric_equal(
        "top_p", _sampling_value(sampling_params, "top_p", 1.0), 1.0
    )
    top_k = _sampling_value(sampling_params, "top_k", None)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k not in (-1, 0):
        raise RuntimeError(
            "Rollout-logprob reuse requires disabled top_k (0 or -1); "
            f"got {top_k!r}."
        )
    _require_numeric_equal(
        "min_p", _sampling_value(sampling_params, "min_p", 0.0), 0.0
    )
    _require_numeric_equal(
        "presence_penalty",
        _sampling_value(sampling_params, "presence_penalty", 0.0),
        0.0,
    )
    _require_numeric_equal(
        "frequency_penalty",
        _sampling_value(sampling_params, "frequency_penalty", 0.0),
        0.0,
    )
    _require_numeric_equal(
        "repetition_penalty",
        _sampling_value(sampling_params, "repetition_penalty", 1.0),
        1.0,
    )
    _require_numeric_equal(
        "min_tokens", _sampling_value(sampling_params, "min_tokens", 0), 0.0
    )

    best_of = _sampling_value(sampling_params, "best_of", None)
    if best_of not in (None, 1):
        raise RuntimeError("Rollout-logprob reuse requires best_of unset or 1.")
    if bool(_sampling_value(sampling_params, "use_beam_search", False)):
        raise RuntimeError("Rollout-logprob reuse does not support beam search.")
    if bool(_sampling_value(sampling_params, "ignore_eos", False)):
        raise RuntimeError("Rollout-logprob reuse requires ignore_eos=false.")
    truncate_prompt_tokens = _sampling_value(
        sampling_params, "truncate_prompt_tokens", None
    )
    if truncate_prompt_tokens is not None:
        raise RuntimeError(
            "Rollout-logprob reuse requires truncate_prompt_tokens to be unset."
        )

    for field in (
        "allowed_token_ids",
        "bad_words",
        "extra_args",
        "guided_decoding",
        "logit_bias",
        "logprob_token_ids",
        "logits_processors",
        "prompt_logprobs",
        "repetition_detection",
        "structured_outputs",
        "thinking_token_budget",
    ):
        value = _sampling_value(sampling_params, field, None)
        if not _is_unset_or_empty_collection(value):
            raise RuntimeError(
                f"Rollout-logprob reuse requires {field} to be empty; got {value!r}."
            )

    if bool(_sampling_value(sampling_params, "flat_logprobs", False)):
        raise RuntimeError(
            "Rollout-logprob reuse requires flat_logprobs=false."
        )

    return {
        "mode": mode,
        "logprobs": logprobs,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": top_k,
        "min_p": 0.0,
    }


def validate_official_vllm_engine_logprob_contract(
    inference_engine: Any,
    mode: str,
) -> dict[str, Any]:
    """Require the effective engine-level logprob mode to be raw policy logits."""

    if not rollout_logprob_reuse_enabled(mode):
        return {"mode": mode}

    candidates = []
    direct_config = getattr(inference_engine, "model_config", None)
    if direct_config is not None:
        candidates.append(("model_config", direct_config))
    llm_engine = getattr(inference_engine, "llm_engine", None)
    nested_config = getattr(llm_engine, "model_config", None)
    if nested_config is not None:
        candidates.append(("llm_engine.model_config", nested_config))
    if not candidates:
        raise RuntimeError(
            "Rollout-logprob reuse could not read the official vLLM "
            "ModelConfig from the constructed inference engine."
        )

    readback = {}
    for path, model_config in candidates:
        value = getattr(model_config, "logprobs_mode", None)
        readback[path] = value
        if value != "raw_logprobs":
            raise RuntimeError(
                "Rollout-logprob reuse requires effective vLLM engine "
                "logprobs_mode='raw_logprobs'; "
                f"{path} reported {value!r}."
            )
    return {
        "mode": mode,
        "logprobs_mode": "raw_logprobs",
        "readback_paths": sorted(readback),
    }


def _logprob_value(entry: Any, *, position: int, token_id: int) -> float:
    if isinstance(entry, Mapping):
        if "logprob" not in entry:
            raise RuntimeError(
                f"Sampled logprob entry at position {position} token {token_id} "
                "has no 'logprob' field."
            )
        raw_value = entry["logprob"]
    elif hasattr(entry, "logprob"):
        raw_value = entry.logprob
    else:
        raw_value = entry
    if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
        raise RuntimeError(
            f"Sampled logprob at position {position} token {token_id} is not "
            f"numeric: {raw_value!r}."
        )
    value = float(raw_value)
    if not math.isfinite(value):
        raise RuntimeError(
            f"Sampled logprob at position {position} token {token_id} is not finite."
        )
    return value


def extract_sampled_token_logprobs(
    token_ids: Sequence[int],
    logprob_rows: Sequence[Mapping[int, Any]] | None,
) -> list[float]:
    """Extract the probability of each exact sampled token from vLLM output."""

    if not isinstance(token_ids, list) or any(type(token_id) is not int for token_id in token_ids):
        raise RuntimeError("Sampled token IDs must be a raw list[int].")
    if not isinstance(logprob_rows, list):
        raise RuntimeError("Official vLLM output is missing per-token logprob rows.")
    if len(logprob_rows) != len(token_ids):
        raise RuntimeError(
            "Official vLLM token/logprob length mismatch: "
            f"tokens={len(token_ids)} logprob_rows={len(logprob_rows)}."
        )

    sampled_logprobs = []
    for position, (token_id, row) in enumerate(zip(token_ids, logprob_rows)):
        if not isinstance(row, Mapping):
            raise RuntimeError(
                f"Official vLLM logprob row {position} is not a mapping."
            )
        if token_id not in row:
            raise RuntimeError(
                "Official vLLM logprob row is missing the sampled token: "
                f"position={position} token_id={token_id}."
            )
        sampled_logprobs.append(
            _logprob_value(row[token_id], position=position, token_id=token_id)
        )
    return sampled_logprobs


def build_sampled_token_logprob_evidence(
    *,
    prompt_token_ids: Sequence[int],
    response_token_ids: Sequence[int],
    logprob_rows: Sequence[Mapping[int, Any]] | None,
) -> dict[str, Any]:
    """Bind sampled logprobs to the exact conditioning and response tokens."""

    if not isinstance(prompt_token_ids, list) or any(
        type(token_id) is not int for token_id in prompt_token_ids
    ):
        raise RuntimeError("Generation prompt token IDs must be a raw list[int].")
    sampled_logprobs = extract_sampled_token_logprobs(
        response_token_ids, logprob_rows
    )
    return {
        _EVIDENCE_PROMPT_TOKEN_IDS: list(prompt_token_ids),
        _EVIDENCE_RESPONSE_TOKEN_IDS: list(response_token_ids),
        _EVIDENCE_LOG_PROBS: sampled_logprobs,
    }


def build_official_vllm_sampled_logprob_evidence(
    *,
    request_output: Any,
    expected_prompt_token_ids: Sequence[int],
    normalized_response_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Bind one official RequestOutput to the exact PPO prompt and response."""

    if not isinstance(expected_prompt_token_ids, list) or any(
        type(token_id) is not int for token_id in expected_prompt_token_ids
    ):
        raise RuntimeError("Expected prompt token IDs must be a raw list[int].")
    backend_prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    if not isinstance(backend_prompt_token_ids, list) or any(
        type(token_id) is not int for token_id in backend_prompt_token_ids
    ):
        raise RuntimeError("Official vLLM output is missing raw prompt token IDs.")
    if backend_prompt_token_ids != expected_prompt_token_ids:
        raise RuntimeError("Official vLLM conditioned on different prompt tokens.")

    candidates = getattr(request_output, "outputs", None)
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise RuntimeError(
            "Rollout-logprob reuse requires exactly one official vLLM candidate."
        )
    candidate = candidates[0]
    candidate_token_ids = getattr(candidate, "token_ids", None)
    if not isinstance(candidate_token_ids, list) or any(
        type(token_id) is not int for token_id in candidate_token_ids
    ):
        raise RuntimeError(
            "Official vLLM candidate token IDs must be a raw list[int]."
        )
    if candidate_token_ids != normalized_response_token_ids:
        raise RuntimeError(
            "Official vLLM response tokens changed before PPO evidence binding."
        )
    return build_sampled_token_logprob_evidence(
        prompt_token_ids=backend_prompt_token_ids,
        response_token_ids=candidate_token_ids,
        logprob_rows=getattr(candidate, "logprobs", None),
    )


def validate_rollout_logprob_rows(
    prompt_token_rows: Sequence[Sequence[int]],
    response_token_rows: Sequence[Sequence[int]],
    rollout_logprob_evidence_rows: Sequence[Mapping[str, Any]],
) -> list[list[float]]:
    row_counts = {
        "prompts": len(prompt_token_rows),
        "responses": len(response_token_rows),
        "evidence": len(rollout_logprob_evidence_rows),
    }
    if len(set(row_counts.values())) != 1:
        raise RuntimeError(
            f"Rollout prompt/response/logprob row-count mismatch: {row_counts}."
        )
    normalized = []
    for row_index, (prompt_row, response_row, evidence) in enumerate(
        zip(
            prompt_token_rows,
            response_token_rows,
            rollout_logprob_evidence_rows,
        )
    ):
        if not isinstance(evidence, Mapping):
            raise RuntimeError(
                f"Rollout logprob evidence row {row_index} must be a mapping."
            )
        evidence_prompt = evidence.get(_EVIDENCE_PROMPT_TOKEN_IDS)
        evidence_response = evidence.get(_EVIDENCE_RESPONSE_TOKEN_IDS)
        if evidence_prompt != list(prompt_row):
            raise RuntimeError(
                "Rollout logprob prompt binding mismatch: "
                f"row={row_index}."
            )
        if evidence_response != list(response_row):
            raise RuntimeError(
                "Rollout logprob response binding mismatch: "
                f"row={row_index}."
            )
        logprob_row = evidence.get(_EVIDENCE_LOG_PROBS)
        if not isinstance(logprob_row, list):
            raise RuntimeError(
                f"Rollout logprobs row {row_index} must be a list."
            )
        if len(response_row) != len(logprob_row):
            raise RuntimeError(
                "Rollout response/logprob token-count mismatch: "
                f"row={row_index} tokens={len(response_row)} "
                f"logprobs={len(logprob_row)}."
            )
        normalized_row = []
        for position, value in enumerate(logprob_row):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RuntimeError(
                    f"Rollout logprob row={row_index} position={position} is not numeric."
                )
            value = float(value)
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Rollout logprob row={row_index} position={position} is not finite."
                )
            normalized_row.append(value)
        normalized.append(normalized_row)
    return normalized
