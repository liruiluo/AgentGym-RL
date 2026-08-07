from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Any, Mapping, Sequence


CONTINUOUS_AGENT_SCHEMA_V1 = "agentmemory_continuous_agent_step_v1"
CONTINUOUS_AGENT_CONTEXT_POLICY_V1 = "policy_authored_compaction"
CONTINUOUS_AGENT_OBSERVATION_SCHEMA_V1 = (
    "agentmemory_continuous_observation_v1"
)
CONTINUOUS_AGENT_HORIZON_SCHEMA_V1 = "agentmemory_continuous_horizon_v1"
ENVIRONMENT_ACTION_ROW = "environment_action"
COMPACTION_ROW = "compaction"
CONTINUOUS_AGENT_ROW_KINDS = frozenset(
    {ENVIRONMENT_ACTION_ROW, COMPACTION_ROW}
)

POLICY_COMPACTION_REQUEST = (
    "The conversation is nearing its context limit. Write the continuation "
    "state you want to retain after the earlier interaction is removed. Your "
    "response will be preserved verbatim and will not be sent to the "
    "environment. Include only information you choose to carry forward."
)
POLICY_CONTINUATION_MARKER = "Continue the same task in the unchanged workspace."


class ContinuousAgentV1Error(ValueError):
    pass


def build_policy_generation_loss_masks(
    prompt_token_ids: Sequence[int],
    response_token_ids: Sequence[int],
) -> tuple[list[int], list[int], list[int]]:
    """Mask only sampled policy tokens into the actor and critic losses."""

    prompt_length = len(prompt_token_ids)
    response_length = len(response_token_ids)
    prompt_loss_mask = [0] * prompt_length
    response_loss_mask = [1] * response_length
    return (
        prompt_loss_mask + response_loss_mask,
        prompt_loss_mask,
        response_loss_mask,
    )


def continuous_prompt_capacity(
    *,
    max_prompt_tokens: int,
    max_model_tokens: int,
    max_response_tokens: int,
) -> int:
    """Return the largest prompt that can be sampled and packed unchanged."""

    values = {
        "max_prompt_tokens": max_prompt_tokens,
        "max_model_tokens": max_model_tokens,
        "max_response_tokens": max_response_tokens,
    }
    for name, value in values.items():
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ContinuousAgentV1Error(f"{name} must be a positive integer")
    capacity = min(
        int(max_prompt_tokens),
        int(max_model_tokens) - int(max_response_tokens),
    )
    if capacity <= 0:
        raise ContinuousAgentV1Error(
            "model capacity must exceed the per-action response budget"
        )
    return capacity


def build_continuous_runtime_capacity_readback(
    *,
    configured_max_prompt_tokens: int,
    configured_max_model_tokens: int,
    configured_max_response_tokens: int,
    engine_max_model_tokens: int,
    sampling_max_response_tokens: int,
    max_observation_tokens: int,
) -> dict[str, int]:
    """Bind continuous-context accounting to the effective runtime limits."""

    configured_capacity = continuous_prompt_capacity(
        max_prompt_tokens=configured_max_prompt_tokens,
        max_model_tokens=configured_max_model_tokens,
        max_response_tokens=configured_max_response_tokens,
    )
    effective_capacity = continuous_prompt_capacity(
        max_prompt_tokens=configured_max_prompt_tokens,
        max_model_tokens=engine_max_model_tokens,
        max_response_tokens=sampling_max_response_tokens,
    )
    if int(engine_max_model_tokens) != int(configured_max_model_tokens):
        raise ContinuousAgentV1Error(
            "vLLM max_model_len readback drifted from the rollout config: "
            f"engine={engine_max_model_tokens} "
            f"configured={configured_max_model_tokens}"
        )
    if int(sampling_max_response_tokens) != int(configured_max_response_tokens):
        raise ContinuousAgentV1Error(
            "sampling max_tokens readback drifted from the packed response "
            f"width: sampling={sampling_max_response_tokens} "
            f"configured={configured_max_response_tokens}"
        )
    if configured_capacity != effective_capacity:
        raise ContinuousAgentV1Error(
            "continuous prompt capacity drifted across runtime layers: "
            f"configured={configured_capacity} effective={effective_capacity}"
        )
    if (
        isinstance(max_observation_tokens, bool)
        or int(max_observation_tokens) != max_observation_tokens
        or int(max_observation_tokens) <= 0
    ):
        raise ContinuousAgentV1Error(
            "max_observation_tokens must be a positive integer"
        )
    return {
        "configured_max_prompt_tokens": int(configured_max_prompt_tokens),
        "configured_max_model_tokens": int(configured_max_model_tokens),
        "configured_max_response_tokens": int(configured_max_response_tokens),
        "engine_max_model_tokens": int(engine_max_model_tokens),
        "sampling_max_response_tokens": int(sampling_max_response_tokens),
        "max_observation_tokens": int(max_observation_tokens),
        "effective_prompt_capacity": effective_capacity,
    }


def should_request_policy_compaction(
    *,
    action_prompt_token_count: int,
    compaction_prompt_token_count: int,
    max_prompt_tokens: int,
    max_model_tokens: int,
    max_response_tokens: int,
    max_observation_tokens: int,
    action_observation_envelope_tokens: int,
) -> bool:
    """Trigger while the compaction request can still be sampled without truncation.

    The reserve covers the next sampled action, the largest policy-visible
    observation, the chat-template role envelope, and the neutral compaction
    request. This is capacity accounting rather than a learned-frequency or
    task-specific compaction-count knob.
    """

    capacity = continuous_prompt_capacity(
        max_prompt_tokens=max_prompt_tokens,
        max_model_tokens=max_model_tokens,
        max_response_tokens=max_response_tokens,
    )
    counts = {
        "action_prompt_token_count": action_prompt_token_count,
        "compaction_prompt_token_count": compaction_prompt_token_count,
        "max_observation_tokens": max_observation_tokens,
    }
    for name, value in counts.items():
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ContinuousAgentV1Error(f"{name} must be a positive integer")
    action_count = int(action_prompt_token_count)
    compaction_count = int(compaction_prompt_token_count)
    envelope_count = int(action_observation_envelope_tokens)
    if (
        isinstance(action_observation_envelope_tokens, bool)
        or envelope_count != action_observation_envelope_tokens
        or envelope_count < 0
    ):
        raise ContinuousAgentV1Error(
            "action_observation_envelope_tokens must be a non-negative integer"
        )
    if compaction_count <= action_count:
        raise ContinuousAgentV1Error(
            "compaction request must extend the current action prompt"
        )
    if action_count > capacity or compaction_count > capacity:
        raise ContinuousAgentV1Error(
            "continuous history reached the prompt cap before a trainable "
            "compaction could be sampled"
        )
    compaction_request_tokens = compaction_count - action_count
    projected_next_compaction_prompt = (
        action_count
        + int(max_response_tokens)
        + int(max_observation_tokens)
        + envelope_count
        + compaction_request_tokens
    )
    return projected_next_compaction_prompt >= capacity


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_digest(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def build_compaction_evidence_v1(
    *,
    pre_request_action_prompt_token_ids: Sequence[int],
    pre_compaction_prompt_token_ids: Sequence[int],
    immutable_framing_token_ids: Sequence[int],
    summary_token_ids: Sequence[int],
    post_compaction_prompt_token_ids: Sequence[int],
    workspace_continuity_id: str | int,
) -> dict[str, Any]:
    """Build self-verifying evidence from the exact prompts used by rollout."""

    pre_request_action_prompt = [
        int(token_id) for token_id in pre_request_action_prompt_token_ids
    ]
    pre_prompt = [int(token_id) for token_id in pre_compaction_prompt_token_ids]
    immutable_framing = [int(token_id) for token_id in immutable_framing_token_ids]
    summary = [int(token_id) for token_id in summary_token_ids]
    post_prompt = [int(token_id) for token_id in post_compaction_prompt_token_ids]
    for name, values in (
        ("pre_request_action_prompt_token_ids", pre_request_action_prompt),
        ("pre_compaction_prompt_token_ids", pre_prompt),
        ("immutable_framing_token_ids", immutable_framing),
        ("summary_token_ids", summary),
        ("post_compaction_prompt_token_ids", post_prompt),
    ):
        if not values:
            raise ContinuousAgentV1Error(f"{name} must not be empty")
    return {
        "request_text_sha256": text_sha256(POLICY_COMPACTION_REQUEST),
        "continuation_marker_sha256": text_sha256(POLICY_CONTINUATION_MARKER),
        "pre_request_action_prompt_token_ids": pre_request_action_prompt,
        "pre_request_action_prompt_length": len(pre_request_action_prompt),
        "pre_request_action_prompt_digest": token_digest(
            pre_request_action_prompt
        ),
        "immutable_framing_token_ids": immutable_framing,
        "immutable_framing_length": len(immutable_framing),
        "immutable_framing_digest": token_digest(immutable_framing),
        "pre_compaction_prompt_length": len(pre_prompt),
        "pre_compaction_prompt_digest": token_digest(pre_prompt),
        "summary_token_count": len(summary),
        "summary_token_digest": token_digest(summary),
        "post_compaction_prompt_token_ids": post_prompt,
        "post_compaction_prompt_length": len(post_prompt),
        "post_compaction_prompt_digest": token_digest(post_prompt),
        "workspace_continuity_id": str(workspace_continuity_id),
    }


def build_observation_evidence_v1(
    *,
    full_text: str,
    full_token_ids: Sequence[int],
    policy_visible_text: str,
    policy_visible_token_ids: Sequence[int],
    post_observation_prompt_token_ids: Sequence[int],
    max_observation_tokens: int,
    truncated: bool,
    head_token_count: int,
    tail_token_count: int,
    truncation_marker: str | None,
) -> dict[str, Any]:
    """Bind a bounded policy observation to its complete server result."""

    if not isinstance(full_text, str) or not isinstance(policy_visible_text, str):
        raise ContinuousAgentV1Error("observation text must be text")
    full_ids = [int(token_id) for token_id in full_token_ids]
    visible_ids = [int(token_id) for token_id in policy_visible_token_ids]
    post_prompt_ids = [
        int(token_id) for token_id in post_observation_prompt_token_ids
    ]
    if not full_ids or not visible_ids or not post_prompt_ids:
        raise ContinuousAgentV1Error(
            "observation token evidence and post-observation prompt must not be empty"
        )
    maximum = int(max_observation_tokens)
    if (
        isinstance(max_observation_tokens, bool)
        or maximum != max_observation_tokens
        or maximum <= 0
    ):
        raise ContinuousAgentV1Error(
            "max_observation_tokens must be a positive integer"
        )
    head_count = int(head_token_count)
    tail_count = int(tail_token_count)
    if (
        isinstance(head_token_count, bool)
        or isinstance(tail_token_count, bool)
        or head_count != head_token_count
        or tail_count != tail_token_count
        or min(head_count, tail_count) < 0
    ):
        raise ContinuousAgentV1Error(
            "observation head/tail counts must be non-negative integers"
        )
    if len(visible_ids) > maximum:
        raise ContinuousAgentV1Error(
            "policy-visible observation exceeds its token bound"
        )
    if type(truncated) is not bool:
        raise ContinuousAgentV1Error("observation truncated flag must be boolean")
    if truncated:
        if len(full_ids) <= maximum:
            raise ContinuousAgentV1Error(
                "observation cannot be marked truncated below its token bound"
            )
        if not isinstance(truncation_marker, str) or not truncation_marker:
            raise ContinuousAgentV1Error(
                "truncated observation requires an explicit marker"
            )
        if truncation_marker not in policy_visible_text:
            raise ContinuousAgentV1Error(
                "truncated observation marker is absent from policy-visible text"
            )
        if head_count + tail_count >= len(full_ids):
            raise ContinuousAgentV1Error(
                "truncated observation must omit at least one original token"
            )
    else:
        if truncation_marker is not None:
            raise ContinuousAgentV1Error(
                "untruncated observation must not carry a truncation marker"
            )
        if full_text != policy_visible_text or full_ids != visible_ids:
            raise ContinuousAgentV1Error(
                "untruncated observation must preserve exact text and tokens"
            )
        if head_count != len(full_ids) or tail_count != 0:
            raise ContinuousAgentV1Error(
                "untruncated observation head/tail evidence is inconsistent"
            )
    return {
        "schema_version": CONTINUOUS_AGENT_OBSERVATION_SCHEMA_V1,
        "full_text_sha256": text_sha256(full_text),
        "full_text_utf8_bytes": len(full_text.encode("utf-8")),
        "full_token_count": len(full_ids),
        "full_token_digest": token_digest(full_ids),
        "policy_visible_text_sha256": text_sha256(policy_visible_text),
        "policy_visible_text_utf8_bytes": len(
            policy_visible_text.encode("utf-8")
        ),
        "policy_visible_token_ids": visible_ids,
        "policy_visible_token_count": len(visible_ids),
        "policy_visible_token_digest": token_digest(visible_ids),
        "post_observation_prompt_token_ids": post_prompt_ids,
        "post_observation_prompt_length": len(post_prompt_ids),
        "post_observation_prompt_digest": token_digest(post_prompt_ids),
        "max_observation_tokens": maximum,
        "truncated": truncated,
        "head_token_count": head_count,
        "tail_token_count": tail_count,
        "truncation_marker": truncation_marker,
        "truncation_marker_sha256": (
            None if truncation_marker is None else text_sha256(truncation_marker)
        ),
    }


def build_horizon_evidence_v1(
    *,
    environment_id: int,
    environment_step: int,
    native_environment_call_count: int,
    policy_step_reward: float,
    horizon_reward: float,
    environment_result: str,
) -> dict[str, Any]:
    """Record hidden grading caused by exhausting the unified policy horizon."""

    combined_reward = float(policy_step_reward) + float(horizon_reward)
    if not all(
        math.isfinite(value)
        for value in (
            float(policy_step_reward),
            float(horizon_reward),
            combined_reward,
        )
    ):
        raise ContinuousAgentV1Error("horizon rewards must be finite")
    return {
        "schema_version": CONTINUOUS_AGENT_HORIZON_SCHEMA_V1,
        "environment_id": int(environment_id),
        "environment_step": int(environment_step),
        "native_environment_call_count": int(native_environment_call_count),
        "policy_step_reward": float(policy_step_reward),
        "horizon_reward": float(horizon_reward),
        "combined_reward": combined_reward,
        "environment_result": str(environment_result),
        "environment_result_sha256": text_sha256(str(environment_result)),
        "done": True,
    }


def extract_sampled_token_logprobs(
    token_ids: Sequence[int], raw_logprobs: Any
) -> list[float]:
    """Read the sampled token's official-vLLM logprob at every position."""

    sampled_ids = [int(token_id) for token_id in token_ids]
    if not isinstance(raw_logprobs, list) or len(raw_logprobs) != len(sampled_ids):
        raise ContinuousAgentV1Error(
            "official-vLLM sampled logprob rows do not align with token IDs"
        )
    sampled: list[float] = []
    for position, (token_id, candidates) in enumerate(
        zip(sampled_ids, raw_logprobs)
    ):
        if not isinstance(candidates, Mapping):
            raise ContinuousAgentV1Error(
                f"sampled logprob position {position} is not a mapping"
            )
        value = candidates.get(token_id)
        if value is None:
            value = candidates.get(str(token_id))
        if value is None:
            raise ContinuousAgentV1Error(
                f"sampled token {token_id} is absent from logprobs at position {position}"
            )
        if hasattr(value, "logprob"):
            value = value.logprob
        elif isinstance(value, Mapping):
            value = value.get("logprob")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ContinuousAgentV1Error(
                f"sampled logprob at position {position} is not numeric"
            ) from exc
        if not math.isfinite(numeric):
            raise ContinuousAgentV1Error(
                f"sampled logprob at position {position} is not finite"
            )
        sampled.append(numeric)
    return sampled


def validate_neutral_compaction_text() -> None:
    combined = f"{POLICY_COMPACTION_REQUEST}\n{POLICY_CONTINUATION_MARKER}".lower()
    forbidden = (
        "memory.md",
        ".agent_memory",
        "todo.md",
        "workspace file",
        "current progress is",
        "next step is",
        "gold patch",
        "test answer",
    )
    leaked = [value for value in forbidden if value in combined]
    if leaked:
        raise ContinuousAgentV1Error(
            f"compaction control text contains semantic/path hints: {leaked}"
        )


def _raw_int_list(value: Any, *, name: str, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ContinuousAgentV1Error(f"{name} must be a raw list[int]")
    if not allow_empty and not value:
        raise ContinuousAgentV1Error(f"{name} must not be empty")
    return list(value)


def _finite_float_list(value: Any, *, name: str, expected: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise ContinuousAgentV1Error(
            f"{name} must contain exactly {expected} values"
        )
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ContinuousAgentV1Error(f"{name} must be numeric") from exc
    if not all(math.isfinite(item) for item in values):
        raise ContinuousAgentV1Error(f"{name} must be finite")
    return values


def build_continuous_agent_step_v1(
    *,
    row_kind: str,
    task_name: str,
    content: str,
    score: float,
    item_id: str,
    data_idx: int,
    parent_index: int,
    parent_group_uid: str,
    replica_index: int,
    trajectory_uid: str,
    exact_state_uid: str,
    prompt_token_ids: Sequence[int],
    response_token_ids: Sequence[int],
    sampled_token_logprobs: Sequence[float],
    generation_record: Mapping[str, Any],
    environment_id: int,
    environment_step_before: int,
    environment_step_after: int,
    native_environment_call_count_before: int,
    native_environment_call_count_after: int,
    context_epoch_before: int,
    context_epoch_after: int,
    done: bool,
    environment_result: str,
    compaction_evidence: Mapping[str, Any] | None = None,
    observation_evidence: Mapping[str, Any] | None = None,
    horizon_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one trainable generation row for a continuous native agent."""

    validate_neutral_compaction_text()
    prompt_ids = [int(token_id) for token_id in prompt_token_ids]
    response_ids = [int(token_id) for token_id in response_token_ids]
    logprobs = [float(value) for value in sampled_token_logprobs]
    record = {
        "schema_version": CONTINUOUS_AGENT_SCHEMA_V1,
        "row_kind": str(row_kind),
        "task_name": str(task_name),
        "content": str(content),
        "score": float(score),
        "item_id": str(item_id),
        "data_idx": int(data_idx),
        "parent_index": int(parent_index),
        "parent_group_uid": str(parent_group_uid),
        "replica_index": int(replica_index),
        "trajectory_uid": str(trajectory_uid),
        "exact_state_uid": str(exact_state_uid),
        "prompt_token_ids": prompt_ids,
        "response_token_ids": response_ids,
        "sampled_token_logprobs": logprobs,
        "response_token_count": int(generation_record["response_token_count"]),
        "max_response_tokens": int(generation_record["max_response_tokens"]),
        "finish_reason": str(generation_record["finish_reason"]),
        "finish_reason_source": str(generation_record["finish_reason_source"]),
        "stop_reason": generation_record.get("stop_reason"),
        "generation_backend_source": str(generation_record["backend_source"]),
        "generation_eos_token_ids": [
            int(token_id)
            for token_id in generation_record["configured_eos_token_ids"]
        ],
        "tokenizer_primary_eos_token_id": generation_record[
            "primary_eos_token_id"
        ],
        "tokenizer_pad_token_id": generation_record["tokenizer_pad_token_id"],
        "generation_token_ids_are_exact": bool(
            generation_record["token_ids_are_exact"]
        ),
        "backend_token_ids_are_exact": bool(
            generation_record["backend_token_ids_are_exact"]
        ),
        "truncated": bool(generation_record["truncated"]),
        "prompt_history_policy": CONTINUOUS_AGENT_CONTEXT_POLICY_V1,
        "environment_id": int(environment_id),
        "environment_step_before": int(environment_step_before),
        "environment_step_after": int(environment_step_after),
        "native_environment_call_count_before": int(
            native_environment_call_count_before
        ),
        "native_environment_call_count_after": int(
            native_environment_call_count_after
        ),
        "environment_action_dispatched": row_kind == ENVIRONMENT_ACTION_ROW,
        "context_epoch_before": int(context_epoch_before),
        "context_epoch_after": int(context_epoch_after),
        "done": bool(done),
        "environment_result": str(environment_result),
        "compaction": (
            None if compaction_evidence is None else deepcopy(dict(compaction_evidence))
        ),
        "observation": (
            None
            if observation_evidence is None
            else deepcopy(dict(observation_evidence))
        ),
        "horizon_finalization": (
            None if horizon_evidence is None else deepcopy(dict(horizon_evidence))
        ),
    }
    validate_continuous_agent_step_v1(record)
    return record


def validate_continuous_agent_step_v1(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "row_kind",
        "task_name",
        "content",
        "score",
        "item_id",
        "data_idx",
        "parent_index",
        "parent_group_uid",
        "replica_index",
        "trajectory_uid",
        "exact_state_uid",
        "prompt_token_ids",
        "response_token_ids",
        "sampled_token_logprobs",
        "response_token_count",
        "max_response_tokens",
        "finish_reason",
        "finish_reason_source",
        "stop_reason",
        "generation_backend_source",
        "generation_eos_token_ids",
        "tokenizer_primary_eos_token_id",
        "tokenizer_pad_token_id",
        "generation_token_ids_are_exact",
        "backend_token_ids_are_exact",
        "truncated",
        "prompt_history_policy",
        "environment_id",
        "environment_step_before",
        "environment_step_after",
        "native_environment_call_count_before",
        "native_environment_call_count_after",
        "environment_action_dispatched",
        "context_epoch_before",
        "context_epoch_after",
        "done",
        "environment_result",
        "compaction",
        "observation",
        "horizon_finalization",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ContinuousAgentV1Error(
            "continuous-agent row is missing fields: " + ", ".join(missing)
        )
    if record["schema_version"] != CONTINUOUS_AGENT_SCHEMA_V1:
        raise ContinuousAgentV1Error("unsupported continuous-agent schema")
    row_kind = record["row_kind"]
    if row_kind not in CONTINUOUS_AGENT_ROW_KINDS:
        raise ContinuousAgentV1Error(f"unsupported row_kind: {row_kind!r}")
    if record["prompt_history_policy"] != CONTINUOUS_AGENT_CONTEXT_POLICY_V1:
        raise ContinuousAgentV1Error("continuous-agent context policy mismatch")
    if not isinstance(record["content"], str):
        raise ContinuousAgentV1Error("content must be text")
    prompt_ids = _raw_int_list(record["prompt_token_ids"], name="prompt_token_ids")
    response_ids = _raw_int_list(
        record["response_token_ids"], name="response_token_ids"
    )
    logprobs = _finite_float_list(
        record["sampled_token_logprobs"],
        name="sampled_token_logprobs",
        expected=len(response_ids),
    )
    if len(response_ids) != int(record["response_token_count"]):
        raise ContinuousAgentV1Error("response token count mismatch")
    if response_ids != list(record["response_token_ids"]) or not logprobs:
        raise ContinuousAgentV1Error("sampled response evidence is empty")
    if not prompt_ids:
        raise ContinuousAgentV1Error("generation prompt is empty")
    if not math.isfinite(float(record["score"])):
        raise ContinuousAgentV1Error("score must be finite")
    exact_state_uid = str(record["exact_state_uid"])
    if ":statev1:" not in exact_state_uid or exact_state_uid.rsplit(
        ":statev1:", 1
    )[1] != token_digest(prompt_ids):
        raise ContinuousAgentV1Error(
            "exact state identity must bind the generation prompt digest"
        )
    before_step = int(record["environment_step_before"])
    after_step = int(record["environment_step_after"])
    before_native_call = int(record["native_environment_call_count_before"])
    after_native_call = int(record["native_environment_call_count_after"])
    before_epoch = int(record["context_epoch_before"])
    after_epoch = int(record["context_epoch_after"])
    if min(
        before_step,
        after_step,
        before_native_call,
        after_native_call,
        before_epoch,
        after_epoch,
    ) < 0:
        raise ContinuousAgentV1Error("step and context epoch must be non-negative")
    if after_step != before_step + 1:
        raise ContinuousAgentV1Error(
            "every policy action, including compaction, must consume exactly one "
            "environment step"
        )
    if before_native_call > before_step or after_native_call > after_step:
        raise ContinuousAgentV1Error(
            "native environment calls cannot exceed unified policy steps"
        )

    dispatched = record["environment_action_dispatched"]
    if type(dispatched) is not bool:
        raise ContinuousAgentV1Error("environment_action_dispatched must be boolean")
    horizon_evidence = record["horizon_finalization"]
    if horizon_evidence is not None:
        _validate_horizon_evidence_v1(record, horizon_evidence)

    if row_kind == ENVIRONMENT_ACTION_ROW:
        if not dispatched or after_native_call != before_native_call + 1:
            raise ContinuousAgentV1Error(
                "environment action must dispatch exactly one native environment call"
            )
        if after_epoch != before_epoch or record["compaction"] is not None:
            raise ContinuousAgentV1Error(
                "environment action must not change the compaction epoch"
            )
        observation_evidence = record["observation"]
        if not isinstance(observation_evidence, Mapping):
            raise ContinuousAgentV1Error(
                "environment action requires bounded observation evidence"
            )
        _validate_observation_evidence_v1(record, observation_evidence)
    else:
        if dispatched or after_native_call != before_native_call:
            raise ContinuousAgentV1Error(
                "compaction must consume a step without dispatching a native "
                "environment call"
            )
        if after_epoch != before_epoch + 1:
            raise ContinuousAgentV1Error(
                "compaction must advance exactly one context epoch"
            )
        evidence = record["compaction"]
        if not isinstance(evidence, Mapping):
            raise ContinuousAgentV1Error("compaction row requires evidence")
        required_evidence = {
            "request_text_sha256",
            "continuation_marker_sha256",
            "pre_request_action_prompt_token_ids",
            "pre_request_action_prompt_length",
            "pre_request_action_prompt_digest",
            "immutable_framing_token_ids",
            "immutable_framing_length",
            "immutable_framing_digest",
            "pre_compaction_prompt_length",
            "pre_compaction_prompt_digest",
            "summary_token_count",
            "summary_token_digest",
            "post_compaction_prompt_token_ids",
            "post_compaction_prompt_length",
            "post_compaction_prompt_digest",
            "workspace_continuity_id",
        }
        evidence_missing = sorted(required_evidence - set(evidence))
        if evidence_missing:
            raise ContinuousAgentV1Error(
                "compaction evidence is missing fields: "
                + ", ".join(evidence_missing)
            )
        if evidence["request_text_sha256"] != text_sha256(
            POLICY_COMPACTION_REQUEST
        ):
            raise ContinuousAgentV1Error("compaction request text drifted")
        if evidence["continuation_marker_sha256"] != text_sha256(
            POLICY_CONTINUATION_MARKER
        ):
            raise ContinuousAgentV1Error("continuation marker drifted")
        if str(evidence["workspace_continuity_id"]) != str(record["environment_id"]):
            raise ContinuousAgentV1Error("workspace continuity identity drifted")
        if record["observation"] is not None:
            raise ContinuousAgentV1Error(
                "compaction must not fabricate policy-visible observation evidence"
            )
        if record["environment_result"]:
            raise ContinuousAgentV1Error(
                "compaction must not fabricate a native environment result"
            )
        pre_request_action_ids = _raw_int_list(
            evidence["pre_request_action_prompt_token_ids"],
            name="pre_request_action_prompt_token_ids",
        )
        if int(evidence["pre_request_action_prompt_length"]) != len(
            pre_request_action_ids
        ):
            raise ContinuousAgentV1Error(
                "pre-request action prompt length mismatch"
            )
        if evidence["pre_request_action_prompt_digest"] != token_digest(
            pre_request_action_ids
        ):
            raise ContinuousAgentV1Error(
                "pre-request action prompt digest mismatch"
            )
        if len(pre_request_action_ids) >= len(prompt_ids):
            raise ContinuousAgentV1Error(
                "compaction request must extend its action prompt"
            )
        if int(evidence["pre_compaction_prompt_length"]) != len(prompt_ids):
            raise ContinuousAgentV1Error("pre-compaction prompt length mismatch")
        if evidence["pre_compaction_prompt_digest"] != token_digest(prompt_ids):
            raise ContinuousAgentV1Error("pre-compaction prompt digest mismatch")
        immutable_framing_ids = _raw_int_list(
            evidence["immutable_framing_token_ids"],
            name="immutable_framing_token_ids",
        )
        if int(evidence["immutable_framing_length"]) != len(
            immutable_framing_ids
        ):
            raise ContinuousAgentV1Error("immutable framing length mismatch")
        if evidence["immutable_framing_digest"] != token_digest(
            immutable_framing_ids
        ):
            raise ContinuousAgentV1Error("immutable framing digest mismatch")
        if int(evidence["summary_token_count"]) != len(response_ids):
            raise ContinuousAgentV1Error("compaction summary token count mismatch")
        if evidence["summary_token_digest"] != token_digest(response_ids):
            raise ContinuousAgentV1Error("compaction summary token digest mismatch")
        post_prompt_ids = _raw_int_list(
            evidence["post_compaction_prompt_token_ids"],
            name="post_compaction_prompt_token_ids",
        )
        if int(evidence["post_compaction_prompt_length"]) != len(post_prompt_ids):
            raise ContinuousAgentV1Error("post-compaction prompt length mismatch")
        if evidence["post_compaction_prompt_digest"] != token_digest(
            post_prompt_ids
        ):
            raise ContinuousAgentV1Error("post-compaction prompt digest mismatch")
        if horizon_evidence is None and (
            bool(record["done"]) or float(record["score"]) != 0.0
        ):
            raise ContinuousAgentV1Error(
                "nonterminal compaction cannot receive immediate environment reward"
            )


def validate_continuous_packed_rows_v1(
    records: Sequence[Mapping[str, Any]],
    *,
    packed_prompt_token_ids: Sequence[Sequence[int]],
    packed_response_token_ids: Sequence[Sequence[int]],
) -> None:
    """Prove that PPO receives the exact sampled continuous-agent tokens."""

    lengths = {
        "records": len(records),
        "packed_prompt_token_ids": len(packed_prompt_token_ids),
        "packed_response_token_ids": len(packed_response_token_ids),
    }
    if len(set(lengths.values())) != 1:
        raise ContinuousAgentV1Error(
            f"continuous packed-row length mismatch: {lengths}"
        )
    for index, record in enumerate(records):
        validate_continuous_agent_step_v1(record)
        packed_prompt = [int(value) for value in packed_prompt_token_ids[index]]
        packed_response = [
            int(value) for value in packed_response_token_ids[index]
        ]
        if packed_prompt != list(record["prompt_token_ids"]):
            raise ContinuousAgentV1Error(
                f"continuous packed prompt drifted at row {index}"
            )
        if packed_response != list(record["response_token_ids"]):
            raise ContinuousAgentV1Error(
                f"continuous packed response drifted at row {index}"
            )


def _validate_observation_evidence_v1(
    record: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    required = {
        "schema_version",
        "full_text_sha256",
        "full_text_utf8_bytes",
        "full_token_count",
        "full_token_digest",
        "policy_visible_text_sha256",
        "policy_visible_text_utf8_bytes",
        "policy_visible_token_ids",
        "policy_visible_token_count",
        "policy_visible_token_digest",
        "post_observation_prompt_token_ids",
        "post_observation_prompt_length",
        "post_observation_prompt_digest",
        "max_observation_tokens",
        "truncated",
        "head_token_count",
        "tail_token_count",
        "truncation_marker",
        "truncation_marker_sha256",
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise ContinuousAgentV1Error(
            "observation evidence is missing fields: " + ", ".join(missing)
        )
    if evidence["schema_version"] != CONTINUOUS_AGENT_OBSERVATION_SCHEMA_V1:
        raise ContinuousAgentV1Error("unsupported observation evidence schema")
    visible_text = str(record["environment_result"])
    if evidence["policy_visible_text_sha256"] != text_sha256(visible_text):
        raise ContinuousAgentV1Error("policy-visible observation text digest mismatch")
    if int(evidence["policy_visible_text_utf8_bytes"]) != len(
        visible_text.encode("utf-8")
    ):
        raise ContinuousAgentV1Error("policy-visible observation byte count mismatch")
    visible_ids = _raw_int_list(
        evidence["policy_visible_token_ids"],
        name="policy_visible_token_ids",
    )
    if int(evidence["policy_visible_token_count"]) != len(visible_ids):
        raise ContinuousAgentV1Error("policy-visible observation token count mismatch")
    if evidence["policy_visible_token_digest"] != token_digest(visible_ids):
        raise ContinuousAgentV1Error("policy-visible observation token digest mismatch")
    maximum = int(evidence["max_observation_tokens"])
    if maximum <= 0 or len(visible_ids) > maximum:
        raise ContinuousAgentV1Error("policy-visible observation exceeds token bound")
    full_count = int(evidence["full_token_count"])
    if full_count <= 0 or int(evidence["full_text_utf8_bytes"]) < 0:
        raise ContinuousAgentV1Error("full observation counts are invalid")
    post_prompt_ids = _raw_int_list(
        evidence["post_observation_prompt_token_ids"],
        name="post_observation_prompt_token_ids",
    )
    if int(evidence["post_observation_prompt_length"]) != len(post_prompt_ids):
        raise ContinuousAgentV1Error("post-observation prompt length mismatch")
    if evidence["post_observation_prompt_digest"] != token_digest(post_prompt_ids):
        raise ContinuousAgentV1Error("post-observation prompt digest mismatch")
    truncated = evidence["truncated"]
    if type(truncated) is not bool:
        raise ContinuousAgentV1Error("observation truncated flag must be boolean")
    head_count = int(evidence["head_token_count"])
    tail_count = int(evidence["tail_token_count"])
    if min(head_count, tail_count) < 0:
        raise ContinuousAgentV1Error("observation head/tail counts are invalid")
    marker = evidence["truncation_marker"]
    marker_digest = evidence["truncation_marker_sha256"]
    if truncated:
        if full_count <= maximum or head_count + tail_count >= full_count:
            raise ContinuousAgentV1Error("truncated observation counts are invalid")
        if not isinstance(marker, str) or not marker or marker not in visible_text:
            raise ContinuousAgentV1Error("observation truncation marker is absent")
        if marker_digest != text_sha256(marker):
            raise ContinuousAgentV1Error("observation truncation marker drifted")
    else:
        if marker is not None or marker_digest is not None:
            raise ContinuousAgentV1Error(
                "untruncated observation must not have a truncation marker"
            )
        if full_count != len(visible_ids) or head_count != full_count or tail_count != 0:
            raise ContinuousAgentV1Error("untruncated observation counts drifted")
        if evidence["full_text_sha256"] != evidence["policy_visible_text_sha256"]:
            raise ContinuousAgentV1Error("untruncated observation text digest drifted")
        if evidence["full_token_digest"] != evidence["policy_visible_token_digest"]:
            raise ContinuousAgentV1Error("untruncated observation token digest drifted")


def _validate_horizon_evidence_v1(
    record: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    required = {
        "schema_version",
        "environment_id",
        "environment_step",
        "native_environment_call_count",
        "policy_step_reward",
        "horizon_reward",
        "combined_reward",
        "environment_result",
        "environment_result_sha256",
        "done",
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise ContinuousAgentV1Error(
            "horizon evidence is missing fields: " + ", ".join(missing)
        )
    if evidence["schema_version"] != CONTINUOUS_AGENT_HORIZON_SCHEMA_V1:
        raise ContinuousAgentV1Error("unsupported horizon evidence schema")
    if int(evidence["environment_id"]) != int(record["environment_id"]):
        raise ContinuousAgentV1Error("horizon workspace identity drifted")
    if int(evidence["environment_step"]) != int(record["environment_step_after"]):
        raise ContinuousAgentV1Error("horizon task step drifted")
    if int(evidence["native_environment_call_count"]) != int(
        record["native_environment_call_count_after"]
    ):
        raise ContinuousAgentV1Error("horizon native-call count drifted")
    result = str(evidence["environment_result"])
    if evidence["environment_result_sha256"] != text_sha256(result):
        raise ContinuousAgentV1Error("horizon result digest drifted")
    values = [
        float(evidence["policy_step_reward"]),
        float(evidence["horizon_reward"]),
        float(evidence["combined_reward"]),
    ]
    if not all(math.isfinite(value) for value in values):
        raise ContinuousAgentV1Error("horizon rewards must be finite")
    if not math.isclose(values[0] + values[1], values[2], rel_tol=0.0, abs_tol=1e-12):
        raise ContinuousAgentV1Error("horizon combined reward is inconsistent")
    if not math.isclose(float(record["score"]), values[2], rel_tol=0.0, abs_tol=1e-12):
        raise ContinuousAgentV1Error("row reward does not match horizon evidence")
    if evidence["done"] is not True or record["done"] is not True:
        raise ContinuousAgentV1Error("horizon finalization must terminate the trajectory")


def validate_continuous_agent_trajectory_v1(
    records: Sequence[Mapping[str, Any]], *, require_terminal: bool = True
) -> None:
    """Validate one continuous trajectory across actions and compactions."""

    if not isinstance(records, Sequence) or not records:
        raise ContinuousAgentV1Error("continuous-agent trajectory must not be empty")
    if type(require_terminal) is not bool:
        raise ContinuousAgentV1Error("require_terminal must be boolean")
    identity_fields = (
        "task_name",
        "item_id",
        "data_idx",
        "parent_index",
        "parent_group_uid",
        "replica_index",
        "trajectory_uid",
        "environment_id",
    )
    first = records[0]
    previous: Mapping[str, Any] | None = None
    for index, record in enumerate(records):
        validate_continuous_agent_step_v1(record)
        for field in identity_fields:
            if record[field] != first[field]:
                raise ContinuousAgentV1Error(
                    f"continuous trajectory identity drifted at row {index}: {field}"
                )
        if int(record["environment_step_before"]) != index:
            raise ContinuousAgentV1Error(
                "continuous trajectory environment steps must start at zero and be dense"
            )
        if previous is not None:
            if bool(previous["done"]):
                raise ContinuousAgentV1Error("trajectory continued after a terminal row")
            for before_field, after_field in (
                ("environment_step_before", "environment_step_after"),
                (
                    "native_environment_call_count_before",
                    "native_environment_call_count_after",
                ),
                ("context_epoch_before", "context_epoch_after"),
            ):
                if int(record[before_field]) != int(previous[after_field]):
                    raise ContinuousAgentV1Error(
                        f"continuous trajectory counter drifted at row {index}: {before_field}"
                    )
            if previous["row_kind"] == COMPACTION_ROW:
                expected_action_prompt = previous["compaction"][
                    "post_compaction_prompt_token_ids"
                ]
            else:
                expected_action_prompt = previous["observation"][
                    "post_observation_prompt_token_ids"
                ]
            if record["row_kind"] == COMPACTION_ROW:
                current_action_prompt = record["compaction"][
                    "pre_request_action_prompt_token_ids"
                ]
            else:
                current_action_prompt = record["prompt_token_ids"]
            if list(current_action_prompt) != list(expected_action_prompt):
                raise ContinuousAgentV1Error(
                    f"continuous prompt history drifted before row {index}"
                )
        if record["horizon_finalization"] is not None and index != len(records) - 1:
            raise ContinuousAgentV1Error(
                "horizon finalization may appear only on the final row"
            )
        previous = record
    if any(bool(record["done"]) for record in records[:-1]):
        raise ContinuousAgentV1Error("done appears before the final trajectory row")
    if require_terminal and not bool(records[-1]["done"]):
        raise ContinuousAgentV1Error(
            "continuous trajectory reached packing without a terminal outcome"
        )
