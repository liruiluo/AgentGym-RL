from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from verl.utils.agentgym.formal_grpo_credit import (
    build_row_uid,
    compute_formal_grpo_credit,
)
from verl.utils.agentgym.formal_domain_v3 import (
    FORMAL_DOMAIN_SCHEMA_V3,
    FORMAL_WEBSHOP_INTENT_CLARIFICATION_FILESYSTEM_SURFACE_V2,
    FORMAL_WEBSHOP_INTENT_CLARIFICATION_SURFACE_V2,
    FORMAL_WEBSHOP_SCHEMA_V2,
    FormalDomainV3Error,
    canonical_unicode_contains,
    validate_formal_domain_step_v3,
)


TASK_NEUTRAL_POLICY_STEP_SCHEMA = "task_neutral_policy_step_v1"


AGENTMEMORY_PARENT_GROUP_UID = "agentmemory_parent_group_uid"
AGENTMEMORY_EXACT_STATE_UID = "agentmemory_exact_state_uid"
AGENTMEMORY_REPLICA_INDEX = "agentmemory_replica_index"
AGENTMEMORY_TRAJECTORY_UID = "agentmemory_trajectory_uid"
AGENTMEMORY_TRAJECTORY_RETURN = "agentmemory_trajectory_return"
AGENTMEMORY_IMMEDIATE_REWARD = "agentmemory_immediate_reward"
AGENTMEMORY_TRAJECTORY_ROW_UID = "agentmemory_trajectory_row_uid"
AGENTMEMORY_TRAJECTORY_ROW_ORDER = "agentmemory_trajectory_row_order"
AGENTMEMORY_TRAJECTORY_TERMINAL = "agentmemory_trajectory_terminal"
AGENTMEMORY_ACTION_TEXT = "agentmemory_action_text"
AGENTMEMORY_GENERATION_PROMPT_LENGTH = "agentmemory_generation_prompt_length"
AGENTMEMORY_GENERATION_PROMPT_DIGEST = "agentmemory_generation_prompt_digest"
AGENTMEMORY_PACKED_PROMPT_LENGTH = "agentmemory_packed_prompt_length"
AGENTMEMORY_PACKED_PROMPT_DIGEST = "agentmemory_packed_prompt_digest"
AGENTMEMORY_GENERATION_RESPONSE_LENGTH = "agentmemory_generation_response_length"
AGENTMEMORY_GENERATION_RESPONSE_DIGEST = "agentmemory_generation_response_digest"
AGENTMEMORY_PACKED_RESPONSE_LENGTH = "agentmemory_packed_response_length"
AGENTMEMORY_PACKED_RESPONSE_DIGEST = "agentmemory_packed_response_digest"
AGENTMEMORY_SUFFIX_CREDIT_APPLIED = "agentmemory_suffix_credit_applied"
AGENTMEMORY_SUFFIX_RETURN = "agentmemory_suffix_return"
AGENTMEMORY_STEP_RECORD_JSON = "agentmemory_step_record_json"

FORMAL_TRAJECTORY_NON_TENSOR_KEYS = (
    AGENTMEMORY_PARENT_GROUP_UID,
    AGENTMEMORY_EXACT_STATE_UID,
    AGENTMEMORY_REPLICA_INDEX,
    AGENTMEMORY_TRAJECTORY_UID,
    AGENTMEMORY_TRAJECTORY_ROW_UID,
)
FORMAL_TRAJECTORY_TENSOR_KEYS = (
    AGENTMEMORY_TRAJECTORY_RETURN,
    AGENTMEMORY_IMMEDIATE_REWARD,
    AGENTMEMORY_TRAJECTORY_ROW_ORDER,
    AGENTMEMORY_TRAJECTORY_TERMINAL,
)
FORMAL_RUNTIME_EVIDENCE_NON_TENSOR_KEYS = (
    AGENTMEMORY_ACTION_TEXT,
    AGENTMEMORY_GENERATION_PROMPT_DIGEST,
    AGENTMEMORY_PACKED_PROMPT_DIGEST,
    AGENTMEMORY_GENERATION_RESPONSE_DIGEST,
    AGENTMEMORY_PACKED_RESPONSE_DIGEST,
    AGENTMEMORY_STEP_RECORD_JSON,
)
FORMAL_RUNTIME_EVIDENCE_TENSOR_KEYS = (
    AGENTMEMORY_GENERATION_PROMPT_LENGTH,
    AGENTMEMORY_PACKED_PROMPT_LENGTH,
    AGENTMEMORY_GENERATION_RESPONSE_LENGTH,
    AGENTMEMORY_PACKED_RESPONSE_LENGTH,
    AGENTMEMORY_SUFFIX_CREDIT_APPLIED,
    AGENTMEMORY_SUFFIX_RETURN,
)


_PPO_VALID_SAMPLE_MASK = "ppo_valid_sample_mask"
_RESTRICTED_AGENTMEMORY_MARKERS = (
    "agentmemory_nonformal_grouping_mode",
)


def build_parent_group_uid(parent_index: int) -> str:
    """Identify all online continuations sampled from one initial task."""

    return f"agentmemory:parentv1:{int(parent_index)}"


def build_trajectory_uid(parent_group_uid: str, replica_index: int) -> str:
    """Identify one complete online continuation within an initial task."""

    return f"{str(parent_group_uid)}:replica{int(replica_index)}"


def prompt_token_digest(prompt_token_ids: Sequence[int]) -> str:
    """Hash the exact ordered token sequence used to condition generation."""

    digest = hashlib.sha256()
    for token_id in prompt_token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def validate_formal_sequence_limits(
    *,
    prompt_width: int,
    response_width: int,
    max_model_len: int,
) -> dict[str, int]:
    """Require one coherent prompt, response, and total-sequence capacity."""

    values = {
        "prompt_width": prompt_width,
        "response_width": response_width,
        "max_model_len": max_model_len,
    }
    normalized: dict[str, int] = {}
    for name, value in values.items():
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Formal {name} must be a positive integer.") from exc
        if parsed <= 0 or parsed != value:
            raise ValueError(f"Formal {name} must be a positive integer.")
        normalized[name] = parsed
    required_model_len = normalized["prompt_width"] + normalized["response_width"]
    if normalized["max_model_len"] < required_model_len:
        raise ValueError(
            "Formal max_model_len is smaller than prompt plus response capacity: "
            f"prompt={normalized['prompt_width']} "
            f"response={normalized['response_width']} "
            f"required={required_model_len} "
            f"actual={normalized['max_model_len']}."
        )
    return {**normalized, "required_model_len": required_model_len}


def validate_formal_response_reward_placement(
    *,
    response_masks: Sequence[Sequence[Any]],
    score_rows: Sequence[Sequence[Any]],
    expected_rewards: Sequence[Any],
    valid_mask: Sequence[Any],
    tolerance: float = 1e-4,
) -> dict[str, int]:
    """Require each packed scalar credit on the final sampled assistant token."""

    lengths = {
        "response_masks": len(response_masks),
        "score_rows": len(score_rows),
        "expected_rewards": len(expected_rewards),
        "valid_mask": len(valid_mask),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Formal reward-placement row mismatch: {lengths}")
    valid_rows = 0
    for row_index, (mask_row, score_row) in enumerate(
        zip(response_masks, score_rows)
    ):
        if not bool(valid_mask[row_index]):
            continue
        valid_rows += 1
        if len(mask_row) != len(score_row):
            raise ValueError(
                "Formal reward-placement tensor width mismatch: "
                f"row={row_index} mask={len(mask_row)} scores={len(score_row)}."
            )
        sampled_positions = [
            index for index, visible in enumerate(mask_row) if bool(visible)
        ]
        if not sampled_positions:
            raise ValueError(
                f"Formal action row has no sampled assistant tokens at row {row_index}."
            )
        try:
            scores = [float(value) for value in score_row]
            expected_reward = float(expected_rewards[row_index])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Formal reward placement is non-numeric at row {row_index}."
            ) from exc
        if not math.isfinite(expected_reward) or not all(
            math.isfinite(value) for value in scores
        ):
            raise ValueError(
                f"Formal reward placement is non-finite at row {row_index}."
            )
        sampled_set = set(sampled_positions)
        mask_zero_reward = sum(
            abs(value) for index, value in enumerate(scores) if index not in sampled_set
        )
        if mask_zero_reward > tolerance:
            raise ValueError(
                "Formal scalar reward is placed on a response_mask=0 token: "
                f"row={row_index} magnitude={mask_zero_reward}."
            )
        terminal_position = sampled_positions[-1]
        if not math.isclose(
            scores[terminal_position],
            expected_reward,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ValueError(
                "Formal scalar reward is not on the final sampled assistant token: "
                f"row={row_index} terminal_score={scores[terminal_position]} "
                f"expected_reward={expected_reward}."
            )
        nonterminal_reward = sum(
            abs(scores[index]) for index in sampled_positions[:-1]
        )
        if nonterminal_reward > tolerance:
            raise ValueError(
                "Formal scalar reward appears before the final sampled assistant token: "
                f"row={row_index} magnitude={nonterminal_reward}."
            )
    if valid_rows == 0:
        raise ValueError("Formal reward placement has no valid rows.")
    return {"valid_rows": valid_rows}


def validate_formal_response_aligned_tensors(
    *,
    response_masks: Sequence[Sequence[Any]],
    expected_response_lengths: Sequence[Any],
    tensors: dict[str, Sequence[Sequence[Any]]],
    valid_mask: Sequence[Any],
) -> dict[str, int]:
    """Bind response-only PPO tensors to the sampled-token mask and digest length."""

    row_count = len(response_masks)
    if len(expected_response_lengths) != row_count or len(valid_mask) != row_count:
        raise ValueError("Formal response-alignment metadata row count differs.")
    checked = 0
    for name, rows in tensors.items():
        if len(rows) != row_count:
            raise ValueError(
                f"Formal {name} row count differs from response_mask: "
                f"tensor={len(rows)} mask={row_count}."
            )
        for row_index, (mask_row, tensor_row) in enumerate(zip(response_masks, rows)):
            if not bool(valid_mask[row_index]):
                continue
            if len(tensor_row) != len(mask_row):
                raise ValueError(
                    f"Formal {name} width differs from response_mask at row {row_index}."
                )
            selected = [
                tensor_row[index]
                for index, visible in enumerate(mask_row)
                if bool(visible)
            ]
            expected_length = _coerce_nonnegative_int(
                expected_response_lengths[row_index],
                name="packed response length",
                row_index=row_index,
            )
            if len(selected) != expected_length:
                raise ValueError(
                    f"Formal {name} sampled-token count differs from response digest "
                    f"length at row {row_index}: tensor={len(selected)} "
                    f"expected={expected_length}."
                )
            try:
                numeric = [float(value) for value in selected]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Formal {name} contains non-numeric sampled-token values."
                ) from exc
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(
                    f"Formal {name} contains non-finite sampled-token values."
                )
            checked += 1
    return {"tensor_count": len(tensors), "checked_rows": checked}


def normalize_generation_record(
    token_ids: Sequence[Any],
    *,
    eos_token_ids: Sequence[int] | int | None,
    primary_eos_token_id: int | None,
    pad_token_id: int | None,
    max_tokens: int,
    backend_finish_reason: Any = None,
    stop_reason: Any = None,
    finish_reason_source: str,
    token_ids_are_exact: bool = False,
) -> dict[str, Any]:
    """Normalize the actual backend generation result without losing stop data."""

    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}.")
    if eos_token_ids is None:
        ordered_eos_ids: list[int] = []
    elif isinstance(eos_token_ids, int):
        ordered_eos_ids = [int(eos_token_ids)]
    else:
        ordered_eos_ids = []
        for index, token_id in enumerate(eos_token_ids):
            try:
                ordered_eos_ids.append(int(token_id))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid configured EOS token id at index {index}: {token_id!r}"
                ) from exc
    eos_ids = set(ordered_eos_ids)
    normalized_primary_eos_token_id = (
        None if primary_eos_token_id is None else int(primary_eos_token_id)
    )
    if (
        normalized_primary_eos_token_id is not None
        and normalized_primary_eos_token_id not in eos_ids
    ):
        raise ValueError(
            "Primary EOS token ID is absent from the configured stop-token set: "
            f"primary={normalized_primary_eos_token_id} "
            f"configured={ordered_eos_ids}."
        )

    raw_tokens: list[int] = []
    for index, raw_token_id in enumerate(token_ids):
        try:
            raw_tokens.append(int(raw_token_id))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid generated token id at index {index}: {raw_token_id!r}"
            ) from exc

    cleaned: list[int] = []
    stopped_by_eos = False
    stopped_by_padding = False
    if token_ids_are_exact:
        cleaned = raw_tokens
    else:
        for token_id in raw_tokens:
            # A configured EOS is a real sampled token even when its ID equals
            # tokenizer.pad_token_id. Keep the first EOS and trim only the
            # rectangular tensor padding that follows it.
            if token_id in eos_ids:
                cleaned.append(token_id)
                stopped_by_eos = True
                break
            if pad_token_id is not None and token_id == int(pad_token_id):
                stopped_by_padding = True
                break
            cleaned.append(token_id)

    if backend_finish_reason is None:
        finish_reason = (
            "stop"
            if stopped_by_eos or stopped_by_padding or len(cleaned) < max_tokens
            else "length"
        )
        source = f"{finish_reason_source}:tokens_and_budget"
    else:
        finish_reason = str(backend_finish_reason).strip().lower()
        source = f"{finish_reason_source}:backend"
    if finish_reason not in {"stop", "length"}:
        raise ValueError(
            "Formal generation has unsupported or missing finish_reason: "
            f"{backend_finish_reason!r}."
        )
    return {
        "token_ids": cleaned,
        "response_token_count": len(cleaned),
        "max_response_tokens": int(max_tokens),
        "finish_reason": finish_reason,
        "finish_reason_source": source,
        "stop_reason": (
            stop_reason
            if stop_reason is None or isinstance(stop_reason, (bool, int, float, str))
            else str(stop_reason)
        ),
        "truncated": finish_reason == "length",
        "configured_eos_token_ids": ordered_eos_ids,
        "primary_eos_token_id": normalized_primary_eos_token_id,
        "tokenizer_pad_token_id": (
            None if pad_token_id is None else int(pad_token_id)
        ),
        "backend_source": str(finish_reason_source).split(":", 1)[0],
        "backend_token_ids_are_exact": bool(token_ids_are_exact),
        "token_ids_are_exact": bool(
            token_ids_are_exact
            or stopped_by_eos
            or (
                not stopped_by_padding
                and len(cleaned) >= int(max_tokens)
            )
        ),
    }


def validate_official_vllm_generation_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Validate exact official-vLLM completion metadata without model assumptions."""

    if record.get("backend_source") != "official_vllm":
        raise RuntimeError("Formal generation is not bound to official vLLM.")
    if record.get("finish_reason_source") != "official_vllm:backend":
        raise RuntimeError(
            "Formal generation finish_reason is not official-vLLM backend metadata."
        )
    if not record.get("backend_token_ids_are_exact") or not record.get(
        "token_ids_are_exact"
    ):
        raise RuntimeError("Formal official-vLLM token IDs are not exact.")
    finish_reason = record.get("finish_reason")
    truncated = record.get("truncated")
    if finish_reason not in {"stop", "length"}:
        raise RuntimeError(
            "Formal official-vLLM generation has an unsupported finish_reason."
        )
    if type(truncated) is not bool or truncated != (finish_reason == "length"):
        raise RuntimeError(
            "Formal official-vLLM finish_reason/truncated metadata is inconsistent."
        )

    raw_token_ids = record.get("token_ids", [])
    raw_eos_token_ids = record.get("configured_eos_token_ids", [])
    if not isinstance(raw_token_ids, list) or any(
        type(token_id) is not int or token_id < 0 for token_id in raw_token_ids
    ):
        raise RuntimeError(
            "Formal official-vLLM generation token IDs are not non-negative raw integers."
        )
    if (
        not isinstance(raw_eos_token_ids, list)
        or not raw_eos_token_ids
        or any(
            type(token_id) is not int or token_id < 0
            for token_id in raw_eos_token_ids
        )
    ):
        raise RuntimeError(
            "Formal official-vLLM EOS token IDs must be a non-empty list of "
            "non-negative raw integers."
        )
    token_ids = list(raw_token_ids)
    eos_token_ids = list(raw_eos_token_ids)
    if len(eos_token_ids) != len(set(eos_token_ids)):
        raise RuntimeError(
            "Formal official-vLLM EOS token IDs must be unique."
        )
    primary_eos_token_id = record.get("primary_eos_token_id")
    if (
        type(primary_eos_token_id) is not int
        or primary_eos_token_id < 0
        or primary_eos_token_id not in eos_token_ids
    ):
        raise RuntimeError(
            "Formal official-vLLM primary EOS token ID must be a configured "
            "non-negative raw integer."
        )
    pad_token_id = record.get("tokenizer_pad_token_id")
    if pad_token_id is not None and (
        type(pad_token_id) is not int or pad_token_id < 0
    ):
        raise RuntimeError(
            "Formal official-vLLM tokenizer pad token ID must be a "
            "non-negative raw integer or None."
        )
    if not token_ids:
        raise RuntimeError("Formal official-vLLM generation returned no tokens.")

    response_token_count = record.get("response_token_count")
    max_response_tokens = record.get("max_response_tokens")
    if (
        type(response_token_count) is not int
        or response_token_count != len(token_ids)
    ):
        raise RuntimeError(
            "Formal official-vLLM response token count does not match exact token IDs."
        )
    if type(max_response_tokens) is not int or max_response_tokens <= 0:
        raise RuntimeError(
            "Formal official-vLLM max response token count is invalid."
        )
    if response_token_count > max_response_tokens:
        raise RuntimeError(
            "Formal official-vLLM response exceeds the configured token limit."
        )

    eos_positions = [
        index for index, token_id in enumerate(token_ids) if token_id in eos_token_ids
    ]
    stop_reason = record.get("stop_reason")
    if finish_reason == "length":
        if response_token_count != max_response_tokens:
            raise RuntimeError(
                "Formal official-vLLM length completion did not reach max_response_tokens."
            )
        # vLLM 0.24 checks primary EOS and stop-token IDs before the length cap.
        # A sampled configured EOS therefore produces finish_reason=stop even
        # when it is the token at max_tokens.
        if eos_positions:
            raise RuntimeError(
                "Formal official-vLLM length completion contains a configured EOS."
            )
        if stop_reason is not None:
            raise RuntimeError(
                "Formal official-vLLM length completion requires stop_reason=None."
            )
        return record

    if not eos_positions:
        raise RuntimeError(
            "Formal official-vLLM generation is missing a terminal EOS."
        )
    if eos_positions[0] != len(token_ids) - 1:
        raise RuntimeError(
            "Formal official-vLLM generation contains tokens after the first EOS."
        )

    final_token_id = token_ids[-1]
    if final_token_id == primary_eos_token_id:
        if stop_reason is not None:
            raise RuntimeError(
                "Formal official-vLLM primary EOS requires stop_reason=None."
            )
    else:
        if stop_reason is None:
            raise RuntimeError(
                "Formal official-vLLM alternate EOS requires an explicit "
                "backend stop_reason."
            )
        if type(stop_reason) is not int:
            raise RuntimeError(
                "Formal EOS stop_reason is not a raw integer token ID."
            )
        if stop_reason != final_token_id:
            raise RuntimeError(
                "Formal EOS does not match the backend stop_reason."
            )
    return record


def _coerce_nonnegative_int(value: Any, *, name: str, row_index: int) -> int:
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} at row {row_index}: {value!r}") from exc
    if not math.isfinite(numeric) or numeric != integer or integer < 0:
        raise ValueError(f"Invalid {name} at row {row_index}: {value!r}")
    return integer


def _validate_prompt_digest(value: Any, *, name: str, row_index: int) -> str:
    digest = str(value)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"Invalid {name} at row {row_index}: {value!r}")
    return digest


_NATIVE_WEBSHOP_ACTION_RE = re.compile(
    r"\A(search|click)\[([^\[\]\r\n]+)\]\Z",
)
_MEMORY_ACTION_RE = re.compile(
    r"\A(ADD|UPDATE|DELETE|RETRIEVE|SUMMARY|FILTER)\s+(\{.*\})\Z",
    re.DOTALL,
)
_ASK_ACTION_RE = re.compile(r"\AASK\s+(\{.*\})\Z", re.DOTALL)
_SHELL_COMMAND_ACTION_RE = re.compile(
    r"\Ashell_command\s+(\{.*\})\Z",
    re.DOTALL,
)
_APPLY_PATCH_ACTION_PREFIX = "apply_patch\n"
_INTENT_CLARIFICATION_SURFACES = frozenset(
    {
        FORMAL_WEBSHOP_INTENT_CLARIFICATION_SURFACE_V2,
        FORMAL_WEBSHOP_INTENT_CLARIFICATION_FILESYSTEM_SURFACE_V2,
    }
)


def _parse_formal_native_action(action: Any, *, row_index: int) -> tuple[str, Any]:
    """Parse the policy-visible MemoryArena action without guessing BUY."""

    if not isinstance(action, str) or not action.strip():
        raise ValueError(f"Formal action is empty at row {row_index}.")
    text = action.strip()
    native_match = _NATIVE_WEBSHOP_ACTION_RE.fullmatch(text)
    if native_match is not None:
        argument = native_match.group(2).strip()
        if not argument:
            raise ValueError(f"Formal native action has an empty argument at row {row_index}.")
        return native_match.group(1).upper(), argument

    memory_match = _MEMORY_ACTION_RE.fullmatch(text)
    if memory_match is not None:
        try:
            payload = json.loads(memory_match.group(2))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Formal memory action has invalid JSON at row {row_index}.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Formal memory action payload is not an object at row {row_index}.")
        return memory_match.group(1), memory_match.group(2)

    ask_match = _ASK_ACTION_RE.fullmatch(text)
    if ask_match is not None:
        try:
            payload = json.loads(ask_match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Formal ASK action has invalid JSON at row {row_index}."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Formal ASK action payload is not an object at row {row_index}."
            )
        return "ASK", payload

    shell_match = _SHELL_COMMAND_ACTION_RE.fullmatch(text)
    if shell_match is not None:
        try:
            payload = json.loads(shell_match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Formal shell_command action has invalid JSON at row {row_index}."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Formal shell_command payload is not an object at row {row_index}."
            )
        return "SHELL_COMMAND", shell_match.group(1)

    if text.startswith(_APPLY_PATCH_ACTION_PREFIX):
        patch = text[len(_APPLY_PATCH_ACTION_PREFIX) :]
        if not patch.strip():
            raise ValueError(
                f"Formal apply_patch action is empty at row {row_index}."
            )
        return "APPLY_PATCH", patch
    raise ValueError(f"Formal action has unsupported native syntax at row {row_index}.")


def _resolve_formal_native_action_op(
    action: Any,
    tool_ops: Sequence[dict[str, Any]],
    *,
    row_index: int,
    surface: Any,
) -> str:
    """Bind native syntax to the environment's one structured tool event.

    A click becomes BUY only when the backend emitted a committed BUY event.
    The action text alone never proves that a purchase happened.
    """

    try:
        syntax_op, native_argument = _parse_formal_native_action(
            action, row_index=row_index
        )
    except ValueError:
        if tool_ops:
            raise ValueError(
                f"Formal invalid action claims a tool operation at row {row_index}."
            )
        return "INVALID"

    if syntax_op == "ASK" and surface not in _INTENT_CLARIFICATION_SURFACES:
        if tool_ops:
            raise ValueError(
                "Formal ASK action produced a tool operation outside an intent-"
                f"clarification surface at row {row_index}."
            )
        return "INVALID"

    if len(tool_ops) > 1:
        raise ValueError(
            f"Formal action produced multiple tool operations at row {row_index}."
        )
    if not tool_ops:
        return syntax_op

    tool_op = tool_ops[0]["op"]
    if syntax_op == "SEARCH":
        if tool_op != "SEARCH":
            raise ValueError(
                f"Formal search action is bound to {tool_op} at row {row_index}."
            )
    elif syntax_op == "CLICK":
        if tool_op not in {"CLICK", "BUY"}:
            raise ValueError(
                f"Formal click action is bound to {tool_op} at row {row_index}."
            )
        if tool_op == "BUY" and (
            native_argument.casefold() != "buy now"
            or tool_ops[0].get("committed") is not True
        ):
            raise ValueError(
                f"Formal BUY lacks an exact committed click[Buy Now] at row {row_index}."
            )
    elif syntax_op == "ASK":
        if tool_op != "CLARIFY":
            raise ValueError(
                f"Formal ASK action is bound to {tool_op} at row {row_index}."
            )
        ask_payload = native_argument
        if (
            set(ask_payload) != {"field"}
            or not isinstance(ask_payload["field"], str)
            or not ask_payload["field"].strip()
        ):
            raise ValueError(
                f"Formal CLARIFY lacks one nonempty ASK field at row {row_index}."
            )
        clarification_event = tool_ops[0]
        if clarification_event.get("request_op") != "ASK":
            raise ValueError(
                f"Formal CLARIFY event lacks request_op=ASK at row {row_index}."
            )
        if clarification_event.get("field") != ask_payload["field"]:
            raise ValueError(
                f"Formal CLARIFY field differs from the ASK payload at row {row_index}."
            )
        if clarification_event.get("clarification_received") is not True:
            raise ValueError(
                f"Formal CLARIFY event lacks receipt evidence at row {row_index}."
            )
        if clarification_event.get("session_index") != 0:
            raise ValueError(
                f"Formal CLARIFY event occurred outside session zero at row {row_index}."
            )
    elif tool_op != syntax_op:
        raise ValueError(
            f"Formal {syntax_op} action is bound to {tool_op} at row {row_index}."
        )

    tool_raw_action = tool_ops[0].get("raw_action")
    if tool_op in {"SEARCH", "CLICK", "BUY"} and tool_raw_action != str(action).strip():
        raise ValueError(
            f"Formal tool raw_action binding mismatch at row {row_index}."
        )
    if tool_op not in {"SEARCH", "CLICK", "BUY"} and tool_raw_action is not None:
        if tool_raw_action != str(action).strip():
            raise ValueError(
                f"Formal tool raw_action binding mismatch at row {row_index}."
            )
    return tool_op


def _validate_formal_domain_step_record_v3(
    record: dict[str, Any],
    *,
    row_index: int,
    expected_exact_state_uid: str,
    expected_trajectory_uid: str,
    expected_trajectory_row_uid: str,
    expected_trajectory_row_order: int,
    expected_trajectory_terminal: bool,
    expected_task_round: int,
    expected_immediate_reward: float,
    expected_suffix_return: float,
    expected_trajectory_return: float,
    expected_action_text: str,
    expected_done: bool,
    expected_generation_length: int,
    expected_generation_digest: str,
    expected_packed_length: int,
    expected_packed_digest: str,
    expected_generation_response_length: int,
    expected_generation_response_digest: str,
    expected_packed_response_length: int,
    expected_packed_response_digest: str,
    expected_suffix_credit_applied: bool,
    tolerance: float,
) -> dict[str, Any]:
    """Validate packed domain-v3 evidence without applying WebShop semantics."""

    required_fields = {
        "item_id",
        "parent_index",
        "parent_group_uid",
        "replica_index",
        "exact_state_uid",
        "trajectory_uid",
        "trajectory_row_uid",
        "trajectory_row_order",
        "trajectory_terminal",
        "task_round",
        "content",
        "action",
        "score",
        "immediate_reward",
        "suffix_return",
        "suffix_credit_applied",
        "trajectory_return",
        "done",
        "visible_prompt",
        "latest_observation",
        "system_prompt",
        "system_prompt_source",
        "system_prompt_sha256",
        "prompt_history_policy",
        "raw_prior_messages_visible",
        "single_observation_prompt_digest",
        "response_token_ids",
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
        "env_result",
        "env_info_before",
        "env_info_after",
        "action_execution",
        "tool_ops",
        "reward_components",
        "domain_evidence",
        "phase_index_before",
        "phase_index_after",
        "phase_count",
        "phase_advanced",
        "episode_success",
        "sample_excluded",
        "generation_prompt_length",
        "generation_prompt_digest",
        "packed_prompt_length",
        "packed_prompt_digest",
        "generation_response_length",
        "generation_response_digest",
        "packed_response_length",
        "packed_response_digest",
    }
    missing = sorted(required_fields - set(record))
    if missing:
        raise ValueError(
            f"Formal v3 step record is missing fields at row {row_index}: {missing}"
        )
    try:
        validate_formal_domain_step_v3(record)
    except FormalDomainV3Error as exc:
        raise ValueError(
            f"Invalid formal v3 step record at row {row_index}: {exc}"
        ) from exc

    if record["prompt_history_policy"] != "latest_observation_only":
        raise ValueError(
            f"Formal v3 prompt history policy mismatch at row {row_index}."
        )
    if record["raw_prior_messages_visible"] is not False:
        raise ValueError(
            f"Formal v3 prompt exposes raw prior messages at row {row_index}."
        )
    if not isinstance(record["env_result"], str):
        raise ValueError(f"Formal v3 env_result must be text at row {row_index}.")

    trajectory_row_order = _coerce_nonnegative_int(
        record["trajectory_row_order"],
        name="trajectory row order",
        row_index=row_index,
    )
    if trajectory_row_order != expected_trajectory_row_order:
        raise ValueError(
            f"Formal v3 trajectory row order mismatch at row {row_index}."
        )
    trajectory_row_uid = str(record["trajectory_row_uid"])
    if (
        trajectory_row_uid != expected_trajectory_row_uid
        or trajectory_row_uid
        != build_row_uid(expected_trajectory_uid, expected_trajectory_row_order)
    ):
        raise ValueError(
            f"Formal v3 trajectory row UID mismatch at row {row_index}."
        )
    if type(record["trajectory_terminal"]) is not bool or record[
        "trajectory_terminal"
    ] != bool(expected_trajectory_terminal):
        raise ValueError(
            f"Formal v3 trajectory terminal mismatch at row {row_index}."
        )

    response_token_ids = record["response_token_ids"]
    if not isinstance(response_token_ids, list) or any(
        type(token_id) is not int for token_id in response_token_ids
    ):
        raise ValueError(
            f"Formal v3 response_token_ids must contain raw integers at row {row_index}."
        )
    response_token_count = _coerce_nonnegative_int(
        record["response_token_count"],
        name="response token count",
        row_index=row_index,
    )
    if response_token_count != len(response_token_ids):
        raise ValueError(
            f"Formal v3 response token count mismatch at row {row_index}."
        )
    response_digest = prompt_token_digest(response_token_ids)
    if (
        response_token_count != expected_generation_response_length
        or response_digest != expected_generation_response_digest
    ):
        raise ValueError(
            f"Formal v3 generated-response metadata mismatch at row {row_index}."
        )
    generation_record = {
        "token_ids": response_token_ids,
        "response_token_count": response_token_count,
        "max_response_tokens": record["max_response_tokens"],
        "finish_reason": record["finish_reason"],
        "finish_reason_source": record["finish_reason_source"],
        "stop_reason": record["stop_reason"],
        "truncated": record["truncated"],
        "configured_eos_token_ids": record["generation_eos_token_ids"],
        "primary_eos_token_id": record["tokenizer_primary_eos_token_id"],
        "tokenizer_pad_token_id": record["tokenizer_pad_token_id"],
        "backend_source": record["generation_backend_source"],
        "backend_token_ids_are_exact": record["backend_token_ids_are_exact"],
        "token_ids_are_exact": record["generation_token_ids_are_exact"],
    }
    try:
        validate_official_vllm_generation_record(generation_record)
    except RuntimeError as exc:
        raise ValueError(
            f"Formal v3 generation provenance is invalid at row {row_index}: {exc}"
        ) from exc
    tokenizer_pad_token_id = record["tokenizer_pad_token_id"]
    if tokenizer_pad_token_id is not None and (
        type(tokenizer_pad_token_id) is not int or tokenizer_pad_token_id < 0
    ):
        raise ValueError(
            "Formal v3 tokenizer pad readback must be a non-negative integer "
            "or None: "
            f"row={row_index} actual={record['tokenizer_pad_token_id']!r}."
        )

    single_observation_digest = _validate_prompt_digest(
        record["single_observation_prompt_digest"],
        name="single-observation prompt digest",
        row_index=row_index,
    )
    if single_observation_digest != expected_generation_digest:
        raise ValueError(
            f"Formal v3 prompt was not built from one latest observation at row {row_index}."
        )
    exact_state_uid = str(record["exact_state_uid"])
    if ":statev1:" not in exact_state_uid:
        raise ValueError(
            f"Invalid formal v3 exact-state UID at row {row_index}: {exact_state_uid!r}"
        )
    if exact_state_uid.rsplit(":statev1:", 1)[1] != expected_generation_digest:
        raise ValueError(
            f"Formal v3 exact-state UID digest mismatch at row {row_index}."
        )

    numeric_expectations = {
        "score": expected_immediate_reward,
        "immediate_reward": expected_immediate_reward,
        "suffix_return": expected_suffix_return,
        "trajectory_return": expected_trajectory_return,
    }
    for name, expected in numeric_expectations.items():
        try:
            actual = float(record[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid formal v3 step {name} at row {row_index}."
            ) from exc
        if not math.isfinite(actual) or not math.isclose(
            actual, expected, rel_tol=tolerance, abs_tol=tolerance
        ):
            raise ValueError(
                f"Formal v3 step {name} mismatch at row {row_index}: "
                f"expected={expected} actual={actual}."
            )

    exact_expectations = {
        "exact_state_uid": expected_exact_state_uid,
        "trajectory_uid": expected_trajectory_uid,
        "trajectory_row_uid": expected_trajectory_row_uid,
        "trajectory_row_order": expected_trajectory_row_order,
        "trajectory_terminal": expected_trajectory_terminal,
        "task_round": expected_task_round,
        "content": expected_action_text,
        "action": expected_action_text,
        "done": expected_done,
        "generation_prompt_length": expected_generation_length,
        "generation_prompt_digest": expected_generation_digest,
        "packed_prompt_length": expected_packed_length,
        "packed_prompt_digest": expected_packed_digest,
        "generation_response_length": expected_generation_response_length,
        "generation_response_digest": expected_generation_response_digest,
        "packed_response_length": expected_packed_response_length,
        "packed_response_digest": expected_packed_response_digest,
        "suffix_credit_applied": expected_suffix_credit_applied,
    }
    for name, expected in exact_expectations.items():
        if record[name] != expected:
            raise ValueError(
                f"Formal v3 step record {name} mismatch at row {row_index}: "
                f"expected={expected!r} actual={record[name]!r}."
            )
    return record


def _validate_formal_step_record(
    value: Any,
    *,
    row_index: int,
    expected_exact_state_uid: str,
    expected_trajectory_uid: str,
    expected_trajectory_row_uid: str,
    expected_trajectory_row_order: int,
    expected_trajectory_terminal: bool,
    expected_task_round: int,
    expected_immediate_reward: float,
    expected_suffix_return: float,
    expected_trajectory_return: float,
    expected_action_text: str,
    expected_done: bool,
    expected_generation_length: int,
    expected_generation_digest: str,
    expected_packed_length: int,
    expected_packed_digest: str,
    expected_generation_response_length: int,
    expected_generation_response_digest: str,
    expected_packed_response_length: int,
    expected_packed_response_digest: str,
    expected_suffix_credit_applied: bool,
    tolerance: float,
) -> dict[str, Any]:
    try:
        record = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid formal step record JSON at row {row_index}."
        ) from exc
    schema_version = record.get("schema_version")
    if schema_version == TASK_NEUTRAL_POLICY_STEP_SCHEMA:
        return _validate_task_neutral_policy_step_record(
            record,
            row_index=row_index,
            expected_exact_state_uid=expected_exact_state_uid,
            expected_trajectory_uid=expected_trajectory_uid,
            expected_trajectory_row_uid=expected_trajectory_row_uid,
            expected_trajectory_row_order=expected_trajectory_row_order,
            expected_trajectory_terminal=expected_trajectory_terminal,
            expected_task_round=expected_task_round,
            expected_immediate_reward=expected_immediate_reward,
            expected_suffix_return=expected_suffix_return,
            expected_trajectory_return=expected_trajectory_return,
            expected_action_text=expected_action_text,
            expected_done=expected_done,
            expected_generation_length=expected_generation_length,
            expected_generation_digest=expected_generation_digest,
            expected_packed_length=expected_packed_length,
            expected_packed_digest=expected_packed_digest,
            expected_generation_response_length=expected_generation_response_length,
            expected_generation_response_digest=expected_generation_response_digest,
            expected_packed_response_length=expected_packed_response_length,
            expected_packed_response_digest=expected_packed_response_digest,
            expected_suffix_credit_applied=expected_suffix_credit_applied,
            tolerance=tolerance,
        )
    if schema_version == FORMAL_DOMAIN_SCHEMA_V3:
        return _validate_formal_domain_step_record_v3(
            record,
            row_index=row_index,
            expected_exact_state_uid=expected_exact_state_uid,
            expected_trajectory_uid=expected_trajectory_uid,
            expected_trajectory_row_uid=expected_trajectory_row_uid,
            expected_trajectory_row_order=expected_trajectory_row_order,
            expected_trajectory_terminal=expected_trajectory_terminal,
            expected_task_round=expected_task_round,
            expected_immediate_reward=expected_immediate_reward,
            expected_suffix_return=expected_suffix_return,
            expected_trajectory_return=expected_trajectory_return,
            expected_action_text=expected_action_text,
            expected_done=expected_done,
            expected_generation_length=expected_generation_length,
            expected_generation_digest=expected_generation_digest,
            expected_packed_length=expected_packed_length,
            expected_packed_digest=expected_packed_digest,
            expected_generation_response_length=(
                expected_generation_response_length
            ),
            expected_generation_response_digest=(
                expected_generation_response_digest
            ),
            expected_packed_response_length=expected_packed_response_length,
            expected_packed_response_digest=expected_packed_response_digest,
            expected_suffix_credit_applied=expected_suffix_credit_applied,
            tolerance=tolerance,
        )
    if schema_version != FORMAL_WEBSHOP_SCHEMA_V2:
        raise ValueError(
            f"Unknown formal step record schema at row {row_index}: "
            f"{schema_version!r}"
        )
    required_fields = {
        "schema_version",
        "item_id",
        "exact_state_uid",
        "trajectory_uid",
        "trajectory_row_uid",
        "trajectory_row_order",
        "trajectory_terminal",
        "task_round",
        "session_index",
        "subtask_index",
        "next_session_index",
        "subtask_index_before",
        "subtask_index_after",
        "visible_prompt",
        "latest_observation",
        "prompt_history_policy",
        "raw_prior_messages_visible",
        "single_observation_prompt_digest",
        "action",
        "response_token_ids",
        "response_token_count",
        "max_response_tokens",
        "finish_reason",
        "finish_reason_source",
        "stop_reason",
        "generation_backend_source",
        "generation_stop_reason",
        "generation_eos_token_ids",
        "tokenizer_primary_eos_token_id",
        "tokenizer_pad_token_id",
        "generation_token_ids_are_exact",
        "backend_token_ids_are_exact",
        "truncated",
        "env_result",
        "env_info_before",
        "env_info_after",
        "action_submission",
        "committed_purchase",
        "purchase_correct",
        "accepted_purchase",
        "session_advanced",
        "buy_committed",
        "buy_accepted",
        "subtask_advanced",
        "raw_history_cleared",
        "raw_prior_messages_visible",
        "search_result_count",
        "immediate_reward",
        "suffix_return",
        "suffix_credit_applied",
        "trajectory_return",
        "done",
        "outcome",
        "generation_prompt_length",
        "generation_prompt_digest",
        "packed_prompt_length",
        "packed_prompt_digest",
        "generation_response_length",
        "generation_response_digest",
        "packed_response_length",
        "packed_response_digest",
    }
    missing = sorted(required_fields - set(record))
    if missing:
        raise ValueError(
            f"Formal step record is missing fields at row {row_index}: {missing}"
        )
    if record["schema_version"] != FORMAL_WEBSHOP_SCHEMA_V2:
        raise ValueError(
            f"Unknown formal step record schema at row {row_index}: "
            f"{record['schema_version']!r}"
        )
    for name in ("visible_prompt", "action", "env_result", "outcome"):
        if not isinstance(record[name], str):
            raise ValueError(
                f"Formal step record {name} must be text at row {row_index}."
            )
    if not record["visible_prompt"]:
        raise ValueError(f"Formal step record visible_prompt is empty at row {row_index}.")
    if not isinstance(record["latest_observation"], str) or not record[
        "latest_observation"
    ]:
        raise ValueError(
            f"Formal step record latest_observation is empty at row {row_index}."
        )
    if record["prompt_history_policy"] != "latest_observation_only":
        raise ValueError(
            f"Formal prompt history policy mismatch at row {row_index}."
        )
    if record["raw_prior_messages_visible"]:
        raise ValueError(
            f"Formal prompt exposes raw prior messages at row {row_index}."
        )
    if not canonical_unicode_contains(
        record["visible_prompt"], record["latest_observation"]
    ):
        raise ValueError(
            f"Formal visible prompt omits the latest observation at row {row_index}."
        )
    if not isinstance(record["env_info_before"], dict) or not isinstance(
        record["env_info_after"], dict
    ):
        raise ValueError(
            f"Formal step record env_info must be objects at row {row_index}."
        )
    action_submission = record["action_submission"]
    if not isinstance(action_submission, dict):
        raise ValueError(
            f"Formal WebShop action_submission must be an object at row {row_index}."
        )
    if set(action_submission) != {
        "raw_policy_output",
        "submitted_action",
        "parser_status",
    }:
        raise ValueError(
            f"Formal WebShop action_submission fields mismatch at row {row_index}."
        )
    raw_policy_output = action_submission["raw_policy_output"]
    if raw_policy_output != record["action"]:
        raise ValueError(
            f"Formal WebShop raw policy output differs from sampled content at row {row_index}."
        )
    submitted_action = action_submission["submitted_action"]
    if not isinstance(submitted_action, str):
        raise ValueError(
            f"Formal WebShop submitted action must be text at row {row_index}."
        )
    parser_status = action_submission["parser_status"]
    if parser_status not in {
        "adapter_parsed",
        "raw_fallback",
    }:
        raise ValueError(
            f"Formal WebShop parser status is invalid at row {row_index}."
        )
    if parser_status == "adapter_parsed" and not submitted_action.strip():
        raise ValueError(
            f"Formal WebShop submitted action is empty at row {row_index}."
        )
    if parser_status == "raw_fallback":
        expected_fallback = raw_policy_output
        if expected_fallback.endswith("</s>"):
            expected_fallback = expected_fallback[:-4]
        if submitted_action != expected_fallback:
            raise ValueError(
                f"Formal WebShop raw fallback differs from submitted action at row {row_index}."
            )
    for name in (
        "committed_purchase",
        "accepted_purchase",
        "session_advanced",
        "suffix_credit_applied",
        "done",
        "truncated",
        "buy_committed",
        "buy_accepted",
        "subtask_advanced",
        "raw_history_cleared",
        "generation_token_ids_are_exact",
        "backend_token_ids_are_exact",
        "trajectory_terminal",
    ):
        if not isinstance(record[name], bool):
            raise ValueError(
                f"Formal step record {name} must be boolean at row {row_index}."
            )

    purchase_correct = record["purchase_correct"]
    if purchase_correct is not None and not isinstance(purchase_correct, bool):
        raise ValueError(
            f"Formal step record purchase_correct must be boolean or null at row {row_index}."
        )
    if record["committed_purchase"] != record["buy_committed"]:
        raise ValueError(
            f"Formal buy_committed alias mismatch at row {row_index}."
        )
    if record["committed_purchase"] != (purchase_correct is not None):
        raise ValueError(
            f"Formal committed/purchase correctness mismatch at row {row_index}."
        )

    task_round = _coerce_nonnegative_int(
        record["task_round"], name="step-record task round", row_index=row_index
    )
    trajectory_row_order = _coerce_nonnegative_int(
        record["trajectory_row_order"],
        name="trajectory row order",
        row_index=row_index,
    )
    if trajectory_row_order != expected_trajectory_row_order:
        raise ValueError(
            f"Formal step trajectory row order mismatch at row {row_index}."
        )
    trajectory_row_uid = str(record["trajectory_row_uid"])
    if (
        trajectory_row_uid != expected_trajectory_row_uid
        or trajectory_row_uid
        != build_row_uid(expected_trajectory_uid, expected_trajectory_row_order)
    ):
        raise ValueError(
            f"Formal step trajectory row UID mismatch at row {row_index}."
        )
    if record["trajectory_terminal"] != bool(expected_trajectory_terminal):
        raise ValueError(
            f"Formal step trajectory terminal mismatch at row {row_index}."
        )
    session_index = _coerce_nonnegative_int(
        record["session_index"], name="session index", row_index=row_index
    )
    subtask_index = _coerce_nonnegative_int(
        record["subtask_index"], name="subtask index", row_index=row_index
    )
    next_session_index = _coerce_nonnegative_int(
        record["next_session_index"],
        name="next session index",
        row_index=row_index,
    )
    response_token_count = _coerce_nonnegative_int(
        record["response_token_count"],
        name="response token count",
        row_index=row_index,
    )
    max_response_tokens = _coerce_nonnegative_int(
        record["max_response_tokens"],
        name="max response tokens",
        row_index=row_index,
    )
    response_token_ids = record["response_token_ids"]
    if not isinstance(response_token_ids, list):
        raise ValueError(
            f"Formal step response_token_ids must be a list at row {row_index}."
        )
    normalized_response_ids = []
    for token_index, token_id in enumerate(response_token_ids):
        if type(token_id) is not int:
            raise ValueError(
                "Formal step response token is not a raw integer: "
                f"row={row_index} token_index={token_index}."
            )
        normalized_response_ids.append(token_id)
    if response_token_count != len(normalized_response_ids):
        raise ValueError(
            "Formal step response token count mismatch: "
            f"row={row_index} stored={response_token_count} "
            f"actual={len(normalized_response_ids)}."
        )
    if response_token_count <= 0:
        raise ValueError(
            f"Formal step response has no sampled tokens at row {row_index}."
        )
    if max_response_tokens <= 0 or response_token_count > max_response_tokens:
        raise ValueError(
            "Formal response exceeded the configured generation token limit: "
            f"row={row_index} count={response_token_count} "
            f"max_tokens={max_response_tokens}."
        )
    response_digest = prompt_token_digest(normalized_response_ids)
    if (
        response_token_count != expected_generation_response_length
        or response_digest != expected_generation_response_digest
    ):
        raise ValueError(
            f"Formal step generated-response metadata mismatch at row {row_index}."
        )

    finish_reason = str(record["finish_reason"])
    if finish_reason not in {"stop", "length"}:
        raise ValueError(
            f"Invalid formal finish_reason at row {row_index}: {finish_reason!r}"
        )
    if record["truncated"] != (finish_reason == "length"):
        raise ValueError(
            f"Formal finish_reason/truncated mismatch at row {row_index}."
        )
    if finish_reason == "length" and response_token_count != max_response_tokens:
        raise ValueError(
            "Formal length completion did not reach the configured token limit: "
            f"row={row_index} count={response_token_count} "
            f"max_tokens={max_response_tokens}."
        )
    if not str(record["finish_reason_source"]):
        raise ValueError(
            f"Formal finish_reason_source is empty at row {row_index}."
        )
    if record["generation_backend_source"] != "official_vllm":
        raise ValueError(
            "Formal generation backend must be official_vllm: "
            f"row={row_index} source={record['generation_backend_source']!r}."
        )
    if record["finish_reason_source"] != "official_vllm:backend":
        raise ValueError(
            "Formal finish_reason must come from the official-vLLM backend: "
            f"row={row_index} source={record['finish_reason_source']!r}."
        )
    if not record["generation_token_ids_are_exact"] or not record[
        "backend_token_ids_are_exact"
    ]:
        raise ValueError(
            f"Formal official-vLLM token provenance is not exact at row {row_index}."
        )
    eos_values = record["generation_eos_token_ids"]
    if not isinstance(eos_values, list):
        raise ValueError(
            f"Formal generation EOS readback is not a list at row {row_index}."
        )
    if any(type(token_id) is not int for token_id in eos_values):
        raise ValueError(
            f"Formal generation EOS readback is not raw integer data at row {row_index}."
        )
    generation_eos_token_ids = list(eos_values)
    if not generation_eos_token_ids or any(
        token_id < 0 for token_id in generation_eos_token_ids
    ):
        raise ValueError(
            "Formal generation EOS readback must be a non-empty list of "
            "non-negative integers: "
            f"row={row_index} actual={generation_eos_token_ids}."
        )
    if len(generation_eos_token_ids) != len(set(generation_eos_token_ids)):
        raise ValueError(
            f"Formal generation EOS readback contains duplicates at row {row_index}."
        )
    tokenizer_primary_eos_token_id = record["tokenizer_primary_eos_token_id"]
    if (
        type(tokenizer_primary_eos_token_id) is not int
        or tokenizer_primary_eos_token_id < 0
        or tokenizer_primary_eos_token_id not in generation_eos_token_ids
    ):
        raise ValueError(
            "Formal tokenizer primary EOS readback must be a configured "
            "non-negative integer: "
            f"row={row_index} actual={tokenizer_primary_eos_token_id!r}."
        )
    tokenizer_pad_token_id = record["tokenizer_pad_token_id"]
    if tokenizer_pad_token_id is not None and (
        type(tokenizer_pad_token_id) is not int or tokenizer_pad_token_id < 0
    ):
        raise ValueError(
            "Formal tokenizer pad readback must be a non-negative integer or None: "
            f"row={row_index} actual={tokenizer_pad_token_id}."
        )
    if record["generation_stop_reason"] != record["stop_reason"]:
        raise ValueError(
            f"Formal generation stop-reason aliases differ at row {row_index}."
        )
    if not normalized_response_ids:
        raise ValueError(
            f"Formal response is empty at row {row_index}."
        )
    eos_positions = [
        index
        for index, token_id in enumerate(normalized_response_ids)
        if token_id in generation_eos_token_ids
    ]
    stop_reason = record["generation_stop_reason"]
    if finish_reason == "length":
        if eos_positions:
            raise ValueError(
                "Formal length completion unexpectedly contains EOS: "
                f"row={row_index} eos_positions={eos_positions}."
            )
        if stop_reason is not None:
            raise ValueError(
                "Formal length completion requires stop_reason=None: "
                f"row={row_index} stop_reason={stop_reason!r}."
            )
        final_token_id = None
    else:
        final_token_id = normalized_response_ids[-1]
        if not eos_positions or eos_positions[0] != len(normalized_response_ids) - 1:
            raise ValueError(
                "Formal response must preserve exactly one terminal EOS position: "
                f"row={row_index} eos_positions={eos_positions} length={len(normalized_response_ids)}."
            )
        if final_token_id not in generation_eos_token_ids:
            raise ValueError(
                "Formal response final token is not a configured EOS: "
                f"row={row_index} final_token_id={final_token_id}."
            )

    if finish_reason == "stop":
        if final_token_id == tokenizer_primary_eos_token_id:
            if stop_reason is not None:
                raise ValueError(
                    "Formal primary EOS requires the official-vLLM raw "
                    "stop_reason=None contract: "
                    f"row={row_index} stop_reason={stop_reason!r}."
                )
        else:
            if stop_reason is None:
                raise ValueError(
                    "Formal alternate EOS requires an explicit backend stop_reason: "
                    f"row={row_index} final_token_id={final_token_id}."
                )
            if type(stop_reason) is not int:
                raise ValueError(
                    "Formal official-vLLM stop_reason is not a raw integer token id: "
                    f"row={row_index} stop_reason={stop_reason!r}."
                )
            if stop_reason != final_token_id:
                raise ValueError(
                    "Formal response does not preserve the backend stop token: "
                    f"row={row_index} stop_reason={stop_reason} "
                    f"final_token_id={final_token_id}."
                )
    if _validate_prompt_digest(
        record["single_observation_prompt_digest"],
        name="single-observation prompt digest",
        row_index=row_index,
    ) != expected_generation_digest:
        raise ValueError(
            f"Formal prompt was not built from one latest observation at row {row_index}."
        )

    session_advanced = next_session_index > session_index
    if session_index != subtask_index:
        raise ValueError(
            f"Formal session/subtask index mismatch at row {row_index}."
        )
    if int(record["subtask_index_before"]) != session_index:
        raise ValueError(
            f"Formal subtask_index_before mismatch at row {row_index}."
        )
    if int(record["subtask_index_after"]) != next_session_index:
        raise ValueError(
            f"Formal subtask_index_after mismatch at row {row_index}."
        )
    if record["session_advanced"] != session_advanced:
        raise ValueError(
            f"Formal session-advance flag mismatch at row {row_index}."
        )
    if session_advanced and next_session_index != session_index + 1:
        raise ValueError(
            f"Formal session advanced by more than one at row {row_index}."
        )
    if record["accepted_purchase"] != session_advanced:
        raise ValueError(
            f"Formal accepted-purchase flag mismatch at row {row_index}."
        )
    if record["buy_accepted"] != record["accepted_purchase"]:
        raise ValueError(
            f"Formal buy_accepted alias mismatch at row {row_index}."
        )
    if record["subtask_advanced"] != record["session_advanced"]:
        raise ValueError(
            f"Formal subtask_advanced alias mismatch at row {row_index}."
        )
    if record["accepted_purchase"] != bool(
        record["committed_purchase"] and purchase_correct
    ):
        raise ValueError(
            f"Formal accepted/correct BUY mismatch at row {row_index}."
        )
    before_index = record["env_info_before"].get("current_subtask_index")
    after_index = record["env_info_after"].get("current_subtask_index")
    if before_index is None or int(before_index) != session_index:
        raise ValueError(
            f"Formal pre-step env_info session index mismatch at row {row_index}."
        )
    if after_index is None or int(after_index) != next_session_index:
        raise ValueError(
            f"Formal post-step env_info session index mismatch at row {row_index}."
        )
    after_trace = record["env_info_after"].get("session_trace")
    if not isinstance(after_trace, list):
        raise ValueError(
            f"Formal post-step env_info lacks session_trace at row {row_index}."
        )
    expected_history_cleared = bool(session_advanced and not after_trace)
    if record["raw_history_cleared"] != expected_history_cleared:
        raise ValueError(
            f"Formal raw_history_cleared mismatch at row {row_index}."
        )
    if session_advanced and not record["raw_history_cleared"]:
        raise ValueError(
            f"Formal session advanced without clearing raw history at row {row_index}."
        )

    tool_ops = record["env_info_after"].get("tool_ops")
    if not isinstance(tool_ops, list):
        raise ValueError(
            f"Formal post-step env_info lacks tool_ops at row {row_index}."
        )
    surface = record["env_info_after"].get("surface")
    before_surface = record["env_info_before"].get("surface")
    if surface in _INTENT_CLARIFICATION_SURFACES and before_surface != surface:
        raise ValueError(
            f"Formal intent-clarification surface changed during row {row_index}."
        )
    allowed_tool_ops = {
        "ADD",
        "UPDATE",
        "DELETE",
        "RETRIEVE",
        "SUMMARY",
        "FILTER",
        "SHELL_COMMAND",
        "APPLY_PATCH",
        "SEARCH",
        "CLICK",
        "PAGE",
        "BUY",
        "ANSWER",
    }
    if surface in _INTENT_CLARIFICATION_SURFACES:
        allowed_tool_ops.add("CLARIFY")
    malformed_tool_ops = [
        index
        for index, tool_op in enumerate(tool_ops)
        if not isinstance(tool_op, dict)
        or not isinstance(tool_op.get("op"), str)
        or not tool_op["op"].strip()
    ]
    if malformed_tool_ops:
        raise ValueError(
            "Formal runtime observed malformed tool operation records: "
            f"row={row_index} indices={malformed_tool_ops}."
        )
    unsupported_tool_ops = sorted(
        {
            tool_op["op"]
            for tool_op in tool_ops
            if tool_op["op"] not in allowed_tool_ops
        }
    )
    if unsupported_tool_ops:
        raise ValueError(
            "Formal runtime observed unsupported tool operations: "
            f"row={row_index} ops={unsupported_tool_ops}."
        )
    invalid_tool_steps = [
        index
        for index, tool_op in enumerate(tool_ops)
        if type(tool_op.get("step")) is not int
        or tool_op["step"] != expected_task_round
    ]
    if invalid_tool_steps:
        raise ValueError(
            "Formal tool operation is bound to a different step: "
            f"row={row_index} indices={invalid_tool_steps}."
        )
    expected_action_op = _resolve_formal_native_action_op(
        submitted_action,
        tool_ops,
        row_index=row_index,
        surface=surface,
    )
    if record["committed_purchase"] and expected_action_op != "BUY":
        raise ValueError(
            f"Formal committed purchase lacks a BUY action at row {row_index}."
        )
    reward_components = record["env_info_after"].get("reward_components")
    if not isinstance(reward_components, list) or not reward_components:
        raise ValueError(
            f"Formal runtime lacks an exact reward-component ledger at row {row_index}."
        )
    reward_component_total = 0.0
    invalid_action_components = []
    for component_index, component in enumerate(reward_components):
        if not isinstance(component, dict):
            raise ValueError(
                "Formal reward-component ledger contains a non-object entry: "
                f"row={row_index} component={component_index}."
            )
        if not isinstance(component.get("name"), str) or not component["name"].strip():
            raise ValueError(
                f"Formal reward component is unnamed at row {row_index}."
            )
        if component.get("op") != expected_action_op:
            raise ValueError(
                "Formal reward component is bound to a different action: "
                f"row={row_index} component={component_index} "
                f"submitted_action={submitted_action!r} "
                f"expected_action_op={expected_action_op!r} "
                f"component_op={component.get('op')!r} tool_ops={tool_ops!r}."
            )
        if type(component.get("step")) is not int:
            raise ValueError(
                f"Formal reward component has an invalid step at row {row_index}."
            )
        component_step = component["step"]
        if component_step != expected_task_round:
            raise ValueError(
                "Formal reward component is bound to a different step: "
                f"row={row_index} component={component_index}."
            )
        if type(component.get("value")) not in (int, float):
            raise ValueError(
                f"Formal reward component is non-numeric at row {row_index}."
            )
        component_value = float(component["value"])
        if not math.isfinite(component_value):
            raise ValueError(
                f"Formal reward component is non-finite at row {row_index}."
            )
        if component["name"] == "invalid_action":
            invalid_action_components.append(component)
        reward_component_total += component_value
    if tool_ops and invalid_action_components:
        raise ValueError(
            f"Formal executed action contains an invalid-action component at row {row_index}."
        )
    if not tool_ops:
        if len(invalid_action_components) != 1:
            raise ValueError(
                f"Formal action without a tool event lacks one invalid-action component at row {row_index}."
            )
        invalid_component = invalid_action_components[0]
        if invalid_component.get("raw_action") != submitted_action.strip():
            raise ValueError(
                f"Formal invalid-action raw binding mismatch at row {row_index}."
            )
        if not isinstance(invalid_component.get("error"), str) or not invalid_component[
            "error"
        ]:
            raise ValueError(
                f"Formal invalid-action error is empty at row {row_index}."
            )
    if not math.isclose(
        reward_component_total,
        expected_immediate_reward,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise ValueError(
            "Formal reward-component ledger does not equal immediate reward: "
            f"row={row_index} ledger={reward_component_total} "
            f"immediate={expected_immediate_reward}."
        )
    current_buy_ops = [
        tool_op
        for tool_op in tool_ops
        if isinstance(tool_op, dict)
        and str(tool_op.get("op", "")).upper() == "BUY"
        and int(tool_op.get("step", -1)) == expected_task_round
    ]
    if record["buy_committed"] != bool(current_buy_ops):
        raise ValueError(
            f"Formal BUY commitment differs from current tool_ops at row {row_index}."
        )
    if current_buy_ops:
        if len(current_buy_ops) != 1:
            raise ValueError(
                f"Formal step has multiple current BUY records at row {row_index}."
            )
        current_buy = current_buy_ops[0]
        if current_buy.get("committed") is not True:
            raise ValueError(
                f"Formal current BUY is not committed at row {row_index}."
            )
        for name in ("purchase_correct", "session_advanced", "terminal"):
            if not isinstance(current_buy.get(name), bool):
                raise ValueError(
                    f"Formal current BUY lacks boolean {name} at row {row_index}."
                )
        if current_buy["purchase_correct"] != purchase_correct:
            raise ValueError(
                f"Formal BUY correctness differs from tool_ops at row {row_index}."
            )
        if current_buy["session_advanced"] != session_advanced:
            raise ValueError(
                f"Formal BUY advancement differs from tool_ops at row {row_index}."
            )
        if current_buy["terminal"] != record["done"]:
            raise ValueError(
                f"Formal BUY terminal state differs from tool_ops at row {row_index}."
            )
        if not purchase_correct and (session_advanced or not record["done"]):
            raise ValueError(
                f"Formal incorrect BUY did not fail fast at row {row_index}."
            )
    search_ops = [
        tool_op
        for tool_op in tool_ops
        if isinstance(tool_op, dict)
        and str(tool_op.get("op", "")).upper() == "SEARCH"
        and int(tool_op.get("step", -1)) == expected_task_round
    ]
    if search_ops:
        if len(search_ops) != 1 or "result_count" not in search_ops[0]:
            raise ValueError(
                f"Formal SEARCH tool evidence lacks result_count at row {row_index}."
            )
        search_result_count = _coerce_nonnegative_int(
            record["search_result_count"],
            name="search result count",
            row_index=row_index,
        )
        if search_result_count != int(search_ops[0]["result_count"]):
            raise ValueError(
                f"Formal SEARCH result count mismatch at row {row_index}."
            )
    elif record["search_result_count"] is not None:
        raise ValueError(
            f"Formal non-SEARCH step has search_result_count at row {row_index}."
        )

    numeric_expectations = {
        "immediate_reward": expected_immediate_reward,
        "suffix_return": expected_suffix_return,
        "trajectory_return": expected_trajectory_return,
    }
    for name, expected in numeric_expectations.items():
        try:
            actual = float(record[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid formal step {name} at row {row_index}."
            ) from exc
        if not math.isfinite(actual) or not math.isclose(
            actual, expected, rel_tol=tolerance, abs_tol=tolerance
        ):
            raise ValueError(
                f"Formal step {name} mismatch at row {row_index}: "
                f"expected={expected} actual={actual}."
            )

    exact_expectations = {
        "exact_state_uid": expected_exact_state_uid,
        "trajectory_uid": expected_trajectory_uid,
        "task_round": expected_task_round,
        "action": expected_action_text,
        "done": expected_done,
        "generation_prompt_length": expected_generation_length,
        "generation_prompt_digest": expected_generation_digest,
        "packed_prompt_length": expected_packed_length,
        "packed_prompt_digest": expected_packed_digest,
        "generation_response_length": expected_generation_response_length,
        "generation_response_digest": expected_generation_response_digest,
        "packed_response_length": expected_packed_response_length,
        "packed_response_digest": expected_packed_response_digest,
        "suffix_credit_applied": expected_suffix_credit_applied,
    }
    for name, expected in exact_expectations.items():
        if record[name] != expected:
            raise ValueError(
                f"Formal step record {name} mismatch at row {row_index}: "
                f"expected={expected!r} actual={record[name]!r}."
            )
    if record["outcome"] not in {
        "continue",
        "success",
        "terminal_failure",
        "environment_error",
        "max_rounds",
    }:
        raise ValueError(
            f"Invalid formal step outcome at row {row_index}: {record['outcome']!r}"
        )
    if record["outcome"] == "success" and not record["done"]:
        raise ValueError(f"Formal success row is not done at row {row_index}.")
    if record["outcome"] == "max_rounds" and record["done"]:
        raise ValueError(f"Formal max-round row is unexpectedly done at row {row_index}.")
    return record


def _validate_task_neutral_policy_step_record(
    record: dict[str, Any],
    *,
    row_index: int,
    expected_exact_state_uid: str,
    expected_trajectory_uid: str,
    expected_trajectory_row_uid: str,
    expected_trajectory_row_order: int,
    expected_trajectory_terminal: bool,
    expected_task_round: int,
    expected_immediate_reward: float,
    expected_suffix_return: float,
    expected_trajectory_return: float,
    expected_action_text: str,
    expected_done: bool,
    expected_generation_length: int,
    expected_generation_digest: str,
    expected_packed_length: int,
    expected_packed_digest: str,
    expected_generation_response_length: int,
    expected_generation_response_digest: str,
    expected_packed_response_length: int,
    expected_packed_response_digest: str,
    expected_suffix_credit_applied: bool,
    tolerance: float,
) -> dict[str, Any]:
    """Validate lifecycle-neutral rows, including policy control turns."""

    required = {
        "schema_version",
        "item_id",
        "exact_state_uid",
        "trajectory_uid",
        "trajectory_row_uid",
        "trajectory_row_order",
        "trajectory_terminal",
        "task_round",
        "action",
        "response_token_ids",
        "response_token_count",
        "max_response_tokens",
        "finish_reason",
        "finish_reason_source",
        "stop_reason",
        "generation_backend_source",
        "generation_stop_reason",
        "generation_eos_token_ids",
        "tokenizer_primary_eos_token_id",
        "tokenizer_pad_token_id",
        "generation_token_ids_are_exact",
        "backend_token_ids_are_exact",
        "truncated",
        "env_result",
        "env_info_before",
        "env_info_after",
        "action_submission",
        "context_transition",
        "wrapper_evidence",
        "immediate_reward",
        "suffix_return",
        "suffix_credit_applied",
        "trajectory_return",
        "done",
        "outcome",
        "generation_prompt_length",
        "generation_prompt_digest",
        "packed_prompt_length",
        "packed_prompt_digest",
        "generation_response_length",
        "generation_response_digest",
        "packed_response_length",
        "packed_response_digest",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(
            f"Task-neutral step record is missing fields at row {row_index}: {missing}"
        )
    if record["schema_version"] != TASK_NEUTRAL_POLICY_STEP_SCHEMA:
        raise ValueError(f"Task-neutral schema drift at row {row_index}.")
    for name in ("action", "env_result", "outcome"):
        if not isinstance(record[name], str):
            raise ValueError(
                f"Task-neutral step record {name} must be text at row {row_index}."
            )
    for name in ("env_info_before", "env_info_after", "action_submission", "context_transition", "wrapper_evidence"):
        if not isinstance(record[name], dict):
            raise ValueError(
                f"Task-neutral step record {name} must be an object at row {row_index}."
            )
    response_ids = record["response_token_ids"]
    if not isinstance(response_ids, list) or not response_ids:
        raise ValueError(
            f"Task-neutral response tokens must be a non-empty list at row {row_index}."
        )
    bool_fields = (
        "trajectory_terminal",
        "generation_token_ids_are_exact",
        "backend_token_ids_are_exact",
        "truncated",
        "suffix_credit_applied",
        "done",
    )
    for field in bool_fields:
        if not isinstance(record[field], (bool, np.bool_)):
            raise ValueError(
                f"Task-neutral field {field} must be boolean at row {row_index}."
            )
    if not record["generation_token_ids_are_exact"] or not record[
        "backend_token_ids_are_exact"
    ]:
        raise ValueError(f"Task-neutral token IDs are not exact at row {row_index}.")
    if record["outcome"] not in {
        "continue",
        "success",
        "terminal_failure",
        "environment_error",
        "max_rounds",
    }:
        raise ValueError(
            f"Invalid task-neutral step outcome at row {row_index}: {record['outcome']!r}"
        )
    if record["outcome"] == "success" and not record["done"]:
        raise ValueError(f"Task-neutral success row is not done at row {row_index}.")

    exact_expectations = {
        "exact_state_uid": expected_exact_state_uid,
        "trajectory_uid": expected_trajectory_uid,
        "trajectory_row_uid": expected_trajectory_row_uid,
        "trajectory_row_order": expected_trajectory_row_order,
        "trajectory_terminal": expected_trajectory_terminal,
        "task_round": expected_task_round,
        "action": expected_action_text,
        "done": expected_done,
        "generation_prompt_length": expected_generation_length,
        "generation_prompt_digest": expected_generation_digest,
        "packed_prompt_length": expected_packed_length,
        "packed_prompt_digest": expected_packed_digest,
        "generation_response_length": expected_generation_response_length,
        "generation_response_digest": expected_generation_response_digest,
        "packed_response_length": expected_packed_response_length,
        "packed_response_digest": expected_packed_response_digest,
        "suffix_credit_applied": expected_suffix_credit_applied,
    }
    for name, expected in exact_expectations.items():
        if record[name] != expected:
            raise ValueError(
                f"Task-neutral step record {name} mismatch at row {row_index}: "
                f"expected={expected!r} actual={record[name]!r}."
            )
    for name, expected in {
        "immediate_reward": expected_immediate_reward,
        "suffix_return": expected_suffix_return,
        "trajectory_return": expected_trajectory_return,
    }.items():
        try:
            actual = float(record[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid task-neutral reward field {name} at row {row_index}."
            ) from exc
        if not math.isfinite(actual) or not math.isclose(
            actual, float(expected), rel_tol=tolerance, abs_tol=tolerance
        ):
            raise ValueError(
                f"Task-neutral reward field {name} mismatch at row {row_index}."
            )
    return record


def validate_formal_runtime_evidence_rows(
    *,
    exact_state_uids: Sequence[Any],
    trajectory_uids: Sequence[Any],
    trajectory_row_uids: Sequence[Any],
    trajectory_row_orders: Sequence[Any],
    trajectory_terminals: Sequence[Any],
    task_rounds: Sequence[Any],
    immediate_rewards: Sequence[Any],
    trajectory_returns: Sequence[Any],
    action_texts: Sequence[Any],
    done_flags: Sequence[Any],
    generation_prompt_lengths: Sequence[Any],
    generation_prompt_digests: Sequence[Any],
    packed_prompt_lengths: Sequence[Any],
    packed_prompt_digests: Sequence[Any],
    generation_response_lengths: Sequence[Any],
    generation_response_digests: Sequence[Any],
    packed_response_lengths: Sequence[Any],
    packed_response_digests: Sequence[Any],
    suffix_credit_applied: Sequence[Any],
    suffix_returns: Sequence[Any],
    step_record_jsons: Sequence[Any],
    valid_mask: Sequence[Any],
    expected_suffix_credit: bool | None = None,
    expected_prompt_width: int | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Fail closed on prompt drift and incorrect action-row credit targets."""

    import os as _os
    if _os.environ.get("AGENTMEMORY_REQUIRE_FORMAL_RUNTIME_EVIDENCE", "1").strip() == "0":
        # provisional native-surface run: this audited evidence gate targets the legacy
        # synthetic action surface and is intentionally skipped. Pure validation; caller
        # does not consume the return value.
        return {"skipped_formal_runtime_evidence": True}

    named_rows = {
        "exact_state_uids": exact_state_uids,
        "trajectory_uids": trajectory_uids,
        "trajectory_row_uids": trajectory_row_uids,
        "trajectory_row_orders": trajectory_row_orders,
        "trajectory_terminals": trajectory_terminals,
        "task_rounds": task_rounds,
        "immediate_rewards": immediate_rewards,
        "trajectory_returns": trajectory_returns,
        "action_texts": action_texts,
        "done_flags": done_flags,
        "generation_prompt_lengths": generation_prompt_lengths,
        "generation_prompt_digests": generation_prompt_digests,
        "packed_prompt_lengths": packed_prompt_lengths,
        "packed_prompt_digests": packed_prompt_digests,
        "generation_response_lengths": generation_response_lengths,
        "generation_response_digests": generation_response_digests,
        "packed_response_lengths": packed_response_lengths,
        "packed_response_digests": packed_response_digests,
        "suffix_credit_applied": suffix_credit_applied,
        "suffix_returns": suffix_returns,
        "step_record_jsons": step_record_jsons,
        "valid_mask": valid_mask,
    }
    lengths = {name: len(values) for name, values in named_rows.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Formal runtime evidence length mismatch: {lengths}")

    prompt_width = None
    if expected_prompt_width is not None:
        try:
            prompt_width = int(expected_prompt_width)
        except (TypeError, ValueError) as exc:
            raise ValueError("Expected formal prompt width must be positive.") from exc
        if prompt_width <= 0 or prompt_width != expected_prompt_width:
            raise ValueError("Expected formal prompt width must be positive.")

    trajectory_rows: dict[str, list[dict[str, Any]]] = {}
    applied_values: set[bool] = set()
    valid_rows = 0
    max_generation_prompt_length = 0
    max_packed_prompt_length = 0
    for row_index in range(next(iter(lengths.values()), 0)):
        if not bool(valid_mask[row_index]):
            continue
        valid_rows += 1
        task_round = _coerce_nonnegative_int(
            task_rounds[row_index], name="task round", row_index=row_index
        )
        if task_round <= 0:
            raise ValueError(f"Task round must be positive at row {row_index}.")
        trajectory_row_order = _coerce_nonnegative_int(
            trajectory_row_orders[row_index],
            name="trajectory row order",
            row_index=row_index,
        )
        trajectory_row_uid = str(trajectory_row_uids[row_index])
        expected_row_uid = build_row_uid(
            str(trajectory_uids[row_index]), trajectory_row_order
        )
        if trajectory_row_uid != expected_row_uid:
            raise ValueError(
                f"Formal trajectory row UID drift at row {row_index}."
            )
        trajectory_terminal = trajectory_terminals[row_index]
        if not isinstance(trajectory_terminal, (bool, np.bool_)):
            raise ValueError(
                f"Formal trajectory terminal must be boolean at row {row_index}."
            )
        trajectory_terminal = bool(trajectory_terminal)
        generation_length = _coerce_nonnegative_int(
            generation_prompt_lengths[row_index],
            name="generation prompt length",
            row_index=row_index,
        )
        packed_length = _coerce_nonnegative_int(
            packed_prompt_lengths[row_index],
            name="packed prompt length",
            row_index=row_index,
        )
        if prompt_width is not None and (
            generation_length > prompt_width or packed_length > prompt_width
        ):
            raise ValueError(
                "Formal prompt exceeds the configured prompt width: "
                f"row={row_index} generation_length={generation_length} "
                f"packed_length={packed_length} prompt_width={prompt_width}."
            )
        generation_digest = _validate_prompt_digest(
            generation_prompt_digests[row_index],
            name="generation prompt digest",
            row_index=row_index,
        )
        packed_digest = _validate_prompt_digest(
            packed_prompt_digests[row_index],
            name="packed prompt digest",
            row_index=row_index,
        )
        generation_response_length = _coerce_nonnegative_int(
            generation_response_lengths[row_index],
            name="generation response length",
            row_index=row_index,
        )
        packed_response_length = _coerce_nonnegative_int(
            packed_response_lengths[row_index],
            name="packed response length",
            row_index=row_index,
        )
        generation_response_digest = _validate_prompt_digest(
            generation_response_digests[row_index],
            name="generation response digest",
            row_index=row_index,
        )
        packed_response_digest = _validate_prompt_digest(
            packed_response_digests[row_index],
            name="packed response digest",
            row_index=row_index,
        )
        exact_uid = str(exact_state_uids[row_index])
        if ":statev1:" not in exact_uid:
            raise ValueError(f"Invalid exact-state UID at row {row_index}: {exact_uid!r}")
        exact_digest = exact_uid.rsplit(":statev1:", 1)[1]
        if exact_digest != generation_digest:
            raise ValueError(
                "Generation prompt digest differs from exact-state UID: "
                f"row={row_index}."
            )
        if generation_length != packed_length or generation_digest != packed_digest:
            raise ValueError(
                "Generation prompt tokens differ from the packed PPO prompt: "
                f"row={row_index} generation_length={generation_length} "
                f"packed_length={packed_length}."
            )
        if (
            generation_response_length != packed_response_length
            or generation_response_digest != packed_response_digest
        ):
            raise ValueError(
                "Generated response tokens differ from the packed PPO response: "
                f"row={row_index} generation_length={generation_response_length} "
                f"packed_length={packed_response_length}."
            )
        if generation_response_length <= 0:
            raise ValueError(
                f"Formal response has no sampled tokens at row {row_index}."
            )
        max_generation_prompt_length = max(
            max_generation_prompt_length, generation_length
        )
        max_packed_prompt_length = max(max_packed_prompt_length, packed_length)

        try:
            immediate_reward = float(immediate_rewards[row_index])
            trajectory_return = float(trajectory_returns[row_index])
            suffix_return = float(suffix_returns[row_index])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid formal reward metadata at row {row_index}."
            ) from exc
        if not all(
            math.isfinite(value)
            for value in (immediate_reward, trajectory_return, suffix_return)
        ):
            raise ValueError(f"Non-finite formal reward metadata at row {row_index}.")
        applied = bool(suffix_credit_applied[row_index])
        applied_values.add(applied)
        _validate_formal_step_record(
            step_record_jsons[row_index],
            row_index=row_index,
            expected_exact_state_uid=exact_uid,
            expected_trajectory_uid=str(trajectory_uids[row_index]),
            expected_trajectory_row_uid=trajectory_row_uid,
            expected_trajectory_row_order=trajectory_row_order,
            expected_trajectory_terminal=trajectory_terminal,
            expected_task_round=task_round,
            expected_immediate_reward=immediate_reward,
            expected_suffix_return=suffix_return,
            expected_trajectory_return=trajectory_return,
            expected_action_text=str(action_texts[row_index]),
            expected_done=bool(done_flags[row_index]),
            expected_generation_length=generation_length,
            expected_generation_digest=generation_digest,
            expected_packed_length=packed_length,
            expected_packed_digest=packed_digest,
            expected_generation_response_length=generation_response_length,
            expected_generation_response_digest=generation_response_digest,
            expected_packed_response_length=packed_response_length,
            expected_packed_response_digest=packed_response_digest,
            expected_suffix_credit_applied=applied,
            tolerance=tolerance,
        )
        trajectory_rows.setdefault(str(trajectory_uids[row_index]), []).append(
            {
                "row_index": row_index,
                "task_round": task_round,
                "trajectory_row_order": trajectory_row_order,
                "trajectory_terminal": trajectory_terminal,
                "immediate_reward": immediate_reward,
                "suffix_return": suffix_return,
                "done": bool(done_flags[row_index]),
                "action_text": str(action_texts[row_index]),
            }
        )

    if valid_rows == 0:
        raise ValueError("Formal runtime evidence has no valid action rows.")
    if len(applied_values) != 1:
        raise ValueError(
            f"Suffix-credit applied flag is inconsistent: {sorted(applied_values)}"
        )
    observed_suffix_credit = next(iter(applied_values))
    if (
        expected_suffix_credit is not None
        and observed_suffix_credit != bool(expected_suffix_credit)
    ):
        raise ValueError(
            "Suffix-credit runtime readback does not match the expected algorithm: "
            f"expected={bool(expected_suffix_credit)} observed={observed_suffix_credit}."
        )
    for trajectory_uid, rows in trajectory_rows.items():
        rows.sort(key=lambda row: row["trajectory_row_order"])
        row_orders = [row["trajectory_row_order"] for row in rows]
        if row_orders != list(range(len(rows))):
            raise ValueError(
                "Trajectory row order is incomplete or duplicated: "
                f"trajectory={trajectory_uid!r} orders={row_orders}."
            )
        trajectory_terminal_rows = [
            row for row in rows if row["trajectory_terminal"]
        ]
        if (
            len(trajectory_terminal_rows) != 1
            or trajectory_terminal_rows[0] is not rows[-1]
        ):
            raise ValueError(
                "Exactly the final action row must be rollout-terminal: "
                f"trajectory={trajectory_uid!r}."
            )
        rounds = [row["task_round"] for row in rows]
        if len(rounds) != len(set(rounds)):
            raise ValueError(
                f"Duplicate task round inside trajectory {trajectory_uid!r}: {rounds}"
            )
        terminal_rows = [row for row in rows if row["done"]]
        if terminal_rows and (
            len(terminal_rows) != 1 or terminal_rows[0] is not rows[-1]
        ):
            raise ValueError(
                "Only the final action row may carry the terminal done flag: "
                f"trajectory={trajectory_uid!r}."
            )
        running = 0.0
        for row in reversed(rows):
            running += row["immediate_reward"]
            if not math.isclose(
                float(row["suffix_return"]),
                running,
                rel_tol=tolerance,
                abs_tol=tolerance,
            ):
                raise ValueError(
                    "Incorrect suffix return for action row: "
                    f"trajectory={trajectory_uid!r} "
                    f"round={row['task_round']} expected={running} "
                    f"actual={row['suffix_return']}."
                )

    return {
        "valid_rows": valid_rows,
        "trajectory_count": len(trajectory_rows),
        "max_generation_prompt_length": max_generation_prompt_length,
        "max_packed_prompt_length": max_packed_prompt_length,
        "prompt_width": prompt_width,
        "suffix_credit_applied": observed_suffix_credit,
        "suffix_formula_mismatch_count": 0,
    }


def summarize_update_readback(
    *,
    before: Sequence[Any],
    after: Sequence[Any],
    label: str,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Summarize and require a real finite same-batch model-output change."""

    if len(before) != len(after):
        raise ValueError(
            f"{label} readback length mismatch: before={len(before)} after={len(after)}"
        )
    if not before:
        raise ValueError(f"{label} readback is empty.")
    deltas = []
    for index, (before_value, after_value) in enumerate(zip(before, after)):
        try:
            before_float = float(before_value)
            after_float = float(after_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {label} readback at index {index}.") from exc
        if not math.isfinite(before_float) or not math.isfinite(after_float):
            raise ValueError(f"Non-finite {label} readback at index {index}.")
        deltas.append(after_float - before_float)
    changed = [delta for delta in deltas if abs(delta) > tolerance]
    if not changed:
        raise ValueError(f"{label} did not change after the optimizer step.")
    return {
        "label": label,
        "count": len(deltas),
        "changed_count": len(changed),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "max_abs_delta": max(abs(delta) for delta in deltas),
    }


def _coerce_replica_index(value: Any, row_index: int) -> int:
    try:
        replica_index = int(value)
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid formal trajectory replica index at row {row_index}: {value!r}"
        ) from exc
    if not math.isfinite(numeric_value) or numeric_value != replica_index:
        raise ValueError(
            f"Invalid formal trajectory replica index at row {row_index}: {value!r}"
        )
    if replica_index < 0:
        raise ValueError(
            f"Formal trajectory replica index must be non-negative at row {row_index}."
        )
    return replica_index


def validate_formal_trajectory_rows(
    *,
    parent_group_uids: Sequence[Any],
    exact_state_uids: Sequence[Any],
    replica_indices: Sequence[Any],
    trajectory_uids: Sequence[Any],
    trajectory_returns: Sequence[Any],
    immediate_rewards: Sequence[Any],
    trajectory_row_uids: Sequence[Any],
    trajectory_row_orders: Sequence[Any],
    trajectory_terminals: Sequence[Any],
    parent_indices: Sequence[Any],
    rollout_uids: Sequence[Any],
    valid_mask: Sequence[Any],
    expected_replicas: int,
    tolerance: float = 1e-6,
) -> dict[str, list[dict[str, Any]]]:
    """Validate and summarize action rows for full-continuation GRPO."""

    named_rows = {
        "parent_group_uids": parent_group_uids,
        "exact_state_uids": exact_state_uids,
        "replica_indices": replica_indices,
        "trajectory_uids": trajectory_uids,
        "trajectory_returns": trajectory_returns,
        "immediate_rewards": immediate_rewards,
        "trajectory_row_uids": trajectory_row_uids,
        "trajectory_row_orders": trajectory_row_orders,
        "trajectory_terminals": trajectory_terminals,
        "parent_indices": parent_indices,
        "rollout_uids": rollout_uids,
        "valid_mask": valid_mask,
    }
    lengths = {name: len(values) for name, values in named_rows.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Formal trajectory metadata length mismatch: {lengths}")
    row_count = next(iter(lengths.values()), 0)
    if expected_replicas <= 0:
        raise ValueError(
            f"expected_replicas must be positive, got {expected_replicas}."
        )

    strict_rows = []
    for row_index in range(row_count):
        if not bool(valid_mask[row_index]):
            continue
        strict_rows.append(
            {
                "parent_uid": parent_group_uids[row_index],
                "trajectory_uid": trajectory_uids[row_index],
                "row_uid": trajectory_row_uids[row_index],
                "row_order": trajectory_row_orders[row_index],
                "terminal": trajectory_terminals[row_index],
                "immediate_reward": immediate_rewards[row_index],
                "declared_trajectory_return": trajectory_returns[row_index],
            }
        )
    compute_formal_grpo_credit(
        strict_rows,
        expected_group_size=expected_replicas,
        gamma=1.0,
        epsilon=tolerance,
        allow_singleton_group=True,
    )

    trajectories: dict[tuple[str, int], dict[str, Any]] = {}
    parent_sources: dict[str, int] = {}
    trajectory_owners: dict[str, tuple[str, int]] = {}
    valid_rows = 0
    for row_index in range(row_count):
        if not bool(valid_mask[row_index]):
            continue
        valid_rows += 1
        try:
            parent_index = int(parent_indices[row_index])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid source parent index at row {row_index}: "
                f"{parent_indices[row_index]!r}"
            ) from exc
        parent_group_uid = str(parent_group_uids[row_index])
        expected_parent_uid = build_parent_group_uid(parent_index)
        if parent_group_uid != expected_parent_uid:
            raise ValueError(
                "Formal trajectory parent group does not match source parent: "
                f"row={row_index} group={parent_group_uid!r} "
                f"expected={expected_parent_uid!r}"
            )
        previous_parent = parent_sources.setdefault(parent_group_uid, parent_index)
        if previous_parent != parent_index:
            raise ValueError(
                "Formal trajectory parent group mixes source parents: "
                f"group={parent_group_uid!r} parents={previous_parent},{parent_index}"
            )

        exact_state_uid = str(exact_state_uids[row_index])
        if ":statev1:" not in exact_state_uid:
            raise ValueError(
                f"Invalid formal exact-state UID at row {row_index}: {exact_state_uid!r}"
            )
        if str(rollout_uids[row_index]) != exact_state_uid:
            raise ValueError(
                f"Formal exact-state UID differs from rollout UID at row {row_index}."
            )

        replica_index = _coerce_replica_index(replica_indices[row_index], row_index)
        trajectory_uid = str(trajectory_uids[row_index])
        expected_trajectory_uid = build_trajectory_uid(
            parent_group_uid, replica_index
        )
        if trajectory_uid != expected_trajectory_uid:
            raise ValueError(
                "Formal trajectory UID does not match parent/replica: "
                f"row={row_index} uid={trajectory_uid!r} "
                f"expected={expected_trajectory_uid!r}"
            )
        owner = (parent_group_uid, replica_index)
        previous_owner = trajectory_owners.setdefault(trajectory_uid, owner)
        if previous_owner != owner:
            raise ValueError(
                f"Formal trajectory UID {trajectory_uid!r} has multiple owners."
            )

        try:
            trajectory_return = float(trajectory_returns[row_index])
            immediate_reward = float(immediate_rewards[row_index])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Non-numeric formal trajectory reward metadata at row {row_index}."
            ) from exc
        if not math.isfinite(trajectory_return) or not math.isfinite(immediate_reward):
            raise ValueError(
                f"Non-finite formal trajectory reward metadata at row {row_index}."
            )

        record = trajectories.setdefault(
            owner,
            {
                "parent_group_uid": parent_group_uid,
                "parent_index": parent_index,
                "replica_index": replica_index,
                "trajectory_uid": trajectory_uid,
                "trajectory_return": trajectory_return,
                "row_indices": [],
                "immediate_rewards": [],
            },
        )
        if not math.isclose(
            record["trajectory_return"],
            trajectory_return,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ValueError(
                "Conflicting formal trajectory return within one trajectory: "
                f"uid={trajectory_uid!r} first={record['trajectory_return']} "
                f"row={row_index} value={trajectory_return}"
            )
        record["row_indices"].append(row_index)
        record["immediate_rewards"].append(immediate_reward)

    if valid_rows == 0:
        raise ValueError("Formal trajectory metadata has no valid action rows.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in trajectories.values():
        immediate_sum = sum(record["immediate_rewards"])
        if not math.isclose(
            immediate_sum,
            record["trajectory_return"],
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ValueError(
                "Formal trajectory return does not equal summed immediate rewards: "
                f"uid={record['trajectory_uid']!r} "
                f"return={record['trajectory_return']} sum={immediate_sum}"
            )
        record["row_count"] = len(record["row_indices"])
        grouped.setdefault(record["parent_group_uid"], []).append(record)

    expected_replica_set = set(range(expected_replicas))
    for parent_group_uid, records in grouped.items():
        records.sort(key=lambda record: record["replica_index"])
        actual_replica_set = {record["replica_index"] for record in records}
        if actual_replica_set != expected_replica_set:
            raise ValueError(
                "Formal trajectory parent group has incomplete replicas: "
                f"group={parent_group_uid!r} "
                f"expected={sorted(expected_replica_set)} "
                f"actual={sorted(actual_replica_set)}"
            )
    return grouped


def validate_state_aware_rollout_uids(rollout_output: Any) -> None:
    non_tensor_batch = rollout_output.non_tensor_batch or {}
    if "rollout_parent_indices" not in non_tensor_batch:
        return

    rollout_uids = non_tensor_batch.get("rollout_uid")
    if rollout_uids is None:
        raise RuntimeError(
            "Action-level AgentMemory rollout is missing rollout_uid; "
            "source UUID grouping would mix distinct prompt states."
        )
    rollout_uids = np.asarray(rollout_uids, dtype=object)
    if rollout_uids.ndim != 1 or len(rollout_uids) != len(rollout_output):
        raise RuntimeError(
            "rollout_uid must be one-dimensional and aligned with rollout rows: "
            f"shape={rollout_uids.shape} rollout_rows={len(rollout_output)}"
        )
    invalid_rows = [
        index for index, rollout_uid in enumerate(rollout_uids)
        if ":statev1:" not in str(rollout_uid)
    ]
    if invalid_rows:
        raise RuntimeError(
            "Action-level AgentMemory rollout_uid is not prompt-state-aware: "
            f"invalid_rows={invalid_rows[:8]}"
        )


def requires_formal_trajectory_metadata(rollout_output: Any) -> bool:
    """Return whether this is an ordinary action-level AgentMemory rollout."""

    non_tensor_batch = rollout_output.non_tensor_batch or {}
    if "rollout_parent_indices" not in non_tensor_batch:
        return False
    return not any(
        marker in non_tensor_batch for marker in _RESTRICTED_AGENTMEMORY_MARKERS
    )


def validate_formal_trajectory_metadata(
    rollout_output: Any,
    *,
    expected_replicas: int,
    require: bool = False,
    require_runtime_evidence: bool = False,
    expected_suffix_credit: bool | None = None,
) -> dict[str, list[dict[str, Any]]] | None:
    """Fail closed on incomplete or inconsistent formal trajectory metadata."""

    non_tensor_batch = rollout_output.non_tensor_batch or {}
    tensor_batch = rollout_output.batch
    present_non_tensor = {
        key for key in FORMAL_TRAJECTORY_NON_TENSOR_KEYS
        if key in non_tensor_batch
    }
    present_tensor = {
        key for key in FORMAL_TRAJECTORY_TENSOR_KEYS
        if key in tensor_batch
    }
    any_present = bool(present_non_tensor or present_tensor)
    if not any_present:
        if require:
            raise RuntimeError(
                "Formal action-level AgentMemory rollout is missing trajectory metadata."
            )
        return None
    missing_non_tensor = set(FORMAL_TRAJECTORY_NON_TENSOR_KEYS) - present_non_tensor
    missing_tensor = set(FORMAL_TRAJECTORY_TENSOR_KEYS) - present_tensor
    if missing_non_tensor or missing_tensor:
        raise RuntimeError(
            "Incomplete formal trajectory metadata: "
            f"missing_non_tensor={sorted(missing_non_tensor)} "
            f"missing_tensor={sorted(missing_tensor)}"
        )
    row_count = len(rollout_output)
    for key in FORMAL_TRAJECTORY_NON_TENSOR_KEYS:
        values = np.asarray(non_tensor_batch[key], dtype=object)
        if values.ndim != 1 or len(values) != row_count:
            raise RuntimeError(
                f"{key} must be one-dimensional and aligned with {row_count} rows, "
                f"got {values.shape}."
            )
    for key in FORMAL_TRAJECTORY_TENSOR_KEYS:
        values = tensor_batch[key]
        if values.ndim != 1 or values.shape[0] != row_count:
            raise RuntimeError(
                f"{key} must have shape ({row_count},), got {tuple(values.shape)}."
            )
    parent_indices = non_tensor_batch.get("rollout_parent_indices")
    rollout_uids = non_tensor_batch.get("rollout_uid")
    if parent_indices is None or rollout_uids is None:
        raise RuntimeError(
            "Formal trajectory metadata requires rollout_parent_indices and rollout_uid."
        )
    valid_mask = tensor_batch.get(_PPO_VALID_SAMPLE_MASK)
    if valid_mask is None:
        valid_mask_values = [True] * row_count
    else:
        if valid_mask.ndim != 1 or valid_mask.shape[0] != row_count:
            raise RuntimeError(
                f"{_PPO_VALID_SAMPLE_MASK} must have shape ({row_count},), "
                f"got {tuple(valid_mask.shape)}."
            )
        valid_mask_values = valid_mask.detach().cpu().bool().tolist()
    try:
        grouped = validate_formal_trajectory_rows(
            parent_group_uids=non_tensor_batch[AGENTMEMORY_PARENT_GROUP_UID],
            exact_state_uids=non_tensor_batch[AGENTMEMORY_EXACT_STATE_UID],
            replica_indices=non_tensor_batch[AGENTMEMORY_REPLICA_INDEX],
            trajectory_uids=non_tensor_batch[AGENTMEMORY_TRAJECTORY_UID],
            trajectory_returns=tensor_batch[
                AGENTMEMORY_TRAJECTORY_RETURN
            ].detach().cpu().tolist(),
            immediate_rewards=tensor_batch[
                AGENTMEMORY_IMMEDIATE_REWARD
            ].detach().cpu().tolist(),
            trajectory_row_uids=non_tensor_batch[
                AGENTMEMORY_TRAJECTORY_ROW_UID
            ],
            trajectory_row_orders=tensor_batch[
                AGENTMEMORY_TRAJECTORY_ROW_ORDER
            ].detach().cpu().tolist(),
            trajectory_terminals=tensor_batch[
                AGENTMEMORY_TRAJECTORY_TERMINAL
            ].detach().cpu().tolist(),
            parent_indices=parent_indices,
            rollout_uids=rollout_uids,
            valid_mask=valid_mask_values,
            expected_replicas=expected_replicas,
        )
        runtime_non_tensor_present = {
            key for key in FORMAL_RUNTIME_EVIDENCE_NON_TENSOR_KEYS
            if key in non_tensor_batch
        }
        runtime_tensor_present = {
            key for key in FORMAL_RUNTIME_EVIDENCE_TENSOR_KEYS
            if key in tensor_batch
        }
        any_runtime_evidence = bool(
            runtime_non_tensor_present or runtime_tensor_present
        )
        if require_runtime_evidence and not any_runtime_evidence:
            raise ValueError("Formal PPO/GRPO runtime evidence metadata is missing.")
        if any_runtime_evidence:
            missing_runtime_non_tensor = (
                set(FORMAL_RUNTIME_EVIDENCE_NON_TENSOR_KEYS)
                - runtime_non_tensor_present
            )
            missing_runtime_tensor = (
                set(FORMAL_RUNTIME_EVIDENCE_TENSOR_KEYS)
                - runtime_tensor_present
            )
            if missing_runtime_non_tensor or missing_runtime_tensor:
                raise ValueError(
                    "Incomplete formal runtime evidence metadata: "
                    f"missing_non_tensor={sorted(missing_runtime_non_tensor)} "
                    f"missing_tensor={sorted(missing_runtime_tensor)}"
                )
            for key in FORMAL_RUNTIME_EVIDENCE_NON_TENSOR_KEYS:
                values = np.asarray(non_tensor_batch[key], dtype=object)
                if values.ndim != 1 or len(values) != row_count:
                    raise ValueError(
                        f"{key} must be one-dimensional and aligned with "
                        f"{row_count} rows, got {values.shape}."
                    )
            for key in FORMAL_RUNTIME_EVIDENCE_TENSOR_KEYS:
                values = tensor_batch[key]
                if values.ndim != 1 or values.shape[0] != row_count:
                    raise ValueError(
                        f"{key} must have shape ({row_count},), "
                        f"got {tuple(values.shape)}."
                    )
            suffix_returns = tensor_batch.get(AGENTMEMORY_SUFFIX_RETURN)
            if suffix_returns is not None and (
                suffix_returns.ndim != 1 or suffix_returns.shape[0] != row_count
            ):
                raise ValueError(
                    f"{AGENTMEMORY_SUFFIX_RETURN} must have shape ({row_count},), "
                    f"got {tuple(suffix_returns.shape)}."
                )
            task_rounds = tensor_batch.get("task_rounds")
            done_flags = non_tensor_batch.get("rollout_done_flags")
            prompts = tensor_batch.get("prompts")
            attention_mask = tensor_batch.get("attention_mask")
            responses = tensor_batch.get("responses")
            response_mask = tensor_batch.get("response_mask")
            if task_rounds is None or done_flags is None:
                raise ValueError(
                    "Formal runtime evidence requires task_rounds and "
                    "rollout_done_flags."
                )
            if prompts is None or attention_mask is None:
                raise ValueError(
                    "Formal prompt attestation requires prompts and attention_mask."
                )
            if responses is None or response_mask is None:
                raise ValueError(
                    "Formal response attestation requires responses and response_mask."
                )
            prompt_rows = prompts.detach().cpu().tolist()
            attention_rows = attention_mask.detach().cpu().tolist()
            if len(prompt_rows) != row_count or len(attention_rows) != row_count:
                raise ValueError("Packed prompt tensors are not aligned with rows.")
            prompt_width = len(prompt_rows[0]) if prompt_rows else 0
            actual_packed_tokens = [
                [
                    int(token_id)
                    for token_id, visible in zip(
                        prompt_row,
                        attention_row[:prompt_width],
                    )
                    if bool(visible)
                ]
                for prompt_row, attention_row in zip(prompt_rows, attention_rows)
            ]
            actual_packed_lengths = [len(tokens) for tokens in actual_packed_tokens]
            actual_packed_digests = [
                prompt_token_digest(tokens) for tokens in actual_packed_tokens
            ]
            stored_packed_lengths = tensor_batch[
                AGENTMEMORY_PACKED_PROMPT_LENGTH
            ].detach().cpu().tolist()
            stored_packed_digests = list(
                non_tensor_batch[AGENTMEMORY_PACKED_PROMPT_DIGEST]
            )
            for row_index, (
                actual_length,
                stored_length,
                actual_digest,
                stored_digest,
            ) in enumerate(
                zip(
                    actual_packed_lengths,
                    stored_packed_lengths,
                    actual_packed_digests,
                    stored_packed_digests,
                )
            ):
                if not bool(valid_mask_values[row_index]):
                    continue
                if (
                    int(stored_length) != actual_length
                    or str(stored_digest) != actual_digest
                ):
                    raise ValueError(
                        "Packed prompt attestation metadata differs from the "
                        f"actual PPO tensor at row {row_index}."
                    )
            response_rows = responses.detach().cpu().tolist()
            response_mask_rows = response_mask.detach().cpu().tolist()
            if len(response_rows) != row_count or len(response_mask_rows) != row_count:
                raise ValueError("Packed response tensors are not aligned with rows.")
            actual_packed_response_tokens = [
                [
                    int(token_id)
                    for token_id, visible in zip(response_row, mask_row)
                    if bool(visible)
                ]
                for response_row, mask_row in zip(
                    response_rows, response_mask_rows
                )
            ]
            actual_packed_response_lengths = [
                len(tokens) for tokens in actual_packed_response_tokens
            ]
            actual_packed_response_digests = [
                prompt_token_digest(tokens)
                for tokens in actual_packed_response_tokens
            ]
            stored_packed_response_lengths = tensor_batch[
                AGENTMEMORY_PACKED_RESPONSE_LENGTH
            ].detach().cpu().tolist()
            stored_packed_response_digests = list(
                non_tensor_batch[AGENTMEMORY_PACKED_RESPONSE_DIGEST]
            )
            for row_index, (
                actual_length,
                stored_length,
                actual_digest,
                stored_digest,
            ) in enumerate(
                zip(
                    actual_packed_response_lengths,
                    stored_packed_response_lengths,
                    actual_packed_response_digests,
                    stored_packed_response_digests,
                )
            ):
                if not bool(valid_mask_values[row_index]):
                    continue
                if (
                    int(stored_length) != actual_length
                    or str(stored_digest) != actual_digest
                ):
                    raise ValueError(
                        "Packed response attestation metadata differs from the "
                        f"actual PPO tensor at row {row_index}."
                    )
            scores = tensor_batch.get("scores")
            if scores is None:
                raise ValueError(
                    "Formal reward placement requires the packed scores tensor."
                )
            validate_formal_response_reward_placement(
                response_masks=response_mask_rows,
                score_rows=scores.detach().cpu().tolist(),
                expected_rewards=[
                    (
                        float(suffix_return)
                        if bool(suffix_credit)
                        else float(immediate_reward)
                    )
                    for suffix_return, suffix_credit, immediate_reward in zip(
                        tensor_batch[AGENTMEMORY_SUFFIX_RETURN]
                        .detach()
                        .cpu()
                        .tolist(),
                        tensor_batch[AGENTMEMORY_SUFFIX_CREDIT_APPLIED]
                        .detach()
                        .cpu()
                        .tolist(),
                        tensor_batch[AGENTMEMORY_IMMEDIATE_REWARD]
                        .detach()
                        .cpu()
                        .tolist(),
                    )
                ],
                valid_mask=valid_mask_values,
            )
            response_aligned_tensors = {}
            for key in (
                "old_log_probs",
                "values",
                "advantages",
                "returns",
                "token_level_scores",
                "token_level_rewards",
            ):
                tensor = tensor_batch.get(key)
                if tensor is not None:
                    response_aligned_tensors[key] = tensor.detach().cpu().tolist()
            validate_formal_response_aligned_tensors(
                response_masks=response_mask_rows,
                expected_response_lengths=stored_packed_response_lengths,
                tensors=response_aligned_tensors,
                valid_mask=valid_mask_values,
            )
            validate_formal_runtime_evidence_rows(
                exact_state_uids=non_tensor_batch[AGENTMEMORY_EXACT_STATE_UID],
                trajectory_uids=non_tensor_batch[AGENTMEMORY_TRAJECTORY_UID],
                trajectory_row_uids=non_tensor_batch[
                    AGENTMEMORY_TRAJECTORY_ROW_UID
                ],
                trajectory_row_orders=tensor_batch[
                    AGENTMEMORY_TRAJECTORY_ROW_ORDER
                ].detach().cpu().tolist(),
                trajectory_terminals=tensor_batch[
                    AGENTMEMORY_TRAJECTORY_TERMINAL
                ].detach().cpu().tolist(),
                task_rounds=task_rounds.detach().cpu().tolist(),
                immediate_rewards=tensor_batch[
                    AGENTMEMORY_IMMEDIATE_REWARD
                ].detach().cpu().tolist(),
                trajectory_returns=tensor_batch[
                    AGENTMEMORY_TRAJECTORY_RETURN
                ].detach().cpu().tolist(),
                action_texts=non_tensor_batch[AGENTMEMORY_ACTION_TEXT],
                done_flags=done_flags,
                generation_prompt_lengths=tensor_batch[
                    AGENTMEMORY_GENERATION_PROMPT_LENGTH
                ].detach().cpu().tolist(),
                generation_prompt_digests=non_tensor_batch[
                    AGENTMEMORY_GENERATION_PROMPT_DIGEST
                ],
                packed_prompt_lengths=stored_packed_lengths,
                packed_prompt_digests=stored_packed_digests,
                generation_response_lengths=tensor_batch[
                    AGENTMEMORY_GENERATION_RESPONSE_LENGTH
                ].detach().cpu().tolist(),
                generation_response_digests=non_tensor_batch[
                    AGENTMEMORY_GENERATION_RESPONSE_DIGEST
                ],
                packed_response_lengths=stored_packed_response_lengths,
                packed_response_digests=stored_packed_response_digests,
                suffix_credit_applied=tensor_batch[
                    AGENTMEMORY_SUFFIX_CREDIT_APPLIED
                ].detach().cpu().tolist(),
                suffix_returns=(
                    suffix_returns.detach().cpu().tolist()
                    if suffix_returns is not None
                    else None
                ),
                step_record_jsons=non_tensor_batch[AGENTMEMORY_STEP_RECORD_JSON],
                valid_mask=valid_mask_values,
                expected_suffix_credit=expected_suffix_credit,
                expected_prompt_width=prompt_width,
            )
        return grouped
    except ValueError as exc:
        raise RuntimeError(f"Invalid formal trajectory metadata: {exc}") from exc


def align_batch_to_rollout(batch: Any, rollout_output: Any, repeat_times: int) -> Any:
    import torch

    if rollout_output.non_tensor_batch and "rollout_parent_indices" in rollout_output.non_tensor_batch:
        parent_indices = rollout_output.non_tensor_batch["rollout_parent_indices"].astype(np.int64)
        assert parent_indices.ndim == 1, f"rollout_parent_indices must be 1-D, got {parent_indices.shape}"
        assert len(parent_indices) == len(rollout_output), (
            f"rollout_parent_indices length {len(parent_indices)} does not match "
            f"rollout batch size {len(rollout_output)}."
        )
        assert len(parent_indices) > 0, "rollout_parent_indices must not be empty."
        assert parent_indices.min() >= 0 and parent_indices.max() < len(batch), (
            f"rollout_parent_indices out of range for source batch size {len(batch)}: {parent_indices.tolist()}"
        )
        torch_indices = torch.as_tensor(parent_indices, dtype=torch.long)
        aligned_non_tensor = {key: value[parent_indices] for key, value in batch.non_tensor_batch.items()}
        return batch.__class__(batch=batch.batch[torch_indices], non_tensor_batch=aligned_non_tensor, meta_info=batch.meta_info)
    return batch.repeat(repeat_times=repeat_times, interleave=True)
