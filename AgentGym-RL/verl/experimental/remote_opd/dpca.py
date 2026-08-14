"""Remote cross-tokenizer PG-OPD using Dual-Pointer Chunk Alignment.

The teacher scores student-sampled action text with its own tokenizer. DPCA
finds minimal synchronized text chunks, then the semantic-prior rule from
arXiv:2606.09456 projects each teacher chunk likelihood onto student tokens.
"""

from __future__ import annotations

import json
import math
import os
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from verl.utils.agentgym.rollout_context import (
    AGENTMEMORY_ACTION_TEXT,
    AGENTMEMORY_GENERATION_PROMPT_DIGEST,
    AGENTMEMORY_GENERATION_PROMPT_LENGTH,
    AGENTMEMORY_GENERATION_RESPONSE_DIGEST,
    AGENTMEMORY_GENERATION_RESPONSE_LENGTH,
    AGENTMEMORY_PACKED_PROMPT_DIGEST,
    AGENTMEMORY_PACKED_PROMPT_LENGTH,
    AGENTMEMORY_PACKED_RESPONSE_DIGEST,
    AGENTMEMORY_PACKED_RESPONSE_LENGTH,
    AGENTMEMORY_STEP_RECORD_JSON,
    prompt_token_digest,
)


DPCA_OPD_ADVANTAGES = "dpca_opd_advantages"
DPCA_OPD_TOKEN_MASK = "dpca_opd_token_mask"
DPCA_OPD_TARGET_LOG_PROBS = "dpca_opd_target_log_probs"
DPCA_OPD_TEACHER_LOG_PROBS = "dpca_opd_teacher_log_probs"


class DPCAOPDError(RuntimeError):
    """Raised when teacher evidence or token alignment is incomplete."""


@dataclass(frozen=True)
class DPCAOPDSettings:
    enabled: bool = False
    teacher_base_url: str | None = None
    teacher_model: str | None = None
    teacher_tokenizer_path: str | None = None
    api_key_env: str | None = None
    request_batch_size: int = 16
    request_timeout_s: float = 300.0
    max_retries: int = 3
    retry_backoff_s: float = 1.0
    distillation_loss_coef: float = 1.0
    clip_ratio: float = 0.2
    loss_max_clamp: float | None = None
    log_prob_min_clamp: float | None = None
    strict_echo_token_check: bool = True
    alignment_dump_dir: str | None = None
    alignment_dump_rows: int = 4

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "DPCAOPDSettings":
        values = dict(config or {})
        known = cls.__dataclass_fields__
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"unknown DPCA OPD settings: {unknown}")
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.enabled:
            return
        required = {
            "teacher_base_url": self.teacher_base_url,
            "teacher_model": self.teacher_model,
            "teacher_tokenizer_path": self.teacher_tokenizer_path,
        }
        missing = sorted(name for name, value in required.items() if not str(value or "").strip())
        if missing:
            raise ValueError(f"DPCA OPD is missing required settings: {missing}")
        if self.request_batch_size <= 0:
            raise ValueError("DPCA OPD request_batch_size must be positive")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("DPCA OPD request_timeout_s must be positive")
        if self.max_retries < 0:
            raise ValueError("DPCA OPD max_retries must be non-negative")
        if not math.isfinite(self.distillation_loss_coef) or self.distillation_loss_coef <= 0:
            raise ValueError("DPCA OPD distillation_loss_coef must be positive")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("DPCA OPD clip_ratio must be in (0, 1)")
        if self.loss_max_clamp is not None and (
            not math.isfinite(self.loss_max_clamp) or self.loss_max_clamp <= 0
        ):
            raise ValueError("DPCA OPD loss_max_clamp must be positive or null")
        if self.log_prob_min_clamp is not None and (
            not math.isfinite(self.log_prob_min_clamp) or self.log_prob_min_clamp >= 0
        ):
            raise ValueError("DPCA OPD log_prob_min_clamp must be negative or null")
        if self.alignment_dump_rows < 0:
            raise ValueError("DPCA OPD alignment_dump_rows must be non-negative")


@dataclass(frozen=True)
class DPCAChunk:
    student_start: int
    student_end: int
    teacher_start: int
    teacher_end: int
    text: str


@dataclass(frozen=True)
class TeacherRequestRow:
    row_index: int
    full_prompt: str
    prefix_token_count: int
    full_teacher_ids: tuple[int, ...]
    teacher_action_ids: tuple[int, ...]
    teacher_stop_id: int | None
    student_action_ids: tuple[int, ...]
    student_stop_ids: tuple[int, ...]
    action_text: str


@dataclass(frozen=True)
class TeacherScoreRow:
    row_index: int
    teacher_action_ids: tuple[int, ...]
    teacher_action_log_probs: tuple[float, ...]
    teacher_stop_log_prob: float | None
    student_action_ids: tuple[int, ...]
    student_stop_ids: tuple[int, ...]
    action_text: str
    teacher_prompt_tokens: int


@dataclass(frozen=True)
class TeacherScoreBatch:
    rows: tuple[TeacherScoreRow, ...]
    full_batch_size: int
    latency_s: float
    request_count: int
    retry_count: int


@dataclass(frozen=True)
class DPCACreditRow:
    row_index: int
    chunks: tuple[DPCAChunk, ...]
    advantages: tuple[float, ...]
    target_log_probs: tuple[float, ...]
    teacher_log_probs: tuple[float, ...]
    student_stop_ids: tuple[int, ...]
    teacher_stop_log_prob: float | None
    stop_advantage: float | None
    conservation_max_abs_error: float
    clipped_tokens: int


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))


def _split_assistant_thinking(content: str) -> dict[str, str]:
    content = str(content)
    if not content.startswith("<think>"):
        return {"role": "assistant", "content": content}
    close = content.find("</think>", len("<think>"))
    if close < 0:
        return {"role": "assistant", "content": content}
    return {
        "role": "assistant",
        "reasoning_content": content[len("<think>") : close].strip("\n"),
        "content": content[close + len("</think>") :].lstrip("\n"),
    }


def parse_qwen_chatml_generation_prompt(prompt_text: str) -> tuple[list[dict[str, str]], bool]:
    """Parse complete ChatML turns and the final open assistant marker."""

    text = str(prompt_text)
    start_marker = "<|im_start|>"
    end_marker = "<|im_end|>"
    position = 0
    messages: list[dict[str, str]] = []
    generation_thinking: bool | None = None

    while position < len(text):
        while position < len(text) and text[position] in "\r\n \t":
            position += 1
        if position == len(text):
            break
        if not text.startswith(start_marker, position):
            raise DPCAOPDError(f"student prompt is not ChatML at offset {position}")
        role_start = position + len(start_marker)
        role_end = text.find("\n", role_start)
        if role_end < 0:
            raise DPCAOPDError("ChatML role marker has no newline")
        role = text[role_start:role_end].strip()
        if role not in {"system", "user", "assistant", "tool"}:
            raise DPCAOPDError(f"unsupported ChatML role {role!r}")
        content_start = role_end + 1
        content_end = text.find(end_marker, content_start)
        if content_end < 0:
            if role != "assistant" or text.find(start_marker, content_start) >= 0:
                raise DPCAOPDError("only the final assistant ChatML turn may be open")
            open_content = text[content_start:].replace("\r\n", "\n")
            if open_content in {"", "<think>", "<think>\n"}:
                generation_thinking = True
            elif open_content in {
                "<think></think>",
                "<think></think>\n\n",
                "<think>\n\n</think>\n\n",
            }:
                generation_thinking = False
            else:
                raise DPCAOPDError(
                    "final assistant prefix contains policy text before generation"
                )
            position = len(text)
            break
        content = text[content_start:content_end]
        if role == "assistant":
            messages.append(_split_assistant_thinking(content))
        else:
            messages.append({"role": role, "content": content})
        position = content_end + len(end_marker)

    if generation_thinking is None:
        raise DPCAOPDError("student prompt has no open assistant generation turn")
    if not messages:
        raise DPCAOPDError("student prompt has no completed messages")
    return messages, generation_thinking


def _as_step_record(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        value = json.loads(value)
        if isinstance(value, Mapping):
            return value
    raise DPCAOPDError("rollout row has no structured step record")


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    return [int(token_id) for token_id in ids]


def _teacher_stop_token(tokenizer: Any) -> tuple[str, int]:
    """Return the assistant-turn terminator used by the teacher template."""

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        token_id = convert("<|im_end|>")
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(token_id, int) and token_id != unknown_id:
            if _token_ids(tokenizer, "<|im_end|>") == [int(token_id)]:
                return "<|im_end|>", int(token_id)
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise DPCAOPDError("teacher tokenizer has no assistant-turn terminator")
    if not isinstance(eos_token_id, int):
        raise DPCAOPDError("teacher tokenizer has no assistant-turn terminator id")
    if _token_ids(tokenizer, eos_token) != [int(eos_token_id)]:
        raise DPCAOPDError("teacher EOS text is not exactly one tokenizer token")
    return eos_token, int(eos_token_id)


def _batch_row_token_ids(
    batch: Any,
    *,
    row_index: int,
    tensor_key: str,
    length_key: str,
    left_padded: bool,
) -> list[int]:
    token_rows = batch.batch.get(tensor_key)
    lengths = batch.batch.get(length_key)
    if token_rows is None or lengths is None:
        raise DPCAOPDError(
            f"DPCA OPD requires {tensor_key!r} and {length_key!r}"
        )
    if token_rows.ndim != 2 or lengths.ndim != 1:
        raise DPCAOPDError(
            f"DPCA OPD received malformed {tensor_key!r}/{length_key!r} tensors"
        )
    if row_index >= token_rows.shape[0] or row_index >= lengths.shape[0]:
        raise DPCAOPDError(f"row {row_index} exceeds {tensor_key!r} metadata")
    length = int(lengths[row_index].detach().cpu().item())
    width = int(token_rows.shape[1])
    if length <= 0 or length > width:
        raise DPCAOPDError(
            f"row {row_index} has invalid {tensor_key!r} length {length}/{width}"
        )
    row = token_rows[row_index]
    selected = row[-length:] if left_padded else row[:length]
    return [int(token_id) for token_id in selected.detach().cpu().tolist()]


def _require_matching_runtime_lengths(
    batch: Any,
    *,
    row_index: int,
    generation_key: str,
    packed_key: str,
) -> None:
    generation = batch.batch.get(generation_key)
    packed = batch.batch.get(packed_key)
    if generation is None or packed is None:
        raise DPCAOPDError(
            f"DPCA OPD requires {generation_key!r} and {packed_key!r}"
        )
    generation_length = int(generation[row_index].detach().cpu().item())
    packed_length = int(packed[row_index].detach().cpu().item())
    if generation_length != packed_length:
        raise DPCAOPDError(
            f"row {row_index} generation/packed token lengths differ: "
            f"{generation_length} != {packed_length}"
        )


def _require_matching_token_digest(
    batch: Any,
    *,
    row_index: int,
    token_ids: Sequence[int],
    digest_keys: Sequence[str],
) -> None:
    expected = prompt_token_digest(token_ids)
    for key in digest_keys:
        values = batch.non_tensor_batch.get(key)
        if values is None or row_index >= len(values):
            raise DPCAOPDError(f"DPCA OPD requires runtime digest {key!r}")
        if str(values[row_index]) != expected:
            raise DPCAOPDError(
                f"row {row_index} token digest differs for {key!r}"
            )


def _split_student_action_and_stop_tokens(
    tokenizer: Any,
    response_ids: Sequence[int],
    action_text: str,
) -> tuple[list[int], list[int]]:
    ids = [int(token_id) for token_id in response_ids]
    special_ids = set(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []))
    action_end = len(ids)
    while action_end > 0 and ids[action_end - 1] in special_ids:
        action_end -= 1
    action_ids = ids[:action_end]
    stop_ids = ids[action_end:]
    if any(token_id in special_ids for token_id in action_ids):
        raise DPCAOPDError("student response contains an internal special token")
    decoded = tokenizer.decode(action_ids, skip_special_tokens=False)
    if _normalize_text(decoded) != _normalize_text(action_text):
        raise DPCAOPDError(
            "visible student response tokens do not reproduce the policy action text"
        )
    if not action_ids:
        raise DPCAOPDError("policy action has no visible student tokens")
    return action_ids, stop_ids


def build_teacher_requests(
    batch: Any,
    *,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
) -> list[TeacherRequestRow]:
    records = batch.non_tensor_batch.get(AGENTMEMORY_STEP_RECORD_JSON)
    actions = batch.non_tensor_batch.get(AGENTMEMORY_ACTION_TEXT)
    if records is None or actions is None:
        raise DPCAOPDError(
            "DPCA OPD requires task-neutral step records and action texts"
        )
    if len(records) != len(batch) or len(actions) != len(batch):
        raise DPCAOPDError("DPCA OPD rollout metadata length mismatch")
    valid = batch.batch.get("ppo_valid_sample_mask")
    valid_rows = (
        [True] * len(batch)
        if valid is None
        else valid.detach().cpu().to(torch.bool).tolist()
    )

    requests: list[TeacherRequestRow] = []
    for row_index, is_valid in enumerate(valid_rows):
        if not is_valid:
            continue
        record = _as_step_record(records[row_index])
        action_text = str(actions[row_index])
        record_action = record.get("action", record.get("content"))
        if record_action is None or action_text != str(record_action):
            raise DPCAOPDError(
                f"row {row_index} action text differs from step record"
            )

        _require_matching_runtime_lengths(
            batch,
            row_index=row_index,
            generation_key=AGENTMEMORY_GENERATION_PROMPT_LENGTH,
            packed_key=AGENTMEMORY_PACKED_PROMPT_LENGTH,
        )
        prompt_ids = _batch_row_token_ids(
            batch,
            row_index=row_index,
            tensor_key="prompts",
            length_key=AGENTMEMORY_GENERATION_PROMPT_LENGTH,
            left_padded=True,
        )
        _require_matching_token_digest(
            batch,
            row_index=row_index,
            token_ids=prompt_ids,
            digest_keys=(
                AGENTMEMORY_GENERATION_PROMPT_DIGEST,
                AGENTMEMORY_PACKED_PROMPT_DIGEST,
            ),
        )

        _require_matching_runtime_lengths(
            batch,
            row_index=row_index,
            generation_key=AGENTMEMORY_GENERATION_RESPONSE_LENGTH,
            packed_key=AGENTMEMORY_PACKED_RESPONSE_LENGTH,
        )
        response_ids = _batch_row_token_ids(
            batch,
            row_index=row_index,
            tensor_key="responses",
            length_key=AGENTMEMORY_GENERATION_RESPONSE_LENGTH,
            left_padded=False,
        )
        _require_matching_token_digest(
            batch,
            row_index=row_index,
            token_ids=response_ids,
            digest_keys=(
                AGENTMEMORY_GENERATION_RESPONSE_DIGEST,
                AGENTMEMORY_PACKED_RESPONSE_DIGEST,
            ),
        )
        record_response_ids = record.get("response_token_ids")
        if not isinstance(record_response_ids, list) or [
            int(token_id) for token_id in record_response_ids
        ] != response_ids:
            raise DPCAOPDError(
                f"row {row_index} response tokens differ from step record"
            )
        student_action_ids, student_stop_ids = _split_student_action_and_stop_tokens(
            student_tokenizer, response_ids, action_text
        )

        prompt_text = student_tokenizer.decode(
            prompt_ids, skip_special_tokens=False
        )
        messages, thinking = parse_qwen_chatml_generation_prompt(prompt_text)
        teacher_prefix = teacher_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking=thinking,
            preserve_thinking=True,
        )
        if not isinstance(teacher_prefix, str) or not teacher_prefix:
            raise DPCAOPDError("teacher chat template returned an empty prefix")
        action_prompt = teacher_prefix + action_text
        prefix_ids = _token_ids(teacher_tokenizer, teacher_prefix)
        action_full_ids = _token_ids(teacher_tokenizer, action_prompt)
        if action_full_ids[: len(prefix_ids)] != prefix_ids:
            raise DPCAOPDError(
                f"row {row_index} teacher tokenization crosses the action boundary"
            )
        teacher_action_ids = action_full_ids[len(prefix_ids) :]
        if not teacher_action_ids:
            raise DPCAOPDError(f"row {row_index} action has no teacher tokens")
        teacher_text = teacher_tokenizer.decode(
            teacher_action_ids, skip_special_tokens=False
        )
        if _normalize_text(teacher_text) != _normalize_text(action_text):
            raise DPCAOPDError(
                f"row {row_index} teacher action tokens do not reproduce action text"
            )

        full_prompt = action_prompt
        full_ids = action_full_ids
        teacher_stop_id = None
        if student_stop_ids:
            stop_token, stop_token_id = _teacher_stop_token(teacher_tokenizer)
            full_prompt = action_prompt + stop_token
            full_ids = _token_ids(teacher_tokenizer, full_prompt)
            expected_ids = action_full_ids + [stop_token_id]
            if full_ids != expected_ids:
                raise DPCAOPDError(
                    f"row {row_index} teacher stop text does not append exactly one token"
                )
            teacher_stop_id = stop_token_id
        requests.append(
            TeacherRequestRow(
                row_index=row_index,
                full_prompt=full_prompt,
                prefix_token_count=len(prefix_ids),
                full_teacher_ids=tuple(full_ids),
                teacher_action_ids=tuple(teacher_action_ids),
                teacher_stop_id=teacher_stop_id,
                student_action_ids=tuple(student_action_ids),
                student_stop_ids=tuple(student_stop_ids),
                action_text=action_text,
            )
        )
    if not requests:
        raise DPCAOPDError("DPCA OPD batch has no valid rollout rows")
    return requests


def dpca_minimal_chunks(
    student_ids: Sequence[int],
    teacher_ids: Sequence[int],
    *,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
) -> tuple[DPCAChunk, ...]:
    """Find every minimal synchronized chunk with dual greedy pointers."""

    student_ids = [int(token_id) for token_id in student_ids]
    teacher_ids = [int(token_id) for token_id in teacher_ids]
    if not student_ids or not teacher_ids:
        raise DPCAOPDError("DPCA cannot align an empty token sequence")
    student_full = _normalize_text(
        student_tokenizer.decode(student_ids, skip_special_tokens=False)
    )
    teacher_full = _normalize_text(
        teacher_tokenizer.decode(teacher_ids, skip_special_tokens=False)
    )
    if student_full != teacher_full:
        raise DPCAOPDError("student and teacher action tokenizations decode differently")

    chunks: list[DPCAChunk] = []
    student_start = 0
    teacher_start = 0
    while student_start < len(student_ids) and teacher_start < len(teacher_ids):
        student_end = student_start + 1
        teacher_end = teacher_start + 1
        matched = False
        while student_end <= len(student_ids) and teacher_end <= len(teacher_ids):
            student_text = _normalize_text(
                student_tokenizer.decode(
                    student_ids[student_start:student_end],
                    skip_special_tokens=False,
                )
            )
            teacher_text = _normalize_text(
                teacher_tokenizer.decode(
                    teacher_ids[teacher_start:teacher_end],
                    skip_special_tokens=False,
                )
            )
            if student_text == teacher_text and not student_text.endswith("\ufffd"):
                chunks.append(
                    DPCAChunk(
                        student_start=student_start,
                        student_end=student_end,
                        teacher_start=teacher_start,
                        teacher_end=teacher_end,
                        text=student_text,
                    )
                )
                student_start = student_end
                teacher_start = teacher_end
                matched = True
                break

            previous = (student_end, teacher_end)
            student_length = len(student_text)
            teacher_length = len(teacher_text)
            if student_length < teacher_length:
                if student_end < len(student_ids):
                    student_end += 1
                elif teacher_end < len(teacher_ids):
                    teacher_end += 1
            elif student_length > teacher_length:
                if teacher_end < len(teacher_ids):
                    teacher_end += 1
                elif student_end < len(student_ids):
                    student_end += 1
            else:
                student_incomplete = student_text.endswith("\ufffd")
                teacher_incomplete = teacher_text.endswith("\ufffd")
                if student_incomplete and not teacher_incomplete and student_end < len(student_ids):
                    student_end += 1
                elif teacher_incomplete and not student_incomplete and teacher_end < len(teacher_ids):
                    teacher_end += 1
                else:
                    if student_end < len(student_ids):
                        student_end += 1
                    if teacher_end < len(teacher_ids):
                        teacher_end += 1
            if previous == (student_end, teacher_end):
                break
        if not matched:
            raise DPCAOPDError(
                "DPCA failed to find a synchronized chunk despite equal full text"
            )

    if student_start != len(student_ids) or teacher_start != len(teacher_ids):
        raise DPCAOPDError("DPCA did not consume both action token sequences")
    if _normalize_text("".join(chunk.text for chunk in chunks)) != student_full:
        raise DPCAOPDError("DPCA chunk text does not reconstruct the action")
    return tuple(chunks)


def dpca_semantic_prior_credit(
    student_log_probs: Sequence[float],
    teacher_log_probs: Sequence[float],
    chunks: Sequence[DPCAChunk],
    *,
    log_prob_min_clamp: float | None,
    loss_max_clamp: float | None,
) -> tuple[list[float], list[float], list[float], float, int]:
    """Apply equations 8-9 of arXiv:2606.09456 to synchronized chunks."""

    student = [float(value) for value in student_log_probs]
    teacher = [float(value) for value in teacher_log_probs]
    if not all(math.isfinite(value) for value in student + teacher):
        raise DPCAOPDError("DPCA credit received a non-finite log probability")
    if log_prob_min_clamp is not None:
        student = [max(value, log_prob_min_clamp) for value in student]
        teacher = [max(value, log_prob_min_clamp) for value in teacher]

    advantages = [0.0] * len(student)
    targets = [0.0] * len(student)
    aligned_teacher = [0.0] * len(student)
    conservation_max_abs_error = 0.0
    clipped_tokens = 0
    for chunk in chunks:
        student_slice = student[chunk.student_start : chunk.student_end]
        teacher_slice = teacher[chunk.teacher_start : chunk.teacher_end]
        student_total = sum(student_slice)
        teacher_total = sum(teacher_slice)
        if abs(student_total) < 1e-12:
            target_slice = [teacher_total / len(student_slice)] * len(student_slice)
        else:
            scale = teacher_total / student_total
            target_slice = [scale * value for value in student_slice]
        error = abs(sum(target_slice) - teacher_total)
        conservation_max_abs_error = max(conservation_max_abs_error, error)
        for local_index, (student_value, target_value) in enumerate(
            zip(student_slice, target_slice, strict=True)
        ):
            index = chunk.student_start + local_index
            advantage = target_value - student_value
            if loss_max_clamp is not None:
                clipped = max(-loss_max_clamp, min(loss_max_clamp, advantage))
                clipped_tokens += int(clipped != advantage)
                advantage = clipped
            advantages[index] = advantage
            targets[index] = target_value
            aligned_teacher[index] = teacher_total / len(student_slice)
    return (
        advantages,
        targets,
        aligned_teacher,
        conservation_max_abs_error,
        clipped_tokens,
    )


RequestFunction = Callable[[str, Mapping[str, Any], Mapping[str, str], float], Mapping[str, Any]]


def _default_request(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_s: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        value = json.load(response)
    if not isinstance(value, Mapping):
        raise DPCAOPDError("teacher returned a non-object response")
    return value


def _surface_tokens(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    return [
        tokenizer.decode([int(token_id)], skip_special_tokens=False)
        for token_id in token_ids
    ]


class RemoteDPCAOPDScorer:
    """Asynchronously score student actions through a Kimi-compatible endpoint."""

    def __init__(
        self,
        settings: DPCAOPDSettings,
        *,
        student_tokenizer: Any,
        teacher_tokenizer: Any | None = None,
        request_fn: RequestFunction | None = None,
    ) -> None:
        settings.validate()
        if not settings.enabled:
            raise ValueError("RemoteDPCAOPDScorer requires enabled settings")
        self.settings = settings
        self.student_tokenizer = student_tokenizer
        if teacher_tokenizer is None:
            from transformers import AutoTokenizer

            teacher_tokenizer = AutoTokenizer.from_pretrained(
                settings.teacher_tokenizer_path,
                trust_remote_code=True,
            )
        self.teacher_tokenizer = teacher_tokenizer
        self.request_fn = request_fn or _default_request
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="remote-dpca-opd",
        )

    @property
    def completion_url(self) -> str:
        return f"{str(self.settings.teacher_base_url).rstrip('/')}/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key_env:
            key = os.environ.get(self.settings.api_key_env)
            if not key:
                raise DPCAOPDError(
                    f"teacher API key environment variable {self.settings.api_key_env!r} is unset"
                )
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def submit(self, batch: Any) -> Future[TeacherScoreBatch]:
        requests = build_teacher_requests(
            batch,
            student_tokenizer=self.student_tokenizer,
            teacher_tokenizer=self.teacher_tokenizer,
        )
        return self._executor.submit(self._score_requests, requests, len(batch))

    def _request_with_retry(self, payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], int]:
        retries = 0
        while True:
            try:
                return (
                    self.request_fn(
                        self.completion_url,
                        payload,
                        self._headers(),
                        self.settings.request_timeout_s,
                    ),
                    retries,
                )
            except (OSError, TimeoutError, urllib.error.URLError, DPCAOPDError) as exc:
                if retries >= self.settings.max_retries:
                    raise DPCAOPDError(
                        f"remote teacher failed after {retries + 1} attempts"
                    ) from exc
                time.sleep(self.settings.retry_backoff_s * (2**retries))
                retries += 1

    def _score_requests(
        self,
        requests: Sequence[TeacherRequestRow],
        full_batch_size: int,
    ) -> TeacherScoreBatch:
        started = time.monotonic()
        scored_rows: list[TeacherScoreRow] = []
        request_count = 0
        retry_count = 0
        for offset in range(0, len(requests), self.settings.request_batch_size):
            chunk = list(requests[offset : offset + self.settings.request_batch_size])
            payload = {
                "model": self.settings.teacher_model,
                "prompt": [row.full_prompt for row in chunk],
                "max_tokens": 1,
                "temperature": 0,
                "echo": True,
                "logprobs": 1,
            }
            response, retries = self._request_with_retry(payload)
            request_count += 1
            retry_count += retries
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != len(chunk):
                raise DPCAOPDError("teacher choice count does not match prompt count")
            indexed: dict[int, Mapping[str, Any]] = {}
            for fallback_index, choice in enumerate(choices):
                if not isinstance(choice, Mapping):
                    raise DPCAOPDError("teacher returned a non-object choice")
                choice_index = int(choice.get("index", fallback_index))
                if choice_index in indexed:
                    raise DPCAOPDError("teacher returned duplicate choice indices")
                indexed[choice_index] = choice

            for local_index, row in enumerate(chunk):
                choice = indexed.get(local_index)
                if choice is None:
                    raise DPCAOPDError(f"teacher omitted choice {local_index}")
                logprobs = choice.get("logprobs")
                if not isinstance(logprobs, Mapping):
                    raise DPCAOPDError("teacher choice has no logprobs object")
                token_logprobs = logprobs.get("token_logprobs")
                tokens = logprobs.get("tokens")
                if not isinstance(token_logprobs, list) or not isinstance(tokens, list):
                    raise DPCAOPDError("teacher logprobs object is incomplete")
                input_length = len(row.full_teacher_ids)
                if len(token_logprobs) < input_length or len(tokens) < input_length:
                    raise DPCAOPDError(
                        f"teacher echoed fewer tokens than supplied for row {row.row_index}"
                    )
                if self.settings.strict_echo_token_check:
                    expected = _surface_tokens(
                        self.teacher_tokenizer, row.full_teacher_ids
                    )
                    if [_normalize_text(token) for token in tokens[:input_length]] != [
                        _normalize_text(token) for token in expected
                    ]:
                        raise DPCAOPDError(
                            f"teacher token echo differs from local tokenizer for row {row.row_index}"
                        )
                action_end = row.prefix_token_count + len(row.teacher_action_ids)
                action_values = token_logprobs[row.prefix_token_count : action_end]
                if len(action_values) != len(row.teacher_action_ids):
                    raise DPCAOPDError("teacher action logprob length mismatch")
                if any(value is None for value in action_values):
                    raise DPCAOPDError("teacher omitted an action-token logprob")
                action_log_probs = tuple(float(value) for value in action_values)
                if not all(math.isfinite(value) for value in action_log_probs):
                    raise DPCAOPDError("teacher returned a non-finite action logprob")
                teacher_stop_log_prob = None
                if row.teacher_stop_id is not None:
                    if action_end >= input_length:
                        raise DPCAOPDError("teacher EOS token is missing from echoed input")
                    if int(row.full_teacher_ids[action_end]) != row.teacher_stop_id:
                        raise DPCAOPDError("teacher EOS token moved after request construction")
                    stop_value = token_logprobs[action_end]
                    if stop_value is None or not math.isfinite(float(stop_value)):
                        raise DPCAOPDError("teacher omitted a finite EOS logprob")
                    teacher_stop_log_prob = float(stop_value)
                scored_rows.append(
                    TeacherScoreRow(
                        row_index=row.row_index,
                        teacher_action_ids=row.teacher_action_ids,
                        teacher_action_log_probs=action_log_probs,
                        teacher_stop_log_prob=teacher_stop_log_prob,
                        student_action_ids=row.student_action_ids,
                        student_stop_ids=row.student_stop_ids,
                        action_text=row.action_text,
                        teacher_prompt_tokens=row.prefix_token_count,
                    )
                )
        scored_rows.sort(key=lambda row: row.row_index)
        return TeacherScoreBatch(
            rows=tuple(scored_rows),
            full_batch_size=full_batch_size,
            latency_s=time.monotonic() - started,
            request_count=request_count,
            retry_count=retry_count,
        )


def _write_alignment_dump(
    settings: DPCAOPDSettings,
    *,
    global_step: int,
    rows: Sequence[DPCACreditRow],
) -> None:
    if not settings.alignment_dump_dir or settings.alignment_dump_rows <= 0:
        return
    path = Path(settings.alignment_dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dpca_opd_alignment_v1",
        "global_step": int(global_step),
        "settings": {
            "teacher_model": settings.teacher_model,
            "clip_ratio": settings.clip_ratio,
            "distillation_loss_coef": settings.distillation_loss_coef,
            "loss_max_clamp": settings.loss_max_clamp,
            "log_prob_min_clamp": settings.log_prob_min_clamp,
        },
        "rows": [
            {
                **asdict(row),
                "chunks": [asdict(chunk) for chunk in row.chunks],
            }
            for row in rows[: settings.alignment_dump_rows]
        ],
    }
    destination = path / f"step_{int(global_step):06d}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(destination)


def attach_dpca_opd_advantages(
    batch: Any,
    teacher_scores: TeacherScoreBatch,
    settings: DPCAOPDSettings,
    *,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    global_step: int,
) -> dict[str, float]:
    """Attach DPCA token advantages while leaving task rewards/returns untouched."""

    old_log_probs = batch.batch["old_log_probs"].detach().to(torch.float32)
    response_mask = batch.batch["response_mask"].detach().to(torch.bool)
    ppo_valid = batch.batch.get("ppo_valid_sample_mask")
    ppo_valid = (
        torch.ones(old_log_probs.shape[0], dtype=torch.bool, device=old_log_probs.device)
        if ppo_valid is None
        else ppo_valid.detach().to(device=old_log_probs.device, dtype=torch.bool)
    )
    if teacher_scores.full_batch_size != old_log_probs.shape[0]:
        raise DPCAOPDError("teacher score batch size changed before credit assignment")

    advantages = torch.zeros_like(old_log_probs)
    targets = torch.zeros_like(old_log_probs)
    aligned_teacher = torch.zeros_like(old_log_probs)
    token_mask = torch.zeros_like(response_mask)
    credit_rows: list[DPCACreditRow] = []
    expected_rows = {index for index, value in enumerate(ppo_valid.tolist()) if value}
    observed_rows = {row.row_index for row in teacher_scores.rows}
    if expected_rows != observed_rows:
        raise DPCAOPDError(
            f"teacher score coverage mismatch: missing={sorted(expected_rows-observed_rows)} "
            f"extra={sorted(observed_rows-expected_rows)}"
        )

    chunk_count = 0
    one_to_one_chunks = 0
    max_student_chunk = 0
    max_teacher_chunk = 0
    conservation_error = 0.0
    clipped_tokens = 0
    teacher_action_tokens = 0
    student_action_tokens = 0
    teacher_stop_tokens = 0
    student_stop_tokens = 0
    supervised_student_stop_tokens = 0
    teacher_prompt_tokens = 0
    for row in teacher_scores.rows:
        student_count = len(row.student_action_ids)
        student_stop_count = len(row.student_stop_ids)
        response_count = student_count + student_stop_count
        if response_count > old_log_probs.shape[1]:
            raise DPCAOPDError("student action exceeds PPO response tensor width")
        if not response_mask[row.row_index, :response_count].all():
            raise DPCAOPDError("student action or trailing stop contains a masked PPO token")
        if bool(row.student_stop_ids) != (row.teacher_stop_log_prob is not None):
            raise DPCAOPDError("student and teacher stop-token evidence disagree")
        chunks = dpca_minimal_chunks(
            row.student_action_ids,
            row.teacher_action_ids,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
        )
        student_values = (
            old_log_probs[row.row_index, :student_count].detach().cpu().tolist()
        )
        (
            row_advantages,
            row_targets,
            row_teacher,
            row_conservation_error,
            row_clipped_tokens,
        ) = dpca_semantic_prior_credit(
            student_values,
            row.teacher_action_log_probs,
            chunks,
            log_prob_min_clamp=settings.log_prob_min_clamp,
            loss_max_clamp=settings.loss_max_clamp,
        )
        device = old_log_probs.device
        advantages[row.row_index, :student_count] = torch.tensor(
            row_advantages, device=device, dtype=old_log_probs.dtype
        )
        targets[row.row_index, :student_count] = torch.tensor(
            row_targets, device=device, dtype=old_log_probs.dtype
        )
        aligned_teacher[row.row_index, :student_count] = torch.tensor(
            row_teacher, device=device, dtype=old_log_probs.dtype
        )
        token_mask[row.row_index, :student_count] = True
        stop_advantage = None
        if row.student_stop_ids:
            stop_index = student_count
            student_stop_log_prob = float(
                old_log_probs[row.row_index, stop_index].detach().cpu().item()
            )
            teacher_stop_log_prob = float(row.teacher_stop_log_prob)
            if not math.isfinite(student_stop_log_prob):
                raise DPCAOPDError("student stop token has a non-finite logprob")
            if settings.log_prob_min_clamp is not None:
                student_stop_log_prob = max(
                    student_stop_log_prob, settings.log_prob_min_clamp
                )
                teacher_stop_log_prob = max(
                    teacher_stop_log_prob, settings.log_prob_min_clamp
                )
            stop_advantage = teacher_stop_log_prob - student_stop_log_prob
            if settings.loss_max_clamp is not None:
                clipped = max(
                    -settings.loss_max_clamp,
                    min(settings.loss_max_clamp, stop_advantage),
                )
                clipped_tokens += int(clipped != stop_advantage)
                stop_advantage = clipped
            advantages[row.row_index, stop_index] = stop_advantage
            targets[row.row_index, stop_index] = teacher_stop_log_prob
            aligned_teacher[row.row_index, stop_index] = teacher_stop_log_prob
            token_mask[row.row_index, stop_index] = True
            teacher_stop_tokens += 1
            supervised_student_stop_tokens += 1
        chunk_count += len(chunks)
        one_to_one_chunks += sum(
            1
            for chunk in chunks
            if chunk.student_end - chunk.student_start == 1
            and chunk.teacher_end - chunk.teacher_start == 1
        )
        max_student_chunk = max(
            max_student_chunk,
            max(chunk.student_end - chunk.student_start for chunk in chunks),
        )
        max_teacher_chunk = max(
            max_teacher_chunk,
            max(chunk.teacher_end - chunk.teacher_start for chunk in chunks),
        )
        conservation_error = max(conservation_error, row_conservation_error)
        clipped_tokens += row_clipped_tokens
        teacher_action_tokens += len(row.teacher_action_ids)
        student_action_tokens += student_count
        student_stop_tokens += student_stop_count
        teacher_prompt_tokens += row.teacher_prompt_tokens
        credit_rows.append(
            DPCACreditRow(
                row_index=row.row_index,
                chunks=chunks,
                advantages=tuple(row_advantages),
                target_log_probs=tuple(row_targets),
                teacher_log_probs=tuple(row.teacher_action_log_probs),
                student_stop_ids=row.student_stop_ids,
                teacher_stop_log_prob=row.teacher_stop_log_prob,
                stop_advantage=stop_advantage,
                conservation_max_abs_error=row_conservation_error,
                clipped_tokens=row_clipped_tokens,
            )
        )

    if not torch.equal(token_mask & ~response_mask, torch.zeros_like(token_mask)):
        raise DPCAOPDError("DPCA OPD mask includes a non-policy token")
    valid_values = advantages[token_mask]
    if valid_values.numel() == 0 or not torch.isfinite(valid_values).all():
        raise DPCAOPDError("DPCA OPD produced no finite token advantages")
    batch.batch[DPCA_OPD_ADVANTAGES] = advantages
    batch.batch[DPCA_OPD_TOKEN_MASK] = token_mask
    batch.batch[DPCA_OPD_TARGET_LOG_PROBS] = targets
    batch.batch[DPCA_OPD_TEACHER_LOG_PROBS] = aligned_teacher
    batch.meta_info["dpca_opd_distillation_loss_coef"] = float(
        settings.distillation_loss_coef
    )
    batch.meta_info["dpca_opd_clip_ratio"] = float(settings.clip_ratio)
    _write_alignment_dump(
        settings, global_step=global_step, rows=credit_rows
    )

    total_tokens = int(token_mask.sum().item())
    return {
        "dpca_opd/enabled": 1.0,
        "dpca_opd/distillation_loss_coef": float(settings.distillation_loss_coef),
        "dpca_opd/valid_rows": float(len(teacher_scores.rows)),
        "dpca_opd/aligned_student_tokens": float(total_tokens),
        "dpca_opd/student_action_tokens": float(student_action_tokens),
        "dpca_opd/teacher_action_tokens": float(teacher_action_tokens),
        "dpca_opd/student_stop_tokens": float(student_stop_tokens),
        "dpca_opd/supervised_student_stop_tokens": float(
            supervised_student_stop_tokens
        ),
        "dpca_opd/teacher_stop_tokens": float(teacher_stop_tokens),
        "dpca_opd/teacher_prompt_tokens": float(teacher_prompt_tokens),
        "dpca_opd/chunks": float(chunk_count),
        "dpca_opd/one_to_one_chunk_ratio": float(one_to_one_chunks / chunk_count),
        "dpca_opd/max_student_chunk_tokens": float(max_student_chunk),
        "dpca_opd/max_teacher_chunk_tokens": float(max_teacher_chunk),
        "dpca_opd/conservation_max_abs_error": float(conservation_error),
        "dpca_opd/adv_mean": float(valid_values.mean().item()),
        "dpca_opd/adv_std": float(valid_values.std(unbiased=False).item()),
        "dpca_opd/adv_min": float(valid_values.min().item()),
        "dpca_opd/adv_max": float(valid_values.max().item()),
        "dpca_opd/clipped_token_ratio": float(clipped_tokens / total_tokens),
        "dpca_opd/teacher_latency_s": float(teacher_scores.latency_s),
        "dpca_opd/teacher_requests": float(teacher_scores.request_count),
        "dpca_opd/teacher_retries": float(teacher_scores.retry_count),
        "dpca_opd/teacher_failure_rate": 0.0,
    }
