"""Fail-closed exact-token transport for the frozen vLLM endpoint."""

from __future__ import annotations

from threading import Lock
import time
from typing import Any, Mapping

from paired_eval.model_client import JsonTransport, ModelClientError
from paired_eval.serialization import sha256_json


def require_token_ids(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ModelClientError(
            "exact_token_ids_missing",
            f"{label} omitted nonempty exact token ids",
        )
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in value
    ):
        raise ModelClientError(
            "exact_token_ids_malformed",
            f"{label} returned malformed exact token ids",
        )
    return tuple(value)


def tokenization_key(payload: Mapping[str, Any], *, tokenize: bool) -> str:
    model = payload.get("model")
    messages = payload.get("messages")
    template = payload.get("chat_template_kwargs")
    if not isinstance(model, str) or not model:
        raise ModelClientError("request_shape_drift", "model id is missing")
    if not isinstance(messages, list) or not messages:
        raise ModelClientError("request_shape_drift", "messages are missing")
    if not isinstance(template, Mapping):
        raise ModelClientError(
            "request_shape_drift", "chat template arguments are missing"
        )
    if tokenize and payload.get("add_generation_prompt") is not True:
        raise ModelClientError(
            "request_shape_drift",
            "tokenize request must add the generation prompt",
        )
    return sha256_json(
        {
            "model": model,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": dict(template),
        }
    )


def scheduler_request_id(
    *,
    run_id: str,
    task_index: int,
    arm: str,
    generation: int,
    turn_index: int,
) -> str:
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("scheduler run ID is invalid")
    if type(task_index) is not int or task_index < 0:
        raise ValueError("scheduler task index is invalid")
    if not isinstance(arm, str) or not arm:
        raise ValueError("scheduler arm is invalid")
    if type(generation) is not int or generation <= 0:
        raise ValueError("scheduler generation is invalid")
    if type(turn_index) is not int or turn_index < 0:
        raise ValueError("scheduler turn index is invalid")
    return sha256_json(
        {
            "run_id": run_id,
            "task_index": task_index,
            "arm": arm,
            "generation": generation,
            "turn_index": turn_index,
        }
    )


class ExactTokenVllmTransport:
    """Request and reconcile vLLM's prompt and response token identifiers."""

    def __init__(
        self,
        delegate: JsonTransport,
        *,
        request_context: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(getattr(delegate, "post", None)):
            raise TypeError("exact-token transport delegate must implement post")
        self.delegate = delegate
        self.prompt_tokens: dict[str, tuple[int, ...]] = {}
        self.lock = Lock()
        self.request_context = (
            None if request_context is None else dict(request_context)
        )
        if self.request_context is not None:
            expected = {"run_id", "task_index", "arm", "generation"}
            if set(self.request_context) != expected:
                raise ValueError("exact-token scheduler context fields drifted")
            scheduler_request_id(
                **self.request_context,
                turn_index=0,
            )
        self.turn_index = 0
        self.events: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        endpoint = url.rstrip("/")
        if endpoint.endswith("/tokenize"):
            return self.post_tokenize(url, payload, timeout_seconds=timeout_seconds)
        if endpoint.endswith("/chat/completions"):
            return self.post_chat(url, payload, timeout_seconds=timeout_seconds)
        raise ModelClientError(
            "endpoint_shape_drift",
            f"unsupported exact-token endpoint: {url}",
        )

    def post_tokenize(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        key = tokenization_key(payload, tokenize=True)
        started_wall_ns = time.time_ns()
        started_monotonic_ns = time.monotonic_ns()
        response = self.delegate.post(
            url,
            payload,
            timeout_seconds=timeout_seconds,
        )
        token_ids = require_token_ids(response.get("tokens"), "tokenize response")
        with self.lock:
            existing = self.prompt_tokens.get(key)
            if existing is not None and existing != token_ids:
                raise ModelClientError(
                    "tokenization_nondeterministic",
                    "tokenize endpoint changed exact prompt token ids",
                )
            self.prompt_tokens[key] = token_ids
            ended_wall_ns = time.time_ns()
            ended_monotonic_ns = time.monotonic_ns()
            self.events.append(
                {
                    "phase": "tokenize",
                    "semantic_request_sha256": key,
                    "prompt_token_ids": list(token_ids),
                    "started_wall_ns": started_wall_ns,
                    "ended_wall_ns": ended_wall_ns,
                    "started_monotonic_ns": started_monotonic_ns,
                    "ended_monotonic_ns": ended_monotonic_ns,
                    "duration_ns": max(
                        0, ended_monotonic_ns - started_monotonic_ns
                    ),
                }
            )
        return response

    def post_chat(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        key = tokenization_key(payload, tokenize=False)
        with self.lock:
            expected_prompt_ids = self.prompt_tokens.get(key)
        if expected_prompt_ids is None:
            raise ModelClientError(
                "prompt_token_ids_unbound",
                "chat completion has no matching exact tokenize receipt",
            )

        declared = payload.get("return_token_ids")
        if "return_token_ids" in payload and declared is not True:
            raise ModelClientError(
                "request_shape_drift",
                "chat completion attempted to disable exact token ids",
            )
        outgoing = payload if declared is True else {**payload, "return_token_ids": True}
        started_wall_ns = time.time_ns()
        started_monotonic_ns = time.monotonic_ns()
        response = self.delegate.post(
            url,
            outgoing,
            timeout_seconds=timeout_seconds,
        )
        prompt_ids = require_token_ids(
            response.get("prompt_token_ids"),
            "chat completion prompt",
        )
        if prompt_ids != expected_prompt_ids:
            raise ModelClientError(
                "prompt_token_ids_mismatch",
                "chat completion prompt ids disagree with /tokenize",
            )
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ModelClientError(
                "response_token_ids_missing",
                "chat completion must contain exactly one tokenized choice",
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ModelClientError(
                "response_token_ids_missing",
                "chat completion choice is not an object",
            )
        response_ids = require_token_ids(
            choice.get("token_ids"), "chat completion response"
        )
        with self.lock:
            turn_index = self.turn_index
            self.turn_index += 1
            request_id = (
                scheduler_request_id(
                    **self.request_context,
                    turn_index=turn_index,
                )
                if self.request_context is not None
                else sha256_json(
                    {
                        "semantic_request_sha256": sha256_json(outgoing),
                        "turn_index": turn_index,
                    }
                )
            )
            ended_wall_ns = time.time_ns()
            ended_monotonic_ns = time.monotonic_ns()
            self.events.append(
                {
                    "phase": "chat_completion",
                    "request_id": request_id,
                    "turn_index": turn_index,
                    "semantic_request_sha256": sha256_json(outgoing),
                    "prompt_token_ids": list(prompt_ids),
                    "response_token_ids": list(response_ids),
                    "started_wall_ns": started_wall_ns,
                    "ended_wall_ns": ended_wall_ns,
                    "started_monotonic_ns": started_monotonic_ns,
                    "ended_monotonic_ns": ended_monotonic_ns,
                    "duration_ns": max(
                        0, ended_monotonic_ns - started_monotonic_ns
                    ),
                }
            )
        return response


__all__ = [
    "ExactTokenVllmTransport",
    "require_token_ids",
    "scheduler_request_id",
    "tokenization_key",
]
