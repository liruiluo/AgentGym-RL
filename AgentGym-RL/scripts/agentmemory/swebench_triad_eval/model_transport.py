"""Fail-closed exact-token transport for the frozen vLLM endpoint."""

from __future__ import annotations

from threading import Lock
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


class ExactTokenVllmTransport:
    """Request and reconcile vLLM's prompt and response token identifiers."""

    def __init__(self, delegate: JsonTransport) -> None:
        if not callable(getattr(delegate, "post", None)):
            raise TypeError("exact-token transport delegate must implement post")
        self.delegate = delegate
        self.prompt_tokens: dict[str, tuple[int, ...]] = {}
        self.lock = Lock()

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
        require_token_ids(choice.get("token_ids"), "chat completion response")
        return response


__all__ = ["ExactTokenVllmTransport", "require_token_ids", "tokenization_key"]
