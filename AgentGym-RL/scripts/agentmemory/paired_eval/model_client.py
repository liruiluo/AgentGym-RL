"""Deterministic OpenAI-compatible model I/O with exact private evidence."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Protocol, Sequence
from urllib import error, request

from .contracts import DecodingConfig, ModelClientFailure, ModelConfig, ModelOutput
from .evidence import PrivateEvidenceStore
from .serialization import sha256_json


class ModelClientError(ModelClientFailure):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class JsonTransport(Protocol):
    def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        ...


class UrllibJsonTransport:
    def post(self, url, payload, *, timeout_seconds):
        encoded = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                body = response.read()
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            raise ModelClientError("transport_error", str(exc)) from exc
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelClientError(
                "invalid_json", "model endpoint returned invalid JSON"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ModelClientError(
                "invalid_payload", "model endpoint returned non-object JSON"
            )
        return decoded


def exact_token_ids(payload: Any, keys: Sequence[str]) -> Optional[tuple[int, ...]]:
    if not isinstance(payload, Mapping):
        return None
    for key in keys:
        candidate = payload.get(key)
        if not isinstance(candidate, list) or not candidate:
            continue
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in candidate
        ):
            continue
        return tuple(candidate)
    return None


class OpenAICompatibleModelClient:
    """Require a seeded request and exact server-produced token identifiers."""

    def __init__(
        self,
        *,
        base_url: str,
        model_config: ModelConfig,
        transport: JsonTransport,
        evidence_store: PrivateEvidenceStore,
        timeout_seconds: float,
        enable_thinking: bool = False,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be nonempty")
        if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.model_config = model_config
        self.transport = transport
        self.evidence_store = evidence_store
        self.timeout_seconds = float(timeout_seconds)
        self.enable_thinking = bool(enable_thinking)
        self.tokenization_cache: dict[str, tuple[tuple[int, ...], Any]] = {}

    @property
    def api_root(self) -> str:
        return self.base_url if self.base_url.endswith("/v1") else self.base_url + "/v1"

    @property
    def server_root(self) -> str:
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def tokenize(self, messages):
        normalized = [dict(message) for message in messages]
        cache_key = sha256_json(normalized)
        cached = self.tokenization_cache.get(cache_key)
        if cached is not None:
            return cached
        payload = {
            "model": self.model_config.model_id,
            "messages": normalized,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        request_ref = self.evidence_store.put_json("tokenize_requests", payload)
        errors = []
        urls = (self.server_root + "/tokenize", self.api_root + "/tokenize")
        for url in dict.fromkeys(urls):
            try:
                response = self.transport.post(
                    url,
                    payload,
                    timeout_seconds=self.timeout_seconds,
                )
            except Exception as exc:
                errors.append(f"{type(exc).__name__}:{exc}")
                continue
            response_ref = self.evidence_store.put_json("tokenize_responses", response)
            token_ids = exact_token_ids(
                response,
                ("token_ids", "tokens", "prompt_token_ids"),
            )
            if token_ids is None:
                errors.append("tokenize response omitted exact token ids")
                continue
            evidence = self.evidence_store.put_json(
                "tokenization",
                {
                    "request": request_ref.to_payload(),
                    "response": response_ref.to_payload(),
                    "url": url,
                    "prompt_token_ids": list(token_ids),
                },
            )
            result = (token_ids, evidence)
            self.tokenization_cache[cache_key] = result
            return result
        raise ModelClientError(
            "tokenization_unavailable",
            "exact tokenization unavailable: " + " | ".join(errors),
        )

    def count_prompt_tokens(self, messages):
        token_ids, _ = self.tokenize(messages)
        return len(token_ids)

    def complete(self, messages, decoding, seed):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        prompt_token_ids, tokenization_ref = self.tokenize(messages)
        payload = {
            "model": self.model_config.model_id,
            "messages": [dict(message) for message in messages],
            "temperature": float(decoding.temperature),
            "top_p": float(decoding.top_p),
            "max_tokens": decoding.max_output_tokens,
            "seed": seed,
            "n": 1,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if decoding.stop:
            payload["stop"] = list(decoding.stop)
        request_ref = self.evidence_store.put_json("model_requests", payload)
        try:
            response = self.transport.post(
                self.api_root + "/chat/completions",
                payload,
                timeout_seconds=self.timeout_seconds,
            )
        except ModelClientError:
            raise
        except Exception as exc:
            raise ModelClientError("transport_error", str(exc)) from exc
        response_ref = self.evidence_store.put_json("model_responses", response)
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ModelClientError(
                "invalid_choices", "expected exactly one model choice"
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ModelClientError("invalid_choice", "model choice must be an object")
        message = choice.get("message")
        if not isinstance(message, Mapping) or not isinstance(
            message.get("content"), str
        ):
            raise ModelClientError(
                "missing_text", "model choice omitted message content"
            )
        response_token_ids = exact_token_ids(
            choice,
            ("token_ids", "response_token_ids", "tokens"),
        )
        if response_token_ids is None:
            response_token_ids = exact_token_ids(
                response,
                ("response_token_ids", "token_ids"),
            )
        if response_token_ids is None:
            raise ModelClientError(
                "response_token_ids_missing",
                "model response omitted exact generated token ids",
            )
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise ModelClientError("finish_reason_missing", "finish reason is missing")
        return ModelOutput(
            text=message["content"],
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            finish_reason=finish_reason,
            request_ref=request_ref,
            response_ref=response_ref,
            tokenization_ref=tokenization_ref,
            retry_count=0,
        )
