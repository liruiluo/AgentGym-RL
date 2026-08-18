"""Environment-client construction outside veRL's shared rollout implementation."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from requests.exceptions import RequestException

_OPENMLE_IDENTITY_FIELDS = (
    "expected_manifest_sha256",
    "expected_release_revision",
    "expected_outer_commit",
    "expected_inner_commit",
    "expected_role",
    "expected_executor_runtime_digest",
    "expected_materializer_sha256",
    "expected_actions_sha256",
    "expected_max_observation_tokens",
)


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _client_classes() -> dict[str, type]:
    # Import lazily: trainer-only processes register action GAE without loading
    # any environment package or opening an HTTP session.
    from agentenv import envs

    names = {
        "agentmemory": "AgentMemoryEnvClient",
        "webshop": "WebshopEnvClient",
        "swesmith": "SwesmithEnvClient",
        "openmle_fast": "OpenMLEFastEnvClient",
        "searchqa": "SearchQAEnvClient",
    }
    classes: dict[str, type] = {}
    for task_name, class_name in names.items():
        client_cls = getattr(envs, class_name, None)
        if client_cls is not None:
            classes[task_name] = client_cls
    return classes


def create_env_client(config: Any):
    """Create one wrapper-owned environment session from resolved AMG config."""

    task_name = str(_get(config, "task_name", "")).strip().lower()
    env_addr = str(_get(config, "env_addr", "")).rstrip("/")
    parsed = urlparse(env_addr)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"AMG env_addr is not an HTTP endpoint: {env_addr!r}")
    client_cls = _client_classes().get(task_name)
    if client_cls is None:
        raise ValueError(f"unsupported AMG task_name: {task_name!r}")

    timeout = float(_get(config, "timeout", 240.0))
    retries = int(_get(config, "max_retries", 2))
    if timeout <= 0 or retries < 0:
        raise ValueError(
            "AMG client timeout must be positive and max_retries non-negative"
        )

    last_error: Exception | None = None
    client_kwargs: dict[str, Any] = {
        "env_server_base": env_addr,
        "data_len": None,
        "timeout": timeout,
    }
    if task_name == "openmle_fast":
        for field in _OPENMLE_IDENTITY_FIELDS:
            value = _get(config, field)
            if value is None or value == "":
                raise ValueError(
                    f"OpenMLE-fast client identity is missing required field {field!r}"
                )
            client_kwargs[field] = value

    for attempt in range(retries + 1):
        try:
            return client_cls(**client_kwargs)
        except (
            RequestException
        ) as exc:  # endpoint startup races are bounded by max_retries
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(5.0, 0.5 * (2**attempt)))
    assert last_error is not None
    raise RuntimeError(
        f"failed to create AMG {task_name} client after {retries + 1} attempts"
    ) from last_error
