"""Immutable per-row environment routing for AMG's shared AgentLoop.

The registry selects a wrapper client and its policy limits.  It deliberately
contains no scheduler, queue, rollout, reward, or environment lifecycle logic;
those remain owned by upstream veRL and the selected environment wrapper.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse

_ROUTE_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_AGENT_NAME = "amg_task_neutral_async"
_REGISTRY_SCHEMA = "amg_route_registry_v1"
_CANONICAL_ROUTE_IDS = (
    "webshop",
    "swesmith",
    "literesearcher",
    "openmle_fast",
)
_CONTROL_FIELDS = {
    "route_registry_path",
    "route_registry_sha256",
    "route_registry_expected_ids",
    "route_id",
}


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _plain_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    items = getattr(value, "items", None)
    if callable(items):
        return {str(key): item for key, item in items()}
    raise TypeError(f"{field} must be a mapping")


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a positive integer, not bool")
    try:
        integer = int(value)
        exact = float(value) == float(integer)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive integer, got {value!r}") from exc
    if not exact or integer <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return integer


def _sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _route_id(value: Any, *, field: str = "route_id") -> str:
    route_id = str(value or "").strip().lower()
    if not _ROUTE_ID.fullmatch(route_id):
        raise ValueError(f"{field} is invalid: {value!r}")
    return route_id


def _loopback_endpoint(value: Any, *, field: str) -> str:
    endpoint = str(value or "").rstrip("/")
    parsed = urlparse(endpoint)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field} has an invalid port: {endpoint!r}") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in _LOOPBACK_HOSTS
        or port is None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            f"{field} must be a same-Pod loopback HTTP endpoint, got {endpoint!r}"
        )
    return endpoint


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


def normalize_policy_framing(messages: Any) -> tuple[Mapping[str, str], ...]:
    """Normalize legacy ``from/value`` and modern ``role/content`` messages."""

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("policy framing must be a non-empty message sequence")
    normalized: list[Mapping[str, str]] = []
    role_aliases = {"human": "user", "gpt": "assistant"}
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            try:
                message = dict(message)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"policy framing message {index} must be a mapping"
                ) from exc
        role = message.get("role")
        if role is None:
            role = message.get("from")
        content = message.get("content")
        if content is None:
            content = message.get("value")
        role = role_aliases.get(role, role)
        if role not in {"system", "user", "assistant"} or not isinstance(
            content, str
        ):
            raise ValueError(
                f"policy framing message {index} has invalid role/content"
            )
        normalized.append(MappingProxyType({"role": str(role), "content": content}))
    if not normalized:
        raise ValueError("policy framing must not be empty")
    return tuple(normalized)


def canonical_policy_framing_sha256(messages: Any) -> str:
    """Hash one wrapper-owned policy framing using a stable JSON encoding."""

    normalized = [dict(message) for message in normalize_policy_framing(messages)]
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RouteSpec:
    """One immutable route from a dataset row to a wrapper client."""

    route_id: str
    max_rounds: int
    max_observation_tokens: int
    policy_framing_sha256: str | None
    route_attestation_sha256: str | None
    client_config: Mapping[str, Any]


class RouteRegistry:
    """Validated immutable route registry with per-row selection."""

    def __init__(
        self,
        *,
        routes: Sequence[RouteSpec],
        sha256: str | None,
        source_path: Path | None,
        agent_name: str = _AGENT_NAME,
    ) -> None:
        if agent_name != _AGENT_NAME:
            raise ValueError(
                "AMG route registry agent_name must be "
                f"{_AGENT_NAME!r}, got {agent_name!r}"
            )
        if not routes:
            raise ValueError("AMG route registry must contain at least one route")
        by_id: dict[str, RouteSpec] = {}
        for route in routes:
            if route.route_id in by_id:
                raise ValueError(f"duplicate route_id {route.route_id!r}")
            by_id[route.route_id] = route
        self._routes = tuple(routes)
        self._by_id = MappingProxyType(by_id)
        self.sha256 = sha256
        self.source_path = source_path
        self.agent_name = agent_name

    @property
    def route_ids(self) -> tuple[str, ...]:
        return tuple(route.route_id for route in self._routes)

    @property
    def routes(self) -> tuple[RouteSpec, ...]:
        return self._routes

    def resolve(self, route_id: Any) -> RouteSpec:
        normalized = _route_id(route_id)
        try:
            return self._by_id[normalized]
        except KeyError as exc:
            raise ValueError(f"unknown AMG route_id {normalized!r}") from exc

    def resolve_row(self, row: Mapping[str, Any]) -> RouteSpec:
        top = row.get("route_id")
        extra = row.get("extra_info")
        nested = extra.get("route_id") if isinstance(extra, Mapping) else None
        if top is not None and nested is not None:
            top_id = _route_id(top, field="row.route_id")
            nested_id = _route_id(nested, field="row.extra_info.route_id")
            if top_id != nested_id:
                raise ValueError(
                    "AMG schedule route_id drift: "
                    f"row={top_id!r} extra_info={nested_id!r}"
                )
            return self.resolve(top_id)
        selected = top if top is not None else nested
        if selected is None:
            if len(self._routes) == 1:
                return self._routes[0]
            raise ValueError("AMG multi-environment schedule row is missing route_id")
        return self.resolve(selected)


def _route_spec(raw: Any, *, require_attestation: bool) -> RouteSpec:
    route = _plain_mapping(raw, field="route registry entry")
    route_id = _route_id(route.get("route_id"))
    max_rounds = _positive_int(
        route.get("max_rounds"), field=f"route {route_id} max_rounds"
    )
    max_observation_tokens = _positive_int(
        route.get("max_observation_tokens"),
        field=f"route {route_id} max_observation_tokens",
    )
    framing_digest_raw = route.get("policy_framing_sha256")
    framing_digest = (
        _sha256(
            framing_digest_raw,
            field=f"route {route_id} policy_framing_sha256",
        )
        if framing_digest_raw is not None
        else None
    )
    attestation_raw = route.get("route_attestation_sha256")
    attestation = (
        _sha256(
            attestation_raw,
            field=f"route {route_id} route_attestation_sha256",
        )
        if attestation_raw is not None
        else None
    )
    if require_attestation and framing_digest is None:
        raise ValueError(f"route {route_id} is missing policy_framing_sha256")
    client = _plain_mapping(route.get("client"), field=f"route {route_id} client")
    task_name = str(client.get("task_name", "")).strip().lower()
    if not task_name:
        raise ValueError(f"route {route_id} client.task_name is missing")
    client["task_name"] = task_name
    client["env_addr"] = _loopback_endpoint(
        client.get("env_addr"), field=f"route {route_id} client.env_addr"
    )
    client["timeout"] = float(client.get("timeout", 240.0))
    if not math.isfinite(client["timeout"]) or client["timeout"] <= 0:
        raise ValueError(f"route {route_id} client.timeout must be positive")
    retries = client.get("max_retries", 2)
    if isinstance(retries, bool):
        raise TypeError(f"route {route_id} client.max_retries must be non-negative")
    try:
        retries_int = int(retries)
        retries_exact = float(retries) == float(retries_int)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"route {route_id} client.max_retries must be non-negative"
        ) from exc
    if not retries_exact or retries_int < 0:
        raise ValueError(f"route {route_id} client.max_retries must be non-negative")
    client["max_retries"] = retries_int
    if require_attestation:
        client_attestation_raw = client.pop("route_attestation_sha256", None)
        client_attestation = (
            _sha256(
                client_attestation_raw,
                field=f"route {route_id} client.route_attestation_sha256",
            )
            if client_attestation_raw is not None
            else None
        )
        if (
            attestation is not None
            and client_attestation is not None
            and attestation != client_attestation
        ):
            raise ValueError(
                f"route {route_id} route attestation drift between route and client"
            )
        if attestation is None:
            attestation = client_attestation
        if attestation is None:
            raise ValueError(f"route {route_id} is missing route_attestation_sha256")
    return RouteSpec(
        route_id=route_id,
        max_rounds=max_rounds,
        max_observation_tokens=max_observation_tokens,
        policy_framing_sha256=framing_digest,
        route_attestation_sha256=attestation,
        client_config=_freeze(client),
    )


def load_route_registry(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_route_ids: Sequence[str] | None = None,
) -> RouteRegistry:
    """Load a pinned regular JSON registry and validate every route."""

    registry_path = Path(path)
    if registry_path.is_symlink() or not registry_path.is_file():
        raise ValueError(
            f"AMG route registry must be a regular non-symlink file: {registry_path}"
        )
    expected_digest = _sha256(expected_sha256, field="route registry expected sha256")
    observed_digest = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    if observed_digest != expected_digest:
        raise ValueError(
            "AMG route registry sha256 mismatch: "
            f"expected {expected_digest}, got {observed_digest}"
        )
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid AMG route registry JSON: {registry_path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != _REGISTRY_SCHEMA:
        raise ValueError(f"AMG route registry schema must be {_REGISTRY_SCHEMA!r}")
    raw_routes = payload.get("routes")
    if isinstance(raw_routes, (str, bytes)) or not isinstance(raw_routes, Sequence):
        raise TypeError("AMG route registry routes must be a sequence")
    registry = RouteRegistry(
        routes=[_route_spec(route, require_attestation=True) for route in raw_routes],
        sha256=observed_digest,
        source_path=registry_path.resolve(),
        agent_name=str(payload.get("agent_name", "")),
    )
    route_ids = registry.route_ids
    unknown = tuple(
        route_id for route_id in route_ids if route_id not in _CANONICAL_ROUTE_IDS
    )
    canonical_subset = tuple(
        route_id for route_id in _CANONICAL_ROUTE_IDS if route_id in route_ids
    )
    if unknown or route_ids != canonical_subset:
        raise ValueError(
            "AMG route registry must be a non-empty canonical ordered subset: "
            f"{route_ids!r}; canonical={_CANONICAL_ROUTE_IDS!r}"
        )
    if expected_route_ids is not None:
        if isinstance(expected_route_ids, (str, bytes)):
            raise TypeError("expected_route_ids must be a sequence, not a string")
        expected = tuple(
            _route_id(value, field="expected route_id")
            for value in expected_route_ids
        )
        if expected != route_ids:
            raise ValueError(
                "expected route IDs differ from the registry route order: "
                f"{expected!r} != {route_ids!r}"
            )
    return registry


def route_registry_from_agentgym_config(config: Any) -> RouteRegistry:
    """Resolve either a pinned multi-route registry or the legacy one-route config."""

    path = _get(config, "route_registry_path")
    digest = _get(config, "route_registry_sha256")
    if (path is None) != (digest is None):
        raise ValueError(
            "AMG route_registry_path and route_registry_sha256 must be "
            "provided together"
        )
    if path is not None:
        expected_ids = _get(config, "route_registry_expected_ids")
        return load_route_registry(
            path,
            expected_sha256=str(digest),
            expected_route_ids=expected_ids,
        )

    raw = _plain_mapping(config, field="agentgym config")
    route_id = _route_id(raw.get("route_id", raw.get("task_name")))
    max_rounds = raw.get("max_rounds", 30)
    max_observation_tokens = raw.get("max_observation_tokens", 8192)
    client = {key: value for key, value in raw.items() if key not in _CONTROL_FIELDS}
    return RouteRegistry(
        routes=[
            _route_spec(
                {
                    "route_id": route_id,
                    "max_rounds": max_rounds,
                    "max_observation_tokens": max_observation_tokens,
                    "client": client,
                },
                require_attestation=False,
            )
        ],
        sha256=None,
        source_path=None,
    )
