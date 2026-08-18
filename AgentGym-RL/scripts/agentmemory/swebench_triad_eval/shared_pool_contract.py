"""Exact structural contract for a live shared-model-pool snapshot."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SHARED_MODEL_POOL_ASSIGNMENT = "uint64_be(sha256(task_id)[:8]) % 8"
SHARED_MODEL_POOL_CLEANUP = "retain_external_pool"
SHARED_MODEL_POOL_LISTENER_CENSUS_FIELDS = frozenset(
    {"source", "family", "address", "port", "inode", "owner_pids"}
)
SHARED_MODEL_POOL_LISTENER_SOURCE = "/proc/net/tcp"
SHARED_MODEL_POOL_LISTENER_FAMILY = "ipv4"
SHARED_MODEL_POOL_LISTENER_ADDRESS = "127.0.0.1"
SHARED_MODEL_POOL_SNAPSHOT_FIELDS = frozenset(
    {
        "status",
        "owner",
        "readiness_sha256",
        "marker_lease_sha256",
        "replica_index",
        "replica_count",
        "gpu_index",
        "gpu_uuid",
        "model_id",
        "model_revision",
        "model_port",
        "proxy_port",
        "server_pid",
        "server_start_ticks",
        "server_target_pids",
        "server_listener_pids",
        "server_listener_census",
        "proxy_pid",
        "proxy_start_ticks",
        "proxy_target_pids",
        "proxy_listener_pids",
        "proxy_listener_census",
        "proxy_route",
        "assigned_gpu_process_pids",
        "all_replicas_alive",
        "all_endpoints_healthy",
        "assignment_algorithm",
        "cleanup_policy",
    }
)
SHARED_MODEL_POOL_PROXY_ROUTE_FIELDS = frozenset(
    {
        "config_path",
        "config_sha256",
        "proxy_source_sha256",
        "runtime_sha256",
        "tokenizer_sha256",
        "upstream_base_url",
        "upstream_base_url_sha256",
    }
)


class SharedModelPoolSnapshotError(RuntimeError):
    """The shared-model-pool snapshot violated its exact contract."""


def _fail(label: str, detail: str) -> None:
    raise SharedModelPoolSnapshotError(f"{label} {detail}")


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_pid_list(value: Any, label: str, field: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(pid) is not int or pid <= 0 for pid in value)
        or value != sorted(set(value))
    ):
        _fail(label, f"{field} drifted")
    return value


def _listener_census(
    value: Any,
    label: str,
    field: str,
    *,
    expected_port: int,
    expected_owner_pids: list[int],
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != SHARED_MODEL_POOL_LISTENER_CENSUS_FIELDS
    ):
        _fail(label, f"{field} listener census fields drifted")
    if (
        value["source"] != SHARED_MODEL_POOL_LISTENER_SOURCE
        or value["family"] != SHARED_MODEL_POOL_LISTENER_FAMILY
        or value["address"] != SHARED_MODEL_POOL_LISTENER_ADDRESS
        or value["port"] != expected_port
        or type(value["inode"]) is not int
        or value["inode"] <= 0
    ):
        _fail(label, f"{field} listener census endpoint drifted")
    owners = _positive_pid_list(
        value["owner_pids"], label, f"{field} listener census owner PIDs"
    )
    if owners != expected_owner_pids:
        _fail(label, f"{field} listener census ownership drifted")
    return value


def validate_shared_model_pool_snapshot(
    value: Any,
    label: str = "shared model pool snapshot",
    *,
    listener_reference: Any = None,
) -> Mapping[str, Any]:
    """Validate the one exact shape emitted into all durable receipts."""

    if not isinstance(value, Mapping):
        _fail(label, "is missing")
    if set(value) != SHARED_MODEL_POOL_SNAPSHOT_FIELDS:
        _fail(label, "fields drifted")
    for field in ("owner", "gpu_uuid", "model_id", "model_revision"):
        if not isinstance(value[field], str) or not value[field]:
            _fail(label, f"{field} drifted")
    for field in ("readiness_sha256", "marker_lease_sha256"):
        if not _sha256(value[field]):
            _fail(label, f"{field} drifted")
    for field in (
        "replica_count",
        "model_port",
        "proxy_port",
        "server_pid",
        "server_start_ticks",
        "proxy_pid",
        "proxy_start_ticks",
    ):
        if type(value[field]) is not int or value[field] <= 0:
            _fail(label, f"{field} drifted")
    for field in ("replica_index", "gpu_index"):
        if type(value[field]) is not int or value[field] < 0:
            _fail(label, f"{field} drifted")
    if (
        value["replica_index"] >= value["replica_count"]
        or value["gpu_index"] >= value["replica_count"]
        or value["gpu_index"] != value["replica_index"]
    ):
        _fail(label, "replica index drifted")
    if not 1 <= value["model_port"] <= 65535 or not 1 <= value[
        "proxy_port"
    ] <= 65535:
        _fail(label, "port drifted")
    if value["model_port"] == value["proxy_port"]:
        _fail(label, "ports collided")
    if (
        value["status"] != "PASS"
        or value["all_replicas_alive"] is not True
        or value["all_endpoints_healthy"] is not True
        or value["assignment_algorithm"] != SHARED_MODEL_POOL_ASSIGNMENT
        or value["cleanup_policy"] != SHARED_MODEL_POOL_CLEANUP
    ):
        _fail(label, "identity drifted")

    server_targets = _positive_pid_list(
        value["server_target_pids"], label, "server target PIDs"
    )
    server_listeners = _positive_pid_list(
        value["server_listener_pids"], label, "server listener PIDs"
    )
    proxy_targets = _positive_pid_list(
        value["proxy_target_pids"], label, "proxy target PIDs"
    )
    proxy_listeners = _positive_pid_list(
        value["proxy_listener_pids"], label, "proxy listener PIDs"
    )
    _positive_pid_list(
        value["assigned_gpu_process_pids"], label, "assigned GPU process PIDs"
    )
    if not set(server_listeners).issubset(server_targets):
        _fail(label, "server listener binding drifted")
    if not set(proxy_listeners).issubset(proxy_targets):
        _fail(label, "proxy listener binding drifted")
    _listener_census(
        value["server_listener_census"],
        label,
        "server",
        expected_port=value["model_port"],
        expected_owner_pids=server_listeners,
    )
    _listener_census(
        value["proxy_listener_census"],
        label,
        "proxy",
        expected_port=value["proxy_port"],
        expected_owner_pids=proxy_listeners,
    )
    if listener_reference is not None:
        if not isinstance(listener_reference, Mapping):
            _fail(label, "listener census reference is missing")
        for field in ("server_listener_census", "proxy_listener_census"):
            if value[field] != listener_reference.get(field):
                _fail(label, "listener census drifted")

    route = value["proxy_route"]
    if (
        not isinstance(route, Mapping)
        or set(route) != SHARED_MODEL_POOL_PROXY_ROUTE_FIELDS
    ):
        _fail(label, "proxy route fields drifted")
    if not isinstance(route["config_path"], str) or not Path(
        route["config_path"]
    ).is_absolute():
        _fail(label, "proxy route config path drifted")
    for field in (
        "config_sha256",
        "proxy_source_sha256",
        "runtime_sha256",
        "tokenizer_sha256",
        "upstream_base_url_sha256",
    ):
        if not _sha256(route[field]):
            _fail(label, f"proxy route {field} drifted")
    upstream = route["upstream_base_url"]
    expected_upstream = f"http://127.0.0.1:{value['model_port']}"
    if (
        upstream != expected_upstream
        or route["upstream_base_url_sha256"]
        != hashlib.sha256(upstream.encode("utf-8")).hexdigest()
    ):
        _fail(label, "proxy route upstream binding drifted")
    return value


__all__ = [
    "SHARED_MODEL_POOL_ASSIGNMENT",
    "SHARED_MODEL_POOL_CLEANUP",
    "SHARED_MODEL_POOL_LISTENER_ADDRESS",
    "SHARED_MODEL_POOL_LISTENER_CENSUS_FIELDS",
    "SHARED_MODEL_POOL_LISTENER_FAMILY",
    "SHARED_MODEL_POOL_LISTENER_SOURCE",
    "SHARED_MODEL_POOL_PROXY_ROUTE_FIELDS",
    "SHARED_MODEL_POOL_SNAPSHOT_FIELDS",
    "SharedModelPoolSnapshotError",
    "validate_shared_model_pool_snapshot",
]
