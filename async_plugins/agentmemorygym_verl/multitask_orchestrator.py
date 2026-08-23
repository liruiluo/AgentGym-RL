# ruff: noqa: BLE001
"""Thin process orchestrator for one four-route native fully-async run.

Environment launch commands and identities live in a hash-pinned external
registry.  This module validates those inputs, supervises the existing
endpoint entrypoints, and delegates all training behavior to ``launch.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol
from urllib.parse import urlparse

from .config_contract import inspect_schedule
from .identity import EXPECTED_VERL_COMMIT
from .launch import LaunchInputs, _load_multitask_identity
from .orchestrator_lifecycle import (
    _signal_process_identity,
    acquire_marker_transaction,
    prepare_marker_transaction,
    process_identity_alive,
    process_start_ticks,
    restore_marker_transaction,
)
from .routes import RouteRegistry, load_route_registry

EXPECTED_ROUTE_IDS = (
    "webshop",
    "swesmith",
    "literesearcher",
    "openmle_fast",
)
_CONFIG_SCHEMA = "amg_multitask400_orchestrator_config_v1"
_ENDPOINT_REGISTRY_SCHEMA = "amg_multitask_endpoint_registry_v1"
_GATE_RECEIPT_SCHEMA = "amg_single_card_optimizer_update_gate_v1"
_GATE_ENVIRONMENT_NAMES = {
    "webshop": "webshop",
    "swesmith": "swesmith",
    "literesearcher": "literesearcher",
    "openmle_fast": "openmle-fast",
}
_HOLDER_LEASE_SCHEMA = "amg_holder_lease_v1"
_PREFLIGHT_SCHEMA = "amg_multitask_orchestrator_preflight_v1"
_ORCHESTRATOR_RECEIPT_SCHEMA = "amg_multitask_orchestrator_receipt_v1"
_IMPLEMENTATION_BASE_COMMIT = "4d8ce04b5d40c2e79abb01b46051a230c7ab3973"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_HOLDER_MARKER_PATHS = {
    "cpu": Path("/tmp/agentmemory-formal-cpu-active"),
    "gpu": Path("/tmp/crg-holder-yield"),
}
_HEX = frozenset("0123456789abcdef")
_RECEIPT_SOURCE_COMMIT_FIELDS = {
    "outer": frozenset(
        {
            "source.environment_outer_source_commit",
            "source.shared_runtime_source_commit",
        }
    ),
    "inner": frozenset({"source.environment_source_commit"}),
}


class OrchestratorError(RuntimeError):
    """Fail-closed launch-contract violation."""


class _TerminationRequested(RuntimeError):
    def __init__(self, signum: int) -> None:
        super().__init__(f"termination requested by signal {signum}")
        self.signum = signum


@contextlib.contextmanager
def _termination_guard() -> Any:
    watched = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.getsignal(signum) for signum in watched}
    previous_sigchld = signal.getsignal(signal.SIGCHLD)

    def request_termination(signum: int, _frame: Any) -> None:
        for watched_signal in watched:
            signal.signal(watched_signal, signal.SIG_IGN)
        raise _TerminationRequested(signum)

    try:
        # A dead direct child must remain as a zombie until the supervisor has
        # authenticated and drained its process group.  Auto-reaping would
        # discard the only safe PGID anchor after an external SIGKILL/OOM.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        for signum in watched:
            signal.signal(signum, request_termination)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        signal.signal(signal.SIGCHLD, previous_sigchld)


@dataclass(frozen=True)
class OrchestratorConfig:
    source_path: Path
    sha256: str
    route_order: tuple[str, ...]
    optimizer_updates: int
    samples_per_update: int
    total_episodes: int
    trainer_gpus: int
    standalone_rollout_gpus: int
    rollout_n: int
    critic_active_token_budget: int
    trigger_parameter_sync_step: int
    actor_use_fused_kernels: bool
    critic_use_fused_kernels: bool
    require_exact_per_update_route_split: bool
    sampling_order: str
    holder_lock_path: Path


@dataclass(frozen=True)
class EndpointLaunchSpec:
    route_id: str
    route_attestation_sha256: str
    endpoint: str
    gate_receipt_path: Path
    gate_receipt_sha256: str
    launcher_path: Path
    launcher_sha256: str
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    working_directory: Path
    readiness_url: str
    readiness_expected: Mapping[str, Any]
    readiness_sha256: str | None
    ready_timeout_seconds: float
    poll_seconds: float
    request_timeout_seconds: float
    cleanup_timeout_seconds: float

    @classmethod
    def for_test(cls, *, route_id: str, endpoint: str) -> EndpointLaunchSpec:
        return cls(
            route_id=route_id,
            route_attestation_sha256="0" * 64,
            endpoint=endpoint,
            gate_receipt_path=Path("/tmp/gate-receipt.json"),
            gate_receipt_sha256="0" * 64,
            launcher_path=Path("/bin/true"),
            launcher_sha256="0" * 64,
            argv=(),
            environment={},
            working_directory=Path("/tmp"),
            readiness_url=f"{endpoint}/metadata",
            readiness_expected={},
            readiness_sha256=None,
            ready_timeout_seconds=1.0,
            poll_seconds=0.01,
            request_timeout_seconds=0.1,
            cleanup_timeout_seconds=1.0,
        )


@dataclass(frozen=True)
class ProcessLease:
    name: str
    pid: int
    start_ticks: str
    process: Any
    log_handle: BinaryIO | None
    cleanup_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class MarkerLease:
    name: str
    path: Path
    original_value: str | None
    original_pid: int
    original_start_ticks: str


@dataclass(frozen=True)
class HolderLease:
    source_path: Path
    sha256: str
    markers: tuple[MarkerLease, ...]
    yield_checks: tuple[Mapping[str, str], ...]
    restore_checks: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class LaunchPlan:
    config: OrchestratorConfig
    outer_root: Path
    verl_root: Path
    schedule: Path
    route_registry_path: Path
    route_registry_sha256: str
    multitask_source_lock: Path
    multitask_schedule_certificate: Path
    endpoint_registry_path: Path
    endpoint_registry_sha256: str
    run_dir: Path
    experiment_name: str
    endpoints: tuple[EndpointLaunchSpec, ...]
    endpoint_report: Mapping[str, Any]
    schedule_report: Mapping[str, Any]
    launch_identity: Mapping[str, Any]
    generic_launcher: Path
    resolve_only: bool
    holder_lease: HolderLease | None = None

    @classmethod
    def for_test(
        cls,
        *,
        resolve_only: bool,
        config: OrchestratorConfig | None = None,
    ) -> LaunchPlan:
        if config is None:
            config = OrchestratorConfig(
                source_path=Path("/config.yaml"),
                sha256="0" * 64,
                route_order=EXPECTED_ROUTE_IDS,
                optimizer_updates=400,
                samples_per_update=64,
                total_episodes=25_600,
                trainer_gpus=6,
                standalone_rollout_gpus=2,
                rollout_n=1,
                critic_active_token_budget=32_768,
                trigger_parameter_sync_step=1,
                actor_use_fused_kernels=False,
                critic_use_fused_kernels=False,
                require_exact_per_update_route_split=False,
                sampling_order="round_robin",
                holder_lock_path=Path("/tmp/amg-holder-marker-transaction.lock"),
            )
        return cls(
            config=config,
            outer_root=Path("/outer"),
            verl_root=Path("/verl"),
            schedule=Path("/schedule.jsonl"),
            route_registry_path=Path("/route-registry.json"),
            route_registry_sha256="1" * 64,
            multitask_source_lock=Path("/source-lock.json"),
            multitask_schedule_certificate=Path("/schedule-certificate.json"),
            endpoint_registry_path=Path("/endpoint-registry.json"),
            endpoint_registry_sha256="2" * 64,
            run_dir=Path("/run"),
            experiment_name="multitask400-test",
            endpoints=(),
            endpoint_report={},
            schedule_report={"sha256": "3" * 64, "count": 25_600},
            launch_identity={},
            generic_launcher=Path(
                "/outer/async_plugins/scripts/launch_amg_fully_async.sh"
            ),
            resolve_only=resolve_only,
            holder_lease=None,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, *, field: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise OrchestratorError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OrchestratorError(f"{field} must be a positive integer")
    return value


def _positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise OrchestratorError(f"{field} must be a positive number")
    return float(value)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OrchestratorError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OrchestratorError(f"{field} must be a sequence")
    return value


def _absolute_path(value: Any, *, field: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        raise OrchestratorError(f"{field} must be an absolute path")
    return path


def _regular_file(value: Any, *, field: str, executable: bool = False) -> Path:
    path = _absolute_path(value, field=field)
    if path.is_symlink() or not path.is_file():
        raise OrchestratorError(f"{field} is missing or not a regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise OrchestratorError(f"{field} is not executable: {path}")
    return path.resolve()


def _directory(value: Any, *, field: str) -> Path:
    path = _absolute_path(value, field=field)
    if path.is_symlink() or not path.is_dir():
        raise OrchestratorError(f"{field} is missing or not a directory: {path}")
    return path.resolve()


def _load_json_file(path: Path, *, field: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OrchestratorError(f"{field} is missing or not regular: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestratorError(f"{field} is not valid JSON: {path}: {exc}") from exc
    return _mapping(payload, field=field)


def _load_yaml_file(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OrchestratorError(f"orchestrator config is missing or symlinked: {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - publication runtime owns PyYAML
        raise OrchestratorError(
            "PyYAML is required for the orchestrator config"
        ) from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OrchestratorError(f"invalid orchestrator config {path}: {exc}") from exc
    return _mapping(payload, field="orchestrator config")


def load_orchestrator_config(path: Path) -> OrchestratorConfig:
    """Load the reviewed experiment contract and reject every tunable drift."""

    path = Path(path)
    payload = _load_yaml_file(path)
    source = _mapping(payload.get("source"), field="config.source")
    experiment = _mapping(payload.get("experiment"), field="config.experiment")
    budget = _mapping(payload.get("budget"), field="config.budget")
    routing = _mapping(payload.get("routing"), field="config.routing")
    r38 = _mapping(payload.get("r38"), field="config.r38")
    runtime_inputs = _mapping(
        payload.get("runtime_inputs"), field="config.runtime_inputs"
    )
    holders = _mapping(
        payload.get("holder_transaction"), field="config.holder_transaction"
    )
    required_runtime_inputs = {
        "route_registry": "cli:--route-registry",
        "route_registry_sha256": "cli:--route-registry-sha256",
        "multitask_manifest": "cli:--schedule",
        "multitask_schedule_certificate": "cli:--multitask-schedule-certificate",
        "multitask_source_lock": "cli:--multitask-source-lock",
        "endpoint_registry": "cli:--endpoint-registry",
        "endpoint_registry_sha256": "cli:--endpoint-registry-sha256",
        "gate_receipts": "endpoint_registry.routes[*].gate_receipt",
        "holder_lease": "cli:--holder-lease",
        "holder_lease_sha256": "cli:--holder-lease-sha256",
    }
    exact = {
        "schema": (payload.get("schema"), _CONFIG_SCHEMA),
        "implementation base commit": (
            source.get("implementation_base_commit"),
            _IMPLEMENTATION_BASE_COMMIT,
        ),
        "veRL commit": (source.get("verl_commit"), EXPECTED_VERL_COMMIT),
        "mode": (experiment.get("mode"), "formal"),
        "model family": (experiment.get("model_family"), "Qwen3.5-4B"),
        "fresh model": (experiment.get("fresh_model"), True),
        "resume mode": (experiment.get("resume_mode"), "disable"),
        "agent name": (experiment.get("agent_name"), "amg_task_neutral_async"),
        "shared actor count": (experiment.get("shared_actor_count"), 1),
        "shared critic count": (experiment.get("shared_critic_count"), 1),
        "checkpoint lineage count": (
            experiment.get("checkpoint_lineage_count"),
            1,
        ),
        "optimizer updates": (budget.get("optimizer_updates"), 400),
        "episodes per update": (
            budget.get("consumed_episodes_per_update"),
            64,
        ),
        "total episodes": (budget.get("total_episodes"), 25_600),
        "route order": (tuple(routing.get("order", ())), EXPECTED_ROUTE_IDS),
        "sampling": (routing.get("sampling"), "round_robin"),
        "per-update route quota": (
            routing.get("require_exact_per_update_route_split"),
            False,
        ),
        "learner/hybrid GPUs": (r38.get("learner_hybrid_gpus"), 6),
        "standalone rollout GPUs": (r38.get("standalone_rollout_gpus"), 2),
        "rollout.n": (r38.get("rollout_n"), 1),
        "critic active token budget": (
            r38.get("critic_active_token_budget"),
            32_768,
        ),
        "parameter sync trigger": (
            r38.get("trigger_parameter_sync_step"),
            1,
        ),
        "actor fused kernels": (r38.get("actor_use_fused_kernels"), False),
        "critic fused kernels": (r38.get("critic_use_fused_kernels"), False),
        "runtime input contract": (dict(runtime_inputs), required_runtime_inputs),
        "marker transaction schema": (
            holders.get("schema"),
            "amg_marker_transaction_v1",
        ),
        "required holder markers": (
            tuple(holders.get("required_markers", ())),
            ("cpu", "gpu"),
        ),
        "holder marker paths": (
            holders.get("marker_paths"),
            {name: str(path) for name, path in _HOLDER_MARKER_PATHS.items()},
        ),
    }
    for field, (observed, expected) in exact.items():
        if observed != expected:
            raise OrchestratorError(
                f"reviewed Multitask400 {field} drifted: {observed!r} != {expected!r}"
            )
    optimizer_updates = _positive_int(
        budget["optimizer_updates"], field="optimizer updates"
    )
    samples_per_update = _positive_int(
        budget["consumed_episodes_per_update"], field="episodes per update"
    )
    total_episodes = _positive_int(budget["total_episodes"], field="total episodes")
    if optimizer_updates * samples_per_update != total_episodes:
        raise OrchestratorError(
            "Multitask400 arithmetic drift: optimizer_updates * "
            "consumed_episodes_per_update != total_episodes"
        )
    return OrchestratorConfig(
        source_path=path.resolve(),
        sha256=_sha256(path),
        route_order=EXPECTED_ROUTE_IDS,
        optimizer_updates=optimizer_updates,
        samples_per_update=samples_per_update,
        total_episodes=total_episodes,
        trainer_gpus=6,
        standalone_rollout_gpus=2,
        rollout_n=1,
        critic_active_token_budget=32_768,
        trigger_parameter_sync_step=1,
        actor_use_fused_kernels=False,
        critic_use_fused_kernels=False,
        require_exact_per_update_route_split=False,
        sampling_order="round_robin",
        holder_lock_path=_absolute_path(
            holders.get("lock_path"), field="config holder lock_path"
        ),
    )


def _nested_value(payload: Mapping[str, Any], dotted: str) -> Any:
    value: Any = payload
    for component in dotted.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise OrchestratorError(f"receipt is missing required field {dotted}")
        value = value[component]
    return value


def _assert_expected_subset(actual: Any, expected: Any, *, field: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise OrchestratorError(f"{field} must be a mapping")
        for key, value in expected.items():
            if key not in actual:
                raise OrchestratorError(f"{field}.{key} is missing")
            _assert_expected_subset(actual[key], value, field=f"{field}.{key}")
        return
    if actual != expected:
        raise OrchestratorError(f"{field} mismatch: {actual!r} != {expected!r}")


def _validate_gate_receipt(
    *,
    route_id: str,
    receipt_path: Path,
    expected_sha256: str,
    gate_launcher_sha256: str,
    runtime_manifest_sha256: str,
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    observed_sha256 = _sha256(receipt_path)
    if observed_sha256 != expected_sha256:
        raise OrchestratorError(
            f"{route_id} gate receipt sha256 mismatch: "
            f"{observed_sha256} != {expected_sha256}"
        )
    receipt = _load_json_file(receipt_path, field=f"{route_id} gate receipt")
    required = {
        "schema": _GATE_RECEIPT_SCHEMA,
        "environment": _GATE_ENVIRONMENT_NAMES[route_id],
        "status": "pass",
        "execution.gpu_count": 1,
        "training.optimizer_update_count": 1,
        "training.trainer_exit_code": 0,
        "training.update1_completed": True,
        "training.actor_parameter_delta_nonzero": True,
        "training.critic_parameter_delta_nonzero": True,
        "runtime.environment_ready": True,
        "runtime.asset_hashes_verified": True,
        "runtime.fatal_error_count": 0,
        "runtime.forwarding_process_count": 0,
        "runtime.listener_scope": "same_pod_loopback_only",
        "cleanup.residue_after_cleanup": 0,
        "cleanup.markers_cleared": True,
        "cleanup.checkpoint_readback": True,
        "source.launcher_sha256": gate_launcher_sha256,
        "source.runtime_manifest_sha256": runtime_manifest_sha256,
    }
    if route_id == "swesmith":
        required.update(
            {
                "runtime.formal_eligible": True,
                "runtime.sandbox_backend": "LinuxNamespaceEpisodeSandbox",
                "runtime.sandbox_contract": ("swesmith_linux_namespace_oci_rootfs_v1"),
                "runtime.rootfs_contract": (
                    "digest_pinned_oci_profile_rootfs_read_only"
                ),
                "runtime.network_contract": (
                    "new_namespace_loopback_only_no_external_routes"
                ),
                "cleanup.sandbox_mount_count_after_cleanup": 0,
                "cleanup.holders_restored": True,
                "cleanup.temporary_path_count_after_cleanup": 0,
                "source.source_worktree_dirty": False,
                "source.environment_source_detached": True,
                "source.shared_runtime_source_detached": True,
                "source.shared_runtime_worktree_dirty": False,
            }
        )
    for field, expected_value in required.items():
        observed = _nested_value(receipt, field)
        if observed != expected_value:
            raise OrchestratorError(
                f"{route_id} gate receipt {field} mismatch: "
                f"{observed!r} != {expected_value!r}"
            )
    gpu_indices = _nested_value(receipt, "execution.gpu_indices")
    if not isinstance(gpu_indices, list) or len(gpu_indices) != 1:
        raise OrchestratorError(
            f"{route_id} gate receipt must contain exactly one GPU index"
        )
    if not str(_nested_value(receipt, "execution.pod_host")):
        raise OrchestratorError(f"{route_id} gate receipt pod_host is empty")
    if not str(receipt.get("run_id") or ""):
        raise OrchestratorError(f"{route_id} gate receipt run_id is empty")
    for field in (
        "training.actor_parameter_delta_l2",
        "training.critic_parameter_delta_l2",
        "training.trajectory_row_count",
    ):
        value = _nested_value(receipt, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise OrchestratorError(f"{route_id} gate receipt {field} must be positive")
    if route_id == "swesmith":
        audit_count = _nested_value(receipt, "runtime.formal_episode_audit_count")
        if (
            isinstance(audit_count, bool)
            or not isinstance(audit_count, int)
            or audit_count < 8
        ):
            raise OrchestratorError(
                "swesmith gate receipt runtime.formal_episode_audit_count "
                "must be at least 8"
            )
        if not str(_nested_value(receipt, "source.environment_outer_source_commit")):
            raise OrchestratorError(
                "swesmith gate receipt outer source commit is empty"
            )
        for field in (
            "timing.startup_seconds",
            "timing.optimizer_update_wall_seconds",
            "timing.total_wall_seconds",
        ):
            value = _nested_value(receipt, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise OrchestratorError(
                    f"swesmith gate receipt {field} must be nonnegative"
                )
    _assert_expected_subset(receipt, expected, field=f"{route_id} gate receipt")
    return receipt


def _verify_git_source(source: Mapping[str, Any], *, route_id: str) -> dict[str, str]:
    name = str(source.get("name", ""))
    if name not in {"outer", "inner"}:
        raise OrchestratorError(
            f"{route_id} source name must be 'outer' or 'inner', got {name!r}"
        )
    root = _directory(source.get("root"), field=f"{route_id} {name} source root")
    commit = str(source.get("commit", ""))
    if len(commit) != 40 or any(character not in _HEX for character in commit):
        raise OrchestratorError(f"{route_id} {name} source commit is invalid")
    try:
        observed = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise OrchestratorError(
            f"cannot verify {route_id} {name} source: {exc.output.strip()}"
        ) from exc
    if observed != commit:
        raise OrchestratorError(
            f"{route_id} {name} source commit mismatch: {observed} != {commit}"
        )
    if status:
        raise OrchestratorError(f"{route_id} {name} source tree is dirty: {status}")
    return {
        "name": name,
        "root": str(root.resolve()),
        "commit": commit,
    }


def _bind_source_evidence(
    source: Mapping[str, Any],
    verified: Mapping[str, str],
    *,
    route_id: str,
    gate_receipt: Mapping[str, Any],
    gate_receipt_path: Path,
    gate_receipt_sha256: str,
) -> dict[str, str]:
    receipt_field_value = source.get("receipt_field")
    source_lock_value = source.get("source_lock")
    if (receipt_field_value is None) == (source_lock_value is None):
        raise OrchestratorError(
            f"{route_id} {verified['name']} source must select exactly one "
            "receipt_field or immutable source_lock"
        )

    if receipt_field_value is not None:
        commit_field = str(receipt_field_value)
        allowed_fields = _RECEIPT_SOURCE_COMMIT_FIELDS[verified["name"]]
        if commit_field not in allowed_fields:
            raise OrchestratorError(
                f"{route_id} {verified['name']} source receipt_field must name "
                f"one of its canonical gate-receipt fields {sorted(allowed_fields)!r}"
            )
        observed_commit = _nested_value(gate_receipt, commit_field)
        evidence = {
            "evidence_kind": "gate_receipt",
            "evidence_path": str(gate_receipt_path.resolve()),
            "evidence_sha256": gate_receipt_sha256,
            "commit_field": commit_field,
        }
    else:
        source_lock = _mapping(
            source_lock_value,
            field=f"{route_id} {verified['name']} immutable source lock",
        )
        source_lock_path = _regular_file(
            source_lock.get("path"),
            field=f"{route_id} {verified['name']} immutable source lock",
        )
        source_lock_sha256 = _digest(
            source_lock.get("sha256"),
            field=f"{route_id} {verified['name']} source lock sha256",
        )
        observed_sha256 = _sha256(source_lock_path)
        if observed_sha256 != source_lock_sha256:
            raise OrchestratorError(
                f"{route_id} {verified['name']} source lock sha256 mismatch: "
                f"{observed_sha256} != {source_lock_sha256}"
            )
        commit_field = str(source_lock.get("commit_field", ""))
        if (
            not commit_field
            or any(not component for component in commit_field.split("."))
            or any(character in commit_field for character in ("\0", "\n", "\r"))
        ):
            raise OrchestratorError(
                f"{route_id} {verified['name']} source lock commit_field is invalid"
            )
        source_lock_payload = _load_json_file(
            source_lock_path,
            field=f"{route_id} {verified['name']} immutable source lock",
        )
        observed_commit = _nested_value(source_lock_payload, commit_field)
        evidence = {
            "evidence_kind": "source_lock",
            "evidence_path": str(source_lock_path.resolve()),
            "evidence_sha256": source_lock_sha256,
            "commit_field": commit_field,
        }

    if observed_commit != verified["commit"]:
        raise OrchestratorError(
            f"{route_id} {verified['name']} {evidence['evidence_kind'].replace('_', ' ')} "
            f"source commit mismatch: {observed_commit!r} != {verified['commit']!r}"
        )
    return {**verified, **evidence}


def _parse_endpoint(value: Any, *, field: str) -> tuple[str, str, int]:
    rendered = str(value or "").rstrip("/")
    parsed = urlparse(rendered)
    try:
        port = parsed.port
    except ValueError as exc:
        raise OrchestratorError(f"{field} has an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in _LOOPBACK_HOSTS
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise OrchestratorError(f"{field} must be a same-Pod loopback HTTP endpoint")
    return rendered, str(parsed.hostname), int(port)


def load_endpoint_registry(
    path: Path,
    *,
    expected_sha256: str,
    route_registry: RouteRegistry,
) -> tuple[tuple[EndpointLaunchSpec, ...], dict[str, Any]]:
    """Validate all endpoint identities before any process or holder mutation."""

    path = _regular_file(path, field="endpoint registry")
    expected_digest = _digest(expected_sha256, field="endpoint registry sha256")
    observed_digest = _sha256(path)
    if observed_digest != expected_digest:
        raise OrchestratorError(
            f"endpoint registry sha256 mismatch: {observed_digest} != {expected_digest}"
        )
    payload = _load_json_file(path, field="endpoint registry")
    if (
        payload.get("schema") != _ENDPOINT_REGISTRY_SCHEMA
        or payload.get("status") != "pass"
    ):
        raise OrchestratorError("endpoint registry is not a completed v1 registry")
    route_order = tuple(
        str(value)
        for value in _sequence(
            payload.get("route_order"), field="endpoint registry route_order"
        )
    )
    if route_order != EXPECTED_ROUTE_IDS or route_order != route_registry.route_ids:
        raise OrchestratorError(
            f"endpoint registry route order mismatch: {route_order!r}"
        )
    raw_routes = _sequence(payload.get("routes"), field="endpoint registry routes")
    if len(raw_routes) != len(EXPECTED_ROUTE_IDS):
        raise OrchestratorError("endpoint registry must contain exactly four routes")
    specs: list[EndpointLaunchSpec] = []
    receipt_report: dict[str, Any] = {}
    source_report: dict[str, Any] = {}
    asset_report: dict[str, Any] = {}
    ports: set[tuple[str, int]] = set()
    for position, (expected_route_id, raw_route) in enumerate(
        zip(EXPECTED_ROUTE_IDS, raw_routes)
    ):
        route = _mapping(raw_route, field=f"endpoint registry route {position}")
        route_id = str(route.get("route_id", ""))
        if route_id != expected_route_id:
            raise OrchestratorError(
                f"endpoint registry route order mismatch at {position}: "
                f"{route_id!r} != {expected_route_id!r}"
            )
        registry_route = route_registry.resolve(route_id)
        attestation = _digest(
            route.get("route_attestation_sha256"),
            field=f"{route_id} route attestation sha256",
        )
        if attestation != registry_route.route_attestation_sha256:
            raise OrchestratorError(f"{route_id} route attestation mismatch")
        endpoint_value = str(route.get("endpoint") or "").rstrip("/")
        if endpoint_value != str(registry_route.client_config["env_addr"]):
            raise OrchestratorError(
                f"{route_id} route registry endpoint mismatch: "
                f"{endpoint_value!r} != "
                f"{registry_route.client_config['env_addr']!r}"
            )
        endpoint, host, port = _parse_endpoint(
            endpoint_value, field=f"{route_id} endpoint"
        )
        if (host, port) in ports:
            raise OrchestratorError(f"duplicate endpoint listener {host}:{port}")
        ports.add((host, port))

        gate_receipt = _mapping(
            route.get("gate_receipt"), field=f"{route_id} gate receipt binding"
        )
        gate_launcher = _mapping(
            route.get("gate_launcher"), field=f"{route_id} gate launcher binding"
        )
        runtime_manifest = _mapping(
            route.get("runtime_manifest"),
            field=f"{route_id} runtime manifest binding",
        )
        gate_receipt_path = _regular_file(
            gate_receipt.get("path"), field=f"{route_id} gate receipt"
        )
        gate_receipt_sha256 = _digest(
            gate_receipt.get("sha256"), field=f"{route_id} gate receipt sha256"
        )
        gate_launcher_path = _regular_file(
            gate_launcher.get("path"),
            field=f"{route_id} gate launcher",
            executable=True,
        )
        gate_launcher_sha256 = _digest(
            gate_launcher.get("sha256"), field=f"{route_id} gate launcher sha256"
        )
        if _sha256(gate_launcher_path) != gate_launcher_sha256:
            raise OrchestratorError(f"{route_id} gate launcher sha256 mismatch")
        runtime_manifest_path = _regular_file(
            runtime_manifest.get("path"), field=f"{route_id} runtime manifest"
        )
        runtime_manifest_sha256 = _digest(
            runtime_manifest.get("sha256"),
            field=f"{route_id} runtime manifest sha256",
        )
        if _sha256(runtime_manifest_path) != runtime_manifest_sha256:
            raise OrchestratorError(f"{route_id} runtime manifest sha256 mismatch")
        receipt_expected = _mapping(
            gate_receipt.get("expected", {}),
            field=f"{route_id} gate receipt expected values",
        )
        receipt = _validate_gate_receipt(
            route_id=route_id,
            receipt_path=gate_receipt_path,
            expected_sha256=gate_receipt_sha256,
            gate_launcher_sha256=gate_launcher_sha256,
            runtime_manifest_sha256=runtime_manifest_sha256,
            expected=receipt_expected,
        )

        raw_sources = _sequence(
            route.get("sources"), field=f"{route_id} source identities"
        )
        source_contracts = [
            _mapping(source, field=f"{route_id} source identity")
            for source in raw_sources
        ]
        verified_sources = [
            _bind_source_evidence(
                source,
                _verify_git_source(source, route_id=route_id),
                route_id=route_id,
                gate_receipt=receipt,
                gate_receipt_path=gate_receipt_path,
                gate_receipt_sha256=gate_receipt_sha256,
            )
            for source in source_contracts
        ]
        source_names = tuple(source["name"] for source in verified_sources)
        if source_names != ("outer", "inner"):
            raise OrchestratorError(
                f"{route_id} must bind exact outer and inner source identities"
            )
        if not any(
            source["evidence_kind"] == "gate_receipt" for source in verified_sources
        ):
            raise OrchestratorError(
                f"{route_id} gate receipt must bind at least one launched source"
            )

        raw_assets = _sequence(route.get("assets"), field=f"{route_id} assets")
        if not raw_assets:
            raise OrchestratorError(f"{route_id} assets must not be empty")
        verified_assets: list[dict[str, str]] = []
        for raw_asset in raw_assets:
            asset = _mapping(raw_asset, field=f"{route_id} asset")
            asset_path = _regular_file(asset.get("path"), field=f"{route_id} asset")
            asset_sha256 = _digest(
                asset.get("sha256"), field=f"{route_id} asset sha256"
            )
            if _sha256(asset_path) != asset_sha256:
                raise OrchestratorError(
                    f"{route_id} asset sha256 mismatch: {asset_path}"
                )
            verified_assets.append(
                {"path": str(asset_path.resolve()), "sha256": asset_sha256}
            )

        launcher = _mapping(
            route.get("endpoint_launcher"),
            field=f"{route_id} endpoint launcher",
        )
        launcher_path = _regular_file(
            launcher.get("path"),
            field=f"{route_id} endpoint launcher",
            executable=True,
        )
        launcher_sha256 = _digest(
            launcher.get("sha256"), field=f"{route_id} endpoint launcher sha256"
        )
        if _sha256(launcher_path) != launcher_sha256:
            raise OrchestratorError(f"{route_id} endpoint launcher sha256 mismatch")
        if launcher.get("process_contract") != "foreground_supervisor_v1":
            raise OrchestratorError(
                f"{route_id} endpoint launcher must own a foreground supervisor"
            )
        argv = tuple(
            str(value)
            for value in _sequence(
                launcher.get("argv", ()), field=f"{route_id} endpoint argv"
            )
        )
        if any("\x00" in value or "\n" in value or "\r" in value for value in argv):
            raise OrchestratorError(f"{route_id} endpoint argv contains unsafe text")
        raw_environment = _mapping(
            launcher.get("environment", {}),
            field=f"{route_id} endpoint environment",
        )
        environment = {str(key): str(value) for key, value in raw_environment.items()}
        if any(
            not key or "=" in key or "\x00" in key or "\x00" in value
            for key, value in environment.items()
        ):
            raise OrchestratorError(f"{route_id} endpoint environment is unsafe")
        working_directory = _directory(
            launcher.get("working_directory"),
            field=f"{route_id} endpoint working directory",
        )

        readiness = _mapping(
            route.get("readiness"), field=f"{route_id} readiness contract"
        )
        readiness_url, _, _ = _parse_endpoint(
            readiness.get("url"), field=f"{route_id} readiness URL"
        )
        if not readiness_url.startswith(endpoint + "/"):
            raise OrchestratorError(
                f"{route_id} readiness URL is not below its route endpoint"
            )
        readiness_expected = _mapping(
            readiness.get("expected"), field=f"{route_id} readiness expected"
        )
        if not readiness_expected:
            raise OrchestratorError(f"{route_id} readiness expected must not be empty")
        readiness_sha256_raw = readiness.get("response_sha256")
        readiness_sha256 = (
            _digest(
                readiness_sha256_raw,
                field=f"{route_id} readiness response sha256",
            )
            if readiness_sha256_raw is not None
            else None
        )
        specs.append(
            EndpointLaunchSpec(
                route_id=route_id,
                route_attestation_sha256=attestation,
                endpoint=endpoint,
                gate_receipt_path=gate_receipt_path.resolve(),
                gate_receipt_sha256=gate_receipt_sha256,
                launcher_path=launcher_path.resolve(),
                launcher_sha256=launcher_sha256,
                argv=argv,
                environment=environment,
                working_directory=working_directory.resolve(),
                readiness_url=readiness_url,
                readiness_expected=dict(readiness_expected),
                readiness_sha256=readiness_sha256,
                ready_timeout_seconds=_positive_float(
                    readiness.get("timeout_seconds"),
                    field=f"{route_id} readiness timeout_seconds",
                ),
                poll_seconds=_positive_float(
                    readiness.get("poll_seconds"),
                    field=f"{route_id} readiness poll_seconds",
                ),
                request_timeout_seconds=_positive_float(
                    readiness.get("request_timeout_seconds"),
                    field=f"{route_id} readiness request_timeout_seconds",
                ),
                cleanup_timeout_seconds=_positive_float(
                    route.get("cleanup_timeout_seconds"),
                    field=f"{route_id} cleanup timeout_seconds",
                ),
            )
        )
        receipt_report[route_id] = {
            "path": str(gate_receipt_path.resolve()),
            "sha256": gate_receipt_sha256,
            "run_id": receipt.get("run_id"),
        }
        source_report[route_id] = verified_sources
        asset_report[route_id] = verified_assets
    return tuple(specs), {
        "schema": _ENDPOINT_REGISTRY_SCHEMA,
        "path": str(path.resolve()),
        "sha256": observed_digest,
        "route_order": list(route_order),
        "gate_receipts": receipt_report,
        "sources": source_report,
        "assets": asset_report,
    }


def _bindable_host(host: str) -> str:
    return "::1" if host == "::1" else "127.0.0.1"


def assert_ports_available(specs: Sequence[EndpointLaunchSpec]) -> None:
    """Reject an occupied endpoint port before any endpoint is spawned."""

    reservations: list[socket.socket] = []
    try:
        for spec in specs:
            _, host, port = _parse_endpoint(
                spec.endpoint, field=f"{spec.route_id} endpoint"
            )
            family = socket.AF_INET6 if host == "::1" else socket.AF_INET
            probe = socket.socket(family, socket.SOCK_STREAM)
            try:
                probe.bind((_bindable_host(host), port))
            except OSError as exc:
                probe.close()
                raise OrchestratorError(
                    f"endpoint port collision for {spec.route_id} on {host}:{port}: {exc}"
                ) from exc
            reservations.append(probe)
    finally:
        for reservation in reservations:
            reservation.close()


def _replace_tokens(value: str, replacements: Mapping[str, str]) -> str:
    rendered = value
    for token, replacement in replacements.items():
        rendered = rendered.replace(token, replacement)
    if "@" in rendered:
        for token in (
            "@RUN_ID@",
            "@RUN_DIR@",
            "@ENDPOINT_RUN_DIR@",
            "@ENDPOINT_URL@",
            "@ENDPOINT_HOST@",
            "@ENDPOINT_PORT@",
            "@PARENT_PID@",
            "@PARENT_START_TICKS@",
        ):
            if token in rendered:
                raise OrchestratorError(f"unresolved endpoint launcher token {token}")
    return rendered


@contextlib.contextmanager
def _blocked_termination_signals() -> Any:
    watched = {signal.SIGINT, signal.SIGTERM}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _process_identity_state(pid: int, start_ticks: str) -> tuple[str, int] | None:
    try:
        fields = (
            Path(f"/proc/{pid}/stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .split()
        )
        if fields[19] != str(start_ticks):
            return None
        return fields[0], int(fields[2])
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _active_process_group_identities(
    process_group: int,
) -> tuple[tuple[int, str], ...]:
    identities: list[tuple[int, str]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise OrchestratorError("exact process-group teardown requires Linux /proc")
    for candidate in proc_root.iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            fields = (
                (candidate / "stat")
                .read_text(encoding="utf-8")
                .rsplit(")", 1)[1]
                .split()
            )
            if fields[0] in {"Z", "X", "x"} or int(fields[2]) != process_group:
                continue
            identities.append((int(candidate.name), fields[19]))
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
    return tuple(sorted(identities))


class ExactProcessSupervisor:
    """Own foreground process groups through exact leader PID/start-tick leases."""

    def __init__(self) -> None:
        self._owned: dict[tuple[int, str], ProcessLease] = {}

    @property
    def owned_leases(self) -> tuple[ProcessLease, ...]:
        return tuple(self._owned.values())

    def _register(self, lease: ProcessLease) -> None:
        identity = (lease.pid, lease.start_ticks)
        if identity in self._owned:
            raise OrchestratorError(f"duplicate process lease registration: {identity}")
        self._owned[identity] = lease

    def _release(self, lease: ProcessLease) -> None:
        self._owned.pop((lease.pid, lease.start_ticks), None)

    def _capture_ticks(self, process: subprocess.Popen[bytes]) -> str:
        for _ in range(100):
            ticks = process_start_ticks(process.pid)
            if ticks:
                try:
                    process_group = os.getpgid(process.pid)
                except ProcessLookupError:
                    break
                if process_group != process.pid:
                    raise OrchestratorError(
                        f"child pid {process.pid} is not its process-group leader"
                    )
                return ticks
            if process.poll() is not None:
                break
            time.sleep(0.02)
        raise OrchestratorError(
            f"failed to capture Linux start ticks for child pid {process.pid}"
        )

    def start_command(
        self,
        *,
        name: str,
        command: Sequence[str],
        working_directory: Path,
        environment: Mapping[str, str],
        log_handle: BinaryIO,
        identity_path: Path,
        cleanup_timeout_seconds: float,
    ) -> ProcessLease:
        """Start a command only after its exact lease is durably published."""

        if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            raise OrchestratorError(
                "managed process spawn requires SIGCHLD=SIG_DFL so an exited "
                "process-group leader remains an authenticated zombie anchor"
            )
        bootstrap = _regular_file(
            Path(__file__).with_name("process_bootstrap.py"),
            field="process bootstrap",
        )
        read_fd, write_fd = os.pipe()
        process: subprocess.Popen[bytes] | None = None
        lease: ProcessLease | None = None
        acknowledged = False
        try:
            with _blocked_termination_signals():
                bootstrap_command = [
                    sys.executable,
                    str(bootstrap),
                    "--ack-fd",
                    str(read_fd),
                    "--parent-pid",
                    str(os.getpid()),
                    "--parent-start-ticks",
                    str(process_start_ticks(os.getpid()) or ""),
                    "--cleanup-timeout-seconds",
                    str(cleanup_timeout_seconds),
                    "--",
                    *command,
                ]
                process = subprocess.Popen(
                    bootstrap_command,
                    cwd=working_directory,
                    env=dict(environment),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(read_fd,),
                )
                os.close(read_fd)
                read_fd = -1
                ticks = self._capture_ticks(process)
                lease = ProcessLease(
                    name=name,
                    pid=process.pid,
                    start_ticks=ticks,
                    process=process,
                    log_handle=log_handle,
                    cleanup_timeout_seconds=cleanup_timeout_seconds,
                )
                self._register(lease)
                _atomic_json(
                    identity_path,
                    {
                        "name": name,
                        "pid": lease.pid,
                        "start_ticks": lease.start_ticks,
                        "process_group": lease.pid,
                        "bootstrap": str(bootstrap),
                        "command": list(command),
                    },
                )
                if os.write(write_fd, b"1") != 1:
                    raise OrchestratorError(
                        f"failed to release {name} process bootstrap"
                    )
                acknowledged = True
                os.close(write_fd)
                write_fd = -1
            return lease
        except Exception as start_error:
            if read_fd >= 0:
                os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
            cleanup_error: Exception | None = None
            if lease is not None and acknowledged:
                try:
                    self.stop(lease, timeout_seconds=cleanup_timeout_seconds)
                except Exception as error:
                    cleanup_error = error
            elif process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as error:
                    cleanup_error = OrchestratorError(
                        f"unreleased {name} bootstrap did not exit after pipe EOF"
                    )
                    cleanup_error.__cause__ = error
                if lease is not None and process.returncode is not None:
                    self._release(lease)
            if not log_handle.closed:
                log_handle.close()
            if cleanup_error is not None:
                raise OrchestratorError(
                    f"{name} start failed and cleanup failed: "
                    f"{start_error}; {cleanup_error}"
                ) from start_error
            raise

    def start(
        self,
        spec: EndpointLaunchSpec,
        *,
        run_dir: Path,
        parent_pid: int,
        parent_start_ticks: str,
    ) -> ProcessLease:
        if (
            parent_pid != os.getpid()
            or process_start_ticks(parent_pid) != parent_start_ticks
        ):
            raise OrchestratorError(
                f"{spec.route_id} endpoint parent PID/start-ticks mismatch"
            )
        endpoint_dir = run_dir / "endpoints" / spec.route_id
        endpoint_dir.mkdir(parents=True, exist_ok=False)
        parsed = urlparse(spec.endpoint)
        replacements = {
            "@RUN_ID@": run_dir.name,
            "@RUN_DIR@": str(run_dir),
            "@ENDPOINT_RUN_DIR@": str(endpoint_dir),
            "@ENDPOINT_URL@": spec.endpoint,
            "@ENDPOINT_HOST@": str(parsed.hostname),
            "@ENDPOINT_PORT@": str(parsed.port),
            "@PARENT_PID@": str(parent_pid),
            "@PARENT_START_TICKS@": parent_start_ticks,
        }
        command = [
            str(spec.launcher_path),
            *(_replace_tokens(value, replacements) for value in spec.argv),
        ]
        environment = dict(os.environ)
        environment.update(
            {
                key: _replace_tokens(value, replacements)
                for key, value in spec.environment.items()
            }
        )
        environment.update(
            {
                "AMG_MULTITASK_RUN_ID": run_dir.name,
                "AMG_MULTITASK_ROUTE_ID": spec.route_id,
                "AMG_MULTITASK_ENDPOINT_RUN_DIR": str(endpoint_dir),
                "AMG_MULTITASK_ENDPOINT_URL": spec.endpoint,
                "AMG_MULTITASK_PARENT_PID": str(parent_pid),
                "AMG_MULTITASK_PARENT_START_TICKS": parent_start_ticks,
            }
        )
        log_handle = (endpoint_dir / "launcher.log").open("ab", buffering=0)
        return self.start_command(
            name=spec.route_id,
            command=command,
            working_directory=spec.working_directory,
            environment=environment,
            log_handle=log_handle,
            identity_path=endpoint_dir / "process-identity.json",
            cleanup_timeout_seconds=spec.cleanup_timeout_seconds,
        )

    def alive(self, lease: ProcessLease) -> bool:
        state = _process_identity_state(lease.pid, lease.start_ticks)
        if state is None or state[1] != lease.pid:
            return False
        return state[0] not in {"Z", "X", "x"}

    def poll(self, lease: ProcessLease) -> int | None:
        """Poll without reaping an exited leader while it still anchors children."""

        leader = _process_identity_state(lease.pid, lease.start_ticks)
        group_members = _active_process_group_identities(lease.pid)
        descendants = tuple(
            identity for identity in group_members if identity[0] != lease.pid
        )
        if leader is None:
            if descendants:
                raise OrchestratorError(
                    f"{lease.name} lost its process-group anchor with live "
                    f"descendants {descendants}"
                )
            return_code = lease.process.poll()
            if return_code is None:
                raise OrchestratorError(
                    f"{lease.name} exact process identity disappeared"
                )
            return int(return_code)
        if leader[1] != lease.pid:
            raise OrchestratorError(f"{lease.name} process group identity drifted")
        if leader[0] not in {"Z", "X", "x"}:
            return None
        if descendants:
            raise OrchestratorError(
                f"{lease.name} leader exited with live descendants {descendants}"
            )
        return int(lease.process.wait(timeout=0))

    def wait(
        self,
        lease: ProcessLease,
        *,
        timeout_seconds: float | None = None,
        poll_seconds: float = 0.02,
    ) -> int:
        """Wait through identity-aware polling without prematurely reaping a leader."""

        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        while True:
            return_code = self.poll(lease)
            if return_code is not None:
                return return_code
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(lease.name, timeout_seconds)
            time.sleep(poll_seconds)

    def stop(self, lease: ProcessLease, *, timeout_seconds: float) -> None:
        try:
            process_group = lease.pid
            leader = _process_identity_state(lease.pid, lease.start_ticks)
            group_members = _active_process_group_identities(process_group)
            if leader is None or leader[1] != process_group:
                if group_members:
                    raise OrchestratorError(
                        f"{lease.name} leader identity is no longer alive; refusing "
                        f"to signal unauthenticated process group {process_group} "
                        f"with members {group_members}"
                    )
                self._release(lease)
                return
            if leader[0] not in {"Z", "X", "x"}:
                _signal_process_identity(lease.pid, lease.start_ticks, signal.SIGTERM)
            else:
                for pid, ticks in group_members:
                    if pid != lease.pid:
                        _signal_process_identity(pid, ticks, signal.SIGTERM)

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                leader = _process_identity_state(lease.pid, lease.start_ticks)
                group_members = _active_process_group_identities(process_group)
                active_descendants = tuple(
                    identity for identity in group_members if identity[0] != lease.pid
                )
                if not active_descendants and (
                    leader is None or leader[0] in {"Z", "X", "x"}
                ):
                    break
                time.sleep(0.02)

            leader = _process_identity_state(lease.pid, lease.start_ticks)
            active_descendants = tuple(
                identity
                for identity in _active_process_group_identities(process_group)
                if identity[0] != lease.pid
            )
            if active_descendants:
                if leader is None or leader[1] != process_group:
                    raise OrchestratorError(
                        f"{lease.name} lost its authenticated process-group anchor"
                    )
                for pid, ticks in active_descendants:
                    _signal_process_identity(pid, ticks, signal.SIGKILL)
                kill_deadline = time.monotonic() + 5.0
                while time.monotonic() < kill_deadline:
                    active_descendants = tuple(
                        identity
                        for identity in _active_process_group_identities(process_group)
                        if identity[0] != lease.pid
                    )
                    if not active_descendants:
                        break
                    for pid, ticks in active_descendants:
                        _signal_process_identity(pid, ticks, signal.SIGKILL)
                    time.sleep(0.02)
                if active_descendants:
                    raise OrchestratorError(
                        f"{lease.name} process-group descendants did not terminate"
                    )
            leader = _process_identity_state(lease.pid, lease.start_ticks)
            if leader is not None and leader[0] not in {"Z", "X", "x"}:
                _signal_process_identity(lease.pid, lease.start_ticks, signal.SIGKILL)
            try:
                lease.process.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise OrchestratorError(
                    f"{lease.name} process-group leader did not terminate"
                ) from exc
            remaining = _active_process_group_identities(process_group)
            if remaining:
                raise OrchestratorError(
                    f"{lease.name} process group did not terminate: {remaining}"
                )
            self._release(lease)
        finally:
            if lease.log_handle is not None and not lease.log_handle.closed:
                lease.log_handle.close()

    def stop_all(self, *, exclude: Sequence[ProcessLease] = ()) -> None:
        errors: list[str] = []
        excluded = {(lease.pid, lease.start_ticks) for lease in exclude}
        for lease in reversed(self.owned_leases):
            if (lease.pid, lease.start_ticks) in excluded:
                continue
            try:
                self.stop(lease, timeout_seconds=lease.cleanup_timeout_seconds)
            except Exception as exc:
                errors.append(f"{lease.name}: {exc}")
        if errors:
            raise OrchestratorError(
                "supervised process cleanup failed: " + "; ".join(errors)
            )


def start_endpoint_processes(
    specs: Sequence[EndpointLaunchSpec],
    *,
    run_dir: Path,
    parent_pid: int,
    parent_start_ticks: str,
    supervisor: Any,
) -> tuple[ProcessLease, ...]:
    """Start all launchers before readiness waits; rollback a partial start."""

    leases: list[ProcessLease] = []
    try:
        for spec in specs:
            with _blocked_termination_signals():
                lease = supervisor.start(
                    spec,
                    run_dir=run_dir,
                    parent_pid=parent_pid,
                    parent_start_ticks=parent_start_ticks,
                )
                leases.append(lease)
    except Exception as start_error:
        cleanup_errors: list[str] = []
        for spec, lease in reversed(list(zip(specs, leases))):
            try:
                supervisor.stop(lease, timeout_seconds=spec.cleanup_timeout_seconds)
            except Exception as cleanup_error:
                cleanup_errors.append(f"{lease.name}: {cleanup_error}")
        if cleanup_errors:
            raise OrchestratorError(
                "partial endpoint startup rollback failed after "
                f"{start_error}: {'; '.join(cleanup_errors)}"
            ) from start_error
        raise
    return tuple(leases)


def _readiness_probe(spec: EndpointLaunchSpec) -> tuple[bytes, Mapping[str, Any]]:
    request = urllib.request.Request(spec.readiness_url, method="GET")
    with urllib.request.urlopen(
        request, timeout=spec.request_timeout_seconds
    ) as response:
        body = response.read()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OrchestratorError(
            f"{spec.route_id} readiness response is not JSON"
        ) from exc
    payload = _mapping(payload, field=f"{spec.route_id} readiness response")
    if spec.readiness_sha256 is not None:
        observed = hashlib.sha256(body).hexdigest()
        if observed != spec.readiness_sha256:
            raise OrchestratorError(
                f"{spec.route_id} readiness response sha256 mismatch"
            )
    _assert_expected_subset(
        payload,
        spec.readiness_expected,
        field=f"{spec.route_id} readiness response",
    )
    return body, payload


def wait_for_endpoints(
    specs: Sequence[EndpointLaunchSpec],
    leases: Sequence[ProcessLease],
    *,
    run_dir: Path,
    supervisor: ExactProcessSupervisor,
) -> dict[str, Any]:
    if len(specs) != len(leases):
        raise OrchestratorError("endpoint spec/process lease count mismatch")
    pending = {
        spec.route_id: (spec, lease, time.monotonic())
        for spec, lease in zip(specs, leases)
    }
    reports: dict[str, Any] = {}
    while pending:
        progressed = False
        for route_id, (spec, lease, started) in tuple(pending.items()):
            if not supervisor.alive(lease):
                raise OrchestratorError(
                    f"{route_id} endpoint launcher exited before readiness"
                )
            try:
                body, payload = _readiness_probe(spec)
            except (OSError, urllib.error.URLError, TimeoutError):
                if time.monotonic() - started >= spec.ready_timeout_seconds:
                    raise OrchestratorError(f"{route_id} endpoint readiness timed out")
                continue
            endpoint_dir = run_dir / "endpoints" / route_id
            metadata_path = endpoint_dir / "metadata.json"
            _atomic_bytes(metadata_path, body)
            reports[route_id] = {
                "endpoint": spec.endpoint,
                "pid": lease.pid,
                "start_ticks": lease.start_ticks,
                "metadata_path": str(metadata_path),
                "metadata_sha256": hashlib.sha256(body).hexdigest(),
                "metadata": dict(payload),
                "startup_seconds": time.monotonic() - started,
            }
            del pending[route_id]
            progressed = True
        if pending and not progressed:
            time.sleep(min(spec.poll_seconds for spec, _, _ in pending.values()))
    return reports


def load_holder_lease(path: Path, *, expected_sha256: str) -> HolderLease:
    path = _regular_file(path, field="holder lease")
    expected_digest = _digest(expected_sha256, field="holder lease sha256")
    observed_digest = _sha256(path)
    if observed_digest != expected_digest:
        raise OrchestratorError("holder lease sha256 mismatch")
    payload = _load_json_file(path, field="holder lease")
    if payload.get("schema") != _HOLDER_LEASE_SCHEMA or payload.get("status") != "pass":
        raise OrchestratorError("holder lease is not a completed v1 lease")
    raw_markers = _sequence(payload.get("markers"), field="holder lease markers")
    markers: list[MarkerLease] = []
    for raw in raw_markers:
        marker = _mapping(raw, field="holder marker")
        name = str(marker.get("name", ""))
        original_value = marker.get("original_value")
        if original_value is not None and not isinstance(original_value, str):
            raise OrchestratorError(f"holder marker {name} original_value is invalid")
        pid = marker.get("original_pid", 0)
        ticks = str(marker.get("original_start_ticks", ""))
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
            raise OrchestratorError(f"holder marker {name} original_pid is invalid")
        if original_value is not None and not process_identity_alive(pid, ticks):
            raise OrchestratorError(
                f"holder marker {name} exact owner PID/start-ticks is not alive"
            )
        marker_path = _absolute_path(marker.get("path"), field=f"holder marker {name}")
        if marker_path != _HOLDER_MARKER_PATHS.get(name):
            raise OrchestratorError(
                f"holder marker {name} path mismatch: {marker_path}"
            )
        markers.append(
            MarkerLease(
                name=name,
                path=marker_path,
                original_value=original_value,
                original_pid=pid,
                original_start_ticks=ticks,
            )
        )
    if tuple(marker.name for marker in markers) != ("cpu", "gpu"):
        raise OrchestratorError("holder lease must contain cpu then gpu markers")

    def checks(name: str) -> tuple[Mapping[str, str], ...]:
        raw_checks = _sequence(payload.get(name), field=f"holder lease {name}")
        normalized: list[Mapping[str, str]] = []
        if not raw_checks:
            raise OrchestratorError(f"holder lease {name} must not be empty")
        for raw in raw_checks:
            check = _mapping(raw, field=f"holder lease {name} entry")
            check_path = _absolute_path(
                check.get("path"), field=f"holder lease {name} path"
            )
            contains = str(check.get("contains", ""))
            if not contains or any(
                character in contains for character in ("\0", "\n", "\r")
            ):
                raise OrchestratorError(
                    f"holder lease {name} contains value is invalid"
                )
            normalized.append({"path": str(check_path), "contains": contains})
        return tuple(normalized)

    return HolderLease(
        source_path=path.resolve(),
        sha256=observed_digest,
        markers=tuple(markers),
        yield_checks=checks("yield_checks"),
        restore_checks=checks("restore_checks"),
    )


def _wait_file_checks(
    checks: Sequence[Mapping[str, str]], *, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    missing: list[str] = []
    while True:
        missing = []
        for check in checks:
            path = Path(check["path"])
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                content = ""
            if check["contains"] not in content:
                missing.append(str(path))
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise OrchestratorError(
                "holder state transition timed out for: " + ", ".join(missing)
            )
        time.sleep(0.25)


def build_generic_launch_command(
    plan: LaunchPlan,
    *,
    resolve_only: bool,
    orchestrator_preflight: Path | None = None,
) -> list[str]:
    command = [
        str(plan.generic_launcher),
        "--mode",
        "formal",
        "--verl-root",
        str(plan.verl_root),
        "--schedule",
        str(plan.schedule),
        "--run-dir",
        str(plan.run_dir if not resolve_only else plan.run_dir / "resolve-only"),
        "--experiment-name",
        plan.experiment_name,
        "--route-registry",
        str(plan.route_registry_path),
        "--route-registry-sha256",
        plan.route_registry_sha256,
        "--multitask-source-lock",
        str(plan.multitask_source_lock),
        "--multitask-schedule-certificate",
        str(plan.multitask_schedule_certificate),
        "--trainer-gpus",
        str(plan.config.trainer_gpus),
        "--standalone-rollout-gpus",
        str(plan.config.standalone_rollout_gpus),
    ]
    if plan.config.actor_use_fused_kernels:
        command.append("--actor-use-fused-kernels")
    if plan.config.critic_use_fused_kernels:
        command.append("--critic-use-fused-kernels")
    if resolve_only:
        command.extend(("--resolve-only", "--skip-runtime-preflight"))
    else:
        if orchestrator_preflight is None:
            raise OrchestratorError(
                "full launch requires an endpoint orchestration preflight receipt"
            )
        command.extend(
            ("--multitask-orchestrator-preflight", str(orchestrator_preflight))
        )
    return command


def _generic_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


class Backend(Protocol):
    def resolve(self, plan: LaunchPlan) -> None: ...

    def acquire_holders(self, plan: LaunchPlan) -> Any: ...

    def start_endpoints(self, plan: LaunchPlan) -> Any: ...

    def start_trainer(self, plan: LaunchPlan) -> Any: ...


def execute_launch_plan(plan: LaunchPlan, *, backend: Backend) -> int:
    """Execute a preflighted plan; resolve-only exits before all runtime owners."""

    backend.resolve(plan)
    if plan.resolve_only:
        return 0
    holder = backend.acquire_holders(plan)
    endpoints: Any = None
    trainer: Any = None
    try:
        endpoints = backend.start_endpoints(plan)
        trainer = backend.start_trainer(plan)
        return int(backend.wait_trainer(plan, trainer, endpoints, holder))
    finally:
        if trainer is not None:
            backend.stop_trainer(plan, trainer)
        if endpoints is not None:
            backend.stop_endpoints(plan, endpoints)
        backend.restore_holders(plan, holder)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _validate_fresh_run_dir(path: Path, *, outer_root: Path) -> Path:
    if path.is_symlink():
        raise OrchestratorError(f"refusing to reuse run directory: {path}")
    path = path.resolve()
    if path == outer_root or outer_root in path.parents:
        raise OrchestratorError("run directory must be outside the exact source tree")
    if path.exists() or path.is_symlink():
        raise OrchestratorError(f"refusing to reuse run directory: {path}")
    return path


def _source_clean(path: Path, *, label: str) -> None:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain=v1"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise OrchestratorError(
            f"cannot inspect {label}: {exc.output.strip()}"
        ) from exc
    if status:
        raise OrchestratorError(f"{label} must be clean: {status}")


def build_launch_plan(args: argparse.Namespace) -> LaunchPlan:
    config_path = _regular_file(args.config, field="orchestrator config")
    config = load_orchestrator_config(config_path)
    outer_root = _directory(args.outer_root, field="outer source root")
    verl_root = _directory(args.verl_root, field="veRL source root")
    route_registry_path = _regular_file(args.route_registry, field="route registry")
    route_registry = load_route_registry(
        route_registry_path,
        expected_sha256=args.route_registry_sha256,
        expected_route_ids=config.route_order,
    )
    schedule = _regular_file(args.schedule, field="multitask schedule")
    source_lock = _regular_file(
        args.multitask_source_lock, field="multitask source lock"
    )
    certificate = _regular_file(
        args.multitask_schedule_certificate,
        field="multitask schedule certificate",
    )
    run_dir = _validate_fresh_run_dir(args.run_dir, outer_root=outer_root)
    schedule_report = inspect_schedule(
        schedule,
        expected_count=config.total_episodes,
        expected_role="train_pool",
        expected_route_ids=config.route_order,
        expected_route_registry_sha256=route_registry.sha256,
    )
    identity_inputs = LaunchInputs(
        mode="formal",
        verl_root=verl_root,
        outer_root=outer_root,
        schedule=schedule,
        env_addr=None,
        run_dir=run_dir,
        experiment_name=args.experiment_name,
        endpoint_source_lock=None,
        endpoint_contract_tool=None,
        publication_receipt=None,
        formal_schedule_certificate=None,
        trainer_gpus=config.trainer_gpus,
        standalone_rollout_gpus=config.standalone_rollout_gpus,
        actor_use_fused_kernels=config.actor_use_fused_kernels,
        critic_use_fused_kernels=config.critic_use_fused_kernels,
        route_registry=route_registry_path,
        route_registry_sha256=route_registry.sha256,
        multitask_source_lock=source_lock,
        multitask_schedule_certificate=certificate,
    )
    launch_identity = _load_multitask_identity(
        identity_inputs, schedule_report=schedule_report
    )
    budget = _mapping(
        launch_identity.get("budget_contract"), field="multitask budget contract"
    )
    required_budget = {
        "optimizer_updates": config.optimizer_updates,
        "samples_per_update": config.samples_per_update,
        "episodes": config.total_episodes,
        "trigger_parameter_sync_step": config.trigger_parameter_sync_step,
    }
    for field, expected in required_budget.items():
        if budget.get(field) != expected:
            raise OrchestratorError(
                f"multitask launch budget {field} mismatch: "
                f"{budget.get(field)!r} != {expected!r}"
            )
    endpoint_registry_path = _regular_file(
        args.endpoint_registry, field="endpoint registry"
    )
    endpoints, endpoint_report = load_endpoint_registry(
        endpoint_registry_path,
        expected_sha256=args.endpoint_registry_sha256,
        route_registry=route_registry,
    )
    assert_ports_available(endpoints)
    generic_launcher = _regular_file(
        outer_root / "async_plugins/scripts/launch_amg_fully_async.sh",
        field="generic fully-async launcher",
        executable=True,
    )
    holder_lease = None
    if not args.resolve_only:
        if args.holder_lease is None or args.holder_lease_sha256 is None:
            raise OrchestratorError(
                "full launch requires --holder-lease and --holder-lease-sha256"
            )
        holder_lease = load_holder_lease(
            _regular_file(args.holder_lease, field="holder lease"),
            expected_sha256=args.holder_lease_sha256,
        )
    elif (args.holder_lease is None) != (args.holder_lease_sha256 is None):
        raise OrchestratorError(
            "holder lease path and sha256 must be provided together"
        )
    if not args.resolve_only:
        _source_clean(outer_root, label="outer launch source")
        _source_clean(verl_root, label="veRL source")
        _source_clean(outer_root / "AgentGym", label="AgentGym source")
    return LaunchPlan(
        config=config,
        outer_root=outer_root,
        verl_root=verl_root,
        schedule=schedule,
        route_registry_path=route_registry_path,
        route_registry_sha256=route_registry.sha256 or "",
        multitask_source_lock=source_lock,
        multitask_schedule_certificate=certificate,
        endpoint_registry_path=endpoint_registry_path,
        endpoint_registry_sha256=args.endpoint_registry_sha256,
        run_dir=run_dir,
        experiment_name=args.experiment_name,
        endpoints=endpoints,
        endpoint_report=endpoint_report,
        schedule_report=schedule_report,
        launch_identity=launch_identity,
        generic_launcher=generic_launcher,
        resolve_only=args.resolve_only,
        holder_lease=holder_lease,
    )


@dataclass
class _HolderHandle:
    state_path: Path
    watcher_receipt: Path
    watcher: ProcessLease


class LocalBackend:
    def __init__(self) -> None:
        self.supervisor = ExactProcessSupervisor()
        self.parent_pid = os.getpid()
        self.parent_start_ticks = process_start_ticks(self.parent_pid) or ""
        self.endpoint_leases: tuple[ProcessLease, ...] = ()
        self.endpoint_runtime: dict[str, Any] = {}
        self.trainer: ProcessLease | None = None
        self.holder_handle: _HolderHandle | None = None

    def resolve(self, plan: LaunchPlan) -> None:
        plan.run_dir.mkdir(parents=True, exist_ok=False)
        command = build_generic_launch_command(plan, resolve_only=True)
        log = plan.run_dir / "resolve-only.log"
        with log.open("wb") as output:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=_generic_environment(),
                cwd=plan.outer_root,
                check=False,
            )
        if completed.returncode != 0:
            raise OrchestratorError(
                f"generic fully-async resolve-only preflight failed with "
                f"exit {completed.returncode}; see {log}"
            )
        _atomic_json(
            plan.run_dir / "resolve-only-receipt.json",
            {
                "schema": _ORCHESTRATOR_RECEIPT_SCHEMA,
                "status": "resolved",
                "endpoints_spawned": 0,
                "trainer_spawned": False,
                "command": command,
                "config_sha256": plan.config.sha256,
                "endpoint_registry_sha256": plan.endpoint_registry_sha256,
                "route_registry_sha256": plan.route_registry_sha256,
                "schedule_sha256": plan.schedule_report["sha256"],
            },
        )

    def acquire_holders(self, plan: LaunchPlan) -> _HolderHandle:
        if not self.parent_start_ticks:
            raise OrchestratorError("full multitask orchestration requires Linux /proc")
        lease = plan.holder_lease
        if lease is None:
            raise OrchestratorError("full launch has no holder lease")
        state_dir = plan.run_dir / "holder-transaction"
        state_dir.mkdir(parents=True, exist_ok=False)
        state_path = state_dir / "state.json"
        ready_path = state_dir / "watcher-ready.json"
        receipt_path = state_dir / "watcher-exit.json"
        markers = [
            {
                "name": marker.name,
                "path": str(marker.path),
                "original_value": marker.original_value,
                "original_identity": {
                    "pid": marker.original_pid,
                    "start_ticks": marker.original_start_ticks,
                },
            }
            for marker in lease.markers
        ]
        watcher: ProcessLease | None = None
        watcher_log: BinaryIO | None = None
        prepared = False
        try:
            prepare_marker_transaction(
                state_path=state_path,
                lock_path=plan.config.holder_lock_path,
                run_id=plan.experiment_name,
                parent_pid=self.parent_pid,
                parent_start_ticks=self.parent_start_ticks,
                markers=markers,
            )
            prepared = True
            lifecycle = Path(__file__).with_name("orchestrator_lifecycle.py")
            watcher_log = (state_dir / "watcher.log").open("ab", buffering=0)
            watcher_command = [
                sys.executable,
                str(lifecycle),
                "marker-watch",
                "--state",
                str(state_path),
                "--lock",
                str(plan.config.holder_lock_path),
                "--parent-pid",
                str(self.parent_pid),
                "--parent-start-ticks",
                self.parent_start_ticks,
                "--ready",
                str(ready_path),
                "--receipt",
                str(receipt_path),
                "--poll-seconds",
                "0.1",
                "--restore-timeout-seconds",
                "300",
            ]
            watcher = self.supervisor.start_command(
                name="holder-watcher",
                command=watcher_command,
                working_directory=plan.outer_root,
                environment=dict(os.environ),
                log_handle=watcher_log,
                identity_path=state_dir / "watcher-process-identity.json",
                cleanup_timeout_seconds=5,
            )
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if ready_path.is_file():
                    break
                if not self.supervisor.alive(watcher):
                    raise OrchestratorError("holder transaction watcher exited early")
                time.sleep(0.1)
            if not ready_path.is_file():
                raise OrchestratorError(
                    "holder transaction watcher readiness timed out"
                )
            ready = _load_json_file(ready_path, field="holder watcher readiness")
            if (
                ready.get("status") != "ready"
                or ready.get("signal_handlers_installed") is not True
            ):
                raise OrchestratorError("holder watcher did not arm signal handling")
            acquire_marker_transaction(state_path, plan.config.holder_lock_path)
            _wait_file_checks(lease.yield_checks, timeout_seconds=30)
            handle = _HolderHandle(
                state_path=state_path,
                watcher_receipt=receipt_path,
                watcher=watcher,
            )
            self.holder_handle = handle
            return handle
        except Exception as acquisition_error:
            cleanup_errors: list[str] = []
            restored = False
            if prepared:
                try:
                    restore_marker_transaction(state_path, plan.config.holder_lock_path)
                    restored = True
                except Exception as restore_error:
                    cleanup_errors.append(f"marker restore: {restore_error}")
            if watcher is not None and restored:
                try:
                    self.supervisor.wait(watcher, timeout_seconds=15)
                except (OrchestratorError, subprocess.TimeoutExpired):
                    pass
                try:
                    self.supervisor.stop(watcher, timeout_seconds=5)
                except Exception as stop_error:
                    cleanup_errors.append(f"watcher stop: {stop_error}")
            if watcher_log is not None and not watcher_log.closed:
                watcher_log.close()
            if cleanup_errors:
                raise OrchestratorError(
                    "holder acquisition failed and rollback was incomplete: "
                    f"{acquisition_error}; {'; '.join(cleanup_errors)}"
                ) from acquisition_error
            self.holder_handle = None
            raise

    def start_endpoints(self, plan: LaunchPlan) -> tuple[ProcessLease, ...]:
        self.endpoint_leases = start_endpoint_processes(
            plan.endpoints,
            run_dir=plan.run_dir,
            parent_pid=self.parent_pid,
            parent_start_ticks=self.parent_start_ticks,
            supervisor=self.supervisor,
        )
        try:
            self.endpoint_runtime = wait_for_endpoints(
                plan.endpoints,
                self.endpoint_leases,
                run_dir=plan.run_dir,
                supervisor=self.supervisor,
            )
        except Exception:
            self.stop_endpoints(plan, self.endpoint_leases)
            self.endpoint_leases = ()
            raise
        return self.endpoint_leases

    def _preflight_receipt(self, plan: LaunchPlan, holder: _HolderHandle) -> Path:
        if not self.supervisor.alive(holder.watcher):
            raise OrchestratorError(
                "holder transaction watcher exited before trainer preflight"
            )
        route_registry = load_route_registry(
            plan.route_registry_path,
            expected_sha256=plan.route_registry_sha256,
            expected_route_ids=plan.config.route_order,
        )
        revalidated_endpoints, endpoint_identity = load_endpoint_registry(
            plan.endpoint_registry_path,
            expected_sha256=plan.endpoint_registry_sha256,
            route_registry=route_registry,
        )
        if revalidated_endpoints != plan.endpoints:
            raise OrchestratorError(
                "endpoint launch contract changed after initial preflight"
            )
        endpoint_entries = []
        if len(plan.endpoints) != len(self.endpoint_leases):
            raise OrchestratorError("endpoint spec/process lease count mismatch")
        for spec, lease in zip(plan.endpoints, self.endpoint_leases):
            runtime = self.endpoint_runtime[spec.route_id]
            endpoint_entries.append(
                {
                    "route_id": spec.route_id,
                    "route_attestation_sha256": spec.route_attestation_sha256,
                    "endpoint": spec.endpoint,
                    "gate_receipt_path": str(spec.gate_receipt_path),
                    "gate_receipt_sha256": spec.gate_receipt_sha256,
                    "launcher_path": str(spec.launcher_path),
                    "launcher_sha256": spec.launcher_sha256,
                    "pid": lease.pid,
                    "start_ticks": lease.start_ticks,
                    "metadata_path": runtime["metadata_path"],
                    "metadata_sha256": runtime["metadata_sha256"],
                    "startup_seconds": runtime["startup_seconds"],
                }
            )
        source_lock_sha256 = _sha256(plan.multitask_source_lock)
        certificate_sha256 = _sha256(plan.multitask_schedule_certificate)
        path = plan.run_dir / "orchestrator-preflight.json"
        _atomic_json(
            path,
            {
                "schema": _PREFLIGHT_SCHEMA,
                "status": "pass",
                "config_path": str(plan.config.source_path),
                "config_sha256": plan.config.sha256,
                "endpoint_registry_path": str(plan.endpoint_registry_path),
                "endpoint_registry_sha256": plan.endpoint_registry_sha256,
                "route_registry_path": str(plan.route_registry_path),
                "route_registry_sha256": plan.route_registry_sha256,
                "route_order": list(plan.config.route_order),
                "schedule_path": str(plan.schedule),
                "schedule_sha256": plan.schedule_report["sha256"],
                "schedule_count": plan.schedule_report["count"],
                "multitask_source_lock_path": str(plan.multitask_source_lock),
                "multitask_source_lock_sha256": source_lock_sha256,
                "multitask_schedule_certificate_path": str(
                    plan.multitask_schedule_certificate
                ),
                "multitask_schedule_certificate_sha256": certificate_sha256,
                "budget": {
                    "optimizer_updates": plan.config.optimizer_updates,
                    "samples_per_update": plan.config.samples_per_update,
                    "episodes": plan.config.total_episodes,
                },
                "holder_transaction": {
                    "status": "acquired",
                    "lease_path": (
                        str(plan.holder_lease.source_path)
                        if plan.holder_lease
                        else None
                    ),
                    "lease_sha256": plan.holder_lease.sha256
                    if plan.holder_lease
                    else None,
                    "state_path": str(holder.state_path),
                    "watcher_pid": holder.watcher.pid,
                    "watcher_start_ticks": holder.watcher.start_ticks,
                },
                "endpoint_identity": endpoint_identity,
                "endpoints": endpoint_entries,
            },
        )
        return path

    def start_trainer(self, plan: LaunchPlan) -> ProcessLease:
        raise OrchestratorError("holder handle is required before trainer start")

    def start_trainer_with_holder(
        self, plan: LaunchPlan, holder: _HolderHandle
    ) -> ProcessLease:
        preflight = self._preflight_receipt(plan, holder)
        command = build_generic_launch_command(
            plan,
            resolve_only=False,
            orchestrator_preflight=preflight,
        )
        log_handle = (plan.run_dir / "trainer.log").open("ab", buffering=0)
        with _blocked_termination_signals():
            trainer = self.supervisor.start_command(
                name="generic-fully-async-launcher",
                command=command,
                working_directory=plan.outer_root,
                environment=_generic_environment(),
                log_handle=log_handle,
                identity_path=plan.run_dir / "trainer-process-identity.json",
                cleanup_timeout_seconds=30,
            )
            self.trainer = trainer
        return trainer

    def wait_trainer(
        self,
        plan: LaunchPlan,
        trainer: ProcessLease,
        endpoints: Sequence[ProcessLease],
        holder: _HolderHandle,
    ) -> int:
        while True:
            return_code = self.supervisor.poll(trainer)
            if return_code is not None:
                return return_code
            for lease in endpoints:
                if not self.supervisor.alive(lease):
                    raise OrchestratorError(
                        f"endpoint {lease.name} exited while trainer was active"
                    )
            if not self.supervisor.alive(holder.watcher):
                raise OrchestratorError(
                    "holder transaction watcher exited while trainer was active"
                )
            time.sleep(0.5)

    def stop_trainer(self, plan: LaunchPlan, trainer: ProcessLease) -> None:
        self.supervisor.stop(trainer, timeout_seconds=30)
        self.trainer = None

    def stop_endpoints(
        self, plan: LaunchPlan, endpoints: Sequence[ProcessLease]
    ) -> None:
        errors: list[str] = []
        by_route = {spec.route_id: spec for spec in plan.endpoints}
        for lease in reversed(tuple(endpoints)):
            try:
                self.supervisor.stop(
                    lease,
                    timeout_seconds=by_route[lease.name].cleanup_timeout_seconds,
                )
            except Exception as exc:
                errors.append(f"{lease.name}: {exc}")
        self.endpoint_leases = ()
        for spec in plan.endpoints:
            try:
                assert_ports_available((spec,))
            except Exception as exc:
                errors.append(f"{spec.route_id} listener cleanup: {exc}")
        if errors:
            raise OrchestratorError("endpoint cleanup failed: " + "; ".join(errors))

    def restore_holders(self, plan: LaunchPlan, holder: _HolderHandle) -> None:
        errors: list[str] = []
        try:
            restore_marker_transaction(holder.state_path, plan.config.holder_lock_path)
        except Exception as exc:
            errors.append(f"marker restore: {exc}")
        try:
            self.supervisor.wait(holder.watcher, timeout_seconds=15)
        except (OrchestratorError, subprocess.TimeoutExpired):
            pass
        try:
            self.supervisor.stop(holder.watcher, timeout_seconds=5)
        except Exception as exc:
            errors.append(f"watcher stop: {exc}")
        finally:
            if (
                holder.watcher.log_handle is not None
                and not holder.watcher.log_handle.closed
            ):
                holder.watcher.log_handle.close()
        try:
            receipt = _load_json_file(
                holder.watcher_receipt, field="holder watcher exit receipt"
            )
            if receipt.get("status") != "pass":
                errors.append("holder watcher did not report pass")
        except Exception as exc:
            errors.append(str(exc))
        if plan.holder_lease is not None:
            try:
                _wait_file_checks(plan.holder_lease.restore_checks, timeout_seconds=30)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise OrchestratorError("holder restoration failed: " + "; ".join(errors))
        self.holder_handle = None


def _execute_local(plan: LaunchPlan) -> int:
    backend = LocalBackend()
    backend.resolve(plan)
    if plan.resolve_only:
        return 0
    holder: _HolderHandle | None = None
    endpoints: tuple[ProcessLease, ...] = ()
    trainer: ProcessLease | None = None
    trainer_rc = 125
    cleanup_errors: list[str] = []
    try:
        holder = backend.acquire_holders(plan)
        endpoints = backend.start_endpoints(plan)
        trainer = backend.start_trainer_with_holder(plan, holder)
        trainer_rc = backend.wait_trainer(plan, trainer, endpoints, holder)
    finally:
        owned_trainer = backend.trainer if backend.trainer is not None else trainer
        if owned_trainer is not None:
            try:
                backend.stop_trainer(plan, owned_trainer)
            except Exception as exc:
                cleanup_errors.append(f"trainer: {exc}")
        owned_endpoints = backend.endpoint_leases or endpoints
        if owned_endpoints:
            try:
                backend.stop_endpoints(plan, owned_endpoints)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        owned_holder = (
            backend.holder_handle if backend.holder_handle is not None else holder
        )
        supervisor = getattr(backend, "supervisor", None)
        holder_watcher = getattr(owned_holder, "watcher", None)
        holder_watchers = (holder_watcher,) if holder_watcher is not None else ()
        holder_watcher_identities = {
            (lease.pid, lease.start_ticks) for lease in holder_watchers
        }
        if supervisor is not None and any(
            (lease.pid, lease.start_ticks) not in holder_watcher_identities
            for lease in supervisor.owned_leases
        ):
            try:
                supervisor.stop_all(exclude=holder_watchers)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if owned_holder is not None:
            try:
                backend.restore_holders(plan, owned_holder)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if supervisor is not None and supervisor.owned_leases:
            try:
                supervisor.stop_all()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        generic_receipt = plan.run_dir / "launch-receipt.json"
        _atomic_json(
            plan.run_dir / "orchestrator-receipt.json",
            {
                "schema": _ORCHESTRATOR_RECEIPT_SCHEMA,
                "status": (
                    "pass" if trainer_rc == 0 and not cleanup_errors else "fail"
                ),
                "trainer_exit_code": trainer_rc,
                "cleanup_errors": cleanup_errors,
                "config_sha256": plan.config.sha256,
                "endpoint_registry_sha256": plan.endpoint_registry_sha256,
                "route_registry_sha256": plan.route_registry_sha256,
                "schedule_sha256": plan.schedule_report["sha256"],
                "generic_launch_receipt": (
                    {
                        "path": str(generic_receipt),
                        "sha256": _sha256(generic_receipt),
                    }
                    if generic_receipt.is_file()
                    else None
                ),
            },
        )
    if cleanup_errors:
        raise OrchestratorError("; ".join(cleanup_errors))
    return trainer_rc


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    outer_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=outer_default / "async_plugins/config/amg_multitask400.yaml",
    )
    parser.add_argument("--outer-root", type=Path, default=outer_default)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--route-registry", type=Path, required=True)
    parser.add_argument("--route-registry-sha256", required=True)
    parser.add_argument("--multitask-source-lock", type=Path, required=True)
    parser.add_argument("--multitask-schedule-certificate", type=Path, required=True)
    parser.add_argument("--endpoint-registry", type=Path, required=True)
    parser.add_argument("--endpoint-registry-sha256", required=True)
    parser.add_argument("--holder-lease", type=Path)
    parser.add_argument("--holder-lease-sha256")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        plan = build_launch_plan(args)
        with _termination_guard():
            return _execute_local(plan)
    except _TerminationRequested as exc:
        return 128 + exc.signum
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"multitask orchestrator failed closed: {exc}", file=sys.stderr)
        return 72


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ROUTE_IDS",
    "EndpointLaunchSpec",
    "ExactProcessSupervisor",
    "LaunchPlan",
    "LocalBackend",
    "OrchestratorConfig",
    "OrchestratorError",
    "ProcessLease",
    "assert_ports_available",
    "build_generic_launch_command",
    "build_launch_plan",
    "execute_launch_plan",
    "load_endpoint_registry",
    "load_holder_lease",
    "load_orchestrator_config",
    "start_endpoint_processes",
]
