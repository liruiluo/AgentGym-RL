"""Eight-replica production coordinator for SWE-bench Verified triads.

This module is deployment-only.  It shards benchmark tasks by the frozen pool
hash, keeps all three treatment arms on one replica, runs the canonical task-0
gate before full work, and aggregates exactly 500 x 3 official outcomes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import statistics
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paired_eval.serialization import canonical_json_bytes

from . import ARMS
from .atomic import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes as atomic_json_bytes,
    ensure_private_directory,
    fsync_directory,
    exclusive_lock,
    read_json,
)
from .cli import driver_from_config, validate_preflight_snapshot
from .identity import verify_image_index
from .production import (
    SHARED_POOL_RUN_CONFIG_SCHEMA,
    ProductionRunConfig,
)
from .shared_pool_contract import (
    SHARED_MODEL_POOL_ASSIGNMENT,
    validate_shared_model_pool_snapshot,
)
from .state import CellKey, RuntimeLaneToken, sha256_json

INDEX_SCHEMA = "amg_swebench_shared_pool_index_v1"
ASSIGNMENT_SCHEMA = "amg_swebench_shared_pool_assignment_v1"
SUMMARY_SCHEMA = "amg_swebench_shared_pool_official_summary_v1"
WORKERS_COMPLETE_SCHEMA = "amg_swebench_shared_pool_workers_complete_v3"
SHA256_PREFIXED_LENGTH = 71
TASK_SLOTS_PER_REPLICA = 2
STARTUP_BARRIER_SCHEMA = "amg_swebench_shared_pool_startup_barrier_v1"
DIGEST_OCCUPANT_SCHEMA = "amg_swebench_image_digest_occupant_v1"
DIGEST_RECONCILIATION_SCHEMA = "amg_swebench_image_digest_reconciliation_v1"
TIMING_CONTRACT_SCHEMA = "amg_swebench_c2_timing_contract_v1"
TIMING_GATE_SCHEMA = "amg_swebench_c2_timing_gate_v1"
TIMING_BUDGET_SECONDS = 28_800.0
TIMING_REQUIRED_METRICS = (
    "setup_materialization",
    "queue_wait",
    "digest_wait",
    "model_generation",
    "environment_tool_execution",
    "grading",
    "publication",
    "per_cell_wall",
    "replica_makespan",
)


def assigned_replica(task_id: str, replica_count: int = 8) -> int:
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task ID is invalid")
    if type(replica_count) is not int or replica_count <= 0:
        raise ValueError("replica count is invalid")
    return (
        int.from_bytes(hashlib.sha256(task_id.encode("utf-8")).digest()[:8], "big")
        % replica_count
    )


def load_canonical_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {label}") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != payload:
        raise RuntimeError(f"{label} is not a canonical object")
    return value


def load_atomic_object(path: Path, label: str) -> Mapping[str, Any]:
    """Load JSON written by :func:`atomic_write_json` (canonical plus LF)."""

    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {label}") from error
    if not isinstance(value, Mapping) or atomic_json_bytes(value) != payload:
        raise RuntimeError(f"{label} is not a canonical atomic object")
    return value


def _digest_paths(root: Path, image_digest: str) -> tuple[Path, Path]:
    if (
        len(image_digest) != SHA256_PREFIXED_LENGTH
        or not image_digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in image_digest[7:])
    ):
        raise ValueError("image digest lease identity is invalid")
    leases = ensure_private_directory(root / "control" / "image-leases")
    stem = image_digest[7:]
    return leases / f"{stem}.lock", leases / f"{stem}.occupant.json"


def _validate_digest_occupant(value: Any, *, path: Path) -> Mapping[str, Any]:
    expected = {
        "schema",
        "status",
        "image_config_digest",
        "task_index",
        "replica_index",
        "slot_index",
        "driver_key",
        "driver_lease_id",
        "driver_owner",
        "lane_generation",
        "lane_fencing_token_sha256",
        "startup_barrier_sha256",
        "acquired_at_unix_ns",
    }
    digest = "sha256:" + path.name.removesuffix(".occupant.json")
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or value.get("schema") != DIGEST_OCCUPANT_SCHEMA
        or value.get("status") != "ACTIVE"
        or value.get("image_config_digest") != digest
        or type(value.get("task_index")) is not int
        or value["task_index"] < 0
        or type(value.get("replica_index")) is not int
        or not 0 <= value["replica_index"] < 8
        or type(value.get("slot_index")) is not int
        or not 0 <= value["slot_index"] < TASK_SLOTS_PER_REPLICA
        or not isinstance(value.get("driver_key"), str)
        or len(value["driver_key"]) != 64
        or not isinstance(value.get("driver_lease_id"), str)
        or len(value["driver_lease_id"]) != 64
        or not isinstance(value.get("driver_owner"), Mapping)
        or type(value.get("lane_generation")) is not int
        or value["lane_generation"] <= 0
        or not isinstance(value.get("lane_fencing_token_sha256"), str)
        or len(value["lane_fencing_token_sha256"]) != 64
        or not isinstance(value.get("startup_barrier_sha256"), str)
        or len(value["startup_barrier_sha256"]) != 64
        or type(value.get("acquired_at_unix_ns")) is not int
        or value["acquired_at_unix_ns"] <= 0
    ):
        raise RuntimeError("durable image-digest occupant is invalid")
    return value


@contextmanager
def digest_lease_admission(
    *,
    coordinator_root: Path,
    image_digest: str,
    task_index: int,
    replica_index: int,
    startup_barrier_sha256: str,
    slot: RuntimeLaneToken,
):
    """Fence duplicate OCI-digest work across owner death and SIGKILL."""

    lock_path, occupant_path = _digest_paths(coordinator_root, image_digest)
    wait_started_wall_ns = time.time_ns()
    wait_started_monotonic_ns = time.monotonic_ns()
    with exclusive_lock(lock_path):
        wait_ended_wall_ns = time.time_ns()
        wait_ended_monotonic_ns = time.monotonic_ns()
        if occupant_path.exists() or occupant_path.is_symlink():
            occupant = load_atomic_object(
                occupant_path, "durable image-digest occupant"
            )
            _validate_digest_occupant(occupant, path=occupant_path)
            raise RuntimeError(
                "durable image-digest occupant requires all-eight reconciliation"
            )
        occupant = {
            "schema": DIGEST_OCCUPANT_SCHEMA,
            "status": "ACTIVE",
            "image_config_digest": image_digest,
            "task_index": task_index,
            "replica_index": replica_index,
            "slot_index": slot.slot_index,
            "driver_key": slot.driver_key,
            "driver_lease_id": slot.lease_id,
            "driver_owner": slot.owner.to_payload(),
            "lane_generation": slot.generation,
            "lane_fencing_token_sha256": hashlib.sha256(
                slot.fencing_token.encode("ascii")
            ).hexdigest(),
            "startup_barrier_sha256": startup_barrier_sha256,
            "acquired_at_unix_ns": time.time_ns(),
        }
        atomic_write_json(occupant_path, occupant)
        completed = False
        try:
            yield {
                "phase": "image_digest_wait",
                "status": "PASS",
                "started_wall_ns": wait_started_wall_ns,
                "ended_wall_ns": wait_ended_wall_ns,
                "started_monotonic_ns": wait_started_monotonic_ns,
                "ended_monotonic_ns": wait_ended_monotonic_ns,
                "duration_ns": max(
                    0, wait_ended_monotonic_ns - wait_started_monotonic_ns
                ),
            }
            completed = True
        finally:
            if completed:
                current = load_atomic_object(
                    occupant_path, "durable image-digest occupant"
                )
                if current != occupant:
                    raise RuntimeError("durable image-digest occupant was fenced")
                occupant_path.unlink()
                fsync_directory(occupant_path.parent)


def reconcile_digest_occupants(root: Path) -> Mapping[str, Any]:
    """Clear stale occupants only after caller completed all-eight reconciliation."""

    leases = ensure_private_directory(root / "control" / "image-leases")
    cleared = []
    for path in sorted(leases.glob("*.occupant.json")):
        before = path.read_bytes()
        occupant = load_atomic_object(path, "durable image-digest occupant")
        _validate_digest_occupant(occupant, path=path)
        lock_path, expected_path = _digest_paths(root, occupant["image_config_digest"])
        if expected_path != path:
            raise RuntimeError("durable image-digest occupant path drifted")
        with exclusive_lock(lock_path):
            if path.read_bytes() != before:
                raise RuntimeError("durable image-digest occupant changed during reconciliation")
            path.unlink()
            fsync_directory(path.parent)
        cleared.append(
            {
                "image_config_digest": occupant["image_config_digest"],
                "task_index": occupant["task_index"],
                "replica_index": occupant["replica_index"],
                "slot_index": occupant["slot_index"],
                "occupant_sha256": hashlib.sha256(before).hexdigest(),
            }
        )
    return {
        "schema": DIGEST_RECONCILIATION_SCHEMA,
        "status": "PASS",
        "all_eight_replica_lanes_held": True,
        "stale_occupants": len(cleared),
        "cleared": cleared,
    }


def image_lock_rows(
    production: ProductionRunConfig,
    task_ids: Mapping[int, str],
) -> tuple[dict[str, Any], ...]:
    """Bind each task to its frozen OCI config digest for cross-shard locking."""

    index_path = Path(production.section("assets")["image_index"])
    verify_image_index(index_path)
    try:
        rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot load the frozen image index") from error
    if len(rows) != 500 or sorted(task_ids) != list(range(500)):
        raise ValueError("image-lock denominator drifted")

    result = []
    for task_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("image-lock row is not an object")
        image = row.get("image")
        config = row.get("config")
        config_digest = config.get("digest") if isinstance(config, Mapping) else None
        task_id = task_ids[task_index]
        suffix = task_id.split("__", 1)[-1]
        if (
            not isinstance(image, str)
            or not image.endswith(f"{suffix}:latest")
            or not isinstance(config_digest, str)
            or len(config_digest) != SHA256_PREFIXED_LENGTH
            or not config_digest.startswith("sha256:")
            or any(
                character not in "0123456789abcdef" for character in config_digest[7:]
            )
        ):
            raise ValueError("image-lock task identity drifted")
        result.append(
            {
                "task_index": task_index,
                "task_id": task_id,
                "image": image,
                "image_config_digest": config_digest,
            }
        )
    return tuple(result)


@dataclass(frozen=True)
class ReplicaConfig:
    replica_index: int
    gpu_uuid: str
    path: Path
    production: ProductionRunConfig
    task_indices: tuple[int, ...]


@dataclass(frozen=True)
class CoordinatorConfig:
    path: Path
    root: Path
    replicas: tuple[ReplicaConfig, ...]
    assignment: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, path: Path | str) -> CoordinatorConfig:
        index_path = Path(path)
        if not index_path.is_absolute():
            raise ValueError("coordinator index path must be absolute")
        value = load_canonical_object(index_path, "coordinator index")
        if set(value) != {"schema", "root", "replicas"}:
            raise ValueError("coordinator index fields drifted")
        if value["schema"] != INDEX_SCHEMA:
            raise ValueError("coordinator index schema drifted")
        root = Path(value["root"])
        if not root.is_absolute():
            raise ValueError("coordinator root must be absolute")
        rows = value["replicas"]
        if not isinstance(rows, list) or len(rows) != 8:
            raise ValueError("coordinator requires exactly eight replicas")

        productions: list[tuple[int, str, Path, ProductionRunConfig]] = []
        common: bytes | None = None
        unique_runtime_rows: list[tuple[Any, ...]] = []
        network_ports: list[int] = []
        for expected_index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != {
                "replica_index",
                "gpu_uuid",
                "config_path",
                "config_sha256",
            }:
                raise ValueError("coordinator replica fields drifted")
            if row["replica_index"] != expected_index:
                raise ValueError("coordinator replica order drifted")
            config_path = Path(row["config_path"])
            if not config_path.is_absolute():
                raise ValueError("replica config path must be absolute")
            if (
                hashlib.sha256(config_path.read_bytes()).hexdigest()
                != row["config_sha256"]
            ):
                raise ValueError("replica config SHA-256 drifted")
            production = ProductionRunConfig.load(config_path)
            shared = production.shared_model_pool
            if (
                production.payload["schema"] != SHARED_POOL_RUN_CONFIG_SCHEMA
                or shared is None
            ):
                raise ValueError("replica is not a shared-pool production config")
            if (
                shared["replica_index"] != expected_index
                or shared["gpu_uuid"] != row["gpu_uuid"]
            ):
                raise ValueError("replica config identity drifted")
            pod = production.section("pod")
            runtime = production.section("runtime")
            grader = production.section("grader")
            serving = production.section("serving")
            identity = canonical_json_bytes(
                {
                    "manifest_path": production.payload["manifest_path"],
                    "manifest_sha256": production.payload["manifest_sha256"],
                    "source": production.section("source"),
                    "assets": production.section("assets"),
                    "pod": {name: pod[name] for name in pod if name != "gpu_uuid"},
                    "docker": production.section("docker"),
                    "task4_receipt": production.section("task4_receipt"),
                    "runtime": {
                        name: runtime[name]
                        for name in runtime
                        if name not in {"pod_local_root", "server_ports"}
                    },
                    "grader": {
                        name: grader[name] for name in grader if name != "output_root"
                    },
                    "shared_model_pool": {
                        name: shared[name]
                        for name in shared
                        if name
                        not in {
                            "replica_index",
                            "gpu_index",
                            "gpu_uuid",
                            "model_port",
                            "proxy_port",
                        }
                    },
                }
            )
            if common is None:
                common = identity
            elif identity != common:
                raise ValueError("replica configs do not share one frozen runtime")
            if (
                runtime["task_slots_per_replica"]
                != TASK_SLOTS_PER_REPLICA
                or len(runtime["server_ports"]) != TASK_SLOTS_PER_REPLICA
            ):
                raise ValueError("replica task-slot lattice drifted")
            network_ports.extend(runtime["server_ports"])
            network_ports.extend((shared["model_port"], shared["proxy_port"]))
            unique_runtime_rows.append(
                (
                    production.run_root,
                    production.evidence_root,
                    Path(runtime["pod_local_root"]),
                    tuple(runtime["server_ports"]),
                    Path(grader["output_root"]),
                    shared["gpu_uuid"],
                    shared["model_port"],
                    shared["proxy_port"],
                    serving["pid"],
                    serving["start_ticks"],
                    serving["pid_file"],
                    serving["receipt_path"],
                )
            )
            productions.append(
                (expected_index, row["gpu_uuid"], config_path, production)
            )

        for column in zip(*unique_runtime_rows):
            if len(set(column)) != 8:
                raise ValueError("replica-local runtime identities are not unique")
        if len(set(network_ports)) != 4 * len(productions):
            raise ValueError("replica-local network ports are not globally unique")

        first = productions[0][3]
        task_ids: dict[int, str] = {}
        for config in first.configs:
            previous = task_ids.setdefault(config.task.task_index, config.task.task_id)
            if previous != config.task.task_id:
                raise ValueError("manifest task identity drifted")
        if sorted(task_ids) != list(range(500)):
            raise ValueError("coordinator manifest is not the full 500 tasks")
        image_rows = image_lock_rows(first, task_ids)
        assignment_rows = [
            {
                "task_index": task_index,
                "task_id": task_ids[task_index],
                "replica_index": assigned_replica(task_ids[task_index]),
                "image": image_rows[task_index]["image"],
                "image_config_digest": image_rows[task_index]["image_config_digest"],
            }
            for task_index in range(500)
        ]
        for replica_index in range(8):
            replica_tasks = [
                row for row in assignment_rows if row["replica_index"] == replica_index
            ]
            for position, row in enumerate(replica_tasks):
                row["slot_index"] = position % TASK_SLOTS_PER_REPLICA
        assignment = tuple(assignment_rows)
        replicas = []
        for replica_index, gpu_uuid, config_path, production in productions:
            tasks = tuple(
                row["task_index"]
                for row in assignment
                if row["replica_index"] == replica_index
            )
            if not tasks:
                raise ValueError("a shared-model replica received no tasks")
            replicas.append(
                ReplicaConfig(
                    replica_index=replica_index,
                    gpu_uuid=gpu_uuid,
                    path=config_path,
                    production=production,
                    task_indices=tasks,
                )
            )
        return cls(index_path, root, tuple(replicas), assignment)

    @property
    def task_zero_replica(self) -> ReplicaConfig:
        replica = self.assignment[0]["replica_index"]
        return self.replicas[replica]

    def write_assignment(self) -> Mapping[str, Any]:
        payload = {
            "schema": ASSIGNMENT_SCHEMA,
            "algorithm": SHARED_MODEL_POOL_ASSIGNMENT,
            "replica_count": 8,
            "paired_arms_same_replica": True,
            "task_slots_per_replica": TASK_SLOTS_PER_REPLICA,
            "deterministic_slot_assignment": "sorted_shard_position_mod_2",
            "task_count": 500,
            "cell_count": 1500,
            "tasks": list(self.assignment),
        }
        atomic_write_json(self.root / "control" / "assignment.json", payload)
        return payload


def release_driver(driver: Any) -> None:
    registry = getattr(driver, "lease_registry", None)
    if registry is not None:
        registry.release()


def validate_live_pool_snapshot(
    value: Any,
    replica: ReplicaConfig,
    label: str,
    *,
    listener_reference: Any = None,
) -> Mapping[str, Any]:
    expected = replica.production.shared_model_pool
    serving = replica.production.section("serving")
    if expected is None:
        raise RuntimeError(f"{label} shared-model pool snapshot is missing")
    value = validate_shared_model_pool_snapshot(
        value,
        f"{label} shared-model pool snapshot",
        listener_reference=listener_reference,
    )
    exact = {
        "status": "PASS",
        "owner": expected["owner"],
        "readiness_sha256": expected["readiness_sha256"],
        "marker_lease_sha256": expected["marker_lease_sha256"],
        "replica_index": replica.replica_index,
        "replica_count": 8,
        "gpu_index": expected["gpu_index"],
        "gpu_uuid": replica.gpu_uuid,
        "model_id": expected["model_id"],
        "model_revision": expected["model_revision"],
        "model_port": expected["model_port"],
        "proxy_port": expected["proxy_port"],
        "server_pid": serving["pid"],
        "server_start_ticks": serving["start_ticks"],
        "assignment_algorithm": SHARED_MODEL_POOL_ASSIGNMENT,
        "cleanup_policy": "retain_external_pool",
        "all_replicas_alive": True,
        "all_endpoints_healthy": True,
    }
    if any(value.get(name) != expected_value for name, expected_value in exact.items()):
        raise RuntimeError(f"{label} shared-model pool identity drifted")
    route = value["proxy_route"]
    expected_upstream = f"http://127.0.0.1:{expected['model_port']}"
    if (
        route.get("upstream_base_url") != expected_upstream
        or route.get("upstream_base_url_sha256")
        != hashlib.sha256(expected_upstream.encode("utf-8")).hexdigest()
    ):
        raise RuntimeError(f"{label} proxy route binding drifted")
    return value


def validated_preflight_pool_snapshot(
    replica: ReplicaConfig,
) -> Mapping[str, Any]:
    snapshot_path = (
        replica.production.run_root / "control" / "preflight-snapshot.json"
    )
    receipt_path = replica.production.run_root / "control" / "preflight-PASS.json"
    try:
        snapshot = read_json(snapshot_path)
        receipt = read_json(receipt_path)
        expected_receipt = validate_preflight_snapshot(
            snapshot, replica.production.preflight_expectations
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "coordinator preflight pool reference is invalid"
        ) from error
    if receipt != expected_receipt:
        raise RuntimeError("coordinator preflight pool reference drifted")
    return validate_live_pool_snapshot(
        snapshot.get("shared_model_pool"), replica, "preflight"
    )


def _reconciliation_cell(value: Any, expected_tasks: set[int]) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"task_index", "arm"}
        or type(value.get("task_index")) is not int
        or value["task_index"] not in expected_tasks
        or value.get("arm") not in ARMS
    ):
        raise RuntimeError("reconciliation cell identity drifted")


def _task_index_list(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or any(type(task_index) is not int or task_index < 0 for task_index in value)
        or value != sorted(set(value))
    ):
        raise RuntimeError(f"{label} drifted")
    return value


def _extract_startup_reconciliation(
    value: Any, expected_task_indices: Sequence[int]
) -> Mapping[str, Any]:
    """Validate every lifecycle receipt and return its one startup receipt."""

    if type(value) is not list or not value:
        raise RuntimeError("lifecycle reconciliation is not a complete list")
    expected_tasks = set(expected_task_indices)
    startup: Mapping[str, Any] | None = None
    for position, raw_row in enumerate(value):
        if not isinstance(raw_row, Mapping):
            raise RuntimeError("lifecycle reconciliation row is not an object")
        fields = set(raw_row)
        if fields == {"startup"}:
            if startup is not None or position != len(value) - 1:
                raise RuntimeError("lifecycle reconciliation must end in one startup receipt")
            raw_startup = raw_row["startup"]
            startup_fields = {
                "schema",
                "task_indices",
                "reconciled_graders",
                "evicted_images",
                "removed_task_roots",
                "foreign_staged_tasks",
                "foreign_loaded_images",
                "residue",
                "slots",
            }
            if not isinstance(raw_startup, Mapping) or set(raw_startup) != startup_fields:
                raise RuntimeError("startup reconciliation receipt fields drifted")
            if raw_startup.get("schema") != "swebench_triad_startup_reconciliation_v1":
                raise RuntimeError("startup reconciliation schema drifted")
            task_indices = _task_index_list(
                raw_startup["task_indices"], "startup reconciliation task indices"
            )
            if task_indices != list(expected_task_indices):
                raise RuntimeError("startup reconciliation task shard drifted")
            for name in (
                "reconciled_graders",
                "evicted_images",
                "foreign_loaded_images",
            ):
                if not isinstance(raw_startup[name], list):
                    raise RuntimeError(f"startup reconciliation {name} drifted")
            removed = _task_index_list(
                raw_startup["removed_task_roots"],
                "startup reconciliation removed task roots",
            )
            if not set(removed).issubset(expected_tasks):
                raise RuntimeError("startup reconciliation removed a foreign task root")
            _task_index_list(
                raw_startup["foreign_staged_tasks"],
                "startup reconciliation foreign staged tasks",
            )
            if not isinstance(raw_startup["residue"], Mapping):
                raise RuntimeError("startup reconciliation residue drifted")
            slots = raw_startup["slots"]
            if (
                not isinstance(slots, list)
                or len(slots) != TASK_SLOTS_PER_REPLICA
                or any(
                    not isinstance(row, Mapping)
                    for row in slots
                )
            ):
                raise RuntimeError("startup reconciliation slot lattice drifted")
            if (
                [row.get("slot_index") for row in slots] != [0, 1]
                or any(
                    set(row)
                    != {"slot_index", "server_port", "lane_generation"}
                    or type(row["server_port"]) is not int
                    or not 1 <= row["server_port"] <= 65535
                    or type(row["lane_generation"]) is not int
                    or row["lane_generation"] <= 0
                    for row in slots
                )
                or len({row["server_port"] for row in slots}) != len(slots)
            ):
                raise RuntimeError("startup reconciliation slot lattice drifted")
            startup = raw_startup
            continue
        if fields == {"cell", "generation", "accepted_recovered", "runtime"}:
            _reconciliation_cell(raw_row["cell"], expected_tasks)
            if type(raw_row["generation"]) is not int or raw_row["generation"] <= 0:
                raise RuntimeError("cell reconciliation generation drifted")
            if type(raw_row["accepted_recovered"]) is not bool:
                raise RuntimeError("cell reconciliation acceptance drifted")
            if not isinstance(raw_row["runtime"], Mapping):
                raise RuntimeError("cell reconciliation runtime receipt drifted")
            continue
        if fields == {"cell", "grade_claim_generation", "grader"}:
            _reconciliation_cell(raw_row["cell"], expected_tasks)
            if (
                type(raw_row["grade_claim_generation"]) is not int
                or raw_row["grade_claim_generation"] <= 0
            ):
                raise RuntimeError("grader reconciliation generation drifted")
            if not isinstance(raw_row["grader"], Mapping):
                raise RuntimeError("grader reconciliation receipt drifted")
            continue
        raise RuntimeError("lifecycle reconciliation row fields drifted")
    if startup is None:
        raise RuntimeError("lifecycle reconciliation startup receipt is missing")
    return startup


def preflight_all(config: CoordinatorConfig) -> list[dict[str, Any]]:
    """Reconcile every replica before validating any shared-daemon snapshot.

    The eight run roots are disjoint, but Docker is shared.  Acquire all eight
    runtime lanes before mutating residue, then let each root remove only its
    own durable residue while temporarily tolerating images bound to another
    replica.  Validation runs only after all roots have reconciled, when the
    shared daemon must be empty again.
    """

    coordinator_sha256 = hashlib.sha256(config.path.read_bytes()).hexdigest()
    replica_config_sha256s = [
        hashlib.sha256(replica.path.read_bytes()).hexdigest()
        for replica in config.replicas
    ]
    barrier_path = config.root / "control" / "preflight-all.json"
    atomic_write_json(
        barrier_path,
        {
            "schema": STARTUP_BARRIER_SCHEMA,
            "status": "RECONCILING",
            "coordinator_index_sha256": coordinator_sha256,
            "replica_config_sha256s": replica_config_sha256s,
        },
    )
    receipts = []
    drivers = []
    try:
        for replica in config.replicas:
            driver = driver_from_config(
                replica.path, assigned_task_indices=replica.task_indices
            )
            drivers.append((replica, driver))
            for slot_index in range(TASK_SLOTS_PER_REPLICA):
                driver.acquire_runtime_lane(None, slot_index=slot_index)
        reconciliations = []
        startup_reconciliations = []
        for replica, driver in drivers:
            receipt = driver.reconcile_dead_work(
                allow_foreign_loaded_images=True
            )
            startup = _extract_startup_reconciliation(
                receipt, replica.task_indices
            )
            startup_reconciliations.append(startup)
            reconciliations.append(
                {
                    "replica_index": replica.replica_index,
                    "receipt": receipt,
                    "receipt_sha256": sha256_json(receipt),
                }
            )
        for startup in startup_reconciliations:
            foreign_staged = startup["foreign_staged_tasks"]
            if not isinstance(foreign_staged, list) or foreign_staged:
                raise RuntimeError("cross-replica durable stage binding drifted")
        shared_image_reconciliation = (
            drivers[0][1].operations.reconcile_unbound_loaded_images()
        )
        if (
            not isinstance(shared_image_reconciliation, Mapping)
            or shared_image_reconciliation.get("status") != "PASS"
            or shared_image_reconciliation.get("remaining_images") != 0
        ):
            raise RuntimeError("shared Docker image reconciliation failed")
        digest_lease_reconciliation = reconcile_digest_occupants(config.root)
        for replica, driver in drivers:
            receipt = driver.preflight()
            receipts.append(
                {
                    "replica_index": replica.replica_index,
                    "gpu_uuid": replica.gpu_uuid,
                    "receipt": receipt,
                    "receipt_sha256": sha256_json(receipt),
                }
            )
    finally:
        for _, driver in reversed(drivers):
            release_driver(driver)
    result = {
        "schema": STARTUP_BARRIER_SCHEMA,
        "status": "PASS",
        "replica_count": 8,
        "task_slots_per_replica": TASK_SLOTS_PER_REPLICA,
        "all_slots_held_during_reconciliation": True,
        "startup_reconciliation_complete": True,
        "coordinator_index_sha256": coordinator_sha256,
        "replica_config_sha256s": replica_config_sha256s,
        "reconciliations": reconciliations,
        "shared_image_reconciliation": shared_image_reconciliation,
        "digest_lease_reconciliation": digest_lease_reconciliation,
        "replicas": receipts,
    }
    atomic_write_json(barrier_path, result)
    return receipts


def require_startup_barrier(
    coordinator_root: Path,
    *,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    path = coordinator_root / "control" / "preflight-all.json"
    value = load_atomic_object(path, "startup reconciliation barrier")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256
    ):
        raise RuntimeError("startup reconciliation barrier digest drifted")
    if (
        value.get("schema") != STARTUP_BARRIER_SCHEMA
        or value.get("status") != "PASS"
        or value.get("replica_count") != 8
        or value.get("task_slots_per_replica") != TASK_SLOTS_PER_REPLICA
        or value.get("all_slots_held_during_reconciliation") is not True
        or value.get("startup_reconciliation_complete") is not True
        or not isinstance(value.get("coordinator_index_sha256"), str)
        or len(value["coordinator_index_sha256"]) != 64
        or not isinstance(value.get("replica_config_sha256s"), list)
        or len(value["replica_config_sha256s"]) != 8
        or any(
            not isinstance(digest, str) or len(digest) != 64
            for digest in value["replica_config_sha256s"]
        )
        or not isinstance(value.get("reconciliations"), list)
        or len(value["reconciliations"]) != 8
        or not isinstance(value.get("replicas"), list)
        or len(value["replicas"]) != 8
        or not isinstance(value.get("shared_image_reconciliation"), Mapping)
        or value["shared_image_reconciliation"].get("status") != "PASS"
        or value["shared_image_reconciliation"].get("remaining_images") != 0
        or not isinstance(value.get("digest_lease_reconciliation"), Mapping)
        or value["digest_lease_reconciliation"].get("schema")
        != DIGEST_RECONCILIATION_SCHEMA
        or value["digest_lease_reconciliation"].get("status") != "PASS"
        or value["digest_lease_reconciliation"].get(
            "all_eight_replica_lanes_held"
        )
        is not True
    ):
        raise RuntimeError("startup reconciliation barrier is invalid")
    return value


def validated_startup_barrier(
    config: CoordinatorConfig,
    *,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    value = require_startup_barrier(
        config.root,
        expected_sha256=expected_sha256,
    )
    expected_configs = [
        hashlib.sha256(replica.path.read_bytes()).hexdigest()
        for replica in config.replicas
    ]
    if (
        value.get("coordinator_index_sha256")
        != hashlib.sha256(config.path.read_bytes()).hexdigest()
        or value.get("replica_config_sha256s") != expected_configs
    ):
        raise RuntimeError("startup reconciliation barrier config binding drifted")
    return value


def cleanup_all(config: CoordinatorConfig) -> Mapping[str, Any]:
    """Reconcile shared residue while one coordinator holds all eight leases."""

    replicas = preflight_all(config)
    preflight_path = config.root / "control" / "preflight-all.json"
    result = {
        "schema": "amg_swebench_shared_pool_cleanup_v1",
        "status": "PASS",
        "replica_count": 8,
        "all_replica_leases_held_during_reconciliation": True,
        "preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        "replicas": replicas,
        "external_model_pool_retained": True,
        "allocation_retained": True,
    }
    atomic_write_json(config.root / "control" / "cleanup-all.json", result)
    return result


def run_gate(config: CoordinatorConfig) -> Mapping[str, Any]:
    validated_startup_barrier(config)
    replica = config.task_zero_replica
    driver = driver_from_config(replica.path, assigned_task_indices=(0,))
    try:
        gate = driver.gate(auto_run_full=False)
    finally:
        release_driver(driver)
    result = {
        "schema": "amg_swebench_shared_pool_gate_v1",
        "status": "PASS",
        "replica_index": replica.replica_index,
        "gpu_uuid": replica.gpu_uuid,
        "gate": gate,
        "gate_sha256": sha256_json(gate),
    }
    atomic_write_json(config.root / "control" / "gate.json", result)
    return result


def _worker(
    config_path: str,
    task_indices: tuple[int, ...],
    task_image_digests: tuple[str, ...],
    task_slot_indices: tuple[int, ...],
    replica_index: int,
    coordinator_root: str,
    startup_barrier_sha256: str,
) -> dict[str, Any]:
    if not (
        len(task_indices)
        == len(task_image_digests)
        == len(task_slot_indices)
    ):
        raise ValueError("worker task/image lock lattice drifted")
    if tuple(sorted(set(task_indices))) != task_indices or any(
        slot_index not in range(TASK_SLOTS_PER_REPLICA)
        for slot_index in task_slot_indices
    ):
        raise ValueError("worker deterministic slot lattice drifted")
    root = Path(coordinator_root)
    require_startup_barrier(
        root,
        expected_sha256=startup_barrier_sha256,
    )
    driver = driver_from_config(Path(config_path), assigned_task_indices=task_indices)
    progress_path = root / "progress" / f"replica-{replica_index}.json"
    started = time.monotonic()
    completed: set[int] = set()
    progress_lock = threading.Lock()
    digest_locks: dict[str, Any] = {}
    try:
        driver.ensure_driver_lease()
        driver._read_validated_preflight()
        ensure_private_directory(root / "control" / "image-leases")
        queues: list[list[tuple[int, str, int, int]]] = [
            [] for _ in range(TASK_SLOTS_PER_REPLICA)
        ]
        for task_index, image_digest, slot_index in zip(
            task_indices,
            task_image_digests,
            task_slot_indices,
        ):
            if (
                len(image_digest) != SHA256_PREFIXED_LENGTH
                or not image_digest.startswith("sha256:")
            ):
                raise ValueError("worker image lock digest drifted")
            queues[slot_index].append(
                (task_index, image_digest, time.time_ns(), time.monotonic_ns())
            )
            digest_locks.setdefault(
                image_digest,
                threading.Lock(),
            )

        def run_slot(slot_index: int) -> dict[str, Any]:
            slot_started = time.monotonic()
            slot_completed = []
            for task_index, image_digest, queued_wall_ns, queued_monotonic_ns in queues[
                slot_index
            ]:
                slot_dequeued_wall_ns = time.time_ns()
                slot_dequeued_monotonic_ns = time.monotonic_ns()
                with digest_locks[image_digest]:
                    driver.run_task(
                        task_index,
                        gate=task_index == 0,
                        slot_index=slot_index,
                        admission=lambda slot: digest_lease_admission(
                            coordinator_root=root,
                            image_digest=image_digest,
                            task_index=task_index,
                            replica_index=replica_index,
                            startup_barrier_sha256=startup_barrier_sha256,
                            slot=slot,
                        ),
                        queued_wall_ns=queued_wall_ns,
                        queued_monotonic_ns=queued_monotonic_ns,
                        slot_dequeued_wall_ns=slot_dequeued_wall_ns,
                        slot_dequeued_monotonic_ns=slot_dequeued_monotonic_ns,
                    )
                slot_completed.append(task_index)
                with progress_lock:
                    completed.add(task_index)
                    atomic_write_json(
                        progress_path,
                        {
                            "schema": "amg_swebench_shared_pool_progress_v2",
                            "status": "RUNNING",
                            "replica_index": replica_index,
                            "task_slots_per_replica": TASK_SLOTS_PER_REPLICA,
                            "completed_task_indices": sorted(completed),
                            "completed_tasks": len(completed),
                            "total_tasks": len(task_indices),
                            "last_task_index": task_index,
                            "last_slot_index": slot_index,
                            "last_image_config_digest": image_digest,
                            "wall_seconds": round(
                                time.monotonic() - started, 6
                            ),
                        },
                    )
            return {
                "slot_index": slot_index,
                "completed_task_indices": slot_completed,
                "wall_seconds": round(time.monotonic() - slot_started, 6),
            }

        with ThreadPoolExecutor(
            max_workers=TASK_SLOTS_PER_REPLICA
        ) as executor:
            slot_results = list(executor.map(run_slot, range(TASK_SLOTS_PER_REPLICA)))
        result = {
            "schema": "amg_swebench_shared_pool_worker_v2",
            "status": "PASS",
            "replica_index": replica_index,
            "task_slots_per_replica": TASK_SLOTS_PER_REPLICA,
            "completed_task_indices": sorted(completed),
            "completed_tasks": len(completed),
            "total_tasks": len(task_indices),
            "slots": slot_results,
            "wall_seconds": round(time.monotonic() - started, 6),
        }
        atomic_write_json(progress_path, result)
        return result
    finally:
        release_driver(driver)


def _require_sha256_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} is not a SHA-256")
    return value


def _require_git_oid(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} is not a Git object ID")
    return value


def timing_contract_path(config: CoordinatorConfig) -> Path:
    return config.root / "control" / "timing-contract.json"


def load_timing_contract(config: CoordinatorConfig) -> Mapping[str, Any]:
    path = timing_contract_path(config)
    value = load_canonical_object(path, "C=2 timing contract")
    expected_fields = {
        "schema",
        "status",
        "budget_seconds",
        "task_slots_per_replica",
        "panel_selection",
        "panel_tasks",
        "required_metrics",
        "projection",
        "bindings",
    }
    if (
        set(value) != expected_fields
        or value.get("schema") != TIMING_CONTRACT_SCHEMA
        or value.get("status") != "FROZEN"
        or value.get("budget_seconds") != TIMING_BUDGET_SECONDS
        or value.get("task_slots_per_replica") != TASK_SLOTS_PER_REPLICA
        or value.get("panel_selection")
        != "per_replica_slotwise_deterministic_spread_distinct_digest_v1"
        or value.get("required_metrics") != list(TIMING_REQUIRED_METRICS)
    ):
        raise RuntimeError("C=2 timing contract fields drifted")
    projection = value.get("projection")
    if (
        not isinstance(projection, Mapping)
        or set(projection)
        != {
            "formula",
            "straggler_percentile",
            "straggler_margin_floor",
            "full_task_count",
            "full_cell_count",
        }
        or projection.get("formula")
        != "max(panel_replica_makespan*ceil(shard_tasks/2))*max(1.10,p95_task/median_task)"
        or projection.get("straggler_percentile") != 0.95
        or projection.get("straggler_margin_floor") != 1.10
        or projection.get("full_task_count") != 500
        or projection.get("full_cell_count") != 1500
    ):
        raise RuntimeError("C=2 timing projection contract drifted")
    bindings = value.get("bindings")
    expected_binding_fields = {
        "coordinator_index_sha256",
        "replica_config_sha256s",
        "manifest_sha256",
        "deployment_commit",
        "deployment_tree",
        "inner_commit",
        "assignment_algorithm",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_binding_fields:
        raise RuntimeError("C=2 timing contract bindings drifted")
    for name in ("coordinator_index_sha256", "manifest_sha256"):
        _require_sha256_text(bindings.get(name), f"timing binding {name}")
    for name in ("deployment_commit", "deployment_tree", "inner_commit"):
        _require_git_oid(bindings.get(name), f"timing binding {name}")
    expected_configs = [
        hashlib.sha256(replica.path.read_bytes()).hexdigest()
        for replica in config.replicas
    ]
    first = config.replicas[0].production
    source = first.section("source")
    if (
        bindings["coordinator_index_sha256"]
        != hashlib.sha256(config.path.read_bytes()).hexdigest()
        or bindings.get("replica_config_sha256s") != expected_configs
        or bindings["manifest_sha256"] != first.payload["manifest_sha256"]
        or bindings["deployment_commit"] != source["deployment_commit"]
        or bindings["inner_commit"] != source["inner_commit"]
        or bindings["assignment_algorithm"] != SHARED_MODEL_POOL_ASSIGNMENT
    ):
        raise RuntimeError("C=2 timing contract source binding drifted")
    rows = value.get("panel_tasks")
    if not isinstance(rows, list) or len(rows) != 16:
        raise RuntimeError("C=2 timing panel denominator drifted")
    seen_tasks = set()
    per_replica: dict[int, list[Mapping[str, Any]]] = {
        replica: [] for replica in range(8)
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "replica_index",
            "slot_index",
            "task_index",
            "task_id",
            "image_config_digest",
        }:
            raise RuntimeError("C=2 timing panel row fields drifted")
        task_index = row["task_index"]
        replica_index = row["replica_index"]
        if (
            type(task_index) is not int
            or not 0 <= task_index < 500
            or task_index == 0
            or task_index in seen_tasks
            or type(replica_index) is not int
            or not 0 <= replica_index < 8
            or row["slot_index"] not in range(TASK_SLOTS_PER_REPLICA)
        ):
            raise RuntimeError("C=2 timing panel identity drifted")
        assignment = config.assignment[task_index]
        if any(
            row.get(name) != assignment[name]
            for name in (
                "replica_index",
                "slot_index",
                "task_id",
                "image_config_digest",
            )
        ):
            raise RuntimeError("C=2 timing panel assignment drifted")
        seen_tasks.add(task_index)
        per_replica[replica_index].append(row)
    for replica_index, panel in per_replica.items():
        if (
            len(panel) != 2
            or sorted(row["slot_index"] for row in panel) != [0, 1]
            or len({row["image_config_digest"] for row in panel}) != 2
            or any(row["replica_index"] != replica_index for row in panel)
        ):
            raise RuntimeError("C=2 timing panel diversity drifted")
    return value


def _validated_gate(config: CoordinatorConfig) -> Mapping[str, Any]:
    validated_startup_barrier(config)
    gate = read_json(config.root / "control" / "gate.json")
    task_zero = config.task_zero_replica
    if (
        not isinstance(gate, Mapping)
        or set(gate)
        != {"schema", "status", "replica_index", "gpu_uuid", "gate", "gate_sha256"}
        or gate.get("schema") != "amg_swebench_shared_pool_gate_v1"
        or gate.get("status") != "PASS"
        or gate.get("replica_index") != task_zero.replica_index
        or gate.get("gpu_uuid") != task_zero.gpu_uuid
        or not isinstance(gate.get("gate"), Mapping)
        or gate.get("gate_sha256") != sha256_json(gate["gate"])
    ):
        raise RuntimeError("full run requires the canonical task-0 gate")
    gate_driver = driver_from_config(task_zero.path, assigned_task_indices=(0,))
    try:
        if not gate_driver.gate_path.is_file():
            raise RuntimeError("full run requires the canonical task-0 gate")
        canonical_gate = gate_driver.gate(auto_run_full=False)
    finally:
        release_driver(gate_driver)
    if gate["gate"] != canonical_gate:
        raise RuntimeError("full run requires the canonical task-0 gate")
    return gate


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile input is invalid")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_summary(values: Sequence[float]) -> Mapping[str, Any]:
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise RuntimeError("timing metric values are invalid")
    return {
        "count": len(values),
        "p50_seconds": round(_percentile(values, 0.50), 6),
        "p95_seconds": round(_percentile(values, 0.95), 6),
        "max_seconds": round(max(values), 6),
    }


def _phase_map(rows: Any, label: str) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{label} phase timings are missing")
    result = {}
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("phase"), str)
            or row.get("status") != "PASS"
            or type(row.get("duration_ns")) is not int
            or row["duration_ns"] < 0
            or row["phase"] in result
        ):
            raise RuntimeError(f"{label} phase timing is invalid")
        result[row["phase"]] = row
    return result


def _load_cell_timing(
    replica: ReplicaConfig,
    driver: Any,
    task_index: int,
    arm: str,
) -> Mapping[str, Any]:
    accepted = driver.store.read_accepted(CellKey(task_index, arm))
    generation = accepted["attempt_generation"]
    runtime_path = (
        replica.production.run_root
        / "control"
        / "cells"
        / f"{task_index:04d}-{arm}"
        / f"generation-{generation:08d}.json"
    )
    runtime = load_atomic_object(runtime_path, "timing-panel cell runtime")
    if (
        runtime.get("schema") != "swebench_triad_cell_runtime_v1"
        or runtime.get("status") != "PASS"
        or runtime.get("task_index") != task_index
        or runtime.get("arm") != arm
        or runtime.get("generation") != generation
    ):
        raise RuntimeError("timing-panel cell runtime identity drifted")
    publication_path = runtime_path.with_name(
        runtime_path.stem + ".publication.json"
    )
    publication = load_atomic_object(
        publication_path, "timing-panel cell publication"
    )
    if (
        publication.get("schema")
        != "swebench_triad_cell_publication_timing_v1"
        or publication.get("status") != "PASS"
        or publication.get("cell_status") != "PASS"
        or publication.get("task_index") != task_index
        or publication.get("arm") != arm
        or publication.get("generation") != generation
        or publication.get("runtime_receipt_path") != str(runtime_path)
        or publication.get("runtime_receipt_sha256")
        != hashlib.sha256(runtime_path.read_bytes()).hexdigest()
        or type(publication.get("duration_ns")) is not int
        or publication["duration_ns"] < 0
    ):
        raise RuntimeError("timing-panel cell publication binding drifted")
    phases = _phase_map(runtime.get("phase_timings"), "cell runtime")
    policy = phases.get("policy_and_model_execution")
    if policy is None:
        raise RuntimeError("timing-panel policy phase is missing")
    events = runtime.get("model_transport_events")
    if not isinstance(events, list) or not events:
        raise RuntimeError("timing-panel model transport events are missing")
    model_ns = 0
    for event in events:
        if (
            not isinstance(event, Mapping)
            or event.get("phase") not in {"tokenize", "chat_completion"}
            or type(event.get("duration_ns")) is not int
            or event["duration_ns"] < 0
        ):
            raise RuntimeError("timing-panel model transport event is invalid")
        if event["phase"] == "chat_completion":
            model_ns += event["duration_ns"]
    if model_ns <= 0 or model_ns > policy["duration_ns"]:
        raise RuntimeError("timing-panel model/environment decomposition is invalid")
    environment_ns = policy["duration_ns"] - model_ns
    for name, phase in phases.items():
        if name.startswith(("environment_", "cgroup_")):
            environment_ns += phase["duration_ns"]
    started = min(row["started_monotonic_ns"] for row in phases.values())
    ended = max(row["ended_monotonic_ns"] for row in phases.values())
    cell_wall_ns = ended - started + publication["duration_ns"]
    return {
        "task_index": task_index,
        "arm": arm,
        "generation": generation,
        "model_generation_seconds": model_ns / 1e9,
        "environment_tool_execution_seconds": environment_ns / 1e9,
        "publication_seconds": publication["duration_ns"] / 1e9,
        "cell_wall_seconds": cell_wall_ns / 1e9,
        "runtime_receipt_sha256": hashlib.sha256(
            runtime_path.read_bytes()
        ).hexdigest(),
        "publication_receipt_sha256": hashlib.sha256(
            publication_path.read_bytes()
        ).hexdigest(),
    }


def _collect_timing_gate(
    config: CoordinatorConfig,
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    panel_rows = contract["panel_tasks"]
    task_rows = []
    cell_rows = []
    replica_makespans = []
    phase_values: dict[str, list[float]] = {
        name: [] for name in TIMING_REQUIRED_METRICS
    }
    for replica in config.replicas:
        selected = [
            row for row in panel_rows if row["replica_index"] == replica.replica_index
        ]
        driver = driver_from_config(
            replica.path,
            assigned_task_indices=replica.task_indices,
        )
        replica_intervals = []
        try:
            for panel in selected:
                task_index = panel["task_index"]
                completion = driver.load_task_completion(task_index)
                timing_path = Path(completion["timing_receipt"]["path"])
                timing = load_atomic_object(timing_path, "timing-panel task timing")
                if (
                    timing.get("schema") != "swebench_triad_task_phase_timing_v1"
                    or timing.get("status") != "READY_FOR_PUBLICATION"
                    or timing.get("task_index") != task_index
                    or timing.get("slot_index") != panel["slot_index"]
                ):
                    raise RuntimeError("timing-panel task timing identity drifted")
                phases = _phase_map(timing.get("phases"), "task")
                required_task_phases = {
                    "task_slot_queue",
                    "runtime_lane_wait",
                    "image_digest_wait",
                    "oci_stage",
                    "official_grade_native",
                    "official_grade_amg_compaction_only",
                    "official_grade_amg_memory",
                }
                if not required_task_phases.issubset(phases):
                    raise RuntimeError("timing-panel task phases are incomplete")
                publication_path = driver.task_publication_path(task_index)
                publication = read_json(publication_path)
                if (
                    publication.get("schema")
                    != "swebench_triad_task_publication_timing_v1"
                    or publication.get("status") != "PASS"
                    or publication.get("task_index") != task_index
                    or publication.get("completion_sha256")
                    != hashlib.sha256(
                        driver.task_completion_path(task_index).read_bytes()
                    ).hexdigest()
                ):
                    raise RuntimeError("timing-panel task publication drifted")
                setup = phases["oci_stage"]["duration_ns"] / 1e9
                queue_wait = phases["task_slot_queue"]["duration_ns"] / 1e9
                digest_wait = phases["image_digest_wait"]["duration_ns"] / 1e9
                grading = sum(
                    phases[f"official_grade_{arm}"]["duration_ns"]
                    for arm in ARMS
                ) / 1e9
                publication_seconds = publication["duration_ns"] / 1e9
                task_cells = [
                    _load_cell_timing(replica, driver, task_index, arm)
                    for arm in ARMS
                ]
                cell_rows.extend(task_cells)
                model_generation = sum(
                    row["model_generation_seconds"] for row in task_cells
                )
                environment = sum(
                    row["environment_tool_execution_seconds"] for row in task_cells
                )
                publication_seconds += sum(
                    row["publication_seconds"] for row in task_cells
                )
                task_seconds = timing["duration_ns"] / 1e9 + publication["duration_ns"] / 1e9
                task_row = {
                    "replica_index": replica.replica_index,
                    "slot_index": panel["slot_index"],
                    "task_index": task_index,
                    "task_id": panel["task_id"],
                    "setup_materialization_seconds": setup,
                    "queue_wait_seconds": queue_wait,
                    "digest_wait_seconds": digest_wait,
                    "model_generation_seconds": model_generation,
                    "environment_tool_execution_seconds": environment,
                    "grading_seconds": grading,
                    "publication_seconds": publication_seconds,
                    "task_wall_seconds": task_seconds,
                    "task_timing_sha256": hashlib.sha256(
                        timing_path.read_bytes()
                    ).hexdigest(),
                    "task_publication_sha256": hashlib.sha256(
                        publication_path.read_bytes()
                    ).hexdigest(),
                }
                task_rows.append(task_row)
                replica_intervals.append(
                    (timing["started_monotonic_ns"], timing["ended_monotonic_ns"])
                )
                phase_values["setup_materialization"].append(setup)
                phase_values["queue_wait"].append(queue_wait)
                phase_values["digest_wait"].append(digest_wait)
                phase_values["model_generation"].append(model_generation)
                phase_values["environment_tool_execution"].append(environment)
                phase_values["grading"].append(grading)
                phase_values["publication"].append(publication_seconds)
            overlap_ns = min(end for _, end in replica_intervals) - max(
                start for start, _ in replica_intervals
            )
            if len(replica_intervals) != 2 or overlap_ns <= 0:
                raise RuntimeError("timing panel did not exercise C=2 on every replica")
            makespan = (
                max(end for _, end in replica_intervals)
                - min(start for start, _ in replica_intervals)
            ) / 1e9
            replica_makespans.append(
                {
                    "replica_index": replica.replica_index,
                    "panel_tasks": [row["task_index"] for row in selected],
                    "panel_makespan_seconds": makespan,
                    "overlap_seconds": overlap_ns / 1e9,
                    "full_shard_tasks": len(replica.task_indices),
                    "projected_waves": math.ceil(
                        len(replica.task_indices) / TASK_SLOTS_PER_REPLICA
                    ),
                }
            )
            phase_values["replica_makespan"].append(makespan)
        finally:
            release_driver(driver)
    for cell in cell_rows:
        phase_values["per_cell_wall"].append(cell["cell_wall_seconds"])
    task_seconds = [row["task_wall_seconds"] for row in task_rows]
    median_task = statistics.median(task_seconds)
    if median_task <= 0:
        raise RuntimeError("timing panel task median is nonpositive")
    p95_task = _percentile(task_seconds, 0.95)
    straggler_margin = max(1.10, p95_task / median_task)
    projected_without_margin = max(
        row["panel_makespan_seconds"] * row["projected_waves"]
        for row in replica_makespans
    )
    projected = projected_without_margin * straggler_margin
    status_value = "PASS" if projected <= TIMING_BUDGET_SECONDS else "FAIL_CLOSED"
    barrier_path = config.root / "control" / "preflight-all.json"
    gate_path = config.root / "control" / "gate.json"
    assignment_path = config.root / "control" / "assignment.json"
    result = {
        "schema": TIMING_GATE_SCHEMA,
        "status": status_value,
        "budget_seconds": TIMING_BUDGET_SECONDS,
        "panel_task_count": 16,
        "panel_cell_count": 48,
        "task_slots_per_replica": TASK_SLOTS_PER_REPLICA,
        "metrics": {
            name: _metric_summary(values)
            for name, values in phase_values.items()
        },
        "tasks": task_rows,
        "cells": cell_rows,
        "replicas": replica_makespans,
        "projection": {
            "formula": contract["projection"]["formula"],
            "task_p95_seconds": p95_task,
            "task_median_seconds": median_task,
            "straggler_margin": straggler_margin,
            "projected_without_margin_seconds": projected_without_margin,
            "projected_full_makespan_seconds": projected,
            "within_budget": projected <= TIMING_BUDGET_SECONDS,
        },
        "bindings": {
            "timing_contract_sha256": hashlib.sha256(
                timing_contract_path(config).read_bytes()
            ).hexdigest(),
            "coordinator_index_sha256": hashlib.sha256(
                config.path.read_bytes()
            ).hexdigest(),
            "assignment_sha256": hashlib.sha256(
                assignment_path.read_bytes()
            ).hexdigest(),
            "startup_barrier_sha256": hashlib.sha256(
                barrier_path.read_bytes()
            ).hexdigest(),
            "gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            "replica_config_sha256s": [
                hashlib.sha256(replica.path.read_bytes()).hexdigest()
                for replica in config.replicas
            ],
            "deployment_commit": contract["bindings"]["deployment_commit"],
            "deployment_tree": contract["bindings"]["deployment_tree"],
            "inner_commit": contract["bindings"]["inner_commit"],
        },
    }
    atomic_write_json(config.root / "control" / "timing-gate.json", result)
    if status_value != "PASS":
        raise RuntimeError(
            f"SWE timing projection exceeds 28800 seconds: {projected:.3f}"
        )
    return result


def run_timing(config: CoordinatorConfig) -> Mapping[str, Any]:
    _validated_gate(config)
    contract = load_timing_contract(config)
    barrier_path = config.root / "control" / "preflight-all.json"
    barrier_sha256 = hashlib.sha256(barrier_path.read_bytes()).hexdigest()
    panel_rows = contract["panel_tasks"]
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {}
        for replica in config.replicas:
            selected = sorted(
                (
                    row
                    for row in panel_rows
                    if row["replica_index"] == replica.replica_index
                ),
                key=lambda row: row["task_index"],
            )
            futures[
                executor.submit(
                    _worker,
                    str(replica.path),
                    tuple(row["task_index"] for row in selected),
                    tuple(row["image_config_digest"] for row in selected),
                    tuple(row["slot_index"] for row in selected),
                    replica.replica_index,
                    str(config.root),
                    barrier_sha256,
                )
            ] = replica.replica_index
        for future in as_completed(futures):
            results.append(future.result())
    if (
        len(results) != 8
        or sum(row.get("completed_tasks", 0) for row in results) != 16
        or any(row.get("status") != "PASS" for row in results)
    ):
        raise RuntimeError("C=2 timing-panel worker completion drifted")
    return _collect_timing_gate(config, contract)


def validated_timing_gate(config: CoordinatorConfig) -> Mapping[str, Any]:
    contract = load_timing_contract(config)
    path = config.root / "control" / "timing-gate.json"
    gate = load_atomic_object(path, "C=2 timing gate")
    bindings = gate.get("bindings")
    projection = gate.get("projection")
    expected_binding_fields = {
        "timing_contract_sha256",
        "coordinator_index_sha256",
        "assignment_sha256",
        "startup_barrier_sha256",
        "gate_sha256",
        "replica_config_sha256s",
        "deployment_commit",
        "deployment_tree",
        "inner_commit",
    }
    if (
        gate.get("schema") != TIMING_GATE_SCHEMA
        or gate.get("status") != "PASS"
        or gate.get("budget_seconds") != TIMING_BUDGET_SECONDS
        or gate.get("panel_task_count") != 16
        or gate.get("panel_cell_count") != 48
        or gate.get("task_slots_per_replica") != TASK_SLOTS_PER_REPLICA
        or not isinstance(gate.get("metrics"), Mapping)
        or set(gate["metrics"]) != set(TIMING_REQUIRED_METRICS)
        or not isinstance(gate.get("tasks"), list)
        or len(gate["tasks"]) != 16
        or not isinstance(gate.get("cells"), list)
        or len(gate["cells"]) != 48
        or not isinstance(gate.get("replicas"), list)
        or len(gate["replicas"]) != 8
        or not isinstance(bindings, Mapping)
        or set(bindings) != expected_binding_fields
        or not isinstance(projection, Mapping)
        or projection.get("formula") != contract["projection"]["formula"]
        or projection.get("within_budget") is not True
        or not isinstance(projection.get("projected_full_makespan_seconds"), (int, float))
        or projection["projected_full_makespan_seconds"] > TIMING_BUDGET_SECONDS
    ):
        raise RuntimeError("C=2 timing gate is incomplete or over budget")
    try:
        task_seconds = [float(row["task_wall_seconds"]) for row in gate["tasks"]]
        replica_bases = [
            float(row["panel_makespan_seconds"]) * int(row["projected_waves"])
            for row in gate["replicas"]
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("C=2 timing gate projection inputs are invalid") from error
    if (
        any(value <= 0 or not math.isfinite(value) for value in task_seconds)
        or any(value <= 0 or not math.isfinite(value) for value in replica_bases)
    ):
        raise RuntimeError("C=2 timing gate projection inputs are invalid")
    expected_p95 = _percentile(task_seconds, 0.95)
    expected_median = statistics.median(task_seconds)
    expected_margin = max(1.10, expected_p95 / expected_median)
    expected_without_margin = max(replica_bases)
    expected_projected = expected_without_margin * expected_margin
    numeric_projection = {
        "task_p95_seconds": expected_p95,
        "task_median_seconds": expected_median,
        "straggler_margin": expected_margin,
        "projected_without_margin_seconds": expected_without_margin,
        "projected_full_makespan_seconds": expected_projected,
    }
    if any(
        not math.isclose(
            float(projection.get(name, -1)), expected_value, rel_tol=1e-12, abs_tol=1e-9
        )
        for name, expected_value in numeric_projection.items()
    ):
        raise RuntimeError("C=2 timing gate projection arithmetic drifted")
    for name, summary in gate["metrics"].items():
        expected_count = 48 if name == "per_cell_wall" else 8 if name == "replica_makespan" else 16
        if (
            not isinstance(summary, Mapping)
            or set(summary) != {"count", "p50_seconds", "p95_seconds", "max_seconds"}
            or summary.get("count") != expected_count
            or any(
                not isinstance(summary.get(field), (int, float))
                or summary[field] < 0
                or not math.isfinite(float(summary[field]))
                for field in ("p50_seconds", "p95_seconds", "max_seconds")
            )
        ):
            raise RuntimeError("C=2 timing gate metric summary drifted")

    expected = {
        "timing_contract_sha256": hashlib.sha256(
            timing_contract_path(config).read_bytes()
        ).hexdigest(),
        "coordinator_index_sha256": hashlib.sha256(
            config.path.read_bytes()
        ).hexdigest(),
        "assignment_sha256": hashlib.sha256(
            (config.root / "control" / "assignment.json").read_bytes()
        ).hexdigest(),
        "startup_barrier_sha256": hashlib.sha256(
            (config.root / "control" / "preflight-all.json").read_bytes()
        ).hexdigest(),
        "gate_sha256": hashlib.sha256(
            (config.root / "control" / "gate.json").read_bytes()
        ).hexdigest(),
        "replica_config_sha256s": [
            hashlib.sha256(replica.path.read_bytes()).hexdigest()
            for replica in config.replicas
        ],
        "deployment_commit": contract["bindings"]["deployment_commit"],
        "deployment_tree": contract["bindings"]["deployment_tree"],
        "inner_commit": contract["bindings"]["inner_commit"],
    }
    if dict(bindings) != expected:
        raise RuntimeError("C=2 timing gate binding drifted")
    return gate


def run_full(config: CoordinatorConfig) -> list[dict[str, Any]]:
    _validated_gate(config)
    validated_timing_gate(config)
    barrier_path = config.root / "control" / "preflight-all.json"
    barrier_sha256 = hashlib.sha256(barrier_path.read_bytes()).hexdigest()
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                _worker,
                str(replica.path),
                replica.task_indices,
                tuple(
                    config.assignment[task_index]["image_config_digest"]
                    for task_index in replica.task_indices
                ),
                tuple(
                    config.assignment[task_index]["slot_index"]
                    for task_index in replica.task_indices
                ),
                replica.replica_index,
                str(config.root),
                barrier_sha256,
            ): replica.replica_index
            for replica in config.replicas
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["replica_index"])
    for replica, result in zip(config.replicas, results):
        if (
            result.get("schema") != "amg_swebench_shared_pool_worker_v2"
            or result.get("status") != "PASS"
            or result.get("replica_index") != replica.replica_index
            or result.get("completed_tasks") != len(replica.task_indices)
            or result.get("total_tasks") != len(replica.task_indices)
            or result.get("task_slots_per_replica")
            != TASK_SLOTS_PER_REPLICA
            or result.get("completed_task_indices")
            != list(replica.task_indices)
        ):
            raise RuntimeError("shared-pool worker completion drifted")

    audits = []
    audit_drivers = []
    try:
        for replica in config.replicas:
            driver = driver_from_config(
                replica.path, assigned_task_indices=replica.task_indices
            )
            audit_drivers.append((replica, driver))
            for slot_index in range(TASK_SLOTS_PER_REPLICA):
                driver.acquire_runtime_lane(None, slot_index=slot_index)
        for replica, driver in audit_drivers:
            audit = driver.operations.final_audit()
            if (
                not isinstance(audit, Mapping)
                or audit.get("status") != "PASS"
                or audit.get("allocation_retained") is not True
                or not isinstance(audit.get("residue"), Mapping)
                or any(audit["residue"].values())
            ):
                raise RuntimeError("shared-pool final runtime audit failed")
            validate_live_pool_snapshot(
                audit.get("shared_model_pool"),
                replica,
                "final audit",
                listener_reference=validated_preflight_pool_snapshot(replica),
            )
            audits.append(
                {
                    "replica_index": replica.replica_index,
                    "receipt": dict(audit),
                    "receipt_sha256": sha256_json(audit),
                }
            )
    finally:
        for _, driver in reversed(audit_drivers):
            release_driver(driver)

    assignment_path = config.root / "control" / "assignment.json"
    gate_path = config.root / "control" / "gate.json"
    atomic_write_json(
        config.root / "control" / "workers-complete.json",
        {
            "schema": WORKERS_COMPLETE_SCHEMA,
            "status": "PASS",
            "coordinator_index_sha256": hashlib.sha256(
                config.path.read_bytes()
            ).hexdigest(),
            "assignment_sha256": hashlib.sha256(
                assignment_path.read_bytes()
            ).hexdigest(),
            "startup_barrier_sha256": barrier_sha256,
            "gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            "workers": results,
            "final_audits": audits,
        },
    )
    return results


def validated_workers_complete(config: CoordinatorConfig) -> Mapping[str, Any]:
    path = config.root / "control" / "workers-complete.json"
    value = load_atomic_object(path, "workers-complete receipt")
    expected_fields = {
        "schema",
        "status",
        "coordinator_index_sha256",
        "assignment_sha256",
        "startup_barrier_sha256",
        "gate_sha256",
        "workers",
        "final_audits",
    }
    assignment_path = config.root / "control" / "assignment.json"
    gate_path = config.root / "control" / "gate.json"
    if (
        set(value) != expected_fields
        or value.get("schema") != WORKERS_COMPLETE_SCHEMA
        or value.get("status") != "PASS"
        or value.get("coordinator_index_sha256")
        != hashlib.sha256(config.path.read_bytes()).hexdigest()
        or value.get("assignment_sha256")
        != hashlib.sha256(assignment_path.read_bytes()).hexdigest()
        or value.get("startup_barrier_sha256")
        != hashlib.sha256(
            (config.root / "control" / "preflight-all.json").read_bytes()
        ).hexdigest()
        or value.get("gate_sha256")
        != hashlib.sha256(gate_path.read_bytes()).hexdigest()
    ):
        raise RuntimeError("workers-complete receipt binding drifted")
    workers = value.get("workers")
    audits = value.get("final_audits")
    if not isinstance(workers, list) or not isinstance(audits, list):
        raise RuntimeError("workers-complete receipt lattice drifted")
    if len(workers) != 8 or len(audits) != 8:
        raise RuntimeError("workers-complete receipt denominator drifted")
    for replica, worker, audit_row in zip(config.replicas, workers, audits):
        if (
            not isinstance(worker, Mapping)
            or worker.get("schema") != "amg_swebench_shared_pool_worker_v2"
            or worker.get("status") != "PASS"
            or worker.get("replica_index") != replica.replica_index
            or worker.get("completed_tasks") != len(replica.task_indices)
            or worker.get("total_tasks") != len(replica.task_indices)
            or worker.get("task_slots_per_replica")
            != TASK_SLOTS_PER_REPLICA
            or worker.get("completed_task_indices")
            != list(replica.task_indices)
            or not isinstance(audit_row, Mapping)
            or set(audit_row) != {"replica_index", "receipt", "receipt_sha256"}
            or audit_row.get("replica_index") != replica.replica_index
            or not isinstance(audit_row.get("receipt"), Mapping)
            or audit_row.get("receipt_sha256")
            != sha256_json(audit_row["receipt"])
            or audit_row["receipt"].get("status") != "PASS"
            or audit_row["receipt"].get("allocation_retained") is not True
            or not isinstance(audit_row["receipt"].get("residue"), Mapping)
            or any(audit_row["receipt"]["residue"].values())
        ):
            raise RuntimeError("workers-complete receipt row drifted")
        validate_live_pool_snapshot(
            audit_row["receipt"].get("shared_model_pool"),
            replica,
            "final audit",
            listener_reference=validated_preflight_pool_snapshot(replica),
        )
    return value


def aggregate(config: CoordinatorConfig) -> Mapping[str, Any]:
    validated_workers_complete(config)
    rows = []
    per_arm = {arm: 0 for arm in ARMS}
    drivers = {
        replica.replica_index: driver_from_config(
            replica.path, assigned_task_indices=replica.task_indices
        )
        for replica in config.replicas
    }
    try:
        listener_references = {
            replica.replica_index: validated_preflight_pool_snapshot(replica)
            for replica in config.replicas
        }
        for task in config.assignment:
            replica = config.replicas[task["replica_index"]]
            driver = drivers[replica.replica_index]
            for arm in ARMS:
                key = CellKey(task["task_index"], arm)
                accepted = driver.store.read_accepted(key)
                outcome = driver.store.read_official_outcome(key)
                generation = accepted["attempt_generation"]
                runtime_path = (
                    replica.production.run_root
                    / "control"
                    / "cells"
                    / key.slug
                    / f"generation-{generation:08d}.json"
                )
                runtime = read_json(runtime_path)
                if (
                    not isinstance(runtime, Mapping)
                    or runtime.get("schema") != "swebench_triad_cell_runtime_v1"
                    or runtime.get("status") != "PASS"
                    or runtime.get("task_index") != task["task_index"]
                    or runtime.get("instance_id") != task["task_id"]
                    or runtime.get("arm") != arm
                    or runtime.get("generation") != generation
                    or atomic_json_bytes(runtime) != runtime_path.read_bytes()
                ):
                    raise RuntimeError("cell runtime receipt identity drifted")
                validate_live_pool_snapshot(
                    runtime.get("shared_model_pool"),
                    replica,
                    "cell runtime",
                    listener_reference=listener_references[
                        replica.replica_index
                    ],
                )
                resolved = outcome["resolved"]
                per_arm[arm] += int(resolved)
                rows.append(
                    {
                        "task_index": task["task_index"],
                        "task_id": task["task_id"],
                        "arm": arm,
                        "resolved": resolved,
                        "failure_class": outcome["failure_class"],
                        "report_sha256": outcome["report_sha256"],
                        "prediction_sha256": outcome["prediction_sha256"],
                        "attempt_generation": generation,
                        "replica_index": replica.replica_index,
                        "gpu_uuid": replica.gpu_uuid,
                        "runtime_receipt_sha256": hashlib.sha256(
                            runtime_path.read_bytes()
                        ).hexdigest(),
                    }
                )
    finally:
        for driver in drivers.values():
            release_driver(driver)
    identities = {(row["task_index"], row["arm"]) for row in rows}
    expected = {(task, arm) for task in range(500) for arm in ARMS}
    if len(rows) != 1500 or identities != expected:
        raise RuntimeError("aggregate is not the complete 1,500-cell lattice")
    results_path = config.root / "results" / "official-outcomes.jsonl"
    ensure_private_directory(results_path.parent)
    atomic_write_bytes(
        results_path, b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )
    rates = {arm: per_arm[arm] / 500.0 for arm in ARMS}
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "PASS",
        "tasks": 500,
        "cells": 1500,
        "resolved": per_arm,
        "rates": rates,
        "deltas": {
            "10-00": rates["amg_compaction_only"] - rates["native"],
            "11-10": rates["amg_memory"] - rates["amg_compaction_only"],
            "11-00": rates["amg_memory"] - rates["native"],
        },
        "results_path": str(results_path),
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
    }
    atomic_write_json(config.root / "results" / "official-summary.json", summary)
    return summary


def status(config: CoordinatorConfig) -> Mapping[str, Any]:
    replicas = []
    total_tasks = 0
    total_cells = 0
    for replica in config.replicas:
        completed = 0
        cells = 0
        driver = driver_from_config(
            replica.path, assigned_task_indices=replica.task_indices
        )
        try:
            for task_index in replica.task_indices:
                completion_path = driver.task_completion_path(task_index)
                if completion_path.exists():
                    if not driver.task_complete(task_index):
                        raise RuntimeError("task completion state is incomplete")
                    driver.load_task_completion(task_index)
                    completed += 1
                for arm in ARMS:
                    key = CellKey(task_index, arm)
                    accepted_exists = driver.store.accepted_path(key).exists()
                    outcome_exists = driver.store.outcome_path(key).exists()
                    if accepted_exists != outcome_exists:
                        raise RuntimeError("cell status state is only partially committed")
                    if outcome_exists:
                        driver.store.read_accepted(key)
                        driver.store.read_official_outcome(key)
                        cells += 1
        finally:
            release_driver(driver)
        total_tasks += completed
        total_cells += cells
        replicas.append(
            {
                "replica_index": replica.replica_index,
                "completed_tasks": completed,
                "total_tasks": len(replica.task_indices),
                "official_outcomes": cells,
            }
        )
    return {
        "schema": "amg_swebench_shared_pool_status_v1",
        "completed_tasks": total_tasks,
        "total_tasks": 500,
        "official_outcomes": total_cells,
        "total_cells": 1500,
        "replicas": replicas,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="swebench-shared-pool")
    value.add_argument(
        "command",
        choices=("preflight", "gate", "timing", "run", "status", "aggregate", "cleanup"),
    )
    value.add_argument("--index", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    config = CoordinatorConfig.load(arguments.index)
    ensure_private_directory(config.root)
    ensure_private_directory(config.root / "control")
    ensure_private_directory(config.root / "progress")
    config.write_assignment()
    if arguments.command == "preflight":
        result: Any = preflight_all(config)
    elif arguments.command == "gate":
        result = run_gate(config)
    elif arguments.command == "timing":
        result = run_timing(config)
    elif arguments.command == "run":
        result = run_full(config)
    elif arguments.command == "cleanup":
        result = cleanup_all(config)
    elif arguments.command == "aggregate":
        result = aggregate(config)
    else:
        result = status(config)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
