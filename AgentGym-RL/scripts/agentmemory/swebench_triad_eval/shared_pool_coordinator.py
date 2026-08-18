"""Eight-replica production coordinator for SWE-bench Verified triads.

This module is deployment-only.  It shards benchmark tasks by the frozen pool
hash, keeps all three treatment arms on one replica, runs the canonical task-0
gate before full work, and aggregates exactly 500 x 3 official outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paired_eval.serialization import canonical_json_bytes

from . import ARMS
from .atomic import (
    atomic_write_bytes,
    atomic_write_json,
    ensure_private_directory,
    exclusive_lock,
    read_json,
)
from .cli import driver_from_config
from .identity import verify_image_index
from .production import (
    SHARED_POOL_RUN_CONFIG_SCHEMA,
    ProductionRunConfig,
)
from .shared_pool_contract import (
    SHARED_MODEL_POOL_ASSIGNMENT,
    validate_shared_model_pool_snapshot,
)
from .state import CellKey, sha256_json

INDEX_SCHEMA = "amg_swebench_shared_pool_index_v1"
ASSIGNMENT_SCHEMA = "amg_swebench_shared_pool_assignment_v1"
SUMMARY_SCHEMA = "amg_swebench_shared_pool_official_summary_v1"
WORKERS_COMPLETE_SCHEMA = "amg_swebench_shared_pool_workers_complete_v2"
SHA256_PREFIXED_LENGTH = 71


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
                        if name not in {"pod_local_root", "server_port"}
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
            network_ports.extend(
                (
                    runtime["server_port"],
                    shared["model_port"],
                    shared["proxy_port"],
                )
            )
            unique_runtime_rows.append(
                (
                    production.run_root,
                    production.evidence_root,
                    Path(runtime["pod_local_root"]),
                    runtime["server_port"],
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
        if len(set(network_ports)) != 3 * len(productions):
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
        assignment = tuple(
            {
                "task_index": task_index,
                "task_id": task_ids[task_index],
                "replica_index": assigned_replica(task_ids[task_index]),
                "image": image_rows[task_index]["image"],
                "image_config_digest": image_rows[task_index]["image_config_digest"],
            }
            for task_index in range(500)
        )
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
    value: Any, replica: ReplicaConfig, label: str
) -> Mapping[str, Any]:
    expected = replica.production.shared_model_pool
    serving = replica.production.section("serving")
    if expected is None:
        raise RuntimeError(f"{label} shared-model pool snapshot is missing")
    value = validate_shared_model_pool_snapshot(
        value, f"{label} shared-model pool snapshot"
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

    receipts = []
    drivers = []
    try:
        for replica in config.replicas:
            driver = driver_from_config(
                replica.path, assigned_task_indices=replica.task_indices
            )
            drivers.append((replica, driver))
            driver.acquire_runtime_lane(None)
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
        "schema": "amg_swebench_shared_pool_preflight_v1",
        "status": "PASS",
        "replica_count": 8,
        "reconciliations": reconciliations,
        "shared_image_reconciliation": shared_image_reconciliation,
        "replicas": receipts,
    }
    atomic_write_json(config.root / "control" / "preflight-all.json", result)
    return receipts


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
    replica_index: int,
    coordinator_root: str,
) -> dict[str, Any]:
    if len(task_indices) != len(task_image_digests):
        raise ValueError("worker task/image lock lattice drifted")
    driver = driver_from_config(Path(config_path), assigned_task_indices=task_indices)
    progress_path = (
        Path(coordinator_root) / "progress" / f"replica-{replica_index}.json"
    )
    started = time.monotonic()
    completed = 0
    try:
        driver.ensure_driver_lease()
        driver._read_validated_preflight()
        image_locks = ensure_private_directory(
            Path(coordinator_root) / "control" / "image-leases"
        )
        for task_index, image_digest in zip(task_indices, task_image_digests):
            if len(
                image_digest
            ) != SHA256_PREFIXED_LENGTH or not image_digest.startswith("sha256:"):
                raise ValueError("worker image lock digest drifted")
            with exclusive_lock(image_locks / f"{image_digest[7:]}.lock"):
                driver.run_task(task_index, gate=task_index == 0)
            completed += 1
            atomic_write_json(
                progress_path,
                {
                    "schema": "amg_swebench_shared_pool_progress_v1",
                    "status": "RUNNING",
                    "replica_index": replica_index,
                    "completed_tasks": completed,
                    "total_tasks": len(task_indices),
                    "last_task_index": task_index,
                    "last_image_config_digest": image_digest,
                    "wall_seconds": round(time.monotonic() - started, 6),
                },
            )
        result = {
            "schema": "amg_swebench_shared_pool_worker_v1",
            "status": "PASS",
            "replica_index": replica_index,
            "completed_tasks": completed,
            "total_tasks": len(task_indices),
            "wall_seconds": round(time.monotonic() - started, 6),
        }
        atomic_write_json(progress_path, result)
        return result
    finally:
        release_driver(driver)


def run_full(config: CoordinatorConfig) -> list[dict[str, Any]]:
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
    gate_driver = driver_from_config(
        task_zero.path, assigned_task_indices=(0,)
    )
    try:
        if not gate_driver.gate_path.is_file():
            raise RuntimeError("full run requires the canonical task-0 gate")
        canonical_gate = gate_driver.gate(auto_run_full=False)
    finally:
        release_driver(gate_driver)
    if gate["gate"] != canonical_gate:
        raise RuntimeError("full run requires the canonical task-0 gate")
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
                replica.replica_index,
                str(config.root),
            ): replica.replica_index
            for replica in config.replicas
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["replica_index"])
    for replica, result in zip(config.replicas, results):
        if (
            result.get("schema") != "amg_swebench_shared_pool_worker_v1"
            or result.get("status") != "PASS"
            or result.get("replica_index") != replica.replica_index
            or result.get("completed_tasks") != len(replica.task_indices)
            or result.get("total_tasks") != len(replica.task_indices)
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
            driver.acquire_runtime_lane(None)
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
            "gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            "workers": results,
            "final_audits": audits,
        },
    )
    return results


def validated_workers_complete(config: CoordinatorConfig) -> Mapping[str, Any]:
    path = config.root / "control" / "workers-complete.json"
    value = load_canonical_object(path, "workers-complete receipt")
    expected_fields = {
        "schema",
        "status",
        "coordinator_index_sha256",
        "assignment_sha256",
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
            or worker.get("schema") != "amg_swebench_shared_pool_worker_v1"
            or worker.get("status") != "PASS"
            or worker.get("replica_index") != replica.replica_index
            or worker.get("completed_tasks") != len(replica.task_indices)
            or worker.get("total_tasks") != len(replica.task_indices)
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
                    or canonical_json_bytes(runtime) != runtime_path.read_bytes()
                ):
                    raise RuntimeError("cell runtime receipt identity drifted")
                validate_live_pool_snapshot(
                    runtime.get("shared_model_pool"), replica, "cell runtime"
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
        choices=("preflight", "gate", "run", "status", "aggregate", "cleanup"),
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
        preflight_all(config)
        result = run_gate(config)
    elif arguments.command == "run":
        preflight_all(config)
        run_gate(config)
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
