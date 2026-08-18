from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from paired_eval.serialization import canonical_json_bytes
from swebench_triad_eval.model_transport import scheduler_request_id
from swebench_triad_eval.atomic import atomic_write_json, canonical_json_bytes as atomic_json_bytes
from swebench_triad_eval.state import (
    OwnerIdentity,
    RuntimeLaneToken,
    sha256_json,
)
from swebench_triad_eval.shared_pool_coordinator import (
    INDEX_SCHEMA,
    ETA_PROGRESS_SCHEMA,
    ETA_RECEIPT_SCHEMA,
    FULL_RUN_TIMING_SCHEMA,
    STOP_MARKER_SCHEMA,
    TIMING_BUDGET_SECONDS,
    WORKERS_COMPLETE_SCHEMA,
    TIMING_REQUIRED_METRICS,
    CoordinatorConfig,
    ReplicaConfig,
    _collect_timing_gate,
    _eta_progress_path,
    _eta_receipt_from_progress,
    _eta_receipt_path,
    _eta_trigger_reasons,
    _extract_startup_reconciliation,
    _load_cell_timing,
    _publish_eta_check,
    _reconcile_eta_history,
    _validate_eta_cadence,
    _validated_full_run_timing,
    _worker,
    aggregate,
    assigned_replica,
    cleanup_all,
    digest_lease_admission,
    image_lock_rows,
    load_atomic_object,
    load_timing_contract,
    reconcile_digest_occupants,
    preflight_all,
    reconcile_all_eight_before_workers,
    run_full,
    validated_timing_gate,
    validated_workers_complete,
    validate_live_pool_snapshot,
)
from test_swebench_triad_eval_cli import production_config

OWNER = "amg-external-eval-g-dp8-swe-0818"
MODEL_REVISION = "3" * 40
READINESS_SHA = "1" * 64
MARKER_SHA = "2" * 64
IMAGE_DIGEST = "sha256:" + "a" * 64


def startup_reconciliation(task_index: int) -> list[dict[str, object]]:
    return [
        {
            "startup": {
                "schema": "swebench_triad_startup_reconciliation_v1",
                "task_indices": [task_index],
                "reconciled_graders": [],
                "evicted_images": [],
                "removed_task_roots": [],
                "foreign_staged_tasks": [],
                "foreign_loaded_images": [],
                "residue": {},
                "slots": [
                    {
                        "slot_index": slot_index,
                        "server_port": 18100 + task_index * 2 + slot_index,
                        "lane_generation": 1,
                    }
                    for slot_index in range(2)
                ],
            }
        }
    ]


def make_shared_coordinator(root: Path) -> Path:
    template_root = root / "template"
    template_root.mkdir()
    _, template = production_config(template_root)
    rows = []
    for replica in range(8):
        replica_root = root / f"replica-{replica}"
        config = copy.deepcopy(template)
        config["schema"] = "amg_swebench_triad_run_config_shared_pool_v3"
        config["run_root"] = str(replica_root / "run")
        config["evidence_root"] = str(replica_root / "evidence")
        config["pod"]["gpu_uuid"] = f"GPU-shared-{replica}"
        config["serving"].update(
            {
                "base_url": f"http://127.0.0.1:{16380 + replica}/v1",
                "pid_file": str(replica_root / "model.pid"),
                "pid": 300 + replica,
                "start_ticks": 3_000 + replica,
                "receipt_path": str(replica_root / "serving.json"),
            }
        )
        config["runtime"]["pod_local_root"] = str(replica_root / "pod-local")
        config["runtime"].pop("server_port")
        config["runtime"].update(
            {
                "task_slots_per_replica": 2,
                "server_ports": [18_100 + replica, 18_108 + replica],
            }
        )
        config["grader"]["output_root"] = str(replica_root / "grader")
        config["grader"].update(
            {
                "global_max_concurrency": 8,
                "semaphore_root": str(root / "grader-semaphore"),
            }
        )
        config["shared_model_pool"] = {
            "owner": OWNER,
            "readiness_path": str(root / "pool-readiness.json"),
            "readiness_sha256": READINESS_SHA,
            "marker_lease_path": str(root / "marker-lease.json"),
            "marker_lease_sha256": MARKER_SHA,
            "replica_index": replica,
            "replica_count": 8,
            "gpu_index": replica,
            "gpu_uuid": f"GPU-shared-{replica}",
            "model_id": "Qwen3.5-4B",
            "model_revision": MODEL_REVISION,
            "model_port": 18_018 + replica,
            "proxy_port": 16_380 + replica,
            "assignment_algorithm": "uint64_be(sha256(task_id)[:8]) % 8",
            "cleanup_policy": "retain_external_pool",
        }
        config_path = root / f"run-config-{replica}.json"
        payload = canonical_json_bytes(config)
        config_path.write_bytes(payload)
        rows.append(
            {
                "replica_index": replica,
                "gpu_uuid": f"GPU-shared-{replica}",
                "config_path": str(config_path),
                "config_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    index = root / "coordinator.json"
    index.write_bytes(
        canonical_json_bytes(
            {
                "schema": INDEX_SCHEMA,
                "root": str(root / "coordinator-root"),
                "replicas": rows,
            }
        )
    )
    return index


def fake_image_rows(_production, task_ids):
    return tuple(
        {
            "task_index": task_index,
            "task_id": task_ids[task_index],
            "image": f"swebench/image-{task_index}:latest",
            "image_config_digest": "sha256:" + f"{task_index:064x}",
        }
        for task_index in range(500)
    )


def write_startup_barrier(
    root: Path, config: CoordinatorConfig | None = None
) -> str:
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "amg_swebench_shared_pool_startup_barrier_v1",
        "status": "PASS",
        "replica_count": 8,
        "task_slots_per_replica": 2,
        "all_slots_held_during_reconciliation": True,
        "startup_reconciliation_complete": True,
        "coordinator_index_sha256": (
            hashlib.sha256(config.path.read_bytes()).hexdigest()
            if config is not None
            else "a" * 64
        ),
        "replica_config_sha256s": (
            [
                hashlib.sha256(replica.path.read_bytes()).hexdigest()
                for replica in config.replicas
            ]
            if config is not None
            else [f"{index:064x}" for index in range(8)]
        ),
        "reconciliations": [{"replica_index": index} for index in range(8)],
        "shared_image_reconciliation": {
            "status": "PASS",
            "remaining_images": 0,
        },
        "digest_lease_reconciliation": {
            "schema": "amg_swebench_image_digest_reconciliation_v1",
            "status": "PASS",
            "all_eight_replica_lanes_held": True,
            "stale_occupants": 0,
            "cleared": [],
        },
        "replicas": [{"replica_index": index} for index in range(8)],
    }
    path = control / "preflight-all.json"
    path.write_bytes(atomic_json_bytes(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_timing_contract(config: CoordinatorConfig) -> dict[str, object]:
    panel = []
    for replica in config.replicas:
        for slot_index in range(2):
            row = next(
                config.assignment[task_index]
                for task_index in replica.task_indices
                if task_index != 0
                and config.assignment[task_index]["slot_index"] == slot_index
            )
            panel.append(
                {
                    name: row[name]
                    for name in (
                        "replica_index",
                        "slot_index",
                        "task_index",
                        "task_id",
                        "image_config_digest",
                    )
                }
            )
    production = config.replicas[0].production
    source = production.section("source")
    contract = {
        "schema": "amg_swebench_c2_timing_contract_v1",
        "status": "FROZEN",
        "budget_seconds": TIMING_BUDGET_SECONDS,
        "task_slots_per_replica": 2,
        "panel_selection": (
            "per_replica_slotwise_deterministic_spread_distinct_digest_v1"
        ),
        "panel_tasks": panel,
        "required_metrics": list(TIMING_REQUIRED_METRICS),
        "projection": {
            "formula": (
                "max(panel_replica_makespan*ceil(shard_tasks/2))*"
                "max(1.10,p95_task/median_task)"
            ),
            "straggler_percentile": 0.95,
            "straggler_margin_floor": 1.10,
            "full_task_count": 500,
            "full_cell_count": 1500,
        },
        "bindings": {
            "coordinator_index_sha256": hashlib.sha256(
                config.path.read_bytes()
            ).hexdigest(),
            "replica_config_sha256s": [
                hashlib.sha256(replica.path.read_bytes()).hexdigest()
                for replica in config.replicas
            ],
            "manifest_sha256": production.payload["manifest_sha256"],
            "deployment_commit": source["deployment_commit"],
            "deployment_tree": "8" * 40,
            "inner_commit": source["inner_commit"],
            "assignment_algorithm": "uint64_be(sha256(task_id)[:8]) % 8",
        },
    }
    path = config.root / "control" / "timing-contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(contract))
    return contract


def write_timing_gate(config: CoordinatorConfig, contract: dict[str, object]) -> Path:
    assignment_path = config.root / "control" / "assignment.json"
    barrier_path = config.root / "control" / "preflight-all.json"
    gate_path = config.root / "control" / "gate.json"
    if not assignment_path.exists():
        config.write_assignment()
    if not barrier_path.exists():
        write_startup_barrier(config.root, config)
    if not gate_path.exists():
        atomic_write_json(gate_path, {"schema": "fixture", "status": "PASS"})
    task_seconds = [100.0 + index for index in range(16)]
    ordered = sorted(task_seconds)
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p95 = ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    median = (ordered[7] + ordered[8]) / 2
    margin = max(1.10, p95 / median)
    replica_rows = [
        {
            "replica_index": replica,
            "panel_tasks": [replica * 2, replica * 2 + 1],
            "panel_makespan_seconds": 120.0 + replica,
            "overlap_seconds": 50.0,
            "full_shard_tasks": len(config.replicas[replica].task_indices),
            "projected_waves": 32,
        }
        for replica in range(8)
    ]
    without_margin = max(
        row["panel_makespan_seconds"] * row["projected_waves"]
        for row in replica_rows
    )
    projected = without_margin * margin
    summaries = {}
    for name in TIMING_REQUIRED_METRICS:
        count = 48 if name == "per_cell_wall" else 8 if name == "replica_makespan" else 16
        summaries[name] = {
            "count": count,
            "p50_seconds": 1.0,
            "p95_seconds": 2.0,
            "max_seconds": 3.0,
        }
    payload = {
        "schema": "amg_swebench_c2_timing_gate_v1",
        "status": "PASS",
        "budget_seconds": TIMING_BUDGET_SECONDS,
        "panel_task_count": 16,
        "panel_cell_count": 48,
        "task_slots_per_replica": 2,
        "metrics": summaries,
        "tasks": [
            {"task_index": index, "task_wall_seconds": seconds}
            for index, seconds in enumerate(task_seconds)
        ],
        "cells": [{"cell": index} for index in range(48)],
        "replicas": replica_rows,
        "projection": {
            "formula": contract["projection"]["formula"],
            "task_p95_seconds": p95,
            "task_median_seconds": median,
            "straggler_margin": margin,
            "projected_without_margin_seconds": without_margin,
            "projected_full_makespan_seconds": projected,
            "within_budget": True,
        },
        "bindings": {
            "timing_contract_sha256": hashlib.sha256(
                (config.root / "control" / "timing-contract.json").read_bytes()
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
    path = config.root / "control" / "timing-gate.json"
    atomic_write_json(path, payload)
    return path


class CoordinatorConfigTest(unittest.TestCase):
    def test_assignment_is_deterministic_complete_and_runtime_is_disjoint(self):
        with tempfile.TemporaryDirectory() as raw, patch(
            "swebench_triad_eval.shared_pool_coordinator.image_lock_rows",
            side_effect=fake_image_rows,
        ):
            config = CoordinatorConfig.load(make_shared_coordinator(Path(raw)))
        self.assertEqual(len(config.assignment), 500)
        self.assertEqual(
            sorted(
                task for replica in config.replicas for task in replica.task_indices
            ),
            list(range(500)),
        )
        self.assertEqual(
            len({replica.production.run_root for replica in config.replicas}), 8
        )
        for row in config.assignment:
            self.assertEqual(row["replica_index"], assigned_replica(row["task_id"]))
            self.assertTrue(row["image_config_digest"].startswith("sha256:"))
        self.assertTrue(all(replica.task_indices for replica in config.replicas))
        for replica in config.replicas:
            rows = [
                config.assignment[task_index]
                for task_index in replica.task_indices
            ]
            self.assertEqual(
                [row["slot_index"] for row in rows],
                [position % 2 for position in range(len(rows))],
            )
            self.assertEqual(replica.production.task_slots_per_replica, 2)
        all_ports = [
            port
            for replica in config.replicas
            for port in (
                *replica.production.server_ports,
                replica.production.shared_model_pool["model_port"],
                replica.production.shared_model_pool["proxy_port"],
            )
        ]
        self.assertEqual(len(all_ports), len(set(all_ports)))

    def test_common_runtime_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            index = make_shared_coordinator(root)
            value = json.loads(index.read_text())
            config_path = Path(value["replicas"][4]["config_path"])
            payload = json.loads(config_path.read_text())
            payload["runtime"]["model_timeout_seconds"] += 1
            encoded = canonical_json_bytes(payload)
            config_path.write_bytes(encoded)
            value["replicas"][4]["config_sha256"] = hashlib.sha256(encoded).hexdigest()
            index.write_bytes(canonical_json_bytes(value))
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.image_lock_rows",
                side_effect=fake_image_rows,
            ), self.assertRaisesRegex(ValueError, "one frozen runtime"):
                CoordinatorConfig.load(index)

    def test_replica_local_collision_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            index = make_shared_coordinator(root)
            value = json.loads(index.read_text())
            paths = [Path(row["config_path"]) for row in value["replicas"]]
            first = json.loads(paths[0].read_text())
            second = json.loads(paths[1].read_text())
            second["runtime"]["server_ports"][0] = first["runtime"][
                "server_ports"
            ][0]
            encoded = canonical_json_bytes(second)
            paths[1].write_bytes(encoded)
            value["replicas"][1]["config_sha256"] = hashlib.sha256(encoded).hexdigest()
            index.write_bytes(canonical_json_bytes(value))
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.image_lock_rows",
                side_effect=fake_image_rows,
            ), self.assertRaisesRegex(ValueError, "globally unique"):
                CoordinatorConfig.load(index)

    def test_cross_category_port_collision_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            index = make_shared_coordinator(root)
            value = json.loads(index.read_text())
            paths = [Path(row["config_path"]) for row in value["replicas"]]
            first = json.loads(paths[0].read_text())
            second = json.loads(paths[1].read_text())
            second["runtime"]["server_ports"][0] = first[
                "shared_model_pool"
            ]["model_port"]
            encoded = canonical_json_bytes(second)
            paths[1].write_bytes(encoded)
            value["replicas"][1]["config_sha256"] = hashlib.sha256(
                encoded
            ).hexdigest()
            index.write_bytes(canonical_json_bytes(value))
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.image_lock_rows",
                side_effect=fake_image_rows,
            ), self.assertRaisesRegex(ValueError, "globally unique"):
                CoordinatorConfig.load(index)

    def test_config_digest_drift_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as raw:
            index = make_shared_coordinator(Path(raw))
            value = json.loads(index.read_text())
            value["replicas"][0]["config_sha256"] = "f" * 64
            index.write_bytes(canonical_json_bytes(value))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                CoordinatorConfig.load(index)


class TimingGateContractTest(unittest.TestCase):
    def load_config(self, root: Path) -> CoordinatorConfig:
        with patch(
            "swebench_triad_eval.shared_pool_coordinator.image_lock_rows",
            side_effect=fake_image_rows,
        ):
            return CoordinatorConfig.load(make_shared_coordinator(root))

    def test_frozen_panel_has_two_distinct_slots_and_digests_per_replica(self):
        with tempfile.TemporaryDirectory() as raw:
            config = self.load_config(Path(raw))
            contract = write_timing_contract(config)
            loaded = load_timing_contract(config)
        self.assertEqual(loaded, contract)
        for replica in range(8):
            rows = [
                row for row in loaded["panel_tasks"]
                if row["replica_index"] == replica
            ]
            self.assertEqual(sorted(row["slot_index"] for row in rows), [0, 1])
            self.assertEqual(len({row["image_config_digest"] for row in rows}), 2)

    def test_validated_timing_gate_recomputes_projection_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            config = self.load_config(Path(raw))
            contract = write_timing_contract(config)
            gate_path = write_timing_gate(config, contract)
            expected = json.loads(gate_path.read_text())
            with patch(
                "swebench_triad_eval.shared_pool_coordinator._collect_timing_gate",
                return_value=expected,
            ):
                accepted = validated_timing_gate(config)
            self.assertTrue(accepted["projection"]["within_budget"])
            tampered = json.loads(gate_path.read_text())
            tampered["projection"]["projected_full_makespan_seconds"] -= 1
            atomic_write_json(gate_path, tampered)
            with self.assertRaisesRegex(RuntimeError, "arithmetic"):
                validated_timing_gate(config)

    def test_validated_timing_gate_rejects_nested_evidence_substitution(self):
        with tempfile.TemporaryDirectory() as raw:
            config = self.load_config(Path(raw))
            contract = write_timing_contract(config)
            gate_path = write_timing_gate(config, contract)
            expected = json.loads(gate_path.read_text())
            substituted = copy.deepcopy(expected)
            substituted["tasks"][0]["task_index"] = 10_000
            substituted["cells"][0] = {"cell": "fabricated"}
            atomic_write_json(gate_path, substituted)
            with patch(
                "swebench_triad_eval.shared_pool_coordinator._collect_timing_gate",
                return_value=expected,
            ), self.assertRaisesRegex(RuntimeError, "evidence recomputation"):
                validated_timing_gate(config)

    def test_synthetic_16_task_panel_collects_all_required_metrics(self):
        with tempfile.TemporaryDirectory() as raw:
            config = self.load_config(Path(raw))
            contract = write_timing_contract(config)
            config.write_assignment()
            write_startup_barrier(config.root, config)
            atomic_write_json(
                config.root / "control" / "gate.json",
                {"schema": "fixture", "status": "PASS"},
            )
            fake_drivers = {}
            for replica in config.replicas:
                run_root = replica.production.run_root

                class Driver:
                    lease_registry = None

                    def __init__(self, selected_replica, selected_root):
                        self.replica = selected_replica
                        self.root = selected_root

                    def task_completion_path(self, task_index):
                        return self.root / "full" / f"task-{task_index:04d}.json"

                    def task_publication_path(self, task_index):
                        return (
                            self.root
                            / "full"
                            / f"task-{task_index:04d}.publication.json"
                        )

                    def load_task_completion(self, task_index):
                        timing = self.root / "timings" / f"task-{task_index:04d}.json"
                        return {
                            "timing_receipt": {
                                "status": "READY_FOR_PUBLICATION",
                                "path": str(timing),
                                "sha256": hashlib.sha256(
                                    timing.read_bytes()
                                ).hexdigest(),
                            }
                        }

                driver = Driver(replica, run_root)
                fake_drivers[str(replica.path)] = driver
                selected = [
                    row
                    for row in contract["panel_tasks"]
                    if row["replica_index"] == replica.replica_index
                ]
                for offset, panel in enumerate(selected):
                    task_index = panel["task_index"]
                    start = (
                        replica.replica_index * 1_000_000_000_000
                        + offset * 10_000_000_000
                    )
                    end = start + 100_000_000_000
                    phases = []
                    cursor = start
                    for name, duration in (
                        ("task_slot_queue", 1_000_000_000),
                        ("runtime_lane_wait", 100_000_000),
                        ("image_digest_wait", 2_000_000_000),
                        ("oci_stage", 3_000_000_000),
                        ("official_grade_native", 4_000_000_000),
                        ("official_grade_amg_compaction_only", 4_000_000_000),
                        ("official_grade_amg_memory", 4_000_000_000),
                    ):
                        phases.append(
                            {
                                "phase": name,
                                "status": "PASS",
                                "started_wall_ns": cursor,
                                "ended_wall_ns": cursor + duration,
                                "started_monotonic_ns": cursor,
                                "ended_monotonic_ns": cursor + duration,
                                "duration_ns": duration,
                            }
                        )
                        cursor += duration
                    timing = run_root / "timings" / f"task-{task_index:04d}.json"
                    atomic_write_json(
                        timing,
                        {
                            "schema": "swebench_triad_task_phase_timing_v1",
                            "status": "READY_FOR_PUBLICATION",
                            "task_index": task_index,
                            "task_id": panel["task_id"],
                            "task_seed": next(
                                config.task.seed
                                for config in replica.production.configs
                                if config.task.task_index == task_index
                            ),
                            "slot_index": panel["slot_index"],
                            "server_port": replica.production.section("runtime")[
                                "server_ports"
                            ][panel["slot_index"]],
                            "lane_generation": 1,
                            "lane_fencing_token_sha256": "f" * 64,
                            "started_wall_ns": start,
                            "ended_wall_ns": end,
                            "started_monotonic_ns": start,
                            "ended_monotonic_ns": end,
                            "duration_ns": end - start,
                            "identity": {
                                "deployment_commit": "d" * 40,
                                "inner_commit": "e" * 40,
                                "source_identity_sha256": "a" * 64,
                                "run_config_sha256": "b" * 64,
                                "manifest_sha256": "c" * 64,
                                "replica_index": replica.replica_index,
                                "gpu_uuid": replica.gpu_uuid,
                            },
                            "phases": phases,
                            "phase_durations_are_non_additive": True,
                        },
                    )
                    completion = driver.task_completion_path(task_index)
                    atomic_write_json(completion, {"task_index": task_index})
                    atomic_write_json(
                        driver.task_publication_path(task_index),
                        {
                            "schema": "swebench_triad_task_publication_timing_v2",
                            "status": "PASS",
                            "recovered_after_crash": False,
                            "task_index": task_index,
                            "completion_path": str(completion),
                            "completion_sha256": hashlib.sha256(
                                completion.read_bytes()
                            ).hexdigest(),
                            "timing_receipt_sha256": hashlib.sha256(
                                timing.read_bytes()
                            ).hexdigest(),
                            "started_wall_ns": end,
                            "ended_wall_ns": end + 100_000_000,
                            "started_monotonic_ns": end,
                            "ended_monotonic_ns": end + 100_000_000,
                            "duration_ns": 100_000_000,
                        },
                    )

            def cell_timing(_replica, _driver, task_index, arm, **_identity):
                return {
                    "task_index": task_index,
                    "task_id": config.assignment[task_index]["task_id"],
                    "arm": arm,
                    "generation": 1,
                    "replica_index": config.assignment[task_index]["replica_index"],
                    "gpu_uuid": config.replicas[
                        config.assignment[task_index]["replica_index"]
                    ].gpu_uuid,
                    "slot_index": config.assignment[task_index]["slot_index"],
                    "server_port": 65_000,
                    "lane_generation": 1,
                    "lane_fencing_token_sha256": "f" * 64,
                    "shared_model_pool_sha256": "c" * 64,
                    "model_generation_seconds": 5.0,
                    "environment_tool_execution_seconds": 6.0,
                    "publication_seconds": 0.01,
                    "cell_wall_seconds": 12.0,
                    "runtime_receipt_sha256": "a" * 64,
                    "publication_receipt_sha256": "b" * 64,
                }

            with (
                patch(
                    "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
                    side_effect=lambda path, **_kwargs: fake_drivers[str(path)],
                ),
                patch(
                    "swebench_triad_eval.shared_pool_coordinator.validated_preflight_pool_snapshot",
                    side_effect=lambda replica: {
                        "replica_index": replica.replica_index
                    },
                ),
                patch(
                    "swebench_triad_eval.shared_pool_coordinator._expected_task_timing_identity",
                    side_effect=lambda replica: {
                        "deployment_commit": "d" * 40,
                        "inner_commit": "e" * 40,
                        "source_identity_sha256": "a" * 64,
                        "run_config_sha256": "b" * 64,
                        "manifest_sha256": "c" * 64,
                        "replica_index": replica.replica_index,
                        "gpu_uuid": replica.gpu_uuid,
                    },
                ),
                patch(
                    "swebench_triad_eval.shared_pool_coordinator._load_cell_timing",
                    side_effect=cell_timing,
                ),
            ):
                receipt = _collect_timing_gate(config, contract)
                validated = validated_timing_gate(config)
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(validated, receipt)
            self.assertEqual(receipt["panel_task_count"], 16)
            self.assertEqual(receipt["panel_cell_count"], 48)
            self.assertEqual(set(receipt["metrics"]), set(TIMING_REQUIRED_METRICS))
            self.assertTrue(
                all(
                    row["panel_makespan_seconds"] == 110.1
                    for row in receipt["replicas"]
                )
            )
            self.assertTrue(
                all(row["task_wall_seconds"] == 100.1 for row in receipt["tasks"])
            )
            self.assertLessEqual(
                receipt["projection"]["projected_full_makespan_seconds"],
                TIMING_BUDGET_SECONDS,
            )

    def test_over_budget_timing_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            config = self.load_config(Path(raw))
            contract = write_timing_contract(config)
            gate_path = write_timing_gate(config, contract)
            over = json.loads(gate_path.read_text())
            scale = 10.0
            for row in over["replicas"]:
                row["panel_makespan_seconds"] *= scale
            over["projection"]["projected_without_margin_seconds"] *= scale
            over["projection"]["projected_full_makespan_seconds"] *= scale
            over["projection"]["within_budget"] = False
            over["status"] = "FAIL_CLOSED"
            atomic_write_json(gate_path, over)
            with self.assertRaisesRegex(RuntimeError, "incomplete or over budget"):
                validated_timing_gate(config)


class ExactTimingReceiptIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runtime_path = (
            self.root / "control/cells/0007-native/generation-00000001.json"
        )
        self.runtime_path.parent.mkdir(parents=True)
        policy = {
            "phase": "policy_and_model_execution",
            "status": "PASS",
            "started_wall_ns": 100,
            "ended_wall_ns": 300,
            "started_monotonic_ns": 100,
            "ended_monotonic_ns": 300,
            "duration_ns": 200,
        }
        run_id = "amg-sbv-0007-native-g00000001"
        self.runtime = {
            "schema": "swebench_triad_cell_runtime_v1",
            "status": "PASS",
            "task_index": 7,
            "instance_id": "owner__task-0007",
            "arm": "native",
            "generation": 1,
            "container_name": "fixture-container",
            "run_id": run_id,
            "run_capability_sha256": "a" * 64,
            "slot_index": 1,
            "server_port": 18101,
            "lane_generation": 3,
            "lane_fencing_token_sha256": "b" * 64,
            "phase_timings": [policy],
            "shared_model_pool": {"fixture": True},
            "rootfs_before": {},
            "cgroup_prepare": {},
            "container_id": "c" * 12,
            "cgroup_descendants_before": {},
            "metadata_before": {},
            "model_transport_events": [
                {
                    "phase": "tokenize",
                    "semantic_request_sha256": "d" * 64,
                    "prompt_token_ids": [1],
                    "started_wall_ns": 110,
                    "ended_wall_ns": 120,
                    "started_monotonic_ns": 110,
                    "ended_monotonic_ns": 120,
                    "duration_ns": 10,
                },
                {
                    "phase": "chat_completion",
                    "request_id": scheduler_request_id(
                        run_id=run_id,
                        task_index=7,
                        arm="native",
                        generation=1,
                        turn_index=0,
                    ),
                    "turn_index": 0,
                    "semantic_request_sha256": "e" * 64,
                    "prompt_token_ids": [1],
                    "response_token_ids": [2],
                    "started_wall_ns": 130,
                    "ended_wall_ns": 150,
                    "started_monotonic_ns": 130,
                    "ended_monotonic_ns": 150,
                    "duration_ns": 20,
                },
            ],
            "metadata_after": {},
            "cgroup_descendants_after": {},
            "rootfs_after": {},
            "container_logs": {},
            "container_cleanup": {},
            "cgroup_teardown": {},
        }
        atomic_write_json(self.runtime_path, self.runtime)
        self.publication_path = self.runtime_path.with_name(
            self.runtime_path.stem + ".publication.json"
        )
        self.publication = {
            "schema": "swebench_triad_cell_publication_timing_v1",
            "status": "PASS",
            "cell_status": "PASS",
            "task_index": 7,
            "arm": "native",
            "generation": 1,
            "runtime_receipt_path": str(self.runtime_path),
            "runtime_receipt_sha256": hashlib.sha256(
                self.runtime_path.read_bytes()
            ).hexdigest(),
            "started_wall_ns": 300,
            "ended_wall_ns": 320,
            "started_monotonic_ns": 300,
            "ended_monotonic_ns": 320,
            "duration_ns": 20,
        }
        atomic_write_json(self.publication_path, self.publication)
        self.replica = SimpleNamespace(
            replica_index=0,
            gpu_uuid="GPU-0",
            production=SimpleNamespace(run_root=self.root),
        )
        self.driver = SimpleNamespace(
            store=SimpleNamespace(read_accepted=lambda _key: {"attempt_generation": 1})
        )
        self.task_row = {
            "task_index": 7,
            "task_id": "owner__task-0007",
            "replica_index": 0,
            "slot_index": 1,
        }
        self.task_timing = {
            "server_port": 18101,
            "lane_generation": 3,
            "lane_fencing_token_sha256": "b" * 64,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def collect(self):
        with patch(
            "swebench_triad_eval.shared_pool_coordinator.validate_live_pool_snapshot"
        ):
            return _load_cell_timing(
                self.replica,
                self.driver,
                7,
                "native",
                task_row=self.task_row,
                task_timing=self.task_timing,
                listener_reference={"fixture": True},
            )

    def rewrite_runtime(self, value):
        atomic_write_json(self.runtime_path, value)
        publication = copy.deepcopy(self.publication)
        publication["runtime_receipt_sha256"] = hashlib.sha256(
            self.runtime_path.read_bytes()
        ).hexdigest()
        atomic_write_json(self.publication_path, publication)

    def test_exact_task_replica_slot_lane_and_event_identity_passes(self):
        row = self.collect()
        self.assertEqual(row["task_id"], self.task_row["task_id"])
        self.assertEqual(row["replica_index"], 0)
        self.assertEqual(row["slot_index"], 1)
        self.assertEqual(row["server_port"], 18101)
        self.assertEqual(row["lane_generation"], 3)

    def test_one_field_identity_and_timestamp_drifts_fail_closed(self):
        mutations = (
            ("instance_id", "wrong"),
            ("slot_index", 99),
            ("server_port", -1),
            ("lane_generation", 99),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.runtime)
                changed[field] = value
                self.rewrite_runtime(changed)
                with self.assertRaisesRegex(RuntimeError, "identity drifted"):
                    self.collect()
        changed = copy.deepcopy(self.runtime)
        changed["phase_timings"][0]["duration_ns"] += 1
        self.rewrite_runtime(changed)
        with self.assertRaisesRegex(RuntimeError, "timestamp arithmetic"):
            self.collect()
        changed = copy.deepcopy(self.runtime)
        changed["model_transport_events"][1]["request_id"] = "0" * 64
        self.rewrite_runtime(changed)
        with self.assertRaisesRegex(RuntimeError, "chat event identity"):
            self.collect()

    def test_publication_requires_exact_timestamp_schema(self):
        changed = copy.deepcopy(self.publication)
        changed.pop("started_wall_ns")
        atomic_write_json(self.publication_path, changed)
        with self.assertRaisesRegex(RuntimeError, "publication"):
            self.collect()


class ImageLockRowsTest(unittest.TestCase):
    def test_image_rows_bind_task_order_and_preserve_duplicate_config_digest(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            index = root / "images.jsonl"
            tasks = {i: f"owner__task-{i:04d}" for i in range(500)}
            rows = []
            for task_index in range(500):
                digest = (
                    IMAGE_DIGEST
                    if task_index in {183, 185}
                    else "sha256:" + f"{task_index:064x}"
                )
                rows.append(
                    {
                        "image": f"swebench/prefix_task-{task_index:04d}:latest",
                        "config": {"digest": digest},
                    }
                )
            index.write_text("".join(json.dumps(row) + "\n" for row in rows))
            production = SimpleNamespace(
                section=lambda name: {"image_index": str(index)}
            )
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.verify_image_index"
            ):
                result = image_lock_rows(production, tasks)
        self.assertEqual(result[183]["image_config_digest"], IMAGE_DIGEST)
        self.assertEqual(result[185]["image_config_digest"], IMAGE_DIGEST)
        self.assertEqual(len(result), 500)

    def test_image_task_misalignment_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            index = root / "images.jsonl"
            rows = [
                {
                    "image": f"swebench/wrong-{task_index:04d}:latest",
                    "config": {"digest": "sha256:" + f"{task_index:064x}"},
                }
                for task_index in range(500)
            ]
            index.write_text("".join(json.dumps(row) + "\n" for row in rows))
            production = SimpleNamespace(
                section=lambda name: {"image_index": str(index)}
            )
            tasks = {i: f"owner__task-{i:04d}" for i in range(500)}
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.verify_image_index"
            ), self.assertRaisesRegex(ValueError, "task identity"):
                image_lock_rows(production, tasks)


class AtomicReceiptContractTest(unittest.TestCase):
    def test_atomic_writer_round_trips_through_atomic_loader(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "receipt.json"
            payload = {"schema": "fixture_v1", "status": "PASS"}
            atomic_write_json(path, payload)
            self.assertEqual(path.read_bytes(), atomic_json_bytes(payload))
            self.assertEqual(load_atomic_object(path, "fixture"), payload)

    def test_atomic_loader_rejects_no_lf_external_canonical_bytes(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "receipt.json"
            path.write_bytes(canonical_json_bytes({"status": "PASS"}))
            with self.assertRaisesRegex(RuntimeError, "canonical atomic"):
                load_atomic_object(path, "fixture")


def fake_runtime_slot(task_index: int, slot_index: int) -> RuntimeLaneToken:
    return RuntimeLaneToken(
        driver_key="d" * 64,
        lease_id="e" * 64,
        owner=OwnerIdentity("host", "boot", 123, 456),
        task_index=task_index,
        slot_index=slot_index,
        server_port=18100 + slot_index,
        generation=task_index + 1,
        fencing_token="f" * 64,
    )


class DurableDigestLeaseTest(unittest.TestCase):
    def admission(self, root: Path, task_index: int = 183):
        return digest_lease_admission(
            coordinator_root=root,
            image_digest=IMAGE_DIGEST,
            task_index=task_index,
            replica_index=4 if task_index == 183 else 7,
            startup_barrier_sha256="b" * 64,
            slot=fake_runtime_slot(task_index, 0),
        )

    def test_failed_owner_leaves_occupant_and_waiter_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(RuntimeError, "owner failed"):
                with self.admission(root):
                    raise RuntimeError("owner failed")
            occupants = list(
                (root / "control" / "image-leases").glob("*.occupant.json")
            )
            self.assertEqual(len(occupants), 1)
            with self.assertRaisesRegex(RuntimeError, "all-eight reconciliation"):
                with self.admission(root, 185):
                    self.fail("stale digest occupant admitted a successor")
            receipt = reconcile_digest_occupants(root)
            self.assertEqual(receipt["stale_occupants"], 1)
            self.assertFalse(occupants[0].exists())
            with self.admission(root, 185):
                pass

    def test_cleanup_failure_preserves_occupant(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            original_unlink = Path.unlink

            def fail_occupant(path, *args, **kwargs):
                if path.name.endswith(".occupant.json"):
                    raise OSError("injected occupant cleanup failure")
                return original_unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", fail_occupant), self.assertRaisesRegex(
                OSError, "cleanup failure"
            ):
                with self.admission(root):
                    pass
            self.assertEqual(
                len(list((root / "control" / "image-leases").glob("*.occupant.json"))),
                1,
            )

    def test_sigkill_releases_flock_but_not_durable_occupant(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ready = root / "ready"
            code = r"""
import pathlib
import time
from swebench_triad_eval.shared_pool_coordinator import digest_lease_admission
from swebench_triad_eval.state import OwnerIdentity, RuntimeLaneToken
root = pathlib.Path(__import__('sys').argv[1])
ready = pathlib.Path(__import__('sys').argv[2])
slot = RuntimeLaneToken(
    driver_key='d' * 64, lease_id='e' * 64,
    owner=OwnerIdentity('host', 'boot', 123, 456),
    task_index=183, slot_index=0, server_port=18100,
    generation=184, fencing_token='f' * 64,
)
with digest_lease_admission(
    coordinator_root=root, image_digest='sha256:' + 'a' * 64,
    task_index=183, replica_index=4,
    startup_barrier_sha256='b' * 64, slot=slot,
):
    ready.write_text('ready')
    while True:
        time.sleep(1)
"""
            process = subprocess.Popen([sys.executable, "-c", code, str(root), str(ready)])
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and process.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("SIGKILL fixture did not acquire the digest lease")
                    time.sleep(0.01)
                os.kill(process.pid, signal.SIGKILL)
                self.assertEqual(process.wait(timeout=5), -signal.SIGKILL)
                with self.assertRaisesRegex(RuntimeError, "all-eight reconciliation"):
                    with self.admission(root, 185):
                        self.fail("post-SIGKILL waiter was admitted")
                receipt = reconcile_digest_occupants(root)
                self.assertEqual(receipt["stale_occupants"], 1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)


class WorkerTest(unittest.TestCase):
    def test_worker_progress_is_monotonic_over_prior_durable_completions(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            completion_root = root / "replica-run" / "full"
            completion_root.mkdir(parents=True)
            (completion_root / "task-0000.json").write_text("{}")
            loaded = []

            class Driver:
                lease_registry = None

                @staticmethod
                def ensure_driver_lease():
                    return None

                @staticmethod
                def _read_validated_preflight():
                    return None

                @staticmethod
                def task_completion_path(task_index):
                    return completion_root / f"task-{task_index:04d}.json"

                @staticmethod
                def load_task_completion(task_index):
                    loaded.append(task_index)
                    return {"task_index": task_index}

                @staticmethod
                def run_task(task_index, *, slot_index, admission, **_kwargs):
                    with admission(fake_runtime_slot(task_index, slot_index)):
                        return {"task_index": task_index}

            barrier_sha256 = write_startup_barrier(root)
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
                return_value=Driver(),
            ):
                result = _worker(
                    "/tmp/config.json",
                    (4, 9),
                    ("sha256:" + "4" * 64, "sha256:" + "9" * 64),
                    (0, 1),
                    3,
                    raw,
                    barrier_sha256,
                    (0, 4, 9),
                )
            progress = load_atomic_object(
                root / "progress" / "replica-3.json", "worker progress"
            )
            self.assertEqual(loaded, [0])
            self.assertEqual(result["completed_task_indices"], [4, 9])
            self.assertEqual(progress["completed_task_indices"], [0, 4, 9])
            self.assertEqual(progress["total_tasks"], 3)

    def test_worker_uses_digest_lock_and_does_not_repeat_startup_reconciliation(self):
        calls = []

        class Driver:
            lease_registry = None

            @staticmethod
            def task_completion_path(task_index):
                return Path(f"/nonexistent-amg-worker-fixture-{task_index}")

            @staticmethod
            def load_task_completion(_task_index):
                raise AssertionError("nonexistent completion was loaded")

            def ensure_driver_lease(self):
                calls.append("lease")

            def _read_validated_preflight(self):
                calls.append("preflight")

            def reconcile_dead_work(self):
                raise AssertionError("coordinator preflight owns reconciliation")

            def run_task(self, task_index, *, gate, slot_index, admission, **_timing):
                with admission(fake_runtime_slot(task_index, slot_index)):
                    calls.append((task_index, gate, slot_index))

        with tempfile.TemporaryDirectory() as raw, patch(
            "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
            return_value=Driver(),
        ):
            barrier_sha256 = write_startup_barrier(Path(raw))
            result = _worker(
                "/tmp/config.json",
                (4, 9),
                (IMAGE_DIGEST, IMAGE_DIGEST),
                (0, 1),
                3,
                raw,
                barrier_sha256,
            )
            lock_root = Path(raw) / "control" / "image-leases"
            self.assertEqual(len(list(lock_root.glob("*.lock"))), 1)
        self.assertEqual(calls[:2], ["lease", "preflight"])
        self.assertEqual(
            sorted(calls[2:]), [(4, False, 0), (9, False, 1)]
        )
        self.assertEqual(result["completed_tasks"], 2)

    def test_worker_stops_before_the_next_task_at_publication_boundary(self):
        calls = []

        class Driver:
            lease_registry = None

            @staticmethod
            def task_completion_path(task_index):
                return Path(f"/nonexistent-amg-worker-stop-{task_index}")

            @staticmethod
            def load_task_completion(_task_index):
                raise AssertionError("nonexistent completion was loaded")

            @staticmethod
            def ensure_driver_lease():
                return None

            @staticmethod
            def _read_validated_preflight():
                return None

            @staticmethod
            def run_task(*_args, **_kwargs):
                calls.append("run")

        with tempfile.TemporaryDirectory() as raw, patch(
            "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
            return_value=Driver(),
        ):
            root = Path(raw)
            barrier_sha256 = write_startup_barrier(root)
            atomic_write_json(
                root / "control" / "stop-after-publication.json",
                {
                    "schema": STOP_MARKER_SCHEMA,
                    "status": "STOP_AT_PUBLICATION_BOUNDARY",
                    "reason": "two_consecutive_eta_checks_above_1_5x_budget",
                    "consecutive_over_budget_checks": 2,
                    "latest_eta_receipt_sha256": "f" * 64,
                },
            )
            result = _worker(
                "/tmp/config.json",
                (4, 9),
                (IMAGE_DIGEST, IMAGE_DIGEST),
                (0, 1),
                3,
                raw,
                barrier_sha256,
            )
        self.assertEqual(calls, [])
        self.assertEqual(result["status"], "STOPPED_AT_PUBLICATION_BOUNDARY")
        self.assertEqual(result["completed_task_indices"], [])

    def test_worker_rejects_task_image_lattice_drift(self):
        with self.assertRaisesRegex(ValueError, "lattice"):
            _worker(
                "/tmp/config.json",
                (1,),
                (),
                (0,),
                0,
                "/tmp/root",
                "a" * 64,
            )

    def test_worker_admits_at_most_two_tasks_and_never_shares_a_slot(self):
        active = 0
        maximum = 0
        active_slots = set()
        slot_maximum = {0: 0, 1: 0}
        lock = threading.Lock()
        first_wave = threading.Barrier(2)

        class Driver:
            lease_registry = None

            @staticmethod
            def task_completion_path(task_index):
                return Path(f"/nonexistent-amg-worker-c2-{task_index}")

            @staticmethod
            def load_task_completion(_task_index):
                raise AssertionError("nonexistent completion was loaded")

            @staticmethod
            def ensure_driver_lease():
                return None

            @staticmethod
            def _read_validated_preflight():
                return None

            def run_task(self, task_index, *, gate, slot_index, admission, **_timing):
                nonlocal active, maximum
                self.assertFalse(gate)
                with admission(fake_runtime_slot(task_index, slot_index)):
                    with lock:
                        if slot_index in active_slots:
                            raise AssertionError("one slot admitted two live tasks")
                        active_slots.add(slot_index)
                        active += 1
                        maximum = max(maximum, active)
                        slot_maximum[slot_index] = max(
                            slot_maximum[slot_index], 1
                        )
                    try:
                        if task_index in {4, 9}:
                            first_wave.wait(timeout=2)
                        time.sleep(0.01)
                    finally:
                        with lock:
                            active -= 1
                            active_slots.remove(slot_index)

            @staticmethod
            def assertFalse(value):
                if value:
                    raise AssertionError("unexpected gate task")

        with tempfile.TemporaryDirectory() as raw, patch(
            "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
            return_value=Driver(),
        ):
            barrier_sha256 = write_startup_barrier(Path(raw))
            result = _worker(
                "/tmp/config.json",
                (4, 9, 12, 15),
                tuple("sha256:" + f"{index:064x}" for index in range(4)),
                (0, 1, 0, 1),
                3,
                raw,
                barrier_sha256,
            )
        self.assertEqual(maximum, 2)
        self.assertEqual(slot_maximum, {0: 1, 1: 1})
        self.assertEqual(result["completed_task_indices"], [4, 9, 12, 15])
        self.assertEqual(
            [row["completed_task_indices"] for row in result["slots"]],
            [[4, 12], [9, 15]],
        )

    def test_duplicate_digest_183_185_is_whole_task_exclusive(self):
        active = 0
        maximum = 0
        lock = threading.Lock()

        class Driver:
            lease_registry = None

            @staticmethod
            def task_completion_path(task_index):
                return Path(f"/nonexistent-amg-worker-digest-{task_index}")

            @staticmethod
            def load_task_completion(_task_index):
                raise AssertionError("nonexistent completion was loaded")

            @staticmethod
            def ensure_driver_lease():
                return None

            @staticmethod
            def _read_validated_preflight():
                return None

            def run_task(self, task_index, *, gate, slot_index, admission, **_timing):
                del gate
                nonlocal active, maximum
                with admission(fake_runtime_slot(task_index, slot_index)):
                    with lock:
                        active += 1
                        maximum = max(maximum, active)
                    time.sleep(0.04)
                    with lock:
                        active -= 1

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            barrier_sha256 = write_startup_barrier(root)
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
                side_effect=lambda *_args, **_kwargs: Driver(),
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(
                            _worker,
                            f"/tmp/config-{task_index}.json",
                            (task_index,),
                            (IMAGE_DIGEST,),
                            (0,),
                            replica_index,
                            raw,
                            barrier_sha256,
                        )
                        for task_index, replica_index in ((183, 4), (185, 7))
                    ]
                    for future in futures:
                        future.result()
        self.assertEqual(maximum, 1)

    def test_worker_refuses_to_claim_before_global_startup_barrier(self):
        with tempfile.TemporaryDirectory() as raw, patch(
            "swebench_triad_eval.shared_pool_coordinator.driver_from_config"
        ) as driver:
            with self.assertRaisesRegex(RuntimeError, "startup reconciliation"):
                _worker(
                    "/tmp/config.json",
                    (183,),
                    (IMAGE_DIGEST,),
                    (0,),
                    4,
                    raw,
                    "a" * 64,
                )
            driver.assert_not_called()


class SharedPoolPreflightTest(unittest.TestCase):
    def test_complete_reconciliation_list_has_exactly_one_final_startup(self):
        startup = startup_reconciliation(0)[0]
        receipts = [
            {
                "cell": {"task_index": 0, "arm": "native"},
                "generation": 2,
                "accepted_recovered": True,
                "runtime": {"status": "PASS"},
            },
            {
                "cell": {"task_index": 0, "arm": "amg_memory"},
                "grade_claim_generation": 3,
                "grader": {"status": "PASS"},
            },
            startup,
        ]
        self.assertIs(
            _extract_startup_reconciliation(receipts, (0,)), startup["startup"]
        )
        for invalid in (
            receipts[:-1],
            [startup, startup],
            [{"unknown": True}, startup],
            [startup, receipts[0]],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                _extract_startup_reconciliation(invalid, (0,))

    def test_all_lanes_are_acquired_before_cross_replica_reconciliation(self):
        events = []
        drivers = {}

        class Registry:
            def __init__(self, index):
                self.index = index

            def release(self):
                events.append(("release", self.index))

        class Driver:
            def __init__(self, index):
                self.index = index
                self.lease_registry = Registry(index)
                self.operations = self

            def acquire_runtime_lane(self, task_index, *, slot_index):
                self.assert_task_none(task_index)
                events.append(("acquire", self.index, slot_index))

            @staticmethod
            def assert_task_none(task_index):
                if task_index is not None:
                    raise AssertionError("shared preflight must acquire a global lane")

            def reconcile_dead_work(self, *, allow_foreign_loaded_images):
                self.assertEqual_all_acquired()
                if allow_foreign_loaded_images is not True:
                    raise AssertionError("foreign images must be deferred across roots")
                events.append(("reconcile", self.index))
                return startup_reconciliation(self.index)

            @staticmethod
            def assertEqual_all_acquired():
                acquired = [row for row in events if row[0] == "acquire"]
                if len(acquired) != 16:
                    raise AssertionError("reconciliation began before all lanes")

            def reconcile_unbound_loaded_images(self):
                if self.index != 0:
                    raise AssertionError("one coordinator owner must evict orphan images")
                if len([row for row in events if row[0] == "reconcile"]) != 8:
                    raise AssertionError("orphan eviction began before all reconciliation")
                events.append(("shared-images", self.index))
                return {"status": "PASS", "remaining_images": 0}

            def preflight(self):
                if len([row for row in events if row[0] == "shared-images"]) != 1:
                    raise AssertionError("validation began before orphan reconciliation")
                events.append(("preflight", self.index))
                return {"status": "PASS", "replica_index": self.index}

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            replicas = tuple(
                ReplicaConfig(
                    replica_index=index,
                    gpu_uuid=f"GPU-{index}",
                    path=root / f"config-{index}.json",
                    production=SimpleNamespace(),
                    task_indices=(index,),
                )
                for index in range(8)
            )
            (root / "index.json").write_text("index")
            for replica in replicas:
                replica.path.write_text(f"config-{replica.replica_index}")
            config = CoordinatorConfig(root / "index.json", root, replicas, ())
            for replica in replicas:
                drivers[str(replica.path)] = Driver(replica.replica_index)
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
                side_effect=lambda path, **_kwargs: drivers[str(path)],
            ):
                receipts = preflight_all(config)
        self.assertEqual(len(receipts), 8)
        self.assertEqual([row[0] for row in events[:16]], ["acquire"] * 16)
        self.assertEqual([row[0] for row in events[16:24]], ["reconcile"] * 8)
        self.assertEqual(events[24], ("shared-images", 0))
        self.assertEqual([row[0] for row in events[25:33]], ["preflight"] * 8)

    def test_worker_retry_reconciles_all_eight_without_rewriting_preflight(self):
        events = []
        drivers = {}

        class Registry:
            def __init__(self, index):
                self.index = index

            def release(self):
                events.append(("release", self.index))

        class Driver:
            def __init__(self, index):
                self.index = index
                self.lease_registry = Registry(index)
                self.operations = self

            def acquire_runtime_lane(self, task_index, *, slot_index):
                self.assertIsNone(task_index)
                events.append(("acquire", self.index, slot_index))

            @staticmethod
            def assertIsNone(value):
                if value is not None:
                    raise AssertionError("retry reconciliation must fence whole lanes")

            def reconcile_dead_work(self, *, allow_foreign_loaded_images):
                if len([row for row in events if row[0] == "acquire"]) != 16:
                    raise AssertionError("retry reconciled before all lanes were held")
                if allow_foreign_loaded_images is not True:
                    raise AssertionError("cross-root images must be deferred")
                events.append(("reconcile", self.index))
                return startup_reconciliation(self.index)

            def reconcile_unbound_loaded_images(self):
                if len([row for row in events if row[0] == "reconcile"]) != 8:
                    raise AssertionError("shared images reconciled too early")
                events.append(("shared-images", self.index))
                return {"status": "PASS", "remaining_images": 0}

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "control").mkdir()
            index_path = root / "index.json"
            index_path.write_text("index")
            replicas = tuple(
                ReplicaConfig(
                    replica_index=index,
                    gpu_uuid=f"GPU-{index}",
                    path=root / f"config-{index}.json",
                    production=SimpleNamespace(),
                    task_indices=(index,),
                )
                for index in range(8)
            )
            for replica in replicas:
                replica.path.write_text("config")
                drivers[str(replica.path)] = Driver(replica.replica_index)
            config = CoordinatorConfig(index_path, root, replicas, ())

            def reconcile_digest(_root):
                events.append(("digest-occupants", 0))
                return {
                    "schema": "amg_swebench_image_digest_reconciliation_v1",
                    "status": "PASS",
                    "all_eight_replica_lanes_held": True,
                    "stale_occupants": 1,
                    "cleared": [],
                }

            with patch(
                "swebench_triad_eval.shared_pool_coordinator.validated_startup_barrier",
                return_value={"status": "PASS"},
            ) as barrier, patch(
                "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
                side_effect=lambda path, **_kwargs: drivers[str(path)],
            ), patch(
                "swebench_triad_eval.shared_pool_coordinator.reconcile_digest_occupants",
                side_effect=reconcile_digest,
            ):
                receipt = reconcile_all_eight_before_workers(
                    config,
                    phase="full",
                    startup_barrier_sha256="b" * 64,
                )
            self.assertEqual(barrier.call_count, 2)
            self.assertEqual([row[0] for row in events[:16]], ["acquire"] * 16)
            self.assertEqual([row[0] for row in events[16:24]], ["reconcile"] * 8)
            self.assertEqual(events[24:26], [("shared-images", 0), ("digest-occupants", 0)])
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["digest_lease_reconciliation"]["stale_occupants"], 1)
            self.assertTrue(
                (root / "control" / "full-worker-reconciliation.json").is_file()
            )

    def test_lane_acquisition_failure_releases_the_failing_driver_too(self):
        released = []

        class Registry:
            def __init__(self, index):
                self.index = index

            def release(self):
                released.append(self.index)

        class Driver:
            def __init__(self, index):
                self.index = index
                self.lease_registry = Registry(index)

            def acquire_runtime_lane(self, _task_index, *, slot_index):
                if self.index == 3:
                    raise RuntimeError("simulated live driver")

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            replicas = tuple(
                ReplicaConfig(
                    replica_index=index,
                    gpu_uuid=f"GPU-{index}",
                    path=root / f"config-{index}.json",
                    production=SimpleNamespace(),
                    task_indices=(index,),
                )
                for index in range(8)
            )
            (root / "index.json").write_text("index")
            for replica in replicas:
                replica.path.write_text(f"config-{replica.replica_index}")
            config = CoordinatorConfig(root / "index.json", root, replicas, ())
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
                side_effect=lambda path, **_kwargs: Driver(
                    int(Path(path).stem.rsplit("-", 1)[1])
                ),
            ), self.assertRaisesRegex(RuntimeError, "simulated live driver"):
                preflight_all(config)
        self.assertEqual(released, [3, 2, 1, 0])


class LivePoolSnapshotValidationTest(unittest.TestCase):
    def test_live_pool_snapshot_binds_ports_processes_and_listener_owners(self):
        production = SimpleNamespace(
            shared_model_pool={
                "owner": OWNER,
                "readiness_sha256": READINESS_SHA,
                "marker_lease_sha256": MARKER_SHA,
                "gpu_index": 3,
                "model_id": "Qwen/Qwen3.5-4B",
                "model_revision": MODEL_REVISION,
                "model_port": 18021,
                "proxy_port": 16383,
            },
            section=lambda name: {"pid": 303, "start_ticks": 3003}
            if name == "serving"
            else {},
        )
        replica = ReplicaConfig(3, "GPU-3", Path("/tmp/config.json"), production, (3,))
        upstream = "http://127.0.0.1:18021"
        snapshot = {
            "status": "PASS",
            "owner": OWNER,
            "readiness_sha256": READINESS_SHA,
            "marker_lease_sha256": MARKER_SHA,
            "replica_index": 3,
            "replica_count": 8,
            "gpu_index": 3,
            "gpu_uuid": "GPU-3",
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": MODEL_REVISION,
            "model_port": 18021,
            "proxy_port": 16383,
            "server_pid": 303,
            "server_start_ticks": 3003,
            "server_target_pids": [303],
            "server_listener_pids": [303],
            "server_listener_census": {
                "source": "/proc/net/tcp",
                "family": "ipv4",
                "address": "127.0.0.1",
                "port": 18021,
                "inode": 99,
                "owner_pids": [303],
            },
            "proxy_pid": 403,
            "proxy_start_ticks": 4003,
            "proxy_target_pids": [403],
            "proxy_listener_pids": [403],
            "proxy_listener_census": {
                "source": "/proc/net/tcp",
                "family": "ipv4",
                "address": "127.0.0.1",
                "port": 16383,
                "inode": 100,
                "owner_pids": [403],
            },
            "proxy_route": {
                "config_path": "/tmp/proxy-config.json",
                "config_sha256": "4" * 64,
                "proxy_source_sha256": "5" * 64,
                "runtime_sha256": "6" * 64,
                "tokenizer_sha256": "7" * 64,
                "upstream_base_url": upstream,
                "upstream_base_url_sha256": hashlib.sha256(
                    upstream.encode("utf-8")
                ).hexdigest(),
            },
            "assigned_gpu_process_pids": [505],
            "all_replicas_alive": True,
            "all_endpoints_healthy": True,
            "assignment_algorithm": "uint64_be(sha256(task_id)[:8]) % 8",
            "cleanup_policy": "retain_external_pool",
        }
        self.assertIs(
            validate_live_pool_snapshot(snapshot, replica, "test"), snapshot
        )
        drifted = copy.deepcopy(snapshot)
        drifted["proxy_route"]["upstream_base_url"] = (
            "http://127.0.0.1:18018"
        )
        with self.assertRaisesRegex(RuntimeError, "proxy route"):
            validate_live_pool_snapshot(drifted, replica, "test")
        failed = {**snapshot, "status": "FAIL"}
        with self.assertRaisesRegex(RuntimeError, "identity"):
            validate_live_pool_snapshot(failed, replica, "test")
        extra = {**snapshot, "unexpected": True}
        with self.assertRaisesRegex(RuntimeError, "fields"):
            validate_live_pool_snapshot(extra, replica, "test")
        extra_route = copy.deepcopy(snapshot)
        extra_route["proxy_route"]["unexpected"] = True
        with self.assertRaisesRegex(RuntimeError, "proxy route fields"):
            validate_live_pool_snapshot(extra_route, replica, "test")

        for field, value in (
            ("source", "/proc/net/tcp6"),
            ("family", "ipv6"),
            ("address", "::1"),
            ("port", 18022),
        ):
            with self.subTest(field=field):
                forged = copy.deepcopy(snapshot)
                forged["server_listener_census"][field] = value
                with self.assertRaisesRegex(RuntimeError, "listener census"):
                    validate_live_pool_snapshot(forged, replica, "test")

        inode_drift = copy.deepcopy(snapshot)
        inode_drift["server_listener_census"]["inode"] = 101
        with self.assertRaisesRegex(RuntimeError, "listener census drifted"):
            validate_live_pool_snapshot(
                inode_drift,
                replica,
                "test",
                listener_reference=snapshot,
            )

        for change in ("missing", "extra"):
            with self.subTest(change=change):
                malformed = copy.deepcopy(snapshot)
                census = malformed["proxy_listener_census"]
                if change == "missing":
                    census.pop("inode")
                else:
                    census["unexpected"] = True
                with self.assertRaisesRegex(RuntimeError, "listener census fields"):
                    validate_live_pool_snapshot(malformed, replica, "test")


class SharedPoolCleanupTest(unittest.TestCase):
    def test_cleanup_delegates_to_the_all_lease_preflight_reconciler(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "control").mkdir()
            index = root / "index.json"
            index.write_text("{}")
            config = CoordinatorConfig(index, root, (), ())
            replicas = [{"replica_index": index} for index in range(8)]
            preflight = {
                "schema": "amg_swebench_shared_pool_preflight_v1",
                "status": "PASS",
            }

            def reconcile(_config):
                (root / "control" / "preflight-all.json").write_bytes(
                    canonical_json_bytes(preflight)
                )
                return replicas

            with patch(
                "swebench_triad_eval.shared_pool_coordinator.preflight_all",
                side_effect=reconcile,
            ) as all_leases:
                receipt = cleanup_all(config)
            all_leases.assert_called_once_with(config)
            self.assertTrue(
                receipt["all_replica_leases_held_during_reconciliation"]
            )
            self.assertTrue(receipt["external_model_pool_retained"])
            self.assertTrue(receipt["allocation_retained"])



class FullRunTransactionRestartTest(unittest.TestCase):
    def journal(self, root: Path) -> dict[str, object]:
        return {
            "schema": "amg_swebench_full_run_transaction_v1",
            "status": "RUNNING",
            "started_wall_ns": 1_000_000_000,
            "updated_wall_ns": 1_000_000_000,
            "baseline_task_indices": [],
            "remaining_cells_at_launch": 1500,
            "timing_gate_sha256": "a" * 64,
            "eta_checks": [],
            "last_elapsed_seconds": 0.0,
            "last_completed_cells": 0,
            "consecutive_over_budget_checks": 0,
            "full_run_timing_sha256": None,
            "workers_complete_sha256": None,
        }

    def test_restart_rolls_forward_orphan_progress_and_global_sequence(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = SimpleNamespace(root=root)
            journal = self.journal(root)
            progress_path = _eta_progress_path(root, 1)
            progress = {
                "schema": ETA_PROGRESS_SCHEMA,
                "status": "PASS",
                "check_index": 1,
                "observed_wall_ns": 11_000_000_000,
                "elapsed_seconds": 10.0,
                "baseline_task_indices_sha256": sha256_json(
                    {"baseline_task_indices": []}
                ),
                "completed_task_indices": list(range(25)),
                "new_completed_task_indices": list(range(25)),
                "baseline_completed_tasks": 0,
                "baseline_completed_cells": 0,
                "new_completed_tasks": 25,
                "new_completed_cells": 75,
                "remaining_cells_at_launch": 1500,
                "trigger_reasons": ["cell_interval"],
                "timing_gate_sha256": "a" * 64,
            }
            atomic_write_json(progress_path, progress)
            rows, elapsed, cells, consecutive = _reconcile_eta_history(config, journal)
            self.assertTrue(_eta_receipt_path(root, 1).is_file())
            self.assertEqual((elapsed, cells, consecutive), (10.0, 75, 0))
            second = _publish_eta_check(
                config,
                journal=journal,
                check_index=2,
                observed_wall_ns=21_000_000_000,
                completed_tasks=set(range(50)),
                trigger_reasons=["cell_interval"],
                prior_consecutive=0,
            )
            self.assertEqual(second["receipt"]["check_index"], 2)
            self.assertEqual(len(rows), 1)
            self.assertNotEqual(rows[0]["path"], second["path"])

    def test_gap_duplicate_and_reordered_progress_are_rejected(self):
        base = {
            "new_completed_cells": 300,
            "remaining_cells_at_launch": 1500,
            "elapsed_seconds": 4000.0,
            "trigger_reasons": ["elapsed_interval", "cell_interval"],
        }
        with self.assertRaisesRegex(RuntimeError, "cadence"):
            _validate_eta_cadence(base, prior_elapsed=0.0, prior_cells=0)
        elapsed_only = dict(base)
        elapsed_only.update(
            {
                "new_completed_cells": 3,
                "elapsed_seconds": 8000.0,
                "trigger_reasons": ["elapsed_interval"],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "cadence"):
            _validate_eta_cadence(elapsed_only, prior_elapsed=0.0, prior_cells=0)
        cells_only = dict(base)
        cells_only.update(
            {
                "new_completed_cells": 300,
                "elapsed_seconds": 10.0,
                "trigger_reasons": ["cell_interval"],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "cadence"):
            _validate_eta_cadence(cells_only, prior_elapsed=0.0, prior_cells=0)
        duplicate = dict(base)
        duplicate.update(
            {
                "new_completed_cells": 75,
                "elapsed_seconds": 10.0,
                "trigger_reasons": ["cell_interval"],
            }
        )
        _validate_eta_cadence(duplicate, prior_elapsed=0.0, prior_cells=0)
        with self.assertRaisesRegex(RuntimeError, "trigger"):
            _validate_eta_cadence(duplicate, prior_elapsed=10.0, prior_cells=75)
        with self.assertRaisesRegex(RuntimeError, "reordered"):
            _validate_eta_cadence(duplicate, prior_elapsed=20.0, prior_cells=100)

    def test_polling_overshoot_is_published_and_restart_consistent(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = SimpleNamespace(root=root)
            journal = self.journal(root)
            self.assertEqual(
                _eta_trigger_reasons(
                    elapsed_seconds=9.0,
                    prior_elapsed_seconds=0.0,
                    completed_cells=72,
                    prior_completed_cells=0,
                    remaining_cells_at_launch=1500,
                    workers_pending=True,
                ),
                [],
            )
            reasons = _eta_trigger_reasons(
                elapsed_seconds=10.0,
                prior_elapsed_seconds=0.0,
                completed_cells=78,
                prior_completed_cells=0,
                remaining_cells_at_launch=1500,
                workers_pending=True,
            )
            self.assertEqual(reasons, ["cell_interval"])
            published = _publish_eta_check(
                config,
                journal=journal,
                check_index=1,
                observed_wall_ns=11_000_000_000,
                completed_tasks=set(range(26)),
                trigger_reasons=reasons,
                prior_consecutive=0,
            )
            self.assertEqual(published["progress"]["new_completed_cells"], 78)
            rows, elapsed, cells, consecutive = _reconcile_eta_history(
                config, journal
            )
            self.assertEqual((len(rows), elapsed, cells, consecutive), (1, 10.0, 78, 0))

    def test_final_coverage_is_durable_while_workers_are_pending(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = SimpleNamespace(root=root)
            journal = self.journal(root)
            journal["baseline_task_indices"] = list(range(475))
            journal["remaining_cells_at_launch"] = 75
            reasons = _eta_trigger_reasons(
                elapsed_seconds=10.0,
                prior_elapsed_seconds=0.0,
                completed_cells=75,
                prior_completed_cells=0,
                remaining_cells_at_launch=75,
                workers_pending=True,
            )
            self.assertEqual(reasons, ["cell_interval", "final_completion"])
            published = _publish_eta_check(
                config,
                journal=journal,
                check_index=1,
                observed_wall_ns=11_000_000_000,
                completed_tasks=set(range(500)),
                trigger_reasons=reasons,
                prior_consecutive=0,
            )
            self.assertEqual(
                published["progress"]["trigger_reasons"],
                ["cell_interval", "final_completion"],
            )
            rows, elapsed, cells, consecutive = _reconcile_eta_history(
                config, journal
            )
            self.assertEqual((len(rows), elapsed, cells, consecutive), (1, 10.0, 75, 0))

    def test_publisher_rejects_omitted_cadence_before_immutable_write(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = SimpleNamespace(root=root)
            journal = self.journal(root)
            with self.assertRaisesRegex(RuntimeError, "mandatory cadence"):
                _publish_eta_check(
                    config,
                    journal=journal,
                    check_index=1,
                    observed_wall_ns=11_000_000_000,
                    completed_tasks=set(range(100)),
                    trigger_reasons=["cell_interval"],
                    prior_consecutive=0,
                )
            self.assertFalse(_eta_progress_path(root, 1).exists())

    def test_partial_stop_boundary_is_never_labeled_final_completion(self):
        partial = _eta_trigger_reasons(
            elapsed_seconds=1810.0,
            prior_elapsed_seconds=0.0,
            completed_cells=75,
            prior_completed_cells=0,
            remaining_cells_at_launch=150,
            workers_pending=False,
        )
        self.assertEqual(partial, ["elapsed_interval", "cell_interval"])
        complete = _eta_trigger_reasons(
            elapsed_seconds=1810.0,
            prior_elapsed_seconds=0.0,
            completed_cells=150,
            prior_completed_cells=75,
            remaining_cells_at_launch=150,
            workers_pending=False,
        )
        self.assertEqual(
            complete,
            ["elapsed_interval", "cell_interval", "final_completion"],
        )


class FullRunTimingBindingTest(unittest.TestCase):
    def _write_receipts(self, root: Path):
        timing_gate = {"projection": {"projected_full_makespan_seconds": 100.0}}
        timing_gate_path = root / "control" / "timing-gate.json"
        atomic_write_json(timing_gate_path, timing_gate)
        timing_gate_sha256 = hashlib.sha256(timing_gate_path.read_bytes()).hexdigest()
        progress_path = root / "control" / "eta" / "progress-000001.json"
        progress = {
            "schema": ETA_PROGRESS_SCHEMA,
            "status": "PASS",
            "check_index": 1,
            "observed_wall_ns": 10_000_000_001,
            "elapsed_seconds": 10.0,
            "baseline_task_indices_sha256": sha256_json(
                {"baseline_task_indices": list(range(475))}
            ),
            "completed_task_indices": list(range(500)),
            "new_completed_task_indices": list(range(475, 500)),
            "baseline_completed_tasks": 475,
            "baseline_completed_cells": 1425,
            "new_completed_tasks": 25,
            "new_completed_cells": 75,
            "remaining_cells_at_launch": 75,
            "trigger_reasons": ["cell_interval", "final_completion"],
            "timing_gate_sha256": timing_gate_sha256,
        }
        atomic_write_json(progress_path, progress)
        eta_path = root / "control" / "eta" / "check-000001.json"
        eta = {
            "schema": ETA_RECEIPT_SCHEMA,
            "status": "WITHIN_STOP_THRESHOLD",
            "check_index": 1,
            "progress_snapshot_path": str(progress_path),
            "progress_snapshot_sha256": hashlib.sha256(
                progress_path.read_bytes()
            ).hexdigest(),
            "observed_wall_ns": 10_000_000_001,
            "elapsed_seconds": 10.0,
            "baseline_completed_tasks": 475,
            "baseline_completed_cells": 1425,
            "new_completed_tasks": 25,
            "new_completed_cells": 75,
            "remaining_cells_at_launch": 75,
            "trigger_reasons": ["cell_interval", "final_completion"],
            "projected_remaining_makespan_seconds": 10.0,
            "stop_threshold_seconds": TIMING_BUDGET_SECONDS * 1.5,
            "consecutive_over_budget_checks": 0,
            "timing_gate_sha256": timing_gate_sha256,
        }
        atomic_write_json(eta_path, eta)
        full = {
            "schema": FULL_RUN_TIMING_SCHEMA,
            "status": "PASS",
            "started_wall_ns": 1,
            "ended_wall_ns": 10_000_000_001,
            "actual_wall_seconds": 10.0,
            "initial_projected_full_makespan_seconds": 100.0,
            "timing_gate_sha256": timing_gate_sha256,
            "baseline_task_indices": list(range(475)),
            "eta_checks": [
                {
                    "path": str(eta_path),
                    "sha256": hashlib.sha256(eta_path.read_bytes()).hexdigest(),
                }
            ],
            "final_projected_remaining_makespan_seconds": 10.0,
        }
        atomic_write_json(root / "control" / "full-run-timing.json", full)
        return timing_gate, timing_gate_sha256, eta_path

    def test_full_run_timing_recomputes_eta_and_rejects_nested_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            timing_gate, digest, eta_path = self._write_receipts(root)
            config = SimpleNamespace(root=root)
            receipt = _validated_full_run_timing(
                config,
                timing_gate=timing_gate,
                timing_gate_sha256=digest,
            )
            self.assertEqual(receipt["actual_wall_seconds"], 10.0)
            tampered = json.loads(eta_path.read_text())
            tampered["new_completed_cells"] = 6
            atomic_write_json(eta_path, tampered)
            full_path = root / "control" / "full-run-timing.json"
            full = json.loads(full_path.read_text())
            full["eta_checks"][0]["sha256"] = hashlib.sha256(
                eta_path.read_bytes()
            ).hexdigest()
            atomic_write_json(full_path, full)
            with self.assertRaisesRegex(RuntimeError, "ETA receipt identity"):
                _validated_full_run_timing(
                    config,
                    timing_gate=timing_gate,
                    timing_gate_sha256=digest,
                )

    def test_closure_rejects_omitted_time_and_cell_checkpoints(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            timing_gate, digest, eta_path = self._write_receipts(root)
            progress_path = root / "control/eta/progress-000001.json"
            progress = json.loads(progress_path.read_text())
            progress["observed_wall_ns"] = 4_000_000_000_001
            progress["elapsed_seconds"] = 4000.0
            progress["trigger_reasons"] = [
                "elapsed_interval",
                "cell_interval",
                "final_completion",
            ]
            atomic_write_json(progress_path, progress)
            receipt = _eta_receipt_from_progress(
                progress,
                progress_path=progress_path,
                prior_consecutive=0,
            )
            atomic_write_json(eta_path, receipt)
            full_path = root / "control/full-run-timing.json"
            full = json.loads(full_path.read_text())
            full["ended_wall_ns"] = 4_000_000_000_001
            full["actual_wall_seconds"] = 4000.0
            full["eta_checks"][0]["sha256"] = hashlib.sha256(
                eta_path.read_bytes()
            ).hexdigest()
            full["final_projected_remaining_makespan_seconds"] = receipt[
                "projected_remaining_makespan_seconds"
            ]
            atomic_write_json(full_path, full)
            with self.assertRaisesRegex(RuntimeError, "mandatory cadence"):
                _validated_full_run_timing(
                    SimpleNamespace(root=root),
                    timing_gate=timing_gate,
                    timing_gate_sha256=digest,
                )




class WorkersCompleteTimingBindingTest(unittest.TestCase):
    def test_workers_complete_binds_timing_gate_and_full_run_receipt(self):
        with tempfile.TemporaryDirectory() as raw, patch(
            "swebench_triad_eval.shared_pool_coordinator.image_lock_rows",
            side_effect=fake_image_rows,
        ):
            config = CoordinatorConfig.load(make_shared_coordinator(Path(raw)))
            config.write_assignment()
            write_startup_barrier(config.root, config)
            gate_path = config.root / "control" / "gate.json"
            timing_gate_path = config.root / "control" / "timing-gate.json"
            full_timing_path = config.root / "control" / "full-run-timing.json"
            atomic_write_json(gate_path, {"schema": "fixture", "status": "PASS"})
            atomic_write_json(
                timing_gate_path,
                {
                    "schema": "fixture-timing",
                    "projection": {"projected_full_makespan_seconds": 1.0},
                },
            )
            atomic_write_json(
                full_timing_path,
                {"schema": "fixture-full-timing", "status": "PASS"},
            )
            workers = [
                {
                    "schema": "amg_swebench_shared_pool_worker_v2",
                    "status": "PASS",
                    "replica_index": replica.replica_index,
                    "completed_tasks": len(replica.task_indices),
                    "total_tasks": len(replica.task_indices),
                    "task_slots_per_replica": 2,
                    "completed_task_indices": list(replica.task_indices),
                }
                for replica in config.replicas
            ]
            audits = []
            for replica in config.replicas:
                receipt = {
                    "status": "PASS",
                    "allocation_retained": True,
                    "residue": {"owned": 0},
                    "shared_model_pool": {},
                }
                audits.append(
                    {
                        "replica_index": replica.replica_index,
                        "receipt": receipt,
                        "receipt_sha256": sha256_json(receipt),
                    }
                )
            workers_path = config.root / "control" / "workers-complete.json"
            value = {
                "schema": WORKERS_COMPLETE_SCHEMA,
                "status": "PASS",
                "coordinator_index_sha256": hashlib.sha256(
                    config.path.read_bytes()
                ).hexdigest(),
                "assignment_sha256": hashlib.sha256(
                    (config.root / "control" / "assignment.json").read_bytes()
                ).hexdigest(),
                "startup_barrier_sha256": hashlib.sha256(
                    (config.root / "control" / "preflight-all.json").read_bytes()
                ).hexdigest(),
                "gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
                "timing_gate_sha256": hashlib.sha256(
                    timing_gate_path.read_bytes()
                ).hexdigest(),
                "full_run_timing_sha256": hashlib.sha256(
                    full_timing_path.read_bytes()
                ).hexdigest(),
                "workers": workers,
                "final_audits": audits,
            }
            atomic_write_json(workers_path, value)
            with self.assertRaisesRegex(RuntimeError, "transaction journal"):
                validated_workers_complete(config)
            journal_path = config.root / "control" / "full-run-transaction.json"
            atomic_write_json(journal_path, {"schema": "fixture-journal"})
            journal = {
                "status": "CLOSING",
                "full_run_timing_sha256": hashlib.sha256(
                    full_timing_path.read_bytes()
                ).hexdigest(),
                "workers_complete_sha256": None,
            }
            timing_gate = {
                "projection": {"projected_full_makespan_seconds": 1.0}
            }
            patches = (
                patch(
                    "swebench_triad_eval.shared_pool_coordinator.validated_timing_gate",
                    return_value=timing_gate,
                ),
                patch(
                    "swebench_triad_eval.shared_pool_coordinator._validated_full_run_timing",
                    return_value={"status": "PASS"},
                ),
                patch(
                    "swebench_triad_eval.shared_pool_coordinator.validated_preflight_pool_snapshot",
                    return_value={},
                ),
                patch(
                    "swebench_triad_eval.shared_pool_coordinator.validate_live_pool_snapshot",
                    return_value={},
                ),
                patch(
                    "swebench_triad_eval.shared_pool_coordinator._load_or_create_full_run_journal",
                    return_value=journal,
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                self.assertEqual(validated_workers_complete(config), value)
                tampered = copy.deepcopy(value)
                tampered["timing_gate_sha256"] = "0" * 64
                atomic_write_json(workers_path, tampered)
                with self.assertRaisesRegex(RuntimeError, "binding drifted"):
                    validated_workers_complete(config)
                tampered = copy.deepcopy(value)
                tampered["full_run_timing_sha256"] = "1" * 64
                atomic_write_json(workers_path, tampered)
                with self.assertRaisesRegex(RuntimeError, "binding drifted"):
                    validated_workers_complete(config)


class GateBindingTest(unittest.TestCase):
    def test_run_full_rejects_unbound_gate_before_spawning_workers(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "control").mkdir()
            production = SimpleNamespace(run_root=root / "replica-run")
            replicas = tuple(
                ReplicaConfig(
                    replica_index=index,
                    gpu_uuid=f"GPU-{index}",
                    path=root / f"config-{index}.json",
                    production=production,
                    task_indices=(index,),
                )
                for index in range(8)
            )
            assignment = tuple(
                {
                    "task_index": index,
                    "task_id": f"task-{index}",
                    "replica_index": index % 8,
                    "image": f"image-{index}",
                    "image_config_digest": "sha256:" + f"{index:064x}",
                }
                for index in range(500)
            )
            (root / "index.json").write_text("index")
            for replica in replicas:
                replica.path.write_text(f"config-{replica.replica_index}")
            config = CoordinatorConfig(root / "index.json", root, replicas, assignment)
            write_startup_barrier(root, config)
            (root / "control" / "gate.json").write_bytes(
                canonical_json_bytes({"status": "PASS"})
            )
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.ProcessPoolExecutor"
            ) as executor, self.assertRaisesRegex(
                RuntimeError, "canonical task-0 gate"
            ):
                run_full(config)
            executor.assert_not_called()

    def test_run_full_rejects_fabricated_nested_gate_before_spawning_workers(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "control").mkdir()
            production = SimpleNamespace(run_root=root / "replica-run")
            replicas = tuple(
                ReplicaConfig(
                    replica_index=index,
                    gpu_uuid=f"GPU-{index}",
                    path=root / f"config-{index}.json",
                    production=production,
                    task_indices=(index,),
                )
                for index in range(8)
            )
            assignment = tuple(
                {
                    "task_index": index,
                    "task_id": f"task-{index}",
                    "replica_index": index % 8,
                    "image": f"image-{index}",
                    "image_config_digest": "sha256:" + f"{index:064x}",
                }
                for index in range(500)
            )
            (root / "index.json").write_text("index")
            for replica in replicas:
                replica.path.write_text(f"config-{replica.replica_index}")
            config = CoordinatorConfig(root / "index.json", root, replicas, assignment)
            write_startup_barrier(root, config)
            fabricated = {"fabricated": True}
            (root / "control" / "gate.json").write_bytes(
                canonical_json_bytes(
                    {
                        "schema": "amg_swebench_shared_pool_gate_v1",
                        "status": "PASS",
                        "replica_index": 0,
                        "gpu_uuid": "GPU-0",
                        "gate": fabricated,
                        "gate_sha256": sha256_json(fabricated),
                    }
                )
            )

            class Driver:
                lease_registry = None
                gate_path = root / "replica-run" / "gate" / "PASS.json"

                def gate(self, *, auto_run_full):
                    self.assertFalse(auto_run_full)
                    return {"canonical": True}

                @staticmethod
                def assertFalse(value):
                    if value:
                        raise AssertionError("full validation must not recurse")

            Driver.gate_path.parent.mkdir(parents=True)
            Driver.gate_path.write_text("{}")
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
                return_value=Driver(),
            ), patch(
                "swebench_triad_eval.shared_pool_coordinator.ProcessPoolExecutor"
            ) as executor, self.assertRaisesRegex(
                RuntimeError, "canonical task-0 gate"
            ):
                run_full(config)
            executor.assert_not_called()

    def test_run_full_requires_fresh_under_budget_timing_gate(self):
        with tempfile.TemporaryDirectory() as raw, patch(
            "swebench_triad_eval.shared_pool_coordinator.image_lock_rows",
            side_effect=fake_image_rows,
        ):
            config = CoordinatorConfig.load(make_shared_coordinator(Path(raw)))
            config.write_assignment()
            write_startup_barrier(config.root, config)
            canonical_gate = {"canonical": True}
            atomic_write_json(
                config.root / "control" / "gate.json",
                {
                    "schema": "amg_swebench_shared_pool_gate_v1",
                    "status": "PASS",
                    "replica_index": config.task_zero_replica.replica_index,
                    "gpu_uuid": config.task_zero_replica.gpu_uuid,
                    "gate": canonical_gate,
                    "gate_sha256": sha256_json(canonical_gate),
                },
            )

            class Driver:
                lease_registry = None
                gate_path = config.root / "gate-PASS.json"

                @staticmethod
                def gate(*, auto_run_full):
                    if auto_run_full:
                        raise AssertionError("full validation must not recurse")
                    return canonical_gate

            Driver.gate_path.write_text("{}")
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
                return_value=Driver(),
            ), patch(
                "swebench_triad_eval.shared_pool_coordinator.ProcessPoolExecutor"
            ) as executor, self.assertRaisesRegex(
                RuntimeError, "C=2 timing contract"
            ):
                run_full(config)
            executor.assert_not_called()

    def test_aggregate_rejects_outcomes_before_workers_and_cleanup_complete(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "control").mkdir()
            index = root / "index.json"
            index.write_bytes(canonical_json_bytes({"index": True}))
            replicas = tuple(
                ReplicaConfig(
                    replica_index=value,
                    gpu_uuid=f"GPU-{value}",
                    path=root / f"config-{value}.json",
                    production=SimpleNamespace(),
                    task_indices=(value,),
                )
                for value in range(8)
            )
            config = CoordinatorConfig(index, root, replicas, ())
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.driver_from_config"
            ) as driver, self.assertRaisesRegex(
                RuntimeError, "workers-complete"
            ):
                aggregate(config)
            driver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
