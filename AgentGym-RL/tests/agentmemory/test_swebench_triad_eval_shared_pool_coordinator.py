from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paired_eval.serialization import canonical_json_bytes
from swebench_triad_eval.state import sha256_json
from swebench_triad_eval.shared_pool_coordinator import (
    INDEX_SCHEMA,
    CoordinatorConfig,
    ReplicaConfig,
    _worker,
    aggregate,
    assigned_replica,
    cleanup_all,
    image_lock_rows,
    preflight_all,
    run_full,
    validate_live_pool_snapshot,
)
from test_swebench_triad_eval_cli import production_config

OWNER = "amg-external-eval-g-dp8-swe-0818"
MODEL_REVISION = "3" * 40
READINESS_SHA = "1" * 64
MARKER_SHA = "2" * 64
IMAGE_DIGEST = "sha256:" + "a" * 64


def make_shared_coordinator(root: Path) -> Path:
    template_root = root / "template"
    template_root.mkdir()
    _, template = production_config(template_root)
    rows = []
    for replica in range(8):
        replica_root = root / f"replica-{replica}"
        config = copy.deepcopy(template)
        config["schema"] = "amg_swebench_triad_run_config_shared_pool_v2"
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
        config["runtime"]["server_port"] = 18_100 + replica
        config["grader"]["output_root"] = str(replica_root / "grader")
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
            second["runtime"]["server_port"] = first["runtime"]["server_port"]
            encoded = canonical_json_bytes(second)
            paths[1].write_bytes(encoded)
            value["replicas"][1]["config_sha256"] = hashlib.sha256(encoded).hexdigest()
            index.write_bytes(canonical_json_bytes(value))
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.image_lock_rows",
                side_effect=fake_image_rows,
            ), self.assertRaisesRegex(ValueError, "not unique"):
                CoordinatorConfig.load(index)

    def test_cross_category_port_collision_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            index = make_shared_coordinator(root)
            value = json.loads(index.read_text())
            paths = [Path(row["config_path"]) for row in value["replicas"]]
            first = json.loads(paths[0].read_text())
            second = json.loads(paths[1].read_text())
            second["runtime"]["server_port"] = first["shared_model_pool"]["model_port"]
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


class WorkerTest(unittest.TestCase):
    def test_worker_uses_digest_lock_and_does_not_repeat_startup_reconciliation(self):
        calls = []

        class Driver:
            lease_registry = None

            def ensure_driver_lease(self):
                calls.append("lease")

            def _read_validated_preflight(self):
                calls.append("preflight")

            def reconcile_dead_work(self):
                raise AssertionError("coordinator preflight owns reconciliation")

            def run_task(self, task_index, *, gate):
                calls.append((task_index, gate))

        with tempfile.TemporaryDirectory() as raw, patch(
            "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
            return_value=Driver(),
        ):
            result = _worker(
                "/tmp/config.json",
                (4, 9),
                (IMAGE_DIGEST, IMAGE_DIGEST),
                3,
                raw,
            )
            lock_root = Path(raw) / "control" / "image-leases"
            self.assertEqual(len(list(lock_root.glob("*.lock"))), 1)
        self.assertEqual(calls, ["lease", "preflight", (4, False), (9, False)])
        self.assertEqual(result["completed_tasks"], 2)

    def test_worker_rejects_task_image_lattice_drift(self):
        with self.assertRaisesRegex(ValueError, "lattice"):
            _worker("/tmp/config.json", (1,), (), 0, "/tmp/root")


class SharedPoolPreflightTest(unittest.TestCase):
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

            def acquire_runtime_lane(self, task_index):
                self.assert_task_none(task_index)
                events.append(("acquire", self.index))

            @staticmethod
            def assert_task_none(task_index):
                if task_index is not None:
                    raise AssertionError("shared preflight must acquire a global lane")

            def reconcile_dead_work(self, *, allow_foreign_loaded_images):
                self.assertEqual_all_acquired()
                if allow_foreign_loaded_images is not True:
                    raise AssertionError("foreign images must be deferred across roots")
                events.append(("reconcile", self.index))
                return {"foreign_staged_tasks": [], "replica_index": self.index}

            @staticmethod
            def assertEqual_all_acquired():
                acquired = [row for row in events if row[0] == "acquire"]
                if len(acquired) != 8:
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
            config = CoordinatorConfig(root / "index.json", root, replicas, ())
            for replica in replicas:
                drivers[str(replica.path)] = Driver(replica.replica_index)
            with patch(
                "swebench_triad_eval.shared_pool_coordinator.driver_from_config",
                side_effect=lambda path, **_kwargs: drivers[str(path)],
            ):
                receipts = preflight_all(config)
        self.assertEqual(len(receipts), 8)
        self.assertEqual([row[0] for row in events[:8]], ["acquire"] * 8)
        self.assertEqual([row[0] for row in events[8:16]], ["reconcile"] * 8)
        self.assertEqual(events[16], ("shared-images", 0))
        self.assertEqual([row[0] for row in events[17:25]], ["preflight"] * 8)

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

            def acquire_runtime_lane(self, _task_index):
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
            "proxy_pid": 403,
            "proxy_start_ticks": 4003,
            "proxy_target_pids": [403],
            "proxy_listener_pids": [403],
            "proxy_route": {
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
            config = CoordinatorConfig(root / "index.json", root, replicas, assignment)
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
            config = CoordinatorConfig(root / "index.json", root, replicas, assignment)
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
