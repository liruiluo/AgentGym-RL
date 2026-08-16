from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from test_swebench_triad_eval_cli import production_config

from paired_eval.serialization import canonical_json_bytes
from swebench_triad_eval.official_grader import (
    DOCKER_SOCKET,
    RetryableGraderError,
)
from swebench_triad_eval.identity import (
    HARNESS_COMMIT,
    HARNESS_TREE,
    PRODUCTION_DATASET_PINS,
)
from swebench_triad_eval.production import (
    LinuxProductionRuntime,
    ProductionLifecycleOperations,
    ProductionRunConfig,
    accepted_rows_for_eviction,
    summarize_task4_receipt,
)
from swebench_triad_eval.state import CellKey, sha256_json


class RecordingProductionRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def preflight(self):
        self.calls.append(("preflight",))
        return {"snapshot": True}

    def stage_task(self, task_index):
        self.calls.append(("stage", task_index))
        return {"task_index": task_index}

    def reconcile_cell(self, config, *, generation, before_preflight):
        self.calls.append(
            (
                "reconcile_cell",
                config.task.task_index,
                config.capability.arm.value,
                generation,
                before_preflight,
            )
        )
        return {"reconciled": True}

    def reconcile_grade(self, **kwargs):
        self.calls.append(("reconcile_grade", kwargs["key"]))
        return {"reconciled": True}

    def reconcile_startup(self, *, task_indices):
        self.calls.append(("reconcile_startup", tuple(task_indices)))
        return {"reconciled": True}

    def run_cell(self, config, stage, *, generation):
        self.calls.append(
            (
                "run",
                config.task.task_index,
                config.capability.arm.value,
                stage["task_index"],
                generation,
            )
        )
        return {"endpoint": True}

    def grade(self, **kwargs):
        self.calls.append(("grade", kwargs["key"]))
        return {"outcome": True}

    def audit_residue(self, task_index):
        self.calls.append(("audit", task_index))
        return {"containers": 0}

    def evict_task(self, task_index, stage):
        self.calls.append(("evict", task_index, stage))
        return {"evicted": True}

    def cleanup(self):
        self.calls.append(("cleanup",))
        return {"owned_residue": 0, "allocation_retained": True}

    def final_audit(self):
        self.calls.append(("final_audit",))
        return {"status": "PASS"}


class ProductionOperationsBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        config_path, _ = production_config(Path(self.temporary.name))
        self.config = ProductionRunConfig.load(config_path)
        self.runtime = RecordingProductionRuntime()
        self.operations = ProductionLifecycleOperations(
            self.config,
            self.config.configs,
            runtime=self.runtime,
        )

    def test_every_lifecycle_boundary_is_concrete_and_generation_bound(self):
        stage = self.operations.stage_task(7)
        config = self.config.configs[7 * 3 + 2]
        self.assertEqual(self.operations.preflight(), {"snapshot": True})
        self.assertEqual(
            self.operations.reconcile_cell(
                config, generation=19, before_preflight=True
            ),
            {"reconciled": True},
        )
        self.assertEqual(
            self.operations.reconcile_startup(task_indices=(7,)),
            {"reconciled": True},
        )
        self.assertEqual(
            self.operations.run_cell(config, stage, generation=19),
            {"endpoint": True},
        )
        key = CellKey(7, "amg_memory")
        self.assertEqual(
            self.operations.grade(
                key=key,
                accepted={},
                prediction={},
                handoff={},
            ),
            {"outcome": True},
        )
        self.assertEqual(self.operations.audit_residue(7), {"containers": 0})
        self.assertEqual(
            self.operations.evict_task(7, stage), {"evicted": True}
        )
        self.assertEqual(
            self.operations.cleanup(),
            {"owned_residue": 0, "allocation_retained": True},
        )
        self.assertEqual(self.operations.final_audit(), {"status": "PASS"})
        self.assertIn(("run", 7, "amg_memory", 7, 19), self.runtime.calls)


class ProductionReceiptAdapterTest(unittest.TestCase):
    def test_nested_accepted_cells_are_adapted_strictly_for_eviction(self):
        accepted = [
            {
                "schema": "swebench_triad_accepted_cell_v1",
                "cell": {"task_index": 3, "arm": arm},
                "instance_id": "owner__repo-3",
            }
            for arm in ("native", "amg_compaction_only", "amg_memory")
        ]
        self.assertEqual(
            accepted_rows_for_eviction(3, "owner__repo-3", accepted),
            [
                {
                    "instance_id": "owner__repo-3",
                    "status": "accepted",
                    "arm": arm,
                }
                for arm in ("native", "amg_compaction_only", "amg_memory")
            ],
        )
        accepted[1]["cell"] = {"task_index": 4, "arm": "amg_compaction_only"}
        with self.assertRaisesRegex(ValueError, "accepted cell identity"):
            accepted_rows_for_eviction(3, "owner__repo-3", accepted)

    def test_task4_probe_summary_requires_all_five_negative_boundaries(self):
        receipt = {
            "schema": "amg_swebench_task4_live_negative_probes_v1",
            "status": "PASS",
            "network_downloads": 0,
            "memory_probe": {"teardown": {"memory_failcnt": 1}},
            "pids_probe": {"teardown": {"pids_max_events": 1}},
            "byte_quota_probe": {"outcome": {"errno": 28}},
            "inode_quota_probe": {"outcome": {"errno": 28}},
            "rootfs_mutation_probe": {"detected": True},
            "cgroup_residue": {"absent": True},
            "tmpfs_residue": {"absent": True},
            "docker_after": {
                name: {"count": 0, "ids": []}
                for name in ("containers", "images", "volumes")
            },
        }
        summary = summarize_task4_receipt(receipt, receipt_sha256="a" * 64)
        self.assertTrue(summary["memory_exhaustion_blocked"])
        self.assertTrue(summary["fork_exhaustion_blocked"])
        self.assertTrue(summary["byte_quota_blocked"])
        self.assertTrue(summary["inode_quota_blocked"])
        self.assertTrue(summary["rootfs_mutation_detected"])

        receipt["pids_probe"] = {"teardown": {"pids_max_events": 0}}
        with self.assertRaisesRegex(ValueError, "fork exhaustion"):
            summarize_task4_receipt(receipt, receipt_sha256="a" * 64)


class LinuxProductionRuntimeTest(unittest.TestCase):
    def make_runtime(self, root: Path) -> LinuxProductionRuntime:
        config_path, _ = production_config(root)
        config = ProductionRunConfig.load(config_path)
        return LinuxProductionRuntime(config, config.configs)

    def test_model_snapshot_uses_the_canonical_file_ledger_digest(self):
        with tempfile.TemporaryDirectory() as raw:
            config_path, _ = production_config(Path(raw))
            config = ProductionRunConfig.load(config_path)
            runtime = LinuxProductionRuntime(config, config.configs)
            with patch(
                "swebench_triad_eval.production.verify_model_files",
                return_value={
                    "file_count": 14,
                    "file_ledger_sha256": "f" * 64,
                },
            ):
                self.assertEqual(
                    runtime.model_snapshot(),
                    {"file_count": 14, "file_ledger_sha256": "f" * 64},
                )

    def test_blob_snapshot_reads_the_real_top_level_revalidation_schema(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config_path, payload = production_config(root)
            blob_root = root / "blobs"
            blob_root.mkdir()
            for index in range(1158):
                path = blob_root / f"sha256:{index:064x}"
                path.touch()
            first = next(blob_root.iterdir())
            with first.open("r+b") as stream:
                stream.truncate(117637519356)
            certificate = canonical_json_bytes(
                {
                    "descriptor_count": 1158,
                    "total_bytes": 117637519356,
                    "verified_bad_count": 0,
                }
            )
            revalidation = canonical_json_bytes(
                {
                    "descriptor_count": 1158,
                    "total_bytes": 117637519356,
                    "downloaded_count": 0,
                    "verified_bad_count": 0,
                }
            )
            certificate_path = root / "certificate.json"
            revalidation_path = root / "revalidation.json"
            certificate_path.write_bytes(certificate)
            revalidation_path.write_bytes(revalidation)
            payload["assets"] = dict(payload["assets"])
            payload["assets"].update(
                {
                    "blob_root": str(blob_root),
                    "blob_certificate": str(certificate_path),
                    "blob_certificate_sha256": hashlib.sha256(
                        certificate
                    ).hexdigest(),
                    "blob_revalidation_receipt": str(revalidation_path),
                    "blob_revalidation_sha256": hashlib.sha256(
                        revalidation
                    ).hexdigest(),
                }
            )
            config_path.write_bytes(canonical_json_bytes(payload))
            config = ProductionRunConfig.load(config_path)
            runtime = LinuxProductionRuntime(config, config.configs)
            snapshot = runtime.blob_snapshot()
            self.assertEqual(snapshot["file_count"], 1158)
            self.assertEqual(snapshot["total_bytes"], 117637519356)
            self.assertEqual(snapshot["downloaded_count"], 0)

    def test_official_grader_retries_in_attempt_bound_generation_namespaces(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config_path, payload = production_config(root)
            payload["docker"] = dict(payload["docker"])
            payload["docker"]["socket"] = str(DOCKER_SOCKET)
            config_path.write_bytes(canonical_json_bytes(payload))
            config = ProductionRunConfig.load(config_path)
            runtime = LinuxProductionRuntime(config, config.configs)

            prediction = {
                "instance_id": "owner__repo-0000",
                "model_name_or_path": "Qwen3.5-4B",
                "model_patch": "",
            }
            handoff = {
                "prediction_sha256": sha256_json(prediction),
                "official_resolved": None,
                "grader_revision": (
                    "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
                ),
            }
            accepted = {
                "schema": "swebench_triad_accepted_cell_v1",
                "cell": {"task_index": 0, "arm": "native"},
                "instance_id": "owner__repo-0000",
                "manifest_cell_sha256": "a" * 64,
                "attempt_generation": 23,
                "endpoint_sha256": "b" * 64,
                "prediction_sha256": sha256_json(prediction),
                "handoff_sha256": sha256_json(handoff),
            }
            attempts = []

            def grade(_config, request):
                attempts.append((request.generation, request.grader_attempt))
                if request.grader_attempt == 1:
                    raise RetryableGraderError(
                        "retry",
                        failure_class="test_retry",
                        attempt_directory=root / "attempt-1",
                    )
                return {
                    "instance_id": "owner__repo-0000",
                    "arm": "native",
                    "resolved": False,
                    "failure_class": None,
                    "report_sha256": "d" * 64,
                }

            with patch(
                "swebench_triad_eval.production.run_official_grader",
                side_effect=grade,
            ):
                outcome = runtime.grade(
                    key=CellKey(0, "native"),
                    accepted=accepted,
                    prediction=prediction,
                    handoff=handoff,
                )
            self.assertFalse(outcome["resolved"])
            self.assertEqual(attempts, [(23, 1), (23, 2)])

    def test_spawn_failure_grade_reconcile_keeps_one_terminal_ledger_event(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            config_path, payload = production_config(root)
            payload["docker"] = dict(payload["docker"])
            payload["docker"]["socket"] = str(DOCKER_SOCKET)
            payload["grader"] = dict(payload["grader"])
            payload["grader"]["max_attempts"] = 1
            payload["grader"]["python_executable"] = str(Path(sys.executable))
            config_path.write_bytes(canonical_json_bytes(payload))
            dataset_path = Path(payload["assets"]["dataset_jsonl"])
            dataset_path.write_text(
                json.dumps({"instance_id": "owner__repo-0000"}) + "\n",
                encoding="utf-8",
            )
            Path(payload["assets"]["harness_root"]).mkdir()
            config = ProductionRunConfig.load(config_path)
            runtime = LinuxProductionRuntime(config, config.configs)
            prediction = {
                "instance_id": "owner__repo-0000",
                "model_name_or_path": "Qwen3.5-4B",
                "model_patch": "",
            }
            handoff = {
                "prediction_sha256": sha256_json(prediction),
                "official_resolved": None,
                "grader_revision": HARNESS_COMMIT,
            }
            accepted = {
                "schema": "swebench_triad_accepted_cell_v1",
                "cell": {"task_index": 0, "arm": "native"},
                "instance_id": "owner__repo-0000",
                "manifest_cell_sha256": "a" * 64,
                "attempt_generation": 23,
                "endpoint_sha256": "b" * 64,
                "prediction_sha256": sha256_json(prediction),
                "handoff_sha256": sha256_json(handoff),
            }
            accepted_root = config.run_root / "state" / "accepted"
            accepted_root.mkdir(parents=True)
            (accepted_root / "0000-native.json").write_bytes(
                canonical_json_bytes(accepted)
            )
            environment_receipt = {
                "harness_commit": HARNESS_COMMIT,
                "harness_tree": HARNESS_TREE,
                "dataset_sha256": PRODUCTION_DATASET_PINS.jsonl_sha256,
                "docker_socket": str(DOCKER_SOCKET),
            }
            with patch(
                "swebench_triad_eval.official_grader.verify_grader_environment",
                return_value=environment_receipt,
            ), patch(
                "swebench_triad_eval.official_grader.subprocess.Popen",
                side_effect=OSError("simulated spawn failure"),
            ) as popen, patch.object(
                runtime, "grader_container_ids", return_value=[]
            ), self.assertRaises(RetryableGraderError) as raised:
                runtime.grade(
                    key=CellKey(0, "native"),
                    accepted=accepted,
                    prediction=prediction,
                    handoff=handoff,
                )
            self.assertEqual(raised.exception.failure_class, "grader_spawn_failure")
            self.assertEqual(popen.call_count, 1)
            ledger_path = config.run_root / "full" / "command-exit-ledger.jsonl"
            events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in events], ["start", "exit"])
            with self.assertRaisesRegex(
                RuntimeError, "does not cover every accepted cell"
            ):
                runtime.command_ledger_audit()

    def test_task_residue_uses_process_mount_and_rootfs_censuses(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            with patch.object(
                runtime, "owned_container_ids", return_value=["container-1"]
            ), patch.object(
                runtime, "task_image_identity", return_value=None
            ), patch.object(
                runtime, "cgroup_paths", return_value=["0000-native"]
            ), patch.object(
                runtime, "cgroup_process_ids", return_value=[101, 102]
            ), patch.object(
                runtime,
                "mount_records_under",
                return_value=[
                    {"fs_type": "tmpfs", "mount_point": "/task/workspace"},
                    {"fs_type": "bind", "mount_point": "/task/source"},
                ],
            ), patch.object(
                runtime, "verify_task_rootfs", return_value={"status": "PASS"}
            ):
                residue = runtime.audit_residue(0)
            self.assertEqual(residue["containers"], 1)
            self.assertEqual(residue["processes"], 2)
            self.assertEqual(residue["tmpfs_mounts"], 1)
            self.assertEqual(residue["mounts"], 2)
            self.assertTrue(residue["rootfs_attested"])

    def test_container_cleanup_requires_exact_owner_labels_and_name(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            docker = Mock()
            with patch.object(runtime, "docker", return_value=docker), patch.object(
                runtime,
                "container_record",
                return_value={
                    "Name": "/foreign-amg-sbv-triad-0000-native-g00000001",
                    "Config": {"Labels": {}},
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "unowned"):
                    runtime.remove_owned_container_id("foreign")
            docker.run.assert_not_called()

    def test_task_root_cleanup_refuses_any_surviving_mount(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            task_root = runtime.task_root_path(0)
            task_root.mkdir(parents=True)
            with patch.object(
                runtime, "verify_task_rootfs", return_value={"status": "PASS"}
            ), patch.object(
                runtime,
                "mount_records_under",
                return_value=[{"fs_type": "tmpfs", "mount_point": str(task_root)}],
            ):
                with self.assertRaisesRegex(RuntimeError, "live mount"):
                    runtime.remove_inactive_task_root(0)
            self.assertTrue(task_root.is_dir())

    def test_startup_reconciliation_evicts_images_and_task_roots(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            with patch.object(
                runtime, "owned_container_ids", return_value=[]
            ), patch.object(
                runtime, "cgroup_paths", return_value=[]
            ), patch.object(
                runtime, "cgroup_process_ids", return_value=[]
            ), patch.object(
                runtime, "mount_records_under", return_value=[]
            ), patch.object(
                runtime, "staged_task_indices", return_value=[0]
            ), patch.object(
                runtime, "task_root_indices", return_value=[0]
            ), patch.object(
                runtime,
                "loaded_task_image_identities",
                return_value=[
                    ("swebench/image:task", "sha256:" + "a" * 64)
                ],
            ), patch.object(
                runtime,
                "task_image_identity",
                return_value=("swebench/image:task", "sha256:" + "a" * 64),
            ), patch.object(
                runtime, "evict_image", return_value={"status": "evicted"}
            ) as evict, patch.object(
                runtime, "remove_inactive_task_root", return_value=True
            ), patch.object(
                runtime,
                "global_residue_snapshot",
                return_value={"owned_containers": 0},
            ):
                receipt = runtime.reconcile_startup(task_indices=(0,))
            evict.assert_called_once()
            self.assertEqual(receipt["removed_task_roots"], [0])

    def test_loaded_image_census_does_not_depend_on_a_stage_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            image = "swebench/image:task"
            image_id = "sha256:" + "a" * 64
            docker = Mock()
            docker.run.return_value = subprocess.CompletedProcess(
                ["docker", "image", "ls"],
                0,
                f"{image}\t{image_id}\n",
                "",
            )
            with patch.object(runtime, "docker", return_value=docker), patch.object(
                runtime,
                "certified_image_identities",
                return_value={image: image_id},
            ):
                self.assertEqual(
                    runtime.loaded_task_image_identities(), [(image, image_id)]
                )

    def test_cleanup_restores_holders_before_certifying_retention(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            pod = {
                "job": runtime.section("pod")["job"],
                "boot_id": runtime.section("pod")["boot_id"],
            }
            zero = {
                "active_owned_processes": 0,
                "active_cgroups": 0,
                "active_tmpfs_mounts": 0,
                "active_mounts": 0,
                "active_scratch_paths": 0,
                "loaded_task_images": 0,
                "owned_containers": 0,
            }
            with patch.object(
                runtime, "owned_container_ids", return_value=[]
            ), patch.object(
                runtime, "cgroup_paths", return_value=[]
            ), patch.object(
                runtime, "reconcile_startup", return_value={"status": "PASS"}
            ), patch.object(
                runtime, "stop_model_process", return_value={"status": "stopped"}
            ), patch.object(
                runtime,
                "restore_holders",
                return_value={"status": "PASS", "snapshot": {}},
            ) as restore, patch.object(
                runtime, "pod_snapshot", return_value=pod
            ), patch.object(
                runtime, "global_residue_snapshot", return_value=zero
            ):
                receipt = runtime.cleanup()
            restore.assert_called_once_with()
            self.assertTrue(receipt["holders_restored"])
            self.assertEqual(receipt["owned_residue"], 0)

    def test_command_ledger_requires_start_success_pair_and_hashes_ledger(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            accepted_root = runtime.config.run_root / "state" / "accepted"
            accepted_root.mkdir(parents=True)
            accepted = {
                "schema": "swebench_triad_accepted_cell_v1",
                "cell": {"task_index": 0, "arm": "native"},
                "instance_id": "owner__repo-0000",
                "manifest_cell_sha256": "a" * 64,
                "attempt_generation": 7,
                "endpoint_sha256": "b" * 64,
                "prediction_sha256": "c" * 64,
                "handoff_sha256": "d" * 64,
            }
            (accepted_root / "0000-native.json").write_bytes(
                canonical_json_bytes(accepted)
            )
            ledger_path = runtime.config.run_root / "full" / "command-exit-ledger.jsonl"
            ledger_path.parent.mkdir(parents=True)
            binding_sha256 = sha256_json(
                {
                    "schema": "swebench_triad_grader_binding_v1",
                    "task_index": 0,
                    "arm": "native",
                    "generation": 7,
                    "grader_attempt": 1,
                    "instance_id": "owner__repo-0000",
                    "prediction_sha256": "c" * 64,
                    "harness_commit": HARNESS_COMMIT,
                    "harness_tree": HARNESS_TREE,
                    "dataset_sha256": PRODUCTION_DATASET_PINS.jsonl_sha256,
                    "namespace": "swebench",
                    "timeout_seconds": 1_800,
                }
            )
            common = {
                "schema": "swebench_triad_command_exit_event_v1",
                "binding_sha256": binding_sha256,
                "task_index": 0,
                "arm": "native",
                "generation": 7,
                "grader_attempt": 1,
                "prediction_sha256": "c" * 64,
            }
            exit_row = {
                **common,
                "event_id": binding_sha256 + ":exit",
                "event": "exit",
                "process_result": {
                    "schema": "swebench_triad_grader_process_v1",
                    "status": "completed",
                    "returncode": 0,
                    "stdout_sha256": "f" * 64,
                    "stderr_sha256": "0" * 64,
                },
            }
            ledger_path.write_bytes(canonical_json_bytes(exit_row) + b"\n")
            with self.assertRaisesRegex(RuntimeError, "start/terminal"):
                runtime.command_ledger_audit()

            start_row = {
                **common,
                "event_id": binding_sha256 + ":start",
                "event": "start",
                "command": [],
                "cwd": "",
                "environment": {},
            }
            attempt_root = (
                Path(runtime.section("grader")["output_root"])
                / "0000-native"
                / "generation-00000007"
                / f"attempt-000001-{binding_sha256}"
            )
            start_row["command"] = [
                str(runtime.section("grader")["python_executable"]),
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                str(runtime.section("assets")["dataset_jsonl"]),
                "--split",
                "test",
                "--instance_ids",
                "owner__repo-0000",
                "--predictions_path",
                str(attempt_root / "prediction.jsonl"),
                "--max_workers",
                "1",
                "--timeout",
                "1800",
                "--force_rebuild",
                "false",
                "--cache_level",
                "instance",
                "--clean",
                "false",
                "--namespace",
                "swebench",
                "--run_id",
                "amg-sbv-0000-native-g00000007-a000001-" + binding_sha256[:16],
            ]
            start_row["cwd"] = str(attempt_root)
            start_row["environment"] = {
                "PYTHONPATH": str(runtime.section("assets")["harness_root"]),
                "PYTHONNOUSERSITE": "1",
                "DOCKER_HOST": "unix://" + str(runtime.section("docker")["socket"]),
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            ledger_bytes = (
                canonical_json_bytes(start_row)
                + b"\n"
                + canonical_json_bytes(exit_row)
                + b"\n"
            )
            ledger_path.write_bytes(ledger_bytes)
            receipt = runtime.command_ledger_audit()
            self.assertEqual(
                receipt["sha256"], hashlib.sha256(ledger_bytes).hexdigest()
            )
            self.assertEqual(receipt["covered_cells"], 1)

    def test_model_cleanup_rejects_surviving_engine_worker(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            serving = runtime.section("serving")
            snapshot = {
                "pgid": serving["pid"],
                "processes": {
                    str(serving["pid"]): serving["start_ticks"],
                    str(serving["pid"] + 1): serving["start_ticks"] + 1,
                },
                "gpu_pids": [serving["pid"] + 1],
            }
            residue = {
                "live_processes": [serving["pid"] + 1],
                "live_gpu_pids": [serving["pid"] + 1],
                "live_matching_processes": [serving["pid"] + 1],
                "endpoint_alive": False,
            }
            with patch(
                "swebench_triad_eval.production.linux_process_start_ticks",
                return_value=serving["start_ticks"],
            ), patch(
                "swebench_triad_eval.production.require_process_identity",
                return_value=(
                    str(runtime.section("assets")["model_root"])
                    + " "
                    + str(serving["model_id"])
                ),
            ), patch.object(
                runtime, "model_process_tree_snapshot", return_value=snapshot
            ), patch.object(
                runtime, "model_shutdown_residue", return_value=residue
            ), patch(
                "swebench_triad_eval.production.os.killpg"
            ) as killpg, self.assertRaisesRegex(RuntimeError, "process tree"):
                runtime.stop_model_process(timeout_seconds=0.0)
            killpg.assert_any_call(serving["pid"], signal.SIGTERM)

    def test_model_shutdown_residue_freshly_censuses_late_gpu_member(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            serving = runtime.section("serving")
            late_pid = serving["pid"] + 17
            snapshot = {
                "pgid": serving["pid"],
                "processes": {str(serving["pid"]): serving["start_ticks"]},
                "gpu_pids": [serving["pid"]],
            }
            with patch.object(
                runtime, "process_group_members", return_value={late_pid}
            ), patch.object(
                runtime, "gpu_compute_pids", return_value={late_pid}
            ), patch.object(
                runtime, "matching_model_processes", return_value=set()
            ), patch.object(runtime, "endpoint_is_alive", return_value=False):
                residue = runtime.model_shutdown_residue(snapshot)
            self.assertEqual(residue["live_processes"], [late_pid])
            self.assertEqual(residue["live_gpu_pids"], [late_pid])

    def test_model_process_group_excludes_zombie_members(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            pgid = runtime.section("serving")["pid"]
            zombie_pid = pgid + 17
            with patch(
                "swebench_triad_eval.production.Path.iterdir",
                return_value=[Path(f"/proc/{zombie_pid}")],
            ), patch.object(
                runtime, "process_group_id", return_value=pgid
            ), patch.object(
                runtime, "process_state", return_value="Z", create=True
            ):
                self.assertEqual(runtime.process_group_members(pgid), set())

    def test_model_cleanup_terminates_group_when_recorded_leader_is_dead(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            serving = runtime.section("serving")
            child = serving["pid"] + 1
            groups = iter(({child}, {child}, set()))

            with patch(
                "swebench_triad_eval.production.linux_process_start_ticks",
                side_effect=RuntimeError("leader is gone"),
            ), patch.object(
                runtime, "process_group_members", side_effect=lambda _pgid: next(groups)
            ), patch.object(
                runtime, "gpu_compute_pids", return_value={child}
            ), patch.object(
                runtime, "matching_model_processes", return_value=set()
            ), patch.object(
                runtime, "endpoint_is_alive", return_value=False
            ), patch(
                "swebench_triad_eval.production.os.killpg"
            ) as killpg:
                receipt = runtime.stop_model_process(timeout_seconds=0.0)

            self.assertEqual(receipt["status"], "stopped")
            killpg.assert_any_call(serving["pid"], signal.SIGTERM)
            killpg.assert_any_call(serving["pid"], signal.SIGKILL)

    def test_model_cleanup_treats_zombie_leader_as_dead_and_reaps_live_child(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            serving = runtime.section("serving")
            child = serving["pid"] + 1
            groups = iter(({child}, {child}, set()))

            with patch(
                "swebench_triad_eval.production.linux_process_start_ticks",
                return_value=serving["start_ticks"],
            ), patch.object(
                runtime, "process_state", return_value="Z"
            ), patch.object(
                runtime, "process_group_members", side_effect=lambda _pgid: next(groups)
            ), patch.object(
                runtime, "gpu_compute_pids", return_value={child}
            ), patch.object(
                runtime, "matching_model_processes", return_value=set()
            ), patch.object(
                runtime, "endpoint_is_alive", return_value=False
            ), patch(
                "swebench_triad_eval.production.require_process_identity"
            ) as identity, patch(
                "swebench_triad_eval.production.os.killpg"
            ) as killpg:
                receipt = runtime.stop_model_process(timeout_seconds=0.0)

            identity.assert_not_called()
            self.assertEqual(receipt["status"], "stopped")
            killpg.assert_any_call(serving["pid"], signal.SIGTERM)
            killpg.assert_any_call(serving["pid"], signal.SIGKILL)

    def test_holder_snapshot_rejects_stale_auto_state_and_unknown_gpu_pid(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self.make_runtime(root)
            auto_path = root / "auto.state"
            fallback_path = root / "fallback.json"
            auto_path.write_text(
                time.strftime("%Y-%m-%d %H:%M:%S")
                + " mode=hold pid=10 gpu=1 cpu=1 work=4096^2\n"
            )
            fallback_path.write_bytes(
                canonical_json_bytes(
                    {
                        "timestamp": int(time.time()),
                        "parent_pid": 20,
                        "cpu_workers": {"0": 21},
                        "gpu_workers": {"0": 22},
                        "cpu_duty": 0.9,
                        "gpu_duty": 0.18,
                        "mode": "hold",
                    }
                )
            )
            old = time.time() - 300
            os.utime(auto_path, (old, old))
            with self.assertRaisesRegex(RuntimeError, "auto-yield holder state is stale"):
                runtime.holder_snapshot(
                    auto_state_path=auto_path,
                    fallback_state_path=fallback_path,
                    sample_count=1,
                    sample_gap=0.0,
                )

            os.utime(auto_path, None)
            cpu_ticks = {12: 100, 21: 200}

            def next_ticks(pid):
                cpu_ticks[pid] += 1
                return cpu_ticks[pid]

            with patch(
                "swebench_triad_eval.production.require_process_identity",
                side_effect=lambda pid, _ticks, _label: (
                    "_heavy_holder.py" if pid == 10 else "non_yield_gpu_cpu_fallback_holder.py"
                ),
            ), patch(
                "swebench_triad_eval.production.linux_process_start_ticks",
                side_effect=lambda pid: pid + 1000,
            ), patch.object(
                runtime,
                "direct_process_children",
                side_effect=lambda pid: {11, 12} if pid == 10 else {21, 22},
            ), patch.object(
                runtime,
                "process_command_line",
                return_value="python -c from multiprocessing.spawn import spawn_main",
            ), patch.object(
                runtime, "process_parent_pid", side_effect=lambda pid: 10 if pid < 20 else 20
            ), patch.object(
                runtime, "process_cpu_ticks", side_effect=next_ticks
            ), patch.object(
                runtime, "gpu_compute_pids", return_value={11, 22, 99}
            ), patch.object(
                runtime, "gpu_utilization_sample", return_value={0: 15}
            ), self.assertRaisesRegex(RuntimeError, "unknown GPU holder process"):
                runtime.holder_snapshot(
                    auto_state_path=auto_path,
                    fallback_state_path=fallback_path,
                    sample_count=2,
                    sample_gap=0.0,
                )

            cpu_ticks = {12: 100, 21: 200}

            def slow_ticks(pid):
                cpu_ticks[pid] += 1
                return cpu_ticks[pid]

            with patch(
                "swebench_triad_eval.production.require_process_identity",
                side_effect=lambda pid, _ticks, _label: (
                    "_heavy_holder.py"
                    if pid == 10
                    else "non_yield_gpu_cpu_fallback_holder.py"
                ),
            ), patch(
                "swebench_triad_eval.production.linux_process_start_ticks",
                side_effect=lambda pid: pid + 1000,
            ), patch.object(
                runtime,
                "direct_process_children",
                side_effect=lambda pid: {11, 12} if pid == 10 else {21, 22},
            ), patch.object(
                runtime,
                "process_command_line",
                return_value="python -c from multiprocessing.spawn import spawn_main",
            ), patch.object(
                runtime,
                "process_parent_pid",
                side_effect=lambda pid: 10 if pid < 20 else 20,
            ), patch.object(
                runtime, "process_cpu_ticks", side_effect=slow_ticks
            ), patch.object(
                runtime, "gpu_compute_pids", return_value={11, 22}
            ), patch.object(
                runtime, "gpu_utilization_sample", return_value={0: 6}
            ), patch.object(
                runtime, "cpu_capacity_count", return_value=10, create=True
            ), patch(
                "swebench_triad_eval.production.os.sysconf", return_value=100
            ), patch(
                "swebench_triad_eval.production.time.monotonic",
                side_effect=(100.0, 101.0, 102.0),
            ), patch(
                "swebench_triad_eval.production.time.sleep"
            ), self.assertRaisesRegex(RuntimeError, "retention floor"):
                runtime.holder_snapshot(
                    auto_state_path=auto_path,
                    fallback_state_path=fallback_path,
                    sample_count=3,
                    sample_gap=1.0,
                )

    def test_holder_retention_requires_measured_cpu_and_gpu_floors(self):
        with self.assertRaisesRegex(RuntimeError, "CPU.*retention floor"):
            LinuxProductionRuntime.require_holder_retention(
                [4.9, 20.0],
                [{0: 15}, {0: 15}, {0: 15}],
                {0},
            )
        with self.assertRaisesRegex(RuntimeError, "GPU.*retention floor"):
            LinuxProductionRuntime.require_holder_retention(
                [20.0, 20.0],
                [{0: 6}, {0: 6}, {0: 6}],
                {0},
            )


if __name__ == "__main__":
    unittest.main()
