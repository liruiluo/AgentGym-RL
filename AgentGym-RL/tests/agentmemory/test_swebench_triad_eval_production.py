from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import time
import unittest
from typing import Any
from unittest.mock import Mock, patch

import test_swebench_triad_eval_cli as cli_test_support
from test_swebench_triad_eval_cli import (
    preflight_expectations,
    production_config,
    valid_preflight_snapshot,
)

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
    exact_token_proxy_target,
    shared_model_pool_snapshot_receipt,
    summarize_task4_receipt,
    validate_exact_token_proxy_config,
)
from swebench_triad_eval.cli import validate_preflight_snapshot
from swebench_triad_eval.state import (
    CellKey,
    OwnerIdentity,
    RuntimeLaneToken,
    sha256_json,
)


def runtime_slot(runtime, task_index=None, slot_index=None):
    if slot_index is None:
        slot_index = 0 if task_index is None else runtime.task_slots[task_index]
    return RuntimeLaneToken(
        driver_key="a" * 64,
        lease_id="b" * 64,
        owner=OwnerIdentity("host", "boot", 101, 1001),
        task_index=task_index,
        slot_index=slot_index,
        server_port=runtime.config.server_port(slot_index),
        generation=1,
        fencing_token="c" * 64,
    )


def startup_slots(runtime):
    return tuple(
        runtime_slot(runtime, slot_index=slot_index)
        for slot_index in range(runtime.config.task_slots_per_replica)
    )


class RecordingProductionRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def preflight(self):
        self.calls.append(("preflight",))
        return {"snapshot": True}

    def stage_task(self, task_index, *, slot):
        self.calls.append(("stage", task_index))
        return {"task_index": task_index}

    def reconcile_cell(self, config, *, generation, before_preflight, slot):
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

    def reconcile_startup(self, *, task_indices, slots):
        self.calls.append(("reconcile_startup", tuple(task_indices)))
        return {"reconciled": True}

    def reconcile_unbound_loaded_images(self):
        self.calls.append(("reconcile_unbound_loaded_images",))
        return {"status": "PASS", "remaining_images": 0}

    def run_cell(self, config, stage, *, generation, slot):
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

    def audit_residue(self, task_index, *, slot):
        self.calls.append(("audit", task_index))
        return {"containers": 0}

    def evict_task(self, task_index, stage, *, slot):
        self.calls.append(("evict", task_index, stage))
        return {"evicted": True}

    def cleanup(self, *, slots):
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
        slot = RuntimeLaneToken(
            driver_key="a" * 64,
            lease_id="b" * 64,
            owner=OwnerIdentity("host", "boot", 101, 1001),
            task_index=7,
            slot_index=0,
            server_port=self.config.server_port(0),
            generation=1,
            fencing_token="c" * 64,
        )
        global_slot = RuntimeLaneToken(
            **{**slot.__dict__, "task_index": None}
        )
        stage = self.operations.stage_task(7, slot=slot)
        config = self.config.configs[7 * 3 + 2]
        self.assertEqual(self.operations.preflight(), {"snapshot": True})
        self.assertEqual(
            self.operations.reconcile_cell(
                config, generation=19, before_preflight=True, slot=slot
            ),
            {"reconciled": True},
        )
        self.assertEqual(
            self.operations.reconcile_startup(
                task_indices=(7,), slots=(global_slot,)
            ),
            {"reconciled": True},
        )
        self.assertEqual(
            self.operations.run_cell(
                config, stage, generation=19, slot=slot
            ),
            {"endpoint": True},
        )
        key = CellKey(7, "amg_memory")
        self.assertEqual(
            self.operations.grade(
                key=key,
                accepted={},
                prediction={},
                handoff={},
                slot=slot,
            ),
            {"outcome": True},
        )
        self.assertEqual(
            self.operations.audit_residue(7, slot=slot), {"containers": 0}
        )
        self.assertEqual(
            self.operations.evict_task(7, stage, slot=slot), {"evicted": True}
        )
        self.assertEqual(
            self.operations.cleanup(slots=(global_slot,)),
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

    def test_cgroup_path_census_holds_the_shared_structure_lock(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            with patch(
                "swebench_triad_eval.production.cgroup_structure_lock"
            ) as structure_lock, patch.object(
                runtime,
                "_cgroup_paths_unlocked",
                return_value=["0000-native"],
            ) as census:
                self.assertEqual(runtime.cgroup_paths(0), ["0000-native"])
            structure_lock.assert_called_once_with()
            census.assert_called_once_with(0)

    def test_cgroup_process_census_holds_the_shared_structure_lock(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            with patch(
                "swebench_triad_eval.production.cgroup_structure_lock"
            ) as structure_lock, patch.object(
                runtime,
                "_cgroup_paths_unlocked",
                return_value=[],
            ) as paths:
                self.assertEqual(runtime.cgroup_process_ids(0), [])
            structure_lock.assert_called_once_with()
            paths.assert_called_once_with(0)

    def test_image_eviction_uses_certified_id_when_census_alias_is_missing(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            image_id = "sha256:" + "c" * 64
            aliases = {
                "swebench/task-a:latest": image_id,
                "swebench/task-b:latest": image_id,
            }
            present = True
            docker = Mock()
            docker.output_text.side_effect = lambda value: value
            docker.is_missing_image.side_effect = lambda result: result.returncode != 0

            def inspect(reference):
                if reference == image_id and present:
                    return subprocess.CompletedProcess(
                        ["docker", "inspect", reference], 0, image_id, ""
                    )
                return subprocess.CompletedProcess(
                    ["docker", "inspect", reference],
                    1,
                    "",
                    "No such image",
                )

            def run(*arguments):
                nonlocal present
                self.assertEqual(arguments, ("image", "rm", "--force", image_id))
                present = False
                return subprocess.CompletedProcess(["docker", *arguments], 0, image_id, "")

            docker.inspect.side_effect = inspect
            docker.run.side_effect = run
            with patch.object(
                runtime, "certified_image_identities", return_value=aliases
            ), patch.object(runtime, "docker", return_value=docker), patch.object(
                runtime, "image_container_ids", return_value=[]
            ):
                receipt = runtime.evict_image("swebench/task-a:latest", image_id)
            self.assertEqual(receipt["status"], "evicted")
            self.assertEqual(receipt["certified_aliases_removed"], sorted(aliases))
            docker.run.assert_called_once_with("image", "rm", "--force", image_id)

    def test_unbound_loaded_image_reconciliation_evicts_every_orphan(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            identities = [
                ("swebench/image-a:latest", "sha256:" + "a" * 64),
                ("swebench/image-b:latest", "sha256:" + "b" * 64),
            ]
            with patch.object(
                runtime,
                "loaded_task_image_identities",
                side_effect=[identities, []],
            ), patch.object(
                runtime,
                "evict_image",
                side_effect=lambda image, image_id: {
                    "status": "evicted",
                    "image": image,
                    "image_id": image_id,
                },
            ) as evict:
                receipt = runtime.reconcile_unbound_loaded_images()
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["remaining_images"], 0)
            self.assertEqual(
                evict.call_args_list,
                [
                    unittest.mock.call(*identities[0]),
                    unittest.mock.call(*identities[1]),
                ],
            )

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
                    slot=runtime_slot(runtime, 0),
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
                    slot=runtime_slot(runtime, 0),
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
                residue = runtime.audit_residue(
                    0, slot=runtime_slot(runtime, 0)
                )
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

    def test_container_cleanup_accepts_exact_slot_fencing_labels(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            docker = Mock()
            docker.run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"", stderr=b""
            )
            labels = {
                "foreign.label": "retained",
                "amg.owner": "amg-swebench-triad-eval-0816",
                "amg.task_index": "0000",
                "amg.arm": "native",
                "amg.generation": "00000001",
                "amg.slot_index": "0",
                "amg.server_port": str(runtime.config.server_port(0)),
                "amg.lane_generation": "00000003",
            }
            with patch.object(runtime, "docker", return_value=docker), patch.object(
                runtime,
                "container_record",
                return_value={
                    "Name": "/amg-sbv-triad-0000-native-g00000001",
                    "Config": {"Labels": labels},
                },
            ):
                receipt = runtime.remove_owned_container_id("owned")
            self.assertTrue(receipt["removed"])
            docker.run.assert_called_once_with(
                "container", "rm", "--force", "owned"
            )

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
                receipt = runtime.reconcile_startup(
                    task_indices=(0,), slots=startup_slots(runtime)
                )
            evict.assert_called_once()
            self.assertEqual(receipt["removed_task_roots"], [0])

    def test_shared_pool_startup_defers_only_foreign_loaded_images(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            foreign = ("swebench/foreign:task", "sha256:" + "b" * 64)
            patches = (
                patch.object(runtime, "owned_container_ids", return_value=[]),
                patch.object(runtime, "cgroup_paths", return_value=[]),
                patch.object(runtime, "cgroup_process_ids", return_value=[]),
                patch.object(runtime, "mount_records_under", return_value=[]),
                patch.object(runtime, "staged_task_indices", return_value=[]),
                patch.object(runtime, "task_root_indices", return_value=[]),
                patch.object(
                    runtime, "loaded_task_image_identities", return_value=[foreign]
                ),
                patch.object(
                    runtime,
                    "global_residue_snapshot",
                    return_value={"loaded_task_images": 1},
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[
                5
            ], patches[6], patches[7]:
                with self.assertRaisesRegex(RuntimeError, "no durable task-stage"):
                    runtime.reconcile_startup(
                        task_indices=(0,), slots=startup_slots(runtime)
                    )
                receipt = runtime.reconcile_startup(
                    task_indices=(0,),
                    allow_foreign_loaded_images=True,
                    slots=startup_slots(runtime),
                )
            self.assertEqual(
                receipt["foreign_loaded_images"],
                [{"image": foreign[0], "image_id": foreign[1]}],
            )

    def test_startup_reconciliation_matches_duplicate_aliases_by_config_digest(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            image_id = "sha256:" + "c" * 64
            staged = ("swebench/task-b:latest", image_id)
            canonical_census = ("swebench/task-a:latest", image_id)
            with patch.object(
                runtime, "owned_container_ids", return_value=[]
            ), patch.object(runtime, "cgroup_paths", return_value=[]), patch.object(
                runtime, "cgroup_process_ids", return_value=[]
            ), patch.object(runtime, "mount_records_under", return_value=[]), patch.object(
                runtime, "staged_task_indices", return_value=[0]
            ), patch.object(runtime, "task_root_indices", return_value=[0]), patch.object(
                runtime,
                "loaded_task_image_identities",
                return_value=[canonical_census],
            ), patch.object(
                runtime, "task_image_identity", return_value=staged
            ), patch.object(
                runtime, "evict_image", return_value={"status": "evicted"}
            ) as evict, patch.object(
                runtime, "remove_inactive_task_root", return_value=True
            ), patch.object(
                runtime,
                "global_residue_snapshot",
                return_value={"owned_containers": 0},
            ):
                receipt = runtime.reconcile_startup(
                    task_indices=(0,), slots=startup_slots(runtime)
                )
            evict.assert_called_once_with(*staged)
            self.assertEqual(receipt["foreign_loaded_images"], [])

    def test_startup_reconciliation_ignores_retired_duplicate_digest_history(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            image_id = "sha256:" + "d" * 64
            stages = runtime.config.run_root / "control" / "stages"
            evictions = runtime.config.run_root / "control" / "evictions"
            stages.mkdir(parents=True)
            evictions.mkdir(parents=True)
            for task_index, image in (
                (0, "swebench/retired:latest"),
                (1, "swebench/resuming:latest"),
            ):
                (stages / f"task-{task_index:04d}.json").write_bytes(
                    canonical_json_bytes(
                        {
                            "schema": "swebench_triad_task_stage_v1",
                            "task_index": task_index,
                            "binding": {
                                "image": image,
                                "config_digest": image_id,
                            },
                        }
                    )
                )
            (evictions / "task-0000.json").write_bytes(
                canonical_json_bytes(
                    {
                        "schema": "swebench_triad_task_eviction_v1",
                        "task_index": 0,
                        "instance_id": runtime.by_task[0][0].task.task_id,
                        "readiness": {"status": "ready"},
                        "image": {"status": "evicted"},
                        "task_root_removed": True,
                        "certified_blobs_retained": True,
                        "repository_mirror_retained": True,
                        "slot_index": 0,
                        "server_port": runtime.config.server_port(0),
                        "lane_generation": 1,
                    }
                )
            )
            with patch.object(
                runtime, "owned_container_ids", return_value=[]
            ), patch.object(runtime, "cgroup_paths", return_value=[]), patch.object(
                runtime, "cgroup_process_ids", return_value=[]
            ), patch.object(runtime, "mount_records_under", return_value=[]), patch.object(
                runtime, "task_root_indices", return_value=[1]
            ), patch.object(
                runtime,
                "loaded_task_image_identities",
                return_value=[("swebench/retired:latest", image_id)],
            ), patch.object(
                runtime, "evict_image", return_value={"status": "evicted"}
            ) as evict, patch.object(
                runtime, "remove_inactive_task_root", return_value=True
            ), patch.object(
                runtime,
                "global_residue_snapshot",
                return_value={"owned_containers": 0},
            ):
                receipt = runtime.reconcile_startup(
                    task_indices=(0, 1), slots=startup_slots(runtime)
                )

            evict.assert_called_once_with("swebench/resuming:latest", image_id)
            self.assertEqual(receipt["removed_task_roots"], [1])
            self.assertEqual(receipt["foreign_loaded_images"], [])

    def test_startup_reconciliation_rejects_two_active_duplicate_digest_claims(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = self.make_runtime(Path(raw))
            image_id = "sha256:" + "e" * 64
            identities = {
                0: ("swebench/active-a:latest", image_id),
                1: ("swebench/active-b:latest", image_id),
            }
            with patch.object(
                runtime, "owned_container_ids", return_value=[]
            ), patch.object(runtime, "cgroup_paths", return_value=[]), patch.object(
                runtime, "cgroup_process_ids", return_value=[]
            ), patch.object(runtime, "mount_records_under", return_value=[]), patch.object(
                runtime, "staged_task_indices", return_value=[0, 1]
            ), patch.object(
                runtime, "retired_task_indices", return_value=[]
            ), patch.object(runtime, "task_root_indices", return_value=[0, 1]), patch.object(
                runtime,
                "loaded_task_image_identities",
                return_value=[identities[0]],
            ), patch.object(
                runtime,
                "task_image_identity",
                side_effect=lambda task_index: identities[task_index],
            ), self.assertRaisesRegex(RuntimeError, "multiple active task leases"):
                runtime.reconcile_startup(
                    task_indices=(0, 1), slots=startup_slots(runtime)
                )

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
                receipt = runtime.cleanup(slots=startup_slots(runtime))
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


class SharedModelPoolProductionTest(unittest.TestCase):
    PROC_TCP_HEADER = (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when "
        "retrnsmt   uid  timeout inode"
    )

    @classmethod
    def proc_tcp_row(
        cls,
        address: str,
        *,
        port: str = "4665",
        inode: int = 99,
        state: str = "0A",
    ) -> str:
        remote_address = "0" * len(address)
        return (
            f"   0: {address}:{port} {remote_address}:0000 {state} "
            "00000000:00000000 00:00000000 00000000 1000 0 "
            f"{inode} 1 0000000000000000 100 0 0 10 0"
        )

    @classmethod
    def proc_table_reader(
        cls,
        *,
        tcp_rows: tuple[str, ...] = (),
        tcp6_rows: tuple[str, ...] = (),
    ):
        tables = {
            Path("/proc/net/tcp"): "\n".join(
                (cls.PROC_TCP_HEADER, *tcp_rows, "")
            ),
            Path("/proc/net/tcp6"): "\n".join(
                (cls.PROC_TCP_HEADER, *tcp6_rows, "")
            ),
        }

        def read_text(path, *, encoding=None, errors=None):
            if encoding != "ascii" or errors is not None:
                raise AssertionError("proc TCP tables require strict ASCII reads")
            return tables[path]

        return read_text

    @staticmethod
    def shared_config(root: Path) -> tuple[ProductionRunConfig, dict[str, Any]]:
        config_path, payload = production_config(root)
        payload["schema"] = "amg_swebench_triad_run_config_shared_pool_v3"
        payload["runtime"] = dict(payload["runtime"])
        payload["runtime"].pop("server_port")
        payload["runtime"].update(
            {
                "task_slots_per_replica": 2,
                "server_ports": [18103, 18111],
            }
        )
        payload["grader"] = {
            **payload["grader"],
            "global_max_concurrency": 8,
            "semaphore_root": str(root / "grader-semaphore"),
        }
        payload["pod"] = {
            **payload["pod"],
            "gpu_uuid": "GPU-shared-3",
        }
        payload["serving"] = {
            **payload["serving"],
            "base_url": "http://127.0.0.1:16383/v1",
            "model_id": "Qwen/Qwen3.5-4B",
            "pid": 303,
            "start_ticks": 3003,
        }
        payload["shared_model_pool"] = {
            "owner": "amg-external-eval-g-dp8-swe-0818",
            "readiness_path": str(root / "pool-readiness.json"),
            "readiness_sha256": "1" * 64,
            "marker_lease_path": str(root / "marker-lease.json"),
            "marker_lease_sha256": "2" * 64,
            "replica_index": 3,
            "replica_count": 8,
            "gpu_index": 3,
            "gpu_uuid": "GPU-shared-3",
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "3" * 40,
            "model_port": 18021,
            "proxy_port": 16383,
            "assignment_algorithm": "uint64_be(sha256(task_id)[:8]) % 8",
            "cleanup_policy": "retain_external_pool",
        }
        config_path.write_bytes(canonical_json_bytes(payload))
        return ProductionRunConfig.load(config_path), payload

    def test_cell_reconciliation_is_bound_to_its_exact_slot(self):
        with tempfile.TemporaryDirectory() as raw:
            config, _ = self.shared_config(Path(raw))
            runtime = LinuxProductionRuntime(config, config.configs)
            task_index = next(
                index for index, slot_index in runtime.task_slots.items()
                if slot_index == 1
            )
            cell = runtime.by_task[task_index][0]
            slot = runtime_slot(runtime, task_index)
            with patch.object(
                runtime, "owned_container_ids", return_value=[]
            ) as containers, patch.object(
                runtime, "cgroup_paths", return_value=[]
            ), patch.object(
                runtime, "mount_records_under", return_value=[]
            ), patch.object(
                runtime, "cgroup_process_ids", return_value=[]
            ):
                receipt = runtime.reconcile_cell(
                    cell,
                    generation=2,
                    before_preflight=False,
                    slot=slot,
                )
            containers.assert_called_once_with(
                task_index, cell.capability.arm.value, 1
            )
            self.assertEqual(receipt["slot_index"], 1)
            self.assertEqual(receipt["server_port"], config.server_port(1))

    def test_cell_reconciliation_rejects_a_future_lane_generation(self):
        with tempfile.TemporaryDirectory() as raw:
            config, _ = self.shared_config(Path(raw))
            runtime = LinuxProductionRuntime(config, config.configs)
            task_index = next(
                index for index, slot_index in runtime.task_slots.items()
                if slot_index == 1
            )
            cell = runtime.by_task[task_index][0]
            slot = runtime_slot(runtime, task_index)
            labels = {
                "amg.owner": "amg-swebench-triad-eval-0816",
                "amg.task_index": f"{task_index:04d}",
                "amg.arm": cell.capability.arm.value,
                "amg.generation": "00000001",
                "amg.slot_index": "1",
                "amg.server_port": str(slot.server_port),
                "amg.lane_generation": str(slot.generation + 1),
            }
            with patch.object(
                runtime, "owned_container_ids", return_value=["future"]
            ), patch.object(
                runtime,
                "container_record",
                return_value={"Config": {"Labels": labels}},
            ):
                with self.assertRaisesRegex(RuntimeError, "generation drifted"):
                    runtime.reconcile_cell(
                        cell,
                        generation=2,
                        before_preflight=False,
                        slot=slot,
                    )

    def test_exact_token_proxy_config_binds_listen_and_upstream_ports(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "gaia_vllm_token_proxy.py"
            source.write_text("print('proxy')\n", encoding="utf-8")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            config_path = root / "proxy-config.json"
            config = {
                "schema": "gaia_vllm_exact_token_proxy_config_v1",
                "listen_host": "127.0.0.1",
                "listen_port": 16383,
                "upstream_base_url": "http://127.0.0.1:18021",
                "upstream_model_id": "Qwen/Qwen3.5-4B",
                "upstream_model_revision": "3" * 40,
                "proxy_source_sha256": source_sha,
                "runtime_sha256": "4" * 64,
                "tokenizer_sha256": "5" * 64,
            }
            config_path.write_bytes(canonical_json_bytes(config))
            target = [sys.executable, str(source), "--config", str(config_path)]
            supervisor = [
                sys.executable,
                "/tmp/supervisor.py",
                "park-exec",
                "--command-json",
                json.dumps(target, separators=(",", ":")),
            ]
            self.assertEqual(exact_token_proxy_target(supervisor), target)
            receipt = validate_exact_token_proxy_config(
                target,
                model_port=18021,
                proxy_port=16383,
                model_id="Qwen/Qwen3.5-4B",
                model_revision="3" * 40,
                proxy_source_sha256=source_sha,
            )
            self.assertEqual(
                receipt["upstream_base_url"], "http://127.0.0.1:18021"
            )
            config["upstream_base_url"] = "http://127.0.0.1:18020"
            config_path.write_bytes(canonical_json_bytes(config))
            with self.assertRaisesRegex(RuntimeError, "route config"):
                validate_exact_token_proxy_config(
                    target,
                    model_port=18021,
                    proxy_port=16383,
                    model_id="Qwen/Qwen3.5-4B",
                    model_revision="3" * 40,
                    proxy_source_sha256=source_sha,
                )

    def test_listener_owner_must_belong_to_the_expected_process_tree(self):
        with tempfile.TemporaryDirectory() as raw:
            config, _ = self.shared_config(Path(raw))
            runtime = LinuxProductionRuntime(config, config.configs)
            with patch.object(
                runtime,
                "tcp_listener_census",
                return_value={
                    "source": "/proc/net/tcp",
                    "family": "ipv4",
                    "address": "127.0.0.1",
                    "port": 18021,
                    "inode": 99,
                },
            ), patch.object(
                runtime,
                "listener_inode_owners",
                return_value={99: {202}},
            ):
                self.assertEqual(
                    runtime.listener_census(
                        18021, {101, 202}, "model server"
                    )["owner_pids"],
                    [202],
                )
                with self.assertRaisesRegex(RuntimeError, "escaped"):
                    runtime.listener_census(18021, {101}, "model server")

    def test_tcp_listener_census_decodes_exact_ipv4_loopback(self):
        row = self.proc_tcp_row("0100007F")
        with patch.object(
            Path,
            "read_text",
            autospec=True,
            side_effect=self.proc_table_reader(tcp_rows=(row,)),
        ):
            self.assertEqual(
                LinuxProductionRuntime.tcp_listener_census(18021),
                {
                    "source": "/proc/net/tcp",
                    "family": "ipv4",
                    "address": "127.0.0.1",
                    "port": 18021,
                    "inode": 99,
                },
            )

    def test_tcp_listener_census_rejects_nonexact_endpoint_rows(self):
        cases = {
            "wildcard IPv4": ((self.proc_tcp_row("00000000"),), ()),
            "non-loopback IPv4": ((self.proc_tcp_row("0200000A"),), ()),
            "wildcard IPv6": ((), (self.proc_tcp_row("0" * 32),)),
            "loopback IPv6": (
                (),
                (self.proc_tcp_row("00000000000000000000000001000000"),),
            ),
            "IPv4-mapped IPv6": (
                (),
                (self.proc_tcp_row("0000000000000000FFFF00000100007F"),),
            ),
            "other IPv6": (
                (),
                (self.proc_tcp_row("B80D0120000000000000000001000000"),),
            ),
            "wrong port": (
                (self.proc_tcp_row("0100007F", port="4666"),),
                (),
            ),
            "duplicate row": (
                (
                    self.proc_tcp_row("0100007F"),
                    self.proc_tcp_row("0100007F"),
                ),
                (),
            ),
            "conflicting row": (
                (
                    self.proc_tcp_row("0100007F"),
                    self.proc_tcp_row("00000000", inode=100),
                ),
                (),
            ),
            "dual-stack conflicting row": (
                (self.proc_tcp_row("0100007F"),),
                (self.proc_tcp_row("00000000000000000000000001000000"),),
            ),
            "zero inode": ((self.proc_tcp_row("0100007F", inode=0),), ()),
        }
        for name, (tcp_rows, tcp6_rows) in cases.items():
            with self.subTest(name=name), patch.object(
                Path,
                "read_text",
                autospec=True,
                side_effect=self.proc_table_reader(
                    tcp_rows=tcp_rows, tcp6_rows=tcp6_rows
                ),
            ), self.assertRaises(RuntimeError):
                LinuxProductionRuntime.tcp_listener_census(18021)

    def test_listener_owner_census_scans_every_process_descriptor(self):
        descriptors = {
            Path("/proc/202/fd"): [
                Path("/proc/202/fd/3"),
                Path("/proc/202/fd/5"),
            ],
            Path("/proc/999/fd"): [Path("/proc/999/fd/4")],
        }

        def iterdir(path):
            if path == Path("/proc"):
                return iter((Path("/proc/202"), Path("/proc/999"), Path("/proc/sys")))
            return iter(descriptors[path])

        links = {
            Path("/proc/202/fd/3"): "socket:[99]",
            Path("/proc/202/fd/5"): "socket:[100]",
            Path("/proc/999/fd/4"): "socket:[99]",
        }
        with patch.object(
            Path, "iterdir", autospec=True, side_effect=iterdir
        ), patch(
            "swebench_triad_eval.production.os.readlink",
            side_effect=lambda path: links[path],
        ):
            self.assertEqual(
                LinuxProductionRuntime.listener_inode_owners({99, 100}),
                {99: {202, 999}, 100: {202}},
            )

    def test_listener_census_rejects_foreign_owner_and_omitted_inode(self):
        with tempfile.TemporaryDirectory() as raw:
            config, _ = self.shared_config(Path(raw))
            runtime = LinuxProductionRuntime(config, config.configs)
            with patch.object(
                runtime,
                "tcp_listener_census",
                return_value={
                    "source": "/proc/net/tcp",
                    "family": "ipv4",
                    "address": "127.0.0.1",
                    "port": 18021,
                    "inode": 99,
                },
            ), patch.object(
                runtime,
                "listener_inode_owners",
                return_value={99: {202, 999}},
            ), self.assertRaisesRegex(RuntimeError, "foreign owner"):
                runtime.listener_census(18021, {202}, "model server")

            with patch.object(
                runtime,
                "tcp_listener_census",
                return_value={
                    "source": "/proc/net/tcp",
                    "family": "ipv4",
                    "address": "127.0.0.1",
                    "port": 18021,
                    "inode": 99,
                },
            ), patch.object(
                runtime,
                "listener_inode_owners",
                return_value={},
            ), self.assertRaisesRegex(RuntimeError, "incomplete"):
                runtime.listener_census(18021, {202}, "model server")

            with patch.object(
                runtime,
                "tcp_listener_census",
                return_value={
                    "source": "/proc/net/tcp",
                    "family": "ipv4",
                    "address": "127.0.0.1",
                    "port": 18021,
                    "inode": 99,
                },
            ), patch.object(
                runtime,
                "listener_inode_owners",
                return_value={99: set()},
            ), self.assertRaisesRegex(RuntimeError, "incomplete"):
                runtime.listener_census(18021, {202}, "model server")

    def test_shared_pool_producer_output_passes_the_exact_preflight_validator(self):
        with tempfile.TemporaryDirectory() as raw:
            config, _ = self.shared_config(Path(raw))
            shared = config.shared_model_pool
            assert shared is not None
            upstream = "http://127.0.0.1:18021"
            pool = shared_model_pool_snapshot_receipt(
                shared,
                readiness_sha256=shared["readiness_sha256"],
                marker_lease_sha256=shared["marker_lease_sha256"],
                selected={
                    "server": {"pid": 303, "start_ticks": 3003},
                    "proxy": {"pid": 403, "start_ticks": 4003},
                },
                selected_live={
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
                },
                assigned_gpu_process_pids=[503],
                live_replica_count=8,
            )

            snapshot = valid_preflight_snapshot()
            expectations = preflight_expectations()
            expectations.update(
                {
                    "gpu_count": 8,
                    "gpu_uuid": shared["gpu_uuid"],
                    "model_id": shared["model_id"],
                    "model_pid": 303,
                    "model_start_ticks": 3003,
                    "shared_model_pool": config.preflight_expectations[
                        "shared_model_pool"
                    ],
                }
            )
            snapshot["pod"] = {
                **snapshot["pod"],
                "gpu_uuid": shared["gpu_uuid"],
                "gpu_count": 8,
            }
            snapshot["model_process"] = {
                **snapshot["model_process"],
                "pid": 303,
                "start_ticks": 3003,
            }
            snapshot["vllm"] = {
                **snapshot["vllm"],
                "model_id": shared["model_id"],
            }
            snapshot["shared_model_pool"] = pool
            self.assertEqual(
                validate_preflight_snapshot(snapshot, expectations)["status"],
                "PASS",
            )

    def test_preflight_listener_reference_is_receipt_hash_bound(self):
        with tempfile.TemporaryDirectory() as raw:
            config, _ = self.shared_config(Path(raw))
            runtime = LinuxProductionRuntime(config, config.configs)
            snapshot, _ = cli_test_support.SharedModelPoolPreflightTest.fixture()
            control = config.run_root / "control"
            control.mkdir(parents=True)
            (control / "preflight-snapshot.json").write_bytes(
                canonical_json_bytes(snapshot)
            )
            receipt = {
                "schema": "swebench_triad_preflight_pass_v1",
                "status": "PASS",
                "snapshot_sha256": sha256_json(snapshot),
                "deployment_commit": "d" * 40,
                "inner_commit": "a" * 40,
                "boot_id": "boot-id",
                "gpu_uuid": "GPU-expected",
                "docker_daemon_id": "docker-daemon-id",
                "model_id": "Qwen/Qwen3.5-4B",
            }
            receipt_path = control / "preflight-PASS.json"
            receipt_path.write_bytes(canonical_json_bytes(receipt))
            reference = runtime.preflight_shared_model_pool_snapshot()
            self.assertEqual(
                reference["server_listener_census"]["inode"], 99
            )

            receipt["snapshot_sha256"] = "0" * 64
            receipt_path.write_bytes(canonical_json_bytes(receipt))
            with self.assertRaisesRegex(RuntimeError, "reference drifted"):
                runtime.preflight_shared_model_pool_snapshot()

    def test_each_shared_pool_cell_requires_a_fresh_exact_pool_snapshot(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config, _ = self.shared_config(root)
            runtime = LinuxProductionRuntime(config, config.configs)
            shared = config.shared_model_pool
            assert shared is not None
            task_index = next(
                index
                for index, rows in runtime.by_task.items()
                if int.from_bytes(
                    hashlib.sha256(rows[0].task.task_id.encode("utf-8")).digest()[:8],
                    "big",
                )
                % shared["replica_count"]
                == shared["replica_index"]
            )
            task_root = root / "pod-task"
            task_root.mkdir()
            stage = SimpleNamespace(task_root=task_root)
            with patch.object(
                runtime, "require_stage", return_value=stage
            ), patch.object(
                runtime, "owned_container_ids", return_value=[]
            ), patch.object(
                runtime, "cgroup_paths", return_value=[]
            ), patch.object(
                runtime,
                "shared_model_pool_snapshot",
                return_value={"status": "PASS"},
            ) as live_probe, self.assertRaisesRegex(
                RuntimeError, "fields drifted"
            ):
                runtime.run_cell(
                    runtime.by_task[task_index][0],
                    stage,
                    generation=1,
                    slot=runtime_slot(runtime, task_index),
                )
            live_probe.assert_called_once_with(require_preflight_binding=True)

    def test_final_audit_reprobes_and_embeds_the_live_shared_pool(self):
        with tempfile.TemporaryDirectory() as raw:
            config, _ = self.shared_config(Path(raw))
            runtime = LinuxProductionRuntime(config, config.configs)
            pool = {"status": "PASS", "replica_index": 3}
            residue = {
                "active_owned_processes": 0,
                "active_cgroups": 0,
                "active_tmpfs_mounts": 0,
                "active_mounts": 0,
                "active_scratch_paths": 0,
                "loaded_task_images": 0,
                "owned_containers": 0,
            }
            with patch.object(
                runtime, "global_residue_snapshot", return_value=residue
            ), patch.object(
                runtime, "shared_model_pool_snapshot", return_value=pool
            ) as live_probe, patch(
                "swebench_triad_eval.production.validate_shared_model_pool_snapshot",
                return_value=pool,
            ) as exact_contract, patch.object(
                runtime,
                "pod_snapshot",
                return_value={
                    "job": config.section("pod")["job"],
                    "boot_id": config.section("pod")["boot_id"],
                },
            ), patch.object(
                runtime,
                "command_ledger_audit",
                return_value={"status": "PASS"},
            ):
                receipt = runtime.final_audit()
            live_probe.assert_called_once_with(require_preflight_binding=True)
            exact_contract.assert_called_once_with(
                pool, "final audit shared model pool snapshot"
            )
            self.assertEqual(receipt["shared_model_pool"], pool)

    def test_shared_pool_config_is_strict_and_exposes_eight_gpu_expectation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config, payload = self.shared_config(root)
            self.assertEqual(config.preflight_expectations["gpu_count"], 8)
            self.assertEqual(
                config.preflight_expectations["shared_model_pool"]["replica_index"],
                3,
            )
            payload["serving"]["base_url"] = "http://127.0.0.1:16382/v1"
            (root / "run-config.json").write_bytes(canonical_json_bytes(payload))
            with self.assertRaisesRegex(ValueError, "exact-token proxy"):
                ProductionRunConfig.load(root / "run-config.json")

    def test_pod_snapshot_requires_all_eight_unique_gpus_and_selected_uuid(self):
        with tempfile.TemporaryDirectory() as raw:
            config, _ = self.shared_config(Path(raw))
            runtime = LinuxProductionRuntime(config, config.configs)
            values = [f"GPU-shared-{index}" for index in range(8)]
            with patch(
                "swebench_triad_eval.production.socket.gethostname",
                return_value="pod-host",
            ), patch(
                "swebench_triad_eval.production.Path.read_text",
                return_value="pod-boot\n",
            ), patch(
                "swebench_triad_eval.production.command_output",
                return_value="\n".join(values),
            ):
                snapshot = runtime.pod_snapshot()
            self.assertEqual(snapshot["gpu_count"], 8)
            self.assertEqual(snapshot["gpu_uuid"], "GPU-shared-3")

            values[3] = "GPU-wrong"
            with patch(
                "swebench_triad_eval.production.socket.gethostname",
                return_value="pod-host",
            ), patch(
                "swebench_triad_eval.production.Path.read_text",
                return_value="pod-boot\n",
            ), patch(
                "swebench_triad_eval.production.command_output",
                return_value="\n".join(values),
            ), self.assertRaisesRegex(RuntimeError, "replica GPU"):
                runtime.pod_snapshot()

    def test_shared_pool_vllm_probe_reads_registry_from_upstream_model_port(self):
        with tempfile.TemporaryDirectory() as raw:
            config, _ = self.shared_config(Path(raw))
            runtime = LinuxProductionRuntime(config, config.configs)
            chat = {
                "prompt_token_ids": [1, 2],
                "choices": [
                    {
                        "token_ids": [3, 4],
                        "message": {"content": "AMG_OK"},
                    }
                ],
            }
            with patch(
                "swebench_triad_eval.production.http_json",
                side_effect=[
                    {"data": [{"id": "Qwen/Qwen3.5-4B"}]},
                    {"tokens": [1, 2]},
                    chat,
                    chat,
                ],
            ) as probe:
                snapshot = runtime.vllm_snapshot()
            self.assertEqual(
                probe.call_args_list[0].args[0],
                "http://127.0.0.1:18021/v1/models",
            )
            self.assertEqual(
                probe.call_args_list[1].args[0],
                "http://127.0.0.1:16383/tokenize",
            )
            self.assertEqual(
                probe.call_args_list[2].args[0],
                "http://127.0.0.1:16383/v1/chat/completions",
            )
            self.assertTrue(snapshot["repeat_text_equal"])

    def test_shared_pool_cleanup_fails_before_any_global_mutation(self):
        with tempfile.TemporaryDirectory() as raw:
            config, _ = self.shared_config(Path(raw))
            runtime = LinuxProductionRuntime(config, config.configs)
            with patch.object(
                runtime, "owned_container_ids"
            ) as containers, patch.object(
                runtime, "cgroup_paths"
            ) as cgroups, patch.object(
                runtime, "reconcile_startup"
            ) as reconcile, patch.object(
                runtime, "stop_model_process"
            ) as stop_model, patch.object(
                runtime, "restore_holders"
            ) as restore_holders, self.assertRaisesRegex(
                RuntimeError, "eight-replica coordinator"
            ):
                runtime.cleanup(slots=startup_slots(runtime))
            containers.assert_not_called()
            cgroups.assert_not_called()
            reconcile.assert_not_called()
            stop_model.assert_not_called()
            restore_holders.assert_not_called()

if __name__ == "__main__":
    unittest.main()
