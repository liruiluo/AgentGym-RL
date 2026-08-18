from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from test_paired_eval_support import Arm, make_config

from paired_eval.contracts import capability_for_arm
from paired_eval.evidence import PrivateEvidenceStore
from paired_eval.serialization import canonical_json_bytes
from swebench_triad_eval.cli import (
    LifecycleDriver,
    PreflightContractError,
    build_parser,
    driver_from_config,
    main,
    read_private_json,
    validate_preflight_snapshot,
)
from swebench_triad_eval.identity import (
    PRODUCTION_DATASET_PINS,
    PRODUCTION_IMAGE_INDEX_PINS,
)
from swebench_triad_eval.state import CellKey, DriverLeaseRegistry, OwnerIdentity


HARNESS_COMMIT = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
INNER_COMMIT = "a0cc3ecf989ee89ba19a8e979617b4ec38909331"
DEPLOYMENT_COMMIT = "d" * 40
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def preflight_expectations() -> dict[str, object]:
    return {
        "deployment_commit": DEPLOYMENT_COMMIT,
        "inner_commit": INNER_COMMIT,
        "job": "luolirui-1-swesmith-4b-1c-0806",
        "pod": "luolirui-1-swesmith-4b-1c-0806-master-0",
        "hostname": "luolirui-1-swesmith-4b-1c-0806-master-0",
        "boot_id": "boot-id",
        "gpu_uuid": "GPU-expected",
        "docker_daemon_id": "docker-daemon-id",
        "docker_pid": 101,
        "docker_start_ticks": 1001,
        "model_pid": 202,
        "model_start_ticks": 2002,
        "model_id": "Qwen3.5-4B",
        "blob_certificate_sha256": SHA_A,
        "blob_revalidation_sha256": SHA_B,
        "docker_receipt_sha256": SHA_C,
        "task4_receipt_sha256": SHA_D,
        "rootfs_prefix": "/root/.local/state/amg-swebench-triad-eval-20260816/",
    }


def valid_preflight_snapshot() -> dict[str, object]:
    return {
        "source": {
            "deployment_commit": DEPLOYMENT_COMMIT,
            "inner_commit": INNER_COMMIT,
            "deployment_clean": True,
            "inner_clean": True,
            "protected_diff_zero": True,
        },
        "dataset": {
            "rows": 500,
            "jsonl_sha256": PRODUCTION_DATASET_PINS.jsonl_sha256,
            "id_ledger_sha256": PRODUCTION_DATASET_PINS.id_ledger_sha256,
        },
        "image_index": {
            "rows": 500,
            "index_sha256": PRODUCTION_IMAGE_INDEX_PINS.index_sha256,
            "tag_ledger_sha256": (
                PRODUCTION_IMAGE_INDEX_PINS.tag_ledger_sha256
            ),
            "digest_tsv_sha256": (
                PRODUCTION_IMAGE_INDEX_PINS.digest_tsv_sha256
            ),
        },
        "model": {
            "file_count": 14,
            "file_ledger_sha256": "model-ledger",
        },
        "blob_cache": {
            "certificate_sha256": SHA_A,
            "revalidation_sha256": SHA_B,
            "descriptor_count": 1158,
            "file_count": 1158,
            "total_bytes": 117637519356,
            "downloaded_count": 0,
            "verified_bad_count": 0,
        },
        "pod": {
            "job": "luolirui-1-swesmith-4b-1c-0806",
            "pod": "luolirui-1-swesmith-4b-1c-0806-master-0",
            "hostname": "luolirui-1-swesmith-4b-1c-0806-master-0",
            "boot_id": "boot-id",
            "gpu_uuid": "GPU-expected",
            "gpu_count": 1,
        },
        "docker": {
            "receipt_sha256": SHA_C,
            "daemon_id": "docker-daemon-id",
            "pid": 101,
            "start_ticks": 1001,
            "version": "27.5.1",
            "api_version": "1.47",
            "cgroup_version": "1",
            "cgroup_driver": "cgroupfs",
            "storage_driver": "vfs",
            "containers": 0,
            "images": 0,
            "volumes": 0,
        },
        "task4_negative_probes": {
            "receipt_sha256": SHA_D,
            "schema": "amg_swebench_task4_live_negative_probes_v1",
            "status": "PASS",
            "network_downloads": 0,
            "memory_exhaustion_blocked": True,
            "fork_exhaustion_blocked": True,
            "byte_quota_blocked": True,
            "inode_quota_blocked": True,
            "rootfs_mutation_detected": True,
            "cgroup_residue_absent": True,
            "tmpfs_residue_absent": True,
            "docker_residue_absent": True,
        },
        "model_process": {
            "pid": 202,
            "start_ticks": 2002,
            "alive": True,
            "command_matches": True,
        },
        "vllm": {
            "model_id": "Qwen3.5-4B",
            "prompt_token_ids": [1, 2, 3],
            "response_token_ids": [4, 5],
            "repeat_prompt_token_ids": [1, 2, 3],
            "repeat_response_token_ids": [4, 5],
            "repeat_text_equal": True,
        },
        "swe_metadata": {
            "schema": "swebench_verified_external_patch_episode_v1",
            "task_count": 500,
            "full_benchmark_task_count": 500,
            "supported_arms": [
                "native",
                "amg_compaction_only",
                "amg_memory",
            ],
            "active_slot_count": 0,
            "active_workspace_count": 0,
            "official_grading_inside_adapter": False,
            "evaluation_max_policy_turns": 250,
            "max_native_actions": 250,
            "max_observation_tokens": 8192,
        },
        "residue": {
            "active_owned_processes": 0,
            "active_cgroups": 0,
            "active_tmpfs_mounts": 0,
            "active_mounts": 0,
            "active_scratch_paths": 0,
            "loaded_task_images": 0,
            "owned_containers": 0,
        },
        "rootfs": {
            "path": (
                "/root/.local/state/amg-swebench-triad-eval-20260816/"
                "oci-rootfs"
            ),
            "pod_local": True,
        },
    }


def configs_for_two_tasks():
    configs = []
    for task_index in range(2):
        base = make_config(
            benchmark="swebench_verified",
            protocol="swebench-verified@v4.1.0",
            task_id=f"owner__repo-{task_index}",
            task_index=task_index,
            artifact_type="patch",
        )
        base = replace(
            base,
            grader=replace(
                base.grader,
                name="swebench-v4.1.0",
                revision=HARNESS_COMMIT,
            ),
        )
        for arm in Arm:
            configs.append(
                replace(base, capability=capability_for_arm(arm))
            )
    return configs


def triad_verification(rows):
    return {
        "status": "PASS",
        "row_count": len(rows),
        "triad_count": len(rows) // 3,
    }


def formal_manifest() -> dict[str, object]:
    return {
        "schema": "amg.paired_eval.manifest",
        "schema_version": "2.0.0",
        "run_id": "amg-swebench-triad-eval-0816",
        "arms": [
            "native",
            "amg_compaction_only",
            "amg_memory",
        ],
        "common": {
            "model": {
                "model_id": "Qwen3.5-4B",
                "revision": "shared-snapshot",
                "tokenizer_sha256": SHA_A,
            },
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 2048,
                "stop": [],
            },
            "budgets": {
                "max_policy_turns": 30,
                "max_total_tokens": 8388608,
                "max_tool_calls": 30,
                "max_wall_seconds": 1800.0,
                "max_prompt_tokens": 30720,
                "max_model_tokens": 32768,
                "max_observation_tokens": 8192,
                "action_observation_envelope_tokens": 0,
            },
            "compaction": {
                "policy": "policy_authored_task_neutral_v1",
                "trigger": "wrapper_token_pressure_v1",
                "summary_max_tokens": 2048,
                "summary_instruction_sha256": SHA_A,
                "context_pressure_policy_sha256": SHA_B,
                "context_transition_schema": (
                    "agentmemory_task_neutral_context_transition_v1"
                ),
                "action_accounting": "global_policy_action_budget_v1",
                "config_sha256": SHA_C,
            },
            "source": {
                "outer_commit": (
                    "aa2e9c80d572b513b5849c6d9b37a8dc4698bbc3"
                ),
                "inner_commit": INNER_COMMIT,
                "adapter_sha256": SHA_A,
                "runner_sha256": SHA_B,
            },
            "runtime": {
                "image_digest": "sha256:" + SHA_C,
                "runtime_sha256": SHA_D,
                "compute_class": "1xB200",
            },
            "grader": {
                "name": "swebench-v4.1.0",
                "revision": HARNESS_COMMIT,
                "config_sha256": SHA_A,
            },
        },
        "tasks": [
            {
                "benchmark": "swebench_verified",
                "protocol": "swebench-verified@v4.1.0",
                "task_id": f"owner__repo-{index:04d}",
                "task_index": index,
                "seed": 0,
                "native_tools": ["shell_command", "apply_patch", "final"],
                "artifact_type": "patch",
            }
            for index in range(500)
        ],
    }


def production_config(root: Path) -> tuple[Path, dict[str, object]]:
    manifest_path = root / "manifest.json"
    manifest_payload = canonical_json_bytes(formal_manifest())
    manifest_path.write_bytes(manifest_payload)

    def path(name: str) -> str:
        return str(root / name)

    config = {
        "schema": "amg_swebench_triad_run_config_v1",
        "run_root": path("run"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "evidence_root": path("evidence"),
        "source": {
            "root": path("source"),
            "integration_commit": (
                "aa2e9c80d572b513b5849c6d9b37a8dc4698bbc3"
            ),
            "deployment_commit": DEPLOYMENT_COMMIT,
            "inner_commit": INNER_COMMIT,
        },
        "assets": {
            "dataset_manifest": path("dataset-manifest.json"),
            "dataset_jsonl": path("dataset.jsonl"),
            "image_index": path("image-index.jsonl"),
            "image_digests": path("image-digests.tsv"),
            "image_manifests": path("image-manifests"),
            "blob_root": path("blob-root"),
            "blob_certificate": path("blob-certificate.json"),
            "blob_certificate_sha256": SHA_A,
            "blob_revalidation_receipt": path("blob-revalidation.json"),
            "blob_revalidation_sha256": SHA_B,
            "exact_identity_receipt": path("identity-receipt.json"),
            "harness_root": path("harness"),
            "model_root": path("model"),
            "rg_binary": path("rg"),
            "rg_sha256": SHA_A,
        },
        "pod": {
            "job": "luolirui-1-swesmith-4b-1c-0806",
            "pod": "luolirui-1-swesmith-4b-1c-0806-master-0",
            "hostname": "pod-host",
            "boot_id": "pod-boot",
            "gpu_uuid": "GPU-pinned",
        },
        "docker": {
            "socket": path("docker.sock"),
            "executable": path("docker"),
            "pid_file": path("dockerd.pid"),
            "readiness_receipt": path("docker-readiness.json"),
            "readiness_receipt_sha256": SHA_A,
            "daemon_id": "daemon-id",
            "pid": 101,
            "start_ticks": 1001,
        },
        "task4_receipt": {
            "path": path("task4.json"),
            "sha256": SHA_B,
        },
        "serving": {
            "base_url": "http://127.0.0.1:18000/v1",
            "model_id": "Qwen3.5-4B",
            "pid_file": path("model.pid"),
            "pid": 202,
            "start_ticks": 2002,
            "receipt_path": path("serving.json"),
            "receipt_sha256": SHA_C,
        },
        "runtime": {
            "pod_local_root": path("pod-local"),
            "mirrors_root": path("mirrors"),
            "server_port": 18100,
            "container_python": "python",
            "model_timeout_seconds": 1800.0,
            "environment_timeout_seconds": 1800,
            "memory_bytes": 68719476736,
            "max_processes": 1024,
            "workspace_bytes": 2147483648,
            "workspace_inodes": 250000,
            "external_memory_bytes": 1073741824,
            "external_memory_inodes": 100000,
        },
        "grader": {
            "python_executable": path("grader-python"),
            "output_root": path("grader-output"),
            "max_attempts": 3,
        },
    }
    config_path = root / "run-config.json"
    config_path.write_bytes(canonical_json_bytes(config))
    return config_path, config


class FakeLifecycleOperations:
    def __init__(self, evidence_root: Path) -> None:
        self.evidence = PrivateEvidenceStore(evidence_root)
        self.run_calls: list[tuple[int, str]] = []
        self.grade_calls: list[tuple[int, str]] = []
        self.staged: list[int] = []
        self.evicted: list[int] = []
        self.cleanup_calls = 0
        self.preflight_calls = 0
        self.stale_residue = False
        self.events: list[tuple[object, ...]] = []

    def preflight(self):
        self.preflight_calls += 1
        snapshot = valid_preflight_snapshot()
        if self.stale_residue:
            snapshot["residue"] = dict(snapshot["residue"])
            snapshot["residue"]["owned_containers"] = 1
        return snapshot

    def stage_task(self, task_index: int, *, slot):
        self.assert_slot(slot, task_index)
        self.staged.append(task_index)
        self.events.append(("stage", task_index))
        return {"task_index": task_index, "status": "staged"}

    def reconcile_cell(
        self, config, *, generation, before_preflight, slot
    ):
        self.assert_slot(slot, config.task.task_index)
        self.events.append(
            (
                "reconcile_cell",
                config.task.task_index,
                config.capability.arm.value,
                generation,
                before_preflight,
            )
        )
        if before_preflight:
            self.stale_residue = False
        return {"status": "reconciled"}

    def reconcile_grade(self, *, key, accepted, prediction, handoff, slot):
        self.assert_slot(slot, key.task_index)
        del accepted, prediction, handoff
        self.events.append(("reconcile_grade", key.task_index, key.arm))
        return {"status": "reconciled"}

    def reconcile_startup(self, *, task_indices, slots):
        self.assertTrue(slots)
        self.events.append(("reconcile_startup", tuple(task_indices)))
        return {"status": "reconciled"}

    def run_cell(self, config, _stage, *, generation, slot):
        self.assert_slot(slot, config.task.task_index)
        key = (config.task.task_index, config.capability.arm.value)
        self.run_calls.append(key)
        self.events.append(("run", *key, generation))
        prediction = {
            "instance_id": config.task.task_id,
            "model_name_or_path": "Qwen3.5-4B",
            "model_patch": "",
        }
        prediction_ref = self.evidence.put_json(
            "swebench_predictions", prediction
        )
        queue = {
            "artifact_sha256": prediction_ref.sha256,
            "official_resolved": None,
            "grader": {"revision": HARNESS_COMMIT},
        }
        scorer_ref = self.evidence.put_json(
            "receipts", {"grader_receipt": queue}
        )
        return {
            "marker": "valid-endpoint",
            "task_id": config.task.task_id,
            "task_index": config.task.task_index,
            "arm": config.capability.arm.value,
            "comparable": True,
            "failure": {"class": None},
            "termination": {"reason": "horizon"},
            "final_artifact": {
                "protected_ref": prediction_ref.protected_ref,
                "sha256": prediction_ref.sha256,
            },
            "scorer": {
                "receipt_ref": scorer_ref.protected_ref,
                "public_metrics": {"official_resolved": None},
            },
            "lifecycle": {"close_receipt_ref": "evidence://close/" + SHA_A},
        }

    def grade(self, *, key, accepted, prediction, handoff, slot):
        self.assert_slot(slot, key.task_index)
        del accepted, prediction, handoff
        self.grade_calls.append((key.task_index, key.arm))
        self.events.append(("grade", key.task_index, key.arm))
        return {
            "instance_id": f"owner__repo-{key.task_index}",
            "arm": key.arm,
            "resolved": key.arm == "amg_memory",
            "failure_class": None,
            "report_sha256": SHA_D,
        }

    def audit_residue(self, task_index: int, *, slot):
        self.assert_slot(slot, task_index)
        return {
            "task_index": task_index,
            "active_slots": 0,
            "active_workspaces": 0,
            "containers": 0,
            "processes": 0,
            "cgroups": 0,
            "tmpfs_mounts": 0,
            "mounts": 0,
            "rootfs_attested": True,
        }

    def evict_task(self, task_index: int, _stage, *, slot):
        self.assert_slot(slot, task_index)
        self.evicted.append(task_index)
        self.events.append(("evict", task_index, _stage is None))
        return {"task_index": task_index, "status": "evicted"}

    def cleanup(self, *, slots):
        self.assertTrue(slots)
        self.cleanup_calls += 1
        return {"owned_residue": 0, "allocation_retained": True}

    def final_audit(self):
        return {"status": "PASS", "residue": {}}

    @staticmethod
    def assert_slot(slot, task_index):
        if slot.task_index not in {None, task_index}:
            raise AssertionError("operation received another task's slot")

    @staticmethod
    def assertTrue(value):
        if not value:
            raise AssertionError("expected a nonempty value")

    @staticmethod
    def timing_identity():
        return {
            "deployment_commit": "d" * 40,
            "run_config_sha256": "c" * 64,
            "replica_index": 0,
            "gpu_uuid": "GPU-test",
        }


class PreflightValidationTest(unittest.TestCase):
    def test_exact_preflight_snapshot_passes(self) -> None:
        receipt = validate_preflight_snapshot(
            valid_preflight_snapshot(), preflight_expectations()
        )
        self.assertEqual(receipt["status"], "PASS")

    def test_every_required_live_boundary_fails_closed(self) -> None:
        mutations = (
            ("source", "deployment_clean", False),
            ("source", "protected_diff_zero", False),
            ("dataset", "rows", 499),
            ("image_index", "index_sha256", SHA_A),
            ("blob_cache", "downloaded_count", 1),
            ("blob_cache", "verified_bad_count", 1),
            ("pod", "gpu_uuid", "GPU-wrong"),
            ("docker", "daemon_id", "wrong-daemon"),
            ("docker", "start_ticks", 1002),
            ("docker", "containers", 1),
            ("task4_negative_probes", "status", "FAIL"),
            ("task4_negative_probes", "fork_exhaustion_blocked", False),
            ("model_process", "start_ticks", 2003),
            ("vllm", "model_id", "wrong-model"),
            ("vllm", "repeat_response_token_ids", [9]),
            ("swe_metadata", "active_workspace_count", 1),
            ("swe_metadata", "official_grading_inside_adapter", True),
            ("residue", "active_cgroups", 1),
            ("rootfs", "pod_local", False),
        )
        for section, field, value in mutations:
            with self.subTest(section=section, field=field):
                snapshot = valid_preflight_snapshot()
                snapshot[section] = dict(snapshot[section])
                snapshot[section][field] = value
                with self.assertRaises(PreflightContractError):
                    validate_preflight_snapshot(
                        snapshot, preflight_expectations()
                    )


class PrivateEvidenceReadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "evidence"
        self.store = PrivateEvidenceStore(self.root)

    def test_dereference_is_digest_bound_and_rejects_symlinks(self) -> None:
        value = {"instance_id": "owner__repo-0", "model_patch": ""}
        reference = self.store.put_json("predictions", value)
        self.assertEqual(
            read_private_json(self.root, reference.protected_ref), value
        )

        path = self.root / "predictions" / f"{reference.sha256}.json"
        payload = path.read_bytes()
        path.write_bytes(payload + b" ")
        with self.assertRaises(PreflightContractError):
            read_private_json(self.root, reference.protected_ref)

        path.unlink()
        target = self.root / "target.json"
        target.write_bytes(payload)
        os.symlink(target, path)
        with self.assertRaises(PreflightContractError):
            read_private_json(self.root, reference.protected_ref)


class LifecycleDriverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.operations = FakeLifecycleOperations(self.root / "evidence")
        self.driver = LifecycleDriver(
            root=self.root / "run",
            configs=configs_for_two_tasks(),
            owner=OwnerIdentity("host", "boot", 101, 1001),
            owner_is_alive=lambda _owner: False,
            operations=self.operations,
            evidence_root=self.root / "evidence",
            endpoint_validator=lambda row: (
                None
                if row.get("marker") == "valid-endpoint"
                else (_ for _ in ()).throw(ValueError("invalid endpoint"))
            ),
            triad_validator=lambda rows: (
                self.operations.events.append(("verify_triad",))
                or triad_verification(rows)
            ),
            preflight_expectations=preflight_expectations(),
        )

    def test_gate_cells_are_canonical_and_auto_continue_without_rerun(self) -> None:
        summary = self.driver.gate(auto_run_full=True)
        self.assertEqual(len(self.operations.run_calls), 6)
        self.assertEqual(len(self.operations.grade_calls), 6)
        self.assertEqual(self.operations.staged, [0, 1])
        self.assertEqual(self.operations.evicted, [0, 1])
        self.assertEqual(summary["denominator_per_arm"], 2)
        self.assertTrue((self.root / "run/gate/PASS.json").is_file())

        before_runs = list(self.operations.run_calls)
        before_grades = list(self.operations.grade_calls)
        resumed = self.driver.resume()
        self.assertEqual(self.operations.run_calls, before_runs)
        self.assertEqual(self.operations.grade_calls, before_grades)
        self.assertEqual(resumed["denominator_per_arm"], 2)
        self.assertGreaterEqual(self.operations.preflight_calls, 3)

    def test_reconcile_dead_work_returns_the_complete_production_list_shape(self):
        slots = [self.driver.acquire_runtime_lane(None, slot_index=0)]
        receipts = self.driver.reconcile_dead_work()
        self.driver.release_runtime_lane(slots[0])

        self.assertIs(type(receipts), list)
        self.assertEqual(receipts, [{"startup": {"status": "reconciled"}}])
        persisted = json.loads(
            (self.root / "run/control/latest-reconciliation.json").read_text()
        )
        self.assertEqual(persisted["receipts"], receipts)

    def test_existing_gate_is_exactly_bound_to_current_canonical_state(self):
        self.driver.gate(auto_run_full=False)
        gate_path = self.root / "run/gate/PASS.json"
        original = json.loads(gate_path.read_text())
        mutations = {
            "missing-field": {
                key: value
                for key, value in original.items()
                if key != "canonical_cells"
            },
            "canonical-cells": {
                **original,
                "canonical_cells": [
                    {"task_index": 0, "arm": "native"},
                ],
            },
            "accepted-count": {**original, "accepted_cells": 2},
            "outcome-count": {**original, "official_outcomes": 2},
            "triad": {**original, "triad_verification_sha256": SHA_A},
            "preflight": {**original, "preflight_sha256": SHA_A},
            "score-flag": {**original, "standalone_benchmark_score": True},
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                gate_path.write_bytes(canonical_json_bytes(mutation))
                with self.assertRaises(RuntimeError):
                    self.driver.gate(auto_run_full=False)
                gate_path.write_bytes(canonical_json_bytes(original))

    def test_complete_audit_requires_gate_and_its_immutable_preflight(self):
        self.driver.gate(auto_run_full=True)
        gate_path = self.root / "run/gate/PASS.json"
        preflight_path = self.root / "run/control/preflight-PASS.json"
        gate = json.loads(gate_path.read_text())
        preflight = json.loads(preflight_path.read_text())
        self.assertEqual(self.driver.audit()["status"], "PASS")

        gate_path.unlink()
        self.assertFalse(self.driver.status()["gate_pass"])
        with self.assertRaisesRegex(RuntimeError, "gate"):
            self.driver.audit()

        gate_path.write_bytes(canonical_json_bytes(gate))
        gate_path.write_bytes(
            canonical_json_bytes(
                {**gate, "triad_verification_sha256": SHA_A}
            )
        )
        with self.assertRaisesRegex(RuntimeError, "gate"):
            self.driver.status()
        with self.assertRaisesRegex(RuntimeError, "gate"):
            self.driver.audit()

        gate_path.write_bytes(canonical_json_bytes(gate))
        preflight_path.write_bytes(
            canonical_json_bytes(
                {**preflight, "snapshot_sha256": SHA_A}
            )
        )
        with self.assertRaisesRegex(RuntimeError, "preflight"):
            self.driver.status()
        with self.assertRaisesRegex(RuntimeError, "preflight"):
            self.driver.audit()

    def test_dead_owner_residue_is_reconciled_before_gate_preflight(self):
        key = CellKey(0, "native")
        self.driver.store.acquire(key)
        recovered_operations = FakeLifecycleOperations(self.root / "evidence")
        recovered_operations.stale_residue = True
        recovered = LifecycleDriver(
            root=self.root / "run",
            configs=configs_for_two_tasks(),
            owner=OwnerIdentity("host", "boot", 202, 2002),
            owner_is_alive=lambda _owner: False,
            operations=recovered_operations,
            evidence_root=self.root / "evidence",
            endpoint_validator=lambda row: None,
            triad_validator=lambda _rows: {"status": "PASS"},
            preflight_expectations=preflight_expectations(),
        )
        recovered.gate(auto_run_full=False)
        reconcile = next(
            event
            for event in recovered_operations.events
            if event[:3] == ("reconcile_cell", 0, "native")
            and event[-1] is True
        )
        self.assertEqual(reconcile[3], 2)
        self.assertFalse(recovered_operations.stale_residue)

    def test_attempt_generation_reaches_runtime_and_triad_precedes_grading(self):
        self.driver.run_task(0, gate=True)
        task_events = [event for event in self.operations.events if 0 in event]
        run_events = [event for event in task_events if event[0] == "run"]
        grade_events = [event for event in task_events if event[0] == "grade"]
        self.assertEqual([event[-1] for event in run_events], [1, 1, 1])
        self.assertLess(
            max(task_events.index(event) for event in run_events),
            min(task_events.index(event) for event in grade_events),
        )
        self.assertLess(
            self.operations.events.index(("verify_triad",)),
            self.operations.events.index(grade_events[0]),
        )
        self.assertEqual(
            [event[2] for event in run_events],
            ["native", "amg_compaction_only", "amg_memory"],
        )

    def test_task_completion_can_be_validated_without_acquiring_a_runtime_lane(self):
        completion = self.driver.run_task(0, gate=True)
        self.assertEqual(self.driver.load_task_completion(0), completion)

    def test_task_phase_timing_binds_slot_port_and_nonnegative_boundaries(self):
        completion = self.driver.run_task(0, gate=True)
        timing_ref = completion["timing_receipt"]
        timing = json.loads(Path(timing_ref["path"]).read_text())
        self.assertEqual(timing["status"], "READY_FOR_PUBLICATION")
        self.assertEqual(timing["slot_index"], 0)
        self.assertEqual(timing["server_port"], 18100)
        self.assertEqual(
            timing["task_seed"], self.driver.by_task[0][0].task.seed
        )
        self.assertTrue(timing["phase_durations_are_non_additive"])
        phases = [row["phase"] for row in timing["phases"]]
        self.assertEqual(
            phases,
            [
                "task_slot_queue",
                "oci_stage",
                "cell_native",
                "cell_amg_compaction_only",
                "cell_amg_memory",
                "triad_validation",
                "official_grade_native",
                "official_grade_amg_compaction_only",
                "official_grade_amg_memory",
                "residue_audit",
                "task_eviction",
                "publication_ready",
            ],
        )
        for row in timing["phases"]:
            self.assertGreaterEqual(row["duration_ns"], 0)
            self.assertGreaterEqual(
                row["ended_monotonic_ns"], row["started_monotonic_ns"]
            )
            self.assertGreaterEqual(row["ended_wall_ns"], row["started_wall_ns"])

    def test_coordinator_queue_digest_and_publication_timings_are_durable(self):
        queued_wall = time.time_ns() - 5_000_000_000
        queued_mono = time.monotonic_ns() - 5_000_000_000
        dequeued_wall = queued_wall + 2_000_000_000
        dequeued_mono = queued_mono + 2_000_000_000

        @contextmanager
        def admission(_slot):
            start_wall = time.time_ns()
            start_mono = time.monotonic_ns()
            end_wall = start_wall + 1_000_000
            end_mono = start_mono + 1_000_000
            yield {
                "phase": "image_digest_wait",
                "status": "PASS",
                "started_wall_ns": start_wall,
                "ended_wall_ns": end_wall,
                "started_monotonic_ns": start_mono,
                "ended_monotonic_ns": end_mono,
                "duration_ns": 1_000_000,
            }

        completion = self.driver.run_task(
            0,
            gate=True,
            admission=admission,
            queued_wall_ns=queued_wall,
            queued_monotonic_ns=queued_mono,
            slot_dequeued_wall_ns=dequeued_wall,
            slot_dequeued_monotonic_ns=dequeued_mono,
        )
        timing = json.loads(Path(completion["timing_receipt"]["path"]).read_text())
        phases = {row["phase"]: row for row in timing["phases"]}
        self.assertEqual(phases["task_slot_queue"]["duration_ns"], 2_000_000_000)
        self.assertIn("runtime_lane_wait", phases)
        self.assertEqual(phases["image_digest_wait"]["duration_ns"], 1_000_000)
        publication_path = self.driver.task_publication_path(0)
        publication = json.loads(publication_path.read_text())
        completion_path = self.driver.task_completion_path(0)
        self.assertEqual(
            publication["completion_sha256"],
            hashlib.sha256(completion_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            publication["timing_receipt_sha256"],
            completion["timing_receipt"]["sha256"],
        )
        self.assertGreaterEqual(publication["duration_ns"], 0)
        self.assertFalse(publication["recovered_after_crash"])
        self.assertEqual(self.driver.load_task_completion(0), completion)

    def test_resume_recovers_missing_task_publication_sidecar(self):
        completion = self.driver.run_task(0, gate=True)
        publication_path = self.driver.task_publication_path(0)
        publication_path.unlink()

        resumed = self.driver.run_task(0, gate=True)

        self.assertEqual(resumed, completion)
        publication = json.loads(publication_path.read_text())
        self.assertEqual(
            publication["schema"], "swebench_triad_task_publication_timing_v2"
        )
        self.assertTrue(publication["recovered_after_crash"])
        timing = json.loads(Path(completion["timing_receipt"]["path"]).read_text())
        self.assertEqual(publication["started_wall_ns"], timing["ended_wall_ns"])
        self.assertEqual(
            publication["started_monotonic_ns"],
            timing["ended_monotonic_ns"],
        )
        self.assertGreaterEqual(publication["duration_ns"], 0)
        self.assertEqual(
            publication["completion_sha256"],
            hashlib.sha256(
                self.driver.task_completion_path(0).read_bytes()
            ).hexdigest(),
        )
        self.assertGreaterEqual(
            publication["ended_wall_ns"],
            self.driver.task_completion_path(0).stat().st_mtime_ns,
        )

    def test_recovery_observation_includes_delayed_durable_tail(self):
        completion = self.driver.run_task(0, gate=True)
        publication_path = self.driver.task_publication_path(0)
        publication_path.unlink()
        timing = json.loads(Path(completion["timing_receipt"]["path"]).read_text())
        delayed_wall = timing["ended_wall_ns"] + 250_000_000
        delayed_monotonic = timing["ended_monotonic_ns"] + 250_000_000
        with (
            patch("swebench_triad_eval.cli.time.time_ns", return_value=delayed_wall),
            patch(
                "swebench_triad_eval.cli.time.monotonic_ns",
                return_value=delayed_monotonic,
            ),
        ):
            self.driver._recover_task_publication(0, completion)
        publication = json.loads(publication_path.read_text())
        self.assertTrue(publication["recovered_after_crash"])
        self.assertEqual(publication["ended_wall_ns"], delayed_wall)
        self.assertEqual(publication["ended_monotonic_ns"], delayed_monotonic)
        self.assertEqual(publication["duration_ns"], 250_000_000)

    def test_grade_all_rejects_noncanonical_accepted_triad_before_outcomes(self):
        slot = self.driver.acquire_runtime_lane(1, slot_index=0)
        stage = self.operations.stage_task(1, slot=slot)
        for config in self.driver.by_task[1]:
            self.driver._accepted_or_run(config, stage, slot=slot)
        self.driver.release_runtime_lane(slot)

        def reject_task_one(rows):
            if rows[0]["task_index"] == 1:
                raise ValueError("task-1 triad drifted")
            return {"status": "PASS"}

        self.driver.triad_validator = reject_task_one
        with self.assertRaisesRegex(RuntimeError, "task triad validation"):
            self.driver.grade_all()
        self.assertEqual(self.operations.grade_calls, [])

    def test_complete_audit_revalidates_the_global_manifest(self):
        self.driver.gate(auto_run_full=True)

        def reject_task_one(rows):
            if any(row["task_index"] == 1 for row in rows):
                raise ValueError("task-1 triad drifted")
            return triad_verification(rows)

        self.driver.triad_validator = reject_task_one
        with self.assertRaisesRegex(RuntimeError, "global triad validation"):
            self.driver.audit()

    def test_complete_audit_rejects_cross_task_namespace_and_root_reuse(self):
        original_run_cell = self.operations.run_cell

        def run_cell_with_cross_task_reuse(
            config, stage, *, generation, slot
        ):
            row = original_run_cell(
                config, stage, generation=generation, slot=slot
            )
            row["audit_namespace"] = config.capability.arm.value
            row["audit_root"] = f"root-{config.capability.arm.value}"
            return row

        def reject_global_reuse(rows):
            namespaces = [row["audit_namespace"] for row in rows]
            roots = [row["audit_root"] for row in rows]
            if len(namespaces) != len(set(namespaces)):
                raise ValueError("namespace was reused across tasks")
            if len(roots) != len(set(roots)):
                raise ValueError("lifecycle root was reused across tasks")
            return triad_verification(rows)

        self.operations.run_cell = run_cell_with_cross_task_reuse
        self.driver.triad_validator = reject_global_reuse
        self.driver.gate(auto_run_full=True)
        with self.assertRaisesRegex(RuntimeError, "global triad validation"):
            self.driver.audit()

    def test_complete_outcomes_resume_eviction_without_restage_or_rerun(self):
        original_evict = self.operations.evict_task

        def fail_once(task_index, stage, *, slot):
            self.operations.evict_task = original_evict
            raise RuntimeError("simulated crash before eviction receipt")

        self.operations.evict_task = fail_once
        with self.assertRaisesRegex(RuntimeError, "before eviction"):
            self.driver.run_task(0, gate=True)
        self.assertTrue(self.driver.task_complete(0))
        self.assertFalse((self.root / "run/full/task-0000.json").exists())

        resumed_operations = FakeLifecycleOperations(self.root / "evidence")
        resumed = LifecycleDriver(
            root=self.root / "run",
            configs=configs_for_two_tasks(),
            owner=OwnerIdentity("host", "boot", 202, 2002),
            owner_is_alive=lambda _owner: False,
            operations=resumed_operations,
            evidence_root=self.root / "evidence",
            endpoint_validator=lambda row: None,
            triad_validator=lambda _rows: {"status": "PASS"},
            preflight_expectations=preflight_expectations(),
        )
        result = resumed.run_task(0, gate=True)
        self.assertEqual(resumed_operations.staged, [])
        self.assertEqual(resumed_operations.run_calls, [])
        self.assertEqual(resumed_operations.grade_calls, [])
        self.assertEqual(resumed_operations.evicted, [0])
        self.assertEqual(result["accepted_cells"], 3)

    def test_every_task_phase_crash_resumes_without_duplicate_publication(self):
        checkpoints = (
            "stage",
            "arm_native",
            "arm_amg_compaction_only",
            "arm_amg_memory",
            "triad",
            "grade_native",
            "grade_amg_compaction_only",
            "grade_amg_memory",
            "eviction",
            "publication",
        )

        def build(case_root, operations, pid):
            return LifecycleDriver(
                root=case_root / "run",
                configs=configs_for_two_tasks(),
                owner=OwnerIdentity("host", "boot", pid, pid * 10),
                owner_is_alive=lambda _owner: False,
                operations=operations,
                evidence_root=case_root / "evidence",
                endpoint_validator=lambda row: (
                    None
                    if row.get("marker") == "valid-endpoint"
                    else (_ for _ in ()).throw(ValueError("invalid endpoint"))
                ),
                triad_validator=triad_verification,
                preflight_expectations=preflight_expectations(),
                assigned_task_indices=(0,),
            )

        for offset, checkpoint in enumerate(checkpoints, start=1):
            with self.subTest(checkpoint=checkpoint), tempfile.TemporaryDirectory() as raw:
                case_root = Path(raw)
                operations = FakeLifecycleOperations(case_root / "evidence")
                driver = build(case_root, operations, 500 + offset)
                crashed = False

                def crash_once():
                    nonlocal crashed
                    if not crashed:
                        crashed = True
                        raise RuntimeError(f"crash after {checkpoint}")

                if checkpoint == "stage":
                    original = operations.stage_task

                    def stage(task_index, *, slot):
                        result = original(task_index, slot=slot)
                        crash_once()
                        return result

                    operations.stage_task = stage
                elif checkpoint.startswith("arm_"):
                    target = checkpoint.removeprefix("arm_")
                    original = driver._accepted_or_run

                    def accepted(config, stage, *, slot):
                        result = original(config, stage, slot=slot)
                        if config.capability.arm.value == target:
                            crash_once()
                        return result

                    driver._accepted_or_run = accepted
                elif checkpoint == "triad":
                    def verify(rows):
                        result = triad_verification(rows)
                        crash_once()
                        return result

                    driver.triad_validator = verify
                elif checkpoint.startswith("grade_"):
                    target = checkpoint.removeprefix("grade_")
                    original = driver._grade_if_missing

                    def grade(key, *, slot):
                        result = original(key, slot=slot)
                        if key.arm == target:
                            crash_once()
                        return result

                    driver._grade_if_missing = grade
                elif checkpoint == "eviction":
                    original = operations.evict_task

                    def evict(task_index, stage, *, slot):
                        result = original(task_index, stage, slot=slot)
                        crash_once()
                        return result

                    operations.evict_task = evict
                else:
                    original = driver.task_result

                    def publish(*args, **kwargs):
                        result = original(*args, **kwargs)
                        crash_once()
                        return result

                    driver.task_result = publish

                with self.assertRaisesRegex(RuntimeError, "crash after"):
                    driver.run_task(0, gate=True)

                resumed_operations = FakeLifecycleOperations(case_root / "evidence")
                resumed = build(case_root, resumed_operations, 700 + offset)
                result = resumed.run_task(0, gate=True)
                self.assertEqual(result["accepted_cells"], 3)
                self.assertEqual(result["official_outcomes"], 3)
                self.assertEqual(
                    len(list((case_root / "run/state/accepted").glob("0000-*.json"))),
                    3,
                )
                self.assertEqual(
                    len(list((case_root / "run/state/outcomes").glob("0000-*.json"))),
                    3,
                )
                self.assertTrue((case_root / "run/full/task-0000.json").is_file())

    def test_real_runner_task_id_is_accepted_and_private_payloads_are_used(self):
        self.driver.run_task(0, gate=True)
        endpoint_path = self.root / (
            "run/state/attempts/0000-native/00000001/endpoint.json"
        )
        endpoint = json.loads(endpoint_path.read_text())
        self.assertEqual(endpoint["task_id"], "owner__repo-0")
        self.assertNotIn("instance_id", endpoint)
        prediction_path = self.root / (
            "run/state/attempts/0000-native/00000001/prediction.json"
        )
        prediction = json.loads(prediction_path.read_text())
        self.assertEqual(prediction["instance_id"], "owner__repo-0")

    def test_heartbeat_keeps_the_heartbeat_schema(self) -> None:
        heartbeat = self.driver.write_heartbeat()
        self.assertEqual(heartbeat["schema"], "swebench_triad_heartbeat_v1")

    def test_cleanup_is_operations_owned_and_retains_allocation(self) -> None:
        receipt = self.driver.cleanup()
        self.assertEqual(receipt["owned_residue"], 0)
        self.assertTrue(receipt["allocation_retained"])
        self.assertEqual(self.operations.cleanup_calls, 1)

    def test_privacy_audit_rejects_protected_patch_fields(self) -> None:
        full = self.root / "run" / "full"
        full.mkdir(parents=True, exist_ok=True)
        (full / "results.jsonl").write_text(
            '{"model_patch":"secret"}\n', encoding="utf-8"
        )
        (full / "official-summary.json").write_text(
            "{}\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "protected fields"):
            self.driver.privacy_audit()

    def test_privacy_audit_requires_both_public_artifacts(self) -> None:
        full = self.root / "run" / "full"
        full.mkdir(parents=True, exist_ok=True)
        paths = (
            full / "results.jsonl",
            full / "official-summary.json",
        )
        for missing_index in range(2):
            for path in paths:
                path.write_text("{}\n", encoding="utf-8")
            paths[missing_index].unlink()
            with self.subTest(missing=paths[missing_index].name):
                with self.assertRaisesRegex(RuntimeError, "requires"):
                    self.driver.privacy_audit()

    def test_complete_audit_requires_exact_canonical_public_artifacts(self):
        self.driver.gate(auto_run_full=True)
        self.assertEqual(self.driver.audit()["status"], "PASS")
        paths = (
            self.root / "run/full/results.jsonl",
            self.root / "run/full/official-summary.json",
        )
        originals = {path: path.read_bytes() for path in paths}
        for path in paths:
            with self.subTest(path=path.name, mutation="missing"):
                path.unlink()
                with self.assertRaisesRegex(RuntimeError, "canonical"):
                    self.driver.audit()
                path.write_bytes(originals[path])
            with self.subTest(path=path.name, mutation="stale"):
                path.write_bytes(originals[path] + b" ")
                with self.assertRaisesRegex(RuntimeError, "canonical"):
                    self.driver.audit()
                path.write_bytes(originals[path])

    def test_explicit_shard_cannot_execute_an_unassigned_task(self) -> None:
        shard = LifecycleDriver(
            root=self.root / "sharded-run",
            configs=configs_for_two_tasks(),
            owner=OwnerIdentity("host", "boot", 303, 3003),
            owner_is_alive=lambda _owner: False,
            operations=FakeLifecycleOperations(self.root / "sharded-evidence"),
            evidence_root=self.root / "sharded-evidence",
            endpoint_validator=lambda row: None,
            triad_validator=lambda _rows: {"status": "PASS"},
            preflight_expectations=preflight_expectations(),
            assigned_task_indices=(0,),
        )
        with self.assertRaisesRegex(ValueError, "outside the assigned shard"):
            shard.run_task(1)

    def test_disjoint_live_driver_startup_reconciliation_is_shard_scoped(self):
        lease_root = self.root / "leased-run" / "state" / "leases"
        owner_a = OwnerIdentity("host-a", "boot-a", 401, 4001)
        owner_b = OwnerIdentity("host-b", "boot-b", 402, 4002)
        lease_a = DriverLeaseRegistry(
            lease_root,
            owner=owner_a,
            assigned_task_indices=(0,),
            local_owner_is_alive=lambda _owner: True,
        )
        lease_b = DriverLeaseRegistry(
            lease_root,
            owner=owner_b,
            assigned_task_indices=(1,),
            local_owner_is_alive=lambda _owner: True,
        )
        lease_a.start_heartbeat()
        self.addCleanup(lease_a.release)
        self.addCleanup(lease_b.release)
        operations = FakeLifecycleOperations(self.root / "leased-evidence")
        driver = LifecycleDriver(
            root=self.root / "leased-run",
            configs=configs_for_two_tasks(),
            owner=owner_b,
            owner_is_alive=lease_b.owner_is_alive,
            operations=operations,
            evidence_root=self.root / "leased-evidence",
            endpoint_validator=lambda row: None,
            triad_validator=lambda _rows: {"status": "PASS"},
            preflight_expectations=preflight_expectations(),
            assigned_task_indices=(1,),
            lease_registry=lease_b,
        )
        driver.live_preflight()
        self.assertIn(("reconcile_startup", (1,)), operations.events)
        self.assertNotIn(("reconcile_startup", (0,)), operations.events)


class CliParserTest(unittest.TestCase):
    def test_all_lifecycle_commands_are_declared(self) -> None:
        parser = build_parser()
        for command in (
            "preflight",
            "gate",
            "run",
            "resume",
            "grade",
            "status",
            "audit",
            "cleanup",
        ):
            with self.subTest(command=command):
                arguments = parser.parse_args(
                    [command, "--config", "/tmp/run-config.json"]
                )
                self.assertEqual(arguments.command, command)

    def test_gate_command_cannot_stop_before_the_full_manifest(self) -> None:
        class Driver:
            def __init__(self):
                self.values = []

            def gate(self, *, auto_run_full):
                self.values.append(auto_run_full)
                return {"status": "continued"}

        driver = Driver()
        with patch(
            "swebench_triad_eval.cli.driver_from_config", return_value=driver
        ), patch("swebench_triad_eval.cli.write_stdout"):
            self.assertEqual(
                main(["gate", "--config", "/tmp/run-config.json"]), 0
            )
        self.assertEqual(driver.values, [True])

    def test_gate_command_rejects_a_partial_task_shard(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "gate",
                    "--config",
                    "/tmp/run-config.json",
                    "--task-range",
                    "0:1",
                ]
            )


class ProductionBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path, self.config = production_config(self.root)
        self.operations = FakeLifecycleOperations(self.root / "evidence")

    def build_driver(self):
        return driver_from_config(
            self.config_path,
            owner=OwnerIdentity("host", "boot", 101, 1001),
            owner_is_alive=lambda _owner: False,
            operations_factory=lambda _config, _configs: self.operations,
        )

    def test_production_config_binds_all_1500_cells_to_concrete_operations(
        self,
    ) -> None:
        driver = self.build_driver()
        self.assertEqual(len(driver.configs), 1500)
        self.assertEqual(len(driver.by_task), 500)
        self.assertIs(driver.operations, self.operations)
        self.assertEqual(driver.root, (self.root / "run").resolve())

    def test_config_and_manifest_digests_fail_closed(self) -> None:
        bad = dict(self.config)
        bad["unexpected"] = True
        self.config_path.write_bytes(canonical_json_bytes(bad))
        with self.assertRaisesRegex(ValueError, "fields drifted"):
            self.build_driver()

        self.config_path, self.config = production_config(self.root)
        self.config["manifest_sha256"] = SHA_D
        self.config_path.write_bytes(canonical_json_bytes(self.config))
        with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
            self.build_driver()


class SharedModelPoolPreflightTest(unittest.TestCase):
    @staticmethod
    def fixture():
        expectations = preflight_expectations()
        expectations["gpu_count"] = 8
        expectations["model_id"] = "Qwen/Qwen3.5-4B"
        expectations["model_pid"] = 303
        expectations["model_start_ticks"] = 3003
        expectations["shared_model_pool"] = {
            "owner": "amg-external-eval-g-dp8-swe-0818",
            "readiness_sha256": SHA_A,
            "marker_lease_sha256": SHA_B,
            "replica_index": 3,
            "replica_count": 8,
            "gpu_index": 3,
            "gpu_uuid": "GPU-expected",
            "model_revision": "3" * 40,
            "model_port": 18021,
            "proxy_port": 16383,
        }
        snapshot = valid_preflight_snapshot()
        snapshot["pod"] = {**snapshot["pod"], "gpu_count": 8}
        snapshot["model_process"] = {
            **snapshot["model_process"],
            "pid": 303,
            "start_ticks": 3003,
        }
        snapshot["vllm"] = {
            **snapshot["vllm"],
            "model_id": "Qwen/Qwen3.5-4B",
        }
        snapshot["shared_model_pool"] = {
            "status": "PASS",
            "owner": "amg-external-eval-g-dp8-swe-0818",
            "readiness_sha256": SHA_A,
            "marker_lease_sha256": SHA_B,
            "replica_index": 3,
            "replica_count": 8,
            "gpu_index": 3,
            "gpu_uuid": "GPU-expected",
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "3" * 40,
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
                "upstream_base_url": "http://127.0.0.1:18021",
                "upstream_base_url_sha256": hashlib.sha256(
                    b"http://127.0.0.1:18021"
                ).hexdigest(),
            },
            "assigned_gpu_process_pids": [503],
            "all_replicas_alive": True,
            "all_endpoints_healthy": True,
            "assignment_algorithm": "uint64_be(sha256(task_id)[:8]) % 8",
            "cleanup_policy": "retain_external_pool",
        }
        return snapshot, expectations

    def test_shared_pool_preflight_is_exact_and_fail_closed(self):
        snapshot, expectations = self.fixture()
        self.assertEqual(
            validate_preflight_snapshot(snapshot, expectations)["status"], "PASS"
        )
        for field, value in (
            ("replica_index", 2),
            ("all_replicas_alive", False),
            ("cleanup_policy", "stop_external_pool"),
            ("assigned_gpu_process_pids", []),
            ("server_listener_pids", [999]),
        ):
            with self.subTest(field=field):
                broken = dict(snapshot)
                broken["shared_model_pool"] = dict(snapshot["shared_model_pool"])
                broken["shared_model_pool"][field] = value
                with self.assertRaises(PreflightContractError):
                    validate_preflight_snapshot(broken, expectations)

        extra = dict(snapshot)
        extra["shared_model_pool"] = {
            **snapshot["shared_model_pool"],
            "unexpected": True,
        }
        with self.assertRaises(PreflightContractError):
            validate_preflight_snapshot(extra, expectations)

    def test_shared_pool_listener_census_is_exact_and_fail_closed(self):
        snapshot, expectations = self.fixture()
        self.assertEqual(
            validate_preflight_snapshot(snapshot, expectations)["status"],
            "PASS",
        )
        mutations = (
            ("source", "/proc/net/tcp6"),
            ("family", "ipv6"),
            ("address", "0.0.0.0"),
            ("port", 18022),
            ("inode", 0),
            ("owner_pids", [999]),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                broken = copy.deepcopy(snapshot)
                broken["shared_model_pool"]["server_listener_census"][
                    field
                ] = value
                with self.assertRaises(PreflightContractError):
                    validate_preflight_snapshot(broken, expectations)

        for change in ("missing", "extra"):
            with self.subTest(change=change):
                broken = copy.deepcopy(snapshot)
                census = broken["shared_model_pool"][
                    "server_listener_census"
                ]
                if change == "missing":
                    census.pop("inode")
                else:
                    census["unexpected"] = True
                with self.assertRaises(PreflightContractError):
                    validate_preflight_snapshot(broken, expectations)


if __name__ == "__main__":
    unittest.main()
