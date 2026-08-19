from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

from swebench_triad_eval import resource_guard
from swebench_triad_eval.resource_guard import (
    CgroupV1CellEnvelope,
    CgroupV1Limits,
    GuardedEpisodeSandboxMixin,
    QuotaMountSpec,
    ResourceGuardError,
    RootfsMutationGuard,
    TmpfsQuotaMounts,
)


class CgroupV1NamespaceHelperTest(unittest.TestCase):
    request = {
        "relative_path": (
            "amg-external-eval-container-runtime-v1/"
            "swebench-triad-v1/0000-native"
        ),
        "memory_bytes": 8 * 1024 * 1024,
        "max_processes": 16,
    }

    def call_prepare(self, controller_roots):
        written_values = {}

        def cgroup_directory(controller, relative_path):
            return controller_roots[controller] / relative_path

        def cgroup_write(path, value):
            written_values[path] = value

        def cgroup_read(path):
            return str(written_values[path])

        with (
            mock.patch.object(
                resource_guard,
                "_cgroup_directory",
                side_effect=cgroup_directory,
            ),
            mock.patch.object(
                resource_guard,
                "_cgroup_write",
                side_effect=cgroup_write,
            ),
            mock.patch.object(
                resource_guard,
                "_cgroup_read",
                side_effect=cgroup_read,
            ),
            mock.patch.object(
                resource_guard,
                "_read_pid_file",
                return_value=[],
            ),
        ):
            return resource_guard._helper_prepare(self.request)

    def test_prepare_creates_only_the_missing_owned_parent_hierarchy(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            controller_roots = {
                "memory": root / "memory",
                "pids": root / "pids",
            }
            for controller_root in controller_roots.values():
                controller_root.mkdir()

            receipt = self.call_prepare(controller_roots)

            self.assertTrue(receipt["limits_applied_before_tasks"])
            expected_directories = set()
            for controller in controller_roots:
                runtime_parent = (
                    Path(controller) / "amg-external-eval-container-runtime-v1"
                )
                evaluation_parent = runtime_parent / "swebench-triad-v1"
                expected_directories.update(
                    {
                        runtime_parent,
                        evaluation_parent,
                        evaluation_parent / "0000-native",
                    }
                )
            actual_directories = {
                path.relative_to(root)
                for path in root.rglob("*")
                if path.is_dir() and path not in controller_roots.values()
            }
            self.assertEqual(actual_directories, expected_directories)
            for controller in controller_roots:
                runtime_parent = (
                    controller_roots[controller]
                    / "amg-external-eval-container-runtime-v1"
                )
                self.assertTrue(runtime_parent.is_dir())

    def test_prepare_rejects_symlinked_runtime_parents(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outside = root / "outside"
            controller_roots = {
                "memory": root / "memory",
                "pids": root / "pids",
            }
            for controller, controller_root in controller_roots.items():
                controller_root.mkdir()
                outside_runtime = outside / controller
                (outside_runtime / resource_guard.CGROUP_EVALUATION_PARENT).mkdir(
                    parents=True
                )
                (controller_root / resource_guard.CGROUP_RUNTIME_PARENT).symlink_to(
                    outside_runtime,
                    target_is_directory=True,
                )

            with self.assertRaisesRegex(ResourceGuardError, "real directory"):
                self.call_prepare(controller_roots)

            for controller, controller_root in controller_roots.items():
                self.assertTrue(
                    (controller_root / resource_guard.CGROUP_RUNTIME_PARENT).is_symlink()
                )
                self.assertFalse(
                    (
                        outside
                        / controller
                        / resource_guard.CGROUP_EVALUATION_PARENT
                        / "0000-native"
                    ).exists()
                )

    def test_prepare_rejects_symlinked_evaluation_parents(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outside = root / "outside"
            controller_roots = {
                "memory": root / "memory",
                "pids": root / "pids",
            }
            for controller, controller_root in controller_roots.items():
                evaluation_parent = (
                    controller_root
                    / resource_guard.CGROUP_RUNTIME_PARENT
                    / resource_guard.CGROUP_EVALUATION_PARENT
                )
                evaluation_parent.parent.mkdir(parents=True)
                outside_evaluation = outside / controller
                (outside_evaluation / "0000-native").mkdir(parents=True)
                evaluation_parent.symlink_to(
                    outside_evaluation,
                    target_is_directory=True,
                )

            with self.assertRaisesRegex(ResourceGuardError, "real directory"):
                self.call_prepare(controller_roots)

            for controller_root in controller_roots.values():
                self.assertTrue(
                    (
                        controller_root
                        / resource_guard.CGROUP_RUNTIME_PARENT
                        / resource_guard.CGROUP_EVALUATION_PARENT
                    ).is_symlink()
                )

    def test_prepare_rejects_symlinked_cell_leaves(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outside = root / "outside"
            controller_roots = {
                "memory": root / "memory",
                "pids": root / "pids",
            }
            for controller, controller_root in controller_roots.items():
                cell_parent = (
                    controller_root
                    / resource_guard.CGROUP_RUNTIME_PARENT
                    / resource_guard.CGROUP_EVALUATION_PARENT
                )
                cell_parent.mkdir(parents=True)
                outside_cell = outside / controller
                outside_cell.mkdir(parents=True)
                (cell_parent / "0000-native").symlink_to(
                    outside_cell,
                    target_is_directory=True,
                )

            with self.assertRaisesRegex(ResourceGuardError, "real directory"):
                self.call_prepare(controller_roots)

            for controller, controller_root in controller_roots.items():
                cell = (
                    controller_root
                    / resource_guard.CGROUP_RUNTIME_PARENT
                    / resource_guard.CGROUP_EVALUATION_PARENT
                    / "0000-native"
                )
                self.assertTrue(cell.is_symlink())
                self.assertEqual(list((outside / controller).iterdir()), [])

    def test_prepare_rolls_back_memory_for_symlinked_pids_cell(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outside_cell = root / "outside" / "pids-cell"
            outside_cell.mkdir(parents=True)
            controller_roots = {
                "memory": root / "memory",
                "pids": root / "pids",
            }
            for controller_root in controller_roots.values():
                controller_root.mkdir()
            pids_cell_parent = (
                controller_roots["pids"]
                / resource_guard.CGROUP_RUNTIME_PARENT
                / resource_guard.CGROUP_EVALUATION_PARENT
            )
            pids_cell_parent.mkdir(parents=True)
            (pids_cell_parent / "0000-native").symlink_to(
                outside_cell,
                target_is_directory=True,
            )

            with self.assertRaisesRegex(ResourceGuardError, "real directory"):
                self.call_prepare(controller_roots)

            self.assertFalse(
                (
                    controller_roots["memory"]
                    / resource_guard.CGROUP_RUNTIME_PARENT
                ).exists()
            )
            self.assertTrue((pids_cell_parent / "0000-native").is_symlink())
            self.assertEqual(list(outside_cell.iterdir()), [])

    def test_prepare_rolls_back_created_controller_hierarchy_on_rejection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outside_runtime = root / "outside" / "pids"
            (outside_runtime / resource_guard.CGROUP_EVALUATION_PARENT).mkdir(
                parents=True
            )
            controller_roots = {
                "memory": root / "memory",
                "pids": root / "pids",
            }
            for controller_root in controller_roots.values():
                controller_root.mkdir()
            (
                controller_roots["pids"] / resource_guard.CGROUP_RUNTIME_PARENT
            ).symlink_to(outside_runtime, target_is_directory=True)

            with self.assertRaisesRegex(ResourceGuardError, "real directory"):
                self.call_prepare(controller_roots)

            self.assertFalse(
                (
                    controller_roots["memory"]
                    / resource_guard.CGROUP_RUNTIME_PARENT
                ).exists()
            )
            self.assertFalse(
                (
                    outside_runtime
                    / resource_guard.CGROUP_EVALUATION_PARENT
                    / "0000-native"
                ).exists()
            )

    def test_snapshot_aggregates_nested_failure_counters(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            memory = root / "memory"
            pids = root / "pids"
            memory_child = memory / "docker-cell"
            pids_child = pids / "docker-cell"
            for directory in (memory, pids, memory_child, pids_child):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "tasks").write_text(
                    "101\n" if directory.name == "docker-cell" else "",
                    encoding="ascii",
                )
                (directory / "cgroup.procs").write_text(
                    "101\n" if directory.name == "docker-cell" else "",
                    encoding="ascii",
                )
            for directory, peak, failures in (
                (memory, 1024, 0),
                (memory_child, 4096, 3),
            ):
                (directory / "memory.max_usage_in_bytes").write_text(
                    str(peak), encoding="ascii"
                )
                (directory / "memory.failcnt").write_text(
                    str(failures), encoding="ascii"
                )
            (memory / "memory.limit_in_bytes").write_text(
                str(8 * 1024 * 1024), encoding="ascii"
            )
            (memory / "memory.memsw.limit_in_bytes").write_text(
                str(8 * 1024 * 1024), encoding="ascii"
            )
            for directory, current, events in (
                (pids, 1, 0),
                (pids_child, 1, 2),
            ):
                (directory / "pids.current").write_text(
                    str(current), encoding="ascii"
                )
                (directory / "pids.events").write_text(
                    f"max {events}\n", encoding="ascii"
                )
            (pids / "pids.max").write_text("16", encoding="ascii")

            def cgroup_directory(controller, _relative):
                return {"memory": memory, "pids": pids}[controller]

            with (
                mock.patch.object(
                    resource_guard, "_cgroup_directory", side_effect=cgroup_directory
                ),
                mock.patch.object(
                    resource_guard,
                    "_pid_memberships",
                    return_value={
                        "memory": "/owned/docker-cell",
                        "pids": "/owned/docker-cell",
                    },
                ),
            ):
                snapshot = resource_guard._helper_snapshot(self.request)

            self.assertEqual(snapshot["memory"]["max_usage_in_bytes"], 4096)
            self.assertEqual(snapshot["memory"]["failcnt"], 3)
            self.assertEqual(snapshot["pids"]["max_events"], 2)

    def test_prepare_rejects_a_path_outside_the_exact_owned_hierarchy(self) -> None:
        with mock.patch.object(resource_guard, "_cgroup_directory") as directory:
            with self.assertRaisesRegex(ResourceGuardError, "outside the owned scope"):
                resource_guard._helper_prepare(
                    {
                        "relative_path": (
                            "amg-external-eval-container-runtime-v1/"
                            "swebench-triad-v1/../escape"
                        ),
                        "memory_bytes": 8 * 1024 * 1024,
                        "max_processes": 16,
                    }
                )
        directory.assert_not_called()


class FakeCgroupBackend:
    def __init__(self) -> None:
        self.events = []
        self.prepared = None
        self.snapshot_value = {
            "schema": "swebench_cgroup_v1_snapshot_v1",
            "memory": {
                "tasks": [101, 102],
                "cgroup_procs": [101, 102],
                "limit_in_bytes": 8 * 1024 * 1024,
                "memsw_limit_in_bytes": 8 * 1024 * 1024,
                "max_usage_in_bytes": 7 * 1024 * 1024,
                "failcnt": 3,
            },
            "pids": {
                "tasks": [101, 102],
                "cgroup_procs": [101, 102],
                "max": 16,
                "current": 2,
                "max_events": 4,
            },
            "memberships": {
                "101": {
                    "memory": "/amg-external-eval-container-runtime-v1/"
                    "swebench-triad-v1/0000-native/docker-a",
                    "pids": "/amg-external-eval-container-runtime-v1/"
                    "swebench-triad-v1/0000-native/docker-a",
                },
                "102": {
                    "memory": "/amg-external-eval-container-runtime-v1/"
                    "swebench-triad-v1/0000-native/docker-a",
                    "pids": "/amg-external-eval-container-runtime-v1/"
                    "swebench-triad-v1/0000-native/docker-a",
                },
            },
        }
        self.removed = False

    def prepare(self, relative_path, limits):
        self.events.append(("prepare", relative_path, limits))
        self.prepared = (relative_path, limits)
        return {
            "schema": "swebench_cgroup_v1_prepare_v1",
            "relative_path": relative_path,
            "limits_applied_before_tasks": True,
            "memory": {
                "limit_in_bytes": limits.memory_bytes,
                "memsw_limit_in_bytes": limits.memory_bytes,
                "swappiness": 0,
                "use_hierarchy": 1,
                "tasks": [],
            },
            "pids": {"max": limits.max_processes, "tasks": []},
        }

    def snapshot(self, relative_path):
        self.events.append(("snapshot", relative_path))
        return self.snapshot_value

    def remove(self, relative_path):
        self.events.append(("remove", relative_path))
        self.removed = True
        return {
            "schema": "swebench_cgroup_v1_remove_v1",
            "relative_path": relative_path,
            "memory_tasks_empty": True,
            "pids_tasks_empty": True,
            "removed": True,
        }


class CgroupV1StructureLockTest(unittest.TestCase):
    def test_mount_namespace_requests_serialize_helper_invocations(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            backend = resource_guard.MountNamespaceCgroupV1Backend(
                namespace_pid=os.getpid(),
            )
            start = threading.Barrier(3)
            counter_lock = threading.Lock()
            active = 0
            max_active = 0
            errors: list[BaseException] = []

            def run_helper(*args, **kwargs):
                nonlocal active, max_active
                with counter_lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)
                with counter_lock:
                    active -= 1
                return subprocess.CompletedProcess(
                    args[0],
                    0,
                    '{"schema":"swebench_cgroup_v1_snapshot_v1"}',
                    "",
                )

            def request() -> None:
                try:
                    start.wait(timeout=2)
                    backend.snapshot(
                        resource_guard.CGROUP_RELATIVE_PREFIX + "/0000-native"
                    )
                except BaseException as error:  # propagate worker failures below
                    errors.append(error)

            with (
                mock.patch.object(
                    resource_guard,
                    "CGROUP_STRUCTURE_LOCK_PATH",
                    Path(raw) / "cgroup-structure.lock",
                ),
                mock.patch.object(Path, "exists", return_value=True),
                mock.patch.object(
                    resource_guard.subprocess,
                    "run",
                    side_effect=run_helper,
                ) as helper,
            ):
                threads = [threading.Thread(target=request) for _ in range(2)]
                for thread in threads:
                    thread.start()
                start.wait(timeout=2)
                for thread in threads:
                    thread.join(timeout=2)

            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(helper.call_count, 2)
            self.assertEqual(max_active, 1)



class CgroupV1CellEnvelopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeCgroupBackend()
        self.limits = CgroupV1Limits(
            memory_bytes=8 * 1024 * 1024,
            max_processes=16,
        )
        self.envelope = CgroupV1CellEnvelope(
            cell_name="0000-native",
            limits=self.limits,
            backend=self.backend,
        )

    def test_limits_are_applied_before_docker_can_attach_or_exec(self) -> None:
        with self.assertRaises(ResourceGuardError):
            self.envelope.docker_resource_arguments()

        receipt = self.envelope.prepare()

        self.assertTrue(receipt["limits_applied_before_tasks"])
        self.assertEqual(receipt["memory"]["tasks"], [])
        self.assertEqual(receipt["pids"]["tasks"], [])
        self.assertEqual(
            self.envelope.docker_resource_arguments(),
            (
                "--cgroup-parent=/amg-external-eval-container-runtime-v1/"
                "swebench-triad-v1/0000-native",
                "--memory=8388608",
                "--memory-swap=8388608",
                "--memory-swappiness=0",
                "--pids-limit=16",
            ),
        )
        self.assertEqual(self.backend.events[0][0], "prepare")

    def test_all_memory_and_pids_descendants_must_inherit_the_cell_parent(self) -> None:
        self.envelope.prepare()
        verified = self.envelope.verify_descendants(container_init_pid=101)
        self.assertEqual(verified["descendant_pids"], [101, 102])
        self.assertTrue(verified["memory_pids_task_sets_equal"])

        self.backend.snapshot_value["memberships"]["102"]["pids"] = "/escaped"
        with self.assertRaisesRegex(ResourceGuardError, "escaped"):
            self.envelope.verify_descendants(container_init_pid=101)

    def test_kubernetes_prefix_preserves_owned_subtree_membership(self) -> None:
        self.envelope.prepare()
        prefix = "/kubepods/burstable/pod-test/container-test"
        for membership in self.backend.snapshot_value["memberships"].values():
            for controller in ("memory", "pids"):
                membership[controller] = prefix + membership[controller]

        verified = self.envelope.verify_descendants(container_init_pid=101)

        self.assertEqual(verified["descendant_pids"], [101, 102])
        self.backend.snapshot_value["memberships"]["102"]["memory"] = (
            prefix
            + "/amg-external-eval-container-runtime-v1-shadow/"
            "swebench-triad-v1/0000-native"
        )
        with self.assertRaisesRegex(ResourceGuardError, "escaped"):
            self.envelope.verify_descendants(container_init_pid=101)

    def test_peak_and_failure_counters_are_recorded_before_empty_teardown(self) -> None:
        self.envelope.prepare()
        self.envelope.verify_descendants(container_init_pid=101)
        self.backend.snapshot_value["memory"]["tasks"] = []
        self.backend.snapshot_value["memory"]["cgroup_procs"] = []
        self.backend.snapshot_value["pids"]["tasks"] = []
        self.backend.snapshot_value["pids"]["cgroup_procs"] = []
        self.backend.snapshot_value["memberships"] = {}

        receipt = self.envelope.teardown()

        self.assertEqual(receipt["memory_peak_bytes"], 7 * 1024 * 1024)
        self.assertEqual(receipt["memory_failcnt"], 3)
        self.assertEqual(receipt["pids_peak"], 2)
        self.assertEqual(receipt["pids_max_events"], 4)
        self.assertTrue(receipt["memory_tasks_empty"])
        self.assertTrue(receipt["pids_tasks_empty"])
        self.assertTrue(self.backend.removed)

    def test_live_counter_observation_survives_child_cgroup_removal(self) -> None:
        self.envelope.prepare()
        observed = self.envelope.observe()

        self.assertEqual(observed["memory_failcnt"], 3)
        self.assertEqual(observed["pids_max_events"], 4)

        self.backend.snapshot_value["memory"]["tasks"] = []
        self.backend.snapshot_value["memory"]["cgroup_procs"] = []
        self.backend.snapshot_value["memory"]["max_usage_in_bytes"] = 0
        self.backend.snapshot_value["memory"]["failcnt"] = 0
        self.backend.snapshot_value["pids"]["tasks"] = []
        self.backend.snapshot_value["pids"]["cgroup_procs"] = []
        self.backend.snapshot_value["pids"]["current"] = 0
        self.backend.snapshot_value["pids"]["max_events"] = 0
        self.backend.snapshot_value["memberships"] = {}

        receipt = self.envelope.teardown()

        self.assertEqual(receipt["memory_peak_bytes"], 7 * 1024 * 1024)
        self.assertEqual(receipt["memory_failcnt"], 3)
        self.assertEqual(receipt["pids_peak"], 2)
        self.assertEqual(receipt["pids_max_events"], 4)
        self.assertTrue(self.backend.removed)

    def test_nonempty_memory_or_pids_group_forbids_removal(self) -> None:
        self.envelope.prepare()
        with self.assertRaisesRegex(ResourceGuardError, "nonempty"):
            self.envelope.teardown()
        self.assertFalse(self.backend.removed)

    def test_limit_and_cell_inputs_fail_closed(self) -> None:
        invalid_limits = (
            {"memory_bytes": True, "max_processes": 16},
            {"memory_bytes": 4097, "max_processes": 16},
            {"memory_bytes": 4096, "max_processes": 0},
        )
        for values in invalid_limits:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                CgroupV1Limits(**values)
        for name in ("../escape", "native/x", "", ".hidden"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                CgroupV1CellEnvelope(
                    cell_name=name,
                    limits=self.limits,
                    backend=self.backend,
                )


class FakeMountBackend:
    def __init__(self) -> None:
        self.next_id = 10
        self.mounts = {}
        self.events = []
        self.reported_bytes_delta = 0
        self.reported_inodes_delta = 0

    def mount_tmpfs(self, target, spec, *, owner_uid, owner_gid):
        self.events.append(("mount", target, spec, owner_uid, owner_gid))
        mount_id = self.next_id
        self.next_id += 1
        evidence = {
            "schema": "swebench_tmpfs_mount_v1",
            "mount_id": mount_id,
            "mount_point": str(target),
            "fs_type": "tmpfs",
            "total_bytes": spec.byte_limit + self.reported_bytes_delta,
            "total_inodes": spec.inode_limit + self.reported_inodes_delta,
            "options": [
                "nodev",
                "nosuid",
                f"size={spec.byte_limit}",
                f"nr_inodes={spec.inode_limit}",
            ],
            "owner_uid": owner_uid,
            "owner_gid": owner_gid,
        }
        self.mounts[str(target)] = evidence
        return evidence

    def unmount_tmpfs(self, target, mount_id):
        self.events.append(("unmount", target, mount_id))
        current = self.mounts.get(str(target))
        if current is None or current["mount_id"] != mount_id:
            raise ResourceGuardError("refusing to unmount an unowned mount")
        del self.mounts[str(target)]
        return {"mount_id": mount_id, "unmounted": True}


class TmpfsQuotaMountsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = FakeMountBackend()
        self.workspace_spec = QuotaMountSpec(
            byte_limit=8 * 1024 * 1024,
            inode_limit=4096,
            purpose="workspace",
        )
        self.memory_spec = QuotaMountSpec(
            byte_limit=1024 * 1024,
            inode_limit=256,
            purpose="external-memory",
        )

    def test_workspace_and_external_memory_get_exact_byte_and_inode_limits(self) -> None:
        workspace = self.root / "workspace"
        workspace.mkdir()
        (workspace / "kept.txt").write_text("preserved", encoding="utf-8")
        memory = self.root / "memory"
        memory.mkdir()
        mounts = TmpfsQuotaMounts(self.backend)

        workspace_receipt = mounts.mount_preserving(
            workspace,
            self.workspace_spec,
            owner_uid=os.getuid(),
            owner_gid=os.getgid(),
        )
        memory_receipt = mounts.mount_preserving(
            memory,
            self.memory_spec,
            owner_uid=os.getuid(),
            owner_gid=os.getgid(),
        )

        self.assertEqual(workspace_receipt["total_bytes"], 8 * 1024 * 1024)
        self.assertEqual(workspace_receipt["total_inodes"], 4096)
        self.assertEqual(memory_receipt["total_bytes"], 1024 * 1024)
        self.assertEqual(memory_receipt["total_inodes"], 256)
        self.assertEqual((workspace / "kept.txt").read_text(), "preserved")
        close = mounts.close()
        self.assertEqual(close["unmounted_count"], 2)
        self.assertFalse(self.backend.mounts)

    def test_quota_rounding_or_mount_identity_drift_fails_closed(self) -> None:
        target = self.root / "workspace"
        target.mkdir()
        self.backend.reported_inodes_delta = 1
        mounts = TmpfsQuotaMounts(self.backend)
        with self.assertRaisesRegex(ResourceGuardError, "quota"):
            mounts.mount_preserving(
                target,
                self.workspace_spec,
                owner_uid=os.getuid(),
                owner_gid=os.getgid(),
            )
        self.assertFalse(self.backend.mounts)

    def test_close_unmounts_only_mounts_created_by_this_owner(self) -> None:
        target = self.root / "workspace"
        target.mkdir()
        mounts = TmpfsQuotaMounts(self.backend)
        receipt = mounts.mount_preserving(
            target,
            self.workspace_spec,
            owner_uid=os.getuid(),
            owner_gid=os.getgid(),
        )
        self.backend.mounts[str(target.resolve())]["mount_id"] = receipt["mount_id"] + 1
        with self.assertRaisesRegex(ResourceGuardError, "unowned"):
            mounts.close()


class RootfsMutationGuardTest(unittest.TestCase):
    def test_full_rootfs_receipt_must_remain_identical_before_every_execution(self) -> None:
        state = {"tree_sha256": "a" * 64, "path_count": 72061}
        calls = []

        def attest(_path):
            calls.append(dict(state))
            return {
                "schema": "swebench_verified_rootfs_attestation_v1",
                "status": "pass",
                "image": "swebench/sweb.eval.x:latest",
                "manifest_digest": "sha256:" + "b" * 64,
                "config_digest": "sha256:" + "c" * 64,
                **state,
            }

        guard = RootfsMutationGuard(Path("/cache/task0"), attestor=attest)
        guard.attest()
        self.assertEqual(len(calls), 2)
        state["tree_sha256"] = "d" * 64
        with self.assertRaisesRegex(ResourceGuardError, "mutated"):
            guard.attest()


class DummySandboxBase:
    def __init__(self) -> None:
        self.model_uid = os.getuid()
        self.model_gid = os.getgid()
        self.calls = []

    def attach_workspace(self, root):
        self.calls.append(("attach_workspace", Path(root)))
        return "workspace-snapshot"

    def attach_external_memory(self, root):
        self.calls.append(("attach_external_memory", Path(root)))
        return None

    def _run_namespace(self, workspace_root, **kwargs):
        self.calls.append(("run", Path(workspace_root), kwargs))
        return "ran"

    def close(self):
        self.calls.append(("base_close",))


class DummyGuardedSandbox(GuardedEpisodeSandboxMixin, DummySandboxBase):
    pass


class GuardedEpisodeSandboxMixinTest(unittest.TestCase):
    def test_mounts_and_full_rootfs_attestation_precede_the_cell_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = root / "workspace"
            memory = root / "memory"
            workspace.mkdir()
            memory.mkdir()
            backend = FakeMountBackend()
            mounts = TmpfsQuotaMounts(backend)
            attests = []

            def attest(_path):
                attests.append(True)
                return {
                    "schema": "swebench_verified_rootfs_attestation_v1",
                    "status": "pass",
                    "image": "image",
                    "manifest_digest": "sha256:" + "a" * 64,
                    "config_digest": "sha256:" + "b" * 64,
                    "path_count": 1,
                    "tree_sha256": "c" * 64,
                }

            sandbox = DummyGuardedSandbox()
            sandbox.configure_deployment_guards(
                quota_mounts=mounts,
                workspace_quota=QuotaMountSpec(
                    byte_limit=4096,
                    inode_limit=16,
                    purpose="workspace",
                ),
                external_memory_quota=QuotaMountSpec(
                    byte_limit=4096,
                    inode_limit=16,
                    purpose="external-memory",
                ),
                rootfs_guard=RootfsMutationGuard(root / "cache", attestor=attest),
            )
            self.assertEqual(sandbox.attach_workspace(workspace), "workspace-snapshot")
            sandbox.attach_external_memory(memory)
            self.assertEqual(sandbox._run_namespace(workspace, command="true"), "ran")
            self.assertEqual(sandbox._run_namespace(workspace, command="true"), "ran")
            self.assertEqual(len(attests), 2)
            self.assertEqual(backend.events[0][0], "mount")
            self.assertEqual(sandbox.calls[0][0], "attach_workspace")
            sandbox.close()
            self.assertEqual(sandbox.calls[-1][0], "base_close")
            self.assertFalse(backend.mounts)


if __name__ == "__main__":
    unittest.main()
