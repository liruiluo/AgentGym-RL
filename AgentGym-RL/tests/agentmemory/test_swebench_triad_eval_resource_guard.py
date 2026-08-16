from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from swebench_triad_eval.resource_guard import (
    CgroupV1CellEnvelope,
    CgroupV1Limits,
    GuardedEpisodeSandboxMixin,
    QuotaMountSpec,
    ResourceGuardError,
    RootfsMutationGuard,
    TmpfsQuotaMounts,
)


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
