from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agentmemorygym_verl import orchestrator_lifecycle as lifecycle

MODULE = Path(lifecycle.__file__).resolve()


class TestMarkerTransactions(unittest.TestCase):
    def _marker(self, name: str, path: Path, value: str | None) -> dict:
        return lifecycle._marker_record(name, path, value, 123, "456")

    def test_partial_acquisition_rolls_back_first_marker(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cpu = root / "cpu"
            gpu = root / "gpu"
            cpu.write_text("cpu-owner\n", encoding="utf-8")
            gpu.write_text("gpu-owner\n", encoding="utf-8")
            state = root / "state.json"
            lock = root / "lock"
            with mock.patch.object(
                lifecycle, "process_identity_alive", return_value=True
            ):
                lifecycle.prepare_marker_transaction(
                    state_path=state,
                    lock_path=lock,
                    run_id="test-run",
                    parent_pid=999,
                    parent_start_ticks="1",
                    markers=(
                        self._marker("cpu", cpu, "cpu-owner"),
                        self._marker("gpu", gpu, "gpu-owner"),
                    ),
                )
                original_cas = lifecycle._cas_marker

                def fail_second(
                    path: Path,
                    expected: str | None,
                    replacement: str | None,
                    **kwargs,
                ) -> None:
                    if path == gpu and replacement == "test-run":
                        raise lifecycle.LifecycleError("injected second-marker failure")
                    return original_cas(path, expected, replacement, **kwargs)

                with mock.patch.object(
                    lifecycle, "_cas_marker", side_effect=fail_second
                ):
                    with self.assertRaises(lifecycle.LifecycleError):
                        lifecycle.acquire_marker_transaction(state, lock)
            self.assertEqual(cpu.read_text(encoding="utf-8").strip(), "cpu-owner")
            self.assertEqual(gpu.read_text(encoding="utf-8").strip(), "gpu-owner")
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "acquisition_rolled_back")
            self.assertTrue(saved["markers"][0]["restored"])
            self.assertTrue(saved["markers"][1]["restored"])

    def test_noncooperating_foreign_claim_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            marker.write_text("known-owner\n", encoding="utf-8")
            transition_id = "foreign-race"
            expected_identity = lifecycle._owned_marker_identity(
                lifecycle._marker_observation(marker)
            )
            claim_identity = lifecycle._create_marker_claim(
                marker, "our-run", transition_id=transition_id
            )
            original_install = lifecycle._install_marker_claim

            def race(path: Path, value: str, **kwargs):
                path.write_text("foreign-owner\n", encoding="utf-8")
                return original_install(path, value, **kwargs)

            with mock.patch.object(
                lifecycle, "_install_marker_claim", side_effect=race
            ):
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle._cas_marker(
                        marker,
                        "known-owner",
                        "our-run",
                        transition_id=transition_id,
                        expected_identity=expected_identity,
                        replacement_claim_identity=claim_identity,
                    )
            self.assertEqual(
                marker.read_text(encoding="utf-8").strip(), "foreign-owner"
            )
            backups = list(root.glob(".marker.*.transition"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(
                backups[0].read_text(encoding="utf-8").strip(), "known-owner"
            )

    def test_foreign_transition_backup_race_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            marker.write_text("known-owner\n", encoding="utf-8")
            transition_id = "foreign-backup-race"
            backup = lifecycle._marker_transition_backup(marker, transition_id)
            expected_identity = lifecycle._owned_marker_identity(
                lifecycle._marker_observation(marker)
            )
            claim_identity = lifecycle._create_marker_claim(
                marker, "our-run", transition_id=transition_id
            )
            original_quarantine = lifecycle._quarantine_marker_noreplace

            def race(path: Path, destination: Path) -> None:
                destination.write_text("foreign-backup\n", encoding="utf-8")
                original_quarantine(path, destination)

            with mock.patch.object(
                lifecycle, "_quarantine_marker_noreplace", side_effect=race
            ):
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle._cas_marker(
                        marker,
                        "known-owner",
                        "our-run",
                        transition_id=transition_id,
                        expected_identity=expected_identity,
                        replacement_claim_identity=claim_identity,
                    )
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), "known-owner")
            self.assertEqual(
                backup.read_text(encoding="utf-8").strip(), "foreign-backup"
            )

    def test_transition_claim_is_fully_written_before_public_install(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "marker"
            transition_id = "fully-written-claim"
            claim = lifecycle._marker_transition_claim(marker, transition_id)
            original_link = lifecycle.os.link
            observed_destinations: list[Path] = []

            def inspect_then_link(source, destination, **kwargs):
                destination = Path(destination)
                observed_destinations.append(destination)
                self.assertEqual(Path(source).read_text(encoding="utf-8"), "our-run\n")
                if destination == claim:
                    self.assertFalse(marker.exists())
                return original_link(source, destination, **kwargs)

            with mock.patch.object(lifecycle.os, "link", side_effect=inspect_then_link):
                claim_identity = lifecycle._create_marker_claim(
                    marker, "our-run", transition_id=transition_id
                )
                lifecycle._install_marker_claim(
                    marker,
                    "our-run",
                    transition_id=transition_id,
                    claim_identity=claim_identity,
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "our-run\n")
            self.assertIn(claim, observed_destinations)
            self.assertIn(marker, observed_destinations)
            self.assertTrue(claim.exists())
            self.assertTrue(
                list(marker.parent.glob(".marker.*.claim-stage.*")),
                "retained claim stage must pin the expected inode",
            )

    def test_restore_retry_recovers_crash_after_marker_commit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cpu = root / "cpu"
            gpu = root / "gpu"
            state = root / "state.json"
            lock = root / "lock"
            with mock.patch.object(
                lifecycle, "process_identity_alive", return_value=True
            ):
                lifecycle.prepare_marker_transaction(
                    state_path=state,
                    lock_path=lock,
                    run_id="crash-window",
                    parent_pid=999,
                    parent_start_ticks="1",
                    markers=(
                        self._marker("cpu", cpu, None),
                        self._marker("gpu", gpu, None),
                    ),
                )
                lifecycle.acquire_marker_transaction(state, lock)
            original_cas = lifecycle._cas_marker

            def crash_after_commit(
                path: Path,
                expected: str | None,
                replacement: str | None,
                **kwargs,
            ) -> None:
                original_cas(path, expected, replacement, **kwargs)
                if path == cpu and replacement is None:
                    raise SystemExit(137)

            with mock.patch.object(
                lifecycle, "_cas_marker", side_effect=crash_after_commit
            ):
                with self.assertRaises(SystemExit):
                    lifecycle.restore_marker_transaction(state, lock)
            restored = lifecycle.restore_marker_transaction(state, lock)
            self.assertEqual(restored["status"], "restored")
            self.assertFalse(cpu.exists())
            self.assertFalse(gpu.exists())

    def test_restore_recovers_crash_after_acquire_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "cpu"
            marker.write_text("cpu-owner\n", encoding="utf-8")
            state_path = root / "state.json"
            lock = root / "lock"
            run_id = "acquire-quarantine-crash"
            transition_id = f"{run_id}:cpu:acquire"
            backup = lifecycle._marker_transition_backup(marker, transition_id)
            with mock.patch.object(
                lifecycle, "process_identity_alive", return_value=True
            ):
                lifecycle.prepare_marker_transaction(
                    state_path=state_path,
                    lock_path=lock,
                    run_id=run_id,
                    parent_pid=999,
                    parent_start_ticks="1",
                    markers=(self._marker("cpu", marker, "cpu-owner"),),
                )
                state = json.loads(state_path.read_text(encoding="utf-8"))
                claim_identity = lifecycle._create_marker_claim(
                    marker, run_id, transition_id=transition_id
                )
                state["status"] = "acquiring"
                state["markers"][0]["acquire_claim_identity"] = claim_identity
                state["markers"][0]["acquire_claim_path"] = str(
                    lifecycle._marker_transition_claim(marker, transition_id)
                )
                state["markers"][0]["acquire_started"] = True
                lifecycle._atomic_write_json(state_path, state)
                lifecycle._rename_noreplace(marker, backup)

                restored = lifecycle.restore_marker_transaction(state_path, lock)

            self.assertEqual(restored["status"], "restored")
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), "cpu-owner")
            self.assertTrue(backup.exists())
            self.assertEqual(backup.read_text(encoding="utf-8").strip(), "cpu-owner")

    def test_restore_preserves_foreign_acquire_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "cpu"
            marker.write_text("cpu-owner\n", encoding="utf-8")
            state_path = root / "state.json"
            lock = root / "lock"
            run_id = "foreign-acquire-quarantine"
            transition_id = f"{run_id}:cpu:acquire"
            backup = lifecycle._marker_transition_backup(marker, transition_id)
            with mock.patch.object(
                lifecycle, "process_identity_alive", return_value=True
            ):
                lifecycle.prepare_marker_transaction(
                    state_path=state_path,
                    lock_path=lock,
                    run_id=run_id,
                    parent_pid=999,
                    parent_start_ticks="1",
                    markers=(self._marker("cpu", marker, "cpu-owner"),),
                )
                state = json.loads(state_path.read_text(encoding="utf-8"))
                claim_identity = lifecycle._create_marker_claim(
                    marker, run_id, transition_id=transition_id
                )
                state["status"] = "acquiring"
                state["markers"][0]["acquire_claim_identity"] = claim_identity
                state["markers"][0]["acquire_claim_path"] = str(
                    lifecycle._marker_transition_claim(marker, transition_id)
                )
                state["markers"][0]["acquire_started"] = True
                lifecycle._atomic_write_json(state_path, state)
                marker.unlink()
                backup.write_text("foreign-owner\n", encoding="utf-8")
                foreign_inode = backup.stat().st_ino

                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.restore_marker_transaction(state_path, lock)

            self.assertFalse(marker.exists())
            self.assertTrue(backup.exists())
            self.assertEqual(backup.stat().st_ino, foreign_inode)
            self.assertEqual(
                backup.read_text(encoding="utf-8").strip(), "foreign-owner"
            )

    def test_restore_rejects_missing_owned_marker(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cpu = root / "cpu"
            gpu = root / "gpu"
            state = root / "state.json"
            lock = root / "lock"
            with mock.patch.object(
                lifecycle, "process_identity_alive", return_value=True
            ):
                lifecycle.prepare_marker_transaction(
                    state_path=state,
                    lock_path=lock,
                    run_id="test-run",
                    parent_pid=999,
                    parent_start_ticks="1",
                    markers=(
                        self._marker("cpu", cpu, None),
                        self._marker("gpu", gpu, None),
                    ),
                )
                lifecycle.acquire_marker_transaction(state, lock)
                cpu.unlink()
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.restore_marker_transaction(state, lock)
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "restore_failed")
            self.assertFalse(saved["markers"][0]["restored"])
            self.assertTrue(saved["markers"][1]["restored"])
            self.assertFalse(gpu.exists())

    def test_same_value_inode_replacement_is_detected_as_ownership_loss(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            state_path = root / "state.json"
            lock = root / "lock"
            with mock.patch.object(
                lifecycle, "process_identity_alive", return_value=True
            ):
                lifecycle.prepare_marker_transaction(
                    state_path=state_path,
                    lock_path=lock,
                    run_id="test-run",
                    parent_pid=999,
                    parent_start_ticks="1",
                    markers=(self._marker("cpu", marker, None),),
                )
                lifecycle.acquire_marker_transaction(state_path, lock)
            original = json.loads(state_path.read_text(encoding="utf-8"))
            replacement = root / "replacement"
            replacement.write_text("test-run\n", encoding="utf-8")
            os.replace(replacement, marker)

            drifts = lifecycle._owned_marker_drifts(original)

            self.assertEqual(len(drifts), 1)
            self.assertFalse(drifts[0]["value_mismatch"])
            self.assertTrue(drifts[0]["identity_mismatch"])
            self.assertEqual(drifts[0]["observation"]["value"], "test-run")

    def test_same_value_claim_after_prepare_is_not_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            state = root / "state.json"
            lock = root / "lock"
            with mock.patch.object(
                lifecycle, "process_identity_alive", return_value=True
            ):
                lifecycle.prepare_marker_transaction(
                    state_path=state,
                    lock_path=lock,
                    run_id="test-run",
                    parent_pid=999,
                    parent_start_ticks="1",
                    markers=(self._marker("cpu", marker, None),),
                )
                marker.write_text("test-run\n", encoding="utf-8")
                foreign_inode = marker.stat().st_ino
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.acquire_marker_transaction(state, lock)
            self.assertTrue(marker.exists())
            self.assertEqual(marker.stat().st_ino, foreign_inode)
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), "test-run")

    def test_same_value_replacement_before_restore_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            state = root / "state.json"
            lock = root / "lock"
            with mock.patch.object(
                lifecycle, "process_identity_alive", return_value=True
            ):
                lifecycle.prepare_marker_transaction(
                    state_path=state,
                    lock_path=lock,
                    run_id="test-run",
                    parent_pid=999,
                    parent_start_ticks="1",
                    markers=(self._marker("cpu", marker, None),),
                )
                acquired = lifecycle.acquire_marker_transaction(state, lock)
                owned_inode = acquired["markers"][0]["owned_identity"]["inode"]
                replacement = root / "replacement"
                replacement.write_text("test-run\n", encoding="utf-8")
                foreign_inode = replacement.stat().st_ino
                self.assertNotEqual(owned_inode, foreign_inode)
                os.replace(replacement, marker)
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.restore_marker_transaction(state, lock)
            self.assertTrue(marker.exists())
            self.assertEqual(marker.stat().st_ino, foreign_inode)
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), "test-run")

    def test_foreign_replacement_before_atomic_quarantine_is_preserved(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            marker.write_text("known-owner\n", encoding="utf-8")
            transition_id = "unlink-window"
            expected_identity = lifecycle._owned_marker_identity(
                lifecycle._marker_observation(marker)
            )
            claim_identity = lifecycle._create_marker_claim(
                marker, "our-run", transition_id=transition_id
            )
            original_rename = lifecycle._rename_noreplace
            evidence: dict[str, int] = {}

            def race(source: Path, destination: Path) -> None:
                if source == marker:
                    replacement = root / "foreign"
                    replacement.write_text("foreign-owner\n", encoding="utf-8")
                    evidence["inode"] = replacement.stat().st_ino
                    os.replace(replacement, source)
                original_rename(source, destination)

            with mock.patch.object(
                lifecycle, "_rename_noreplace", side_effect=race
            ):
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle._cas_marker(
                        marker,
                        "known-owner",
                        "our-run",
                        transition_id=transition_id,
                        expected_identity=expected_identity,
                        replacement_claim_identity=claim_identity,
                    )
            self.assertTrue(marker.exists())
            self.assertEqual(marker.stat().st_ino, evidence["inode"])
            self.assertEqual(
                marker.read_text(encoding="utf-8").strip(), "foreign-owner"
            )

    def test_foreign_backup_replacement_after_install_is_never_unlinked(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            marker.write_text("known-owner\n", encoding="utf-8")
            transition_id = "post-install-backup-race"
            backup = lifecycle._marker_transition_backup(marker, transition_id)
            expected_identity = lifecycle._owned_marker_identity(
                lifecycle._marker_observation(marker)
            )
            claim_identity = lifecycle._create_marker_claim(
                marker, "our-run", transition_id=transition_id
            )
            original_install = lifecycle._install_marker_claim
            evidence: dict[str, int] = {}

            def replace_backup_after_install(path: Path, value: str, **kwargs):
                identity = original_install(path, value, **kwargs)
                foreign = root / "foreign-backup"
                foreign.write_text("foreign-backup\n", encoding="utf-8")
                evidence["inode"] = foreign.stat().st_ino
                os.replace(foreign, backup)
                return identity

            with mock.patch.object(
                lifecycle,
                "_install_marker_claim",
                side_effect=replace_backup_after_install,
            ):
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle._cas_marker(
                        marker,
                        "known-owner",
                        "our-run",
                        transition_id=transition_id,
                        expected_identity=expected_identity,
                        replacement_claim_identity=claim_identity,
                    )

            self.assertTrue(backup.exists())
            self.assertEqual(backup.stat().st_ino, evidence["inode"])
            self.assertEqual(
                backup.read_text(encoding="utf-8").strip(), "foreign-backup"
            )

    def test_foreign_backup_replacement_during_recovery_is_never_unlinked(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            marker.write_text("known-owner\n", encoding="utf-8")
            transition_id = "recovery-backup-race"
            backup = lifecycle._marker_transition_backup(marker, transition_id)
            expected_identity = lifecycle._owned_marker_identity(
                lifecycle._marker_observation(marker)
            )
            claim_identity = lifecycle._create_marker_claim(
                marker, "our-run", transition_id=transition_id
            )
            lifecycle._rename_noreplace(marker, backup)
            original_install = lifecycle._install_marker_claim
            evidence: dict[str, int] = {}

            def replace_backup_after_install(path: Path, value: str, **kwargs):
                identity = original_install(path, value, **kwargs)
                foreign = root / "foreign-backup"
                foreign.write_text("foreign-backup\n", encoding="utf-8")
                evidence["inode"] = foreign.stat().st_ino
                os.replace(foreign, backup)
                return identity

            with mock.patch.object(
                lifecycle,
                "_install_marker_claim",
                side_effect=replace_backup_after_install,
            ):
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle._cas_marker(
                        marker,
                        "known-owner",
                        "our-run",
                        transition_id=transition_id,
                        expected_identity=expected_identity,
                        replacement_claim_identity=claim_identity,
                    )

            self.assertTrue(backup.exists())
            self.assertEqual(backup.stat().st_ino, evidence["inode"])
            self.assertEqual(
                backup.read_text(encoding="utf-8").strip(), "foreign-backup"
            )

    def test_restore_recovers_crash_after_acquire_commit_before_state_save(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            state = root / "state.json"
            lock = root / "lock"
            with mock.patch.object(
                lifecycle, "process_identity_alive", return_value=True
            ):
                lifecycle.prepare_marker_transaction(
                    state_path=state,
                    lock_path=lock,
                    run_id="acquire-postcommit-crash",
                    parent_pid=999,
                    parent_start_ticks="1",
                    markers=(self._marker("cpu", marker, None),),
                )
                original_cas = lifecycle._cas_marker

                def crash_after_acquire_commit(
                    path: Path,
                    expected: str | None,
                    replacement: str | None,
                    **kwargs,
                ):
                    result = original_cas(path, expected, replacement, **kwargs)
                    if replacement == "acquire-postcommit-crash":
                        raise SystemExit(137)
                    return result

                with mock.patch.object(
                    lifecycle, "_cas_marker", side_effect=crash_after_acquire_commit
                ):
                    with self.assertRaises(SystemExit):
                        lifecycle.acquire_marker_transaction(state, lock)

                saved_after_crash = json.loads(state.read_text(encoding="utf-8"))
                self.assertTrue(saved_after_crash["markers"][0]["acquire_started"])
                self.assertFalse(saved_after_crash["markers"][0]["acquired"])
                self.assertEqual(
                    marker.read_text(encoding="utf-8").strip(),
                    "acquire-postcommit-crash",
                )

                restored = lifecycle.restore_marker_transaction(state, lock)

            self.assertEqual(restored["status"], "restored")
            self.assertFalse(marker.exists())

    def test_acquire_postcommit_recovery_rejects_same_value_foreign_inode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            state = root / "state.json"
            lock = root / "lock"
            run_id = "acquire-postcommit-foreign"
            with mock.patch.object(
                lifecycle, "process_identity_alive", return_value=True
            ):
                lifecycle.prepare_marker_transaction(
                    state_path=state,
                    lock_path=lock,
                    run_id=run_id,
                    parent_pid=999,
                    parent_start_ticks="1",
                    markers=(self._marker("cpu", marker, None),),
                )
                original_cas = lifecycle._cas_marker

                def crash_after_acquire_commit(
                    path: Path,
                    expected: str | None,
                    replacement: str | None,
                    **kwargs,
                ):
                    result = original_cas(path, expected, replacement, **kwargs)
                    if replacement == run_id:
                        raise SystemExit(137)
                    return result

                with mock.patch.object(
                    lifecycle, "_cas_marker", side_effect=crash_after_acquire_commit
                ):
                    with self.assertRaises(SystemExit):
                        lifecycle.acquire_marker_transaction(state, lock)

                foreign = root / "foreign"
                foreign.write_text(f"{run_id}\n", encoding="utf-8")
                foreign_inode = foreign.stat().st_ino
                os.replace(foreign, marker)
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.restore_marker_transaction(state, lock)

            self.assertTrue(marker.exists())
            self.assertEqual(marker.stat().st_ino, foreign_inode)
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), run_id)

    def test_non_linux_without_pidfds_fails_closed_without_os_kill(self) -> None:
        with (
            mock.patch.object(lifecycle, "process_identity_alive", return_value=True),
            mock.patch.object(lifecycle.sys, "platform", "darwin"),
            mock.patch.object(lifecycle.os, "pidfd_open", None, create=True),
            mock.patch.object(lifecycle.signal, "pidfd_send_signal", None, create=True),
            mock.patch.object(lifecycle.os, "kill") as unsafe_kill,
        ):
            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle._signal_process_identity(999, "1", signal.SIGTERM)
        unsafe_kill.assert_not_called()

    def test_pidfd_syscall_error_fails_closed_without_os_kill(self) -> None:
        class FailingSyscall:
            restype = None

            def __call__(self, *_args):
                lifecycle.ctypes.set_errno(lifecycle.errno.ENOSYS)
                return -1

        fake_libc = SimpleNamespace(syscall=FailingSyscall())
        with (
            mock.patch.object(lifecycle.sys, "platform", "linux"),
            mock.patch.object(lifecycle.os, "pidfd_open", None, create=True),
            mock.patch.object(lifecycle.ctypes, "CDLL", return_value=fake_libc),
            mock.patch.object(lifecycle.os, "kill") as unsafe_kill,
        ):
            with self.assertRaisesRegex(lifecycle.LifecycleError, "pidfd_open"):
                lifecycle._signal_process_identity(999, "1", signal.SIGTERM)
        unsafe_kill.assert_not_called()

    def test_watcher_installs_signal_handlers_before_ready_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state.json"
            lock = root / "lock"
            ready = root / "ready.json"
            receipt = root / "receipt.json"
            lifecycle._atomic_write_json(
                state,
                {
                    "schema": "amg_marker_transaction_v1",
                    "run_id": "handler-order",
                    "status": "prepared",
                    "parent": {"pid": 999, "start_ticks": "1"},
                    "lock_path": str(lock),
                    "markers": [self._marker("cpu", root / "cpu", None)],
                },
            )
            order: list[str] = []
            original_write = lifecycle._atomic_write_json

            def record_write(path: Path, value, mode: int = 0o600):
                if path == ready:
                    order.append("ready")
                    self.assertTrue(value["signal_handlers_installed"])
                    raise RuntimeError("stop after ready")
                return original_write(path, value, mode=mode)

            def record_handler(_signal, _handler):
                order.append("handler")

            with (
                mock.patch.object(lifecycle, "process_start_ticks", return_value="2"),
                mock.patch.object(lifecycle.signal, "signal", side_effect=record_handler),
                mock.patch.object(
                    lifecycle, "_atomic_write_json", side_effect=record_write
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after ready"):
                    lifecycle.watch_marker_transaction(
                        state_path=state,
                        lock_path=lock,
                        parent_pid=999,
                        parent_start_ticks="1",
                        ready_path=ready,
                        receipt_path=receipt,
                        poll_seconds=0.01,
                        restore_timeout_seconds=1,
                    )
            self.assertEqual(order, ["handler", "handler", "ready"])

    def _run_marker_cas(
        self, marker: Path, transition_id: str
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(MODULE.parent.parent)
        program = (
            "from pathlib import Path\n"
            "import sys\n"
            "from agentmemorygym_verl import orchestrator_lifecycle as lifecycle\n"
            "path = Path(sys.argv[1])\n"
            "transition_id = sys.argv[2]\n"
            "observation = lifecycle._marker_observation(path)\n"
            "expected_identity = (\n"
            "    lifecycle._owned_marker_identity(observation)\n"
            "    if observation.get('value') == 'known-owner' and not observation.get('error')\n"
            "    else {'device': -1, 'inode': -1, 'ctime_ns': -1}\n"
            ")\n"
            "claim_identity = lifecycle._create_marker_claim(\n"
            "    path, 'our-run', transition_id=transition_id\n"
            ")\n"
            "lifecycle._cas_marker(\n"
            "    path, 'known-owner', 'our-run',\n"
            "    transition_id=transition_id,\n"
            "    expected_identity=expected_identity,\n"
            "    replacement_claim_identity=claim_identity,\n"
            ")\n"
        )
        return subprocess.run(
            [sys.executable, "-c", program, str(marker), transition_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
            env=environment,
        )

    def test_fifo_transition_backup_is_rejected_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            marker.write_text("known-owner\n", encoding="utf-8")
            transition_id = "fifo-backup"
            backup = lifecycle._marker_transition_backup(marker, transition_id)
            os.mkfifo(backup, mode=0o600)

            result = self._run_marker_cas(marker, transition_id)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("marker must be a regular file", result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), "known-owner")
            self.assertTrue(backup.is_fifo())

    def test_fifo_marker_is_rejected_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            os.mkfifo(marker, mode=0o600)
            transition_id = "fifo-marker"

            result = self._run_marker_cas(marker, transition_id)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("marker must be a regular file", result.stderr)
            self.assertTrue(marker.is_fifo())
            self.assertFalse(
                lifecycle._marker_transition_backup(marker, transition_id).exists()
            )

    @unittest.skipUnless(Path("/proc/self/stat").is_file(), "requires Linux /proc")
    def test_live_marker_takeover_stops_parent_and_preserves_foreign_owner(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cpu = root / "cpu"
            gpu = root / "gpu"
            state = root / "state.json"
            lock = root / "lock"
            ready = root / "ready.json"
            receipt = root / "receipt.json"
            parent = subprocess.Popen(["sleep", "60"])
            watcher: subprocess.Popen[str] | None = None
            try:
                ticks = lifecycle.process_start_ticks(parent.pid)
                self.assertIsNotNone(ticks)
                markers = (
                    lifecycle._marker_record("cpu", cpu, None, 0, ""),
                    lifecycle._marker_record("gpu", gpu, None, 0, ""),
                )
                lifecycle.prepare_marker_transaction(
                    state_path=state,
                    lock_path=lock,
                    run_id="watcher-run",
                    parent_pid=parent.pid,
                    parent_start_ticks=str(ticks),
                    markers=markers,
                )
                watcher = subprocess.Popen(
                    [
                        sys.executable,
                        str(MODULE),
                        "marker-watch",
                        "--state",
                        str(state),
                        "--lock",
                        str(lock),
                        "--parent-pid",
                        str(parent.pid),
                        "--parent-start-ticks",
                        str(ticks),
                        "--ready",
                        str(ready),
                        "--receipt",
                        str(receipt),
                        "--poll-seconds",
                        "0.05",
                        "--restore-timeout-seconds",
                        "1",
                    ],
                    text=True,
                )
                deadline = time.monotonic() + 5
                while not ready.is_file() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready.is_file())
                lifecycle.acquire_marker_transaction(state, lock)
                cpu.write_text("foreign-run\n", encoding="utf-8")

                self.assertEqual(parent.wait(timeout=5), -signal.SIGTERM)
                self.assertNotEqual(watcher.wait(timeout=5), 0)
                report = json.loads(receipt.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "fail")
                self.assertEqual(report["mode"], "marker_ownership_lost")
                loss = report["ownership_loss"]
                self.assertEqual(loss["run_id"], "watcher-run")
                self.assertEqual(loss["markers"][0]["name"], "cpu")
                self.assertEqual(loss["markers"][0]["expected_value"], "watcher-run")
                self.assertEqual(
                    loss["markers"][0]["observation"]["value"], "foreign-run"
                )
                self.assertEqual(cpu.read_text(encoding="utf-8").strip(), "foreign-run")
                self.assertFalse(gpu.exists())
                saved = json.loads(state.read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "restore_failed")
                self.assertFalse(saved["markers"][0]["restored"])
                self.assertTrue(saved["markers"][1]["restored"])
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait()
                if watcher is not None and watcher.poll() is None:
                    watcher.kill()
                    watcher.wait()

    @unittest.skipUnless(Path("/proc/self/stat").is_file(), "requires Linux /proc")
    def test_normal_restore_and_watcher_exit_do_not_signal_live_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cpu = root / "cpu"
            gpu = root / "gpu"
            state = root / "state.json"
            lock = root / "lock"
            ready = root / "ready.json"
            receipt = root / "receipt.json"
            parent = subprocess.Popen(["sleep", "60"])
            watcher: subprocess.Popen[str] | None = None
            try:
                ticks = lifecycle.process_start_ticks(parent.pid)
                self.assertIsNotNone(ticks)
                lifecycle.prepare_marker_transaction(
                    state_path=state,
                    lock_path=lock,
                    run_id="normal-restore",
                    parent_pid=parent.pid,
                    parent_start_ticks=str(ticks),
                    markers=(
                        lifecycle._marker_record("cpu", cpu, None, 0, ""),
                        lifecycle._marker_record("gpu", gpu, None, 0, ""),
                    ),
                )
                watcher = subprocess.Popen(
                    [
                        sys.executable,
                        str(MODULE),
                        "marker-watch",
                        "--state",
                        str(state),
                        "--lock",
                        str(lock),
                        "--parent-pid",
                        str(parent.pid),
                        "--parent-start-ticks",
                        str(ticks),
                        "--ready",
                        str(ready),
                        "--receipt",
                        str(receipt),
                        "--poll-seconds",
                        "0.01",
                        "--restore-timeout-seconds",
                        "1",
                    ],
                    text=True,
                )
                deadline = time.monotonic() + 5
                while not ready.is_file() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.is_file())
                lifecycle.acquire_marker_transaction(state, lock)
                restored = lifecycle.restore_marker_transaction(state, lock)
                self.assertEqual(restored["status"], "restored")
                self.assertEqual(watcher.wait(timeout=5), 0)
                self.assertIsNone(parent.poll())
                report = json.loads(receipt.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "pass")
                self.assertEqual(report["mode"], "explicit_restore")
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait()
                if watcher is not None and watcher.poll() is None:
                    watcher.kill()
                    watcher.wait()

    @unittest.skipUnless(Path("/proc/self/stat").is_file(), "requires Linux /proc")
    def test_parent_death_watcher_restores_both_markers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cpu = root / "cpu"
            gpu = root / "gpu"
            state = root / "state.json"
            lock = root / "lock"
            ready = root / "ready.json"
            receipt = root / "receipt.json"
            parent = subprocess.Popen(["sleep", "60"])
            watcher: subprocess.Popen[str] | None = None
            try:
                ticks = lifecycle.process_start_ticks(parent.pid)
                self.assertIsNotNone(ticks)
                markers = (
                    lifecycle._marker_record("cpu", cpu, None, 0, ""),
                    lifecycle._marker_record("gpu", gpu, None, 0, ""),
                )
                lifecycle.prepare_marker_transaction(
                    state_path=state,
                    lock_path=lock,
                    run_id="watcher-run",
                    parent_pid=parent.pid,
                    parent_start_ticks=str(ticks),
                    markers=markers,
                )
                watcher = subprocess.Popen(
                    [
                        sys.executable,
                        str(MODULE),
                        "marker-watch",
                        "--state",
                        str(state),
                        "--lock",
                        str(lock),
                        "--parent-pid",
                        str(parent.pid),
                        "--parent-start-ticks",
                        str(ticks),
                        "--ready",
                        str(ready),
                        "--receipt",
                        str(receipt),
                        "--poll-seconds",
                        "0.05",
                        "--restore-timeout-seconds",
                        "3",
                    ],
                    text=True,
                )
                deadline = time.monotonic() + 5
                while not ready.is_file() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready.is_file())
                lifecycle.acquire_marker_transaction(state, lock)
                self.assertEqual(cpu.read_text(encoding="utf-8").strip(), "watcher-run")
                self.assertEqual(gpu.read_text(encoding="utf-8").strip(), "watcher-run")
                parent.terminate()
                parent.wait(timeout=5)
                self.assertEqual(watcher.wait(timeout=8), 0)
                self.assertFalse(cpu.exists())
                self.assertFalse(gpu.exists())
                saved = json.loads(state.read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "restored")
                watched = json.loads(receipt.read_text(encoding="utf-8"))
                self.assertEqual(watched["status"], "pass")
                self.assertEqual(watched["mode"], "parent_death_restore")
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait()
                if watcher is not None and watcher.poll() is None:
                    watcher.kill()
                    watcher.wait()


@unittest.skipUnless(Path("/proc/self/stat").is_file(), "requires Linux /proc")
class TestProcessIdentity(unittest.TestCase):
    def test_unreaped_zombie_is_not_alive(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"])
        try:
            ticks = lifecycle.process_start_ticks(child.pid)
            self.assertIsNotNone(ticks)
            deadline = time.monotonic() + 5
            state = None
            while time.monotonic() < deadline:
                try:
                    fields = (
                        Path(f"/proc/{child.pid}/stat")
                        .read_text()
                        .rsplit(")", 1)[1]
                        .split()
                    )
                except FileNotFoundError:
                    break
                state = fields[0]
                if state == "Z":
                    break
                time.sleep(0.01)
            self.assertEqual(state, "Z")
            self.assertIsNone(lifecycle.process_start_ticks(child.pid))
            self.assertFalse(lifecycle.process_identity_alive(child.pid, str(ticks)))
        finally:
            child.wait(timeout=5)


class TestCapacityAdmission(unittest.TestCase):
    def test_memory_cgroup_headroom_is_recorded_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            volatile = root / "volatile"
            persistent = root / "persistent"
            volatile.mkdir()
            persistent.mkdir()
            usage_path = root / "memory.usage_in_bytes"
            limit_path = root / "memory.limit_in_bytes"
            usage_path.write_text("800\n", encoding="utf-8")
            limit_path.write_text("1000\n", encoding="utf-8")
            output = root / "capacity.json"
            usage = shutil_usage(total=10_000, used=1_000, free=9_000)
            with mock.patch.object(lifecycle.shutil, "disk_usage", return_value=usage):
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.capacity_admission(
                        volatile_path=volatile,
                        persistent_path=persistent,
                        checkpoint_bytes=100,
                        volatile_checkpoint_copies=1,
                        persistent_checkpoint_copies=0,
                        volatile_margin_bytes=0,
                        persistent_margin_bytes=0,
                        memory_cgroup_usage_path=usage_path,
                        memory_cgroup_limit_path=limit_path,
                        memory_cgroup_checkpoint_copies=2,
                        memory_cgroup_margin_bytes=1,
                        output_path=output,
                    )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["memory_cgroup"]["usage_bytes"], 800)
            self.assertEqual(report["memory_cgroup"]["limit_bytes"], 1000)
            self.assertEqual(report["memory_cgroup"]["headroom_bytes"], 200)
            self.assertEqual(
                report["memory_cgroup"]["required_headroom_bytes"], 201
            )
            self.assertIn("memory cgroup headroom=200 required=201", report["failures"])

    def test_memory_cgroup_headroom_passes_independently_of_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            volatile = root / "volatile"
            persistent = root / "persistent"
            volatile.mkdir()
            persistent.mkdir()
            usage_path = root / "memory.usage_in_bytes"
            limit_path = root / "memory.limit_in_bytes"
            usage_path.write_text("500\n", encoding="utf-8")
            limit_path.write_text("1000\n", encoding="utf-8")
            output = root / "capacity.json"
            usage = shutil_usage(total=10_000, used=1_000, free=9_000)
            with mock.patch.object(lifecycle.shutil, "disk_usage", return_value=usage):
                report = lifecycle.capacity_admission(
                    volatile_path=volatile,
                    persistent_path=persistent,
                    checkpoint_bytes=100,
                    volatile_checkpoint_copies=1,
                    persistent_checkpoint_copies=0,
                    volatile_margin_bytes=0,
                    persistent_margin_bytes=0,
                    memory_cgroup_usage_path=usage_path,
                    memory_cgroup_limit_path=limit_path,
                    memory_cgroup_checkpoint_copies=2,
                    memory_cgroup_margin_bytes=100,
                    output_path=output,
                )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["memory_cgroup"]["headroom_bytes"], 500)
            self.assertEqual(
                report["memory_cgroup"]["required_headroom_bytes"], 300
            )

    def test_memory_cgroup_paths_must_be_supplied_together(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            usage_path = root / "memory.usage_in_bytes"
            usage_path.write_text("0\n", encoding="utf-8")
            with self.assertRaisesRegex(
                lifecycle.LifecycleError, "usage and limit paths"
            ):
                lifecycle.capacity_admission(
                    volatile_path=root,
                    persistent_path=root,
                    checkpoint_bytes=100,
                    volatile_checkpoint_copies=0,
                    persistent_checkpoint_copies=0,
                    volatile_margin_bytes=0,
                    persistent_margin_bytes=0,
                    memory_cgroup_usage_path=usage_path,
                    output_path=root / "capacity.json",
                )

    def test_enospc_is_recorded_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            volatile = root / "volatile"
            persistent = root / "persistent"
            volatile.mkdir()
            persistent.mkdir()
            output = root / "capacity.json"
            usage = shutil_usage(total=1000, used=900, free=100)
            with mock.patch.object(lifecycle.shutil, "disk_usage", return_value=usage):
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.capacity_admission(
                        volatile_path=volatile,
                        persistent_path=persistent,
                        checkpoint_bytes=100,
                        volatile_checkpoint_copies=2,
                        persistent_checkpoint_copies=1,
                        volatile_margin_bytes=1,
                        persistent_margin_bytes=1,
                        output_path=output,
                    )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "fail")
            self.assertEqual(len(report["failures"]), 1)
            self.assertTrue(report["shared_filesystem"])

    def test_shared_filesystem_combines_both_capacity_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "capacity.json"
            usage = shutil_usage(total=1000, used=850, free=150)
            with mock.patch.object(lifecycle.shutil, "disk_usage", return_value=usage):
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.capacity_admission(
                        volatile_path=root,
                        persistent_path=root,
                        checkpoint_bytes=100,
                        volatile_checkpoint_copies=1,
                        persistent_checkpoint_copies=1,
                        volatile_margin_bytes=0,
                        persistent_margin_bytes=0,
                        output_path=output,
                    )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["shared_filesystem"])
            self.assertEqual(report["shared_filesystem_required_bytes"], 200)
            self.assertEqual(report["status"], "fail")

    def test_distinct_mounted_nfs_persistent_path_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            volatile = root / "volatile"
            persistent = root / "persistent"
            volatile.mkdir()
            persistent.mkdir()
            output = root / "capacity.json"
            usage = shutil_usage(total=10_000, used=1_000, free=9_000)
            devices = {volatile: 11, persistent: 22}
            with (
                mock.patch.object(lifecycle.shutil, "disk_usage", return_value=usage),
                mock.patch.object(
                    lifecycle,
                    "_filesystem_device",
                    side_effect=lambda path: devices[path],
                ),
                mock.patch.object(
                    lifecycle,
                    "_filesystem_identity",
                    return_value={
                        "mountpoint": str(persistent),
                        "filesystem_type": "nfs",
                        "source": "server:/durable",
                    },
                ),
            ):
                report = lifecycle.capacity_admission(
                    volatile_path=volatile,
                    persistent_path=persistent,
                    checkpoint_bytes=100,
                    volatile_checkpoint_copies=1,
                    persistent_checkpoint_copies=1,
                    volatile_margin_bytes=10,
                    persistent_margin_bytes=10,
                    output_path=output,
                    require_distinct_filesystems=True,
                    expected_persistent_filesystem_types=("nfs",),
                )
            self.assertEqual(report["status"], "pass")
            self.assertFalse(report["shared_filesystem"])
            self.assertEqual(report["persistent_filesystem"]["filesystem_type"], "nfs")

    def test_unmounted_rootfs_persistent_path_fails_nfs_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            volatile = root / "volatile"
            persistent = root / "persistent"
            volatile.mkdir()
            persistent.mkdir()
            output = root / "capacity.json"
            usage = shutil_usage(total=10_000, used=1_000, free=9_000)
            devices = {volatile: 11, persistent: 22}
            with (
                mock.patch.object(lifecycle.shutil, "disk_usage", return_value=usage),
                mock.patch.object(
                    lifecycle,
                    "_filesystem_device",
                    side_effect=lambda path: devices[path],
                ),
                mock.patch.object(
                    lifecycle,
                    "_filesystem_identity",
                    return_value={
                        "mountpoint": "/",
                        "filesystem_type": "overlay",
                        "source": "overlay",
                    },
                ),
            ):
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.capacity_admission(
                        volatile_path=volatile,
                        persistent_path=persistent,
                        checkpoint_bytes=100,
                        volatile_checkpoint_copies=1,
                        persistent_checkpoint_copies=1,
                        volatile_margin_bytes=10,
                        persistent_margin_bytes=10,
                        output_path=output,
                        require_distinct_filesystems=True,
                        expected_persistent_filesystem_types=("nfs",),
                    )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "fail")
            self.assertIn("filesystem type='overlay'", report["failures"][0])

    def test_same_filesystem_fails_distinct_attestation_even_when_nfs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            volatile = root / "volatile"
            persistent = root / "persistent"
            volatile.mkdir()
            persistent.mkdir()
            output = root / "capacity.json"
            usage = shutil_usage(total=10_000, used=1_000, free=9_000)
            with (
                mock.patch.object(lifecycle.shutil, "disk_usage", return_value=usage),
                mock.patch.object(lifecycle, "_filesystem_device", return_value=11),
                mock.patch.object(
                    lifecycle,
                    "_filesystem_identity",
                    return_value={
                        "mountpoint": str(root),
                        "filesystem_type": "nfs",
                        "source": "server:/durable",
                    },
                ),
            ):
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.capacity_admission(
                        volatile_path=volatile,
                        persistent_path=persistent,
                        checkpoint_bytes=100,
                        volatile_checkpoint_copies=1,
                        persistent_checkpoint_copies=1,
                        volatile_margin_bytes=10,
                        persistent_margin_bytes=10,
                        output_path=output,
                        require_distinct_filesystems=True,
                        expected_persistent_filesystem_types=("nfs",),
                    )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "fail")
            self.assertIn(
                "volatile and persistent paths resolve to the same filesystem",
                report["failures"],
            )


def shutil_usage(*, total: int, used: int, free: int):
    usage_type = type(lifecycle.shutil.disk_usage(Path.cwd()))
    return usage_type(total, used, free)


class TestPublicationSelection(unittest.TestCase):
    def _make_publication(
        self, root: Path, version: int, receipt: bytes, lock: bytes, cert: bytes
    ) -> Path:
        publication = root / f"openmle-fast-rich-v{version}-publication"
        artifacts = publication / "artifacts"
        artifacts.mkdir(parents=True)
        path = publication / "publication-receipt.json"
        path.write_bytes(receipt)
        (artifacts / "source-lock.json").write_bytes(lock)
        (artifacts / "formal100-schedule-certificate.json").write_bytes(cert)
        return path

    def test_new_sealed_publication_is_detected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            receipt = b'{"status":"pass"}\n'
            lock = b'{"integration":{"pod_root":"/tmp/pod"}}\n'
            cert = (
                b'{"task_count":2,"source_family_count":2,'
                b'"scheduled_episode_count":4,"optimizer_updates":1,'
                b'"manifest_sha256":"m","output_sha256":"s"}\n'
            )
            fixture_receipt = root / "fixture-receipt.json"
            fixture_lock = root / "fixture-lock.json"
            fixture_cert = root / "fixture-cert.json"
            fixture_receipt.write_bytes(receipt)
            fixture_lock.write_bytes(lock)
            fixture_cert.write_bytes(cert)
            self._make_publication(root, 8, receipt, lock, cert)
            first = root / "first.json"
            lifecycle.select_latest_publication(
                registry_root=root,
                receipt_glob=str(
                    root / "openmle-fast-rich-v*-publication/publication-receipt.json"
                ),
                fixture_receipt=fixture_receipt,
                fixture_lock=fixture_lock,
                fixture_certificate=fixture_cert,
                output_path=first,
            )
            self._make_publication(root, 9, receipt, lock, cert)
            second = root / "second.json"
            lifecycle.select_latest_publication(
                registry_root=root,
                receipt_glob=str(
                    root / "openmle-fast-rich-v*-publication/publication-receipt.json"
                ),
                fixture_receipt=fixture_receipt,
                fixture_lock=fixture_lock,
                fixture_certificate=fixture_cert,
                output_path=second,
            )
            race = root / "race.json"
            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.assert_publication_selection_unchanged(first, second, race)
            report = json.loads(race.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["first"]["version"], 8)
            self.assertEqual(report["second"]["version"], 9)

    def test_symlinked_publication_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            receipt = b'{"status":"pass"}\n'
            lock = b'{"integration":{"pod_root":"/tmp/pod"}}\n'
            cert = (
                b'{"task_count":2,"source_family_count":2,'
                b'"scheduled_episode_count":4,"optimizer_updates":1,'
                b'"manifest_sha256":"m","output_sha256":"s"}\n'
            )
            fixture_receipt = root / "fixture-receipt.json"
            fixture_lock = root / "fixture-lock.json"
            fixture_cert = root / "fixture-cert.json"
            fixture_receipt.write_bytes(receipt)
            fixture_lock.write_bytes(lock)
            fixture_cert.write_bytes(cert)
            real_receipt = self._make_publication(root, 8, receipt, lock, cert)
            os.symlink(
                real_receipt.parent,
                root / "openmle-fast-rich-v99-publication",
                target_is_directory=True,
            )

            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.select_latest_publication(
                    registry_root=root,
                    receipt_glob=str(
                        root
                        / "openmle-fast-rich-v*-publication/publication-receipt.json"
                    ),
                    fixture_receipt=fixture_receipt,
                    fixture_lock=fixture_lock,
                    fixture_certificate=fixture_cert,
                    output_path=root / "selection.json",
                )

    def test_symlink_dotdot_cannot_escape_publication_registry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            registry = root / "registry"
            escape = root / "escape"
            registry.mkdir()
            (escape / "inner").mkdir(parents=True)
            os.symlink(escape / "inner", registry / "jump", target_is_directory=True)
            receipt = b'{"status":"pass"}\n'
            lock = b'{"integration":{"pod_root":"/tmp/pod"}}\n'
            cert = (
                b'{"task_count":2,"source_family_count":2,'
                b'"scheduled_episode_count":4,"optimizer_updates":1,'
                b'"manifest_sha256":"m","output_sha256":"s"}\n'
            )
            self._make_publication(escape, 9, receipt, lock, cert)

            # These in-registry decoys satisfy the old normalized lstat walk,
            # while kernel path resolution through jump/.. reads from escape.
            decoy = registry / "openmle-fast-rich-v9-publication"
            (decoy / "artifacts").mkdir(parents=True)
            (decoy / "publication-receipt.json").write_bytes(b"not-json\n")
            (decoy / "artifacts/source-lock.json").write_bytes(b"decoy\n")
            (decoy / "artifacts/formal100-schedule-certificate.json").write_bytes(
                b"decoy\n"
            )
            fixture_receipt = root / "fixture-receipt.json"
            fixture_lock = root / "fixture-lock.json"
            fixture_cert = root / "fixture-cert.json"
            fixture_receipt.write_bytes(receipt)
            fixture_lock.write_bytes(lock)
            fixture_cert.write_bytes(cert)

            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.select_latest_publication(
                    registry_root=registry,
                    receipt_glob=str(
                        registry
                        / "jump/../openmle-fast-rich-v*-publication"
                        / "publication-receipt.json"
                    ),
                    fixture_receipt=fixture_receipt,
                    fixture_lock=fixture_lock,
                    fixture_certificate=fixture_cert,
                    output_path=root / "selection.json",
                )

    def test_symlinked_registry_ancestor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            registry = root / "registry"
            registry.mkdir()
            real = registry / "real"
            real.mkdir()
            alias = registry / "alias"
            os.symlink(real, alias, target_is_directory=True)
            receipt = b'{"status":"pass"}\n'
            lock = b'{"integration":{"pod_root":"/tmp/pod"}}\n'
            cert = (
                b'{"task_count":2,"source_family_count":2,'
                b'"scheduled_episode_count":4,"optimizer_updates":1,'
                b'"manifest_sha256":"m","output_sha256":"s"}\n'
            )
            publication = real / "openmle-fast-rich-v8-publication"
            (publication / "artifacts").mkdir(parents=True)
            (publication / "publication-receipt.json").write_bytes(receipt)
            (publication / "artifacts/source-lock.json").write_bytes(lock)
            (publication / "artifacts/formal100-schedule-certificate.json").write_bytes(
                cert
            )
            fixture_receipt = root / "fixture-receipt.json"
            fixture_lock = root / "fixture-lock.json"
            fixture_cert = root / "fixture-cert.json"
            fixture_receipt.write_bytes(receipt)
            fixture_lock.write_bytes(lock)
            fixture_cert.write_bytes(cert)

            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.select_latest_publication(
                    registry_root=registry,
                    receipt_glob=str(
                        alias
                        / "openmle-fast-rich-v*-publication/publication-receipt.json"
                    ),
                    fixture_receipt=fixture_receipt,
                    fixture_lock=fixture_lock,
                    fixture_certificate=fixture_cert,
                    output_path=root / "selection.json",
                )

    def test_symlinked_publication_artifacts_ancestor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            receipt = b'{"status":"pass"}\n'
            lock = b'{"integration":{"pod_root":"/tmp/pod"}}\n'
            cert = (
                b'{"task_count":2,"source_family_count":2,'
                b'"scheduled_episode_count":4,"optimizer_updates":1,'
                b'"manifest_sha256":"m","output_sha256":"s"}\n'
            )
            fixture_receipt = root / "fixture-receipt.json"
            fixture_lock = root / "fixture-lock.json"
            fixture_cert = root / "fixture-cert.json"
            fixture_receipt.write_bytes(receipt)
            fixture_lock.write_bytes(lock)
            fixture_cert.write_bytes(cert)
            publication = root / "openmle-fast-rich-v8-publication"
            publication.mkdir()
            (publication / "publication-receipt.json").write_bytes(receipt)
            external_artifacts = root / "external-artifacts"
            external_artifacts.mkdir()
            (external_artifacts / "source-lock.json").write_bytes(lock)
            (external_artifacts / "formal100-schedule-certificate.json").write_bytes(
                cert
            )
            os.symlink(
                external_artifacts,
                publication / "artifacts",
                target_is_directory=True,
            )

            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.select_latest_publication(
                    registry_root=root,
                    receipt_glob=str(
                        root
                        / "openmle-fast-rich-v*-publication/publication-receipt.json"
                    ),
                    fixture_receipt=fixture_receipt,
                    fixture_lock=fixture_lock,
                    fixture_certificate=fixture_cert,
                    output_path=root / "selection.json",
                )

    def test_symlinked_fixture_ancestor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            receipt = b'{"status":"pass"}\n'
            lock = b'{"integration":{"pod_root":"/tmp/pod"}}\n'
            cert = (
                b'{"task_count":2,"source_family_count":2,'
                b'"scheduled_episode_count":4,"optimizer_updates":1,'
                b'"manifest_sha256":"m","output_sha256":"s"}\n'
            )
            self._make_publication(root, 8, receipt, lock, cert)
            real_fixtures = root / "real-fixtures"
            real_fixtures.mkdir()
            (real_fixtures / "publication-receipt.json").write_bytes(receipt)
            (real_fixtures / "source-lock.json").write_bytes(lock)
            (real_fixtures / "schedule.json").write_bytes(cert)
            fixture_alias = root / "fixture-alias"
            os.symlink(real_fixtures, fixture_alias, target_is_directory=True)

            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.select_latest_publication(
                    registry_root=root,
                    receipt_glob=str(
                        root
                        / "openmle-fast-rich-v*-publication/publication-receipt.json"
                    ),
                    fixture_receipt=fixture_alias / "publication-receipt.json",
                    fixture_lock=fixture_alias / "source-lock.json",
                    fixture_certificate=fixture_alias / "schedule.json",
                    output_path=root / "selection.json",
                )

    def test_checked_exec_reselects_live_publication_before_exec(self) -> None:
        class ExecCalled(BaseException):
            pass

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            receipt = b'{"status":"pass"}\n'
            lock = b'{"integration":{"pod_root":"/tmp/pod"}}\n'
            cert = (
                b'{"task_count":2,"source_family_count":2,'
                b'"scheduled_episode_count":4,"optimizer_updates":1,'
                b'"manifest_sha256":"m","output_sha256":"s"}\n'
            )
            fixture_receipt = root / "fixture-receipt.json"
            fixture_lock = root / "fixture-lock.json"
            fixture_cert = root / "fixture-cert.json"
            fixture_receipt.write_bytes(receipt)
            fixture_lock.write_bytes(lock)
            fixture_cert.write_bytes(cert)
            self._make_publication(root, 8, receipt, lock, cert)
            first = root / "first.json"
            publication_glob = str(
                root / "openmle-fast-rich-v*-publication/publication-receipt.json"
            )
            lifecycle.select_latest_publication(
                registry_root=root,
                receipt_glob=publication_glob,
                fixture_receipt=fixture_receipt,
                fixture_lock=fixture_lock,
                fixture_certificate=fixture_cert,
                output_path=first,
            )
            registry_lock = root / "publication-registry.lock"
            args = SimpleNamespace(
                exec_command=["--", "echo", "ok"],
                registry_root=str(root),
                receipt_glob=publication_glob,
                fixture_receipt=str(fixture_receipt),
                fixture_lock=str(fixture_lock),
                fixture_certificate=str(fixture_cert),
                selection_output=str(root / "preexec.json"),
                check_output=str(root / "check.json"),
                registry_lock=str(registry_lock),
                first=str(first),
                unset_env=["PYTHONPATH"],
            )

            def assert_lock_then_exec(*_args, **_kwargs):
                descriptor = os.open(registry_lock, os.O_RDWR)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(descriptor)
                raise ExecCalled

            with mock.patch.object(
                lifecycle.os, "execvpe", side_effect=assert_lock_then_exec
            ) as execute:
                with self.assertRaises(ExecCalled):
                    lifecycle._cmd_exec_after_publication_check(args)
            execute.assert_called_once()
            descriptor = os.open(registry_lock, os.O_RDWR)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
            check = json.loads((root / "check.json").read_text(encoding="utf-8"))
            self.assertEqual(
                check["linearization_point"],
                "final_selection_under_shared_registry_lock",
            )
            self._make_publication(root, 9, receipt, lock, cert)
            with mock.patch.object(lifecycle.os, "execvpe") as execute:
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle._cmd_exec_after_publication_check(args)
            execute.assert_not_called()


class TestAtomicPublication(unittest.TestCase):
    def _make_run(self, root: Path, run_id: str) -> Path:
        run = root / run_id
        checkpoint = run / "checkpoints/global_step_1"
        checkpoint.mkdir(parents=True)
        (run / "launcher-exit.env").write_text(
            "trainer_exit_code=0\n"
            "cleanup_status=pass\n"
            "publication_status=ready_for_atomic_publication\n"
            f"run_id={run_id}\n"
            "utc=2026-08-19T00:00:00Z\n",
            encoding="utf-8",
        )
        (run / "evidence.json").write_text('{"status":"pass"}\n', encoding="utf-8")
        (checkpoint / "actor.bin").write_bytes(b"actor")
        (checkpoint / "critic.bin").write_bytes(b"critic")
        (run / "checkpoints/latest_checkpointed_iteration.txt").write_text(
            "1\n", encoding="utf-8"
        )
        return run

    def _make_recovery_ready(self, run: Path, run_id: str) -> str:
        recovery = run / "recovery"
        recovery.mkdir()
        receipt = recovery / "RECOVERY-RECEIPT.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema": "amg_test_recovery_receipt_v1",
                    "status": "pass",
                    "run_id": run_id,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        receipt_digest = lifecycle._sha256(receipt)
        post = recovery / "POST-RECOVERY-STATE.json"
        post.write_text(
            json.dumps(
                {
                    "schema": "amg_test_recovery_post_state_v1",
                    "status": "ready_for_atomic_publication",
                    "run_id": run_id,
                    "recovery_receipt_sha256": receipt_digest,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (recovery / "sealed-original.txt").write_text(
            "original\n", encoding="utf-8"
        )
        (run / "finalization.json").write_text(
            '{"errors": [], "status": "pass"}\n', encoding="utf-8"
        )
        (run / "trainer-exit-code").write_text("0\n", encoding="utf-8")
        (run / "persistent-evidence-path").write_text(
            f"/persist/{run_id}\n", encoding="utf-8"
        )
        manifest = recovery / "RECOVERY-SHA256SUMS"
        manifest_rows = []
        for path in sorted(
            item
            for item in recovery.rglob("*")
            if item.is_file()
            and item.name not in {"RECOVERY-SHA256SUMS", "RECOVERY-COMMIT.json"}
        ):
            manifest_rows.append(
                f"{lifecycle._sha256(path)}  {path.relative_to(recovery)}\n"
            )
        manifest.write_text("".join(manifest_rows), encoding="utf-8")
        launcher_contract = {
            "trainer_exit_code": "0",
            "cleanup_status": "pass",
            "publication_status": "ready_for_atomic_publication",
            "run_id": run_id,
            "recovery_mode": "post_run_evidence_preserving",
            "recovery_fix_commit": "a" * 40,
            "recovery_receipt_sha256": receipt_digest,
        }
        artifact_paths = {
            "recovery_receipt": "recovery/RECOVERY-RECEIPT.json",
            "post_recovery_state": "recovery/POST-RECOVERY-STATE.json",
            "recovery_manifest": "recovery/RECOVERY-SHA256SUMS",
            "finalization": "finalization.json",
            "trainer_exit_code": "trainer-exit-code",
            "persistent_evidence_path": "persistent-evidence-path",
        }
        commit = {
            "schema": "amg_recovery_publication_commit_v1",
            "status": "ready_for_atomic_publication",
            "run_id": run_id,
            "launcher_contract": launcher_contract,
            "artifacts": {
                name: {
                    "path": relative,
                    "sha256": lifecycle._sha256(run / relative),
                }
                for name, relative in artifact_paths.items()
            },
        }
        commit_path = recovery / "RECOVERY-COMMIT.json"
        commit_path.write_text(
            json.dumps(commit, sort_keys=True) + "\n", encoding="utf-8"
        )
        commit_digest = lifecycle._sha256(commit_path)
        launcher = [
            *(f"{key}={value}" for key, value in launcher_contract.items()),
            f"recovery_commit_sha256={commit_digest}",
            "utc=2026-08-20T00:00:00Z",
        ]
        (run / "launcher-exit.env").write_text(
            "\n".join(launcher) + "\n", encoding="utf-8"
        )
        return commit_digest

    def test_formal_tree_appears_complete_with_internal_hash_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = self._make_run(root, "formal-run")
            persist = root / "persist"
            persist.mkdir()
            report = lifecycle.atomic_publish_run(
                run_dir=run,
                persist_root=persist,
                run_id="formal-run",
                mode="formal",
                checkpoint_step=1,
                discard_gate_checkpoints=False,
            )
            final = persist / "formal-run"
            self.assertEqual(report["status"], "complete")
            self.assertTrue((final / "PUBLICATION-COMPLETE.json").is_file())
            self.assertTrue((final / "TREE-SHA256SUMS").is_file())
            self.assertTrue((final / "launcher-exit.env").is_file())
            self.assertTrue((final / "checkpoints/global_step_1/actor.bin").is_file())
            self.assertFalse(
                any(p.name.startswith(".formal-run.publish") for p in persist.iterdir())
            )
            for row in (
                (final / "TREE-SHA256SUMS").read_text(encoding="utf-8").splitlines()
            ):
                digest, relative = row.split("  ", 1)
                self.assertEqual(lifecycle._sha256(final / relative), digest)

    def test_missing_launcher_exit_never_acquires_public_destination(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = self._make_run(root, "no-exit-run")
            (run / "launcher-exit.env").unlink()
            persist = root / "persist"
            persist.mkdir()
            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.atomic_publish_run(
                    run_dir=run,
                    persist_root=persist,
                    run_id="no-exit-run",
                    mode="formal",
                    checkpoint_step=1,
                    discard_gate_checkpoints=False,
                )
            self.assertFalse((persist / "no-exit-run").exists())

    def test_recovery_launcher_without_commit_never_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = self._make_run(root, "unbound-recovery-run")
            with (run / "launcher-exit.env").open("a", encoding="utf-8") as stream:
                stream.write(
                    "recovery_mode=post_run_evidence_preserving\n"
                    f"recovery_receipt_sha256={'0' * 64}\n"
                )
            persist = root / "persist"
            persist.mkdir()
            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.atomic_publish_run(
                    run_dir=run,
                    persist_root=persist,
                    run_id="unbound-recovery-run",
                    mode="formal",
                    checkpoint_step=1,
                    discard_gate_checkpoints=False,
                )
            self.assertFalse((persist / "unbound-recovery-run").exists())

    def test_recovery_artifact_drift_never_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = self._make_run(root, "drifted-recovery-run")
            self._make_recovery_ready(run, "drifted-recovery-run")
            (run / "recovery/RECOVERY-RECEIPT.json").write_text(
                '{"run_id":"drifted-recovery-run","status":"fail"}\n',
                encoding="utf-8",
            )
            persist = root / "persist"
            persist.mkdir()
            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.atomic_publish_run(
                    run_dir=run,
                    persist_root=persist,
                    run_id="drifted-recovery-run",
                    mode="formal",
                    checkpoint_step=1,
                    discard_gate_checkpoints=False,
                )
            self.assertFalse((persist / "drifted-recovery-run").exists())

    def test_recovery_manifest_must_cover_every_recovery_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = self._make_run(root, "extra-recovery-file-run")
            self._make_recovery_ready(run, "extra-recovery-file-run")
            (run / "recovery/uncommitted.txt").write_text(
                "not committed\n", encoding="utf-8"
            )
            persist = root / "persist"
            persist.mkdir()
            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.atomic_publish_run(
                    run_dir=run,
                    persist_root=persist,
                    run_id="extra-recovery-file-run",
                    mode="formal",
                    checkpoint_step=1,
                    discard_gate_checkpoints=False,
                )
            self.assertFalse((persist / "extra-recovery-file-run").exists())

    def test_recovery_artifact_drift_during_staging_never_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_id = "staging-drift-recovery-run"
            run = self._make_run(root, run_id)
            self._make_recovery_ready(run, run_id)
            persist = root / "persist"
            persist.mkdir()
            real_rsync = lifecycle._run_rsync
            mutated = False

            def mutate_before_first_copy(arguments):
                nonlocal mutated
                if not mutated:
                    mutated = True
                    (run / "recovery/RECOVERY-RECEIPT.json").write_text(
                        json.dumps({"run_id": run_id, "status": "fail"}) + "\n",
                        encoding="utf-8",
                    )
                real_rsync(arguments)

            with mock.patch.object(
                lifecycle, "_run_rsync", side_effect=mutate_before_first_copy
            ), mock.patch.object(lifecycle.os, "_exit") as exit_process:
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.atomic_publish_run(
                        run_dir=run,
                        persist_root=persist,
                        run_id=run_id,
                        mode="formal",
                        checkpoint_step=1,
                        discard_gate_checkpoints=False,
                        terminal_exit=True,
                    )
            self.assertTrue(mutated)
            exit_process.assert_not_called()
            self.assertFalse((persist / run_id).exists())

    def test_recovery_manifest_growth_during_staging_never_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_id = "staging-growth-recovery-run"
            run = self._make_run(root, run_id)
            self._make_recovery_ready(run, run_id)
            persist = root / "persist"
            persist.mkdir()
            real_rsync = lifecycle._run_rsync
            mutated = False

            def mutate_before_first_copy(arguments):
                nonlocal mutated
                if not mutated:
                    mutated = True
                    (run / "recovery/unmanifested-after-validation.txt").write_text(
                        "not committed\n", encoding="utf-8"
                    )
                real_rsync(arguments)

            with mock.patch.object(
                lifecycle, "_run_rsync", side_effect=mutate_before_first_copy
            ):
                with self.assertRaises(lifecycle.LifecycleError):
                    lifecycle.atomic_publish_run(
                        run_dir=run,
                        persist_root=persist,
                        run_id=run_id,
                        mode="formal",
                        checkpoint_step=1,
                        discard_gate_checkpoints=False,
                    )
            self.assertTrue(mutated)
            self.assertFalse((persist / run_id).exists())

    def test_recovery_publication_binds_commit_into_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = self._make_run(root, "bound-recovery-run")
            commit_digest = self._make_recovery_ready(run, "bound-recovery-run")
            persist = root / "persist"
            persist.mkdir()
            report = lifecycle.atomic_publish_run(
                run_dir=run,
                persist_root=persist,
                run_id="bound-recovery-run",
                mode="formal",
                checkpoint_step=1,
                discard_gate_checkpoints=False,
            )
            final = persist / "bound-recovery-run"
            self.assertEqual(report["recovery_commit_sha256"], commit_digest)
            publication = json.loads(
                (final / "PUBLICATION-COMPLETE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                publication["recovery_commit_sha256"], commit_digest
            )

    def test_source_symlink_never_acquires_public_destination(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = self._make_run(root, "bad-run")
            os.symlink(run / "evidence.json", run / "evidence-link")
            persist = root / "persist"
            persist.mkdir()
            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.atomic_publish_run(
                    run_dir=run,
                    persist_root=persist,
                    run_id="bad-run",
                    mode="formal",
                    checkpoint_step=1,
                    discard_gate_checkpoints=False,
                )
            self.assertFalse((persist / "bad-run").exists())

    def test_source_fifo_never_acquires_public_destination(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = self._make_run(root, "fifo-run")
            os.mkfifo(run / "orphan.fifo", mode=0o600)
            persist = root / "persist"
            persist.mkdir()
            with self.assertRaises(lifecycle.LifecycleError):
                lifecycle.atomic_publish_run(
                    run_dir=run,
                    persist_root=persist,
                    run_id="fifo-run",
                    mode="formal",
                    checkpoint_step=1,
                    discard_gate_checkpoints=False,
                )
            self.assertFalse((persist / "fifo-run").exists())

    def test_terminal_publisher_has_complete_tree_before_immediate_exit(self) -> None:
        class ExitCalled(BaseException):
            pass

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = self._make_run(root, "terminal-run")
            persist = root / "persist"
            persist.mkdir()
            with mock.patch.object(
                lifecycle.os, "_exit", side_effect=ExitCalled
            ) as exit_process:
                with self.assertRaises(ExitCalled):
                    lifecycle.terminal_atomic_publish_run(
                        run_dir=run,
                        persist_root=persist,
                        run_id="terminal-run",
                        mode="formal",
                        checkpoint_step=1,
                        discard_gate_checkpoints=False,
                    )
            exit_process.assert_called_once_with(0)
            final = persist / "terminal-run"
            self.assertTrue((final / "PUBLICATION-COMPLETE.json").is_file())
            self.assertTrue((final / "TERMINAL-PUBLISHER.json").is_file())
            self.assertTrue((final / "TREE-SHA256SUMS").is_file())
            receipt = json.loads(
                (final / "TERMINAL-PUBLISHER.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["linearization_point"], "atomic_directory_rename")
            self.assertEqual(receipt["post_rename_work"], "none")
            self.assertEqual(
                receipt["launcher_exit_sha256"],
                lifecycle._sha256(final / "launcher-exit.env"),
            )
            self.assertFalse(
                any(
                    p.name.startswith(".terminal-run.publish")
                    for p in persist.iterdir()
                )
            )

    def test_sigkill_immediately_after_rename_leaves_terminal_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = self._make_run(root, "post-rename-kill-run")
            persist = root / "persist"
            persist.mkdir()
            program = (
                "import os, signal, sys\n"
                "from pathlib import Path\n"
                "from agentmemorygym_verl import orchestrator_lifecycle as lifecycle\n"
                "def kill_after_rename(_code):\n"
                "    os.kill(os.getpid(), signal.SIGKILL)\n"
                "lifecycle.os._exit = kill_after_rename\n"
                "lifecycle.terminal_atomic_publish_run(\n"
                "    run_dir=Path(sys.argv[1]), persist_root=Path(sys.argv[2]),\n"
                "    run_id='post-rename-kill-run', mode='formal',\n"
                "    checkpoint_step=1, discard_gate_checkpoints=False)\n"
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(MODULE.parent.parent)
            result = subprocess.run(
                [sys.executable, "-c", program, str(run), str(persist)],
                check=False,
                env=environment,
            )
            self.assertEqual(result.returncode, -signal.SIGKILL)
            final = persist / "post-rename-kill-run"
            self.assertTrue((final / "PUBLICATION-COMPLETE.json").is_file())
            self.assertTrue((final / "TERMINAL-PUBLISHER.json").is_file())
            self.assertTrue((final / "TREE-SHA256SUMS").is_file())
            self.assertFalse(
                any(
                    p.name.startswith(".post-rename-kill-run.publish")
                    for p in persist.iterdir()
                )
            )

    def test_gate_checkpoint_is_hashed_then_deleted_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = self._make_run(root, "gate-run")
            persist = root / "persist"
            persist.mkdir()
            lifecycle.atomic_publish_run(
                run_dir=run,
                persist_root=persist,
                run_id="gate-run",
                mode="gate",
                checkpoint_step=None,
                discard_gate_checkpoints=True,
            )
            final = persist / "gate-run"
            self.assertFalse((run / "checkpoints").exists())
            self.assertFalse((final / "checkpoints").exists())
            self.assertTrue((final / "gate-checkpoint-before-delete.sha256").is_file())
            deletion = json.loads(
                (final / "gate-checkpoint-deletion.json").read_text(encoding="utf-8")
            )
            self.assertTrue(deletion["deleted"])


class TestGpuMonitorFailClosed(unittest.TestCase):
    def test_deadline_crossing_never_requests_a_negative_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            identities = iter((True, True, True, False))
            monotonic_values = iter((0.0, 0.0, 0.0, 0.05, 0.11, 0.12))
            sleeps: list[float] = []

            def record_sleep(seconds: float) -> None:
                self.assertGreaterEqual(seconds, 0.0)
                sleeps.append(seconds)

            sample = subprocess.CompletedProcess(
                args=["nvidia-smi"], returncode=0, stdout="ok\n", stderr=""
            )
            with (
                mock.patch.object(
                    lifecycle,
                    "process_identity_alive",
                    side_effect=lambda *_args: next(identities, False),
                ),
                mock.patch.object(
                    lifecycle,
                    "_run_gpu_sample",
                    return_value=(sample, False, []),
                ),
                mock.patch.object(
                    lifecycle.time,
                    "monotonic",
                    side_effect=lambda: next(monotonic_values),
                ),
                mock.patch.object(
                    lifecycle.time,
                    "sleep",
                    side_effect=record_sleep,
                ),
            ):
                rc = lifecycle.run_gpu_monitor(
                    parent_pid=1,
                    parent_start_ticks="1",
                    output_path=root / "gpu.csv",
                    stderr_path=root / "gpu.stderr",
                    ready_path=root / "ready.json",
                    receipt_path=root / "receipt.json",
                    nvidia_smi="nvidia-smi",
                    interval_seconds=0.1,
                    command_timeout_seconds=0.01,
                )

            report = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 0)
            self.assertEqual(report["status"], "pass")
            self.assertTrue(all(seconds >= 0 for seconds in sleeps))

    def test_unknown_sampler_exception_cannot_emit_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            identities = iter((True, True, False))

            def identity(*_args) -> bool:
                return next(identities, False)

            with (
                mock.patch.object(
                    lifecycle, "process_identity_alive", side_effect=identity
                ),
                mock.patch.object(
                    lifecycle,
                    "_run_gpu_sample",
                    side_effect=OSError("unknown sampler cleanup state"),
                ),
            ):
                rc = lifecycle.run_gpu_monitor(
                    parent_pid=1,
                    parent_start_ticks="1",
                    output_path=root / "gpu.csv",
                    stderr_path=root / "gpu.stderr",
                    ready_path=root / "ready.json",
                    receipt_path=root / "receipt.json",
                    nvidia_smi="nvidia-smi",
                    interval_seconds=0.01,
                    command_timeout_seconds=0.01,
                )
            report = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(rc, 1)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["sampler_cleanup_uncertain"], 1)


@unittest.skipUnless(Path("/proc/self/stat").is_file(), "requires Linux /proc")
class TestBoundedGpuMonitor(unittest.TestCase):
    def test_hung_sampler_has_bounded_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fake = root / "nvidia-smi"
            child_pid_path = root / "sampler-child.pid"
            fake.write_text(
                '#!/bin/sh\nsleep 30 &\necho "$!" > "$AMG_TEST_CHILD_PID_FILE"\nwait\n',
                encoding="utf-8",
            )
            fake.chmod(0o700)
            ready = root / "ready.json"
            receipt = root / "receipt.json"
            ticks = lifecycle.process_start_ticks(os.getpid())
            self.assertIsNotNone(ticks)
            environment = os.environ.copy()
            environment["AMG_TEST_CHILD_PID_FILE"] = str(child_pid_path)
            monitor = subprocess.Popen(
                [
                    sys.executable,
                    str(MODULE),
                    "gpu-monitor",
                    "--parent-pid",
                    str(os.getpid()),
                    "--parent-start-ticks",
                    str(ticks),
                    "--output",
                    str(root / "gpu.csv"),
                    "--stderr",
                    str(root / "gpu.stderr"),
                    "--ready",
                    str(ready),
                    "--receipt",
                    str(receipt),
                    "--nvidia-smi",
                    str(fake),
                    "--interval-seconds",
                    "0.1",
                    "--command-timeout-seconds",
                    "0.2",
                ],
                env=environment,
            )
            try:
                deadline = time.monotonic() + 5
                while (
                    not (ready.is_file() and child_pid_path.is_file())
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                self.assertTrue(ready.is_file())
                self.assertTrue(child_pid_path.is_file())
                child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
                child_ticks = lifecycle.process_start_ticks(child_pid)
                self.assertIsNotNone(child_ticks)
                started = time.monotonic()
                monitor.send_signal(signal.SIGTERM)
                self.assertEqual(monitor.wait(timeout=4), 0)
                self.assertLess(time.monotonic() - started, 3)
                report = json.loads(receipt.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "pass")
                self.assertEqual(report["active_sampler_process_groups"], 0)
                deadline = time.monotonic() + 2
                while (
                    lifecycle.process_identity_alive(child_pid, str(child_ticks))
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                self.assertFalse(
                    lifecycle.process_identity_alive(child_pid, str(child_ticks))
                )
            finally:
                if monitor.poll() is None:
                    monitor.kill()
                    monitor.wait()


class TestShellOrchestratorContract(unittest.TestCase):
    def test_formal_capacity_models_two_checkpoint_peak_and_memory_cgroup(self) -> None:
        script = MODULE.parent.parent / "scripts/orchestrate_openmle_fully_async.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("VOLATILE_CHECKPOINT_COPIES=2", text)
        self.assertNotIn("VOLATILE_CHECKPOINT_COPIES=3", text)
        self.assertIn("MEMORY_CGROUP_CHECKPOINT_COPIES=2", text)
        self.assertIn("MEMORY_CGROUP_RUNTIME_MARGIN_BYTES=274877906944", text)
        self.assertEqual(text.count("--memory-cgroup-usage-path"), 2)
        self.assertEqual(text.count("--memory-cgroup-limit-path"), 2)
        self.assertEqual(text.count("--memory-cgroup-checkpoint-copies"), 2)
        self.assertEqual(text.count("--memory-cgroup-margin-bytes"), 2)

    def test_resident_probe_reuses_mode_specific_runtime_preflight_manifest(
        self,
    ) -> None:
        script = MODULE.parent.parent / "scripts/orchestrate_openmle_fully_async.sh"
        text = script.read_text(encoding="utf-8")
        probe_start = text.index('eval "$("$PY" - "$LOCK" "$PREFLIGHT"')
        probe_end = text.index('if [ "$MODE" = gate ]; then', probe_start)
        probe = text[probe_start:probe_end]
        self.assertIn("'MANIFEST':p['manifest_path']", probe)
        self.assertIn("'MANIFEST_SHA':p['manifest_sha256']", probe)
        self.assertNotIn("['manifests']['train_pool']", probe)

    def test_crash_guards_and_terminal_publication_order_are_static(self) -> None:
        script = MODULE.parent.parent / "scripts/orchestrate_openmle_fully_async.sh"
        text = script.read_text(encoding="utf-8")
        prepare = text.index('"$PY" "$LIFECYCLE" marker-prepare')
        watcher = text.index('"$PY" "$LIFECYCLE" marker-watch')
        acquire = text.index('"$PY" "$LIFECYCLE" marker-acquire')
        self.assertLess(prepare, watcher)
        self.assertLess(watcher, acquire)
        cpu_yield = text.index("CPU holder did not reach state=yielded")
        gpu_yield = text.index("GPU holder did not reach mode=yield")
        endpoint_start = text.index('"$START_ENDPOINT" "$ENDPOINT_CONTRACT"')
        endpoint_probe = text.index("verify_openmle_fast_resident_endpoint.py")
        self.assertLess(acquire, cpu_yield)
        self.assertLess(cpu_yield, gpu_yield)
        self.assertLess(gpu_yield, endpoint_start)
        self.assertLess(endpoint_start, endpoint_probe)
        capacity = text.index('--output "$RUN_DIR/capacity-pretrainer.json"')
        reselect = text.index('"$PY" "$LIFECYCLE" exec-after-publication-check')
        trainer = text.index(
            '"$PLUGIN_OUTER/async_plugins/scripts/launch_amg_fully_async.sh"',
            reselect,
        )
        self.assertLess(capacity, reselect)
        self.assertLess(reselect, trainer)
        launcher_exit = text.index("write_launcher_exit 0 ready_for_atomic_publication")
        freeze = text.index("freeze_run_dir_logging", launcher_exit)
        trap_off = text.index("trap - EXIT INT TERM", freeze)
        publish = text.index('exec "$PY" "$LIFECYCLE" terminal-publish', trap_off)
        self.assertLess(launcher_exit, freeze)
        self.assertLess(freeze, trap_off)
        self.assertLess(trap_off, publish)
        self.assertNotIn('"$PY" "$LIFECYCLE" atomic-publish', text)
        self.assertNotIn("PUBLICATION_COMPLETE=1", text)
        self.assertNotIn("cpu_worker_identity", text)
        self.assertNotIn('find "$PERSIST" -type f', text)
        alive_start = text.index("process_alive_exact()")
        alive_end = text.index("\n}", alive_start)
        alive_function = text[alive_start:alive_end]
        self.assertIn('"$PY" "$LIFECYCLE" process-identity-alive', alive_function)
        self.assertEqual(text.count('--registry-root "$PUBLICATION_REGISTRY_ROOT"'), 2)

    def test_cleanup_failure_hard_stops_before_prepublication(self) -> None:
        script = MODULE.parent.parent / "scripts/orchestrate_openmle_fully_async.sh"
        text = script.read_text(encoding="utf-8")
        cleanup_call = text.index("cleanup_before_publication || exit $?")
        prepublication = text.index("PERSIST=$PERSIST_ROOT/$RUN_ID")
        self.assertLess(cleanup_call, prepublication)
        self.assertNotIn(
            '[ "$RUNTIME_CLEANED" -eq 1 ] && [ "$CLEANUP_STATUS" = pass ]',
            text,
        )
        cleanup_start = text.index("cleanup_runtime() {")
        cleanup_end = text.index(
            "\n}\n\ncleanup_before_publication()", cleanup_start
        )
        cleanup_body = text[cleanup_start:cleanup_end]
        self.assertIn('if [ "$CLEANUP_STATUS" = pass ]; then', cleanup_body)
        self.assertTrue(cleanup_body.rstrip().endswith("return 1"))

    def test_cleanup_failure_exits_125_before_publication_at_runtime(self) -> None:
        script = MODULE.parent.parent / "scripts/orchestrate_openmle_fully_async.sh"
        text = script.read_text(encoding="utf-8")
        start = text.index("cleanup_before_publication() {")
        end = text.index("\n}\n\ncleanup()", start) + 2
        function = text[start:end]
        program = f"""#!/usr/bin/env bash
set +e
{function}
cleanup_runtime() {{ return 1; }}
RUNTIME_CLEANED=0
CLEANUP_STATUS=fail
cleanup_before_publication
rc=$?
printf 'rc=%s\\n' "$rc"
if [ "$rc" -eq 0 ]; then touch "$1"; fi
exit "$rc"
"""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            probe = root / "probe.sh"
            sentinel = root / "published"
            probe.write_text(program, encoding="utf-8")
            result = subprocess.run(
                ["bash", str(probe), str(sentinel)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 125, result.stderr)
            self.assertEqual(result.stdout.strip(), "rc=125")
            self.assertFalse(sentinel.exists())

    def test_dead_marker_watcher_stops_live_trainer_at_runtime(self) -> None:
        script = MODULE.parent.parent / "scripts/orchestrate_openmle_fully_async.sh"
        text = script.read_text(encoding="utf-8")
        start = text.index("wait_trainer_with_marker_watcher() {")
        end = text.index("\n}\n\nstop_endpoint()", start) + 2
        function = text[start:end]
        program = f"""#!/usr/bin/env bash
set +e
{function}
process_alive_exact() {{ kill -0 "$1" 2>/dev/null; }}
stop_exact_child() {{
  local _name=$1 pid_var=$2 ticks_var=$3
  local pid=${{!pid_var}}
  kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  printf -v "$pid_var" ''
  printf -v "$ticks_var" ''
}}
CLEANUP_STATUS=pass
sleep 30 & TRAIN_PID=$!
ORIGINAL_TRAIN_PID=$TRAIN_PID
trap 'kill -KILL "$ORIGINAL_TRAIN_PID" 2>/dev/null || true' EXIT
TRAIN_TICKS=unused
sleep 0.05 & MARKER_WATCH_PID=$!
MARKER_WATCH_TICKS=unused
wait_trainer_with_marker_watcher
rc=$?
if kill -0 "$ORIGINAL_TRAIN_PID" 2>/dev/null; then alive=1; else alive=0; fi
printf 'rc=%s cleanup=%s trainer_alive=%s\\n' "$rc" "$CLEANUP_STATUS" "$alive"
exit "$rc"
"""
        with tempfile.TemporaryDirectory() as raw:
            probe = Path(raw) / "probe.sh"
            probe.write_text(program, encoding="utf-8")
            result = subprocess.run(
                ["bash", str(probe)],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 125, result.stderr)
            self.assertIn("cleanup=fail", result.stdout)
            self.assertIn("trainer_alive=0", result.stdout)

    def test_shell_marker_reads_use_nonblocking_lifecycle_cli(self) -> None:
        script = MODULE.parent.parent / "scripts/orchestrate_openmle_fully_async.sh"
        text = script.read_text(encoding="utf-8")
        self.assertNotIn('cat "$CPU_MARKER"', text)
        self.assertNotIn('cat "$GPU_MARKER"', text)
        self.assertEqual(
            text.count('"$PY" "$LIFECYCLE" marker-read --path'),
            2,
        )

    def test_marker_read_cli_rejects_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            marker = root / "marker"
            os.mkfifo(marker, mode=0o600)
            result = subprocess.run(
                [sys.executable, str(MODULE), "marker-read", "--path", str(marker)],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("marker must be a regular file", result.stderr)

            missing = subprocess.run(
                [
                    sys.executable,
                    str(MODULE),
                    "marker-read",
                    "--path",
                    str(root / "missing"),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
            self.assertEqual(missing.returncode, 0, missing.stderr)
            self.assertEqual(missing.stdout, "")


if __name__ == "__main__":
    unittest.main()
