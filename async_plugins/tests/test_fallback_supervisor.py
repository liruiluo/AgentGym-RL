from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from agentmemorygym_verl import fallback_supervisor as fallback


MODULE = Path(fallback.__file__).resolve()
BOOTSTRAP = MODULE.with_name("process_bootstrap.py")
WATCHDOG_WRAPPER = MODULE.parents[1] / "scripts/amg_fallback_supervisor_watchdog.sh"


def _supports_exact_crash_recovery_fixture() -> bool:
    if not Path("/proc/self/stat").is_file():
        return False
    try:
        descriptor = fallback._pidfd_open_exact(os.getpid())
    except fallback.FallbackError:
        return False
    else:
        os.close(descriptor)
    return getattr(fallback.ctypes.CDLL(None), "renameat2", None) is not None


class TestFallbackWrapperSource(unittest.TestCase):
    def test_production_default_uses_restart_stable_interpreter(self) -> None:
        source = WATCHDOG_WRAPPER.read_text(encoding="utf-8")
        self.assertIn(
            'FALLBACK_SUPERVISOR_PYTHON:-/opt/conda/envs/py312/bin/python3',
            source,
        )
        self.assertNotIn(
            'PYTHON:-/dev/shm/qwen35-runtime-verl-main-sglang-fsdp-tf553-fla052-v2',
            source,
        )


class TestPendingResumeReleaseTransaction(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pause = self.root / "pause.json"
        self.request = self.root / "release.json"
        self.token = "crash-boundary"
        marker = fallback._create_pause_marker(
            self.pause,
            {"schema": fallback._PAUSE_SCHEMA, "token": self.token},
        )
        request_binding = fallback._create_release_request(
            self.request,
            {"schema": fallback._RELEASE_REQUEST_SCHEMA, "token": self.token},
        )
        self.pending = {
            "schema": fallback._PENDING_RESUME_SCHEMA,
            "token": self.token,
            "marker_binding": fallback._marker_binding_view(marker),
            "request_binding": request_binding,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _complete_with_portable_file_ops(self) -> None:
        def release(path: Path, *, token: str, expected: dict) -> bool:
            observed = fallback._marker_observation(path)
            self.assertEqual(token, self.token)
            self.assertEqual(
                fallback._marker_binding_view(observed),
                expected,
            )
            path.unlink()
            return True

        def consume(path: Path, expected: dict, *, token: str) -> None:
            payload, observed = fallback._bound_json_file(path)
            self.assertEqual(token, self.token)
            self.assertEqual(payload["token"], self.token)
            self.assertEqual(observed, expected)
            path.unlink()

        with (
            mock.patch.object(fallback, "_release_pause_marker", side_effect=release),
            mock.patch.object(fallback, "_remove_bound_json", side_effect=consume),
        ):
            fallback._complete_pending_resume_release(
                pause_path=self.pause,
                release_request_path=self.request,
                pending_resume=self.pending,
            )

    def test_crash_after_authorization_persisted_finishes_both_artifacts(self) -> None:
        self._complete_with_portable_file_ops()
        self.assertFalse(self.pause.exists())
        self.assertFalse(self.request.exists())

    def test_crash_after_marker_removed_consumes_bound_request(self) -> None:
        self.pause.unlink()
        self._complete_with_portable_file_ops()
        self.assertFalse(self.request.exists())

    def test_crash_after_request_consumed_is_idempotent(self) -> None:
        self.pause.unlink()
        self.request.unlink()
        self._complete_with_portable_file_ops()
        fallback._assert_pending_resume_publishable(
            pause_path=self.pause,
            release_request_path=self.request,
            pending_resume=self.pending,
        )

    def test_crash_after_marker_quarantine_rename_finishes_release(self) -> None:
        quarantine = fallback._pause_marker_quarantine(self.pause, self.token)
        self.pause.rename(quarantine)
        self._complete_with_portable_file_ops()
        self.assertFalse(quarantine.exists())
        self.assertFalse(self.request.exists())

    def test_crash_after_request_quarantine_rename_finishes_consumption(self) -> None:
        self.pause.unlink()
        quarantine = fallback._release_request_quarantine(
            self.request, self.token
        )
        self.request.rename(quarantine)
        self._complete_with_portable_file_ops()
        self.assertFalse(quarantine.exists())

    def test_foreign_quarantine_cannot_be_completed(self) -> None:
        self.pause.unlink()
        quarantine = fallback._release_request_quarantine(
            self.request, self.token
        )
        self.request.rename(quarantine)
        quarantine.write_text(
            json.dumps(
                {
                    "schema": fallback._RELEASE_REQUEST_SCHEMA,
                    "token": self.token,
                    "foreign": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            fallback.FallbackError,
            "foreign/replaced release-request quarantine",
        ):
            self._complete_with_portable_file_ops()
        self.assertTrue(quarantine.is_file())

    def test_same_inode_marker_mutation_during_quarantine_fails_closed(self) -> None:
        expected = dict(self.pending["marker_binding"])
        calls = 0

        def mutate_then_rename(source: Path, destination: Path) -> None:
            nonlocal calls
            if calls == 0:
                payload = json.loads(source.read_text(encoding="utf-8"))
                payload["same_inode_mutation"] = True
                source.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self.assertEqual(source.stat().st_ino, expected["inode"])
            calls += 1
            source.rename(destination)

        with (
            mock.patch.object(
                fallback, "_rename_noreplace", side_effect=mutate_then_rename
            ),
            self.assertRaisesRegex(fallback.FallbackError, "changed during release"),
        ):
            fallback._complete_pending_resume_release(
                pause_path=self.pause,
                release_request_path=self.request,
                pending_resume=self.pending,
            )

        observed = fallback._marker_observation(self.pause)
        self.assertEqual(observed["payload"]["token"], self.token)
        self.assertNotEqual(observed["sha256"], expected["sha256"])
        self.assertFalse(
            fallback._pause_marker_quarantine(self.pause, self.token).exists()
        )
        self.assertTrue(self.request.is_file())

    def test_same_inode_request_mutation_during_quarantine_fails_closed(self) -> None:
        self.pause.unlink()
        expected = dict(self.pending["request_binding"])
        calls = 0

        def mutate_then_rename(source: Path, destination: Path) -> None:
            nonlocal calls
            if calls == 0:
                payload = json.loads(source.read_text(encoding="utf-8"))
                payload["same_inode_mutation"] = True
                source.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self.assertEqual(source.stat().st_ino, expected["inode"])
            calls += 1
            source.rename(destination)

        with (
            mock.patch.object(
                fallback, "_rename_noreplace", side_effect=mutate_then_rename
            ),
            self.assertRaisesRegex(
                fallback.FallbackError, "changed during consumption"
            ),
        ):
            fallback._complete_pending_resume_release(
                pause_path=self.pause,
                release_request_path=self.request,
                pending_resume=self.pending,
            )

        payload, observed = fallback._bound_json_file(self.request)
        self.assertEqual(payload["token"], self.token)
        self.assertNotEqual(observed["sha256"], expected["sha256"])
        self.assertFalse(
            fallback._release_request_quarantine(self.request, self.token).exists()
        )

    def test_holder_attestation_cannot_publish_holding_with_stale_request(self) -> None:
        self.pause.unlink()
        with self.assertRaisesRegex(
            fallback.FallbackError,
            "release transaction is not complete",
        ):
            fallback._assert_pending_resume_publishable(
                pause_path=self.pause,
                release_request_path=self.request,
                pending_resume=self.pending,
            )


@unittest.skipUnless(Path("/proc/self/stat").is_file(), "requires Linux /proc")
class TestFallbackSupervisor(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        # The production watchdog canonicalizes the interpreter before it is
        # pinned.  ``sys.executable`` may itself be a symlink (as it is in the
        # 9N py312 environment), while the immutable-file guard intentionally
        # rejects symlinks.
        self.python = str(Path(sys.executable).resolve(strict=True))
        self.holder = self.root / "holder.py"
        self.original = self.root / "watchdog.sh"
        self.wrapper = self.root / "platform-watchdog.sh"
        self.pid_file = self.root / "holder.pid"
        self.holder_state = self.root / "holder-state.json"
        self.pause = self.root / "pause.json"
        self.release_request = self.root / "release-request.json"
        self.supervisor_state = self.root / "supervisor-state.json"
        self.supervisor_lock = self.root / "supervisor.lock"
        self.transaction_lock = self.root / "transaction.lock"
        self.watchdog_log = self.root / "watchdog.log"
        self.supervisor_log = self.root / "supervisor.log"
        self.crash_wrapper = self.root / "crash_supervisor_at_boundary.py"
        self.auxiliary_processes: list[subprocess.Popen[bytes]] = []
        self.formal_inventory_args: list[str] | None = None
        self.holder.write_text(
            """#!/usr/bin/env python3
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--pid-file', required=True); p.add_argument('--state-file', required=True)
p.add_argument('--cpu-workers'); p.add_argument('--cpu-duty'); p.add_argument('--gpu-count')
p.add_argument('--gpu-duty'); p.add_argument('--gpu-hold-mb'); p.add_argument('--gpu-matrix-size')
p.add_argument('--gpu-burst-matmuls'); a=p.parse_args()
pid=Path(a.pid_file); state=Path(a.state_file); stopping=False
workers=[subprocess.Popen([sys.executable,'-c','import time; time.sleep(3600)']) for _ in range(8)]
workers += [subprocess.Popen([sys.executable,'-c','x=0\\nwhile True: x+=1']) for _ in range(20)]
def stop(*_):
 global stopping; stopping=True
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
pid.write_text(str(os.getpid())+'\\n')
state.write_text(json.dumps({'parent_pid':os.getpid(),'mode':'hold','gpu_workers':{str(i):workers[i].pid for i in range(8)},'cpu_workers':{str(i):workers[i+8].pid for i in range(20)}})+'\\n')
while not stopping: time.sleep(0.02)
for worker in workers: worker.terminate()
for worker in workers: worker.wait()
pid.unlink(missing_ok=True); state.unlink(missing_ok=True)
""",
            encoding="utf-8",
        )
        self.original.write_text(
            """#!/usr/bin/env bash
set -u
child=
cleanup() { trap - EXIT INT TERM; if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; exit "${1:-0}"; }
trap 'cleanup $?' EXIT; trap 'cleanup 130' INT; trap 'cleanup 143' TERM
while true; do
  "$PYTHON" "$HOLDER" --cpu-workers "$CPU_WORKERS" --cpu-duty "$CPU_DUTY" --gpu-count "$GPU_COUNT" --gpu-duty "$GPU_DUTY" --gpu-hold-mb "$GPU_HOLD_MB" --gpu-matrix-size "$GPU_MATRIX_SIZE" --gpu-burst-matmuls "$GPU_BURST_MATMULS" --pid-file "$PID_FILE" --state-file "$STATE_FILE" &
  child=$!; wait "$child"; child=; sleep 0.05
done
""",
            encoding="utf-8",
        )
        self.original.chmod(0o700)
        self.wrapper.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        self.wrapper.chmod(0o700)
        self.crash_wrapper.write_text(
            f"""import os, signal, sys
from pathlib import Path
sys.path.insert(0, {str(MODULE.parents[1])!r})
from agentmemorygym_verl import fallback_supervisor as target

boundary=sys.argv[1]
boundary_receipt=Path(sys.argv[2])
arguments=sys.argv[3:]

def crash():
    os.kill(os.getpid(), signal.SIGKILL)

if boundary == 'authorization_persisted':
    original=target._atomic_json
    def wrapped(path, payload):
        original(path, payload)
        if payload.get('mode') == 'resume_authorized':
            crash()
    target._atomic_json=wrapped
elif boundary == 'marker_removed':
    original=target._release_pause_marker
    def wrapped(*args, **kwargs):
        result=original(*args, **kwargs)
        crash()
        return result
    target._release_pause_marker=wrapped
elif boundary == 'request_consumed':
    original=target._remove_bound_json
    def wrapped(*args, **kwargs):
        result=original(*args, **kwargs)
        crash()
        return result
    target._remove_bound_json=wrapped
elif boundary == 'holder_attested':
    original=target._holder_resource_attestation
    def wrapped(*args, **kwargs):
        result=original(*args, **kwargs)
        target._atomic_json(boundary_receipt, {{
            'schema': 'amg_fallback_sigkill_fixture_boundary_v1',
            'boundary': boundary,
            'holder': target._holder_lease_snapshot(args[0]),
        }})
        crash()
        return result
    target._holder_resource_attestation=wrapped
else:
    raise SystemExit(f'unknown crash boundary: {{boundary}}')

raise SystemExit(target.main(arguments))
""",
            encoding="utf-8",
        )
        self.process: subprocess.Popen[bytes] | None = None

    def tearDown(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for process in self.auxiliary_processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.temporary.cleanup()

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _supervisor_command(self) -> list[str]:
        if self.formal_inventory_args is None:
            inventory = {
                "--nvidia-smi": "/usr/bin/nvidia-smi",
                "--nvidia-smi-sha256": self._digest(
                    Path("/usr/bin/nvidia-smi")
                ),
                "--auto-gpu-holder-state": str(
                    self.root / "auto-gpu-holder.state"
                ),
                "--auto-gpu-holder-command-fragment": "test_auto_gpu_holder.py",
                "--auto-cpu-holder-state": str(
                    self.root / "auto-cpu-holder.json"
                ),
                "--auto-cpu-holder-command-fragment": "test_auto_cpu_holder.py",
            }
        else:
            inventory = dict(
                zip(
                    self.formal_inventory_args[0::2],
                    self.formal_inventory_args[1::2],
                    strict=True,
                )
            )
        return [
            self.python,
            str(MODULE),
            "supervise",
            "--python",
            self.python,
            "--python-sha256",
            self._digest(Path(self.python)),
            "--original",
            str(self.original),
            "--original-sha256",
            self._digest(self.original),
            "--holder",
            str(self.holder),
            "--holder-sha256",
            self._digest(self.holder),
            "--watchdog-wrapper",
            str(self.wrapper),
            "--watchdog-wrapper-sha256",
            self._digest(self.wrapper),
            "--supervisor-script",
            str(MODULE),
            "--supervisor-script-sha256",
            self._digest(MODULE),
            "--bootstrap",
            str(BOOTSTRAP),
            "--bootstrap-sha256",
            self._digest(BOOTSTRAP),
            "--watchdog-identity-file",
            str(self.root / "watchdog-process-identity.json"),
            "--pid-file",
            str(self.pid_file),
            "--holder-state-file",
            str(self.holder_state),
            "--pause-marker",
            str(self.pause),
            "--release-request-file",
            str(self.release_request),
            "--supervisor-state",
            str(self.supervisor_state),
            "--supervisor-lock",
            str(self.supervisor_lock),
            "--transaction-lock",
            str(self.transaction_lock),
            "--watchdog-log",
            str(self.watchdog_log),
            "--nvidia-smi",
            inventory["--nvidia-smi"],
            "--nvidia-smi-sha256",
            inventory["--nvidia-smi-sha256"],
            "--auto-gpu-holder-state",
            inventory["--auto-gpu-holder-state"],
            "--auto-gpu-holder-command-fragment",
            inventory["--auto-gpu-holder-command-fragment"],
            "--auto-cpu-holder-state",
            inventory["--auto-cpu-holder-state"],
            "--auto-cpu-holder-command-fragment",
            inventory["--auto-cpu-holder-command-fragment"],
            "--poll-seconds",
            "0.02",
            "--restart-delay-seconds",
            "0.02",
            "--stop-timeout-seconds",
            "3",
        ]

    def _wrapper_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.pop("PYTHON", None)
        environment.update({
            "FALLBACK_SUPERVISOR_PYTHON": self.python,
            "FALLBACK_SUPERVISOR_MODULE": str(MODULE),
            "FALLBACK_PROCESS_BOOTSTRAP": str(BOOTSTRAP),
            "FALLBACK_WATCHDOG_ORIGINAL": str(self.original),
            "HOLDER": str(self.holder),
            "FALLBACK_WATCHDOG_WRAPPER": str(WATCHDOG_WRAPPER),
            "FALLBACK_ORIGINAL_SHA256": self._digest(self.original),
            "FALLBACK_HOLDER_SHA256": self._digest(self.holder),
            "FALLBACK_PYTHON_SHA256": self._digest(Path(self.python)),
            "FALLBACK_MODULE_SHA256": self._digest(MODULE),
            "FALLBACK_BOOTSTRAP_SHA256": self._digest(BOOTSTRAP),
            "FALLBACK_WATCHDOG_WRAPPER_SHA256": self._digest(WATCHDOG_WRAPPER),
            "FALLBACK_PID_FILE": str(self.pid_file),
            "FALLBACK_HOLDER_STATE_FILE": str(self.holder_state),
            "FALLBACK_PAUSE_MARKER": str(self.pause),
            "FALLBACK_RELEASE_REQUEST_FILE": str(self.release_request),
            "FALLBACK_SUPERVISOR_STATE": str(self.supervisor_state),
            "FALLBACK_SUPERVISOR_LOCK": str(self.supervisor_lock),
            "FALLBACK_TRANSACTION_LOCK": str(self.transaction_lock),
            "FALLBACK_WATCHDOG_IDENTITY_FILE": str(
                self.root / "watchdog-process-identity.json"
            ),
            "FALLBACK_WATCHDOG_LOG": str(self.watchdog_log),
            "FALLBACK_NVIDIA_SMI": "/usr/bin/nvidia-smi",
            "FALLBACK_NVIDIA_SMI_SHA256": self._digest(
                Path("/usr/bin/nvidia-smi")
            ),
            "FALLBACK_AUTO_GPU_HOLDER_STATE": str(
                self.root / "auto-gpu-holder.state"
            ),
            "FALLBACK_AUTO_GPU_HOLDER_COMMAND_FRAGMENT": "test_auto_gpu_holder.py",
            "FALLBACK_AUTO_CPU_HOLDER_STATE": str(
                self.root / "auto-cpu-holder.json"
            ),
            "FALLBACK_AUTO_CPU_HOLDER_COMMAND_FRAGMENT": "test_auto_cpu_holder.py",
        })
        return environment

    def _start_platform_wrapper(self) -> subprocess.Popen[bytes]:
        with self.supervisor_log.open("ab", buffering=0) as output:
            process = subprocess.Popen(
                ["bash", str(WATCHDOG_WRAPPER)],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=self._wrapper_environment(),
            )
        self.process = process
        return process

    def _start_supervisor(self) -> dict:
        with self.supervisor_log.open("ab", buffering=0) as output:
            self.process = subprocess.Popen(
                self._supervisor_command(),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return fallback._wait_supervisor_state(
            self.supervisor_state, modes={"holding"}, timeout_seconds=5
        )

    def _start_crashing_supervisor(self, boundary: str) -> dict:
        self._formal_inventory_args()
        command = self._supervisor_command()
        boundary_receipt = self.root / f"{boundary}.boundary.json"
        with self.supervisor_log.open("ab", buffering=0) as output:
            self.process = subprocess.Popen(
                [
                    self.python,
                    str(self.crash_wrapper),
                    boundary,
                    str(boundary_receipt),
                    *command[2:],
                ],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return fallback._wait_supervisor_state(
            self.supervisor_state,
            modes={"holding"},
            timeout_seconds=10,
            require_live_holder=True,
        )

    def _assert_actual_sigkill_resume_recovers(self, *, boundary: str) -> None:
        first = self._start_crashing_supervisor(boundary)
        first_holder = dict(first["holder"])
        owner = fallback._capture_identity(os.getpid())
        token = f"actual-sigkill-{boundary}"
        marker = fallback._create_pause_marker(
            self.pause,
            self._marker_payload(token, owner),
        )
        paused = fallback._wait_supervisor_state(
            self.supervisor_state,
            modes={"paused"},
            timeout_seconds=10,
            pause_token=token,
            require_holder_files_clear=True,
            expected_contract=first["immutable_contract"],
        )
        fallback._publish_release_request(
            self.release_request,
            marker=marker,
            paused_state=paused,
            owner=owner,
            expected_contract=first["immutable_contract"],
            mode="test_owner_release",
        )
        assert self.process is not None
        self.process.wait(timeout=20)
        self.assertEqual(self.process.returncode, -signal.SIGKILL)
        self.process = None
        crashed = json.loads(self.supervisor_state.read_text(encoding="utf-8"))
        self.assertIsNotNone(crashed["pending_resume"])
        self.assertFalse(fallback._identity_alive(first_holder))
        crashed_holder = fallback._holder_identity(
            self.pid_file,
            self.holder_state,
            watchdog_identity=crashed.get("watchdog"),
            holder_path=self.holder,
        )
        if boundary == "authorization_persisted":
            self.assertTrue(self.pause.is_file())
            self.assertTrue(self.release_request.is_file())
            self.assertIsNone(crashed_holder)
        elif boundary == "marker_removed":
            self.assertFalse(self.pause.exists())
            self.assertTrue(self.release_request.is_file())
            self.assertIsNone(crashed_holder)
        elif boundary == "request_consumed":
            self.assertFalse(self.pause.exists())
            self.assertFalse(self.release_request.exists())
            self.assertIsNone(crashed_holder)
        else:
            self.assertEqual(boundary, "holder_attested")
            self.assertFalse(self.pause.exists())
            self.assertFalse(self.release_request.exists())
            boundary_receipt = fallback._load_json(
                self.root / f"{boundary}.boundary.json"
            )
            self.assertEqual(
                boundary_receipt.get("schema"),
                "amg_fallback_sigkill_fixture_boundary_v1",
            )
            self.assertEqual(boundary_receipt.get("boundary"), boundary)
            attested_holder = boundary_receipt.get("holder")
            self.assertIsInstance(attested_holder, dict)
            self.assertIsInstance(attested_holder.get("parent"), dict)
            self.assertEqual(len(attested_holder.get("gpu_workers", ())), 8)
            self.assertEqual(len(attested_holder.get("cpu_workers", ())), 20)
            # process_bootstrap arms PDEATHSIG and may finish draining this
            # holder before the test reaps the SIGKILLed supervisor.  The
            # durable boundary receipt proves attestation occurred; if a live
            # holder is still observable, it must be that exact lease.
            if crashed_holder is not None:
                self.assertEqual(
                    attested_holder,
                    fallback._holder_lease_snapshot(crashed_holder),
                )

        restored = self._start_supervisor()
        self.assertEqual(restored["mode"], "holding")
        self.assertIsNone(restored["pending_resume"])
        self.assertFalse(self.pause.exists())
        self.assertFalse(self.release_request.exists())
        if crashed_holder is not None:
            self.assertFalse(fallback._identity_alive(crashed_holder))
        if boundary == "holder_attested":
            self.assertFalse(fallback._identity_alive(attested_holder["parent"]))
            self.assertNotEqual(
                attested_holder,
                fallback._holder_lease_snapshot(restored["holder"]),
            )
        self.assertTrue(fallback._identity_alive(restored["holder"]))
        self.assertNotEqual(
            (restored["holder"]["pid"], restored["holder"]["start_ticks"]),
            (first_holder["pid"], first_holder["start_ticks"]),
        )

    def _formal_inventory_args(self) -> list[str]:
        if self.formal_inventory_args is not None:
            return list(self.formal_inventory_args)
        gpu_script = self.root / "test_auto_gpu_holder.py"
        gpu_children = self.root / "test-auto-gpu-children.json"
        gpu_state = self.root / "test-auto-gpu.state"
        gpu_script.write_text(
            """import json, signal, subprocess, sys, time
from pathlib import Path
children_path=Path(sys.argv[1]); state_path=Path(sys.argv[2]); stopping=False
children=[subprocess.Popen([sys.executable,'-c','import time; time.sleep(3600)']) for _ in range(8)]
def stop(*_):
 global stopping; stopping=True
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
children_path.write_text(json.dumps([p.pid for p in children]))
state_path.write_text(f'fixture mode=hold pid={__import__("os").getpid()} gpu=8 cpu=0 work=test\\n')
while not stopping: time.sleep(0.02)
for child in children: child.terminate()
for child in children: child.wait()
""",
            encoding="utf-8",
        )
        cpu_script = self.root / "test_auto_cpu_holder.py"
        cpu_state = self.root / "test-auto-cpu.json"
        cpu_script.write_text(
            """import json, os, signal, subprocess, sys, time
from pathlib import Path
state_path=Path(sys.argv[1]); stopping=False
worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(3600)'])
def stop(*_):
 global stopping; stopping=True
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
state_path.write_text(json.dumps({'state':'holding','parent_pid':os.getpid(),'worker_count_requested':1,'worker_pids':[worker.pid]})+'\\n')
while not stopping: time.sleep(0.02)
worker.terminate(); worker.wait()
""",
            encoding="utf-8",
        )
        gpu_parent = subprocess.Popen(
            [self.python, str(gpu_script), str(gpu_children), str(gpu_state)],
            start_new_session=True,
        )
        cpu_parent = subprocess.Popen(
            [self.python, str(cpu_script), str(cpu_state)], start_new_session=True
        )
        self.auxiliary_processes.extend((gpu_parent, cpu_parent))
        deadline = time.monotonic() + 5
        while (
            (not gpu_children.is_file() or not gpu_state.is_file() or not cpu_state.is_file())
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        self.assertTrue(gpu_children.is_file())
        self.assertTrue(gpu_state.is_file())
        self.assertTrue(cpu_state.is_file())

        nvidia_smi = self.root / "nvidia-smi"
        nvidia_smi.write_text(
            f"""#!{self.python}
import json, sys
from pathlib import Path
args=' '.join(sys.argv[1:])
if '--query-gpu=uuid,utilization.gpu' in args:
  for index in range(8): print(f'GPU-{{index}}, 10')
elif '--query-gpu=' in args:
  for index in range(8): print(f'{{index}}, NVIDIA B300 TEST, GPU-{{index}}')
elif '--query-compute-apps=' in args:
  pids=[]
  state=Path({str(self.holder_state)!r})
  if state.is_file():
    pids.extend(int(v) for v in json.loads(state.read_text()).get('gpu_workers',{{}}).values())
  children=Path({str(gpu_children)!r})
  if children.is_file(): pids.extend(int(v) for v in json.loads(children.read_text()))
  for index,pid in enumerate(pids): print(f'GPU-{{index % 8}}, {{pid}}, python, 512')
else:
  raise SystemExit(2)
""",
            encoding="utf-8",
        )
        nvidia_smi.chmod(0o700)
        self.formal_inventory_args = [
            "--nvidia-smi",
            str(nvidia_smi),
            "--nvidia-smi-sha256",
            self._digest(nvidia_smi),
            "--auto-gpu-holder-state",
            str(gpu_state),
            "--auto-gpu-holder-command-fragment",
            str(gpu_script),
            "--auto-cpu-holder-state",
            str(cpu_state),
            "--auto-cpu-holder-command-fragment",
            str(cpu_script),
        ]
        return list(self.formal_inventory_args)

    def _marker_payload(self, token: str, owner: dict, **extra) -> dict:
        supervisor = json.loads(self.supervisor_state.read_text(encoding="utf-8"))
        return {
            "schema": fallback._PAUSE_SCHEMA,
            "token": token,
            "owner": owner,
            "run_id": extra.get("run_id", ""),
            "run_dir": str(extra.get("run_dir", self.root / "no-run")),
            "ports": extra.get("ports", []),
            "protected_identities": extra.get("protected_identities", []),
            "recovery_receipt": str(self.root / f"{token}.recovery.json"),
            "release_request_file": str(self.release_request),
            "immutable_contract": supervisor["immutable_contract"],
            **(
                {"orchestrator_identity": str(extra["orchestrator_identity"])}
                if extra.get("orchestrator_identity")
                else {}
            ),
        }

    def test_signal_identity_fails_closed_when_pidfd_is_unavailable(self) -> None:
        process = subprocess.Popen(["sleep", "60"])
        try:
            identity = fallback._capture_identity(process.pid)
            with mock.patch.object(
                fallback,
                "_pidfd_open_exact",
                side_effect=fallback.FallbackError("synthetic pidfd unavailable"),
            ):
                with self.assertRaisesRegex(
                    fallback.FallbackError, "synthetic pidfd unavailable"
                ):
                    fallback._signal_identity(identity, signal.SIGTERM)
            self.assertIsNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_orphan_drain_refuses_stale_numeric_process_group(self) -> None:
        foreign = subprocess.Popen(["sleep", "60"], start_new_session=True)
        try:
            stale = {
                "pid": 99_999_999,
                "start_ticks": "1",
                "pgrp": foreign.pid,
            }
            with self.assertRaisesRegex(
                fallback.FallbackError, "stale numeric process group"
            ):
                fallback._drain_orphaned_watchdog_group(
                    stale, timeout_seconds=0.1
                )
            self.assertIsNone(foreign.poll())
        finally:
            if foreign.poll() is None:
                foreign.kill()
                foreign.wait()

    def test_process_inventory_fails_closed_on_unreadable_environment(self) -> None:
        with mock.patch.object(
            Path,
            "read_bytes",
            side_effect=PermissionError(13, "synthetic unreadable environ"),
        ):
            with self.assertRaisesRegex(
                fallback.FallbackError, "cannot audit process environment"
            ):
                fallback._proc_environment_for_inventory(os.getpid())

    def test_relevant_process_classifier_ignores_runtime_path_and_audit_text(
        self,
    ) -> None:
        runtime_python = (
            "/dev/shm/qwen35-runtime-verl-main-sglang-fsdp/bin/python3.12"
        )
        self.assertFalse(fallback._looks_like_relevant_workload((runtime_python, "-")))
        self.assertFalse(
            fallback._looks_like_relevant_workload(
                (
                    "bash",
                    "-c",
                    "sed -n '1,80p' /tmp/swesmith/multitask_orchestrator.py",
                )
            )
        )
        self.assertTrue(
            fallback._looks_like_relevant_workload(
                (runtime_python, "-m", "sglang.launch_server")
            )
        )
        self.assertTrue(
            fallback._looks_like_relevant_workload(
                (runtime_python, "/srv/amg/multitask_orchestrator.py")
            )
        )
        self.assertTrue(
            fallback._looks_like_relevant_workload(("ray::FullyAsyncTrainer",))
        )

    def test_persisted_watchdog_identity_replacement_is_rejected(self) -> None:
        state = self._start_supervisor()
        identity_path = self.root / "watchdog-process-identity.json"
        self.assertTrue(identity_path.is_file())
        assert self.process is not None
        os.kill(self.process.pid, signal.SIGSTOP)
        foreign = subprocess.Popen(["sleep", "60"], start_new_session=True)
        try:
            foreign_identity = fallback._capture_identity(foreign.pid)
            replacement = {
                "schema": fallback._PROCESS_IDENTITY_SCHEMA,
                "name": "fallback-holder-watchdog",
                "pid": foreign_identity["pid"],
                "start_ticks": foreign_identity["start_ticks"],
                "process_group": foreign_identity["pid"],
                "bootstrap": str(BOOTSTRAP),
                "command": ["bash", str(self.original)],
                "immutable_contract": state["immutable_contract"],
            }
            fallback._atomic_json(identity_path, replacement)
            with self.assertRaisesRegex(
                fallback.FallbackError, "watchdog identity was replaced"
            ):
                fallback._load_bound_previous_watchdog(
                    state_path=self.supervisor_state,
                    identity_path=identity_path,
                    expected_bootstrap=BOOTSTRAP,
                    expected_command=("bash", str(self.original)),
                    expected_contract=state["immutable_contract"],
                )
            self.assertIsNone(foreign.poll())
        finally:
            if self.process is not None and self.process.poll() is None:
                os.kill(self.process.pid, signal.SIGCONT)
            if foreign.poll() is None:
                foreign.kill()
                foreign.wait()

    def test_failed_generation_drain_preserves_all_recovery_artifacts(self) -> None:
        identity_path = self.root / "watchdog-process-identity.json"
        live_identity = fallback._capture_identity(os.getpid())
        fallback._atomic_json(
            identity_path,
            {
                "schema": fallback._PROCESS_IDENTITY_SCHEMA,
                "name": "fallback-holder-watchdog",
                "pid": live_identity["pid"],
                "start_ticks": live_identity["start_ticks"],
                "process_group": live_identity["pid"],
                "bootstrap": str(BOOTSTRAP),
                "command": ["bash", str(self.original)],
                "immutable_contract": {},
            },
        )
        self.pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
        self.holder_state.write_text(
            json.dumps({"parent_pid": os.getpid(), "mode": "hold"}) + "\n",
            encoding="utf-8",
        )
        lease = fallback._process_identity_file_binding(identity_path)

        with self.assertRaisesRegex(
            fallback.FallbackError, "live fallback watchdog identity"
        ):
            fallback._clear_drained_watchdog_artifacts(
                identity_path=identity_path,
                expected_lease=lease,
                expected_identity=live_identity,
                pid_file=self.pid_file,
                holder_state_file=self.holder_state,
                holder_path=self.holder,
            )

        self.assertTrue(identity_path.is_file())
        self.assertTrue(self.pid_file.is_file())
        self.assertTrue(self.holder_state.is_file())

    def test_replaced_generation_identity_preserves_all_recovery_artifacts(
        self,
    ) -> None:
        identity_path = self.root / "watchdog-process-identity.json"
        payload = {
            "schema": fallback._PROCESS_IDENTITY_SCHEMA,
            "name": "fallback-holder-watchdog",
            "pid": 99999999,
            "start_ticks": "1",
            "process_group": 99999999,
            "bootstrap": str(BOOTSTRAP),
            "command": ["bash", str(self.original)],
            "immutable_contract": {},
        }
        fallback._atomic_json(identity_path, payload)
        lease = fallback._process_identity_file_binding(identity_path)
        displaced = self.root / "watchdog-process-identity.old"
        identity_path.rename(displaced)
        fallback._atomic_json(identity_path, payload)
        self.pid_file.write_text("99999999\n", encoding="utf-8")
        self.holder_state.write_text(
            json.dumps({"parent_pid": 99999999, "mode": "hold"}) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            fallback.FallbackError, "replaced fallback watchdog identity"
        ):
            fallback._clear_drained_watchdog_artifacts(
                identity_path=identity_path,
                expected_lease=lease,
                expected_identity={"pid": 99999999, "start_ticks": "1"},
                pid_file=self.pid_file,
                holder_state_file=self.holder_state,
                holder_path=self.holder,
            )

        self.assertTrue(identity_path.is_file())
        self.assertTrue(self.pid_file.is_file())
        self.assertTrue(self.holder_state.is_file())

    def test_contract_drift_restart_never_releases_dead_owner_marker(self) -> None:
        state = self._start_supervisor()
        owner = subprocess.Popen(["sleep", "60"])
        try:
            token = "contract-drift-owner-death"
            fallback._create_pause_marker(
                self.pause,
                self._marker_payload(token, fallback._capture_identity(owner.pid)),
            )
            fallback._wait_supervisor_state(
                self.supervisor_state,
                modes={"paused"},
                timeout_seconds=5,
                pause_token=token,
                require_holder_files_clear=True,
                expected_contract=state["immutable_contract"],
            )
            owner.terminate()
            owner.wait(timeout=5)
            assert self.process is not None
            self.process.kill()
            self.process.wait(timeout=5)
            self.wrapper.write_text("#!/usr/bin/env bash\n# drift\nexit 0\n", encoding="utf-8")
            with self.supervisor_log.open("ab", buffering=0) as output:
                self.process = subprocess.Popen(
                    self._supervisor_command(),
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            drifted = fallback._wait_supervisor_state(
                self.supervisor_state,
                modes={"pause_contract_error"},
                timeout_seconds=5,
            )
            self.assertEqual(drifted["pause_token"], token)
            self.assertTrue(self.pause.exists())
            self.assertNotEqual(
                drifted["immutable_contract"], state["immutable_contract"]
            )
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait()

    def test_pod_preflight_rejects_foreign_gpu_process(self) -> None:
        foreign = subprocess.Popen(["sleep", "60"], start_new_session=True)
        try:
            foreign_identity = fallback._capture_identity(foreign.pid)
            holder_parent = fallback._capture_identity(os.getpid())
            hardware = {
                "devices": tuple(
                    {"index": index, "name": "B300", "uuid": f"GPU-{index}"}
                    for index in range(8)
                ),
                "applications": (
                    {
                        "gpu_uuid": "GPU-0",
                        "pid": foreign.pid,
                        "process_name": "python",
                        "used_gpu_memory_mib": 512,
                        "identity": foreign_identity,
                        "cmdline": "sleep 60",
                    },
                ),
            }
            with (
                mock.patch.object(
                    fallback,
                    "_auto_gpu_holder_inventory",
                    return_value={"parent": holder_parent, "descendants": ()},
                ),
                mock.patch.object(
                    fallback,
                    "_auto_cpu_holder_inventory",
                    return_value={"parent": holder_parent, "workers": ()},
                ),
                mock.patch.object(fallback, "_nvidia_inventory", return_value=hardware),
                mock.patch.object(fallback, "_relevant_processes", return_value=()),
            ):
                report = fallback._pod_preflight_inventory(
                    nvidia_smi=self.root / "unused",
                    nvidia_smi_sha256="unused",
                    auto_gpu_holder_state=self.root / "unused-gpu",
                    auto_gpu_holder_command_fragment="unused",
                    auto_cpu_holder_state=self.root / "unused-cpu",
                    auto_cpu_holder_command_fragment="unused",
                    allowed_identities=(holder_parent,),
                )
            self.assertFalse(report["safe"])
            self.assertEqual(
                report["foreign_gpu_applications"][0]["pid"], foreign.pid
            )
        finally:
            if foreign.poll() is None:
                foreign.kill()
                foreign.wait()

    def test_holder_identity_rejects_cross_role_pid_reuse(self) -> None:
        self.pid_file.write_text("100\n", encoding="utf-8")
        gpu_workers = {str(i): 200 + i for i in range(8)}
        cpu_workers = {str(i): 208 + i for i in range(20)}
        cpu_workers["0"] = gpu_workers["0"]
        self.holder_state.write_text(
            json.dumps(
                {
                    "parent_pid": 100,
                    "mode": "hold",
                    "gpu_workers": gpu_workers,
                    "cpu_workers": cpu_workers,
                }
            ),
            encoding="utf-8",
        )

        def identity(pid: int) -> dict:
            return {
                "pid": pid,
                "ppid": 99 if pid == 100 else 100,
                "pgrp": 100,
                "state": "R",
                "start_ticks": str(pid * 10),
            }

        with (
            mock.patch.object(fallback, "_proc_identity", side_effect=identity),
            mock.patch.object(
                fallback, "_cmdline", return_value=f"python {self.holder}"
            ),
        ):
            self.assertIsNone(
                fallback._holder_identity(
                    self.pid_file,
                    self.holder_state,
                    watchdog_identity=identity(100),
                    holder_path=self.holder,
                )
            )

    def test_holder_activity_requires_every_cpu_worker_and_real_gpu_memory(
        self,
    ) -> None:
        gpu_workers = tuple(
            {"pid": 300 + i, "start_ticks": str(3000 + i)} for i in range(8)
        )
        cpu_workers = tuple(
            {"pid": 400 + i, "start_ticks": str(4000 + i)} for i in range(20)
        )
        holder = {
            "gpu_worker_identities": gpu_workers,
            "cpu_worker_identities": cpu_workers,
        }
        with (
            mock.patch.object(
                fallback,
                "_process_cpu_ticks",
                side_effect=([0] * 20 + [1] + [0] * 19),
            ),
            mock.patch.object(fallback.time, "sleep"),
        ):
            with self.assertRaisesRegex(fallback.FallbackError, "did not all"):
                fallback._holder_resource_attestation(
                    holder,
                    nvidia_smi=Path("/unused"),
                    nvidia_smi_sha256="unused",
                )

        gpu_rows = tuple(
            {
                "gpu_uuid": f"GPU-{i}",
                "pid": worker["pid"],
                "process_name": "python",
                "used_gpu_memory_mib": 0,
                "identity": dict(worker),
                "cmdline": "fixture",
            }
            for i, worker in enumerate(gpu_workers)
        )
        with (
            mock.patch.object(
                fallback,
                "_process_cpu_ticks",
                side_effect=([0] * 20 + [1] * 20),
            ),
            mock.patch.object(fallback.time, "sleep"),
            mock.patch.object(
                fallback,
                "_nvidia_inventory",
                return_value={"devices": (), "applications": gpu_rows},
            ),
        ):
            with self.assertRaisesRegex(fallback.FallbackError, "not resident"):
                fallback._holder_resource_attestation(
                    holder,
                    nvidia_smi=Path("/unused"),
                    nvidia_smi_sha256="unused",
                )

    def test_pod_preflight_rejects_allowed_pid_with_reused_identity(self) -> None:
        old_gpu = {"pid": 501, "start_ticks": "old"}
        reused_gpu = {"pid": 501, "start_ticks": "new", "state": "R"}
        with (
            mock.patch.object(
                fallback,
                "_auto_gpu_holder_inventory",
                return_value={"parent": old_gpu, "descendants": ()},
            ),
            mock.patch.object(
                fallback,
                "_auto_cpu_holder_inventory",
                return_value={
                    "parent": {"pid": 601, "start_ticks": "cpu"},
                    "workers": (),
                },
            ),
            mock.patch.object(
                fallback,
                "_nvidia_inventory",
                return_value={
                    "devices": (),
                    "applications": (
                        {
                            "pid": 501,
                            "identity": reused_gpu,
                            "gpu_uuid": "GPU-0",
                            "process_name": "foreign",
                            "used_gpu_memory_mib": 512,
                            "cmdline": "foreign",
                        },
                    ),
                },
            ),
            mock.patch.object(fallback, "_relevant_processes", return_value=()),
        ):
            report = fallback._pod_preflight_inventory(
                nvidia_smi=Path("/unused"),
                nvidia_smi_sha256="unused",
                auto_gpu_holder_state=Path("/unused-gpu"),
                auto_gpu_holder_command_fragment="gpu-holder",
                auto_cpu_holder_state=Path("/unused-cpu"),
                auto_cpu_holder_command_fragment="cpu-holder",
                allowed_identities=(),
            )
        self.assertFalse(report["safe"])
        self.assertEqual(len(report["foreign_gpu_applications"]), 1)

    def test_inner_holder_restore_requires_pass_receipt_and_exact_markers(self) -> None:
        run_dir = self.root / "inner-run"
        transaction = run_dir / "holder-transaction"
        transaction.mkdir(parents=True)
        cpu = self.root / "cpu-marker"
        gpu = self.root / "gpu-marker"
        cpu.write_text("cpu-owner\n", encoding="utf-8")
        state = {
            "schema": "amg_marker_transaction_v1",
            "status": "restored",
            "markers": [
                {
                    "name": "cpu",
                    "path": str(cpu),
                    "restored": True,
                    "restore_target_set": True,
                    "restore_target": "cpu-owner",
                },
                {
                    "name": "gpu",
                    "path": str(gpu),
                    "restored": True,
                    "restore_target_set": True,
                    "restore_target": None,
                },
            ],
        }
        (transaction / "state.json").write_text(
            json.dumps(state) + "\n", encoding="utf-8"
        )
        receipt_path = transaction / "watcher-exit.json"
        receipt = {
            "schema": "amg_marker_watcher_exit_v1",
            "status": "pass",
            "mode": "explicit_restore",
            "state_status": "restored",
        }
        receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
        with mock.patch.dict(
            fallback._INNER_HOLDER_MARKERS,
            {"cpu": cpu, "gpu": gpu},
            clear=True,
        ):
            report = fallback._inner_holder_restore_report(run_dir)
            self.assertTrue(report["safe"])
            receipt["status"] = "fail"
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                fallback.FallbackError, "did not attest restored state"
            ):
                fallback._inner_holder_restore_report(run_dir)

    def test_migrate_validates_bootstrap_before_creating_pause_marker(self) -> None:
        receipt = self.root / "migration-invalid-bootstrap.json"
        args = argparse.Namespace(
            python=self.python,
            python_sha256=self._digest(Path(self.python)),
            transaction_lock=str(self.transaction_lock),
            pause_marker=str(self.pause),
            release_request_file=str(self.release_request),
            supervisor_state=str(self.supervisor_state),
            supervisor_lock=str(self.supervisor_lock),
            watchdog_log=str(self.watchdog_log),
            pid_file=str(self.pid_file),
            holder_state_file=str(self.holder_state),
            receipt=str(receipt),
            original=str(self.original),
            original_sha256=self._digest(self.original),
            holder=str(self.holder),
            holder_sha256=self._digest(self.holder),
            watchdog_wrapper=str(self.wrapper),
            watchdog_wrapper_sha256=self._digest(self.wrapper),
            supervisor_script=str(MODULE),
            supervisor_script_sha256=self._digest(MODULE),
            bootstrap=str(BOOTSTRAP),
            bootstrap_sha256="0" * 64,
            watchdog_identity_file=str(
                self.root / "watchdog-process-identity.json"
            ),
            nvidia_smi="/usr/bin/nvidia-smi",
            nvidia_smi_sha256=self._digest(Path("/usr/bin/nvidia-smi")),
            auto_gpu_holder_state=str(self.root / "auto-gpu-holder.state"),
            auto_gpu_holder_command_fragment="test_auto_gpu_holder.py",
            auto_cpu_holder_state=str(self.root / "auto-cpu-holder.json"),
            auto_cpu_holder_command_fragment="test_auto_cpu_holder.py",
            stop_timeout_seconds=3,
            poll_seconds=0.02,
            restart_delay_seconds=0.02,
            supervisor_restart_timeout_seconds=8,
        )
        with self.assertRaisesRegex(fallback.FallbackError, "sha256 mismatch"):
            fallback.migrate(args)
        self.assertFalse(self.pause.exists())
        self.assertFalse(receipt.exists())

    def test_live_protected_child_blocks_owner_death_recovery(self) -> None:
        self._formal_inventory_args()
        self._start_supervisor()
        protected = subprocess.Popen(["sleep", "60"])
        try:
            protected_identity = fallback._capture_identity(protected.pid)
            dead_owner = {"pid": 99999999, "start_ticks": "1"}
            token = "protected-child"
            marker = self._marker_payload(
                token,
                dead_owner,
                protected_identities=[protected_identity],
            )
            fallback._create_pause_marker(self.pause, marker)
            blocked = fallback._wait_supervisor_state(
                self.supervisor_state,
                modes={"owner_death_recovery_blocked"},
                timeout_seconds=5,
                pause_token=token,
                require_holder_files_clear=True,
            )
            self.assertFalse(blocked["pause_owner_alive"])
            self.assertFalse(blocked["recovery_drain"]["safe"])
            time.sleep(0.2)
            self.assertTrue(self.pause.exists())

            protected.terminate()
            protected.wait(timeout=5)
            restored = fallback._wait_supervisor_state(
                self.supervisor_state, modes={"holding"}, timeout_seconds=5
            )
            self.assertIsNotNone(restored["holder"])
            self.assertFalse(self.pause.exists())
        finally:
            if protected.poll() is None:
                protected.kill()
                protected.wait()

    def test_platform_watchdog_wrapper_execs_pinned_supervisor(self) -> None:
        self._start_platform_wrapper()
        state = fallback._wait_supervisor_state(
            self.supervisor_state,
            modes={"holding"},
            timeout_seconds=5,
            require_live_holder=True,
        )
        self.assertEqual(state["immutable_contract"]["bootstrap"], str(BOOTSTRAP))
        self.assertEqual(
            state["immutable_contract"]["watchdog_wrapper"],
            str(WATCHDOG_WRAPPER),
        )

    def test_platform_wrapper_uses_source_module_digest_without_override(self) -> None:
        environment = self._wrapper_environment()
        environment.pop("FALLBACK_MODULE_SHA256")
        self.assertNotIn("FALLBACK_MODULE_SHA256", environment)
        with self.supervisor_log.open("ab", buffering=0) as output:
            self.process = subprocess.Popen(
                ["bash", str(WATCHDOG_WRAPPER)],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
        state = fallback._wait_supervisor_state(
            self.supervisor_state,
            modes={"holding"},
            timeout_seconds=5,
            require_live_holder=True,
        )
        self.assertEqual(
            state["immutable_contract"]["supervisor_script_sha256"],
            self._digest(MODULE),
        )

    def test_platform_wrapper_recovers_without_training_python_environment(self) -> None:
        environment = self._wrapper_environment()
        self.assertNotIn("PYTHON", environment)
        with self.supervisor_log.open("ab", buffering=0) as output:
            self.process = subprocess.Popen(
                ["bash", str(WATCHDOG_WRAPPER)],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
        state = fallback._wait_supervisor_state(
            self.supervisor_state,
            modes={"holding"},
            timeout_seconds=5,
            require_live_holder=True,
        )
        self.assertEqual(state["immutable_contract"]["python"], self.python)

    def test_platform_restart_after_supervisor_sigkill_drains_old_holder(self) -> None:
        first_process = self._start_platform_wrapper()
        first = fallback._wait_supervisor_state(
            self.supervisor_state,
            modes={"holding"},
            timeout_seconds=5,
            require_live_holder=True,
        )
        first_holder = first["holder"]
        first_watchdog = first["watchdog"]
        first_process.kill()
        first_process.wait(timeout=5)

        replacement = self._start_platform_wrapper()
        self.assertIs(self.process, replacement)
        second = fallback._wait_supervisor_state(
            self.supervisor_state,
            modes={"holding"},
            timeout_seconds=10,
            require_live_holder=True,
        )
        self.assertFalse(fallback._identity_alive(first_watchdog))
        self.assertFalse(fallback._identity_alive(first_holder))
        self.assertNotEqual(first_holder["pid"], second["holder"]["pid"])

    @unittest.skipUnless(
        _supports_exact_crash_recovery_fixture(),
        "requires Linux pidfd and renameat2",
    )
    def test_sigkill_recovery_after_resume_authorization(self) -> None:
        self._assert_actual_sigkill_resume_recovers(
            boundary="authorization_persisted"
        )

    @unittest.skipUnless(
        _supports_exact_crash_recovery_fixture(),
        "requires Linux pidfd and renameat2",
    )
    def test_sigkill_recovery_after_marker_removal(self) -> None:
        self._assert_actual_sigkill_resume_recovers(boundary="marker_removed")

    @unittest.skipUnless(
        _supports_exact_crash_recovery_fixture(),
        "requires Linux pidfd and renameat2",
    )
    def test_sigkill_recovery_after_request_consumption(self) -> None:
        self._assert_actual_sigkill_resume_recovers(boundary="request_consumed")

    @unittest.skipUnless(
        _supports_exact_crash_recovery_fixture(),
        "requires Linux pidfd and renameat2",
    )
    def test_sigkill_recovery_after_holder_attestation(self) -> None:
        self._assert_actual_sigkill_resume_recovers(boundary="holder_attested")

    def test_pause_then_resume_has_exactly_one_holder(self) -> None:
        first = self._start_supervisor()
        first_holder = first["holder"]
        owner = fallback._capture_identity(os.getpid())
        token = "normal-pause"
        observation = fallback._create_pause_marker(
            self.pause, self._marker_payload(token, owner)
        )
        fallback._wait_supervisor_state(
            self.supervisor_state,
            modes={"paused"},
            timeout_seconds=5,
            pause_token=token,
            require_holder_files_clear=True,
        )
        self.assertFalse(fallback._identity_alive(first_holder))
        fallback._release_pause_marker(
            self.pause,
            token=token,
            expected=fallback._marker_binding_view(observation),
        )
        second = fallback._wait_supervisor_state(
            self.supervisor_state, modes={"holding"}, timeout_seconds=5
        )
        self.assertNotEqual(first_holder["pid"], second["holder"]["pid"])
        self.assertTrue(fallback._identity_alive(second["holder"]))

    def test_pause_release_preserves_existing_quarantine(self) -> None:
        token = "foreign-quarantine"
        observation = fallback._create_pause_marker(
            self.pause,
            {
                "schema": fallback._PAUSE_SCHEMA,
                "token": token,
            },
        )
        quarantine = self.pause.with_name(f".{self.pause.name}.released.{token}")
        quarantine.write_text("foreign\n", encoding="utf-8")

        with self.assertRaisesRegex(
            fallback.FallbackError, "existing pause-marker quarantine"
        ):
            fallback._release_pause_marker(
                self.pause,
                token=token,
                expected=fallback._marker_binding_view(observation),
            )

        self.assertTrue(self.pause.is_file())
        self.assertEqual(quarantine.read_text(encoding="utf-8"), "foreign\n")

    def test_unexpected_watchdog_death_drains_orphan_before_restart(self) -> None:
        first = self._start_supervisor()
        first_watchdog = first["watchdog"]
        first_holder = first["holder"]
        self.assertTrue(fallback._signal_identity(first_watchdog, signal.SIGKILL))

        deadline = time.monotonic() + 8
        replacement: dict | None = None
        while time.monotonic() < deadline:
            try:
                state = json.loads(self.supervisor_state.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.05)
                continue
            if state.get("mode") == "holding" and state.get("generation", 0) > 1:
                replacement = state
                break
            time.sleep(0.05)
        self.assertIsNotNone(replacement)
        assert replacement is not None
        self.assertFalse(fallback._identity_alive(first_watchdog))
        self.assertFalse(fallback._identity_alive(first_holder))
        self.assertTrue(fallback._identity_alive(replacement["watchdog"]))
        self.assertTrue(fallback._identity_alive(replacement["holder"]))
        self.assertEqual(
            int(replacement["holder"]["pgrp"]),
            int(replacement["watchdog"]["pid"]),
        )

    def test_migration_waits_for_external_platform_restart(self) -> None:
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHON": self.python,
                "HOLDER": str(self.holder),
                "CPU_WORKERS": "20",
                "CPU_DUTY": "0.9",
                "GPU_COUNT": "8",
                "GPU_DUTY": "0.18",
                "GPU_HOLD_MB": "512",
                "GPU_MATRIX_SIZE": "4096",
                "GPU_BURST_MATMULS": "8",
                "PID_FILE": str(self.pid_file),
                "STATE_FILE": str(self.holder_state),
            }
        )
        old = subprocess.Popen(
            ["bash", str(self.original)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        deadline = time.monotonic() + 5
        while not self.pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(self.pid_file.is_file())

        inventory_args = self._formal_inventory_args()
        inventory = dict(
            zip(inventory_args[0::2], inventory_args[1::2], strict=True)
        )

        restart_error: list[BaseException] = []

        def platform_restart() -> None:
            try:
                old.wait(timeout=8)
                with self.supervisor_log.open("ab", buffering=0) as output:
                    self.process = subprocess.Popen(
                        self._supervisor_command(),
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
            except BaseException as error:  # surfaced in the test thread
                restart_error.append(error)

        thread = threading.Thread(target=platform_restart, daemon=True)
        thread.start()
        receipt = self.root / "migration-receipt.json"
        args = argparse.Namespace(
            python=self.python,
            python_sha256=self._digest(Path(self.python)),
            transaction_lock=str(self.transaction_lock),
            pause_marker=str(self.pause),
            release_request_file=str(self.release_request),
            supervisor_state=str(self.supervisor_state),
            supervisor_lock=str(self.supervisor_lock),
            watchdog_log=str(self.watchdog_log),
            pid_file=str(self.pid_file),
            holder_state_file=str(self.holder_state),
            receipt=str(receipt),
            original=str(self.original),
            original_sha256=self._digest(self.original),
            holder=str(self.holder),
            holder_sha256=self._digest(self.holder),
            watchdog_wrapper=str(self.wrapper),
            watchdog_wrapper_sha256=self._digest(self.wrapper),
            supervisor_script=str(MODULE),
            supervisor_script_sha256=self._digest(MODULE),
            bootstrap=str(BOOTSTRAP),
            bootstrap_sha256=self._digest(BOOTSTRAP),
            watchdog_identity_file=str(
                self.root / "watchdog-process-identity.json"
            ),
            nvidia_smi=inventory["--nvidia-smi"],
            nvidia_smi_sha256=inventory["--nvidia-smi-sha256"],
            auto_gpu_holder_state=inventory["--auto-gpu-holder-state"],
            auto_gpu_holder_command_fragment=inventory[
                "--auto-gpu-holder-command-fragment"
            ],
            auto_cpu_holder_state=inventory["--auto-cpu-holder-state"],
            auto_cpu_holder_command_fragment=inventory[
                "--auto-cpu-holder-command-fragment"
            ],
            stop_timeout_seconds=3,
            poll_seconds=0.02,
            restart_delay_seconds=0.02,
            supervisor_restart_timeout_seconds=8,
        )
        try:
            self.assertEqual(fallback.migrate(args), 0)
            thread.join(timeout=1)
            self.assertFalse(restart_error)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["bootstrap_sha256"], self._digest(BOOTSTRAP))
            self.assertEqual(
                payload["immutable_contract"],
                json.loads(self.supervisor_state.read_text())["immutable_contract"],
            )
            state = fallback._wait_supervisor_state(
                self.supervisor_state,
                modes={"holding"},
                timeout_seconds=5,
                require_live_holder=True,
            )
            self.assertTrue(fallback._identity_alive(state["holder"]))
            self.assertFalse(self.pause.exists())
        finally:
            if old.poll() is None:
                old.terminate()
                old.wait(timeout=5)

    def test_formal_owner_lock_and_sigkill_recovery(self) -> None:
        inventory_args = self._formal_inventory_args()
        self._start_supervisor()
        run_id = "formal-owner-fixture"
        run_dir = self.root / "run"
        attempt = self.root / "attempt"
        console = self.root / "console.log"
        command = [
            self.python,
            str(MODULE),
            "formal",
            "--transaction-lock",
            str(self.transaction_lock),
            "--python",
            self.python,
            "--python-sha256",
            self._digest(Path(self.python)),
            "--pause-marker",
            str(self.pause),
            "--release-request-file",
            str(self.release_request),
            "--supervisor-state",
            str(self.supervisor_state),
            "--supervisor-lock",
            str(self.supervisor_lock),
            "--watchdog-log",
            str(self.watchdog_log),
            "--watchdog-identity-file",
            str(self.root / "watchdog-process-identity.json"),
            "--pid-file",
            str(self.pid_file),
            "--holder-state-file",
            str(self.holder_state),
            "--original",
            str(self.original),
            "--original-sha256",
            self._digest(self.original),
            "--holder",
            str(self.holder),
            "--holder-sha256",
            self._digest(self.holder),
            "--watchdog-wrapper",
            str(self.wrapper),
            "--watchdog-wrapper-sha256",
            self._digest(self.wrapper),
            "--supervisor-script",
            str(MODULE),
            "--supervisor-script-sha256",
            self._digest(MODULE),
            "--bootstrap-sha256",
            self._digest(BOOTSTRAP),
            "--poll-seconds",
            "0.02",
            "--restart-delay-seconds",
            "0.02",
            "--stop-timeout-seconds",
            "3",
            "--attempt-dir",
            str(attempt),
            "--run-id",
            run_id,
            "--run-dir",
            str(run_dir),
            "--bootstrap",
            str(BOOTSTRAP),
            "--cwd",
            str(self.root),
            "--console",
            str(console),
            *inventory_args,
            "--cleanup-timeout-seconds",
            "5",
            "--",
            self.python,
            "-c",
            "import time; time.sleep(60)",
        ]
        owner = subprocess.Popen(command)
        second: subprocess.CompletedProcess[bytes] | None = None
        try:
            identity_path = attempt / "orchestrator-process-identity.json"
            deadline = time.monotonic() + 8
            while not identity_path.is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(identity_path.is_file())
            second_command = command.copy()
            second_command[second_command.index(str(attempt))] = str(
                self.root / "attempt-two"
            )
            second = subprocess.run(second_command, check=False, timeout=5)
            self.assertEqual(second.returncode, 73)

            owner.kill()
            owner.wait(timeout=5)
            recovery = attempt / "fallback-recovery.json"
            deadline = time.monotonic() + 10
            while (
                (self.pause.exists() or not recovery.is_file())
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            self.assertFalse(self.pause.exists())
            self.assertTrue(recovery.is_file())
            restored = fallback._wait_supervisor_state(
                self.supervisor_state,
                modes={"holding"},
                timeout_seconds=10,
                require_live_holder=True,
            )
            self.assertIsNotNone(restored["holder"])
            self.assertEqual(json.loads(recovery.read_text())["status"], "pass")
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait()

    def test_owner_sigkill_drains_stubborn_setsid_descendant_before_restore(
        self,
    ) -> None:
        inventory_args = self._formal_inventory_args()
        self._start_supervisor()
        run_id = "formal-owner-delayed-drain"
        attempt = self.root / "attempt-delayed"
        child_pid_file = self.root / "stubborn-child.pid"
        payload = (
            "import os,signal,subprocess,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"p=subprocess.Popen([{self.python!r},'-c',"
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(60)'], start_new_session=True); "
            f"open({str(child_pid_file)!r},'w').write(str(p.pid)); "
            "time.sleep(60)"
        )
        command = [
            self.python,
            str(MODULE),
            "formal",
            "--transaction-lock",
            str(self.transaction_lock),
            "--python",
            self.python,
            "--python-sha256",
            self._digest(Path(self.python)),
            "--pause-marker",
            str(self.pause),
            "--release-request-file",
            str(self.release_request),
            "--supervisor-state",
            str(self.supervisor_state),
            "--supervisor-lock",
            str(self.supervisor_lock),
            "--watchdog-log",
            str(self.watchdog_log),
            "--watchdog-identity-file",
            str(self.root / "watchdog-process-identity.json"),
            "--pid-file",
            str(self.pid_file),
            "--holder-state-file",
            str(self.holder_state),
            "--original",
            str(self.original),
            "--original-sha256",
            self._digest(self.original),
            "--holder",
            str(self.holder),
            "--holder-sha256",
            self._digest(self.holder),
            "--watchdog-wrapper",
            str(self.wrapper),
            "--watchdog-wrapper-sha256",
            self._digest(self.wrapper),
            "--supervisor-script",
            str(MODULE),
            "--supervisor-script-sha256",
            self._digest(MODULE),
            "--bootstrap-sha256",
            self._digest(BOOTSTRAP),
            "--poll-seconds",
            "0.02",
            "--restart-delay-seconds",
            "0.02",
            "--stop-timeout-seconds",
            "3",
            "--attempt-dir",
            str(attempt),
            "--run-id",
            run_id,
            "--run-dir",
            str(self.root / "run-delayed"),
            "--bootstrap",
            str(BOOTSTRAP),
            "--cwd",
            str(self.root),
            "--console",
            str(self.root / "delayed.log"),
            *inventory_args,
            "--cleanup-timeout-seconds",
            "1",
            "--",
            self.python,
            "-c",
            payload,
        ]
        owner = subprocess.Popen(command)
        try:
            deadline = time.monotonic() + 8
            while not child_pid_file.is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(child_pid_file.is_file())
            detached_pid = int(child_pid_file.read_text(encoding="utf-8"))
            owner.kill()
            owner.wait(timeout=5)
            time.sleep(0.2)
            self.assertTrue(self.pause.exists())
            state = json.loads(self.supervisor_state.read_text(encoding="utf-8"))
            self.assertEqual(state["mode"], "owner_death_recovery_blocked")
            self.assertFalse(state["recovery_drain"]["safe"])
            restored = fallback._wait_supervisor_state(
                self.supervisor_state,
                modes={"holding"},
                timeout_seconds=8,
                require_live_holder=True,
            )
            self.assertTrue(fallback._identity_alive(restored["holder"]))
            self.assertFalse(self.pause.exists())
            self.assertIsNone(fallback._proc_identity(detached_pid))
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait()


if __name__ == "__main__":
    unittest.main()
