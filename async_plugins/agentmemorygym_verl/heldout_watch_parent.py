"""Independent parent-death cleanup guard for CAMG held-out evaluation.

The guard deliberately does not use the orchestrator's process bootstrap: it
must survive an abrupt orchestrator death long enough to stop processes that
inherit the exact attempt identity.  Normal shutdown sends SIGTERM to the
guard, in which case it records a clean signal exit and performs no cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _start_ticks(pid: int) -> str | None:
    try:
        fields = (
            Path(f"/proc/{pid}/stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .split()
        )
        if fields[0] in {"Z", "X", "x"}:
            return None
        return fields[19]
    except (FileNotFoundError, IndexError, OSError):
        return None


def _alive(pid: int, ticks: str) -> bool:
    return pid > 1 and bool(ticks) and _start_ticks(pid) == str(ticks)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _signal_exact(pid: int, ticks: str, signum: int) -> bool:
    if not _alive(pid, ticks):
        return False
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if not callable(pidfd_open) or not callable(pidfd_send_signal):
        raise RuntimeError("held-out watch-parent requires pidfd signalling")
    descriptor = pidfd_open(pid, 0)
    try:
        if not _alive(pid, ticks):
            return False
        pidfd_send_signal(descriptor, signum, None, 0)
    except ProcessLookupError:
        return False
    finally:
        os.close(descriptor)
    return True


def _owned_processes(owner_id: str) -> list[tuple[int, str]]:
    needles = {
        f"AGENTMEMORY_ATTEMPT_ID={owner_id}",
        f"AMG_MULTITASK_RUN_ID={owner_id}",
    }
    result: list[tuple[int, str]] = []
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        pid = int(candidate.name)
        if pid == os.getpid():
            continue
        try:
            environment = {
                value.decode("utf-8", "replace")
                for value in (candidate / "environ").read_bytes().split(b"\0")
                if value
            }
        except OSError:
            continue
        if not (needles & environment):
            continue
        ticks = _start_ticks(pid)
        if ticks:
            result.append((pid, ticks))
    return sorted(result)


def _terminate_owned(owner_id: str, timeout_seconds: float) -> dict[str, Any]:
    initial = _owned_processes(owner_id)
    signalled_term = [
        [pid, ticks]
        for pid, ticks in initial
        if _signal_exact(pid, ticks, signal.SIGTERM)
    ]
    deadline = time.monotonic() + timeout_seconds
    remaining = _owned_processes(owner_id)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = _owned_processes(owner_id)
    signalled_kill: list[list[Any]] = []
    for pid, ticks in remaining:
        if _signal_exact(pid, ticks, signal.SIGKILL):
            signalled_kill.append([pid, ticks])
    deadline = time.monotonic() + 5.0
    remaining = _owned_processes(owner_id)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = _owned_processes(owner_id)
    return {
        "initial": [[pid, ticks] for pid, ticks in initial],
        "sigterm": signalled_term,
        "sigkill": signalled_kill,
        "remaining": [[pid, ticks] for pid, ticks in remaining],
    }


def _mounts_below(roots: list[Path]) -> list[Path]:
    rendered = [str(root.resolve()) for root in roots]
    findings: list[Path] = []
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        value = fields[4]
        for encoded, decoded in (
            ("\\040", " "),
            ("\\011", "\t"),
            ("\\012", "\n"),
            ("\\134", "\\"),
        ):
            value = value.replace(encoded, decoded)
        if any(value == root or value.startswith(root + os.sep) for root in rendered):
            findings.append(Path(value))
    return sorted(set(findings), key=lambda item: len(str(item)), reverse=True)


def _unmount_roots(roots: list[Path]) -> dict[str, Any]:
    attempted: list[dict[str, Any]] = []
    for mount in _mounts_below(roots):
        completed = subprocess.run(
            ["/bin/umount", str(mount)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        attempted.append(
            {
                "path": str(mount),
                "returncode": completed.returncode,
                "output": completed.stdout[-2000:],
            }
        )
    return {
        "attempted": attempted,
        "remaining": [str(path) for path in _mounts_below(roots)],
    }


def run(args: argparse.Namespace) -> int:
    if not Path("/proc/self/stat").is_file():
        raise RuntimeError("held-out watch-parent requires Linux /proc")
    owner_id = str(args.owner_id)
    roots = [Path(value) for value in args.cleanup_root]
    if any(not root.is_absolute() for root in roots):
        raise RuntimeError("watch-parent cleanup roots must be absolute")
    stopped = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    watcher_ticks = _start_ticks(os.getpid())
    _atomic_json(
        args.ready,
        {
            "schema": "camg_heldout_watch_parent_start_v1",
            "status": "ready",
            "pid": os.getpid(),
            "start_ticks": watcher_ticks,
            "parent_pid": args.parent_pid,
            "parent_start_ticks": args.parent_start_ticks,
            "owner_id": owner_id,
            "cleanup_roots": [str(root) for root in roots],
        },
    )
    while not stopped and _alive(args.parent_pid, args.parent_start_ticks):
        time.sleep(args.poll_seconds)
    if stopped:
        _atomic_json(
            args.receipt,
            {
                "schema": "camg_heldout_watch_parent_exit_v1",
                "status": "pass",
                "mode": "signal",
                "owner_id": owner_id,
                "pid": os.getpid(),
                "start_ticks": watcher_ticks,
            },
        )
        return 0

    processes = _terminate_owned(owner_id, args.term_timeout_seconds)
    mounts = _unmount_roots(roots)
    status = "pass" if not processes["remaining"] and not mounts["remaining"] else "fail"
    _atomic_json(
        args.receipt,
        {
            "schema": "camg_heldout_watch_parent_exit_v1",
            "status": status,
            "mode": "parent_death",
            "owner_id": owner_id,
            "pid": os.getpid(),
            "start_ticks": watcher_ticks,
            "process_cleanup": processes,
            "mount_cleanup": mounts,
        },
    )
    return 0 if status == "pass" else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-start-ticks", required=True)
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cleanup-root", type=Path, action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--term-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    try:
        return run(_parse_args())
    except Exception as exc:
        print(f"held-out watch-parent failed closed: {exc}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
