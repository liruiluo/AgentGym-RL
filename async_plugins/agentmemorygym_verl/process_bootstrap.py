# ruff: noqa: BLE001
"""Signal-safe process-group anchor for the multitask launch orchestrator.

The parent starts this helper in a new session while SIGINT and SIGTERM are
blocked.  The helper cannot start the requested command until the parent has
recorded an exact PID/start-ticks lease and acknowledges a one-byte pipe.
After release it remains the process-group leader until the command and every
same-group descendant have exited.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

_WATCHED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_PR_SET_PDEATHSIG = 1
_PR_SET_CHILD_SUBREAPER = 36
_PIDFD_SEND_SIGNAL_SYSCALL = 424
_PIDFD_OPEN_SYSCALL = 434

_ZOMBIE_STATES = {"Z", "X", "x"}


def _process_start_ticks(pid: int) -> str | None:
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


def _active_group_members(
    process_group: int, *, exclude: int
) -> list[tuple[int, str]]:
    members: list[tuple[int, str]] = []
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            fields = (
                (candidate / "stat")
                .read_text(encoding="utf-8")
                .rsplit(")", 1)[1]
                .split()
            )
        except (FileNotFoundError, IndexError, OSError):
            continue
        pid = int(candidate.name)
        if (
            pid != exclude
            and fields[0] not in {"Z", "X", "x"}
            and int(fields[2]) == process_group
        ):
            members.append((pid, fields[19]))
    return sorted(members)


def _process_identity(pid: int) -> tuple[int, str, int, str] | None:
    """Return ``(pid, start_ticks, ppid, state)`` from one proc snapshot."""

    try:
        fields = (
            Path(f"/proc/{pid}/stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .split()
        )
        return pid, fields[19], int(fields[1]), fields[0]
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def _active_descendants(root_pid: int) -> list[tuple[int, str]]:
    """Snapshot every live descendant, including processes that called setsid."""

    snapshots: dict[int, tuple[int, str, int, str]] = {}
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        identity = _process_identity(int(candidate.name))
        if identity is not None:
            snapshots[identity[0]] = identity
    descendants: dict[int, tuple[int, str]] = {}
    frontier = {root_pid}
    while frontier:
        parents = set(frontier)
        frontier.clear()
        for pid, start_ticks, ppid, state in snapshots.values():
            if pid in descendants or ppid not in parents:
                continue
            if state not in _ZOMBIE_STATES:
                descendants[pid] = (pid, start_ticks)
                frontier.add(pid)
    return sorted(descendants.values())


def _reap_adopted_children(*, exclude: int) -> None:
    """Reap dead grandchildren adopted through the subreaper contract."""

    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        pid = int(candidate.name)
        if pid == exclude:
            continue
        identity = _process_identity(pid)
        if identity is None or identity[2] != os.getpid() or identity[3] != "Z":
            continue
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass


def _pidfd_open_exact(pid: int) -> int:
    pidfd_open = getattr(os, "pidfd_open", None)
    if callable(pidfd_open):
        return int(pidfd_open(pid, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    descriptor = int(syscall(_PIDFD_OPEN_SYSCALL, pid, 0))
    if descriptor >= 0:
        return descriptor
    error_number = ctypes.get_errno()
    if error_number == errno.ESRCH:
        raise ProcessLookupError(pid)
    raise OSError(error_number, os.strerror(error_number))


def _pidfd_send_signal_exact(descriptor: int, signum: int) -> None:
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if callable(pidfd_send_signal):
        pidfd_send_signal(descriptor, signum, None, 0)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    result = int(
        syscall(
            _PIDFD_SEND_SIGNAL_SYSCALL,
            descriptor,
            signum,
            ctypes.c_void_p(0),
            0,
        )
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.ESRCH:
        raise ProcessLookupError
    raise OSError(error_number, os.strerror(error_number))


def _signal_group_member_exact(identity: tuple[int, str], signum: int) -> bool:
    pid, start_ticks = identity
    if _process_start_ticks(pid) != start_ticks:
        return False
    descriptor = -1
    try:
        descriptor = _pidfd_open_exact(pid)
        if _process_start_ticks(pid) != start_ticks:
            return False
        _pidfd_send_signal_exact(descriptor, signum)
    except ProcessLookupError:
        return False
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return True


def _arm_parent_death_signal() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            f"prctl(PR_SET_PDEATHSIG) failed: {os.strerror(error_number)}",
        )


def _arm_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            f"prctl(PR_SET_CHILD_SUBREAPER) failed: {os.strerror(error_number)}",
        )


def _render_return_code(return_code: int) -> int:
    if return_code >= 0:
        return min(return_code, 255)
    return min(128 + abs(return_code), 255)


def run(
    *,
    ack_fd: int,
    parent_pid: int,
    parent_start_ticks: str,
    cleanup_timeout_seconds: float,
    command: Sequence[str],
) -> int:
    if not sys.platform.startswith("linux") or not Path("/proc/self/stat").is_file():
        raise RuntimeError("process bootstrap requires Linux /proc")
    if os.getpgrp() != os.getpid():
        raise RuntimeError("process bootstrap must be its own process-group leader")
    if not command:
        raise RuntimeError("process bootstrap received an empty command")

    termination_signal: int | None = None

    def request_termination(signum: int, _frame: object) -> None:
        nonlocal termination_signal
        if termination_signal is None:
            termination_signal = signum

    for signum in _WATCHED_SIGNALS:
        signal.signal(signum, request_termination)
    _arm_parent_death_signal()
    # cgroup delegation is unavailable in the 9N containers.  Becoming a
    # subreaper keeps daemonized/setsid descendants attached to this exact,
    # lease-bound root so cleanup is not limited to the original process group.
    _arm_child_subreaper()

    try:
        acknowledged = os.read(ack_fd, 1)
    finally:
        os.close(ack_fd)
    if acknowledged != b"1":
        return 125

    signal.pthread_sigmask(signal.SIG_UNBLOCK, _WATCHED_SIGNALS)
    if _process_start_ticks(parent_pid) != parent_start_ticks:
        termination_signal = signal.SIGTERM
    if termination_signal is not None:
        return 128 + termination_signal

    child = subprocess.Popen(command, start_new_session=False)
    forwarded = False
    kill_deadline = 0.0
    next_kill_retry = 0.0
    while True:
        if termination_signal is not None and not forwarded:
            for signum in _WATCHED_SIGNALS:
                signal.signal(signum, signal.SIG_IGN)
            forwarded = True
            kill_deadline = time.monotonic() + cleanup_timeout_seconds

        return_code = child.poll()
        _reap_adopted_children(exclude=child.pid)
        descendants = _active_descendants(os.getpid())
        if return_code is not None and not descendants:
            if termination_signal is not None:
                return 128 + termination_signal
            return _render_return_code(return_code)
        if forwarded:
            now = time.monotonic()
            descendant_signal = (
                signal.SIGTERM if now < kill_deadline else signal.SIGKILL
            )
            if now >= next_kill_retry:
                for descendant in descendants:
                    _signal_group_member_exact(descendant, descendant_signal)
                next_kill_retry = now + 0.25
        time.sleep(0.02)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ack-fd", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-start-ticks", required=True)
    parser.add_argument("--cleanup-timeout-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run(
            ack_fd=args.ack_fd,
            parent_pid=args.parent_pid,
            parent_start_ticks=args.parent_start_ticks,
            cleanup_timeout_seconds=args.cleanup_timeout_seconds,
            command=args.command,
        )
    except Exception as exc:
        print(f"multitask process bootstrap failed closed: {exc}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
