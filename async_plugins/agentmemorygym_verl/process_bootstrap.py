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
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

_WATCHED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_PR_SET_PDEATHSIG = 1


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


def _active_group_members(process_group: int, *, exclude: int) -> list[int]:
    members: list[int] = []
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
            members.append(pid)
    return sorted(members)


def _arm_parent_death_signal() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            f"prctl(PR_SET_PDEATHSIG) failed: {os.strerror(error_number)}",
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
    while True:
        if termination_signal is not None and not forwarded:
            for signum in _WATCHED_SIGNALS:
                signal.signal(signum, signal.SIG_IGN)
            try:
                # This process is the still-live group leader, so its own PGID
                # cannot have been reused when it signals the group it anchors.
                os.killpg(os.getpid(), termination_signal)
            except ProcessLookupError:
                pass
            forwarded = True
            kill_deadline = time.monotonic() + cleanup_timeout_seconds

        return_code = child.poll()
        members = _active_group_members(os.getpid(), exclude=os.getpid())
        if return_code is not None and not members:
            if termination_signal is not None:
                return 128 + termination_signal
            return _render_return_code(return_code)
        if forwarded and time.monotonic() >= kill_deadline:
            # The group is still authenticated by this live leader.  SIGKILL
            # includes the helper itself and therefore cannot leave an anchor.
            os.killpg(os.getpid(), signal.SIGKILL)
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
