#!/usr/bin/env python3
"""Thin signal-safe entrypoints around the reviewed OpenMLE-fast services."""

from __future__ import annotations

import argparse
import signal
import sys
from typing import Optional, Sequence

from agentenv_openmle_fast import launch


def run_private_grader() -> int:
    service = launch.build_private_grader_from_environment()

    def stop_service(_signum, _frame) -> None:
        service.shutdown()

    signal.signal(signal.SIGINT, stop_service)
    signal.signal(signal.SIGTERM, stop_service)
    try:
        service.serve_forever()
    finally:
        service.shutdown()
    return 0


def run_public_endpoint() -> int:
    launch.launch()
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("private", "public"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.service == "private":
        return run_private_grader()
    return run_public_endpoint()


if __name__ == "__main__":
    sys.exit(main())
