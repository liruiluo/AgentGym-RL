#!/usr/bin/env python3
"""Restore the four pinned CAMG held-out evaluator sources from git bundles."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class RestoreError(RuntimeError):
    """Raised when a source bundle or restored checkout is not exact."""


@dataclass(frozen=True)
class SourceSpec:
    label: str
    bundle: Path
    commit: str
    relative_destination: Path


def _run(*arguments: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            list(arguments),
            cwd=cwd,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RestoreError(
            f"command failed ({' '.join(arguments)}): {exc.output.strip()}"
        ) from exc


def _validate_specs(specs: tuple[SourceSpec, ...]) -> None:
    if tuple(spec.label for spec in specs) != (
        "outer",
        "inner",
        "literesearcher_endpoint",
        "verl",
    ):
        raise RestoreError("source specs must use the canonical four-source order")
    destinations: set[str] = set()
    for spec in specs:
        if not spec.bundle.is_absolute() or spec.bundle.is_symlink() or not spec.bundle.is_file():
            raise RestoreError(f"{spec.label} bundle must be an absolute regular file")
        if not _COMMIT.fullmatch(spec.commit):
            raise RestoreError(f"{spec.label} commit must be a full lowercase git commit")
        destination = spec.relative_destination
        if (
            destination.is_absolute()
            or destination == Path(".")
            or ".." in destination.parts
        ):
            raise RestoreError(f"unsafe {spec.label} relative destination")
        key = destination.as_posix()
        if key in destinations:
            raise RestoreError(f"duplicate source destination: {key}")
        destinations.add(key)


def verify_sources(root: Path, specs: tuple[SourceSpec, ...]) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    # Verify nested repositories before the outer superproject so the latter's
    # submodule status is meaningful.
    for spec in (specs[1], specs[0], *specs[2:]):
        checkout = root / spec.relative_destination
        if checkout.is_symlink() or not checkout.is_dir():
            raise RestoreError(f"missing restored {spec.label} checkout: {checkout}")
        observed = _run("git", "rev-parse", "HEAD", cwd=checkout)
        if observed != spec.commit:
            raise RestoreError(
                f"{spec.label} commit mismatch: {observed} != {spec.commit}"
            )
        status = _run(
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            cwd=checkout,
        )
        if status:
            raise RestoreError(f"{spec.label} checkout is dirty: {status}")
        records.append(
            {
                "label": spec.label,
                "path": str(checkout.resolve()),
                "commit": observed,
            }
        )
    return {
        "schema": "camg_heldout_source_restore_v1",
        "status": "pass",
        "root": str(root.resolve()),
        "sources": sorted(records, key=lambda item: item["label"]),
    }


def restore_sources(root: Path, specs: tuple[SourceSpec, ...]) -> dict[str, Any]:
    if not root.is_absolute() or root.is_symlink():
        raise RestoreError("target root must be an absolute non-symlink path")
    _validate_specs(specs)
    if root.exists():
        result = verify_sources(root, specs)
        result["publication"] = "reused_verified"
        return result

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.with_name(f".{root.name}.staging-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise RestoreError(f"source staging path already exists: {staging}")
    staging.mkdir(mode=0o700)
    for spec in specs:
        destination = staging / spec.relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", "--no-checkout", str(spec.bundle), str(destination))
        _run("git", "bundle", "verify", str(spec.bundle), cwd=destination)
        _run("git", "checkout", "--detach", spec.commit, cwd=destination)
    verify_sources(staging, specs)
    os.replace(staging, root)
    result = verify_sources(root, specs)
    result["publication"] = "created"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--outer-bundle", type=Path, required=True)
    parser.add_argument("--outer-commit", required=True)
    parser.add_argument("--inner-bundle", type=Path, required=True)
    parser.add_argument("--inner-commit", required=True)
    parser.add_argument("--literesearcher-bundle", type=Path, required=True)
    parser.add_argument("--literesearcher-commit", required=True)
    parser.add_argument("--verl-bundle", type=Path, required=True)
    parser.add_argument("--verl-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    specs = (
        SourceSpec(
            "outer", arguments.outer_bundle, arguments.outer_commit, Path("AgentGym-RL")
        ),
        SourceSpec(
            "inner",
            arguments.inner_bundle,
            arguments.inner_commit,
            Path("AgentGym-RL/AgentGym"),
        ),
        SourceSpec(
            "literesearcher_endpoint",
            arguments.literesearcher_bundle,
            arguments.literesearcher_commit,
            Path("LiteResearcher-endpoint"),
        ),
        SourceSpec("verl", arguments.verl_bundle, arguments.verl_commit, Path("verl")),
    )
    result = restore_sources(arguments.target_root, specs)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RestoreError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}), file=sys.stderr)
        raise SystemExit(2)
