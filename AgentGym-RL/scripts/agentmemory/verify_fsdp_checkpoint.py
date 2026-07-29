#!/usr/bin/env python3
"""Verify that a VERL FSDP actor/critic checkpoint is complete."""

from __future__ import annotations

import argparse
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROLES = ("actor", "critic")
SHARD_STEMS = ("model", "optim", "extra_state")


class CheckpointVerificationError(RuntimeError):
    pass


def _regular_nonempty_file(path: Path, *, label: str) -> int:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise CheckpointVerificationError(f"missing {label}: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise CheckpointVerificationError(f"{label} is not a regular file: {path}")
    if metadata.st_size <= 0:
        raise CheckpointVerificationError(f"empty {label}: {path}")
    return metadata.st_size


def _verify_shard_family(
    role_directory: Path,
    *,
    stem: str,
    world_size: int,
) -> dict[str, Any]:
    expected_names = {
        f"{stem}_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)
    }
    matching_paths = list(role_directory.glob(f"{stem}_world_size_*_rank_*.pt"))
    matching_names = {path.name for path in matching_paths}
    missing = sorted(expected_names - matching_names)
    unexpected = sorted(matching_names - expected_names)
    if missing:
        raise CheckpointVerificationError(
            f"missing {role_directory.name}/{stem} shards: {', '.join(missing)}"
        )
    if unexpected:
        raise CheckpointVerificationError(
            f"unexpected {role_directory.name}/{stem} shards: {', '.join(unexpected)}"
        )

    rank_pattern = re.compile(
        rf"^{re.escape(stem)}_world_size_{world_size}_rank_(\d+)\.pt$"
    )
    ranked_paths: list[tuple[int, Path]] = []
    for path in matching_paths:
        match = rank_pattern.fullmatch(path.name)
        if match is None:
            raise CheckpointVerificationError(f"invalid shard name: {path}")
        ranked_paths.append((int(match.group(1)), path))
    ranked_paths.sort()

    ranks: list[int] = []
    total_bytes = 0
    for rank, path in ranked_paths:
        ranks.append(rank)
        total_bytes += _regular_nonempty_file(path, label="checkpoint shard")
    expected_ranks = list(range(world_size))
    if ranks != expected_ranks:
        raise CheckpointVerificationError(
            f"wrong {role_directory.name}/{stem} rank set: {ranks} != {expected_ranks}"
        )
    return {"files": len(ranks), "ranks": ranks, "bytes": total_bytes}


def verify_checkpoint(
    checkpoint_root: str | Path,
    *,
    step: int,
    world_size: int,
    roles: Iterable[str] = DEFAULT_ROLES,
) -> dict[str, Any]:
    root = Path(checkpoint_root).expanduser().resolve()
    if step < 0:
        raise CheckpointVerificationError(f"step must be non-negative: {step}")
    if world_size <= 0:
        raise CheckpointVerificationError(
            f"world size must be positive: {world_size}"
        )
    if not root.is_dir():
        raise CheckpointVerificationError(f"checkpoint root does not exist: {root}")

    tracker_path = root / "latest_checkpointed_iteration.txt"
    _regular_nonempty_file(tracker_path, label="checkpoint tracker")
    tracker = tracker_path.read_text(encoding="utf-8").strip()
    if tracker != str(step):
        raise CheckpointVerificationError(
            f"checkpoint tracker mismatch: {tracker!r} != {step!r}"
        )

    step_directory = root / f"global_step_{step}"
    if not step_directory.is_dir():
        raise CheckpointVerificationError(
            f"global-step directory does not exist: {step_directory}"
        )
    data_bytes = _regular_nonempty_file(
        step_directory / "data.pt", label="checkpoint data"
    )

    role_inventory: dict[str, Any] = {}
    normalized_roles = tuple(roles)
    if not normalized_roles:
        raise CheckpointVerificationError("at least one checkpoint role is required")
    for role in normalized_roles:
        role_directory = step_directory / role
        if not role_directory.is_dir():
            raise CheckpointVerificationError(
                f"checkpoint role directory does not exist: {role_directory}"
            )
        role_inventory[role] = {
            stem: _verify_shard_family(
                role_directory, stem=stem, world_size=world_size
            )
            for stem in SHARD_STEMS
        }

    return {
        "status": "pass",
        "checkpoint_root": str(root),
        "step": step,
        "world_size": world_size,
        "tracker": int(tracker),
        "global_step_dir": str(step_directory),
        "data_pt_bytes": data_bytes,
        "roles": role_inventory,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--role", action="append", dest="roles")
    arguments = parser.parse_args(argv)
    payload = verify_checkpoint(
        arguments.checkpoint_root,
        step=arguments.step,
        world_size=arguments.world_size,
        roles=arguments.roles or DEFAULT_ROLES,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CheckpointVerificationError, OSError, UnicodeError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
