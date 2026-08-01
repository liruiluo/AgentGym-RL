#!/usr/bin/env python3
"""Seed and attest the stable Triton cache used by Qwen3.5 PPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


AUTOTUNE_SUFFIX = ".autotune.json"
DEFAULT_REQUIRED_KERNELS = (
    "l2norm_fwd_kernel",
    "l2norm_bwd_kernel",
    "layer_norm_gated_fwd_kernel",
    "layer_norm_gated_bwd_kernel",
)


class CacheToolError(RuntimeError):
    pass


def _cache_root(path: str | os.PathLike[str], *, must_exist: bool) -> Path:
    root = Path(path).expanduser().resolve()
    if must_exist and not root.is_dir():
        raise CacheToolError(f"cache directory does not exist: {root}")
    if root.exists() and not root.is_dir():
        raise CacheToolError(f"cache path is not a directory: {root}")
    return root


def _canonical_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _autotune_records(
    root: Path,
) -> tuple[dict[tuple[str, str, str], list[str]], list[str], Counter[str]]:
    records: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    invalid: list[str] = []
    kernel_counts: Counter[str] = Counter()
    for path in sorted(root.rglob(f"*{AUTOTUNE_SUFFIX}")):
        relative = str(path.relative_to(root))
        variant = str(path.parent.relative_to(root))
        kernel = path.name[: -len(AUTOTUNE_SUFFIX)]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = _canonical_key(payload["key"])
            timings = payload["configs_timings"]
            if not isinstance(timings, list) or not timings:
                raise ValueError("configs_timings must be a non-empty list")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid.append(relative)
            continue
        # The parent directory is Triton's compiled-source hash. Different
        # hashes may legitimately use the same function name and autotune key;
        # collapsing them makes a required variant JIT again on every run.
        records[(variant, kernel, key)].append(relative)
        kernel_counts[kernel] += 1
    return records, invalid, kernel_counts


def inventory_cache(path: str | os.PathLike[str]) -> dict[str, Any]:
    root = _cache_root(path, must_exist=True)
    records, invalid, kernel_counts = _autotune_records(root)
    duplicates = [
        {
            "variant": variant,
            "kernel": kernel,
            "key": json.loads(key),
            "paths": paths,
        }
        for (variant, kernel, key), paths in sorted(records.items())
        if len(paths) > 1
    ]
    function_records: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (_, kernel, key), paths in records.items():
        function_records[(kernel, key)].extend(paths)
    cross_variant_keys = [
        {
            "kernel": kernel,
            "key": json.loads(key),
            "paths": sorted(paths),
        }
        for (kernel, key), paths in sorted(function_records.items())
        if len(paths) > 1
    ]
    files = [entry for entry in root.rglob("*") if entry.is_file()]
    return {
        "cache_dir": str(root),
        "files": len(files),
        "bytes": sum(entry.stat().st_size for entry in files),
        "autotune_files": sum(kernel_counts.values()),
        "unique_function_keys": len(function_records),
        "unique_variant_keys": len(records),
        "kernel_counts": dict(sorted(kernel_counts.items())),
        "invalid_autotune_files": invalid,
        "cross_variant_function_keys": cross_variant_keys,
        "duplicate_function_keys": duplicates,
    }


def assert_prewarmer_ready(
    inventory: dict[str, Any],
    *,
    required_kernels: Iterable[str] = DEFAULT_REQUIRED_KERNELS,
    min_autotune_files: int = 1,
) -> dict[str, Any]:
    if inventory["invalid_autotune_files"]:
        raise CacheToolError(
            "invalid autotune files: "
            + ", ".join(inventory["invalid_autotune_files"][:8])
        )
    if inventory["duplicate_function_keys"]:
        raise CacheToolError(
            "duplicate (variant, kernel, key) autotune records: "
            f"{len(inventory['duplicate_function_keys'])}"
        )
    missing = sorted(
        set(required_kernels) - set(inventory["kernel_counts"])
    )
    if missing:
        raise CacheToolError("missing kernels: " + ", ".join(missing))
    if inventory["autotune_files"] < min_autotune_files:
        raise CacheToolError(
            "insufficient autotune files: "
            f"{inventory['autotune_files']} < {min_autotune_files}"
        )
    return inventory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_match(source: Path, destination: Path) -> bool:
    return (
        destination.is_file()
        and not destination.is_symlink()
        and source.stat().st_size == destination.stat().st_size
        and _sha256(source) == _sha256(destination)
    )


def seed_cache(
    source_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
) -> dict[str, Any]:
    source = _cache_root(source_path, must_exist=True)
    destination = _cache_root(destination_path, must_exist=False)
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise CacheToolError("source and destination cache trees must be disjoint")
    destination.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    copied_bytes = 0
    reused_files = 0
    for source_file in sorted(source.rglob("*")):
        if source_file.is_dir():
            continue
        if source_file.is_symlink() or not source_file.is_file():
            raise CacheToolError(f"unsupported cache entry: {source_file}")
        relative = source_file.relative_to(source)
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        if destination_file.exists():
            if not _files_match(source_file, destination_file):
                raise CacheToolError(f"cache content conflict: {relative}")
            reused_files += 1
            continue

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_file.name}.",
            dir=destination_file.parent,
        )
        os.close(descriptor)
        temporary_file = Path(temporary_name)
        try:
            shutil.copy2(source_file, temporary_file)
            try:
                os.link(temporary_file, destination_file)
                copied_files += 1
                copied_bytes += source_file.stat().st_size
            except FileExistsError:
                if not _files_match(source_file, destination_file):
                    raise CacheToolError(f"cache content conflict: {relative}")
                reused_files += 1
        finally:
            temporary_file.unlink(missing_ok=True)

    return {
        "source_dir": str(source),
        "destination_dir": str(destination),
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "reused_files": reused_files,
    }


def verify_reference_coverage(
    reference_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
) -> dict[str, Any]:
    reference = _cache_root(reference_path, must_exist=True)
    cache = _cache_root(cache_path, must_exist=True)
    reference_records, reference_invalid, _ = _autotune_records(reference)
    cache_records, cache_invalid, _ = _autotune_records(cache)
    if reference_invalid:
        raise CacheToolError("reference cache has invalid autotune files")
    if cache_invalid:
        raise CacheToolError("destination cache has invalid autotune files")
    if any(len(paths) > 1 for paths in reference_records.values()):
        raise CacheToolError("reference cache has duplicate variant keys")
    if any(len(paths) > 1 for paths in cache_records.values()):
        raise CacheToolError("destination cache has duplicate variant keys")
    missing = sorted(set(reference_records) - set(cache_records))
    return {
        "reference_dir": str(reference),
        "cache_dir": str(cache),
        "reference_function_keys": len(reference_records),
        "cache_function_keys": len(cache_records),
        "missing_function_keys": [
            {
                "variant": variant,
                "kernel": kernel,
                "key": json.loads(key),
            }
            for variant, kernel, key in missing
        ],
    }


def _required_kernels(arguments: argparse.Namespace) -> tuple[str, ...]:
    return tuple(arguments.require_kernel or DEFAULT_REQUIRED_KERNELS)


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--cache-dir", required=True)
    inventory_parser.add_argument("--require-kernel", action="append")
    inventory_parser.add_argument("--min-autotune-files", type=int, default=4)

    seed_parser = subparsers.add_parser("seed")
    seed_parser.add_argument("--source", required=True)
    seed_parser.add_argument("--destination", required=True)
    seed_parser.add_argument("--require-kernel", action="append")
    seed_parser.add_argument("--min-autotune-files", type=int, default=4)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--cache-dir", required=True)
    verify_parser.add_argument("--reference-cache")
    verify_parser.add_argument("--require-kernel", action="append")
    verify_parser.add_argument("--min-autotune-files", type=int, default=4)

    arguments = parser.parse_args(argv)
    if arguments.command == "inventory":
        inventory = inventory_cache(arguments.cache_dir)
        assert_prewarmer_ready(
            inventory,
            required_kernels=_required_kernels(arguments),
            min_autotune_files=arguments.min_autotune_files,
        )
        _print(inventory)
        return 0

    if arguments.command == "seed":
        source_inventory = inventory_cache(arguments.source)
        assert_prewarmer_ready(
            source_inventory,
            required_kernels=_required_kernels(arguments),
            min_autotune_files=arguments.min_autotune_files,
        )
        seed = seed_cache(arguments.source, arguments.destination)
        inventory = inventory_cache(arguments.destination)
        assert_prewarmer_ready(
            inventory,
            required_kernels=_required_kernels(arguments),
            min_autotune_files=arguments.min_autotune_files,
        )
        coverage = verify_reference_coverage(
            arguments.source, arguments.destination
        )
        if coverage["missing_function_keys"]:
            raise CacheToolError("destination cache misses reference keys")
        _print(
            {
                "source_inventory": source_inventory,
                "seed": seed,
                "inventory": inventory,
                "coverage": coverage,
            }
        )
        return 0

    inventory = inventory_cache(arguments.cache_dir)
    assert_prewarmer_ready(
        inventory,
        required_kernels=_required_kernels(arguments),
        min_autotune_files=arguments.min_autotune_files,
    )
    payload: dict[str, Any] = {"inventory": inventory}
    if arguments.reference_cache:
        coverage = verify_reference_coverage(
            arguments.reference_cache, arguments.cache_dir
        )
        if coverage["missing_function_keys"]:
            raise CacheToolError("destination cache misses reference keys")
        payload["coverage"] = coverage
    _print(payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CacheToolError as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
