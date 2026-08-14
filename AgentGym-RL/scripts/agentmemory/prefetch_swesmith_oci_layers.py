#!/usr/bin/env python3
"""Prefetch digest-pinned SWE-smith OCI layers with resumable HTTP transfers."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prepare_swesmith_oci_rootfs import ImageBinding, parse_binding

_DIGEST_PREFIX = "sha256:"
_EVIDENCE_SCHEMA = "swesmith_oci_resumable_layer_prefetch_v1"
_VERIFIED_BLOB_FILES: set[tuple[str, int, int, int]] = set()


@dataclass(frozen=True)
class LayerDescriptor:
    digest: str
    size: int


@dataclass(frozen=True)
class LayerPlan:
    descriptor: LayerDescriptor
    source_images: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, raw: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _transport_prefixes(primary: str, fallbacks: Iterable[str]) -> tuple[str, ...]:
    prefixes: list[str] = []
    for raw in (primary, *fallbacks):
        prefix = raw.strip().rstrip("/")
        if not prefix or "://" in prefix or "/" in prefix:
            raise ValueError(f"invalid registry hostname: {raw!r}")
        if prefix not in prefixes:
            prefixes.append(prefix)
    return tuple(prefixes)


def parse_manifest_layers(raw: bytes, binding: ImageBinding) -> tuple[LayerDescriptor, ...]:
    if f"{_DIGEST_PREFIX}{hashlib.sha256(raw).hexdigest()}" != binding.digest:
        raise RuntimeError(f"manifest bytes do not match {binding.digest}")
    manifest = json.loads(raw)
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise RuntimeError(f"manifest has no layer descriptors: {binding.source_image}")

    descriptors: list[LayerDescriptor] = []
    sizes: dict[str, int] = {}
    for raw_descriptor in layers:
        if not isinstance(raw_descriptor, dict):
            raise TypeError("OCI layer descriptor must be an object")
        digest = raw_descriptor.get("digest")
        size = raw_descriptor.get("size")
        if (
            not isinstance(digest, str)
            or not digest.startswith(_DIGEST_PREFIX)
            or len(digest) != len(_DIGEST_PREFIX) + 64
            or any(character not in "0123456789abcdef" for character in digest[7:])
        ):
            raise RuntimeError(f"invalid OCI layer digest: {digest!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise RuntimeError(f"invalid OCI layer size for {digest}: {size!r}")
        previous = sizes.setdefault(digest, size)
        if previous != size:
            raise RuntimeError(f"conflicting sizes for OCI layer {digest}")
        if previous == size and any(item.digest == digest for item in descriptors):
            continue
        descriptors.append(LayerDescriptor(digest=digest, size=size))
    return tuple(descriptors)


def build_layer_plan(
    bindings: tuple[ImageBinding, ...],
    manifests: dict[str, bytes],
) -> tuple[dict[str, LayerPlan], dict[str, tuple[LayerDescriptor, ...]]]:
    sources: dict[str, list[str]] = {}
    descriptors: dict[str, LayerDescriptor] = {}
    image_layers: dict[str, tuple[LayerDescriptor, ...]] = {}
    for binding in bindings:
        layers = parse_manifest_layers(manifests[binding.digest], binding)
        image_layers[binding.digest] = layers
        for layer in layers:
            previous = descriptors.setdefault(layer.digest, layer)
            if previous.size != layer.size:
                raise RuntimeError(f"conflicting sizes for shared layer {layer.digest}")
            repositories = sources.setdefault(layer.digest, [])
            if binding.source_image not in repositories:
                repositories.append(binding.source_image)
    plan = {
        digest: LayerPlan(descriptor=descriptors[digest], source_images=tuple(sources[digest]))
        for digest in sorted(descriptors)
    }
    return plan, image_layers


def _is_valid_blob(path: Path, descriptor: LayerDescriptor) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    if stat.st_size != descriptor.size:
        return False
    identity = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity in _VERIFIED_BLOB_FILES:
        return True
    if _sha256_file(path) != descriptor.digest.removeprefix(_DIGEST_PREFIX):
        return False
    _VERIFIED_BLOB_FILES.add(identity)
    return True


def _has_expected_size(path: Path, descriptor: LayerDescriptor) -> bool:
    return path.is_file() and path.stat().st_size == descriptor.size


def _install_verified_blob(source: Path, target: Path, descriptor: LayerDescriptor) -> str:
    if not _is_valid_blob(source, descriptor):
        raise RuntimeError(f"source layer is not digest-valid: {source}")
    if _is_valid_blob(target, descriptor):
        source_stat = source.stat()
        target_stat = target.stat()
        if source_stat.st_dev == target_stat.st_dev and source_stat.st_ino != target_stat.st_ino:
            temporary = target.with_name(f".{target.name}.relink-{os.getpid()}")
            temporary.unlink(missing_ok=True)
            try:
                os.link(source, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            relinked = target.stat()
            _VERIFIED_BLOB_FILES.add(
                (str(target), relinked.st_ino, relinked.st_size, relinked.st_mtime_ns)
            )
            return "relinked"
        return "reused"
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = target.with_name(f".{target.name}.install-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
            method = "hardlink"
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, temporary)
            method = "copy"
        if not _is_valid_blob(temporary, descriptor):
            raise RuntimeError(f"installed layer failed digest verification: {temporary}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return method


def _run_command(
    argv: list[str],
    *,
    environment: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        env=environment,
        timeout=timeout,
    )


def _fetch_manifest(
    binding: ImageBinding,
    *,
    crane: Path,
    prefixes: tuple[str, ...],
    environment: dict[str, str],
    manifest_cache_root: Path,
) -> tuple[bytes, dict[str, Any]]:
    target = manifest_cache_root / f"{binding.cache_name}.json"
    if target.is_file():
        raw = target.read_bytes()
        parse_manifest_layers(raw, binding)
        return raw, {"status": "reused", "path": str(target)}

    failures: list[str] = []
    for prefix in prefixes:
        reference = f"{prefix}/{binding.source_image}@{binding.digest}"
        try:
            completed = _run_command(
                [str(crane), "manifest", reference],
                environment=environment,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{prefix}: timeout")
            continue
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace")[-500:]
            failures.append(f"{prefix}: rc={completed.returncode} {error}")
            continue
        try:
            parse_manifest_layers(completed.stdout, binding)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{prefix}: {exc}")
            continue
        _atomic_write(target, completed.stdout)
        return completed.stdout, {
            "status": "downloaded",
            "path": str(target),
            "transport": prefix,
        }
    raise RuntimeError(
        f"failed to fetch manifest for {binding.source_image}: " + " | ".join(failures)
    )


def _authorization_header(
    *,
    crane: Path,
    prefix: str,
    source_image: str,
    environment: dict[str, str],
) -> str:
    completed = _run_command(
        [str(crane), "auth", "token", "-H", f"{prefix}/{source_image}"],
        environment=environment,
        timeout=45,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"registry token request failed: rc={completed.returncode} {error}")
    header = completed.stdout.decode("utf-8", errors="strict").strip()
    if (
        not header.lower().startswith("authorization: bearer ")
        or "\n" in header
        or "\r" in header
        or '"' in header
        or "\\" in header
    ):
        raise RuntimeError("registry token helper returned an unsafe header")
    return header


def _download_layer(
    plan: LayerPlan,
    *,
    shared_cache_root: Path,
    crane: Path,
    curl: Path,
    prefixes: tuple[str, ...],
    environment: dict[str, str],
    attempts: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    descriptor = plan.descriptor
    target = shared_cache_root / descriptor.digest
    partial_root = shared_cache_root / ".partials"
    lock_root = shared_cache_root / ".locks"
    partial = partial_root / f"{descriptor.digest}.partial"
    lock_path = lock_root / f"{descriptor.digest}.lock"
    partial_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if _is_valid_blob(target, descriptor):
            return {"status": "reused", "bytes": descriptor.size, "attempts": 0}
        if target.exists():
            target.unlink()
        if partial.exists() and partial.stat().st_size > descriptor.size:
            partial.unlink()
        if _is_valid_blob(partial, descriptor):
            os.replace(partial, target)
            return {"status": "resumed_complete", "bytes": descriptor.size, "attempts": 0}
        if partial.exists() and partial.stat().st_size == descriptor.size:
            partial.unlink()

        records: list[dict[str, Any]] = []
        choices = [
            (prefix, source_image)
            for source_image in plan.source_images
            for prefix in prefixes
        ]
        transfer_attempt = 0
        selection_attempt = 0
        while transfer_attempt < attempts and selection_attempt < attempts * len(choices):
            prefix, source_image = choices[selection_attempt % len(choices)]
            selection_attempt += 1
            before = partial.stat().st_size if partial.exists() else 0
            started = time.monotonic()
            try:
                header = _authorization_header(
                    crane=crane,
                    prefix=prefix,
                    source_image=source_image,
                    environment=environment,
                )
                transfer_attempt += 1
                descriptor_fd, descriptor_name = tempfile.mkstemp(
                    prefix=".curl-auth-", dir=partial_root
                )
                auth_config = Path(descriptor_name)
                try:
                    os.fchmod(descriptor_fd, 0o600)
                    with os.fdopen(descriptor_fd, "w", encoding="utf-8") as handle:
                        handle.write(f'header = "{header}"\n')
                    url = (
                        f"https://{prefix}/v2/{source_image}/blobs/"
                        f"{descriptor.digest}"
                    )
                    completed = _run_command(
                        [
                            str(curl),
                            "--config",
                            str(auth_config),
                            "--silent",
                            "--show-error",
                            "--fail",
                            "--location",
                            "--continue-at",
                            "-",
                            "--connect-timeout",
                            "15",
                            "--max-time",
                            str(timeout_seconds),
                            "--speed-time",
                            "30",
                            "--speed-limit",
                            "1024",
                            "--output",
                            str(partial),
                            url,
                        ],
                        environment=environment,
                        timeout=timeout_seconds + 30,
                    )
                finally:
                    auth_config.unlink(missing_ok=True)
                after = partial.stat().st_size if partial.exists() else 0
                record = {
                    "selection_attempt": selection_attempt,
                    "transfer_attempt": transfer_attempt,
                    "transport": prefix,
                    "source_image": source_image,
                    "returncode": completed.returncode,
                    "before_bytes": before,
                    "after_bytes": after,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
                if completed.returncode != 0:
                    record["error"] = completed.stderr.decode(
                        "utf-8", errors="replace"
                    )[-500:]
                records.append(record)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                records.append(
                    {
                        "selection_attempt": selection_attempt,
                        "transfer_attempt": None,
                        "transport": prefix,
                        "source_image": source_image,
                        "returncode": None,
                        "before_bytes": before,
                        "after_bytes": partial.stat().st_size if partial.exists() else 0,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "error": str(exc),
                    }
                )

            if partial.exists() and partial.stat().st_size > descriptor.size:
                partial.unlink()
            if _is_valid_blob(partial, descriptor):
                os.replace(partial, target)
                return {
                    "status": "downloaded",
                    "bytes": descriptor.size,
                    "attempts": transfer_attempt,
                    "selection_attempts": selection_attempt,
                    "records": records,
                }
            if partial.exists() and partial.stat().st_size == descriptor.size:
                partial.unlink()
            time.sleep(min(5, transfer_attempt + 1))

    raise RuntimeError(
        f"resumable download failed for {descriptor.digest}: "
        + json.dumps(records, sort_keys=True)
    )


def _acquire_shard_locks(shard_roots: tuple[Path, ...]) -> ExitStack:
    stack = ExitStack()
    try:
        for root in shard_roots:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            handle = stack.enter_context((root / ".pull.lock").open("a+b"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"layer cache is active and cannot be seeded: {root}") from exc
        return stack
    except BaseException:
        stack.close()
        raise


def _seed_shared_from_shards(
    plan: dict[str, LayerPlan],
    *,
    shared_cache_root: Path,
    shard_roots: tuple[Path, ...],
) -> dict[str, int]:
    seeded = 0
    reused = 0
    for layer in plan.values():
        target = shared_cache_root / layer.descriptor.digest
        if _has_expected_size(target, layer.descriptor):
            reused += 1
            continue
        for root in shard_roots:
            candidate = root / layer.descriptor.digest
            if _is_valid_blob(candidate, layer.descriptor):
                _install_verified_blob(candidate, target, layer.descriptor)
                seeded += 1
                break
    return {"seeded": seeded, "reused": reused}


def _seed_shards(
    bindings: tuple[ImageBinding, ...],
    image_layers: dict[str, tuple[LayerDescriptor, ...]],
    *,
    shared_cache_root: Path,
    shard_roots: tuple[Path, ...],
) -> dict[str, int]:
    installed = 0
    reused = 0
    methods: dict[str, int] = {}
    for index, binding in enumerate(bindings):
        root = shard_roots[index % len(shard_roots)]
        for descriptor in image_layers[binding.digest]:
            source = shared_cache_root / descriptor.digest
            target = root / descriptor.digest
            method = _install_verified_blob(source, target, descriptor)
            if method == "reused":
                reused += 1
            else:
                installed += 1
                methods[method] = methods.get(method, 0) + 1
    return {"installed": installed, "reused": reused, **methods}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", action="append", type=parse_binding, required=True)
    parser.add_argument("--shared-layer-cache-root", type=Path, required=True)
    parser.add_argument("--manifest-cache-root", type=Path, required=True)
    parser.add_argument("--shard-layer-cache-root", action="append", type=Path, default=[])
    parser.add_argument("--crane", type=Path, required=True)
    parser.add_argument("--curl", type=Path, default=Path("/usr/bin/curl"))
    parser.add_argument("--transport-prefix", required=True)
    parser.add_argument("--fallback-transport-prefix", action="append", default=[])
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--download-attempts", type=int, default=12)
    parser.add_argument("--download-timeout-seconds", type=int, default=300)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_workers <= 0 or args.download_attempts <= 0:
        parser.error("worker and attempt counts must be positive")
    if args.download_timeout_seconds <= 0:
        parser.error("download timeout must be positive")
    try:
        prefixes = _transport_prefixes(
            args.transport_prefix, args.fallback_transport_prefix
        )
    except ValueError as exc:
        parser.error(str(exc))
    bindings = tuple(args.binding)
    if len({binding.source_image for binding in bindings}) != len(bindings):
        parser.error("source image bindings must be unique")
    crane = args.crane.expanduser().resolve()
    curl = args.curl.expanduser().resolve()
    for label, executable in (("crane", crane), ("curl", curl)):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            parser.error(f"{label} is not executable: {executable}")

    started = time.monotonic()
    environment = dict(os.environ)
    shared_cache_root = args.shared_layer_cache_root.expanduser().resolve()
    manifest_cache_root = args.manifest_cache_root.expanduser().resolve()
    shard_roots = tuple(path.expanduser().resolve() for path in args.shard_layer_cache_root)
    shared_cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    manifests: dict[str, bytes] = {}
    manifest_results: dict[str, dict[str, Any]] = {}
    manifest_failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_manifest,
                binding,
                crane=crane,
                prefixes=prefixes,
                environment=environment,
                manifest_cache_root=manifest_cache_root,
            ): binding
            for binding in bindings
        }
        for future in as_completed(futures):
            binding = futures[future]
            try:
                raw, result = future.result()
                manifests[binding.digest] = raw
                manifest_results[binding.digest] = result
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                manifest_failures[binding.digest] = str(exc)
    if manifest_failures:
        evidence = {
            "schema": _EVIDENCE_SCHEMA,
            "status": "failed",
            "phase": "manifests",
            "failures": manifest_failures,
        }
        _atomic_write(args.evidence_output.expanduser().resolve(), _json_bytes(evidence))
        raise RuntimeError(f"manifest fetch failed for {len(manifest_failures)} images")

    plan, image_layers = build_layer_plan(bindings, manifests)
    lock_stack = _acquire_shard_locks(shard_roots) if shard_roots else ExitStack()
    with lock_stack:
        seed_result = _seed_shared_from_shards(
            plan,
            shared_cache_root=shared_cache_root,
            shard_roots=shard_roots,
        )
        layer_results: dict[str, dict[str, Any]] = {}
        layer_failures: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    _download_layer,
                    layer,
                    shared_cache_root=shared_cache_root,
                    crane=crane,
                    curl=curl,
                    prefixes=prefixes,
                    environment=environment,
                    attempts=args.download_attempts,
                    timeout_seconds=args.download_timeout_seconds,
                ): digest
                for digest, layer in plan.items()
            }
            for future in as_completed(futures):
                digest = futures[future]
                try:
                    layer_results[digest] = future.result()
                    print(
                        json.dumps(
                            {"digest": digest, **layer_results[digest]},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    layer_failures[digest] = str(exc)

        shard_result: dict[str, int] = {}
        if not layer_failures and shard_roots:
            shard_result = _seed_shards(
                bindings,
                image_layers,
                shared_cache_root=shared_cache_root,
                shard_roots=shard_roots,
            )

    evidence = {
        "schema": _EVIDENCE_SCHEMA,
        "status": "failed" if layer_failures else "pass",
        "image_count": len(bindings),
        "manifest_results": manifest_results,
        "unique_layer_count": len(plan),
        "unique_layer_bytes": sum(layer.descriptor.size for layer in plan.values()),
        "shared_seed": seed_result,
        "layer_results": layer_results,
        "layer_failures": layer_failures,
        "shard_seed": shard_result,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _atomic_write(args.evidence_output.expanduser().resolve(), _json_bytes(evidence))
    if layer_failures:
        raise RuntimeError(f"layer download failed for {len(layer_failures)} digests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
