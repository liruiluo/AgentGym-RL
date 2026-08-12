#!/usr/bin/env python3
"""Materialize digest-pinned SWE-smith OCI root filesystems out of band."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_CACHE_SCHEMA = "swesmith_oci_rootfs_cache_v1"
_MANIFEST_SCHEMA = "swesmith_oci_image_manifest_v1"


@dataclass(frozen=True)
class ImageBinding:
    source_image: str
    profile_image: str
    digest: str

    @property
    def cache_name(self) -> str:
        return self.digest.replace(":", "-")


def parse_binding(raw: str) -> ImageBinding:
    try:
        images, digest = raw.rsplit("@", 1)
        source_image, profile_image = images.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "binding must be SOURCE_IMAGE=PROFILE_IMAGE@sha256:DIGEST"
        ) from exc
    source_image = source_image.strip()
    profile_image = profile_image.strip()
    digest = digest.strip().lower()
    if not source_image or not profile_image or _DIGEST_RE.fullmatch(digest) is None:
        raise argparse.ArgumentTypeError(
            "binding must be SOURCE_IMAGE=PROFILE_IMAGE@sha256:DIGEST"
        )
    if any(character.isspace() for character in source_image + profile_image):
        raise argparse.ArgumentTypeError("image names must not contain whitespace")
    return ImageBinding(source_image, profile_image, digest)


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


def _run_bytes(argv: list[str], *, environment: dict[str, str]) -> bytes:
    completed = subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=900,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"command failed ({completed.returncode}): {argv!r}\n{error}")
    return completed.stdout


def _transport_prefixes(primary: str, fallbacks: Iterable[str]) -> tuple[str, ...]:
    prefixes: list[str] = []
    for raw in (primary, *fallbacks):
        prefix = raw.strip().rstrip("/")
        if not prefix:
            raise ValueError("transport prefixes must not be empty")
        if prefix not in prefixes:
            prefixes.append(prefix)
    return tuple(prefixes)


def _pull_cached_tarball(
    binding: ImageBinding,
    *,
    partial: Path,
    crane: Path,
    transport_prefixes: tuple[str, ...],
    layer_cache_root: Path,
    download_attempts: int,
    environment: dict[str, str],
) -> tuple[str, str, bytes, bytes, int]:
    """Pull one digest-pinned image while sharing immutable layers across images."""
    image_tarball = partial / "image.tar"
    pull_log = partial / "crane-pull.stderr.log"
    lock_path = layer_cache_root / ".pull.lock"
    layer_cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    failures: list[str] = []
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        for attempt in range(download_attempts):
            prefix = transport_prefixes[attempt % len(transport_prefixes)]
            reference = f"{prefix}/{binding.source_image}@{binding.digest}"
            image_tarball.unlink(missing_ok=True)
            try:
                manifest_raw = _run_bytes(
                    [str(crane), "manifest", reference], environment=environment
                )
                if f"sha256:{hashlib.sha256(manifest_raw).hexdigest()}" != binding.digest:
                    raise RuntimeError(
                        f"registry manifest bytes do not match {binding.digest}"
                    )
                manifest = json.loads(manifest_raw)
                config_raw = _run_bytes(
                    [str(crane), "config", reference], environment=environment
                )
                config_sha = hashlib.sha256(config_raw).hexdigest()
                if manifest.get("config", {}).get("digest") != f"sha256:{config_sha}":
                    raise RuntimeError(
                        "registry config bytes do not match the manifest descriptor"
                    )
                with pull_log.open("ab") as error_handle:
                    error_handle.write(
                        f"attempt={attempt + 1} reference={reference}\n".encode("utf-8")
                    )
                    completed = subprocess.run(
                        [
                            str(crane),
                            "pull",
                            "--format=tarball",
                            "--cache_path",
                            str(layer_cache_root),
                            reference,
                            str(image_tarball),
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=error_handle,
                        env=environment,
                        timeout=1800,
                    )
                if completed.returncode != 0 or not image_tarball.is_file():
                    raise RuntimeError(f"crane pull exited {completed.returncode}")
                return reference, prefix, manifest_raw, config_raw, attempt + 1
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                failures.append(f"attempt {attempt + 1} ({reference}): {exc}")
                if attempt + 1 < download_attempts:
                    time.sleep(min(30, 2 ** attempt))
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    raise RuntimeError("cached OCI pull failed:\n" + "\n".join(failures))


def _measure_rootfs(rootfs: Path) -> tuple[int, int]:
    total_bytes = 0
    regular_files = 0
    pending = [rootfs]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    regular_files += 1
                    total_bytes += entry.stat(follow_symlinks=False).st_size
    return total_bytes, regular_files


def _require_rootfs_contract(rootfs: Path) -> tuple[Path, Path]:
    for relative in (
        "testbed",
        "tmp",
        "var/tmp",
        "dev",
        "proc",
        "run",
    ):
        path = rootfs / relative
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"OCI rootfs is missing required directory: /{relative}")
    required_files = (
        "bin/bash",
        "usr/bin/setpriv",
        "usr/bin/prlimit",
        "usr/bin/env",
        "bin/sleep",
        "usr/bin/cut",
        "opt/miniconda3/bin/python3.12",
        "opt/miniconda3/envs/testbed/bin/python",
    )
    for relative in required_files:
        path = rootfs / relative
        if not path.is_file():
            raise RuntimeError(f"OCI rootfs is missing required file: /{relative}")
    return rootfs / "bin/bash", rootfs / "opt/miniconda3/bin/python3.12"


def _validate_complete_cache(cache_dir: Path, binding: ImageBinding) -> dict[str, Any]:
    if (cache_dir / ".complete").read_text(encoding="ascii") != "complete\n":
        raise RuntimeError(f"invalid completion marker: {cache_dir}")
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != _CACHE_SCHEMA:
        raise RuntimeError(f"unsupported cache metadata: {cache_dir}")
    if metadata.get("resolved_digest") != binding.digest:
        raise RuntimeError(f"cache digest mismatch: {cache_dir}")
    if metadata.get("repo_profile_image") != binding.profile_image:
        raise RuntimeError(f"cache profile image mismatch: {cache_dir}")
    manifest_sha = _sha256_file(cache_dir / "manifest.json")
    if f"sha256:{manifest_sha}" != binding.digest:
        raise RuntimeError(f"cache manifest digest mismatch: {cache_dir}")
    config_sha = _sha256_file(cache_dir / "config.json")
    if metadata.get("config_sha256") != config_sha:
        raise RuntimeError(f"cache config digest mismatch: {cache_dir}")
    _require_rootfs_contract(cache_dir / "rootfs")
    return metadata


def _materialize_one(
    binding: ImageBinding,
    *,
    cache_root: Path,
    crane: Path,
    transport_prefix: str,
    environment: dict[str, str],
    dataset_revision: str,
    source_revision: str,
    layer_cache_root: Path | None = None,
    transport_fallbacks: tuple[str, ...] = (),
    download_attempts: int = 1,
) -> dict[str, Any]:
    target = cache_root / binding.cache_name
    if target.is_dir() and (target / ".complete").is_file():
        metadata = _validate_complete_cache(target, binding)
        return {"status": "reused", "cache_dir": str(target), "metadata": metadata}
    if target.exists():
        raise RuntimeError(f"refusing to overwrite incomplete cache path: {target}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    partial = cache_root / f"{binding.cache_name}.partial-{stamp}-{os.getpid()}"
    if partial.exists():
        raise RuntimeError(f"partial cache path already exists: {partial}")
    rootfs = partial / "rootfs"
    rootfs.mkdir(parents=True, mode=0o755)
    prefixes = _transport_prefixes(transport_prefix, transport_fallbacks)
    reference = f"{prefixes[0]}/{binding.source_image}@{binding.digest}"
    selected_transport = prefixes[0]
    pull_attempts = 1
    if layer_cache_root is None:
        manifest_raw = _run_bytes(
            [str(crane), "manifest", reference], environment=environment
        )
        if f"sha256:{hashlib.sha256(manifest_raw).hexdigest()}" != binding.digest:
            raise RuntimeError(f"registry manifest bytes do not match {binding.digest}")
        manifest = json.loads(manifest_raw)
        config_raw = _run_bytes(
            [str(crane), "config", reference], environment=environment
        )
    else:
        (
            reference,
            selected_transport,
            manifest_raw,
            config_raw,
            pull_attempts,
        ) = _pull_cached_tarball(
            binding,
            partial=partial,
            crane=crane,
            transport_prefixes=prefixes,
            layer_cache_root=layer_cache_root,
            download_attempts=download_attempts,
            environment=environment,
        )
        manifest = json.loads(manifest_raw)
    config_sha = hashlib.sha256(config_raw).hexdigest()
    if manifest.get("config", {}).get("digest") != f"sha256:{config_sha}":
        raise RuntimeError("registry config bytes do not match the manifest descriptor")
    config = json.loads(config_raw)
    if config.get("architecture") != "amd64" or config.get("os") != "linux":
        raise RuntimeError("SWE-smith image must be Linux amd64")
    if config.get("config", {}).get("WorkingDir") != "/testbed":
        raise RuntimeError("SWE-smith image must use /testbed as WorkingDir")
    _atomic_write(partial / "manifest.json", manifest_raw)
    _atomic_write(partial / "config.json", config_raw)

    export_stderr = partial / "crane-export.stderr.log"
    tar_stderr = partial / "tar-extract.stderr.log"
    image_tarball = partial / "image.tar"
    with export_stderr.open("wb") as export_error, tar_stderr.open("wb") as tar_error:
        image_input = image_tarball.open("rb") if layer_cache_root is not None else None
        exporter = subprocess.Popen(
            [str(crane), "export", "-" if image_input is not None else reference, "-"],
            stdin=image_input,
            stdout=subprocess.PIPE,
            stderr=export_error,
            env=environment,
        )
        assert exporter.stdout is not None
        extractor = subprocess.Popen(
            [
                "tar",
                "--extract",
                "--preserve-permissions",
                "--numeric-owner",
                "--file=-",
                "--directory",
                str(rootfs),
            ],
            stdin=exporter.stdout,
            stdout=subprocess.DEVNULL,
            stderr=tar_error,
        )
        exporter.stdout.close()
        tar_returncode = extractor.wait()
        export_returncode = exporter.wait()
        if image_input is not None:
            image_input.close()
    if export_returncode != 0 or tar_returncode != 0:
        raise RuntimeError(
            f"OCI export failed: crane={export_returncode} tar={tar_returncode}; "
            f"partial={partial}"
        )
    image_tarball.unlink(missing_ok=True)

    bash, python312 = _require_rootfs_contract(rootfs)
    rootfs_bytes, regular_files = _measure_rootfs(rootfs)
    crane_version = _run_bytes([str(crane), "version"], environment=environment).decode(
        "utf-8", errors="replace"
    ).strip()
    metadata = {
        "schema": _CACHE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_image": binding.source_image,
        "repo_profile_image": binding.profile_image,
        "pull_ref": reference,
        "transport_mirror": selected_transport,
        "transport_attempts": pull_attempts,
        "layer_cache_enabled": layer_cache_root is not None,
        "resolved_digest": binding.digest,
        "manifest_sha256": binding.digest.removeprefix("sha256:"),
        "config_sha256": config_sha,
        "crane": {
            "version": crane_version,
            "binary_sha256": _sha256_file(crane),
        },
        "upstream": {
            "huggingface_revision": dataset_revision,
            "swe_smith_commit": source_revision,
        },
        "rootfs": {
            "working_dir": "/testbed",
            "bytes": rootfs_bytes,
            "regular_files": regular_files,
            "bash_sha256": _sha256_file(bash),
            "python312_sha256": _sha256_file(python312),
        },
    }
    _atomic_write(partial / "metadata.json", _json_bytes(metadata))
    _atomic_write(partial / ".complete", b"complete\n")
    os.replace(partial, target)
    _validate_complete_cache(target, binding)
    return {"status": "created", "cache_dir": str(target), "metadata": metadata}


def build_image_manifest(
    bindings: Iterable[ImageBinding],
    *,
    dataset_revision: str,
    source_revision: str,
) -> dict[str, Any]:
    entries = sorted(
        ({"image": binding.profile_image, "digest": binding.digest} for binding in bindings),
        key=lambda entry: entry["image"],
    )
    if not entries or len({entry["image"] for entry in entries}) != len(entries):
        raise ValueError("image bindings must have unique profile names")
    if len({entry["digest"] for entry in entries}) != len(entries):
        raise ValueError("image bindings must have unique digests")
    return {
        "schema_version": _MANIFEST_SCHEMA,
        "images": entries,
        "upstream": {
            "repository": "SWE-bench/SWE-smith",
            "dataset_revision": dataset_revision,
            "source_revision": source_revision,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", action="append", type=parse_binding, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--crane", type=Path, required=True)
    parser.add_argument("--transport-prefix", required=True)
    parser.add_argument("--fallback-transport-prefix", action="append", default=[])
    parser.add_argument("--layer-cache-root", type=Path)
    parser.add_argument("--download-attempts", type=int, default=6)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--image-manifest-output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    for label, revision in (
        ("dataset", args.dataset_revision),
        ("source", args.source_revision),
    ):
        if _REVISION_RE.fullmatch(revision) is None:
            parser.error(f"{label} revision must be a full lowercase Git commit")
    if args.max_workers <= 0 or args.download_attempts <= 0:
        parser.error("--max-workers and --download-attempts must be positive")
    try:
        transport_prefixes = _transport_prefixes(
            args.transport_prefix, args.fallback_transport_prefix
        )
    except ValueError as exc:
        parser.error(str(exc))
    bindings = tuple(args.binding)
    if len({binding.source_image for binding in bindings}) != len(bindings):
        parser.error("source image bindings must be unique")
    args.cache_root.mkdir(parents=True, exist_ok=True, mode=0o755)
    crane = args.crane.expanduser().resolve()
    if not crane.is_file() or not os.access(crane, os.X_OK):
        parser.error(f"crane is not executable: {crane}")

    environment = dict(os.environ)
    layer_cache_root = (
        args.layer_cache_root.expanduser().resolve()
        if args.layer_cache_root is not None
        else None
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _materialize_one,
                binding,
                cache_root=args.cache_root.expanduser().resolve(),
                crane=crane,
                transport_prefix=args.transport_prefix,
                environment=environment,
                dataset_revision=args.dataset_revision,
                source_revision=args.source_revision,
                layer_cache_root=layer_cache_root,
                transport_fallbacks=transport_prefixes[1:],
                download_attempts=args.download_attempts,
            ): binding
            for binding in bindings
        }
        for future in as_completed(futures):
            binding = futures[future]
            result = future.result()
            results.append({"profile_image": binding.profile_image, **result})
            print(
                json.dumps(
                    {
                        "profile_image": binding.profile_image,
                        "status": result["status"],
                        "cache_dir": result["cache_dir"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    image_manifest = build_image_manifest(
        bindings,
        dataset_revision=args.dataset_revision,
        source_revision=args.source_revision,
    )
    manifest_raw = _json_bytes(image_manifest)
    _atomic_write(args.image_manifest_output.expanduser().resolve(), manifest_raw)
    print(
        json.dumps(
            {
                "status": "pass",
                "image_count": len(bindings),
                "image_manifest": str(args.image_manifest_output.expanduser().resolve()),
                "image_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "cache_results": sorted(results, key=lambda result: result["profile_image"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
