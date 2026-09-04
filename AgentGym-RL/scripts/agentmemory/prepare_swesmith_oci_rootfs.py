#!/usr/bin/env python3
"""Materialize digest-pinned SWE-smith OCI root filesystems out of band."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
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
_OFFLINE_ASSET_SCHEMA = "camg_swesmith_offline_image_assets_v1"
_OFFLINE_PRESTAGE_SCHEMA = "camg_swesmith_final128_oci_metadata_prestage_v1"
_VERIFIED_LAYER_CACHE_FILES: set[tuple[str, int, int, int]] = set()


@dataclass(frozen=True)
class ImageBinding:
    source_image: str
    profile_image: str
    digest: str

    @property
    def cache_name(self) -> str:
        return self.digest.replace(":", "-")


@dataclass(frozen=True)
class OfflineImageAsset:
    image_tarball: Path
    image_tarball_sha256: str
    image_tarball_bytes: int
    manifest_raw: bytes
    config_raw: bytes


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


def _run_bytes(
    argv: list[str],
    *,
    environment: dict[str, str],
    timeout_seconds: int = 180,
) -> bytes:
    completed = subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=timeout_seconds,
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


def _bound_regular_file(
    path: Path,
    *,
    root: Path,
    expected_sha256: str,
    expected_bytes: int | None = None,
    label: str,
) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {expanded}")
    resolved = (expanded if expanded.is_absolute() else root / expanded).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the offline asset package: {resolved}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not an immutable regular file: {resolved}")
    if expected_bytes is not None and resolved.stat().st_size != expected_bytes:
        raise RuntimeError(f"{label} byte count drifted: {resolved}")
    if _sha256_file(resolved) != expected_sha256:
        raise RuntimeError(f"{label} digest drifted: {resolved}")
    return resolved


def load_offline_image_assets(
    path: Path,
    bindings: tuple[ImageBinding, ...],
) -> dict[str, OfflineImageAsset]:
    """Load a complete, hash-bound offline image set for this materialization."""

    expanded_manifest_path = path.expanduser()
    if expanded_manifest_path.is_symlink():
        raise RuntimeError(
            f"offline image asset manifest must not be a symlink: {expanded_manifest_path}"
        )
    manifest_path = expanded_manifest_path.resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"offline image asset manifest is invalid: {manifest_path}")
    package_root = manifest_path.parent.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("images")
    if (
        payload.get("schema") != _OFFLINE_ASSET_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("network_required_at_launch") is not False
        or not isinstance(records, list)
        or int(payload.get("image_count", -1)) != len(records)
    ):
        raise RuntimeError("offline image asset manifest is not launch-ready")

    prestage_record = payload.get("source_metadata_prestage")
    if not isinstance(prestage_record, dict):
        raise RuntimeError("offline image asset manifest lacks metadata provenance")
    prestage_path = _bound_regular_file(
        Path(str(prestage_record.get("path", ""))),
        root=package_root,
        expected_sha256=str(prestage_record.get("sha256", "")),
        label="offline metadata prestage",
    )
    prestage = json.loads(prestage_path.read_text(encoding="utf-8"))
    prestage_images = prestage.get("images")
    if (
        prestage.get("schema") != _OFFLINE_PRESTAGE_SCHEMA
        or prestage.get("status") != "pass"
        or int(prestage.get("missing_layer_count", -1)) != 0
        or prestage.get("bad_layers") != []
        or not isinstance(prestage_images, list)
    ):
        raise RuntimeError("offline metadata prestage is not complete")
    provenance_by_profile = {
        str(record.get("profile_image", "")): record for record in prestage_images
    }

    records_by_profile: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("offline image asset entry is not an object")
        profile_image = str(record.get("profile_image", ""))
        if not profile_image or profile_image in records_by_profile:
            raise RuntimeError("offline image asset profiles must be nonempty and unique")
        records_by_profile[profile_image] = record

    expected_profiles = {binding.profile_image for binding in bindings}
    if set(records_by_profile) != expected_profiles:
        missing = sorted(expected_profiles - set(records_by_profile))
        extra = sorted(set(records_by_profile) - expected_profiles)
        raise RuntimeError(
            f"offline image assets differ from selected bindings: missing={missing}, extra={extra}"
        )

    assets: dict[str, OfflineImageAsset] = {}
    for binding in bindings:
        record = records_by_profile[binding.profile_image]
        provenance = provenance_by_profile.get(binding.profile_image)
        if not isinstance(provenance, dict):
            raise RuntimeError(
                f"offline image metadata is missing for {binding.profile_image}"
            )
        if (
            record.get("source_image") != binding.source_image
            or record.get("digest") != binding.digest
            or provenance.get("source_image") != binding.source_image
            or provenance.get("digest") != binding.digest
        ):
            raise RuntimeError(
                f"offline image identity drifted for {binding.profile_image}"
            )
        image_tar = record.get("image_tar")
        if not isinstance(image_tar, dict):
            raise RuntimeError(f"offline image tar is missing for {binding.profile_image}")
        tarball = _bound_regular_file(
            Path(str(image_tar.get("path", ""))),
            root=package_root,
            expected_sha256=str(image_tar.get("sha256", "")),
            expected_bytes=int(image_tar.get("bytes", -1)),
            label=f"offline image tar for {binding.profile_image}",
        )
        manifest_path_for_image = _bound_regular_file(
            Path(str(provenance.get("manifest", ""))),
            root=package_root,
            expected_sha256=binding.digest.removeprefix("sha256:"),
            label=f"OCI manifest for {binding.profile_image}",
        )
        config_path = _bound_regular_file(
            Path(str(provenance.get("config", ""))),
            root=package_root,
            expected_sha256=str(record.get("config_sha256", "")),
            label=f"OCI config for {binding.profile_image}",
        )
        manifest_raw = manifest_path_for_image.read_bytes()
        config_raw = config_path.read_bytes()
        manifest = json.loads(manifest_raw)
        config_sha256 = hashlib.sha256(config_raw).hexdigest()
        if (
            record.get("manifest_sha256") != binding.digest.removeprefix("sha256:")
            or manifest.get("config", {}).get("digest") != f"sha256:{config_sha256}"
        ):
            raise RuntimeError(
                f"offline OCI metadata drifted for {binding.profile_image}"
            )
        expected_members = {
            "manifest.json",
            f"sha256:{config_sha256}",
            *{
                f"{layer['digest'].removeprefix('sha256:')}.tar.gz"
                for layer in provenance.get("layers", [])
            },
        }
        with tarfile.open(tarball, "r:") as archive:
            members = archive.getmembers()
            if any(not member.isfile() for member in members):
                raise RuntimeError(
                    f"offline image tar contains non-regular entries: {binding.profile_image}"
                )
            if {member.name for member in members} != expected_members:
                raise RuntimeError(
                    f"offline image tar member set drifted: {binding.profile_image}"
                )
            docker_manifest_handle = archive.extractfile("manifest.json")
            config_handle = archive.extractfile(f"sha256:{config_sha256}")
            if docker_manifest_handle is None or config_handle is None:
                raise RuntimeError(
                    f"offline image tar metadata is missing: {binding.profile_image}"
                )
            docker_manifest = json.load(docker_manifest_handle)
            archived_config = config_handle.read()
        expected_layers = [
            f"{layer['digest'].removeprefix('sha256:')}.tar.gz"
            for layer in provenance.get("layers", [])
        ]
        if (
            archived_config != config_raw
            or not isinstance(docker_manifest, list)
            or len(docker_manifest) != 1
            or docker_manifest[0].get("Config") != f"sha256:{config_sha256}"
            or docker_manifest[0].get("Layers") != expected_layers
        ):
            raise RuntimeError(
                f"offline Docker archive contract drifted: {binding.profile_image}"
            )
        assets[binding.profile_image] = OfflineImageAsset(
            image_tarball=tarball,
            image_tarball_sha256=str(image_tar.get("sha256")),
            image_tarball_bytes=int(image_tar.get("bytes")),
            manifest_raw=manifest_raw,
            config_raw=config_raw,
        )
    return assets


def _purge_invalid_cached_layers(layer_cache_root: Path) -> tuple[str, ...]:
    """Remove corrupt crane cache entries before they poison another retry."""
    removed: list[str] = []
    for path in layer_cache_root.iterdir():
        if not path.is_file() or _DIGEST_RE.fullmatch(path.name) is None:
            continue
        stat = path.stat()
        identity = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity in _VERIFIED_LAYER_CACHE_FILES:
            continue
        expected = path.name.removeprefix("sha256:")
        if _sha256_file(path) != expected:
            path.unlink()
            removed.append(path.name)
            continue
        _VERIFIED_LAYER_CACHE_FILES.add(identity)
    return tuple(sorted(removed))


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
            invalid_layers = _purge_invalid_cached_layers(layer_cache_root)
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
                    if invalid_layers:
                        error_handle.write(
                            (
                                "purged_invalid_layer_cache="
                                + ",".join(invalid_layers)
                                + "\n"
                            ).encode("ascii")
                        )
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


def _require_rootfs_traversable(rootfs: Path) -> None:
    if not rootfs.is_dir() or rootfs.is_symlink():
        raise RuntimeError(f"OCI rootfs is not a secure directory: {rootfs}")
    if stat.S_IMODE(rootfs.stat().st_mode) & 0o555 != 0o555:
        raise RuntimeError(
            "OCI rootfs top directory must be traversable by the unprivileged policy"
        )


def _make_rootfs_traversable(rootfs: Path) -> None:
    # The sealed eval launcher intentionally runs with umask 077.  mkdir(0755)
    # is therefore insufficient, and OCI extraction may also restore a root
    # directory mode from the image.  Normalize only the chroot top directory;
    # the containing cache remains private and image-internal modes stay intact.
    os.chmod(rootfs, 0o755)
    _require_rootfs_traversable(rootfs)


def _require_rootfs_contract(rootfs: Path) -> Path:
    _require_rootfs_traversable(rootfs)
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
    )
    for relative in required_files:
        path = rootfs / relative
        if not path.is_file():
            raise RuntimeError(f"OCI rootfs is missing required file: /{relative}")
    return rootfs / "bin/bash"


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
    transport_prefix: str | None,
    environment: dict[str, str],
    dataset_revision: str,
    source_revision: str,
    layer_cache_root: Path | None = None,
    transport_fallbacks: tuple[str, ...] = (),
    download_attempts: int = 1,
    offline_image_asset: OfflineImageAsset | None = None,
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
    if offline_image_asset is not None:
        prefixes: tuple[str, ...] = ()
        reference = f"{binding.source_image}@{binding.digest}"
        selected_transport = "offline-image-asset"
        pull_attempts = 0
        manifest_raw = offline_image_asset.manifest_raw
        config_raw = offline_image_asset.config_raw
        manifest = json.loads(manifest_raw)
    else:
        if transport_prefix is None:
            raise RuntimeError("online OCI materialization requires a transport prefix")
        prefixes = _transport_prefixes(transport_prefix, transport_fallbacks)
        reference = f"{prefixes[0]}/{binding.source_image}@{binding.digest}"
        selected_transport = prefixes[0]
        pull_attempts = 1
    if offline_image_asset is None and layer_cache_root is None:
        manifest_raw = _run_bytes(
            [str(crane), "manifest", reference], environment=environment
        )
        if f"sha256:{hashlib.sha256(manifest_raw).hexdigest()}" != binding.digest:
            raise RuntimeError(f"registry manifest bytes do not match {binding.digest}")
        manifest = json.loads(manifest_raw)
        config_raw = _run_bytes(
            [str(crane), "config", reference], environment=environment
        )
    elif offline_image_asset is None:
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
    image_tarball = (
        offline_image_asset.image_tarball
        if offline_image_asset is not None
        else partial / "image.tar"
    )
    with export_stderr.open("wb") as export_error, tar_stderr.open("wb") as tar_error:
        image_input = (
            image_tarball.open("rb")
            if offline_image_asset is not None or layer_cache_root is not None
            else None
        )
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
    if offline_image_asset is None:
        image_tarball.unlink(missing_ok=True)

    _make_rootfs_traversable(rootfs)
    bash = _require_rootfs_contract(rootfs)
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
        "offline_image_asset": (
            {
                "path": str(offline_image_asset.image_tarball),
                "bytes": offline_image_asset.image_tarball_bytes,
                "sha256": offline_image_asset.image_tarball_sha256,
            }
            if offline_image_asset is not None
            else None
        ),
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


def select_materialization_bindings(
    bindings: tuple[ImageBinding, ...],
    requested_profile_images: Iterable[str],
) -> tuple[ImageBinding, ...]:
    """Select the root filesystems needed by one frozen evaluation panel.

    The complete binding tuple still defines the generated image manifest.  This
    selector only limits which digest-pinned root filesystems are materialized,
    so a formal subset does not silently rewrite the frozen image identity.
    """

    requested = tuple(value.strip() for value in requested_profile_images)
    if not requested:
        return bindings
    if any(not value for value in requested):
        raise ValueError("materialized profile image names must not be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("materialized profile image names must be unique")
    available = {binding.profile_image for binding in bindings}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(
            "materialized profile images are absent from the frozen bindings: "
            + ", ".join(unknown)
        )
    selected = set(requested)
    return tuple(
        binding for binding in bindings if binding.profile_image in selected
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", action="append", type=parse_binding, required=True)
    parser.add_argument("--materialize-profile-image", action="append", default=[])
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--crane", type=Path, required=True)
    parser.add_argument("--offline-image-asset-manifest", type=Path)
    parser.add_argument("--transport-prefix")
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
    bindings = tuple(args.binding)
    if len({binding.source_image for binding in bindings}) != len(bindings):
        parser.error("source image bindings must be unique")
    try:
        materialization_bindings = select_materialization_bindings(
            bindings, args.materialize_profile_image
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.offline_image_asset_manifest is not None:
        if (
            args.transport_prefix is not None
            or args.fallback_transport_prefix
            or args.layer_cache_root is not None
        ):
            parser.error(
                "offline image assets cannot be combined with registry transports "
                "or a download cache"
            )
        offline_assets = load_offline_image_assets(
            args.offline_image_asset_manifest, materialization_bindings
        )
        transport_prefixes: tuple[str, ...] = ()
    else:
        if args.transport_prefix is None:
            parser.error(
                "either --offline-image-asset-manifest or --transport-prefix is required"
            )
        try:
            transport_prefixes = _transport_prefixes(
                args.transport_prefix, args.fallback_transport_prefix
            )
        except ValueError as exc:
            parser.error(str(exc))
        offline_assets = {}
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
                offline_image_asset=offline_assets.get(binding.profile_image),
            ): binding
            for binding in materialization_bindings
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
                "materialized_image_count": len(materialization_bindings),
                "materialized_profile_images": sorted(
                    binding.profile_image for binding in materialization_bindings
                ),
                "offline_image_assets": args.offline_image_asset_manifest is not None,
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
