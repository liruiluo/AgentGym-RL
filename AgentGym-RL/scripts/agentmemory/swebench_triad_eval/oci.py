"""Cache-local OCI staging for SWE-bench Verified task images."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import secrets
import shutil
import stat
import subprocess
import tarfile
from typing import Any, Callable, Mapping, Sequence

from . import ARMS
from .atomic import (
    atomic_write_json,
    canonical_json_bytes,
    ensure_private_directory,
    exclusive_lock,
    fsync_directory,
)


class OciCacheError(RuntimeError):
    """Raised when immutable OCI or staging evidence fails closed."""


SHA256_PATTERN = re.compile(r"^sha256:([0-9a-f]{64})$")
REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$"
)
CONFIG_MEDIA_TYPES = {
    "application/vnd.docker.container.image.v1+json",
    "application/vnd.oci.image.config.v1+json",
}
GZIP_LAYER_MEDIA_TYPES = {
    "application/vnd.docker.image.rootfs.diff.tar.gzip",
    "application/vnd.oci.image.layer.v1.tar+gzip",
}
PLAIN_LAYER_MEDIA_TYPES = {
    "application/vnd.oci.image.layer.v1.tar",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_sha256_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise OciCacheError(f"{label} must be a lowercase sha256 digest")
    return value


def require_real_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as error:
        raise OciCacheError(f"{label} is unavailable: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise OciCacheError(f"{label} must be a real regular file: {path}")
    return path.resolve(strict=True)


def require_real_directory(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as error:
        raise OciCacheError(f"{label} is unavailable: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise OciCacheError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def read_json_file(path: Path, label: str) -> Any:
    resolved = require_real_file(path, label)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OciCacheError(f"{label} is invalid JSON: {path}") from error


def path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


@dataclass(frozen=True)
class OciLayer:
    media_type: str
    digest: str
    size: int
    diff_id: str
    blob_path: Path


@dataclass(frozen=True)
class OciImageBinding:
    image: str
    manifest_digest: str
    manifest_path: Path
    manifest_sha256: str
    config_digest: str
    config_path: Path
    config_sha256: str
    working_dir: str
    layers: tuple[OciLayer, ...]

    @property
    def cache_name(self) -> str:
        return f"sha256-{self.manifest_digest[7:]}"

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "swebench_verified_oci_binding_v1",
            "image": self.image,
            "manifest_digest": self.manifest_digest,
            "manifest_sha256": self.manifest_sha256,
            "config_digest": self.config_digest,
            "config_sha256": self.config_sha256,
            "working_dir": self.working_dir,
            "layers": [
                {
                    "media_type": layer.media_type,
                    "digest": layer.digest,
                    "size": layer.size,
                    "diff_id": layer.diff_id,
                }
                for layer in self.layers
            ],
        }


def parse_descriptor(
    value: Any,
    *,
    label: str,
    allowed_media_types: set[str],
) -> tuple[str, str, int]:
    if not isinstance(value, Mapping):
        raise OciCacheError(f"{label} must be an object")
    media_type = value.get("mediaType")
    digest = value.get("digest")
    size = value.get("size")
    if media_type not in allowed_media_types:
        raise OciCacheError(f"{label} has an unsupported media type")
    require_sha256_digest(digest, f"{label} digest")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise OciCacheError(f"{label} size must be a non-negative integer")
    return media_type, digest, size


def verify_blob(path: Path, digest: str, size: int, label: str) -> Path:
    resolved = require_real_file(path, label)
    if resolved.stat().st_size != size:
        raise OciCacheError(f"{label} size drifted")
    if sha256_file(resolved) != digest[7:]:
        raise OciCacheError(f"{label} SHA-256 drifted")
    return resolved


def normalize_working_directory(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise OciCacheError("OCI WorkingDir must be an absolute path")
    if "\x00" in value or any(part == ".." for part in PurePosixPath(value).parts):
        raise OciCacheError("OCI WorkingDir is unsafe")
    normalized = posixpath.normpath(value)
    if not normalized.startswith("/"):
        raise OciCacheError("OCI WorkingDir escaped the rootfs")
    return normalized


class CachedOciStore:
    """Resolve one image exclusively from the certified local OCI cache."""

    def __init__(
        self,
        *,
        index_path: Path | str,
        manifest_root: Path | str,
        blob_root: Path | str,
    ) -> None:
        self.index_path = Path(index_path)
        self.manifest_root = Path(manifest_root)
        self.blob_root = Path(blob_root)

    def rows(self) -> list[Mapping[str, Any]]:
        index = require_real_file(self.index_path, "OCI manifest index")
        try:
            payload = index.read_bytes()
        except OSError as error:
            raise OciCacheError("cannot read OCI manifest index") from error
        if not payload or not payload.endswith(b"\n"):
            raise OciCacheError("OCI manifest index must be newline terminated")
        rows: list[Mapping[str, Any]] = []
        aliases: set[str] = set()
        for line_number, raw_line in enumerate(payload.splitlines(), start=1):
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise OciCacheError(
                    f"OCI manifest index row {line_number} is invalid JSON"
                ) from error
            if not isinstance(row, Mapping):
                raise OciCacheError(
                    f"OCI manifest index row {line_number} must be an object"
                )
            image = row.get("image")
            if not isinstance(image, str) or not image:
                raise OciCacheError("OCI image alias must be nonempty text")
            if image in aliases:
                raise OciCacheError(f"duplicate OCI image alias: {image}")
            aliases.add(image)
            rows.append(row)
        return rows

    def resolve(self, image: str) -> OciImageBinding:
        matches = [row for row in self.rows() if row.get("image") == image]
        if len(matches) != 1:
            raise OciCacheError(
                f"OCI image alias did not resolve exactly once: {image}"
            )
        row = matches[0]
        if row.get("platform") != "linux/amd64":
            raise OciCacheError("OCI image platform drifted")

        manifest_digest = require_sha256_digest(
            row.get("digest"), "OCI manifest digest"
        )
        if row.get("manifest_sha256") != manifest_digest[7:]:
            raise OciCacheError("OCI manifest index digest fields disagree")
        manifest_root = require_real_directory(
            self.manifest_root, "OCI manifest directory"
        )
        manifest_path = manifest_root / f"sha256-{manifest_digest[7:]}.json"
        manifest_path = require_real_file(manifest_path, "OCI manifest")
        try:
            manifest_raw = manifest_path.read_bytes()
            manifest = json.loads(manifest_raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OciCacheError("OCI manifest is invalid JSON") from error
        if hashlib.sha256(manifest_raw).hexdigest() != manifest_digest[7:]:
            raise OciCacheError("OCI manifest SHA-256 drifted")
        if not isinstance(manifest, Mapping) or manifest.get("schemaVersion") != 2:
            raise OciCacheError("OCI manifest schema drifted")
        if manifest.get("mediaType") != row.get("media_type"):
            raise OciCacheError("OCI manifest media type drifted")

        config_value = manifest.get("config")
        row_config = row.get("config")
        if config_value != row_config:
            raise OciCacheError("OCI config descriptor drifted from the index")
        _, config_digest, config_size = parse_descriptor(
            config_value,
            label="OCI config descriptor",
            allowed_media_types=CONFIG_MEDIA_TYPES,
        )
        blob_root = require_real_directory(self.blob_root, "OCI blob cache")
        config_path = verify_blob(
            blob_root / config_digest,
            config_digest,
            config_size,
            "OCI config blob",
        )
        try:
            config_raw = config_path.read_bytes()
            config = json.loads(config_raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OciCacheError("OCI config blob is invalid JSON") from error
        if not isinstance(config, Mapping):
            raise OciCacheError("OCI config must be an object")
        if config.get("architecture") != "amd64" or config.get("os") != "linux":
            raise OciCacheError("OCI config platform drifted")
        rootfs = config.get("rootfs")
        if not isinstance(rootfs, Mapping) or rootfs.get("type") != "layers":
            raise OciCacheError("OCI config rootfs declaration drifted")
        diff_ids = rootfs.get("diff_ids")
        if not isinstance(diff_ids, list) or not diff_ids:
            raise OciCacheError("OCI config diff-ID list is empty")
        for index, diff_id in enumerate(diff_ids):
            require_sha256_digest(diff_id, f"OCI layer {index} diff ID")

        layer_values = manifest.get("layers")
        if layer_values != row.get("layers"):
            raise OciCacheError("OCI layer descriptors drifted from the index")
        if not isinstance(layer_values, list) or len(layer_values) != len(diff_ids):
            raise OciCacheError("OCI layer and diff-ID counts disagree")
        layers: list[OciLayer] = []
        for index, (descriptor, diff_id) in enumerate(zip(layer_values, diff_ids)):
            media_type, digest, size = parse_descriptor(
                descriptor,
                label=f"OCI layer {index} descriptor",
                allowed_media_types=GZIP_LAYER_MEDIA_TYPES | PLAIN_LAYER_MEDIA_TYPES,
            )
            blob_path = verify_blob(
                blob_root / digest,
                digest,
                size,
                f"OCI layer {index} blob",
            )
            layers.append(
                OciLayer(
                    media_type=media_type,
                    digest=digest,
                    size=size,
                    diff_id=diff_id,
                    blob_path=blob_path,
                )
            )
        compressed_bytes = row.get("compressed_layer_bytes")
        if (
            isinstance(compressed_bytes, bool)
            or not isinstance(compressed_bytes, int)
            or compressed_bytes != sum(layer.size for layer in layers)
        ):
            raise OciCacheError("OCI compressed layer byte count drifted")

        config_section = config.get("config")
        if not isinstance(config_section, Mapping):
            raise OciCacheError("OCI runtime config section is missing")
        working_dir = normalize_working_directory(
            config_section.get("WorkingDir", "/")
        )
        return OciImageBinding(
            image=image,
            manifest_digest=manifest_digest,
            manifest_path=manifest_path,
            manifest_sha256=manifest_digest[7:],
            config_digest=config_digest,
            config_path=config_path,
            config_sha256=config_digest[7:],
            working_dir=working_dir,
            layers=tuple(layers),
        )


def canonical_member_path(name: str, label: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise OciCacheError(f"{label} has an invalid path")
    if name.startswith("/"):
        raise OciCacheError(f"{label} uses an absolute path")
    parts = PurePosixPath(name).parts
    if any(part == ".." for part in parts):
        raise OciCacheError(f"{label} traverses outside the rootfs")
    normalized = posixpath.normpath(name)
    if normalized in ("", ".") or normalized.startswith("../"):
        raise OciCacheError(f"{label} has an empty or escaping path")
    try:
        normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise OciCacheError(f"{label} path is not valid UTF-8") from error
    return normalized


def validate_symlink_target(member_path: str, target: str) -> None:
    if not isinstance(target, str) or not target or "\x00" in target:
        raise OciCacheError("OCI symlink has an invalid target")
    target_parts = PurePosixPath(target).parts
    if target.startswith("/"):
        if any(part == ".." for part in target_parts):
            raise OciCacheError("OCI absolute symlink target is not canonical")
        return
    resolved = posixpath.normpath(
        posixpath.join(posixpath.dirname(member_path), target)
    )
    if resolved == ".." or resolved.startswith("../"):
        raise OciCacheError("OCI symlink escapes the extracted rootfs")


def validate_hardlink_target(target: str) -> str:
    return canonical_member_path(target, "OCI hardlink target")


def remove_owned_path(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def ensure_parent_directories(rootfs: Path, relative_path: str) -> None:
    current = rootfs
    for part in PurePosixPath(relative_path).parts[:-1]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            current.mkdir(mode=0o755)
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OciCacheError("OCI member parent is not a real directory")


def whiteout_kind(member_path: str) -> tuple[str, str] | None:
    basename = posixpath.basename(member_path)
    parent = posixpath.dirname(member_path)
    if basename == ".wh..wh..opq":
        return "opaque", parent
    if basename.startswith(".wh."):
        target_name = basename[4:]
        if not target_name:
            raise OciCacheError("OCI whiteout target is empty")
        return "remove", posixpath.join(parent, target_name)
    return None


def validate_tar_members(
    members: Sequence[tarfile.TarInfo],
) -> list[tuple[tarfile.TarInfo, str, tuple[str, str] | None]]:
    validated: list[tuple[tarfile.TarInfo, str, tuple[str, str] | None]] = []
    paths: set[str] = set()
    for index, member in enumerate(members):
        member_path = canonical_member_path(member.name, f"OCI member {index}")
        if member_path in paths:
            raise OciCacheError(f"OCI layer contains an aliased path: {member_path}")
        paths.add(member_path)
        whiteout = whiteout_kind(member_path)
        if whiteout is not None:
            if not member.isreg() and not (
                member.ischr() and member.devmajor == 0 and member.devminor == 0
            ):
                raise OciCacheError("OCI whiteout has an unsupported type")
            validated.append((member, member_path, whiteout))
            continue
        if member.issym():
            validate_symlink_target(member_path, member.linkname)
        elif member.islnk():
            validate_hardlink_target(member.linkname)
        elif not (member.isdir() or member.isreg()):
            raise OciCacheError(f"OCI member has an unsafe type: {member_path}")
        validated.append((member, member_path, None))
    return validated


def apply_whiteouts(
    rootfs: Path,
    members: Sequence[tuple[tarfile.TarInfo, str, tuple[str, str] | None]],
) -> None:
    whiteouts = [item[2] for item in members if item[2] is not None]
    whiteouts.sort(key=lambda item: 0 if item[0] == "opaque" else 1)
    for kind, target in whiteouts:
        marker_path = target if kind == "remove" else posixpath.join(target, "marker")
        ensure_parent_directories(rootfs, marker_path)
        target_path = rootfs / target
        if kind == "remove":
            remove_owned_path(target_path)
            continue
        try:
            info = target_path.lstat()
        except FileNotFoundError:
            target_path.mkdir(mode=0o755)
            info = target_path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OciCacheError("OCI opaque whiteout target is not a directory")
        for child in target_path.iterdir():
            remove_owned_path(child)


def write_regular_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    target: Path,
) -> None:
    remove_owned_path(target)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, member.mode & 0o7777)
    written = 0
    try:
        source = archive.extractfile(member)
        if source is None:
            raise OciCacheError(f"OCI regular member has no data: {member.name}")
        with source:
            while True:
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                view = memoryview(block)
                while view:
                    count = os.write(descriptor, view)
                    if count <= 0:
                        raise OSError("short write while extracting OCI layer")
                    written += count
                    view = view[count:]
        if written != member.size:
            raise OciCacheError(f"OCI regular member size drifted: {member.name}")
        os.fchmod(descriptor, member.mode & 0o7777)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        target.unlink(missing_ok=True)
        raise
    os.close(descriptor)


def extract_layer(rootfs: Path, layer_tar: Path) -> None:
    try:
        archive = tarfile.open(layer_tar, mode="r:")
    except (OSError, tarfile.TarError) as error:
        raise OciCacheError("OCI layer is not a valid tar archive") from error
    with archive:
        try:
            members = validate_tar_members(archive.getmembers())
            apply_whiteouts(rootfs, members)
            hardlinks: list[tuple[tarfile.TarInfo, str]] = []
            for member, member_path, whiteout in members:
                if whiteout is not None:
                    continue
                ensure_parent_directories(rootfs, member_path)
                target = rootfs / member_path
                if member.isdir():
                    try:
                        info = target.lstat()
                    except FileNotFoundError:
                        target.mkdir(mode=member.mode & 0o7777)
                    else:
                        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                            remove_owned_path(target)
                            target.mkdir(mode=member.mode & 0o7777)
                    os.chmod(target, member.mode & 0o7777, follow_symlinks=False)
                elif member.isreg():
                    write_regular_member(archive, member, target)
                elif member.issym():
                    remove_owned_path(target)
                    os.symlink(member.linkname, target)
                elif member.islnk():
                    hardlinks.append((member, member_path))
            for member, member_path in hardlinks:
                target = rootfs / member_path
                source = rootfs / validate_hardlink_target(member.linkname)
                try:
                    source_info = source.lstat()
                except OSError as error:
                    raise OciCacheError("OCI hardlink target is unavailable") from error
                if stat.S_ISLNK(source_info.st_mode) or not stat.S_ISREG(
                    source_info.st_mode
                ):
                    raise OciCacheError("OCI hardlink target is not a regular file")
                remove_owned_path(target)
                os.link(source, target, follow_symlinks=False)
        except (OciCacheError, OSError, tarfile.TarError) as error:
            if isinstance(error, OciCacheError):
                raise
            raise OciCacheError("OCI layer extraction failed") from error


def decompress_layer(layer: OciLayer, destination: Path) -> Path:
    verify_blob(layer.blob_path, layer.digest, layer.size, "OCI layer blob")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    digest = hashlib.sha256()
    try:
        if layer.media_type in GZIP_LAYER_MEDIA_TYPES:
            source = gzip.open(layer.blob_path, "rb")
        elif layer.media_type in PLAIN_LAYER_MEDIA_TYPES:
            source = layer.blob_path.open("rb")
        else:
            raise OciCacheError("OCI layer media type is unsupported")
        with source:
            while True:
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                view = memoryview(block)
                while view:
                    count = os.write(descriptor, view)
                    if count <= 0:
                        raise OSError("short write while decompressing OCI layer")
                    view = view[count:]
        os.fsync(descriptor)
    except BaseException as error:
        os.close(descriptor)
        destination.unlink(missing_ok=True)
        if isinstance(error, OciCacheError):
            raise
        raise OciCacheError("OCI layer decompression failed") from error
    os.close(descriptor)
    if digest.hexdigest() != layer.diff_id[7:]:
        destination.unlink(missing_ok=True)
        raise OciCacheError("OCI layer uncompressed diff ID drifted")
    return destination


def resolve_rootfs_path(rootfs: Path, relative_path: str) -> Path:
    pending = list(PurePosixPath(relative_path).parts)
    resolved_parts: list[str] = []
    symlink_count = 0
    while pending:
        part = pending.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            if not resolved_parts:
                raise OciCacheError("rootfs symlink escaped the root")
            resolved_parts.pop()
            continue
        candidate = rootfs.joinpath(*resolved_parts, part)
        try:
            info = candidate.lstat()
        except OSError as error:
            raise OciCacheError(
                f"required rootfs path is missing: {relative_path}"
            ) from error
        if stat.S_ISLNK(info.st_mode):
            symlink_count += 1
            if symlink_count > 40:
                raise OciCacheError("rootfs symlink chain is too deep")
            target = os.readlink(candidate)
            if target.startswith("/"):
                resolved_parts = []
            target_parts = list(PurePosixPath(target).parts)
            if target.startswith("/") and target_parts and target_parts[0] == "/":
                target_parts = target_parts[1:]
            pending = target_parts + pending
            continue
        resolved_parts.append(part)
        if pending and not stat.S_ISDIR(info.st_mode):
            raise OciCacheError("required rootfs parent is not a directory")
    return rootfs.joinpath(*resolved_parts)


def validate_runtime_rootfs(rootfs: Path) -> None:
    for relative_path in ("testbed", "tmp", "var/tmp", "dev", "proc", "run"):
        resolved = resolve_rootfs_path(rootfs, relative_path)
        if not stat.S_ISDIR(resolved.lstat().st_mode):
            raise OciCacheError(f"required rootfs directory drifted: {relative_path}")
    for relative_path in (
        "bin/bash",
        "usr/bin/setpriv",
        "usr/bin/prlimit",
        "usr/bin/env",
        "bin/sleep",
        "usr/bin/cut",
    ):
        resolved = resolve_rootfs_path(rootfs, relative_path)
        info = resolved.lstat()
        if not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o111:
            raise OciCacheError(f"required rootfs executable drifted: {relative_path}")


def rootfs_tree_entries(rootfs: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, prefix: str) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda child: child.name)
        except OSError as error:
            raise OciCacheError("cannot enumerate extracted rootfs") from error
        with_children = children
        for child in with_children:
            try:
                child.name.encode("utf-8", errors="strict")
                info = child.stat(follow_symlinks=False)
            except (OSError, UnicodeEncodeError) as error:
                raise OciCacheError("rootfs contains an unreadable path") from error
            relative_path = child.name if not prefix else f"{prefix}/{child.name}"
            common = {
                "path": relative_path,
                "mode": stat.S_IMODE(info.st_mode),
                "size": info.st_size,
                "inode": info.st_ino,
                "ctime_ns": info.st_ctime_ns,
            }
            child_path = Path(child.path)
            if stat.S_ISREG(info.st_mode):
                entries.append(
                    {
                        **common,
                        "type": "file",
                        "sha256": sha256_file(child_path),
                    }
                )
            elif stat.S_ISDIR(info.st_mode):
                entries.append({**common, "type": "directory"})
                visit(child_path, relative_path)
            elif stat.S_ISLNK(info.st_mode):
                entries.append(
                    {
                        **common,
                        "type": "symlink",
                        "target": os.readlink(child_path),
                    }
                )
            else:
                raise OciCacheError(
                    f"rootfs contains a forbidden special file: {relative_path}"
                )

    visit(rootfs, "")
    return entries


def fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for current, directory_names, _ in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.append(current_path)
        retained: list[str] = []
        for name in directory_names:
            child = current_path / name
            if not stat.S_ISLNK(child.lstat().st_mode):
                retained.append(name)
        directory_names[:] = retained
    for directory in reversed(directories):
        fsync_directory(directory)


def recover_stale_partials(
    cache_root: Path | str,
    manifest_digest: str,
) -> tuple[str, ...]:
    digest = require_sha256_digest(manifest_digest, "OCI manifest digest")
    root = ensure_private_directory(cache_root)
    prefix = f".sha256-{digest[7:]}.partial-"
    return recover_scoped_partials(root, prefix, label="OCI rootfs cache")


def attest_rootfs(cache_directory: Path | str) -> dict[str, Any]:
    cache = require_real_directory(Path(cache_directory), "OCI rootfs cache entry")
    expected = read_json_file(cache / "rootfs-manifest.json", "rootfs manifest")
    binding = read_json_file(cache / "binding.json", "rootfs binding")
    if not isinstance(expected, Mapping) or expected.get("schema") != (
        "swebench_verified_rootfs_manifest_v1"
    ):
        raise OciCacheError("rootfs manifest schema drifted")
    if not isinstance(binding, Mapping) or binding.get("schema") != (
        "swebench_verified_oci_binding_v1"
    ):
        raise OciCacheError("rootfs binding schema drifted")
    rootfs = require_real_directory(cache / "rootfs", "extracted rootfs")
    entries = rootfs_tree_entries(rootfs)
    tree_sha256 = hashlib.sha256(canonical_json_bytes(entries)).hexdigest()
    if expected.get("entries") != entries:
        raise OciCacheError("rootfs full-tree attestation drifted")
    if expected.get("tree_sha256") != tree_sha256:
        raise OciCacheError("rootfs tree digest drifted")
    if expected.get("manifest_digest") != binding.get("manifest_digest"):
        raise OciCacheError("rootfs manifest binding drifted")
    return {
        "schema": "swebench_verified_rootfs_attestation_v1",
        "status": "pass",
        "image": binding.get("image"),
        "manifest_digest": binding.get("manifest_digest"),
        "config_digest": binding.get("config_digest"),
        "path_count": len(entries),
        "tree_sha256": tree_sha256,
    }


def materialize_rootfs(
    binding: OciImageBinding,
    cache_root: Path | str,
) -> Path:
    if not isinstance(binding, OciImageBinding):
        raise TypeError("rootfs materialization requires an OCI image binding")
    root = ensure_private_directory(cache_root)
    final = root / binding.cache_name
    lock = root / f".{binding.cache_name}.lock"
    with exclusive_lock(lock):
        recover_stale_partials(root, binding.manifest_digest)
        if path_exists(final):
            info = final.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise OciCacheError("OCI rootfs cache entry is not a real directory")
            receipt = attest_rootfs(final)
            if (
                receipt["image"] != binding.image
                or receipt["manifest_digest"] != binding.manifest_digest
                or receipt["config_digest"] != binding.config_digest
            ):
                raise OciCacheError("OCI rootfs cache is bound to another image")
            return final

        partial = root / (
            f".{binding.cache_name}.partial-{os.getpid()}-{secrets.token_hex(8)}"
        )
        partial.mkdir(mode=0o700)
        rootfs = partial / "rootfs"
        rootfs.mkdir(mode=0o755)
        try:
            for index, layer in enumerate(binding.layers):
                layer_tar = partial / f".layer-{index:04d}.tar"
                decompress_layer(layer, layer_tar)
                try:
                    extract_layer(rootfs, layer_tar)
                finally:
                    layer_tar.unlink(missing_ok=True)
            validate_runtime_rootfs(rootfs)
            entries = rootfs_tree_entries(rootfs)
            manifest = {
                "schema": "swebench_verified_rootfs_manifest_v1",
                "image": binding.image,
                "manifest_digest": binding.manifest_digest,
                "config_digest": binding.config_digest,
                "path_count": len(entries),
                "tree_sha256": hashlib.sha256(
                    canonical_json_bytes(entries)
                ).hexdigest(),
                "entries": entries,
            }
            atomic_write_json(partial / "binding.json", binding.receipt())
            atomic_write_json(partial / "rootfs-manifest.json", manifest)
            fsync_tree(partial)
            os.replace(partial, final)
            fsync_directory(root)
            attest_rootfs(final)
            return final
        except BaseException:
            if path_exists(partial):
                partial_info = partial.lstat()
                if stat.S_ISLNK(partial_info.st_mode) or not stat.S_ISDIR(
                    partial_info.st_mode
                ):
                    raise OciCacheError("OCI partial rootfs changed type")
                shutil.rmtree(partial)
                fsync_directory(root)
            raise


def add_bytes_to_tar(
    archive: tarfile.TarFile,
    name: str,
    payload: bytes,
) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mode = 0o644
    member.uid = 0
    member.gid = 0
    member.mtime = 0
    archive.addfile(member, io.BytesIO(payload))


def add_file_to_tar(
    archive: tarfile.TarFile,
    name: str,
    path: Path,
) -> None:
    member = tarfile.TarInfo(name)
    member.size = path.stat().st_size
    member.mode = 0o644
    member.uid = 0
    member.gid = 0
    member.mtime = 0
    with path.open("rb") as stream:
        archive.addfile(member, stream)


def build_docker_archive(
    binding: OciImageBinding,
    archive_path: Path | str,
) -> Path:
    """Build a Docker-loadable archive using only verified cached blobs."""

    if not isinstance(binding, OciImageBinding):
        raise TypeError("Docker archive construction requires an OCI binding")
    target = Path(archive_path)
    parent = ensure_private_directory(target.parent)
    target = parent / target.name
    if path_exists(target):
        raise OciCacheError(f"Docker archive already exists: {target}")
    temporary = parent / (
        f".{target.name}.partial-{os.getpid()}-{secrets.token_hex(8)}"
    )
    layer_names: list[str] = []
    try:
        with tarfile.open(temporary, mode="w") as archive:
            config_raw = require_real_file(
                binding.config_path, "OCI config blob"
            ).read_bytes()
            if hashlib.sha256(config_raw).hexdigest() != binding.config_digest[7:]:
                raise OciCacheError("OCI config drifted before archive construction")
            config_name = f"{binding.config_digest[7:]}.json"
            add_bytes_to_tar(archive, config_name, config_raw)
            for index, layer in enumerate(binding.layers):
                uncompressed = parent / (
                    f".{target.name}.layer-{index:04d}-{secrets.token_hex(8)}.tar"
                )
                decompress_layer(layer, uncompressed)
                try:
                    layer_name = f"{index:04d}-{layer.digest[7:]}/layer.tar"
                    add_file_to_tar(archive, layer_name, uncompressed)
                    layer_names.append(layer_name)
                finally:
                    uncompressed.unlink(missing_ok=True)
            docker_manifest = [
                {
                    "Config": config_name,
                    "RepoTags": [binding.image],
                    "Layers": layer_names,
                }
            ]
            add_bytes_to_tar(
                archive,
                "manifest.json",
                canonical_json_bytes(docker_manifest),
            )
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        os.chmod(target, 0o600, follow_symlinks=False)
        fsync_directory(parent)
        atomic_write_json(
            parent / f"{target.name}.receipt.json",
            {
                "schema": "swebench_verified_docker_archive_v1",
                "image": binding.image,
                "manifest_digest": binding.manifest_digest,
                "config_digest": binding.config_digest,
                "archive_size": target.stat().st_size,
                "archive_sha256": sha256_file(target),
            },
        )
        return target
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


Executor = Callable[[list[str]], subprocess.CompletedProcess]


class DockerCli:
    """Fail-closed wrapper for the evaluation's isolated Docker daemon."""

    def __init__(
        self,
        *,
        socket_path: Path | str,
        executable: str = "docker",
        executor: Executor | None = None,
        verify_socket: bool = True,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.executable = executable
        self.executor = executor or self.default_executor
        if verify_socket:
            try:
                info = self.socket_path.lstat()
            except OSError as error:
                raise OciCacheError("isolated Docker socket is unavailable") from error
            if not stat.S_ISSOCK(info.st_mode):
                raise OciCacheError("isolated Docker path is not a Unix socket")

    @staticmethod
    def default_executor(argv: list[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise OciCacheError("Docker CLI execution failed") from error

    def command(self, *arguments: str) -> list[str]:
        return [
            self.executable,
            "--host",
            f"unix://{self.socket_path}",
            *arguments,
        ]

    def run(self, *arguments: str) -> subprocess.CompletedProcess:
        result = self.executor(self.command(*arguments))
        if not isinstance(result, subprocess.CompletedProcess):
            raise OciCacheError("Docker executor returned an invalid result")
        return result

    def inspect(self, image: str) -> subprocess.CompletedProcess:
        return self.run("image", "inspect", "--format", "{{.Id}}", image)

    @staticmethod
    def output_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value if isinstance(value, str) else ""

    @classmethod
    def is_missing_image(cls, result: subprocess.CompletedProcess) -> bool:
        if result.returncode == 0:
            return False
        stderr = cls.output_text(result.stderr).lower()
        return "no such image" in stderr or "no such object" in stderr

    def ensure_loaded(
        self,
        binding: OciImageBinding,
        archive_path: Path | str,
    ) -> dict[str, Any]:
        current = self.inspect(binding.image)
        if current.returncode == 0:
            image_id = self.output_text(current.stdout).strip()
            if image_id != binding.config_digest:
                raise OciCacheError("Docker image alias resolved to another config ID")
            return {
                "schema": "swebench_verified_docker_load_v1",
                "status": "already_loaded",
                "image": binding.image,
                "image_id": image_id,
            }
        if not self.is_missing_image(current):
            raise OciCacheError("Docker image inspect failed unexpectedly")
        archive = require_real_file(Path(archive_path), "Docker image archive")
        loaded = self.run("image", "load", "--input", str(archive))
        if loaded.returncode != 0:
            raise OciCacheError("Docker image load failed")
        verified = self.inspect(binding.image)
        image_id = self.output_text(verified.stdout).strip()
        if verified.returncode != 0 or image_id != binding.config_digest:
            raise OciCacheError("Docker loaded image config ID drifted")
        return {
            "schema": "swebench_verified_docker_load_v1",
            "status": "loaded",
            "image": binding.image,
            "image_id": image_id,
            "archive_sha256": sha256_file(archive),
        }

    def evict_loaded(self, binding: OciImageBinding) -> dict[str, Any]:
        current = self.inspect(binding.image)
        if current.returncode != 0:
            if self.is_missing_image(current):
                return {
                    "schema": "swebench_verified_docker_eviction_v1",
                    "status": "already_absent",
                    "image": binding.image,
                }
            raise OciCacheError("Docker image inspect failed during eviction")
        if self.output_text(current.stdout).strip() != binding.config_digest:
            raise OciCacheError("refusing to evict a mismatched Docker image alias")
        containers = self.run(
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            f"ancestor={binding.image}",
        )
        if containers.returncode != 0:
            raise OciCacheError("Docker container census failed")
        if self.output_text(containers.stdout).strip():
            raise OciCacheError("Docker image still has task containers")
        removed = self.run("image", "rm", binding.image)
        if removed.returncode != 0:
            raise OciCacheError("Docker image eviction failed")
        after = self.inspect(binding.image)
        if not self.is_missing_image(after):
            raise OciCacheError("Docker image alias survived eviction")
        return {
            "schema": "swebench_verified_docker_eviction_v1",
            "status": "evicted",
            "image": binding.image,
            "image_id": binding.config_digest,
        }


def run_git(argv: list[str], label: str) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise OciCacheError(f"{label} could not run git") from error
    if result.returncode != 0:
        raise OciCacheError(f"{label} failed: {result.stderr.strip()}")
    return result


def recover_scoped_partials(
    root: Path,
    prefix: str,
    *,
    label: str,
) -> tuple[str, ...]:
    removed: list[str] = []
    for candidate in sorted(root.iterdir(), key=lambda path: path.name):
        if not candidate.name.startswith(prefix):
            continue
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OciCacheError(f"{label} partial is not a real directory")
        shutil.rmtree(candidate)
        removed.append(candidate.name)
    if removed:
        fsync_directory(root)
    return tuple(removed)


def ensure_repository_mirror(
    source_rootfs: Path | str,
    mirror_root: Path | str,
    *,
    repo: str,
    base_commit: str,
) -> Path:
    if REPOSITORY_PATTERN.fullmatch(repo) is None:
        raise OciCacheError("repository identity is unsafe")
    if re.fullmatch(r"[0-9a-f]{40}", base_commit) is None:
        raise OciCacheError("dataset base commit must be lowercase hexadecimal")
    rootfs = require_real_directory(Path(source_rootfs), "source rootfs")
    testbed = require_real_directory(rootfs / "testbed", "rootfs testbed")
    require_real_directory(testbed / ".git", "rootfs Git metadata")
    inside = run_git(
        ["git", "-C", str(testbed), "rev-parse", "--is-inside-work-tree"],
        "rootfs repository check",
    ).stdout.strip()
    if inside != "true":
        raise OciCacheError("rootfs testbed is not a Git worktree")
    resolved = run_git(
        ["git", "-C", str(testbed), "rev-parse", f"{base_commit}^{{commit}}"],
        "rootfs base-commit check",
    ).stdout.strip()
    head = run_git(
        ["git", "-C", str(testbed), "rev-parse", "HEAD^{commit}"],
        "rootfs HEAD check",
    ).stdout.strip()
    if resolved != base_commit:
        raise OciCacheError("rootfs dataset base commit drifted")
    run_git(
        [
            "git",
            "-C",
            str(testbed),
            "merge-base",
            "--is-ancestor",
            base_commit,
            head,
        ],
        "rootfs base commit ancestry check",
    )

    root = ensure_private_directory(mirror_root)
    mirror_name = repo.replace("/", "__")
    mirror = root / mirror_name
    lock = root / f".{mirror_name}.lock"
    with exclusive_lock(lock):
        recover_scoped_partials(
            root,
            f".{mirror_name}.partial-",
            label="repository mirror",
        )
        if path_exists(mirror):
            require_real_directory(mirror, "repository mirror")
            bare = run_git(
                ["git", "--git-dir", str(mirror), "rev-parse", "--is-bare-repository"],
                "repository mirror check",
            ).stdout.strip()
            if bare != "true":
                raise OciCacheError("repository mirror is not bare")
            probe = subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(mirror),
                    "cat-file",
                    "-e",
                    f"{base_commit}^{{commit}}",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if probe.returncode != 0:
                run_git(
                    [
                        "git",
                        "--git-dir",
                        str(mirror),
                        "fetch",
                        "--force",
                        "--no-tags",
                        str(testbed),
                        base_commit,
                    ],
                    "repository mirror local fetch",
                )
                run_git(
                    [
                        "git",
                        "--git-dir",
                        str(mirror),
                        "update-ref",
                        f"refs/amg-staged/{base_commit}",
                        base_commit,
                    ],
                    "repository mirror base ref",
                )
        else:
            partial = root / (
                f".{mirror_name}.partial-{os.getpid()}-{secrets.token_hex(8)}"
            )
            run_git(
                [
                    "git",
                    "clone",
                    "--mirror",
                    "--local",
                    "--no-hardlinks",
                    "--",
                    str(testbed),
                    str(partial),
                ],
                "repository mirror clone",
            )
            os.chmod(partial, 0o700, follow_symlinks=False)
            os.replace(partial, mirror)
            fsync_directory(root)
        mirror_commit = run_git(
            ["git", "--git-dir", str(mirror), "rev-parse", f"{base_commit}^{{commit}}"],
            "repository mirror base-commit verification",
        ).stdout.strip()
        if mirror_commit != base_commit:
            raise OciCacheError("repository mirror base commit drifted")
        return mirror


def require_task_eviction_ready(
    instance_id: str,
    accepted_cells: Sequence[Mapping[str, Any]],
    official_outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(instance_id, str) or not instance_id:
        raise OciCacheError("task eviction requires an instance ID")
    accepted_by_arm: dict[str, Mapping[str, Any]] = {}
    for row in accepted_cells:
        if not isinstance(row, Mapping):
            raise OciCacheError("accepted cell is not an object")
        if row.get("instance_id") != instance_id or row.get("status") != "accepted":
            raise OciCacheError("accepted cell is not bound to the task")
        arm = row.get("arm")
        if arm not in ARMS or arm in accepted_by_arm:
            raise OciCacheError("accepted triad arms are incomplete or duplicated")
        accepted_by_arm[arm] = row
    if tuple(accepted_by_arm) != ARMS:
        if set(accepted_by_arm) != set(ARMS):
            raise OciCacheError("accepted triad is incomplete")

    outcomes_by_arm: dict[str, Mapping[str, Any]] = {}
    for row in official_outcomes:
        if not isinstance(row, Mapping) or row.get("instance_id") != instance_id:
            raise OciCacheError("official outcome is not bound to the task")
        arm = row.get("arm")
        if arm not in ARMS or arm in outcomes_by_arm:
            raise OciCacheError("official outcome arms are incomplete or duplicated")
        if type(row.get("resolved")) is not bool:
            raise OciCacheError("official task outcome must be boolean")
        outcomes_by_arm[arm] = row
    if set(outcomes_by_arm) != set(ARMS):
        raise OciCacheError("official task outcomes are incomplete")
    return {
        "schema": "swebench_verified_task_eviction_ready_v1",
        "status": "pass",
        "instance_id": instance_id,
        "arms": list(ARMS),
        "resolved": [outcomes_by_arm[arm]["resolved"] for arm in ARMS],
    }


__all__ = [
    "CachedOciStore",
    "DockerCli",
    "OciCacheError",
    "OciImageBinding",
    "OciLayer",
    "attest_rootfs",
    "build_docker_archive",
    "ensure_repository_mirror",
    "materialize_rootfs",
    "recover_stale_partials",
    "require_task_eviction_ready",
]
