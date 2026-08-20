"""Immutable veRL/model identity and dynamic publication validation helpers.

The reviewed veRL commit and Qwen3.5 model bytes are treatment invariants.  The
OpenMLE publication, AgentGym commits, runtime paths, task counts, and schedule
hashes are *not* constants here: the launcher derives them from the selected,
completed publication and cross-checks every binding before launch.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

# Updated only after the generic veRL runtime-evidence, fused-forward
# instance-scope, upstream masked fused-head, and variable-response alignment
# patches are reviewed and committed. It is intentionally not caller-selectable.
EXPECTED_VERL_COMMIT = "88e17fbb07088b6085c5949e33cdc3b0f0ebc53d"

# Files needed to load the exact Qwen3.5-4B text model and tokenizer.  The model
# root itself comes from the publication's training_runtime section, so a newer
# publication may relocate the same immutable model without a plugin edit.
TRL_WHEEL_RELATIVE_PATH = "async_plugins/vendor/trl-0.9.6-py3-none-any.whl"
TRL_WHEEL_SHA256 = (
    "4753f190c94c11488fcc46ec74b2128e53fbc61d51f0887b7204ec4dc333af4b"
)


LOCKED_MODEL_FILE_SHA256 = {
    "model-00001-of-00002.safetensors": (
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"
    ),
    "model-00002-of-00002.safetensors": (
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"
    ),
    "model.safetensors.index.json": (
        "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d"
    ),
    "config.json": ("ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"),
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "tokenizer_config.json": (
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
    ),
    "chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    ),
}


_AMBIENT_IDENTITY_NAMES = frozenset(
    {
        "PYTHONPATH",
        "VERL_USE_EXTERNAL_MODULES",
        "VERL_USE_EXTERNAL_PLUGINS",
        "VERL_FILE_LOGGER_PATH",
        "VERL_FULLY_ASYNC_RUNTIME_RECEIPT_PATH",
        "AMG_ENDPOINT_CLIENT_CONFIG_JSON",
        "AMG_ENV_ADDR",
        "AGENTMEMORY_MODEL_PATH",
        "MODEL_PATH",
        "OPENMLE_FAST_TASK_MANIFEST_SHA256",
        "OPENMLE_FAST_RELEASE_REVISION",
        "OPENMLE_FAST_RUNTIME_OUTER_COMMIT",
        "OPENMLE_FAST_RUNTIME_INNER_COMMIT",
        "OPENMLE_FAST_MANIFEST_ROLE",
        "OPENMLE_FAST_EXECUTOR_RUNTIME_DIGEST",
        "OPENMLE_FAST_MATERIALIZER_SHA256",
        "OPENMLE_FAST_ACTIONS_SHA256",
        "OPENMLE_FAST_MAX_OBSERVATION_TOKENS",
    }
)


def sha256_file(path: Path) -> str:
    """Return a file digest without loading multi-gigabyte weights in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_ambient_identity(environment: Mapping[str, str]) -> None:
    """Reject inherited values that could replace a publication-derived identity."""

    conflicts = sorted(name for name in _AMBIENT_IDENTITY_NAMES if name in environment)
    if conflicts:
        raise RuntimeError("ambient identity conflict: " + ", ".join(conflicts))


def validate_outer_change_paths(
    *,
    locked_outer_commit: str,
    ancestor_is_locked: bool,
    committed_paths: Sequence[str],
    dirty_paths: Sequence[str],
    require_clean: bool,
) -> dict[str, Any]:
    """Prove that all outer changes after the publication commit are plugin-only."""

    if not ancestor_is_locked:
        raise RuntimeError(
            f"publication outer commit {locked_outer_commit} is not an ancestor of HEAD"
        )

    normalized_committed = tuple(str(path) for path in committed_paths if str(path))
    normalized_dirty = tuple(str(path) for path in dirty_paths if str(path))
    for path in (*normalized_committed, *normalized_dirty):
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or len(pure.parts) < 2
            or pure.parts[0] != "async_plugins"
        ):
            raise RuntimeError(f"outer source change is outside async_plugins/: {path}")
    if require_clean and normalized_dirty:
        raise RuntimeError("AMG plugin runtime tree must be clean")

    return {
        "publication_outer_commit": locked_outer_commit,
        "publication_ancestor": True,
        "committed_paths": list(normalized_committed),
        "committed_path_count": len(normalized_committed),
        "dirty_paths": list(normalized_dirty),
        "dirty_path_count": len(normalized_dirty),
        "clean_required": require_clean,
    }


def _absolute_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise RuntimeError(f"training runtime {field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"training runtime {field} must be absolute: {value!r}")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"training runtime {field} must be a lowercase SHA-256")
    return value


def validate_training_runtime_lock(observed: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the publication-owned training runtime identity.

    This deliberately validates shape and current AMG hardware requirements
    rather than comparing against one dated publication's paths or digest.
    """

    if not isinstance(observed, Mapping):
        raise TypeError("training runtime identity must be a mapping")
    normalized = {
        "base_model": _absolute_path(observed.get("base_model"), field="base_model"),
        "python": _absolute_path(observed.get("python"), field="python"),
        "site_packages": _absolute_path(
            observed.get("site_packages"), field="site_packages"
        ),
        "bundle_sha256": _sha256(observed.get("bundle_sha256"), field="bundle_sha256"),
        "bundle_sha256_file": _absolute_path(
            observed.get("bundle_sha256_file"), field="bundle_sha256_file"
        ),
    }
    gpu_count = observed.get("gpu_count")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count != 8:
        raise RuntimeError(f"training runtime gpu_count must be 8, got {gpu_count!r}")
    gpu_type = observed.get("gpu_type")
    if gpu_type != "B200":
        raise RuntimeError(
            f"training runtime gpu_type must be 'B200', got {gpu_type!r}"
        )
    normalized.update(gpu_count=gpu_count, gpu_type=gpu_type)
    return normalized


def verify_hash_manifest(
    root: str | os.PathLike[str], expected: Mapping[str, str]
) -> dict[str, str]:
    """Fail closed unless every selected regular file has its expected digest."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"hash-manifest root is missing: {root_path}")
    verified: dict[str, str] = {}
    for relative, expected_digest in expected.items():
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError(f"unsafe hash-manifest path: {relative!r}")
        if len(expected_digest) != 64 or any(
            character not in "0123456789abcdef" for character in expected_digest
        ):
            raise ValueError(f"invalid expected hash for {relative!r}")
        path = root_path.joinpath(*pure.parts)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"locked file is missing or not regular: {path}")
        observed = sha256_file(path)
        if observed != expected_digest:
            raise RuntimeError(
                f"hash mismatch for {path}: expected {expected_digest}, got {observed}"
            )
        verified[str(relative)] = observed
    return verified


__all__ = [
    "EXPECTED_VERL_COMMIT",
    "LOCKED_MODEL_FILE_SHA256",
    "TRL_WHEEL_RELATIVE_PATH",
    "TRL_WHEEL_SHA256",
    "reject_ambient_identity",
    "sha256_file",
    "validate_outer_change_paths",
    "validate_training_runtime_lock",
    "verify_hash_manifest",
]
