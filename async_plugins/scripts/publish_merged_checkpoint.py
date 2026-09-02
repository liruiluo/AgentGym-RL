#!/usr/bin/env python3
"""Verify, merge, and immutably publish one CAMG FSDP checkpoint.

The held-out evaluator consumes a Hugging Face checkpoint plus a byte-level
manifest.  This helper keeps that publication fail-closed: it verifies the
complete actor/critic FSDP checkpoint first, merges only the actor into a
same-filesystem staging directory, atomically installs the model directory,
and finally writes the manifest expected by ``heldout_eval.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


OUTER_ROOT = Path(__file__).resolve().parents[2]
ASYNC_PLUGINS = OUTER_ROOT / "async_plugins"
CHECKPOINT_VERIFIER = (
    OUTER_ROOT / "AgentGym-RL/scripts/agentmemory/verify_fsdp_checkpoint.py"
)
MODEL_MANIFEST_SCHEMA = "camg_merged_hf_checkpoint_manifest_v1"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class PublicationError(RuntimeError):
    """Raised when a checkpoint publication is incomplete or ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise PublicationError("manifest path must be an absolute non-symlink path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise PublicationError(f"temporary manifest path already exists: {temporary}")
    temporary.write_bytes(payload)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _load_checkpoint_verifier() -> ModuleType:
    if not CHECKPOINT_VERIFIER.is_file() or CHECKPOINT_VERIFIER.is_symlink():
        raise PublicationError(
            f"pinned FSDP checkpoint verifier is unavailable: {CHECKPOINT_VERIFIER}"
        )
    spec = importlib.util.spec_from_file_location(
        "camg_publish_verify_fsdp_checkpoint", CHECKPOINT_VERIFIER
    )
    if spec is None or spec.loader is None:
        raise PublicationError("cannot import the pinned FSDP checkpoint verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_checkpoint(
    checkpoint_root: Path, *, checkpoint_step: int, world_size: int
) -> dict[str, Any]:
    if not checkpoint_root.is_absolute() or checkpoint_root.is_symlink():
        raise PublicationError(
            "checkpoint root must be an absolute non-symlink directory"
        )
    verifier = _load_checkpoint_verifier()
    try:
        report = verifier.verify_checkpoint(
            checkpoint_root,
            step=checkpoint_step,
            world_size=world_size,
        )
    except Exception as exc:
        raise PublicationError(f"FSDP checkpoint verification failed: {exc}") from exc
    actor = checkpoint_root / f"global_step_{checkpoint_step}" / "actor"
    fsdp_config = actor / "fsdp_config.json"
    try:
        payload = json.loads(fsdp_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"invalid actor fsdp_config.json: {exc}") from exc
    if payload.get("world_size") != world_size:
        raise PublicationError(
            "actor fsdp_config world size does not match the publication contract"
        )
    return report


def inventory_model(model_path: Path) -> list[dict[str, Any]]:
    if not model_path.is_absolute() or model_path.is_symlink() or not model_path.is_dir():
        raise PublicationError("merged model path must be an absolute non-symlink directory")
    records: list[dict[str, Any]] = []
    for candidate in sorted(model_path.rglob("*")):
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PublicationError(f"merged model contains a symlink: {candidate}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise PublicationError(
                f"merged model contains a non-regular or empty payload: {candidate}"
            )
        records.append(
            {
                "path": candidate.relative_to(model_path).as_posix(),
                "bytes": metadata.st_size,
                "sha256": sha256_file(candidate),
            }
        )
    names = {record["path"] for record in records}
    if "config.json" not in names:
        raise PublicationError("merged model payload lacks config.json")
    if not any(name.endswith(".safetensors") for name in names):
        raise PublicationError("merged model payload lacks safetensors weights")
    return records


def _verify_existing_publication(
    *,
    model_path: Path,
    manifest_path: Path,
    checkpoint_step: int,
    training_run_id: str,
    source_commits: dict[str, str],
) -> dict[str, Any]:
    sys.path.insert(0, str(ASYNC_PLUGINS))
    from agentmemorygym_verl.heldout_eval import verify_model_manifest

    manifest_sha256 = sha256_file(manifest_path)
    verified = verify_model_manifest(
        manifest_path,
        expected_manifest_sha256=manifest_sha256,
        expected_checkpoint_step=checkpoint_step,
        expected_training_run_id=training_run_id,
        expected_source_commits=source_commits,
    )
    if verified.path != model_path.resolve():
        raise PublicationError("existing model manifest points at another model path")
    return {
        "schema": "camg_merged_hf_checkpoint_publication_result_v1",
        "status": "pass",
        "publication": "reused_verified",
        "model_path": str(verified.path),
        "model_manifest": str(verified.manifest_path),
        "model_manifest_sha256": manifest_sha256,
        "file_count": verified.file_count,
        "checkpoint_step": verified.checkpoint_step,
    }


def publish_checkpoint(
    *,
    checkpoint_root: Path,
    checkpoint_step: int,
    world_size: int,
    model_path: Path,
    manifest_path: Path,
    merge_log_path: Path,
    training_run_id: str,
    source_commits: dict[str, str],
    verl_root: Path,
    python_executable: Path,
) -> dict[str, Any]:
    for label, path in (
        ("checkpoint root", checkpoint_root),
        ("model path", model_path),
        ("manifest path", manifest_path),
        ("merge log path", merge_log_path),
        ("veRL root", verl_root),
        ("Python executable", python_executable),
    ):
        if not path.is_absolute():
            raise PublicationError(f"{label} must be an absolute path")
    if checkpoint_step <= 0 or world_size <= 0:
        raise PublicationError("checkpoint step and world size must be positive")
    if not training_run_id.strip():
        raise PublicationError("training run id must not be empty")
    if set(source_commits) != {"outer", "inner", "verl"}:
        raise PublicationError("source commits must contain outer, inner, and verl")
    if any(not _COMMIT.fullmatch(value) for value in source_commits.values()):
        raise PublicationError("source commits must be full lowercase git commits")
    if not verl_root.is_dir() or verl_root.is_symlink():
        raise PublicationError("veRL root must be a non-symlink directory")
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        raise PublicationError("Python executable is unavailable")
    if manifest_path == model_path or model_path in manifest_path.parents:
        raise PublicationError("model manifest must live outside the model payload")
    if merge_log_path == model_path or model_path in merge_log_path.parents:
        raise PublicationError("merge log must live outside the model payload")

    model_exists = model_path.exists() or model_path.is_symlink()
    manifest_exists = manifest_path.exists() or manifest_path.is_symlink()
    if model_exists or manifest_exists:
        if not model_exists or not manifest_exists:
            raise PublicationError(
                "partial publication exists; model and manifest must appear together"
            )
        return _verify_existing_publication(
            model_path=model_path,
            manifest_path=manifest_path,
            checkpoint_step=checkpoint_step,
            training_run_id=training_run_id,
            source_commits=source_commits,
        )

    checkpoint_report = verify_checkpoint(
        checkpoint_root,
        checkpoint_step=checkpoint_step,
        world_size=world_size,
    )
    actor_path = checkpoint_root / f"global_step_{checkpoint_step}" / "actor"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    staging = model_path.with_name(f".{model_path.name}.staging-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise PublicationError(f"merge staging path already exists: {staging}")
    merge_log_path.parent.mkdir(parents=True, exist_ok=True)
    if merge_log_path.exists() or merge_log_path.is_symlink():
        raise PublicationError(f"merge log already exists: {merge_log_path}")

    command = [
        str(python_executable),
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_path),
        "--target_dir",
        str(staging),
    ]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(verl_root)
        if not existing_pythonpath
        else str(verl_root) + os.pathsep + existing_pythonpath
    )
    with merge_log_path.open("xb") as log_handle:
        subprocess.run(
            command,
            cwd=verl_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    files = inventory_model(staging)
    os.replace(staging, model_path)

    manifest = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "checkpoint_step": checkpoint_step,
        "training_run_id": training_run_id,
        "source_commits": source_commits,
        "checkpoint_source": {
            "checkpoint_root": str(checkpoint_root.resolve()),
            "global_step_dir": checkpoint_report["global_step_dir"],
            "world_size": world_size,
            "tracker": checkpoint_report["tracker"],
        },
        "merge": {
            "backend": "fsdp",
            "verl_root": str(verl_root.resolve()),
            "python_executable": str(python_executable.resolve()),
            "log_path": str(merge_log_path.resolve()),
            "log_sha256": sha256_file(merge_log_path),
        },
        "model_path": str(model_path.resolve()),
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write(manifest_path, canonical_json_bytes(manifest))
    result = _verify_existing_publication(
        model_path=model_path,
        manifest_path=manifest_path,
        checkpoint_step=checkpoint_step,
        training_run_id=training_run_id,
        source_commits=source_commits,
    )
    result["publication"] = "created"
    result["checkpoint_verification"] = checkpoint_report
    result["merge_log_sha256"] = manifest["merge"]["log_sha256"]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=200)
    parser.add_argument("--world-size", type=int, default=6)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--merge-log", type=Path, required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--training-outer-commit", required=True)
    parser.add_argument("--training-inner-commit", required=True)
    parser.add_argument("--training-verl-commit", required=True)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = publish_checkpoint(
        checkpoint_root=arguments.checkpoint_root,
        checkpoint_step=arguments.checkpoint_step,
        world_size=arguments.world_size,
        model_path=arguments.model_path,
        manifest_path=arguments.manifest,
        merge_log_path=arguments.merge_log,
        training_run_id=arguments.training_run_id,
        source_commits={
            "outer": arguments.training_outer_commit,
            "inner": arguments.training_inner_commit,
            "verl": arguments.training_verl_commit,
        },
        verl_root=arguments.verl_root,
        python_executable=arguments.python,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        PublicationError,
        OSError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(json.dumps({"status": "fail", "error": str(error)}), file=sys.stderr)
        raise SystemExit(2)
