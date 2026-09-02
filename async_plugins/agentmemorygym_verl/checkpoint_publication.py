"""Fail-closed helpers for publishing an AMG FSDP actor as merged HF files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

MODEL_MANIFEST_SCHEMA = "camg_merged_hf_checkpoint_manifest_v1"
CHECKPOINT_INSPECTION_SCHEMA = "camg_fsdp_actor_checkpoint_inspection_v1"
EXPECTED_CHECKPOINT_STEP = 400
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, field: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} must be a regular non-symlink file: {path}")
    return path


def _absolute_directory(path: str | os.PathLike[str], *, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"{field} must be an absolute non-symlink directory")
    return candidate.resolve()


def _commit(value: str, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _COMMIT.fullmatch(normalized):
        raise ValueError(f"{field} must be a 40-character lowercase Git commit")
    return normalized


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def inspect_fsdp_actor_checkpoint(
    run_dir: str | os.PathLike[str],
    *,
    checkpoint_step: int = EXPECTED_CHECKPOINT_STEP,
) -> dict[str, Any]:
    """Verify the complete FSDP actor shard set at the declared endpoint."""

    if checkpoint_step != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("native held-out publication is permitted only at update400")
    run = _absolute_directory(run_dir, field="training run directory")
    latest = _regular_file(
        run / "checkpoints/latest_checkpointed_iteration.txt",
        field="latest checkpoint marker",
    )
    try:
        latest_step = int(latest.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ValueError("latest checkpoint marker is not an integer") from exc
    if latest_step != checkpoint_step:
        raise ValueError(
            f"latest checkpoint is {latest_step}, expected {checkpoint_step}"
        )

    actor = _absolute_directory(
        run / f"checkpoints/global_step_{checkpoint_step}/actor",
        field="FSDP actor checkpoint",
    )
    fsdp_config_path = _regular_file(
        actor / "fsdp_config.json", field="FSDP actor config"
    )
    try:
        fsdp_config = json.loads(fsdp_config_path.read_text(encoding="utf-8"))
        world_size = int(fsdp_config["world_size"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid FSDP actor config") from exc
    if world_size <= 0:
        raise ValueError("FSDP actor world_size must be positive")

    expected_model_names = {
        f"model_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
    }
    observed_model_names = {
        path.name for path in actor.glob("model_world_size_*_rank_*.pt")
    }
    if observed_model_names != expected_model_names:
        raise ValueError(
            "FSDP model shard set mismatch: "
            f"missing={sorted(expected_model_names - observed_model_names)!r} "
            f"extra={sorted(observed_model_names - expected_model_names)!r}"
        )
    model_files = []
    for name in sorted(expected_model_names):
        path = _regular_file(actor / name, field=f"FSDP model shard {name}")
        if path.stat().st_size <= 0:
            raise ValueError(f"FSDP model shard is empty: {name}")
        model_files.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )

    huggingface = _absolute_directory(
        actor / "huggingface", field="checkpoint Hugging Face metadata"
    )
    _regular_file(huggingface / "config.json", field="checkpoint config.json")
    hf_files = []
    for path in sorted(huggingface.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint Hugging Face metadata has a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(actor).as_posix()
            hf_files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not hf_files:
        raise ValueError("checkpoint Hugging Face metadata is empty")
    return {
        "schema": CHECKPOINT_INSPECTION_SCHEMA,
        "status": "pass",
        "run_dir": str(run),
        "checkpoint_step": checkpoint_step,
        "actor_path": str(actor),
        "world_size": world_size,
        "model_shards": model_files,
        "huggingface_metadata": hf_files,
        "latest_checkpoint_marker": {
            "path": str(latest),
            "bytes": latest.stat().st_size,
            "sha256": sha256_file(latest),
        },
    }


def build_merged_hf_manifest(
    model_path: str | os.PathLike[str],
    *,
    training_run_id: str,
    checkpoint_step: int,
    outer_commit: str,
    inner_commit: str,
    verl_commit: str,
) -> dict[str, Any]:
    """Hash an exact merged-HF directory using the evaluator's schema."""

    if checkpoint_step != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("native held-out publication is permitted only at update400")
    if not _RUN_ID.fullmatch(training_run_id):
        raise ValueError("training_run_id contains unsupported characters or length")
    model = _absolute_directory(model_path, field="merged-HF model directory")
    files = []
    for path in sorted(model.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"merged-HF model contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(model).as_posix()
            if relative.startswith(".") or "/." in relative:
                raise ValueError(f"merged-HF model contains hidden metadata: {relative}")
            if path.stat().st_size <= 0:
                raise ValueError(f"merged-HF model contains an empty file: {relative}")
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    names = {entry["path"] for entry in files}
    if "config.json" not in names:
        raise ValueError("merged-HF model lacks config.json")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("merged-HF model lacks safetensors weights")
    return {
        "schema": MODEL_MANIFEST_SCHEMA,
        "checkpoint_step": checkpoint_step,
        "training_run_id": training_run_id,
        "source_commits": {
            "outer": _commit(outer_commit, field="outer commit"),
            "inner": _commit(inner_commit, field="inner commit"),
            "verl": _commit(verl_commit, field="veRL commit"),
        },
        "model_path": str(model),
        "files": files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-checkpoint")
    inspect.add_argument("--run-dir", type=Path, required=True)
    inspect.add_argument("--checkpoint-step", type=int, default=EXPECTED_CHECKPOINT_STEP)
    inspect.add_argument("--output", type=Path)
    manifest = commands.add_parser("build-manifest")
    manifest.add_argument("--model-path", type=Path, required=True)
    manifest.add_argument("--training-run-id", required=True)
    manifest.add_argument("--checkpoint-step", type=int, default=EXPECTED_CHECKPOINT_STEP)
    manifest.add_argument("--outer-commit", required=True)
    manifest.add_argument("--inner-commit", required=True)
    manifest.add_argument("--verl-commit", required=True)
    manifest.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-checkpoint":
        result = inspect_fsdp_actor_checkpoint(
            args.run_dir, checkpoint_step=args.checkpoint_step
        )
        if args.output:
            _atomic_json(args.output, result)
    else:
        result = build_merged_hf_manifest(
            args.model_path,
            training_run_id=args.training_run_id,
            checkpoint_step=args.checkpoint_step,
            outer_commit=args.outer_commit,
            inner_commit=args.inner_commit,
            verl_commit=args.verl_commit,
        )
        if args.output.exists():
            raise ValueError(f"refusing to overwrite model manifest: {args.output}")
        _atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_INSPECTION_SCHEMA",
    "EXPECTED_CHECKPOINT_STEP",
    "MODEL_MANIFEST_SCHEMA",
    "build_merged_hf_manifest",
    "inspect_fsdp_actor_checkpoint",
]
