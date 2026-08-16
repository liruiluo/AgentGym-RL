"""Frozen identities and immutable manifest construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from paired_eval.manifest import expand_manifest

from . import ARMS


OUTER_COMMIT = "aa2e9c80d572b513b5849c6d9b37a8dc4698bbc3"
INNER_COMMIT = "a0cc3ecf989ee89ba19a8e979617b4ec38909331"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
HARNESS_COMMIT = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
HARNESS_TREE = "f178530b37202c549b1b2b3300db2da90da648db"


@dataclass(frozen=True)
class DatasetPins:
    row_count: int
    jsonl_sha256: str
    id_ledger_sha256: str


PRODUCTION_DATASET_PINS = DatasetPins(
    row_count=500,
    jsonl_sha256=(
        "392529c5e79ca273bf0b073be35169beb68c604a26d9aef5514912fc584fa6cb"
    ),
    id_ledger_sha256=(
        "a6b0fd7c8c2969a0eef892e032250adcfa6d32362d395c246930e61b575ac9b9"
    ),
)


@dataclass(frozen=True)
class ImageIndexPins:
    row_count: int
    index_sha256: str
    tag_ledger_sha256: str
    digest_tsv_sha256: str


PRODUCTION_IMAGE_INDEX_PINS = ImageIndexPins(
    row_count=500,
    index_sha256=(
        "f2c1fb29457b66034cb04067f93707833125c8284b93771c924c10878ad9cd9b"
    ),
    tag_ledger_sha256=(
        "b69e618cfcfd2a59c3897e3f4856dbd88c4eeb921a5b24467a90bff6fa48581a"
    ),
    digest_tsv_sha256=(
        "b327b313612adefbc12161e2bf1e63e54925cbfcdccc26a416c1f7e94686af6b"
    ),
)


@dataclass(frozen=True)
class ModelFilePin:
    size: int
    sha256: str


PRODUCTION_MODEL_FILE_PINS = {
    ".gitattributes": ModelFilePin(
        1570,
        "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    ),
    "LICENSE": ModelFilePin(
        11544,
        "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
    ),
    "README.md": ModelFilePin(
        77661,
        "1406be1b6b8fd8a6545870da516912804756593628a1d0fb0a7965211e82a7bb",
    ),
    "chat_template.jinja": ModelFilePin(
        7756,
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    ),
    "config.json": ModelFilePin(
        3161,
        "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    ),
    "merges.txt": ModelFilePin(
        3353259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    "model.safetensors-00001-of-00002.safetensors": ModelFilePin(
        5329398688,
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    ),
    "model.safetensors-00002-of-00002.safetensors": ModelFilePin(
        3990429408,
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
    ),
    "model.safetensors.index.json": ModelFilePin(
        76196,
        "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d",
    ),
    "preprocessor_config.json": ModelFilePin(
        390,
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    "tokenizer.json": ModelFilePin(
        12807982,
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    "tokenizer_config.json": ModelFilePin(
        16710,
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
    ),
    "video_preprocessor_config.json": ModelFilePin(
        385,
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    ),
    "vocab.json": ModelFilePin(
        6722759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    ),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_regular_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a real regular file")
    return path.resolve(strict=True)


def require_directory(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a real directory")
    return path.resolve(strict=True)


def verify_source_identity(outer_commit: str, inner_commit: str) -> dict[str, str]:
    if outer_commit != OUTER_COMMIT:
        raise ValueError("outer source commit drifted")
    if inner_commit != INNER_COMMIT:
        raise ValueError("inner source commit drifted")
    return {
        "status": "pass",
        "outer_commit": outer_commit,
        "inner_commit": inner_commit,
    }


def parse_jsonl(path: Path, label: str) -> tuple[bytes, list[Mapping[str, Any]]]:
    resolved = require_regular_file(path, label)
    payload = resolved.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be nonempty and newline terminated")
    rows: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{label} row {line_number} is invalid JSON") from error
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} row {line_number} must be an object")
        rows.append(row)
    return payload, rows


def verify_dataset(
    path: Path | str,
    *,
    pins: DatasetPins = PRODUCTION_DATASET_PINS,
) -> dict[str, Any]:
    payload, rows = parse_jsonl(Path(path), "dataset JSONL")
    if len(rows) != pins.row_count:
        raise ValueError("dataset row count drifted")
    if sha256_bytes(payload) != pins.jsonl_sha256:
        raise ValueError("dataset JSONL SHA-256 drifted")
    instance_ids = [row.get("instance_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in instance_ids):
        raise ValueError("dataset instance IDs must be nonempty text")
    if instance_ids != sorted(instance_ids):
        raise ValueError("dataset instance IDs must be sorted")
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("dataset instance IDs must be unique")
    ledger = "".join(value + "\n" for value in instance_ids).encode("utf-8")
    if sha256_bytes(ledger) != pins.id_ledger_sha256:
        raise ValueError("dataset instance-ID ledger drifted")
    return {
        "schema": "swebench_verified_dataset_identity_v1",
        "rows": len(rows),
        "jsonl_sha256": sha256_bytes(payload),
        "id_ledger_sha256": sha256_bytes(ledger),
        "instance_ids": instance_ids,
    }


def verify_image_index(
    path: Path | str,
    *,
    pins: ImageIndexPins = PRODUCTION_IMAGE_INDEX_PINS,
) -> dict[str, Any]:
    payload, rows = parse_jsonl(Path(path), "image manifest index")
    if len(rows) != pins.row_count:
        raise ValueError("image index row count drifted")
    if sha256_bytes(payload) != pins.index_sha256:
        raise ValueError("image index SHA-256 drifted")
    tags: list[str] = []
    digests: list[str] = []
    for row in rows:
        tag = row.get("image")
        digest = row.get("digest")
        if not isinstance(tag, str) or not tag:
            raise ValueError("image index tag must be nonempty text")
        if (
            not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
        ):
            raise ValueError("image index digest must be lowercase sha256")
        if row.get("platform") != "linux/amd64":
            raise ValueError("image index platform drifted")
        tags.append(tag)
        digests.append(digest)
    if tags != sorted(tags):
        raise ValueError("image index tags must be sorted")
    if len(set(tags)) != len(tags):
        raise ValueError("image index tags must be unique")
    tag_ledger = "".join(tag + "\n" for tag in tags).encode("utf-8")
    digest_tsv = "".join(
        tag + "\t" + digest + "\n" for tag, digest in zip(tags, digests)
    ).encode("utf-8")
    if sha256_bytes(tag_ledger) != pins.tag_ledger_sha256:
        raise ValueError("image tag ledger drifted")
    if sha256_bytes(digest_tsv) != pins.digest_tsv_sha256:
        raise ValueError("image digest TSV drifted")
    return {
        "schema": "swebench_verified_image_index_identity_v1",
        "rows": len(rows),
        "index_sha256": sha256_bytes(payload),
        "tag_ledger_sha256": sha256_bytes(tag_ledger),
        "digest_tsv_sha256": sha256_bytes(digest_tsv),
        "digest_tsv": digest_tsv,
    }


def verify_model_files(
    root: Path | str,
    *,
    pins: Mapping[str, ModelFilePin] = PRODUCTION_MODEL_FILE_PINS,
) -> dict[str, Any]:
    model_root = require_directory(Path(root), "model root")
    actual_names = {
        item.name
        for item in model_root.iterdir()
        if item.is_file() or item.is_symlink()
    }
    if actual_names != set(pins):
        raise ValueError("model file set drifted")
    ledger = []
    for name in sorted(pins):
        path = require_regular_file(model_root / name, f"model file {name}")
        expected = pins[name]
        if path.stat().st_size != expected.size:
            raise ValueError(f"model file size drifted: {name}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected.sha256:
            raise ValueError(f"model file SHA-256 drifted: {name}")
        ledger.append(
            {"path": name, "size": expected.size, "sha256": actual_sha256}
        )
    ledger_payload = json.dumps(
        ledger,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "qwen35_model_file_identity_v1",
        "file_count": len(ledger),
        "file_ledger_sha256": sha256_bytes(ledger_payload),
        "files": ledger,
    }


def validate_frozen_common(common: Mapping[str, Any]) -> None:
    decoding = common.get("decoding")
    budgets = common.get("budgets")
    source = common.get("source")
    if not isinstance(decoding, Mapping) or decoding != {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 2048,
        "stop": [],
    }:
        raise ValueError("decoding settings drifted")
    expected_budgets = {
        "max_policy_turns": 250,
        "max_total_tokens": 8388608,
        "max_tool_calls": 250,
        "max_wall_seconds": 1800.0,
        "max_prompt_tokens": 30720,
        "max_model_tokens": 32768,
        "max_observation_tokens": 8192,
        "action_observation_envelope_tokens": 0,
    }
    if not isinstance(budgets, Mapping) or budgets != expected_budgets:
        raise ValueError("policy budgets drifted")
    if not isinstance(source, Mapping):
        raise ValueError("source identity is missing")
    verify_source_identity(source.get("outer_commit"), source.get("inner_commit"))


def build_manifest(
    instance_ids: Sequence[str],
    *,
    common: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    ids = list(instance_ids)
    if len(ids) != 500:
        raise ValueError("formal manifest requires exactly 500 tasks")
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("manifest task IDs must be nonempty text")
    if ids != sorted(ids):
        raise ValueError("manifest task IDs must be sorted")
    if len(set(ids)) != len(ids):
        raise ValueError("manifest task IDs must be unique")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("manifest run ID must be nonempty text")
    common_copy = json.loads(
        json.dumps(common, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    validate_frozen_common(common_copy)
    manifest = {
        "schema": "amg.paired_eval.manifest",
        "schema_version": "2.0.0",
        "run_id": run_id,
        "arms": list(ARMS),
        "common": common_copy,
        "tasks": [
            {
                "benchmark": "swebench_verified",
                "protocol": "swebench-verified@v4.1.0",
                "task_id": instance_id,
                "task_index": index,
                "seed": 0,
                "native_tools": ["shell_command", "apply_patch", "final"],
                "artifact_type": "patch",
            }
            for index, instance_id in enumerate(ids)
        ],
    }
    configs = expand_manifest(manifest)
    if len(configs) != 1500:
        raise ValueError("formal manifest did not expand to 1,500 cells")
    for offset in range(0, len(configs), 3):
        triad = configs[offset : offset + 3]
        if [config.capability.arm.value for config in triad] != list(ARMS):
            raise ValueError("formal manifest arm order drifted")
        excluded = {
            config.treatment_excluded_config_sha256 for config in triad
        }
        if len(excluded) != 1:
            raise ValueError("formal manifest treatment-excluded identity drifted")
    return manifest


__all__ = [
    "DatasetPins",
    "ImageIndexPins",
    "ModelFilePin",
    "PRODUCTION_DATASET_PINS",
    "PRODUCTION_IMAGE_INDEX_PINS",
    "PRODUCTION_MODEL_FILE_PINS",
    "build_manifest",
    "sha256_file",
    "verify_dataset",
    "verify_image_index",
    "verify_model_files",
    "verify_source_identity",
]
