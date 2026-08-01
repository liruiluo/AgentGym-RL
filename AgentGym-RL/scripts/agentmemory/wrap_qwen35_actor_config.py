#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


TEXT_CONFIG_RUNTIME_FIELDS = {
    "architectures",
    "bos_token_id",
    "dtype",
    "eos_token_id",
    "pad_token_id",
    "partial_rotary_factor",
    "transformers_version",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def validate_compatible_configs(actor_config: dict, base_config: dict) -> None:
    if actor_config.get("model_type") != "qwen3_5_text":
        raise SystemExit(
            f"unexpected actor model_type: {actor_config.get('model_type')!r}"
        )
    if base_config.get("model_type") != "qwen3_5":
        raise SystemExit(
            f"unexpected base model_type: {base_config.get('model_type')!r}"
        )
    if base_config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise SystemExit(
            f"unexpected base architecture: {base_config.get('architectures')!r}"
        )
    base_text_config = base_config.get("text_config") or {}
    actor_architecture = {
        key: value
        for key, value in actor_config.items()
        if key not in TEXT_CONFIG_RUNTIME_FIELDS
    }
    base_architecture = {
        key: value
        for key, value in base_text_config.items()
        if key not in TEXT_CONFIG_RUNTIME_FIELDS
    }
    if actor_architecture != base_architecture:
        mismatched_keys = sorted(
            key
            for key in actor_architecture.keys() | base_architecture.keys()
            if actor_architecture.get(key) != base_architecture.get(key)
        )
        details = {
            key: {
                "actor": actor_architecture.get(key),
                "base": base_architecture.get(key),
            }
            for key in mismatched_keys[:10]
        }
        raise SystemExit(
            "actor/base text architecture mismatch: "
            f"keys={mismatched_keys[:10]!r} details={details!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-dir", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    args = parser.parse_args()

    actor_config_path = args.actor_dir / "config.json"
    text_backup_path = args.actor_dir / "config.text_only.json"
    base_config_path = args.base_dir / "config.json"
    generation_config_path = args.actor_dir / "generation_config.json"
    manifest_path = args.actor_dir / "config.wrapper_manifest.json"

    actor_config = read_json(actor_config_path)
    base_config = read_json(base_config_path)
    if actor_config.get("model_type") == "qwen3_5":
        if not text_backup_path.is_file():
            raise SystemExit("wrapped actor is missing config.text_only.json")
        text_config = read_json(text_backup_path)
        validate_compatible_configs(text_config, base_config)
        if actor_config.get("text_config") != text_config:
            raise SystemExit("wrapped actor text_config differs from its backup")
    else:
        text_config = actor_config
        validate_compatible_configs(text_config, base_config)
        if text_backup_path.exists():
            if read_json(text_backup_path) != text_config:
                raise SystemExit(
                    "existing text-only config backup differs from actor config"
                )
        else:
            write_json(text_backup_path, text_config)
        wrapped_config = dict(base_config)
        wrapped_config["text_config"] = text_config
        write_json(actor_config_path, wrapped_config)
        actor_config = wrapped_config

    manifest = {
        "format": "qwen3_5_full_config_with_trained_text_actor_v1",
        "actor_config": str(actor_config_path),
        "actor_config_sha256": sha256(actor_config_path),
        "base_config": str(base_config_path),
        "base_config_sha256": sha256(base_config_path),
        "generation_config_sha256": (
            sha256(generation_config_path)
            if generation_config_path.is_file()
            else None
        ),
        "text_config_backup": str(text_backup_path),
        "text_config_sha256": sha256(text_backup_path),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
