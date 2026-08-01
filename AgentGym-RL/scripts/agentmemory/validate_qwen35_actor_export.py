#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from verl.workers.qwen35_runtime import resolve_qwen3_5_text_config


def normalize_weight_key(key: str) -> str:
    if key.startswith("model.language_model."):
        return key.replace("model.language_model.", "model.", 1)
    return key


def load_weight_shapes(actor_dir: Path) -> tuple[dict[str, tuple[int, ...]], int]:
    weight_files = sorted(actor_dir.glob("*.safetensors"))
    if not weight_files:
        raise SystemExit("merged actor has no safetensors files")
    shapes: dict[str, tuple[int, ...]] = {}
    raw_key_count = 0
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            for raw_key in handle.keys():
                raw_key_count += 1
                key = normalize_weight_key(raw_key)
                if key in shapes:
                    raise SystemExit(
                        "duplicate normalized key across safetensors files: "
                        f"{key!r}"
                    )
                shapes[key] = tuple(handle.get_slice(raw_key).get_shape())
    return shapes, raw_key_count


def verify_weight_shapes(
    actual_shapes: dict[str, tuple[int, ...]],
    expected_shapes: dict[str, tuple[int, ...]],
) -> None:
    missing = sorted(expected_shapes.keys() - actual_shapes.keys())
    unexpected = sorted(actual_shapes.keys() - expected_shapes.keys())
    shape_mismatches = {
        key: {"actual": actual_shapes[key], "expected": expected_shapes[key]}
        for key in sorted(actual_shapes.keys() & expected_shapes.keys())
        if actual_shapes[key] != expected_shapes[key]
    }
    if missing or unexpected or shape_mismatches:
        raise SystemExit(
            f"weight mismatch missing={missing[:10]!r} "
            f"unexpected={unexpected[:10]!r} "
            f"shape_mismatches={dict(list(shape_mismatches.items())[:10])!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-dir", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--max-model-len", default=131072, type=int)
    args = parser.parse_args()

    full_config = AutoConfig.from_pretrained(args.actor_dir)
    base_config = AutoConfig.from_pretrained(args.base_dir)
    if full_config.model_type != "qwen3_5":
        raise SystemExit(f"unexpected actor outer model type: {full_config.model_type!r}")
    if base_config.model_type != "qwen3_5":
        raise SystemExit(f"unexpected base outer model type: {base_config.model_type!r}")
    text_config = resolve_qwen3_5_text_config(full_config)
    if text_config.model_type != "qwen3_5_text":
        raise SystemExit(f"unexpected actor text model type: {text_config.model_type!r}")

    with torch.device("meta"):
        actor_model = AutoModelForCausalLM.from_config(text_config)
    expected_shapes = {
        key: tuple(tensor.shape)
        for key, tensor in actor_model.state_dict().items()
    }
    actual_shapes, raw_weight_keys = load_weight_shapes(args.actor_dir)
    verify_weight_shapes(actual_shapes, expected_shapes)

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_dir)
    actor_tokenizer = AutoTokenizer.from_pretrained(args.actor_dir)
    probes = [
        'SEARCH {"query": "red shoes"}',
        'ADD {"key": "source", "value": "ASIN B001"}',
        "Available Options:\nA. alpha\nB. beta\nC. gamma",
        "Progress: 1/6",
    ]
    if base_tokenizer.get_vocab() != actor_tokenizer.get_vocab():
        raise SystemExit("actor/base tokenizer vocabulary mismatch")
    if base_tokenizer.chat_template != actor_tokenizer.chat_template:
        raise SystemExit("actor/base chat template mismatch")
    if base_tokenizer.all_special_ids != actor_tokenizer.all_special_ids:
        raise SystemExit("actor/base tokenizer special IDs mismatch")
    for probe in probes:
        if base_tokenizer.encode(probe) != actor_tokenizer.encode(probe):
            raise SystemExit(f"actor/base tokenizer probe mismatch: {probe!r}")

    generation = GenerationConfig.from_pretrained(args.actor_dir)
    base_text_config = resolve_qwen3_5_text_config(base_config)
    if generation.eos_token_id != base_text_config.eos_token_id:
        raise SystemExit(
            "merged generation EOS differs from frozen base generation EOS: "
            f"{generation.eos_token_id!r} != {base_text_config.eos_token_id!r}"
        )
    if actor_tokenizer.eos_token_id != base_tokenizer.eos_token_id:
        raise SystemExit("actor/base tokenizer primary EOS mismatch")
    if actor_tokenizer.pad_token_id != base_tokenizer.pad_token_id:
        raise SystemExit("actor/base tokenizer pad ID mismatch")

    from vllm.engine.arg_utils import EngineArgs
    from vllm.renderers.registry import renderer_from_config

    engine_args = EngineArgs(
        model=str(args.actor_dir),
        tokenizer=str(args.actor_dir),
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        language_model_only=True,
    )
    vllm_config = engine_args.create_engine_config()
    renderer = renderer_from_config(vllm_config)

    result = {
        "status": "pass",
        "outer_model_type": full_config.model_type,
        "text_model_type": text_config.model_type,
        "vllm_architectures": vllm_config.model_config.architectures,
        "renderer": type(renderer).__name__,
        "weight_keys": raw_weight_keys,
        "parameters": sum(parameter.numel() for parameter in actor_model.parameters()),
        "tokenizer_eos": actor_tokenizer.eos_token_id,
        "tokenizer_pad": actor_tokenizer.pad_token_id,
        "generation_eos": generation.eos_token_id,
        "generation_pad": generation.pad_token_id,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
