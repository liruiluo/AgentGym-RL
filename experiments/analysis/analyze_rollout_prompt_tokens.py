#!/usr/bin/env python3
"""Reconstruct actual AgentMemory rollout prompt lengths from 0.json logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "agentmemory_rollout_prompt_token_telemetry_v1"
SAMPLE_PATTERN = re.compile(r"steptest_batch_(\d+)_sample_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--runtime-source",
        type=Path,
        required=True,
        help="AgentGym-RL root containing verl/workers/rollout/schemas.py",
    )
    parser.add_argument(
        "--reply-mode",
        choices=("bare", "reasoning", "thinking"),
        required=True,
        help="Must match AGENTMEMORY_ALLOW_REASONING/ENABLE_THINKING in the run.",
    )
    parser.add_argument(
        "--prompt-caps",
        type=int,
        nargs="+",
        default=(32256, 61440, 124928),
    )
    parser.add_argument("--expected-replicas", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_percentile(values: list[int], percentile: int) -> int:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    index = int(math.floor(position + 0.5))
    return ordered[index]


def summarize(values: list[int], caps: Iterable[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50": nearest_percentile(values, 50),
        "p95": nearest_percentile(values, 95),
        "p99": nearest_percentile(values, 99),
        "max": max(values),
        "over_caps": {
            str(cap): sum(value > cap for value in values) for cap in caps
        },
    }


def sample_sort_key(path: Path) -> tuple[int, int, str]:
    match = SAMPLE_PATTERN.search(str(path))
    if match is None:
        return (sys.maxsize, sys.maxsize, str(path))
    return (int(match.group(1)), int(match.group(2)), str(path))


def validate_seed_pair(conversations: list[dict[str, Any]], source: str) -> None:
    if len(conversations) < 3:
        raise ValueError(f"{source}: conversation is too short")
    if conversations[0].get("role") != "user":
        raise ValueError(f"{source}: conversation_start user turn is missing")
    if conversations[1] != {"role": "assistant", "content": "Ok."}:
        raise ValueError(
            f"{source}: unexpected conversation_start assistant turn; "
            "refusing to guess which turn was sampled"
        )


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    model = args.model.resolve()
    runtime_source = args.runtime_source.resolve()
    output = (args.output or run_dir / "prompt_token_telemetry.json").resolve()
    caps = sorted(set(args.prompt_caps))
    if not caps or caps[0] < 1:
        raise SystemExit("prompt caps must be positive")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ["AGENTMEMORY_ENABLE_THINKING"] = (
        "1" if args.reply_mode == "thinking" else "0"
    )
    os.environ["AGENTMEMORY_ALLOW_REASONING"] = (
        "1" if args.reply_mode == "reasoning" else "0"
    )
    sys.path.insert(0, str(runtime_source))

    from transformers import AutoTokenizer
    from verl.workers.rollout.schemas import (
        agentmemory_action_system_prompt,
        apply_chat_template,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=True,
    )
    system_prompt = agentmemory_action_system_prompt()
    log_paths = sorted(
        run_dir.glob(
            "results/*/executer_logs/steptest_batch_*_sample_*/0.json"
        ),
        key=sample_sort_key,
    )
    if not log_paths:
        raise SystemExit(f"no rollout 0.json files found under {run_dir}")
    if args.expected_replicas is not None and len(log_paths) != args.expected_replicas:
        raise SystemExit(
            f"expected {args.expected_replicas} replicas, found {len(log_paths)}"
        )

    all_lengths: list[int] = []
    replica_summaries: list[dict[str, Any]] = []
    trajectory_count = 0
    for path in log_paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path}: expected a non-empty list")
        trajectory_count += len(rows)
        lengths: list[int] = []
        expected_count = 0
        for row_index, row in enumerate(rows):
            conversations = row.get("conversations")
            if not isinstance(conversations, list):
                raise ValueError(f"{path}: row {row_index} has no conversations list")
            validate_seed_pair(conversations, f"{path}: row {row_index}")
            expected_count += int(row["task_rounds"])
            for message_index, message in enumerate(conversations[2:], start=2):
                if message.get("role") != "assistant":
                    continue
                previous = conversations[message_index - 1]
                if previous.get("role") != "user":
                    raise ValueError(
                        f"{path}: assistant turn {message_index} is not preceded by user"
                    )
                prompt_ids = apply_chat_template(
                    tokenizer,
                    [
                        {"role": "system", "content": system_prompt},
                        previous,
                    ],
                )
                lengths.append(len(prompt_ids))
        if len(lengths) != expected_count:
            raise ValueError(
                f"{path}: counted {len(lengths)} sampled turns, "
                f"but sum(task_rounds)={expected_count}"
            )
        summary = summarize(lengths, caps)
        summary.update(
            {
                "rollout_log": str(path.relative_to(run_dir)),
                "trajectory_count": len(rows),
            }
        )
        replica_summaries.append(summary)
        all_lengths.extend(lengths)

    model_config = model / "config.json"
    model_payload = json.loads(model_config.read_text(encoding="utf-8"))
    native_context = model_payload.get("text_config", {}).get(
        "max_position_embeddings"
    ) or model_payload.get("max_position_embeddings")
    result = {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "reply_mode": args.reply_mode,
        "seed_pair_policy": "exclude_fixed_conversation_start_user_assistant_pair",
        "count_invariant": "actual_generation_prompts_equals_sum_task_rounds",
        "percentile_method": "nearest_observed_value",
        "prompt_caps": caps,
        "model": {
            "path": str(model),
            "config_sha256": sha256_file(model_config),
            "native_context_length": native_context,
        },
        "runtime": {
            "source": str(runtime_source),
            "schemas_sha256": sha256_file(
                runtime_source / "verl/workers/rollout/schemas.py"
            ),
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
        },
        "replica_count": len(log_paths),
        "trajectory_count": trajectory_count,
        "aggregate": summarize(all_lengths, caps),
        "replicas": replica_summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(json.dumps(result["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
