#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from verl.workers.rollout.agent_vllm_rollout.rollout_timing import (
    analyze_rollout_timing_documents,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze synchronous AgentMemory rollout critical-path traces."
    )
    parser.add_argument("step_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(args.step_dir.glob("*.json"), key=lambda path: int(path.stem))
    documents = [json.loads(path.read_text()) for path in paths]
    summary = analyze_rollout_timing_documents(documents)
    rendered = json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
