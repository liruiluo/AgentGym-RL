#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


LIFECYCLE_SOP_FRAGMENTS = (
    "use ADD before click[Buy Now]",
    "At the start of every later shopping session",
    "use RETRIEVE",
    "memory_id:string for exact readback",
    "does not reject an otherwise correct purchase when ADD was skipped",
)
LATENT_PREFERENCE_SOP_FRAGMENTS = (
    "confirmed choice as preference evidence",
    "customer-profile memory",
    "preference axis",
    "inferred value",
    "use UPDATE",
    "Do not assume a fixed number",
    "later application sessions",
)


def build_attestation(
    *,
    prompt: str,
    memory_prompt_mode: str,
    ltm_inventory_mode: str,
    thinking_enabled: bool,
    reasoning_enabled: bool,
    require_lifecycle_sop: bool,
    require_latent_preference_sop: bool = False,
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("effective AgentMemory system prompt is empty")
    missing = [
        fragment for fragment in LIFECYCLE_SOP_FRAGMENTS if fragment not in prompt
    ]
    if require_lifecycle_sop and missing:
        raise RuntimeError(
            "effective AgentMemory system prompt is missing the required "
            f"memory lifecycle SOP: {missing}"
        )
    missing_latent_preference = [
        fragment
        for fragment in LATENT_PREFERENCE_SOP_FRAGMENTS
        if fragment not in prompt
    ]
    if require_latent_preference_sop and missing_latent_preference:
        raise RuntimeError(
            "effective AgentMemory system prompt is missing the required "
            f"latent-preference SOP: {missing_latent_preference}"
        )
    return {
        "memory_prompt_mode": memory_prompt_mode,
        "ltm_inventory_mode": ltm_inventory_mode,
        "thinking_enabled": bool(thinking_enabled),
        "reasoning_enabled": bool(reasoning_enabled),
        "require_lifecycle_sop": bool(require_lifecycle_sop),
        "lifecycle_sop_present": not missing,
        "missing_lifecycle_sop_fragments": missing,
        "require_latent_preference_sop": bool(require_latent_preference_sop),
        "latent_preference_sop_present": not missing_latent_preference,
        "missing_latent_preference_sop_fragments": missing_latent_preference,
        "system_prompt_chars": len(prompt),
        "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "system_prompt": prompt,
    }


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def load_effective_prompt() -> tuple[str, str, str]:
    from verl.workers.rollout.schemas import (
        agentmemory_action_system_prompt,
        agentmemory_ltm_inventory_mode,
        agentmemory_memory_prompt_mode,
    )

    return (
        agentmemory_action_system_prompt(),
        agentmemory_memory_prompt_mode(),
        agentmemory_ltm_inventory_mode(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attest the effective AgentMemory rollout system prompt."
    )
    parser.add_argument("--expect-memory-prompt-mode")
    parser.add_argument("--expect-ltm-inventory-mode")
    parser.add_argument("--require-lifecycle-sop", action="store_true")
    parser.add_argument("--require-latent-preference-sop", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt, memory_prompt_mode, ltm_inventory_mode = load_effective_prompt()
    if (
        args.expect_memory_prompt_mode is not None
        and memory_prompt_mode != args.expect_memory_prompt_mode
    ):
        raise RuntimeError(
            "effective memory prompt mode mismatch: "
            f"expected={args.expect_memory_prompt_mode!r} actual={memory_prompt_mode!r}"
        )
    if (
        args.expect_ltm_inventory_mode is not None
        and ltm_inventory_mode != args.expect_ltm_inventory_mode
    ):
        raise RuntimeError(
            "effective LTM inventory mode mismatch: "
            f"expected={args.expect_ltm_inventory_mode!r} actual={ltm_inventory_mode!r}"
        )
    attestation = build_attestation(
        prompt=prompt,
        memory_prompt_mode=memory_prompt_mode,
        ltm_inventory_mode=ltm_inventory_mode,
        thinking_enabled=_env_enabled("AGENTMEMORY_ENABLE_THINKING"),
        reasoning_enabled=_env_enabled("AGENTMEMORY_ALLOW_REASONING"),
        require_lifecycle_sop=args.require_lifecycle_sop,
        require_latent_preference_sop=args.require_latent_preference_sop,
    )
    rendered = (
        json.dumps(attestation, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
