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
QUERY_TOP1_SURFACES = frozenset(
    {
        "agentmemory_webshop_distractor_robustness_top1_train_v1",
        "agentmemory_webshop_compositional_recall_top1_train_v1",
        "agentmemory_webshop_intent_clarification_train_v1",
        "agentmemory_webshop_selective_memory_use_top1_train_v1",
    }
)
INTENT_CLARIFICATION_SURFACE = (
    "agentmemory_webshop_intent_clarification_train_v1"
)
SELECTIVE_MEMORY_USE_SURFACE = (
    "agentmemory_webshop_selective_memory_use_top1_train_v1"
)
FILESYSTEM_SURFACE = (
    "agentmemory_webshop_procedural_natural_chain_filesystem_v2"
)
QUERY_TOP1_REQUIRED_FRAGMENTS = (
    "RETRIEVE requires exactly query:string",
    "returns exactly one highest-ranked matching memory",
    "memory_id and top_k are forbidden",
)
QUERY_TOP1_FORBIDDEN_FRAGMENTS = (
    "optional top_k:int",
    "memory_id:string for exact readback",
)
INTENT_CLARIFICATION_FRAGMENTS = (
    "ASK requires field:string",
    "CLARIFY observation",
)
SELECTIVE_MEMORY_SOP_FRAGMENTS = (
    "First decide whether the current request already states every attribute needed",
    "explicit current requirements override profile history",
    "should not ADD or RETRIEVE merely by habit",
    "current request omits the customer's profile preference",
    "use RETRIEVE to expose the saved current profile",
    "Store new memory only when",
)
FILESYSTEM_REQUIRED_FRAGMENTS = (
    'shell_command {"command":"rg -n pattern ."',
    "apply_patch is followed on the next line",
    "*** Begin Patch",
    "*** End Patch",
    "workspace persists across shopping sessions within this episode",
    "Workspace actions have zero task reward",
    "has no network",
    "no host-path access",
    "no dedicated memory API",
    "workspace starts empty and contains only files that you create",
    "Before an action that advances to a new shopping session",
    "Copy the exact policy-visible field name and value needed later",
    "do not search, click, or write a replacement fact until you have used",
    "locate the candidate note and then read that file",
    "never infer or recreate the missing value from the choice table",
    "Listing a directory or reading a different file is not reading the note",
    "After an Add File action returns Done!, keep that successful note unchanged",
    "new note, use Add File with a new path",
    "never append Result or feedback text to the action",
    "separate turns and must never be emitted together",
    "Earlier turn (complete reply):\napply_patch",
    "*** Add File: .agent_memory/example.md\n+service port: 4317",
    "Later turn (complete reply):\n"
    "shell_command {\"command\":\"rg -n 'service port' .agent_memory/example.md\"",
)
FILESYSTEM_FORBIDDEN_FRAGMENTS = (
    'Read {"path"',
    'Write {"path"',
    'Edit {"path"',
    'Grep {"pattern"',
    'Glob {"pattern"',
    "ADD requires",
    "RETRIEVE accepts",
    "memory_id:string",
    "use ADD before",
    "use RETRIEVE",
    "Long-term memory persists",
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
    surface: str | None = None,
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("effective AgentMemory system prompt is empty")
    filesystem_required = surface == FILESYSTEM_SURFACE
    if (memory_prompt_mode == "natural_filesystem") != filesystem_required:
        raise RuntimeError(
            "natural_filesystem prompt mode and filesystem surface must be paired"
        )
    missing_filesystem = [
        fragment for fragment in FILESYSTEM_REQUIRED_FRAGMENTS if fragment not in prompt
    ]
    forbidden_filesystem = [
        fragment for fragment in FILESYSTEM_FORBIDDEN_FRAGMENTS if fragment in prompt
    ]
    if filesystem_required and (missing_filesystem or forbidden_filesystem):
        raise RuntimeError(
            "effective AgentMemory filesystem prompt contract is invalid: "
            f"missing={missing_filesystem} forbidden={forbidden_filesystem}"
        )
    query_top1_required = surface in QUERY_TOP1_SURFACES
    required_lifecycle_fragments = tuple(
        fragment
        for fragment in LIFECYCLE_SOP_FRAGMENTS
        if not (
            query_top1_required
            and fragment == "memory_id:string for exact readback"
        )
    )
    missing = [
        fragment for fragment in required_lifecycle_fragments if fragment not in prompt
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
    missing_query_top1 = [
        fragment for fragment in QUERY_TOP1_REQUIRED_FRAGMENTS if fragment not in prompt
    ]
    forbidden_query_top1 = [
        fragment for fragment in QUERY_TOP1_FORBIDDEN_FRAGMENTS if fragment in prompt
    ]
    if query_top1_required and (missing_query_top1 or forbidden_query_top1):
        raise RuntimeError(
            "effective AgentMemory system prompt violates the query-only top1 "
            "RETRIEVE contract: "
            f"missing={missing_query_top1} forbidden={forbidden_query_top1}"
        )
    intent_clarification_required = surface == INTENT_CLARIFICATION_SURFACE
    missing_intent_clarification = [
        fragment
        for fragment in INTENT_CLARIFICATION_FRAGMENTS
        if fragment not in prompt
    ]
    unexpected_intent_clarification = [
        fragment
        for fragment in INTENT_CLARIFICATION_FRAGMENTS
        if fragment in prompt
    ]
    if intent_clarification_required and missing_intent_clarification:
        raise RuntimeError(
            "effective AgentMemory system prompt is missing the intent "
            f"clarification contract: {missing_intent_clarification}"
        )
    if (
        surface in QUERY_TOP1_SURFACES
        and not intent_clarification_required
        and unexpected_intent_clarification
    ):
        raise RuntimeError(
            "effective AgentMemory system prompt leaks the intent clarification "
            f"contract onto another surface: {unexpected_intent_clarification}"
        )
    selective_memory_required = surface == SELECTIVE_MEMORY_USE_SURFACE
    missing_selective_memory = [
        fragment
        for fragment in SELECTIVE_MEMORY_SOP_FRAGMENTS
        if fragment not in prompt
    ]
    unexpected_selective_memory = [
        fragment
        for fragment in SELECTIVE_MEMORY_SOP_FRAGMENTS
        if fragment in prompt
    ]
    if selective_memory_required and missing_selective_memory:
        raise RuntimeError(
            "effective AgentMemory system prompt is missing the selective-memory "
            f"SOP: {missing_selective_memory}"
        )
    if (
        surface in QUERY_TOP1_SURFACES
        and not selective_memory_required
        and unexpected_selective_memory
    ):
        raise RuntimeError(
            "effective AgentMemory system prompt leaks the selective-memory SOP "
            f"onto another surface: {unexpected_selective_memory}"
        )
    return {
        "surface": surface,
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
        "query_top1_required": query_top1_required,
        "query_top1_present": not missing_query_top1 and not forbidden_query_top1,
        "missing_query_top1_fragments": missing_query_top1,
        "forbidden_query_top1_fragments_present": forbidden_query_top1,
        "intent_clarification_required": intent_clarification_required,
        "intent_clarification_present": not missing_intent_clarification,
        "missing_intent_clarification_fragments": missing_intent_clarification,
        "selective_memory_required": selective_memory_required,
        "selective_memory_present": not missing_selective_memory,
        "missing_selective_memory_fragments": missing_selective_memory,
        "filesystem_required": filesystem_required,
        "filesystem_present": not missing_filesystem and not forbidden_filesystem,
        "missing_filesystem_fragments": missing_filesystem,
        "forbidden_filesystem_fragments_present": forbidden_filesystem,
        "system_prompt_chars": len(prompt),
        "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "system_prompt": prompt,
    }


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def load_effective_prompt() -> tuple[str, str, str, str | None]:
    from verl.workers.rollout.schemas import (
        agentmemory_action_system_prompt,
        agentmemory_ltm_inventory_mode,
        agentmemory_memory_prompt_mode,
    )

    surface = os.environ.get("AGENTMEMORY_SURFACE")
    if surface is not None:
        surface = surface.strip() or None
    return (
        agentmemory_action_system_prompt(surface=surface),
        agentmemory_memory_prompt_mode(),
        agentmemory_ltm_inventory_mode(),
        surface,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attest the effective AgentMemory rollout system prompt."
    )
    parser.add_argument("--expect-memory-prompt-mode")
    parser.add_argument("--expect-ltm-inventory-mode")
    parser.add_argument("--expect-surface")
    parser.add_argument("--require-lifecycle-sop", action="store_true")
    parser.add_argument("--require-latent-preference-sop", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt, memory_prompt_mode, ltm_inventory_mode, surface = load_effective_prompt()
    if args.expect_surface is not None and surface != args.expect_surface:
        raise RuntimeError(
            "effective AgentMemory surface mismatch: "
            f"expected={args.expect_surface!r} actual={surface!r}"
        )
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
        surface=surface,
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
