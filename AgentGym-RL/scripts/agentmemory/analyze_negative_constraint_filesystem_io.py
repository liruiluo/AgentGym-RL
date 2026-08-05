#!/usr/bin/env python3
"""Audit standing-exclusion behavior from a filesystem exact-I/O audit."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BRANCH_RULES = {
    "allow_black": ("color", "black", ("gray", "red")),
    "allow_gray": ("color", "gray", ("black", "red")),
    "allow_red": ("color", "red", ("black", "gray")),
    "allow_floral": ("pattern", "floral", ("geometric", "solid")),
    "allow_geometric": ("pattern", "geometric", ("floral", "solid")),
    "allow_solid": ("pattern", "solid", ("floral", "geometric")),
}


def _normalize(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _has_word(text: str, value: str) -> bool:
    return re.search(
        rf"(?<![a-z0-9_]){re.escape(value)}(?![a-z0-9_])", text
    ) is not None


def _episode_branch(episode: dict[str, Any]) -> str | None:
    for record in episode.get("filesystem_io", []):
        branch = (
            (record.get("env_response") or {})
            .get("info", {})
            .get("branch_kind")
        )
        if isinstance(branch, str) and branch:
            return branch
    return None


def _contains_exact_rule(
    contents: list[str],
    *,
    axis: str,
    allowed: str,
    forbidden: tuple[str, str],
) -> bool:
    lines = [
        _normalize(line)
        for content in contents
        for line in content.splitlines()
        if _normalize(line).startswith("standing exclusions:")
    ]
    return any(
        _has_word(line, axis)
        and all(_has_word(line, value) for value in forbidden)
        and not _has_word(line, allowed)
        for line in lines
    )


def _analyze_episode(episode: dict[str, Any]) -> dict[str, Any]:
    branch = _episode_branch(episode)
    rule = BRANCH_RULES.get(branch or "")
    exact_sources = []
    if rule is not None:
        axis, allowed, forbidden = rule
        for source in episode.get("source_write_timing_candidates", []):
            contents = [
                str(item.get("content") or "") for item in source.get("files", [])
            ]
            if source.get("phase") == 0 and _contains_exact_rule(
                contents,
                axis=axis,
                allowed=allowed,
                forbidden=forbidden,
            ):
                exact_sources.append(source)

    exact_source_turns = {int(source["turn"]) for source in exact_sources}
    readback_links = []
    strict_links = []
    for link in episode.get("later_shell_timing_links", []):
        if (
            int(link.get("source_turn", -1)) not in exact_source_turns
            or int(link.get("source_phase", -1)) != 0
            or int(link.get("shell_phase", 0)) <= 0
            or not link.get("source_content_observed_in_stdout")
        ):
            continue
        readback_links.append(link)
        if link.get("later_correct_buy_turn") is not None:
            strict_links.append(link)

    unique_readbacks = {
        (int(link["source_turn"]), int(link["shell_turn"])) for link in readback_links
    }
    unique_strict = {
        (
            int(link["source_turn"]),
            int(link["shell_turn"]),
            int(link["later_correct_buy_turn"]),
        )
        for link in strict_links
    }
    compact_sources = [
        {
            "turn": int(source["turn"]),
            "files": [
                {
                    "path": item.get("path"),
                    "sha256": item.get("sha256"),
                    "content": item.get("content"),
                }
                for item in source.get("files", [])
            ],
        }
        for source in exact_sources
    ]
    compact_readbacks = [
        {
            "source_turn": int(link["source_turn"]),
            "shell_turn": int(link["shell_turn"]),
            "shell_phase": int(link["shell_phase"]),
            "command": link.get("command"),
            "stdout": link.get("stdout"),
            "later_correct_buy_turn": link.get("later_correct_buy_turn"),
        }
        for link in readback_links
    ]
    rule_payload = None
    if rule is not None:
        rule_payload = {
            "axis": rule[0],
            "allowed": rule[1],
            "forbidden": list(rule[2]),
        }
    return {
        "episode_path": episode.get("episode_path"),
        "rollout_step": episode.get("rollout_step"),
        "policy_updates_before_rollout": episode.get("policy_updates_before_rollout"),
        "trajectory_uid": episode.get("trajectory_uid"),
        "data_idx": episode.get("data_idx"),
        "branch_kind": branch,
        "expected_rule": rule_payload,
        "final_phase_progress": episode.get("final_phase_progress"),
        "episode_success": bool(episode.get("episode_success")),
        "exact_source_rule_action_count": len(exact_sources),
        "exact_source_rule_turns": sorted(exact_source_turns),
        "exact_source_rules": compact_sources,
        "exact_rule_readback_count": len(unique_readbacks),
        "exact_rule_readbacks": compact_readbacks,
        "strict_exclusion_chain_count": len(unique_strict),
        "strict_exclusion_chains": [
            {
                "source_turn": source_turn,
                "shell_turn": shell_turn,
                "later_correct_buy_turn": buy_turn,
            }
            for source_turn, shell_turn, buy_turn in sorted(unique_strict)
        ],
    }


def _summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    progress = Counter(int(episode.get("final_phase_progress") or 0) for episode in episodes)
    return {
        "trajectory_count": len(episodes),
        "branch_resolved_trajectory_count": sum(
            episode["branch_kind"] in BRANCH_RULES for episode in episodes
        ),
        "exact_source_rule_action_count": sum(
            episode["exact_source_rule_action_count"] for episode in episodes
        ),
        "exact_source_rule_trajectory_count": sum(
            episode["exact_source_rule_action_count"] > 0 for episode in episodes
        ),
        "exact_rule_readback_count": sum(
            episode["exact_rule_readback_count"] for episode in episodes
        ),
        "exact_rule_readback_trajectory_count": sum(
            episode["exact_rule_readback_count"] > 0 for episode in episodes
        ),
        "strict_exclusion_chain_count": sum(
            episode["strict_exclusion_chain_count"] for episode in episodes
        ),
        "strict_exclusion_chain_trajectory_count": sum(
            episode["strict_exclusion_chain_count"] > 0 for episode in episodes
        ),
        "final_phase_progress_counts": {
            str(key): value for key, value in sorted(progress.items())
        },
        "progress_ge_1_count": sum(
            int(episode.get("final_phase_progress") or 0) >= 1 for episode in episodes
        ),
        "progress_ge_2_count": sum(
            int(episode.get("final_phase_progress") or 0) >= 2 for episode in episodes
        ),
        "episode_success_count": sum(episode["episode_success"] for episode in episodes),
    }


def analyze_audit(audit: dict[str, Any]) -> dict[str, Any]:
    if audit.get("schema") != "agentmemory_filesystem_exact_io_audit_v1":
        raise ValueError("input is not a filesystem exact-I/O audit v1")
    raw_episodes = audit.get("episodes")
    if not isinstance(raw_episodes, list):
        raise ValueError("input audit has no episodes list")
    episodes = [_analyze_episode(episode) for episode in raw_episodes]
    rollout_steps = sorted(
        {
            int(episode["rollout_step"])
            for episode in episodes
            if episode["rollout_step"] is not None
        }
    )
    return {
        "schema": "agentmemory_negative_constraint_filesystem_io_audit_v1",
        "source_audit_schema": audit["schema"],
        "run_dir": audit.get("run_dir"),
        "verdict_boundary": (
            "training-side short-gate evidence only; correct BUY means native phase "
            "advancement and is not a held-out or multitask capability result"
        ),
        "summary": _summarize(episodes),
        "summary_by_rollout_step": {
            str(step): _summarize(
                [episode for episode in episodes if episode["rollout_step"] == step]
            )
            for step in rollout_steps
        },
        "episodes": episodes,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Negative-constraint filesystem exact-I/O audit",
        "",
        f"Run: `{audit.get('run_dir')}`",
        "",
        "## Verdict boundary",
        "",
        "- This is a single-task RL admission gate, not a held-out or multitask endpoint.",
        "- A strict chain requires an exact `Standing exclusions:` record with the axis and both forbidden values, a later-session shell readback of that same file content, and a native correct BUY after the readback.",
        "- The allowed value appearing in the exclusion record invalidates the source write.",
        "",
        "## Aggregate",
        "",
        f"- Exact source rules: {summary['exact_source_rule_action_count']} actions in {summary['exact_source_rule_trajectory_count']}/{summary['trajectory_count']} trajectories.",
        f"- Exact later-session readbacks: {summary['exact_rule_readback_count']} in {summary['exact_rule_readback_trajectory_count']} trajectories.",
        f"- Strict exclusion chains: {summary['strict_exclusion_chain_count']} in {summary['strict_exclusion_chain_trajectory_count']} trajectories.",
        f"- Final progress distribution: `{summary['final_phase_progress_counts']}`; full successes: {summary['episode_success_count']}.",
        "",
        "## Per rollout step",
        "",
    ]
    for step, item in audit["summary_by_rollout_step"].items():
        updates = max(int(step) - 1, 0)
        policy = "base policy" if updates == 0 else f"after {updates} update(s)"
        lines.extend(
            [
                f"- Step {step} ({policy}): exact source rules in {item['exact_source_rule_trajectory_count']}/{item['trajectory_count']} trajectories; strict chains {item['strict_exclusion_chain_count']} in {item['strict_exclusion_chain_trajectory_count']} trajectories; progress `{item['final_phase_progress_counts']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Nonzero strict chains prove that the filesystem contract can express and train the intended standing-exclusion behavior.",
            "- Low full-session success remains a model-quality gap for multitask training and per-task held-out evaluation; extending this single task to 100 updates is not the default remedy.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.audit.read_text(encoding="utf-8"))
    result = analyze_audit(source)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(render_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
