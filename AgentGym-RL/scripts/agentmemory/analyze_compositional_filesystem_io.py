#!/usr/bin/env python3
"""Audit two-hop profile behavior from a filesystem exact-I/O audit."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROFILE_TOKEN_PATTERN = r"pt[.][a-f0-9]{16}"


def _normalize(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _phase_inputs(episode: dict[str, Any], phase: int) -> list[str]:
    texts = []
    for record in episode.get("filesystem_io", []):
        if int(record.get("phase_before") or 0) != phase:
            continue
        exact_input = record.get("exact_model_input")
        if isinstance(exact_input, str):
            texts.append(exact_input)
        for message in record.get("request_messages", []):
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                texts.append(content)
    return texts


def _parse_mapping_expectation(
    episode: dict[str, Any],
) -> tuple[str, str] | None:
    for text in _phase_inputs(episode, 0):
        customer = re.search(r"Customer profile:\s*([^\s\n]+)", text)
        token = re.search(
            rf"active shopping profile token is\s+({PROFILE_TOKEN_PATTERN})",
            text,
            flags=re.IGNORECASE,
        )
        if customer and token:
            return customer.group(1), token.group(1)
    return None


def _parse_directory_expectation(
    episode: dict[str, Any],
) -> tuple[tuple[str, str, str], tuple[str, str, str]] | None:
    pattern = re.compile(
        rf"^-\s*({PROFILE_TOKEN_PATTERN}):\s*([^\n:]+?)\s+is\s+([^\n]+?)\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    for text in _phase_inputs(episode, 1):
        entries = [
            (match.group(1), match.group(2).strip(), match.group(3).strip())
            for match in pattern.finditer(text)
        ]
        unique = []
        for entry in entries:
            if entry not in unique:
                unique.append(entry)
        if (
            len(unique) == 2
            and len({entry[0] for entry in unique}) == 2
            and len({entry[1].casefold() for entry in unique}) == 1
            and len({entry[2].casefold() for entry in unique}) == 2
        ):
            return unique[0], unique[1]
    return None


def _labeled_lines(contents: list[str], prefix: str) -> list[str]:
    normalized_prefix = prefix.casefold()
    return [
        _normalize(line)
        for content in contents
        for line in content.splitlines()
        if _normalize(line).startswith(normalized_prefix)
    ]


def _mapping_content_is_exact(
    contents: list[str],
    expectation: tuple[str, str],
    directory: tuple[tuple[str, str, str], tuple[str, str, str]] | None,
) -> bool:
    customer, active_token = expectation
    directory_values = () if directory is None else tuple(entry[2] for entry in directory)
    return any(
        _normalize(customer) in line
        and _normalize(active_token) in line
        and all(_normalize(value) not in line for value in directory_values)
        for line in _labeled_lines(contents, "customer-to-profile:")
    )


def _directory_content_is_exact(
    contents: list[str],
    expectation: tuple[tuple[str, str, str], tuple[str, str, str]],
    customer: str | None,
) -> bool:
    required = {
        _normalize(value)
        for entry in expectation
        for value in entry
    }
    return any(
        all(value in line for value in required)
        and (customer is None or _normalize(customer) not in line)
        for line in _labeled_lines(contents, "profile-directory:")
    )


def _analyze_episode(episode: dict[str, Any]) -> dict[str, Any]:
    mapping = _parse_mapping_expectation(episode)
    directory = _parse_directory_expectation(episode)
    exact_mapping_sources = []
    exact_directory_sources = []
    for source in episode.get("source_write_timing_candidates", []):
        contents = [
            str(item.get("content") or "") for item in source.get("files", [])
        ]
        phase = int(source.get("phase", -1))
        if (
            phase == 0
            and mapping is not None
            and _mapping_content_is_exact(contents, mapping, directory)
        ):
            exact_mapping_sources.append(source)
        if (
            phase == 1
            and directory is not None
            and _directory_content_is_exact(
                contents,
                directory,
                None if mapping is None else mapping[0],
            )
        ):
            exact_directory_sources.append(source)

    source_roles = {
        **{int(source["turn"]): "customer_to_profile" for source in exact_mapping_sources},
        **{int(source["turn"]): "profile_directory" for source in exact_directory_sources},
    }
    readbacks = []
    compact_readbacks = []
    roles_by_buy: dict[tuple[int, int], set[str]] = defaultdict(set)
    shell_turns_by_buy: dict[tuple[int, int], set[int]] = defaultdict(set)
    for link in episode.get("later_shell_timing_links", []):
        role = source_roles.get(int(link.get("source_turn", -1)))
        shell_phase = int(link.get("shell_phase", 0))
        buy_turn = link.get("later_correct_buy_turn")
        if (
            role is None
            or shell_phase < 2
            or not link.get("source_content_observed_in_stdout")
        ):
            continue
        readbacks.append(
            (role, int(link["source_turn"]), int(link["shell_turn"]), shell_phase)
        )
        compact_readbacks.append(
            {
                "role": role,
                "source_turn": int(link["source_turn"]),
                "shell_turn": int(link["shell_turn"]),
                "shell_phase": shell_phase,
                "command": link.get("command"),
                "stdout": link.get("stdout"),
                "later_correct_buy_turn": buy_turn,
            }
        )
        if buy_turn is not None:
            key = (shell_phase, int(buy_turn))
            roles_by_buy[key].add(role)
            shell_turns_by_buy[key].add(int(link["shell_turn"]))

    strict = [
        {
            "shell_phase": shell_phase,
            "later_correct_buy_turn": buy_turn,
            "roles_observed": sorted(roles_by_buy[(shell_phase, buy_turn)]),
            "shell_turns": sorted(shell_turns_by_buy[(shell_phase, buy_turn)]),
        }
        for shell_phase, buy_turn in sorted(roles_by_buy)
        if roles_by_buy[(shell_phase, buy_turn)]
        == {"customer_to_profile", "profile_directory"}
    ]
    expected = None
    if mapping is not None and directory is not None:
        expected = {
            "customer": mapping[0],
            "active_profile_token": mapping[1],
            "directory": [
                {"profile_token": token, "axis": axis, "value": value}
                for token, axis, value in directory
            ],
        }
    def compact_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "turn": int(source["turn"]),
                "phase": int(source["phase"]),
                "files": [
                    {
                        "path": item.get("path"),
                        "sha256": item.get("sha256"),
                        "content": item.get("content"),
                    }
                    for item in source.get("files", [])
                ],
            }
            for source in sources
        ]
    return {
        "episode_path": episode.get("episode_path"),
        "rollout_step": episode.get("rollout_step"),
        "policy_updates_before_rollout": episode.get("policy_updates_before_rollout"),
        "trajectory_uid": episode.get("trajectory_uid"),
        "data_idx": episode.get("data_idx"),
        "final_phase_progress": episode.get("final_phase_progress"),
        "episode_success": bool(episode.get("episode_success")),
        "expected_facts": expected,
        "expectation_parse_complete": expected is not None,
        "exact_mapping_source_count": len(exact_mapping_sources),
        "exact_mapping_sources": compact_sources(exact_mapping_sources),
        "exact_directory_source_count": len(exact_directory_sources),
        "exact_directory_sources": compact_sources(exact_directory_sources),
        "exact_two_source_hops": bool(exact_mapping_sources and exact_directory_sources),
        "exact_source_readback_count": len(set(readbacks)),
        "exact_source_readbacks": compact_readbacks,
        "strict_two_hop_chain_count": len(strict),
        "strict_two_hop_chains": strict,
    }


def _summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    progress = Counter(int(episode.get("final_phase_progress") or 0) for episode in episodes)
    return {
        "trajectory_count": len(episodes),
        "expectation_parse_complete_count": sum(
            episode["expectation_parse_complete"] for episode in episodes
        ),
        "exact_mapping_source_trajectory_count": sum(
            episode["exact_mapping_source_count"] > 0 for episode in episodes
        ),
        "exact_directory_source_trajectory_count": sum(
            episode["exact_directory_source_count"] > 0 for episode in episodes
        ),
        "exact_two_source_hop_trajectory_count": sum(
            episode["exact_two_source_hops"] for episode in episodes
        ),
        "exact_source_readback_count": sum(
            episode["exact_source_readback_count"] for episode in episodes
        ),
        "strict_two_hop_chain_count": sum(
            episode["strict_two_hop_chain_count"] for episode in episodes
        ),
        "strict_two_hop_chain_trajectory_count": sum(
            episode["strict_two_hop_chain_count"] > 0 for episode in episodes
        ),
        "final_phase_progress_counts": {
            str(key): value for key, value in sorted(progress.items())
        },
        "progress_ge_2_count": sum(
            int(episode.get("final_phase_progress") or 0) >= 2 for episode in episodes
        ),
        "progress_ge_3_count": sum(
            int(episode.get("final_phase_progress") or 0) >= 3 for episode in episodes
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
        "schema": "agentmemory_compositional_filesystem_io_audit_v1",
        "source_audit_schema": audit["schema"],
        "run_dir": audit.get("run_dir"),
        "verdict_boundary": (
            "training-side short-gate evidence only; two exact labeled facts may be "
            "read in one or multiple shell actions before the same native correct BUY"
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
        "# Compositional filesystem exact-I/O audit",
        "",
        f"Run: `{audit.get('run_dir')}`",
        "",
        "## Verdict boundary",
        "",
        "- This is a single-task RL admission gate, not a held-out or multitask endpoint.",
        "- A strict chain requires an exact customer-to-profile record written in session 0, an exact two-entry profile directory written in session 1, later-session shell output that exposes both records, and a native correct BUY after both reads.",
        "- The two records may be read by one or multiple shell commands before the same BUY; the audit tests semantic composition, not a one-call retrieval scaffold.",
        "",
        "## Aggregate",
        "",
        f"- Exact mapping sources: {summary['exact_mapping_source_trajectory_count']}/{summary['trajectory_count']} trajectories.",
        f"- Exact directory sources: {summary['exact_directory_source_trajectory_count']}/{summary['trajectory_count']} trajectories.",
        f"- Both exact source hops: {summary['exact_two_source_hop_trajectory_count']}/{summary['trajectory_count']} trajectories.",
        f"- Strict two-hop chains: {summary['strict_two_hop_chain_count']} in {summary['strict_two_hop_chain_trajectory_count']} trajectories.",
        f"- Final progress distribution: `{summary['final_phase_progress_counts']}`; full successes: {summary['episode_success_count']}.",
        "",
        "## Per rollout step",
        "",
    ]
    for step, item in audit["summary_by_rollout_step"].items():
        updates = max(int(step) - 1, 0)
        policy = "base policy" if updates == 0 else f"after {updates} update(s)"
        lines.append(
            f"- Step {step} ({policy}): both source hops in {item['exact_two_source_hop_trajectory_count']}/{item['trajectory_count']} trajectories; strict chains {item['strict_two_hop_chain_count']} in {item['strict_two_hop_chain_trajectory_count']} trajectories; progress `{item['final_phase_progress_counts']}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Nonzero strict chains prove that the filesystem surface can express the intended two-hop composition without a dedicated memory API.",
            "- Remaining six-session success is a multitask training and per-task held-out question; it is not a reason to extend this single-task gate to 100 updates by default.",
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
