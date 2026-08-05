#!/usr/bin/env python3
"""Audit filesystem behavior samples from exact model and environment I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TRAINING_DIAGNOSTIC_RE = re.compile(r"ppo_batch_step([0-9]+)_post_adv[.]json$")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _component_names(step: dict[str, Any]) -> set[str]:
    return {
        str(item.get("name", ""))
        for item in step.get("reward_components", [])
        if isinstance(item, dict)
    }


def _workspace_event(step: dict[str, Any]) -> dict[str, Any] | None:
    event = step.get("env_info_after", {}).get("workspace_latest_event")
    return event if isinstance(event, dict) else None


def _user_input(step: dict[str, Any]) -> str:
    messages = step.get("request_messages", [])
    if not messages:
        return ""
    return str(messages[-1].get("content", ""))


def _first_feedback_line(step: dict[str, Any]) -> str:
    observation = str(step.get("env_response", {}).get("observation", ""))
    return observation.splitlines()[0] if observation else ""


def _parse_add_file_contents(patch: str) -> dict[str, str]:
    files: dict[str, str] = {}
    current_path: str | None = None
    lines: list[str] = []

    def finish() -> None:
        nonlocal current_path, lines
        if current_path is not None:
            files[current_path] = "\n".join(lines) + ("\n" if lines else "")
        current_path = None
        lines = []

    for line in patch.splitlines():
        if line.startswith("*** Add File: "):
            finish()
            current_path = line.removeprefix("*** Add File: ").strip()
        elif line.startswith("*** "):
            finish()
        elif current_path is not None and line.startswith("+"):
            lines.append(line[1:])
    finish()
    return files


def _parse_cards(observation: str) -> list[dict[str, str]]:
    cards = []
    pattern = re.compile(
        r"- Product: ([^\n]+)\n\s+Confirmed ([^:\n]+): ([^\n]+)"
    )
    for match in pattern.finditer(observation):
        value = match.group(3).split(" [SEP]", 1)[0].strip()
        cards.append(
            {
                "title": match.group(1).strip(),
                "attribute_name": match.group(2).strip(),
                "attribute_value": value,
            }
        )
    return cards


def _page_title(observation: str) -> str | None:
    matches = re.findall(r"\[SEP\] ([^\n\[]+?) \[SEP\] Price:", observation)
    return matches[-1].strip() if matches else None


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _selected_card(step: dict[str, Any]) -> dict[str, str] | None:
    observation = _user_input(step)
    page_title = _page_title(observation)
    if page_title is None:
        return None
    normalized_page = _normalize(page_title)
    for card in _parse_cards(observation):
        if _normalize(card["title"]) == normalized_page:
            return card
    return None


def _semantic_source_evidence(
    contents: Iterable[str], card: dict[str, str] | None, phase: int
) -> dict[str, Any]:
    content = "\n".join(contents)
    lowered = _normalize(content)
    if card is None:
        return {
            "status": "selected_card_unresolved",
            "expected_value_present": False,
            "expected_field_present": False,
            "explicit_attribute_conflict": False,
            "refers_only_to_earlier_session": False,
        }

    expected_value = _normalize(card["attribute_value"])
    expected_field = _normalize(card["attribute_name"])
    value_present = expected_value in lowered
    field_present = expected_field in lowered or expected_field.removeprefix("listed ") in lowered
    assignments = re.findall(
        r"(?:attribute|flavor|colour|color|finish|material)\s*[:=]\s*([^\n]+)",
        content,
        flags=re.IGNORECASE,
    )
    explicit_conflict = bool(assignments) and all(
        expected_value not in _normalize(assignment) for assignment in assignments
    )
    mentioned_sessions = {
        int(value) for value in re.findall(r"\bsession\s*([0-9]+)\b", content, re.I)
    }
    current_session = phase + 1
    earlier_only = bool(mentioned_sessions) and all(
        value < current_session for value in mentioned_sessions
    )

    if earlier_only:
        status = "earlier_session_fact_not_current_purchase"
    elif not value_present:
        status = "wrong_or_irrelevant"
    elif explicit_conflict:
        status = "expected_value_present_but_attribute_conflicts"
    elif field_present:
        status = "exact_field_value"
    else:
        status = "expected_value_present_implicitly"
    return {
        "status": status,
        "expected_value_present": value_present,
        "expected_field_present": field_present,
        "explicit_attribute_conflict": explicit_conflict,
        "refers_only_to_earlier_session": earlier_only,
    }


def _content_observed(contents: Iterable[str], stdout: str) -> bool:
    visible = _normalize(stdout)
    evidence_lines = [
        _normalize(line)
        for content in contents
        for line in content.splitlines()
        if len(_normalize(line)) >= 4
    ]
    return bool(evidence_lines) and any(line in visible for line in evidence_lines)


def _shell_command(step: dict[str, Any], event: dict[str, Any]) -> str:
    command = event.get("command")
    if isinstance(command, str) and command:
        return command
    action = str(step.get("action_submitted", ""))
    match = re.fullmatch(r"shell_command\s+(\{.*\})", action, flags=re.DOTALL)
    if match is None:
        return ""
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return ""
    parsed = payload.get("command") if isinstance(payload, dict) else None
    return parsed if isinstance(parsed, str) else ""


def _correct_buys(steps: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    buys = []
    for index, step in enumerate(steps):
        before = int(step.get("env_info_before", {}).get("current_subtask_index", 0))
        after = int(step.get("env_info_after", {}).get("current_subtask_index", before))
        if after > before:
            buys.append((index, before, step))
    return buys


def _source_writes(
    steps: list[dict[str, Any]], correct_buys: list[tuple[int, int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    sources = []
    for index, step in enumerate(steps):
        event = _workspace_event(step)
        if not event or event.get("op") != "APPLY_PATCH":
            continue
        diff = event.get("workspace_diff", {})
        versions = list(diff.get("added", [])) + list(diff.get("modified", []))
        if not versions:
            continue
        phase = int(step.get("env_info_before", {}).get("current_subtask_index", 0))
        next_buy = next(
            (
                (buy_index, buy_step)
                for buy_index, buy_phase, buy_step in correct_buys
                if buy_phase == phase and buy_index > index
            ),
            None,
        )
        if next_buy is None:
            continue
        add_contents = _parse_add_file_contents(str(step.get("action_submitted", "")))
        files = []
        for version in versions:
            path = str(version.get("path", ""))
            files.append(
                {
                    "path": path,
                    "sha256": version.get("sha256"),
                    "bytes": version.get("bytes"),
                    "content": add_contents.get(path),
                    "content_reconstruction": (
                        "complete_add_file" if path in add_contents else "unavailable"
                    ),
                }
            )
        card = _selected_card(next_buy[1])
        evidence = _semantic_source_evidence(
            [item["content"] or "" for item in files], card, phase
        )
        sources.append(
            {
                "step_index": index,
                "turn": int(step.get("turn", index + 1)),
                "phase": phase,
                "next_correct_buy_turn": int(next_buy[1].get("turn", 0)),
                "next_correct_buy_step_index": next_buy[0],
                "model_text": step.get("model_text"),
                "files": files,
                "selected_card_at_buy": card,
                "semantic_evidence": evidence,
            }
        )
    return sources


def _later_shell_links(
    steps: list[dict[str, Any]],
    correct_buys: list[tuple[int, int, dict[str, Any]]],
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    links = []
    for index, step in enumerate(steps):
        event = _workspace_event(step)
        if not event or event.get("op") != "SHELL_COMMAND":
            continue
        phase = int(step.get("env_info_before", {}).get("current_subtask_index", 0))
        snapshot_versions = {
            (str(item.get("path", "")), str(item.get("sha256", "")))
            for item in step.get("env_info_after", {})
            .get("workspace_snapshot", {})
            .get("files", [])
        }
        for source in sources:
            source_versions = {
                (str(item["path"]), str(item["sha256"])) for item in source["files"]
            }
            if not (
                source["phase"] < phase
                and source["next_correct_buy_step_index"] < index
                and source_versions.intersection(snapshot_versions)
            ):
                continue
            later_buy = next(
                (
                    buy_step
                    for buy_index, buy_phase, buy_step in correct_buys
                    if buy_phase == phase and buy_index > index
                ),
                None,
            )
            command = _shell_command(step, event)
            stdout = str(event.get("stdout", ""))
            contents = [item["content"] or "" for item in source["files"]]
            content_seen = _content_observed(contents, stdout)
            path_mentioned = any(item["path"] in command for item in source["files"])
            source_status = str(source["semantic_evidence"]["status"])
            semantic_ok = source_status in {
                "exact_field_value",
                "expected_value_present_implicitly",
            }
            strict = bool(content_seen and later_buy is not None and semantic_ok)
            links.append(
                {
                    "source_turn": source["turn"],
                    "source_phase": source["phase"],
                    "source_paths": [item["path"] for item in source["files"]],
                    "source_semantic_status": source_status,
                    "shell_turn": int(step.get("turn", index + 1)),
                    "shell_phase": phase,
                    "command": command,
                    "exit_code": event.get("exit_code"),
                    "stdout": stdout,
                    "stderr": event.get("stderr"),
                    "source_path_explicitly_referenced": path_mentioned,
                    "source_content_observed_in_stdout": content_seen,
                    "later_correct_buy_turn": (
                        int(later_buy.get("turn", 0)) if later_buy is not None else None
                    ),
                    "strict_content_chain": strict,
                }
            )
    return links


def _io_record(step: dict[str, Any]) -> dict[str, Any]:
    exact_model_input = step.get("exact_model_input")
    return {
        "turn": step.get("turn"),
        "phase_before": step.get("env_info_before", {}).get("current_subtask_index"),
        "phase_after": step.get("env_info_after", {}).get("current_subtask_index"),
        "exact_model_input_sha256": (
            hashlib.sha256(exact_model_input.encode("utf-8")).hexdigest()
            if isinstance(exact_model_input, str)
            else None
        ),
        "exact_model_input": exact_model_input,
        "request_messages_sha256": _sha256_json(step.get("request_messages", [])),
        "request_messages": step.get("request_messages", []),
        "raw_model_response": step.get("raw_model_response"),
        "model_text": step.get("model_text"),
        "action_submitted": step.get("action_submitted"),
        "action_submission": step.get("action_submission"),
        "environment_step_request": step.get("environment_step_request"),
        "env_response": step.get("env_response"),
        "reward": step.get("reward"),
        "reward_components": step.get("reward_components", []),
        "workspace_event": _workspace_event(step),
    }


def _analyze_episode_data(
    episode: dict[str, Any], episode_path: str
) -> dict[str, Any]:
    steps = list(episode.get("steps", []))
    buys = _correct_buys(steps)
    sources = _source_writes(steps, buys)
    links = _later_shell_links(steps, buys, sources)
    filesystem_io = []
    invalid_io = []
    wrong_buy_io = []
    for step in steps:
        text = str(step.get("model_text", ""))
        event = _workspace_event(step)
        components = _component_names(step)
        mentions_filesystem = "apply_patch" in text.casefold() or "shell_command" in text.casefold()
        if mentions_filesystem or event is not None:
            filesystem_io.append(_io_record(step))
        if "invalid_action" in components:
            invalid_io.append(_io_record(step))
        if "buy_committed_incorrect" in components:
            wrong_buy_io.append(_io_record(step))

    post_transition_writes = [
        step
        for step in filesystem_io
        if step["workspace_event"]
        and step["workspace_event"].get("op") == "APPLY_PATCH"
        and int(step["phase_before"] or 0) > 0
        and (
            step["workspace_event"].get("workspace_diff", {}).get("added")
            or step["workspace_event"].get("workspace_diff", {}).get("modified")
        )
    ]
    return {
        "episode_path": episode_path,
        "rollout_step": episode.get("rollout_step"),
        "policy_updates_before_rollout": episode.get("policy_updates_before_rollout"),
        "trajectory_uid": episode.get("trajectory_uid"),
        "data_idx": episode.get("data_idx"),
        "final_phase_progress": episode.get("final_phase_progress"),
        "episode_return": episode.get("episode_return"),
        "episode_success": episode.get("episode_success"),
        "timed_out": episode.get("timed_out"),
        "turn_count": len(steps),
        "correct_buy_turns": [int(step.get("turn", 0)) for _, _, step in buys],
        "source_write_timing_candidates": sources,
        "later_shell_timing_links": links,
        "strict_content_chain_count": sum(link["strict_content_chain"] for link in links),
        "post_transition_content_write_count": len(post_transition_writes),
        "filesystem_io": filesystem_io,
        "invalid_io": invalid_io,
        "wrong_buy_io": wrong_buy_io,
    }


def _analyze_episode(path: Path, run_dir: Path) -> dict[str, Any]:
    episode = json.loads(path.read_text(encoding="utf-8"))
    return _analyze_episode_data(episode, str(path.relative_to(run_dir)))


def _training_step_from_row(row: dict[str, Any]) -> dict[str, Any]:
    record = row.get("formal_step_record")
    if not isinstance(record, dict):
        raise ValueError("training row has no formal_step_record")
    submission = record.get("action_submission")
    if not isinstance(submission, dict):
        submission = {}
    raw_output = submission.get("raw_policy_output", record.get("action", ""))
    submitted = submission.get("submitted_action", record.get("action", ""))
    env_info_before = record.get("env_info_before")
    env_info_after = record.get("env_info_after")
    if not isinstance(env_info_before, dict) or not isinstance(env_info_after, dict):
        raise ValueError("training formal_step_record lacks environment state")
    latest_observation = str(record.get("latest_observation", ""))
    immediate_reward = record.get("immediate_reward", row.get("agentmemory_immediate_reward"))
    env_result = str(record.get("env_result", ""))
    return {
        "turn": int(record.get("trajectory_row_order", 0)) + 1,
        "exact_model_input": record.get("visible_prompt"),
        "request_messages": [{"role": "user", "content": latest_observation}],
        "raw_model_response": {
            "raw_policy_output": raw_output,
            "generation_response_digest": record.get("generation_response_digest"),
            "generation_response_length": record.get("generation_response_length"),
            "finish_reason": record.get("finish_reason"),
            "generation_stop_reason": record.get("generation_stop_reason"),
        },
        "model_text": raw_output,
        "action_submitted": submitted,
        "action_submission": submission,
        "environment_step_request": {
            "action": submitted,
            "item_id": record.get("item_id"),
        },
        "env_response": {
            "observation": env_result,
            "reward": immediate_reward,
            "done": record.get("done"),
            "info": env_info_after,
        },
        "env_info_before": env_info_before,
        "env_info_after": env_info_after,
        "reward": immediate_reward,
        "reward_components": env_info_after.get("reward_components", []),
    }


def _training_episodes(path: Path, run_dir: Path) -> list[dict[str, Any]]:
    match = TRAINING_DIAGNOSTIC_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Cannot infer rollout step from {path}")
    rollout_step = int(match.group(1))
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Training diagnostic has no rows list: {path}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("ppo_valid_sample", True):
            continue
        record = row.get("formal_step_record")
        if not isinstance(record, dict):
            continue
        trajectory_uid = str(
            record.get("trajectory_uid") or row.get("agentmemory_trajectory_uid") or ""
        )
        if not trajectory_uid:
            raise ValueError(f"Training row has no trajectory uid: {path}")
        grouped.setdefault(trajectory_uid, []).append(_training_step_from_row(row))
        metadata[trajectory_uid] = {
            "data_idx": record.get("parent_index", row.get("parent_index")),
            "trajectory_return": record.get(
                "trajectory_return", row.get("agentmemory_trajectory_return")
            ),
        }

    episodes = []
    for trajectory_uid, steps in sorted(grouped.items()):
        steps.sort(key=lambda step: int(step["turn"]))
        last = steps[-1]
        final_info = last["env_info_after"]
        final_progress = int(final_info.get("current_subtask_index", 0))
        episode = {
            "rollout_step": rollout_step,
            "policy_updates_before_rollout": max(rollout_step - 1, 0),
            "trajectory_uid": trajectory_uid,
            "data_idx": metadata[trajectory_uid]["data_idx"],
            "final_phase_progress": final_progress,
            "episode_return": metadata[trajectory_uid]["trajectory_return"],
            "episode_success": bool(final_info.get("episode_success", False)),
            "timed_out": str(last.get("env_response", {}).get("observation", "")).startswith(
                "Maximum rounds reached"
            ),
            "steps": steps,
        }
        episode_path = (
            f"{path.relative_to(run_dir)}#trajectory_uid={trajectory_uid}"
        )
        episodes.append(_analyze_episode_data(episode, episode_path))
    return episodes


def analyze_run(run_dir: Path, layout: str = "auto") -> dict[str, Any]:
    paths = sorted(run_dir.glob("replicas/*/output/episode_*.json"))
    diagnostic_paths = [
        path
        for path in run_dir.glob("diagnostics/ppo_batch_step*_post_adv.json")
        if TRAINING_DIAGNOSTIC_RE.fullmatch(path.name)
    ]
    diagnostic_paths.sort(
        key=lambda path: int(TRAINING_DIAGNOSTIC_RE.fullmatch(path.name).group(1))
    )
    if layout not in {"auto", "eval", "training"}:
        raise ValueError(f"Unsupported layout: {layout}")
    if layout == "eval" or (layout == "auto" and paths):
        if not paths:
            raise ValueError(f"No eval episode JSON files found under {run_dir}")
        source_layout = "eval_replicas"
        episodes = [_analyze_episode(path, run_dir) for path in paths]
    else:
        if not diagnostic_paths:
            raise ValueError(f"No training diagnostic JSON files found under {run_dir}")
        source_layout = "training_post_adv"
        episodes = [
            episode
            for path in diagnostic_paths
            for episode in _training_episodes(path, run_dir)
        ]

    invalid_reasons: Counter[str] = Counter()
    accepted_patch_count = 0
    accepted_shell_count = 0
    patch_intent_count = 0
    shell_intent_count = 0
    patch_prefix_count = 0
    shell_prefix_count = 0
    for episode in episodes:
        for record in episode["filesystem_io"]:
            text = str(record.get("model_text", ""))
            if "apply_patch" in text.casefold():
                patch_intent_count += 1
            if "shell_command" in text.casefold():
                shell_intent_count += 1
            if text.startswith("apply_patch\n"):
                patch_prefix_count += 1
            if text.startswith("shell_command "):
                shell_prefix_count += 1
            event = record.get("workspace_event")
            if event and event.get("op") == "APPLY_PATCH":
                accepted_patch_count += 1
            elif event and event.get("op") == "SHELL_COMMAND":
                accepted_shell_count += 1
        for record in episode["invalid_io"]:
            invalid_reasons[_first_feedback_line(record)] += 1

    sources = [
        source
        for episode in episodes
        for source in episode["source_write_timing_candidates"]
    ]
    links = [
        link for episode in episodes for link in episode["later_shell_timing_links"]
    ]
    source_episodes = sum(
        bool(episode["source_write_timing_candidates"]) for episode in episodes
    )
    strict_episodes = sum(bool(episode["strict_content_chain_count"]) for episode in episodes)
    unique_later_shells = {
        (episode["episode_path"], link["shell_turn"])
        for episode in episodes
        for link in episode["later_shell_timing_links"]
    }
    rollout_steps = sorted(
        {
            int(episode["rollout_step"])
            for episode in episodes
            if episode["rollout_step"] is not None
        }
    )
    summary = {
        "source_layout": source_layout,
        "trajectory_count": len(episodes),
        "trajectory_count_by_rollout_step": {
            str(step): count
            for step, count in sorted(
                Counter(
                    episode["rollout_step"]
                    for episode in episodes
                    if episode["rollout_step"] is not None
                ).items()
            )
        },
        "strict_content_chain_count_by_rollout_step": {
            str(step): sum(
                int(episode["strict_content_chain_count"])
                for episode in episodes
                if episode["rollout_step"] == step
            )
            for step in rollout_steps
        },
        "filesystem_intent_record_count": sum(
            len(episode["filesystem_io"]) for episode in episodes
        ),
        "apply_patch_mention_count": patch_intent_count,
        "submitted_apply_patch_prefix_count": patch_prefix_count,
        "accepted_apply_patch_count": accepted_patch_count,
        "shell_command_mention_count": shell_intent_count,
        "submitted_shell_command_prefix_count": shell_prefix_count,
        "accepted_shell_command_count": accepted_shell_count,
        "invalid_action_count": sum(len(episode["invalid_io"]) for episode in episodes),
        "invalid_action_first_line_counts": dict(invalid_reasons.most_common()),
        "source_write_action_count": len(sources),
        "source_write_trajectory_count": source_episodes,
        "source_write_trajectory_rate": source_episodes / len(episodes),
        "source_semantic_status_counts": dict(
            Counter(source["semantic_evidence"]["status"] for source in sources)
        ),
        "later_shell_timing_link_count": len(links),
        "later_shell_timing_action_count": len(unique_later_shells),
        "later_shell_explicit_source_path_count": sum(
            link["source_path_explicitly_referenced"] for link in links
        ),
        "later_shell_source_content_observed_count": sum(
            link["source_content_observed_in_stdout"] for link in links
        ),
        "strict_content_chain_count": sum(link["strict_content_chain"] for link in links),
        "strict_content_chain_trajectory_count": strict_episodes,
        "post_transition_content_write_count": sum(
            episode["post_transition_content_write_count"] for episode in episodes
        ),
        "session1_failure_trajectory_count": sum(
            not episode["correct_buy_turns"] for episode in episodes
        ),
        "wrong_buy_count": sum(len(episode["wrong_buy_io"]) for episode in episodes),
    }
    aggregate_path = run_dir / "aggregate.json"
    aggregate = (
        json.loads(aggregate_path.read_text(encoding="utf-8"))
        if aggregate_path.exists()
        else None
    )
    return {
        "schema": "agentmemory_filesystem_exact_io_audit_v1",
        "run_dir": str(run_dir),
        "run_aggregate": aggregate,
        "summary": summary,
        "episodes": episodes,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    status_counts = summary["source_semantic_status_counts"]
    non_exact_sources = sum(
        count for status, count in status_counts.items() if status != "exact_field_value"
    )
    if not summary["source_write_action_count"]:
        source_semantics = (
            "- Source semantics: no source-write timing candidate was observed."
        )
    elif non_exact_sources:
        source_semantics = (
            f"- Source semantics: `{status_counts}`. {non_exact_sources} source-write "
            "timing candidates did not contain the selected card's exact field/value pair."
        )
    else:
        source_semantics = (
            f"- Source semantics: `{status_counts}`. Every source-write timing candidate "
            "contained the selected card's exact field/value pair; any remaining failure is "
            "in timing, later readback, or the dependent purchase."
        )
    training_rollout_lines = []
    if summary.get("source_layout") == "training_post_adv":
        training_rollout_lines = [
            "- Training rollout steps use pre-update sampling semantics: step 1 is the base "
            "policy, and step N is sampled after N-1 optimizer updates.",
            f"- Trajectories by rollout step: "
            f"`{summary['trajectory_count_by_rollout_step']}`; strict chains by rollout "
            f"step: `{summary['strict_content_chain_count_by_rollout_step']}`.",
        ]
    lines = [
        "# Filesystem behavior exact I/O audit",
        "",
        f"Run: `{audit['run_dir']}`",
        "",
        "## Verdict",
        "",
        f"- Source layout: `{summary.get('source_layout', 'eval_replicas')}`.",
        f"- Strict same-content cross-session chains: **{summary['strict_content_chain_count']}**.",
        f"- Timing-only source writes: {summary['source_write_action_count']} actions in "
        f"{summary['source_write_trajectory_count']}/{summary['trajectory_count']} trajectories.",
        f"- Timing-only later-shell actions: {summary['later_shell_timing_action_count']} "
        f"({summary['later_shell_timing_link_count']} source/action pairs); "
        f"source content actually observed: {summary['later_shell_source_content_observed_count']}.",
        *training_rollout_lines,
        "- A file merely remaining in the workspace does not count as retrieval.",
        "",
        "## Counts",
        "",
        f"- apply_patch mentions / accepted: {summary['apply_patch_mention_count']} / "
        f"{summary['accepted_apply_patch_count']}",
        f"- bare apply_patch-prefix submissions: "
        f"{summary['submitted_apply_patch_prefix_count']}",
        f"- shell_command mentions / accepted: {summary['shell_command_mention_count']} / "
        f"{summary['accepted_shell_command_count']}",
        f"- bare shell_command-prefix submissions: "
        f"{summary['submitted_shell_command_prefix_count']}",
        f"- post-transition content writes: {summary['post_transition_content_write_count']}",
        f"- session-1 failures: {summary['session1_failure_trajectory_count']}",
        f"- wrong BUYs: {summary['wrong_buy_count']}",
        "",
        "## Observed failure modes",
        "",
        f"- Only {summary['source_write_trajectory_count']}/{summary['trajectory_count']} "
        "trajectories wrote anything before a later accepted BUY in that session; the "
        f"{summary['source_write_action_count']} reported source writes are actions, not "
        "trajectories.",
        f"- {summary['post_transition_content_write_count']} accepted content writes happened "
        "after at least one session transition, when the preceding session trace had already "
        "been removed from model input.",
        source_semantics,
        f"- {summary['submitted_apply_patch_prefix_count'] - summary['accepted_apply_patch_count']}"
        f"/{summary['submitted_apply_patch_prefix_count']} bare apply_patch submissions were "
        "rejected; malformed Update File hunks, repeated Add File paths, appended feedback, "
        "and non-canonical wrappers recur in the raw I/O.",
        f"- {summary['session1_failure_trajectory_count']} trajectories failed before memory was "
        "needed, so native search/navigation and one-action formatting remain a separate "
        "failure surface.",
        "",
        "## Source-write timing candidates",
        "",
    ]
    for episode in audit["episodes"]:
        for source in episode["source_write_timing_candidates"]:
            files = ", ".join(item["path"] for item in source["files"])
            card = source["selected_card_at_buy"] or {}
            lines.extend(
                [
                    f"- `{episode['episode_path']}` turn {source['turn']} phase "
                    f"{source['phase']}: `{files}`",
                    f"  - semantic status: `{source['semantic_evidence']['status']}`; "
                    f"expected `{card.get('attribute_name', '?')} = "
                    f"{card.get('attribute_value', '?')}`",
                    f"  - output: `{str(source['model_text']).replace(chr(10), ' | ')}`",
                ]
            )
    lines.extend(["", "## Later-shell timing links", ""])
    for episode in audit["episodes"]:
        for link in episode["later_shell_timing_links"]:
            lines.extend(
                [
                    f"- `{episode['episode_path']}` source turn {link['source_turn']} -> "
                    f"shell turn {link['shell_turn']}: `{link['command']}`",
                    f"  - source path referenced: `{link['source_path_explicitly_referenced']}`; "
                    f"source content observed: `{link['source_content_observed_in_stdout']}`; "
                    f"later correct BUY: `{link['later_correct_buy_turn']}`",
                    f"  - stdout: `{str(link['stdout']).replace(chr(10), ' | ')}`",
                ]
            )
    if summary["strict_content_chain_count"]:
        interpretation = (
            "The run demonstrates at least one strict same-content cross-session chain. "
            "Inspect `exact-io-audit.json` for the exact request messages, raw completion, "
            "submitted action, environment feedback, and workspace event."
        )
    elif summary["source_write_action_count"]:
        interpretation = (
            "The run activates legal filesystem writes, but it does not demonstrate a "
            "functional memory chain. Inspect `exact-io-audit.json` for the exact request "
            "messages, raw completion, submitted action, environment feedback, and workspace "
            "event for every filesystem-related turn."
        )
    else:
        interpretation = (
            "The run does not demonstrate a source write or a functional memory chain. "
            "Inspect `exact-io-audit.json` for the exact request messages, raw completion, "
            "submitted action, environment feedback, and workspace event."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--layout", choices=("auto", "eval", "training"), default="auto")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    audit = analyze_run(run_dir, layout=args.layout)
    json_text = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown_text = render_markdown(audit)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text, encoding="utf-8")
    else:
        print(json_text, end="")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown_text, encoding="utf-8")


if __name__ == "__main__":
    main()
