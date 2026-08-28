#!/usr/bin/env python3
"""Audit SWE-smith's receipt-backed filesystem context-compaction contract.

The operational chain is:

1. a normal policy-authored workspace action changes a bounded checkpoint file;
2. the wrapper verifies that exact file and mechanically emits replace_messages;
3. a later normal policy action reads the same non-empty file identity; and
4. a still-later accepted task action continues the episode.

The last step proves behavioral continuation, not causal benefit. Task success is
reported separately and causal lift remains a formal-run comparison question.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import shlex
from pathlib import Path

CHECKPOINT_SCHEMA = "agentmemory_filesystem_checkpoint_receipt_v1"
TRANSITION_SCHEMA = "agentmemory_task_neutral_context_transition_v1"


def normalize_path(value):
    value = str(value or "").strip().strip("'\"`),;:").replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    if value.startswith("/testbed/"):
        value = value[len("/testbed/") :]
    return value.rstrip("/")


def positive_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def sha256_text(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def actor_credit(record):
    for holder in (
        record.get("wrapper_evidence") or {},
        record.get("env_info_after") or {},
    ):
        credit = holder.get("actor_credit") or {}
        if isinstance(credit, dict) and credit.get("basis"):
            return credit
    return {}


def workspace_action(record):
    submission = record.get("action_submission") or {}
    info = record.get("env_info_after") or {}
    operation = str(submission.get("op") or "").upper()
    kind = str(submission.get("kind") or "").lower()
    action_kind = str(info.get("action_kind") or "").lower()
    raw = str(submission.get("raw_policy_output") or record.get("action") or "")
    return (
        kind == "workspace"
        or operation in {"SHELL_COMMAND", "APPLY_PATCH"}
        or action_kind in {"shell_command", "apply_patch"}
        or raw.startswith("shell_command ")
        or raw.startswith("apply_patch\n")
    )


def action_completed(record):
    credit = actor_credit(record)
    if credit.get("positive_eligible") is False:
        return False
    if credit.get("basis") in {"parser_rejected", "executor_rejected"}:
        return False
    info = record.get("env_info_after") or {}
    if info.get("action_kind") == "parser_error" or info.get("action_status") in {
        "parser_error",
        "failed",
    }:
        return False
    receipt = info.get("filesystem_checkpoint")
    if isinstance(receipt, dict):
        return receipt.get("action_completed") is True
    execution = info.get("execution") or {}
    if not isinstance(execution, dict):
        return False
    return execution.get("status") in (None, "completed", "success") and execution.get(
        "exit_code"
    ) in (None, 0)


def parse_shell_command(record):
    submission = record.get("action_submission") or {}
    raw = submission.get("raw_policy_output")
    if not isinstance(raw, str) or raw != record.get("action"):
        return None
    prefix = "shell_command "
    if not raw.startswith(prefix):
        return None
    try:
        payload = json.loads(raw[len(prefix) :])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    command = payload.get("command") if isinstance(payload, dict) else None
    return command if isinstance(command, str) else None


def parse_shell_words(record):
    command = parse_shell_command(record)
    if command is None:
        return None
    try:
        return shlex.split(command)
    except ValueError:
        return None


_READ_COMMANDS = {"awk", "cat", "grep", "head", "less", "more", "nl", "sed", "tail", "wc"}
_SHELL_SEGMENT_SEPARATOR = re.compile(r"(?:&&|\|\||[;\n])")


def _command_mentions_path(command, path):
    variants = {path, f"./{path}", f"/testbed/{path}"}
    return any(variant in command for variant in variants)


def _segment_reads_path(segment, path):
    """Conservatively recognize ordinary shell readers of one exact file.

    This deliberately does not infer a read from a path mention alone.  In
    particular, ``cat > PATH`` and heredoc writes remain non-reads.
    """

    try:
        words = shlex.split(segment)
    except ValueError:
        return False
    while words and ("=" in words[0] and not words[0].startswith(("/", "./"))):
        words.pop(0)
    if words[:1] == ["command"]:
        words.pop(0)
    if not words:
        return False
    executable = words[0].rsplit("/", 1)[-1]
    if executable not in _READ_COMMANDS:
        if executable.startswith("python") and _command_mentions_path(segment, path):
            return bool(re.search(r"\b(?:open|read_text|read_bytes)\s*\(", segment))
        return False
    for index, word in enumerate(words[1:], start=1):
        if normalize_path(word) != path:
            continue
        previous = words[index - 1] if index else ""
        if previous in {">", ">>", "1>", "1>>", "2>", "2>>", "<<", "<<<"}:
            continue
        if word.startswith((">", ">>")):
            continue
        return True
    return False


def checkpoint_read_class(record, checkpoint):
    """Classify exact, broader attested, and mention-only checkpoint actions."""

    path, digest, size = checkpoint
    command = parse_shell_command(record)
    if command is None:
        return None
    mentioned = _command_mentions_path(command, path)
    info = record.get("env_info_after") or {}
    evidence = record.get("wrapper_evidence") or {}
    transition = record.get("context_transition") or {}
    observed = valid_receipt(info.get("filesystem_checkpoint"), require_changed=False)
    eligible = (
        observed == (path, digest, size)
        and isinstance(evidence, dict)
        and evidence.get("event") == "native_action"
        and actor_credit(record).get("positive_eligible") is True
        and isinstance(transition, dict)
        and transition.get("operation") == "append_observation"
        and action_completed(record)
    )
    words = parse_shell_words(record)
    if eligible and words in (["cat", path], ["cat", "--", path]):
        return "strict_exact"
    if eligible and any(
        _segment_reads_path(segment, path)
        for segment in _SHELL_SEGMENT_SEPARATOR.split(command)
        if segment.strip()
    ):
        return "broader_attested"
    if mentioned:
        return "path_mentioned_only"
    return None


def valid_receipt(value, *, require_changed=None):
    if not isinstance(value, dict) or value.get("schema") != CHECKPOINT_SCHEMA:
        return None
    if value.get("action_completed") is not True:
        return None
    if value.get("action_kind") not in {"shell_command", "apply_patch"}:
        return None
    if value.get("exists") is not True or value.get("regular_file") is not True:
        return None
    if require_changed is not None and value.get("changed") is not require_changed:
        return None
    path = normalize_path(value.get("path"))
    digest = value.get("sha256")
    size = positive_int(value.get("size_bytes"))
    if not path or not sha256_text(digest) or size is None:
        return None
    return path, digest, size


def checkpoint_write(record):
    evidence = record.get("wrapper_evidence") or {}
    info = record.get("env_info_after") or {}
    transition = record.get("context_transition") or {}
    if not all(isinstance(value, dict) for value in (evidence, info, transition)):
        return None
    evidence_receipt = evidence.get("checkpoint_receipt")
    info_receipt = info.get("filesystem_checkpoint")
    identity = valid_receipt(evidence_receipt, require_changed=True)
    if identity is None or evidence_receipt != info_receipt:
        return None
    path, _digest, size = identity
    maximum = positive_int(evidence.get("checkpoint_max_bytes", 8192))
    messages = transition.get("messages")
    if (
        maximum is None
        or size > maximum
        or evidence.get("event") != "context_compaction"
        or evidence.get("context_replaced") is not True
        or evidence.get("continuation_persisted") is not True
        or normalize_path(evidence.get("continuation_path")) != path
        or evidence.get("replacement_contains_policy_output") is not False
        or evidence.get("replacement_contains_native_observation") is not False
        or evidence.get("sampled_policy_output_preserved_in_ledger") is not True
        or evidence.get("native_observation_preserved_in_ledger") is not True
        or transition.get("schema") != TRANSITION_SCHEMA
        or transition.get("operation") != "replace_messages"
        or not isinstance(messages, list)
        or not isinstance(record.get("control_request"), str)
        or not record["control_request"].strip()
    ):
        return None
    return identity


def is_compaction_event(record):
    evidence = record.get("wrapper_evidence") or {}
    return isinstance(evidence, dict) and evidence.get("event") == "context_compaction"


def is_context_replacement(record):
    transition = record.get("context_transition") or {}
    return isinstance(transition, dict) and transition.get("operation") == "replace_messages"


def explicitly_nonterminal(record):
    flags = [
        record[key]
        for key in ("rollout_done_flag", "trajectory_terminal")
        if key in record
    ]
    return bool(flags) and all(flag is False for flag in flags)


def valid_failed_checkpoint_feedback(record):
    """Validate v3 failed-checkpoint evidence without hiding the failed turn."""

    evidence = record.get("wrapper_evidence") or {}
    transition = record.get("context_transition") or {}
    retry_pending = evidence.get("retry_pending")
    retry_exhausted = evidence.get("retry_exhausted")
    attempt = positive_int(evidence.get("checkpoint_attempt_count"))
    maximum = positive_int(evidence.get("checkpoint_max_attempts"))
    return (
        isinstance(evidence, dict)
        and evidence.get("event") == "context_compaction"
        and evidence.get("continuation_persisted") is False
        and evidence.get("context_replaced") is False
        and evidence.get("retry_context_restored") is False
        and evidence.get("retry_feedback_preserved") is True
        and type(retry_pending) is bool
        and type(retry_exhausted) is bool
        and retry_pending is not retry_exhausted
        and attempt is not None
        and maximum is not None
        and attempt <= maximum
        and retry_pending is (attempt < maximum)
        and explicitly_nonterminal(record)
        and isinstance(transition, dict)
        and transition.get("schema") == TRANSITION_SCHEMA
        and transition.get("operation") == "append_observation"
        and isinstance(record.get("control_request"), str)
        and bool(record["control_request"].strip())
    )


def checkpoint_read(record, checkpoint):
    return checkpoint_read_class(record, checkpoint) == "strict_exact"


def successful_task_action(record, checkpoint):
    if not workspace_action(record) or not action_completed(record):
        return False
    evidence = record.get("wrapper_evidence") or {}
    transition = record.get("context_transition") or {}
    return (
        isinstance(evidence, dict)
        and evidence.get("event") == "native_action"
        and actor_credit(record).get("positive_eligible") is True
        and isinstance(transition, dict)
        and transition.get("operation") == "append_observation"
        and checkpoint_read_class(record, checkpoint)
        not in {"strict_exact", "broader_attested"}
    )


def load_groups(run_dir, through=None):
    groups = collections.defaultdict(list)
    files = []
    rollout_dir = Path(run_dir) / "rollout_data"
    candidates = [path for path in rollout_dir.glob("*.jsonl") if path.stem.isdecimal()]
    for path in sorted(candidates, key=lambda candidate: int(candidate.stem)):
        update = int(path.stem)
        if through is not None and update > through:
            continue
        files.append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        with path.open(errors="replace") as handle:
            for line in handle:
                try:
                    outer = json.loads(line)
                    record = json.loads(outer["step_record_json"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                record["_update"] = update
                groups[(update, record.get("trajectory_uid"))].append(record)
    for rows in groups.values():
        rows.sort(key=lambda record: int(record.get("trajectory_row_order", 0)))
    return groups, files


def compaction_segments(rows):
    segments = []
    current = []
    for index, record in enumerate(rows):
        if is_compaction_event(record):
            current.append((index, record))
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def compact_attempt(update, uid, record):
    receipt = (record.get("wrapper_evidence") or {}).get("checkpoint_receipt")
    return {
        "update": update,
        "trajectory_uid": uid,
        "item_id": record.get("item_id"),
        "row_order": record.get("trajectory_row_order"),
        "action": str(record.get("action") or "")[:1000],
        "workspace_action": workspace_action(record),
        "action_completed": action_completed(record),
        "replace_messages": is_context_replacement(record),
        "retry_control": (
            (record.get("context_transition") or {}).get("operation")
            == "retry_control"
        ),
        "valid_failed_checkpoint_feedback": valid_failed_checkpoint_feedback(record),
        "receipt": receipt if isinstance(receipt, dict) else None,
        "valid_write_and_replacement": checkpoint_write(record) is not None,
        "response_token_count": record.get("response_token_count"),
        "actor_credit_basis": actor_credit(record).get("basis"),
        "trajectory_terminal_at_attempt": bool(record.get("trajectory_terminal")),
    }


def compact_case_row(record):
    return {
        "row_order": record.get("trajectory_row_order"),
        "action": str(record.get("action") or "")[:1600],
        "control_request": str(record.get("control_request") or "")[:1000] or None,
        "context_operation": (record.get("context_transition") or {}).get("operation"),
        "event": (record.get("wrapper_evidence") or {}).get("event"),
        "actor_credit_basis": actor_credit(record).get("basis"),
        "outcome": record.get("outcome"),
        "trajectory_terminal": bool(record.get("trajectory_terminal")),
        "trajectory_return": record.get("trajectory_return"),
    }


def build_case(groups, key, label, focus_orders=()):
    if key is None or key not in groups:
        return None
    rows = groups[key]
    focus = set(focus_orders)
    selected = [record for record in rows if record.get("trajectory_row_order") in focus]
    if rows and rows[-1] not in selected:
        selected.append(rows[-1])
    if not focus:
        selected = rows[-5:]
    return {
        "label": label,
        "update": key[0],
        "trajectory_uid": key[1],
        "item_id": rows[0].get("item_id") if rows else None,
        "row_count": len(rows),
        "terminal_outcome": rows[-1].get("outcome") if rows else None,
        "trajectory_return": rows[-1].get("trajectory_return") if rows else None,
        "selected_rows": [compact_case_row(record) for record in selected],
    }


def analyze(run_dir, through=None, route="swesmith"):
    groups, files = load_groups(run_dir, through)
    if route is not None:
        groups = {
            key: rows
            for key, rows in groups.items()
            if rows and rows[0].get("data_source") in (None, route)
        }

    attempts = []
    opportunities = []
    successful_replacements = []
    invalid_replacements = []
    failed_checkpoint_feedback = []
    retry_exhausted_failures = []
    invalid_retry_transitions = []
    strict_chains = []
    attested_chains = []
    strict_behavioral_chains = []
    attested_behavioral_chains = []
    path_mention_only = []
    proactive_write_trajectories = set()
    strict_read_trajectories = set()
    attested_read_trajectories = set()
    successful_trajectories = {
        key
        for key, rows in groups.items()
        if rows and rows[-1].get("outcome") == "success"
    }

    for (update, uid), rows in sorted(groups.items()):
        for record in rows:
            receipt = valid_receipt(
                (record.get("env_info_after") or {}).get("filesystem_checkpoint"),
                require_changed=True,
            )
            if receipt is not None and not is_compaction_event(record):
                proactive_write_trajectories.add((update, uid))

        for opportunity_index, segment in enumerate(compaction_segments(rows), start=1):
            opportunity = {
                "update": update,
                "trajectory_uid": uid,
                "item_id": rows[0].get("item_id") if rows else None,
                "opportunity_index": opportunity_index,
                "first_row_order": segment[0][1].get("trajectory_row_order"),
                "last_row_order": segment[-1][1].get("trajectory_row_order"),
                "attempt_count": len(segment),
                "successful_replacement_count": 0,
            }
            for index, record in segment:
                attempt = compact_attempt(update, uid, record)
                attempts.append(attempt)
                identity = checkpoint_write(record)
                if is_context_replacement(record) and identity is None:
                    invalid_replacements.append(attempt)
                if identity is None:
                    if valid_failed_checkpoint_feedback(record):
                        failed_checkpoint_feedback.append(attempt)
                        evidence = record.get("wrapper_evidence") or {}
                        if evidence.get("retry_exhausted") is True:
                            retry_exhausted_failures.append(attempt)
                    elif explicitly_nonterminal(record):
                        invalid_retry_transitions.append(attempt)
                    continue
                opportunity["successful_replacement_count"] += 1
                path, digest, size = identity
                replacement = {
                    "update": update,
                    "trajectory_uid": uid,
                    "item_id": record.get("item_id"),
                    "row_order": record.get("trajectory_row_order"),
                    "path": path,
                    "sha256": digest,
                    "size_bytes": size,
                }
                successful_replacements.append(replacement)

                strict_read = None
                attested_read = None
                mention_only_record = None
                for candidate_index in range(index + 1, len(rows)):
                    candidate = rows[candidate_index]
                    read_class = checkpoint_read_class(candidate, identity)
                    if read_class == "strict_exact" and strict_read is None:
                        strict_read = (candidate_index, candidate)
                    if read_class in {"strict_exact", "broader_attested"}:
                        if attested_read is None:
                            attested_read = (candidate_index, candidate, read_class)
                    elif read_class == "path_mentioned_only" and mention_only_record is None:
                        mention_only_record = candidate
                if mention_only_record is not None:
                    path_mention_only.append(
                        {
                            **replacement,
                            "mention_row_order": mention_only_record.get(
                                "trajectory_row_order"
                            ),
                            "mention_action": str(
                                mention_only_record.get("action") or ""
                            )[:1000],
                        }
                    )
                if strict_read is not None:
                    strict_index, strict_record = strict_read
                    strict_read_trajectories.add((update, uid))
                    strict_chain = {
                        **replacement,
                        "read_class": "strict_exact",
                        "read_row_order": strict_record.get("trajectory_row_order"),
                        "write_response_token_count": record.get("response_token_count"),
                        "read_response_token_count": strict_record.get(
                            "response_token_count"
                        ),
                        "trajectory_return": rows[-1].get("trajectory_return"),
                        "terminal_outcome": rows[-1].get("outcome"),
                    }
                    strict_chains.append(strict_chain)
                    strict_task = next(
                        (
                            candidate
                            for candidate in rows[strict_index + 1 :]
                            if successful_task_action(candidate, identity)
                        ),
                        None,
                    )
                    if strict_task is not None:
                        strict_behavioral_chains.append(
                            {
                                **strict_chain,
                                "next_task_action_row_order": strict_task.get(
                                    "trajectory_row_order"
                                ),
                                "next_task_action": str(
                                    strict_task.get("action") or ""
                                )[:1000],
                            }
                        )
                if attested_read is None:
                    continue
                read_index, read_record, read_class = attested_read
                attested_read_trajectories.add((update, uid))
                chain = {
                    **replacement,
                    "read_class": read_class,
                    "read_row_order": read_record.get("trajectory_row_order"),
                    "write_response_token_count": record.get("response_token_count"),
                    "read_response_token_count": read_record.get("response_token_count"),
                    "trajectory_return": rows[-1].get("trajectory_return"),
                    "terminal_outcome": rows[-1].get("outcome"),
                }
                attested_chains.append(chain)
                task_record = next(
                    (
                        candidate
                        for candidate in rows[read_index + 1 :]
                        if successful_task_action(candidate, identity)
                    ),
                    None,
                )
                if task_record is not None:
                    attested_behavioral_chains.append(
                        {
                            **chain,
                            "next_task_action_row_order": task_record.get(
                                "trajectory_row_order"
                            ),
                            "next_task_action": str(task_record.get("action") or "")[
                                :1000
                            ],
                        }
                    )
            opportunities.append(opportunity)

    unresolved = [
        opportunity
        for opportunity in opportunities
        if opportunity["successful_replacement_count"] == 0
    ]
    strict_chain_trajectories = {
        (chain["update"], chain["trajectory_uid"]) for chain in strict_chains
    }
    strict_behavioral_chain_trajectories = {
        (chain["update"], chain["trajectory_uid"])
        for chain in strict_behavioral_chains
    }
    attested_chain_trajectories = {
        (chain["update"], chain["trajectory_uid"]) for chain in attested_chains
    }
    attested_behavioral_chain_trajectories = {
        (chain["update"], chain["trajectory_uid"])
        for chain in attested_behavioral_chains
    }
    ordered_keys = sorted(groups)
    native_success_key = next(
        (key for key in ordered_keys if key in successful_trajectories), None
    )
    ordinary_failure_key = next(
        (
            key
            for key in ordered_keys
            if key not in successful_trajectories
            and not any(is_compaction_event(record) for record in groups[key])
        ),
        None,
    )
    failed_attempt = next(
        (attempt for attempt in failed_checkpoint_feedback),
        next(
            (
                attempt
                for attempt in attempts
                if not attempt["valid_write_and_replacement"]
            ),
            None,
        ),
    )
    failed_key = (
        (failed_attempt["update"], failed_attempt["trajectory_uid"])
        if failed_attempt
        else None
    )
    successful_full_chain = next(
        (
            chain
            for chain in attested_behavioral_chains
            if (chain["update"], chain["trajectory_uid"]) in successful_trajectories
        ),
        None,
    )
    failed_full_chain = next(
        (
            chain
            for chain in attested_behavioral_chains
            if (chain["update"], chain["trajectory_uid"])
            not in successful_trajectories
        ),
        None,
    )

    def chain_case(chain, label):
        if chain is None:
            return None
        key = (chain["update"], chain["trajectory_uid"])
        rows = groups[key]
        focus_orders = tuple(
            record.get("trajectory_row_order")
            for record in rows
            if record.get("trajectory_row_order") >= chain["row_order"]
        )
        return build_case(groups, key, label, focus_orders)

    output = {
        "schema": "amg_swesmith_filesystem_compaction_audit_v3",
        "claim_scope": (
            "Receipt-backed operational behavior only; downstream actions and task "
            "success do not by themselves prove causal memory benefit."
        ),
        "run_dir": str(run_dir),
        "through_update": max([update for update, _uid in groups] or [0]),
        "rollout_files": files,
        "trajectory_count": len(groups),
        "task_success_count": len(successful_trajectories),
        "compaction_action_attempt_count": len(attempts),
        "compaction_opportunity_count": len(opportunities),
        "failed_checkpoint_action_attempt_count": (
            len(attempts) - len(successful_replacements)
        ),
        "successful_replacement_count": len(successful_replacements),
        "invalid_replacement_count": len(invalid_replacements),
        "feedback_preserving_failed_attempt_count": len(failed_checkpoint_feedback),
        "retry_exhausted_failed_attempt_count": len(retry_exhausted_failures),
        "invalid_retry_transition_count": len(invalid_retry_transitions),
        "unresolved_compaction_opportunity_count": len(unresolved),
        "strict_write_compaction_read_chain_count": len(strict_chains),
        "strict_chain_trajectory_count": len(strict_chain_trajectories),
        "attested_write_compaction_read_chain_count": len(attested_chains),
        "attested_chain_trajectory_count": len(attested_chain_trajectories),
        "broader_nonexact_read_chain_count": sum(
            chain["read_class"] == "broader_attested" for chain in attested_chains
        ),
        "path_mentioned_without_attested_read_count": len(path_mention_only),
        # Compatibility fields retain their strict-exact v2 meaning.
        "behavioral_continuation_chain_count": len(strict_behavioral_chains),
        "behavioral_continuation_trajectory_count": len(
            strict_behavioral_chain_trajectories
        ),
        "attested_behavioral_continuation_chain_count": len(
            attested_behavioral_chains
        ),
        "attested_behavioral_continuation_trajectory_count": len(
            attested_behavioral_chain_trajectories
        ),
        "strict_chain_task_success_count": sum(
            (chain["update"], chain["trajectory_uid"]) in successful_trajectories
            for chain in strict_chains
        ),
        "attested_chain_task_success_count": sum(
            (chain["update"], chain["trajectory_uid"]) in successful_trajectories
            for chain in attested_chains
        ),
        "proactive_checkpoint_write_trajectory_count": len(
            proactive_write_trajectories
        ),
        "post_compaction_read_trajectory_count": len(attested_read_trajectories),
        "strict_post_compaction_read_trajectory_count": len(
            strict_read_trajectories
        ),
        # Compatibility aliases for existing monitors, with v2 receipt semantics.
        "compaction_event_count": len(attempts),
        "valid_executed_checkpoint_write_count": len(successful_replacements),
        "transition_without_valid_write_count": len(invalid_replacements),
        "forced_checkpoint_events": attempts,
        "compaction_opportunities": opportunities,
        "successful_replacements": successful_replacements,
        "invalid_replacements": invalid_replacements,
        "feedback_preserving_failed_attempts": failed_checkpoint_feedback,
        "retry_exhausted_failed_attempts": retry_exhausted_failures,
        "invalid_retry_transitions": invalid_retry_transitions,
        "unresolved_compaction_opportunities": unresolved,
        "strict_chains": strict_chains,
        "attested_chains": attested_chains,
        "path_mentioned_without_attested_read": path_mention_only,
        "behavioral_continuation_chains": strict_behavioral_chains,
        "attested_behavioral_continuation_chains": attested_behavioral_chains,
        "case_examples": {
            "native_success": build_case(
                groups, native_success_key, "native_success"
            ),
            "ordinary_failure_without_compaction": build_case(
                groups,
                ordinary_failure_key,
                "ordinary_failure_without_compaction",
            ),
            "checkpoint_failure": build_case(
                groups,
                failed_key,
                "checkpoint_failure_with_preserved_feedback",
                (failed_attempt["row_order"],) if failed_attempt else (),
            ),
            "full_write_replace_read_continue_success": chain_case(
                successful_full_chain,
                "full_write_replace_read_continue_success",
            ),
            "full_write_replace_read_continue_failure": chain_case(
                failed_full_chain,
                "full_write_replace_read_continue_failure",
            ),
        },
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--through", type=int)
    parser.add_argument("--route", default="swesmith")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.through, args.route)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(output))
    keys = (
        "through_update",
        "trajectory_count",
        "task_success_count",
        "compaction_action_attempt_count",
        "compaction_opportunity_count",
        "failed_checkpoint_action_attempt_count",
        "successful_replacement_count",
        "invalid_replacement_count",
        "feedback_preserving_failed_attempt_count",
        "retry_exhausted_failed_attempt_count",
        "invalid_retry_transition_count",
        "unresolved_compaction_opportunity_count",
        "strict_write_compaction_read_chain_count",
        "attested_write_compaction_read_chain_count",
        "broader_nonexact_read_chain_count",
        "path_mentioned_without_attested_read_count",
        "behavioral_continuation_chain_count",
        "attested_behavioral_continuation_chain_count",
        "strict_chain_task_success_count",
        "attested_chain_task_success_count",
    )
    print(json.dumps({key: result[key] for key in keys}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
