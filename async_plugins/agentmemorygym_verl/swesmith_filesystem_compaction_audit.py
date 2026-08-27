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


def parse_shell_words(record):
    submission = record.get("action_submission") or {}
    raw = submission.get("raw_policy_output")
    if not isinstance(raw, str) or raw != record.get("action"):
        return None
    prefix = "shell_command "
    if not raw.startswith(prefix):
        return None
    try:
        payload = json.loads(raw[len(prefix) :])
        if not isinstance(payload, dict) or not isinstance(payload.get("command"), str):
            return None
        return shlex.split(payload["command"])
    except (json.JSONDecodeError, TypeError, ValueError):
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


def checkpoint_read(record, checkpoint):
    path, digest, size = checkpoint
    info = record.get("env_info_after") or {}
    evidence = record.get("wrapper_evidence") or {}
    transition = record.get("context_transition") or {}
    observed = valid_receipt(info.get("filesystem_checkpoint"), require_changed=False)
    words = parse_shell_words(record)
    return (
        observed == (path, digest, size)
        and isinstance(evidence, dict)
        and evidence.get("event") == "native_action"
        and actor_credit(record).get("positive_eligible") is True
        and isinstance(transition, dict)
        and transition.get("operation") == "append_observation"
        and words in (["cat", path], ["cat", "--", path])
    )


def successful_task_action(record, checkpoint_path):
    if not workspace_action(record) or not action_completed(record):
        return False
    evidence = record.get("wrapper_evidence") or {}
    transition = record.get("context_transition") or {}
    words = parse_shell_words(record)
    return (
        isinstance(evidence, dict)
        and evidence.get("event") == "native_action"
        and actor_credit(record).get("positive_eligible") is True
        and isinstance(transition, dict)
        and transition.get("operation") == "append_observation"
        and words not in (["cat", checkpoint_path], ["cat", "--", checkpoint_path])
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
        "receipt": receipt if isinstance(receipt, dict) else None,
        "valid_write_and_replacement": checkpoint_write(record) is not None,
        "response_token_count": record.get("response_token_count"),
        "actor_credit_basis": actor_credit(record).get("basis"),
        "trajectory_terminal_at_attempt": bool(record.get("trajectory_terminal")),
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
    strict_chains = []
    behavioral_chains = []
    proactive_write_trajectories = set()
    read_trajectories = set()
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

                read_index = None
                read_record = None
                for candidate_index in range(index + 1, len(rows)):
                    if checkpoint_read(rows[candidate_index], identity):
                        read_index = candidate_index
                        read_record = rows[candidate_index]
                        break
                if read_record is None:
                    continue
                read_trajectories.add((update, uid))
                chain = {
                    **replacement,
                    "read_row_order": read_record.get("trajectory_row_order"),
                    "write_response_token_count": record.get("response_token_count"),
                    "read_response_token_count": read_record.get("response_token_count"),
                    "trajectory_return": rows[-1].get("trajectory_return"),
                    "terminal_outcome": rows[-1].get("outcome"),
                }
                strict_chains.append(chain)

                task_record = next(
                    (
                        candidate
                        for candidate in rows[read_index + 1 :]
                        if successful_task_action(candidate, path)
                    ),
                    None,
                )
                if task_record is not None:
                    behavioral_chains.append(
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
    behavioral_chain_trajectories = {
        (chain["update"], chain["trajectory_uid"]) for chain in behavioral_chains
    }
    output = {
        "schema": "amg_swesmith_filesystem_compaction_audit_v2",
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
        "unresolved_compaction_opportunity_count": len(unresolved),
        "strict_write_compaction_read_chain_count": len(strict_chains),
        "strict_chain_trajectory_count": len(strict_chain_trajectories),
        "behavioral_continuation_chain_count": len(behavioral_chains),
        "behavioral_continuation_trajectory_count": len(
            behavioral_chain_trajectories
        ),
        "strict_chain_task_success_count": sum(
            (chain["update"], chain["trajectory_uid"]) in successful_trajectories
            for chain in strict_chains
        ),
        "proactive_checkpoint_write_trajectory_count": len(
            proactive_write_trajectories
        ),
        "post_compaction_read_trajectory_count": len(read_trajectories),
        # Compatibility aliases for existing monitors, with v2 receipt semantics.
        "compaction_event_count": len(attempts),
        "valid_executed_checkpoint_write_count": len(successful_replacements),
        "transition_without_valid_write_count": len(invalid_replacements),
        "forced_checkpoint_events": attempts,
        "compaction_opportunities": opportunities,
        "successful_replacements": successful_replacements,
        "invalid_replacements": invalid_replacements,
        "unresolved_compaction_opportunities": unresolved,
        "strict_chains": strict_chains,
        "behavioral_continuation_chains": behavioral_chains,
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
        "unresolved_compaction_opportunity_count",
        "strict_write_compaction_read_chain_count",
        "behavioral_continuation_chain_count",
        "strict_chain_task_success_count",
    )
    print(json.dumps({key: result[key] for key in keys}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
