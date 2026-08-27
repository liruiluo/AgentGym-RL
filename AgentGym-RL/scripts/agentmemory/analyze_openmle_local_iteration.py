#!/usr/bin/env python3
"""Audit OpenMLE-fast local validation and filesystem experiment-memory chains."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

STEP_RE = re.compile(r"ppo_batch_step(\d+)_post_adv\.json$")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
VALIDATION_CONTEXT_RE = re.compile(
    r"(?i)(?:\b(?:validation|valid|holdout|cross[-_ ]?validation|cv|fold)\b|"
    r"\bval(?:idation)?[_ :\-])"
)
METRIC_RE = re.compile(
    r"(?i)\b(?:accuracy|auc|auroc|f1|rmse|rmsle|mae|mse|log[-_ ]?loss|"
    r"map|ndcg|pearson|spearman)\b"
)
EXPLICIT_VALIDATION_RE = re.compile(
    r"(?im)^\s*validation_([A-Za-z][A-Za-z0-9_.-]{0,63})\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
DOCUMENT_SUFFIXES = frozenset({".md", ".txt", ".rst", ".log", ".yaml", ".yml", ".toml"})
CODE_SUFFIXES = frozenset({".py", ".ipynb", ".sh", ".r", ".jl"})
IGNORED_DOCUMENTS = frozenset({"TASK.md"})
SUBMISSION_PATH = "submission.csv"
CONTINUATION_PATH = ".agent_memory/CONTINUATION.md"
LEGACY_CONTINUATION_PATHS = frozenset(
    {".agent_memory/OPENMLE_CONTINUATION.md"}
)
KNOWN_CONTINUATION_PATHS = frozenset(
    {CONTINUATION_PATH, *LEGACY_CONTINUATION_PATHS}
)
CHECKPOINT_RECEIPT_SCHEMA = "agentmemory_filesystem_checkpoint_receipt_v1"
CHECKPOINT_READ_RECEIPT_SCHEMA = (
    "agentmemory_filesystem_checkpoint_read_receipt_v1"
)
CHECKPOINT_MAX_BYTES = 8 * 1024
CHECKPOINT_MARKER_PREFIX = (
    "Earlier conversation was removed after the continuation snapshot write "
    "succeeded. The workspace persists, but "
    "`.agent_memory/CONTINUATION.md` was not copied into this prompt. Use the "
    "next normal action to read that file, then continue from its evidence and "
    "next action. Other workspace files remain available and may still be read "
    "or updated normally."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--require-chain", action="store_true")
    return parser.parse_args()


def _step_paths(directory: Path, max_step: int | None) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("ppo_batch_step*_post_adv.json"):
        match = STEP_RE.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if max_step is None or step <= max_step:
            candidates.append((step, path))
    candidates.sort()
    if not candidates:
        raise ValueError("no OpenMLE PPO diagnostic dumps found")
    steps = {step for step, _ in candidates}
    contiguous = 0
    while contiguous + 1 in steps:
        contiguous += 1
    if contiguous == 0:
        raise ValueError("diagnostic dumps do not start at step 1")
    return [(step, path) for step, path in candidates if step <= contiguous]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _document_paths(changed_paths: Iterable[Any]) -> tuple[str, ...]:
    paths: list[str] = []
    for raw in changed_paths:
        if not isinstance(raw, str) or not raw or raw in IGNORED_DOCUMENTS:
            continue
        path = Path(raw)
        if raw in KNOWN_CONTINUATION_PATHS or path.suffix.lower() in DOCUMENT_SUFFIXES:
            paths.append(raw)
    return tuple(sorted(set(paths)))


def _code_paths(changed_paths: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                raw
                for raw in changed_paths
                if isinstance(raw, str)
                and raw
                and Path(raw).suffix.lower() in CODE_SUFFIXES
            }
        )
    )


def _is_completed_shell(row: Mapping[str, Any]) -> bool:
    info = _mapping(row.get("env_info_after"))
    execution = _mapping(info.get("execution"))
    return (
        info.get("action_kind") == "shell_command"
        and info.get("action_status") == "completed"
        and execution.get("status") == "completed"
        and execution.get("exit_code") == 0
    )


def _execution_attempt_delta(row: Mapping[str, Any]) -> int:
    info = _mapping(row.get("env_info_after"))
    execution = _mapping(info.get("execution"))
    counter_delta = _mapping(info.get("counter_delta"))
    value = counter_delta.get(
        "execution_attempt_count", execution.get("execution_attempt_delta")
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _is_completed_managed_execution(row: Mapping[str, Any]) -> bool:
    return _is_completed_shell(row) and _execution_attempt_delta(row) >= 1


def _validation_excerpts(stdout: str) -> list[str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    excerpts = [
        match.group(0).strip() for match in EXPLICIT_VALIDATION_RE.finditer(stdout)
    ]
    for index in range(len(lines)):
        for width in (1, 2):
            excerpt = "\n".join(lines[index : index + width])
            if not excerpt or excerpt in excerpts:
                continue
            if (
                VALIDATION_CONTEXT_RE.search(excerpt)
                and METRIC_RE.search(excerpt)
                and NUMBER_RE.search(excerpt)
            ):
                excerpts.append(excerpt)
    return excerpts


def _local_validation_evidence(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if not _is_completed_managed_execution(row) or row.get("terminal") is True:
        return None
    execution = _mapping(_mapping(row.get("env_info_after")).get("execution"))
    stdout = execution.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    excerpts = _validation_excerpts(stdout)
    if not excerpts:
        return None
    matched_text = "\n".join(excerpts[:5])
    return {
        "row_order": row.get("row_order"),
        "validation_terms": [
            match.group(0) for match in VALIDATION_CONTEXT_RE.finditer(matched_text)
        ][:8],
        "metric_terms": [match.group(0) for match in METRIC_RE.finditer(matched_text)][:8],
        "numbers": [match.group(0) for match in NUMBER_RE.finditer(matched_text)][:12],
        "matched_excerpts": excerpts[:5],
        "stdout_excerpt": stdout[:1000],
    }


def _normalize_row(step: int, row: Mapping[str, Any]) -> dict[str, Any] | None:
    record = row.get("formal_step_record")
    if not isinstance(record, Mapping):
        return None
    uid = record.get("trajectory_uid") or row.get("agentmemory_trajectory_uid")
    if not isinstance(uid, str) or not uid:
        return None
    info = _mapping(record.get("env_info_after"))
    execution = _mapping(info.get("execution"))
    changed_paths = tuple(
        path for path in _sequence(execution.get("changed_paths")) if isinstance(path, str)
    )
    action = record.get("action")
    if not isinstance(action, str):
        action = ""
    return {
        "step": step,
        "trajectory_uid": uid,
        "row_order": record.get("trajectory_row_order"),
        "item_id": record.get("item_id"),
        "action": action,
        "action_submission": record.get("action_submission"),
        "terminal": bool(record.get("trajectory_terminal")),
        "trajectory_return": record.get("trajectory_return"),
        "immediate_reward": record.get("immediate_reward"),
        "wrapper_evidence": dict(_mapping(record.get("wrapper_evidence"))),
        "context_transition": record.get("context_transition"),
        "control_request": record.get("control_request"),
        "native_step_before": record.get("native_step_before"),
        "native_step_after": record.get("native_step_after"),
        "native_call_count_before": record.get("native_call_count_before"),
        "native_call_count_after": record.get("native_call_count_after"),
        "policy_step_before": record.get("policy_step_before"),
        "policy_step_after": record.get("policy_step_after"),
        "context_epoch_before": record.get("context_epoch_before"),
        "context_epoch_after": record.get("context_epoch_after"),
        "env_info_after": dict(info),
        "changed_paths": changed_paths,
        "document_paths": _document_paths(changed_paths),
        "code_paths": _code_paths(changed_paths),
    }


def _first_after(rows: list[dict[str, Any]], order: int, predicate) -> dict[str, Any] | None:
    for row in rows:
        row_order = row.get("row_order")
        if isinstance(row_order, int) and row_order > order and predicate(row):
            return row
    return None


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _unit_counter_increment(row: Mapping[str, Any], prefix: str) -> bool:
    before = row.get(f"{prefix}_before")
    after = row.get(f"{prefix}_after")
    return bool(
        isinstance(before, int)
        and not isinstance(before, bool)
        and isinstance(after, int)
        and not isinstance(after, bool)
        and after == before + 1
    )


def _canonical_checkpoint_receipt(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    size = value.get("size_bytes")
    if (
        set(value)
        != {
            "schema",
            "path",
            "action_kind",
            "action_completed",
            "changed",
            "exists",
            "regular_file",
            "size_bytes",
            "sha256",
        }
        or value.get("schema") != CHECKPOINT_RECEIPT_SCHEMA
        or value.get("path") != CONTINUATION_PATH
        or value.get("action_kind") not in {"shell_command", "apply_patch"}
        or value.get("action_completed") is not True
        or value.get("changed") is not True
        or value.get("exists") is not True
        or value.get("regular_file") is not True
        or isinstance(size, bool)
        or not isinstance(size, int)
        or not 0 < size <= CHECKPOINT_MAX_BYTES
        or not _valid_sha256(value.get("sha256"))
    ):
        return None
    return value


def _canonical_read_receipt(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    size = value.get("size_bytes")
    if (
        set(value) != {"schema", "path", "observed", "size_bytes", "sha256"}
        or value.get("schema") != CHECKPOINT_READ_RECEIPT_SCHEMA
        or value.get("path") != CONTINUATION_PATH
        or value.get("observed") is not True
        or isinstance(size, bool)
        or not isinstance(size, int)
        or not 0 < size <= CHECKPOINT_MAX_BYTES
        or not _valid_sha256(value.get("sha256"))
    ):
        return None
    return value


def _safe_checkpoint_successor(
    row: Mapping[str, Any], receipt: Mapping[str, Any]
) -> bool:
    transition = row.get("context_transition")
    messages = transition.get("messages") if isinstance(transition, Mapping) else None
    if (
        not isinstance(transition, Mapping)
        or transition.get("schema")
        != "agentmemory_task_neutral_context_transition_v1"
        or transition.get("operation") != "replace_messages"
        or not isinstance(messages, Sequence)
        or isinstance(messages, (str, bytes))
        or not messages
    ):
        return False
    normalized: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            return False
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            return False
        normalized.append((role, content))
    marker = (
        f"{CHECKPOINT_MARKER_PREFIX} Verified receipt: "
        f"size_bytes={receipt['size_bytes']}, sha256={receipt['sha256']}."
    )
    if normalized[-1][0] != "user" or not normalized[-1][1].endswith(marker):
        return False
    successor = "\n".join(content for _role, content in normalized)
    return bool(
        row.get("action") not in successor
        and row.get("control_request") not in successor
    )


def _canonical_compaction_receipt(
    row: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    evidence = _mapping(row.get("wrapper_evidence"))
    receipt = _canonical_checkpoint_receipt(evidence.get("checkpoint_receipt"))
    endpoint = _canonical_checkpoint_receipt(
        _mapping(_mapping(row.get("env_info_after")).get("execution")).get(
            "filesystem_checkpoint"
        )
    )
    submission = _mapping(row.get("action_submission"))
    if receipt is None or endpoint is None or dict(receipt) != dict(endpoint):
        return None
    if not (
        evidence.get("event") == "context_compaction"
        and evidence.get("continuation_path") == CONTINUATION_PATH
        and evidence.get("continuation_persisted") is True
        and evidence.get("checkpoint_failure_reason") is None
        and evidence.get("context_replaced") is True
        and evidence.get("retry_pending") is False
        and evidence.get("preserved_policy_output") is True
        and evidence.get("preserved_native_observation") is True
        and evidence.get("checkpoint_action_in_successor_context") is False
        and evidence.get("checkpoint_observation_in_successor_context") is False
        and evidence.get("checkpoint_content_in_successor_context") is False
        and isinstance(row.get("action"), str)
        and CONTINUATION_PATH in row["action"]
        and submission.get("raw_policy_output") == row["action"]
        and isinstance(row.get("control_request"), str)
        and bool(row["control_request"].strip())
        and _unit_counter_increment(row, "native_step")
        and _unit_counter_increment(row, "native_call_count")
        and _unit_counter_increment(row, "policy_step")
        and _unit_counter_increment(row, "context_epoch")
        and _safe_checkpoint_successor(row, receipt)
    ):
        return None
    return receipt


def _matching_checkpoint_read(
    row: Mapping[str, Any], receipt: Mapping[str, Any]
) -> bool:
    evidence = _mapping(row.get("wrapper_evidence"))
    read = _canonical_read_receipt(evidence.get("filesystem_checkpoint_read"))
    endpoint = _canonical_read_receipt(
        _mapping(_mapping(row.get("env_info_after")).get("execution")).get(
            "filesystem_checkpoint_read"
        )
    )
    context_before = row.get("context_epoch_before")
    context_after = row.get("context_epoch_after")
    return bool(
        read is not None
        and endpoint is not None
        and dict(read) == dict(endpoint)
        and read.get("size_bytes") == receipt.get("size_bytes")
        and read.get("sha256") == receipt.get("sha256")
        and evidence.get("memory_event") == "read"
        and evidence.get("document_read_observed") is True
        and _is_completed_shell(row)
        and _unit_counter_increment(row, "native_step")
        and _unit_counter_increment(row, "native_call_count")
        and _unit_counter_increment(row, "policy_step")
        and isinstance(context_before, int)
        and not isinstance(context_before, bool)
        and context_after == context_before
    )


def _canonical_checkpoint_sequences(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    sequences: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for compaction in rows:
        receipt = _canonical_compaction_receipt(compaction)
        order = compaction.get("row_order")
        if receipt is None or not isinstance(order, int):
            continue
        read = _first_after(
            rows,
            order,
            lambda row, expected=receipt: _matching_checkpoint_read(row, expected),
        )
        if read is not None:
            sequences.append((compaction, read))
    return sequences


def _post_compaction_read(
    rows: list[dict[str, Any]],
    compactions: list[dict[str, Any]],
    write: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    write_order = write.get("row_order")
    if not isinstance(write_order, int):
        return None
    paths = frozenset(write["document_paths"])
    compaction = next(
        (
            row
            for row in compactions
            if isinstance(row.get("row_order"), int)
            and row["row_order"] >= write_order
        ),
        None,
    )
    if compaction is None:
        return None
    read = _first_after(
        rows,
        compaction["row_order"],
        lambda row, expected_paths=paths: _is_completed_shell(row)
        and any(path in row["action"] for path in expected_paths)
        and bool(
            str(
                _mapping(_mapping(row["env_info_after"]).get("execution")).get(
                    "stdout", ""
                )
            ).strip()
        ),
    )
    return (compaction, read) if read is not None else None


def _trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows.sort(
        key=lambda row: row["row_order"]
        if isinstance(row.get("row_order"), int)
        else 10**9
    )
    local = [evidence for row in rows if (evidence := _local_validation_evidence(row))]
    doc_writes = [row for row in rows if row["document_paths"]]
    compactions = [
        row for row in rows if row["wrapper_evidence"].get("event") == "context_compaction"
    ]
    terminal_submit = next(
        (
            row
            for row in reversed(rows)
            if row["terminal"]
            and _mapping(row["env_info_after"]).get("action_kind") == "submit"
        ),
        None,
    )
    read_sequences = [
        (write, result[0], result[1])
        for write in doc_writes
        if (result := _post_compaction_read(rows, compactions, write)) is not None
    ]
    canonical_checkpoint_sequences = _canonical_checkpoint_sequences(rows)
    continuation_writes = [
        row
        for row in doc_writes
        if CONTINUATION_PATH in row["document_paths"]
    ]
    legacy_continuation_writes = [
        row
        for row in doc_writes
        if any(
            path in LEGACY_CONTINUATION_PATHS
            for path in row["document_paths"]
        )
    ]
    continuation_read_sequences = [
        (compaction, compaction, read)
        for compaction, read in canonical_checkpoint_sequences
    ]
    legacy_continuation_read_sequences = [
        sequence
        for sequence in read_sequences
        if any(
            path in LEGACY_CONTINUATION_PATHS
            for path in sequence[0]["document_paths"]
        )
    ]

    chain: dict[str, Any] | None = None
    for validation in local:
        validation_order = validation.get("row_order")
        if not isinstance(validation_order, int):
            continue
        for write, compaction, read in continuation_read_sequences:
            write_order = write.get("row_order")
            if not isinstance(write_order, int) or write_order <= validation_order:
                continue
            read_order = read["row_order"]
            edit = _first_after(rows, read_order, lambda row: bool(row["code_paths"]))
            if edit is None:
                continue
            rerun = _first_after(rows, edit["row_order"], _is_completed_managed_execution)
            if rerun is None or terminal_submit is None:
                continue
            submit_order = terminal_submit.get("row_order")
            if not isinstance(submit_order, int) or submit_order <= rerun["row_order"]:
                continue
            chain = {
                "validation_order": validation_order,
                "document_write_order": write_order,
                "document_paths": list(write["document_paths"]),
                "compaction_order": compaction["row_order"],
                "document_read_order": read_order,
                "code_edit_order": edit["row_order"],
                "code_paths": list(edit["code_paths"]),
                "rerun_order": rerun["row_order"],
                "submit_order": submit_order,
            }
            break
        if chain is not None:
            break

    terminal = next((row for row in reversed(rows) if row["terminal"]), rows[-1])
    terminal_info = _mapping(terminal["env_info_after"])
    grade = _mapping(terminal_info.get("grade"))
    counters = _mapping(terminal_info.get("counters"))
    return {
        "step": rows[0]["step"],
        "trajectory_uid": rows[0]["trajectory_uid"],
        "item_id": rows[0].get("item_id"),
        "action_rows": len(rows),
        "local_validation_rows": len(local),
        "local_validation_evidence": local[:5],
        "document_write_rows": len(doc_writes),
        "continuation_write_rows": len(continuation_writes),
        "legacy_continuation_write_rows": len(legacy_continuation_writes),
        "document_paths": sorted(
            {path for row in doc_writes for path in row["document_paths"]}
        ),
        "compaction_rows": len(compactions),
        "post_compaction_document_read": bool(read_sequences),
        "post_compaction_continuation_read": bool(continuation_read_sequences),
        "post_compaction_legacy_continuation_read": bool(
            legacy_continuation_read_sequences
        ),
        "terminal_submit": terminal_submit is not None,
        "terminal_reason": terminal_info.get("terminal_reason"),
        "submission_valid": grade.get("submission_valid"),
        "baseline_beat": terminal_info.get("episode_success"),
        "normalized_reward": grade.get("normalized_reward"),
        "trajectory_return": terminal.get("trajectory_return"),
        "execution_attempt_count": counters.get("execution_attempt_count"),
        "execution_completed_count": counters.get("execution_completed_count"),
        "fit_count": counters.get("fit_count"),
        "complete_iteration_memory_chain": chain is not None,
        "chain": chain,
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite_number(value)) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate_trajectories(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(trajectories)
    valid = [row for row in trajectories if row.get("submission_valid") is True]
    valid_count = len(valid)
    baseline_beats = sum(row.get("baseline_beat") is True for row in trajectories)
    valid_baseline_beats = sum(row.get("baseline_beat") is True for row in valid)
    terminal_submits = sum(row.get("terminal_submit") is True for row in trajectories)
    online_ps = _rate(baseline_beats, total)
    return {
        "trajectory_count": total,
        "valid_submission_count": valid_count,
        "terminal_submit_count": terminal_submits,
        "vsr": _rate(valid_count, total),
        "terminal_submit_rate": _rate(terminal_submits, total),
        "ps": online_ps,
        "bbr_all": online_ps,
        "bbr_valid": _rate(valid_baseline_beats, valid_count),
        "mean_normalized_reward_valid": _mean(
            row.get("normalized_reward") for row in valid
        ),
        "mean_trajectory_return": _mean(
            row.get("trajectory_return") for row in trajectories
        ),
        "local_validation_rate": _rate(
            sum(int(row.get("local_validation_rows", 0) > 0) for row in trajectories),
            total,
        ),
        "document_write_rate": _rate(
            sum(int(row.get("document_write_rows", 0) > 0) for row in trajectories),
            total,
        ),
        "continuation_write_rate": _rate(
            sum(int(row.get("continuation_write_rows", 0) > 0) for row in trajectories),
            total,
        ),
        "post_compaction_document_read_rate": _rate(
            sum(row.get("post_compaction_document_read") is True for row in trajectories),
            total,
        ),
        "post_compaction_continuation_read_rate": _rate(
            sum(row.get("post_compaction_continuation_read") is True for row in trajectories),
            total,
        ),
        "complete_iteration_memory_chain_rate": _rate(
            sum(row.get("complete_iteration_memory_chain") is True for row in trajectories),
            total,
        ),
        "mean_execution_attempt_count": _mean(
            row.get("execution_attempt_count") for row in trajectories
        ),
        "mean_execution_completed_count": _mean(
            row.get("execution_completed_count") for row in trajectories
        ),
        "mean_fit_count": _mean(row.get("fit_count") for row in trajectories),
    }


def analyze_documents(
    documents: Iterable[tuple[int, Mapping[str, Any]]],
    *,
    block_size: int = 10,
) -> dict[str, Any]:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    steps: set[int] = set()
    row_count = 0
    for step, document in documents:
        steps.add(step)
        for raw in _sequence(document.get("rows")):
            if not isinstance(raw, Mapping):
                continue
            row = _normalize_row(step, raw)
            if row is None:
                continue
            grouped[(step, row["trajectory_uid"])].append(row)
            row_count += 1
    trajectories = [_trajectory_summary(rows) for _, rows in sorted(grouped.items())]
    counts = Counter()
    for trajectory in trajectories:
        for key in (
            "terminal_submit",
            "post_compaction_document_read",
            "post_compaction_continuation_read",
            "post_compaction_legacy_continuation_read",
            "complete_iteration_memory_chain",
        ):
            counts[key] += int(trajectory[key])
        counts["has_local_validation"] += int(trajectory["local_validation_rows"] > 0)
        counts["has_document_write"] += int(trajectory["document_write_rows"] > 0)
        counts["has_continuation_write"] += int(
            trajectory["continuation_write_rows"] > 0
        )
        counts["has_legacy_continuation_write"] += int(
            trajectory["legacy_continuation_write_rows"] > 0
        )
        counts["has_compaction"] += int(trajectory["compaction_rows"] > 0)

    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        step = trajectory["step"]
        by_step[step].append(trajectory)
        by_block[(step - 1) // block_size].append(trajectory)
    step_summaries = [
        {"step": step, **_aggregate_trajectories(rows)}
        for step, rows in sorted(by_step.items())
    ]
    block_summaries = [
        {
            "step_start": block * block_size + 1,
            "step_end": (block + 1) * block_size,
            "observed_steps": sorted({row["step"] for row in rows}),
            **_aggregate_trajectories(rows),
        }
        for block, rows in sorted(by_block.items())
    ]
    return {
        "schema": "openmle_local_iteration_memory_audit_v1",
        "steps": sorted(steps),
        "block_size": block_size,
        "trajectory_count": len(trajectories),
        "action_row_count": row_count,
        "counts": dict(sorted(counts.items())),
        "aggregate": _aggregate_trajectories(trajectories),
        "step_summaries": step_summaries,
        "block_summaries": block_summaries,
        "complete_chain_cases": [
            trajectory
            for trajectory in trajectories
            if trajectory["complete_iteration_memory_chain"]
        ],
        "trajectories": trajectories,
    }


def main() -> int:
    args = parse_args()
    paths = _step_paths(args.diagnostics_dir, args.max_step)
    documents = [(step, json.loads(path.read_text(encoding="utf-8"))) for step, path in paths]
    result = analyze_documents(documents, block_size=args.block_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "counts": result["counts"]}, sort_keys=True))
    if args.require_chain and result["counts"].get(
        "complete_iteration_memory_chain", 0
    ) < 1:
        raise SystemExit("no complete OpenMLE local-iteration memory chain found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
