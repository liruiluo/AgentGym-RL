#!/usr/bin/env python3
"""Attest one complete OpenMLE-fast PPO update from frozen evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


MANIFEST_SCHEMA = "openmle_fast_public_manifest_v1"
UPDATE_SCHEMA = "openmle_fast_ppo_update_evidence_v1"
STEP_SCHEMA = "task_neutral_policy_step_v1"
ENDPOINT_SCHEMA = "openmle_fast_resident_endpoint_probe_v1"
METADATA_SCHEMA = "openmle_fast_public_metadata_v1"
GRADE_SCHEMA = "openmle_fast_grade_response_v1"
DOMAIN_ID = "openmle_fast"
CONTRACT_VERSION = "openmle_fast_v1"
EXPECTED_OPENMLE_TASKS_REVISION = "f56e4b31252a9b81d95fea100098cd49b7290398"
EXPECTED_TASK_COUNT = 64
EXPECTED_PANEL_ID = "openmle-fast-g64-v1"
EXPECTED_ROLE = "gate_only"
COUNTER_KEYS = (
    "action_count",
    "execution_action_count",
    "execution_attempt_count",
    "execution_completed_count",
    "nested_subprocess_count",
    "fit_count",
    "grading_count",
)
FORBIDDEN_PUBLIC_KEYS = (
    "credential",
    "detail_token",
    "grader_socket",
    "private_manifest",
    "private_path",
    "private_root",
    "secret",
    "traceback",
)
POLICY_TERMINAL_REASONS = frozenset(
    {
        "action_budget_exhausted",
        "managed_runtime_limit",
        "episode_wall_limit",
    }
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--endpoint-probe", type=Path, required=True)
    parser.add_argument("--expected-outer-commit", required=True)
    parser.add_argument("--expected-inner-commit", required=True)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--forbidden-canaries-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_forbidden_canaries(path: Path) -> list[str]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("forbidden canaries must come from a real file")
    if path.stat().st_mode & 0o077:
        raise ValueError("forbidden canaries file must have mode 0600 or stricter")
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [line for line in raw.splitlines() if line]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError("forbidden canaries file must contain nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError("forbidden canaries must be distinct")
    return value


def canonical_sha256(value: Any, *, newline: bool = False) -> str:
    suffix = "\n" if newline else ""
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + suffix).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, bool):
        matches = value is expected
    else:
        matches = value == expected
    if not matches:
        raise AssertionError(f"{label} mismatch: {value!r} != {expected!r}")


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AssertionError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise AssertionError(f"{label} must be finite")
    return normalized


def _sha256_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise AssertionError(f"{label} must be a lowercase SHA-256 string")
    return value


def _identity_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{label} must be a nonempty string")
    return value


def _boolean(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise AssertionError(f"{label} must be boolean")
    return value


def require_public_safe(
    value: Any,
    *,
    label: str,
    forbidden_canaries: list[str],
) -> None:
    def walk(current: Any, path: str) -> None:
        if isinstance(current, dict):
            for key, child in current.items():
                key_text = str(key)
                lowered = key_text.lower()
                if any(fragment in lowered for fragment in FORBIDDEN_PUBLIC_KEYS):
                    raise AssertionError(
                        f"{label} is not public-safe: forbidden key {path}.{key_text}"
                    )
                walk(child, f"{path}.{key_text}")
            return
        if isinstance(current, list):
            for index, child in enumerate(current):
                walk(child, f"{path}[{index}]")
            return
        if isinstance(current, str):
            for canary in forbidden_canaries:
                if canary and canary in current:
                    raise AssertionError(
                        f"{label} is not public-safe: canary at {path}"
                    )

    walk(value, "$")


def validate_manifest(
    document: dict[str, Any],
    manifest_sha256: str,
    manifest_bytes: bytes | None = None,
) -> dict[int, dict[str, Any]]:
    _require_exact(document.get("schema"), MANIFEST_SCHEMA, label="manifest schema")
    _require_exact(
        (
            canonical_sha256(document, newline=True)
            if manifest_bytes is None
            else hashlib.sha256(manifest_bytes).hexdigest()
        ),
        manifest_sha256,
        label="manifest SHA-256",
    )
    _require_exact(
        document.get("openmle_tasks_revision"),
        EXPECTED_OPENMLE_TASKS_REVISION,
        label="manifest OpenMLE revision",
    )
    _require_exact(document.get("panel_id"), EXPECTED_PANEL_ID, label="gate panel_id")
    _require_exact(document.get("role"), EXPECTED_ROLE, label="gate role")
    records = document.get("records")
    if not isinstance(records, list):
        raise AssertionError("manifest records are missing")
    _require_exact(document.get("task_count"), len(records), label="task_count")
    _require_exact(len(records), EXPECTED_TASK_COUNT, label="gate task_count")

    records_by_index: dict[int, dict[str, Any]] = {}
    task_ids: set[str] = set()
    source_families: set[str] = set()
    for expected_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise AssertionError(f"manifest record {expected_index} is not an object")
        data_idx = record.get("data_idx")
        if isinstance(data_idx, bool) or data_idx != expected_index:
            raise AssertionError(
                f"manifest data_idx {data_idx!r} is not contiguous at {expected_index}"
            )
        task_id = record.get("task_id")
        source_family = record.get("source_family")
        if not isinstance(task_id, str) or not task_id:
            raise AssertionError(f"manifest record {expected_index} lacks task_id")
        if not isinstance(source_family, str) or not source_family:
            raise AssertionError(
                f"manifest record {expected_index} lacks source_family"
            )
        _require_exact(
            record.get("role"),
            EXPECTED_ROLE,
            label=f"manifest record {expected_index} role",
        )
        if task_id in task_ids:
            raise AssertionError(f"duplicate manifest task_id: {task_id}")
        if source_family in source_families:
            raise AssertionError(f"duplicate manifest source_family: {source_family}")
        task_ids.add(task_id)
        source_families.add(source_family)
        records_by_index[expected_index] = record
    return records_by_index


def validate_endpoint_metadata(
    metadata: Any,
    document: dict[str, Any],
    manifest_sha256: str,
    *,
    expected_outer_commit: str,
    expected_inner_commit: str,
    expected_prompt_sha256: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise AssertionError(f"{label} metadata is missing")
    expected = {
        "schema": METADATA_SCHEMA,
        "domain_id": DOMAIN_ID,
        "contract_version": CONTRACT_VERSION,
        "panel_id": document["panel_id"],
        "role": document["role"],
        "task_count": document["task_count"],
        "task_manifest_sha256": manifest_sha256,
        "openmle_tasks_revision": document["openmle_tasks_revision"],
        "task_id_list_sha256": document["task_id_list_sha256"],
        "compact_panel_sha256": document["compact_panel_sha256"],
        "policy_prompt_sha256": expected_prompt_sha256,
    }
    for key, value in expected.items():
        _require_exact(metadata.get(key), value, label=f"{label} {key}")
    runtime_source = metadata.get("runtime_source")
    if not isinstance(runtime_source, dict):
        raise AssertionError(f"{label} runtime_source is missing")
    validate_runtime_identity(
        runtime_source,
        expected_outer_commit=expected_outer_commit,
        expected_inner_commit=expected_inner_commit,
        label=label,
    )
    return metadata


def validate_endpoint_probe(
    endpoint_probe: dict[str, Any],
    document: dict[str, Any],
    manifest_sha256: str,
    *,
    expected_outer_commit: str,
    expected_inner_commit: str,
    expected_prompt_sha256: str,
) -> None:
    _require_exact(
        endpoint_probe.get("schema"),
        ENDPOINT_SCHEMA,
        label="endpoint probe schema",
    )
    _require_exact(endpoint_probe.get("status"), "pass", label="endpoint status")
    _require_exact(
        endpoint_probe.get("manifest_sha256"),
        manifest_sha256,
        label="endpoint manifest SHA-256",
    )
    _require_exact(
        endpoint_probe.get("idempotent_close_verified"),
        True,
        label="endpoint idempotent close",
    )
    probe_indices = endpoint_probe.get("probe_indices")
    if (
        not isinstance(probe_indices, list)
        or len(probe_indices) != 2
        or len(set(probe_indices)) != 2
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= document["task_count"]
            for index in probe_indices
        )
    ):
        raise AssertionError("endpoint probe indices are invalid")
    _require_exact(endpoint_probe.get("reset_count"), 4, label="endpoint reset count")
    _require_exact(
        endpoint_probe.get("slot_cleanup_count"),
        3,
        label="endpoint slot cleanup count",
    )
    metadata_before = validate_endpoint_metadata(
        endpoint_probe.get("metadata_before"),
        document,
        manifest_sha256,
        expected_outer_commit=expected_outer_commit,
        expected_inner_commit=expected_inner_commit,
        expected_prompt_sha256=expected_prompt_sha256,
        label="endpoint metadata_before",
    )
    metadata_active = validate_endpoint_metadata(
        endpoint_probe.get("metadata_active"),
        document,
        manifest_sha256,
        expected_outer_commit=expected_outer_commit,
        expected_inner_commit=expected_inner_commit,
        expected_prompt_sha256=expected_prompt_sha256,
        label="endpoint metadata_active",
    )
    metadata_after = validate_endpoint_metadata(
        endpoint_probe.get("metadata_after"),
        document,
        manifest_sha256,
        expected_outer_commit=expected_outer_commit,
        expected_inner_commit=expected_inner_commit,
        expected_prompt_sha256=expected_prompt_sha256,
        label="endpoint metadata_after",
    )
    require_idle(metadata_before, label="endpoint probe before")
    for key in (
        "active_slot_count",
        "active_environment_count",
        "active_workspace_count",
    ):
        _require_exact(
            metadata_active.get(key),
            2,
            label=f"endpoint active {key}",
        )
    require_idle(metadata_after, label="endpoint probe")


def require_idle(metadata: dict[str, Any], *, label: str) -> None:
    for key in (
        "active_slot_count",
        "active_environment_count",
        "active_workspace_count",
    ):
        value = _integer(metadata.get(key), label=f"{label} {key}")
        if value != 0:
            raise AssertionError(f"{label} {key} is not zero: {value}")


def validate_runtime_identity(
    value: dict[str, Any],
    *,
    expected_outer_commit: str,
    expected_inner_commit: str,
    label: str,
) -> None:
    _require_exact(
        value.get("outer_commit"),
        expected_outer_commit,
        label=f"{label} outer_commit",
    )
    _require_exact(
        value.get("inner_commit"),
        expected_inner_commit,
        label=f"{label} inner_commit",
    )


def normalize_counters(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} counters are missing")
    counters = {
        key: _integer(value.get(key), label=f"{label} {key}") for key in COUNTER_KEYS
    }
    if counters["execution_action_count"] > counters["action_count"]:
        raise AssertionError(f"{label} execution actions exceed policy actions")
    if counters["execution_completed_count"] > counters["execution_attempt_count"]:
        raise AssertionError(f"{label} completed executions exceed execution attempts")
    if counters["grading_count"] > 1:
        raise AssertionError(f"{label} grading count exceeds one")
    return counters


def validate_token_row(row: dict[str, Any], *, label: str) -> int:
    _require_exact(
        row.get("generation_token_ids_are_exact"),
        True,
        label=f"{label} generation token identity",
    )
    _require_exact(
        row.get("backend_token_ids_are_exact"),
        True,
        label=f"{label} backend token identity",
    )
    sampled_tokens = row.get("sampled_response_token_ids")
    packed_tokens = row.get("packed_token_ids")
    response_mask = row.get("response_mask")
    sampled_logprobs = row.get("sampled_old_logprobs")
    packed_logprobs = row.get("packed_old_logprobs")
    if not isinstance(sampled_tokens, list) or not sampled_tokens:
        raise AssertionError(f"{label} sampled response tokens are empty")
    if not isinstance(packed_tokens, list) or not packed_tokens:
        raise AssertionError(f"{label} packed tokens are empty")
    if not isinstance(response_mask, list):
        raise AssertionError(f"{label} response mask is missing")
    if not isinstance(sampled_logprobs, list):
        raise AssertionError(f"{label} sampled logprobs are missing")
    if not isinstance(packed_logprobs, list):
        raise AssertionError(f"{label} packed logprobs are missing")
    if not (len(packed_tokens) == len(response_mask) == len(packed_logprobs)):
        raise AssertionError(f"{label} packed token evidence lengths differ")
    if len(sampled_tokens) != len(sampled_logprobs):
        raise AssertionError(f"{label} sampled token/logprob lengths differ")

    for index, token in enumerate(sampled_tokens):
        _integer(token, label=f"{label} sampled token {index}")
    for index, token in enumerate(packed_tokens):
        _integer(token, label=f"{label} packed token {index}")
    for index, mask in enumerate(response_mask):
        if isinstance(mask, bool) or mask not in (0, 1):
            raise AssertionError(f"{label} response mask {index} is not binary")
    normalized_sampled_logprobs = [
        _finite(value, label=f"{label} sampled logprob {index}")
        for index, value in enumerate(sampled_logprobs)
    ]
    normalized_packed_logprobs = [
        _finite(value, label=f"{label} packed logprob {index}")
        for index, value in enumerate(packed_logprobs)
    ]

    masked_tokens = [
        token for token, mask in zip(packed_tokens, response_mask) if mask == 1
    ]
    masked_logprobs = [
        value
        for value, mask in zip(normalized_packed_logprobs, response_mask)
        if mask == 1
    ]
    _require_exact(
        masked_tokens,
        sampled_tokens,
        label=f"{label} sampled/packed response token IDs",
    )
    _require_exact(
        masked_logprobs,
        normalized_sampled_logprobs,
        label=f"{label} sampled/packed logprobs",
    )

    digest_fields = (
        ("sampled_response_token_ids_sha256", sampled_tokens),
        ("packed_token_ids_sha256", packed_tokens),
        ("response_mask_sha256", response_mask),
        ("sampled_old_logprobs_sha256", sampled_logprobs),
    )
    for field, value in digest_fields:
        _require_exact(
            row.get(field),
            canonical_sha256(value),
            label=f"{label} {field}",
        )
    return len(sampled_tokens)


def validate_grade_receipt(
    receipt: Any,
    record: dict[str, Any],
    reward: float,
    *,
    action_submission: dict[str, Any],
    episode_id: str,
    terminal_reason: Any,
    terminal_classification: Any,
    runtime_success: Any,
    episode_success: Any,
    truncated: bool,
    label: str,
) -> None:
    if not isinstance(receipt, dict):
        raise AssertionError(f"{label} terminal grade receipt is missing")
    required_fields = {
        "schema",
        "contract_version",
        "request_id",
        "episode_id",
        "task_id",
        "submission_sha256",
        "submission_valid",
        "native_score",
        "higher_is_better",
        "normalized_reward",
        "improved_over_baseline",
        "runtime_success",
        "terminal_reason",
        "classification",
        "audit_digest",
    }
    if set(receipt) != required_fields:
        raise AssertionError(f"{label} grade receipt fields do not match the contract")
    _require_exact(
        receipt.get("schema"),
        GRADE_SCHEMA,
        label=f"{label} grade schema",
    )
    _require_exact(
        receipt.get("contract_version"),
        CONTRACT_VERSION,
        label=f"{label} grade contract version",
    )
    request_id = _identity_string(
        receipt.get("request_id"), label=f"{label} grade request_id"
    )
    _require_exact(
        receipt.get("episode_id"), episode_id, label=f"{label} grade episode_id"
    )
    _require_exact(
        receipt.get("task_id"),
        record["task_id"],
        label=f"{label} grade task_id",
    )
    submission_sha256 = _sha256_string(
        receipt.get("submission_sha256"),
        label=f"{label} grade submission SHA-256",
    )
    submission_valid = _boolean(
        receipt.get("submission_valid"),
        label=f"{label} submission_valid",
    )
    higher_is_better = _boolean(
        receipt.get("higher_is_better"),
        label=f"{label} higher_is_better",
    )
    improved = _boolean(
        receipt.get("improved_over_baseline"),
        label=f"{label} improved_over_baseline",
    )
    grade_runtime_success = _boolean(
        receipt.get("runtime_success"),
        label=f"{label} grade runtime_success",
    )
    del higher_is_better  # Direction is attested as Boolean but remains opaque here.
    classification = receipt.get("classification")
    reason = receipt.get("terminal_reason")
    native_score = receipt.get("native_score")
    normalized_reward = receipt.get("normalized_reward")
    if classification == "graded":
        _finite(native_score, label=f"{label} native_score")
        normalized = _finite(
            normalized_reward,
            label=f"{label} normalized_reward",
        )
        if not -1.0 <= normalized <= 1.0:
            raise AssertionError(f"{label} normalized reward is outside [-1, 1]")
        if not submission_valid or not grade_runtime_success:
            raise AssertionError(f"{label} graded receipt is internally inconsistent")
        _require_exact(reason, "graded_submission", label=f"{label} terminal reason")
    elif classification == "invalid_submission":
        _require_exact(native_score, None, label=f"{label} invalid native score")
        _require_exact(
            normalized_reward,
            -1.0,
            label=f"{label} invalid normalized reward",
        )
        if submission_valid or improved or grade_runtime_success:
            raise AssertionError(
                f"{label} invalid-submission receipt is internally inconsistent"
            )
        _require_exact(reason, "invalid_submission", label=f"{label} terminal reason")
        normalized = -1.0
    elif classification == "infrastructure_fault":
        _require_exact(native_score, None, label=f"{label} fault native score")
        _require_exact(
            normalized_reward,
            None,
            label=f"{label} fault normalized reward",
        )
        if submission_valid or improved or grade_runtime_success:
            raise AssertionError(
                f"{label} infrastructure-fault receipt is internally inconsistent"
            )
        _require_exact(
            reason,
            "grader_infrastructure_fault",
            label=f"{label} terminal reason",
        )
        raise AssertionError(
            f"{label} infrastructure fault cannot attest a complete PPO update"
        )
    else:
        raise AssertionError(f"{label} grade classification is invalid")
    if improved and not submission_valid:
        raise AssertionError(f"{label} invalid submission cannot improve over baseline")
    _require_exact(normalized, reward, label=f"{label} terminal reward")
    _require_exact(terminal_reason, reason, label=f"{label} row terminal_reason")
    _require_exact(
        terminal_classification,
        classification,
        label=f"{label} row terminal classification",
    )
    _require_exact(
        runtime_success,
        grade_runtime_success,
        label=f"{label} row runtime_success",
    )
    _require_exact(
        episode_success,
        submission_valid and improved,
        label=f"{label} row episode_success",
    )
    _require_exact(truncated, False, label=f"{label} completed terminal truncation")
    for field, expected in (
        ("request_id", request_id),
        ("episode_id", episode_id),
        ("submission_sha256", submission_sha256),
    ):
        _require_exact(
            action_submission.get(field),
            expected,
            label=f"{label} action submission {field}",
        )
    _sha256_string(receipt.get("audit_digest"), label=f"{label} grade audit digest")


def validate_ungraded_policy_terminal(
    row: dict[str, Any],
    *,
    reward: float,
    action: str,
    action_submission: dict[str, Any],
    grading_increment: int,
    counters_after: dict[str, int],
    max_policy_actions: int,
    runtime_success: bool,
    episode_success: bool,
    label: str,
) -> None:
    reason = row.get("terminal_reason")
    if reason not in POLICY_TERMINAL_REASONS:
        raise AssertionError(f"{label} ungraded policy terminal reason is invalid")
    if (
        reward != -1.0
        or row.get("grade_receipt") is not None
        or row.get("terminal_classification") is not None
        or row.get("truncated") is not False
        or runtime_success
        or episode_success
    ):
        raise AssertionError(f"{label} ungraded policy terminal is inconsistent")
    if set(action_submission) != {"raw_policy_output"}:
        raise AssertionError(
            f"{label} ungraded policy terminal fabricated submission identity"
        )
    if reason == "action_budget_exhausted" and (
        counters_after["action_count"] != max_policy_actions
    ):
        raise AssertionError(
            f"{label} action-budget terminal ended before max_policy_actions"
        )
    if grading_increment == 0:
        if counters_after["grading_count"] != 0:
            raise AssertionError(f"{label} ungraded terminal grade ledger drifted")
    elif not (
        grading_increment == 1
        and counters_after["grading_count"] == 1
        and action.strip() == "submit"
        and reason == "episode_wall_limit"
    ):
        raise AssertionError(
            f"{label} receipt-free grade attempt is not a bounded submit timeout"
        )


def validate_task_rows(
    rows: list[dict[str, Any]],
    record: dict[str, Any],
    manifest_sha256: str,
    max_policy_actions: int,
) -> dict[str, Any]:
    index = record["data_idx"]
    rows.sort(
        key=lambda value: _integer(
            value.get("task_round"), label="task_round", minimum=1
        )
    )
    rounds = [row["task_round"] for row in rows]
    _require_exact(
        rounds,
        list(range(1, len(rows) + 1)),
        label=f"data_idx {index} task rounds",
    )
    if not rows:
        raise AssertionError(f"data_idx {index} has no action rows")
    if len(rows) > max_policy_actions:
        raise AssertionError(
            f"data_idx {index} exceeds max_policy_actions={max_policy_actions}"
        )

    item_ids: set[str] = set()
    episode_ids: set[str] = set()
    previous_after = {key: 0 for key in COUNTER_KEYS}
    sampled_token_count = 0
    graded_terminal = False
    for offset, row in enumerate(rows):
        label = f"data_idx {index} row {offset + 1}"
        _require_exact(row.get("schema"), STEP_SCHEMA, label=f"{label} schema")
        _require_exact(row.get("parent_index"), index, label=f"{label} parent_index")
        _require_exact(row.get("data_idx"), index, label=f"{label} data_idx")
        for key in ("task_id", "source_family"):
            _require_exact(row.get(key), record[key], label=f"{label} {key}")
        _require_exact(
            row.get("task_manifest_sha256"),
            manifest_sha256,
            label=f"{label} manifest SHA-256",
        )
        episode_id = _identity_string(
            row.get("episode_id"), label=f"{label} episode_id"
        )
        episode_ids.add(episode_id)
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise AssertionError(f"{label} item_id is empty")
        item_ids.add(item_id)
        action = row.get("action")
        if not isinstance(action, str) or not action:
            raise AssertionError(f"{label} action is empty")
        submission = row.get("action_submission")
        if not isinstance(submission, dict):
            raise AssertionError(f"{label} action submission is missing")
        _require_exact(
            submission.get("raw_policy_output"),
            action,
            label=f"{label} raw action",
        )

        sampled_token_count += validate_token_row(row, label=label)
        before = normalize_counters(row.get("counters_before"), label=label)
        after = normalize_counters(row.get("counters_after"), label=label)
        if before != previous_after:
            raise AssertionError(
                f"{label} counter continuity failed: {before!r} != {previous_after!r}"
            )
        for key in COUNTER_KEYS:
            if after[key] < before[key]:
                raise AssertionError(f"{label} counter {key} regressed")
        counter_delta = row.get("counter_delta")
        if not isinstance(counter_delta, dict) or set(counter_delta) != set(
            COUNTER_KEYS
        ):
            raise AssertionError(f"{label} counter_delta fields are incomplete")
        expected_delta = {key: after[key] - before[key] for key in COUNTER_KEYS}
        for key, expected in expected_delta.items():
            _require_exact(
                _integer(
                    counter_delta.get(key),
                    label=f"{label} counter_delta {key}",
                ),
                expected,
                label=f"{label} counter_delta {key}",
            )
        _require_exact(
            after["action_count"],
            before["action_count"] + 1,
            label=f"{label} one-action accounting",
        )

        terminal = offset == len(rows) - 1
        _require_exact(bool(row.get("done")), terminal, label=f"{label} done")
        if not isinstance(row.get("done"), bool):
            raise AssertionError(f"{label} done is not boolean")
        if not isinstance(row.get("truncated"), bool):
            raise AssertionError(f"{label} truncated is not boolean")
        runtime_success = _boolean(
            row.get("runtime_success"), label=f"{label} runtime_success"
        )
        episode_success = _boolean(
            row.get("episode_success"), label=f"{label} episode_success"
        )
        reward = _finite(row.get("reward"), label=f"{label} reward")
        grading_increment = after["grading_count"] - before["grading_count"]
        if not terminal:
            if (
                reward != 0.0
                or row.get("grade_receipt") is not None
                or grading_increment != 0
                or row.get("terminal_reason") is not None
                or row.get("terminal_classification") is not None
                or runtime_success
                or episode_success
            ):
                raise AssertionError(
                    f"{label} violates terminal-only grading and reward"
                )
        else:
            if row.get("grade_receipt") is None:
                validate_ungraded_policy_terminal(
                    row,
                    reward=reward,
                    action=action,
                    action_submission=submission,
                    grading_increment=grading_increment,
                    counters_after=after,
                    max_policy_actions=max_policy_actions,
                    runtime_success=runtime_success,
                    episode_success=episode_success,
                    label=label,
                )
            else:
                _require_exact(
                    grading_increment,
                    1,
                    label=f"{label} terminal-only grading count",
                )
                validate_grade_receipt(
                    row.get("grade_receipt"),
                    record,
                    reward,
                    action_submission=submission,
                    episode_id=episode_id,
                    terminal_reason=row.get("terminal_reason"),
                    terminal_classification=row.get("terminal_classification"),
                    runtime_success=runtime_success,
                    episode_success=episode_success,
                    truncated=row["truncated"],
                    label=label,
                )
                graded_terminal = True
        previous_after = after

    _require_exact(len(item_ids), 1, label=f"data_idx {index} item_id continuity")
    _require_exact(
        len(episode_ids),
        1,
        label=f"data_idx {index} episode_id continuity",
    )
    return {
        "data_idx": index,
        "item_id": next(iter(item_ids)),
        "action_row_count": len(rows),
        "sampled_response_token_count": sampled_token_count,
        "final_reward": float(rows[-1]["reward"]),
        "graded_terminal": graded_terminal,
    }


def validate_optimizer_readback(
    readback: Any,
    *,
    global_step: int,
) -> dict[str, Any]:
    if not isinstance(readback, dict):
        raise AssertionError("optimizer readback is missing")
    _require_exact(
        readback.get("role"),
        "same_batch_post_optimizer_readback",
        label="optimizer readback role",
    )
    _require_exact(
        readback.get("global_step"),
        global_step,
        label="optimizer readback global_step",
    )
    result: dict[str, Any] = {}
    step_pairs: list[tuple[int, int]] = []
    for role in ("actor", "critic"):
        payload = readback.get(role)
        if not isinstance(payload, dict):
            raise AssertionError(f"{role} optimizer readback is missing")
        before = _integer(
            payload.get("optimizer_step_before"),
            label=f"{role} optimizer_step_before",
        )
        after = _integer(
            payload.get("optimizer_step_after"),
            label=f"{role} optimizer_step_after",
        )
        _require_exact(after, before + 1, label=f"{role} optimizer step delta")
        delta_l2 = _finite(
            payload.get("parameter_delta_l2"),
            label=f"{role} parameter_delta_l2",
        )
        max_abs_delta = _finite(
            payload.get("max_abs_delta"),
            label=f"{role} max_abs_delta",
        )
        changed = _integer(
            payload.get("parameter_probe_changed_count"),
            label=f"{role} changed parameter count",
            minimum=1,
        )
        if delta_l2 <= 0 or max_abs_delta <= 0:
            raise AssertionError(f"{role} parameter movement is not positive")
        step_pairs.append((before, after))
        result[role] = {
            "optimizer_step_before": before,
            "optimizer_step_after": after,
            "parameter_delta_l2": delta_l2,
            "max_abs_delta": max_abs_delta,
            "parameter_probe_changed_count": changed,
        }
    _require_exact(step_pairs[0], step_pairs[1], label="actor/critic optimizer steps")
    return result


def validate_cleanup(
    evidence: dict[str, Any],
    document: dict[str, Any],
    manifest_sha256: str,
    *,
    expected_outer_commit: str,
    expected_inner_commit: str,
    expected_prompt_sha256: str,
) -> dict[str, Any]:
    metadata = validate_endpoint_metadata(
        evidence.get("metadata_after"),
        document,
        manifest_sha256,
        expected_outer_commit=expected_outer_commit,
        expected_inner_commit=expected_inner_commit,
        expected_prompt_sha256=expected_prompt_sha256,
        label="metadata_after",
    )
    require_idle(metadata, label="PPO cleanup")

    cleanup = evidence.get("cleanup")
    if not isinstance(cleanup, dict):
        raise AssertionError("cleanup evidence is missing")
    _require_exact(
        cleanup.get("schema"),
        "openmle_fast_owned_cleanup_evidence_v1",
        label="cleanup schema",
    )
    run_id = _identity_string(cleanup.get("run_id"), label="cleanup run_id")
    process_owner = _identity_string(
        cleanup.get("process_owner"), label="cleanup process_owner"
    )
    _require_exact(
        cleanup.get("client_close_count"),
        EXPECTED_TASK_COUNT,
        label="client close count",
    )
    processes = cleanup.get("owned_processes")
    if not isinstance(processes, list) or len(processes) != 2:
        raise AssertionError("owned process cleanup evidence is missing")
    expected_roles = {"public-environment", "private-grader"}
    roles: set[str] = set()
    pids: set[int] = set()
    process_identities: set[tuple[int, int]] = set()
    for offset, process in enumerate(processes):
        if not isinstance(process, dict):
            raise AssertionError(f"owned process {offset} is not an object")
        role = process.get("role")
        if role not in expected_roles or role in roles:
            raise AssertionError(f"owned process {offset} has invalid role")
        roles.add(role)
        _require_exact(
            process.get("run_id"), run_id, label=f"owned process {role} run_id"
        )
        _require_exact(
            process.get("process_owner"),
            process_owner,
            label=f"owned process {role} owner",
        )
        pid = _integer(process.get("pid"), label=f"owned process {role} pid", minimum=1)
        start_ticks = _integer(
            process.get("start_ticks"),
            label=f"owned process {role} start_ticks",
            minimum=1,
        )
        if pid in pids or (pid, start_ticks) in process_identities:
            raise AssertionError("owned process identities are not unique")
        pids.add(pid)
        process_identities.add((pid, start_ticks))
        _require_exact(
            process.get("alive"),
            False,
            label=f"owned process {role} cleanup",
        )
        _require_exact(
            process.get("exit_code"),
            0,
            label=f"owned process {role} exit status",
        )
    _require_exact(roles, expected_roles, label="owned cleanup roles")

    census = cleanup.get("process_census")
    if not isinstance(census, dict):
        raise AssertionError("cleanup process census is missing")
    _require_exact(census.get("complete"), True, label="process census completeness")
    _require_exact(census.get("run_id"), run_id, label="process census run_id")
    _require_exact(
        census.get("process_owner"),
        process_owner,
        label="process census owner",
    )
    matched_pids = census.get("matched_pids")
    if not isinstance(matched_pids, list) or any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid < 1
        for pid in matched_pids
    ):
        raise AssertionError("process census matched_pids are invalid")
    _require_exact(
        matched_pids,
        sorted(pids),
        label="process census exact matched PIDs",
    )
    _require_exact(
        census.get("residual_pids"),
        [],
        label="process census residual PIDs",
    )

    checkpoint = cleanup.get("checkpoint_disposition")
    if not isinstance(checkpoint, dict):
        raise AssertionError("gate checkpoint disposition is missing")
    _require_exact(
        checkpoint.get("schema"),
        "openmle_fast_gate_checkpoint_disposition_v1",
        label="checkpoint disposition schema",
    )
    _require_exact(
        checkpoint.get("policy"),
        "discard_after_readback",
        label="checkpoint disposition policy",
    )
    _require_exact(
        checkpoint.get("checkpoint_reuse_allowed"),
        False,
        label="checkpoint reuse policy",
    )
    _require_exact(
        checkpoint.get("remaining_checkpoint_paths"),
        [],
        label="remaining gate checkpoints",
    )
    return {
        "client_close_count": EXPECTED_TASK_COUNT,
        "owned_process_count": 2,
        "process_census_complete": True,
        "active_slot_count": 0,
        "active_environment_count": 0,
        "active_workspace_count": 0,
    }


def verify_ppo_gate(
    evidence: dict[str, Any],
    document: dict[str, Any],
    manifest_sha256: str,
    endpoint_probe: dict[str, Any],
    *,
    expected_outer_commit: str,
    expected_inner_commit: str,
    expected_prompt_sha256: str,
    forbidden_canaries: list[str],
    manifest_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Verify exact task rows, one optimizer update, and complete cleanup."""

    require_public_safe(
        evidence,
        label="PPO evidence",
        forbidden_canaries=forbidden_canaries,
    )
    require_public_safe(
        endpoint_probe,
        label="endpoint evidence",
        forbidden_canaries=forbidden_canaries,
    )
    records = validate_manifest(document, manifest_sha256, manifest_bytes)
    validate_endpoint_probe(
        endpoint_probe,
        document,
        manifest_sha256,
        expected_outer_commit=expected_outer_commit,
        expected_inner_commit=expected_inner_commit,
        expected_prompt_sha256=expected_prompt_sha256,
    )
    _require_exact(evidence.get("schema"), UPDATE_SCHEMA, label="update schema")
    gate_contract = evidence.get("gate_contract")
    if not isinstance(gate_contract, dict):
        raise AssertionError("gate contract evidence is missing")
    _require_exact(
        gate_contract.get("schema"),
        "openmle_fast_gate_contract_v1",
        label="gate contract schema",
    )
    _require_exact(gate_contract.get("role"), EXPECTED_ROLE, label="gate contract role")
    _require_exact(
        gate_contract.get("optimizer_update_limit"),
        1,
        label="gate optimizer-update limit",
    )
    _require_exact(
        gate_contract.get("initialization"),
        "fresh_base_checkpoint",
        label="gate initialization",
    )
    _require_exact(
        gate_contract.get("resume_checkpoint"),
        None,
        label="gate resume checkpoint",
    )
    _require_exact(
        gate_contract.get("checkpoint_reuse_allowed"),
        False,
        label="gate checkpoint reuse",
    )
    _require_exact(
        evidence.get("manifest_sha256"),
        manifest_sha256,
        label="update manifest SHA-256",
    )
    _require_exact(
        evidence.get("policy_prompt_sha256"),
        expected_prompt_sha256,
        label="update prompt SHA-256",
    )
    runtime_source = evidence.get("runtime_source")
    if not isinstance(runtime_source, dict):
        raise AssertionError("update runtime_source is missing")
    validate_runtime_identity(
        runtime_source,
        expected_outer_commit=expected_outer_commit,
        expected_inner_commit=expected_inner_commit,
        label="update",
    )

    update = evidence.get("update")
    if not isinstance(update, dict):
        raise AssertionError("update evidence is missing")
    first_global_step = _integer(
        update.get("first_global_step"),
        label="first_global_step",
        minimum=1,
    )
    last_global_step = _integer(
        update.get("last_global_step"),
        label="last_global_step",
        minimum=1,
    )
    _require_exact(last_global_step, first_global_step, label="one-update step range")
    _require_exact(
        update.get("optimizer_update_count"),
        1,
        label="optimizer_update_count",
    )
    rows = update.get("rows")
    if not isinstance(rows, list) or not rows:
        raise AssertionError("sampled action rows are missing")
    rows_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for offset, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AssertionError(f"sampled action row {offset} is not an object")
        index = _integer(row.get("data_idx"), label=f"row {offset} data_idx")
        if index not in records:
            raise AssertionError(f"sampled action row has unknown data_idx: {index}")
        rows_by_index[index].append(row)
    if set(rows_by_index) != set(records):
        missing = sorted(set(records) - set(rows_by_index))
        extra = sorted(set(rows_by_index) - set(records))
        raise AssertionError(
            "one complete update must contain receipts for all manifest tasks: "
            f"missing={missing}, extra={extra}"
        )

    max_policy_actions = _integer(
        document.get("max_policy_actions"),
        label="manifest max_policy_actions",
        minimum=1,
    )
    _require_exact(max_policy_actions, 30, label="manifest max_policy_actions")
    task_receipts = [
        validate_task_rows(
            rows_by_index[index],
            records[index],
            manifest_sha256,
            max_policy_actions,
        )
        for index in sorted(records)
    ]
    item_ids = [receipt["item_id"] for receipt in task_receipts]
    if len(set(item_ids)) != len(item_ids):
        raise AssertionError("task receipts contain duplicate opaque item IDs")

    readback = validate_optimizer_readback(
        evidence.get("optimizer_readback"),
        global_step=last_global_step,
    )
    cleanup = validate_cleanup(
        evidence,
        document,
        manifest_sha256,
        expected_outer_commit=expected_outer_commit,
        expected_inner_commit=expected_inner_commit,
        expected_prompt_sha256=expected_prompt_sha256,
    )
    return {
        "schema": "openmle_fast_ppo_gate_attestation_v1",
        "status": "pass",
        "manifest_sha256": manifest_sha256,
        "global_step": last_global_step,
        "optimizer_update_count": 1,
        "checkpoint_reuse_allowed": False,
        "task_receipt_count": len(task_receipts),
        "sampled_action_row_count": len(rows),
        "sampled_response_token_count": sum(
            receipt["sampled_response_token_count"] for receipt in task_receipts
        ),
        "graded_terminal_count": sum(
            int(receipt["graded_terminal"]) for receipt in task_receipts
        ),
        "ungraded_policy_terminal_count": sum(
            int(not receipt["graded_terminal"]) for receipt in task_receipts
        ),
        "dataset_indices": [receipt["data_idx"] for receipt in task_receipts],
        "optimizer_readback": readback,
        "cleanup": cleanup,
        "endpoint_probe_status": "pass",
    }


def main() -> None:
    args = parse_args()
    manifest_bytes = args.manifest.read_bytes()
    document = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(document, dict):
        raise TypeError("manifest must contain a JSON object")
    attestation = verify_ppo_gate(
        load_json(args.evidence),
        document,
        args.manifest_sha256,
        load_json(args.endpoint_probe),
        expected_outer_commit=args.expected_outer_commit,
        expected_inner_commit=args.expected_inner_commit,
        expected_prompt_sha256=args.expected_prompt_sha256,
        forbidden_canaries=load_forbidden_canaries(args.forbidden_canaries_file),
        manifest_bytes=manifest_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "output": str(args.output)}))


if __name__ == "__main__":
    main()
