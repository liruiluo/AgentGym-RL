"""Validation helpers for one-action agent SFT records.

The policy-visible training input is limited to ``system_prompt`` and
``observation``.  Everything else in the record is execution evidence used to
reject fabricated or unsuccessful demonstrations before tokenization.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any


AGENT_ACTION_SCHEMA_V1 = "agentmemory_agent_action_sft_v1"
AGENT_ACTION_KINDS = (
    "native_search",
    "native_click",
    "workspace_shell_command",
    "workspace_apply_patch",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
_NATIVE_ACTION_RE = re.compile(r"^(search|click)\[([^\[\]\r\n]+)\]$")
_SHELL_ACTION_RE = re.compile(r"^shell_command\s+(\{.*\})$", re.DOTALL)
_PATCH_ACTION_RE = re.compile(
    r"^apply_patch\n\*\*\* Begin Patch\n.+\n\*\*\* End Patch$",
    re.DOTALL,
)


class AgentActionSchemaError(ValueError):
    """Raised when an SFT record lacks executable-action evidence."""


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise AgentActionSchemaError("text_sha256 requires text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def finalize_agent_action_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with a content-addressed ``record_sha256`` field."""

    result = dict(record)
    result.pop("record_sha256", None)
    result["record_sha256"] = canonical_json_sha256(result)
    return result


def validate_agent_action_record(record: Any) -> dict[str, str]:
    """Validate one executed action and return the three model-visible fields."""

    if not isinstance(record, Mapping):
        raise AgentActionSchemaError("agent action record must be an object")
    if record.get("schema") != AGENT_ACTION_SCHEMA_V1:
        raise AgentActionSchemaError(
            f"schema must be {AGENT_ACTION_SCHEMA_V1!r}"
        )

    system_prompt = _nonempty_text(record, "system_prompt")
    observation = _nonempty_text(record, "observation")
    action = _nonempty_text(record, "assistant_action")
    if action != action.strip():
        raise AgentActionSchemaError(
            "assistant_action must not contain leading or trailing whitespace"
        )
    action_kind = record.get("action_kind")
    if action_kind not in AGENT_ACTION_KINDS:
        raise AgentActionSchemaError(
            f"action_kind must be one of {AGENT_ACTION_KINDS!r}"
        )
    _validate_action_syntax(action_kind, action)

    template = _mapping(record, "chat_template")
    if template.get("add_generation_prompt") is not True:
        raise AgentActionSchemaError("chat_template must add a generation prompt")
    if template.get("enable_thinking") is not False:
        raise AgentActionSchemaError("agent_action_v1 requires enable_thinking=false")
    if template.get("assistant_terminator") != "<|im_end|>":
        raise AgentActionSchemaError(
            "agent_action_v1 requires assistant_terminator='<|im_end|>'"
        )

    execution = _mapping(record, "execution")
    if execution.get("accepted") is not True:
        raise AgentActionSchemaError("execution.accepted must be true")
    if execution.get("action_effect_verified") is not True:
        raise AgentActionSchemaError(
            "execution.action_effect_verified must be true"
        )
    if execution.get("submitted_action") != action:
        raise AgentActionSchemaError(
            "execution.submitted_action disagrees with assistant_action"
        )
    observation_after = _nonempty_text(
        execution, "observation_after", prefix="execution"
    )
    reward = execution.get("reward")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
    ):
        raise AgentActionSchemaError("execution.reward must be finite numeric")
    for key in ("terminated", "truncated"):
        if type(execution.get(key)) is not bool:
            raise AgentActionSchemaError(f"execution.{key} must be boolean")
    info_before = _mapping(execution, "info_before")
    info_after = _mapping(execution, "info_after")
    receipt = {
        "submitted_action": action,
        "observation_after": observation_after,
        "reward": reward,
        "terminated": execution["terminated"],
        "truncated": execution["truncated"],
        "info_after": info_after,
    }
    if execution.get("receipt_sha256") != canonical_json_sha256(receipt):
        raise AgentActionSchemaError("execution.receipt_sha256 mismatch")

    task = _mapping(record, "task")
    for key in (
        "surface",
        "task_family",
        "task_id",
        "orbit_id",
        "scenario_id",
        "split",
    ):
        _nonempty_text(task, key, prefix="task")
    for key in (
        "data_index",
        "orbit_index",
        "branch_index",
        "phase_index",
        "turn_index",
    ):
        value = task.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AgentActionSchemaError(
                f"task.{key} must be a non-negative integer"
            )

    before_phase = _nonnegative_int(
        info_before, "current_subtask_index", prefix="execution.info_before"
    )
    after_phase = _nonnegative_int(
        info_after, "current_subtask_index", prefix="execution.info_after"
    )
    before_phase_count = _positive_int(
        info_before, "phase_count", prefix="execution.info_before"
    )
    after_phase_count = _positive_int(
        info_after, "phase_count", prefix="execution.info_after"
    )
    if before_phase_count != after_phase_count:
        raise AgentActionSchemaError(
            "execution phase_count changed across one action"
        )
    if before_phase >= before_phase_count:
        raise AgentActionSchemaError(
            "execution.info_before.current_subtask_index exceeds phase_count"
        )
    if before_phase != task["phase_index"]:
        raise AgentActionSchemaError(
            "task.phase_index disagrees with execution.info_before"
        )
    for key in ("surface", "task_family", "split", "scenario_id"):
        if info_before.get(key) != task[key] or info_after.get(key) != task[key]:
            raise AgentActionSchemaError(
                f"task.{key} disagrees with execution environment info"
            )

    before_snapshot = _mapping(info_before, "workspace_snapshot")
    after_snapshot = _mapping(info_after, "workspace_snapshot")
    before_tree = _require_sha256(
        before_snapshot, "tree_sha256", "execution.info_before.workspace_snapshot"
    )
    after_tree = _require_sha256(
        after_snapshot, "tree_sha256", "execution.info_after.workspace_snapshot"
    )
    before_event_count = _nonnegative_int(
        info_before,
        "workspace_audit_event_count",
        prefix="execution.info_before",
    )
    after_event_count = _nonnegative_int(
        info_after,
        "workspace_audit_event_count",
        prefix="execution.info_after",
    )

    workspace_audit = _mapping(record, "workspace_audit")
    native_audit = _mapping(record, "native_audit")
    workspace_action = action_kind.startswith("workspace_")
    if workspace_audit.get("applicable") is not workspace_action:
        raise AgentActionSchemaError(
            "workspace_audit.applicable disagrees with action_kind"
        )
    if native_audit.get("applicable") is not (not workspace_action):
        raise AgentActionSchemaError(
            "native_audit.applicable disagrees with action_kind"
        )
    if workspace_audit.get("tree_sha256_before") != before_tree or (
        workspace_audit.get("tree_sha256_after") != after_tree
    ):
        raise AgentActionSchemaError(
            "workspace audit tree hashes disagree with environment snapshots"
        )
    if workspace_action:
        if workspace_audit.get("committed") is not True:
            raise AgentActionSchemaError(
                "successful workspace actions require committed=true"
            )
        if reward != 0 or execution["terminated"] or execution["truncated"]:
            raise AgentActionSchemaError(
                "workspace demonstrations must be zero-reward nonterminal actions"
            )
        if before_phase != after_phase:
            raise AgentActionSchemaError(
                "workspace demonstrations must not advance the shopping phase"
            )
        if after_event_count != before_event_count + 1:
            raise AgentActionSchemaError(
                "workspace demonstration must append exactly one audit event"
            )
        event = _mapping(workspace_audit, "event")
        expected_op = (
            "SHELL_COMMAND"
            if action_kind == "workspace_shell_command"
            else "APPLY_PATCH"
        )
        if event.get("op") != expected_op or event.get("status") != "executed":
            raise AgentActionSchemaError(
                "workspace audit event does not prove the executed action kind"
            )
        if event.get("workspace_tree_sha256_before") != workspace_audit.get(
            "tree_sha256_before"
        ) or event.get("workspace_tree_sha256_after") != workspace_audit.get(
            "tree_sha256_after"
        ):
            raise AgentActionSchemaError(
                "workspace audit tree hashes disagree with the exact event"
            )
        if event.get("event_id") != before_event_count or event.get(
            "phase_index"
        ) != before_phase:
            raise AgentActionSchemaError(
                "workspace audit event ordering or phase binding mismatch"
            )
        _nonempty_text(event, "episode_id", prefix="workspace_audit.event")
        _validate_workspace_request_binding(action_kind, action, event)
        workspace_ops = info_after.get("workspace_ops")
        if (
            not isinstance(workspace_ops, list)
            or len(workspace_ops) != 1
            or workspace_ops[0] != event
        ):
            raise AgentActionSchemaError(
                "execution.info_after must contain the exact workspace event"
            )
        tool_ops = info_after.get("tool_ops")
        if not isinstance(tool_ops, list) or tool_ops != [event]:
            raise AgentActionSchemaError(
                "execution.info_after tool_ops must contain the exact workspace event"
            )
        if info_after.get("workspace_latest_event") != event:
            raise AgentActionSchemaError(
                "execution.info_after latest workspace event mismatch"
            )
        if native_audit.get("event") is not None or (
            native_audit.get("purchase_receipt") is not None
        ) or native_audit.get("target_asin") is not None:
            raise AgentActionSchemaError(
                "workspace demonstration must not carry native execution evidence"
            )
        if action_kind == "workspace_shell_command":
            if event.get("exit_code") != 0 or event.get("timed_out") is not False:
                raise AgentActionSchemaError(
                    "workspace shell demonstration must exit successfully"
                )
        else:
            if event.get("transactional") is not True:
                raise AgentActionSchemaError(
                    "workspace patch demonstration must be transactional"
                )
            changed_paths = event.get("changed_paths")
            if not isinstance(changed_paths, list) or not changed_paths:
                raise AgentActionSchemaError(
                    "workspace patch demonstration must change at least one path"
                )
            if before_tree == after_tree:
                raise AgentActionSchemaError(
                    "workspace patch demonstration must change the workspace tree"
                )
    else:
        if workspace_audit.get("committed") is not False or (
            workspace_audit.get("event") is not None
        ):
            raise AgentActionSchemaError(
                "native demonstration must not claim a workspace event"
            )
        if before_tree != after_tree or before_event_count != after_event_count:
            raise AgentActionSchemaError(
                "native demonstration unexpectedly mutated the workspace"
            )
        if info_after.get("workspace_ops") != [] or (
            info_after.get("workspace_latest_event") is not None
        ):
            raise AgentActionSchemaError(
                "native demonstration unexpectedly reported a workspace event"
            )
        target_asin = _nonempty_text(native_audit, "target_asin", prefix="native_audit")
        if _ASIN_RE.fullmatch(target_asin) is None:
            raise AgentActionSchemaError(
                "native_audit.target_asin must be one uppercase ASIN"
            )
        event = _mapping(native_audit, "event")
        tool_ops = info_after.get("tool_ops")
        if not isinstance(tool_ops, list) or len(tool_ops) != 1 or tool_ops[0] != event:
            raise AgentActionSchemaError(
                "execution.info_after must contain the exact native event"
            )
        if event.get("raw_action") != action:
            raise AgentActionSchemaError(
                "native audit event disagrees with assistant_action"
            )
        native_match = _NATIVE_ACTION_RE.fullmatch(action)
        if native_match is None:  # pragma: no cover - syntax validation owns this.
            raise AgentActionSchemaError("invalid native action syntax")
        native_op, native_argument = native_match.groups()
        purchase_receipt = native_audit.get("purchase_receipt")
        if native_op == "search":
            if event.get("op") != "SEARCH":
                raise AgentActionSchemaError("native search lacks SEARCH evidence")
            if (
                isinstance(event.get("result_count"), bool)
                or not isinstance(event.get("result_count"), int)
                or event["result_count"] < 1
            ):
                raise AgentActionSchemaError(
                    "native search must return at least one catalog result"
                )
            _require_zero_reward_nonterminal(execution, reward, "native search")
            if before_phase != after_phase or purchase_receipt is not None:
                raise AgentActionSchemaError(
                    "native search changed phase or carried a purchase receipt"
                )
        elif native_argument.casefold() != "buy now":
            if event.get("op") != "CLICK" or native_argument.upper() != target_asin:
                raise AgentActionSchemaError(
                    "native product click lacks exact target-ASIN evidence"
                )
            _require_zero_reward_nonterminal(execution, reward, "native click")
            if before_phase != after_phase or purchase_receipt is not None:
                raise AgentActionSchemaError(
                    "native product click changed phase or carried a purchase receipt"
                )
        else:
            if event.get("op") != "BUY":
                raise AgentActionSchemaError("native BUY lacks BUY evidence")
            for key in ("committed", "purchase_correct", "session_advanced"):
                if event.get(key) is not True:
                    raise AgentActionSchemaError(
                        f"native BUY requires event.{key}=true"
                    )
            purchase = _mapping(native_audit, "purchase_receipt")
            for key in (
                "op",
                "raw_action",
                "committed",
                "purchase_correct",
                "session_advanced",
                "terminal",
                "step",
                "session_index",
            ):
                if purchase.get(key) != event.get(key):
                    raise AgentActionSchemaError(
                        "private BUY receipt disagrees with the public native event"
                    )
            if purchase.get("actual_asin") != target_asin or (
                purchase.get("purchase_correct") is not True
            ):
                raise AgentActionSchemaError(
                    "native BUY receipt disagrees with the target ASIN"
                )
            if after_phase != before_phase + 1:
                raise AgentActionSchemaError(
                    "correct native BUY must advance exactly one phase"
                )
            final_purchase = after_phase == before_phase_count
            expected_reward = 2.0 if final_purchase else 1.0
            if float(reward) != expected_reward:
                raise AgentActionSchemaError(
                    "correct native BUY reward disagrees with phase position"
                )
            if execution["terminated"] is not final_purchase or execution["truncated"]:
                raise AgentActionSchemaError(
                    "native BUY terminal flags disagree with phase position"
                )
            if event.get("terminal") is not final_purchase:
                raise AgentActionSchemaError(
                    "native BUY event terminal flag disagrees with phase position"
                )

    provenance = _mapping(record, "provenance")
    expected_hashes = {
        "system_prompt_sha256": text_sha256(system_prompt),
        "observation_sha256": text_sha256(observation),
        "assistant_action_sha256": text_sha256(action),
        "observation_after_sha256": text_sha256(observation_after),
        "env_info_before_sha256": canonical_json_sha256(info_before),
        "env_info_after_sha256": canonical_json_sha256(info_after),
    }
    for key, expected in expected_hashes.items():
        if provenance.get(key) != expected:
            raise AgentActionSchemaError(f"provenance.{key} mismatch")
    for key in ("outer_source_commit", "agentgym_source_commit"):
        _require_git_object(provenance, key, "provenance")
    for key in (
        "provider_proof_sha256",
        "product_pool_sha256",
        "task_semantic_sha256",
        "target_product_record_sha256",
    ):
        _require_sha256(provenance, key, "provenance")

    record_sha256 = record.get("record_sha256")
    if not isinstance(record_sha256, str) or not _SHA256_RE.fullmatch(
        record_sha256
    ):
        raise AgentActionSchemaError("record_sha256 must be lowercase SHA-256")
    unhashed = dict(record)
    unhashed.pop("record_sha256", None)
    if canonical_json_sha256(unhashed) != record_sha256:
        raise AgentActionSchemaError("record_sha256 mismatch")

    return {
        "system_prompt": system_prompt,
        "observation": observation,
        "assistant_action": action,
    }


def _validate_action_syntax(action_kind: str, action: str) -> None:
    if action_kind.startswith("native_"):
        match = _NATIVE_ACTION_RE.fullmatch(action)
        expected = action_kind.removeprefix("native_")
        if match is None or match.group(1) != expected:
            raise AgentActionSchemaError(
                f"{action_kind} has invalid native action syntax"
            )
        return
    if action_kind == "workspace_apply_patch":
        if _PATCH_ACTION_RE.fullmatch(action) is None:
            raise AgentActionSchemaError("invalid apply_patch action syntax")
        return
    match = _SHELL_ACTION_RE.fullmatch(action)
    if match is None:
        raise AgentActionSchemaError("invalid shell_command action syntax")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise AgentActionSchemaError("shell_command payload is not JSON") from exc
    if not isinstance(payload, dict) or set(payload) - {
        "command",
        "workdir",
        "timeout_ms",
    }:
        raise AgentActionSchemaError(
            "shell_command payload contains unsupported fields"
        )
    command = payload.get("command")
    if not isinstance(command, str) or not command.strip():
        raise AgentActionSchemaError("shell_command requires command:string")
    if "workdir" in payload and not isinstance(payload["workdir"], str):
        raise AgentActionSchemaError("shell_command workdir must be text")
    if "timeout_ms" in payload and (
        isinstance(payload["timeout_ms"], bool)
        or not isinstance(payload["timeout_ms"], int)
        or payload["timeout_ms"] <= 0
    ):
        raise AgentActionSchemaError(
            "shell_command timeout_ms must be a positive integer"
        )


def _validate_workspace_request_binding(
    action_kind: str,
    action: str,
    event: Mapping[str, Any],
) -> None:
    """Bind the supervised action bytes to the authoritative runtime event."""

    if action_kind == "workspace_apply_patch":
        patch_text = action.removeprefix("apply_patch\n")
        patch_bytes = patch_text.encode("utf-8")
        expected_request_sha256 = hashlib.sha256(patch_bytes).hexdigest()
        if event.get("request_sha256") != expected_request_sha256 or event.get(
            "patch_sha256"
        ) != expected_request_sha256:
            raise AgentActionSchemaError(
                "workspace apply_patch event is not bound to assistant_action"
            )
        if event.get("patch_bytes") != len(patch_bytes):
            raise AgentActionSchemaError(
                "workspace apply_patch event has the wrong request size"
            )
        return

    match = _SHELL_ACTION_RE.fullmatch(action)
    if match is None:  # pragma: no cover - syntax validation owns this.
        raise AgentActionSchemaError("invalid shell_command action syntax")
    payload = json.loads(match.group(1))
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    command = payload["command"]
    command_bytes = command.encode("utf-8")
    if event.get("request_sha256") != text_sha256(canonical_payload):
        raise AgentActionSchemaError(
            "workspace shell event is not bound to assistant_action"
        )
    if event.get("command_sha256") != hashlib.sha256(command_bytes).hexdigest() or (
        event.get("command_bytes") != len(command_bytes)
    ):
        raise AgentActionSchemaError(
            "workspace shell event is not bound to the submitted command"
        )


def _mapping(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise AgentActionSchemaError(f"{key} must be an object")
    return value


def _nonempty_text(
    record: Mapping[str, Any], key: str, *, prefix: str | None = None
) -> str:
    value = record.get(key)
    label = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, str) or not value.strip():
        raise AgentActionSchemaError(f"{label} must be non-empty text")
    return value


def _require_sha256(record: Mapping[str, Any], key: str, prefix: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AgentActionSchemaError(f"{prefix}.{key} must be lowercase SHA-256")
    return value


def _require_git_object(record: Mapping[str, Any], key: str, prefix: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or _GIT_OBJECT_RE.fullmatch(value) is None:
        raise AgentActionSchemaError(
            f"{prefix}.{key} must be a full 40- or 64-hex git object ID"
        )
    return value


def _nonnegative_int(
    record: Mapping[str, Any], key: str, *, prefix: str
) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgentActionSchemaError(f"{prefix}.{key} must be a non-negative integer")
    return value


def _positive_int(record: Mapping[str, Any], key: str, *, prefix: str) -> int:
    value = _nonnegative_int(record, key, prefix=prefix)
    if value == 0:
        raise AgentActionSchemaError(f"{prefix}.{key} must be positive")
    return value


def _require_zero_reward_nonterminal(
    execution: Mapping[str, Any], reward: int | float, label: str
) -> None:
    if reward != 0 or execution["terminated"] or execution["truncated"]:
        raise AgentActionSchemaError(
            f"{label} must be a zero-reward nonterminal action"
        )
