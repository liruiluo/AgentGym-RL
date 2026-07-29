from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from copy import deepcopy
from typing import Any, Mapping, Sequence


FORMAL_DOMAIN_SCHEMA_V3 = "agentmemory_formal_step_v3"
FORMAL_WEBSHOP_SCHEMA_V2 = "agentmemory_formal_step_v2"
FORMAL_WEBSHOP_SURFACE_V2 = "memoryarena_webshop_native_v1"
LTM_INVENTORY_MODES = ("hidden", "keys")
MEMORY_PROMPT_MODES = (
    "legacy",
    "neutral",
    "neutral_horizon",
    "neutral_horizon_responsibility",
)
ACTION_LISTING_MODES = ("separate", "unified")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class FormalDomainV3Error(ValueError):
    pass


def canonical_unicode_contains(text: str, fragment: str) -> bool:
    """Compare decoded prompt text across canonically equivalent Unicode forms."""

    return unicodedata.normalize("NFC", fragment) in unicodedata.normalize("NFC", text)


def validate_webshop_ltm_inventory_mode(
    metadata: Mapping[str, Any],
    *,
    expected_mode: str,
) -> None:
    """Keep the WebShop observation and rollout prompt on the same interface."""

    if expected_mode not in LTM_INVENTORY_MODES:
        raise FormalDomainV3Error(
            f"unsupported rollout LTM inventory mode: {expected_mode!r}"
        )
    server_mode = metadata.get("ltm_inventory_mode")
    if server_mode is None:
        if expected_mode != "hidden":
            raise FormalDomainV3Error(
                "WebShop runtime metadata is missing ltm_inventory_mode"
            )
        return
    if server_mode not in LTM_INVENTORY_MODES:
        raise FormalDomainV3Error(
            f"unsupported server LTM inventory mode: {server_mode!r}"
        )
    if server_mode != expected_mode:
        raise FormalDomainV3Error(
            "WebShop server and rollout LTM inventory modes disagree: "
            f"server={server_mode!r} rollout={expected_mode!r}"
        )


def validate_webshop_memory_prompt_mode(
    metadata: Mapping[str, Any],
    *,
    expected_mode: str,
) -> None:
    """Keep the WebShop server, adapter, and rollout prompt on one mode."""

    if expected_mode not in MEMORY_PROMPT_MODES:
        raise FormalDomainV3Error(
            f"unsupported rollout memory prompt mode: {expected_mode!r}"
        )
    server_mode = metadata.get("memory_prompt_mode")
    if server_mode is None:
        if expected_mode != "legacy":
            raise FormalDomainV3Error(
                "WebShop runtime metadata is missing memory_prompt_mode"
            )
        return
    if server_mode not in MEMORY_PROMPT_MODES:
        raise FormalDomainV3Error(
            f"unsupported server memory prompt mode: {server_mode!r}"
        )
    if server_mode != expected_mode:
        raise FormalDomainV3Error(
            "WebShop server and rollout memory prompt modes disagree: "
            f"server={server_mode!r} rollout={expected_mode!r}"
        )


def validate_webshop_action_listing_mode(
    metadata: Mapping[str, Any],
    *,
    expected_mode: str,
) -> None:
    """Keep the rendered WebShop action interface on the requested variant."""

    if expected_mode not in ACTION_LISTING_MODES:
        raise FormalDomainV3Error(
            f"unsupported rollout action listing mode: {expected_mode!r}"
        )
    server_mode = metadata.get("action_listing_mode")
    if server_mode is None:
        if expected_mode != "separate":
            raise FormalDomainV3Error(
                "WebShop runtime metadata is missing action_listing_mode"
            )
        return
    if server_mode not in ACTION_LISTING_MODES:
        raise FormalDomainV3Error(
            f"unsupported server action listing mode: {server_mode!r}"
        )
    if server_mode != expected_mode:
        raise FormalDomainV3Error(
            "WebShop server and rollout action listing modes disagree: "
            f"server={server_mode!r} rollout={expected_mode!r}"
        )


def resolve_formal_runtime_contract(
    metadata: Mapping[str, Any],
    *,
    webshop_v2_system_prompt: str,
) -> tuple[str, str, str]:
    """Resolve the formal schema and the exact system prompt used for sampling."""

    if not isinstance(metadata, Mapping):
        raise FormalDomainV3Error("formal runtime metadata must be an object")
    surface = metadata.get("surface")
    schema_version = metadata.get("formal_schema_version")
    if schema_version in (None, FORMAL_WEBSHOP_SCHEMA_V2):
        if surface != FORMAL_WEBSHOP_SURFACE_V2:
            raise FormalDomainV3Error(
                "formal runtime metadata without v3 schema must be the WebShop v2 surface"
            )
        if not isinstance(webshop_v2_system_prompt, str) or not webshop_v2_system_prompt.strip():
            raise FormalDomainV3Error("WebShop v2 system prompt must not be empty")
        return (
            FORMAL_WEBSHOP_SCHEMA_V2,
            webshop_v2_system_prompt,
            "rollout_webshop_v2",
        )
    if schema_version != FORMAL_DOMAIN_SCHEMA_V3:
        raise FormalDomainV3Error(
            f"unsupported formal runtime schema: {schema_version!r}"
        )
    system_prompt = metadata.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise FormalDomainV3Error(
            "formal v3 runtime metadata requires a non-empty system_prompt"
        )
    return FORMAL_DOMAIN_SCHEMA_V3, system_prompt, "server_metadata"


def validate_formal_env_schema(
    schema_version: str,
    env_info: Mapping[str, Any],
    *,
    boundary: str,
) -> None:
    """Fail closed when metadata and transition evidence disagree on schema."""

    if not isinstance(env_info, Mapping):
        raise FormalDomainV3Error(f"{boundary} env_info must be an object")
    observed = env_info.get("formal_schema_version")
    if schema_version == FORMAL_DOMAIN_SCHEMA_V3:
        if observed != FORMAL_DOMAIN_SCHEMA_V3:
            raise FormalDomainV3Error(
                f"{boundary} env_info schema does not match v3 metadata"
            )
        return
    if schema_version != FORMAL_WEBSHOP_SCHEMA_V2:
        raise FormalDomainV3Error(f"unsupported formal schema: {schema_version!r}")
    if observed not in (None, FORMAL_WEBSHOP_SCHEMA_V2):
        raise FormalDomainV3Error(
            f"{boundary} env_info schema does not match WebShop v2 metadata"
        )


def build_formal_domain_step_v3(
    *,
    content: str,
    score: float,
    task_round: int,
    done: bool,
    item_id: str,
    parent_index: int,
    parent_group_uid: str,
    replica_index: int,
    trajectory_uid: str,
    exact_state_uid: str,
    prompt_token_ids: Sequence[int],
    response_token_ids: Sequence[int],
    latest_observation: str,
    visible_prompt: str,
    system_prompt: str,
    single_observation_prompt_digest: str,
    env_result: str,
    generation_record: Mapping[str, Any],
    env_info_before: Mapping[str, Any],
    env_info_after: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one domain-neutral formal row from authoritative server evidence."""

    done = _require_boolean_value(done, "done")
    before = deepcopy(dict(env_info_before))
    after = deepcopy(dict(env_info_after))
    _validate_env_info_pair(before, after, score=score, task_round=task_round, done=done)
    phase_before = int(before["phase_index"])
    phase_after = int(after["phase_index"])
    phase_advanced = phase_after > phase_before
    episode_success = after["episode_success"]
    generation_token_ids_are_exact = _require_boolean_value(
        generation_record.get("token_ids_are_exact"),
        "generation_record.token_ids_are_exact",
    )
    backend_token_ids_are_exact = _require_boolean_value(
        generation_record.get("backend_token_ids_are_exact"),
        "generation_record.backend_token_ids_are_exact",
    )
    truncated = _require_boolean_value(
        generation_record.get("truncated"),
        "generation_record.truncated",
    )
    outcome = (
        "success"
        if done and episode_success
        else "terminal_failure" if done else "continue"
    )
    action_execution = deepcopy(after["action_execution"])
    record = {
        "schema_version": FORMAL_DOMAIN_SCHEMA_V3,
        "content": content,
        "score": float(score),
        "task_round": int(task_round),
        "done": done,
        "outcome": outcome,
        "item_id": str(item_id),
        "parent_index": int(parent_index),
        "parent_group_uid": str(parent_group_uid),
        "replica_index": int(replica_index),
        "trajectory_uid": str(trajectory_uid),
        "exact_state_uid": str(exact_state_uid),
        "domain_id": str(after["domain_id"]),
        "surface": str(after["surface"]),
        "contract_id": str(after["contract_id"]),
        "contract_sha256": str(after["contract_sha256"]),
        "phase_index_before": phase_before,
        "phase_index_after": phase_after,
        "phase_count": after.get("phase_count"),
        "phase_advanced": phase_advanced,
        "episode_success": episode_success,
        "prompt_token_ids": [int(token) for token in prompt_token_ids],
        "response_token_ids": [int(token) for token in response_token_ids],
        "latest_observation": latest_observation,
        "visible_prompt": visible_prompt,
        "system_prompt": system_prompt,
        "system_prompt_source": "server_metadata",
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "single_observation_prompt_digest": single_observation_prompt_digest,
        "prompt_history_policy": "latest_observation_only",
        "raw_prior_messages_visible": False,
        "response_token_count": int(generation_record["response_token_count"]),
        "max_response_tokens": int(generation_record["max_response_tokens"]),
        "finish_reason": str(generation_record["finish_reason"]),
        "finish_reason_source": str(generation_record["finish_reason_source"]),
        "stop_reason": generation_record.get("stop_reason"),
        "generation_backend_source": str(generation_record["backend_source"]),
        "generation_eos_token_ids": [
            int(token) for token in generation_record["configured_eos_token_ids"]
        ],
        "tokenizer_primary_eos_token_id": generation_record[
            "primary_eos_token_id"
        ],
        "tokenizer_pad_token_id": generation_record["tokenizer_pad_token_id"],
        "generation_token_ids_are_exact": generation_token_ids_are_exact,
        "backend_token_ids_are_exact": backend_token_ids_are_exact,
        "truncated": truncated,
        "env_result": str(env_result),
        "env_info_before": before,
        "env_info_after": after,
        "action_execution": action_execution,
        "tool_ops": deepcopy(after["tool_ops"]),
        "reward_components": deepcopy(after["reward_components"]),
        "domain_evidence": deepcopy(after["domain_evidence"]),
        "sample_excluded": after["sample_excluded"],
    }
    validate_formal_domain_step_v3(record)
    return record


def validate_formal_domain_step_v3(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "content",
        "score",
        "task_round",
        "done",
        "outcome",
        "domain_id",
        "surface",
        "contract_id",
        "contract_sha256",
        "phase_index_before",
        "phase_index_after",
        "phase_advanced",
        "episode_success",
        "raw_prior_messages_visible",
        "generation_token_ids_are_exact",
        "backend_token_ids_are_exact",
        "tokenizer_primary_eos_token_id",
        "truncated",
        "sample_excluded",
        "system_prompt",
        "system_prompt_source",
        "system_prompt_sha256",
        "single_observation_prompt_digest",
        "response_token_ids",
        "action_execution",
        "reward_components",
        "domain_evidence",
        "env_info_before",
        "env_info_after",
    }
    missing = sorted(required - set(record))
    if missing:
        raise FormalDomainV3Error(
            "formal v3 record missing field(s): " + ", ".join(missing)
        )
    if record["schema_version"] != FORMAL_DOMAIN_SCHEMA_V3:
        raise FormalDomainV3Error("unsupported formal domain schema version")
    content = record.get("content")
    if not isinstance(content, str):
        raise FormalDomainV3Error("content must be a string")
    _require_nonempty_text(record, "domain_id")
    _require_nonempty_text(record, "surface")
    _require_nonempty_text(record, "contract_id")
    system_prompt = _require_nonempty_text(record, "system_prompt")
    if record["system_prompt_source"] != "server_metadata":
        raise FormalDomainV3Error("formal v3 system prompt must come from server metadata")
    expected_system_prompt_sha256 = hashlib.sha256(
        system_prompt.encode("utf-8")
    ).hexdigest()
    if record["system_prompt_sha256"] != expected_system_prompt_sha256:
        raise FormalDomainV3Error("system_prompt_sha256 disagrees with system_prompt")
    if _SHA256_RE.fullmatch(str(record["single_observation_prompt_digest"])) is None:
        raise FormalDomainV3Error("single_observation_prompt_digest must be SHA256 hex")
    if not isinstance(record.get("latest_observation"), str) or not record[
        "latest_observation"
    ]:
        raise FormalDomainV3Error("latest_observation must be non-empty text")
    if not isinstance(record.get("visible_prompt"), str) or not record["visible_prompt"]:
        raise FormalDomainV3Error("visible_prompt must be non-empty text")
    if not canonical_unicode_contains(record["visible_prompt"], system_prompt):
        raise FormalDomainV3Error("visible_prompt omits the server system prompt")
    if not canonical_unicode_contains(
        record["visible_prompt"], record["latest_observation"]
    ):
        raise FormalDomainV3Error("visible_prompt omits the latest observation")
    contract_sha256 = str(record["contract_sha256"])
    if _SHA256_RE.fullmatch(contract_sha256) is None:
        raise FormalDomainV3Error("contract_sha256 must be lowercase SHA256 hex")
    task_round = _require_nonnegative_int(record, "task_round", positive=True)
    phase_before = _require_nonnegative_int(record, "phase_index_before")
    phase_after = _require_nonnegative_int(record, "phase_index_after")
    if phase_after < phase_before or phase_after > phase_before + 1:
        raise FormalDomainV3Error("phase index must stay put or advance exactly once")
    phase_advanced = _require_boolean_value(
        record["phase_advanced"],
        "phase_advanced",
    )
    if phase_advanced != (phase_after > phase_before):
        raise FormalDomainV3Error("phase_advanced disagrees with phase indices")
    phase_count = record.get("phase_count")
    if phase_count is not None:
        if isinstance(phase_count, bool) or not isinstance(phase_count, int):
            raise FormalDomainV3Error("phase_count must be an integer or null")
        if phase_count < 1 or phase_after > phase_count:
            raise FormalDomainV3Error("phase_count is inconsistent with phase index")
    score = _require_finite_number(record, "score")
    components = record["reward_components"]
    if not isinstance(components, list) or not components:
        raise FormalDomainV3Error("reward_components must be a non-empty list")
    component_sum = 0.0
    for component in components:
        if not isinstance(component, Mapping):
            raise FormalDomainV3Error("reward component must be an object")
        component_sum += _require_finite_number(component, "value")
        if int(component.get("step", -1)) != task_round:
            raise FormalDomainV3Error("reward component step disagrees with task_round")
    if not math.isclose(component_sum, score, rel_tol=0.0, abs_tol=1e-8):
        raise FormalDomainV3Error(
            f"reward ledger sum mismatch: components={component_sum} score={score}"
        )
    execution = record["action_execution"]
    if not isinstance(execution, Mapping):
        raise FormalDomainV3Error("action_execution must be an object")
    if execution.get("raw_policy_output") != content:
        raise FormalDomainV3Error("raw_policy_output must equal sampled content")
    submitted_action = execution.get("submitted_action")
    if not isinstance(submitted_action, str):
        raise FormalDomainV3Error("submitted_action must be a string")
    op = _require_nonempty_text(execution, "op")
    status = _require_nonempty_text(execution, "status")
    invalid_action = op.upper() == "INVALID" or status.lower() == "invalid"
    if (not content.strip() or not submitted_action.strip()) and not invalid_action:
        raise FormalDomainV3Error(
            "empty sampled/submitted action is valid only for an INVALID outcome"
        )
    if int(execution.get("step", -1)) != task_round:
        raise FormalDomainV3Error("action_execution step disagrees with task_round")
    if not isinstance(record["response_token_ids"], list) or not record[
        "response_token_ids"
    ]:
        raise FormalDomainV3Error("response_token_ids must be a non-empty list")
    raw_prior_messages_visible = _require_boolean_value(
        record["raw_prior_messages_visible"],
        "raw_prior_messages_visible",
    )
    if raw_prior_messages_visible:
        raise FormalDomainV3Error("raw_prior_messages_visible must be false")
    generation_token_ids_are_exact = _require_boolean_value(
        record["generation_token_ids_are_exact"],
        "generation_token_ids_are_exact",
    )
    if not generation_token_ids_are_exact:
        raise FormalDomainV3Error("generation token ids must be exact")
    backend_token_ids_are_exact = _require_boolean_value(
        record["backend_token_ids_are_exact"],
        "backend_token_ids_are_exact",
    )
    if not backend_token_ids_are_exact:
        raise FormalDomainV3Error("backend token ids must be exact")
    _require_boolean_value(record["truncated"], "truncated")
    done = _require_boolean_value(record["done"], "done")
    episode_success = _require_boolean_value(
        record["episode_success"],
        "episode_success",
    )
    sample_excluded = _require_boolean_value(
        record["sample_excluded"],
        "sample_excluded",
    )
    expected_outcome = (
        "success" if done and episode_success else "terminal_failure" if done else "continue"
    )
    if record["outcome"] != expected_outcome:
        raise FormalDomainV3Error("outcome disagrees with done/episode_success")
    if episode_success and not done:
        raise FormalDomainV3Error("episode_success requires done=True")
    if sample_excluded and not done:
        raise FormalDomainV3Error("sample_excluded requires done=True")
    if not isinstance(record["domain_evidence"], Mapping):
        raise FormalDomainV3Error("domain_evidence must be an object")
    _validate_env_info_pair(
        record["env_info_before"],
        record["env_info_after"],
        score=score,
        task_round=task_round,
        done=done,
    )
    before = record["env_info_before"]
    after = record["env_info_after"]
    identity_fields = ("domain_id", "surface", "contract_id", "contract_sha256")
    for field in identity_fields:
        if record[field] != after[field]:
            raise FormalDomainV3Error(
                f"{field} disagrees with authoritative env_info_after"
            )
    if phase_before != int(before["phase_index"]):
        raise FormalDomainV3Error(
            "phase_index_before disagrees with authoritative env_info_before"
        )
    if phase_after != int(after["phase_index"]):
        raise FormalDomainV3Error(
            "phase_index_after disagrees with authoritative env_info_after"
        )
    if record.get("phase_count") != after.get("phase_count"):
        raise FormalDomainV3Error(
            "phase_count disagrees with authoritative env_info_after"
        )
    if episode_success != after["episode_success"]:
        raise FormalDomainV3Error(
            "episode_success disagrees with authoritative env_info_after"
        )
    authoritative_bindings = {
        "action_execution": after.get("action_execution"),
        "tool_ops": after.get("tool_ops"),
        "reward_components": after.get("reward_components"),
        "domain_evidence": after.get("domain_evidence"),
    }
    for field, authoritative_value in authoritative_bindings.items():
        if record.get(field) != authoritative_value:
            raise FormalDomainV3Error(
                f"{field} disagrees with authoritative env_info_after"
            )
    if sample_excluded != after["sample_excluded"]:
        raise FormalDomainV3Error(
            "sample_excluded disagrees with authoritative env_info_after"
        )


def bind_generic_timeout_v3(
    record: dict[str, Any],
    *,
    max_policy_turns: int,
    penalty: float = -0.01,
) -> None:
    """Bind a policy-turn ceiling to the last nonterminal formal v3 row."""

    validate_formal_domain_step_v3(record)
    if record["done"]:
        raise FormalDomainV3Error("cannot bind timeout to a terminal row")
    if int(record["task_round"]) != int(max_policy_turns):
        raise FormalDomainV3Error(
            "timeout row must be the declared policy-turn ceiling"
        )
    component = {
        "name": "policy_turn_ceiling_failure",
        "value": float(penalty),
        "op": str(record["action_execution"]["op"]),
        "step": int(record["task_round"]),
        "max_policy_turns": int(max_policy_turns),
    }
    record["reward_components"].append(component)
    record["score"] = float(record["score"]) + float(penalty)
    record["done"] = True
    record["outcome"] = "terminal_failure"
    record["episode_success"] = False
    record["env_info_after"]["reward_components"] = deepcopy(
        record["reward_components"]
    )
    record["env_info_after"]["done"] = True
    record["env_info_after"]["episode_success"] = False
    validate_formal_domain_step_v3(record)


def _validate_env_info_pair(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    score: float,
    task_round: int,
    done: bool,
) -> None:
    done = _require_boolean_value(done, "done")
    required = {
        "formal_schema_version",
        "domain_id",
        "surface",
        "contract_id",
        "contract_sha256",
        "phase_index",
        "phase_count",
        "episode_success",
        "sample_excluded",
    }
    for label, info in (("before", before), ("after", after)):
        if not isinstance(info, Mapping):
            raise FormalDomainV3Error(f"env_info_{label} must be an object")
        missing = sorted(required - set(info))
        if missing:
            raise FormalDomainV3Error(
                f"env_info_{label} missing field(s): " + ", ".join(missing)
            )
        if info["formal_schema_version"] != FORMAL_DOMAIN_SCHEMA_V3:
            raise FormalDomainV3Error(f"env_info_{label} schema is not v3")
        _require_boolean_value(
            info["episode_success"],
            f"env_info_{label}.episode_success",
        )
        _require_boolean_value(
            info["sample_excluded"],
            f"env_info_{label}.sample_excluded",
        )
        if "done" in info:
            _require_boolean_value(info["done"], f"env_info_{label}.done")
    for key in ("domain_id", "surface", "contract_id", "contract_sha256"):
        if before[key] != after[key]:
            raise FormalDomainV3Error(f"{key} changed within one transition")
    if int(after["phase_index"]) < int(before["phase_index"]):
        raise FormalDomainV3Error("phase index regressed")
    components = after.get("reward_components")
    if not isinstance(components, list) or not components:
        raise FormalDomainV3Error("env_info_after requires reward_components")
    component_sum = sum(_require_finite_number(item, "value") for item in components)
    if not math.isclose(component_sum, float(score), rel_tol=0.0, abs_tol=1e-8):
        raise FormalDomainV3Error("env_info_after reward ledger disagrees with score")
    execution = after.get("action_execution")
    if not isinstance(execution, Mapping):
        raise FormalDomainV3Error("env_info_after requires action_execution")
    if int(execution.get("step", -1)) != int(task_round):
        raise FormalDomainV3Error("env action step disagrees with task_round")
    if "done" in after and after["done"] != done:
        raise FormalDomainV3Error("env_info_after done disagrees with step output")


def _require_nonempty_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FormalDomainV3Error(f"{key} must be a non-empty string")
    return value


def _require_boolean_value(value: Any, key: str) -> bool:
    if type(value) is not bool:
        raise FormalDomainV3Error(f"{key} must be boolean")
    return value


def _require_nonnegative_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    positive: bool = False,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormalDomainV3Error(f"{key} must be an integer")
    if value < (1 if positive else 0):
        raise FormalDomainV3Error(f"{key} is out of range")
    return value


def _require_finite_number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalDomainV3Error(f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise FormalDomainV3Error(f"{key} must be finite")
    return value
