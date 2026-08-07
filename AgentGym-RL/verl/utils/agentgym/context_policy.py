from __future__ import annotations

import os
from typing import Any


AGENTMEMORY_RAW_HISTORY_ERROR = (
    "AgentMemoryGym requires an ephemeral/latest-observation rollout context. "
    "The current AgentGym-RL vLLM rollout appends the full raw conversation "
    "history, which lets the policy bypass memory tools by reading earlier "
    "observations/actions. This is allowed only for explicitly marked "
    "diagnostic smoke runs. Set agentgym.allow_raw_history_for_agentmemory=true "
    "or AGENTMEMORY_ALLOW_RAW_HISTORY=1 if you intentionally want that "
    "diagnostic behavior; do not use it for formal training."
)

CONTINUOUS_AGENT_CONTEXT_POLICY = "policy_authored_compaction"
CONTINUOUS_AGENT_RAW_HISTORY_ERROR = (
    "Native long-horizon SWE-smith training requires policy-authored context "
    "compaction. Raw conversation history is allowed only for an explicitly "
    "marked diagnostic run. Set rollout_context_policy=policy_authored_compaction "
    "for training, or AGENTMEMORY_ALLOW_RAW_HISTORY=1 only when intentionally "
    "running the raw-history diagnostic."
)


def assert_rollout_context_supported(agentgym_config: Any) -> None:
    task_name = str(read_config(agentgym_config, "task_name", "")).lower()
    if task_name == "swesmith":
        if rollout_context_policy(agentgym_config) == CONTINUOUS_AGENT_CONTEXT_POLICY:
            return
        if allow_raw_history_for_agentmemory(agentgym_config):
            return
        raise RuntimeError(CONTINUOUS_AGENT_RAW_HISTORY_ERROR)
    if task_name != "agentmemory":
        return
    if rollout_context_policy(agentgym_config) == CONTINUOUS_AGENT_CONTEXT_POLICY:
        if allow_policy_authored_compaction_for_agentmemory(agentgym_config):
            return
        raise RuntimeError(
            "AgentMemory policy-authored compaction is diagnostic-only until "
            "allow_policy_authored_compaction_for_agentmemory=true is set."
        )
    if rollout_context_policy(agentgym_config) == "latest_observation_only":
        return
    if allow_raw_history_for_agentmemory(agentgym_config):
        return
    raise RuntimeError(AGENTMEMORY_RAW_HISTORY_ERROR)


def rollout_context_policy(agentgym_config: Any) -> str:
    task_name = str(read_config(agentgym_config, "task_name", "")).lower()
    policy = str(read_config(agentgym_config, "rollout_context_policy", "")).strip().lower()
    if policy:
        return policy
    if task_name == "agentmemory" and allow_raw_history_for_agentmemory(agentgym_config):
        return "raw_history"
    if task_name == "agentmemory":
        return "latest_observation_only"
    return "raw_history"


def allow_raw_history_for_agentmemory(agentgym_config: Any) -> bool:
    env_value = os.environ.get("AGENTMEMORY_ALLOW_RAW_HISTORY")
    if env_value is not None:
        return parse_bool(env_value)
    return parse_bool(read_config(agentgym_config, "allow_raw_history_for_agentmemory", False))


def allow_policy_authored_compaction_for_agentmemory(agentgym_config: Any) -> bool:
    """Gate WebShop compaction so it cannot silently become formal training."""

    env_value = os.environ.get("AGENTMEMORY_ALLOW_POLICY_AUTHORED_COMPACTION")
    if env_value is not None:
        return parse_bool(env_value)
    return parse_bool(
        read_config(
            agentgym_config,
            "allow_policy_authored_compaction_for_agentmemory",
            False,
        )
    )


def read_config(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")
