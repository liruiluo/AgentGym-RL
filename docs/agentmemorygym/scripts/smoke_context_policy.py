from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_context_policy_module():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "AgentGym-RL" / "verl" / "utils" / "agentgym" / "context_policy.py"
    spec = spec_from_file_location("agentmemory_context_policy_smoke", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


context_policy_module = load_context_policy_module()
assert_rollout_context_supported = context_policy_module.assert_rollout_context_supported
rollout_context_policy = context_policy_module.rollout_context_policy


def expect_blocked() -> None:
    try:
        assert_rollout_context_supported({"task_name": "agentmemory", "rollout_context_policy": "raw_history"})
    except RuntimeError as exc:
        assert "full raw conversation history" in str(exc)
        return
    raise AssertionError("AgentMemory raw-history rollout should be blocked by default.")


def expect_allowed_by_config() -> None:
    assert rollout_context_policy(
        {"task_name": "agentmemory", "allow_raw_history_for_agentmemory": True}
    ) == "raw_history"
    assert_rollout_context_supported(
        {"task_name": "agentmemory", "allow_raw_history_for_agentmemory": True}
    )


def expect_allowed_by_env() -> None:
    old_value = os.environ.get("AGENTMEMORY_ALLOW_RAW_HISTORY")
    os.environ["AGENTMEMORY_ALLOW_RAW_HISTORY"] = "1"
    try:
        assert_rollout_context_supported({"task_name": "agentmemory"})
    finally:
        if old_value is None:
            os.environ.pop("AGENTMEMORY_ALLOW_RAW_HISTORY", None)
        else:
            os.environ["AGENTMEMORY_ALLOW_RAW_HISTORY"] = old_value


def main() -> None:
    assert_rollout_context_supported({"task_name": "webshop"})
    assert rollout_context_policy({"task_name": "agentmemory"}) == "latest_observation_only"
    assert_rollout_context_supported({"task_name": "agentmemory"})
    expect_blocked()
    expect_allowed_by_config()
    expect_allowed_by_env()
    print("AGENTMEMORY_CONTEXT_POLICY_SMOKE_OK")


if __name__ == "__main__":
    main()
