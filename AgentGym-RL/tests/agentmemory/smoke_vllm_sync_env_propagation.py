#!/usr/bin/env python3
"""Pure propagation smoke for main process -> Ray main_task -> worker actor."""

import importlib.util
import os
from pathlib import Path
from unittest.mock import patch


def load_module(name, override_env, default_module):
    override = os.environ.get(override_env)
    if not override:
        return __import__(default_module, fromlist=["*"])
    spec = importlib.util.spec_from_file_location(name, override)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main_ppo = load_module(
    "staged_agentmemory_main_ppo",
    "AMG_STAGED_MAIN_PPO",
    "verl.agent_trainer.main_ppo",
)
ray_base = load_module(
    "staged_agentmemory_ray_base",
    "AMG_STAGED_RAY_BASE",
    "verl.single_controller.ray.base",
)


SYNC_DIR = "VERL_AGENTMEMORY_VLLM_SYNC_EVIDENCE_DIR"
REQUIRE_CHANGE = "VERL_AGENTMEMORY_REQUIRE_VLLM_POST_UPDATE_CHANGE"
EXPECTED = {
    SYNC_DIR: "/tmp/amg-sync-evidence-propagation-smoke",
    REQUIRE_CHANGE: "1",
}


def expect_runtime_error(call, contains):
    try:
        call()
    except RuntimeError as error:
        assert contains in str(error), error
    else:
        raise AssertionError(f"expected RuntimeError containing {contains!r}")


def main():
    shell_env = {
        **EXPECTED,
        "AGENTMEMORY_BUY_SEMANTICS": "terminate",
        "VERL_AGENTMEMORY_UNKNOWN_MUST_NOT_PROPAGATE": "forbidden",
    }
    with patch.dict(os.environ, shell_env, clear=True):
        ray_runtime_env = main_ppo._ray_runtime_env_vars()

    # This is the env_vars payload supplied to ray.init(runtime_env=...).
    for key, value in EXPECTED.items():
        assert ray_runtime_env[key] == value
    assert "VERL_AGENTMEMORY_UNKNOWN_MUST_NOT_PROPAGATE" not in ray_runtime_env
    assert ray_runtime_env["AGENTMEMORY_BUY_SEMANTICS"] == "terminate"

    # Simulate main_task's process environment, then build the exact env_vars
    # payload passed to every Ray actor via RayClassWithInitArgs.update_options.
    with patch.dict(os.environ, ray_runtime_env, clear=True):
        worker_runtime_env = ray_base._agentmemory_worker_runtime_env_vars()
    for key, value in EXPECTED.items():
        assert worker_runtime_env[key] == value
    assert worker_runtime_env["AGENTMEMORY_BUY_SEMANTICS"] == "terminate"

    source_path = Path(ray_base.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    helper_call = source.index("env_vars.update(_agentmemory_worker_runtime_env_vars())")
    actor_options = source.index("ray_cls_with_init.update_options", helper_call)
    assert helper_call < actor_options

    with patch.dict(os.environ, {REQUIRE_CHANGE: "1"}, clear=True):
        expect_runtime_error(main_ppo._ray_runtime_env_vars, SYNC_DIR)
        expect_runtime_error(ray_base._agentmemory_worker_runtime_env_vars, SYNC_DIR)
    with patch.dict(
        os.environ,
        {REQUIRE_CHANGE: "1", SYNC_DIR: "relative/path"},
        clear=True,
    ):
        expect_runtime_error(main_ppo._ray_runtime_env_vars, "absolute")
        expect_runtime_error(ray_base._agentmemory_worker_runtime_env_vars, "absolute")
    with patch.dict(
        os.environ,
        {REQUIRE_CHANGE: "maybe", SYNC_DIR: EXPECTED[SYNC_DIR]},
        clear=True,
    ):
        expect_runtime_error(main_ppo._ray_runtime_env_vars, "must be 0 or 1")
        expect_runtime_error(ray_base._agentmemory_worker_runtime_env_vars, "must be 0 or 1")

    print("vLLM sync env propagation smoke: PASS")


if __name__ == "__main__":
    main()
