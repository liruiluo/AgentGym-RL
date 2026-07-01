from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any


class ActionFormat(Enum):
    REACT = "react"
    FUNCTION_CALLING = "function_calling"
    CODE_AS_ACTION = "code_as_action"


@dataclass
class ActionWithTought:
    thought: str
    action: str


@dataclass
class StepOutput:
    state: str
    reward: float
    done: bool


class BaseAdapter:
    action_parser = staticmethod(lambda text, action_format: text)


class BaseEnvClient:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.action_format = ActionFormat.REACT


class BaseTask:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs


def install_agentenv_controller_stubs() -> None:
    agentenv_module = ModuleType("agentenv")
    controller_module = ModuleType("agentenv.controller")
    types_module = ModuleType("agentenv.controller.types")

    controller_module.BaseAdapter = BaseAdapter
    controller_module.BaseEnvClient = BaseEnvClient
    controller_module.BaseTask = BaseTask
    controller_module.extract_python_code_blocks = lambda text: text
    controller_module.format_code_as_action_prompt = lambda functions: str(functions)
    controller_module.format_function_call_prompt = lambda functions: str(functions)
    controller_module.parse_python_code_comments = lambda code: ""

    types_module.ActionFormat = ActionFormat
    types_module.ActionWithTought = ActionWithTought
    types_module.ConversationMessage = dict
    types_module.StepOutput = StepOutput

    sys.modules["agentenv"] = agentenv_module
    sys.modules["agentenv.controller"] = controller_module
    sys.modules["agentenv.controller.types"] = types_module


def load_agentmemory_module() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "AgentGym" / "agentenv" / "agentenv" / "envs" / "agentmemory.py"
    spec = spec_from_file_location("agentmemory_client_smoke", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()

    install_agentenv_controller_stubs()
    module = load_agentmemory_module()
    client = module.AgentMemoryEnvClient(env_server_base=args.base_url, data_len=None)
    assert len(client) == 3, len(client)
    assert client.metadata["task_count"] == 3, client.metadata
    reset = client.reset(2)
    assert reset["info"]["task_id"] == "monitor_bundle_27", reset
    assert reset["info"]["split"] == "test", reset
    client.close()
    print("AGENTMEMORY_CLIENT_METADATA_SMOKE_OK", len(client), reset["info"]["task_id"])


if __name__ == "__main__":
    main()
