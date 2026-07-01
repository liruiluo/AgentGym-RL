from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys


class FakeTokenizer:
    def apply_chat_template(self, conversations, add_generation_prompt=True, tokenize=True):
        assert tokenize
        rendered = "|".join(f"{item['role']}:{item['content']}" for item in conversations)
        if add_generation_prompt:
            rendered += "|assistant:"
        return [ord(ch) for ch in rendered]


def load_schemas_module():
    transformers_module = ModuleType("transformers")
    transformers_module.PreTrainedTokenizer = object
    sys.modules.setdefault("transformers", transformers_module)

    torch_module = ModuleType("torch")
    torch_module.Tensor = object
    sys.modules.setdefault("torch", torch_module)

    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "AgentGym-RL" / "verl" / "workers" / "rollout" / "schemas.py"
    spec = spec_from_file_location("agentmemory_rollout_schema_smoke", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_schemas_module()
    handler = module.RolloutHandler(
        messages=[
            module.Message(role="user", content="old observation contains TV size 75"),
            module.Message(role="assistant", content='ADD {"key":"tv","value":"75 inch"}'),
            module.Message(role="user", content="current observation asks for compatible mount"),
        ],
        task_name="agentmemory",
        item_id=0,
        score=0,
        done=False,
        input_ids=[1],
        prompt_ids=[1],
        response_ids=[],
        attention_mask=[1],
        prompt_attention_mask=[1],
        response_attention_mask=[],
        position_ids=[0],
        prompt_position_ids=[0],
        response_position_ids=[],
        loss_mask=[0],
        prompt_loss_mask=[0],
        response_loss_mask=[],
    )
    prompt = "".join(chr(token_id) for token_id in handler.get_latest_observation_prompt(FakeTokenizer()))
    assert "current observation asks for compatible mount" in prompt
    assert "old observation contains TV size 75" not in prompt
    assert 'ADD {"key":"tv","value":"75 inch"}' not in prompt
    print("AGENTMEMORY_LATEST_OBSERVATION_PROMPT_SMOKE_OK")


if __name__ == "__main__":
    main()
