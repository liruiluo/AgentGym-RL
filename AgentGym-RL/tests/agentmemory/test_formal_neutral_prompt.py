from __future__ import annotations

import ast
import unittest
from pathlib import Path


SCHEMAS_PATH = Path(__file__).resolve().parents[2] / "verl" / "workers" / "rollout" / "schemas.py"


def extract_static_string_assignments() -> dict[str, str]:
    tree = ast.parse(SCHEMAS_PATH.read_text(encoding="utf-8"))
    values: dict[str, str] = {}

    def resolve(node: ast.expr) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return resolve(node.left) + resolve(node.right)
        raise ValueError(f"unsupported static string expression: {ast.dump(node)}")

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = resolve(node.value)
        except ValueError:
            continue
    return values


class FormalPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        values = extract_static_string_assignments()
        self.no_thinking_prompt = values["AGENTMEMORY_ACTION_SYSTEM_PROMPT"]
        self.thinking_prompt = values["AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING"]
        self.reasoning_prompt = values["AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING"]

    def test_both_prompts_have_native_action_contract(self) -> None:
        for prompt in (self.no_thinking_prompt, self.thinking_prompt, self.reasoning_prompt):
            for fragment in (
                "native WebShop bundled-shopping environment",
                "search[keywords]",
                "click[value]",
                "click[Buy Now]",
                "ADD requires key:string",
                "RETRIEVE requires query:string and top_k=3",
                "Current-session trace clears",
                "Long-term memory persists across shopping sessions",
            ):
                self.assertIn(fragment, prompt)

    def test_both_prompts_have_explicit_memory_lifecycle(self) -> None:
        for prompt in (self.no_thinking_prompt, self.thinking_prompt, self.reasoning_prompt):
            for fragment in (
                "use ADD before click[Buy Now]",
                "At the start of every later shopping session",
                "use RETRIEVE",
                "does not reject an otherwise correct purchase when ADD was skipped",
            ):
                self.assertIn(fragment, prompt)

    def test_reply_rules_match_thinking_mode(self) -> None:
        self.assertIn("Output excludes", self.no_thinking_prompt)
        self.assertIn("<think> blocks", self.no_thinking_prompt)
        self.assertIn("You may first reason inside a single <think>", self.thinking_prompt)
        self.assertIn("After the closing </think>", self.thinking_prompt)
        self.assertIn("Write `Thought:` followed by brief free-form reasoning", self.reasoning_prompt)
        self.assertIn("after the final `Action:` label", self.reasoning_prompt)
        self.assertIn("PPO trains the complete sampled Thought-and-Action response", self.reasoning_prompt)
        self.assertIn("Output excludes markdown and <think> blocks", self.reasoning_prompt)

    def test_prompts_contain_no_removed_action(self) -> None:
        for prompt in (self.no_thinking_prompt, self.thinking_prompt, self.reasoning_prompt):
            for forbidden in (
                "GROUND",
                'SEARCH {"query"',
                'BUY {"product_id"',
                "PAGE accepts",
                "try a different candidate",
            ):
                self.assertNotIn(forbidden, prompt)


if __name__ == "__main__":
    unittest.main()
