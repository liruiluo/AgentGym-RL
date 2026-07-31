from __future__ import annotations

import ast
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_PATH = _REPO_ROOT / "verl/workers/agent_fsdp_workers.py"
_CONFIG_PATH = _REPO_ROOT / "verl/agent_trainer/config/ppo_trainer.yaml"


def _forward_prefetch_expressions() -> list[ast.expr]:
    tree = ast.parse(_WORKER_PATH.read_text(encoding="utf-8"))
    expressions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "forward_prefetch":
                expressions.append(keyword.value)
    return expressions


class FsdpForwardPrefetchTests(unittest.TestCase):
    def test_actor_and_critic_fsdp_constructors_read_the_config(self):
        expressions = _forward_prefetch_expressions()
        self.assertEqual(len(expressions), 2)

        for expression in expressions:
            compiled = compile(ast.Expression(expression), str(_WORKER_PATH), "eval")
            self.assertIs(eval(compiled, {"fsdp_config": {}}), False)
            self.assertIs(
                eval(compiled, {"fsdp_config": {"forward_prefetch": True}}),
                True,
            )
            self.assertIs(
                eval(compiled, {"fsdp_config": {"forward_prefetch": 0}}),
                False,
            )

    def test_trainer_config_keeps_all_fsdp_roles_disabled_by_default(self):
        config = _CONFIG_PATH.read_text(encoding="utf-8")
        self.assertEqual(config.count("forward_prefetch: False"), 3)


if __name__ == "__main__":
    unittest.main()
