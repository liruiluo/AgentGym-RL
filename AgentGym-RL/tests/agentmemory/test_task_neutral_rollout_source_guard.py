import ast
import unittest
from pathlib import Path


ROLLOUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py"
)


class TaskNeutralRolloutSourceGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = ROLLOUT_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_shared_rollout_contains_no_domain_specific_surface(self) -> None:
        lowered = self.source.lower()
        for forbidden in (
            "literesearcher",
            "swesmith",
            "swe-smith",
            "webshop",
            "openmle",
            "native_server",
            "locator",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, lowered)

    def test_generate_sequences_has_one_task_neutral_entrypoint(self) -> None:
        generation_methods = {
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("generate_")
        }
        self.assertEqual(
            generation_methods,
            {"generate_sequences", "generate_task_neutral_policy"},
        )
        generate_sequences = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "generate_sequences"
        )
        task_neutral_calls = [
            node
            for node in ast.walk(generate_sequences)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate_task_neutral_policy"
        ]
        self.assertEqual(len(task_neutral_calls), 1)


if __name__ == "__main__":
    unittest.main()
