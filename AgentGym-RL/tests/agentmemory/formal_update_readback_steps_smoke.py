#!/usr/bin/env python3

import ast
import os
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "verl" / "agent_trainer" / "ppo" / "ray_trainer.py"


def load_helpers():
    source = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {
            "_agentmemory_env_flag",
            "_agentmemory_formal_update_readback_target_steps",
        }
    ]
    if len(selected) != 2:
        raise AssertionError("formal readback helper set drifted")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"os": os}
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace


class FormalUpdateReadbackStepsTests(unittest.TestCase):
    def test_two_steps_are_parsed_exactly(self) -> None:
        helper = load_helpers()["_agentmemory_formal_update_readback_target_steps"]
        with mock.patch.dict(
            os.environ,
            {
                "AGENTMEMORY_FORMAL_UPDATE_READBACK": "1",
                "AGENTMEMORY_FORMAL_UPDATE_READBACK_STEP": "1,2",
            },
            clear=True,
        ):
            self.assertEqual(helper(), frozenset({1, 2}))

    def test_disabled_returns_none(self) -> None:
        helper = load_helpers()["_agentmemory_formal_update_readback_target_steps"]
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(helper())

    def test_invalid_or_missing_steps_fail_closed(self) -> None:
        helper = load_helpers()["_agentmemory_formal_update_readback_target_steps"]
        for value in (None, "", "0", "-1", "1,nope"):
            with self.subTest(value=value):
                environment = {"AGENTMEMORY_FORMAL_UPDATE_READBACK": "1"}
                if value is not None:
                    environment["AGENTMEMORY_FORMAL_UPDATE_READBACK_STEP"] = value
                with mock.patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(RuntimeError):
                        helper()

    def test_fit_tracks_every_requested_step(self) -> None:
        source = TRAINER.read_text(encoding="utf-8")
        for fragment in (
            "self.global_steps in formal_readback_target_steps",
            "formal_readback_observed_steps.add(self.global_steps)",
            "formal_readback_observed_steps != set(formal_readback_target_steps)",
        ):
            self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
