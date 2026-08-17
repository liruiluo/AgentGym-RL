#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
import uuid
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
            "_agentmemory_atomic_json_dump",
            "_agentmemory_env_flag",
            "_agentmemory_formal_update_readback_target_steps",
        }
    ]
    if len(selected) != 3:
        raise AssertionError("formal readback helper set drifted")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"json": json, "os": os, "uuid": uuid}
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace


class FormalUpdateReadbackStepsTests(unittest.TestCase):
    def test_json_artifact_is_atomically_replaced(self) -> None:
        helper = load_helpers()["_agentmemory_atomic_json_dump"]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "readback.json"
            output.write_text('{"version": 1}\n', encoding="utf-8")

            helper({"version": 2, "rows": [1, 2, 3]}, output)

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"version": 2, "rows": [1, 2, 3]},
            )
            self.assertEqual(list(output.parent.glob("readback.json.tmp.*")), [])

    def test_failed_json_write_keeps_previous_complete_artifact(self) -> None:
        helper = load_helpers()["_agentmemory_atomic_json_dump"]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "readback.json"
            output.write_text('{"version": 1}\n', encoding="utf-8")

            def fail_after_partial_write(payload, handle, **kwargs):
                del payload, kwargs
                handle.write('{"version":')
                raise RuntimeError("simulated interrupted write")

            with mock.patch.object(json, "dump", side_effect=fail_after_partial_write):
                with self.assertRaisesRegex(RuntimeError, "interrupted write"):
                    helper({"version": 2}, output)

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"version": 1},
            )
            self.assertEqual(list(output.parent.glob("readback.json.tmp.*")), [])

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
            "_agentmemory_missing_formal_update_readback_steps(",
            "readback target steps completed without",
        ):
            self.assertIn(fragment, source)
        tree = ast.parse(source)
        published_payloads = {
            node.args[0].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_agentmemory_atomic_json_dump"
            and node.args
            and isinstance(node.args[0], ast.Name)
        }
        self.assertEqual(published_payloads, {"summary", "payload"})


if __name__ == "__main__":
    unittest.main()
