#!/usr/bin/env python3

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "verl" / "agent_trainer" / "ppo" / "ray_trainer.py"


def load_helper():
    source = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_formal_update_readback_row_evidence"
    ]
    if len(selected) != 1:
        raise AssertionError("formal readback row helper drifted")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"json": json}
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace["_formal_update_readback_row_evidence"]


class FormalUpdateReadbackTaskRowsTests(unittest.TestCase):
    def test_agentmemory_keeps_canonical_step_records(self) -> None:
        helper = load_helper()
        evidence = helper(
            non_tensor_batch={
                "agentmemory_step_record_json": [
                    json.dumps({"row": 0}),
                    json.dumps({"row": 1}),
                ]
            },
            valid_row_indices=[1],
            task_name="agentmemory",
        )
        self.assertEqual(evidence["schema"], "agentmemory_formal_step_records_v1")
        self.assertEqual(evidence["rows"], [{"row": 1}])

    def test_agentmemory_missing_step_records_fails_closed(self) -> None:
        helper = load_helper()
        with self.assertRaisesRegex(RuntimeError, "canonical step records"):
            helper(
                non_tensor_batch={"index": [0]},
                valid_row_indices=[0],
                task_name="agentmemory",
            )

    def test_swesmith_uses_dataset_indices(self) -> None:
        helper = load_helper()
        evidence = helper(
            non_tensor_batch={"index": [7, 11, 13]},
            valid_row_indices=[0, 2],
            task_name="swesmith",
        )
        self.assertEqual(evidence["schema"], "generic_task_dataset_rows_v1")
        self.assertEqual(evidence["task_name"], "swesmith")
        self.assertEqual(evidence["index_field"], "index")
        self.assertEqual(evidence["dataset_indices"], [7, 13])

    def test_generic_task_requires_valid_dataset_indices(self) -> None:
        helper = load_helper()
        with self.assertRaisesRegex(RuntimeError, "canonical dataset index"):
            helper(
                non_tensor_batch={},
                valid_row_indices=[0],
                task_name="swesmith",
            )
        for value in (True, -1, "not-an-index"):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    helper(
                        non_tensor_batch={"index": [value]},
                        valid_row_indices=[0],
                        task_name="swesmith",
                    )


if __name__ == "__main__":
    unittest.main()
