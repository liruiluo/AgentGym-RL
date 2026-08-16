#!/usr/bin/env python3

import ast
import hashlib
import json
import math
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
    namespace = {"hashlib": hashlib, "json": json, "math": math}
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace["_formal_update_readback_row_evidence"]


class FormalUpdateReadbackTaskRowsTests(unittest.TestCase):
    def test_agentmemory_keeps_canonical_step_records(self) -> None:
        helper = load_helper()
        first = {
            "schema_version": "task_neutral_policy_step_v1",
            "response_token_ids": [101, 102],
            "response_token_count": 2,
            "generation_token_ids_are_exact": True,
            "backend_token_ids_are_exact": True,
        }
        second = {
            "schema_version": "task_neutral_policy_step_v1",
            "response_token_ids": [201],
            "response_token_count": 1,
            "generation_token_ids_are_exact": True,
            "backend_token_ids_are_exact": True,
        }
        evidence = helper(
            non_tensor_batch={
                "agentmemory_step_record_json": [
                    json.dumps(first),
                    json.dumps(second),
                ]
            },
            valid_row_indices=[1],
            task_name="agentmemory",
            response_token_rows=[[101, 102, 0], [201, 0, 0]],
            response_mask_rows=[[1, 1, 0], [1, 0, 0]],
            old_logprob_rows=[[-0.1, -0.2, 0.0], [-0.3, 0.0, 0.0]],
        )
        self.assertEqual(evidence["schema"], "agentmemory_formal_step_records_v1")
        row = evidence["rows"][0]
        self.assertEqual(row["response_token_ids"], [201])
        self.assertEqual(row["sampled_response_token_ids"], [201])
        self.assertEqual(row["packed_token_ids"], [201, 0, 0])
        self.assertEqual(row["response_mask"], [1, 0, 0])
        self.assertEqual(row["sampled_old_logprobs"], [-0.3])
        self.assertEqual(row["packed_old_logprobs"], [-0.3, 0.0, 0.0])
        for field in (
            "sampled_response_token_ids_sha256",
            "packed_token_ids_sha256",
            "response_mask_sha256",
            "sampled_old_logprobs_sha256",
        ):
            self.assertRegex(row[field], r"^[0-9a-f]{64}$")

    def test_agentmemory_missing_step_records_fails_closed(self) -> None:
        helper = load_helper()
        with self.assertRaisesRegex(RuntimeError, "canonical step records"):
            helper(
                non_tensor_batch={"index": [0]},
                valid_row_indices=[0],
                task_name="agentmemory",
                response_token_rows=[[1]],
                response_mask_rows=[[1]],
                old_logprob_rows=[[-0.1]],
            )

    def test_agentmemory_rejects_tensor_token_drift(self) -> None:
        helper = load_helper()
        record = {
            "schema_version": "task_neutral_policy_step_v1",
            "response_token_ids": [7, 8],
            "response_token_count": 2,
            "generation_token_ids_are_exact": True,
            "backend_token_ids_are_exact": True,
        }
        with self.assertRaisesRegex(RuntimeError, "sampled response tokens"):
            helper(
                non_tensor_batch={
                    "agentmemory_step_record_json": [json.dumps(record)]
                },
                valid_row_indices=[0],
                task_name="openmle_fast",
                response_token_rows=[[7, 9]],
                response_mask_rows=[[1, 1]],
                old_logprob_rows=[[-0.1, -0.2]],
            )

    def test_agentmemory_requires_old_logprob_tensor_evidence(self) -> None:
        helper = load_helper()
        record = {
            "schema_version": "task_neutral_policy_step_v1",
            "response_token_ids": [7],
            "response_token_count": 1,
            "generation_token_ids_are_exact": True,
            "backend_token_ids_are_exact": True,
        }
        with self.assertRaisesRegex(RuntimeError, "token/logprob tensor evidence"):
            helper(
                non_tensor_batch={
                    "agentmemory_step_record_json": [json.dumps(record)]
                },
                valid_row_indices=[0],
                task_name="openmle_fast",
                response_token_rows=[[7]],
                response_mask_rows=[[1]],
                old_logprob_rows=None,
            )

    def test_swesmith_uses_dataset_indices(self) -> None:
        helper = load_helper()
        evidence = helper(
            non_tensor_batch={"index": [7, 11, 13]},
            valid_row_indices=[0, 2],
            task_name="swesmith",
            response_token_rows=None,
            response_mask_rows=None,
            old_logprob_rows=None,
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
                response_token_rows=None,
                response_mask_rows=None,
                old_logprob_rows=None,
            )
        for value in (True, -1, "not-an-index"):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    helper(
                        non_tensor_batch={"index": [value]},
                        valid_row_indices=[0],
                        task_name="swesmith",
                        response_token_rows=None,
                        response_mask_rows=None,
                        old_logprob_rows=None,
                    )


if __name__ == "__main__":
    unittest.main()
