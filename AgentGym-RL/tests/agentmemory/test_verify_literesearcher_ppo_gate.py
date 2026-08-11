#!/usr/bin/env python3

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "agentmemory" / "verify_literesearcher_ppo_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_literesearcher_ppo_gate", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load LiteResearcher PPO verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(*, row_order: int = 0, terminal: bool = True) -> dict:
    task_round = row_order + 1
    reward = 1.0 if terminal else 0.0
    return {
        "schema_version": "task_neutral_policy_step_v1",
        "item_id": "literesearcher_0",
        "parent_index": 0,
        "parent_group_uid": "parent-0",
        "exact_state_uid": f"state-{row_order}",
        "replica_index": 0,
        "trajectory_uid": "trajectory-0",
        "trajectory_row_uid": f"trajectory-0:row{row_order}",
        "trajectory_row_order": row_order,
        "trajectory_terminal": terminal,
        "task_round": task_round,
        "action": f"<answer>answer-{row_order}</answer>",
        "score": reward,
        "immediate_reward": reward,
        "suffix_return": reward,
        "suffix_credit_applied": False,
        "trajectory_return": 1.0,
        "response_token_ids": [101 + row_order, 102 + row_order],
        "response_token_count": 2,
        "generation_response_length": 2,
        "packed_response_length": 2,
        "generation_prompt_length": 4 + row_order,
        "generation_prompt_digest": f"generation-{row_order}",
        "packed_prompt_length": 4 + row_order,
        "packed_prompt_digest": f"packed-{row_order}",
        "generation_token_ids_are_exact": True,
        "backend_token_ids_are_exact": True,
        "env_info_after": {"data_idx": 0},
    }


def ppo_row(value: dict) -> dict:
    return {
        "task_round": value["task_round"],
        "agentmemory_parent_group_uid": value["parent_group_uid"],
        "agentmemory_exact_state_uid": value["exact_state_uid"],
        "agentmemory_replica_index": value["replica_index"],
        "agentmemory_trajectory_uid": value["trajectory_uid"],
        "agentmemory_trajectory_row_uid": value["trajectory_row_uid"],
        "agentmemory_trajectory_row_order": value["trajectory_row_order"],
        "agentmemory_trajectory_terminal": value["trajectory_terminal"],
        "agentmemory_trajectory_return": value["trajectory_return"],
        "agentmemory_immediate_reward": value["immediate_reward"],
        "agentmemory_suffix_return": value["suffix_return"],
        "agentmemory_suffix_credit_applied": value["suffix_credit_applied"],
        "agentmemory_action_text": value["action"],
        "agentmemory_generation_prompt_length": value["generation_prompt_length"],
        "agentmemory_generation_prompt_digest": value["generation_prompt_digest"],
        "agentmemory_packed_prompt_length": value["packed_prompt_length"],
        "agentmemory_packed_prompt_digest": value["packed_prompt_digest"],
        "response_mask_sum": len(value["response_token_ids"]),
        "score_sum": value["immediate_reward"],
        "old_logprob_mean": -0.25,
    }


def readback_role() -> dict:
    return {
        "summary": {"max_abs_delta": 0.1},
        "parameter_delta_l2": 0.2,
        "parameter_probe": {"parameter_probe_changed_count": 3},
    }


class LiteResearcherPpoGateTests(unittest.TestCase):
    def test_row_binding_accepts_exact_canonical_record(self) -> None:
        module = load_module()
        value = record()
        module.verify_row_binding(ppo_row(value), value, parent_index=0)

    def test_row_binding_rejects_each_ppo_identity_or_credit_drift(self) -> None:
        module = load_module()
        value = record()
        mutations = {
            "agentmemory_action_text": "<answer>different</answer>",
            "agentmemory_generation_prompt_length": 99,
            "agentmemory_generation_prompt_digest": "different-generation",
            "agentmemory_packed_prompt_length": 99,
            "agentmemory_packed_prompt_digest": "different-packed",
            "agentmemory_trajectory_row_order": 9,
            "agentmemory_trajectory_terminal": False,
            "agentmemory_immediate_reward": 0.0,
            "agentmemory_trajectory_return": 0.0,
        }
        for key, replacement in mutations.items():
            row = ppo_row(value)
            row[key] = replacement
            with self.subTest(key=key), self.assertRaises(AssertionError):
                module.verify_row_binding(row, value, parent_index=0)

    def test_trajectory_contract_requires_contiguous_order_and_exact_return(self) -> None:
        module = load_module()
        first = record(row_order=0, terminal=False)
        second = record(row_order=1, terminal=True)
        ordered = module.verify_trajectory_records(0, [second, first])
        self.assertEqual([value["task_round"] for value in ordered], [1, 2])

        broken = deepcopy(second)
        broken["trajectory_return"] = 2.0
        with self.assertRaises(AssertionError):
            module.verify_trajectory_records(0, [first, broken])

    def test_readback_requires_the_same_ordered_formal_rows(self) -> None:
        module = load_module()
        first = record(row_order=0, terminal=False)
        second = record(row_order=1, terminal=True)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            diagnostics = run_dir / "diagnostics"
            diagnostics.mkdir()
            path = diagnostics / "formal_update_readback_step1.json"

            def write(rows: list[dict]) -> None:
                path.write_text(
                    json.dumps(
                        {
                            "global_step": 1,
                            "role": "same_batch_post_optimizer_readback",
                            "row_evidence": {
                                "schema": "agentmemory_formal_step_records_v1",
                                "task_name": "literesearcher",
                                "rows": rows,
                            },
                            "formal_step_records": rows,
                            "actor": readback_role(),
                            "critic": readback_role(),
                        }
                    ),
                    encoding="utf-8",
                )

            write([first, second])
            result = module.verify_readback(
                run_dir,
                global_step=1,
                expected_indices={0},
                expected_records=[first, second],
            )
            self.assertEqual(result["row_count"], 2)

            write([second, first])
            with self.assertRaises(AssertionError):
                module.verify_readback(
                    run_dir,
                    global_step=1,
                    expected_indices={0},
                    expected_records=[first, second],
                )


if __name__ == "__main__":
    unittest.main()
