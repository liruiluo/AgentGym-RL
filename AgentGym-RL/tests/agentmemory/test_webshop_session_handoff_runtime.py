from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from tests.agentmemory.test_formal_domain_v3 import filesystem_contract_metadata
from verl.workers.rollout.agent_vllm_rollout import vllm_rollout as MODULE


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(value) + 3 for value in str(text).encode("utf-8")]

    @staticmethod
    def decode(token_ids, **kwargs) -> str:
        del kwargs
        return bytes(int(value) - 3 for value in token_ids).decode("utf-8")


class FakeWebShopClient:
    def __init__(self) -> None:
        self.env_id = 101
        self.metadata = filesystem_contract_metadata()
        self.session_index = 0
        self.native_calls: list[str] = []
        self.info = {"metadata": self.metadata, "env_info": {}}

    def _set_info(self, *, tool_ops: list[dict]) -> None:
        self.info = {
            "metadata": self.metadata,
            "env_info": {
                "current_subtask_index": self.session_index,
                "session_trace": [],
                "tool_ops": tool_ops,
            },
        }

    def reset(self, index: int) -> None:
        self.reset_index = int(index)
        self.session_index = 0
        self._set_info(tool_ops=[])

    def observe(self) -> str:
        return f"session-{self.session_index} fresh observation"

    def step(self, action: str) -> SimpleNamespace:
        self.native_calls.append(str(action))
        native_step = len(self.native_calls)
        if str(action) != "click[Buy Now]":
            raise AssertionError(f"unexpected fake action: {action!r}")
        self.session_index += 1
        done = self.session_index == 2
        self._set_info(
            tool_ops=[
                {
                    "op": "BUY",
                    "step": native_step,
                    "committed": True,
                    "purchase_correct": True,
                    "session_advanced": True,
                    "terminal": done,
                }
            ]
        )
        return SimpleNamespace(reward=1.0, done=done, state=self.observe())


class WebShopSessionHandoffRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(
            prefix="amg-webshop-handoff-test-"
        )
        self.addCleanup(self.tempdir.cleanup)
        self.rollout = object.__new__(MODULE.vLLMRollout)
        self.rollout.agentgym_config = {
            "task_name": "agentmemory",
            "rollout_context_policy": "policy_authored_compaction",
            "allow_policy_authored_compaction_for_agentmemory": True,
            "max_observation_tokens": 256,
            "continuous_timeout_penalty": -0.01,
        }
        self.rollout.config = SimpleNamespace(
            n=1,
            response_length=64,
            prompt_length=512,
            max_model_len=576,
            send_interval=0,
            rollout_log_dir=self.tempdir.name,
        )
        self.rollout.sampling_params = SimpleNamespace(max_tokens=64)
        self.rollout.pad_token_id = 0
        self.rollout.tokenizer = FakeTokenizer()
        self.generated_contents = iter(
            ["click[Buy Now]", "notes/state.md", "click[Buy Now]"]
        )
        self.seen_prompt_texts: list[str] = []
        self.current_records: list[dict] = []

        def prompt_from_messages(_self, messages):
            text = "\n".join(
                f"{message.role}:{message.content}" for message in messages
            )
            return FakeTokenizer.encode(text)

        def generate(_self, generation_prompt_idxs, sampling_params):
            del sampling_params
            self.seen_prompt_texts.extend(
                FakeTokenizer.decode(prompt) for prompt in generation_prompt_idxs
            )
            content = next(self.generated_contents)
            token_ids = FakeTokenizer.encode(content)
            self.current_records = [
                {
                    "token_ids": token_ids,
                    "sampled_token_logprobs": [-0.1] * len(token_ids),
                    "response_token_count": len(token_ids),
                    "max_response_tokens": 64,
                    "finish_reason": "stop",
                    "finish_reason_source": "official_vllm:backend",
                    "stop_reason": None,
                    "backend_source": "official_vllm",
                    "configured_eos_token_ids": [1],
                    "primary_eos_token_id": 1,
                    "tokenizer_pad_token_id": 0,
                    "token_ids_are_exact": True,
                    "backend_token_ids_are_exact": True,
                    "truncated": False,
                }
            ]
            return object()

        def output_records(_self, output, sampling_params, **kwargs):
            del output, sampling_params, kwargs
            return list(self.current_records)

        @contextmanager
        def update_sampling_params(_self, **kwargs):
            del kwargs
            yield

        self.rollout._continuous_prompt_from_messages = MethodType(
            prompt_from_messages, self.rollout
        )
        self.rollout._generate_token_ids = MethodType(generate, self.rollout)
        self.rollout._output_generation_records = MethodType(
            output_records, self.rollout
        )
        self.rollout.update_sampling_params = MethodType(
            update_sampling_params, self.rollout
        )

    def test_native_reset_handoff_and_next_buy_are_packed_exactly(self) -> None:
        client = FakeWebShopClient()
        handler = SimpleNamespace(
            item_id="webshop-0",
            data_idx=0,
            agentmemory_local_data_idx=0,
            parent_index=0,
            rollout_replica_index=0,
            done=False,
            score=0.0,
            messages=[],
        )
        with patch.object(torch.distributed, "get_rank", return_value=0), patch.object(
            MODULE,
            "_formal_runtime_contract_for_client",
            return_value=(MODULE.FORMAL_WEBSHOP_SCHEMA_V2, "fake system", "fake"),
        ), patch.object(MODULE, "_validate_runtime_env_schema"):
            output = self.rollout.generate_agentmemory_webshop_session_handoff(
                rollout_handler_ls=[handler],
                env_clients=[client],
                cur_device=torch.device("cpu"),
                max_policy_turns=4,
                sampling_kwargs={"max_tokens": 64},
                global_steps=None,
            )

        self.assertEqual(client.native_calls, ["click[Buy Now]", "click[Buy Now]"])
        self.assertEqual(len(self.seen_prompt_texts), 3)
        self.assertIn("session-0 fresh observation", self.seen_prompt_texts[0])
        self.assertIn(
            MODULE.POLICY_WEBSHOP_SESSION_HANDOFF_REQUEST,
            self.seen_prompt_texts[1],
        )
        self.assertIn("session-1 fresh observation", self.seen_prompt_texts[2])
        self.assertIn("notes/state.md", self.seen_prompt_texts[2])
        self.assertNotIn("session-0 fresh observation", self.seen_prompt_texts[2])
        self.assertNotIn("click[Buy Now]", self.seen_prompt_texts[2])

        records = [
            json.loads(value)
            for value in output.non_tensor_batch[MODULE.AGENTMEMORY_STEP_RECORD_JSON]
        ]
        self.assertEqual(
            [record["row_kind"] for record in records],
            [
                MODULE.ENVIRONMENT_ACTION_ROW,
                MODULE.COMPACTION_ROW,
                MODULE.ENVIRONMENT_ACTION_ROW,
            ],
        )
        self.assertEqual(
            [
                (
                    record["native_environment_call_count_before"],
                    record["native_environment_call_count_after"],
                )
                for record in records
            ],
            [(0, 1), (1, 1), (1, 2)],
        )
        self.assertEqual(
            [
                (record["environment_step_before"], record["environment_step_after"])
                for record in records
            ],
            [(0, 1), (1, 2), (2, 3)],
        )
        self.assertEqual(records[1]["action"], "notes/state.md")
        self.assertEqual(records[1]["compaction"]["session_index_before"], 0)
        self.assertEqual(records[1]["compaction"]["session_index_after"], 1)
        self.assertTrue(records[1]["compaction"]["raw_history_cleared"])
        self.assertEqual(records[2]["buy_evidence"]["buy_record"]["step"], 2)
        self.assertEqual(
            list(output.batch[MODULE.AGENTMEMORY_TRAJECTORY_ROW_ORDER].cpu().tolist()),
            [0, 1, 2],
        )


if __name__ == "__main__":
    unittest.main()
