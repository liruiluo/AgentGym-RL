from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
AGENTENV_ROOT = ROOT.parent / "AgentGym" / "agentenv"
if str(AGENTENV_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTENV_ROOT))

from agentenv.controller import (  # noqa: E402
    StepOutput,
    bind_initial_policy_context,
    complete_policy_turn,
    prepare_policy_turn,
)
from agentenv.controller.env import BaseEnvClient  # noqa: E402
from agentenv.controller.types import (  # noqa: E402
    ActionFormat,
    build_task_neutral_transition_info,
)
from agentenv.envs.agentmemory import AgentMemoryEnvClient  # noqa: E402
from agentenv.envs.swesmith import (  # noqa: E402
    SWE_CONTEXT_COMPACTION_REQUEST,
    SwesmithEnvClient,
    _validate_actor_credit_receipt,
)
from agentenv.envs.webshop_handoff import (  # noqa: E402
    WEBSHOP_SESSION_HANDOFF_REQUEST,
)


def count_prompt_tokens(messages) -> int:
    return sum(len(message["content"].split()) + 2 for message in messages)


def prepare(client, messages, *, capacity: int = 4096):
    response_budget = 32
    return prepare_policy_turn(
        client,
        messages,
        count_prompt_tokens=count_prompt_tokens,
        max_prompt_tokens=capacity,
        max_model_tokens=capacity + response_budget,
        max_response_tokens=response_budget,
        max_observation_tokens=64,
        action_observation_envelope_tokens=4,
    )


class FakeWebShopClient(AgentMemoryEnvClient):
    def __init__(self, responses: list[dict]) -> None:
        BaseEnvClient.__init__(self, action_format=ActionFormat.REACT)
        self.is_v3 = True
        self.is_filesystem = True
        self.metadata = {"surface": "fake_filesystem_webshop"}
        self.env_id = 101
        self.data_len = 1
        self.last_action_submission = None
        self.info = {
            "observation": "session-0 fresh observation",
            "reward": 0.0,
            "done": False,
            "env_info": {
                "current_subtask_index": 0,
                "tool_ops": [],
                "session_trace": [],
            },
            "metadata": self.metadata,
        }
        self._responses = list(responses)
        self.native_calls: list[str] = []
        self._reset_policy_transition_state(self.info["env_info"])
        self.configure_policy_system_prompt("Canonical WebShop tool contract")

    def __len__(self) -> int:
        return 1

    def post(self, path: str, data: dict) -> dict:
        if path != "step":
            raise AssertionError(f"unexpected fake WebShop path: {path}")
        self.native_calls.append(str(data["action"]))
        if not self._responses:
            raise AssertionError("fake WebShop response queue is empty")
        return deepcopy(self._responses.pop(0))

    def reset(self, idx: int = 0) -> dict:
        del idx
        raise AssertionError("test fixture is already reset")


class FakeSwesmithClient(SwesmithEnvClient):
    def __init__(self) -> None:
        BaseEnvClient.__init__(self, action_format=ActionFormat.REACT)
        self.env_id = 202
        self.data_len = 1
        self.metadata = {"task_count": 1}
        self.info = {
            "observation": "Fix the failing parser in this repository.",
            "reward": 0.0,
            "done": False,
            "info": {"step": 0},
        }
        self.native_calls: list[str] = []
        self._reset_policy_transition_state()

    def __len__(self) -> int:
        return 1

    def _request(self, method: str, path: str, **kwargs) -> dict:
        if (method, path) != ("POST", "step"):
            raise AssertionError(f"unexpected fake SWE request: {method} {path}")
        action = str(kwargs["json"]["action"])
        self.native_calls.append(action)
        step = len(self.native_calls)
        return {
            "observation": f"native tool output {step}",
            "reward": 0.0,
            "done": False,
            "info": {
                "step": step,
                "action_kind": "shell_command",
                "actor_credit": {
                    "schema": "task_neutral_actor_credit_v1",
                    "positive_eligible": True,
                    "basis": "shell_executed",
                },
            },
        }

    def reset(self, idx: int = 0) -> dict:
        del idx
        raise AssertionError("test fixture is already reset")


def webshop_search_response() -> dict:
    return {
        "observation": "session-0 search result",
        "reward": 0.0,
        "done": False,
        "info": {
            "current_subtask_index": 0,
            "tool_ops": [{"op": "SEARCH", "step": 1}],
            "session_trace": ["search result"],
        },
    }


def webshop_buy_response() -> dict:
    return {
        "observation": "session-1 fresh observation",
        "reward": 1.0,
        "done": False,
        "info": {
            "current_subtask_index": 1,
            "tool_ops": [
                {
                    "op": "BUY",
                    "step": 2,
                    "committed": True,
                    "purchase_correct": True,
                    "session_advanced": True,
                    "terminal": False,
                }
            ],
            "session_trace": [],
        },
    }


class SharedWrapperPolicyTurnTest(unittest.TestCase):
    @staticmethod
    def bind_webshop(client: FakeWebShopClient) -> list[dict[str, str]]:
        return bind_initial_policy_context(
            client,
            [
                {
                    "role": "user",
                    "content": "Legacy prompt requiring Thought and Action labels",
                },
                {"role": "assistant", "content": "Ok."},
                {"role": "user", "content": client.observe()},
            ],
        )

    def test_webshop_uses_native_action_then_local_handoff(self) -> None:
        client = FakeWebShopClient(
            [webshop_search_response(), webshop_buy_response()]
        )
        messages = self.bind_webshop(client)
        self.assertEqual(
            messages,
            [
                {
                    "role": "system",
                    "content": "Canonical WebShop tool contract",
                },
                {"role": "user", "content": "session-0 fresh observation"},
            ],
        )

        search = prepare(client, messages)
        self.assertIsNone(search.control_request)
        search_output, messages = complete_policy_turn(
            client, search, "search[black mug]"
        )
        self.assertEqual(client.native_calls, ["search[black mug]"])
        self.assertEqual(
            search_output.info["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertEqual(
            messages,
            [
                {
                    "role": "system",
                    "content": "Canonical WebShop tool contract",
                },
                {"role": "user", "content": "session-0 search result"},
            ],
        )
        self.assertNotIn("search[black mug]", str(messages))
        self.assertNotIn("session-0 fresh observation", str(messages))

        buy = prepare(client, messages)
        self.assertIsNone(buy.control_request)
        buy_output, messages = complete_policy_turn(client, buy, "click[Buy Now]")
        self.assertEqual(
            buy_output.info["context_transition"]["operation"], "preserve"
        )
        self.assertIn("click[Buy Now]", str(messages))
        self.assertNotIn("session-1 fresh observation", str(messages))
        self.assertEqual(client.native_calls, ["search[black mug]", "click[Buy Now]"])

        handoff = prepare(client, messages)
        self.assertEqual(handoff.control_request, WEBSHOP_SESSION_HANDOFF_REQUEST)
        self.assertIn("session-0 search result", str(handoff.messages))
        self.assertNotIn("session-1 fresh observation", str(handoff.messages))
        handoff_output, messages = complete_policy_turn(
            client, handoff, "notes/state.md"
        )

        self.assertEqual(client.native_calls, ["search[black mug]", "click[Buy Now]"])
        self.assertEqual(
            (handoff_output.info["native_call_count_before"],
             handoff_output.info["native_call_count_after"]),
            (2, 2),
        )
        self.assertEqual(
            (handoff_output.info["policy_step_before"],
             handoff_output.info["policy_step_after"]),
            (2, 3),
        )
        self.assertEqual(
            (handoff_output.info["context_epoch_before"],
             handoff_output.info["context_epoch_after"]),
            (0, 1),
        )
        handoff_evidence = handoff_output.info["wrapper_evidence"]
        self.assertEqual(
            (
                handoff_evidence["native_call_count_before"],
                handoff_evidence["native_call_count_after"],
            ),
            (2, 2),
        )
        self.assertEqual(
            (
                handoff_evidence["policy_step_before"],
                handoff_evidence["policy_step_after"],
            ),
            (2, 3),
        )
        self.assertEqual(
            (
                handoff_evidence["context_epoch_before"],
                handoff_evidence["context_epoch_after"],
            ),
            (0, 1),
        )
        self.assertIn("session-1 fresh observation", str(messages))
        self.assertIn("notes/state.md", str(messages))
        self.assertNotIn("session-0 search result", str(messages))
        self.assertNotIn("click[Buy Now]", str(messages))
        self.assertNotIn(WEBSHOP_SESSION_HANDOFF_REQUEST, str(messages))

    def test_invalid_webshop_handoff_remains_evidence_but_not_context(self) -> None:
        client = FakeWebShopClient([webshop_buy_response()])
        messages = self.bind_webshop(client)
        buy_output, messages = complete_policy_turn(
            client, prepare(client, messages), "click[Buy Now]"
        )
        self.assertEqual(buy_output.info["session_epoch_after"], 1)

        invalid = "black; cat file:///Users/master/state.md"
        handoff_output, messages = complete_policy_turn(
            client, prepare(client, messages), invalid
        )
        self.assertEqual(
            handoff_output.info["action_submission"]["raw_policy_output"], invalid
        )
        self.assertFalse(
            handoff_output.info["wrapper_evidence"]["handoff_parse"]["valid"]
        )
        self.assertNotIn(invalid, str(messages))
        self.assertIn("session-1 fresh observation", str(messages))
        self.assertEqual(client.native_calls, ["click[Buy Now]"])

    def test_invalid_native_output_is_evidence_not_successor_context(self) -> None:
        client = FakeWebShopClient([webshop_search_response()])
        messages = self.bind_webshop(client)
        invalid = '<ShellCommand {"command":"cat note.md"}>'

        output, messages = complete_policy_turn(
            client,
            prepare(client, messages),
            invalid,
        )

        self.assertEqual(
            output.info["action_submission"]["raw_policy_output"], invalid
        )
        self.assertEqual(
            output.info["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertNotIn(invalid, str(messages))
        self.assertIn("session-0 search result", str(messages))
        self.assertNotIn("session-0 fresh observation", str(messages))

    def test_swesmith_compaction_uses_same_entrypoint_without_native_call(self) -> None:
        client = FakeSwesmithClient()
        initial = client.policy_framing() + [
            {"role": "user", "content": client.observe()},
        ]
        messages = bind_initial_policy_context(client, initial)

        action = prepare(client, messages, capacity=4096)
        self.assertIsNone(action.control_request)
        action_output, messages = complete_policy_turn(
            client,
            action,
            'shell_command {"command":"rg -n parser .","workdir":"."}',
        )
        self.assertEqual(len(client.native_calls), 1)
        self.assertEqual(
            action_output.info["wrapper_evidence"]["workspace_continuity_id"],
            client.env_id,
        )
        self.assertEqual(
            action_output.info["wrapper_evidence"]["actor_credit"],
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": True,
                "basis": "shell_executed",
            },
        )
        self.assertEqual(
            (action_output.info["policy_step_after"],
             action_output.info["native_call_count_after"]),
            (1, 1),
        )
        self.assertIn("native tool output 1", str(messages))

        action_count = count_prompt_tokens(messages)
        candidate_count = count_prompt_tokens(
            messages + [{"role": "user", "content": SWE_CONTEXT_COMPACTION_REQUEST}]
        )
        self.assertGreater(candidate_count, action_count)
        compaction = prepare(client, messages, capacity=candidate_count + 1)
        self.assertEqual(
            compaction.control_request, SWE_CONTEXT_COMPACTION_REQUEST
        )
        summary = "Progress is in .agent_memory/MEMORY.md; inspect parser.py next."
        compaction_output, messages = complete_policy_turn(
            client, compaction, summary
        )

        self.assertEqual(len(client.native_calls), 1)
        self.assertEqual(
            compaction_output.info["wrapper_evidence"]["workspace_continuity_id"],
            client.env_id,
        )
        self.assertEqual(
            compaction_output.info["wrapper_evidence"]["actor_credit"],
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": True,
                "basis": "policy_context_compaction",
            },
        )
        self.assertEqual(
            (compaction_output.info["native_call_count_before"],
             compaction_output.info["native_call_count_after"]),
            (1, 1),
        )
        self.assertEqual(
            (compaction_output.info["policy_step_before"],
             compaction_output.info["policy_step_after"]),
            (1, 2),
        )
        self.assertEqual(
            (compaction_output.info["context_epoch_before"],
             compaction_output.info["context_epoch_after"]),
            (0, 1),
        )
        self.assertIn("Fix the failing parser", str(messages))
        self.assertIn(summary, str(messages))
        self.assertNotIn("native tool output 1", str(messages))
        self.assertNotIn(SWE_CONTEXT_COMPACTION_REQUEST, str(messages))

    def test_swesmith_actor_credit_receipt_fails_closed(self) -> None:
        invalid_receipts = (
            None,
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": "false",
                "basis": "parser_rejected",
            },
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": False,
                "basis": "workspace_changed",
            },
        )
        for receipt in invalid_receipts:
            with self.subTest(receipt=receipt):
                with self.assertRaises(RuntimeError):
                    _validate_actor_credit_receipt(receipt)

    def test_swesmith_replaces_legacy_acknowledgement_with_system_framing(self) -> None:
        client = FakeSwesmithClient()
        messages = bind_initial_policy_context(
            client,
            [
                {"role": "user", "content": "Legacy coding instructions"},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": client.observe()},
            ],
        )

        self.assertEqual(
            messages,
            client.policy_framing()
            + [{"role": "user", "content": client.observe()}],
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertNotIn("Understood.", str(messages))
        system_prompt = messages[0]["content"]
        self.assertIn("Start at byte zero", system_prompt)
        self.assertIn(
            'shell_command {"command":"find . -maxdepth 2 -type f | head -80",'
            '"workdir":".","timeout_ms":120000}',
            system_prompt,
        )
        self.assertIn("Start at byte zero with shell_command or apply_patch", system_prompt)
        self.assertIn("no XML tags", system_prompt)
        self.assertIn("*** Update File: relative/path.py", system_prompt)
        self.assertIn("apply_patch is optional", system_prompt)
        self.assertIn("use shell_command when an exact patch is uncertain", system_prompt)
        self.assertIn("relative to /testbed", system_prompt)
        self.assertIn("never /testbed/src/module.py", system_prompt)
        self.assertIn("Do not use cat > or a here-document", system_prompt)
        self.assertIn("re-inspect the small target region", system_prompt)
        self.assertIn("<think> tag", system_prompt)
        self.assertIn("Think privately", system_prompt)
        self.assertIn("A shell command can edit", system_prompt)
        self.assertIn("workspace intentionally has no .git directory", system_prompt)
        self.assertIn("Do not submit plain text until", system_prompt)
        self.assertIn("Prose before or after a tool action is a parser error", system_prompt)


class RolloutFakeWebShopClient(FakeWebShopClient):
    def reset(self, idx: int = 0) -> dict:
        del idx
        self._reset_policy_transition_state(self.info["env_info"])
        return deepcopy(self.info)

    def close(self) -> None:
        return None


class RolloutFakeSwesmithClient(FakeSwesmithClient):
    def reset(self, idx: int = 0) -> dict:
        del idx
        self._reset_policy_transition_state()
        return deepcopy(self.info)

    def prepare_policy_turn(self, pressure):
        del pressure
        self._selected_policy_control = None
        if self._native_call_count == 0:
            return None
        self._selected_policy_control = "context_compaction"
        return SWE_CONTEXT_COMPACTION_REQUEST

    def finalize_policy_horizon(self) -> StepOutput:
        return StepOutput(
            state="workspace grader completed",
            reward=0.5,
            done=True,
            info=build_task_neutral_transition_info(
                env_info={"resolved": True},
                wrapper_evidence={
                    "event": "horizon_grade",
                    "outcome": "success",
                },
            ),
        )

    def close(self) -> None:
        return None


class FakeRolloutTokenizer:
    pad_token_id = 0

    _responses = {
        (101, 999): "click[Buy Now]",
        (103, 999): "notes/state.md",
        (201, 202, 999): (
            'shell_command {"command":"rg -n parser .","workdir":"."}'
        ),
        (204, 205, 999): (
            "Progress is in .agent_memory/MEMORY.md; inspect parser.py next."
        ),
    }

    def apply_chat_template(self, conversations, **kwargs):
        del kwargs
        token_ids = [7]
        for message in conversations:
            role = str(message["role"])
            content = str(message["content"])
            marker = 50
            if content == WEBSHOP_SESSION_HANDOFF_REQUEST:
                marker = 31
            elif content == SWE_CONTEXT_COMPACTION_REQUEST:
                marker = 41
            elif "WebShop" in content:
                marker = 11
            elif "Coding" in content or "Fix the failing parser" in content:
                marker = 21
            token_ids.extend(
                [
                    {"system": 1, "user": 2, "assistant": 3}[role],
                    marker,
                    100 + len(content),
                ]
            )
        return token_ids

    def decode(self, token_ids, *, skip_special_tokens=True):
        self.assert_skip_special_tokens = skip_special_tokens
        return self._responses[tuple(int(token_id) for token_id in token_ids)]


class SharedRolloutRuntimeTest(unittest.TestCase):
    def test_two_wrappers_share_exact_sampled_and_packed_policy_rows(self) -> None:
        import torch

        from verl.utils.agentgym.rollout_context import (
            AGENTMEMORY_STEP_RECORD_JSON,
            normalize_generation_record,
        )
        from verl.workers.rollout.agent_vllm_rollout.vllm_rollout import (
            vLLMRollout,
        )
        from verl.workers.rollout.schemas import Message, RolloutHandler

        tokenizer = FakeRolloutTokenizer()
        rollout = vLLMRollout.__new__(vLLMRollout)
        rollout.tokenizer = tokenizer
        rollout.pad_token_id = tokenizer.pad_token_id
        rollout.agentgym_config = {"max_observation_tokens": 16}
        rollout.sampling_params = SimpleNamespace(max_tokens=8)

        def make_handler(system_prompt: str, item_id: int) -> RolloutHandler:
            handler = RolloutHandler(
                messages=[Message(role="system", content=system_prompt)],
                task_name="fixture",
                item_id=item_id,
                score=0.0,
                done=False,
                input_ids=[],
                prompt_ids=[],
                response_ids=[],
                attention_mask=[],
                prompt_attention_mask=[],
                response_attention_mask=[],
                position_ids=[],
                prompt_position_ids=[],
                response_position_ids=[],
                loss_mask=[],
                prompt_loss_mask=[],
                response_loss_mask=[],
                max_response_len=8,
                max_model_len=72,
            )
            handler.parent_index = item_id
            handler.rollout_replica_index = 0
            handler.data_idx = 0
            return handler

        webshop = RolloutFakeWebShopClient([webshop_buy_response()])
        swesmith = RolloutFakeSwesmithClient()
        sampled_prompts: dict[str, list[int]] = {}

        def generate_token_ids(*, generation_prompt_idxs, sampling_params):
            del sampling_params
            return [list(prompt_ids) for prompt_ids in generation_prompt_idxs]

        def output_generation_records(generated, sampling_params, **kwargs):
            del sampling_params, kwargs
            records = []
            for prompt_ids in generated:
                markers = set(prompt_ids)
                if 31 in markers:
                    response_ids = [103, 999]
                elif 41 in markers:
                    response_ids = [204, 205, 999]
                elif 11 in markers:
                    response_ids = [101, 999]
                elif 21 in markers:
                    response_ids = [201, 202, 999]
                else:
                    raise AssertionError(f"unroutable fixture prompt: {prompt_ids}")
                action = tokenizer.decode(response_ids, skip_special_tokens=True)
                sampled_prompts[action] = list(prompt_ids)
                records.append(
                    normalize_generation_record(
                        response_ids,
                        eos_token_ids=[999],
                        primary_eos_token_id=999,
                        pad_token_id=0,
                        max_tokens=8,
                        backend_finish_reason="stop",
                        stop_reason=None,
                        finish_reason_source="official_vllm",
                        token_ids_are_exact=True,
                    )
                )
            return records

        rollout._generate_token_ids = generate_token_ids
        rollout._output_generation_records = output_generation_records

        expected_actions = [
            "click[Buy Now]",
            "notes/state.md",
            'shell_command {"command":"rg -n parser .","workdir":"."}',
            "Progress is in .agent_memory/MEMORY.md; inspect parser.py next.",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            rollout.config = SimpleNamespace(
                prompt_length=64,
                response_length=8,
                max_model_len=72,
                n=1,
                send_interval=0,
                rollout_log_dir=tmpdir,
            )
            with mock.patch.dict(
                os.environ,
                {"AGENTMEMORY_LATEST_OBS_SUFFIX_CREDIT": "0"},
            ):
                output = rollout.generate_task_neutral_policy(
                    rollout_handler_ls=[
                        make_handler("WebShop tool contract", 0),
                        make_handler("Coding tool contract", 1),
                    ],
                    env_clients=[webshop, swesmith],
                    cur_device=torch.device("cpu"),
                    max_policy_turns=2,
                    sampling_kwargs={"max_tokens": 8},
                    global_steps=1,
                )

        records = [
            json.loads(value)
            for value in output.non_tensor_batch[AGENTMEMORY_STEP_RECORD_JSON]
        ]
        self.assertEqual([record["action"] for record in records], expected_actions)
        self.assertEqual(len(records), 4)
        prompt_width = rollout.config.prompt_length
        for row_index, record in enumerate(records):
            response_mask = output.batch["response_mask"][row_index].bool()
            packed_response = output.batch["responses"][row_index][
                response_mask
            ].tolist()
            self.assertEqual(packed_response, record["response_token_ids"])
            self.assertGreater(int(response_mask.sum().item()), 0)

            prompt_mask = output.batch["attention_mask"][
                row_index, :prompt_width
            ].bool()
            packed_prompt = output.batch["prompts"][row_index][
                prompt_mask
            ].tolist()
            self.assertEqual(packed_prompt, sampled_prompts[record["action"]])

            sampled_positions = response_mask.nonzero().flatten().tolist()
            score_row = output.batch["scores"][row_index]
            self.assertAlmostEqual(
                float(score_row[sampled_positions[-1]].item()),
                float(record["immediate_reward"]),
            )
            self.assertEqual(
                float(score_row[sampled_positions[:-1]].abs().sum().item()),
                0.0,
            )
            self.assertEqual(
                record["action_submission"]["raw_policy_output"],
                record["action"],
            )

        handoff, compaction = records[1], records[3]
        self.assertEqual(
            handoff["wrapper_evidence"]["event"],
            "webshop_session_handoff",
        )
        self.assertEqual(
            handoff["wrapper_evidence"]["native_call_count_before"],
            handoff["wrapper_evidence"]["native_call_count_after"],
        )
        self.assertEqual(
            handoff["wrapper_evidence"]["policy_step_after"],
            handoff["wrapper_evidence"]["policy_step_before"] + 1,
        )
        self.assertEqual(handoff["context_transition"]["operation"], "replace_messages")
        self.assertEqual(handoff["immediate_reward"], 0.0)
        self.assertEqual(compaction["wrapper_evidence"]["event"], "context_compaction")
        self.assertEqual(
            compaction["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertEqual(compaction["immediate_reward"], 0.5)
        self.assertTrue(compaction["done"])
        self.assertEqual(compaction["outcome"], "success")
        self.assertEqual(compaction["horizon_finalization"]["reward"], 0.5)
        self.assertEqual(webshop.native_calls, ["click[Buy Now]"])
        self.assertEqual(len(swesmith.native_calls), 1)


if __name__ == "__main__":
    unittest.main()
