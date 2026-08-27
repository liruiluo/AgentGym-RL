from __future__ import annotations

from copy import deepcopy
import hashlib
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

import agentenv.envs.openmle_fast as openmle_fast_module  # noqa: E402
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
from agentenv.envs.filesystem_checkpoint import (  # noqa: E402
    FILESYSTEM_CHECKPOINT_PATH,
    FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA,
    FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA,
    filesystem_workspace_action_request_sha256,
)
from agentenv.envs.openmle_fast import (  # noqa: E402
    OPENMLE_CONTEXT_COMPACTION_REQUEST,
    OPENMLE_POLICY_CONTINUATION_MARKER,
    OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
    OpenMLEFastEnvClient,
)
from agentenv.envs.swesmith import (  # noqa: E402
    SWE_CONTEXT_COMPACTION_REQUEST,
    SWE_MEMORY_CONTRACT,
    SWE_POLICY_SYSTEM_PROMPT,
    SwesmithEnvClient,
    _validate_action_progress_receipt,
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


def filesystem_checkpoint_receipt(
    *,
    action_kind: str = "shell_command",
    action_completed: bool = True,
    changed: bool = True,
    exists: bool = True,
    regular_file: bool = True,
    size_bytes: int | None = 37,
) -> dict:
    return {
        "schema": FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA,
        "path": FILESYSTEM_CHECKPOINT_PATH,
        "action_kind": action_kind,
        "action_completed": action_completed,
        "changed": changed,
        "exists": exists,
        "regular_file": regular_file,
        "size_bytes": size_bytes,
        "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
    }


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
        response = deepcopy(self._responses.pop(0))
        info = response.get("info", {})
        latest_event = info.get("workspace_latest_event")
        tool_ops = info.get("tool_ops")
        preserve_identity = bool(info.pop("_preserve_workspace_event_identity", False))
        if (
            not preserve_identity
            and isinstance(latest_event, dict)
            and isinstance(tool_ops, list)
            and tool_ops
            and isinstance(tool_ops[-1], dict)
        ):
            current_step = len(self.native_calls)
            request_sha256 = filesystem_workspace_action_request_sha256(
                data["action"]
            )
            if request_sha256 is not None:
                for event in (latest_event, tool_ops[-1]):
                    event["step"] = current_step
                    event["event_id"] = current_step - 1
                    event["request_sha256"] = request_sha256
        return response

    def reset(self, idx: int = 0) -> dict:
        del idx
        raise AssertionError("test fixture is already reset")


class FakeSwesmithClient(SwesmithEnvClient):
    def __init__(self, *, invalid_action_reward: float = 0.0) -> None:
        if float(invalid_action_reward) != 0.0:
            raise ValueError("SWE-smith invalid_action_reward must be zero")
        BaseEnvClient.__init__(self, action_format=ActionFormat.REACT)
        self.invalid_action_reward = float(invalid_action_reward)
        self.env_id = 202
        self.data_len = 1
        self.metadata = {
            "task_count": 1,
            "memory_contract": SWE_MEMORY_CONTRACT,
        }
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
        if action == "malformed policy output":
            return {
                "observation": "Invalid action: expected one bare tool action.",
                "reward": -0.01,
                "done": True,
                "info": {
                    "step": step,
                    "action_kind": "parser_error",
                    "terminal": True,
                    "episode_success": False,
                    "terminal_reason": "parser_rejected",
                    "actor_credit": {
                        "schema": "task_neutral_actor_credit_v1",
                        "positive_eligible": False,
                        "basis": "parser_rejected",
                    },
                },
            }
        checkpoint_write = (
            FILESYSTEM_CHECKPOINT_PATH in action
            and ("printf" in action or action.lstrip().startswith("apply_patch"))
        )
        checkpoint_read = (
            f"cat {FILESYSTEM_CHECKPOINT_PATH}" in action and not checkpoint_write
        )
        workspace_changed = "printf changed >" in action or checkpoint_write
        terminal_submission = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in action
        action_kind = (
            "apply_patch" if action.lstrip().startswith("apply_patch") else "shell_command"
        )
        return {
            "observation": f"native tool output {step}",
            "reward": float(terminal_submission),
            "done": terminal_submission,
            "info": {
                "step": step,
                "action_kind": action_kind,
                "filesystem_checkpoint": filesystem_checkpoint_receipt(
                    action_kind=action_kind,
                    changed=checkpoint_write,
                    exists=checkpoint_write or checkpoint_read,
                    size_bytes=37 if checkpoint_write or checkpoint_read else None,
                ),
                "filesystem_checkpoint_read": (
                    {
                        "schema": FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA,
                        "path": FILESYSTEM_CHECKPOINT_PATH,
                        "observed": True,
                        "size_bytes": 37,
                        "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                    }
                    if checkpoint_read
                    else None
                ),
                "workspace_changed_paths": (
                    [FILESYSTEM_CHECKPOINT_PATH]
                    if checkpoint_write
                    else (["notes.md"] if "printf changed > notes.md" in action else [])
                ),
                "shell_action_succeeded": action_kind == "shell_command",
                "actor_credit": {
                    "schema": "task_neutral_actor_credit_v1",
                    "positive_eligible": True,
                    "basis": (
                        "terminal_submission"
                        if terminal_submission
                        else "shell_executed"
                    ),
                },
                "action_progress": {
                    "schema": "swesmith_action_progress_v1",
                    "action_fingerprint": hashlib.sha256(
                        action.encode("utf-8")
                    ).hexdigest(),
                    "result_fingerprint": hashlib.sha256(
                        ("result:" + action).encode("utf-8")
                    ).hexdigest(),
                    "workspace_changed": workspace_changed,
                },
            },
        }

    def reset(self, idx: int = 0) -> dict:
        del idx
        raise AssertionError("test fixture is already reset")


class FakeOpenMLEClient(OpenMLEFastEnvClient):
    def __init__(self) -> None:
        BaseEnvClient.__init__(self, action_format=ActionFormat.REACT)
        self.env_id = 303
        self.data_len = 1
        self.metadata = {"max_policy_actions": 30}
        self.info = {
            "observation": "Solve the OpenMLE task from TASK.md.",
            "reward": 0.0,
            "done": False,
            "info": {},
        }
        self._episode_identity = {"episode_id": "fake-openmle"}
        self.native_calls: list[str] = []
        self._reset_transition_state()

    def __len__(self) -> int:
        return 1

    def _request(self, method: str, path: str, **kwargs) -> dict:
        if (method, path) != ("POST", "step"):
            raise AssertionError(f"unexpected fake OpenMLE request: {method} {path}")
        action = str(kwargs["json"]["action"])
        self.native_calls.append(action)
        return {"sentinel": len(self.native_calls)}

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


def webshop_checkpoint_response(
    *,
    changed: bool = True,
    exit_code: int = 0,
    timed_out: bool = False,
    stdout: str = "",
    checkpoint_payload: bytes = b"checkpoint",
    exists: bool | None = None,
) -> dict:
    if exists is None:
        exists = changed
    entry = {
        "path": FILESYSTEM_CHECKPOINT_PATH,
        "bytes": len(checkpoint_payload),
        "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
        "kind": "file",
    }
    return {
        "observation": (
            stdout
            if stdout
            else ("workspace write completed" if changed else "checkpoint unchanged")
        ),
        "reward": 0.0,
        "done": False,
        "info": {
            "current_subtask_index": 1,
            "tool_ops": [{"op": "SHELL_COMMAND", "step": 3}],
            "session_trace": [],
            "workspace_latest_event": {
                "op": "SHELL_COMMAND",
                "tool_name": "shell_command",
                "status": "executed",
                "exit_code": exit_code,
                "timed_out": timed_out,
                "stdout": stdout,
                "workspace_diff": {
                    "added": [entry] if changed else [],
                    "modified": [],
                    "deleted": [],
                },
            },
            "workspace_snapshot": {"files": [entry] if exists else []},
        },
    }


class SharedWrapperPolicyTurnTest(unittest.TestCase):
    def test_swesmith_endpoint_memory_contract_mismatch_fails_closed(self) -> None:
        for endpoint_contract in (None, "policy_compaction_only_v1"):
            calls = []

            class MismatchedSwesmithClient(SwesmithEnvClient):
                def _request(self, method: str, path: str, **kwargs) -> dict:
                    del kwargs
                    calls.append((method, path))
                    if (method, path) == ("GET", "metadata"):
                        metadata = {"task_count": 1}
                        if endpoint_contract is not None:
                            metadata["memory_contract"] = endpoint_contract
                        return metadata
                    raise AssertionError("client must fail before creating an episode")

            with self.subTest(endpoint_contract=endpoint_contract):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "SWE-smith endpoint memory contract mismatch",
                ):
                    MismatchedSwesmithClient("http://unused.invalid")
                self.assertEqual(calls, [("GET", "metadata")])

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

    def test_webshop_handoff_requires_executed_checkpoint_then_real_read(self) -> None:
        checkpoint_body = "objective: finish six purchases; next: search blue mug"
        checkpoint_payload = checkpoint_body.encode("utf-8")
        client = FakeWebShopClient(
            [
                webshop_search_response(),
                webshop_buy_response(),
                webshop_checkpoint_response(checkpoint_payload=checkpoint_payload),
                webshop_checkpoint_response(
                    changed=False,
                    stdout=checkpoint_body,
                    checkpoint_payload=checkpoint_payload,
                    exists=True,
                ),
            ]
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

        buy = prepare(client, messages)
        self.assertIsNone(buy.control_request)
        buy_output, messages = complete_policy_turn(client, buy, "click[Buy Now]")
        self.assertEqual(
            buy_output.info["context_transition"]["operation"], "preserve"
        )
        self.assertIn("click[Buy Now]", str(messages))
        self.assertNotIn("session-1 fresh observation", str(messages))

        handoff = prepare(client, messages)
        self.assertEqual(handoff.control_request, WEBSHOP_SESSION_HANDOFF_REQUEST)
        checkpoint_action = (
            'shell_command {"command":"mkdir -p .agent_memory && printf %s '
            + checkpoint_body
            + ' > .agent_memory/CONTINUATION.md","workdir":"."}'
        )
        handoff_output, messages = complete_policy_turn(
            client, handoff, checkpoint_action
        )

        self.assertEqual(
            client.native_calls,
            ["search[black mug]", "click[Buy Now]", checkpoint_action],
        )
        self.assertEqual(
            (handoff_output.info["native_call_count_before"],
             handoff_output.info["native_call_count_after"]),
            (2, 3),
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
        evidence = handoff_output.info["wrapper_evidence"]
        self.assertEqual(evidence["event"], "webshop_session_handoff")
        self.assertTrue(evidence["continuation_persisted"])
        self.assertEqual(evidence["continuation_path"], FILESYSTEM_CHECKPOINT_PATH)
        self.assertFalse(evidence["checkpoint_action_in_successor_context"])
        self.assertFalse(evidence["checkpoint_content_in_successor_context"])
        self.assertIn("session-1 fresh observation", str(messages))
        self.assertIn(FILESYSTEM_CHECKPOINT_PATH, str(messages))
        self.assertIn(evidence["checkpoint_receipt"]["sha256"], str(messages))
        self.assertNotIn("session-0 search result", str(messages))
        self.assertNotIn("click[Buy Now]", str(messages))
        self.assertNotIn(checkpoint_action, str(messages))
        self.assertNotIn(checkpoint_body, str(messages))
        self.assertNotIn(WEBSHOP_SESSION_HANDOFF_REQUEST, str(messages))

        read = prepare(client, messages)
        self.assertIsNone(read.control_request)
        read_action = (
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md",'
            '"workdir":"."}'
        )
        read_output, messages = complete_policy_turn(client, read, read_action)
        self.assertEqual(client.native_calls[-1], read_action)
        self.assertEqual(read_output.info["context_epoch_after"], 1)
        self.assertTrue(
            read_output.info["wrapper_evidence"]["checkpoint_read_satisfied"]
        )
        self.assertFalse(
            read_output.info["wrapper_evidence"]["checkpoint_read_retry_pending"]
        )
        self.assertEqual(
            read_output.info["wrapper_evidence"]["memory_event"], "read"
        )
        self.assertIn("objective: finish six purchases", str(messages))

    def test_failed_webshop_checkpoint_keeps_context_and_retries(self) -> None:
        client = FakeWebShopClient(
            [webshop_buy_response(), webshop_checkpoint_response(changed=False)]
        )
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
        self.assertEqual(
            handoff_output.info["context_transition"]["operation"],
            "append_observation",
        )
        self.assertEqual(handoff_output.info["context_epoch_after"], 0)
        self.assertFalse(
            handoff_output.info["wrapper_evidence"]["continuation_persisted"]
        )
        self.assertTrue(handoff_output.info["wrapper_evidence"]["retry_pending"])
        self.assertIn(invalid, str(messages))
        self.assertNotIn("checkpoint unchanged", str(messages))
        self.assertEqual(
            messages[-1]["content"],
            "Filesystem checkpoint was not accepted (missing_receipt). "
            "The earlier context is still present. Retry now with exactly one "
            "shell_command or apply_patch that overwrites "
            "`.agent_memory/CONTINUATION.md` with 1 to 8192 bytes.",
        )
        self.assertEqual(
            handoff_output.info["env_info"]["workspace_latest_event"][
                "workspace_diff"
            ]["added"],
            [],
        )
        self.assertEqual(
            handoff_output.info["wrapper_evidence"]["native_wrapper_evidence"][
                "event"
            ],
            "native_action",
        )
        self.assertFalse(
            handoff_output.info["wrapper_evidence"]["native_wrapper_evidence"][
                "raw_history_cleared"
            ]
        )
        retry = prepare(client, messages, capacity=4096)
        self.assertEqual(retry.control_request, WEBSHOP_SESSION_HANDOFF_REQUEST)
        self.assertEqual(
            client.native_calls,
            ["click[Buy Now]", invalid],
        )

    def test_failed_webshop_shell_cannot_authorize_context_replacement(self) -> None:
        client = FakeWebShopClient(
            [
                webshop_buy_response(),
                webshop_checkpoint_response(changed=True, exit_code=7),
            ]
        )
        messages = self.bind_webshop(client)
        _, messages = complete_policy_turn(
            client, prepare(client, messages), "click[Buy Now]"
        )

        output, messages = complete_policy_turn(
            client,
            prepare(client, messages),
            'shell_command {"command":"false > .agent_memory/CONTINUATION.md"}',
        )

        self.assertEqual(
            output.info["context_transition"]["operation"],
            "append_observation",
        )
        self.assertFalse(output.info["wrapper_evidence"]["context_replaced"])
        self.assertEqual(
            output.info["wrapper_evidence"]["checkpoint_failure_reason"],
            "action_not_completed",
        )
        self.assertIn("action_not_completed", messages[-1]["content"])

    def test_stale_webshop_workspace_event_cannot_authorize_handoff(self) -> None:
        stale = webshop_checkpoint_response(changed=True)
        stale_info = stale["info"]
        stale_info["_preserve_workspace_event_identity"] = True
        stale_info["workspace_latest_event"].update(
            step=1,
            event_id=0,
            request_sha256="a" * 64,
        )
        stale_info["tool_ops"][-1].update(
            step=2,
            event_id=1,
            request_sha256="b" * 64,
        )
        client = FakeWebShopClient([webshop_buy_response(), stale])
        messages = self.bind_webshop(client)
        _, messages = complete_policy_turn(
            client, prepare(client, messages), "click[Buy Now]"
        )

        output, messages = complete_policy_turn(
            client,
            prepare(client, messages),
            'shell_command {"command":"printf checkpoint > '
            '.agent_memory/CONTINUATION.md"}',
        )

        self.assertEqual(
            output.info["context_transition"]["operation"],
            "append_observation",
        )
        self.assertFalse(output.info["wrapper_evidence"]["context_replaced"])
        self.assertEqual(
            output.info["wrapper_evidence"]["checkpoint_failure_reason"],
            "missing_receipt",
        )
        self.assertIn("missing_receipt", messages[-1]["content"])

    def test_coherently_stale_webshop_event_is_bound_to_submitted_action(self) -> None:
        stale = webshop_checkpoint_response(changed=True)
        stale_info = stale["info"]
        stale_info["_preserve_workspace_event_identity"] = True
        old_digest = filesystem_workspace_action_request_sha256(
            'shell_command {"command":"printf old > .agent_memory/CONTINUATION.md"}'
        )
        self.assertIsNotNone(old_digest)
        for event in (stale_info["workspace_latest_event"], stale_info["tool_ops"][-1]):
            event.update(step=2, event_id=1, request_sha256=old_digest)
        client = FakeWebShopClient([webshop_buy_response(), stale])
        messages = self.bind_webshop(client)
        _, messages = complete_policy_turn(
            client, prepare(client, messages), "click[Buy Now]"
        )

        output, messages = complete_policy_turn(
            client,
            prepare(client, messages),
            'shell_command {"command":"printf new > '
            '.agent_memory/CONTINUATION.md"}',
        )

        self.assertEqual(
            output.info["context_transition"]["operation"],
            "append_observation",
        )
        self.assertFalse(output.info["wrapper_evidence"]["context_replaced"])
        self.assertEqual(
            output.info["wrapper_evidence"]["checkpoint_failure_reason"],
            "missing_receipt",
        )
        self.assertIn("missing_receipt", messages[-1]["content"])

    def test_webshop_requires_matching_checkpoint_read_before_progress(self) -> None:
        payload = b"objective: continue\nnext: search blue mug"
        client = FakeWebShopClient(
            [
                webshop_buy_response(),
                webshop_checkpoint_response(checkpoint_payload=payload),
                {
                    "observation": "session-1 search result",
                    "reward": 0.0,
                    "done": False,
                    "info": {
                        "current_subtask_index": 1,
                        "tool_ops": [{"op": "SEARCH", "step": 3}],
                        "session_trace": ["search result"],
                    },
                },
                {
                    "observation": "session-1 second search result",
                    "reward": 0.0,
                    "done": False,
                    "info": {
                        "current_subtask_index": 1,
                        "tool_ops": [{"op": "SEARCH", "step": 4}],
                        "session_trace": ["second search result"],
                    },
                },
                webshop_checkpoint_response(
                    changed=False,
                    stdout=payload.decode("utf-8"),
                    checkpoint_payload=payload,
                    exists=True,
                ),
            ]
        )
        messages = self.bind_webshop(client)
        _, messages = complete_policy_turn(
            client, prepare(client, messages), "click[Buy Now]"
        )
        write_action = (
            'shell_command {"command":"printf checkpoint > '
            '.agent_memory/CONTINUATION.md"}'
        )
        _, messages = complete_policy_turn(
            client, prepare(client, messages), write_action
        )

        wrong_output, messages = complete_policy_turn(
            client, prepare(client, messages), "search[blue mug]"
        )
        wrong_evidence = wrong_output.info["wrapper_evidence"]
        self.assertTrue(wrong_evidence["checkpoint_read_required"])
        self.assertFalse(wrong_evidence["checkpoint_read_satisfied"])
        self.assertTrue(wrong_evidence["checkpoint_read_retry_pending"])
        self.assertEqual(
            wrong_output.info["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertIn("Checkpoint read failed", messages[-1]["content"])
        self.assertIn(FILESYSTEM_CHECKPOINT_PATH, str(messages))
        self.assertNotIn("search[blue mug]", str(messages))
        self.assertNotIn("session-1 search result", str(messages))
        self.assertEqual(wrong_output.info["context_epoch_after"], 1)
        self.assertEqual(
            wrong_output.info["action_submission"]["raw_policy_output"],
            "search[blue mug]",
        )

        first_retry_messages = deepcopy(messages)
        second_wrong_output, messages = complete_policy_turn(
            client, prepare(client, messages), "search[red mug]"
        )
        self.assertEqual(
            second_wrong_output.info["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertEqual(messages, first_retry_messages)
        self.assertNotIn("search[red mug]", str(messages))
        self.assertNotIn("session-1 second search result", str(messages))
        self.assertEqual(second_wrong_output.info["context_epoch_after"], 1)

        read_action = (
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md"}'
        )
        read_output, _ = complete_policy_turn(
            client, prepare(client, messages), read_action
        )
        read_evidence = read_output.info["wrapper_evidence"]
        self.assertTrue(read_evidence["checkpoint_read_required"])
        self.assertTrue(read_evidence["checkpoint_read_satisfied"])
        self.assertFalse(read_evidence["checkpoint_read_retry_pending"])
        self.assertEqual(read_evidence["memory_event"], "read")

    def test_webshop_handoff_requires_one_failed_retry_of_headroom(self) -> None:
        client = FakeWebShopClient([webshop_buy_response()])
        messages = self.bind_webshop(client)
        _, messages = complete_policy_turn(
            client, prepare(client, messages), "click[Buy Now]"
        )
        candidate_messages = messages + [
            {"role": "user", "content": WEBSHOP_SESSION_HANDOFF_REQUEST}
        ]
        barely_fits_first_request = count_prompt_tokens(candidate_messages) + 1

        with self.assertRaisesRegex(RuntimeError, "failed checkpoint attempt"):
            prepare(client, messages, capacity=barely_fits_first_request)

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

    def test_openmle_compaction_executes_checkpoint_without_free_read(self) -> None:
        client = FakeOpenMLEClient()
        initial = client.policy_framing() + [
            {"role": "user", "content": client.observe()},
        ]
        messages = bind_initial_policy_context(client, initial)
        messages.extend(
            [
                {"role": "assistant", "content": "old action output"},
                {"role": "user", "content": "large previous execution evidence"},
            ]
        )
        candidate_count = count_prompt_tokens(
            messages
            + [{"role": "user", "content": OPENMLE_CONTEXT_COMPACTION_REQUEST}]
        )
        prepared = prepare(client, messages, capacity=candidate_count + 1)
        self.assertEqual(
            prepared.control_request,
            OPENMLE_CONTEXT_COMPACTION_REQUEST,
        )
        marker = getattr(
            openmle_fast_module,
            "OPENMLE_POLICY_CONTINUATION_MARKER",
            None,
        )
        self.assertIsInstance(marker, str)
        self.assertIn("Earlier conversation was removed", marker)
        self.assertNotIn("but you may instead", prepared.control_request)
        receipt = filesystem_checkpoint_receipt(action_kind="apply_patch")
        env_info = {
            "action_kind": "apply_patch",
            "action_status": "completed",
            "counters": {"action_count": 1},
            "execution": {"filesystem_checkpoint": receipt},
        }
        secret = "next inspect train.csv then revise model"
        action = f"""apply_patch
*** Begin Patch
*** Add File: {FILESYSTEM_CHECKPOINT_PATH}
+{secret}
*** End Patch"""
        with mock.patch.object(
            openmle_fast_module,
            "_validate_step_response",
            return_value=("action_status=completed", 0.0, False, env_info),
        ):
            output, replacement = complete_policy_turn(client, prepared, action)

        self.assertEqual(client.native_calls, [action])
        self.assertEqual(client.metadata["max_policy_actions"], 30)
        self.assertEqual(
            (
                output.info["native_call_count_before"],
                output.info["native_call_count_after"],
                output.info["policy_step_before"],
                output.info["policy_step_after"],
                output.info["context_epoch_before"],
                output.info["context_epoch_after"],
            ),
            (0, 1, 0, 1, 0, 1),
        )
        self.assertEqual(replacement[0], initial[0])
        self.assertIn(initial[-1]["content"], replacement[-1]["content"])
        self.assertIn(FILESYSTEM_CHECKPOINT_PATH, replacement[-1]["content"])
        self.assertIn(receipt["sha256"], replacement[-1]["content"])
        self.assertNotIn("large previous execution evidence", str(replacement))
        self.assertNotIn(action, str(replacement))
        self.assertNotIn(secret, str(replacement))
        evidence = output.info["wrapper_evidence"]
        self.assertEqual(evidence["workspace_continuity_id"], client.env_id)
        self.assertEqual(evidence["event"], "context_compaction")
        self.assertTrue(evidence["continuation_persisted"])
        self.assertEqual(evidence["continuation_path"], FILESYSTEM_CHECKPOINT_PATH)
        self.assertFalse(evidence["checkpoint_action_in_successor_context"])
        self.assertFalse(evidence["checkpoint_observation_in_successor_context"])
        self.assertFalse(evidence["checkpoint_content_in_successor_context"])
        self.assertTrue(evidence["checkpoint_read_required_after"])

        wrong_info = {
            "action_kind": "shell_command",
            "action_status": "completed",
            "counters": {"action_count": 2},
            "counter_delta": {"execution_completed_count": 1},
            "execution": {
                "status": "completed",
                "exit_code": 0,
                "timed_out": False,
                "changed_paths": [],
                "execution_completed_delta": 1,
            },
        }
        wrong_action = 'shell_command {"command":"python train.py"}'
        with mock.patch.object(
            openmle_fast_module,
            "_validate_step_response",
            return_value=("training complete", 0.0, False, wrong_info),
        ):
            wrong_output, replacement = complete_policy_turn(
                client,
                prepare(client, replacement),
                wrong_action,
            )
        self.assertIn("Checkpoint read failed", wrong_output.state)
        self.assertTrue(
            wrong_output.info["wrapper_evidence"]["checkpoint_read_retry_pending"]
        )
        self.assertIn(FILESYSTEM_CHECKPOINT_PATH, str(replacement))
        self.assertEqual(
            wrong_output.info["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertNotIn(wrong_action, str(replacement))
        self.assertNotIn("training complete", str(replacement))
        self.assertEqual(wrong_output.info["context_epoch_after"], 1)

        first_retry_messages = deepcopy(replacement)
        second_wrong_info = deepcopy(wrong_info)
        second_wrong_info["counters"]["action_count"] = 3
        second_wrong_action = 'shell_command {"command":"python second.py"}'
        with mock.patch.object(
            openmle_fast_module,
            "_validate_step_response",
            return_value=("second training output", 0.0, False, second_wrong_info),
        ):
            second_wrong_output, replacement = complete_policy_turn(
                client,
                prepare(client, replacement),
                second_wrong_action,
            )
        self.assertEqual(
            second_wrong_output.info["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertEqual(replacement, first_retry_messages)
        self.assertNotIn(second_wrong_action, str(replacement))
        self.assertNotIn("second training output", str(replacement))
        self.assertEqual(second_wrong_output.info["context_epoch_after"], 1)

        read_receipt = {
            "schema": FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA,
            "path": FILESYSTEM_CHECKPOINT_PATH,
            "observed": True,
            "size_bytes": receipt["size_bytes"],
            "sha256": receipt["sha256"],
        }
        read_info = {
            "action_kind": "shell_command",
            "action_status": "completed",
            "counters": {"action_count": 4},
            "counter_delta": {"execution_completed_count": 1},
            "execution": {
                "status": "completed",
                "exit_code": 0,
                "timed_out": False,
                "changed_paths": [],
                "execution_completed_delta": 1,
                "filesystem_checkpoint_read": read_receipt,
            },
        }
        read_action = (
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md"}'
        )
        with mock.patch.object(
            openmle_fast_module,
            "_validate_step_response",
            return_value=("checkpoint", 0.0, False, read_info),
        ):
            read_output, replacement = complete_policy_turn(
                client,
                prepare(client, replacement),
                read_action,
            )
        self.assertTrue(
            read_output.info["wrapper_evidence"]["checkpoint_read_satisfied"]
        )
        self.assertIsNone(client._pending_checkpoint_read)
        self.assertIn("checkpoint", str(replacement))

    def test_openmle_failed_checkpoint_keeps_context_and_retries(self) -> None:
        client = FakeOpenMLEClient()
        initial = client.policy_framing() + [
            {"role": "user", "content": client.observe()},
        ]
        messages = bind_initial_policy_context(client, initial)
        messages.extend(
            [
                {"role": "assistant", "content": "old action output"},
                {"role": "user", "content": "large previous execution evidence"},
            ]
        )
        candidate_count = count_prompt_tokens(
            messages
            + [{"role": "user", "content": OPENMLE_CONTEXT_COMPACTION_REQUEST}]
        )
        prepared = prepare(client, messages, capacity=candidate_count + 1)
        malformed = (
            'apply_patch {"patch":"*** Begin Patch\n'
            '# state\n*** End Patch"}'
        )
        failed_receipt = filesystem_checkpoint_receipt(
            action_kind="parser_error",
            action_completed=False,
            changed=False,
            exists=False,
            size_bytes=None,
        )
        failed_info = {
            "action_kind": "parser_error",
            "action_status": "parser_error",
            "counters": {"action_count": 1},
            "execution": {"filesystem_checkpoint": failed_receipt},
        }
        with mock.patch.object(
            openmle_fast_module,
            "_validate_step_response",
            return_value=("parser error", -0.01, False, failed_info),
        ):
            output, retry_messages = complete_policy_turn(
                client,
                prepared,
                malformed,
            )

        self.assertEqual(client.native_calls, [malformed])
        self.assertEqual(
            output.info["context_transition"]["operation"],
            "append_observation",
        )
        self.assertEqual(output.info["wrapper_evidence"]["event"], "context_compaction")
        self.assertFalse(output.info["wrapper_evidence"]["continuation_persisted"])
        self.assertTrue(output.info["wrapper_evidence"]["retry_pending"])
        self.assertEqual(output.info["wrapper_evidence"]["checkpoint_failure_reason"], "wrong_action_kind")
        self.assertEqual(
            (
                output.info["native_call_count_before"],
                output.info["native_call_count_after"],
                output.info["policy_step_before"],
                output.info["policy_step_after"],
                output.info["context_epoch_before"],
                output.info["context_epoch_after"],
            ),
            (0, 1, 0, 1, 0, 0),
        )
        self.assertIn("large previous execution evidence", str(retry_messages))
        self.assertTrue(
            any(message["content"] == malformed for message in retry_messages)
        )
        self.assertIn("parser error", str(retry_messages))
        retry = prepare(client, retry_messages, capacity=4096)
        self.assertEqual(retry.control_request, OPENMLE_CONTEXT_COMPACTION_REQUEST)

    def test_openmle_endpoint_attested_memory_events(self) -> None:
        read_receipt = {
            "schema": FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA,
            "path": FILESYSTEM_CHECKPOINT_PATH,
            "observed": True,
            "size_bytes": len(b"checkpoint"),
            "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        }
        cases = (
            (
                "read",
                'shell_command {"command":"cat .agent_memory/CONTINUATION.md"}',
                {
                    "action_kind": "shell_command",
                    "action_status": "completed",
                    "counter_delta": {"execution_completed_count": 1},
                    "execution": {
                        "status": "completed",
                        "exit_code": 0,
                        "timed_out": False,
                        "changed_paths": [],
                        "execution_completed_delta": 1,
                        "filesystem_checkpoint_read": read_receipt,
                    },
                },
            ),
            (
                "modify",
                "apply_patch\n*** Begin Patch\n*** Add File: notes.md\n+x\n*** End Patch",
                {
                    "action_kind": "apply_patch",
                    "action_status": "completed",
                    "counter_delta": {"execution_completed_count": 0},
                    "execution": {
                        "status": "completed",
                        "exit_code": None,
                        "timed_out": False,
                        "changed_paths": ["notes.md"],
                        "execution_completed_delta": 0,
                    },
                },
            ),
            (
                "execute",
                'shell_command {"command":"python train.py"}',
                {
                    "action_kind": "shell_command",
                    "action_status": "completed",
                    "counter_delta": {"execution_completed_count": 1},
                    "execution": {
                        "status": "completed",
                        "exit_code": 0,
                        "timed_out": False,
                        "changed_paths": [],
                        "execution_completed_delta": 1,
                    },
                },
            ),
        )
        for expected_event, action, env_info in cases:
            with self.subTest(expected_event=expected_event):
                client = FakeOpenMLEClient()
                with mock.patch.object(
                    openmle_fast_module,
                    "_validate_step_response",
                    return_value=("result", 0.0, False, env_info),
                ):
                    output = client.step(action)
                self.assertEqual(
                    output.info["wrapper_evidence"]["memory_event"],
                    expected_event,
                )

    def test_openmle_action_text_cannot_forge_memory_evidence(self) -> None:
        client = FakeOpenMLEClient()
        action = (
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md; '
            'printf x > notes.md; python train.py"}'
        )
        env_info = {
            "action_kind": "shell_command",
            "action_status": "failed",
            "counter_delta": {"execution_completed_count": 0},
            "execution": {
                "status": "failed",
                "exit_code": 7,
                "timed_out": False,
                "changed_paths": [],
                "execution_completed_delta": 0,
            },
        }
        with mock.patch.object(
            openmle_fast_module,
            "_validate_step_response",
            return_value=("failed", -0.01, False, env_info),
        ):
            output = client.step(action)
        self.assertNotIn("memory_event", output.info["wrapper_evidence"])

    def test_swesmith_compaction_executes_checkpoint_then_real_read(self) -> None:
        client = FakeSwesmithClient()
        self.assertEqual(
            SWE_MEMORY_CONTRACT,
            "policy_filesystem_checkpoint_then_client_replace_v2",
        )
        self.assertIn("# Durable debugging notes", SWE_POLICY_SYSTEM_PROMPT)
        self.assertIn(
            "maintain a concise evidence ledger incrementally",
            SWE_POLICY_SYSTEM_PROMPT,
        )
        self.assertIn("read the checkpoint with a normal command", SWE_POLICY_SYSTEM_PROMPT)
        self.assertIn(FILESYSTEM_CHECKPOINT_PATH, SWE_CONTEXT_COMPACTION_REQUEST)
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
            (action_output.info["policy_step_after"],
             action_output.info["native_call_count_after"]),
            (1, 1),
        )
        self.assertIn("native tool output 1", str(messages))

        candidate_count = count_prompt_tokens(
            messages + [{"role": "user", "content": SWE_CONTEXT_COMPACTION_REQUEST}]
        )
        compaction = prepare(client, messages, capacity=candidate_count + 1)
        self.assertEqual(compaction.control_request, SWE_CONTEXT_COMPACTION_REQUEST)
        secret = "parser evidence retained; next inspect parser.py"
        checkpoint_action = (
            'shell_command {"command":"mkdir -p .agent_memory && printf %s '
            + secret
            + ' > .agent_memory/CONTINUATION.md","workdir":"."}'
        )
        compaction_output, messages = complete_policy_turn(
            client, compaction, checkpoint_action
        )

        self.assertEqual(len(client.native_calls), 2)
        self.assertEqual(client.native_calls[-1], checkpoint_action)
        evidence = compaction_output.info["wrapper_evidence"]
        self.assertEqual(evidence["workspace_continuity_id"], client.env_id)
        self.assertTrue(evidence["continuation_persisted"])
        self.assertTrue(evidence["checkpoint_read_required_after"])
        self.assertIsNotNone(client._pending_checkpoint_read)
        self.assertEqual(
            (compaction_output.info["native_call_count_before"],
             compaction_output.info["native_call_count_after"]),
            (1, 2),
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
        self.assertIn(FILESYSTEM_CHECKPOINT_PATH, str(messages))
        self.assertNotIn(secret, str(messages))
        self.assertNotIn(checkpoint_action, str(messages))
        self.assertNotIn("native tool output 1", str(messages))
        self.assertNotIn("native tool output 2", str(messages))
        self.assertNotIn(SWE_CONTEXT_COMPACTION_REQUEST, str(messages))

        wrong_action = (
            'shell_command {"command":"python wrong.py","workdir":"."}'
        )
        wrong_output, messages = complete_policy_turn(
            client,
            prepare(client, messages, capacity=4096),
            wrong_action,
        )
        self.assertEqual(
            wrong_output.info["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertTrue(
            wrong_output.info["wrapper_evidence"]["checkpoint_read_retry_pending"]
        )
        self.assertEqual(
            (wrong_output.info["context_epoch_before"],
             wrong_output.info["context_epoch_after"]),
            (1, 1),
        )
        self.assertIn("Checkpoint read failed", str(messages))
        self.assertNotIn(wrong_action, str(messages))
        self.assertNotIn("native tool output 3", str(messages))

        reread = prepare(client, messages, capacity=4096)
        self.assertIsNone(reread.control_request)
        reread_action = (
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md",'
            '"workdir":"."}'
        )
        reread_output, messages = complete_policy_turn(
            client,
            reread,
            reread_action,
        )
        self.assertEqual(client.native_calls[-1], reread_action)
        self.assertEqual(len(client.native_calls), 4)
        self.assertEqual(
            reread_output.info["wrapper_evidence"]["workspace_continuity_id"],
            client.env_id,
        )
        self.assertEqual(
            (reread_output.info["context_epoch_before"],
             reread_output.info["context_epoch_after"]),
            (1, 1),
        )
        self.assertEqual(
            reread_output.info["wrapper_evidence"]["memory_event"], "read"
        )
        self.assertTrue(
            reread_output.info["wrapper_evidence"]["document_read_observed"]
        )
        self.assertTrue(
            reread_output.info["wrapper_evidence"]["checkpoint_read_satisfied"]
        )
        self.assertIsNone(client._pending_checkpoint_read)
        self.assertIn("native tool output 4", str(messages))

    def test_swesmith_endpoint_attested_modify_and_execute_events(self) -> None:
        client = FakeSwesmithClient()
        modify = client.step(
            'shell_command {"command":"printf changed > notes.md","workdir":"."}'
        )
        self.assertEqual(modify.info["wrapper_evidence"]["memory_event"], "modify")
        self.assertEqual(
            modify.info["wrapper_evidence"]["workspace_changed_paths"],
            ["notes.md"],
        )
        execute = client.step(
            'shell_command {"command":"python train.py","workdir":"."}'
        )
        self.assertEqual(execute.info["wrapper_evidence"]["memory_event"], "execute")

    def test_swesmith_action_text_cannot_forge_memory_evidence(self) -> None:
        client = FakeSwesmithClient()
        action = (
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md; '
            'printf x > notes.md; python train.py"}'
        )
        response = {
            "observation": "executor rejected",
            "reward": 0.0,
            "done": False,
            "info": {
                "step": 1,
                "action_kind": "shell_command",
                "actor_credit": {
                    "schema": "task_neutral_actor_credit_v1",
                    "positive_eligible": False,
                    "basis": "executor_rejected",
                },
            },
        }
        with mock.patch.object(client, "_request", return_value=response):
            output = client.step(action)
        self.assertNotIn("memory_event", output.info["wrapper_evidence"])

    def test_swesmith_failed_checkpoint_keeps_epoch_and_retries(self) -> None:
        client = FakeSwesmithClient()
        messages = bind_initial_policy_context(
            client,
            client.policy_framing()
            + [{"role": "user", "content": client.observe()}],
        )
        candidate_count = count_prompt_tokens(
            messages + [{"role": "user", "content": SWE_CONTEXT_COMPACTION_REQUEST}]
        )
        compaction = prepare(client, messages, capacity=candidate_count + 1)
        output, messages = complete_policy_turn(
            client, compaction, 'shell_command {"command":"true"}'
        )
        self.assertEqual(output.reward, 0.0)
        self.assertEqual(output.info["context_epoch_after"], 0)
        self.assertEqual(
            output.info["context_transition"]["operation"], "append_observation"
        )
        self.assertFalse(output.info["wrapper_evidence"]["continuation_persisted"])
        self.assertTrue(output.info["wrapper_evidence"]["retry_pending"])
        self.assertIn("checkpoint_not_changed", str(messages))
        retry = prepare(client, messages, capacity=4096)
        self.assertEqual(retry.control_request, SWE_CONTEXT_COMPACTION_REQUEST)

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

    def test_swesmith_action_progress_receipt_fails_closed(self) -> None:
        invalid_receipts = (
            None,
            {
                "schema": "swesmith_action_progress_v1",
                "action_fingerprint": "not-a-sha",
                "result_fingerprint": "b" * 64,
                "workspace_changed": False,
            },
            {
                "schema": "swesmith_action_progress_v1",
                "action_fingerprint": "a" * 64,
                "result_fingerprint": "b" * 64,
                "workspace_changed": 0,
            },
        )
        for receipt in invalid_receipts:
            with self.subTest(receipt=receipt):
                with self.assertRaises(RuntimeError):
                    _validate_action_progress_receipt(receipt)

    def test_swesmith_terminal_submission_preserves_shell_progress(self) -> None:
        client = FakeSwesmithClient()

        output = client.step(
            'shell_command {"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",'
            '"workdir":"."}'
        )

        evidence = output.info["wrapper_evidence"]
        self.assertTrue(output.done)
        self.assertEqual(output.reward, 1.0)
        self.assertEqual(
            evidence["actor_credit"],
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": True,
                "basis": "terminal_submission",
            },
        )
        self.assertEqual(
            evidence["action_progress"]["schema"],
            "swesmith_action_progress_v1",
        )

    def test_swesmith_zero_progress_repeat_resets_after_workspace_change(self) -> None:
        client = FakeSwesmithClient()
        inspect = 'shell_command {"command":"find . -maxdepth 2 -type f"}'

        first = client.step(inspect)
        repeated = client.step(inspect)
        mutation = client.step(
            'shell_command {"command":"printf changed > notes.txt"}'
        )
        after_mutation = client.step(inspect)

        self.assertTrue(
            first.info["wrapper_evidence"]["actor_credit"]["positive_eligible"]
        )
        self.assertEqual(
            repeated.info["wrapper_evidence"]["actor_credit"],
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": False,
                "basis": "zero_progress_repeat",
            },
        )
        self.assertTrue(
            mutation.info["wrapper_evidence"]["action_progress"][
                "workspace_changed"
            ]
        )
        self.assertTrue(
            after_mutation.info["wrapper_evidence"]["actor_credit"][
                "positive_eligible"
            ]
        )

    def test_swesmith_low_invalid_reward_is_environment_owned(self) -> None:
        client = FakeSwesmithClient()
        inspect = 'shell_command {"command":"find . -maxdepth 2 -type f"}'

        invalid = client.step("malformed policy output")
        first = client.step(inspect)
        repeated = client.step(inspect)

        self.assertEqual(invalid.reward, -0.01)
        self.assertNotIn("reward_overlay", invalid.info["wrapper_evidence"])
        self.assertEqual(first.reward, 0.0)
        self.assertEqual(repeated.reward, 0.0)
        self.assertNotIn("reward_overlay", repeated.info["wrapper_evidence"])

    def test_swesmith_rejects_a_second_invalid_action_penalty_overlay(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be zero"):
            FakeSwesmithClient(invalid_action_reward=-0.01)

    def test_swesmith_zero_progress_repeat_resets_after_compaction(self) -> None:
        client = FakeSwesmithClient()
        messages = bind_initial_policy_context(
            client,
            client.policy_framing()
            + [{"role": "user", "content": client.observe()}],
        )
        inspect = 'shell_command {"command":"find . -maxdepth 2 -type f"}'

        first, messages = complete_policy_turn(client, prepare(client, messages), inspect)
        repeated, messages = complete_policy_turn(
            client, prepare(client, messages), inspect
        )
        self.assertTrue(
            first.info["wrapper_evidence"]["actor_credit"]["positive_eligible"]
        )
        self.assertFalse(
            repeated.info["wrapper_evidence"]["actor_credit"]["positive_eligible"]
        )

        action_count = count_prompt_tokens(messages)
        candidate_count = count_prompt_tokens(
            messages + [{"role": "user", "content": SWE_CONTEXT_COMPACTION_REQUEST}]
        )
        self.assertGreater(candidate_count, action_count)
        compaction = prepare(client, messages, capacity=candidate_count + 1)
        self.assertEqual(compaction.control_request, SWE_CONTEXT_COMPACTION_REQUEST)
        checkpoint_action = (
            'shell_command {"command":"mkdir -p .agent_memory && printf changed > '
            '.agent_memory/CONTINUATION.md","workdir":"."}'
        )
        checkpoint_output, messages = complete_policy_turn(
            client, compaction, checkpoint_action
        )
        self.assertTrue(
            checkpoint_output.info["wrapper_evidence"]["continuation_persisted"]
        )

        after_compaction, _ = complete_policy_turn(
            client, prepare(client, messages), inspect
        )
        self.assertTrue(
            after_compaction.info["wrapper_evidence"]["actor_credit"][
                "positive_eligible"
            ]
        )

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


WEBSHOP_CHECKPOINT_ACTION = (
    'shell_command {"command":"mkdir -p .agent_memory && printf webshop-state > '
    '.agent_memory/CONTINUATION.md","workdir":"."}'
)
SWE_CHECKPOINT_ACTION = (
    'shell_command {"command":"mkdir -p .agent_memory && printf swe-state > '
    '.agent_memory/CONTINUATION.md","workdir":"."}'
)


class FakeRolloutTokenizer:
    pad_token_id = 0

    _responses = {
        (101, 999): "click[Buy Now]",
        (103, 999): WEBSHOP_CHECKPOINT_ACTION,
        (201, 202, 999): (
            'shell_command {"command":"rg -n parser .","workdir":"."}'
        ),
        (204, 205, 999): SWE_CHECKPOINT_ACTION,
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
                # Leave realistic headroom for one failed checkpoint write and
                # its bounded retry request. The synthetic token IDs remain tiny.
                max_model_len=2048,
            )
            handler.parent_index = item_id
            handler.rollout_replica_index = 0
            handler.data_idx = 0
            return handler

        webshop = RolloutFakeWebShopClient(
            [webshop_buy_response(), webshop_checkpoint_response()]
        )
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
            WEBSHOP_CHECKPOINT_ACTION,
            'shell_command {"command":"rg -n parser .","workdir":"."}',
            SWE_CHECKPOINT_ACTION,
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            rollout.config = SimpleNamespace(
                prompt_length=2040,
                response_length=8,
                max_model_len=2048,
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
            handoff["wrapper_evidence"]["native_call_count_after"],
            handoff["wrapper_evidence"]["native_call_count_before"] + 1,
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
        self.assertEqual(
            webshop.native_calls, ["click[Buy Now]", WEBSHOP_CHECKPOINT_ACTION]
        )
        self.assertEqual(len(swesmith.native_calls), 2)


if __name__ == "__main__":
    unittest.main()
