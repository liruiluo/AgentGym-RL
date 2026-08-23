from __future__ import annotations

import ast
import inspect
import json
import unittest
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

from agentenv.controller import StepOutput
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA,
    build_task_neutral_transition_info,
)
from agentmemorygym_verl import agent_loop as agent_loop_module
from agentmemorygym_verl.agent_loop import AMGTaskNeutralAgentLoop
from agentmemorygym_verl.routes import RouteRegistry, RouteSpec


@dataclass
class _MergeResult:
    token_ids: list[int]


class _RecordingContinuousBuilder:
    def __init__(self):
        self.initial_calls = []
        self.non_assistant_calls = []
        self.assistant_calls = []

    @staticmethod
    def _render(messages):
        payload = "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        return [ord(char) for char in payload] + [999]

    def build_initial_tokens(self, messages):
        self.initial_calls.append([dict(message) for message in messages])
        return self._render(messages)

    def merge_non_assistant_tokens(self, previous, updated, runtime):
        self.non_assistant_calls.append((previous, updated, runtime))
        return _MergeResult(self._render(updated))

    def merge_assistant_tokens(self, runtime, assistant):
        self.assistant_calls.append((list(runtime), list(assistant)))
        return _MergeResult(list(runtime)[:-1] + list(assistant))


class _HistoryNormalizingContinuousBuilder(_RecordingContinuousBuilder):
    """Mimic Qwen3.5 dropping generation-only thinking markers on rerender."""

    def build_initial_tokens(self, messages):
        self.initial_calls.append([dict(message) for message in messages])
        if any(message["role"] == "assistant" for message in messages):
            return [700, 701]
        return [700, 702]

    def merge_assistant_tokens(self, runtime, assistant):
        self.assistant_calls.append((list(runtime), list(assistant)))
        return _MergeResult(list(runtime) + list(assistant))

    def merge_non_assistant_tokens(self, previous, updated, runtime):
        self.non_assistant_calls.append((previous, updated, runtime))
        return _MergeResult(list(runtime) + [703])


class _Tokenizer:
    def __init__(self, actions=None):
        self.actions = actions or {}
        self.eos_token_id = 999

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return self.actions[tuple(token_ids)]


class _Server:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    async def generate(self, *, request_id, prompt_ids, sampling_params, priority):
        index = len(self.calls)
        token_id = 100 + index
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
                "priority": priority,
            }
        )
        return SimpleNamespace(
            token_ids=[token_id],
            log_probs=[-0.01 * (index + 1)],
            routed_experts=None,
            num_preempted=0,
            stop_reason="stop",
            extra_fields={"min_global_steps": index, "max_global_steps": index + 1},
        )


class _MemoryChainClient:
    def __init__(self):
        self.actions = []
        self.files = {}
        self.bound_context = None
        self.closed = False

    def reset(self, data_idx):
        self.data_idx = data_idx

    def observe(self):
        return "Task: persist a value across compaction, revise it, and execute it."

    def normalize_initial_policy_context(self, messages):
        return messages

    def bind_policy_context(self, messages, initial=False):
        self.bound_context = messages
        self.initial_bind = initial

    def policy_turn_candidate(self):
        if len(self.actions) == 1:
            return "Compact the current context now. Preserve only a short summary; files remain available."
        return None

    def prepare_policy_turn(self, pressure):
        self.last_pressure = pressure
        return self.policy_turn_candidate()

    def step(self, action):
        self.actions.append(action)
        if action == "WRITE memory.md alpha=2":
            self.files["memory.md"] = "alpha=2"
            return StepOutput(
                state="wrote memory.md",
                reward=0.0,
                done=False,
                info=build_task_neutral_transition_info(
                    wrapper_evidence={"memory_event": "write"}
                ),
            )
        if action == "COMPACT retain memory locator":
            transition = {
                "schema": TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA,
                "operation": CONTEXT_OPERATION_REPLACE,
                "messages": [
                    {
                        "role": "system",
                        "content": "Use ordinary shell and filesystem actions.",
                    },
                    {
                        "role": "user",
                        "content": "Context compacted. Continue using memory.md.",
                    },
                ],
            }
            return StepOutput(
                state="context replaced",
                reward=0.0,
                done=False,
                info=build_task_neutral_transition_info(
                    context_transition=transition,
                    wrapper_evidence={"memory_event": "compaction"},
                ),
            )
        if action == "READ memory.md":
            return StepOutput(
                state=self.files["memory.md"],
                reward=0.0,
                done=False,
                info=build_task_neutral_transition_info(
                    wrapper_evidence={
                        "memory_event": "read",
                        "value": self.files["memory.md"],
                    }
                ),
            )
        if action == "MODIFY memory.md alpha=3":
            if self.files.get("memory.md") != "alpha=2":
                raise AssertionError(
                    "memory was not read from the persisted pre-compaction file"
                )
            self.files["memory.md"] = "alpha=3"
            return StepOutput(
                state="updated memory.md",
                reward=0.0,
                done=False,
                info=build_task_neutral_transition_info(
                    wrapper_evidence={"memory_event": "modify"}
                ),
            )
        if action == "EXECUTE memory.md":
            if self.files.get("memory.md") != "alpha=3":
                raise AssertionError("modified memory was not reused by execution")
            return StepOutput(
                state="execution used alpha=3",
                reward=1.0,
                done=True,
                info=build_task_neutral_transition_info(
                    env_info={"resolved": True},
                    wrapper_evidence={"memory_event": "execute", "outcome": "success"},
                ),
            )
        raise AssertionError(f"unexpected policy action: {action}")

    def close(self):
        self.closed = True


class _ExcludedClient(_MemoryChainClient):
    def __init__(
        self,
        *,
        reason="executor_infrastructure_fault",
        exclude_on_action=1,
    ):
        super().__init__()
        self.sample_excluded = False
        self.reason = reason
        self.exclude_on_action = exclude_on_action

    def step(self, action):
        self.actions.append(action)
        if len(self.actions) < self.exclude_on_action:
            return StepOutput(
                state="valid prefix before infrastructure fault",
                reward=0.25,
                done=False,
                info=build_task_neutral_transition_info(),
            )
        self.sample_excluded = True
        return StepOutput(
            state="infrastructure fault excluded this sample",
            reward=None,
            done=True,
            info=build_task_neutral_transition_info(
                env_info={
                    "truncated": True,
                    "terminal_reason": self.reason,
                },
                wrapper_evidence={
                    "outcome": "environment_error",
                    "terminal_reason": self.reason,
                },
            ),
        )


class _SuccessfulClient(_MemoryChainClient):
    def __init__(self):
        super().__init__()
        self.sample_excluded = False

    def step(self, action):
        self.actions.append(action)
        return StepOutput(
            state="valid terminal sample",
            reward=1.0,
            done=True,
            info=build_task_neutral_transition_info(
                env_info={"resolved": True},
                wrapper_evidence={"outcome": "success"},
            ),
        )


class _HorizonClient(_MemoryChainClient):
    def step(self, action):
        self.actions.append(action)
        return StepOutput(
            state="candidate retained",
            reward=0.25,
            done=False,
            info=build_task_neutral_transition_info(
                wrapper_evidence={"outcome": "continue"}
            ),
        )

    def policy_turn_candidate(self):
        return None

    def finalize_policy_horizon(self):
        return StepOutput(
            state="best candidate accepted at horizon",
            reward=0.75,
            done=True,
            info=build_task_neutral_transition_info(
                env_info={"resolved": True},
                wrapper_evidence={"outcome": "success", "source": "horizon"},
            ),
        )


class _PressureHorizonClient(_HorizonClient):
    def __init__(self):
        super().__init__()
        self.pressures = []

    def policy_turn_candidate(self):
        return "Optional task-neutral control turn."

    def prepare_policy_turn(self, pressure):
        self.pressures.append(pressure)
        return None


class _ErrorClient(_MemoryChainClient):
    def policy_turn_candidate(self):
        return None

    def step(self, action):
        self.actions.append(action)
        raise RuntimeError("deliberate wrapper failure")


class TestTaskNeutralAgentLoopSource(unittest.TestCase):
    def test_shared_loop_contains_no_concrete_environment_literal(self):
        tree = ast.parse(inspect.getsource(agent_loop_module))
        forbidden = ("webshop", "swesmith", "literesearcher", "openmle", "searchqa")
        hits = sorted(
            {
                value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                for value in (node.value.lower(),)
                if any(name in value for name in forbidden)
            }
        )
        self.assertEqual(hits, [])


class TestPromptRendering(unittest.TestCase):
    def _loop(self):
        loop = object.__new__(AMGTaskNeutralAgentLoop)
        loop.continuous_token_builder = _RecordingContinuousBuilder()
        loop.tokenizer = object()
        loop.apply_chat_template_kwargs = {}
        return loop

    def test_control_candidate_is_full_rendered_before_any_assistant_generation(self):
        loop = self._loop()
        current = [{"role": "user", "content": "task"}]
        current_ids = loop._render_prompt_sync(current)
        candidate = current + [{"role": "user", "content": "compact now"}]

        actual = loop._prompt_for_candidate(current, current_ids, candidate)

        self.assertEqual(actual, loop.continuous_token_builder._render(candidate))
        self.assertEqual(loop.continuous_token_builder.non_assistant_calls, [])
        self.assertEqual(loop.continuous_token_builder.initial_calls[-1], candidate)

    def test_append_after_sampled_assistant_uses_upstream_continuous_token_merge(self):
        loop = self._loop()
        prepared = [{"role": "user", "content": "task"}]
        prompt_ids = loop._render_prompt_sync(prepared)
        action_ids = [11, 12]
        next_messages = prepared + [
            {"role": "assistant", "content": "act"},
            {"role": "user", "content": "observation"},
        ]

        actual = loop._next_prompt_ids(
            prepared_messages=prepared,
            prepared_prompt_ids=prompt_ids,
            action="act",
            action_token_ids=action_ids,
            next_messages=next_messages,
        )

        self.assertEqual(actual, loop.continuous_token_builder._render(next_messages))
        self.assertEqual(len(loop.continuous_token_builder.assistant_calls), 1)
        self.assertEqual(len(loop.continuous_token_builder.non_assistant_calls), 1)

    def test_append_keeps_continuous_runtime_for_history_normalizing_template(self):
        loop = self._loop()
        loop.continuous_token_builder = _HistoryNormalizingContinuousBuilder()
        prepared = [{"role": "user", "content": "task"}]
        prompt_ids = loop._render_prompt_sync(prepared)
        next_messages = prepared + [
            {"role": "assistant", "content": "act"},
            {"role": "user", "content": "observation"},
        ]

        actual = loop._next_prompt_ids(
            prepared_messages=prepared,
            prepared_prompt_ids=prompt_ids,
            action="act",
            action_token_ids=[11, 12],
            next_messages=next_messages,
        )

        self.assertEqual(actual, prompt_ids + [11, 12, 703])
        self.assertNotEqual(actual, [700, 701])
        self.assertEqual(loop.continuous_token_builder.initial_calls, [prepared])

    def test_replace_messages_forces_full_rebuild(self):
        loop = self._loop()
        prepared = [{"role": "user", "content": "long context"}]
        prompt_ids = loop._render_prompt_sync(prepared)
        replacement = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "compacted"},
        ]

        actual = loop._next_prompt_ids(
            prepared_messages=prepared,
            prepared_prompt_ids=prompt_ids,
            action="compact summary",
            action_token_ids=[15],
            next_messages=replacement,
        )

        self.assertEqual(actual, loop.continuous_token_builder._render(replacement))
        self.assertEqual(loop.continuous_token_builder.assistant_calls, [])
        self.assertEqual(loop.continuous_token_builder.non_assistant_calls, [])


class TestAMGAgentLoop(IsolatedAsyncioTestCase):
    @staticmethod
    def _registry(*routes):
        if not routes:
            routes = (
                RouteSpec(
                    route_id="openmle-fast",
                    max_rounds=1,
                    max_observation_tokens=32,
                    policy_framing_sha256=None,
                    route_attestation_sha256=None,
                    client_config=MappingProxyType(
                        {
                            "task_name": "openmle_fast",
                            "env_addr": "http://127.0.0.1:65404",
                            "max_retries": 2,
                        }
                    ),
                ),
            )
        return RouteRegistry(routes=routes, sha256=None, source_path=None)

    def _loop(self, actions, *, max_turns, registry=None):
        loop = object.__new__(AMGTaskNeutralAgentLoop)
        loop.agentgym_config = {"max_retries": 2}
        loop._route_registry = registry or self._registry(
            RouteSpec(
                route_id="openmle-fast",
                max_rounds=max_turns,
                max_observation_tokens=32,
                policy_framing_sha256=None,
                route_attestation_sha256=None,
                client_config=MappingProxyType(
                    {
                        "task_name": "openmle_fast",
                        "env_addr": "http://127.0.0.1:65404",
                        "max_retries": 2,
                    }
                ),
            )
        )
        loop._envelope_tokens = 1
        loop.rollout_config = SimpleNamespace(
            response_length=8,
            prompt_length=512,
            max_model_len=520,
            calculate_log_probs=True,
        )
        loop.server_manager = _Server(actions)
        loop.tokenizer = _Tokenizer(
            {(100 + index,): action for index, action in enumerate(actions)}
        )
        loop.continuous_token_builder = _RecordingContinuousBuilder()
        loop._render_prompt_sync = lambda messages: [
            1 + index for index, _ in enumerate(json.dumps(messages, sort_keys=True))
        ]
        return loop

    async def test_one_shared_loop_selects_only_the_row_route_and_labels_every_action(
        self,
    ):
        route_specs = tuple(
            RouteSpec(
                route_id=route_id,
                max_rounds=max_rounds,
                max_observation_tokens=max_observation_tokens,
                policy_framing_sha256=None,
                route_attestation_sha256=str(index + 1) * 64,
                client_config=MappingProxyType(
                    {
                        "task_name": task_name,
                        "env_addr": f"http://127.0.0.1:{65410 + index}",
                        "max_retries": 0,
                    }
                ),
            )
            for index, (
                route_id,
                task_name,
                max_rounds,
                max_observation_tokens,
            ) in enumerate(
                (
                    ("webshop", "webshop", 1, 11),
                    ("swesmith", "swesmith", 2, 22),
                    ("literesearcher", "agentmemory", 3, 33),
                    ("openmle-fast", "openmle_fast", 4, 44),
                )
            )
        )
        registry = self._registry(*route_specs)

        for route in route_specs:
            with self.subTest(route_id=route.route_id):
                client = _SuccessfulClient()
                loop = self._loop(
                    [f"ACTION {route.route_id}"],
                    max_turns=99,
                    registry=registry,
                )
                selected_configs = []

                def select_client(config):
                    selected_configs.append(config)
                    return client

                with mock.patch.object(
                    agent_loop_module,
                    "create_env_client",
                    side_effect=select_client,
                ):
                    outputs = await loop.run(
                        {"max_tokens": 8},
                        item_id=f"task-{route.route_id}",
                        data_idx=0,
                        uid=f"trajectory-{route.route_id}",
                        route_id=route.route_id,
                        extra_info={"route_id": route.route_id},
                        raw_prompt=[{"role": "system", "content": "system"}],
                    )

                self.assertEqual(selected_configs, [route.client_config])
                self.assertTrue(client.closed)
                self.assertEqual(len(outputs), 1)
                self.assertEqual(outputs[0].extra_fields["route_id"], route.route_id)
                self.assertEqual(
                    outputs[0].extra_fields["data_source"], route.route_id
                )
                record = json.loads(outputs[0].extra_fields["step_record_json"])
                self.assertEqual(record["route_id"], route.route_id)
                self.assertEqual(record["data_source"], route.route_id)

    async def test_route_local_horizon_and_observation_budget_are_used(self):
        route = RouteSpec(
            route_id="swesmith",
            max_rounds=2,
            max_observation_tokens=73,
            policy_framing_sha256=None,
            route_attestation_sha256=None,
            client_config=MappingProxyType(
                {
                    "task_name": "swesmith",
                    "env_addr": "http://127.0.0.1:65411",
                    "max_retries": 0,
                }
            ),
        )
        client = _PressureHorizonClient()
        loop = self._loop(
            ["FIRST", "SECOND"],
            max_turns=99,
            registry=self._registry(route),
        )

        with mock.patch.object(
            agent_loop_module, "create_env_client", return_value=client
        ):
            outputs = await loop.run(
                {"max_tokens": 8},
                item_id="route-local-limits",
                data_idx=0,
                route_id="swesmith",
                raw_prompt=[{"role": "system", "content": "system"}],
            )

        self.assertEqual(len(outputs), 2)
        self.assertEqual(client.actions, ["FIRST", "SECOND"])
        self.assertEqual(
            [pressure.max_observation_tokens for pressure in client.pressures],
            [73, 73],
        )
        self.assertTrue(client.closed)

    async def test_selected_client_closes_when_wrapper_raises(self):
        route = RouteSpec(
            route_id="literesearcher",
            max_rounds=1,
            max_observation_tokens=32,
            policy_framing_sha256=None,
            route_attestation_sha256=None,
            client_config=MappingProxyType(
                {
                    "task_name": "agentmemory",
                    "env_addr": "http://127.0.0.1:65412",
                    "max_retries": 0,
                }
            ),
        )
        client = _ErrorClient()
        loop = self._loop(["FAIL"], max_turns=99, registry=self._registry(route))

        with mock.patch.object(
            agent_loop_module, "create_env_client", return_value=client
        ):
            with self.assertRaisesRegex(RuntimeError, "deliberate wrapper failure"):
                await loop.run(
                    {"max_tokens": 8},
                    item_id="route-error",
                    data_idx=0,
                    route_id="literesearcher",
                    raw_prompt=[{"role": "system", "content": "system"}],
                )

        self.assertTrue(client.closed)

    async def test_each_policy_action_stops_at_tokenizer_eos_and_records_reason(self):
        client = _SuccessfulClient()
        loop = self._loop(["KEPT ACTION"], max_turns=1)

        with mock.patch.object(
            agent_loop_module, "create_env_client", return_value=client
        ):
            outputs = await loop.run(
                {"max_tokens": 8, "stop_token_ids": [777, 999, 777]},
                item_id="eos-boundary-task",
                data_idx=9,
                uid="trajectory-eos-boundary",
                raw_prompt=[{"role": "system", "content": "system"}],
            )

        self.assertEqual(
            loop.server_manager.calls[0]["sampling_params"]["stop_token_ids"],
            [777, 999],
        )
        record = json.loads(outputs[0].extra_fields["step_record_json"])
        self.assertEqual(record["generation_stop_reason"], "stop")
        self.assertEqual(outputs[0].extra_fields["generation_stop_reason"], "stop")

    async def test_complete_filesystem_memory_chain_is_one_ordered_ppo_episode(self):
        actions = [
            "WRITE memory.md alpha=2",
            "COMPACT retain memory locator",
            "READ memory.md",
            "MODIFY memory.md alpha=3",
            "EXECUTE memory.md",
        ]
        client = _MemoryChainClient()
        loop = self._loop(actions, max_turns=len(actions))

        with mock.patch.object(
            agent_loop_module, "create_env_client", return_value=client
        ):
            outputs = await loop.run(
                {"max_tokens": 8},
                item_id="memory-task",
                data_idx=4,
                uid="trajectory-4",
                raw_prompt=[{"role": "system", "content": "system"}],
            )

        self.assertEqual(client.actions, actions)
        self.assertEqual(client.files, {"memory.md": "alpha=3"})
        self.assertTrue(client.closed)
        self.assertEqual(len(outputs), len(actions))
        self.assertEqual(
            [output.extra_fields["trajectory_row_order"] for output in outputs],
            list(range(len(actions))),
        )
        self.assertEqual(
            [
                output.extra_fields["wrapper_evidence"]["memory_event"]
                for output in outputs
            ],
            ["write", "compaction", "read", "modify", "execute"],
        )
        self.assertEqual(
            [output.extra_fields["min_global_steps"] for output in outputs],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [output.extra_fields["max_global_steps"] for output in outputs],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [output.response_logprobs for output in outputs],
            [[-0.01], [-0.02], [-0.03], [-0.04], [-0.05]],
        )
        self.assertEqual(
            [output.extra_fields["trajectory_terminal"] for output in outputs],
            [False] * 4 + [True],
        )
        self.assertEqual(outputs[-1].extra_fields["outcome"], "success")
        records = [
            json.loads(output.extra_fields["step_record_json"]) for output in outputs
        ]
        reward_evidence = [
            output.extra_fields["reward_extra_info"] for output in outputs
        ]
        self.assertEqual(
            [json.loads(info["step_record_json"]) for info in reward_evidence],
            records,
        )
        self.assertEqual([info["is_padding"] for info in reward_evidence], [False] * 5)
        self.assertEqual(
            [record["trajectory_row_uid"] for record in records],
            [f"trajectory-4-row-{i}" for i in range(5)],
        )
        self.assertEqual(sum(record["immediate_reward"] for record in records), 1.0)
        self.assertEqual([record["trajectory_return"] for record in records], [1.0] * 5)

    async def test_horizon_receipt_updates_reward_outcome_and_final_evidence(self):
        client = _HorizonClient()
        loop = self._loop(["WRITE candidate.py"], max_turns=1)

        with mock.patch.object(
            agent_loop_module, "create_env_client", return_value=client
        ):
            outputs = await loop.run(
                {"max_tokens": 8},
                item_id="horizon-task",
                data_idx=2,
                uid="trajectory-horizon",
                raw_prompt=[{"role": "system", "content": "system"}],
            )

        self.assertEqual(len(outputs), 1)
        output = outputs[0]
        record = json.loads(output.extra_fields["step_record_json"])
        self.assertAlmostEqual(output.reward_score, 1.0)
        self.assertAlmostEqual(record["immediate_reward"], 1.0)
        self.assertTrue(record["rollout_done_flag"])
        self.assertEqual(record["outcome"], "success")
        self.assertEqual(output.extra_fields["outcome"], "success")
        self.assertEqual(
            record["horizon_finalization"]["wrapper_evidence"]["source"], "horizon"
        )
        self.assertTrue(record["horizon_finalization"]["env_info"]["resolved"])

    async def test_excluded_attempt_is_closed_and_resampled_as_a_whole_trajectory(self):
        excluded = _ExcludedClient(exclude_on_action=2)
        successful = _SuccessfulClient()
        loop = self._loop(
            ["DISCARDED PREFIX", "DISCARDED FAULT", "KEPT ACTION"],
            max_turns=2,
        )
        loop.agentgym_config["max_retries"] = 2

        with mock.patch.object(
            agent_loop_module,
            "create_env_client",
            side_effect=[excluded, successful],
        ) as create_client:
            outputs = await loop.run(
                {"max_tokens": 8},
                item_id="resampled-task",
                data_idx=7,
                uid="trajectory-resampled",
                raw_prompt=[{"role": "system", "content": "system"}],
            )

        self.assertEqual(create_client.call_count, 2)
        self.assertTrue(excluded.closed)
        self.assertTrue(successful.closed)
        self.assertEqual(
            excluded.actions,
            ["DISCARDED PREFIX", "DISCARDED FAULT"],
        )
        self.assertEqual(successful.actions, ["KEPT ACTION"])
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].extra_fields["action_text"], "KEPT ACTION")
        self.assertEqual(outputs[0].response_ids, [102])
        self.assertEqual(outputs[0].response_logprobs, [-0.03])
        self.assertEqual(outputs[0].reward_score, 1.0)
        self.assertEqual(outputs[0].extra_fields["sample_reschedule_attempt"], 1)
        record = json.loads(outputs[0].extra_fields["step_record_json"])
        self.assertEqual(record["sample_reschedule_attempt"], 1)
        self.assertNotIn("DISCARDED PREFIX", outputs[0].extra_fields["step_record_json"])
        self.assertNotIn("DISCARDED FAULT", outputs[0].extra_fields["step_record_json"])

    async def test_repeated_exclusion_fails_closed_after_bounded_retries(self):
        first = _ExcludedClient(reason="executor_infrastructure_fault")
        second = _ExcludedClient(reason="grader_infrastructure_fault")
        loop = self._loop(["FIRST EXCLUDED", "SECOND EXCLUDED"], max_turns=1)
        loop.agentgym_config["max_retries"] = 1

        with mock.patch.object(
            agent_loop_module,
            "create_env_client",
            side_effect=[first, second],
        ) as create_client:
            with self.assertRaisesRegex(
                RuntimeError,
                "excluded after 2 complete trajectory attempts.*grader_infrastructure_fault",
            ):
                await loop.run(
                    {"max_tokens": 8},
                    item_id="always-excluded-task",
                    data_idx=8,
                    uid="trajectory-always-excluded",
                    raw_prompt=[{"role": "system", "content": "system"}],
                )

        self.assertEqual(create_client.call_count, 2)
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        self.assertEqual(first.actions, ["FIRST EXCLUDED"])
        self.assertEqual(second.actions, ["SECOND EXCLUDED"])


if __name__ == "__main__":
    unittest.main()
