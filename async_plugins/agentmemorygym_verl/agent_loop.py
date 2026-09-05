"""Task-neutral multi-action AgentLoop for upstream veRL fully-async PPO."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any
from uuid import uuid4

import numpy as np
from agentenv.controller import (
    bind_initial_policy_context,
    complete_policy_turn,
    prepare_policy_turn,
)
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

from .env_client import create_env_client
from .routes import RouteSpec, route_registry_from_agentgym_config


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return (
        value.item()
        if hasattr(value, "item") and not isinstance(value, (str, bytes))
        else value
    )


def _messages(value: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("AMG raw_prompt must be a sequence of policy messages")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise TypeError(f"AMG policy message {index} is not a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(
                f"invalid AMG policy message at index {index}: {message!r}"
            )
        normalized.append({"role": str(role), "content": content})
    if not normalized:
        raise ValueError("AMG raw_prompt must not be empty")
    return normalized


def _policy_tool_schemas(client: Any) -> list[dict[str, Any]] | None:
    provider = getattr(client, "policy_tool_schemas", None)
    if not callable(provider):
        return None
    raw_schemas = provider()
    if raw_schemas is None:
        return None
    if isinstance(raw_schemas, (str, bytes)) or not isinstance(
        raw_schemas, Sequence
    ):
        raise TypeError("AMG policy tool schemas must be a sequence of mappings")

    schemas: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_schema in enumerate(raw_schemas):
        if not isinstance(raw_schema, Mapping):
            raise TypeError(f"AMG policy tool schema {index} is not a mapping")
        schema = deepcopy(dict(raw_schema))
        function = schema.get("function")
        if schema.get("type") != "function" or not isinstance(function, Mapping):
            raise ValueError(
                f"AMG policy tool schema {index} is not an OpenAI function schema"
            )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"AMG policy tool schema {index} has no function name")
        if name in names:
            raise ValueError(f"AMG policy tool schema repeats function {name!r}")
        names.add(name)
        schemas.append(schema)
    if not schemas:
        raise ValueError("AMG policy tool schemas must not be empty")
    return schemas


def _tool_schema_digest(tools: Sequence[Mapping[str, Any]] | None) -> str:
    payload = json.dumps(
        list(tools or []),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_prefix(
    prefix: Sequence[Mapping[str, str]], values: Sequence[Mapping[str, str]]
) -> bool:
    return len(prefix) <= len(values) and all(
        dict(left) == dict(right) for left, right in zip(prefix, values)
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _digest_token_ids(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


_QWEN_NATIVE_TOOL_TEMPLATE_REPLACEMENTS = (
    (
        "If you choose to call a function ONLY reply in the following format "
        "with NO suffix:",
        "For every response, call exactly one function and ONLY reply in the "
        "following format with NO prefix or suffix:",
    ),
    (
        "- You may provide optional reasoning for your function call in natural "
        "language BEFORE the function call, but NOT after",
        "- Output exactly one function call with no natural-language reasoning "
        "before or after it",
    ),
    (
        "- If there is no function call available, answer the question like "
        "normal with your current knowledge and do not tell the user about "
        "function calls",
        "- Always select exactly one available function; never answer in prose",
    ),
)


def _strict_qwen_tool_chat_template(template: Any) -> str:
    """Remove Qwen's optional prose path for AMG's strict one-call policy."""

    if not isinstance(template, str) or not template:
        raise RuntimeError("AMG native tools require a nonempty Qwen chat template")
    patched = template
    for old, new in _QWEN_NATIVE_TOOL_TEMPLATE_REPLACEMENTS:
        count = patched.count(old)
        if count != 1:
            raise RuntimeError(
                "AMG pinned Qwen tool-template clause drifted: "
                f"expected one occurrence, found {count}: {old!r}"
            )
        patched = patched.replace(old, new, 1)
    return patched


def _native_tool_observation_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    expected_assistant_messages: Sequence[Mapping[str, Any]],
    tool_name: str,
    observation: str,
) -> list[dict[str, Any]]:
    """Represent one executed native call result through Qwen's tool role."""

    normalized = [dict(message) for message in messages]
    assistant_prefix = [dict(message) for message in expected_assistant_messages]
    if (
        len(normalized) != len(assistant_prefix) + 1
        or normalized[:-1] != assistant_prefix
        or normalized[-1].get("role") != "user"
        or normalized[-1].get("content") != observation
    ):
        raise RuntimeError(
            "AMG native tool lifecycle must append exactly one observation after "
            "the sampled assistant call"
        )
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("AMG native tool observation requires a tool name")
    normalized[-1] = {
        "role": "tool",
        "name": tool_name,
        "content": observation,
    }
    return normalized


class _SampleExcluded(Exception):
    """Unwind one infra-faulted environment attempt without emitting PPO rows."""

    def __init__(self, summary: str) -> None:
        super().__init__(summary)
        self.summary = summary


def _sample_exclusion_summary(step_output: Any) -> str:
    payload = {
        "state": _json_safe(getattr(step_output, "state", None)),
        "info": _json_safe(getattr(step_output, "info", {})),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return encoded[:2048]


def _receipt_parts(
    step_output: Any, action: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    info = getattr(step_output, "info", {})
    info = dict(info) if isinstance(info, Mapping) else {}
    env_info = info.get("env_info", info.get("info", {}))
    action_submission = info.get("action_submission", {"raw_policy_output": action})
    context_transition = info.get("context_transition", {})
    wrapper_evidence = info.get("wrapper_evidence", {})
    return (
        deepcopy(dict(env_info)) if isinstance(env_info, Mapping) else {},
        deepcopy(dict(action_submission))
        if isinstance(action_submission, Mapping)
        else {},
        {
            "context_transition": deepcopy(dict(context_transition))
            if isinstance(context_transition, Mapping)
            else {},
            "wrapper_evidence": deepcopy(dict(wrapper_evidence))
            if isinstance(wrapper_evidence, Mapping)
            else {},
        },
    )


def _outcome(
    *,
    done: bool,
    reward: float,
    env_info: Mapping[str, Any],
    wrapper_evidence: Mapping[str, Any],
) -> str:
    if not done:
        return "continue"
    explicit = wrapper_evidence.get("outcome")
    if explicit in {"success", "terminal_failure", "environment_error"}:
        return str(explicit)
    if env_info.get("episode_success") is True or env_info.get("resolved") is True:
        return "success"
    return "success" if reward > 0 else "terminal_failure"


class AMGTaskNeutralAgentLoop(AgentLoopBase):
    """Emit one upstream ``AgentLoopOutput`` per ordinary AMG policy action.

    Environment wrappers retain all lifecycle, parser, reward, compaction, and
    filesystem semantics.  This class only samples an action, sends it through
    ``env.step``, mechanically applies the task-neutral context receipt, and
    records the exact action row for upstream PPO.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.agentgym_config = self.config.actor_rollout_ref.get("agentgym")
        if self.agentgym_config is None:
            raise ValueError(
                "AMGTaskNeutralAgentLoop requires actor_rollout_ref.agentgym"
            )
        self._route_registry = route_registry_from_agentgym_config(
            self.agentgym_config
        )
        self._envelope_tokens_by_tool_schema: dict[str, int] = {}
        self._typed_tool_schemas_by_digest: dict[
            str, tuple[OpenAIFunctionToolSchema, ...]
        ] = {}
        self._native_tool_parser = ToolParser.get_tool_parser(
            "qwen3_coder", self.tokenizer
        )
        self._native_tool_template_sha256: str | None = None

    def _activate_native_tool_template(
        self, tools: Sequence[Mapping[str, Any]] | None
    ) -> None:
        if not tools or self._native_tool_template_sha256 is not None:
            return
        builder_kwargs = dict(self.continuous_token_builder.chat_template_kwargs)
        source_template = builder_kwargs.get(
            "chat_template", getattr(self.tokenizer, "chat_template", None)
        )
        strict_template = _strict_qwen_tool_chat_template(source_template)
        builder_kwargs["chat_template"] = strict_template
        self.continuous_token_builder.chat_template_kwargs = builder_kwargs
        self._native_tool_template_sha256 = hashlib.sha256(
            strict_template.encode("utf-8")
        ).hexdigest()

    def _typed_tool_schemas(
        self, tools: Sequence[Mapping[str, Any]]
    ) -> tuple[OpenAIFunctionToolSchema, ...]:
        digest = _tool_schema_digest(tools)
        typed = self._typed_tool_schemas_by_digest.get(digest)
        if typed is None:
            typed = tuple(
                OpenAIFunctionToolSchema.model_validate(dict(schema))
                for schema in tools
            )
            self._typed_tool_schemas_by_digest[digest] = typed
        return typed

    async def _validated_native_tool_call(
        self,
        *,
        response_ids: list[int],
        action: str,
        tools: Sequence[Mapping[str, Any]] | None,
        action_submission: Mapping[str, Any],
    ) -> FunctionCall | None:
        # A parser-rejected output did not execute a tool. Keep its diagnostic
        # observation as a user correction instead of forging a tool response.
        if action_submission.get("tool_parser_normalized") is not True:
            return None
        if not tools:
            raise RuntimeError(
                "AMG wrapper accepted a native tool call without tool schemas"
            )
        stripped = action.strip()
        if (
            not stripped.startswith("<tool_call>")
            or not stripped.endswith("</tool_call>")
            or stripped.count("<tool_call>") != 1
            or stripped.count("</tool_call>") != 1
        ):
            raise RuntimeError(
                "AMG wrapper accepted output outside the strict native tool envelope"
            )
        content, calls = await self._native_tool_parser.extract_tool_calls(
            response_ids, list(self._typed_tool_schemas(tools))
        )
        if content.strip() or len(calls) != 1:
            raise RuntimeError(
                "AMG wrapper/native veRL parser disagreement for an accepted action"
            )
        allowed_names = {
            str(schema["function"]["name"])
            for schema in tools
        }
        if calls[0].name not in allowed_names:
            raise RuntimeError(
                "AMG wrapper accepted an action outside the native tool schema: "
                f"{calls[0].name!r}"
            )
        return calls[0]

    def _render_prompt_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # Latest veRL always routes AgentLoop tokenization through its native
        # Continuous Token builder. Qwen3.5 selection is inferred from the
        # root Hugging Face model_type by AgentLoopWorker.
        return normalize_token_ids(
            self.continuous_token_builder.build_initial_tokens(messages, tools=tools)
        )

    def _prompt_for_candidate(
        self,
        current_messages: list[dict[str, Any]],
        current_prompt_ids: list[int],
        candidate_messages: Sequence[Mapping[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        candidate = [dict(message) for message in candidate_messages]
        if candidate == current_messages:
            return list(current_prompt_ids)
        # ``current_prompt_ids`` already ends in an assistant-generation marker.
        # A wrapper-selected control request is inserted before any assistant
        # generation, so it is not an append to the runtime token stream.  Render
        # that candidate as a fresh prompt; Continuous Token remains responsible
        # for the ordinary sampled-assistant -> observation append below.
        return self._render_prompt_sync(candidate, tools=tools)

    def _next_prompt_ids(
        self,
        *,
        prepared_messages: list[dict[str, Any]],
        prepared_prompt_ids: list[int],
        action: str,
        action_token_ids: list[int],
        next_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        assistant_messages = prepared_messages + [
            {"role": "assistant", "content": action}
        ]
        if _is_prefix(assistant_messages, next_messages) and len(next_messages) > len(
            assistant_messages
        ):
            appended = next_messages[len(assistant_messages) :]
            if all(message.get("role") != "assistant" for message in appended):
                assistant_runtime = (
                    self.continuous_token_builder.merge_assistant_tokens(
                        prepared_prompt_ids, action_token_ids
                    ).token_ids
                )
                next_prompt = normalize_token_ids(
                    self.continuous_token_builder.merge_non_assistant_tokens(
                        assistant_messages,
                        next_messages,
                        assistant_runtime,
                        tools=tools,
                    ).token_ids
                )
                return next_prompt

        # replace_messages and preserve-without-observation deliberately rebuild
        # through upstream Continuous Token rather than inventing a second merge API.
        return self._render_prompt_sync(next_messages, tools=tools)

    def _action_observation_envelope_tokens(
        self,
        messages: list[dict[str, Any]],
        prompt_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        schema_digest = _tool_schema_digest(tools)
        envelope_tokens = self._envelope_tokens_by_tool_schema.get(schema_digest)
        if envelope_tokens is None:
            observation_message: dict[str, Any] = {
                "role": "user",
                "content": "",
            }
            if tools:
                observation_message = {
                    "role": "tool",
                    "name": str(tools[0]["function"]["name"]),
                    "content": "",
                }
            empty_transition = messages + [
                {"role": "assistant", "content": ""},
                observation_message,
            ]
            envelope_tokens = len(
                self._render_prompt_sync(empty_transition, tools=tools)
            ) - len(prompt_ids)
            if envelope_tokens < 0:
                raise RuntimeError(
                    "AMG chat template shortened across an empty action/observation turn"
                )
            self._envelope_tokens_by_tool_schema[schema_digest] = envelope_tokens
        return envelope_tokens

    @staticmethod
    def _resolve_data_idx(kwargs: Mapping[str, Any]) -> int:
        value = kwargs.get("data_idx")
        if value is None:
            extra_info = kwargs.get("extra_info")
            if isinstance(extra_info, Mapping):
                value = extra_info.get("index")
        if value is None:
            value = kwargs.get("index")
        value = _scalar(value)
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("AMG data_idx must be an integer, not bool")
        try:
            data_idx = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"AMG data_idx is not an integer: {value!r}") from exc
        if data_idx < 0:
            raise ValueError(f"AMG data_idx must be non-negative, got {data_idx}")
        return data_idx

    @staticmethod
    def _resolve_trajectory_uid(kwargs: Mapping[str, Any]) -> str:
        raw_uid = _scalar(kwargs.get("uid"))
        if raw_uid is not None and str(raw_uid):
            return str(raw_uid)
        item_id = _scalar(kwargs.get("item_id"))
        return f"amg-{item_id}-{uuid4().hex}"

    async def run(
        self,
        sampling_params: dict[str, Any],
        priority: int = 0,
        **kwargs: Any,
    ) -> list[AgentLoopOutput]:
        max_retries = int(_get(self.agentgym_config, "max_retries", 2))
        if max_retries < 0:
            raise ValueError("AMG max_retries must be non-negative")

        route = self._route_registry.resolve_row(kwargs)
        attempt_kwargs = dict(kwargs)
        attempt_kwargs["uid"] = self._resolve_trajectory_uid(kwargs)
        data_idx = self._resolve_data_idx(kwargs)
        item_id = str(_scalar(kwargs.get("item_id", data_idx)))
        last_exclusion: _SampleExcluded | None = None
        for sample_reschedule_attempt in range(max_retries + 1):
            try:
                return await self._run_single_attempt(
                    sampling_params,
                    priority,
                    route=route,
                    sample_reschedule_attempt=sample_reschedule_attempt,
                    **attempt_kwargs,
                )
            except _SampleExcluded as exc:
                last_exclusion = exc

        assert last_exclusion is not None
        raise RuntimeError(
            f"AMG sample item_id={item_id!r} data_idx={data_idx} was excluded after "
            f"{max_retries + 1} complete trajectory attempts: "
            f"{last_exclusion.summary}"
        ) from last_exclusion

    async def _run_single_attempt(
        self,
        sampling_params: dict[str, Any],
        priority: int = 0,
        *,
        route: RouteSpec,
        sample_reschedule_attempt: int,
        **kwargs: Any,
    ) -> list[AgentLoopOutput]:
        priority = int(_scalar(priority))
        data_idx = self._resolve_data_idx(kwargs)
        item_id = str(_scalar(kwargs.get("item_id", data_idx)))
        trajectory_uid = self._resolve_trajectory_uid(kwargs)
        raw_prompt = _messages(kwargs["raw_prompt"])

        response_budget = int(
            sampling_params.get("max_tokens", self.rollout_config.response_length)
        )
        if response_budget <= 0 or response_budget > int(
            self.rollout_config.response_length
        ):
            raise ValueError(
                "AMG sampling max_tokens must fit rollout.response_length: "
                f"sampling={response_budget} width={self.rollout_config.response_length}"
            )
        max_prompt_tokens = int(self.rollout_config.prompt_length)
        max_model_tokens = int(
            getattr(
                self.rollout_config,
                "max_model_len",
                max_prompt_tokens + int(self.rollout_config.response_length),
            )
        )
        sampling_params = dict(sampling_params)
        sampling_params["max_tokens"] = response_budget
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("AMG AgentLoop tokenizer must define eos_token_id")
        existing_stop_ids = sampling_params.get("stop_token_ids") or []
        if not isinstance(existing_stop_ids, (list, tuple)):
            raise TypeError("sampling_params.stop_token_ids must be a list or tuple")
        eos_stop_ids = (
            eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
        )
        merged_stop_ids: list[int] = []
        seen_stop_ids: set[int] = set()
        for token_id in [*existing_stop_ids, *eos_stop_ids]:
            normalized_token_id = int(token_id)
            if normalized_token_id not in seen_stop_ids:
                seen_stop_ids.add(normalized_token_id)
                merged_stop_ids.append(normalized_token_id)
        sampling_params["stop_token_ids"] = merged_stop_ids
        if not bool(self.rollout_config.calculate_log_probs):
            raise ValueError(
                "AMG fully-async PPO requires rollout.calculate_log_probs=true"
            )

        client = create_env_client(route.client_config)
        outputs: list[AgentLoopOutput] = []
        rows: list[dict[str, Any]] = []
        try:
            client.reset(data_idx)
            policy_tools = _policy_tool_schemas(client)
            self._activate_native_tool_template(policy_tools)
            initial_messages = raw_prompt + [
                {"role": "user", "content": str(client.observe())}
            ]
            current_messages = bind_initial_policy_context(client, initial_messages)
            current_prompt_ids = self._render_prompt_sync(
                current_messages, tools=policy_tools
            )

            for row_order in range(route.max_rounds):
                if len(current_prompt_ids) > max_prompt_tokens:
                    raise RuntimeError(
                        "AMG prompt exceeded PPO width before a trainable compaction action: "
                        f"item_id={item_id!r} data_idx={data_idx} row_order={row_order} "
                        f"tokens={len(current_prompt_ids)} width={max_prompt_tokens}"
                    )

                prepared = prepare_policy_turn(
                    client,
                    current_messages,
                    count_prompt_tokens=lambda candidate, messages=current_messages, prompt_ids=current_prompt_ids: (
                        len(
                            self._prompt_for_candidate(
                                messages,
                                prompt_ids,
                                candidate,
                                tools=policy_tools,
                            )
                        )
                    ),
                    max_prompt_tokens=max_prompt_tokens,
                    max_model_tokens=max_model_tokens,
                    max_response_tokens=response_budget,
                    max_observation_tokens=route.max_observation_tokens,
                    action_observation_envelope_tokens=self._action_observation_envelope_tokens(
                        current_messages,
                        current_prompt_ids,
                        tools=policy_tools,
                    ),
                )
                prepared_messages = [dict(message) for message in prepared.messages]
                prompt_ids = self._prompt_for_candidate(
                    current_messages,
                    current_prompt_ids,
                    prepared_messages,
                    tools=policy_tools,
                )
                if len(prompt_ids) != prepared.prompt_token_count:
                    raise RuntimeError(
                        "AMG wrapper prompt-token count differs from sampled prompt: "
                        f"wrapper={prepared.prompt_token_count} sampled={len(prompt_ids)}"
                    )
                if len(prompt_ids) > max_prompt_tokens:
                    raise RuntimeError(
                        "AMG wrapper control prompt exceeds PPO prompt width: "
                        f"item_id={item_id!r} data_idx={data_idx} row_order={row_order} "
                        f"tokens={len(prompt_ids)} width={max_prompt_tokens}"
                    )

                request_id = f"{trajectory_uid}-row-{row_order}-{uuid4().hex[:8]}"
                started = time.perf_counter()
                token_output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    priority=priority,
                )
                elapsed = time.perf_counter() - started
                response_ids = [int(token_id) for token_id in token_output.token_ids]
                if not response_ids:
                    raise RuntimeError(
                        "AMG policy generation returned no response tokens"
                    )
                if len(response_ids) > response_budget:
                    raise RuntimeError(
                        "AMG policy generation exceeded the configured response budget: "
                        f"tokens={len(response_ids)} budget={response_budget}"
                    )
                if token_output.log_probs is None or len(token_output.log_probs) != len(
                    response_ids
                ):
                    raise RuntimeError(
                        "AMG PPO requires one rollout behavior logprob per sampled response token"
                    )
                response_logprobs = [float(value) for value in token_output.log_probs]
                action = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                step_output, next_messages = complete_policy_turn(
                    client, prepared, action
                )
                if getattr(client, "sample_excluded", False):
                    if not bool(step_output.done):
                        raise RuntimeError(
                            "AMG sample_excluded requires a terminal environment step"
                        )
                    raise _SampleExcluded(_sample_exclusion_summary(step_output))
                next_messages = [dict(message) for message in next_messages]
                reward = float(step_output.reward)
                done = bool(step_output.done)
                env_info, action_submission, receipt = _receipt_parts(
                    step_output, action
                )
                context_transition = receipt["context_transition"]
                wrapper_evidence = receipt["wrapper_evidence"]
                native_tool_call = await self._validated_native_tool_call(
                    response_ids=response_ids,
                    action=action,
                    tools=policy_tools,
                    action_submission=action_submission,
                )
                native_tool_name = (
                    native_tool_call.name if native_tool_call is not None else None
                )
                native_tool_observation = False
                if (
                    native_tool_name is not None
                    and context_transition.get("operation") == "append_observation"
                    and not done
                ):
                    next_messages = _native_tool_observation_messages(
                        next_messages,
                        expected_assistant_messages=prepared_messages
                        + [{"role": "assistant", "content": action}],
                        tool_name=native_tool_name,
                        observation=str(step_output.state),
                    )
                    native_tool_observation = True

                token_extra = dict(token_output.extra_fields or {})
                if (
                    token_extra.get("min_global_steps") is None
                    or token_extra.get("max_global_steps") is None
                ):
                    raise RuntimeError(
                        "AMG fully-async generation is missing upstream policy-version metadata"
                    )
                metrics = AgentLoopMetrics(
                    generate_sequences=elapsed,
                    num_preempted=(
                        int(token_output.num_preempted)
                        if token_output.num_preempted is not None
                        else -1
                    ),
                )
                row_uid = f"{trajectory_uid}-row-{row_order}"
                row = {
                    "schema": "amg_task_neutral_action_row_v1",
                    "route_id": route.route_id,
                    "data_source": route.route_id,
                    "item_id": item_id,
                    "data_idx": data_idx,
                    "trajectory_uid": trajectory_uid,
                    "trajectory_row_uid": row_uid,
                    "trajectory_row_order": row_order,
                    "sample_reschedule_attempt": sample_reschedule_attempt,
                    "trajectory_terminal": False,
                    "rollout_done_flag": done,
                    "immediate_reward": reward,
                    "trajectory_return": 0.0,
                    "task_round": row_order + 1,
                    "action": action,
                    "action_submission": action_submission,
                    "env_info_after": env_info,
                    "context_transition": context_transition,
                    "wrapper_evidence": wrapper_evidence,
                    "control_request": prepared.control_request,
                    "outcome": _outcome(
                        done=done,
                        reward=reward,
                        env_info=env_info,
                        wrapper_evidence=wrapper_evidence,
                    ),
                    "prompt_token_count": len(prompt_ids),
                    "prompt_token_sha256": _digest_token_ids(prompt_ids),
                    "response_token_count": len(response_ids),
                    "response_token_sha256": _digest_token_ids(response_ids),
                    "generation_stop_reason": _json_safe(token_output.stop_reason),
                    "native_tool_call_name": native_tool_name,
                    "native_tool_observation": native_tool_observation,
                    "native_tool_template_sha256": self._native_tool_template_sha256,
                    "min_global_steps": int(token_extra["min_global_steps"]),
                    "max_global_steps": int(token_extra["max_global_steps"]),
                }
                rows.append(row)
                token_extra.update(
                    {
                        "route_id": route.route_id,
                        "data_source": route.route_id,
                        "trajectory_uid": trajectory_uid,
                        "trajectory_row_uid": row_uid,
                        "trajectory_row_order": row_order,
                        "sample_reschedule_attempt": sample_reschedule_attempt,
                        "trajectory_terminal": False,
                        "rollout_done_flag": done,
                        "immediate_reward": reward,
                        "trajectory_return": 0.0,
                        "item_id": item_id,
                        "data_idx": data_idx,
                        "task_round": row_order + 1,
                        "action_text": action,
                        "generation_stop_reason": _json_safe(token_output.stop_reason),
                        "native_tool_call_name": native_tool_name,
                        "native_tool_observation": native_tool_observation,
                        "native_tool_template_sha256": self._native_tool_template_sha256,
                        "context_transition": context_transition,
                        "wrapper_evidence": wrapper_evidence,
                        "env_info_after": env_info,
                        "control_request": prepared.control_request,
                        "outcome": row["outcome"],
                        "horizon_finalization": None,
                        "turn_scores": [reward],
                        "tool_rewards": [],
                    }
                )
                outputs.append(
                    AgentLoopOutput(
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        response_mask=[1] * len(response_ids),
                        response_logprobs=response_logprobs,
                        routed_experts=token_output.routed_experts,
                        reward_score=reward,
                        num_turns=row_order + 1,
                        metrics=metrics,
                        extra_fields=token_extra,
                    )
                )

                if done:
                    break
                current_prompt_ids = self._next_prompt_ids(
                    prepared_messages=prepared_messages,
                    prepared_prompt_ids=prompt_ids,
                    action=action,
                    action_token_ids=response_ids,
                    next_messages=next_messages,
                    tools=policy_tools,
                )
                current_messages = next_messages

            if not outputs:
                raise RuntimeError("AMG AgentLoop produced no trainable policy action")

            if not rows[-1]["rollout_done_flag"]:
                finalizer = getattr(client, "finalize_policy_horizon", None)
                horizon_output = finalizer() if callable(finalizer) else None
                if horizon_output is not None:
                    if getattr(client, "sample_excluded", False):
                        if not bool(horizon_output.done):
                            raise RuntimeError(
                                "AMG sample_excluded requires terminal horizon finalization"
                            )
                        raise _SampleExcluded(
                            _sample_exclusion_summary(horizon_output)
                        )
                    if not bool(horizon_output.done):
                        raise RuntimeError(
                            "AMG horizon finalization must terminate the episode"
                        )
                    horizon_reward = float(horizon_output.reward)
                    horizon_env_info, horizon_action_submission, horizon_receipt = (
                        _receipt_parts(horizon_output, "")
                    )
                    horizon_context_transition = horizon_receipt["context_transition"]
                    horizon_wrapper_evidence = horizon_receipt["wrapper_evidence"]
                    rows[-1]["immediate_reward"] += horizon_reward
                    rows[-1]["rollout_done_flag"] = True
                    rows[-1]["outcome"] = _outcome(
                        done=True,
                        reward=float(rows[-1]["immediate_reward"]),
                        env_info=horizon_env_info,
                        wrapper_evidence=horizon_wrapper_evidence,
                    )
                    horizon_finalization = _json_safe(
                        {
                            "state": horizon_output.state,
                            "reward": horizon_reward,
                            "done": True,
                            "info": horizon_output.info,
                            "env_info": horizon_env_info,
                            "action_submission": horizon_action_submission,
                            "context_transition": horizon_context_transition,
                            "wrapper_evidence": horizon_wrapper_evidence,
                        }
                    )
                    rows[-1]["horizon_finalization"] = horizon_finalization
                    outputs[-1].reward_score = float(rows[-1]["immediate_reward"])
                    outputs[-1].extra_fields.update(
                        {
                            "immediate_reward": float(rows[-1]["immediate_reward"]),
                            "rollout_done_flag": True,
                            "outcome": rows[-1]["outcome"],
                            "horizon_finalization": horizon_finalization,
                        }
                    )
                else:
                    rows[-1]["outcome"] = "max_rounds"
                    outputs[-1].extra_fields["outcome"] = "max_rounds"

            trajectory_return = sum(float(row["immediate_reward"]) for row in rows)
            for index, (row, output) in enumerate(zip(rows, outputs, strict=True)):
                terminal = index == len(rows) - 1
                row["trajectory_terminal"] = terminal
                row["trajectory_return"] = trajectory_return
                output.extra_fields["trajectory_terminal"] = terminal
                output.extra_fields["trajectory_return"] = trajectory_return
                step_record_json = json.dumps(
                    _json_safe(row),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                output.extra_fields["step_record_json"] = step_record_json
                # veRL already persists fields listed under reward_extra_info in
                # trainer.rollout_data_dir.  Reuse that native path rather than
                # adding an AMG dumper or trainer hook.  Padding utilities replace
                # the flattened is_padding value for synthetic rows after dequeue.
                output.extra_fields["reward_extra_info"] = {
                    "step_record_json": step_record_json,
                    "is_padding": False,
                }
            return outputs
        finally:
            client.close()
