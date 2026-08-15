"""Task-neutral controller adapters for the wrapper-owned policy-turn contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence, Tuple

from .contracts import (
    BudgetConfig,
    CompletedPolicyTurn,
    CONTEXT_OPERATION_APPEND,
    CONTEXT_OPERATION_PRESERVE,
    CONTEXT_OPERATION_REPLACE,
    CONTEXT_OPERATIONS,
    CONTEXT_TRANSITION_SCHEMA,
    PolicyContextPressure,
    PreparedPolicyTurn,
    StepOutput,
)


class PolicyControllerError(RuntimeError):
    pass


class EnvironmentStepError(RuntimeError):
    pass


def normalize_messages(
    messages: Sequence[Mapping[str, str]],
) -> Tuple[Mapping[str, str], ...]:
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"policy message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"policy message {index} has invalid role")
        if not isinstance(content, str):
            raise TypeError(f"policy message {index} content must be text")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("policy context must not be empty")
    return tuple(normalized)


def coerce_step_output(value: Any) -> StepOutput:
    try:
        state = value.state
        reward = value.reward
        done = value.done
        info = value.info
    except AttributeError as exc:
        raise PolicyControllerError(
            "environment returned an invalid step object"
        ) from exc
    return StepOutput(state=state, reward=reward, done=done, info=info)


def context_transition_from_step(step_output: StepOutput) -> dict[str, Any]:
    transition = step_output.info.get("context_transition")
    if not transition:
        return {
            "schema": CONTEXT_TRANSITION_SCHEMA,
            "operation": CONTEXT_OPERATION_APPEND,
            "messages": [],
        }
    if not isinstance(transition, Mapping):
        raise PolicyControllerError("context transition must be a mapping")
    if transition.get("schema") != CONTEXT_TRANSITION_SCHEMA:
        raise PolicyControllerError("context transition schema is unsupported")
    operation = transition.get("operation")
    if operation not in CONTEXT_OPERATIONS:
        raise PolicyControllerError("context transition operation is unsupported")
    replacement = transition.get("messages", ())
    if operation == CONTEXT_OPERATION_REPLACE:
        normalized = normalize_messages(replacement)
        messages = [dict(message) for message in normalized]
    else:
        if replacement:
            raise PolicyControllerError(
                "non-replacement context transition carried messages"
            )
        messages = []
    return {
        "schema": CONTEXT_TRANSITION_SCHEMA,
        "operation": operation,
        "messages": messages,
    }


class DependencyLightPolicyTurnController:
    """Stdlib-only semantic equivalent used by tests and thin clients."""

    def bind_initial(self, client, messages):
        try:
            normalize = getattr(client, "normalize_initial_policy_context", None)
            prepared = normalize(deepcopy(list(messages))) if normalize else messages
            normalized = normalize_messages(prepared)
            bind = getattr(client, "bind_policy_context", None)
            if bind is not None:
                bind(deepcopy(list(normalized)), initial=True)
            return normalized
        except Exception as exc:
            raise PolicyControllerError(
                "failed to bind initial policy context"
            ) from exc

    def prepare(
        self,
        client,
        messages,
        *,
        count_prompt_tokens,
        budgets: BudgetConfig,
        max_response_tokens: int,
    ) -> PreparedPolicyTurn:
        try:
            action_messages = normalize_messages(messages)
            bind = getattr(client, "bind_policy_context", None)
            if bind is not None:
                bind(deepcopy(list(action_messages)), initial=False)
            action_count = int(count_prompt_tokens(action_messages))
            candidate_method = getattr(client, "policy_turn_candidate", None)
            candidate = candidate_method() if candidate_method is not None else None
            if candidate is None:
                return PreparedPolicyTurn(
                    messages=action_messages,
                    prompt_token_count=action_count,
                    control_request=None,
                )
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("policy turn candidate must be nonempty text")
            candidate_messages = normalize_messages(
                list(action_messages) + [{"role": "user", "content": candidate}]
            )
            candidate_count = int(count_prompt_tokens(candidate_messages))
            pressure = PolicyContextPressure(
                action_prompt_tokens=action_count,
                candidate_prompt_tokens=candidate_count,
                max_prompt_tokens=budgets.max_prompt_tokens,
                max_model_tokens=budgets.max_model_tokens,
                max_response_tokens=max_response_tokens,
                max_observation_tokens=budgets.max_observation_tokens,
                action_observation_envelope_tokens=(
                    budgets.action_observation_envelope_tokens
                ),
            )
            select = getattr(client, "prepare_policy_turn", None)
            selected = select(pressure) if select is not None else None
            if selected is None:
                return PreparedPolicyTurn(
                    messages=action_messages,
                    prompt_token_count=action_count,
                    control_request=None,
                )
            if selected != candidate:
                raise ValueError("wrapper selected a different control request")
            if bind is not None:
                bind(deepcopy(list(candidate_messages)), initial=False)
            return PreparedPolicyTurn(
                messages=candidate_messages,
                prompt_token_count=candidate_count,
                control_request=selected,
            )
        except Exception as exc:
            if isinstance(exc, PolicyControllerError):
                raise
            raise PolicyControllerError("failed to prepare policy turn") from exc

    def complete(
        self,
        client,
        prepared: PreparedPolicyTurn,
        policy_output: str,
    ) -> CompletedPolicyTurn:
        if not isinstance(policy_output, str):
            raise TypeError("policy output must be text")
        try:
            raw_step = client.step(policy_output)
        except Exception as exc:
            raise EnvironmentStepError("environment step failed") from exc
        step_output = coerce_step_output(raw_step)
        transition = context_transition_from_step(step_output)
        messages = [dict(message) for message in prepared.messages]
        messages.append({"role": "assistant", "content": policy_output})
        operation = transition["operation"]
        if operation == CONTEXT_OPERATION_APPEND:
            messages.append({"role": "user", "content": step_output.state})
        elif operation == CONTEXT_OPERATION_REPLACE:
            messages = [dict(message) for message in transition["messages"]]
        elif operation != CONTEXT_OPERATION_PRESERVE:
            raise PolicyControllerError("unreachable context operation")
        return CompletedPolicyTurn(
            step_output=step_output,
            messages=normalize_messages(messages),
            context_transition=transition,
        )


class AgentGymPolicyTurnController:
    """Integration adapter that calls the exact AgentGym controller functions."""

    def __init__(
        self,
        *,
        bind_initial_policy_context: Callable[..., Any],
        prepare_policy_turn: Callable[..., Any],
        complete_policy_turn: Callable[..., Any],
    ) -> None:
        self.bind_initial_policy_context = bind_initial_policy_context
        self.prepare_policy_turn = prepare_policy_turn
        self.complete_policy_turn = complete_policy_turn

    @classmethod
    def from_agentenv(cls) -> "AgentGymPolicyTurnController":
        from agentenv.controller import (  # type: ignore
            bind_initial_policy_context,
            complete_policy_turn,
            prepare_policy_turn,
        )

        return cls(
            bind_initial_policy_context=bind_initial_policy_context,
            prepare_policy_turn=prepare_policy_turn,
            complete_policy_turn=complete_policy_turn,
        )

    def bind_initial(self, client, messages):
        try:
            result = self.bind_initial_policy_context(client, messages)
            return normalize_messages(result)
        except Exception as exc:
            raise PolicyControllerError(
                "AgentGym initial context binding failed"
            ) from exc

    def prepare(
        self,
        client,
        messages,
        *,
        count_prompt_tokens,
        budgets: BudgetConfig,
        max_response_tokens: int,
    ) -> PreparedPolicyTurn:
        try:
            native = self.prepare_policy_turn(
                client,
                messages,
                count_prompt_tokens=count_prompt_tokens,
                max_prompt_tokens=budgets.max_prompt_tokens,
                max_model_tokens=budgets.max_model_tokens,
                max_response_tokens=max_response_tokens,
                max_observation_tokens=budgets.max_observation_tokens,
                action_observation_envelope_tokens=(
                    budgets.action_observation_envelope_tokens
                ),
            )
            return PreparedPolicyTurn(
                messages=normalize_messages(native.messages),
                prompt_token_count=int(native.prompt_token_count),
                control_request=native.control_request,
                opaque=native,
            )
        except Exception as exc:
            raise PolicyControllerError(
                "AgentGym policy turn preparation failed"
            ) from exc

    def complete(self, client, prepared, policy_output):
        native_prepared = prepared.opaque
        if native_prepared is None:
            raise PolicyControllerError("exact controller prepared state is missing")
        try:
            raw_step, messages = self.complete_policy_turn(
                client,
                native_prepared,
                policy_output,
            )
        except Exception as exc:
            raise EnvironmentStepError(
                "AgentGym policy turn completion failed"
            ) from exc
        step_output = coerce_step_output(raw_step)
        return CompletedPolicyTurn(
            step_output=step_output,
            messages=normalize_messages(messages),
            context_transition=context_transition_from_step(step_output),
        )
