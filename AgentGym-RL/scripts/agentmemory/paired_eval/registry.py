"""One fail-closed registry joining frozen clients to the shared runner.

Benchmark selection, client construction, artifact extraction, and grader handoff
live here, outside :class:`PairedRunner`.  The client proxy translates only
wrapper-owned structured receipts; it never interprets a policy action language.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from .contracts import (
    ACTION_ACCOUNTING_COUNTERS,
    AdapterClose,
    AdapterReset,
    ArtifactResult,
    BENCHMARK_ROUTE,
    CapabilityRoot,
    CONTEXT_OPERATION_REPLACE,
    EXTERNAL_MEMORY_ROUTE,
    FinalizationContext,
    Namespace,
    POLICY_COMPACTION_ROUTE,
    RunConfig,
    ScorerResult,
    StepOutput,
    TASK_NEUTRAL_RECEIPT_SCHEMA,
    WRAPPER_EXECUTION_SCHEMA,
    capability_root_id,
)
from .evidence import PrivateEvidenceStore
from .manifest import RuntimeBindings
from .serialization import sha256_json, sha256_text


FROZEN_ARMS = ("native", "amg_compaction_only", "amg_memory")
MemoryOperationResolver = Callable[[Mapping[str, Any]], Optional[str]]
RuntimeBuilder = Callable[..., RuntimeBindings]
ArtifactFinalizer = Callable[
    [Any, FinalizationContext, RunConfig, PrivateEvidenceStore], ArtifactResult
]
GraderHandoff = Callable[
    [Any, ArtifactResult, RunConfig, PrivateEvidenceStore], ScorerResult
]


def _require_label(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty normalized text")
    return value


def _normalized_action_accounting_delta(
    value: Any,
    *,
    execution_kind: str,
) -> dict[str, int]:
    if value is None:
        delta = {name: 0 for name in ACTION_ACCOUNTING_COUNTERS}
        if execution_kind == "policy_compaction":
            delta["compactions"] = 1
        elif execution_kind == "external_memory_action":
            delta["parsed_actions"] = 1
            delta["workspace_actions"] = 1
        else:
            # Legacy task-neutral adapters do not expose domain-level parsing.
            # Their benchmark dispatch is conservatively one successful tool call.
            delta["parsed_actions"] = 1
            delta["domain_tool_attempts"] = 1
            delta["successful_backend_calls"] = 1
        return delta
    if not isinstance(value, Mapping) or set(value) != set(
        ACTION_ACCOUNTING_COUNTERS
    ):
        raise ValueError("registered client action accounting is not canonical")
    delta = dict(value)
    if any(type(count) is not int or count < 0 for count in delta.values()):
        raise ValueError("registered client action accounting must be non-negative")
    if sum(delta.values()) == 0:
        raise ValueError("ordinary policy output has no accounting classification")
    return delta


def _load_symbol(module_name: str, attribute_name: str) -> Any:
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as error:
        raise ImportError(
            f"{module_name!r} does not export {attribute_name!r}"
        ) from error


def _gaia_memory_operation(info: Mapping[str, Any]) -> Optional[str]:
    env_info = info.get("env_info")
    submission = info.get("action_submission")
    if (
        isinstance(env_info, Mapping)
        and env_info.get("domain_action") == "workspace"
        and isinstance(submission, Mapping)
        and submission.get("kind") == "workspace"
    ):
        return "read_write"
    return None


def _declared_memory_operation(info: Mapping[str, Any]) -> Optional[str]:
    env_info = info.get("env_info")
    if not isinstance(env_info, Mapping):
        return None
    operation = env_info.get("external_memory_operation")
    if operation is None:
        return None
    if operation not in {"read", "write", "read_write"}:
        raise ValueError("wrapper declared an unsupported memory operation")
    return str(operation)


@dataclass(frozen=True)
class AdapterSpec:
    benchmark: str
    client_module: str
    client_name: str
    arm_parameter: str
    artifact_type: str
    native_tools: Tuple[str, ...]
    memory_prompt_module: str
    memory_prompt_name: str
    memory_operation: MemoryOperationResolver
    arms: Tuple[str, ...] = FROZEN_ARMS

    def __post_init__(self) -> None:
        for name in (
            "benchmark",
            "client_module",
            "client_name",
            "arm_parameter",
            "artifact_type",
            "memory_prompt_module",
            "memory_prompt_name",
        ):
            _require_label(name, getattr(self, name))
        object.__setattr__(self, "native_tools", tuple(self.native_tools))
        object.__setattr__(self, "arms", tuple(self.arms))
        if self.arms != FROZEN_ARMS:
            raise ValueError("adapter spec must expose the exact frozen triad")
        if self.arm_parameter not in {"arm", "mode"}:
            raise ValueError("adapter arm parameter must be arm or mode")
        if not self.native_tools or len(set(self.native_tools)) != len(
            self.native_tools
        ):
            raise ValueError("adapter native tools must be nonempty and unique")
        if not callable(self.memory_operation):
            raise TypeError("adapter memory-operation resolver must be callable")

    @property
    def client_specification(self) -> str:
        return f"{self.client_module}:{self.client_name}"

    def resolve_client_type(self) -> type:
        value = _load_symbol(self.client_module, self.client_name)
        if not isinstance(value, type):
            raise TypeError("registered client target must be a class")
        return value

    def memory_prompt_suffix(self) -> str:
        value = _load_symbol(
            self.memory_prompt_module,
            self.memory_prompt_name,
        )
        if not isinstance(value, str) or not value:
            raise TypeError("registered memory prompt suffix must be nonempty text")
        return value

    def validate_config(self, config: RunConfig) -> None:
        if config.task.benchmark != self.benchmark:
            raise ValueError("adapter spec does not match the configured benchmark")
        if config.capability.arm.value not in self.arms:
            raise ValueError("configured arm is unsupported by the adapter")
        if config.task.artifact_type != self.artifact_type:
            raise ValueError("configured artifact type does not match the adapter")
        if tuple(config.task.native_tools) != self.native_tools:
            raise ValueError("configured ordinary tools do not match the adapter")

    def instantiate_client(
        self,
        config: RunConfig,
        *,
        env_server_base: str,
        client_kwargs: Mapping[str, Any],
    ) -> Any:
        self.validate_config(config)
        arguments = dict(client_kwargs)
        if self.arm_parameter in arguments:
            raise ValueError("client kwargs must not override the configured arm")
        arguments[self.arm_parameter] = config.capability.arm.value
        return self.resolve_client_type()(
            env_server_base=env_server_base,
            **arguments,
        )


DEFAULT_ADAPTER_SPECS = (
    AdapterSpec(
        benchmark="gaia_text",
        client_module="agentenv_gaia_text.client",
        client_name="GaiaTextEnvClient",
        arm_parameter="arm",
        artifact_type="answer",
        native_tools=("search", "visit", "answer"),
        memory_prompt_module="agentenv_gaia_text.client",
        memory_prompt_name="GAIA_TEXT_MEMORY_AFFORDANCE",
        memory_operation=_gaia_memory_operation,
    ),
    AdapterSpec(
        benchmark="swebench_verified",
        client_module="agentenv.envs.swebench_verified",
        client_name="SwebenchVerifiedEnvClient",
        arm_parameter="arm",
        artifact_type="patch",
        native_tools=("shell_command", "apply_patch", "final"),
        memory_prompt_module="agentenv.envs.swebench_verified",
        memory_prompt_name="SBV_MEMORY_ADDENDUM",
        memory_operation=_declared_memory_operation,
    ),
    AdapterSpec(
        benchmark="mlebench_lite",
        client_module="agentenv.envs.mlebench_lite",
        client_name="MLEBenchLiteEnvClient",
        arm_parameter="mode",
        artifact_type="submission",
        native_tools=("inspect", "edit", "shell", "submit"),
        memory_prompt_module="agentenv.envs.mlebench_lite",
        memory_prompt_name="MEMORY_POLICY_ADDITION",
        memory_operation=_declared_memory_operation,
    ),
)


def lifecycle_roots(config: RunConfig) -> Tuple[CapabilityRoot, ...]:
    namespace = Namespace.from_config(config)
    return tuple(
        CapabilityRoot(
            capability_id=capability_id,
            root_kind=root_kind,
            root_id=capability_root_id(namespace, capability_id, root_kind),
            namespace_sha256=namespace.sha256,
        )
        for capability_id, root_kind in config.capability.allowed_routes
    )


def _normalize_messages(
    messages: Sequence[Mapping[str, str]],
) -> Tuple[Mapping[str, str], ...]:
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"policy message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"policy message {index} has an invalid role")
        if not isinstance(content, str):
            raise TypeError(f"policy message {index} content must be text")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("policy messages must not be empty")
    return tuple(normalized)


def treatment_excluded_messages(
    spec: AdapterSpec,
    config: RunConfig,
    initial_messages: Sequence[Mapping[str, str]],
) -> Tuple[Mapping[str, str], ...]:
    initial = _normalize_messages(initial_messages)
    if not config.capability.external_read_write_memory:
        return initial
    if initial[0]["role"] != "system":
        raise ValueError("memory-enabled adapter framing must start with system text")
    suffix = spec.memory_prompt_suffix()
    content = initial[0]["content"]
    if not content.endswith(suffix):
        raise ValueError("adapter memory prompt does not end with its frozen suffix")
    base = content[: -len(suffix)]
    if not base:
        raise ValueError("adapter memory prompt suffix consumed the full system prompt")
    if suffix in base:
        raise ValueError("adapter memory prompt suffix must occur exactly once")
    return (
        {"role": "system", "content": base},
        *({"role": item["role"], "content": item["content"]} for item in initial[1:]),
    )


def _validate_reset_response(raw_client: Any, response: Any) -> Mapping[str, Any]:
    if not isinstance(response, Mapping):
        raise TypeError("registered client reset response must be a mapping")
    required_fields = {"observation", "reward", "done", "info"}
    if set(response) not in (required_fields, required_fields | {"state"}):
        raise ValueError("registered client reset response fields are not canonical")
    observation = response.get("observation")
    reward = response.get("reward")
    done = response.get("done")
    info = response.get("info")
    if not isinstance(observation, str):
        raise TypeError("registered client reset observation must be text")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
    ):
        raise TypeError("registered client reset reward must be finite numeric")
    if float(reward) != 0.0:
        raise ValueError("registered client reset reward must be zero")
    if done is not False:
        raise ValueError("registered client reset must not be terminal")
    if not isinstance(info, Mapping):
        raise TypeError("registered client reset info must be a mapping")
    if "state" in response and response["state"] != observation:
        raise ValueError("registered client reset state differs from observation")

    client_info = getattr(raw_client, "info", None)
    if not isinstance(client_info, Mapping):
        raise TypeError("registered client did not retain reset response info")
    if dict(client_info) != dict(response):
        raise ValueError("registered client reset info differs from its response")
    observed = raw_client.observe()
    if not isinstance(observed, str) or observed != observation:
        raise ValueError("registered client observation differs from reset response")
    return response


def _validate_close_response(response: Any) -> None:
    if response is True:
        return
    if not isinstance(response, Mapping):
        raise TypeError("registered client close response must attest closure")
    if response.get("closed") is not True:
        raise ValueError("registered client close response did not attest closure")


class ClientStepProxy:
    """Decorate a real client step with the runner's typed neutral receipt."""

    def __init__(
        self,
        raw_client: Any,
        *,
        config: RunConfig,
        spec: AdapterSpec,
        roots: Sequence[CapabilityRoot],
    ) -> None:
        self.raw_client = raw_client
        self.config = config
        self.spec = spec
        self.roots = {root.route: root for root in roots}
        self.policy_steps = 0
        self.native_steps = 0
        self.context_epoch = 0
        self.raw_native_calls = 0
        self.raw_policy_steps = 0
        self.raw_context_epoch = 0
        self.raw_session_epoch = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw_client, name)

    @staticmethod
    def _counter_pair(
        info: Mapping[str, Any],
        name: str,
        *,
        expected_before: int,
        expected_delta: int,
    ) -> Tuple[int, int]:
        before = info.get(f"{name}_before")
        after = info.get(f"{name}_after")
        if (
            isinstance(before, bool)
            or not isinstance(before, int)
            or isinstance(after, bool)
            or not isinstance(after, int)
            or before < 0
            or after < 0
        ):
            raise ValueError(f"registered client {name} receipt is invalid")
        if before != expected_before or after != before + expected_delta:
            raise ValueError(f"registered client {name} receipt drifted")
        return before, after

    def _reconcile_raw_receipt(
        self,
        info: Mapping[str, Any],
        *,
        raw_native_delta: int,
        normalized_native_delta: int,
        context_delta: int,
    ) -> Mapping[str, Any]:
        if info.get("schema") != TASK_NEUTRAL_RECEIPT_SCHEMA:
            raise ValueError("registered client receipt schema drifted")
        if not isinstance(info.get("env_info"), Mapping):
            raise TypeError("registered client env_info must be a mapping")
        if not isinstance(info.get("action_submission"), Mapping):
            raise TypeError("registered client action submission must be a mapping")
        if not isinstance(info.get("wrapper_evidence"), Mapping):
            raise TypeError("registered client wrapper evidence must be a mapping")

        native_step = self._counter_pair(
            info,
            "native_step",
            expected_before=self.raw_native_calls,
            expected_delta=raw_native_delta,
        )
        native_calls = self._counter_pair(
            info,
            "native_call_count",
            expected_before=self.raw_native_calls,
            expected_delta=raw_native_delta,
        )
        if native_step != native_calls:
            raise ValueError("registered client native counter receipts disagree")
        policy_steps = self._counter_pair(
            info,
            "policy_step",
            expected_before=self.raw_policy_steps,
            expected_delta=1,
        )
        context_epochs = self._counter_pair(
            info,
            "context_epoch",
            expected_before=self.raw_context_epoch,
            expected_delta=context_delta,
        )

        session_before = info.get("session_epoch_before")
        session_after = info.get("session_epoch_after")
        if session_before is None and session_after is None:
            session_epochs: Optional[Tuple[int, int]] = None
        else:
            session_epochs = self._counter_pair(
                info,
                "session_epoch",
                expected_before=self.raw_session_epoch,
                expected_delta=0,
            )

        self.raw_native_calls = native_calls[1]
        self.raw_policy_steps = policy_steps[1]
        self.raw_context_epoch = context_epochs[1]
        if session_epochs is not None:
            self.raw_session_epoch = session_epochs[1]
        return {
            "raw_native_call_count_before": native_calls[0],
            "raw_native_call_count_after": native_calls[1],
            "raw_policy_step_before": policy_steps[0],
            "raw_policy_step_after": policy_steps[1],
            "raw_context_epoch_before": context_epochs[0],
            "raw_context_epoch_after": context_epochs[1],
            "normalized_native_call_delta": normalized_native_delta,
        }

    def step(self, policy_output: str) -> StepOutput:
        raw_step = self.raw_client.step(policy_output)
        try:
            state = raw_step.state
            reward = raw_step.reward
            done = raw_step.done
            adapter_info = raw_step.info
        except AttributeError as error:
            raise TypeError("registered client returned an invalid step") from error
        if (
            not isinstance(state, str)
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or type(done) is not bool
            or not isinstance(adapter_info, Mapping)
        ):
            raise TypeError("registered client returned an invalid step contract")
        if not math.isfinite(float(reward)):
            raise ValueError("registered client returned a non-finite reward")
        reward = float(reward)
        transition = adapter_info.get("context_transition")
        if not isinstance(transition, Mapping):
            raise ValueError("registered client omitted its context transition")

        operation = transition.get("operation")
        memory_operation = self.spec.memory_operation(adapter_info)
        if operation == CONTEXT_OPERATION_REPLACE:
            if memory_operation is not None:
                raise ValueError("one step cannot be both compaction and memory")
            if not self.config.capability.policy_authored_compaction:
                raise ValueError("native adapter emitted a compaction receipt")
            route = POLICY_COMPACTION_ROUTE
            execution_kind = "policy_compaction"
            execution_receipt = {
                "summary_sha256": sha256_text(policy_output),
                "trigger_sha256": sha256_text(self.config.compaction.trigger),
                "replacement_context_sha256": sha256_json(
                    transition.get("messages", ())
                ),
            }
            context_after = self.context_epoch + 1
            native_after = self.native_steps
        elif memory_operation is not None:
            if not self.config.capability.external_read_write_memory:
                raise ValueError("memory-disabled adapter emitted a memory receipt")
            route = EXTERNAL_MEMORY_ROUTE
            execution_kind = "external_memory_action"
            execution_receipt = {
                "operation": memory_operation,
                "submission_sha256": sha256_text(policy_output),
                "memory_receipt_sha256": sha256_json(adapter_info),
            }
            context_after = self.context_epoch
            native_after = self.native_steps
        else:
            route = BENCHMARK_ROUTE
            execution_kind = "benchmark_action"
            execution_receipt = {
                "submission_sha256": sha256_text(policy_output),
                "adapter_receipt_sha256": sha256_json(adapter_info),
            }
            context_after = self.context_epoch
            native_after = self.native_steps + 1

        raw_counter_receipt = self._reconcile_raw_receipt(
            adapter_info,
            raw_native_delta=(0 if operation == CONTEXT_OPERATION_REPLACE else 1),
            normalized_native_delta=native_after - self.native_steps,
            context_delta=(1 if operation == CONTEXT_OPERATION_REPLACE else 0),
        )

        root = self.roots.get(route)
        if root is None:
            raise ValueError("wrapper used an undeclared capability route")
        execution = {
            "schema": WRAPPER_EXECUTION_SCHEMA,
            "kind": execution_kind,
            "status": "ok",
            "receipt": execution_receipt,
        }
        adapter_wrapper_evidence = adapter_info.get("wrapper_evidence")
        if not isinstance(adapter_wrapper_evidence, Mapping):
            raise TypeError("registered client wrapper evidence must be a mapping")
        action_accounting_delta = _normalized_action_accounting_delta(
            adapter_wrapper_evidence.get("action_accounting_delta"),
            execution_kind=execution_kind,
        )
        info = {
            "schema": adapter_info.get("schema"),
            "env_info": dict(adapter_info),
            "action_submission": {
                "accepted": True,
                "capability_id": root.capability_id,
                "root_kind": root.root_kind,
                "root_id": root.root_id,
                "namespace_sha256": root.namespace_sha256,
                "policy_output_sha256": sha256_text(policy_output),
                "execution": execution,
                "execution_sha256": sha256_json(execution),
            },
            "native_step_before": self.native_steps,
            "native_step_after": native_after,
            "native_call_count_before": self.native_steps,
            "native_call_count_after": native_after,
            "context_epoch_before": self.context_epoch,
            "context_epoch_after": context_after,
            "session_epoch_before": 0,
            "session_epoch_after": 0,
            "policy_step_before": self.policy_steps,
            "policy_step_after": self.policy_steps + 1,
            "context_transition": dict(transition),
            "wrapper_evidence": {
                "adapter_schema": adapter_info.get("schema"),
                "adapter_receipt_sha256": sha256_json(adapter_info),
                "counter_reconciliation": raw_counter_receipt,
                "action_accounting_delta": action_accounting_delta,
            },
        }
        # The public runner schema intentionally matches the AgentGym transition
        # schema string; retain it explicitly instead of trusting arbitrary input.
        info["schema"] = "agentmemory_task_neutral_transition_v1"
        self.policy_steps += 1
        self.native_steps = native_after
        self.context_epoch = context_after
        return StepOutput(state=state, reward=reward, done=done, info=info)


@dataclass(frozen=True)
class AdapterHooks:
    finalize_artifact: ArtifactFinalizer
    handoff_to_grader: GraderHandoff

    def __post_init__(self) -> None:
        if not callable(self.finalize_artifact) or not callable(
            self.handoff_to_grader
        ):
            raise TypeError("adapter artifact and grader hooks must be callable")


class ClientEnvironmentAdapter:
    """Typed lifecycle bridge around one frozen AgentGym client."""

    def __init__(
        self,
        *,
        config: RunConfig,
        spec: AdapterSpec,
        raw_client: Any,
        hooks: AdapterHooks,
        evidence_store: PrivateEvidenceStore,
    ) -> None:
        spec.validate_config(config)
        if type(raw_client) is not spec.resolve_client_type():
            raise TypeError("registered builder returned the wrong client type")
        if (
            getattr(raw_client, spec.arm_parameter, None)
            != config.capability.arm.value
        ):
            raise ValueError("registered client is bound to the wrong evaluation arm")
        self.config = config
        self.spec = spec
        self.raw_client = raw_client
        self.hooks = hooks
        self.evidence_store = evidence_store
        self.namespace = Namespace.from_config(config)
        self.roots = lifecycle_roots(config)
        self.client = ClientStepProxy(
            raw_client,
            config=config,
            spec=spec,
            roots=self.roots,
        )
        self.reset_complete = False
        self.closed = False

    def reset(self, config: RunConfig) -> AdapterReset:
        if config != self.config:
            raise ValueError("adapter cannot reset with a different run config")
        if self.reset_complete:
            raise RuntimeError("adapter reset may run only once")
        response = _validate_reset_response(
            self.raw_client,
            self.raw_client.reset(config.task.task_index),
        )
        framing = self.raw_client.policy_framing()
        if framing is None:
            raise RuntimeError("registered client has no policy framing")
        initial = _normalize_messages(
            [*framing, {"role": "user", "content": response["observation"]}]
        )
        excluded = treatment_excluded_messages(self.spec, config, initial)
        self.reset_complete = True
        return AdapterReset(
            namespace=self.namespace,
            initial_messages=initial,
            treatment_excluded_messages=excluded,
            roots=self.roots,
            receipt={
                "status": "ok",
                "client": self.spec.client_specification,
                "arm_parameter": self.spec.arm_parameter,
                "arm": config.capability.arm.value,
                "reset_response_sha256": sha256_json(response),
            },
        )

    def finalize_artifact(
        self, context: FinalizationContext
    ) -> ArtifactResult:
        return self.hooks.finalize_artifact(
            self.raw_client,
            context,
            self.config,
            self.evidence_store,
        )

    def handoff_to_grader(self, artifact: ArtifactResult) -> ScorerResult:
        return self.hooks.handoff_to_grader(
            self.raw_client,
            artifact,
            self.config,
            self.evidence_store,
        )

    def close(self) -> AdapterClose:
        if self.closed:
            raise RuntimeError("adapter close may run only once")
        receipt = self.raw_client.close()
        _validate_close_response(receipt)
        self.closed = True
        return AdapterClose(
            namespace=self.namespace,
            closed_roots=self.roots,
            receipt={
                "status": "closed",
                "client": self.spec.client_specification,
                "close_response_sha256": sha256_json(receipt),
            },
        )


class PairedEvalRegistry:
    """Immutable adapter catalog plus explicitly injected runtime builders."""

    def __init__(
        self,
        *,
        specs: Sequence[AdapterSpec] = DEFAULT_ADAPTER_SPECS,
        builders: Mapping[str, RuntimeBuilder],
    ) -> None:
        registered_specs = tuple(specs)
        spec_map = {spec.benchmark: spec for spec in registered_specs}
        if len(spec_map) != len(registered_specs):
            raise ValueError("adapter benchmark registrations must be unique")
        if registered_specs != DEFAULT_ADAPTER_SPECS:
            raise ValueError("registry must contain the exact frozen adapter specs")
        builder_map = dict(builders)
        if set(builder_map) != set(spec_map):
            raise ValueError("runtime builders must match the adapter registry exactly")
        if any(not callable(builder) for builder in builder_map.values()):
            raise TypeError("runtime builders must be callable")
        self.specs = MappingProxyType(spec_map)
        self.builders = MappingProxyType(builder_map)

    def resolve(self, benchmark: str) -> AdapterSpec:
        try:
            return self.specs[benchmark]
        except KeyError as error:
            raise KeyError(f"unsupported paired-evaluation benchmark: {benchmark}") from error

    def build_runtime(
        self,
        config: RunConfig,
        *,
        evidence_store: PrivateEvidenceStore,
    ) -> RuntimeBindings:
        spec = self.resolve(config.task.benchmark)
        spec.validate_config(config)
        client_type = spec.resolve_client_type()
        bindings = self.builders[spec.benchmark](
            config=config,
            spec=spec,
            client_type=client_type,
            evidence_store=evidence_store,
        )
        if not isinstance(bindings, RuntimeBindings):
            raise TypeError("runtime builder must return RuntimeBindings")
        adapter = bindings.adapter
        if not isinstance(adapter, ClientEnvironmentAdapter):
            raise TypeError("runtime builder must use ClientEnvironmentAdapter")
        if adapter.config != config or adapter.spec != spec:
            raise ValueError("runtime adapter binding does not match its config")
        if type(adapter.raw_client) is not client_type:
            raise TypeError("runtime adapter bound the wrong client class")
        if (
            getattr(adapter.raw_client, spec.arm_parameter, None)
            != config.capability.arm.value
        ):
            raise ValueError("runtime client bound the wrong evaluation arm")
        if getattr(bindings.model, "model_config", None) != config.model:
            raise ValueError("runtime model binding does not match its config")
        if adapter.evidence_store is not evidence_store:
            raise ValueError("runtime adapter bound a different evidence store")
        return bindings


def make_runtime_factory(
    builders: Mapping[str, RuntimeBuilder],
    *,
    evidence_store: PrivateEvidenceStore,
) -> Callable[[RunConfig], RuntimeBindings]:
    """Bind builders and private evidence into an executable manifest factory."""

    if not isinstance(evidence_store, PrivateEvidenceStore):
        raise TypeError("evidence_store must be a PrivateEvidenceStore")
    registry = PairedEvalRegistry(builders=builders)

    def build_runtime(config: RunConfig) -> RuntimeBindings:
        return registry.build_runtime(config, evidence_store=evidence_store)

    return build_runtime


__all__ = [
    "AdapterHooks",
    "AdapterSpec",
    "ClientEnvironmentAdapter",
    "ClientStepProxy",
    "DEFAULT_ADAPTER_SPECS",
    "FROZEN_ARMS",
    "PairedEvalRegistry",
    "lifecycle_roots",
    "make_runtime_factory",
    "treatment_excluded_messages",
]
