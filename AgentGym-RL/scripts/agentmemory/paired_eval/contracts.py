"""Immutable configuration and typed boundary contracts for paired evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import re
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from .serialization import canonical_json_bytes, sha256_json


RESULT_SCHEMA = "amg.paired_eval.result"
RESULT_SCHEMA_VERSION = "2.1.0"
MANIFEST_SCHEMA = "amg.paired_eval.manifest"
MANIFEST_SCHEMA_VERSION = "2.0.0"
CONTEXT_TRANSITION_SCHEMA = "agentmemory_task_neutral_context_transition_v1"
CONTEXT_OPERATION_APPEND = "append_observation"
CONTEXT_OPERATION_PRESERVE = "preserve"
CONTEXT_OPERATION_REPLACE = "replace_messages"
CONTEXT_OPERATIONS = frozenset(
    {
        CONTEXT_OPERATION_APPEND,
        CONTEXT_OPERATION_PRESERVE,
        CONTEXT_OPERATION_REPLACE,
    }
)
TASK_NEUTRAL_RECEIPT_SCHEMA = "agentmemory_task_neutral_transition_v1"
BENCHMARK_ROUTE = ("benchmark_task", "benchmark_task")
EXTERNAL_MEMORY_ROUTE = ("external_memory", "external_memory")
POLICY_COMPACTION_ROUTE = ("policy_compaction", "policy_context")
GLOBAL_POLICY_ACTION_ACCOUNTING = "global_policy_action_budget_v1"
EXTERNAL_MEMORY_CAPABILITY_SURFACES = (
    "dedicated_memory_namespace",
    "dedicated_memory_root",
    "mount",
    "endpoint",
    "environment_variable",
    "prompt_declaration",
    "tool_schema",
    "parser_dispatch_path",
    "action_receipt",
    "private_evidence_store",
    "cleanup_handle",
)
ROUTE_EXECUTION_KINDS = MappingProxyType(
    {
        BENCHMARK_ROUTE: "benchmark_action",
        EXTERNAL_MEMORY_ROUTE: "external_memory_action",
        POLICY_COMPACTION_ROUTE: "policy_compaction",
    }
)
WRAPPER_EXECUTION_SCHEMA = "amg.paired_eval.wrapper_execution_v1"
FAILURE_CLASSES = frozenset(
    {
        "model_failure",
        "environment_failure",
        "wall_timeout",
        "artifact_failure",
        "scorer_failure",
    }
)
TERMINATION_REASONS = frozenset({"terminal", "horizon", "failure", "timeout"})
HORIZON_CAUSES = frozenset(
    {
        "policy_turn_limit",
        "tool_call_limit",
        "token_limit",
        "prompt_token_limit",
    }
)
SCORER_STATUSES = frozenset({"deferred", "scored"})
ACTION_ACCOUNTING_COUNTERS = (
    "parsed_actions",
    "domain_tool_attempts",
    "successful_backend_calls",
    "workspace_actions",
    "compactions",
    "answers",
    "invalid_actions",
    "parser_corrections",
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def require_commit(name: str, value: Any) -> str:
    if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40-character commit")
    return value


def require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def require_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


class Arm(str, Enum):
    NATIVE = "native"
    AMG_COMPACTION_ONLY = "amg_compaction_only"
    AMG_MEMORY = "amg_memory"


CAPABILITY_LATTICE = MappingProxyType(
    {
        Arm.NATIVE.value: "00",
        Arm.AMG_COMPACTION_ONLY.value: "10",
        Arm.AMG_MEMORY.value: "11",
    }
)


class ModelClientFailure(RuntimeError):
    """Typed model-side failure that survives controller exception wrapping."""


@dataclass(frozen=True)
class CapabilityConfig:
    """The frozen compaction/external-memory capability declaration."""

    arm: Arm
    name: str
    enabled: bool
    policy_authored_compaction: bool
    external_read_write_memory: bool
    tools: Tuple[str, ...]
    prompt_declaration: str
    external_memory_surfaces: Tuple[str, ...]
    implicit_retrieval: bool = False
    hidden_context_injection: bool = False

    def __post_init__(self) -> None:
        arm = Arm(self.arm)
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(
            self,
            "external_memory_surfaces",
            tuple(self.external_memory_surfaces),
        )
        require_text("capability.name", self.name)
        expected_name = {
            Arm.NATIVE: "native_without_amg_capability_v1",
            Arm.AMG_COMPACTION_ONLY: (
                "amg_policy_compaction_without_external_memory_v1"
            ),
            Arm.AMG_MEMORY: "amg_policy_compaction_external_read_write_v1",
        }[arm]
        if self.name != expected_name:
            raise ValueError("capability name does not match the frozen arm")
        expected_enabled = arm is not Arm.NATIVE
        expected_compaction = arm is not Arm.NATIVE
        expected_external_memory = arm is Arm.AMG_MEMORY
        if self.enabled is not expected_enabled:
            raise ValueError("capability enabled flag does not match its arm")
        if self.policy_authored_compaction is not expected_compaction:
            raise ValueError("compaction enablement does not match its arm")
        if self.external_read_write_memory is not expected_external_memory:
            raise ValueError("external memory enablement does not match its arm")
        expected_tools = (
            ("external_memory_read", "external_memory_write")
            if expected_external_memory
            else ()
        )
        if tuple(self.tools) != expected_tools:
            raise ValueError("capability tools do not match the frozen arm")
        expected_declaration = {
            Arm.NATIVE: "",
            Arm.AMG_COMPACTION_ONLY: "",
            Arm.AMG_MEMORY: "adapter_owned_external_memory_declaration_v1",
        }[arm]
        if self.prompt_declaration != expected_declaration:
            raise ValueError("capability prompt does not match the frozen arm")
        expected_surfaces = (
            EXTERNAL_MEMORY_CAPABILITY_SURFACES
            if expected_external_memory
            else ()
        )
        if self.external_memory_surfaces != expected_surfaces:
            raise ValueError(
                "external-memory surfaces must be absent or the exact full bundle"
            )
        if self.implicit_retrieval:
            raise ValueError("implicit retrieval is forbidden")
        if self.hidden_context_injection:
            raise ValueError("hidden context injection is forbidden")

    def to_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "name": self.name,
            "enabled": self.enabled,
            "policy_authored_compaction": self.policy_authored_compaction,
            "external_read_write_memory": self.external_read_write_memory,
            "tools": list(self.tools),
            "prompt_declaration": self.prompt_declaration,
            "external_memory_surfaces": list(self.external_memory_surfaces),
            "implicit_retrieval": self.implicit_retrieval,
            "hidden_context_injection": self.hidden_context_injection,
        }

    @property
    def allowed_routes(self) -> Tuple[Tuple[str, str], ...]:
        """Task-neutral routes a wrapper may attest for this capability."""

        routes = [BENCHMARK_ROUTE]
        if self.external_read_write_memory:
            routes.append(EXTERNAL_MEMORY_ROUTE)
        if self.policy_authored_compaction:
            routes.append(POLICY_COMPACTION_ROUTE)
        return tuple(routes)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CapabilityConfig":
        return cls(
            arm=Arm(payload["arm"]),
            name=payload["name"],
            enabled=payload["enabled"],
            policy_authored_compaction=payload["policy_authored_compaction"],
            external_read_write_memory=payload["external_read_write_memory"],
            tools=tuple(payload.get("tools", ())),
            prompt_declaration=payload["prompt_declaration"],
            external_memory_surfaces=tuple(
                payload.get("external_memory_surfaces", ())
            ),
            implicit_retrieval=payload.get("implicit_retrieval", False),
            hidden_context_injection=payload.get(
                "hidden_context_injection", False
            ),
        )


NATIVE_CAPABILITY = CapabilityConfig(
    arm=Arm.NATIVE,
    name="native_without_amg_capability_v1",
    enabled=False,
    policy_authored_compaction=False,
    external_read_write_memory=False,
    tools=(),
    prompt_declaration="",
    external_memory_surfaces=(),
)

AMG_COMPACTION_ONLY_CAPABILITY = CapabilityConfig(
    arm=Arm.AMG_COMPACTION_ONLY,
    name="amg_policy_compaction_without_external_memory_v1",
    enabled=True,
    policy_authored_compaction=True,
    external_read_write_memory=False,
    tools=(),
    prompt_declaration="",
    external_memory_surfaces=(),
)

AMG_MEMORY_CAPABILITY = CapabilityConfig(
    arm=Arm.AMG_MEMORY,
    name="amg_policy_compaction_external_read_write_v1",
    enabled=True,
    policy_authored_compaction=True,
    external_read_write_memory=True,
    tools=("external_memory_read", "external_memory_write"),
    prompt_declaration="adapter_owned_external_memory_declaration_v1",
    external_memory_surfaces=EXTERNAL_MEMORY_CAPABILITY_SURFACES,
)


def capability_for_arm(arm: Any) -> CapabilityConfig:
    selected = Arm(arm)
    if selected is Arm.NATIVE:
        return NATIVE_CAPABILITY
    if selected is Arm.AMG_COMPACTION_ONLY:
        return AMG_COMPACTION_ONLY_CAPABILITY
    return AMG_MEMORY_CAPABILITY


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float
    top_p: float
    max_output_tokens: int
    stop: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop", tuple(self.stop))
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or float(self.temperature) < 0
        ):
            raise ValueError("temperature must be finite and non-negative")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(float(self.top_p))
            or not 0 < float(self.top_p) <= 1
        ):
            raise ValueError("top_p must be in (0, 1]")
        require_positive_int("max_output_tokens", self.max_output_tokens)
        if any(not isinstance(item, str) or not item for item in self.stop):
            raise ValueError("stop entries must be nonempty text")
        if len(set(self.stop)) != len(self.stop):
            raise ValueError("stop entries must be unique")

    def with_max_output_tokens(self, value: int) -> "DecodingConfig":
        return replace(self, max_output_tokens=value)

    def to_payload(self) -> dict[str, Any]:
        return {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "max_output_tokens": self.max_output_tokens,
            "stop": list(self.stop),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DecodingConfig":
        return cls(
            temperature=payload["temperature"],
            top_p=payload["top_p"],
            max_output_tokens=payload["max_output_tokens"],
            stop=tuple(payload.get("stop", ())),
        )


@dataclass(frozen=True)
class BudgetConfig:
    max_policy_turns: int
    max_total_tokens: int
    max_tool_calls: int
    max_wall_seconds: float
    max_prompt_tokens: int
    max_model_tokens: int
    max_observation_tokens: int
    action_observation_envelope_tokens: int = 0

    def __post_init__(self) -> None:
        require_positive_int("max_policy_turns", self.max_policy_turns)
        require_positive_int("max_total_tokens", self.max_total_tokens)
        require_positive_int("max_tool_calls", self.max_tool_calls)
        require_positive_int("max_prompt_tokens", self.max_prompt_tokens)
        require_positive_int("max_model_tokens", self.max_model_tokens)
        require_positive_int("max_observation_tokens", self.max_observation_tokens)
        require_nonnegative_int(
            "action_observation_envelope_tokens",
            self.action_observation_envelope_tokens,
        )
        if (
            isinstance(self.max_wall_seconds, bool)
            or not isinstance(self.max_wall_seconds, (int, float))
            or not math.isfinite(float(self.max_wall_seconds))
            or float(self.max_wall_seconds) <= 0
        ):
            raise ValueError("max_wall_seconds must be finite and positive")
        if self.max_prompt_tokens >= self.max_model_tokens:
            raise ValueError("max_prompt_tokens must leave response capacity")

    def to_payload(self) -> dict[str, Any]:
        return {
            "max_policy_turns": self.max_policy_turns,
            "max_total_tokens": self.max_total_tokens,
            "max_tool_calls": self.max_tool_calls,
            "max_wall_seconds": float(self.max_wall_seconds),
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_model_tokens": self.max_model_tokens,
            "max_observation_tokens": self.max_observation_tokens,
            "action_observation_envelope_tokens": (
                self.action_observation_envelope_tokens
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BudgetConfig":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CompactionConfig:
    policy: str
    trigger: str
    summary_max_tokens: int
    summary_instruction_sha256: str
    context_pressure_policy_sha256: str
    context_transition_schema: str
    action_accounting: str
    config_sha256: str

    def __post_init__(self) -> None:
        require_text("compaction.policy", self.policy)
        require_text("compaction.trigger", self.trigger)
        require_positive_int("compaction.summary_max_tokens", self.summary_max_tokens)
        require_sha256(
            "compaction.summary_instruction_sha256",
            self.summary_instruction_sha256,
        )
        require_sha256(
            "compaction.context_pressure_policy_sha256",
            self.context_pressure_policy_sha256,
        )
        if self.context_transition_schema != CONTEXT_TRANSITION_SCHEMA:
            raise ValueError(
                "compaction context-transition schema must be task-neutral"
            )
        if self.action_accounting != GLOBAL_POLICY_ACTION_ACCOUNTING:
            raise ValueError(
                "compaction must use the global policy-action budget"
            )
        require_sha256("compaction.config_sha256", self.config_sha256)

    def to_payload(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "trigger": self.trigger,
            "summary_max_tokens": self.summary_max_tokens,
            "summary_instruction_sha256": self.summary_instruction_sha256,
            "context_pressure_policy_sha256": (
                self.context_pressure_policy_sha256
            ),
            "context_transition_schema": self.context_transition_schema,
            "action_accounting": self.action_accounting,
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CompactionConfig":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SourceConfig:
    outer_commit: str
    inner_commit: str
    adapter_sha256: str
    runner_sha256: str

    def __post_init__(self) -> None:
        require_commit("source.outer_commit", self.outer_commit)
        require_commit("source.inner_commit", self.inner_commit)
        require_sha256("source.adapter_sha256", self.adapter_sha256)
        require_sha256("source.runner_sha256", self.runner_sha256)

    def to_payload(self) -> dict[str, Any]:
        return {
            "outer_commit": self.outer_commit,
            "inner_commit": self.inner_commit,
            "adapter_sha256": self.adapter_sha256,
            "runner_sha256": self.runner_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SourceConfig":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RuntimeConfig:
    image_digest: str
    runtime_sha256: str
    compute_class: str

    def __post_init__(self) -> None:
        if not isinstance(self.image_digest, str) or not self.image_digest.startswith(
            "sha256:"
        ):
            raise ValueError("runtime.image_digest must use sha256:<digest>")
        require_sha256("runtime.image_digest", self.image_digest[7:])
        require_sha256("runtime.runtime_sha256", self.runtime_sha256)
        require_text("runtime.compute_class", self.compute_class)

    def to_payload(self) -> dict[str, Any]:
        return {
            "image_digest": self.image_digest,
            "runtime_sha256": self.runtime_sha256,
            "compute_class": self.compute_class,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RuntimeConfig":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str
    tokenizer_sha256: str

    def __post_init__(self) -> None:
        require_text("model.model_id", self.model_id)
        require_text("model.revision", self.revision)
        require_sha256("model.tokenizer_sha256", self.tokenizer_sha256)

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_sha256": self.tokenizer_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ModelConfig":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class GraderConfig:
    name: str
    revision: str
    config_sha256: str

    def __post_init__(self) -> None:
        require_text("grader.name", self.name)
        require_text("grader.revision", self.revision)
        require_sha256("grader.config_sha256", self.config_sha256)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GraderConfig":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class TaskConfig:
    benchmark: str
    protocol: str
    task_id: str
    task_index: int
    seed: int
    native_tools: Tuple[str, ...]
    artifact_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "native_tools", tuple(self.native_tools))
        require_text("task.benchmark", self.benchmark)
        require_text("task.protocol", self.protocol)
        require_text("task.task_id", self.task_id)
        require_nonnegative_int("task.task_index", self.task_index)
        require_nonnegative_int("task.seed", self.seed)
        require_text("task.artifact_type", self.artifact_type)
        if any(not isinstance(tool, str) or not tool for tool in self.native_tools):
            raise ValueError("native tool names must be nonempty text")
        if len(set(self.native_tools)) != len(self.native_tools):
            raise ValueError("native tool names must be unique")

    def to_payload(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "protocol": self.protocol,
            "task_id": self.task_id,
            "task_index": self.task_index,
            "seed": self.seed,
            "native_tools": list(self.native_tools),
            "artifact_type": self.artifact_type,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TaskConfig":
        return cls(
            benchmark=payload["benchmark"],
            protocol=payload["protocol"],
            task_id=payload["task_id"],
            task_index=payload["task_index"],
            seed=payload["seed"],
            native_tools=tuple(payload.get("native_tools", ())),
            artifact_type=payload["artifact_type"],
        )


@dataclass(frozen=True)
class PairKey:
    run_id: str
    benchmark: str
    protocol: str
    task_id: str
    seed: int

    def __post_init__(self) -> None:
        require_text("pair.run_id", self.run_id)
        require_text("pair.benchmark", self.benchmark)
        require_text("pair.protocol", self.protocol)
        require_text("pair.task_id", self.task_id)
        require_nonnegative_int("pair.seed", self.seed)

    @classmethod
    def from_config(cls, config: "RunConfig") -> "PairKey":
        return cls(
            run_id=config.run_id,
            benchmark=config.task.benchmark,
            protocol=config.task.protocol,
            task_id=config.task.task_id,
            seed=config.task.seed,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "protocol": self.protocol,
            "task_id": self.task_id,
            "seed": self.seed,
        }

    def render(self) -> str:
        return canonical_json_bytes(
            [self.run_id, self.benchmark, self.protocol, self.task_id, self.seed]
        ).decode("utf-8")


@dataclass(frozen=True)
class Namespace:
    run_id: str
    benchmark: str
    protocol: str
    task_id: str
    seed: int
    arm: Arm

    def __post_init__(self) -> None:
        require_text("namespace.run_id", self.run_id)
        require_text("namespace.benchmark", self.benchmark)
        require_text("namespace.protocol", self.protocol)
        require_text("namespace.task_id", self.task_id)
        require_nonnegative_int("namespace.seed", self.seed)
        object.__setattr__(self, "arm", Arm(self.arm))

    @classmethod
    def from_config(cls, config: "RunConfig") -> "Namespace":
        return cls(
            run_id=config.run_id,
            benchmark=config.task.benchmark,
            protocol=config.task.protocol,
            task_id=config.task.task_id,
            seed=config.task.seed,
            arm=config.capability.arm,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "protocol": self.protocol,
            "task_id": self.task_id,
            "seed": self.seed,
            "arm": self.arm.value,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())


@dataclass(frozen=True)
class CapabilityRoot:
    """Opaque, namespace-bound wrapper root used by one task-neutral route."""

    capability_id: str
    root_kind: str
    root_id: str
    namespace_sha256: str

    def __post_init__(self) -> None:
        require_text("root.capability_id", self.capability_id)
        require_text("root.root_kind", self.root_kind)
        require_sha256("root.root_id", self.root_id)
        require_sha256("root.namespace_sha256", self.namespace_sha256)

    @property
    def route(self) -> Tuple[str, str]:
        return (self.capability_id, self.root_kind)

    def to_payload(self) -> dict[str, str]:
        return {
            "capability_id": self.capability_id,
            "root_kind": self.root_kind,
            "root_id": self.root_id,
            "namespace_sha256": self.namespace_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CapabilityRoot":
        if not isinstance(payload, Mapping):
            raise TypeError("capability root must be a mapping")
        expected = {
            "capability_id",
            "root_kind",
            "root_id",
            "namespace_sha256",
        }
        if set(payload) != expected:
            raise ValueError("capability root fields are not canonical")
        return cls(**{name: payload[name] for name in expected})


def capability_root_id(
    namespace: Namespace,
    capability_id: str,
    root_kind: str,
) -> str:
    """Derive a logical root id from the full isolated namespace and route."""

    return sha256_json(
        {
            "namespace": namespace.to_payload(),
            "capability_id": capability_id,
            "root_kind": root_kind,
        }
    )


def validate_capability_roots(
    namespace: Namespace,
    roots: Sequence[CapabilityRoot],
) -> Tuple[CapabilityRoot, ...]:
    """Validate namespace binding and uniqueness without interpreting actions."""

    normalized = tuple(roots)
    if any(not isinstance(root, CapabilityRoot) for root in normalized):
        raise TypeError("all lifecycle roots must be CapabilityRoot values")
    if any(root.namespace_sha256 != namespace.sha256 for root in normalized):
        raise ValueError("lifecycle root is bound to the wrong namespace")
    if any(
        root.root_id
        != capability_root_id(
            namespace,
            root.capability_id,
            root.root_kind,
        )
        for root in normalized
    ):
        raise ValueError("lifecycle root id is not bound to its namespace and route")
    routes = [root.route for root in normalized]
    root_ids = [root.root_id for root in normalized]
    if len(routes) != len(set(routes)):
        raise ValueError("lifecycle routes must be unique")
    if len(root_ids) != len(set(root_ids)):
        raise ValueError("lifecycle roots must be distinct")
    return normalized


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    task: TaskConfig
    model: ModelConfig
    decoding: DecodingConfig
    budgets: BudgetConfig
    compaction: CompactionConfig
    source: SourceConfig
    runtime: RuntimeConfig
    grader: GraderConfig
    capability: CapabilityConfig

    def __post_init__(self) -> None:
        require_text("run_id", self.run_id)
        expected_types = {
            "task": TaskConfig,
            "model": ModelConfig,
            "decoding": DecodingConfig,
            "budgets": BudgetConfig,
            "compaction": CompactionConfig,
            "source": SourceConfig,
            "runtime": RuntimeConfig,
            "grader": GraderConfig,
            "capability": CapabilityConfig,
        }
        for name, expected_type in expected_types.items():
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")

    @property
    def pair_key(self) -> str:
        return PairKey.from_config(self).render()

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task.to_payload(),
            "model": self.model.to_payload(),
            "decoding": self.decoding.to_payload(),
            "budgets": self.budgets.to_payload(),
            "compaction": self.compaction.to_payload(),
            "source": self.source.to_payload(),
            "runtime": self.runtime.to_payload(),
            "grader": self.grader.to_payload(),
            "capability": self.capability.to_payload(),
        }

    def treatment_excluded_payload(self) -> dict[str, Any]:
        payload = self.to_payload()
        del payload["capability"]
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RunConfig":
        return cls(
            run_id=payload["run_id"],
            task=TaskConfig.from_payload(payload["task"]),
            model=ModelConfig.from_payload(payload["model"]),
            decoding=DecodingConfig.from_payload(payload["decoding"]),
            budgets=BudgetConfig.from_payload(payload["budgets"]),
            compaction=CompactionConfig.from_payload(payload["compaction"]),
            source=SourceConfig.from_payload(payload["source"]),
            runtime=RuntimeConfig.from_payload(payload["runtime"]),
            grader=GraderConfig.from_payload(payload["grader"]),
            capability=CapabilityConfig.from_payload(payload["capability"]),
        )

    @property
    def full_config_sha256(self) -> str:
        return sha256_json(self.to_payload())

    @property
    def treatment_excluded_config_sha256(self) -> str:
        return sha256_json(self.treatment_excluded_payload())


@dataclass(frozen=True)
class EvidenceReference:
    protected_ref: str
    sha256: str
    byte_count: int
    media_type: str = "application/json"

    def __post_init__(self) -> None:
        if not isinstance(self.protected_ref, str) or not self.protected_ref.startswith(
            "evidence://"
        ):
            raise ValueError("protected evidence references must use evidence://")
        require_sha256("evidence.sha256", self.sha256)
        if self.protected_ref.rsplit("/", 1)[-1] != self.sha256:
            raise ValueError("evidence reference digest does not match SHA-256")
        require_nonnegative_int("evidence.byte_count", self.byte_count)
        require_text("evidence.media_type", self.media_type)

    def to_payload(self) -> dict[str, Any]:
        return {
            "protected_ref": self.protected_ref,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class PolicyContextPressure:
    action_prompt_tokens: int
    candidate_prompt_tokens: int
    max_prompt_tokens: int
    max_model_tokens: int
    max_response_tokens: int
    max_observation_tokens: int
    action_observation_envelope_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "action_prompt_tokens",
            "candidate_prompt_tokens",
            "max_prompt_tokens",
            "max_model_tokens",
            "max_response_tokens",
            "max_observation_tokens",
        ):
            require_positive_int(name, getattr(self, name))
        require_nonnegative_int(
            "action_observation_envelope_tokens",
            self.action_observation_envelope_tokens,
        )
        if self.candidate_prompt_tokens < self.action_prompt_tokens:
            raise ValueError("candidate prompt must include the action prompt")

    @property
    def effective_prompt_capacity(self) -> int:
        capacity = min(
            self.max_prompt_tokens,
            self.max_model_tokens - self.max_response_tokens,
        )
        if capacity <= 0:
            raise ValueError("effective prompt capacity must be positive")
        return capacity


@dataclass(frozen=True)
class BenchmarkActionExecutionReceipt:
    submission_sha256: str
    adapter_receipt_sha256: str

    def __post_init__(self) -> None:
        require_sha256("benchmark submission", self.submission_sha256)
        require_sha256("benchmark adapter receipt", self.adapter_receipt_sha256)

    def to_payload(self) -> dict[str, str]:
        return {
            "submission_sha256": self.submission_sha256,
            "adapter_receipt_sha256": self.adapter_receipt_sha256,
        }


@dataclass(frozen=True)
class ExternalMemoryExecutionReceipt:
    operation: str
    submission_sha256: str
    memory_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.operation not in {"read", "write", "read_write"}:
            raise ValueError(
                "external memory operation must be read, write, or read_write"
            )
        require_sha256("external memory submission", self.submission_sha256)
        require_sha256("external memory receipt", self.memory_receipt_sha256)

    def to_payload(self) -> dict[str, str]:
        return {
            "operation": self.operation,
            "submission_sha256": self.submission_sha256,
            "memory_receipt_sha256": self.memory_receipt_sha256,
        }


@dataclass(frozen=True)
class PolicyCompactionExecutionReceipt:
    summary_sha256: str
    trigger_sha256: str
    replacement_context_sha256: str

    def __post_init__(self) -> None:
        require_sha256("compaction summary", self.summary_sha256)
        require_sha256("compaction trigger", self.trigger_sha256)
        require_sha256(
            "compaction replacement context",
            self.replacement_context_sha256,
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "summary_sha256": self.summary_sha256,
            "trigger_sha256": self.trigger_sha256,
            "replacement_context_sha256": self.replacement_context_sha256,
        }


WrapperExecutionReceipt = Union[
    BenchmarkActionExecutionReceipt,
    ExternalMemoryExecutionReceipt,
    PolicyCompactionExecutionReceipt,
]


def parse_wrapper_execution_receipt(
    execution_kind: str,
    payload: Any,
) -> WrapperExecutionReceipt:
    """Parse the wrapper-owned route receipt without parsing policy text."""

    if not isinstance(payload, Mapping):
        raise TypeError("wrapper execution receipt must be a mapping")
    receipt_types = {
        "benchmark_action": BenchmarkActionExecutionReceipt,
        "external_memory_action": ExternalMemoryExecutionReceipt,
        "policy_compaction": PolicyCompactionExecutionReceipt,
    }
    receipt_type = receipt_types.get(execution_kind)
    if receipt_type is None:
        raise ValueError("wrapper execution kind is unsupported")
    expected = set(receipt_type.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError("wrapper execution receipt fields are not canonical")
    return receipt_type(**{name: payload[name] for name in expected})


@dataclass(frozen=True)
class TaskNeutralStepReceipt:
    """Typed public core of a wrapper-owned task-neutral step receipt."""

    route: CapabilityRoot
    native_step_before: int
    native_step_after: int
    native_call_count_before: int
    native_call_count_after: int
    context_epoch_before: int
    context_epoch_after: int
    session_epoch_before: int
    session_epoch_after: int
    policy_step_before: int
    policy_step_after: int
    policy_output_sha256: str
    execution_kind: str
    execution_sha256: str
    execution_receipt: WrapperExecutionReceipt
    context_transition: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.route, CapabilityRoot):
            raise TypeError("step route must be a CapabilityRoot")
        monotonic_pairs = (
            ("native_step", self.native_step_before, self.native_step_after),
            (
                "native_call_count",
                self.native_call_count_before,
                self.native_call_count_after,
            ),
            ("context_epoch", self.context_epoch_before, self.context_epoch_after),
            ("session_epoch", self.session_epoch_before, self.session_epoch_after),
        )
        for name, before, after in monotonic_pairs:
            require_nonnegative_int(f"{name}_before", before)
            require_nonnegative_int(f"{name}_after", after)
            if after < before:
                raise ValueError(f"{name} cannot decrease during a policy step")
        require_nonnegative_int("policy_step_before", self.policy_step_before)
        require_nonnegative_int("policy_step_after", self.policy_step_after)
        if self.policy_step_after != self.policy_step_before + 1:
            raise ValueError("policy step receipt must advance exactly once")
        require_sha256("step.policy_output_sha256", self.policy_output_sha256)
        require_text("step.execution_kind", self.execution_kind)
        require_sha256("step.execution_sha256", self.execution_sha256)
        if ROUTE_EXECUTION_KINDS.get(self.route.route) != self.execution_kind:
            raise ValueError("wrapper execution kind does not match its route")
        expected_receipt_types = {
            "benchmark_action": BenchmarkActionExecutionReceipt,
            "external_memory_action": ExternalMemoryExecutionReceipt,
            "policy_compaction": PolicyCompactionExecutionReceipt,
        }
        if not isinstance(
            self.execution_receipt,
            expected_receipt_types[self.execution_kind],
        ):
            raise TypeError("wrapper execution receipt has the wrong typed variant")
        if not isinstance(self.context_transition, Mapping):
            raise TypeError("context transition receipt must be a mapping")
        if self.context_transition.get("schema") != CONTEXT_TRANSITION_SCHEMA:
            raise ValueError("context transition receipt schema is unsupported")
        if self.context_transition.get("operation") not in CONTEXT_OPERATIONS:
            raise ValueError("context transition receipt operation is unsupported")

    @classmethod
    def from_info(cls, info: Mapping[str, Any]) -> "TaskNeutralStepReceipt":
        if not isinstance(info, Mapping):
            raise TypeError("environment info must be a mapping")
        if info.get("schema") != TASK_NEUTRAL_RECEIPT_SCHEMA:
            raise ValueError("task-neutral step receipt schema is unsupported")
        if not isinstance(info.get("env_info"), Mapping):
            raise TypeError("task-neutral env_info must be a mapping")
        if not isinstance(info.get("wrapper_evidence"), Mapping):
            raise TypeError("task-neutral wrapper_evidence must be a mapping")
        submission = info.get("action_submission")
        if not isinstance(submission, Mapping):
            raise TypeError("task-neutral action_submission must be a mapping")
        if submission.get("accepted") is not True:
            raise ValueError("wrapper did not accept the ordinary policy step")
        route = CapabilityRoot(
            capability_id=submission.get("capability_id"),
            root_kind=submission.get("root_kind"),
            root_id=submission.get("root_id"),
            namespace_sha256=submission.get("namespace_sha256"),
        )
        policy_output_sha256 = submission.get("policy_output_sha256")
        execution = submission.get("execution")
        if not isinstance(execution, Mapping):
            raise TypeError("wrapper execution attestation must be a mapping")
        if set(execution) != {"schema", "kind", "status", "receipt"}:
            raise ValueError("wrapper execution attestation fields are not canonical")
        if execution.get("schema") != WRAPPER_EXECUTION_SCHEMA:
            raise ValueError("wrapper execution attestation schema is unsupported")
        if execution.get("status") != "ok":
            raise ValueError("wrapper execution did not attest successful dispatch")
        execution_kind = execution.get("kind")
        execution_receipt = parse_wrapper_execution_receipt(
            execution_kind,
            execution.get("receipt"),
        )
        execution_sha256 = submission.get("execution_sha256")
        require_sha256("step.execution_sha256", execution_sha256)
        if execution_sha256 != sha256_json(execution):
            raise ValueError("wrapper execution attestation digest mismatch")
        counter_names = (
            "native_step_before",
            "native_step_after",
            "native_call_count_before",
            "native_call_count_after",
            "context_epoch_before",
            "context_epoch_after",
            "session_epoch_before",
            "session_epoch_after",
            "policy_step_before",
            "policy_step_after",
        )
        try:
            counters = {name: info[name] for name in counter_names}
            transition = info["context_transition"]
        except KeyError as error:
            raise ValueError(
                f"task-neutral step receipt omitted {error.args[0]}"
            ) from error
        return cls(
            route=route,
            policy_output_sha256=policy_output_sha256,
            execution_kind=execution_kind,
            execution_sha256=execution_sha256,
            execution_receipt=execution_receipt,
            context_transition=transition,
            **counters,
        )


@dataclass(frozen=True)
class StepOutput:
    state: str
    reward: float
    done: bool
    info: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.state, str):
            raise TypeError("environment state must be text")
        if (
            isinstance(self.reward, bool)
            or not isinstance(self.reward, (int, float))
            or not math.isfinite(float(self.reward))
        ):
            raise ValueError("environment reward must be finite")
        if type(self.done) is not bool:
            raise TypeError("environment done must be boolean")
        if not isinstance(self.info, Mapping):
            raise TypeError("environment info must be a mapping")
        TaskNeutralStepReceipt.from_info(self.info)

    @property
    def task_neutral_receipt(self) -> TaskNeutralStepReceipt:
        return TaskNeutralStepReceipt.from_info(self.info)


@dataclass(frozen=True)
class PreparedPolicyTurn:
    messages: Tuple[Mapping[str, str], ...]
    prompt_token_count: int
    control_request: Optional[str]
    opaque: Any = None


@dataclass(frozen=True)
class CompletedPolicyTurn:
    step_output: StepOutput
    messages: Tuple[Mapping[str, str], ...]
    context_transition: Mapping[str, Any]


@dataclass(frozen=True)
class ModelOutput:
    text: str
    prompt_token_ids: Tuple[int, ...]
    response_token_ids: Tuple[int, ...]
    finish_reason: str
    request_ref: EvidenceReference
    response_ref: EvidenceReference
    tokenization_ref: EvidenceReference
    retry_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("model output must be text")
        if not self.prompt_token_ids:
            raise ValueError("prompt token ids must not be empty")
        if not self.response_token_ids:
            raise ValueError("response token ids must not be empty")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.prompt_token_ids
        ):
            raise TypeError("prompt token ids must be non-negative integers")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.response_token_ids
        ):
            raise TypeError("response token ids must be non-negative integers")
        require_text("finish_reason", self.finish_reason)
        require_nonnegative_int("retry_count", self.retry_count)


@dataclass(frozen=True)
class AdapterReset:
    namespace: Namespace
    initial_messages: Tuple[Mapping[str, str], ...]
    treatment_excluded_messages: Tuple[Mapping[str, str], ...]
    roots: Tuple[CapabilityRoot, ...]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, Namespace):
            raise TypeError("adapter reset namespace must be a Namespace")
        object.__setattr__(
            self,
            "roots",
            validate_capability_roots(self.namespace, self.roots),
        )
        if not isinstance(self.receipt, Mapping):
            raise TypeError("adapter reset receipt must be a mapping")


@dataclass(frozen=True)
class AdapterClose:
    """Successful wrapper cleanup attestation for every declared root."""

    namespace: Namespace
    closed_roots: Tuple[CapabilityRoot, ...]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, Namespace):
            raise TypeError("adapter close namespace must be a Namespace")
        object.__setattr__(
            self,
            "closed_roots",
            validate_capability_roots(self.namespace, self.closed_roots),
        )
        if not isinstance(self.receipt, Mapping):
            raise TypeError("adapter close receipt must be a mapping")


@dataclass(frozen=True)
class FinalizationContext:
    termination_reason: str
    horizon_cause: Optional[str]
    failure_class: Optional[str]
    timed_out: bool
    policy_turns: int
    tool_calls: int


@dataclass(frozen=True)
class ArtifactResult:
    artifact_type: str
    protected_ref: str
    sha256: str
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_text("artifact_type", self.artifact_type)
        if not isinstance(self.protected_ref, str) or not self.protected_ref.startswith(
            "evidence://"
        ):
            raise ValueError("artifact references must use evidence://")
        require_sha256("artifact.sha256", self.sha256)
        if self.protected_ref.rsplit("/", 1)[-1] != self.sha256:
            raise ValueError("artifact reference digest does not match SHA-256")
        if not isinstance(self.receipt, Mapping):
            raise TypeError("artifact receipt must be a mapping")


@dataclass(frozen=True)
class ScorerResult:
    name: str
    revision: str
    config_sha256: str
    status: str
    input_sha256: str
    output_sha256: Optional[str]
    per_task_correct: Optional[bool]
    public_metrics: Mapping[str, Any]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_text("scorer.name", self.name)
        require_text("scorer.revision", self.revision)
        require_sha256("scorer.config_sha256", self.config_sha256)
        if self.status not in SCORER_STATUSES:
            raise ValueError("scorer status must be deferred or scored")
        require_sha256("scorer.input_sha256", self.input_sha256)
        if not isinstance(self.public_metrics, Mapping):
            raise TypeError("public scorer metrics must be a mapping")
        if self.status == "deferred":
            if self.output_sha256 is not None or self.per_task_correct is not None:
                raise ValueError("deferred scorer cannot contain scored output")
            if self.public_metrics:
                raise ValueError("deferred scorer cannot contain public metrics")
        else:
            require_sha256("scorer.output_sha256", self.output_sha256)
            if type(self.per_task_correct) is not bool:
                raise TypeError("scored result requires a per-task boolean")
        for key, value in self.public_metrics.items():
            require_text("public metric name", key)
            if value is not None and type(value) not in {bool, int, float}:
                raise TypeError("public metrics may contain only scalar numbers/bools")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("public metric floats must be finite")
        if not isinstance(self.receipt, Mapping):
            raise TypeError("scorer receipt must be a mapping")
        if self.receipt.get("status") != self.status:
            raise ValueError("scorer receipt status disagrees with scorer result")


@runtime_checkable
class EnvironmentClientProtocol(Protocol):
    def normalize_initial_policy_context(
        self, messages: Sequence[Mapping[str, str]]
    ) -> Sequence[Mapping[str, str]]:
        ...

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        initial: bool = False,
    ) -> Any:
        ...

    def policy_turn_candidate(self) -> Optional[str]:
        ...

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure
    ) -> Optional[str]:
        ...

    def step(self, policy_output: str) -> Any:
        ...


@runtime_checkable
class EnvironmentAdapterProtocol(Protocol):
    client: EnvironmentClientProtocol

    def reset(self, config: RunConfig) -> AdapterReset:
        ...

    def finalize_artifact(self, context: FinalizationContext) -> ArtifactResult:
        ...

    def handoff_to_grader(self, artifact: ArtifactResult) -> ScorerResult:
        ...

    def close(self) -> AdapterClose:
        ...


@runtime_checkable
class ModelClientProtocol(Protocol):
    model_config: ModelConfig

    def count_prompt_tokens(
        self, messages: Sequence[Mapping[str, str]]
    ) -> int:
        ...

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        decoding: DecodingConfig,
        seed: int,
    ) -> ModelOutput:
        ...


@runtime_checkable
class PolicyTurnControllerProtocol(Protocol):
    def bind_initial(
        self,
        client: EnvironmentClientProtocol,
        messages: Sequence[Mapping[str, str]],
    ) -> Tuple[Mapping[str, str], ...]:
        ...

    def prepare(
        self,
        client: EnvironmentClientProtocol,
        messages: Sequence[Mapping[str, str]],
        *,
        count_prompt_tokens: Callable[[Sequence[Mapping[str, str]]], int],
        budgets: BudgetConfig,
        max_response_tokens: int,
    ) -> PreparedPolicyTurn:
        ...

    def complete(
        self,
        client: EnvironmentClientProtocol,
        prepared: PreparedPolicyTurn,
        policy_output: str,
    ) -> CompletedPolicyTurn:
        ...
