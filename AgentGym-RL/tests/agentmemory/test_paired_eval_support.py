from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "scripts" / "agentmemory"


def install_package_path() -> None:
    import sys

    package_root = str(PACKAGE_ROOT)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


install_package_path()

from paired_eval.contracts import (  # noqa: E402
    AdapterClose,
    AdapterReset,
    Arm,
    ArtifactResult,
    BudgetConfig,
    CapabilityRoot,
    CompactionConfig,
    CONTEXT_TRANSITION_SCHEMA,
    DecodingConfig,
    FinalizationContext,
    GraderConfig,
    ModelConfig,
    ModelOutput,
    Namespace,
    RunConfig,
    RuntimeConfig,
    ROUTE_EXECUTION_KINDS,
    ScorerResult,
    SourceConfig,
    StepOutput,
    TaskConfig,
    WRAPPER_EXECUTION_SCHEMA,
    capability_root_id,
    capability_for_arm,
)
from paired_eval.evidence import PrivateEvidenceStore  # noqa: E402
from paired_eval.model_client import ModelClientError  # noqa: E402
from paired_eval.serialization import sha256_json, sha256_text  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
OUTER_COMMIT = "d5892e63de0f8ad2ebdcedf09be46d3bca4117d1"
INNER_COMMIT = "017ebd2fbc0ab8e53a0ba743f79b50d6e46d1a42"
FAKE_MEMORY_PROMPT_SUFFIX = "\nAdapter-owned durable memory instructions."


def make_config(
    *,
    benchmark: str = "fake_benchmark",
    protocol: str = "fake_protocol@1",
    task_id: str = "task-001",
    task_index: int = 0,
    seed: int = 7,
    arm: Arm = Arm.NATIVE,
    artifact_type: str = "answer",
    max_policy_turns: int = 3,
    max_total_tokens: int = 4096,
    max_tool_calls: Optional[int] = None,
    max_wall_seconds: float = 60.0,
) -> RunConfig:
    tool_limit = max_policy_turns if max_tool_calls is None else max_tool_calls
    return RunConfig(
        run_id="paired-test-run",
        task=TaskConfig(
            benchmark=benchmark,
            protocol=protocol,
            task_id=task_id,
            task_index=task_index,
            seed=seed,
            native_tools=("inspect", "act"),
            artifact_type=artifact_type,
        ),
        model=ModelConfig(
            model_id="test-model",
            revision="test-model-revision",
            tokenizer_sha256=SHA_A,
        ),
        decoding=DecodingConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=64,
        ),
        budgets=BudgetConfig(
            max_policy_turns=max_policy_turns,
            max_total_tokens=max_total_tokens,
            max_tool_calls=tool_limit,
            max_wall_seconds=max_wall_seconds,
            max_prompt_tokens=2048,
            max_model_tokens=2112,
            max_observation_tokens=256,
            action_observation_envelope_tokens=16,
        ),
        compaction=CompactionConfig(
            policy="policy_authored_task_neutral_v1",
            trigger="wrapper_token_pressure_v1",
            summary_max_tokens=256,
            summary_instruction_sha256=SHA_C,
            context_pressure_policy_sha256=SHA_D,
            context_transition_schema=CONTEXT_TRANSITION_SCHEMA,
            action_accounting="global_policy_action_budget_v1",
            config_sha256=SHA_B,
        ),
        source=SourceConfig(
            outer_commit=OUTER_COMMIT,
            inner_commit=INNER_COMMIT,
            adapter_sha256=SHA_C,
            runner_sha256=SHA_D,
        ),
        runtime=RuntimeConfig(
            image_digest="sha256:" + SHA_A,
            runtime_sha256=SHA_B,
            compute_class="cpu-test",
        ),
        grader=GraderConfig(
            name="fake_official_grader",
            revision="grader-revision-1",
            config_sha256=SHA_C,
        ),
        capability=capability_for_arm(arm),
    )


def with_arm(config: RunConfig, arm: Arm) -> RunConfig:
    return replace(config, capability=capability_for_arm(arm))


def lifecycle_roots(config: RunConfig) -> tuple[CapabilityRoot, ...]:
    namespace = Namespace.from_config(config)
    return tuple(
        CapabilityRoot(
            capability_id=capability_id,
            root_kind=root_kind,
            root_id=capability_root_id(
                namespace,
                capability_id,
                root_kind,
            ),
            namespace_sha256=namespace.sha256,
        )
        for capability_id, root_kind in config.capability.allowed_routes
    )


class ManualClock:
    def __init__(self, initial: float = 0.0) -> None:
        self.value = float(initial)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class FakeMemoryService:
    def __init__(self, namespace: Namespace, root: CapabilityRoot) -> None:
        self.namespace = namespace
        self.root_id = root.root_id
        self.values: dict[str, str] = {}
        self.events: list[tuple[str, str, Optional[str]]] = []
        self.reset_calls = 0
        self.closed = False

    def reset(self, namespace: Namespace) -> Mapping[str, Any]:
        if namespace != self.namespace:
            raise RuntimeError("external-memory namespace mismatch")
        self.values.clear()
        self.events.clear()
        self.reset_calls += 1
        return {
            "status": "ok",
            "capability_id": "external_memory",
            "root_kind": "external_memory",
            "root_id": self.root_id,
            "namespace": namespace.to_payload(),
        }

    def execute(self, policy_output: str) -> Mapping[str, Any]:
        if policy_output.startswith("MEMORY_WRITE(") and policy_output.endswith(")"):
            body = policy_output[13:-1]
            if "," not in body:
                raise RuntimeError("invalid fake memory-write syntax")
            key, value = (part.strip() for part in body.split(",", 1))
            if not key or not value:
                raise RuntimeError("fake memory write requires a key and value")
            self.values[key] = value
            self.events.append(("write", key, value))
            return {"operation": "read_write", "key": key, "status": "ok"}
        if policy_output.startswith("MEMORY_READ(") and policy_output.endswith(")"):
            key = policy_output[12:-1].strip()
            if not key or "," in key:
                raise RuntimeError("invalid fake memory-read syntax")
            value = self.values.get(key)
            self.events.append(("read", key, value))
            return {
                "operation": "read_write",
                "key": key,
                "found": value is not None,
                "value": value,
            }
        raise RuntimeError("unrecognized external-memory action")

    def close(self) -> Mapping[str, Any]:
        self.closed = True
        return {
            "status": "closed",
            "capability_id": "external_memory",
            "root_kind": "external_memory",
            "root_id": self.root_id,
        }


class FakeModel:
    def __init__(
        self,
        config: ModelConfig,
        store: PrivateEvidenceStore,
        outputs: Sequence[str],
        *,
        fail_on_call: Optional[int] = None,
        tokenize_fail_on_call: Optional[int] = None,
        clock: Optional[ManualClock] = None,
        advance_seconds: float = 0.0,
    ) -> None:
        self.model_config = config
        self.store = store
        self.outputs = list(outputs)
        self.fail_on_call = fail_on_call
        self.tokenize_fail_on_call = tokenize_fail_on_call
        self.tokenize_calls = 0
        self.clock = clock
        self.advance_seconds = float(advance_seconds)
        self.calls: list[dict[str, Any]] = []

    def count_prompt_tokens(
        self, messages: Sequence[Mapping[str, str]]
    ) -> int:
        self.tokenize_calls += 1
        if self.tokenize_fail_on_call == self.tokenize_calls:
            raise ModelClientError(
                "synthetic_tokenization_failure",
                "synthetic private tokenization failure",
            )
        return sum(len(message["content"].split()) + 2 for message in messages)

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        decoding: DecodingConfig,
        seed: int,
    ) -> ModelOutput:
        call_number = len(self.calls) + 1
        if self.fail_on_call == call_number:
            raise RuntimeError("synthetic model failure containing private text")
        if not self.outputs:
            raise AssertionError("fake model output queue is empty")
        if self.clock is not None:
            self.clock.advance(self.advance_seconds)
        text = self.outputs.pop(0)
        prompt_count = self.count_prompt_tokens(messages)
        prompt_ids = tuple(range(1000, 1000 + prompt_count))
        response_ids = tuple(ord(character) for character in text)
        request = {
            "messages": [dict(message) for message in messages],
            "decoding": decoding.to_payload(),
            "seed": seed,
        }
        response = {
            "text": text,
            "finish_reason": "stop",
            "response_token_ids": list(response_ids),
        }
        request_ref = self.store.put_json("model_requests", request)
        response_ref = self.store.put_json("model_responses", response)
        tokenize_ref = self.store.put_json(
            "tokenization",
            {"messages": request["messages"], "prompt_token_ids": list(prompt_ids)},
        )
        self.calls.append(request)
        return ModelOutput(
            text=text,
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
            finish_reason="stop",
            request_ref=request_ref,
            response_ref=response_ref,
            tokenization_ref=tokenize_ref,
            retry_count=0,
        )


class FakeClient:
    def __init__(
        self,
        plan: Sequence[Mapping[str, Any]],
        *,
        namespace: Namespace,
        roots: Sequence[CapabilityRoot],
        memory_service: Optional[FakeMemoryService],
        allow_compaction: bool,
    ) -> None:
        self.plan = [dict(item) for item in plan]
        self.memory_service = memory_service
        self.allow_compaction = allow_compaction
        self.namespace = namespace
        self.roots_by_route = {root.route: root for root in roots}
        self.step_calls: list[str] = []
        self.reset_calls: list[int] = []
        self.bound_contexts: list[tuple[bool, list[dict[str, str]]]] = []
        self.route_calls: list[dict[str, Any]] = []
        self.step_infos: list[dict[str, Any]] = []

    def reset(self, index: int) -> Mapping[str, Any]:
        self.reset_calls.append(index)
        return {"status": "ok", "index": index}

    def observe(self) -> str:
        return "Public initial observation"

    def normalize_initial_policy_context(
        self, messages: Sequence[Mapping[str, str]]
    ) -> Sequence[Mapping[str, str]]:
        return messages

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        initial: bool = False,
    ) -> None:
        self.bound_contexts.append(
            (initial, [dict(message) for message in messages])
        )

    def policy_turn_candidate(self) -> Optional[str]:
        index = len(self.step_calls)
        if index >= len(self.plan):
            return None
        candidate = self.plan[index].get("control_request")
        if candidate is not None and not self.allow_compaction:
            raise RuntimeError("policy compaction is disabled")
        return None if candidate is None else str(candidate)

    def prepare_policy_turn(self, pressure: Any) -> Optional[str]:
        del pressure
        return self.policy_turn_candidate()

    def step(self, policy_output: str) -> StepOutput:
        index = len(self.step_calls)
        if index >= len(self.plan):
            raise AssertionError("fake environment plan is exhausted")
        item = self.plan[index]
        self.step_calls.append(policy_output)
        error = item.get("raise")
        if error is not None:
            raise error
        operation = item.get("operation", "append_observation")
        if operation == "replace_messages" and not self.allow_compaction:
            raise RuntimeError("policy compaction is disabled")
        memory_action = policy_output.startswith(("MEMORY_WRITE(", "MEMORY_READ("))
        if memory_action:
            if self.memory_service is None:
                raise RuntimeError("external memory is forbidden for this treatment")
            selected_route = ("external_memory", "external_memory")
            memory_receipt = self.memory_service.execute(policy_output)
        elif operation == "replace_messages":
            selected_route = ("policy_compaction", "policy_context")
            memory_receipt = None
        else:
            selected_route = ("benchmark_task", "benchmark_task")
            memory_receipt = None
        selected_root = self.roots_by_route[selected_route]
        route = selected_root.to_payload()
        route.update(dict(item.get("route_override", {})))
        self.route_calls.append(route)
        transition: dict[str, Any] = {
            "schema": "agentmemory_task_neutral_context_transition_v1",
            "operation": operation,
            "messages": [],
        }
        if operation == "replace_messages":
            transition["messages"] = list(
                item.get(
                    "replacement_messages",
                    (
                        {"role": "system", "content": "Public task framing"},
                        {"role": "user", "content": "Continue the same task."},
                    ),
                )
            )
        execution_kind = item.get(
            "execution_kind_override",
            ROUTE_EXECUTION_KINDS.get(
                (route["capability_id"], route["root_kind"])
            ),
        )
        if execution_kind == "external_memory_action":
            execution_receipt = {
                "operation": (
                    None if memory_receipt is None else memory_receipt["operation"]
                ),
                "submission_sha256": sha256_text(policy_output),
                "memory_receipt_sha256": sha256_json(memory_receipt),
            }
        elif execution_kind == "policy_compaction":
            execution_receipt = {
                "summary_sha256": sha256_text(policy_output),
                "trigger_sha256": sha256_text(str(item.get("control_request", ""))),
                "replacement_context_sha256": sha256_json(
                    transition["messages"]
                ),
            }
        else:
            execution_receipt = {
                "submission_sha256": sha256_text(policy_output),
                "adapter_receipt_sha256": sha256_json(
                    {"index": index, "operation": operation}
                ),
            }
        execution = {
            "schema": WRAPPER_EXECUTION_SCHEMA,
            "kind": execution_kind,
            "status": "ok",
            "receipt": execution_receipt,
        }
        info = {
            "schema": "agentmemory_task_neutral_transition_v1",
            "env_info": {},
            "context_transition": transition,
            "action_submission": {
                "accepted": True,
                **route,
                "policy_output_sha256": item.get(
                    "policy_output_sha256_override",
                    sha256_text(policy_output),
                ),
                "execution": execution,
                "execution_sha256": sha256_json(execution),
            },
            "native_step_before": index,
            "native_step_after": index + 1,
            "native_call_count_before": index,
            "native_call_count_after": index + 1,
            "context_epoch_before": index,
            "context_epoch_after": index + 1,
            "session_epoch_before": 0,
            "session_epoch_after": 0,
            "policy_step_before": index,
            "policy_step_after": index + 1,
            "wrapper_evidence": dict(item.get("wrapper_evidence", {})),
        }
        self.step_infos.append(info)
        return StepOutput(
            state=str(item.get("state", f"observation-{index + 1}")),
            reward=float(item.get("reward", 0.0)),
            done=bool(item.get("done", False)),
            info=info,
        )


class FakeAdapter:
    def __init__(
        self,
        config: RunConfig,
        store: PrivateEvidenceStore,
        plan: Sequence[Mapping[str, Any]],
        *,
        artifact_error: Optional[Exception] = None,
        scorer_error: Optional[Exception] = None,
        clock: Optional[ManualClock] = None,
        artifact_advance_seconds: float = 0.0,
        scorer_advance_seconds: float = 0.0,
        close_advance_seconds: float = 0.0,
        prompt_declaration_override: Optional[str] = None,
        invalid_close_result: bool = False,
    ) -> None:
        self.config = config
        self.store = store
        self.namespace = Namespace.from_config(config)
        self.roots = lifecycle_roots(config)
        roots_by_route = {root.route: root for root in self.roots}
        self.memory_service = (
            FakeMemoryService(
                self.namespace,
                roots_by_route[("external_memory", "external_memory")],
            )
            if config.capability.external_read_write_memory
            else None
        )
        self.client = FakeClient(
            plan,
            namespace=self.namespace,
            roots=self.roots,
            memory_service=self.memory_service,
            allow_compaction=config.capability.policy_authored_compaction,
        )
        self.artifact_error = artifact_error
        self.scorer_error = scorer_error
        self.clock = clock
        self.artifact_advance_seconds = float(artifact_advance_seconds)
        self.scorer_advance_seconds = float(scorer_advance_seconds)
        self.close_advance_seconds = float(close_advance_seconds)
        self.prompt_declaration_override = prompt_declaration_override
        self.invalid_close_result = invalid_close_result
        self.finalization_contexts: list[FinalizationContext] = []
        self.close_calls = 0

    def reset(self, config: RunConfig) -> AdapterReset:
        reset_response = self.client.reset(config.task.task_index)
        memory_reset = (
            None
            if self.memory_service is None
            else self.memory_service.reset(self.namespace)
        )
        base_messages = (
            {"role": "system", "content": "Public task framing"},
            {"role": "user", "content": self.client.observe()},
        )
        actual_messages = list(base_messages)
        declaration = (
            FAKE_MEMORY_PROMPT_SUFFIX
            if (
                config.capability.external_read_write_memory
                and self.prompt_declaration_override is None
            )
            else self.prompt_declaration_override or ""
        )
        if declaration:
            actual_messages[0] = {
                "role": "system",
                "content": "Public task framing" + declaration,
            }
        return AdapterReset(
            namespace=Namespace.from_config(config),
            initial_messages=tuple(actual_messages),
            treatment_excluded_messages=base_messages,
            roots=self.roots,
            receipt={
                "status": "ok",
                "namespace": Namespace.from_config(config).to_payload(),
                "external_memory_created": (
                    config.capability.external_read_write_memory
                ),
                "external_memory_reset": memory_reset,
                "hidden_context_injection": False,
                "reset_response": reset_response,
            },
        )

    def finalize_artifact(
        self, context: FinalizationContext
    ) -> ArtifactResult:
        self.finalization_contexts.append(context)
        if self.artifact_error is not None:
            raise self.artifact_error
        if self.clock is not None:
            self.clock.advance(self.artifact_advance_seconds)
        artifact_content = {
            "private_artifact": (
                f"{self.config.task.benchmark}:{self.config.task.task_id}:secret"
            )
        }
        artifact_ref = self.store.put_json("artifacts", artifact_content)
        return ArtifactResult(
            artifact_type=self.config.task.artifact_type,
            protected_ref=artifact_ref.protected_ref,
            sha256=artifact_ref.sha256,
            receipt={"status": "ok", "artifact": artifact_ref.to_payload()},
        )

    def handoff_to_grader(self, artifact: ArtifactResult) -> ScorerResult:
        if self.scorer_error is not None:
            raise self.scorer_error
        if self.clock is not None:
            self.clock.advance(self.scorer_advance_seconds)
        receipt = {
            "status": "ok",
            "artifact_sha256": artifact.sha256,
            "private_grader_detail": "not public",
        }
        return ScorerResult(
            name=self.config.grader.name,
            revision=self.config.grader.revision,
            config_sha256=self.config.grader.config_sha256,
            public_metrics={"score": 1.0, "passed": True},
            receipt=receipt,
        )

    def close(self) -> AdapterClose:
        self.close_calls += 1
        memory_close = (
            None
            if self.memory_service is None
            else self.memory_service.close()
        )
        if self.clock is not None:
            self.clock.advance(self.close_advance_seconds)
        if self.invalid_close_result:
            return None  # type: ignore[return-value]
        return AdapterClose(
            namespace=self.namespace,
            closed_roots=self.roots,
            receipt={
                "status": "closed",
                "namespace": self.namespace.to_payload(),
                "external_memory": memory_close,
            },
        )


def terminal_plan() -> list[dict[str, Any]]:
    return [
        {
            "state": "terminal observation",
            "reward": 1.0,
            "done": True,
            "wrapper_evidence": {"outcome": "success"},
        }
    ]


def make_fake_runtime(
    config: RunConfig,
    store: PrivateEvidenceStore,
    *,
    plan: Optional[Sequence[Mapping[str, Any]]] = None,
    outputs: Optional[Sequence[str]] = None,
    model_fail_on_call: Optional[int] = None,
    tokenize_fail_on_call: Optional[int] = None,
    artifact_error: Optional[Exception] = None,
    scorer_error: Optional[Exception] = None,
    clock: Optional[ManualClock] = None,
    model_advance_seconds: float = 0.0,
    artifact_advance_seconds: float = 0.0,
    scorer_advance_seconds: float = 0.0,
    close_advance_seconds: float = 0.0,
    prompt_declaration_override: Optional[str] = None,
    invalid_close_result: bool = False,
):
    from paired_eval.manifest import RuntimeBindings

    selected_plan = terminal_plan() if plan is None else plan
    selected_outputs = ("ordinary-policy-output",) if outputs is None else outputs
    adapter = FakeAdapter(
        config,
        store,
        selected_plan,
        artifact_error=artifact_error,
        scorer_error=scorer_error,
        clock=clock,
        artifact_advance_seconds=artifact_advance_seconds,
        scorer_advance_seconds=scorer_advance_seconds,
        close_advance_seconds=close_advance_seconds,
        prompt_declaration_override=prompt_declaration_override,
        invalid_close_result=invalid_close_result,
    )
    model = FakeModel(
        config.model,
        store,
        selected_outputs,
        fail_on_call=model_fail_on_call,
        tokenize_fail_on_call=tokenize_fail_on_call,
        clock=clock,
        advance_seconds=model_advance_seconds,
    )
    return RuntimeBindings(adapter=adapter, model=model)
