"""Single task-agnostic sampling loop for paired external evaluation."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Optional

from .contracts import (
    AdapterClose,
    AdapterReset,
    ArtifactResult,
    CapabilityRoot,
    CONTEXT_OPERATION_REPLACE,
    EnvironmentAdapterProtocol,
    FAILURE_CLASSES,
    FinalizationContext,
    ModelClientFailure,
    ModelClientProtocol,
    ModelOutput,
    Namespace,
    POLICY_COMPACTION_ROUTE,
    PolicyTurnControllerProtocol,
    RESULT_SCHEMA,
    RESULT_SCHEMA_VERSION,
    RunConfig,
    ScorerResult,
)
from .controller import normalize_messages
from .evidence import PrivateEvidenceStore
from .serialization import sha256_json, sha256_text, token_ids_sha256


def validate_prompt_binding(
    initial_messages,
    treatment_excluded_messages,
    capability_declaration: str,
):
    """Require the full prompt to be one exact frozen capability transform."""

    excluded = normalize_messages(treatment_excluded_messages)
    declared = normalize_messages(initial_messages)
    expected = [dict(message) for message in excluded]
    if capability_declaration:
        if expected[0]["role"] != "system":
            raise ValueError(
                "capability declaration requires a leading system message"
            )
        expected[0]["content"] += "\n" + capability_declaration
    if declared != tuple(expected):
        raise ValueError(
            "initial prompt differs beyond the frozen capability declaration"
        )
    return declared, excluded


def exception_chain_contains(error: BaseException, kind: type) -> bool:
    """Find a typed cause through controller wrappers without string matching."""

    seen = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        if isinstance(current, kind):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def validate_declared_roots(
    config: RunConfig,
    namespace: Namespace,
    roots: tuple[CapabilityRoot, ...],
) -> None:
    """Match wrapper-declared task-neutral routes to the frozen capability."""

    if {root.route for root in roots} != set(config.capability.allowed_routes):
        raise ValueError("adapter lifecycle routes do not match the run capability")
    if any(root.namespace_sha256 != namespace.sha256 for root in roots):
        raise ValueError("adapter lifecycle root has the wrong namespace")


class PairedRunner:
    """Drive model -> ordinary environment step -> neutral receipt accounting."""

    def __init__(
        self,
        *,
        controller: PolicyTurnControllerProtocol,
        evidence_store: PrivateEvidenceStore,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.controller = controller
        self.evidence_store = evidence_store
        self.clock = clock

    def append_receipt(
        self,
        receipts: list[dict[str, Any]],
        kind: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        reference = self.evidence_store.put_json("receipts", payload)
        envelope = {
            "sequence": len(receipts),
            "kind": kind,
            **reference.to_payload(),
        }
        receipts.append(envelope)
        return envelope

    def record_error(
        self,
        receipts: list[dict[str, Any]],
        *,
        stage: str,
        failure_class: str,
        error: BaseException,
    ) -> dict[str, Any]:
        if failure_class not in FAILURE_CLASSES:
            raise ValueError("unsupported paired-evaluation failure class")
        payload = {
            "stage": stage,
            "failure_class": failure_class,
            "error_type": type(error).__name__,
            "error_code": getattr(error, "code", None),
            "message": str(error),
        }
        reference = self.evidence_store.put_json("errors", payload)
        self.append_receipt(
            receipts,
            "error",
            {
                "stage": stage,
                "failure_class": failure_class,
                "error": reference.to_payload(),
            },
        )
        return {
            "class": failure_class,
            "stage": stage,
            "timed_out": failure_class == "wall_timeout",
            "protected_ref": reference.protected_ref,
            "message_sha256": reference.sha256,
        }

    def run_task(
        self,
        config: RunConfig,
        adapter: EnvironmentAdapterProtocol,
        model: ModelClientProtocol,
    ) -> dict[str, Any]:
        """Execute one configured task without interpreting its action language."""

        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if model.model_config != config.model:
            raise ValueError("model client identity does not match run config")

        started = self.clock()
        expected_namespace = Namespace.from_config(config)
        receipts: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        policy_turns = 0
        tool_calls = 0
        prompt_tokens = 0
        response_tokens = 0
        retry_count = 0
        compaction_receipts = 0
        termination_reason: Optional[str] = None
        horizon_cause: Optional[str] = None
        failure: Optional[dict[str, Any]] = None
        reset_completed = False
        prompt_full_sha256: Optional[str] = None
        prompt_treatment_excluded_sha256: Optional[str] = None
        prompt_full_ref: Optional[str] = None
        prompt_treatment_excluded_ref: Optional[str] = None
        final_artifact: Optional[dict[str, Any]] = None
        scorer: Optional[dict[str, Any]] = None
        declared_roots: tuple[CapabilityRoot, ...] = ()
        closed_roots: Optional[tuple[CapabilityRoot, ...]] = None
        reset_receipt_ref: Optional[str] = None
        close_receipt_ref: Optional[str] = None
        messages = ()

        try:
            try:
                reset_result = adapter.reset(config)
                if not isinstance(reset_result, AdapterReset):
                    raise TypeError("adapter reset must return AdapterReset")
                if reset_result.namespace != expected_namespace:
                    raise ValueError("adapter reset returned the wrong namespace")
                candidate_declared_roots = tuple(reset_result.roots)
                validate_declared_roots(
                    config,
                    expected_namespace,
                    candidate_declared_roots,
                )
                declared_roots = candidate_declared_roots
                declared_messages, excluded_messages = validate_prompt_binding(
                    reset_result.initial_messages,
                    reset_result.treatment_excluded_messages,
                    config.capability.prompt_declaration,
                )
                messages = self.controller.bind_initial(
                    adapter.client,
                    declared_messages,
                )
                if messages != declared_messages:
                    raise ValueError(
                        "client normalization injected undeclared prompt state"
                    )
                full_prompt_evidence = self.evidence_store.put_json(
                    "prompts",
                    [dict(message) for message in messages]
                )
                excluded_prompt_evidence = self.evidence_store.put_json(
                    "prompts",
                    [dict(message) for message in excluded_messages]
                )
                prompt_full_sha256 = full_prompt_evidence.sha256
                prompt_treatment_excluded_sha256 = (
                    excluded_prompt_evidence.sha256
                )
                prompt_full_ref = full_prompt_evidence.protected_ref
                prompt_treatment_excluded_ref = (
                    excluded_prompt_evidence.protected_ref
                )
                reset_receipt = self.append_receipt(
                    receipts,
                    "namespace_reset",
                    {
                        "namespace": expected_namespace.to_payload(),
                        "roots": [root.to_payload() for root in declared_roots],
                        "adapter_receipt": reset_result.receipt,
                        "prompt_full_sha256": prompt_full_sha256,
                        "prompt_full_ref": prompt_full_ref,
                        "prompt_treatment_excluded_sha256": (
                            prompt_treatment_excluded_sha256
                        ),
                        "prompt_treatment_excluded_ref": (
                            prompt_treatment_excluded_ref
                        ),
                    },
                )
                reset_receipt_ref = reset_receipt["protected_ref"]
                reset_completed = True
            except Exception as error:
                failure = self.record_error(
                    receipts,
                    stage="reset",
                    failure_class="environment_failure",
                    error=error,
                )
                termination_reason = "failure"

            while termination_reason is None:
                elapsed = self.clock() - started
                if elapsed >= config.budgets.max_wall_seconds:
                    failure = self.record_error(
                        receipts,
                        stage="before_model",
                        failure_class="wall_timeout",
                        error=TimeoutError("paired evaluation wall budget exhausted"),
                    )
                    termination_reason = "timeout"
                    break
                if policy_turns >= config.budgets.max_policy_turns:
                    termination_reason = "horizon"
                    horizon_cause = "policy_turn_limit"
                    break
                if tool_calls >= config.budgets.max_tool_calls:
                    termination_reason = "horizon"
                    horizon_cause = "tool_call_limit"
                    break
                if prompt_tokens + response_tokens >= config.budgets.max_total_tokens:
                    termination_reason = "horizon"
                    horizon_cause = "token_limit"
                    break

                try:
                    prepared = self.controller.prepare(
                        adapter.client,
                        messages,
                        count_prompt_tokens=model.count_prompt_tokens,
                        budgets=config.budgets,
                        max_response_tokens=config.decoding.max_output_tokens,
                    )
                except Exception as error:
                    model_side = exception_chain_contains(
                        error,
                        ModelClientFailure,
                    )
                    failure = self.record_error(
                        receipts,
                        stage=(
                            "model_tokenization"
                            if model_side
                            else "prepare_policy_turn"
                        ),
                        failure_class=(
                            "model_failure"
                            if model_side
                            else "environment_failure"
                        ),
                        error=error,
                    )
                    termination_reason = "failure"
                    break

                remaining_tokens = config.budgets.max_total_tokens - (
                    prompt_tokens + response_tokens
                )
                if prepared.prompt_token_count > config.budgets.max_prompt_tokens:
                    termination_reason = "horizon"
                    horizon_cause = "prompt_token_limit"
                    break
                if prepared.prompt_token_count >= remaining_tokens:
                    termination_reason = "horizon"
                    horizon_cause = "token_limit"
                    break
                model_response_capacity = (
                    config.budgets.max_model_tokens
                    - prepared.prompt_token_count
                )
                if model_response_capacity <= 0:
                    termination_reason = "horizon"
                    horizon_cause = "prompt_token_limit"
                    break
                response_capacity = min(
                    config.decoding.max_output_tokens,
                    remaining_tokens - prepared.prompt_token_count,
                    model_response_capacity,
                )
                decoding = config.decoding.with_max_output_tokens(response_capacity)

                try:
                    model_output = model.complete(
                        prepared.messages,
                        decoding,
                        config.task.seed,
                    )
                    if not isinstance(model_output, ModelOutput):
                        raise TypeError("model client must return ModelOutput")
                except Exception as error:
                    failure = self.record_error(
                        receipts,
                        stage="model",
                        failure_class="model_failure",
                        error=error,
                    )
                    termination_reason = "failure"
                    break

                policy_turns += 1
                actual_prompt_tokens = len(model_output.prompt_token_ids)
                actual_response_tokens = len(model_output.response_token_ids)
                prompt_tokens += actual_prompt_tokens
                response_tokens += actual_response_tokens
                retry_count += model_output.retry_count
                model_receipt = self.append_receipt(
                    receipts,
                    "model",
                    {
                        "turn": policy_turns,
                        "request": model_output.request_ref.to_payload(),
                        "response": model_output.response_ref.to_payload(),
                        "tokenization": model_output.tokenization_ref.to_payload(),
                        "prompt_token_ids_sha256": token_ids_sha256(
                            model_output.prompt_token_ids
                        ),
                        "response_token_ids_sha256": token_ids_sha256(
                            model_output.response_token_ids
                        ),
                        "finish_reason": model_output.finish_reason,
                        "retry_count": model_output.retry_count,
                    },
                )
                turn = {
                    "sequence": policy_turns,
                    "model_receipt_ref": model_receipt["protected_ref"],
                    "request_ref": model_output.request_ref.protected_ref,
                    "response_ref": model_output.response_ref.protected_ref,
                    "tokenization_ref": model_output.tokenization_ref.protected_ref,
                    "prompt_token_count": actual_prompt_tokens,
                    "response_token_count": actual_response_tokens,
                    "prompt_token_ids_sha256": token_ids_sha256(
                        model_output.prompt_token_ids
                    ),
                    "response_token_ids_sha256": token_ids_sha256(
                        model_output.response_token_ids
                    ),
                    "finish_reason": model_output.finish_reason,
                    "retry_count": model_output.retry_count,
                    "environment_ref": None,
                    "context_transition_ref": None,
                    "context_operation": None,
                    "capability_id": None,
                    "root_kind": None,
                    "root_id": None,
                    "policy_output_sha256": None,
                    "execution_kind": None,
                    "execution_sha256": None,
                    "execution_receipt": None,
                    "reward": None,
                    "done": None,
                }
                turns.append(turn)

                contract_error: Optional[Exception] = None
                if actual_prompt_tokens != prepared.prompt_token_count:
                    contract_error = ValueError(
                        "model prompt-token evidence disagrees with preparation"
                    )
                elif actual_response_tokens > response_capacity:
                    contract_error = ValueError(
                        "model response exceeded the requested token capacity"
                    )
                if contract_error is not None:
                    failure = self.record_error(
                        receipts,
                        stage="model_evidence",
                        failure_class="model_failure",
                        error=contract_error,
                    )
                    termination_reason = "failure"
                    break

                if self.clock() - started >= config.budgets.max_wall_seconds:
                    failure = self.record_error(
                        receipts,
                        stage="after_model",
                        failure_class="wall_timeout",
                        error=TimeoutError("model completed after the wall deadline"),
                    )
                    termination_reason = "timeout"
                    break

                tool_calls += 1
                try:
                    completed = self.controller.complete(
                        adapter.client,
                        prepared,
                        model_output.text,
                    )
                    step_output = completed.step_output
                    step_receipt = step_output.task_neutral_receipt
                    if step_receipt.policy_output_sha256 != sha256_text(
                        model_output.text
                    ):
                        raise ValueError(
                            "wrapper execution receipt is bound to another policy output"
                        )
                    if step_receipt.route not in declared_roots:
                        raise ValueError(
                            "environment step used an undeclared capability root"
                        )
                    if (
                        step_receipt.policy_step_before != policy_turns - 1
                        or step_receipt.policy_step_after != policy_turns
                    ):
                        raise ValueError(
                            "environment policy-step receipt is out of sequence"
                        )
                    if dict(step_receipt.context_transition) != dict(
                        completed.context_transition
                    ):
                        raise ValueError(
                            "environment context-transition receipts disagree"
                        )
                    replacement = (
                        completed.context_transition["operation"]
                        == CONTEXT_OPERATION_REPLACE
                    )
                    if replacement != (
                        step_receipt.route.route == POLICY_COMPACTION_ROUTE
                    ):
                        raise ValueError(
                            "compaction transition and capability route disagree"
                        )
                    environment_reference = self.evidence_store.put_json(
                        "environment_steps",
                        {
                            "turn": policy_turns,
                            "state": step_output.state,
                            "reward": float(step_output.reward),
                            "done": step_output.done,
                            "info": step_output.info,
                        },
                    )
                    environment_receipt = self.append_receipt(
                        receipts,
                        "environment",
                        {
                            "turn": policy_turns,
                            "step": environment_reference.to_payload(),
                        },
                    )
                    transition_reference = self.evidence_store.put_json(
                        "context_transitions",
                        {
                            "turn": policy_turns,
                            "transition": completed.context_transition,
                        },
                    )
                    transition_receipt = self.append_receipt(
                        receipts,
                        "context_transition",
                        {
                            "turn": policy_turns,
                            "transition": transition_reference.to_payload(),
                        },
                    )
                except Exception as error:
                    failure = self.record_error(
                        receipts,
                        stage="environment_step",
                        failure_class="environment_failure",
                        error=error,
                    )
                    termination_reason = "failure"
                    break

                messages = completed.messages
                operation = completed.context_transition["operation"]
                turn.update(
                    {
                        "environment_ref": environment_receipt["protected_ref"],
                        "context_transition_ref": transition_receipt[
                            "protected_ref"
                        ],
                        "context_operation": operation,
                        "capability_id": step_receipt.route.capability_id,
                        "root_kind": step_receipt.route.root_kind,
                        "root_id": step_receipt.route.root_id,
                        "policy_output_sha256": (
                            step_receipt.policy_output_sha256
                        ),
                        "execution_kind": step_receipt.execution_kind,
                        "execution_sha256": step_receipt.execution_sha256,
                        "execution_receipt": (
                            step_receipt.execution_receipt.to_payload()
                        ),
                        "reward": float(step_output.reward),
                        "done": step_output.done,
                    }
                )
                if operation == CONTEXT_OPERATION_REPLACE:
                    compaction_receipts += 1

                if self.clock() - started >= config.budgets.max_wall_seconds:
                    failure = self.record_error(
                        receipts,
                        stage="after_environment",
                        failure_class="wall_timeout",
                        error=TimeoutError(
                            "environment step completed after the wall deadline"
                        ),
                    )
                    termination_reason = "timeout"
                    break
                if step_output.done:
                    termination_reason = "terminal"
                    break

            if termination_reason is None:
                raise RuntimeError("paired runner exited without a termination reason")

            if reset_completed:
                context = FinalizationContext(
                    termination_reason=termination_reason,
                    horizon_cause=horizon_cause,
                    failure_class=None if failure is None else failure["class"],
                    timed_out=False if failure is None else failure["timed_out"],
                    policy_turns=policy_turns,
                    tool_calls=tool_calls,
                )
                try:
                    artifact_result = adapter.finalize_artifact(context)
                    if not isinstance(artifact_result, ArtifactResult):
                        raise TypeError(
                            "adapter artifact finalizer must return ArtifactResult"
                        )
                    if artifact_result.artifact_type != config.task.artifact_type:
                        raise ValueError("adapter returned the wrong artifact type")
                    artifact_receipt = self.append_receipt(
                        receipts,
                        "artifact",
                        {
                            "artifact_type": artifact_result.artifact_type,
                            "artifact": {
                                "protected_ref": artifact_result.protected_ref,
                                "sha256": artifact_result.sha256,
                            },
                            "adapter_receipt": artifact_result.receipt,
                        },
                    )
                    final_artifact = {
                        "type": artifact_result.artifact_type,
                        "protected_ref": artifact_result.protected_ref,
                        "sha256": artifact_result.sha256,
                        "receipt_ref": artifact_receipt["protected_ref"],
                    }
                except Exception as error:
                    artifact_error = self.record_error(
                        receipts,
                        stage="artifact",
                        failure_class="artifact_failure",
                        error=error,
                    )
                    if failure is None:
                        failure = artifact_error

                if final_artifact is not None:
                    try:
                        scorer_result = adapter.handoff_to_grader(artifact_result)
                        if not isinstance(scorer_result, ScorerResult):
                            raise TypeError(
                                "grader handoff must return ScorerResult"
                            )
                        if (
                            scorer_result.name != config.grader.name
                            or scorer_result.revision != config.grader.revision
                            or scorer_result.config_sha256
                            != config.grader.config_sha256
                        ):
                            raise ValueError(
                                "scorer identity does not match grader config"
                            )
                        scorer_receipt = self.append_receipt(
                            receipts,
                            "scorer",
                            {
                                "name": scorer_result.name,
                                "revision": scorer_result.revision,
                                "config_sha256": scorer_result.config_sha256,
                                "public_metrics": dict(
                                    scorer_result.public_metrics
                                ),
                                "grader_receipt": scorer_result.receipt,
                            },
                        )
                        scorer = {
                            "name": scorer_result.name,
                            "revision": scorer_result.revision,
                            "config_sha256": scorer_result.config_sha256,
                            "public_metrics": dict(
                                scorer_result.public_metrics
                            ),
                            "receipt_ref": scorer_receipt["protected_ref"],
                        }
                    except Exception as error:
                        scorer_error = self.record_error(
                            receipts,
                            stage="scorer",
                            failure_class="scorer_failure",
                            error=error,
                        )
                        if failure is None:
                            failure = scorer_error
        finally:
            try:
                close_result = adapter.close()
                if not isinstance(close_result, AdapterClose):
                    raise TypeError("adapter close must return AdapterClose")
                if close_result.namespace != expected_namespace:
                    raise ValueError("adapter close returned the wrong namespace")
                candidate_closed_roots = tuple(close_result.closed_roots)
                if set(candidate_closed_roots) != set(declared_roots):
                    raise ValueError(
                        "adapter close did not clean every declared lifecycle root"
                    )
                close_receipt = self.append_receipt(
                    receipts,
                    "close",
                    {
                        "status": "ok",
                        "namespace": close_result.namespace.to_payload(),
                        "closed_roots": [
                            root.to_payload() for root in candidate_closed_roots
                        ],
                        "adapter_receipt": close_result.receipt,
                    },
                )
                closed_roots = candidate_closed_roots
                close_receipt_ref = close_receipt["protected_ref"]
            except Exception as error:
                close_error = self.record_error(
                    receipts,
                    stage="close",
                    failure_class="environment_failure",
                    error=error,
                )
                if failure is None:
                    failure = close_error
                    if termination_reason is None:
                        termination_reason = "failure"

        if termination_reason is None:
            termination_reason = "failure"
        elapsed = max(0.0, self.clock() - started)
        if elapsed >= config.budgets.max_wall_seconds and failure is None:
            failure = self.record_error(
                receipts,
                stage="after_close",
                failure_class="wall_timeout",
                error=TimeoutError(
                    "finalization completed after the wall deadline"
                ),
            )
            termination_reason = "timeout"
            horizon_cause = None
            elapsed = max(0.0, self.clock() - started)
        comparable = (
            failure is None
            and termination_reason in {"terminal", "horizon"}
            and final_artifact is not None
            and scorer is not None
        )
        config_payload = config.to_payload()
        namespace_payload = expected_namespace.to_payload()
        row = {
            "schema": RESULT_SCHEMA,
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": config.run_id,
            "benchmark": config.task.benchmark,
            "protocol": config.task.protocol,
            "task_id": config.task.task_id,
            "task_index": config.task.task_index,
            "seed": config.task.seed,
            "arm": config.capability.arm.value,
            "pair_key": config.pair_key,
            "namespace": namespace_payload,
            "namespace_sha256": sha256_json(namespace_payload),
            "config": config_payload,
            "full_config_sha256": config.full_config_sha256,
            "treatment_excluded_config_sha256": (
                config.treatment_excluded_config_sha256
            ),
            "source": config.source.to_payload(),
            "model": config.model.to_payload(),
            "runtime": config.runtime.to_payload(),
            "decoding": config.decoding.to_payload(),
            "budgets": config.budgets.to_payload(),
            "grader": config.grader.to_payload(),
            "treatment": config.capability.to_payload(),
            "prompt": {
                "full_sha256": prompt_full_sha256,
                "full_ref": prompt_full_ref,
                "treatment_excluded_sha256": (
                    prompt_treatment_excluded_sha256
                ),
                "treatment_excluded_ref": prompt_treatment_excluded_ref,
                "hidden_context_injection": (
                    config.capability.hidden_context_injection
                ),
            },
            "lifecycle": {
                "declared_roots": [
                    root.to_payload() for root in declared_roots
                ],
                "reset_receipt_ref": reset_receipt_ref,
                "closed_roots": (
                    None
                    if closed_roots is None
                    else [root.to_payload() for root in closed_roots]
                ),
                "close_receipt_ref": close_receipt_ref,
            },
            "receipts": receipts,
            "turns": turns,
            "usage": {
                "policy_turns": policy_turns,
                "tool_calls": tool_calls,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": prompt_tokens + response_tokens,
                "retry_count": retry_count,
                "wall_seconds": elapsed,
            },
            "compaction": {
                **config.compaction.to_payload(),
                "receipt_count": compaction_receipts,
            },
            "termination": {
                "reason": termination_reason,
                "horizon_cause": horizon_cause,
            },
            "failure": failure
            or {
                "class": None,
                "stage": None,
                "timed_out": False,
                "protected_ref": None,
                "message_sha256": None,
            },
            "final_artifact": final_artifact,
            "scorer": scorer,
            "comparable": comparable,
        }
        return row
