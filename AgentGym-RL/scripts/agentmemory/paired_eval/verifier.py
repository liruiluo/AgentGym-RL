"""Fail-closed result validation, pair matching, and safe public summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    Arm,
    CapabilityRoot,
    CONTEXT_OPERATIONS,
    FAILURE_CLASSES,
    HORIZON_CAUSES,
    Namespace,
    PairKey,
    POLICY_COMPACTION_ROUTE,
    ROUTE_EXECUTION_KINDS,
    RESULT_SCHEMA,
    RESULT_SCHEMA_VERSION,
    RunConfig,
    SHA256_PATTERN,
    TERMINATION_REASONS,
    WRAPPER_EXECUTION_SCHEMA,
    capability_root_id,
    parse_wrapper_execution_receipt,
)
from .serialization import canonical_json_bytes, sha256_json


EVIDENCE_PATTERN = re.compile(r"^evidence://[a-z][a-z0-9_]*/[0-9a-f]{64}$")
RESULT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "run_id",
        "benchmark",
        "protocol",
        "task_id",
        "task_index",
        "seed",
        "arm",
        "pair_key",
        "namespace",
        "namespace_sha256",
        "config",
        "full_config_sha256",
        "treatment_excluded_config_sha256",
        "source",
        "model",
        "runtime",
        "decoding",
        "budgets",
        "grader",
        "treatment",
        "prompt",
        "lifecycle",
        "receipts",
        "turns",
        "usage",
        "compaction",
        "termination",
        "failure",
        "final_artifact",
        "scorer",
        "comparable",
    }
)
RECEIPT_KINDS = frozenset(
    {
        "namespace_reset",
        "model",
        "environment",
        "context_transition",
        "artifact",
        "scorer",
        "close",
        "error",
    }
)
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "content",
        "messages",
        "request_messages",
        "model_text",
        "policy_output",
        "observation",
        "state",
        "info",
        "private_grader_detail",
        "grader_detail",
        "gold_answer",
    }
)


class ResultValidationError(ValueError):
    pass


class PairVerificationError(ValueError):
    pass


def require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultValidationError(f"{name} must be an object")
    return value


def require_exact_keys(
    name: str,
    value: Mapping[str, Any],
    expected: Sequence[str],
) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        extra = sorted(actual - wanted)
        raise ResultValidationError(
            f"{name} keys mismatch; missing={missing}, extra={extra}"
        )


def require_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResultValidationError(f"{name} must be a non-negative integer")
    return value


def require_positive_int(name: str, value: Any) -> int:
    number = require_nonnegative_int(name, value)
    if number == 0:
        raise ResultValidationError(f"{name} must be positive")
    return number


def require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ResultValidationError(f"{name} must be a lowercase SHA-256")
    return value


def require_evidence_ref(name: str, value: Any) -> str:
    if not isinstance(value, str) or EVIDENCE_PATTERN.fullmatch(value) is None:
        raise ResultValidationError(f"{name} must be a digest-addressed evidence URI")
    return value


def require_evidence_sha(name: str, reference: Any, digest: Any) -> None:
    protected_ref = require_evidence_ref(f"{name}.protected_ref", reference)
    sha256 = require_sha256(f"{name}.sha256", digest)
    if protected_ref.rsplit("/", 1)[-1] != sha256:
        raise ResultValidationError(f"{name} reference and SHA-256 disagree")


def reject_forbidden_keys(value: Any, path: str = "result") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in FORBIDDEN_PUBLIC_KEYS:
                raise ResultValidationError(
                    f"raw/private field {path}.{key} is forbidden"
                )
            reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_forbidden_keys(child, f"{path}[{index}]")


def validate_public_metrics(value: Any) -> None:
    metrics = require_mapping("scorer.public_metrics", value)
    for name, metric in metrics.items():
        if not isinstance(name, str) or not name:
            raise ResultValidationError("public metric names must be nonempty text")
        if name in FORBIDDEN_PUBLIC_KEYS:
            raise ResultValidationError("private-looking metric name is forbidden")
        if metric is not None and type(metric) not in {bool, int, float}:
            raise ResultValidationError(
                "public metric values must be scalar numbers, booleans, or null"
            )
        if isinstance(metric, float) and not math.isfinite(metric):
            raise ResultValidationError("public metric floats must be finite")


def parse_capability_roots(
    name: str,
    value: Any,
    namespace: Namespace,
) -> tuple[CapabilityRoot, ...]:
    if not isinstance(value, list):
        raise ResultValidationError(f"{name} must be a list")
    roots = []
    try:
        for index, payload in enumerate(value):
            root = CapabilityRoot.from_payload(
                require_mapping(f"{name}[{index}]", payload)
            )
            if root.namespace_sha256 != namespace.sha256:
                raise ResultValidationError(
                    f"{name}[{index}] is bound to the wrong namespace"
                )
            if root.root_id != capability_root_id(
                namespace,
                root.capability_id,
                root.root_kind,
            ):
                raise ResultValidationError(
                    f"{name}[{index}] root id is not namespace/route bound"
                )
            roots.append(root)
    except (TypeError, ValueError) as error:
        if isinstance(error, ResultValidationError):
            raise
        raise ResultValidationError(f"{name} contains an invalid root") from error
    routes = [root.route for root in roots]
    root_ids = [root.root_id for root in roots]
    if len(routes) != len(set(routes)):
        raise ResultValidationError(f"{name} contains duplicate routes")
    if len(root_ids) != len(set(root_ids)):
        raise ResultValidationError(f"{name} reuses a root id")
    return tuple(roots)


def validate_result_row(row: Mapping[str, Any]) -> None:
    """Validate one private JSONL row without resolving protected evidence."""

    result = require_mapping("result", row)
    require_exact_keys("result", result, RESULT_KEYS)
    try:
        canonical_json_bytes(result)
    except (TypeError, ValueError) as error:
        raise ResultValidationError(
            "result is not canonical-JSON serializable"
        ) from error
    reject_forbidden_keys(result)

    if result["schema"] != RESULT_SCHEMA:
        raise ResultValidationError("unsupported result schema")
    if result["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ResultValidationError("unsupported result schema version")

    config_payload = require_mapping("config", result["config"])
    try:
        config = RunConfig.from_payload(config_payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ResultValidationError("invalid embedded run config") from error
    if config.to_payload() != config_payload:
        raise ResultValidationError(
            "embedded run config has unknown or noncanonical fields"
        )

    identity = {
        "run_id": config.run_id,
        "benchmark": config.task.benchmark,
        "protocol": config.task.protocol,
        "task_id": config.task.task_id,
        "task_index": config.task.task_index,
        "seed": config.task.seed,
        "arm": config.capability.arm.value,
        "pair_key": config.pair_key,
    }
    for name, expected in identity.items():
        if result[name] != expected:
            raise ResultValidationError(f"result {name} disagrees with its config")
    if result["full_config_sha256"] != config.full_config_sha256:
        raise ResultValidationError("full config digest mismatch")
    if (
        result["treatment_excluded_config_sha256"]
        != config.treatment_excluded_config_sha256
    ):
        raise ResultValidationError("treatment-excluded config digest mismatch")

    mirrors = {
        "source": config.source.to_payload(),
        "model": config.model.to_payload(),
        "runtime": config.runtime.to_payload(),
        "decoding": config.decoding.to_payload(),
        "budgets": config.budgets.to_payload(),
        "grader": config.grader.to_payload(),
        "treatment": config.capability.to_payload(),
    }
    for name, expected in mirrors.items():
        if result[name] != expected:
            raise ResultValidationError(f"result {name} mirror drifted from config")

    expected_namespace = {
        "run_id": config.run_id,
        "benchmark": config.task.benchmark,
        "protocol": config.task.protocol,
        "task_id": config.task.task_id,
        "seed": config.task.seed,
        "arm": config.capability.arm.value,
    }
    namespace = require_mapping("namespace", result["namespace"])
    if namespace != expected_namespace:
        raise ResultValidationError(
            "namespace does not isolate this run/task/seed/treatment"
        )
    require_sha256("namespace_sha256", result["namespace_sha256"])
    if result["namespace_sha256"] != sha256_json(namespace):
        raise ResultValidationError("namespace digest mismatch")
    namespace_contract = Namespace.from_config(config)

    prompt = require_mapping("prompt", result["prompt"])
    require_exact_keys(
        "prompt",
        prompt,
        {
            "full_sha256",
            "full_ref",
            "treatment_excluded_sha256",
            "treatment_excluded_ref",
            "hidden_context_injection",
        },
    )
    full_prompt = prompt["full_sha256"]
    excluded_prompt = prompt["treatment_excluded_sha256"]
    full_prompt_ref = prompt["full_ref"]
    excluded_prompt_ref = prompt["treatment_excluded_ref"]
    if (full_prompt is None) != (excluded_prompt is None):
        raise ResultValidationError(
            "prompt digests must be both present or both absent"
        )
    if full_prompt is not None:
        require_evidence_sha("prompt.full", full_prompt_ref, full_prompt)
        require_evidence_sha(
            "prompt.treatment_excluded",
            excluded_prompt_ref,
            excluded_prompt,
        )
    elif full_prompt_ref is not None or excluded_prompt_ref is not None:
        raise ResultValidationError(
            "absent prompt digests cannot have protected references"
        )
    if prompt["hidden_context_injection"] is not False:
        raise ResultValidationError("hidden context injection is forbidden")

    lifecycle = require_mapping("lifecycle", result["lifecycle"])
    require_exact_keys(
        "lifecycle",
        lifecycle,
        {
            "declared_roots",
            "reset_receipt_ref",
            "closed_roots",
            "close_receipt_ref",
        },
    )
    declared_roots = parse_capability_roots(
        "lifecycle.declared_roots",
        lifecycle["declared_roots"],
        namespace_contract,
    )
    expected_routes = set(config.capability.allowed_routes)
    declared_routes_match = {
        root.route for root in declared_roots
    } == expected_routes
    if declared_roots and not declared_routes_match:
        raise ResultValidationError(
            "lifecycle routes do not match the frozen capability"
        )
    if lifecycle["closed_roots"] is None:
        closed_roots = None
    else:
        closed_roots = parse_capability_roots(
            "lifecycle.closed_roots",
            lifecycle["closed_roots"],
            namespace_contract,
        )
    reset_receipt_ref = lifecycle["reset_receipt_ref"]
    close_receipt_ref = lifecycle["close_receipt_ref"]
    if reset_receipt_ref is not None:
        require_evidence_ref(
            "lifecycle.reset_receipt_ref",
            reset_receipt_ref,
        )
    if close_receipt_ref is not None:
        require_evidence_ref(
            "lifecycle.close_receipt_ref",
            close_receipt_ref,
        )
    if (closed_roots is None) != (close_receipt_ref is None):
        raise ResultValidationError(
            "closed roots and close receipt reference must appear together"
        )

    receipts = result["receipts"]
    if not isinstance(receipts, list):
        raise ResultValidationError("receipts must be a list")
    receipt_refs = set()
    receipt_by_ref = {}
    for index, receipt_value in enumerate(receipts):
        receipt = require_mapping(f"receipts[{index}]", receipt_value)
        require_exact_keys(
            f"receipts[{index}]",
            receipt,
            {
                "sequence",
                "kind",
                "protected_ref",
                "sha256",
                "byte_count",
                "media_type",
            },
        )
        if receipt["sequence"] != index:
            raise ResultValidationError("receipt sequence is not contiguous")
        if receipt["kind"] not in RECEIPT_KINDS:
            raise ResultValidationError("receipt kind is unsupported")
        require_evidence_sha(
            f"receipts[{index}]",
            receipt["protected_ref"],
            receipt["sha256"],
        )
        reference = receipt["protected_ref"]
        require_nonnegative_int(
            f"receipts[{index}].byte_count", receipt["byte_count"]
        )
        if receipt["media_type"] != "application/json":
            raise ResultValidationError("receipt evidence must be JSON")
        if reference in receipt_refs:
            raise ResultValidationError("duplicate receipt reference")
        receipt_refs.add(reference)
        receipt_by_ref[reference] = receipt
    if full_prompt is not None:
        if not receipts or receipts[0]["kind"] != "namespace_reset":
            raise ResultValidationError(
                "initialized rows must begin with a namespace-reset receipt"
            )
    close_receipts = [receipt for receipt in receipts if receipt["kind"] == "close"]
    if len(close_receipts) > 1:
        raise ResultValidationError("row contains duplicate close receipts")
    if (full_prompt is None) != (reset_receipt_ref is None):
        raise ResultValidationError(
            "prompt binding and lifecycle reset receipt must appear together"
        )
    if reset_receipt_ref is not None:
        if reset_receipt_ref not in receipt_refs:
            raise ResultValidationError("lifecycle reset receipt is missing")
        if receipt_by_ref[reset_receipt_ref]["kind"] != "namespace_reset":
            raise ResultValidationError(
                "lifecycle reset reference has the wrong receipt kind"
            )
    if close_receipt_ref is not None:
        if close_receipt_ref not in receipt_refs:
            raise ResultValidationError("lifecycle close receipt is missing")
        if receipt_by_ref[close_receipt_ref]["kind"] != "close":
            raise ResultValidationError(
                "lifecycle close reference has the wrong receipt kind"
            )
        if set(closed_roots or ()) != set(declared_roots):
            raise ResultValidationError(
                "lifecycle close did not clean every declared root"
            )

    turns = result["turns"]
    if not isinstance(turns, list):
        raise ResultValidationError("turns must be a list")
    prompt_total = 0
    response_total = 0
    retry_total = 0
    completed_steps = 0
    replacement_count = 0
    last_turn_receipt_sequence = 0
    for index, turn_value in enumerate(turns, start=1):
        turn = require_mapping(f"turns[{index}]", turn_value)
        require_exact_keys(
            f"turns[{index}]",
            turn,
            {
                "sequence",
                "model_receipt_ref",
                "request_ref",
                "response_ref",
                "tokenization_ref",
                "prompt_token_count",
                "response_token_count",
                "prompt_token_ids_sha256",
                "response_token_ids_sha256",
                "finish_reason",
                "retry_count",
                "environment_ref",
                "context_transition_ref",
                "context_operation",
                "capability_id",
                "root_kind",
                "root_id",
                "policy_output_sha256",
                "execution_kind",
                "execution_sha256",
                "execution_receipt",
                "reward",
                "done",
            },
        )
        if turn["sequence"] != index:
            raise ResultValidationError("turn sequence is not contiguous")
        for name in (
            "model_receipt_ref",
            "request_ref",
            "response_ref",
            "tokenization_ref",
        ):
            require_evidence_ref(f"turns[{index}].{name}", turn[name])
        if turn["model_receipt_ref"] not in receipt_refs:
            raise ResultValidationError("turn model receipt is missing")
        model_receipt = receipt_by_ref[turn["model_receipt_ref"]]
        if model_receipt["kind"] != "model":
            raise ResultValidationError("turn model reference has the wrong kind")
        if model_receipt["sequence"] <= last_turn_receipt_sequence:
            raise ResultValidationError("model receipts are out of order")
        prompt_total += require_positive_int(
            f"turns[{index}].prompt_token_count", turn["prompt_token_count"]
        )
        response_total += require_positive_int(
            f"turns[{index}].response_token_count", turn["response_token_count"]
        )
        require_sha256(
            f"turns[{index}].prompt_token_ids_sha256",
            turn["prompt_token_ids_sha256"],
        )
        require_sha256(
            f"turns[{index}].response_token_ids_sha256",
            turn["response_token_ids_sha256"],
        )
        if not isinstance(turn["finish_reason"], str) or not turn["finish_reason"]:
            raise ResultValidationError("finish reason must be nonempty text")
        retry_total += require_nonnegative_int(
            f"turns[{index}].retry_count", turn["retry_count"]
        )
        environment_fields = (
            turn["environment_ref"],
            turn["context_transition_ref"],
            turn["context_operation"],
            turn["capability_id"],
            turn["root_kind"],
            turn["root_id"],
            turn["policy_output_sha256"],
            turn["execution_kind"],
            turn["execution_sha256"],
            turn["execution_receipt"],
            turn["reward"],
            turn["done"],
        )
        if turn["environment_ref"] is None:
            if any(value is not None for value in environment_fields):
                raise ResultValidationError("partial environment turn evidence")
        else:
            completed_steps += 1
            require_evidence_ref(
                f"turns[{index}].environment_ref", turn["environment_ref"]
            )
            require_evidence_ref(
                f"turns[{index}].context_transition_ref",
                turn["context_transition_ref"],
            )
            if turn["environment_ref"] not in receipt_refs:
                raise ResultValidationError("turn environment receipt is missing")
            if turn["context_transition_ref"] not in receipt_refs:
                raise ResultValidationError("turn transition receipt is missing")
            environment_receipt = receipt_by_ref[turn["environment_ref"]]
            transition_receipt = receipt_by_ref[turn["context_transition_ref"]]
            if environment_receipt["kind"] != "environment":
                raise ResultValidationError(
                    "turn environment reference has the wrong kind"
                )
            if transition_receipt["kind"] != "context_transition":
                raise ResultValidationError(
                    "turn transition reference has the wrong kind"
                )
            if not (
                model_receipt["sequence"]
                < environment_receipt["sequence"]
                < transition_receipt["sequence"]
            ):
                raise ResultValidationError(
                    "model/environment/context receipts are out of order"
                )
            if turn["context_operation"] not in CONTEXT_OPERATIONS:
                raise ResultValidationError("unsupported context operation")
            route = (
                turn["capability_id"],
                turn["root_kind"],
                turn["root_id"],
            )
            if any(not isinstance(value, str) or not value for value in route):
                raise ResultValidationError(
                    "completed environment turn lacks a typed capability route"
                )
            matching_roots = [
                root
                for root in declared_roots
                if (
                    root.capability_id,
                    root.root_kind,
                    root.root_id,
                )
                == route
            ]
            if len(matching_roots) != 1:
                raise ResultValidationError(
                    "environment turn used an undeclared capability root"
                )
            require_sha256(
                f"turns[{index}].policy_output_sha256",
                turn["policy_output_sha256"],
            )
            require_sha256(
                f"turns[{index}].execution_sha256",
                turn["execution_sha256"],
            )
            if ROUTE_EXECUTION_KINDS.get(route[:2]) != turn["execution_kind"]:
                raise ResultValidationError(
                    "wrapper execution kind does not match its capability route"
                )
            try:
                execution_receipt = parse_wrapper_execution_receipt(
                    turn["execution_kind"],
                    turn["execution_receipt"],
                )
            except (TypeError, ValueError) as error:
                raise ResultValidationError(
                    "wrapper execution receipt is invalid"
                ) from error
            if execution_receipt.to_payload() != turn["execution_receipt"]:
                raise ResultValidationError(
                    "wrapper execution receipt is not canonical"
                )
            execution_attestation = {
                "schema": WRAPPER_EXECUTION_SCHEMA,
                "kind": turn["execution_kind"],
                "status": "ok",
                "receipt": turn["execution_receipt"],
            }
            if sha256_json(execution_attestation) != turn["execution_sha256"]:
                raise ResultValidationError(
                    "wrapper execution attestation digest mismatch"
                )
            bound_submission = getattr(
                execution_receipt,
                "submission_sha256",
                getattr(execution_receipt, "summary_sha256", None),
            )
            if bound_submission != turn["policy_output_sha256"]:
                raise ResultValidationError(
                    "wrapper execution receipt is bound to another policy output"
                )
            replacement = turn["context_operation"] == "replace_messages"
            compaction_route = route[:2] == POLICY_COMPACTION_ROUTE
            if replacement != compaction_route:
                raise ResultValidationError(
                    "compaction operation and capability route disagree"
                )
            reward = turn["reward"]
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(float(reward))
            ):
                raise ResultValidationError("turn reward must be finite")
            if type(turn["done"]) is not bool:
                raise ResultValidationError("turn done must be boolean")
            replacement_count += int(turn["context_operation"] == "replace_messages")
            last_turn_receipt_sequence = transition_receipt["sequence"]
        if turn["environment_ref"] is None:
            last_turn_receipt_sequence = model_receipt["sequence"]

    usage = require_mapping("usage", result["usage"])
    require_exact_keys(
        "usage",
        usage,
        {
            "policy_turns",
            "tool_calls",
            "prompt_tokens",
            "response_tokens",
            "total_tokens",
            "retry_count",
            "wall_seconds",
        },
    )
    policy_turns = require_nonnegative_int("usage.policy_turns", usage["policy_turns"])
    tool_calls = require_nonnegative_int("usage.tool_calls", usage["tool_calls"])
    if policy_turns != len(turns):
        raise ResultValidationError("policy turn accounting mismatch")
    if tool_calls < completed_steps or tool_calls > policy_turns:
        raise ResultValidationError("tool call accounting mismatch")
    if usage["prompt_tokens"] != prompt_total:
        raise ResultValidationError("prompt token accounting mismatch")
    if usage["response_tokens"] != response_total:
        raise ResultValidationError("response token accounting mismatch")
    if usage["total_tokens"] != prompt_total + response_total:
        raise ResultValidationError("total token accounting mismatch")
    if usage["retry_count"] != retry_total:
        raise ResultValidationError("retry accounting mismatch")
    wall_seconds = usage["wall_seconds"]
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) < 0
    ):
        raise ResultValidationError("wall usage must be finite and non-negative")
    if policy_turns > config.budgets.max_policy_turns:
        raise ResultValidationError("policy turn budget exceeded")
    if tool_calls > config.budgets.max_tool_calls:
        raise ResultValidationError("tool call budget exceeded")

    compaction = require_mapping("compaction", result["compaction"])
    expected_compaction = {
        **config.compaction.to_payload(),
        "receipt_count": replacement_count,
    }
    if compaction != expected_compaction:
        raise ResultValidationError("compaction accounting/config mismatch")
    if replacement_count and not config.capability.policy_authored_compaction:
        raise ResultValidationError(
            "disabled policy compaction emitted a replacement receipt"
        )

    termination = require_mapping("termination", result["termination"])
    require_exact_keys("termination", termination, {"reason", "horizon_cause"})
    reason = termination["reason"]
    if reason not in TERMINATION_REASONS:
        raise ResultValidationError("unsupported termination reason")
    if reason == "horizon":
        if termination["horizon_cause"] not in HORIZON_CAUSES:
            raise ResultValidationError("horizon cause is missing or unsupported")
    elif termination["horizon_cause"] is not None:
        raise ResultValidationError("non-horizon row has a horizon cause")
    if reason == "terminal":
        if not turns or turns[-1]["done"] is not True:
            raise ResultValidationError(
                "terminal row lacks a terminal environment step"
            )
    if reason == "horizon" and turns and turns[-1]["done"] is True:
        raise ResultValidationError("horizon row cannot end on a terminal step")

    failure = require_mapping("failure", result["failure"])
    require_exact_keys(
        "failure",
        failure,
        {
            "class",
            "stage",
            "timed_out",
            "protected_ref",
            "message_sha256",
        },
    )
    failure_class = failure["class"]
    if failure_class is None:
        if any(
            failure[name] is not None
            for name in ("stage", "protected_ref", "message_sha256")
        ) or failure["timed_out"] is not False:
            raise ResultValidationError("empty failure envelope is inconsistent")
    else:
        if failure_class not in FAILURE_CLASSES:
            raise ResultValidationError("unsupported failure class")
        if not isinstance(failure["stage"], str) or not failure["stage"]:
            raise ResultValidationError("failure stage must be nonempty text")
        require_evidence_sha(
            "failure", failure["protected_ref"], failure["message_sha256"]
        )
        if failure["timed_out"] is not (failure_class == "wall_timeout"):
            raise ResultValidationError("failure timeout flag mismatch")
        if not any(receipt["kind"] == "error" for receipt in receipts):
            raise ResultValidationError("failed row has no protected error receipt")
    if reason in {"failure", "timeout"} and failure_class is None:
        raise ResultValidationError("failure termination lacks a failure envelope")
    if reason == "timeout" and failure_class != "wall_timeout":
        raise ResultValidationError("timeout termination lacks wall-timeout evidence")
    if failure_class == "wall_timeout" and reason != "timeout":
        raise ResultValidationError("wall timeout has the wrong termination reason")

    artifact = result["final_artifact"]
    if artifact is not None:
        artifact = require_mapping("final_artifact", artifact)
        require_exact_keys(
            "final_artifact",
            artifact,
            {"type", "protected_ref", "sha256", "receipt_ref"},
        )
        if artifact["type"] != config.task.artifact_type:
            raise ResultValidationError("final artifact type mismatch")
        require_evidence_sha(
            "final_artifact", artifact["protected_ref"], artifact["sha256"]
        )
        require_evidence_ref("final_artifact.receipt_ref", artifact["receipt_ref"])
        if artifact["receipt_ref"] not in receipt_refs:
            raise ResultValidationError("artifact receipt is missing")
        if receipt_by_ref[artifact["receipt_ref"]]["kind"] != "artifact":
            raise ResultValidationError("artifact receipt has the wrong kind")

    scorer = result["scorer"]
    if scorer is not None:
        scorer = require_mapping("scorer", scorer)
        require_exact_keys(
            "scorer",
            scorer,
            {
                "name",
                "revision",
                "config_sha256",
                "public_metrics",
                "receipt_ref",
            },
        )
        if (
            scorer["name"] != config.grader.name
            or scorer["revision"] != config.grader.revision
            or scorer["config_sha256"] != config.grader.config_sha256
        ):
            raise ResultValidationError("scorer identity mismatch")
        validate_public_metrics(scorer["public_metrics"])
        require_evidence_ref("scorer.receipt_ref", scorer["receipt_ref"])
        if scorer["receipt_ref"] not in receipt_refs:
            raise ResultValidationError("scorer receipt is missing")
        if receipt_by_ref[scorer["receipt_ref"]]["kind"] != "scorer":
            raise ResultValidationError("scorer receipt has the wrong kind")
    if scorer is not None and artifact is None:
        raise ResultValidationError("scorer cannot exist without an artifact")

    comparable = (
        failure_class is None
        and reason in {"terminal", "horizon"}
        and artifact is not None
        and scorer is not None
    )
    if type(result["comparable"]) is not bool or result["comparable"] != comparable:
        raise ResultValidationError("comparability flag is inconsistent")
    if comparable and not declared_routes_match:
        raise ResultValidationError(
            "comparable row lifecycle routes do not match its capability"
        )
    if comparable and set(closed_roots or ()) != set(declared_roots):
        raise ResultValidationError(
            "comparable row lacks complete lifecycle cleanup"
        )
    if comparable and (
        reset_receipt_ref is None or close_receipt_ref is None
    ):
        raise ResultValidationError(
            "comparable row lacks lifecycle receipt references"
        )
    if comparable and len(close_receipts) != 1:
        raise ResultValidationError("comparable row lacks a close receipt")
    if comparable and float(wall_seconds) >= config.budgets.max_wall_seconds:
        raise ResultValidationError("comparable row exceeded its wall budget")
    if comparable and usage["total_tokens"] > config.budgets.max_total_tokens:
        raise ResultValidationError("comparable row exceeded its token budget")


def verify_pair_completeness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require exactly one matched row for each frozen treatment."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise PairVerificationError("rows must be a sequence")
    if not rows:
        raise PairVerificationError("at least one complete pair is required")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    namespaces = set()
    lifecycle_root_ids = set()
    try:
        for row in rows:
            validate_result_row(row)
            namespace_key = canonical_json_bytes(row["namespace"])
            if namespace_key in namespaces:
                raise PairVerificationError("namespace was reused across result rows")
            namespaces.add(namespace_key)
            for root in row["lifecycle"]["declared_roots"]:
                root_id = root["root_id"]
                if root_id in lifecycle_root_ids:
                    raise PairVerificationError(
                        "lifecycle root was reused across result rows"
                    )
                lifecycle_root_ids.add(root_id)
            grouped[row["pair_key"]].append(row)
    except ResultValidationError as error:
        raise PairVerificationError(str(error)) from error

    expected_treatments = {Arm.NATIVE.value, Arm.AMG_MEMORY.value}
    comparable_pairs = 0
    for pair_key, pair_rows in grouped.items():
        if len(pair_rows) != 2:
            raise PairVerificationError(
                f"pair {pair_key!r} must contain exactly two rows"
            )
        by_treatment = {row["arm"]: row for row in pair_rows}
        if set(by_treatment) != expected_treatments:
            raise PairVerificationError(
                f"pair {pair_key!r} does not contain both frozen treatments"
            )
        native = by_treatment[Arm.NATIVE.value]
        memory = by_treatment[Arm.AMG_MEMORY.value]
        native_config = dict(native["config"])
        memory_config = dict(memory["config"])
        native_config.pop("capability")
        memory_config.pop("capability")
        if native_config != memory_config:
            raise PairVerificationError(
                f"pair {pair_key!r} has treatment-excluded config drift"
            )
        if (
            native["treatment_excluded_config_sha256"]
            != memory["treatment_excluded_config_sha256"]
        ):
            raise PairVerificationError(
                f"pair {pair_key!r} has treatment-excluded digest drift"
            )
        native_prompt = native["prompt"]["treatment_excluded_sha256"]
        memory_prompt = memory["prompt"]["treatment_excluded_sha256"]
        if native_prompt is None or native_prompt != memory_prompt:
            raise PairVerificationError(
                f"pair {pair_key!r} has missing or drifted base prompt evidence"
            )
        native_namespace = dict(native["namespace"])
        memory_namespace = dict(memory["namespace"])
        native_namespace.pop("arm")
        memory_namespace.pop("arm")
        if native_namespace != memory_namespace:
            raise PairVerificationError(f"pair {pair_key!r} namespace base drifted")
        try:
            rendered = PairKey(
                run_id=native["run_id"],
                benchmark=native["benchmark"],
                protocol=native["protocol"],
                task_id=native["task_id"],
                seed=native["seed"],
            ).render()
        except (TypeError, ValueError) as error:
            raise PairVerificationError("invalid pair identity") from error
        if pair_key != rendered:
            raise PairVerificationError("pair key includes drift or treatment state")
        comparable_pairs += int(native["comparable"] and memory["comparable"])

    failure_counts = Counter(
        row["failure"]["class"] or "none" for row in rows
    )
    return {
        "schema": "amg.paired_eval.pair_verification",
        "schema_version": RESULT_SCHEMA_VERSION,
        "row_count": len(rows),
        "pair_count": len(grouped),
        "comparable_pair_count": comparable_pairs,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def build_public_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Project only explicit public scalars; never copy protected references."""

    verify_pair_completeness(rows)
    public_rows = []
    for row in rows:
        validate_result_row(row)
        metrics = {} if row["scorer"] is None else row["scorer"]["public_metrics"]
        public_rows.append(
            {
                "run_id": row["run_id"],
                "benchmark": row["benchmark"],
                "protocol": row["protocol"],
                "task_id": row["task_id"],
                "seed": row["seed"],
                "arm": row["arm"],
                "termination_reason": row["termination"]["reason"],
                "failure_class": row["failure"]["class"],
                "comparable": row["comparable"],
                "public_metrics": dict(metrics),
            }
        )
    return {
        "schema": "amg.paired_eval.public_summary",
        "schema_version": RESULT_SCHEMA_VERSION,
        "row_count": len(public_rows),
        "rows": public_rows,
    }
