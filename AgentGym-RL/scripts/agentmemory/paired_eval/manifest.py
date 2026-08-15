"""Manifest expansion and generic execution through the sole paired runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    Arm,
    BudgetConfig,
    CompactionConfig,
    DecodingConfig,
    EnvironmentAdapterProtocol,
    GraderConfig,
    MANIFEST_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    ModelClientProtocol,
    ModelConfig,
    RunConfig,
    RuntimeConfig,
    SourceConfig,
    TaskConfig,
    capability_for_arm,
    require_text,
)
from .evidence import AppendSafeJsonlWriter
from .runner import PairedRunner
from .verifier import validate_result_row, verify_pair_completeness


COMMON_KEYS = frozenset(
    {
        "model",
        "decoding",
        "budgets",
        "compaction",
        "source",
        "runtime",
        "grader",
    }
)
TASK_KEYS = frozenset(
    {
        "benchmark",
        "protocol",
        "task_id",
        "task_index",
        "seed",
        "native_tools",
        "artifact_type",
    }
)
MANIFEST_KEYS = frozenset(
    {"schema", "schema_version", "run_id", "arms", "common", "tasks"}
)


@dataclass(frozen=True)
class RuntimeBindings:
    adapter: EnvironmentAdapterProtocol
    model: ModelClientProtocol


def exact_keys(name: str, value: Mapping[str, Any], expected: Sequence[str]) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(wanted - actual)}, "
            f"extra={sorted(actual - wanted)}"
        )


def require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def parse_exact_config(name: str, payload: Any, constructor: Callable) -> Any:
    config_payload = require_mapping(name, payload)
    value = constructor(config_payload)
    if value.to_payload() != config_payload:
        raise ValueError(f"{name} has unknown or noncanonical fields")
    return value


def expand_manifest(payload: Mapping[str, Any]) -> list[RunConfig]:
    """Expand task × treatment declarations without benchmark dispatch."""

    manifest = require_mapping("manifest", payload)
    exact_keys("manifest", manifest, MANIFEST_KEYS)
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ValueError("unsupported paired-evaluation manifest schema")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported paired-evaluation manifest version")
    run_id = require_text("manifest.run_id", manifest["run_id"])

    treatment_values = manifest["arms"]
    if not isinstance(treatment_values, list):
        raise TypeError("manifest arms must be a list")
    if len(treatment_values) != 3:
        raise ValueError("manifest must declare exactly three arms")
    try:
        treatments = tuple(Arm(value) for value in treatment_values)
    except ValueError as error:
        raise ValueError("manifest declares an unsupported treatment") from error
    expected_arms = {
        Arm.NATIVE,
        Arm.AMG_COMPACTION_ONLY,
        Arm.AMG_MEMORY,
    }
    if set(treatments) != expected_arms:
        raise ValueError(
            "manifest must declare native, amg_compaction_only, and "
            "amg_memory exactly once"
        )

    common = require_mapping("manifest.common", manifest["common"])
    exact_keys("manifest.common", common, COMMON_KEYS)
    model = parse_exact_config(
        "common.model", common["model"], ModelConfig.from_payload
    )
    decoding = parse_exact_config(
        "common.decoding", common["decoding"], DecodingConfig.from_payload
    )
    budgets = parse_exact_config(
        "common.budgets", common["budgets"], BudgetConfig.from_payload
    )
    compaction = parse_exact_config(
        "common.compaction", common["compaction"], CompactionConfig.from_payload
    )
    source = parse_exact_config(
        "common.source", common["source"], SourceConfig.from_payload
    )
    runtime = parse_exact_config(
        "common.runtime", common["runtime"], RuntimeConfig.from_payload
    )
    grader = parse_exact_config(
        "common.grader", common["grader"], GraderConfig.from_payload
    )

    task_values = manifest["tasks"]
    if not isinstance(task_values, list) or not task_values:
        raise ValueError("manifest tasks must be a nonempty list")
    configs = []
    pair_keys = set()
    for index, task_value in enumerate(task_values):
        task_payload = require_mapping(f"manifest.tasks[{index}]", task_value)
        exact_keys(f"manifest.tasks[{index}]", task_payload, TASK_KEYS)
        task = parse_exact_config(
            f"manifest.tasks[{index}]", task_payload, TaskConfig.from_payload
        )
        for treatment in treatments:
            config = RunConfig(
                run_id=run_id,
                task=task,
                model=model,
                decoding=decoding,
                budgets=budgets,
                compaction=compaction,
                source=source,
                runtime=runtime,
                grader=grader,
                capability=capability_for_arm(treatment),
            )
            configs.append(config)
        if configs[-1].pair_key in pair_keys:
            raise ValueError("manifest contains a duplicate pair identity")
        pair_keys.add(configs[-1].pair_key)
    return configs


def execute_manifest(
    payload: Mapping[str, Any],
    *,
    runner: PairedRunner,
    runtime_factory: Callable[[RunConfig], RuntimeBindings],
    writer: AppendSafeJsonlWriter,
) -> list[dict[str, Any]]:
    """Execute every case through exactly ``PairedRunner.run_task``."""

    configs = expand_manifest(payload)
    if writer.path.exists() and writer.path.stat().st_size:
        raise RuntimeError("manifest execution requires an empty result file")
    rows = []
    for config in configs:
        bindings = runtime_factory(config)
        if not isinstance(bindings, RuntimeBindings):
            raise TypeError("runtime factory must return RuntimeBindings")
        row = runner.run_task(config, bindings.adapter, bindings.model)
        validate_result_row(row)
        rows.append(row)
    verify_pair_completeness(rows)
    writer.append_many(rows, require_empty=True)
    return rows
