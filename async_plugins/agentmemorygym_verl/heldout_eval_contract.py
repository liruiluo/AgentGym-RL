"""Pure-data contract for resumable CAMG native held-out evaluation.

This module deliberately has no torch, Ray, veRL, or environment imports.  It
owns the immutable schedule identity, padding identity, action-row validation,
native per-environment success metrics, and atomic batch receipts.  The GPU
runner in :mod:`agentmemorygym_verl.heldout_eval` only wires these primitives to
the existing task-neutral AgentLoop.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agemem_evidence import summarize_agemem_step_records
from .heldout_method_evidence import summarize_method_step_records

CANONICAL_ROUTES = (
    "webshop",
    "swesmith",
    "literesearcher",
    "openmle_fast",
)
AGENT_NAME = "amg_task_neutral_async"
SPEC_SCHEMA = "camg_heldout_schedule_spec_v2"
SCHEDULE_SCHEMA = "camg_heldout_schedule_manifest_v2"
RUN_SCHEMA = "camg_heldout_eval_run_contract_v1"
BATCH_SCHEMA = "camg_heldout_eval_batch_receipt_v1"
EPISODE_SCHEMA = "camg_heldout_eval_episode_v1"
METRICS_SCHEMA = "camg_heldout_eval_metrics_v1"
ACTION_ROW_SCHEMA = "amg_task_neutral_action_row_v1"
NATIVE_SOURCE_IDENTITY_SCHEMA = "camg_native_episode_source_identity_v1"
COMPLETE_SPLIT_SCHEMA = "camg_formal_eval_split_contract_v5"
COMPLETE_SPLIT_AGGREGATE = (
    "unweighted macro-average of the four environment-level primary success metrics"
)
SWESMITH_FORMAL_EVAL_RUNTIME_SCHEMA = (
    "camg_swesmith_formal_eval_runtime_manifest_v5"
)
SWESMITH_FULL_ADMISSION_SUMMARY_SCHEMA = (
    "camg_swesmith_complete_admission_summary_v2"
)
SWESMITH_FORMAL_EVAL_SELECTION = (
    "deterministic complete-repository subset of the exact-runtime-admitted "
    "held-out candidate pool"
)
SWESMITH_FORMAL_SELECTION_SCHEMA = "camg_swesmith_formal_eval_selection_v5"
SWESMITH_ADMITTED_POOL_SCHEMA = "camg_swesmith_admitted_heldout_pool_manifest_v5"
SWESMITH_EXTENSION_POOL_SCHEMA = "camg_swesmith_extension_pool_manifest_v5"
SWESMITH_HELDOUT_REPOSITORY_COUNT = 36
SWESMITH_RAW_HELDOUT_TASK_COUNT = 10_381
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BATCH_DIRECTORY = re.compile(r"^batch-(\d{6})$")


@dataclass(frozen=True)
class RouteSchedule:
    """One verified route-local held-out schedule."""

    route_id: str
    path: Path
    sha256: str
    expected_rows: int
    rows: tuple[dict[str, Any], ...]
    selection_authority: dict[str, Any] | None = None


def _resolve_file_reference(base: Path, value: Any, *, field: str) -> Path:
    reference = value
    if not isinstance(reference, Mapping):
        raise TypeError(f"{field} must be an object")
    path = _resolve_relative(base, reference.get("path"), field=f"{field} path")
    regular = require_regular_file(path, field=field).resolve()
    expected_digest = require_sha256(
        reference.get("sha256"), field=f"{field} sha256"
    )
    if sha256_file(regular) != expected_digest:
        raise ValueError(f"{field} sha256 mismatch")
    expected_bytes = require_positive_int(
        reference.get("bytes"), field=f"{field} bytes"
    )
    if regular.stat().st_size != expected_bytes:
        raise ValueError(f"{field} byte count mismatch")
    return regular


def verify_complete_split_contract(
    contract_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_route_counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the evaluator to the completed four-environment denominator."""

    path = require_regular_file(
        contract_path, field="complete held-out split contract"
    ).resolve()
    expected_digest = require_sha256(
        expected_sha256, field="complete held-out split contract sha256"
    )
    observed_digest = sha256_file(path)
    if observed_digest != expected_digest:
        raise ValueError(
            "complete held-out split contract sha256 mismatch: "
            f"expected {expected_digest}, got {observed_digest}"
        )
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError("complete held-out split contract must be an object")
    invariants = payload.get("global_invariants")
    environments = payload.get("environments")
    if (
        payload.get("schema") != COMPLETE_SPLIT_SCHEMA
        or payload.get("status") != "ready"
        or not isinstance(invariants, Mapping)
        or invariants.get("active_training_inputs_modified") is not False
        or invariants.get("heldout_evaluation_run") is not False
        or invariants.get("aggregate_metric") != COMPLETE_SPLIT_AGGREGATE
        or not isinstance(environments, Mapping)
    ):
        raise ValueError("complete held-out split contract is not evaluation-ready")
    environment_fields = {
        "webshop": ("shop", "formal_heldout_episode_count"),
        "swesmith": ("coding", "formal_eval_tasks"),
        "literesearcher": ("deep_research", "formal_heldout_task_count"),
        "openmle_fast": ("auto_research", "formal_heldout_task_count"),
    }
    route_counts: dict[str, int] = {}
    for route_id in CANONICAL_ROUTES:
        environment_name, count_field = environment_fields[route_id]
        environment = environments.get(environment_name)
        if (
            not isinstance(environment, Mapping)
            or environment.get("preparation_status") != "ready"
        ):
            raise ValueError(
                f"complete held-out split route {route_id!r} is not ready"
            )
        route_counts[route_id] = require_positive_int(
            environment.get(count_field),
            field=f"complete held-out split route {route_id!r} count",
        )
    if expected_route_counts is not None:
        normalized_expected = {
            route_id: require_positive_int(
                expected_route_counts.get(route_id),
                field=f"expected complete held-out route {route_id!r} count",
            )
            for route_id in CANONICAL_ROUTES
        }
        if set(expected_route_counts) != set(CANONICAL_ROUTES):
            raise ValueError("expected complete held-out route set drift")
        if normalized_expected != route_counts:
            raise ValueError(
                "held-out schedule counts differ from the complete split contract"
            )
    result = {
        "schema": COMPLETE_SPLIT_SCHEMA,
        "path": str(path),
        "sha256": observed_digest,
        "package_id": str(payload.get("package_id", "")),
        "route_counts": route_counts,
    }
    return result


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashes and immutable receipts."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool")
    try:
        normalized = int(value)
        exact = float(value) == float(normalized)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if not exact or normalized < 0:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return normalized


def require_positive_int(value: Any, *, field: str) -> int:
    normalized = require_nonnegative_int(value, field=field)
    if normalized == 0:
        raise ValueError(f"{field} must be positive")
    return normalized


def normalize_route_max_rounds(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("route_max_rounds must be an object")
    if set(value) != set(CANONICAL_ROUTES):
        raise ValueError("route_max_rounds must cover exactly the canonical routes")
    result = {
        route_id: require_positive_int(
            value[route_id], field=f"route {route_id!r} max_rounds"
        )
        for route_id in CANONICAL_ROUTES
    }
    return result


def require_regular_file(path: str | os.PathLike[str], *, field: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{field} must be a regular non-symlink file: {candidate}")
    return candidate


def read_json(path: str | os.PathLike[str]) -> Any:
    regular = require_regular_file(path, field="JSON input")
    try:
        return json.loads(regular.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {regular}") from exc


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    regular = require_regular_file(path, field="JSONL input")
    rows: list[dict[str, Any]] = []
    with regular.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"blank line in {regular} at line {line_number}")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {regular} at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"JSONL row in {regular} at line {line_number} is not an object"
                )
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"JSONL input is empty: {regular}")
    return rows


def verify_swesmith_formal_eval_authority(
    manifest_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_routing_sha256: str,
    expected_admitted_task_count: int,
    selected_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify the full Coding Eval authority and an optional final-panel subset.

    ``runtime-manifest.json`` authorizes the complete 933-task formal Eval
    pool.  Paper tables may use a pre-registered subset of that pool.  In that
    case, ``selected_rows`` binds every scheduled row back to the full formal
    routing without pretending that the 128-row panel is a newly repartitioned
    formal Eval pool.
    """

    path = require_regular_file(
        manifest_path, field="SWE-smith formal Eval runtime manifest"
    ).resolve()
    expected_digest = require_sha256(
        expected_sha256, field="SWE-smith formal Eval runtime manifest sha256"
    )
    observed_digest = sha256_file(path)
    if observed_digest != expected_digest:
        raise ValueError(
            "SWE-smith formal Eval runtime manifest sha256 mismatch: "
            f"expected {expected_digest}, got {observed_digest}"
        )
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError("SWE-smith formal Eval runtime manifest must be an object")
    if (
        payload.get("schema") != SWESMITH_FORMAL_EVAL_RUNTIME_SCHEMA
        or payload.get("status") != "ready"
        or payload.get("selection") != SWESMITH_FORMAL_EVAL_SELECTION
        or payload.get("heldout_evaluation_run") is not False
        or payload.get("active_training_inputs_modified") is not False
        or payload.get("count_cap") is not None
        or payload.get("top_k") is not None
        or payload.get("early_stop") is not False
    ):
        raise ValueError("SWE-smith authority is not a frozen formal Eval subset")
    repositories = require_positive_int(
        payload.get("raw_assigned_repository_count"),
        field="SWE-smith formal Eval raw_assigned_repository_count",
    )
    if repositories != SWESMITH_HELDOUT_REPOSITORY_COUNT:
        raise ValueError(
            "SWE-smith full admission must cover all 36 frozen held-out repositories"
        )
    raw_assigned = require_positive_int(
        payload.get("raw_assigned_task_count"),
        field="SWE-smith formal Eval raw_assigned_task_count",
    )
    if raw_assigned != SWESMITH_RAW_HELDOUT_TASK_COUNT:
        raise ValueError(
            "SWE-smith full admission must classify all 10,381 raw held-out tasks"
        )
    formal = require_positive_int(
        payload.get("task_count"),
        field="SWE-smith formal Eval task_count",
    )
    formal_repositories = require_positive_int(
        payload.get("repository_count"),
        field="SWE-smith formal Eval repository_count",
    )
    admitted = require_positive_int(
        payload.get("complete_admitted_pool_task_count"),
        field="SWE-smith complete admitted-pool task_count",
    )
    admitted_repositories = require_positive_int(
        payload.get("complete_admitted_pool_repository_count"),
        field="SWE-smith complete admitted-pool repository_count",
    )
    extension = require_positive_int(
        payload.get("extension_pool_task_count"),
        field="SWE-smith extension-pool task_count",
    )
    extension_repositories = require_positive_int(
        payload.get("extension_pool_repository_count"),
        field="SWE-smith extension-pool repository_count",
    )
    rejected = require_nonnegative_int(
        payload.get("semantic_exclusion_count"),
        field="SWE-smith admission rejected_task_count",
    )
    if admitted + rejected != raw_assigned:
        raise ValueError(
            "SWE-smith full admission does not account for the complete raw task pool"
        )
    expected_count = require_positive_int(
        expected_admitted_task_count,
        field="expected SWE-smith scheduled task count",
    )
    is_subset = selected_rows is not None and expected_count < formal
    if not is_subset and formal != expected_count:
        raise ValueError(
            "SWE-smith formal Eval task count differs from the held-out schedule"
        )
    if selected_rows is not None and len(selected_rows) != expected_count:
        raise ValueError("SWE-smith selected-row count differs from the schedule")
    if expected_count > formal:
        raise ValueError("SWE-smith held-out schedule exceeds the formal Eval pool")
    if formal + extension != admitted:
        raise ValueError("SWE-smith formal and extension pools do not partition admission")
    if formal_repositories + extension_repositories != admitted_repositories:
        raise ValueError("SWE-smith formal and extension repositories do not partition admission")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("SWE-smith formal Eval runtime files are missing")
    routing_path = _resolve_file_reference(
        path.parent,
        files.get("routing"),
        field="SWE-smith formal Eval routing",
    )
    routing_digest = sha256_file(routing_path)
    expected_routing_digest = require_sha256(
        expected_routing_sha256,
        field="expected SWE-smith scheduled routing sha256",
    )
    if not is_subset and routing_digest != expected_routing_digest:
        raise ValueError(
            "SWE-smith full admission does not bind the held-out routing schedule"
        )
    routing_rows = read_jsonl(routing_path)
    if len(routing_rows) != formal:
        raise ValueError("SWE-smith formal Eval routing count drifted")
    if [row.get("data_idx") for row in routing_rows] != list(range(formal)):
        raise ValueError("SWE-smith formal Eval routing data_idx must be dense")
    if [
        (row.get("extra_info") or {}).get("index") for row in routing_rows
    ] != list(range(formal)):
        raise ValueError("SWE-smith formal Eval routing extra_info.index must be dense")

    selected_identity_sha256: str | None = None
    if is_subset:
        def identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
            extra = row.get("extra_info")
            if not isinstance(extra, Mapping):
                raise ValueError("SWE-smith routing row extra_info is missing")
            item_id = str(row.get("item_id", "")).strip()
            instance_id = str(extra.get("instance_id", "")).strip()
            repository = str(extra.get("base_repository", "")).strip()
            if not item_id or not instance_id or not repository:
                raise ValueError("SWE-smith routing identity is incomplete")
            return (
                item_id,
                require_nonnegative_int(
                    row.get("data_idx"), field="SWE-smith routing data_idx"
                ),
                instance_id,
                repository,
            )

        formal_identities = [identity(row) for row in routing_rows]
        if len(set(formal_identities)) != len(formal_identities):
            raise ValueError("SWE-smith formal Eval routing identity is not unique")
        selected_identities = [identity(row) for row in selected_rows]
        if len(set(selected_identities)) != len(selected_identities):
            raise ValueError("SWE-smith final-panel routing identity is not unique")
        missing = sorted(set(selected_identities) - set(formal_identities))
        if missing:
            raise ValueError(
                "SWE-smith final-panel routing is not a subset of formal Eval"
            )
        selected_identity_sha256 = sha256_bytes(
            canonical_json_bytes(sorted(selected_identities))
        )

    selection_path = _resolve_file_reference(
        path.parent,
        files.get("formal_eval_selection"),
        field="SWE-smith formal Eval selection",
    )
    selection = read_json(selection_path)
    if not isinstance(selection, Mapping):
        raise TypeError("SWE-smith formal Eval selection must be an object")
    active_train = require_positive_int(
        selection.get("active_train_task_count"),
        field="SWE-smith active train task count",
    )
    fraction = formal / (active_train + formal)
    if (
        selection.get("schema") != SWESMITH_FORMAL_SELECTION_SCHEMA
        or selection.get("status") != "frozen"
        or selection.get("formal_eval_task_count") != formal
        or selection.get("formal_eval_repository_count") != formal_repositories
        or selection.get("complete_admitted_heldout_pool_task_count") != admitted
        or selection.get("complete_admitted_heldout_pool_repository_count")
        != admitted_repositories
        or selection.get("extension_pool_task_count") != extension
        or selection.get("extension_pool_repository_count")
        != extension_repositories
        or selection.get("extension_pool_formal_evaluation_role") is not False
        or selection.get("extension_pool_training_role") is not False
        or selection.get("selection_depends_on_model_output_or_reward") is not False
        or selection.get("active_training_inputs_modified") is not False
        or selection.get("heldout_evaluation_run") is not False
        or not (0.18 <= fraction <= 0.22)
        or abs(float(selection.get("formal_eval_fraction", -1.0)) - fraction)
        > 1e-15
    ):
        raise ValueError("SWE-smith formal Eval selection contract drifted")

    pool_path = _resolve_file_reference(
        path.parent,
        files.get("admitted_pool_manifest"),
        field="SWE-smith admitted-pool manifest",
    )
    pool = read_json(pool_path)
    if (
        not isinstance(pool, Mapping)
        or pool.get("schema") != SWESMITH_ADMITTED_POOL_SCHEMA
        or pool.get("status") != "complete"
        or pool.get("role") != "admitted_heldout_candidate_pool"
        or pool.get("task_count") != admitted
        or pool.get("repository_count") != admitted_repositories
        or pool.get("formal_evaluation_role") is not False
        or pool.get("training_role") is not False
    ):
        raise ValueError("SWE-smith admitted-pool manifest drifted")

    extension_path = _resolve_file_reference(
        path.parent,
        files.get("extension_pool_manifest"),
        field="SWE-smith extension-pool manifest",
    )
    extension_payload = read_json(extension_path)
    if (
        not isinstance(extension_payload, Mapping)
        or extension_payload.get("schema") != SWESMITH_EXTENSION_POOL_SCHEMA
        or extension_payload.get("status") != "frozen"
        or extension_payload.get("role") != "admitted_heldout_extension_pool"
        or extension_payload.get("task_count") != extension
        or extension_payload.get("repository_count") != extension_repositories
        or extension_payload.get("formal_evaluation_role") is not False
        or extension_payload.get("training_role") is not False
    ):
        raise ValueError("SWE-smith extension-pool manifest drifted")

    heldout_manifest_path = _resolve_file_reference(
        path.parent,
        files.get("manifest"),
        field="SWE-smith formal Eval dataset manifest",
    )
    heldout_manifest = read_json(heldout_manifest_path)
    heldout_selection = (
        heldout_manifest.get("selection")
        if isinstance(heldout_manifest, Mapping)
        else None
    )
    if (
        not isinstance(heldout_selection, Mapping)
        or heldout_manifest.get("role") != "formal_heldout"
        or heldout_selection.get("mode") != "instance_ids"
        or heldout_selection.get("count") != formal
        or heldout_selection.get("repository_count") != formal_repositories
        or heldout_selection.get("source_admitted_pool_count") != admitted
        or heldout_selection.get("source_admitted_pool_repository_count")
        != admitted_repositories
        or heldout_selection.get("active_training_inputs_modified") is not False
        or heldout_selection.get("count_cap") is not None
        or heldout_selection.get("top_k") is not None
        or heldout_selection.get("early_stop") is not False
    ):
        raise ValueError("SWE-smith formal Eval dataset manifest drifted")

    summary_path = _resolve_file_reference(
        path.parent,
        files.get("admission_summary"),
        field="SWE-smith full admission summary",
    )
    summary = read_json(summary_path)
    if not isinstance(summary, Mapping):
        raise TypeError("SWE-smith full admission summary must be an object")
    if (
        summary.get("schema") != SWESMITH_FULL_ADMISSION_SUMMARY_SCHEMA
        or summary.get("status") != "pass"
        or summary.get("raw_assigned_task_count") != raw_assigned
        or summary.get("classified_task_count") != raw_assigned
        or summary.get("admitted_task_count") != admitted
        or summary.get("excluded_task_count") != rejected
        or summary.get("pending_infrastructure_task_count") != 0
        or summary.get("raw_heldout_repository_count") != repositories
        or summary.get("count_cap") is not None
        or summary.get("top_k") is not None
        or summary.get("early_stop") is not False
        or summary.get("heldout_evaluation_run") is not False
    ):
        raise ValueError(
            "SWE-smith admission summary does not account for the full held-out pool"
        )
    return {
        "schema": SWESMITH_FORMAL_EVAL_RUNTIME_SCHEMA,
        "path": str(path),
        "sha256": observed_digest,
        "selection": SWESMITH_FORMAL_EVAL_SELECTION,
        "raw_assigned_repository_count": repositories,
        "raw_assigned_task_count": raw_assigned,
        "classified_task_count": raw_assigned,
        "formal_eval_task_count": formal,
        "formal_eval_repository_count": formal_repositories,
        "complete_admitted_pool_task_count": admitted,
        "complete_admitted_pool_repository_count": admitted_repositories,
        "extension_pool_task_count": extension,
        "extension_pool_repository_count": extension_repositories,
        "rejected_task_count": rejected,
        "formal_eval_routing_sha256": routing_digest,
        "scheduled_task_count": expected_count,
        "scheduled_routing_sha256": expected_routing_digest,
        "scheduled_identity_sha256": selected_identity_sha256,
        "scheduled_subset_of_formal_eval": is_subset,
        "formal_eval_selection_path": str(selection_path),
        "formal_eval_selection_sha256": sha256_file(selection_path),
        "admission_summary_path": str(summary_path),
        "admission_summary_sha256": sha256_file(summary_path),
    }


def _read_jsonl_allow_empty(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    regular = require_regular_file(path, field="JSONL input")
    if regular.stat().st_size == 0:
        return []
    return read_jsonl(regular)


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload))


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _resolve_relative(base: Path, value: Any, *, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is missing")
    path = Path(text)
    return path if path.is_absolute() else base / path


def _validate_source_row(
    row: Mapping[str, Any],
    *,
    route_id: str,
    source_position: int,
) -> dict[str, Any]:
    normalized = deepcopy(dict(row))
    item_id = str(normalized.get("item_id", "")).strip()
    if not item_id:
        raise ValueError(
            f"route {route_id!r} row {source_position} has no item_id"
        )
    data_idx = require_nonnegative_int(
        normalized.get("data_idx"),
        field=f"route {route_id!r} row {source_position} data_idx",
    )
    extra_raw = normalized.get("extra_info")
    if extra_raw is None:
        extra: dict[str, Any] = {}
    elif isinstance(extra_raw, Mapping):
        extra = deepcopy(dict(extra_raw))
    else:
        raise TypeError(
            f"route {route_id!r} row {source_position} extra_info must be an object"
        )

    for field, value in (
        ("top-level index", normalized.get("index")),
        ("extra_info.index", extra.get("index")),
    ):
        if value is not None and require_nonnegative_int(
            value,
            field=f"route {route_id!r} row {source_position} {field}",
        ) != data_idx:
            raise ValueError(
                f"route {route_id!r} row {source_position} index/data_idx drift"
            )

    for field, value in (
        ("route_id", normalized.get("route_id")),
        ("data_source", normalized.get("data_source")),
        ("extra_info.route_id", extra.get("route_id")),
    ):
        if value is not None and str(value) != route_id:
            raise ValueError(
                f"route {route_id!r} row {source_position} {field} drift"
            )
    configured_agent = normalized.get("agent_name")
    if configured_agent is not None and str(configured_agent) != AGENT_NAME:
        raise ValueError(
            f"route {route_id!r} row {source_position} selects another agent loop"
        )
    if "uid" in normalized or "eval_padding" in normalized:
        raise ValueError(
            f"route {route_id!r} row {source_position} contains reserved eval identity fields"
        )

    normalized["item_id"] = item_id
    normalized["data_idx"] = data_idx
    normalized["extra_info"] = extra
    return normalized


def _load_spec(
    spec_path: str | os.PathLike[str],
    *,
    expected_spec_sha256: str,
) -> tuple[dict[str, Any], tuple[RouteSchedule, ...], str]:
    path = require_regular_file(spec_path, field="held-out schedule spec")
    expected_digest = require_sha256(
        expected_spec_sha256, field="held-out schedule spec expected sha256"
    )
    observed_digest = sha256_file(path)
    if observed_digest != expected_digest:
        raise ValueError(
            "held-out schedule spec sha256 mismatch: "
            f"expected {expected_digest}, got {observed_digest}"
        )
    payload = read_json(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"held-out schedule spec schema must be {SPEC_SCHEMA!r}")
    if payload.get("agent_name") != AGENT_NAME:
        raise ValueError(f"held-out schedule agent_name must be {AGENT_NAME!r}")
    panel_id = str(payload.get("panel_id", "")).strip()
    if not panel_id:
        raise ValueError("held-out schedule panel_id must not be empty")
    route_registry_sha256 = require_sha256(
        payload.get("route_registry_sha256"),
        field="held-out schedule route_registry_sha256",
    )
    complete_split_path = _resolve_relative(
        path.parent,
        payload.get("complete_split_contract"),
        field="complete held-out split contract",
    )
    complete_split_authority = verify_complete_split_contract(
        complete_split_path,
        expected_sha256=payload.get("complete_split_contract_sha256"),
    )
    raw_routes = payload.get("routes")
    if isinstance(raw_routes, (str, bytes)) or not isinstance(raw_routes, Sequence):
        raise TypeError("held-out schedule routes must be a sequence")
    route_ids = tuple(
        str(route.get("route_id", "")) if isinstance(route, Mapping) else ""
        for route in raw_routes
    )
    if route_ids != CANONICAL_ROUTES:
        raise ValueError(
            "held-out schedule routes must use canonical order: "
            f"{route_ids!r} != {CANONICAL_ROUTES!r}"
        )

    sources: list[RouteSchedule] = []
    for raw_route in raw_routes:
        assert isinstance(raw_route, Mapping)
        route_id = str(raw_route["route_id"])
        schedule = _resolve_relative(
            path.parent,
            raw_route.get("schedule"),
            field=f"route {route_id!r} schedule",
        )
        regular = require_regular_file(
            schedule, field=f"route {route_id!r} schedule"
        )
        expected_schedule_digest = require_sha256(
            raw_route.get("schedule_sha256"),
            field=f"route {route_id!r} schedule_sha256",
        )
        observed_schedule_digest = sha256_file(regular)
        if observed_schedule_digest != expected_schedule_digest:
            raise ValueError(
                f"route {route_id!r} schedule sha256 mismatch: "
                f"expected {expected_schedule_digest}, got {observed_schedule_digest}"
            )
        expected_rows = require_positive_int(
            raw_route.get("expected_rows"),
            field=f"route {route_id!r} expected_rows",
        )
        source_rows = tuple(
            _validate_source_row(row, route_id=route_id, source_position=position)
            for position, row in enumerate(read_jsonl(regular))
        )
        if len(source_rows) != expected_rows:
            raise ValueError(
                f"route {route_id!r} row count mismatch: "
                f"expected {expected_rows}, got {len(source_rows)}"
            )
        item_ids = [str(row["item_id"]) for row in source_rows]
        data_indices = [int(row["data_idx"]) for row in source_rows]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError(f"route {route_id!r} has duplicate item_id values")
        if len(set(data_indices)) != len(data_indices):
            raise ValueError(f"route {route_id!r} has duplicate data_idx values")
        selection_authority: dict[str, Any] | None = None
        if route_id == "swesmith":
            authority_path = _resolve_relative(
                path.parent,
                raw_route.get("formal_eval_runtime_manifest"),
                field="SWE-smith formal Eval runtime manifest",
            )
            selection_authority = verify_swesmith_formal_eval_authority(
                authority_path,
                expected_sha256=raw_route.get(
                    "formal_eval_runtime_manifest_sha256"
                ),
                expected_routing_sha256=observed_schedule_digest,
                expected_admitted_task_count=expected_rows,
                selected_rows=source_rows,
            )
        elif any(
            raw_route.get(field) is not None
            for field in (
                "formal_eval_runtime_manifest",
                "formal_eval_runtime_manifest_sha256",
            )
        ):
            raise ValueError(
                f"route {route_id!r} must not declare a SWE-smith formal Eval manifest"
            )
        sources.append(
            RouteSchedule(
                route_id=route_id,
                path=regular.resolve(),
                sha256=observed_schedule_digest,
                expected_rows=expected_rows,
                rows=source_rows,
                selection_authority=selection_authority,
            )
        )

    observed_route_counts = {
        source.route_id: source.expected_rows for source in sources
    }
    if observed_route_counts != complete_split_authority["route_counts"]:
        raise ValueError(
            "held-out schedule counts differ from the complete split contract"
        )

    normalized_spec = dict(payload)
    normalized_spec["panel_id"] = panel_id
    normalized_spec["route_registry_sha256"] = route_registry_sha256
    normalized_spec["complete_split_authority"] = complete_split_authority
    return normalized_spec, tuple(sources), observed_digest


def _eval_uid(
    *, route_id: str, source_item_id: str, data_idx: int, global_index: int
) -> str:
    identity = {
        "data_idx": data_idx,
        "global_index": global_index,
        "route_id": route_id,
        "source_item_id": source_item_id,
    }
    return f"camg-heldout-v1-{sha256_bytes(canonical_json_bytes(identity))}"


def compose_heldout_schedule(
    spec_path: str | os.PathLike[str],
    *,
    expected_spec_sha256: str,
    output_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build one deterministic, exhaustion-aware, route-interleaved schedule."""

    spec, sources, spec_sha256 = _load_spec(
        spec_path,
        expected_spec_sha256=expected_spec_sha256,
    )
    rows: list[dict[str, Any]] = []
    seen_item_ids: set[str] = set()
    seen_uids: set[str] = set()
    max_rows = max(source.expected_rows for source in sources)
    for route_position in range(max_rows):
        for source in sources:
            if route_position >= source.expected_rows:
                continue
            source_row = deepcopy(source.rows[route_position])
            source_item_id = str(source_row["item_id"])
            data_idx = int(source_row["data_idx"])
            global_index = len(rows)
            global_item_id = (
                f"{source.route_id}:{source_item_id}:heldout-{route_position:06d}"
            )
            uid = _eval_uid(
                route_id=source.route_id,
                source_item_id=source_item_id,
                data_idx=data_idx,
                global_index=global_index,
            )
            if global_item_id in seen_item_ids or uid in seen_uids:
                raise ValueError("held-out schedule global identity collision")
            seen_item_ids.add(global_item_id)
            seen_uids.add(uid)

            extra = dict(source_row.get("extra_info") or {})
            source_index = extra.get("index", source_row.get("index", data_idx))
            source_extra = deepcopy(extra)
            extra.update(
                {
                    "index": global_index,
                    "schedule_position": global_index,
                    "role": "heldout",
                    "route_id": source.route_id,
                    "route_registry_sha256": spec["route_registry_sha256"],
                    "source_schedule_sha256": source.sha256,
                    "source_schedule_position": route_position,
                    "source_index": source_index,
                    "source_item_id": source_item_id,
                    "source_extra_info": source_extra,
                    "panel_id": spec["panel_id"],
                }
            )
            source_row.update(
                {
                    "agent_name": AGENT_NAME,
                    "data_source": source.route_id,
                    "eval_padding": False,
                    "index": global_index,
                    "item_id": global_item_id,
                    "route_id": source.route_id,
                    "uid": uid,
                    "extra_info": extra,
                }
            )
            rows.append(source_row)

    schedule_payload = jsonl_bytes(rows)
    atomic_write_bytes(output_path, schedule_payload)
    manifest = {
        "schema": SCHEDULE_SCHEMA,
        "agent_name": AGENT_NAME,
        "panel_id": spec["panel_id"],
        "spec_sha256": spec_sha256,
        "route_registry_sha256": spec["route_registry_sha256"],
        "complete_split_authority": spec["complete_split_authority"],
        "route_order": list(CANONICAL_ROUTES),
        "row_count": len(rows),
        "schedule_sha256": sha256_bytes(schedule_payload),
        "uid_set_sha256": sha256_bytes(
            canonical_json_bytes(sorted(str(row["uid"]) for row in rows))
        ),
        "per_route_rows": {
            source.route_id: source.expected_rows for source in sources
        },
        "sources": {
            source.route_id: {
                "path": str(source.path),
                "schedule_sha256": source.sha256,
                "row_count": source.expected_rows,
                **(
                    {"selection_authority": source.selection_authority}
                    if source.selection_authority is not None
                    else {}
                ),
            }
            for source in sources
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def inspect_heldout_schedule(
    schedule_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    path = require_regular_file(schedule_path, field="held-out schedule")
    expected_digest = require_sha256(
        expected_sha256, field="held-out schedule expected sha256"
    )
    observed_digest = sha256_file(path)
    if observed_digest != expected_digest:
        raise ValueError(
            f"held-out schedule sha256 mismatch: expected {expected_digest}, "
            f"got {observed_digest}"
        )
    rows = read_jsonl(path)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"held-out schedule row count mismatch: expected {expected_count}, "
            f"got {len(rows)}"
        )
    seen_uids: set[str] = set()
    seen_item_ids: set[str] = set()
    per_route_data_indices: dict[str, set[int]] = defaultdict(set)
    for position, row in enumerate(rows):
        if row.get("index") != position:
            raise ValueError(f"held-out schedule index drift at row {position}")
        extra = row.get("extra_info")
        if not isinstance(extra, Mapping) or extra.get("index") != position:
            raise ValueError(
                f"held-out schedule extra_info.index drift at row {position}"
            )
        route_id = str(row.get("route_id", ""))
        if route_id not in CANONICAL_ROUTES:
            raise ValueError(f"held-out schedule has invalid route at row {position}")
        if row.get("data_source") != route_id or extra.get("route_id") != route_id:
            raise ValueError(f"held-out schedule route identity drift at row {position}")
        if row.get("agent_name") != AGENT_NAME:
            raise ValueError(f"held-out schedule agent drift at row {position}")
        if row.get("eval_padding") is not False:
            raise ValueError(f"held-out schedule contains padding at row {position}")
        data_idx = require_nonnegative_int(
            row.get("data_idx"), field=f"held-out schedule row {position} data_idx"
        )
        if data_idx in per_route_data_indices[route_id]:
            raise ValueError(
                f"held-out schedule repeats route-local data_idx {route_id}:{data_idx}"
            )
        per_route_data_indices[route_id].add(data_idx)
        item_id = str(row.get("item_id", ""))
        uid = str(row.get("uid", ""))
        if not item_id or item_id in seen_item_ids:
            raise ValueError(f"held-out schedule item_id collision at row {position}")
        if not uid or uid in seen_uids:
            raise ValueError(f"held-out schedule uid collision at row {position}")
        seen_item_ids.add(item_id)
        seen_uids.add(uid)
        source_item_id = str(extra.get("source_item_id", ""))
        expected_uid = _eval_uid(
            route_id=route_id,
            source_item_id=source_item_id,
            data_idx=data_idx,
            global_index=position,
        )
        if uid != expected_uid:
            raise ValueError(f"held-out schedule uid drift at row {position}")
    return rows


def pad_batch_rows(
    real_rows: Sequence[Mapping[str, Any]],
    *,
    batch_index: int,
    size_divisor: int,
    padding_index_base: int,
) -> list[dict[str, Any]]:
    """Pad prompts with explicit synthetic identities, never positional unpadding."""

    if not real_rows:
        raise ValueError("cannot pad an empty held-out batch")
    divisor = require_positive_int(size_divisor, field="size_divisor")
    normalized_batch_index = require_nonnegative_int(
        batch_index, field="batch_index"
    )
    index_base = require_nonnegative_int(
        padding_index_base, field="padding_index_base"
    )
    rows = [deepcopy(dict(row)) for row in real_rows]
    used_indices: set[int] = set()
    for position, row in enumerate(rows):
        if row.get("eval_padding") is not False or not str(row.get("uid", "")):
            raise ValueError(f"real held-out row {position} lacks explicit eval identity")
        row_index = require_nonnegative_int(
            row.get("index"), field=f"real held-out row {position} index"
        )
        if row_index in used_indices:
            raise ValueError("held-out batch contains duplicate real global indices")
        used_indices.add(row_index)
    pad_count = (-len(rows)) % divisor
    for pad_position in range(pad_count):
        source = deepcopy(rows[pad_position % len(rows)])
        source_uid = str(source["uid"])
        global_index = (
            index_base + normalized_batch_index * divisor + pad_position
        )
        uid_seed = {
            "batch_index": normalized_batch_index,
            "pad_position": pad_position,
            "source_uid": source_uid,
        }
        uid = f"camg-heldout-padding-v1-{sha256_bytes(canonical_json_bytes(uid_seed))}"
        if global_index in used_indices:
            raise ValueError(
                "padding_index_base collides with a real held-out global index"
            )
        used_indices.add(global_index)
        source_item_id = str(source["item_id"])
        source["uid"] = uid
        source["eval_padding"] = True
        source["index"] = global_index
        source["item_id"] = (
            f"padding:{source_item_id}:batch-{normalized_batch_index:06d}:"
            f"slot-{pad_position:04d}"
        )
        extra = dict(source.get("extra_info") or {})
        extra.update(
            {
                "eval_padding": True,
                "index": global_index,
                "schedule_position": global_index,
                "padding_source_uid": source_uid,
                "padding_batch_index": normalized_batch_index,
                "padding_position": pad_position,
            }
        )
        source["extra_info"] = extra
        rows.append(source)
    uids = [str(row["uid"]) for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError("held-out batch contains duplicate real/padding UID")
    return rows


def _sequence_field(
    fields: Mapping[str, Any], key: str, *, expected_length: int | None = None
) -> Sequence[Any]:
    value = fields.get(key)
    if value is None or isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise ValueError(f"generated output is missing row-aligned field {key!r}")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(
            f"generated output field {key!r} has length {len(value)}, "
            f"expected {expected_length}"
        )
    return value


def _plain_scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes, Mapping)):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def _parse_step_record(value: Any, *, output_position: int) -> dict[str, Any]:
    value = _plain_scalar(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid step_record_json at output row {output_position}"
            ) from exc
    if not isinstance(value, Mapping):
        raise TypeError(
            f"step_record_json at output row {output_position} is not an object"
        )
    return deepcopy(dict(value))


def _terminal_env_info(final_row: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    horizon = final_row.get("horizon_finalization")
    if isinstance(horizon, Mapping):
        horizon_env = horizon.get("env_info")
        if isinstance(horizon_env, Mapping):
            return deepcopy(dict(horizon_env)), "horizon_finalization.env_info"
    env_info = final_row.get("env_info_after")
    if not isinstance(env_info, Mapping):
        raise ValueError("terminal action row has no native env_info evidence")
    return deepcopy(dict(env_info)), "terminal_action.env_info_after"


def _adapter_event(row: Mapping[str, Any]) -> str | None:
    wrapper_evidence = row.get("wrapper_evidence")
    if not isinstance(wrapper_evidence, Mapping):
        return None
    adapter = wrapper_evidence.get("agemem_adapter")
    if not isinstance(adapter, Mapping):
        return None
    event = adapter.get("event")
    return str(event) if isinstance(event, str) else None


def _required_identity_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _expected_native_source_identity(
    input_row: Mapping[str, Any],
) -> dict[str, Any]:
    route_id = str(input_row.get("route_id", ""))
    if route_id not in CANONICAL_ROUTES:
        raise ValueError(f"unknown native source identity route {route_id!r}")
    data_idx = require_nonnegative_int(
        input_row.get("data_idx"), field=f"{route_id} native source data_idx"
    )
    extra = input_row.get("extra_info")
    if not isinstance(extra, Mapping):
        raise ValueError(f"{route_id} source identity lacks extra_info")
    source = extra.get("source_extra_info")
    if not isinstance(source, Mapping):
        raise ValueError(f"{route_id} source identity lacks source_extra_info")
    identity: dict[str, Any] = {
        "schema": NATIVE_SOURCE_IDENTITY_SCHEMA,
        "route_id": route_id,
        "data_idx": data_idx,
    }
    if route_id == "webshop":
        identity.update(
            scenario_id=_required_identity_text(
                source.get("scenario_id"), field="Shop scenario_id"
            ),
            orbit_index=require_nonnegative_int(
                source.get("orbit_index"), field="Shop orbit_index"
            ),
        )
    elif route_id == "swesmith":
        identity.update(
            instance_id=_required_identity_text(
                source.get("instance_id"), field="SWE-smith instance_id"
            ),
            base_repository=_required_identity_text(
                source.get("base_repository"),
                field="SWE-smith base_repository",
            ),
        )
    elif route_id == "literesearcher":
        identity.update(
            row_identity=require_sha256(
                source.get("row_identity"),
                field="LiteResearcher row_identity",
            ),
            source_pool_index=require_nonnegative_int(
                source.get("source_pool_index"),
                field="LiteResearcher source_pool_index",
            ),
        )
    else:
        identity.update(
            task_id=_required_identity_text(
                source.get("task_id"), field="AutoResearch task_id"
            ),
            source_family=_required_identity_text(
                source.get("source_family"), field="AutoResearch source_family"
            ),
            manifest_role=_required_identity_text(
                source.get("role"), field="AutoResearch manifest_role"
            ),
            manifest_sha256=require_sha256(
                source.get("manifest_sha256"),
                field="AutoResearch manifest_sha256",
            ),
        )
        if identity["manifest_role"] != "heldout":
            raise ValueError("AutoResearch native source manifest_role must be heldout")
    return identity


def _verify_action_native_source_identity(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    episode_uid: str,
    row_position: int,
) -> None:
    env_info = row.get("env_info_after")
    if not isinstance(env_info, Mapping):
        raise ValueError(
            f"episode {episode_uid} row {row_position} lacks native env_info"
        )
    observed = env_info.get("episode_source_identity")
    if not isinstance(observed, Mapping):
        raise ValueError(
            f"episode {episode_uid} row {row_position} lacks native source identity"
        )
    if dict(observed) != dict(expected):
        raise ValueError(
            f"episode {episode_uid} row {row_position} native source identity drift"
        )
    horizon = row.get("horizon_finalization")
    if isinstance(horizon, Mapping):
        horizon_env = horizon.get("env_info")
        if not isinstance(horizon_env, Mapping):
            raise ValueError(
                f"episode {episode_uid} row {row_position} horizon lacks native env_info"
            )
        horizon_identity = horizon_env.get("episode_source_identity")
        if not isinstance(horizon_identity, Mapping):
            raise ValueError(
                f"episode {episode_uid} row {row_position} horizon lacks native source identity"
            )
        if dict(horizon_identity) != dict(expected):
            raise ValueError(
                f"episode {episode_uid} row {row_position} native source identity drift"
            )


def _native_metric_evidence(
    route_id: str,
    action_rows: Sequence[Mapping[str, Any]],
    *,
    method_id: str = "agemem",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Locate native metric evidence without treating memory-only rows as task steps."""

    final_row = action_rows[-1]
    if route_id != "webshop":
        env_info, source = _terminal_env_info(final_row)
        return env_info, {
            "source": source,
            "action_row_uid": final_row.get("trajectory_row_uid"),
        }

    # The bundled-shopping wrapper has no horizon finalizer.  A method-owned
    # memory action may not call the native environment, so walk back to the
    # latest row carrying the native progress receipt.  This is intentionally
    # independent of any adapter-specific event name.
    for row in reversed(action_rows):
        env_info = row.get("env_info_after")
        if not isinstance(env_info, Mapping):
            continue
        if {"current_subtask_index", "subtask_count"}.issubset(env_info):
            return deepcopy(dict(env_info)), {
                "source": "latest_native_action.env_info_after",
                "action_row_uid": row.get("trajectory_row_uid"),
            }

    adapter_key = {
        "agemem": "agemem_adapter",
        "mem0": "mem0_adapter",
        "letta_code": "letta_code_adapter",
        "qwen35_4b": None,
    }.get(method_id)
    events = []
    for row in action_rows:
        evidence = row.get("wrapper_evidence")
        adapter = (
            evidence.get(adapter_key)
            if adapter_key is not None and isinstance(evidence, Mapping)
            else None
        )
        events.append(adapter.get("event") if isinstance(adapter, Mapping) else None)
    method_only_events = (
        {"memory_tool_action"}
        if method_id == "agemem"
        else {"memory_tool_action", "memory_filesystem_read"}
    )
    if method_id in {"agemem", "letta_code"} and events and all(
        event in method_only_events for event in events
    ):
        return {
            "current_subtask_index": 0,
            "subtask_count": 6,
        }, {
            "source": "initial_state_all_actions_are_memory_tools",
            "action_row_uid": None,
        }
    raise ValueError(
        "Shop trajectory has native actions but no current_subtask_index/"
        "subtask_count evidence"
    )


def native_success_metric(
    route_id: str, final_env_info: Mapping[str, Any]
) -> dict[str, Any]:
    """Extract only the registered native metric; never trust generic outcome."""

    if route_id == "webshop":
        completed = require_nonnegative_int(
            final_env_info.get("current_subtask_index"),
            field="Shop current_subtask_index",
        )
        total = require_positive_int(
            final_env_info.get("subtask_count"), field="Shop subtask_count"
        )
        if total != 6:
            raise ValueError(
                f"Shop held-out contract requires exactly six sessions, got {total}"
            )
        if completed > total:
            raise ValueError("Shop current_subtask_index exceeds subtask_count")
        return {
            "name": "shop_completed_sessions_rate",
            "numerator": completed,
            "denominator": total,
            "value": completed / total,
        }
    if route_id in {"swesmith", "literesearcher"}:
        success = final_env_info.get("episode_success")
        if not isinstance(success, bool):
            raise ValueError(f"{route_id} native episode_success must be boolean")
        return {
            "name": f"{route_id}_episode_success",
            "numerator": int(success),
            "denominator": 1,
            "value": float(success),
        }
    if route_id == "openmle_fast":
        grade = final_env_info.get("grade")
        if grade is None and "grade" in final_env_info:
            # AutoResearch intentionally does not synthesize a private grade
            # when the policy reaches a non-infrastructure terminal before a
            # valid submit (for example, the action or wall-time budget).  It
            # is still a registered native failure, not missing evidence.
            # Keep this fail-closed: only the exact public terminal receipt is
            # accepted; truncated/infrastructure or malformed receipts remain
            # evaluator errors and must be rescheduled upstream.
            counters = final_env_info.get("counters")
            terminal_reason = final_env_info.get("terminal_reason")
            if (
                final_env_info.get("terminal") is not True
                or final_env_info.get("truncated") is not False
                or final_env_info.get("episode_success") is not False
                or final_env_info.get("runtime_success") is not False
                or not isinstance(terminal_reason, str)
                or not terminal_reason
                or not isinstance(counters, Mapping)
            ):
                raise ValueError(
                    "AutoResearch ungraded terminal evidence is malformed"
                )
            grading_count = counters.get("grading_count")
            if (
                isinstance(grading_count, bool)
                or not isinstance(grading_count, int)
                or grading_count < 0
            ):
                raise ValueError(
                    "AutoResearch ungraded terminal lacks a valid grading_count"
                )
            return {
                "name": "autoresearch_beats_baseline_rate",
                "numerator": 0,
                "denominator": 1,
                "value": 0.0,
                "submission_valid": False,
                "improved_over_baseline": None,
                "grade_present": False,
                "terminal_reason": terminal_reason,
            }
        if not isinstance(grade, Mapping):
            raise ValueError("AutoResearch terminal evidence is missing grade")
        submission_valid = grade.get("submission_valid")
        if not isinstance(submission_valid, bool):
            raise ValueError("AutoResearch grade.submission_valid must be boolean")
        improved = grade.get("improved_over_baseline")
        if submission_valid and not isinstance(improved, bool):
            raise ValueError(
                "AutoResearch valid submission lacks boolean improved_over_baseline"
            )
        success = submission_valid and improved is True
        return {
            "name": "autoresearch_beats_baseline_rate",
            "numerator": int(success),
            "denominator": 1,
            "value": float(success),
            "submission_valid": submission_valid,
            "improved_over_baseline": improved if isinstance(improved, bool) else None,
            "grade_present": True,
            "terminal_reason": final_env_info.get("terminal_reason"),
        }
    raise ValueError(f"unknown CAMG held-out route {route_id!r}")


def _episode_record(
    input_row: Mapping[str, Any],
    action_rows: Sequence[Mapping[str, Any]],
    *,
    expected_global_step: int,
    route_max_rounds: Mapping[str, int],
    method_id: str = "agemem",
) -> dict[str, Any]:
    uid = str(input_row["uid"])
    route_id = str(input_row["route_id"])
    item_id = str(input_row["item_id"])
    data_idx = require_nonnegative_int(
        input_row.get("data_idx"), field=f"episode {uid} data_idx"
    )
    global_index = require_nonnegative_int(
        input_row.get("index"), field=f"episode {uid} index"
    )
    if not action_rows:
        raise ValueError(f"episode {uid} produced no action rows")
    verified_native_source_identity = _expected_native_source_identity(input_row)
    expected_orders = list(range(len(action_rows)))
    observed_orders = [
        require_nonnegative_int(
            row.get("trajectory_row_order"), field=f"episode {uid} row order"
        )
        for row in action_rows
    ]
    if observed_orders != expected_orders:
        raise ValueError(
            f"episode {uid} action rows are not contiguous: {observed_orders!r}"
        )
    returns: set[float] = set()
    attempts: set[int] = set()
    policy_steps: set[int] = set()
    for position, row in enumerate(action_rows):
        if row.get("schema") != ACTION_ROW_SCHEMA:
            raise ValueError(f"episode {uid} has an unexpected action-row schema")
        if (
            str(row.get("trajectory_uid")) != uid
            or str(row.get("route_id")) != route_id
            or str(row.get("data_source")) != route_id
            or str(row.get("item_id")) != item_id
            or require_nonnegative_int(
                row.get("data_idx"), field=f"episode {uid} row data_idx"
            )
            != data_idx
        ):
            raise ValueError(f"episode {uid} action-row identity drift")
        if row.get("trajectory_row_uid") != f"{uid}-row-{position}":
            raise ValueError(f"episode {uid} action-row UID drift at {position}")
        _verify_action_native_source_identity(
            row,
            expected=verified_native_source_identity,
            episode_uid=uid,
            row_position=position,
        )
        terminal = row.get("trajectory_terminal")
        if not isinstance(terminal, bool) or terminal != (position == len(action_rows) - 1):
            raise ValueError(f"episode {uid} terminal-row contract failed")
        try:
            returns.add(float(row["trajectory_return"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"episode {uid} has invalid trajectory_return") from exc
        attempts.add(
            require_nonnegative_int(
                row.get("sample_reschedule_attempt"),
                field=f"episode {uid} sample_reschedule_attempt",
            )
        )
        minimum_step = require_nonnegative_int(
            row.get("min_global_steps"),
            field=f"episode {uid} min_global_steps",
        )
        maximum_step = require_nonnegative_int(
            row.get("max_global_steps"),
            field=f"episode {uid} max_global_steps",
        )
        if minimum_step != maximum_step:
            raise ValueError(
                f"episode {uid} mixed policy versions inside action row {position}: "
                f"{minimum_step} != {maximum_step}"
            )
        policy_steps.add(minimum_step)
    if len(returns) != 1 or len(attempts) != 1:
        raise ValueError(f"episode {uid} has inconsistent trajectory metadata")
    if policy_steps != {expected_global_step}:
        raise ValueError(
            f"episode {uid} was not sampled entirely from checkpoint step "
            f"{expected_global_step}: observed {sorted(policy_steps)!r}"
        )
    method_evidence = summarize_method_step_records(
        action_rows,
        method_id=method_id,
    )
    if method_evidence.get("status") != "PASS":
        violations = method_evidence.get("violations")
        detail = violations[0] if isinstance(violations, list) and violations else "unknown"
        label = "AgeMem" if method_id == "agemem" else method_id
        raise ValueError(f"episode {uid} {label} evidence failed: {detail}")

    final_row = action_rows[-1]
    done = final_row.get("rollout_done_flag")
    if done is not True:
        configured_max_rounds = route_max_rounds.get(route_id)
        valid_shop_budget_terminal = (
            route_id == "webshop"
            and done is False
            and final_row.get("outcome") == "max_rounds"
            and configured_max_rounds == len(action_rows)
        )
        if not valid_shop_budget_terminal:
            raise ValueError(f"episode {uid} lacks a terminal wrapper transition")
    final_env_info, metric_evidence = _native_metric_evidence(
        route_id, action_rows, method_id=method_id
    )
    metric = native_success_metric(route_id, final_env_info)
    return {
        "schema": EPISODE_SCHEMA,
        "uid": uid,
        "route_id": route_id,
        "item_id": item_id,
        "data_idx": data_idx,
        "index": global_index,
        "source_identity": deepcopy(dict(input_row.get("extra_info") or {})),
        "verified_native_source_identity": verified_native_source_identity,
        "action_row_count": len(action_rows),
        "trajectory_return": next(iter(returns)),
        "sample_reschedule_attempt": next(iter(attempts)),
        "policy_global_step": expected_global_step,
        "terminal_outcome_informational_only": final_row.get("outcome"),
        "native_metric": metric,
        "native_metric_evidence": metric_evidence,
        "final_env_info": final_env_info,
        "method_id": method_id,
        "method_evidence": method_evidence,
        "final_action_row_uid": final_row["trajectory_row_uid"],
    }


def materialize_generated_batch(
    output_fields: Mapping[str, Any],
    input_rows: Sequence[Mapping[str, Any]],
    *,
    expected_global_step: int,
    route_max_rounds: Mapping[str, int],
    method_id: str = "agemem",
) -> dict[str, Any]:
    """Validate and partition expanded AgentLoop outputs by UID and padding flag."""

    expected_step = require_nonnegative_int(
        expected_global_step, field="expected_global_step"
    )
    normalized_max_rounds = normalize_route_max_rounds(route_max_rounds)
    if not input_rows:
        raise ValueError("generated batch has no input rows")
    expected: dict[str, dict[str, Any]] = {}
    for position, raw_row in enumerate(input_rows):
        row = deepcopy(dict(raw_row))
        uid = str(row.get("uid", ""))
        padding = row.get("eval_padding")
        if not uid or uid in expected:
            raise ValueError(f"input batch UID collision at row {position}")
        if not isinstance(padding, bool):
            raise ValueError(f"input batch eval_padding is not boolean at row {position}")
        expected[uid] = row

    uids = _sequence_field(output_fields, "uid")
    paddings = _sequence_field(
        output_fields, "eval_padding", expected_length=len(uids)
    )
    records = _sequence_field(
        output_fields, "step_record_json", expected_length=len(uids)
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for output_position, (raw_uid, raw_padding, raw_record) in enumerate(
        zip(uids, paddings, records)
    ):
        uid = str(_plain_scalar(raw_uid))
        if uid not in expected:
            raise ValueError(f"generated output contains unknown UID {uid!r}")
        padding = _plain_scalar(raw_padding)
        if not isinstance(padding, (bool,)):
            raise ValueError(
                f"generated output eval_padding is not boolean for UID {uid!r}"
            )
        if padding is not expected[uid]["eval_padding"]:
            raise ValueError(f"generated output padding marker drift for UID {uid!r}")
        record = _parse_step_record(raw_record, output_position=output_position)
        record["eval_padding"] = padding
        record["eval_uid"] = uid
        record["eval_global_index"] = int(expected[uid]["index"])
        grouped[uid].append(record)

    missing = sorted(set(expected) - set(grouped))
    if missing:
        raise ValueError(f"generated output is missing expected UIDs: {missing[:5]!r}")

    real_action_rows: list[dict[str, Any]] = []
    padding_action_rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for input_row in input_rows:
        uid = str(input_row["uid"])
        action_rows = grouped[uid]
        if input_row["eval_padding"]:
            _episode_record(
                input_row,
                action_rows,
                expected_global_step=expected_step,
                route_max_rounds=normalized_max_rounds,
                method_id=method_id,
            )
            padding_action_rows.extend(action_rows)
        else:
            real_action_rows.extend(action_rows)
            episodes.append(
                _episode_record(
                    input_row,
                    action_rows,
                    expected_global_step=expected_step,
                    route_max_rounds=normalized_max_rounds,
                    method_id=method_id,
                )
            )
    real_evidence = summarize_method_step_records(
        real_action_rows, method_id=method_id
    )
    padding_evidence = (
        summarize_method_step_records(padding_action_rows, method_id=method_id)
        if padding_action_rows
        else None
    )
    for label, summary in (("real", real_evidence), ("padding", padding_evidence)):
        if summary is not None and summary.get("status") != "PASS":
            violations = summary.get("violations")
            detail = violations[0] if isinstance(violations, list) and violations else "unknown"
            raise ValueError(f"{label} {method_id} batch evidence failed: {detail}")
    result = {
        "action_rows": real_action_rows,
        "episodes": episodes,
        "padding_action_rows": padding_action_rows,
        "input_uids": [str(row["uid"]) for row in input_rows],
        "real_uids": [
            str(row["uid"]) for row in input_rows if row["eval_padding"] is False
        ],
        "padding_uids": [
            str(row["uid"]) for row in input_rows if row["eval_padding"] is True
        ],
        "method_id": method_id,
        "method_evidence": {
            "real": real_evidence,
            "padding": padding_evidence,
        },
        "route_max_rounds": normalized_max_rounds,
    }
    if method_id == "agemem":
        result["agemem_evidence"] = {
            key: (
                value.get("legacy_agemem_summary")
                if isinstance(value, Mapping)
                else None
            )
            for key, value in result["method_evidence"].items()
        }
    return result


def aggregate_episode_metrics(
    episodes: Sequence[Mapping[str, Any]], *, require_all_routes: bool = True
) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = {
        route_id: {"episodes": 0, "numerator": 0, "denominator": 0}
        for route_id in CANONICAL_ROUTES
    }
    seen_uids: set[str] = set()
    for position, episode in enumerate(episodes):
        if episode.get("schema") != EPISODE_SCHEMA:
            raise ValueError(f"episode record {position} has an invalid schema")
        uid = str(episode.get("uid", ""))
        if not uid or uid in seen_uids:
            raise ValueError(f"episode record {position} has duplicate/blank UID")
        seen_uids.add(uid)
        route_id = str(episode.get("route_id", ""))
        if route_id not in totals:
            raise ValueError(f"episode record {position} has an invalid route")
        metric = episode.get("native_metric")
        if not isinstance(metric, Mapping):
            raise ValueError(f"episode record {position} has no native metric")
        numerator = require_nonnegative_int(
            metric.get("numerator"), field=f"episode {uid} metric numerator"
        )
        denominator = require_positive_int(
            metric.get("denominator"), field=f"episode {uid} metric denominator"
        )
        if numerator > denominator:
            raise ValueError(f"episode {uid} metric numerator exceeds denominator")
        totals[route_id]["episodes"] += 1
        totals[route_id]["numerator"] += numerator
        totals[route_id]["denominator"] += denominator

    route_metrics: dict[str, Any] = {}
    for route_id in CANONICAL_ROUTES:
        total = totals[route_id]
        if require_all_routes and total["episodes"] == 0:
            raise ValueError(f"held-out metrics are missing route {route_id!r}")
        value = (
            total["numerator"] / total["denominator"]
            if total["denominator"]
            else None
        )
        route_metrics[route_id] = {**total, "success_rate": value}
    available = [
        value["success_rate"]
        for value in route_metrics.values()
        if value["success_rate"] is not None
    ]
    return {
        "schema": METRICS_SCHEMA,
        "episode_count": len(episodes),
        "routes": route_metrics,
        "average_success": sum(available) / len(available) if available else None,
        "average_success_weighting": "equal_weight_per_environment",
        "generic_action_outcome_used": False,
    }


def _write_file_and_entry(
    directory: Path, name: str, payload: bytes
) -> dict[str, Any]:
    path = directory / name
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)
    return {"path": name, "bytes": len(payload), "sha256": sha256_bytes(payload)}


def commit_batch(
    run_dir: str | os.PathLike[str],
    *,
    batch_index: int,
    schedule_start: int,
    schedule_stop: int,
    materialized: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one complete batch directory and its content hashes."""

    root = Path(run_dir)
    batches = root / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    index = require_nonnegative_int(batch_index, field="batch_index")
    start = require_nonnegative_int(schedule_start, field="schedule_start")
    stop = require_nonnegative_int(schedule_stop, field="schedule_stop")
    if stop <= start:
        raise ValueError("schedule_stop must be greater than schedule_start")
    action_rows = [dict(row) for row in materialized["action_rows"]]
    episodes = [dict(row) for row in materialized["episodes"]]
    padding_rows = [dict(row) for row in materialized["padding_action_rows"]]
    input_uids = [str(uid) for uid in materialized["input_uids"]]
    real_uids = [str(uid) for uid in materialized["real_uids"]]
    padding_uids = [str(uid) for uid in materialized["padding_uids"]]
    method_id = str(materialized.get("method_id") or "agemem")
    method_evidence = deepcopy(dict(materialized["method_evidence"]))
    route_max_rounds = normalize_route_max_rounds(
        materialized.get("route_max_rounds")
    )
    run_contract = read_json(root / "run-contract.json")
    if not isinstance(run_contract, Mapping):
        raise TypeError("held-out run contract must be an object")
    contract_registry = run_contract.get("route_registry")
    contract_max_rounds = normalize_route_max_rounds(
        contract_registry.get("max_rounds")
        if isinstance(contract_registry, Mapping)
        else None
    )
    if route_max_rounds != contract_max_rounds:
        raise ValueError("batch route max-rounds differ from the run contract")
    if stop - start != len(real_uids) or len(episodes) != len(real_uids):
        raise ValueError(
            "batch schedule interval, real UID count, and episode count differ"
        )
    if input_uids != real_uids + padding_uids:
        raise ValueError("batch input UID order must be real rows followed by padding")
    if len(set(input_uids)) != len(input_uids):
        raise ValueError("batch materialization contains duplicate input UIDs")
    if [str(episode.get("uid", "")) for episode in episodes] != real_uids:
        raise ValueError("batch episode UID order differs from real input UID order")
    final = batches / f"batch-{index:06d}"
    if final.exists():
        raise FileExistsError(f"batch directory already exists: {final}")
    leftovers = list(batches.glob(f".{final.name}.*.tmp"))
    if leftovers:
        raise RuntimeError(f"incomplete atomic batch directories exist: {leftovers!r}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".tmp", dir=batches)
    )
    try:
        files = {
            "action_rows": _write_file_and_entry(
                temporary, "action-rows.jsonl", jsonl_bytes(action_rows)
            ),
            "episodes": _write_file_and_entry(
                temporary, "episodes.jsonl", jsonl_bytes(episodes)
            ),
            "padding_action_rows": _write_file_and_entry(
                temporary, "padding-action-rows.jsonl", jsonl_bytes(padding_rows)
            ),
        }
        receipt = {
            "schema": BATCH_SCHEMA,
            "batch_index": index,
            "schedule_start": start,
            "schedule_stop": stop,
            "real_input_count": len(materialized["real_uids"]),
            "padding_input_count": len(materialized["padding_uids"]),
            "real_action_row_count": len(action_rows),
            "padding_action_row_count": len(padding_rows),
            "episode_count": len(episodes),
            "input_uids": input_uids,
            "real_uids": real_uids,
            "padding_uids": padding_uids,
            "batch_metrics": aggregate_episode_metrics(
                episodes, require_all_routes=False
            ),
            "method_id": method_id,
            "method_evidence": method_evidence,
            "route_max_rounds": route_max_rounds,
            "files": files,
        }
        if method_id == "agemem":
            receipt["agemem_evidence"] = deepcopy(
                dict(materialized["agemem_evidence"])
            )
        _write_file_and_entry(
            temporary, "receipt.json", canonical_json_bytes(receipt)
        )
        os.replace(temporary, final)
        directory_fd = os.open(batches, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return receipt
    except BaseException:
        for child in temporary.iterdir():
            child.unlink(missing_ok=True)
        temporary.rmdir()
        raise


def verify_batch_directory(
    directory: str | os.PathLike[str],
    *,
    expected_batch_index: int | None = None,
) -> dict[str, Any]:
    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"batch path must be a regular directory: {path}")
    receipt = read_json(path / "receipt.json")
    if not isinstance(receipt, Mapping) or receipt.get("schema") != BATCH_SCHEMA:
        raise ValueError(f"invalid batch receipt schema: {path}")
    run_contract = read_json(path.parent.parent / "run-contract.json")
    if not isinstance(run_contract, Mapping):
        raise TypeError("held-out run contract must be an object")
    expected_global_step = require_nonnegative_int(
        run_contract.get("checkpoint_step"), field="run checkpoint_step"
    )
    method_id = str(run_contract.get("method_id") or "agemem")
    contract_registry = run_contract.get("route_registry")
    route_max_rounds = normalize_route_max_rounds(
        contract_registry.get("max_rounds")
        if isinstance(contract_registry, Mapping)
        else None
    )
    if normalize_route_max_rounds(receipt.get("route_max_rounds")) != route_max_rounds:
        raise ValueError("batch route max-rounds differ from the run contract")
    index = require_nonnegative_int(receipt.get("batch_index"), field="batch_index")
    if expected_batch_index is not None and index != expected_batch_index:
        raise ValueError(
            f"batch receipt index mismatch: expected {expected_batch_index}, got {index}"
        )
    files = receipt.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("batch receipt files must be an object")
    expected_file_names = {"receipt.json"}
    for key in ("action_rows", "episodes", "padding_action_rows"):
        entry = files.get(key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"batch receipt lacks file entry {key!r}")
        name = str(entry.get("path", ""))
        if not name or Path(name).name != name:
            raise ValueError(f"batch file entry {key!r} has an invalid path")
        file_path = require_regular_file(path / name, field=f"batch file {key}")
        expected_file_names.add(name)
        if file_path.stat().st_size != require_nonnegative_int(
            entry.get("bytes"), field=f"batch file {key} bytes"
        ):
            raise ValueError(f"batch file {key!r} byte count mismatch")
        if sha256_file(file_path) != require_sha256(
            entry.get("sha256"), field=f"batch file {key} sha256"
        ):
            raise ValueError(f"batch file {key!r} sha256 mismatch")
    observed_names = {child.name for child in path.iterdir()}
    if observed_names != expected_file_names:
        raise ValueError(
            f"batch directory has unexpected/missing files: "
            f"expected {sorted(expected_file_names)!r}, got {sorted(observed_names)!r}"
        )
    action_rows = _read_jsonl_allow_empty(
        path / str(files["action_rows"]["path"])
    )
    episodes = _read_jsonl_allow_empty(path / str(files["episodes"]["path"]))
    padding_rows = _read_jsonl_allow_empty(
        path / str(files["padding_action_rows"]["path"])
    )
    episode_count = require_nonnegative_int(
        receipt.get("episode_count"), field="batch episode_count"
    )
    real_input_count = require_nonnegative_int(
        receipt.get("real_input_count"), field="batch real_input_count"
    )
    padding_input_count = require_nonnegative_int(
        receipt.get("padding_input_count"), field="batch padding_input_count"
    )
    real_action_row_count = require_nonnegative_int(
        receipt.get("real_action_row_count"), field="batch real_action_row_count"
    )
    padding_action_row_count = require_nonnegative_int(
        receipt.get("padding_action_row_count"),
        field="batch padding_action_row_count",
    )
    if len(episodes) != episode_count or episode_count != real_input_count:
        raise ValueError("batch episode count mismatch")
    if len(action_rows) != real_action_row_count:
        raise ValueError("batch real action-row count mismatch")
    if len(padding_rows) != padding_action_row_count:
        raise ValueError("batch padding action-row count mismatch")
    input_uids = list(receipt.get("input_uids", []))
    real_uids = list(receipt.get("real_uids", []))
    padding_uids = list(receipt.get("padding_uids", []))
    if (
        len(real_uids) != real_input_count
        or len(padding_uids) != padding_input_count
        or input_uids != real_uids + padding_uids
        or len(set(input_uids)) != len(input_uids)
    ):
        raise ValueError("batch receipt UID partition mismatch")
    if [episode.get("uid") for episode in episodes] != real_uids:
        raise ValueError("batch episode UID order mismatch")
    real_action_uids = [str(row.get("eval_uid", "")) for row in action_rows]
    padding_action_uids = [str(row.get("eval_uid", "")) for row in padding_rows]
    if any(
        row.get("eval_padding") is not False or uid not in set(real_uids)
        for row, uid in zip(action_rows, real_action_uids)
    ):
        raise ValueError("batch real action-row UID/padding partition mismatch")
    if any(
        row.get("eval_padding") is not True or uid not in set(padding_uids)
        for row, uid in zip(padding_rows, padding_action_uids)
    ):
        raise ValueError("batch padding action-row UID/padding partition mismatch")
    real_counts: dict[str, int] = defaultdict(int)
    for uid in real_action_uids:
        real_counts[uid] += 1
    if any(
        real_counts.get(str(episode["uid"]), 0) != episode.get("action_row_count")
        for episode in episodes
    ):
        raise ValueError("batch episode/action-row count linkage mismatch")
    action_rows_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        action_rows_by_uid[str(row.get("eval_uid", ""))].append(row)
    for episode in episodes:
        uid = str(episode["uid"])
        reconstructed = _episode_record(
            {
                "uid": uid,
                "route_id": episode["route_id"],
                "item_id": episode["item_id"],
                "data_idx": episode["data_idx"],
                "index": episode["index"],
                "extra_info": episode.get("source_identity", {}),
            },
            action_rows_by_uid[uid],
            expected_global_step=expected_global_step,
            route_max_rounds=route_max_rounds,
            method_id=method_id,
        )
        if reconstructed != episode:
            raise ValueError(f"episode {uid} differs from its action-row evidence")
    recomputed_method_evidence = {
        "real": summarize_method_step_records(action_rows, method_id=method_id),
        "padding": (
            summarize_method_step_records(padding_rows, method_id=method_id)
            if padding_rows
            else None
        ),
    }
    if any(
        summary is not None and summary.get("status") != "PASS"
        for summary in recomputed_method_evidence.values()
    ):
        raise ValueError(f"batch {method_id} adapter evidence failed validation")
    stored_evidence = receipt.get("method_evidence")
    if stored_evidence is None and method_id == "agemem":
        # Read-only compatibility with the immutable AgeMem v1 receipts.
        stored_evidence = receipt.get("agemem_evidence")
        recomputed_method_evidence = {
            key: (
                value.get("legacy_agemem_summary")
                if isinstance(value, Mapping)
                else None
            )
            for key, value in recomputed_method_evidence.items()
        }
    if stored_evidence != recomputed_method_evidence:
        raise ValueError(
            f"batch {method_id} evidence differs from the action-row ledger"
        )
    if method_id == "agemem" and "agemem_evidence" in receipt:
        expected_legacy = {
            key: (
                value.get("legacy_agemem_summary")
                if isinstance(value, Mapping)
                else None
            )
            for key, value in {
                "real": summarize_method_step_records(
                    action_rows, method_id="agemem"
                ),
                "padding": (
                    summarize_method_step_records(
                        padding_rows, method_id="agemem"
                    )
                    if padding_rows
                    else None
                ),
            }.items()
        }
        if receipt.get("agemem_evidence") != expected_legacy:
            raise ValueError(
                "batch AgeMem evidence differs from the action-row ledger"
            )
    recomputed_metrics = aggregate_episode_metrics(
        episodes, require_all_routes=False
    )
    if receipt.get("batch_metrics") != recomputed_metrics:
        raise ValueError("batch metrics differ from the episode ledger")
    return dict(receipt)


def inspect_resume_state(
    run_dir: str | os.PathLike[str],
    *,
    schedule_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify a contiguous receipt chain and return the next untouched row."""

    root = Path(run_dir)
    batches = root / "batches"
    if not batches.exists():
        return {"next_batch_index": 0, "next_schedule_position": 0}
    if batches.is_symlink() or not batches.is_dir():
        raise ValueError(f"run batches path must be a regular directory: {batches}")
    leftovers = sorted(batches.glob(".*.tmp"))
    if leftovers:
        raise RuntimeError(f"incomplete atomic batch directories exist: {leftovers!r}")
    directories: list[Path] = []
    for candidate in sorted(batches.iterdir()):
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or _BATCH_DIRECTORY.fullmatch(candidate.name) is None
        ):
            raise ValueError(f"unexpected entry in batch directory: {candidate.name}")
        directories.append(candidate)
    next_position = 0
    expected_uids = [str(row["uid"]) for row in schedule_rows]
    for expected_index, directory in enumerate(directories):
        match = _BATCH_DIRECTORY.fullmatch(directory.name)
        if match is None or int(match.group(1)) != expected_index:
            raise ValueError(f"non-contiguous batch directory: {directory.name}")
        receipt = verify_batch_directory(
            directory, expected_batch_index=expected_index
        )
        start = require_nonnegative_int(
            receipt.get("schedule_start"), field="schedule_start"
        )
        stop = require_nonnegative_int(
            receipt.get("schedule_stop"), field="schedule_stop"
        )
        if start != next_position or stop > len(schedule_rows):
            raise ValueError(f"batch {expected_index} schedule interval is not contiguous")
        if list(receipt["real_uids"]) != expected_uids[start:stop]:
            raise ValueError(f"batch {expected_index} real UID slice drift")
        next_position = stop
    return {
        "next_batch_index": len(directories),
        "next_schedule_position": next_position,
    }


def initialize_run_contract(
    run_dir: str | os.PathLike[str], contract: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(run_dir)
    if root.is_symlink():
        raise ValueError(f"run directory must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    normalized = deepcopy(dict(contract))
    if normalized.get("schema") != RUN_SCHEMA:
        raise ValueError(f"run contract schema must be {RUN_SCHEMA!r}")
    path = root / "run-contract.json"
    if path.exists():
        existing = read_json(path)
        if existing != normalized:
            raise ValueError("existing held-out run contract differs from requested run")
    else:
        unexpected = [child.name for child in root.iterdir()]
        if unexpected:
            raise ValueError(
                "refusing to initialize a run contract in a non-empty directory: "
                f"{unexpected!r}"
            )
        atomic_write_json(path, normalized)
    return normalized


def finalize_run_metrics(
    run_dir: str | os.PathLike[str],
    *,
    expected_episode_count: int,
) -> dict[str, Any]:
    root = Path(run_dir)
    batches = root / "batches"
    if batches.is_symlink() or not batches.is_dir():
        raise ValueError(f"run batches path must be a regular directory: {batches}")
    directories: list[Path] = []
    for candidate in sorted(batches.iterdir()):
        match = _BATCH_DIRECTORY.fullmatch(candidate.name)
        if candidate.is_symlink() or not candidate.is_dir() or match is None:
            raise ValueError(f"unexpected entry in batch directory: {candidate.name}")
        if int(match.group(1)) != len(directories):
            raise ValueError(f"non-contiguous batch directory: {candidate.name}")
        directories.append(candidate)
    episodes: list[dict[str, Any]] = []
    next_position = 0
    for expected_index, directory in enumerate(directories):
        receipt = verify_batch_directory(
            directory, expected_batch_index=expected_index
        )
        start = require_nonnegative_int(
            receipt.get("schedule_start"), field="schedule_start"
        )
        stop = require_nonnegative_int(
            receipt.get("schedule_stop"), field="schedule_stop"
        )
        if start != next_position or stop <= start:
            raise ValueError(
                f"batch {expected_index} schedule interval is not contiguous"
            )
        next_position = stop
        episode_path = directory / receipt["files"]["episodes"]["path"]
        episodes.extend(read_jsonl(episode_path))
    if len(episodes) != expected_episode_count or next_position != expected_episode_count:
        raise ValueError(
            f"held-out run has {len(episodes)} episodes through schedule position "
            f"{next_position}, expected {expected_episode_count}"
        )
    metrics = aggregate_episode_metrics(episodes, require_all_routes=True)
    atomic_write_json(root / "final-metrics.json", metrics)
    return metrics
