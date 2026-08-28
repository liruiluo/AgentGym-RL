"""Fail-closed attestation for native veRL fully-asynchronous AMG runs.

Both receipt generations are supported, but they remain deliberately distinct:
the legacy receipt keeps its publication checks while the multitask receipt binds
only opaque route labels and generic owner-emitted accounting.  No rollout,
queue, or behavior event is synthesized when its owning artifact omitted it.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config_contract import inspect_schedule, verify_resolved_config
from .identity import (
    EXPECTED_VERL_COMMIT,
    LOCKED_MODEL_FILE_SHA256,
    sha256_file,
    validate_training_runtime_lock,
)
from .routes import load_route_registry

_LEGACY_RECEIPT_SCHEMA = "amg_verl_fully_async_launch_receipt_v5"
_MULTITASK_RECEIPT_SCHEMA = "amg_verl_fully_async_multitask_launch_receipt_v1"
_MULTITASK_SOURCE_LOCK_SCHEMA = "amg_multitask_launcher_source_lock_v1"
_MULTITASK_SCHEDULE_CERTIFICATE_SCHEMA = "amg_multitask_schedule_certificate_v1"
_FINAL_STATISTICS_SCHEMA = "verl_fully_async_final_statistics_v1"
_FINAL_STATISTICS_MARKER = "[FullyAsyncTaskRunner][FinalStatistics] "
_FINAL_STATISTICS_VERL_COMMIT = "f3ac28fe54c945e092b9630030f44d236a106a11"
_FINAL_STATISTICS_FIELDS = frozenset(
    {"schema", "queue", "rollouter", "trainer", "queue_cleanup"}
)
_FINAL_STATISTICS_QUEUE_FIELDS = frozenset(
    {
        "queue_size",
        "total_produced",
        "total_consumed",
        "dropped_samples",
        "total_cleared",
        "max_queue_size",
        "enqueued_by_data_source",
        "consumed_by_data_source",
        "evicted_by_data_source",
        "cleared_by_data_source",
        "resident_by_data_source",
    }
)
_FINAL_STATISTICS_ROLLOUTER_FIELDS = frozenset(
    {
        "monitor/active_tasks_size",
        "monitor/queue/pending_queue_size",
        "monitor/queue/mq_queue_size",
        "count/total_generated_samples",
        "count/rollout_dispatched_samples",
        "count/rollout_inflight_samples",
        "count/rollout_completed_samples",
        "count/rollout_failed_samples",
        "count/rollout_cancelled_samples",
        "count/queue_enqueued_samples",
        "count/queue_dequeued_samples",
        "count/queue_overflow_evictions",
        "count/queue_cleared_samples",
        "count/queue_resident_samples",
        "count/staleness_samples",
        "count/dropped_stale_samples",
        "static/max_required_samples",
        "static/required_samples",
        "static/staleness_threshold",
        "static/max_queue_size",
        "static/max_concurrent_samples",
    }
)
_FINAL_STATISTICS_ROUTE_EVENTS = (
    "rollout_dispatched",
    "rollout_inflight",
    "rollout_completed",
    "rollout_failed",
    "rollout_cancelled",
    "queue_enqueued",
    "queue_dequeued",
    "queue_overflow_evicted",
    "queue_cleared",
    "queue_resident",
)
_FINAL_STATISTICS_TRAINER_FIELDS = frozenset(
    {
        "optimizer_consumed_episodes",
        "optimizer_consumed_action_rows",
        "optimizer_consumed_policy_response_tokens",
        "optimizer_consumed_episodes_by_data_source",
        "optimizer_consumed_action_rows_by_data_source",
        "optimizer_consumed_policy_response_tokens_by_data_source",
        "stale_action_rows",
        "stale_action_rows_by_data_source",
        "current_param_version",
    }
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_RAY_LOG_PREFIX = re.compile(r"^\([^()\r\n]* pid=[0-9]+(?:, ip=[^()\r\n]+)?\) ")
_MEMORY_EVENTS = frozenset(
    {"write", "compaction", "read", "reuse", "modify", "execute"}
)
_REQUIRED_RUNTIME_ARTIFACTS = (
    "file_logger",
    "rollout_data",
    "hydra_config",
    "checkpoints",
    "finalization",
)
_LEGACY_RUNTIME_ARTIFACT_RELATIVE_PATHS = {
    "file_logger": Path("metrics.jsonl"),
    "rollout_data": Path("rollout_data"),
    "hydra_config": Path("hydra/.hydra/config.yaml"),
    "checkpoints": Path("checkpoints"),
    "finalization": Path("finalization.json"),
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"required {label} is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"required {label} must be a JSON object: {path}")
    return value


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - the veRL runtime owns PyYAML
        raise RuntimeError("PyYAML is required by the post-run finalizer") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"required {label} is not valid YAML: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"required {label} must be a YAML mapping: {path}")
    return value


def _load_resolved_hydra_yaml(path: Path, label: str) -> Mapping[str, Any]:
    """Load Hydra's persisted config after resolving native interpolations."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - veRL depends on OmegaConf
        raise RuntimeError("OmegaConf is required by the post-run finalizer") from exc
    try:
        value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        raise ValueError(f"required {label} cannot be resolved: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"required {label} must resolve to a mapping: {path}")
    return value


def _jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise ValueError(f"blank row in {label} at {path}:{line_number}")
                value = json.loads(raw)
                if not isinstance(value, Mapping):
                    raise TypeError(
                        f"{label} row is not an object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _at(value: Any, dotted: str, default: Any = None) -> Any:
    current = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _same_path(left: Any, right: Path) -> bool:
    if not isinstance(left, str) or not left:
        return False
    try:
        return Path(left).resolve() == right.resolve()
    except (OSError, RuntimeError):
        return False


def _finite_positive(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and number > 0.0


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _nonnegative_integral(value: Any) -> int | None:
    """Normalize JSON integer metrics, including FileLogger's integral floats."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or int(value) != value:
        return None
    return int(value)


def _nonnegative_int(value: Any) -> int | None:
    """Accept only the JSON integer type emitted by FinalStatistics owners."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _contiguous_positive_prefix(values: Iterable[Any]) -> int:
    """Count the observed 1..N prefix without mistaking a declared budget for work."""

    observed = {
        int(value)
        for value in values
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    }
    completed = 0
    while completed + 1 in observed:
        completed += 1
    return completed


def _lcm(left: int, right: int) -> int:
    return abs(left * right) // math.gcd(left, right)


def _sha256_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _route_ids(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    if any(not isinstance(route_id, str) for route_id in value):
        return None
    normalized = tuple(value)
    if (
        not 1 <= len(normalized) <= 4
        or len(set(normalized)) != len(normalized)
        or any(
            not route_id or len(route_id) > 256 or "\n" in route_id or "\r" in route_id
            for route_id in normalized
        )
    ):
        return None
    return normalized


def _path_within(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute() or candidate.is_symlink():
        return None
    try:
        resolved_root = root.resolve()
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return None
    if resolved != resolved_root and resolved_root not in resolved.parents:
        return None
    return resolved


def _resolved_path_within(root: Path, value: Any) -> Path | None:
    """Resolve a receipt path for collision checks, including symlink targets."""

    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        return None
    try:
        resolved_root = root.resolve()
        resolved = Path(value).resolve()
    except (OSError, RuntimeError):
        return None
    if resolved != resolved_root and resolved_root not in resolved.parents:
        return None
    return resolved


def _receipt_protected_paths(
    launch: Mapping[str, Any],
    run_dir: Path,
    *,
    include_finalization: bool,
) -> tuple[set[Path], set[Path]]:
    """Return receipt-bound input files and directory trees inside ``run_dir``."""

    protected_files = {(run_dir / "launch-receipt.json").resolve()}
    protected_directories: set[Path] = set()
    runtime_artifacts = launch.get("runtime_artifacts")
    if isinstance(runtime_artifacts, Mapping):
        for field, raw_path in runtime_artifacts.items():
            if field == "finalization" and not include_finalization:
                continue
            path = _resolved_path_within(run_dir, raw_path)
            if path is None:
                continue
            if field in {"rollout_data", "checkpoints"}:
                protected_directories.add(path)
            else:
                protected_files.add(path)
    for raw_path in (
        _at(launch, "resolved_config.path"),
        _at(launch, "schedule.path"),
        _at(launch, "inputs.route_registry"),
        _at(launch, "launch_identity.source_lock_path"),
        _at(launch, "launch_identity.schedule_certificate_path"),
        _at(launch, "launch_identity.route_registry_path"),
    ):
        path = _resolved_path_within(run_dir, raw_path)
        if path is not None:
            protected_files.add(path)
    return protected_files, protected_directories


def _path_overlaps_inputs(
    path: Path,
    protected_files: set[Path],
    protected_directories: set[Path],
) -> bool:
    resolved = path.resolve()
    return resolved in protected_files or any(
        resolved == directory or directory in resolved.parents
        for directory in protected_directories
    )


def _bound_runtime_paths(
    launch: Mapping[str, Any], run_dir: Path, *, require_trainer_log: bool
) -> tuple[dict[str, Path], list[str]]:
    raw = launch.get("runtime_artifacts")
    if not isinstance(raw, Mapping):
        return {}, ["launch receipt has no runtime_artifacts mapping"]
    fields = list(_REQUIRED_RUNTIME_ARTIFACTS)
    if require_trainer_log:
        fields.append("trainer_log")
    paths: dict[str, Path] = {}
    errors: list[str] = []
    for field in fields:
        path = _path_within(run_dir, raw.get(field))
        if path is None:
            errors.append(
                f"launch runtime artifact {field} must be an absolute path inside run_dir"
            )
        else:
            paths[field] = path
    if len(set(paths.values())) != len(paths):
        errors.append("launch runtime artifact paths must be distinct")
    return paths, errors


def _load_final_statistics(path: Path) -> Mapping[str, Any]:
    """Load the one exact terminal owner snapshot emitted by pinned veRL."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required trainer log is missing: {path}")
    matches: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                message = _ANSI_ESCAPE.sub("", line.rstrip("\r\n"))
                if not message.startswith(_FINAL_STATISTICS_MARKER):
                    prefix = _RAY_LOG_PREFIX.match(message)
                    if prefix is None:
                        continue
                    message = message[prefix.end() :]
                if not message.startswith(_FINAL_STATISTICS_MARKER):
                    continue
                payload = message.removeprefix(_FINAL_STATISTICS_MARKER)
                try:
                    value = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid FinalStatistics JSON at {path}:{line_number}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise TypeError(
                        f"FinalStatistics must be an object at {path}:{line_number}"
                    )
                matches.append(value)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read trainer log: {path}") from exc
    if len(matches) != 1:
        raise ValueError(
            f"trainer log must contain exactly one FinalStatistics row, got {len(matches)}"
        )
    statistics = matches[0]
    if statistics.get("schema") != _FINAL_STATISTICS_SCHEMA:
        raise ValueError("FinalStatistics schema mismatch")
    if set(statistics) != _FINAL_STATISTICS_FIELDS:
        raise ValueError("FinalStatistics top-level fields differ from pinned veRL")
    return statistics


def _exact_counter(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    normalized: dict[str, int] = {}
    for key, count in value.items():
        route_id = str(key)
        parsed = _nonnegative_int(count)
        if not route_id or parsed is None:
            raise ValueError(f"{label} contains an invalid route/count")
        normalized[route_id] = parsed
    return normalized


def _flat_route_counter(
    value: Mapping[str, Any], *, prefix: str, label: str
) -> dict[str, int]:
    marker = f"{prefix}/data_source/"
    selected = {
        str(key)[len(marker) :]: item
        for key, item in value.items()
        if str(key).startswith(marker)
    }
    return _exact_counter(selected, label=label)


def _normalized_counter(
    observed: Mapping[str, int], route_ids: Sequence[str]
) -> dict[str, int]:
    return {route_id: int(observed.get(route_id, 0)) for route_id in route_ids}


def _metric_route_counter(
    items: Sequence[tuple[str, Any]], *, prefix: str, label: str
) -> dict[str, int]:
    """Parse one unambiguous set of FileLogger per-route integer metrics."""

    observed: dict[str, int] = {}
    for key, value in items:
        if not key.startswith(prefix):
            continue
        route_id = key.removeprefix(prefix)
        if not route_id or route_id in observed:
            raise ValueError(f"{label} contains a missing or repeated route")
        parsed = _nonnegative_integral(value)
        if parsed is None:
            raise ValueError(f"{label} contains a non-integral count")
        observed[route_id] = parsed
    return observed


def _rolling_episode_shares(
    per_update: Sequence[Mapping[str, int]], route_ids: Sequence[str]
) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for end in range(8, len(per_update) + 1):
        counts = {
            route_id: sum(
                int(update.get(route_id, 0)) for update in per_update[end - 8 : end]
            )
            for route_id in route_ids
        }
        total = sum(counts.values())
        shares = {
            route_id: (counts[route_id] / total if total else 0.0)
            for route_id in route_ids
        }
        invalid = [
            route_id
            for route_id, share in shares.items()
            if share < 0.20 or share > 0.30
        ]
        window = {
            "start_update": end - 7,
            "end_update": end,
            "episodes": counts,
            "shares": shares,
            "status": "pass" if not invalid else "fail",
        }
        windows.append(window)
        if invalid:
            violations.append(
                {
                    "start_update": end - 7,
                    "end_update": end,
                    "routes": invalid,
                }
            )
    return {
        "window_size": 8,
        "bounds": {"minimum": 0.20, "maximum": 0.30},
        "first_applicable_update": 8,
        "status": (
            "not_applicable"
            if len(per_update) < 8
            else ("pass" if not violations else "fail")
        ),
        "windows": windows,
        "violations": violations,
    }


def _emitted_memory_events(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return only wrapper-emitted generic memory events for one action row."""

    evidence = record.get("wrapper_evidence")
    if not isinstance(evidence, Mapping):
        return ()
    raw_event = evidence.get("memory_event")

    # Compatibility for the task-neutral compaction receipt predating the
    # generic memory_event field.  Both events are explicitly attested by the
    # wrapper; no action text is parsed.
    legacy_compaction = (
        evidence.get("event") == "context_compaction"
        and evidence.get("continuation_persisted") is True
        and evidence.get("preserved_policy_output") is True
        and evidence.get("preserved_native_observation") is True
        and isinstance(evidence.get("continuation_path"), str)
        and bool(evidence.get("continuation_path"))
        and isinstance(evidence.get("native_action_kind"), str)
        and bool(evidence.get("native_action_kind"))
        and evidence.get("native_action_status") == "completed"
        and _at(record, "env_info_after.action_kind")
        == evidence.get("native_action_kind")
        and _at(record, "env_info_after.action_status") == "completed"
        and _at(record, "context_transition.operation") == "replace_messages"
        and isinstance(_at(record, "context_transition.messages"), Sequence)
        and not isinstance(_at(record, "context_transition.messages"), (str, bytes))
        and isinstance(record.get("control_request"), str)
        and bool(record.get("control_request", "").strip())
    )
    if legacy_compaction:
        # The old receipt explicitly attests that one policy action persisted the
        # continuation document and then replaced the context.
        return ("write", "compaction") if raw_event is None else ()
    if not isinstance(raw_event, str) or raw_event not in _MEMORY_EVENTS:
        return ()
    if raw_event == "read" and not (
        evidence.get("value") is not None
        or evidence.get("document_read_observed") is True
    ):
        return ()
    if raw_event == "execute":
        if evidence.get("outcome") != "success":
            return ()
        info = record.get("env_info_after")
        if info is not None and not isinstance(info, Mapping):
            return ()
        if isinstance(info, Mapping):
            if "action_status" in info and info.get("action_status") != "completed":
                return ()
            counters = info.get("counter_delta")
            if counters is not None and not isinstance(counters, Mapping):
                return ()
            if (
                isinstance(counters, Mapping)
                and "execution_completed_count" in counters
                and _nonnegative_integral(counters.get("execution_completed_count"))
                != 1
            ):
                return ()
            execution = info.get("execution")
            if execution is not None and not isinstance(execution, Mapping):
                return ()
            if isinstance(execution, Mapping):
                if "status" in execution and execution.get("status") != "completed":
                    return ()
                if "exit_code" in execution and execution.get("exit_code") not in {
                    None,
                    0,
                }:
                    return ()
                if (
                    "execution_completed_delta" in execution
                    and _nonnegative_integral(
                        execution.get("execution_completed_delta")
                    )
                    != 1
                ):
                    return ()
    return (raw_event,)


def _has_complete_memory_chain(
    episode: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> bool:
    stage = 0
    for record, _document in episode:
        events = set(_emitted_memory_events(record))
        if stage == 0:
            if "write" in events:
                # Only the legacy context-compaction receipt may attest these
                # two adjacent operations on the same policy action row.
                stage = 2 if "compaction" in events else 1
            continue
        if stage == 1:
            if "compaction" in events:
                stage = 2
            continue
        if stage == 2:
            if "read" in events:
                stage = 3
            continue
        if stage == 3:
            if events.intersection({"reuse", "modify"}):
                stage = 4
            continue
        if stage == 4 and "execute" in events:
            return True
    return False


def _terminal_row_closes_trajectory(
    record: Mapping[str, Any],
    *,
    episode_length: int,
    route_max_rounds: Mapping[str, int],
) -> bool:
    """Distinguish environment completion from a valid policy horizon.

    A rollout can end because the environment returned ``done`` or because the
    route-local AgentLoop exhausted its configured policy-action budget.  The
    latter remains a complete PPO trajectory even though the environment did
    not terminate.  It is accepted only when the terminal receipt explicitly
    says ``max_rounds`` and the contiguous action-row count exactly matches the
    immutable route registry.
    """

    done = record.get("rollout_done_flag")
    if done is True:
        return True
    if done is not False or record.get("outcome") != "max_rounds":
        return False
    route_id = record.get("route_id")
    if not isinstance(route_id, str):
        return False
    return route_max_rounds.get(route_id) == episode_length


class _Audit:
    def __init__(
        self,
        run_dir: Path,
        trainer_exit_code: int,
        *,
        require_trainer_log: bool = True,
    ) -> None:
        self.run_dir = run_dir
        self.trainer_exit_code = int(trainer_exit_code)
        self.require_trainer_log = require_trainer_log
        self.errors: list[str] = []
        self.mode: str | None = None
        self.role: str | None = None
        self.expected: Mapping[str, Any] | None = None
        self.launch: Mapping[str, Any] | None = None
        self.receipt_schema: str | None = None
        self.multitask = False
        self.route_ids: tuple[str, ...] = ()
        self.route_max_rounds: dict[str, int] = {}
        self.runtime_paths: dict[str, Path] = {}
        self.route_summaries: dict[str, Any] = {}
        self.schedule_routes: dict[tuple[str, int], str] = {}
        self.rolling_8: dict[str, Any] = {}
        self.accounting: dict[str, Any] = {}
        self.batch_multiple: int | None = None
        self.staleness_threshold: float | None = None
        self.trainer_world_size: int | None = None
        self.counts: dict[str, int] = {
            "scheduled_episodes": 0,
            "declared_learner_updates": 0,
            "declared_publication_cycles": 0,
            "complete_learner_updates": 0,
            "publication_cycles": 0,
            "real_action_rows": 0,
            "derived_padding_action_rows": 0,
            "real_response_tokens": 0,
            "stale_action_rows": 0,
            "policy_version_min": 0,
            "policy_version_max": 0,
            "validation_events": 0,
            "memory_chains": 0,
            "late_memory_chains": 0,
            "completed_episodes": 0,
        }

    def check(self, condition: bool, message: str) -> bool:
        if not condition:
            self.errors.append(message)
        return condition

    def error(self, label: str, exc: BaseException) -> None:
        self.errors.append(f"{label}: {exc}")

    def audit_launch(self, launch: Mapping[str, Any] | None = None) -> None:
        if launch is None:
            path = self.run_dir / "launch-receipt.json"
            try:
                launch = _load_json(path, "launch receipt")
            except Exception as exc:
                self.error("launch receipt", exc)
                return
        self.launch = launch
        schema = launch.get("schema")
        if schema not in {_LEGACY_RECEIPT_SCHEMA, _MULTITASK_RECEIPT_SCHEMA}:
            self.errors.append(f"launch receipt schema is unsupported: {schema!r}")
            return
        self.receipt_schema = str(schema)
        self.multitask = schema == _MULTITASK_RECEIPT_SCHEMA
        runtime_paths, runtime_errors = _bound_runtime_paths(
            launch,
            self.run_dir,
            require_trainer_log=self.multitask and self.require_trainer_log,
        )
        self.runtime_paths = runtime_paths
        self.errors.extend(runtime_errors)
        if not self.multitask:
            for field, relative_path in _LEGACY_RUNTIME_ARTIFACT_RELATIVE_PATHS.items():
                self.check(
                    runtime_paths.get(field) == (self.run_dir / relative_path).resolve(),
                    f"legacy runtime artifact {field} must equal "
                    f"{self.run_dir / relative_path}",
                )
        finalization_path = runtime_paths.get("finalization")
        protected_files, protected_directories = _receipt_protected_paths(
            launch, self.run_dir, include_finalization=False
        )
        self.check(
            finalization_path is None
            or not _path_overlaps_inputs(
                finalization_path, protected_files, protected_directories
            ),
            "launch finalization path overlaps a receipt-bound input artifact",
        )
        self.check(
            launch.get("schema") in {_LEGACY_RECEIPT_SCHEMA, _MULTITASK_RECEIPT_SCHEMA},
            "launch receipt schema is unsupported",
        )
        self.check(
            launch.get("entrypoint")
            == "verl.experimental.fully_async_policy.fully_async_main",
            "launch receipt entrypoint is not native veRL fully-async",
        )
        mode = _at(launch, "inputs.mode")
        if mode not in {"gate", "formal"}:
            self.errors.append(f"launch mode is unsupported: {mode!r}")
            return
        self.mode = str(mode)

        if self.multitask:
            self._audit_multitask_launch(launch)
            return

        budget_contract = launch.get("budget_contract")
        endpoint = launch.get("endpoint_publication")
        if not isinstance(budget_contract, Mapping):
            self.errors.append(
                "launch receipt has no publication-derived budget contract"
            )
            return
        if not isinstance(endpoint, Mapping):
            self.errors.append("launch receipt has no endpoint publication identity")
            return
        self.expected = budget_contract
        self.role = str(budget_contract.get("role", ""))
        expected_role = "gate_only" if self.mode == "gate" else "train_pool"
        self.check(self.role == expected_role, "launch budget role does not match mode")
        self.check(
            endpoint.get("schema") == "amg_openmle_publication_identity_v3",
            "endpoint publication identity schema mismatch",
        )
        self.check(
            endpoint.get("budget_contract") == budget_contract,
            "endpoint and launch budget contracts differ",
        )
        self.check(
            launch.get("budget", {}).get("schema") == "amg_verl_fully_async_budget_v2",
            "verified launch budget schema mismatch",
        )
        self.check(
            _same_path(_at(launch, "inputs.run_dir"), self.run_dir),
            "launch run_dir does not match the finalized directory",
        )
        episodes = _positive_int(budget_contract.get("episodes"))
        updates = _positive_int(budget_contract.get("optimizer_updates"))
        samples_per_update = _positive_int(budget_contract.get("samples_per_update"))
        publications = _positive_int(budget_contract.get("publication_cycles"))
        sync_step = _positive_int(budget_contract.get("trigger_parameter_sync_step"))
        if None in (episodes, updates, samples_per_update, publications, sync_step):
            self.errors.append("launch budget contains a non-positive integer")
        else:
            self.check(
                episodes == updates * samples_per_update,
                "launch episode budget is not optimizer_updates * samples_per_update",
            )
            self.check(
                updates == publications * sync_step,
                "launch optimizer budget is not publication_cycles * sync cadence",
            )
            self.counts.update(
                scheduled_episodes=episodes,
                declared_learner_updates=updates,
                declared_publication_cycles=publications,
            )

        model_path = str(_at(endpoint, "training_runtime.base_model", ""))
        self.check(
            _at(launch, "inputs.model_path") == model_path,
            "launch model path differs from the selected publication",
        )
        source_checks = (
            ("source.verl_commit", EXPECTED_VERL_COMMIT, "veRL identity"),
            (
                "source.publication_outer_commit",
                endpoint.get("publication_outer_commit"),
                "publication outer identity",
            ),
            (
                "source.agentgym_commit",
                endpoint.get("publication_inner_commit"),
                "AgentGym inner identity",
            ),
            (
                "source.agentgym_expected_commit",
                endpoint.get("publication_inner_commit"),
                "AgentGym gitlink identity",
            ),
            (
                "source.training_runtime",
                endpoint.get("training_runtime"),
                "training runtime identity",
            ),
            (
                "source.model_files_sha256",
                LOCKED_MODEL_FILE_SHA256,
                "model file identity",
            ),
        )
        for dotted, wanted, label in source_checks:
            self.check(_at(launch, dotted) == wanted, f"launch {label} mismatch")

        schedule_checks = (
            ("role", self.role),
            ("count", budget_contract.get("episodes")),
            ("sha256", budget_contract.get("schedule_sha256")),
            ("manifest_digest", budget_contract.get("manifest_sha256")),
        )
        for field, wanted in schedule_checks:
            self.check(
                _at(launch, f"schedule.{field}") == wanted,
                f"launch schedule {field} mismatch",
            )
        endpoint_checks = (
            ("manifest_role", self.role),
            ("manifest_sha256", budget_contract.get("manifest_sha256")),
            ("routing_sha256", budget_contract.get("routing_sha256")),
            ("schedule_count", budget_contract.get("episodes")),
            ("schedule_sha256", budget_contract.get("schedule_sha256")),
            ("task_count", budget_contract.get("task_count")),
            ("source_family_count", budget_contract.get("source_family_count")),
        )
        for field, wanted in endpoint_checks:
            self.check(
                endpoint.get(field) == wanted,
                f"endpoint publication {field} mismatch",
            )
        for path_field, digest_field in (
            ("source_lock_path", "source_lock_sha256"),
            ("contract_tool_path", "contract_tool_sha256"),
            ("publication_receipt_path", "publication_receipt_sha256"),
            ("schedule_certificate_path", "schedule_certificate_sha256"),
        ):
            artifact_path = endpoint.get(path_field)
            expected_digest = endpoint.get(digest_field)
            try:
                artifact = Path(str(artifact_path))
                self.check(
                    artifact.is_file()
                    and not artifact.is_symlink()
                    and sha256_file(artifact) == expected_digest,
                    f"endpoint publication artifact {path_field} drifted",
                )
            except (OSError, ValueError, TypeError) as exc:
                self.error(f"endpoint publication artifact {path_field}", exc)
        self.check(
            launch.get("validation_enabled") is False,
            "launch validation_enabled must be false",
        )

    def _audit_multitask_launch(self, launch: Mapping[str, Any]) -> None:
        budget_contract = launch.get("budget_contract")
        identity = launch.get("launch_identity")
        if not isinstance(budget_contract, Mapping):
            self.errors.append("multitask launch has no budget contract")
            return
        if not isinstance(identity, Mapping):
            self.errors.append("multitask launch has no source identity")
            return
        self.expected = budget_contract
        self.role = str(budget_contract.get("role", ""))
        expected_role = "gate_only" if self.mode == "gate" else "train_pool"
        self.check(self.role == expected_role, "launch budget role does not match mode")
        self.check(
            budget_contract.get("schema") == "amg_verl_multitask_budget_contract_v1",
            "multitask budget contract schema mismatch",
        )
        self.check(
            identity.get("schema") == "amg_multitask_source_identity_v1",
            "multitask source identity schema mismatch",
        )
        self.check(
            launch.get("endpoint_publication") is None,
            "multitask receipt must not carry a single endpoint publication",
        )
        self.check(
            identity.get("budget_contract") == budget_contract,
            "source identity and launch budget contracts differ",
        )
        self.check(
            isinstance(launch.get("budget"), Mapping)
            and launch["budget"].get("schema") == "amg_verl_fully_async_budget_v2",
            "verified launch budget schema mismatch",
        )
        self.check(
            _same_path(_at(launch, "inputs.run_dir"), self.run_dir),
            "launch run_dir does not match the finalized directory",
        )

        routes = _route_ids(identity.get("route_ids"))
        if routes is None:
            self.errors.append(
                "multitask source identity must declare one to four opaque routes"
            )
            return
        self.route_ids = routes
        self.check(
            _route_ids(budget_contract.get("route_ids")) == routes,
            "multitask budget route IDs differ from source identity",
        )
        self.check(
            _route_ids(_at(launch, "budget.route_ids")) == routes,
            "verified budget route IDs differ from source identity",
        )

        episodes = _positive_int(budget_contract.get("episodes"))
        updates = _positive_int(budget_contract.get("optimizer_updates"))
        samples_per_update = _positive_int(budget_contract.get("samples_per_update"))
        publications = _positive_int(budget_contract.get("publication_cycles"))
        sync_step = _positive_int(budget_contract.get("trigger_parameter_sync_step"))
        if None in (episodes, updates, samples_per_update, publications, sync_step):
            self.errors.append("launch budget contains a non-positive integer")
        else:
            self.check(
                episodes == updates * samples_per_update,
                "launch episode budget is not optimizer_updates * samples_per_update",
            )
            self.check(
                updates == publications * sync_step,
                "launch optimizer budget is not publication_cycles * sync cadence",
            )
            self.counts.update(
                scheduled_episodes=episodes,
                declared_learner_updates=updates,
                declared_publication_cycles=publications,
            )

        registry_digest = identity.get("route_registry_sha256")
        self.check(_sha256_text(registry_digest), "route registry digest is invalid")
        self.check(
            budget_contract.get("route_registry_sha256") == registry_digest,
            "multitask budget registry digest mismatch",
        )
        self.check(
            _at(launch, "inputs.route_registry_sha256") == registry_digest,
            "launch input registry digest mismatch",
        )
        registry_path = identity.get("route_registry_path")
        self.check(
            _same_path(_at(launch, "inputs.route_registry"), Path(str(registry_path))),
            "launch input registry path mismatch",
        )
        try:
            registry_file = Path(str(registry_path))
            registry = load_route_registry(
                registry_file,
                expected_sha256=str(registry_digest),
                expected_route_ids=routes,
            )
            self.check(
                registry.route_ids == routes,
                "route registry order differs from source identity",
            )
            self.route_max_rounds = {
                route.route_id: route.max_rounds for route in registry.routes
            }
        except Exception as exc:
            self.error("route registry", exc)

        identity_artifacts: dict[str, Mapping[str, Any]] = {}
        for label, path_field, digest_field in (
            ("source lock", "source_lock_path", "source_lock_sha256"),
            (
                "schedule certificate",
                "schedule_certificate_path",
                "schedule_certificate_sha256",
            ),
        ):
            artifact_path = identity.get(path_field)
            expected_digest = identity.get(digest_field)
            try:
                artifact = Path(str(artifact_path))
                digest_matches = (
                    _sha256_text(expected_digest)
                    and artifact.is_absolute()
                    and artifact.is_file()
                    and not artifact.is_symlink()
                    and sha256_file(artifact) == expected_digest
                )
                self.check(
                    digest_matches,
                    f"multitask {label} digest/path drifted",
                )
                if digest_matches:
                    identity_artifacts[label] = _load_json(artifact, label)
            except (OSError, ValueError, TypeError) as exc:
                self.error(f"multitask {label}", exc)

        outer = identity.get("publication_outer_commit")
        inner = identity.get("publication_inner_commit")
        self.check(_git_revision(outer), "multitask outer identity is invalid")
        self.check(_git_revision(inner), "multitask inner identity is invalid")
        source_checks = (
            ("source.verl_commit", _FINAL_STATISTICS_VERL_COMMIT, "veRL identity"),
            ("source.publication_outer_commit", outer, "publication outer identity"),
            ("source.outer_commit", outer, "runtime outer identity"),
            ("source.agentgym_commit", inner, "runtime inner identity"),
            ("source.agentgym_expected_commit", inner, "inner gitlink identity"),
            (
                "source.training_runtime",
                identity.get("training_runtime"),
                "training runtime identity",
            ),
            (
                "source.model_files_sha256",
                LOCKED_MODEL_FILE_SHA256,
                "model file identity",
            ),
        )
        for dotted, wanted, label in source_checks:
            self.check(_at(launch, dotted) == wanted, f"launch {label} mismatch")
        self.check(
            identity.get("verl_commit") == _FINAL_STATISTICS_VERL_COMMIT,
            "multitask source identity did not select the reviewed FinalStatistics commit",
        )
        self.check(
            _at(launch, "source.outer_diff_paths") == [],
            "multitask runtime outer identity is not exact",
        )
        for field in ("verl_clean", "outer_clean", "agentgym_clean"):
            self.check(
                _at(launch, f"source.{field}") is True,
                f"multitask runtime source {field} is not true",
            )
        self.check(
            _at(launch, "inputs.model_path")
            == _at(identity, "training_runtime.base_model"),
            "launch model path differs from multitask source identity",
        )

        schedule_checks = (
            ("role", self.role),
            ("count", budget_contract.get("episodes")),
            ("sha256", budget_contract.get("schedule_sha256")),
            ("manifest_digest", budget_contract.get("manifest_sha256")),
            ("route_registry_sha256", registry_digest),
            ("route_order", list(routes)),
        )
        for field, wanted in schedule_checks:
            self.check(
                _at(launch, f"schedule.{field}") == wanted,
                f"launch schedule {field} mismatch",
            )
        self.check(
            identity.get("schedule_count") == budget_contract.get("episodes")
            and identity.get("schedule_sha256")
            == budget_contract.get("schedule_sha256"),
            "multitask identity schedule budget mismatch",
        )
        self._audit_multitask_identity_artifacts(
            identity=identity,
            budget_contract=budget_contract,
            routes=routes,
            source_lock=identity_artifacts.get("source lock"),
            certificate=identity_artifacts.get("schedule certificate"),
        )
        self.check(
            launch.get("validation_enabled") is False,
            "launch validation_enabled must be false",
        )

    def _audit_multitask_identity_artifacts(
        self,
        *,
        identity: Mapping[str, Any],
        budget_contract: Mapping[str, Any],
        routes: Sequence[str],
        source_lock: Mapping[str, Any] | None,
        certificate: Mapping[str, Any] | None,
    ) -> None:
        """Cross-bind immutable launcher artifacts to the receipt content."""

        if source_lock is None or certificate is None:
            return

        self.check(
            source_lock.get("schema") == _MULTITASK_SOURCE_LOCK_SCHEMA
            and source_lock.get("status") == "pass",
            "multitask source lock is not a completed v1 lock",
        )
        runtime_source = source_lock.get("runtime_source")
        integration = source_lock.get("integration")
        if not isinstance(runtime_source, Mapping):
            self.errors.append("multitask source lock omitted runtime_source")
            runtime_source = {}
        if not isinstance(integration, Mapping):
            self.errors.append("multitask source lock omitted integration bindings")
            integration = {}

        source_bindings = (
            ("outer_commit", identity.get("publication_outer_commit")),
            ("inner_commit", identity.get("publication_inner_commit")),
            ("verl_commit", identity.get("verl_commit")),
            ("selected_files", identity.get("selected_files")),
        )
        for field, expected in source_bindings:
            self.check(
                runtime_source.get(field) == expected,
                f"multitask source lock {field} differs from launch identity",
            )
        selected_files = runtime_source.get("selected_files")
        selected_files_valid = (
            isinstance(selected_files, Mapping)
            and bool(selected_files)
            and all(
                isinstance(path, str)
                and path.startswith(("outer:", "inner:"))
                and bool(path.removeprefix("outer:").removeprefix("inner:"))
                and _sha256_text(digest)
                for path, digest in selected_files.items()
            )
            and any(str(path).startswith("outer:") for path in selected_files)
            and any(str(path).startswith("inner:") for path in selected_files)
        )
        self.check(
            selected_files_valid,
            "multitask source lock selected_files are invalid",
        )
        try:
            training_runtime = validate_training_runtime_lock(
                source_lock.get("training_runtime")
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            self.error("multitask source lock training runtime", exc)
            training_runtime = None
        self.check(
            training_runtime is not None
            and training_runtime == identity.get("training_runtime"),
            "multitask source lock training runtime differs from launch identity",
        )

        registry_binding = integration.get("route_registry")
        certificate_binding = integration.get("schedule_certificate")
        self.check(
            isinstance(registry_binding, Mapping)
            and registry_binding.get("sha256") == identity.get("route_registry_sha256")
            and _route_ids(registry_binding.get("route_ids")) == tuple(routes),
            "multitask source lock route registry binding differs from launch identity",
        )
        self.check(
            isinstance(certificate_binding, Mapping)
            and certificate_binding.get("sha256")
            == identity.get("schedule_certificate_sha256")
            and certificate_binding.get("schedule_sha256")
            == budget_contract.get("schedule_sha256"),
            "multitask source lock schedule certificate binding differs from launch identity",
        )

        self.check(
            certificate.get("schema") == _MULTITASK_SCHEDULE_CERTIFICATE_SCHEMA,
            "multitask schedule certificate schema mismatch",
        )
        certificate_checks = (
            ("route_registry_sha256", identity.get("route_registry_sha256")),
            ("schedule_sha256", budget_contract.get("schedule_sha256")),
            ("spec_sha256", budget_contract.get("manifest_sha256")),
            ("role", budget_contract.get("role")),
            ("optimizer_updates", budget_contract.get("optimizer_updates")),
            ("samples_per_update", budget_contract.get("samples_per_update")),
            ("row_count", budget_contract.get("episodes")),
            ("route_order", list(routes)),
            ("panel_id", _at(self.launch, "schedule.panel_id")),
            ("agent_name", _at(self.launch, "schedule.agent_name")),
        )
        for field, expected in certificate_checks:
            self.check(
                certificate.get(field) == expected,
                f"multitask schedule certificate {field} differs from launch budget",
            )

        try:
            per_route_rows = _exact_counter(
                certificate.get("per_route_rows"),
                label="multitask schedule certificate per_route_rows",
            )
        except (TypeError, ValueError) as exc:
            self.error("multitask schedule certificate", exc)
            per_route_rows = {}
        expected_route_rows = _at(self.launch, "schedule.per_route_counts")
        self.check(
            set(per_route_rows) == set(routes)
            and sum(per_route_rows.values()) == budget_contract.get("episodes")
            and per_route_rows == expected_route_rows,
            "multitask schedule certificate per-route rows differ from launch schedule",
        )
        self.check(
            identity.get("formal_schedule_contract") == certificate,
            "multitask launch identity formal schedule contract differs from certificate",
        )

    def audit_config(self) -> None:
        if self.launch is None or self.expected is None or self.mode is None:
            return
        resolved_value = _at(self.launch, "resolved_config.path")
        resolved_path = _path_within(self.run_dir, resolved_value)
        hydra_path = self.runtime_paths.get("hydra_config")
        if resolved_path is None or hydra_path is None:
            self.errors.append("resolved/Hydra config paths are not receipt-bound")
            return
        try:
            resolved = _load_yaml(resolved_path, "resolved config")
            hydra = _load_resolved_hydra_yaml(hydra_path, "Hydra config")
        except Exception as exc:
            self.error("resolved/Hydra config", exc)
            return

        self.check(
            _same_path(_at(self.launch, "resolved_config.path"), resolved_path),
            "launch resolved config path mismatch",
        )
        self.check(
            _at(self.launch, "resolved_config.sha256") == sha256_file(resolved_path),
            "launch resolved config sha256 mismatch",
        )
        self.check(resolved == hydra, "Hydra config drifted from the preflight config")
        try:
            budget = verify_resolved_config(
                resolved,
                mode=self.mode,
                expected_budget=self.expected,
            )
        except Exception as exc:
            self.error("resolved config contract", exc)
            budget = None
        if budget is not None:
            self.check(
                self.launch.get("budget") == budget,
                "launch budget does not match the verified resolved config",
            )
            trainer_world_size = _positive_int(budget.get("trainer_gpus"))
            if trainer_world_size is None:
                self.errors.append("verified trainer world size is invalid")
            else:
                self.trainer_world_size = trainer_world_size
        raw_staleness = _at(resolved, "async_training.staleness_threshold")
        if (
            isinstance(raw_staleness, bool)
            or not isinstance(raw_staleness, (int, float))
            or not math.isfinite(float(raw_staleness))
            or float(raw_staleness) < 0.0
        ):
            self.errors.append("resolved staleness_threshold is invalid")
        else:
            self.staleness_threshold = float(raw_staleness)
        self.check(
            _at(resolved, "trainer.logger") == ["console", "file"],
            "resolved config trainer.logger must include console and FileLogger",
        )
        self.check(
            _at(resolved, "trainer.validation_data_dir") is None,
            "resolved config validation_data_dir must be null",
        )
        resolved_endpoint = _at(resolved, "actor_rollout_ref.agentgym")
        if self.multitask:
            identity = self.launch.get("launch_identity")
            self.check(
                isinstance(resolved_endpoint, Mapping)
                and isinstance(identity, Mapping)
                and _same_path(
                    resolved_endpoint.get("route_registry_path"),
                    Path(str(identity.get("route_registry_path"))),
                )
                and resolved_endpoint.get("route_registry_sha256")
                == identity.get("route_registry_sha256")
                and tuple(resolved_endpoint.get("route_registry_expected_ids", ()))
                == self.route_ids,
                "resolved config route registry differs from launch identity",
            )
        else:
            launch_endpoint = _at(self.launch, "endpoint_publication.client_config")
            endpoint_fields = (
                "expected_manifest_sha256",
                "expected_release_revision",
                "expected_outer_commit",
                "expected_inner_commit",
                "expected_role",
                "expected_executor_runtime_digest",
                "expected_materializer_sha256",
                "expected_actions_sha256",
                "expected_max_observation_tokens",
            )
            self.check(
                isinstance(resolved_endpoint, Mapping)
                and isinstance(launch_endpoint, Mapping)
                and {field: resolved_endpoint.get(field) for field in endpoint_fields}
                == {field: launch_endpoint.get(field) for field in endpoint_fields},
                "resolved config endpoint identity differs from the publication",
            )

        train_files = _at(resolved, "data.train_files")
        if isinstance(train_files, str):
            train_paths = [train_files]
        elif isinstance(train_files, Sequence):
            train_paths = [str(path) for path in train_files]
        else:
            train_paths = []
        schedule_value = _at(self.launch, "schedule.path")
        self.check(
            len(train_paths) == 1
            and isinstance(schedule_value, str)
            and _same_path(train_paths[0], Path(schedule_value)),
            "resolved config train_files do not select the launch schedule",
        )
        if len(train_paths) == 1:
            schedule_path = Path(train_paths[0])
            try:
                schedule_report = inspect_schedule(
                    schedule_path,
                    expected_count=int(self.expected["episodes"]),
                    expected_sha256=str(self.expected["schedule_sha256"]),
                    expected_role=str(self.expected["role"]),
                    expected_route_ids=(self.route_ids if self.multitask else None),
                    expected_route_registry_sha256=(
                        str(self.expected["route_registry_sha256"])
                        if self.multitask
                        else None
                    ),
                )
                if self.multitask:
                    self.check(
                        schedule_report == self.launch.get("schedule"),
                        "launch schedule report differs from the frozen schedule",
                    )
                schedule_rows = [
                    json.loads(line)
                    for line in schedule_path.read_text(encoding="utf-8").splitlines()
                ]
                self._schedule_ids = [str(row["item_id"]) for row in schedule_rows]
                self._schedule_instances = [
                    (str(row["item_id"]), int(row["data_idx"])) for row in schedule_rows
                ]
                if self.multitask:
                    self.schedule_routes = {
                        (str(row["item_id"]), int(row["data_idx"])): str(
                            row.get("route_id")
                            or row.get("extra_info", {}).get("route_id")
                        )
                        for row in schedule_rows
                    }
            except Exception as exc:
                self.error("publication schedule", exc)

        actor_batch = _positive_int(
            _at(resolved, "actor_rollout_ref.actor.ppo_mini_batch_size")
        )
        critic_batch = _positive_int(_at(resolved, "critic.ppo_mini_batch_size"))
        if actor_batch is None or critic_batch is None:
            self.errors.append("resolved actor/critic mini-batch is not positive")
        else:
            self.batch_multiple = _lcm(actor_batch, critic_batch)

    def audit_file_logger(self) -> None:
        """Audit metrics emitted by current upstream veRL itself.

        The previous integration added a private runtime receipt and sampled
        parameter probes.  Current upstream has no consumers for those config
        keys, so this audit instead joins the native FileLogger signals that are
        produced by the actual learner/rollouter path: actor and critic gradient
        norms, rollout-correction diagnostics, policy/staleness versions, queue
        counters, rollout JSONL, and the complete optimizer checkpoint.
        """

        path = self.runtime_paths.get("file_logger")
        if path is None:
            self.errors.append("FileLogger path is not receipt-bound")
            return
        try:
            rows = _jsonl(path, "FileLogger JSONL")
        except Exception as exc:
            self.error("FileLogger JSONL", exc)
            return

        validation_metrics = 0
        actor_grad_rows = 0
        critic_grad_rows = 0
        rollout_correction_rows = 0
        current_param_versions: dict[int, int] = {}
        native_stale_action_rows: dict[int, int] = {}
        generated_samples: dict[int, int] = {}
        dropped_samples: dict[int, int] = {}
        queue_sizes: dict[int, int] = {}
        observed_steps: list[int] = []
        rows_by_step: dict[int, list[Mapping[str, Any]]] = {}

        for index, row in enumerate(rows):
            raw_step = row.get("step")
            valid_step = isinstance(raw_step, int) and not isinstance(raw_step, bool)
            self.check(valid_step, f"FileLogger row {index} has no integer step")
            data = row.get("data")
            if not isinstance(data, Mapping):
                self.errors.append(f"FileLogger row {index} has no data mapping")
                continue
            if valid_step:
                step = int(raw_step)
                observed_steps.append(step)
                rows_by_step.setdefault(step, []).append(data)
            for key in data:
                folded = str(key).casefold()
                if "validation" in folded or folded.startswith(
                    ("val/", "val_", "val-")
                ):
                    validation_metrics += 1

        self.check(
            validation_metrics == 0,
            f"FileLogger emitted {validation_metrics} validation metric(s)",
        )
        self.counts["publication_cycles"] = _contiguous_positive_prefix(
            observed_steps
        )
        if self.expected is not None:
            publications = int(self.expected["publication_cycles"])
            samples_per_update = int(self.expected["samples_per_update"])

            step_zero_bad_keys = sorted(
                str(key)
                for data in rows_by_step.get(0, [])
                for key in data
                if not (
                    str(key).startswith("fully_async/rollouter/")
                    or str(key) == "dynamic_resource/rollout_resource_utilization"
                )
            )
            self.check(
                not step_zero_bad_keys,
                "FileLogger step 0 is not rollouter-only: "
                + ", ".join(step_zero_bad_keys),
            )
            out_of_range_steps = sorted(
                step for step in rows_by_step if step < 0 or step > publications
            )
            self.check(
                not out_of_range_steps,
                f"FileLogger has out-of-range publication steps: {out_of_range_steps}",
            )
            positive_steps = [step for step in observed_steps if step > 0]
            self.check(
                positive_steps == sorted(positive_steps),
                "FileLogger publication rows are out of order",
            )
            self.check(
                set(positive_steps) == set(range(1, publications + 1)),
                "FileLogger publication steps are incomplete",
            )

            required_rollout_corr = (
                "rollout_corr/kl",
                "rollout_corr/k3_kl",
                "rollout_corr/log_ppl_abs_diff",
            )
            for step in range(1, publications + 1):
                data_rows = rows_by_step.get(step, [])
                items = [
                    (str(key), value)
                    for data in data_rows
                    for key, value in data.items()
                ]

                for role in ("actor", "critic"):
                    grad_norms = [
                        value for key, value in items if key == f"{role}/grad_norm"
                    ]
                    valid_grad = len(grad_norms) == 1 and _finite_positive(
                        grad_norms[0]
                    )
                    self.check(
                        valid_grad,
                        f"FileLogger publication step {step} has no unique nonzero {role}/grad_norm",
                    )
                    if valid_grad:
                        if role == "actor":
                            actor_grad_rows += 1
                        else:
                            critic_grad_rows += 1

                correction_values: dict[str, float] = {}
                for key in required_rollout_corr:
                    values = [value for item_key, value in items if item_key == key]
                    valid = len(values) == 1 and _finite_number(values[0])
                    self.check(
                        valid,
                        f"FileLogger publication step {step} has no unique finite {key}",
                    )
                    if valid:
                        correction_values[key] = float(values[0])
                if len(correction_values) == len(required_rollout_corr):
                    rollout_correction_rows += 1

                integral_metrics = {
                    "current_param_version": "fully_async/count/current_param_version",
                    "stale_trajectory_processed": "fully_async/count/stale_trajectory_processed",
                    "total_generated_samples": "fully_async/count/total_generated_samples",
                    "dropped_stale_samples": "fully_async/count/dropped_stale_samples",
                    "mq_queue_size": "fully_async/monitor/queue/mq_queue_size",
                    "required_samples": "fully_async/static/required_samples",
                }
                observed_integrals: dict[str, int] = {}
                for label, key in integral_metrics.items():
                    values = [value for item_key, value in items if item_key == key]
                    parsed = (
                        _nonnegative_integral(values[0]) if len(values) == 1 else None
                    )
                    self.check(
                        parsed is not None,
                        f"FileLogger publication step {step} has no unique integral {key}",
                    )
                    if parsed is not None:
                        observed_integrals[label] = parsed

                if observed_integrals.get("required_samples") is not None:
                    self.check(
                        observed_integrals["required_samples"] == samples_per_update,
                        f"FileLogger publication step {step} native required_samples mismatch",
                    )
                if "current_param_version" in observed_integrals:
                    current_param_versions[step] = observed_integrals[
                        "current_param_version"
                    ]
                if "stale_trajectory_processed" in observed_integrals:
                    native_stale_action_rows[step] = observed_integrals[
                        "stale_trajectory_processed"
                    ]
                if "total_generated_samples" in observed_integrals:
                    generated_samples[step] = observed_integrals[
                        "total_generated_samples"
                    ]
                if "dropped_stale_samples" in observed_integrals:
                    dropped_samples[step] = observed_integrals["dropped_stale_samples"]
                if "mq_queue_size" in observed_integrals:
                    queue_sizes[step] = observed_integrals["mq_queue_size"]

            self.check(
                actor_grad_rows == publications,
                "FileLogger actor/grad_norm does not cover every publication cycle",
            )
            self.check(
                critic_grad_rows == publications,
                "FileLogger critic/grad_norm does not cover every publication cycle",
            )
            self.check(
                rollout_correction_rows == publications,
                "FileLogger native rollout-correction diagnostics do not cover every publication cycle",
            )
            if len(current_param_versions) == publications:
                self.check(
                    [
                        current_param_versions[step]
                        for step in range(1, publications + 1)
                    ]
                    == list(range(publications)),
                    "FileLogger current parameter versions do not match publication order",
                )
            for label, values in (
                ("native stale action-row count", native_stale_action_rows),
                ("native total-generated-samples count", generated_samples),
                ("native dropped-stale-samples count", dropped_samples),
            ):
                if len(values) == publications:
                    sequence = [values[step] for step in range(1, publications + 1)]
                    self.check(
                        sequence == sorted(sequence),
                        f"FileLogger {label} is not cumulative",
                    )

        self.counts["validation_events"] += validation_metrics
        current_param_versions_by_update: dict[int, int] = {}
        if self.expected is not None:
            trigger = int(self.expected["trigger_parameter_sync_step"])
            for publication, version in current_param_versions.items():
                first_update = (publication - 1) * trigger + 1
                for update in range(first_update, first_update + trigger):
                    current_param_versions_by_update[update] = version
        self._file_logger_summary = {
            "rows": len(rows),
            "actor_grad_rows": actor_grad_rows,
            "critic_grad_rows": critic_grad_rows,
            "rollout_correction_rows": rollout_correction_rows,
            "current_param_versions_by_update": current_param_versions_by_update,
            "native_stale_action_rows_by_publication": native_stale_action_rows,
            "native_total_generated_samples_by_publication": generated_samples,
            "native_dropped_stale_samples_by_publication": dropped_samples,
            "native_mq_queue_size_by_publication": queue_sizes,
            "rows_by_step": rows_by_step,
        }

    @staticmethod
    def _has_memory_chain(
        episode: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ) -> bool:
        return _has_complete_memory_chain(episode)

    def _audit_optimizer_file_logger_update(
        self,
        update: int,
        *,
        episodes: Mapping[str, int],
        action_rows: Mapping[str, int],
        response_tokens: Mapping[str, int],
        stale_rows: Mapping[str, int],
    ) -> None:
        summary = getattr(self, "_file_logger_summary", {})
        rows_by_step = summary.get("rows_by_step", {})
        data_rows = (
            rows_by_step.get(update, []) if isinstance(rows_by_step, Mapping) else []
        )
        items = [(str(key), value) for row in data_rows for key, value in row.items()]
        cumulative = getattr(
            self,
            "_optimizer_logger_cumulative",
            {
                "episodes": Counter(),
                "action_rows": Counter(),
                "policy_response_tokens": Counter(),
                "stale_action_rows": Counter(),
            },
        )
        specifications = (
            ("episodes", episodes),
            ("action_rows", action_rows),
            ("policy_response_tokens", response_tokens),
        )
        for measure, expected in specifications:
            per_update_prefix = (
                f"fully_async/sum/optimizer_consumed_{measure}/data_source/"
            )
            cumulative_prefix = (
                f"fully_async/count/optimizer_consumed_{measure}/data_source/"
            )
            try:
                observed_update = _metric_route_counter(
                    items,
                    prefix=per_update_prefix,
                    label=(
                        f"FileLogger update {update} optimizer-consumed {measure}"
                    ),
                )
            except ValueError as exc:
                self.error("FileLogger route metric", exc)
                observed_update = {}
            try:
                observed_cumulative = _metric_route_counter(
                    items,
                    prefix=cumulative_prefix,
                    label=(
                        f"FileLogger update {update} cumulative "
                        f"optimizer-consumed {measure}"
                    ),
                )
            except ValueError as exc:
                self.error("FileLogger route metric", exc)
                observed_cumulative = {}
            self.check(
                set(observed_update).issubset(self.route_ids)
                and _normalized_counter(observed_update, self.route_ids)
                == dict(expected),
                f"FileLogger update {update} optimizer-consumed {measure} route totals mismatch",
            )
            global_key = f"fully_async/sum/optimizer_consumed_{measure}"
            global_values = [value for key, value in items if key == global_key]
            self.check(
                len(global_values) == 1
                and _nonnegative_integral(global_values[0]) == sum(expected.values()),
                f"FileLogger update {update} optimizer-consumed {measure} global total mismatch",
            )
            cumulative[measure].update(expected)
            self.check(
                set(observed_cumulative).issubset(self.route_ids)
                and _normalized_counter(observed_cumulative, self.route_ids)
                == _normalized_counter(cumulative[measure], self.route_ids),
                f"FileLogger update {update} cumulative optimizer-consumed {measure} route totals mismatch",
            )
            cumulative_key = f"fully_async/count/optimizer_consumed_{measure}"
            cumulative_values = [value for key, value in items if key == cumulative_key]
            self.check(
                len(cumulative_values) == 1
                and _nonnegative_integral(cumulative_values[0])
                == sum(cumulative[measure].values()),
                f"FileLogger update {update} cumulative optimizer-consumed {measure} global total mismatch",
            )

        cumulative["stale_action_rows"].update(stale_rows)
        stale_prefix = "fully_async/count/stale_action_rows/data_source/"
        try:
            observed_stale = _metric_route_counter(
                items,
                prefix=stale_prefix,
                label=f"FileLogger update {update} cumulative stale rows",
            )
        except ValueError as exc:
            self.error("FileLogger route metric", exc)
            observed_stale = {}
        self.check(
            set(observed_stale).issubset(self.route_ids)
            and _normalized_counter(observed_stale, self.route_ids)
            == _normalized_counter(cumulative["stale_action_rows"], self.route_ids),
            f"FileLogger update {update} cumulative stale-row route totals mismatch",
        )
        self._optimizer_logger_cumulative = cumulative

    def audit_rollouts(self) -> None:
        directory = self.runtime_paths.get("rollout_data")
        if directory is None:
            self.errors.append("rollout data path is not receipt-bound")
            return
        paths: list[Path] = []
        if directory.is_dir():
            candidates = list(directory.glob("*.jsonl"))
            non_numeric = sorted(
                path.name for path in candidates if not path.stem.isdecimal()
            )
            if non_numeric:
                self.errors.append(
                    "rollout JSONL filenames must be numeric optimizer steps: "
                    + ", ".join(non_numeric)
                )
            paths = sorted(
                (path for path in candidates if path.stem.isdecimal()),
                key=lambda path: int(path.stem),
            )
        self.counts["complete_learner_updates"] = _contiguous_positive_prefix(
            int(path.stem) for path in paths
        )
        if not paths:
            self.errors.append(f"required rollout JSONL is missing under: {directory}")
            return
        if self.expected is not None:
            self.check(
                len(paths) == int(self.expected["optimizer_updates"]),
                "rollout JSONL file count does not match optimizer-update horizon",
            )
            expected_names = [
                f"{step}.jsonl"
                for step in range(1, int(self.expected["optimizer_updates"]) + 1)
            ]
            self.check(
                [path.name for path in paths] == expected_names,
                "rollout JSONL filename is not bound to its optimizer step",
            )

        real_rows = 0
        derived_padding_rows = 0
        real_tokens = 0
        stale_action_rows = 0
        stale_action_rows_by_publication: dict[int, int] = {}
        staleness_diffs: Counter[int] = Counter()
        version_pairs: Counter[str] = Counter()
        versions: set[int] = set()
        terminal_ids: list[str] = []
        terminal_instances: list[tuple[str, int]] = []
        memory_chain_updates: list[int] = []
        per_update_episodes: list[dict[str, int]] = []
        per_update_action_rows: list[dict[str, int]] = []
        per_update_response_tokens: list[dict[str, int]] = []
        route_action_rows: Counter[str] = Counter()
        route_response_tokens: Counter[str] = Counter()
        route_stale_rows: Counter[str] = Counter()
        route_episodes: Counter[str] = Counter()
        route_reward: Counter[str] = Counter()
        route_successes: Counter[str] = Counter()
        route_memory_events: dict[str, Counter[str]] = {
            route_id: Counter() for route_id in self.route_ids
        }
        route_memory_chains: Counter[str] = Counter()
        seen_uids: set[str] = set()
        samples_per_update = (
            int(self.expected["samples_per_update"]) if self.expected else 0
        )
        trigger = (
            int(self.expected["trigger_parameter_sync_step"]) if self.expected else 0
        )
        file_logger_summary = getattr(self, "_file_logger_summary", None)
        current_versions_by_update = (
            file_logger_summary.get("current_param_versions_by_update", {})
            if isinstance(file_logger_summary, Mapping)
            else {}
        )

        for ordinal, path in enumerate(paths, start=1):
            try:
                documents = _jsonl(path, "rollout JSONL")
            except Exception as exc:
                self.error("rollout JSONL", exc)
                continue
            file_step_values = {document.get("step") for document in documents}
            self.check(
                file_step_values == {ordinal},
                f"rollout JSONL {path.name} has unexpected optimizer step(s)",
            )
            current_version = current_versions_by_update.get(ordinal)
            if current_version is None:
                self.errors.append(
                    f"rollout update {ordinal} has no FileLogger current parameter version"
                )
            episodes_by_uid: dict[
                str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]
            ] = {}
            completed_episodes: list[
                list[tuple[Mapping[str, Any], Mapping[str, Any]]]
            ] = []
            real_rows_in_file = 0
            update_action_rows: Counter[str] = Counter()
            update_response_tokens: Counter[str] = Counter()
            update_stale_rows: Counter[str] = Counter()
            update_episodes: Counter[str] = Counter()
            for document in documents:
                is_padding = document.get("is_padding")
                if is_padding is not False:
                    if is_padding is True:
                        self.errors.append(
                            "native rollout JSONL contains a synthetic padding row"
                        )
                    else:
                        self.errors.append(
                            "rollout JSONL row must explicitly declare is_padding=false"
                        )
                    continue
                raw_record = document.get("step_record_json")
                if not isinstance(raw_record, str):
                    self.errors.append("rollout JSONL row is missing step_record_json")
                    continue
                try:
                    record = json.loads(raw_record)
                except json.JSONDecodeError:
                    self.errors.append("rollout JSONL step_record_json is invalid JSON")
                    continue
                if not isinstance(record, Mapping):
                    self.errors.append("rollout step record is not an object")
                    continue

                action = record.get("action")
                self.check(
                    isinstance(action, str)
                    and bool(action)
                    and _at(record, "action_submission.raw_policy_output") == action
                    and document.get("output") == action,
                    "rollout action is not bound to raw policy output",
                )
                route_id = record.get("route_id") or record.get("data_source")
                if self.multitask:
                    valid_route = (
                        isinstance(route_id, str)
                        and route_id in self.route_ids
                        and record.get("route_id") == route_id
                        and record.get("data_source") == route_id
                    )
                    self.check(
                        valid_route,
                        "rollout row has an unknown or inconsistent route label",
                    )
                    if not valid_route:
                        route_id = ""
                else:
                    route_id = ""

                real_rows += 1
                real_rows_in_file += 1
                token_count = record.get("response_token_count")
                if (
                    not isinstance(token_count, int)
                    or isinstance(token_count, bool)
                    or token_count <= 0
                ):
                    self.errors.append(
                        "real rollout row has invalid response_token_count"
                    )
                    token_count = 0
                real_tokens += int(token_count)
                if route_id:
                    route_action_rows[route_id] += 1
                    route_response_tokens[route_id] += int(token_count)
                    update_action_rows[route_id] += 1
                    update_response_tokens[route_id] += int(token_count)
                minimum = record.get("min_global_steps")
                maximum = record.get("max_global_steps")
                if (
                    not isinstance(minimum, int)
                    or isinstance(minimum, bool)
                    or not isinstance(maximum, int)
                    or isinstance(maximum, bool)
                    or minimum < 0
                    or maximum < minimum
                ):
                    self.errors.append("rollout policy-version fields are invalid")
                else:
                    versions.update((minimum, maximum))
                    version_pairs[f"{minimum}:{maximum}"] += 1
                    if current_version is not None:
                        staleness = current_version - maximum
                        self.check(
                            staleness >= 0,
                            f"rollout update {ordinal} contains a future policy version",
                        )
                        if staleness >= 0:
                            staleness_diffs[staleness] += 1
                            stale_action_rows += int(staleness >= 1)
                            if route_id and staleness >= 1:
                                route_stale_rows[route_id] += 1
                                update_stale_rows[route_id] += 1

                uid = record.get("trajectory_uid")
                item_id = record.get("item_id")
                if not isinstance(uid, str) or not uid:
                    self.errors.append("real rollout row has no trajectory_uid")
                    continue
                if uid in seen_uids:
                    if uid not in episodes_by_uid:
                        self.errors.append(
                            f"trajectory {uid!r} appears in multiple updates"
                        )
                        episodes_by_uid[uid] = []
                    continue
                if not isinstance(item_id, str) or not item_id:
                    self.errors.append("real rollout row has no item_id")
                    continue
                episodes_by_uid.setdefault(uid, []).append((record, document))

            for uid, episode in episodes_by_uid.items():
                if not episode:
                    continue
                item_ids = {record.get("item_id") for record, _document in episode}
                if len(item_ids) != 1:
                    self.errors.append(
                        f"trajectory identity changed item_id within {uid!r}"
                    )
                data_indices = {record.get("data_idx") for record, _document in episode}
                valid_data_idx = len(data_indices) == 1 and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in data_indices
                )
                if not valid_data_idx:
                    self.errors.append(
                        f"trajectory identity changed or has invalid data_idx within {uid!r}"
                    )
                orders = [record.get("trajectory_row_order") for record, _ in episode]
                valid_orders = all(
                    isinstance(order, int) and not isinstance(order, bool)
                    for order in orders
                )
                sorted_episode = sorted(
                    episode,
                    key=lambda pair: (
                        pair[0].get("trajectory_row_order")
                        if isinstance(pair[0].get("trajectory_row_order"), int)
                        and not isinstance(pair[0].get("trajectory_row_order"), bool)
                        else math.inf
                    ),
                )
                sorted_orders = [
                    record.get("trajectory_row_order")
                    for record, _document in sorted_episode
                ]
                if not valid_orders or sorted_orders != list(range(len(episode))):
                    self.errors.append(
                        f"trajectory {uid!r} action rows are not contiguous: {sorted_orders!r}"
                    )

                terminal_indices: list[int] = []
                for index, (record, _document) in enumerate(sorted_episode):
                    terminal = record.get("trajectory_terminal")
                    done = record.get("rollout_done_flag")
                    if terminal is True:
                        terminal_indices.append(index)
                    elif terminal is not False or done is not False:
                        self.errors.append(
                            f"trajectory {uid!r} nonterminal row has invalid terminal/done flags"
                        )
                if len(terminal_indices) != 1:
                    self.errors.append(
                        f"trajectory {uid!r} has {len(terminal_indices)} terminal rows in {path.name}"
                    )
                elif terminal_indices[0] != len(sorted_episode) - 1:
                    self.errors.append(
                        f"trajectory {uid!r} terminal row is not the maximum action order"
                    )
                elif not self.check(
                    _terminal_row_closes_trajectory(
                        sorted_episode[-1][0],
                        episode_length=len(sorted_episode),
                        route_max_rounds=self.route_max_rounds,
                    ),
                    f"trajectory {uid!r} terminal row is not done or a valid max_rounds horizon",
                ):
                    continue
                else:
                    completed_episodes.append(sorted_episode)
                    terminal_id = sorted_episode[0][0].get("item_id")
                    terminal_data_idx = sorted_episode[0][0].get("data_idx")
                    episode_routes = {
                        record.get("route_id") or record.get("data_source")
                        for record, _document in sorted_episode
                    }
                    if self.multitask:
                        self.check(
                            len(episode_routes) == 1
                            and next(iter(episode_routes), None) in self.route_ids,
                            f"trajectory {uid!r} changed route label",
                        )
                    episode_route = (
                        str(next(iter(episode_routes)))
                        if self.multitask and len(episode_routes) == 1
                        else ""
                    )
                    if episode_route:
                        route_episodes[episode_route] += 1
                        update_episodes[episode_route] += 1
                        terminal_record = sorted_episode[-1][0]
                        reward = terminal_record.get("trajectory_return")
                        if _finite_number(reward):
                            route_reward[episode_route] += float(reward)
                        else:
                            self.errors.append(
                                f"trajectory {uid!r} has invalid trajectory_return"
                            )
                        if terminal_record.get("outcome") == "success":
                            route_successes[episode_route] += 1
                        for record, _document in sorted_episode:
                            route_memory_events[episode_route].update(
                                _emitted_memory_events(record)
                            )
                        if self._has_memory_chain(sorted_episode):
                            route_memory_chains[episode_route] += 1
                    if isinstance(terminal_id, str) and terminal_id:
                        terminal_ids.append(terminal_id)
                        if valid_data_idx:
                            terminal_instances.append((terminal_id, terminal_data_idx))
                            if self.multitask:
                                scheduled_route = self.schedule_routes.get(
                                    (terminal_id, terminal_data_idx)
                                )
                                self.check(
                                    scheduled_route == episode_route,
                                    "rollout route differs from the frozen schedule",
                                )
                seen_uids.add(uid)
            self.check(
                len(completed_episodes) == samples_per_update,
                f"rollout update {ordinal} terminal trajectories per learner update "
                f"do not equal {samples_per_update}",
            )
            if any(self._has_memory_chain(episode) for episode in completed_episodes):
                memory_chain_updates.append(ordinal)
            if self.multitask:
                normalized_episodes = _normalized_counter(
                    update_episodes, self.route_ids
                )
                normalized_actions = _normalized_counter(
                    update_action_rows, self.route_ids
                )
                normalized_tokens = _normalized_counter(
                    update_response_tokens, self.route_ids
                )
                per_update_episodes.append(normalized_episodes)
                per_update_action_rows.append(normalized_actions)
                per_update_response_tokens.append(normalized_tokens)
                self._audit_optimizer_file_logger_update(
                    ordinal,
                    episodes=normalized_episodes,
                    action_rows=normalized_actions,
                    response_tokens=normalized_tokens,
                    stale_rows=_normalized_counter(update_stale_rows, self.route_ids),
                )
            if self.batch_multiple is not None:
                derived_padding_rows += (-real_rows_in_file) % self.batch_multiple
            if trigger > 0 and ordinal % trigger == 0:
                stale_action_rows_by_publication[ordinal // trigger] = stale_action_rows

        if versions and self.expected is not None:
            publication_cycles = int(self.expected["publication_cycles"])
            self.check(
                min(versions) >= 0 and max(versions) <= publication_cycles,
                "rollout policy-version span exceeds the published learner horizon",
            )
        if isinstance(file_logger_summary, Mapping):
            native_stale = file_logger_summary.get(
                "native_stale_action_rows_by_publication"
            )
            if isinstance(native_stale, Mapping):
                for (
                    publication,
                    reconstructed,
                ) in stale_action_rows_by_publication.items():
                    self.check(
                        native_stale.get(publication) == reconstructed,
                        "FileLogger/native stale action-row count mismatch at "
                        f"publication {publication}",
                    )

        memory_chains = len(memory_chain_updates)
        self.check(
            memory_chains > 0,
            "no real non-synthetic policy-authored external-document chain "
            "write -> compaction -> read -> modify/reuse -> execute was found",
        )
        late_memory_chains = 0
        if self.expected is not None and self.mode == "formal":
            late_boundary = max(1, int(self.expected["optimizer_updates"]) * 4 // 5)
            late_memory_chains = sum(
                step > late_boundary for step in memory_chain_updates
            )
            self.check(
                late_memory_chains > 0,
                "external-document memory chain disappeared in the final 20% of formal updates",
            )

        if self.multitask:
            self.rolling_8 = _rolling_episode_shares(
                per_update_episodes, self.route_ids
            )
            self.check(
                self.rolling_8["status"] in {"pass", "not_applicable"},
                "rolling-8 optimizer-consumed episode share left the 20%-30% band",
            )
            self.route_summaries = {
                route_id: {
                    "optimizer_consumed_episodes": route_episodes[route_id],
                    "optimizer_consumed_action_rows": route_action_rows[route_id],
                    "optimizer_consumed_policy_response_tokens": route_response_tokens[
                        route_id
                    ],
                    "stale_action_rows": route_stale_rows[route_id],
                    "reward_sum": float(route_reward[route_id]),
                    "reward_mean": (
                        float(route_reward[route_id]) / route_episodes[route_id]
                        if route_episodes[route_id]
                        else None
                    ),
                    "native_successes": route_successes[route_id],
                    "native_success_rate": (
                        route_successes[route_id] / route_episodes[route_id]
                        if route_episodes[route_id]
                        else None
                    ),
                    "document_writes": route_memory_events[route_id]["write"],
                    "compactions": route_memory_events[route_id]["compaction"],
                    "document_reads": route_memory_events[route_id]["read"],
                    "memory_reuses_or_modifications": (
                        route_memory_events[route_id]["reuse"]
                        + route_memory_events[route_id]["modify"]
                    ),
                    "executions": route_memory_events[route_id]["execute"],
                    "complete_memory_chains": route_memory_chains[route_id],
                }
                for route_id in self.route_ids
            }

        self.counts.update(
            real_action_rows=real_rows,
            derived_padding_action_rows=derived_padding_rows,
            real_response_tokens=real_tokens,
            stale_action_rows=stale_action_rows,
            memory_chains=memory_chains,
            late_memory_chains=late_memory_chains,
            completed_episodes=len(terminal_ids),
        )
        if versions:
            self.counts["policy_version_min"] = min(versions)
            self.counts["policy_version_max"] = max(versions)
        self._rollout_summary = {
            "real_rows": real_rows,
            "derived_padding_rows": derived_padding_rows,
            "real_tokens": real_tokens,
            "stale_action_rows": stale_action_rows,
            "stale_action_rows_by_publication": stale_action_rows_by_publication,
            "staleness_diffs": dict(sorted(staleness_diffs.items())),
            "version_pairs": dict(sorted(version_pairs.items())),
            "versions": sorted(versions),
            "terminal_ids": terminal_ids,
            "collection_files": len(paths),
            "memory_chain_updates": memory_chain_updates,
            "per_update_episodes": per_update_episodes,
            "per_update_action_rows": per_update_action_rows,
            "per_update_response_tokens": per_update_response_tokens,
            "episodes_by_route": dict(route_episodes),
            "action_rows_by_route": dict(route_action_rows),
            "response_tokens_by_route": dict(route_response_tokens),
            "stale_action_rows_by_route": dict(route_stale_rows),
        }
        schedule_ids = getattr(self, "_schedule_ids", None)
        if schedule_ids is not None:
            self.check(
                Counter(terminal_ids) == Counter(schedule_ids),
                "rollout terminal trajectory identity/occurrences differ from the publication schedule",
            )
        schedule_instances = getattr(self, "_schedule_instances", None)
        if schedule_instances is not None:
            self.check(
                Counter(terminal_instances) == Counter(schedule_instances),
                "rollout terminal item_id/data_idx occurrences differ from the publication schedule",
            )

    def audit_final_statistics(self) -> None:
        if not self.multitask or self.expected is None:
            return
        path = self.runtime_paths.get("trainer_log")
        if path is None:
            self.errors.append("trainer log path is not receipt-bound")
            return
        try:
            statistics = _load_final_statistics(path)
        except Exception as exc:
            self.error("FinalStatistics", exc)
            return
        queue = statistics.get("queue")
        rollouter = statistics.get("rollouter")
        trainer = statistics.get("trainer")
        cleanup = statistics.get("queue_cleanup")
        if not all(isinstance(value, Mapping) for value in (queue, rollouter, trainer)):
            self.errors.append("FinalStatistics omitted a true-owner snapshot")
            return
        assert isinstance(queue, Mapping)
        assert isinstance(rollouter, Mapping)
        assert isinstance(trainer, Mapping)
        self.check(
            set(queue) == _FINAL_STATISTICS_QUEUE_FIELDS,
            "FinalStatistics queue fields differ from pinned veRL",
        )
        allowed_route_prefixes = tuple(
            f"count/{event}/data_source/"
            for event in _FINAL_STATISTICS_ROUTE_EVENTS
        )
        rollouter_extra_fields = {
            str(key)
            for key in rollouter
            if key not in _FINAL_STATISTICS_ROLLOUTER_FIELDS
            and not any(
                str(key).startswith(prefix)
                and bool(str(key).removeprefix(prefix))
                for prefix in allowed_route_prefixes
            )
        }
        self.check(
            _FINAL_STATISTICS_ROLLOUTER_FIELDS.issubset(rollouter)
            and not rollouter_extra_fields,
            "FinalStatistics rollouter fields differ from pinned veRL",
        )
        self.check(
            set(trainer) == _FINAL_STATISTICS_TRAINER_FIELDS,
            "FinalStatistics trainer fields differ from pinned veRL",
        )
        self.check(
            cleanup == {"status": "completed"},
            "FinalStatistics queue cleanup did not complete exactly",
        )

        def integer(mapping: Mapping[str, Any], key: str, owner: str) -> int:
            parsed = _nonnegative_int(mapping.get(key))
            if parsed is None:
                self.errors.append(
                    f"FinalStatistics {owner}.{key} is missing or not a JSON integer"
                )
                return 0
            return parsed

        queue_totals = {
            "enqueued": integer(queue, "total_produced", "queue"),
            "dequeued": integer(queue, "total_consumed", "queue"),
            "overflow_evicted": integer(queue, "dropped_samples", "queue"),
            "cleanup_cleared": integer(queue, "total_cleared", "queue"),
            "resident": integer(queue, "queue_size", "queue"),
        }
        queue_capacity = integer(queue, "max_queue_size", "queue")
        queue_fields = {
            "enqueued": "enqueued_by_data_source",
            "dequeued": "consumed_by_data_source",
            "overflow_evicted": "evicted_by_data_source",
            "cleanup_cleared": "cleared_by_data_source",
            "resident": "resident_by_data_source",
        }
        queue_routes: dict[str, dict[str, int]] = {}
        for event, field in queue_fields.items():
            try:
                observed = _exact_counter(queue.get(field), label=f"queue.{field}")
            except Exception as exc:
                self.error(f"FinalStatistics queue.{field}", exc)
                observed = {}
            self.check(
                set(observed).issubset(self.route_ids),
                f"FinalStatistics queue {event} contains an undeclared route",
            )
            normalized = _normalized_counter(observed, self.route_ids)
            queue_routes[event] = normalized
            self.check(
                sum(normalized.values()) == queue_totals[event],
                f"FinalStatistics queue {event} route total mismatch",
            )
        self.check(
            queue_totals["enqueued"]
            == queue_totals["dequeued"]
            + queue_totals["overflow_evicted"]
            + queue_totals["cleanup_cleared"]
            + queue_totals["resident"],
            "FinalStatistics global queue conservation failed",
        )
        for route_id in self.route_ids:
            self.check(
                queue_routes["enqueued"][route_id]
                == queue_routes["dequeued"][route_id]
                + queue_routes["overflow_evicted"][route_id]
                + queue_routes["cleanup_cleared"][route_id]
                + queue_routes["resident"][route_id],
                f"FinalStatistics queue conservation failed for route {route_id!r}",
            )

        lifecycle_events = (
            "rollout_dispatched",
            "rollout_inflight",
            "rollout_completed",
            "rollout_failed",
            "rollout_cancelled",
        )
        lifecycle_totals: dict[str, int] = {}
        lifecycle_routes: dict[str, dict[str, int]] = {}
        for event in lifecycle_events:
            total_key = f"count/{event}_samples"
            lifecycle_totals[event] = integer(rollouter, total_key, "rollouter")
            try:
                observed = _flat_route_counter(
                    rollouter,
                    prefix=f"count/{event}",
                    label=f"rollouter {event}",
                )
            except Exception as exc:
                self.error(f"FinalStatistics rollouter {event}", exc)
                observed = {}
            self.check(
                set(observed).issubset(self.route_ids),
                f"FinalStatistics rollouter {event} contains an undeclared route",
            )
            normalized = _normalized_counter(observed, self.route_ids)
            lifecycle_routes[event] = normalized
            self.check(
                sum(normalized.values()) == lifecycle_totals[event],
                f"FinalStatistics rollouter {event} route total mismatch",
            )
        self.check(
            lifecycle_totals["rollout_dispatched"]
            == lifecycle_totals["rollout_inflight"]
            + lifecycle_totals["rollout_completed"]
            + lifecycle_totals["rollout_failed"]
            + lifecycle_totals["rollout_cancelled"],
            "FinalStatistics global rollout lifecycle conservation failed",
        )
        for route_id in self.route_ids:
            self.check(
                lifecycle_routes["rollout_dispatched"][route_id]
                == lifecycle_routes["rollout_inflight"][route_id]
                + lifecycle_routes["rollout_completed"][route_id]
                + lifecycle_routes["rollout_failed"][route_id]
                + lifecycle_routes["rollout_cancelled"][route_id],
                f"FinalStatistics rollout lifecycle conservation failed for route {route_id!r}",
            )

        rollouter_queue_keys = {
            "enqueued": ("count/queue_enqueued_samples", "count/queue_enqueued"),
            "dequeued": ("count/queue_dequeued_samples", "count/queue_dequeued"),
            "overflow_evicted": (
                "count/queue_overflow_evictions",
                "count/queue_overflow_evicted",
            ),
            "cleanup_cleared": (
                "count/queue_cleared_samples",
                "count/queue_cleared",
            ),
            "resident": ("count/queue_resident_samples", "count/queue_resident"),
        }
        for event, (key, route_prefix) in rollouter_queue_keys.items():
            self.check(
                integer(rollouter, key, "rollouter") == queue_totals[event],
                f"FinalStatistics queue/rollouter {event} totals differ",
            )
            try:
                observed = _flat_route_counter(
                    rollouter,
                    prefix=route_prefix,
                    label=f"rollouter queue {event}",
                )
            except Exception as exc:
                self.error(f"FinalStatistics rollouter queue {event}", exc)
                observed = {}
            self.check(
                set(observed).issubset(self.route_ids)
                and _normalized_counter(observed, self.route_ids)
                == queue_routes[event],
                f"FinalStatistics queue/rollouter {event} route totals differ",
            )

        generated = integer(rollouter, "count/total_generated_samples", "rollouter")
        dropped_stale = integer(rollouter, "count/dropped_stale_samples", "rollouter")
        required_samples = integer(rollouter, "static/required_samples", "rollouter")
        self.check(
            generated
            == lifecycle_totals["rollout_completed"]
            == queue_totals["enqueued"],
            "FinalStatistics generated/completed/enqueued totals differ",
        )
        self.check(
            required_samples == int(self.expected["samples_per_update"]),
            "FinalStatistics required_samples differs from optimizer budget",
        )
        active_tasks = integer(rollouter, "monitor/active_tasks_size", "rollouter")
        pending_tasks = integer(
            rollouter, "monitor/queue/pending_queue_size", "rollouter"
        )
        monitored_queue_size = integer(
            rollouter, "monitor/queue/mq_queue_size", "rollouter"
        )
        staleness_samples = integer(
            rollouter, "count/staleness_samples", "rollouter"
        )
        max_required_samples = integer(
            rollouter, "static/max_required_samples", "rollouter"
        )
        max_queue_size = integer(rollouter, "static/max_queue_size", "rollouter")
        max_concurrent_samples = integer(
            rollouter, "static/max_concurrent_samples", "rollouter"
        )
        raw_staleness_threshold = rollouter.get("static/staleness_threshold")
        final_staleness_threshold = (
            float(raw_staleness_threshold)
            if not isinstance(raw_staleness_threshold, bool)
            and isinstance(raw_staleness_threshold, (int, float))
            and math.isfinite(raw_staleness_threshold)
            and float(raw_staleness_threshold) >= 0.0
            else None
        )
        self.check(
            final_staleness_threshold is not None,
            "FinalStatistics rollouter.static/staleness_threshold is invalid",
        )
        self.check(
            monitored_queue_size == queue_totals["resident"],
            "FinalStatistics monitored/resident queue sizes differ",
        )
        self.check(
            queue_capacity == max_queue_size == max_required_samples,
            "FinalStatistics queue capacity fields differ",
        )
        self.check(
            max_required_samples
            == int(
                required_samples
                * ((final_staleness_threshold or 0.0) + 1.0)
                * int(self.expected["trigger_parameter_sync_step"])
            ),
            "FinalStatistics maximum required samples differ from runtime config",
        )
        self.check(
            self.staleness_threshold is not None
            and final_staleness_threshold == self.staleness_threshold,
            "FinalStatistics staleness threshold differs from resolved config",
        )
        self.check(
            max_concurrent_samples <= max_required_samples,
            "FinalStatistics maximum concurrency exceeds the queue bound",
        )
        for label, value in (
            ("active rollout task", active_tasks),
            ("pending rollout task", pending_tasks),
            ("rollout inflight", lifecycle_totals["rollout_inflight"]),
            ("rollout failed", lifecycle_totals["rollout_failed"]),
            ("rollout cancelled", lifecycle_totals["rollout_cancelled"]),
            ("queue overflow eviction", queue_totals["overflow_evicted"]),
            ("queue resident", queue_totals["resident"]),
            ("dropped stale sample", dropped_stale),
        ):
            self.check(value == 0, f"FinalStatistics has nonzero {label} count")

        trainer_specs = (
            ("episodes", "optimizer_consumed_episodes"),
            ("action_rows", "optimizer_consumed_action_rows"),
            (
                "policy_response_tokens",
                "optimizer_consumed_policy_response_tokens",
            ),
        )
        trainer_totals: dict[str, int] = {}
        trainer_routes: dict[str, dict[str, int]] = {}
        for measure, field in trainer_specs:
            total = integer(trainer, field, "trainer")
            trainer_totals[measure] = total
            map_field = f"{field}_by_data_source"
            try:
                observed = _exact_counter(
                    trainer.get(map_field), label=f"trainer.{map_field}"
                )
            except Exception as exc:
                self.error(f"FinalStatistics trainer.{map_field}", exc)
                observed = {}
            self.check(
                set(observed).issubset(self.route_ids),
                f"FinalStatistics trainer {measure} contains an undeclared route",
            )
            normalized = _normalized_counter(observed, self.route_ids)
            trainer_routes[measure] = normalized
            self.check(
                sum(normalized.values()) == total,
                f"FinalStatistics trainer {measure} route total mismatch",
            )

        stale_total = integer(trainer, "stale_action_rows", "trainer")
        try:
            stale_observed = _exact_counter(
                trainer.get("stale_action_rows_by_data_source"),
                label="trainer.stale_action_rows_by_data_source",
            )
        except Exception as exc:
            self.error("FinalStatistics trainer stale rows", exc)
            stale_observed = {}
        self.check(
            set(stale_observed).issubset(self.route_ids),
            "FinalStatistics trainer stale rows contain an undeclared route",
        )
        stale_routes = _normalized_counter(stale_observed, self.route_ids)
        self.check(
            sum(stale_routes.values()) == stale_total,
            "FinalStatistics trainer stale-row route total mismatch",
        )

        rollout = getattr(self, "_rollout_summary", {})
        expected_routes = {
            "episodes": _normalized_counter(
                rollout.get("episodes_by_route", {}), self.route_ids
            ),
            "action_rows": _normalized_counter(
                rollout.get("action_rows_by_route", {}), self.route_ids
            ),
            "policy_response_tokens": _normalized_counter(
                rollout.get("response_tokens_by_route", {}), self.route_ids
            ),
        }
        expected_totals = {
            "episodes": int(self.counts["completed_episodes"]),
            "action_rows": int(self.counts["real_action_rows"]),
            "policy_response_tokens": int(self.counts["real_response_tokens"]),
        }
        for measure in expected_totals:
            self.check(
                trainer_totals[measure] == expected_totals[measure],
                f"FinalStatistics optimizer-consumed {measure} differs from real rollout evidence",
            )
            self.check(
                trainer_routes[measure] == expected_routes[measure],
                f"FinalStatistics optimizer-consumed {measure} route totals differ from real rollout evidence",
            )
        self.check(
            trainer_totals["episodes"]
            == queue_totals["dequeued"]
            == int(self.expected["episodes"]),
            "FinalStatistics dequeued/optimizer episode total differs from launch budget",
        )
        self.check(
            lifecycle_routes["rollout_completed"] == queue_routes["enqueued"],
            "FinalStatistics completed/enqueued route accounting differs",
        )
        self.check(
            queue_routes["dequeued"] == trainer_routes["episodes"],
            "FinalStatistics dequeued/optimizer route accounting differs",
        )
        self.check(
            stale_total == int(self.counts["stale_action_rows"]),
            "FinalStatistics stale-row total differs from rollout evidence",
        )
        self.check(
            stale_routes
            == _normalized_counter(
                rollout.get("stale_action_rows_by_route", {}), self.route_ids
            ),
            "FinalStatistics stale-row route totals differ from rollout evidence",
        )
        self.check(
            stale_total <= trainer_totals["action_rows"],
            "FinalStatistics stale-row total exceeds real optimizer action rows",
        )
        self.check(
            integer(trainer, "current_param_version", "trainer")
            == int(self.expected["publication_cycles"]),
            "FinalStatistics current parameter version differs from publication budget",
        )

        self.accounting = {
            "schema": _FINAL_STATISTICS_SCHEMA,
            "queue": queue_totals,
            "queue_by_route": queue_routes,
            "rollout_lifecycle": lifecycle_totals,
            "rollout_lifecycle_by_route": lifecycle_routes,
            "optimizer_consumed": trainer_totals,
            "optimizer_consumed_by_route": trainer_routes,
            "stale_action_rows": stale_total,
            "stale_action_rows_by_route": stale_routes,
            "rollout_runtime": {
                "staleness_samples": staleness_samples,
                "max_required_samples": max_required_samples,
                "required_samples": required_samples,
                "staleness_threshold": final_staleness_threshold,
                "max_queue_size": max_queue_size,
                "max_concurrent_samples": max_concurrent_samples,
            },
        }

    def audit_checkpoint(self) -> None:
        if self.expected is None:
            return
        if self.trainer_world_size is None:
            self.errors.append("checkpoint trainer world size is unavailable")
            return
        world_size = self.trainer_world_size
        expected_step = int(self.expected["publication_cycles"])
        root = self.runtime_paths.get("checkpoints")
        if root is None:
            self.errors.append("checkpoint path is not receipt-bound")
            return
        tracker = root / "latest_checkpointed_iteration.txt"
        if not tracker.is_file() or tracker.is_symlink():
            self.errors.append(f"required checkpoint tracker is missing: {tracker}")
            return
        try:
            tracked_step = int(tracker.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeError, ValueError) as exc:
            self.error("checkpoint tracker", exc)
            return
        self.check(tracked_step == expected_step, "checkpoint tracker step mismatch")
        target = root / f"global_step_{expected_step}"
        dataloader = target / "data.pt"
        self.check(
            dataloader.is_file()
            and not dataloader.is_symlink()
            and dataloader.stat().st_size > 0,
            f"dataloader checkpoint is missing at global_step_{expected_step}",
        )
        for role in ("actor", "critic"):
            role_dir = target / role
            missing: list[str] = []
            for kind in ("model", "optim", "extra_state"):
                for rank in range(world_size):
                    filename = f"{kind}_world_size_{world_size}_rank_{rank}.pt"
                    path = role_dir / filename
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or path.stat().st_size <= 0
                    ):
                        missing.append(filename)
            self.check(
                not missing,
                f"checkpoint {role} is incomplete at global_step_{expected_step}: "
                + ", ".join(missing),
            )

    def terminal_path(self) -> str:
        if self.trainer_exit_code != 0:
            return "crash"
        scheduled = self.counts.get("scheduled_episodes", 0)
        completed = self.counts.get("completed_episodes", 0)
        if scheduled > 0 and completed < scheduled:
            return "partial"
        return "success"

    def run(self) -> dict[str, Any]:
        if self.trainer_exit_code != 0:
            self.errors.append(f"trainer exit code {self.trainer_exit_code} is nonzero")
        self.audit_launch()
        self.audit_config()
        self.audit_file_logger()
        self.audit_rollouts()
        self.audit_final_statistics()
        self.audit_checkpoint()
        terminal_path = self.terminal_path()
        if terminal_path == "partial" and not any(
            "underfill" in error.casefold() for error in self.errors
        ):
            self.errors.append("native runtime ended on a partial terminal path")
        return {
            "schema": "amg_verl_fully_async_finalization_v3",
            "status": "pass" if not self.errors else "fail",
            "terminal_path": terminal_path,
            "trainer_exit_code": self.trainer_exit_code,
            "mode": self.mode,
            "role": self.role,
            "launch_receipt_schema": self.receipt_schema,
            "counts": self.counts,
            "routes": self.route_summaries,
            "rolling_8_episode_share": self.rolling_8,
            "final_accounting": self.accounting,
            "errors": self.errors,
        }


def finalize_run(
    run_dir: str | os.PathLike[str], trainer_exit_code: int
) -> dict[str, Any]:
    """Audit one native run and atomically replace its finalization verdict."""

    directory = Path(run_dir).resolve()
    audit = _Audit(directory, trainer_exit_code)
    try:
        verdict = audit.run()
    except Exception as exc:  # finalization itself must fail closed on every path
        audit.error("unexpected finalizer failure", exc)
        verdict = {
            "schema": "amg_verl_fully_async_finalization_v3",
            "status": "fail",
            "terminal_path": audit.terminal_path(),
            "trainer_exit_code": audit.trainer_exit_code,
            "mode": audit.mode,
            "role": audit.role,
            "launch_receipt_schema": audit.receipt_schema,
            "counts": audit.counts,
            "routes": audit.route_summaries,
            "rolling_8_episode_share": audit.rolling_8,
            "final_accounting": audit.accounting,
            "errors": audit.errors,
        }
    protected_files, protected_directories = _receipt_protected_paths(
        audit.launch or {}, directory, include_finalization=False
    )
    requested_output = audit.runtime_paths.get("finalization")
    canonical_output = _path_within(
        directory, str(directory / "finalization.json")
    )
    if audit.receipt_schema == _MULTITASK_RECEIPT_SCHEMA:
        output_candidates = (
            requested_output,
            canonical_output,
            _path_within(
                directory, str(directory / "finalization-fail-closed.json")
            ),
        )
    else:
        # Legacy v5 has a canonical output path.  Never let a malformed receipt
        # redirect a new failure verdict and leave an older canonical PASS live.
        output_candidates = (
            canonical_output,
            _path_within(
                directory, str(directory / "finalization-fail-closed.json")
            ),
        )
    output = next(
        (
            candidate
            for candidate in output_candidates
            if candidate is not None
            and not _path_overlaps_inputs(
                candidate, protected_files, protected_directories
            )
        ),
        None,
    )
    if output is None:
        raise RuntimeError("no safe finalization output path is available")
    _atomic_json(output, verdict)
    return verdict


__all__ = ["finalize_run"]
