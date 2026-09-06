"""Fail-closed update-1 token-credit gate for a live AMG formal run.

The gate observes only run-owned artifacts.  A passing gate publishes one
atomic receipt and leaves the formal owner alone.  A failing gate may request
``SIGTERM`` only after re-authenticating the exact process-bootstrap lease by
receipt bytes/inode, PID start ticks, process group/session, command line, and
both run-id environment tags.  There is deliberately no raw-PID, process-name,
or broad process-group/name fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fallback_supervisor import (
    FallbackError,
    _atomic_json,
    _bound_json_file,
    _signal_identity,
)
from .finalizer import (
    _MULTITASK_RECEIPT_SCHEMA,
    _ROLLOUT_CORRECTION_ESS,
    _ROLLOUT_CORRECTION_FRACTIONS,
    _ROLLOUT_CORRECTION_IDENTITY_TOLERANCE,
    _ROLLOUT_CORRECTION_METRICS,
    _at,
    _Audit,
    _finite_number,
    _finite_positive,
    _load_json,
    _nonnegative_integral,
    _path_overlaps_inputs,
    _path_within,
    _receipt_protected_paths,
    sha256_file,
)
from .online_monitor import _complete_jsonl_rows, _observe_run

_GATE_SCHEMA = "amg_update1_token_credit_gate_v1"
_OWNER_SCHEMA = "amg_fallback_managed_process_identity_v2"
_EXPECTED_OWNER_NAME = "outer-multitask-orchestrator"
_EXPECTED_ROUTES = frozenset({"webshop", "swesmith", "literesearcher", "openmle_fast"})
_ZERO_FAILURE_COUNTERS = (
    "fully_async/count/rollout_failed_samples",
    "fully_async/count/rollout_cancelled_samples",
    "fully_async/count/queue_overflow_evictions",
    "fully_async/count/dropped_stale_samples",
)
_OPTIMIZER_EXECUTION_SUFFIXES = (
    "ppo_epoch_passes_delta",
    "mini_batches_per_epoch",
    "optimizer_steps_delta",
)
_ZOMBIE_STATES = frozenset({"Z", "X", "x"})


class GatePending(RuntimeError):
    """The immutable update-1 evidence is not complete yet."""


class GateFailure(RuntimeError):
    """Complete evidence violates the frozen update-1 contract."""


class OwnerAuthenticationError(GateFailure):
    """The formal owner cannot safely authorize an exact stop."""


@dataclass(frozen=True)
class ExpectedRunIdentity:
    run_id: str
    outer_commit: str
    inner_commit: str
    verl_commit: str

    def as_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "outer_commit": self.outer_commit,
            "inner_commit": self.inner_commit,
            "verl_commit": self.verl_commit,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute_without_symlink_resolution(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def _fail(message: str) -> None:
    raise GateFailure(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _is_hex_revision(value: str) -> bool:
    return len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_expected_identity(expected: ExpectedRunIdentity) -> None:
    _require(bool(expected.run_id), "expected run_id is empty")
    for label, value in (
        ("outer", expected.outer_commit),
        ("inner", expected.inner_commit),
        ("veRL", expected.verl_commit),
    ):
        _require(
            _is_hex_revision(value),
            f"expected {label} commit is not a full lowercase SHA",
        )


def _read_small(path: Path, *, maximum_bytes: int = 1 << 20) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise OwnerAuthenticationError(
            f"cannot stat process evidence {path}: {error}"
        ) from error
    if size > maximum_bytes:
        raise OwnerAuthenticationError(f"process evidence is too large: {path}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise OwnerAuthenticationError(
            f"cannot read process evidence {path}: {error}"
        ) from error
    if len(payload) > maximum_bytes:
        raise OwnerAuthenticationError(f"process evidence is too large: {path}")
    return payload


def _proc_snapshot(proc_root: Path, pid: int) -> dict[str, Any] | None:
    path = proc_root / str(pid) / "stat"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise OwnerAuthenticationError(
            f"cannot audit owner stat {path}: {error}"
        ) from error
    try:
        fields = raw.rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgrp": int(fields[2]),
            "session": int(fields[3]),
            "start_ticks": fields[19],
        }
    except (IndexError, TypeError, ValueError) as error:
        raise OwnerAuthenticationError(f"invalid owner stat record: {path}") from error


def _proc_argv(proc_root: Path, pid: int) -> tuple[str, ...]:
    raw = _read_small(proc_root / str(pid) / "cmdline")
    try:
        return tuple(part.decode("utf-8") for part in raw.split(b"\0") if part)
    except UnicodeDecodeError as error:
        raise OwnerAuthenticationError("owner command line is not UTF-8") from error


def _proc_environment(proc_root: Path, pid: int) -> frozenset[bytes]:
    return frozenset(
        part
        for part in _read_small(proc_root / str(pid) / "environ").split(b"\0")
        if part
    )


def _single_option(command: Sequence[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise OwnerAuthenticationError(f"owner command has no unique {option}")
    value = command[positions[0] + 1]
    if not value or value.startswith("--"):
        raise OwnerAuthenticationError(f"owner command has an invalid {option} value")
    return value


def _binding_view(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: binding[key]
        for key in ("path", "device", "inode", "ctime_ns", "size", "sha256")
    }


def authenticate_owner(
    owner_receipt: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    run_id: str,
    *,
    proc_root: str | os.PathLike[str] = "/proc",
) -> dict[str, Any]:
    """Authenticate the exact live process-bootstrap owner lease."""

    receipt_path = _absolute_without_symlink_resolution(owner_receipt)
    if not receipt_path.exists() and not receipt_path.is_symlink():
        raise GatePending(f"owner identity receipt is not published: {receipt_path}")
    try:
        payload, binding = _bound_json_file(receipt_path)
    except FallbackError as error:
        raise OwnerAuthenticationError(str(error)) from error

    if payload.get("schema") != _OWNER_SCHEMA:
        raise OwnerAuthenticationError("owner identity receipt schema mismatch")
    if payload.get("name") != _EXPECTED_OWNER_NAME:
        raise OwnerAuthenticationError("owner identity receipt name mismatch")
    try:
        pid = int(payload["pid"])
        start_ticks = str(payload["start_ticks"])
        process_group = int(payload["process_group"])
    except (KeyError, TypeError, ValueError) as error:
        raise OwnerAuthenticationError(
            "owner identity receipt fields are invalid"
        ) from error
    if pid <= 0 or process_group != pid or not start_ticks:
        raise OwnerAuthenticationError(
            "owner identity is not an exact process-group leader"
        )

    bootstrap = payload.get("bootstrap")
    raw_command = payload.get("command")
    immutable_contract = payload.get("immutable_contract")
    if (
        not isinstance(bootstrap, str)
        or not bootstrap
        or not Path(bootstrap).is_absolute()
    ):
        raise OwnerAuthenticationError("owner bootstrap path is not absolute")
    if (
        not isinstance(raw_command, list)
        or not raw_command
        or any(not isinstance(value, str) or not value for value in raw_command)
    ):
        raise OwnerAuthenticationError("owner command is not a non-empty string list")
    if not isinstance(immutable_contract, Mapping) or not immutable_contract:
        raise OwnerAuthenticationError("owner immutable contract is missing")
    command = tuple(raw_command)
    command_run_dir = Path(_single_option(command, "--run-dir")).resolve()
    command_run_id = _single_option(command, "--experiment-name")
    if command_run_dir != Path(run_dir).resolve() or command_run_id != run_id:
        raise OwnerAuthenticationError("owner command is not bound to this run")

    root = Path(proc_root)
    observed = _proc_snapshot(root, pid)
    if observed is None or observed["state"] in _ZOMBIE_STATES:
        raise OwnerAuthenticationError("owner process is not live")
    if (
        observed["start_ticks"] != start_ticks
        or observed["pgrp"] != process_group
        or observed["session"] != pid
    ):
        raise OwnerAuthenticationError(
            "owner PID/start-ticks/process-group/session identity mismatch"
        )

    argv = _proc_argv(root, pid)
    if len(argv) < 3 or argv[1] != bootstrap or argv[-len(command) :] != command:
        raise OwnerAuthenticationError(
            "live owner command line differs from its receipt"
        )
    environment = _proc_environment(root, pid)
    required_tags = {
        f"AMG_MULTITASK_RUN_ID={run_id}".encode(),
        f"AGENTMEMORY_RUN_ID={run_id}".encode(),
    }
    if not required_tags.issubset(environment):
        raise OwnerAuthenticationError("live owner environment lacks exact run-id tags")

    return {
        "identity": {
            "pid": pid,
            "start_ticks": start_ticks,
            "pgrp": process_group,
            "session": pid,
        },
        "receipt": {
            "schema": payload.get("schema"),
            "name": payload.get("name"),
            "bootstrap": bootstrap,
            "command": list(command),
            "immutable_contract": dict(immutable_contract),
        },
        "binding": _binding_view(binding),
        "proc_root": str(root),
    }


def _owner_is_live(lease: Mapping[str, Any], proc_root: Path) -> bool:
    identity = lease.get("identity")
    if not isinstance(identity, Mapping):
        return False
    try:
        observed = _proc_snapshot(proc_root, int(identity["pid"]))
    except (KeyError, TypeError, ValueError, OwnerAuthenticationError):
        return False
    return bool(
        observed
        and observed["state"] not in _ZOMBIE_STATES
        and observed["start_ticks"] == str(identity.get("start_ticks"))
        and observed["pgrp"] == int(identity.get("pgrp", -1))
        and observed["session"] == int(identity.get("session", -1))
    )


def _load_required_bound_json(
    path: Path, label: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if not path.exists() and not path.is_symlink():
        raise GatePending(f"required {label} is not published: {path}")
    try:
        payload, binding = _bound_json_file(path)
    except FallbackError as error:
        raise GateFailure(f"invalid {label}: {error}") from error
    return payload, _binding_view(binding)


def _rollout_episode_readiness(path: Path, expected_episodes: int) -> None:
    if not path.exists() and not path.is_symlink():
        raise GatePending(f"update-1 rollout is not published: {path}")
    try:
        rows = _complete_jsonl_rows(path, "update-1 rollout JSONL")
    except (FileNotFoundError, ValueError) as error:
        if "no complete rows" in str(error):
            raise GatePending(str(error)) from error
        raise GateFailure(str(error)) from error
    terminal_uids: set[str] = set()
    observed_uids: set[str] = set()
    for index, row in enumerate(rows):
        raw = row.get("step_record_json")
        if not isinstance(raw, str):
            raise GateFailure(f"update-1 rollout row {index} omitted step_record_json")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise GateFailure(
                f"update-1 rollout row {index} has invalid step record"
            ) from error
        if not isinstance(record, Mapping):
            raise GateFailure(
                f"update-1 rollout row {index} step record is not an object"
            )
        uid = record.get("trajectory_uid")
        if not isinstance(uid, str) or not uid:
            raise GateFailure(f"update-1 rollout row {index} omitted trajectory_uid")
        observed_uids.add(uid)
        if record.get("trajectory_terminal") is True:
            terminal_uids.add(uid)
    if len(terminal_uids) < expected_episodes or len(observed_uids) < expected_episodes:
        raise GatePending(
            f"update-1 rollout has {len(terminal_uids)}/{expected_episodes} terminal episodes"
        )


def _step1_owner_rows(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], int]:
    if not path.exists() and not path.is_symlink():
        raise GatePending(f"FileLogger JSONL is not published: {path}")
    try:
        rows = _complete_jsonl_rows(path, "FileLogger JSONL")
    except (FileNotFoundError, ValueError) as error:
        if "no complete rows" in str(error):
            raise GatePending(str(error)) from error
        raise GateFailure(str(error)) from error
    data_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        step = row.get("step")
        data = row.get("data")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or not isinstance(data, Mapping)
        ):
            raise GateFailure(f"FileLogger row {index} has invalid step/data")
        if step == 1:
            data_rows.append(data)
    learner_rows = [
        data
        for data in data_rows
        if any(
            key in data
            for key in ("actor/grad_norm", "critic/grad_norm", "training/global_step")
        )
    ]
    rollouter_rows = [
        data
        for data in data_rows
        if any(str(key).startswith("fully_async/rollouter/") for key in data)
    ]
    if not learner_rows or not rollouter_rows:
        raise GatePending(
            "FileLogger update 1 has not published both current-owner rows"
        )
    if (
        len(learner_rows) != 1
        or len(rollouter_rows) != 1
        or learner_rows[0] is rollouter_rows[0]
    ):
        raise GateFailure(
            "FileLogger update 1 has no unique distinct learner/rollouter owner rows"
        )
    return learner_rows[0], rollouter_rows[0], len(rows)


def _positive_integral(value: Any, label: str) -> int:
    parsed = _nonnegative_integral(value)
    if parsed is None or parsed <= 0:
        raise GateFailure(f"{label} is not a positive integer")
    return parsed


def _zero_integral(value: Any, label: str) -> int:
    parsed = _nonnegative_integral(value)
    if parsed != 0:
        raise GateFailure(f"{label} is not zero")
    return parsed


def _audit_step1_metrics(path: Path) -> dict[str, Any]:
    learner, rollouter, row_count = _step1_owner_rows(path)
    gradients: dict[str, float] = {}
    for role in ("actor", "critic"):
        key = f"{role}/grad_norm"
        value = learner.get(key)
        if not _finite_positive(value):
            raise GateFailure(f"FileLogger update 1 has no finite nonzero {key}")
        gradients[role] = float(value)

    execution: dict[str, dict[str, int]] = {}
    for role in ("actor", "critic"):
        execution[role] = {
            suffix: _positive_integral(
                learner.get(f"{role}/{suffix}"), f"{role}/{suffix}"
            )
            for suffix in _OPTIMIZER_EXECUTION_SUFFIXES
        }
    actor = execution["actor"]
    critic = execution["critic"]
    if not (
        actor["ppo_epoch_passes_delta"] == 1
        and critic["ppo_epoch_passes_delta"] == 2
        and actor["mini_batches_per_epoch"] == critic["mini_batches_per_epoch"]
        and actor["optimizer_steps_delta"]
        == actor["ppo_epoch_passes_delta"] * actor["mini_batches_per_epoch"]
        and critic["optimizer_steps_delta"]
        == critic["ppo_epoch_passes_delta"] * critic["mini_batches_per_epoch"]
        and critic["optimizer_steps_delta"] == 2 * actor["optimizer_steps_delta"]
    ):
        raise GateFailure("FileLogger update 1 violated actor-K1/critic-K2 execution")

    correction: dict[str, float] = {}
    for canonical_key in _ROLLOUT_CORRECTION_METRICS:
        emitted_key = f"actor/{canonical_key}"
        value = learner.get(emitted_key)
        if not _finite_number(value):
            raise GateFailure(f"FileLogger update 1 lacks finite {emitted_key}")
        correction[canonical_key] = float(value)
    for key in _ROLLOUT_CORRECTION_FRACTIONS:
        if not 0.0 <= correction[key] <= 1.0:
            raise GateFailure(f"FileLogger update 1 has out-of-range actor/{key}")
    ess = correction[_ROLLOUT_CORRECTION_ESS]
    if not 0.0 < ess <= 1.0 + _ROLLOUT_CORRECTION_IDENTITY_TOLERANCE:
        raise GateFailure("FileLogger update 1 has invalid IcePop ESS")
    applied_min = correction["rollout_corr/rollout_is_min"]
    applied_mean = correction["rollout_corr/rollout_is_mean"]
    applied_max = correction["rollout_corr/rollout_is_max"]
    applied_std = correction["rollout_corr/rollout_is_std"]
    if not (
        0.0
        <= applied_min
        <= applied_mean
        <= applied_max
        <= 4.0 + _ROLLOUT_CORRECTION_IDENTITY_TOLERANCE
        and applied_mean > 0.0
        and applied_std >= 0.0
    ):
        raise GateFailure("FileLogger update 1 has invalid IcePop applied weights")
    oob = correction["rollout_corr/rollout_is_oob_ratio"]
    high = correction["rollout_corr/rollout_is_ratio_fraction_high"]
    low = correction["rollout_corr/rollout_is_ratio_fraction_low"]
    if not math.isclose(
        oob,
        high + low,
        rel_tol=0.0,
        abs_tol=_ROLLOUT_CORRECTION_IDENTITY_TOLERANCE,
    ):
        raise GateFailure("FileLogger update 1 IcePop high+low does not equal OOB")

    consumed_version = _nonnegative_integral(
        learner.get("fully_async/count/current_param_version")
    )
    if consumed_version != 0:
        raise GateFailure("FileLogger update 1 consumed parameter version is not 0")
    _zero_integral(
        learner.get("fully_async/count/stale_trajectory_processed"),
        "fully_async/count/stale_trajectory_processed",
    )
    required_samples = _nonnegative_integral(
        learner.get("fully_async/static/required_samples")
    )
    if required_samples != 64:
        raise GateFailure("FileLogger update 1 required_samples is not 64")
    version_time = rollouter.get("fully_async/rollouter/version_time")
    if not _finite_positive(version_time):
        raise GateFailure("FileLogger update 1 lacks positive publication version_time")
    step_generated = _positive_integral(
        rollouter.get("fully_async/rollouter/step_generated_samples"),
        "fully_async/rollouter/step_generated_samples",
    )
    failures = {
        key: _zero_integral(rollouter.get(key), key) for key in _ZERO_FAILURE_COUNTERS
    }
    return {
        "file_logger_complete_rows": row_count,
        "grad_norm": gradients,
        "optimizer_execution": execution,
        "icepop": {
            "kl": correction["rollout_corr/kl"],
            "k3_kl": correction["rollout_corr/k3_kl"],
            "log_ppl_abs_diff": correction["rollout_corr/log_ppl_abs_diff"],
            "oob_ratio": oob,
            "fraction_high": high,
            "fraction_low": low,
            "applied_weight_min": applied_min,
            "applied_weight_mean": applied_mean,
            "applied_weight_max": applied_max,
            "applied_weight_std": applied_std,
            "effective_sample_size": ess,
        },
        "publication": {
            "publication_step": 1,
            "consumed_parameter_version": consumed_version,
            "published_parameter_version": 1,
            "rollouter_version_time": float(version_time),
            "rollouter_step_generated_samples": step_generated,
        },
        "zero_failure_counters": failures,
        "stale_trajectory_processed": 0,
    }


def _freeze_readiness(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    manifest, binding = _load_required_bound_json(
        path, "critic parameter freeze manifest"
    )
    status = manifest.get("status")
    if status in {"pending", "pending_optimizer_step_learning_rates"}:
        raise GatePending(f"critic parameter freeze manifest is still {status}")
    return manifest, binding


def _audit_freeze(audit: _Audit, manifest: Mapping[str, Any]) -> dict[str, Any]:
    before = len(audit.errors)
    audit.audit_critic_parameter_freeze()
    errors = audit.errors[before:]
    if errors:
        raise GateFailure("critic parameter freeze audit failed: " + "; ".join(errors))
    learning_rates = audit.critic_parameter_freeze.get("optimizer_step_learning_rates")
    expected_learning_rates = (5e-7, 1e-6)
    lr_trace_matches = (
        isinstance(learning_rates, list)
        and len(learning_rates) == len(expected_learning_rates)
        and all(
            isinstance(observed, list)
            and bool(observed)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-15)
                for value in observed
            )
            for observed, expected in zip(learning_rates, expected_learning_rates)
        )
    )
    if not lr_trace_matches:
        raise GateFailure("critic optimizer LR trace is not exactly 5e-7 then 1e-6")
    first_step = manifest.get("first_optimizer_step")
    optimizer = manifest.get("optimizer")
    if not isinstance(first_step, Mapping) or not isinstance(optimizer, Mapping):
        raise GateFailure("critic freeze manifest lacks optimizer/probe evidence")
    frozen = first_step.get("frozen")
    trainable = first_step.get("trainable")
    if not isinstance(frozen, Mapping) or not isinstance(trainable, Mapping):
        raise GateFailure("critic freeze manifest lacks parameter delta evidence")
    return {
        **audit.critic_parameter_freeze,
        "optimizer_membership_exact": optimizer.get("membership_exact"),
        "frozen_delta": {
            "changed_count": frozen.get("changed_count"),
            "l2": frozen.get("l2"),
            "max_abs": frozen.get("max_abs"),
        },
        "trainable_delta": {
            "changed_count": trainable.get("changed_count"),
            "l2": trainable.get("l2"),
            "max_abs": trainable.get("max_abs"),
        },
    }


def _source_identity(launch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "outer_commit": _at(launch, "source.outer_commit"),
        "inner_commit": _at(launch, "source.agentgym_commit"),
        "verl_commit": _at(launch, "source.verl_commit"),
    }


def audit_update1(
    run_dir: str | os.PathLike[str], expected: ExpectedRunIdentity
) -> dict[str, Any]:
    """Audit complete update-1 evidence without changing a live process."""

    _validate_expected_identity(expected)
    directory = Path(run_dir).resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise GatePending(f"run directory is not published: {directory}")
    launch_path = directory / "launch-receipt.json"
    launch, launch_binding = _load_required_bound_json(launch_path, "launch receipt")

    observed_source = _source_identity(launch)
    expected_source = {
        "outer_commit": expected.outer_commit,
        "inner_commit": expected.inner_commit,
        "verl_commit": expected.verl_commit,
    }
    _require(
        launch.get("schema") == _MULTITASK_RECEIPT_SCHEMA,
        "launch is not a multitask receipt",
    )
    _require(
        _at(launch, "inputs.experiment_name") == expected.run_id,
        "launch run_id mismatch",
    )
    _require(
        directory.name == expected.run_id, "run directory basename differs from run_id"
    )
    _require(observed_source == expected_source, "launch source identity mismatch")

    audit = _Audit(directory, trainer_exit_code=0, require_trainer_log=False)
    audit.audit_launch(launch)
    audit.audit_config()
    if audit.errors:
        raise GateFailure(
            "launch identity/config audit failed: " + "; ".join(audit.errors)
        )
    if audit.expected is None or audit.resolved_config is None:
        raise GateFailure(
            "launch identity/config audit did not bind a resolved contract"
        )
    _require(audit.mode == "formal", "update-1 gate requires formal mode")
    _require(
        set(audit.route_ids) == _EXPECTED_ROUTES,
        "launch does not bind the four CAMG routes",
    )
    _require(
        audit.expected.get("samples_per_update") == 64, "samples_per_update is not 64"
    )
    _require(
        audit.expected.get("trigger_parameter_sync_step") == 1,
        "trigger_parameter_sync_step is not 1",
    )

    rollout_path = audit.runtime_paths["rollout_data"] / "1.jsonl"
    metrics_path = audit.runtime_paths["file_logger"]
    freeze_path = audit.runtime_paths["critic_parameter_freeze"]
    _rollout_episode_readiness(rollout_path, 64)
    _step1_owner_rows(metrics_path)
    freeze_manifest, freeze_binding = _freeze_readiness(freeze_path)

    try:
        snapshot = _observe_run(directory, 1, launch)
    except Exception as error:
        raise GateFailure(
            f"update-1 rollout/composition audit failed: {error}"
        ) from error
    route_episodes = {
        route_id: snapshot["routes"][route_id]["optimizer_consumed_episodes"]
        for route_id in audit.route_ids
    }
    _require(
        sum(route_episodes.values()) == 64,
        "update 1 did not consume exactly 64 episodes",
    )
    _require(
        all(value > 0 for value in route_episodes.values()),
        "one or more CAMG routes are absent at update 1",
    )

    metrics = _audit_step1_metrics(metrics_path)
    metrics["publication"]["trigger_parameter_sync_step"] = 1
    freeze = _audit_freeze(audit, freeze_manifest)
    config = audit.resolved_config
    config_summary = {
        "adv_estimator": _at(config, "algorithm.adv_estimator"),
        "full_learner_batch_updates": _at(
            config, "algorithm.full_learner_batch_updates"
        ),
        "rollout_is": _at(config, "algorithm.rollout_correction.rollout_is"),
        "rollout_is_threshold": _at(
            config, "algorithm.rollout_correction.rollout_is_threshold"
        ),
        "rollout_is_batch_normalize": _at(
            config, "algorithm.rollout_correction.rollout_is_batch_normalize"
        ),
        "actor_ppo_epochs": _at(config, "actor_rollout_ref.actor.ppo_epochs"),
        "critic_ppo_epochs": _at(config, "critic.ppo_epochs"),
        "critic_lr": _at(config, "critic.optim.lr"),
        "trigger_parameter_sync_step": _at(
            config, "async_training.trigger_parameter_sync_step"
        ),
    }

    evidence_paths = {
        "run_source_config_identity": str(launch_path),
        "rollout_episodes_and_routes": str(rollout_path),
        "gradient_optimizer_icepop_publication_failures": str(metrics_path),
        "critic_freeze_optimizer_delta_and_lr": str(freeze_path),
    }
    return {
        "snapshot_update": 1,
        "expected_identity": expected.as_dict(),
        "observed_identity": {"run_id": expected.run_id, **observed_source},
        "launch_binding": launch_binding,
        "config": config_summary,
        "optimizer_budget": snapshot["optimizer_budget"],
        "route_episodes": route_episodes,
        "rollout_snapshot": snapshot,
        "token_credit": metrics,
        "critic_parameter_freeze": freeze,
        "evidence_paths": evidence_paths,
        "evidence_sha256": {
            "launch_receipt": sha256_file(launch_path),
            "rollout_update1": sha256_file(rollout_path),
            "file_logger_snapshot": sha256_file(metrics_path),
            "critic_parameter_freeze": sha256_file(freeze_path),
        },
        "freeze_binding": freeze_binding,
    }


def _validate_output_path(
    directory: Path,
    output_path: str | os.PathLike[str],
    owner_receipt: Path,
    launch: Mapping[str, Any] | None,
) -> Path:
    requested = _absolute_without_symlink_resolution(output_path)
    if requested.is_symlink():
        raise GateFailure("gate output is a symlink or non-file")
    output = _path_within(directory, str(requested))
    if output is None:
        raise GateFailure("gate output must be an absolute path inside run_dir")
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise GateFailure("gate output is a symlink or non-file")
    try:
        resolved_owner = owner_receipt.resolve()
    except (OSError, RuntimeError) as error:
        raise GateFailure(f"cannot resolve owner receipt path: {error}") from error
    if output == resolved_owner:
        raise GateFailure("gate output overlaps the owner identity receipt")
    if launch is not None:
        protected_files, protected_directories = _receipt_protected_paths(
            launch, directory, include_finalization=True
        )
        if _path_overlaps_inputs(output, protected_files, protected_directories):
            raise GateFailure("gate output overlaps a receipt-bound input artifact")
    elif output == directory / "launch-receipt.json":
        raise GateFailure("gate output overlaps the launch receipt")
    return output


def _load_launch_if_available(directory: Path) -> Mapping[str, Any] | None:
    path = directory / "launch-receipt.json"
    try:
        return _load_json(path, "launch receipt")
    except (OSError, TypeError, ValueError):
        return None


def _same_owner_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return dict(left) == dict(right)


def _rebind_immutable_evidence(directory: Path, evidence: Mapping[str, Any]) -> None:
    expected_launch = evidence.get("launch_binding")
    expected_freeze = evidence.get("freeze_binding")
    if not isinstance(expected_launch, Mapping) or not isinstance(
        expected_freeze, Mapping
    ):
        raise GateFailure("gate evidence omitted immutable input bindings")
    for label, path, expected_binding in (
        (
            "launch receipt",
            directory / "launch-receipt.json",
            expected_launch,
        ),
        (
            "critic parameter freeze manifest",
            Path(str(expected_freeze.get("path", ""))),
            expected_freeze,
        ),
    ):
        _payload, observed = _load_required_bound_json(path, label)
        if dict(observed) != dict(expected_binding):
            raise GateFailure(f"{label} drifted before PASS publication")


def _terminate_authenticated_owner(
    lease: Mapping[str, Any],
    *,
    dry_run: bool,
    proc_root: Path,
    stop_timeout_seconds: float,
    signal_owner: Callable[[Mapping[str, Any], int], bool],
) -> dict[str, Any]:
    identity = lease["identity"]
    if dry_run:
        return {
            "requested": False,
            "signal": None,
            "scope": "exact-owner-identity",
            "status": "suppressed-by-dry-run",
        }
    if not _owner_is_live(lease, proc_root):
        return {
            "requested": False,
            "signal": None,
            "scope": "exact-owner-identity",
            "status": "owner-identity-changed-before-signal",
        }
    sent = signal_owner(identity, signal.SIGTERM)
    if not sent:
        return {
            "requested": True,
            "signal": "SIGTERM",
            "scope": "exact-owner-identity",
            "status": "already-exited",
        }
    deadline = time.monotonic() + max(0.0, stop_timeout_seconds)
    while time.monotonic() < deadline and _owner_is_live(lease, proc_root):
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    stopped = not _owner_is_live(lease, proc_root)
    return {
        "requested": True,
        "signal": "SIGTERM",
        "scope": "exact-owner-identity",
        "status": "confirmed-stopped" if stopped else "signal-sent-not-confirmed",
    }


def run_update1_gate(
    run_dir: str | os.PathLike[str],
    expected: ExpectedRunIdentity,
    *,
    owner_receipt: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    timeout_seconds: float = 1800.0,
    poll_seconds: float = 1.0,
    stop_timeout_seconds: float = 120.0,
    dry_run: bool = False,
    proc_root: str | os.PathLike[str] = "/proc",
    signal_owner: Callable[[Mapping[str, Any], int], bool] = _signal_identity,
) -> dict[str, Any]:
    """Wait for update 1, publish a receipt, and stop only an exact failed owner."""

    if timeout_seconds < 0 or poll_seconds <= 0 or stop_timeout_seconds < 0:
        raise ValueError(
            "timeouts must be nonnegative and poll_seconds must be positive"
        )
    _validate_expected_identity(expected)
    directory = Path(run_dir).resolve()
    output = _absolute_without_symlink_resolution(output_path)
    owner_path = _absolute_without_symlink_resolution(owner_receipt)
    proc_path = Path(proc_root)
    started_at = _utc_now()
    deadline = time.monotonic() + timeout_seconds
    first_lease: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    failure: str | None = None
    last_pending: str | None = None

    while True:
        try:
            lease = authenticate_owner(
                owner_path, directory, expected.run_id, proc_root=proc_path
            )
            if first_lease is None:
                first_lease = lease
            elif not _same_owner_binding(first_lease["binding"], lease["binding"]):
                raise OwnerAuthenticationError(
                    "owner identity receipt drifted during gate"
                )
            evidence = audit_update1(directory, expected)
            break
        except GatePending as error:
            last_pending = str(error)
            if time.monotonic() >= deadline:
                failure = f"timed out waiting for update-1 evidence: {last_pending}"
                break
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
        except GateFailure as error:
            failure = str(error)
            break
        except Exception as error:  # noqa: BLE001 - unexpected evidence faults fail closed
            failure = f"unexpected update-1 gate error: {type(error).__name__}: {error}"
            break

    launch = _load_launch_if_available(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise GateFailure("cannot publish gate receipt before the run directory exists")
    output = _validate_output_path(directory, output, owner_path, launch)

    if failure is None:
        try:
            final_lease = authenticate_owner(
                owner_path, directory, expected.run_id, proc_root=proc_path
            )
            if first_lease is None or not _same_owner_binding(
                first_lease["binding"], final_lease["binding"]
            ):
                raise OwnerAuthenticationError(
                    "owner identity receipt drifted before PASS"
                )
            if evidence is None:
                raise GateFailure("update-1 audit returned no evidence")
            _rebind_immutable_evidence(directory, evidence)
        except GateFailure as error:
            failure = str(error)
        else:
            receipt = {
                "schema": _GATE_SCHEMA,
                "status": "pass",
                "decision": "PASS_CONTINUE",
                "run_id": expected.run_id,
                "snapshot_update": 1,
                "dry_run": dry_run,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "owner": {
                    "identity": final_lease["identity"],
                    "receipt_binding": final_lease["binding"],
                    "receipt_path": str(owner_path),
                },
                "field_evidence_paths": {
                    **evidence["evidence_paths"],
                    "formal_owner_identity": str(owner_path),
                    "live_owner_process": f"{proc_path}/{final_lease['identity']['pid']}",
                },
                "evidence": evidence,
                "termination": {
                    "requested": False,
                    "signal": None,
                    "scope": "exact-owner-identity",
                    "status": "not-requested",
                },
                "errors": [],
            }
            _atomic_json(output, receipt)
            return receipt

    termination = {
        "requested": False,
        "signal": None,
        "scope": "exact-owner-identity",
        "status": "unsafe-to-stop-owner-not-authenticated",
    }
    final_lease: dict[str, Any] | None = None
    if first_lease is not None:
        try:
            candidate = authenticate_owner(
                owner_path, directory, expected.run_id, proc_root=proc_path
            )
            if not _same_owner_binding(first_lease["binding"], candidate["binding"]):
                raise OwnerAuthenticationError(
                    "owner identity receipt drifted before stop"
                )
            final_lease = candidate
        except GateFailure as error:
            failure = f"{failure}; exact stop withheld: {error}"
    if final_lease is not None:
        try:
            termination = _terminate_authenticated_owner(
                final_lease,
                dry_run=dry_run,
                proc_root=proc_path,
                stop_timeout_seconds=stop_timeout_seconds,
                signal_owner=signal_owner,
            )
        except (FallbackError, OSError, RuntimeError) as error:
            termination = {
                "requested": True,
                "signal": "SIGTERM",
                "scope": "exact-owner-identity",
                "status": "signal-error",
                "error": f"{type(error).__name__}: {error}",
            }

    receipt = {
        "schema": _GATE_SCHEMA,
        "status": "fail",
        "decision": "FAIL_STOP" if termination["requested"] else "FAIL_NO_UNSAFE_STOP",
        "run_id": expected.run_id,
        "snapshot_update": 1,
        "dry_run": dry_run,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "owner": (
            {
                "identity": final_lease["identity"],
                "receipt_binding": final_lease["binding"],
                "receipt_path": str(owner_path),
            }
            if final_lease is not None
            else {"receipt_path": str(owner_path), "authenticated": False}
        ),
        "field_evidence_paths": {
            "formal_owner_identity": str(owner_path),
            **(
                evidence.get("evidence_paths", {})
                if isinstance(evidence, Mapping)
                else {}
            ),
        },
        "evidence": evidence,
        "termination": termination,
        "errors": [failure or "unknown update-1 gate failure"],
    }
    _atomic_json(output, receipt)
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed, stop-capable AMG update-1 token-credit gate"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--owner-receipt", type=Path, required=True)
    parser.add_argument("--expected-outer-commit", required=True)
    parser.add_argument("--expected-inner-commit", required=True)
    parser.add_argument("--expected-verl-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--stop-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    expected = ExpectedRunIdentity(
        run_id=args.run_id,
        outer_commit=args.expected_outer_commit,
        inner_commit=args.expected_inner_commit,
        verl_commit=args.expected_verl_commit,
    )
    receipt = run_update1_gate(
        args.run_dir,
        expected,
        owner_receipt=args.owner_receipt,
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
        stop_timeout_seconds=args.stop_timeout_seconds,
        dry_run=args.dry_run,
    )
    return 0 if receipt.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExpectedRunIdentity",
    "GateFailure",
    "GatePending",
    "OwnerAuthenticationError",
    "audit_update1",
    "authenticate_owner",
    "run_update1_gate",
]
