"""Fail-closed attestation for one native veRL fully-asynchronous AMG run.

The finalizer joins artifacts owned by their native layers instead of making the
generic veRL receipt carry OpenMLE fields: launch/publication identity, resolved
Hydra config, FileLogger metrics, real-row rollout dumps, generic runtime queue
snapshots, and native actor/critic checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import shlex
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .config_contract import inspect_schedule, verify_resolved_config
from .identity import (
    EXPECTED_VERL_COMMIT,
    LOCKED_MODEL_FILE_SHA256,
    sha256_file,
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
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


def _lcm(left: int, right: int) -> int:
    return abs(left * right) // math.gcd(left, right)


class _Audit:
    def __init__(self, run_dir: Path, trainer_exit_code: int) -> None:
        self.run_dir = run_dir
        self.trainer_exit_code = int(trainer_exit_code)
        self.errors: list[str] = []
        self.mode: str | None = None
        self.role: str | None = None
        self.expected: Mapping[str, Any] | None = None
        self.launch: Mapping[str, Any] | None = None
        self.runtime: Mapping[str, Any] | None = None
        self.batch_multiple: int | None = None
        self.staleness_threshold: float | None = None
        self.trainer_world_size: int | None = None
        self.counts: dict[str, int] = {
            "scheduled_episodes": 0,
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
        }

    def check(self, condition: bool, message: str) -> bool:
        if not condition:
            self.errors.append(message)
        return condition

    def error(self, label: str, exc: BaseException) -> None:
        self.errors.append(f"{label}: {exc}")

    def audit_launch(self) -> None:
        path = self.run_dir / "launch-receipt.json"
        try:
            launch = _load_json(path, "launch receipt")
        except Exception as exc:
            self.error("launch receipt", exc)
            return
        self.launch = launch
        self.check(
            launch.get("schema") == "amg_verl_fully_async_launch_receipt_v4",
            "launch receipt schema must be amg_verl_fully_async_launch_receipt_v4",
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
        runtime_artifacts = {
            "native_receipt": self.run_dir / "native-runtime-receipt.json",
            "file_logger": self.run_dir / "metrics.jsonl",
            "rollout_data": self.run_dir / "rollout_data",
            "hydra_config": self.run_dir / "hydra" / ".hydra" / "config.yaml",
            "checkpoints": self.run_dir / "checkpoints",
            "finalization": self.run_dir / "finalization.json",
        }
        for field, wanted_path in runtime_artifacts.items():
            self.check(
                _same_path(_at(launch, f"runtime_artifacts.{field}"), wanted_path),
                f"launch runtime artifact {field} path mismatch",
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
                complete_learner_updates=updates,
                publication_cycles=publications,
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

    def audit_config(self) -> None:
        if self.launch is None or self.expected is None or self.mode is None:
            return
        resolved_path = self.run_dir / "resolved-config.yaml"
        hydra_path = self.run_dir / "hydra" / ".hydra" / "config.yaml"
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
        self.check(
            _same_path(
                _at(resolved, "async_training.runtime_receipt_path"),
                self.run_dir / "native-runtime-receipt.json",
            ),
            "resolved config native runtime receipt path mismatch",
        )

        resolved_endpoint = _at(resolved, "actor_rollout_ref.agentgym")
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
                inspect_schedule(
                    schedule_path,
                    expected_count=int(self.expected["episodes"]),
                    expected_sha256=str(self.expected["schedule_sha256"]),
                    expected_role=str(self.expected["role"]),
                )
                self._schedule_ids = [
                    str(json.loads(line)["item_id"])
                    for line in schedule_path.read_text(encoding="utf-8").splitlines()
                ]
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
        path = self.run_dir / "metrics.jsonl"
        try:
            rows = _jsonl(path, "FileLogger JSONL")
        except Exception as exc:
            self.error("FileLogger JSONL", exc)
            return
        validation_metrics = 0
        bypass_evidence_rows = 0
        actor_probe_rows = 0
        critic_probe_rows = 0
        bypass_real_tokens = 0
        current_param_versions: dict[int, int] = {}
        native_stale_action_rows: dict[int, int] = {}
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
            for key, value in data.items():
                folded = str(key).casefold()
                if "validation" in folded or folded.startswith(
                    ("val/", "val_", "val-")
                ):
                    validation_metrics += 1
        self.check(
            validation_metrics == 0,
            f"FileLogger emitted {validation_metrics} validation metric(s)",
        )
        if self.expected is not None:
            publications = int(self.expected["publication_cycles"])

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
                    self.check(
                        len(grad_norms) == 1 and _finite_positive(grad_norms[0]),
                        f"FileLogger publication step {step} has no unique nonzero "
                        f"{role} update metric",
                    )

                bypass_counts = [
                    value
                    for key, value in items
                    if key == "rollout_corr/bypass_real_token_count"
                ]
                bypass_diffs = [
                    value
                    for key, value in items
                    if key == "rollout_corr/bypass_max_abs_diff"
                ]
                self.check(
                    len(bypass_counts) == 1 and _finite_positive(bypass_counts[0]),
                    f"FileLogger publication step {step} has no unique compared real-token count",
                )
                self.check(
                    len(bypass_diffs) == 1 and bypass_diffs[0] in (0, 0.0),
                    f"FileLogger publication step {step} reports an old/rollout logprob mismatch",
                )
                if len(bypass_counts) == 1 and _finite_positive(bypass_counts[0]):
                    bypass_evidence_rows += 1
                    bypass_real_tokens += int(bypass_counts[0])

                actor_probes = [
                    value
                    for key, value in items
                    if key == "parameter_update_probe/actor/changed"
                ]
                critic_probes = [
                    value
                    for key, value in items
                    if key == "parameter_update_probe/critic/changed"
                ]
                self.check(
                    len(actor_probes) == 1
                    and actor_probes[0] in (True, 1, 1.0),
                    f"FileLogger publication step {step} has no unique actor parameter-update probe",
                )
                self.check(
                    len(critic_probes) == 1
                    and critic_probes[0] in (True, 1, 1.0),
                    f"FileLogger publication step {step} has no unique critic parameter-update probe",
                )
                actor_probe_rows += int(
                    len(actor_probes) == 1
                    and actor_probes[0] in (True, 1, 1.0)
                )
                critic_probe_rows += int(
                    len(critic_probes) == 1
                    and critic_probes[0] in (True, 1, 1.0)
                )

                current_versions = [
                    value
                    for key, value in items
                    if key == "fully_async/count/current_param_version"
                ]
                current_version = (
                    _nonnegative_integral(current_versions[0])
                    if len(current_versions) == 1
                    else None
                )
                self.check(
                    current_version is not None,
                    f"FileLogger publication step {step} has no unique integral "
                    "current parameter version",
                )
                if current_version is not None:
                    current_param_versions[step] = current_version

                stale_counts = [
                    value
                    for key, value in items
                    if key == "fully_async/count/stale_trajectory_processed"
                ]
                stale_count = (
                    _nonnegative_integral(stale_counts[0])
                    if len(stale_counts) == 1
                    else None
                )
                self.check(
                    stale_count is not None,
                    f"FileLogger publication step {step} has no unique integral "
                    "native stale action-row count",
                )
                if stale_count is not None:
                    native_stale_action_rows[step] = stale_count

            self.check(
                bypass_evidence_rows == publications,
                "FileLogger old/rollout logprob evidence does not cover every publication cycle",
            )
            self.check(
                actor_probe_rows == publications,
                "FileLogger actor parameter-update probe does not cover every publication cycle",
            )
            self.check(
                critic_probe_rows == publications,
                "FileLogger critic parameter-update probe does not cover every publication cycle",
            )
            if len(current_param_versions) == publications:
                self.check(
                    [current_param_versions[step] for step in range(1, publications + 1)]
                    == list(range(publications)),
                    "FileLogger current parameter versions do not match publication order",
                )
            if len(native_stale_action_rows) == publications:
                stale_sequence = [
                    native_stale_action_rows[step]
                    for step in range(1, publications + 1)
                ]
                self.check(
                    stale_sequence == sorted(stale_sequence),
                    "FileLogger native stale action-row count is not cumulative",
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
            "bypass_evidence_rows": bypass_evidence_rows,
            "actor_probe_rows": actor_probe_rows,
            "critic_probe_rows": critic_probe_rows,
            "bypass_real_tokens": bypass_real_tokens,
            "current_param_versions_by_update": current_param_versions_by_update,
            "native_stale_action_rows_by_publication": native_stale_action_rows,
        }

    @staticmethod
    def _policy_action(
        record: Mapping[str, Any], document: Mapping[str, Any]
    ) -> Any | None:
        action = record.get("action")
        if not (
            isinstance(action, str)
            and action != ""
            and _at(record, "action_submission.raw_policy_output") == action
            and document.get("output") == action
        ):
            return None
        from agentenv_openmle_fast.actions import parse_policy_action

        parsed = parse_policy_action(action)
        if parsed.kind == "parser_error":
            return None
        info = record.get("env_info_after")
        if not isinstance(info, Mapping) or info.get("action_kind") != parsed.kind:
            return None
        return parsed

    @staticmethod
    def _safe_relative_path(value: Any) -> str | None:
        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
            or "\\" in value
        ):
            return None
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            return None
        normalized = path.as_posix()
        return normalized if normalized == value else None

    @staticmethod
    def _successful_action_execution(
        record: Mapping[str, Any], parsed: Any
    ) -> Mapping[str, Any] | None:
        info = record.get("env_info_after")
        if not isinstance(info, Mapping):
            return None
        execution = info.get("execution")
        if not isinstance(execution, Mapping):
            return None
        if (
            info.get("action_kind") != parsed.kind
            or info.get("action_status") != "completed"
            or execution.get("action_kind") != parsed.kind
            or execution.get("status") != "completed"
        ):
            return None
        if parsed.kind == "shell_command":
            if execution.get("exit_code") != 0:
                return None
        elif parsed.kind == "apply_patch":
            if execution.get("exit_code") is not None:
                return None
        else:
            return None
        return execution

    @classmethod
    def _successful_managed_execution(
        cls, record: Mapping[str, Any], parsed: Any
    ) -> Mapping[str, Any] | None:
        if parsed.kind != "shell_command":
            return None
        execution = cls._successful_action_execution(record, parsed)
        if execution is None:
            return None
        info = record["env_info_after"]
        counters = info.get("counter_delta")
        if not isinstance(counters, Mapping):
            return None
        positive_counts = (
            counters.get("execution_action_count"),
            counters.get("execution_attempt_count"),
            counters.get("execution_completed_count"),
            execution.get("execution_action_delta"),
            execution.get("execution_attempt_delta"),
            execution.get("execution_completed_delta"),
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in positive_counts
        ):
            return None
        return execution

    @staticmethod
    def _shell_tokens(parsed: Any) -> list[str] | None:
        arguments = getattr(parsed, "arguments", None)
        command = arguments.get("command") if isinstance(arguments, Mapping) else None
        if not isinstance(command, str):
            return None
        try:
            return shlex.split(command, comments=False, posix=True)
        except ValueError:
            return None

    @classmethod
    def _exact_document_read(cls, parsed: Any, path: str) -> bool:
        tokens = cls._shell_tokens(parsed)
        return tokens in (["cat", path], ["cat", "--", path])

    @classmethod
    def _exact_managed_execution(cls, parsed: Any, path: str) -> bool:
        tokens = cls._shell_tokens(parsed)
        return tokens in (["python", path], ["python3", path])

    @classmethod
    def _continuation_code_path(cls, stdout: Any) -> str | None:
        if not isinstance(stdout, str) or not stdout.strip():
            return None
        fields: dict[str, str] = {}
        for line in stdout.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().casefold()
            value = value.strip()
            if not key or not value or key in fields:
                return None
            fields[key] = value
        required = {"objective", "conclusion", "code_path", "next_action"}
        if not required.issubset(fields):
            return None
        measurement_fields = {
            "measured_validation",
            "measured_validation_or_failure",
            "validation",
        }
        if not measurement_fields.intersection(fields):
            return None
        return cls._safe_relative_path(fields["code_path"])

    @staticmethod
    def _changed_paths(execution: Mapping[str, Any]) -> set[str] | None:
        changed = execution.get("changed_paths")
        if not isinstance(changed, Sequence) or isinstance(changed, (str, bytes)):
            return None
        paths = {str(path) for path in changed}
        return paths if len(paths) == len(changed) else None

    @classmethod
    def _has_memory_chain(
        cls, episode: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ) -> bool:
        parsed_actions = [
            cls._policy_action(record, document) for record, document in episode
        ]
        for compact_index, (compact, _document) in enumerate(episode):
            parsed_compact = parsed_actions[compact_index]
            if parsed_compact is None:
                continue
            evidence = compact.get("wrapper_evidence")
            transition = compact.get("context_transition")
            control_request = compact.get("control_request")
            if not (
                isinstance(evidence, Mapping)
                and evidence.get("event") == "context_compaction"
                and evidence.get("continuation_persisted") is True
                and evidence.get("preserved_policy_output") is True
                and evidence.get("preserved_native_observation") is True
                and evidence.get("native_action_kind") == parsed_compact.kind
                and evidence.get("native_action_status") == "completed"
                and isinstance(transition, Mapping)
                and transition.get("operation") == "replace_messages"
                and isinstance(transition.get("messages"), Sequence)
                and not isinstance(transition.get("messages"), (str, bytes))
                and isinstance(control_request, str)
                and control_request.strip()
            ):
                continue
            external_path = cls._safe_relative_path(
                evidence.get("continuation_path")
            )
            if external_path is None:
                continue
            compact_execution = cls._successful_action_execution(
                compact, parsed_compact
            )
            if compact_execution is None:
                continue
            compact_paths = cls._changed_paths(compact_execution)
            if compact_paths is None or external_path not in compact_paths:
                continue

            for read_index in range(compact_index + 1, len(episode)):
                read = episode[read_index][0]
                parsed_read = parsed_actions[read_index]
                if parsed_read is None or not cls._exact_document_read(
                    parsed_read, external_path
                ):
                    continue
                read_execution = cls._successful_action_execution(read, parsed_read)
                if read_execution is None:
                    continue
                code_path = cls._continuation_code_path(
                    read_execution.get("stdout")
                )
                if code_path is None:
                    continue

                for reuse_index in range(read_index + 1, len(episode)):
                    reuse = episode[reuse_index][0]
                    parsed_reuse = parsed_actions[reuse_index]
                    if parsed_reuse is None:
                        continue
                    reuse_execution = cls._successful_action_execution(
                        reuse, parsed_reuse
                    )
                    if reuse_execution is None:
                        continue
                    reuse_paths = cls._changed_paths(reuse_execution)
                    if reuse_paths is None or code_path not in reuse_paths:
                        continue
                    for execute_index in range(reuse_index + 1, len(episode)):
                        execute = episode[execute_index][0]
                        parsed_execute = parsed_actions[execute_index]
                        if (
                            parsed_execute is not None
                            and cls._exact_managed_execution(
                                parsed_execute, code_path
                            )
                            and cls._successful_managed_execution(
                                execute, parsed_execute
                            )
                            is not None
                        ):
                            return True
        return False

    def audit_rollouts(self) -> None:
        directory = self.run_dir / "rollout_data"
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
        if not paths:
            self.errors.append(f"required rollout JSONL is missing under: {directory}")
            return
        if self.expected is not None:
            self.check(
                len(paths) == int(self.expected["optimizer_updates"]),
                "rollout JSONL file count does not match optimizer-update horizon",
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
        memory_chain_updates: list[int] = []
        seen_uids: set[str] = set()
        samples_per_update = (
            int(self.expected["samples_per_update"]) if self.expected else 0
        )
        trigger = (
            int(self.expected["trigger_parameter_sync_step"])
            if self.expected
            else 0
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
            for document in documents:
                is_padding = document.get("is_padding", False)
                if not isinstance(is_padding, bool):
                    self.errors.append("rollout JSONL row has non-boolean is_padding")
                    continue
                if is_padding:
                    self.errors.append(
                        "native rollout JSONL contains a synthetic padding row"
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

                uid = record.get("trajectory_uid")
                order = record.get("trajectory_row_order")
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
                        self.check(
                            done is True,
                            f"trajectory {uid!r} terminal row is not done",
                        )
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
                else:
                    completed_episodes.append(sorted_episode)
                    terminal_id = sorted_episode[0][0].get("item_id")
                    if isinstance(terminal_id, str) and terminal_id:
                        terminal_ids.append(terminal_id)
                seen_uids.add(uid)
            self.check(
                len(completed_episodes) == samples_per_update,
                f"rollout update {ordinal} terminal trajectories per learner update "
                f"do not equal {samples_per_update}",
            )
            if any(self._has_memory_chain(episode) for episode in completed_episodes):
                memory_chain_updates.append(ordinal)
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
            self.check(
                file_logger_summary.get("bypass_real_tokens") == real_tokens,
                "FileLogger old/rollout logprob real-token total differs from rollout data",
            )
            native_stale = file_logger_summary.get(
                "native_stale_action_rows_by_publication"
            )
            if isinstance(native_stale, Mapping):
                for publication, reconstructed in stale_action_rows_by_publication.items():
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

        self.counts.update(
            real_action_rows=real_rows,
            derived_padding_action_rows=derived_padding_rows,
            real_response_tokens=real_tokens,
            stale_action_rows=stale_action_rows,
            memory_chains=memory_chains,
            late_memory_chains=late_memory_chains,
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
        }
        schedule_ids = getattr(self, "_schedule_ids", None)
        if schedule_ids is not None:
            self.check(
                Counter(terminal_ids) == Counter(schedule_ids),
                "rollout terminal trajectory identity/occurrences differ from the publication schedule",
            )

    @staticmethod
    def _component_statistics(
        runtime: Mapping[str, Any], boundary: str, component: str
    ) -> Mapping[str, Any] | None:
        record = _at(runtime, f"snapshots.{boundary}.{component}")
        if not isinstance(record, Mapping) or record.get("available") is not True:
            return None
        statistics = record.get("statistics")
        return statistics if isinstance(statistics, Mapping) else None

    @staticmethod
    def _queue_conserved(statistics: Mapping[str, Any]) -> bool:
        fields = (
            "real_enqueued",
            "real_consumed",
            "real_evicted",
            "real_cleared",
            "real_resident",
        )
        if any(not isinstance(statistics.get(field), int) for field in fields):
            return False
        return statistics["real_enqueued"] == sum(
            statistics[field]
            for field in (
                "real_consumed",
                "real_evicted",
                "real_cleared",
                "real_resident",
            )
        )

    def audit_runtime(self) -> None:
        path = self.run_dir / "native-runtime-receipt.json"
        try:
            wrapper = _load_json(path, "native runtime receipt")
        except Exception as exc:
            self.error("native runtime receipt", exc)
            return
        runtime = wrapper.get("data")
        self.check(
            isinstance(wrapper.get("step"), int)
            and not isinstance(wrapper.get("step"), bool),
            "native runtime FileLogger wrapper has no integer step",
        )
        if not isinstance(runtime, Mapping):
            self.errors.append("native runtime receipt has no FileLogger data mapping")
            return
        self.runtime = runtime
        self.check(
            runtime.get("schema_version") == 1,
            "native runtime receipt schema_version mismatch",
        )
        self.check(
            runtime.get("outcome") == "success", "native runtime outcome is not success"
        )
        self.check(
            runtime.get("status") == "completed",
            "native runtime status is not completed",
        )
        self.check(
            runtime.get("exception") is None,
            "native runtime exception must be null on success",
        )
        self.check(
            runtime.get("finalization_errors") == [],
            "native runtime finalization_errors must be empty on success",
        )
        timestamps = runtime.get("timestamps")
        self.check(
            isinstance(timestamps, Mapping)
            and all(
                timestamps.get(key)
                for key in ("run_started_at", "finalization_started_at", "finalized_at")
            ),
            "native runtime timestamps are incomplete",
        )

        self.check(
            isinstance(runtime.get("snapshots"), Mapping),
            "native runtime snapshots mapping is missing",
        )
        boundaries: dict[str, dict[str, Mapping[str, Any]]] = {}
        for boundary in ("before_clear", "after_clear"):
            boundaries[boundary] = {}
            for component in ("trainer", "rollouter", "queue"):
                statistics = self._component_statistics(runtime, boundary, component)
                if statistics is None:
                    self.errors.append(
                        f"native runtime {boundary}.{component} statistics are unavailable"
                    )
                    statistics = {}
                boundaries[boundary][component] = statistics
        before = boundaries["before_clear"]
        after = boundaries["after_clear"]

        flags = runtime.get("queue_conservation")
        self.check(
            isinstance(flags, Mapping),
            "native runtime queue_conservation mapping is missing",
        )
        for flag in ("before_clear", "after_clear", "clear_delta_matches_resident"):
            self.check(
                isinstance(flags, Mapping) and flags.get(flag) is True,
                f"native runtime queue_conservation.{flag} did not pass",
            )
        for boundary, stats in (
            ("before_clear", before["queue"]),
            ("after_clear", after["queue"]),
        ):
            self.check(
                self._queue_conserved(stats),
                f"native queue accounting is not conserved at {boundary}",
            )
        self.check(
            after["queue"].get("real_resident") == 0,
            "native queue retained samples after clear",
        )
        self.check(
            after["queue"].get("real_cleared", 0)
            - before["queue"].get("real_cleared", 0)
            == before["queue"].get("real_resident", 0),
            "native queue clear delta does not match its resident samples",
        )

        if self.expected is None:
            return
        episodes = int(self.expected["episodes"])
        publications = int(self.expected["publication_cycles"])
        samples_per_update = int(self.expected["samples_per_update"])
        trainer = before["trainer"]
        rollouter = before["rollouter"]
        queue = before["queue"]
        self.check(
            runtime.get("trainer_step") == publications
            and wrapper.get("step") == publications,
            "native runtime FileLogger wrapper step/trainer_step mismatch",
        )
        for field, wanted in (
            ("global_steps", int(self.expected["optimizer_updates"]) + 1),
            ("current_param_version", publications),
            ("total_train_steps", publications),
            ("local_trigger_step", 1),
            ("processed_samples", episodes),
            ("terminal_underfill_events", 0),
            ("terminal_underfill_samples", 0),
            ("pending_rollout_dump_writes", 0),
        ):
            self.check(
                trainer.get(field) == wanted,
                f"native trainer {field} mismatch",
            )
        stale_processed = trainer.get("stale_trajectory_processed")
        valid_stale_processed = (
            isinstance(stale_processed, int)
            and not isinstance(stale_processed, bool)
            and 0 <= stale_processed <= self.counts["real_action_rows"]
        )
        self.check(
            valid_stale_processed,
            "native trainer stale_trajectory_processed is not a valid action-row count",
        )
        # veRL retains the historical field name, but AMG expands every trajectory
        # into action rows before PPO.  Upstream therefore evaluates one
        # trajectory_param_versions entry per action row here.
        self.check(
            valid_stale_processed
            and stale_processed == self.counts["stale_action_rows"],
            "native trainer stale_trajectory_processed differs from the "
            "independently reconstructed stale action-row count",
        )
        bypass = trainer.get("latest_bypass_log_prob_evidence")
        self.check(
            isinstance(bypass, Mapping)
            and _finite_positive(bypass.get("rollout_corr/bypass_real_token_count"))
            and bypass.get("rollout_corr/bypass_max_abs_diff") in (0, 0.0),
            "native trainer latest bypass real-token count/old/rollout logprob evidence failed",
        )
        probes = trainer.get("latest_parameter_update_probe")
        for role in ("actor", "critic"):
            evidence = probes.get(role) if isinstance(probes, Mapping) else None
            self.check(
                isinstance(evidence, Mapping)
                and evidence.get("changed") is True
                and _finite_positive(evidence.get("changed_elements"))
                and _finite_positive(evidence.get("sampled_elements"))
                and evidence.get("worker_count") == self.trainer_world_size,
                f"native trainer latest {role} parameter-update probe failed",
            )

        for field, wanted in (
            ("monitor/active_tasks_size", 0),
            ("monitor/queue/pending_queue_size", 0),
            ("monitor/queue/mq_queue_size", 0),
            ("count/total_generated_samples", episodes),
            ("count/dropped_stale_samples", 0),
            ("static/required_samples", samples_per_update),
        ):
            self.check(
                rollouter.get(field) == wanted,
                f"native rollouter {field} mismatch",
            )
        staleness_samples = rollouter.get("count/staleness_samples")
        self.check(
            isinstance(staleness_samples, int)
            and not isinstance(staleness_samples, bool)
            and 0 <= staleness_samples <= episodes,
            "native rollouter count/staleness_samples is invalid",
        )
        if self.staleness_threshold is not None:
            expected_max_required = int(
                samples_per_update
                * (self.staleness_threshold + 1.0)
                * int(self.expected["trigger_parameter_sync_step"])
            )
            self.check(
                rollouter.get("static/staleness_threshold") == self.staleness_threshold,
                "native rollouter static/staleness_threshold mismatch",
            )
            self.check(
                rollouter.get("static/max_required_samples") == expected_max_required,
                "native rollouter static/max_required_samples mismatch",
            )
            self.check(
                rollouter.get("static/max_queue_size") == expected_max_required,
                "native rollouter static/max_queue_size mismatch",
            )
            max_concurrent = rollouter.get("static/max_concurrent_samples")
            self.check(
                isinstance(max_concurrent, int)
                and not isinstance(max_concurrent, bool)
                and 0 < max_concurrent <= expected_max_required,
                "native rollouter static/max_concurrent_samples is invalid",
            )

        queue_expectations = (
            ("total_produced", episodes),
            ("total_consumed", episodes),
            ("real_enqueued", episodes),
            ("real_consumed", episodes),
            ("real_evicted", 0),
            ("real_cleared", 0),
            ("real_resident", 0),
            ("queue_size", 0),
            ("dropped_samples", 0),
            ("closed", True),
            ("control_signals_enqueued", 0),
        )
        for field, wanted in queue_expectations:
            self.check(
                queue.get(field) == wanted,
                f"native queue {field} mismatch",
            )
        # Native popleft eviction is a valid freshness mechanism in general.  This
        # publication has no replacement horizon, so every eviction/drop would
        # remove one exact scheduled occurrence and must fail treatment accounting.
        self.check(
            rollouter.get("count/dropped_stale_samples") == 0,
            "native rollouter count/dropped_stale_samples violates the no-replacement budget",
        )

        self.check(
            before["trainer"] == after["trainer"],
            "native before/after trainer statistics changed during finalization",
        )
        self.check(
            before["rollouter"] == after["rollouter"],
            "native before/after rollouter statistics changed during finalization",
        )

    def audit_checkpoint(self) -> None:
        if self.expected is None:
            return
        if self.trainer_world_size is None:
            self.errors.append("checkpoint trainer world size is unavailable")
            return
        world_size = self.trainer_world_size
        expected_step = int(self.expected["publication_cycles"])
        root = self.run_dir / "checkpoints"
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
        if self.runtime is not None:
            outcome = self.runtime.get("outcome")
            status = self.runtime.get("status")
            if outcome == "terminal_underfill" or status == "partial":
                return "partial"
            if outcome != "success" or status != "completed":
                return "crash"
        return "success"

    def run(self) -> dict[str, Any]:
        if self.trainer_exit_code != 0:
            self.errors.append(f"trainer exit code {self.trainer_exit_code} is nonzero")
        self.audit_launch()
        self.audit_config()
        self.audit_file_logger()
        self.audit_rollouts()
        self.audit_runtime()
        self.audit_checkpoint()
        terminal_path = self.terminal_path()
        if terminal_path == "partial" and not any(
            "underfill" in error.casefold() for error in self.errors
        ):
            self.errors.append("native runtime ended on a partial terminal path")
        return {
            "schema": "amg_verl_fully_async_finalization_v2",
            "status": "pass" if not self.errors else "fail",
            "terminal_path": terminal_path,
            "trainer_exit_code": self.trainer_exit_code,
            "mode": self.mode,
            "role": self.role,
            "counts": self.counts,
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
            "schema": "amg_verl_fully_async_finalization_v2",
            "status": "fail",
            "terminal_path": audit.terminal_path(),
            "trainer_exit_code": audit.trainer_exit_code,
            "mode": audit.mode,
            "role": audit.role,
            "counts": audit.counts,
            "errors": audit.errors,
        }
    _atomic_json(directory / "finalization.json", verdict)
    return verdict


__all__ = ["finalize_run"]
