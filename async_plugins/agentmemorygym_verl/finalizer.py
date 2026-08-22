"""Fail-closed attestation for one native veRL fully-asynchronous AMG run.

The finalizer joins artifacts owned by their native layers instead of making the
generic veRL receipt carry OpenMLE fields: launch/publication identity, resolved
Hydra config, native FileLogger metrics, real-row rollout dumps, and native
actor/critic checkpoints.  It deliberately does not depend on the retired local
runtime-receipt / parameter-probe patch that current upstream veRL does not consume.
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
            "completed_episodes": 0,
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
            launch.get("schema") == "amg_verl_fully_async_launch_receipt_v5",
            "launch receipt schema must be amg_verl_fully_async_launch_receipt_v5",
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
                schedule_rows = [
                    json.loads(line)
                    for line in schedule_path.read_text(encoding="utf-8").splitlines()
                ]
                self._schedule_ids = [str(row["item_id"]) for row in schedule_rows]
                self._schedule_instances = [
                    (str(row["item_id"]), int(row["data_idx"]))
                    for row in schedule_rows
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
        """Audit metrics emitted by current upstream veRL itself.

        The previous integration added a private runtime receipt and sampled
        parameter probes.  Current upstream has no consumers for those config
        keys, so this audit instead joins the native FileLogger signals that are
        produced by the actual learner/rollouter path: actor and critic gradient
        norms, rollout-correction diagnostics, policy/staleness versions, queue
        counters, rollout JSONL, and the complete optimizer checkpoint.
        """

        path = self.run_dir / "metrics.jsonl"
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
                if "validation" in folded or folded.startswith(("val/", "val_", "val-")):
                    validation_metrics += 1

        self.check(
            validation_metrics == 0,
            f"FileLogger emitted {validation_metrics} validation metric(s)",
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
                "FileLogger step 0 is not rollouter-only: " + ", ".join(step_zero_bad_keys),
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
                items = [(str(key), value) for data in data_rows for key, value in data.items()]

                for role in ("actor", "critic"):
                    grad_norms = [value for key, value in items if key == f"{role}/grad_norm"]
                    valid_grad = len(grad_norms) == 1 and _finite_positive(grad_norms[0])
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
                    parsed = _nonnegative_integral(values[0]) if len(values) == 1 else None
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
                    current_param_versions[step] = observed_integrals["current_param_version"]
                if "stale_trajectory_processed" in observed_integrals:
                    native_stale_action_rows[step] = observed_integrals["stale_trajectory_processed"]
                if "total_generated_samples" in observed_integrals:
                    generated_samples[step] = observed_integrals["total_generated_samples"]
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
                    [current_param_versions[step] for step in range(1, publications + 1)]
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
        code_path = cls._safe_relative_path(fields["code_path"])
        return code_path if code_path == "train.py" else None

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
        terminal_instances: list[tuple[str, int]] = []
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
                data_indices = {
                    record.get("data_idx") for record, _document in episode
                }
                valid_data_idx = (
                    len(data_indices) == 1
                    and all(
                        isinstance(value, int) and not isinstance(value, bool) and value >= 0
                        for value in data_indices
                    )
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
                    terminal_data_idx = sorted_episode[0][0].get("data_idx")
                    if isinstance(terminal_id, str) and terminal_id:
                        terminal_ids.append(terminal_id)
                        if valid_data_idx:
                            terminal_instances.append((terminal_id, terminal_data_idx))
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
            "schema": "amg_verl_fully_async_finalization_v3",
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
