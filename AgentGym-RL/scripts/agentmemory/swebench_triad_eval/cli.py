"""Guarded lifecycle and resume driver for the formal SWE-bench triad.

The generic paired runner deliberately knows nothing about benchmark lifecycle.
This deployment-only module owns fenced cell state, protected-artifact
dereferencing, official grading, gate-to-full transition, and owned cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from paired_eval.contracts import RunConfig
from paired_eval.serialization import canonical_json_bytes

from . import ARMS
from .atomic import (
    atomic_write_bytes,
    atomic_write_json,
    ensure_private_directory,
    read_json,
    write_immutable_json,
)
from .identity import PRODUCTION_DATASET_PINS, PRODUCTION_IMAGE_INDEX_PINS
from .state import (
    AlreadyAcceptedError,
    AlreadyGradedError,
    CellKey,
    CellStateStore,
    DriverLeaseRegistry,
    ManifestCell,
    OwnerIdentity,
    sha256_json,
)


PREFLIGHT_SCHEMA = "swebench_triad_preflight_snapshot_v1"
PREFLIGHT_PASS_SCHEMA = "swebench_triad_preflight_pass_v1"
GATE_PASS_SCHEMA = "swebench_triad_gate_pass_v1"
HEARTBEAT_SCHEMA = "swebench_triad_heartbeat_v1"
HARNESS_COMMIT = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
EVIDENCE_REFERENCE_RE = re.compile(
    r"\Aevidence://([a-z][a-z0-9_]*)/([0-9a-f]{64})\Z"
)
MAX_PRIVATE_JSON_BYTES = 64 * 1024 * 1024


class PreflightContractError(RuntimeError):
    """A required frozen or live identity failed closed."""


class LifecycleOperations(Protocol):
    def preflight(self) -> Mapping[str, Any]: ...

    def stage_task(self, task_index: int) -> Any: ...

    def reconcile_cell(
        self,
        config: RunConfig,
        *,
        generation: int,
        before_preflight: bool,
    ) -> Mapping[str, Any]: ...

    def reconcile_grade(
        self,
        *,
        key: CellKey,
        accepted: Mapping[str, Any],
        prediction: Mapping[str, Any],
        handoff: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def reconcile_startup(
        self, *, task_indices: Sequence[int]
    ) -> Mapping[str, Any]: ...

    def run_cell(
        self,
        config: RunConfig,
        stage: Any,
        *,
        generation: int,
    ) -> Mapping[str, Any]: ...

    def grade(
        self,
        *,
        key: CellKey,
        accepted: Mapping[str, Any],
        prediction: Mapping[str, Any],
        handoff: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def audit_residue(self, task_index: int) -> Mapping[str, Any]: ...

    def evict_task(self, task_index: int, stage: Any) -> Mapping[str, Any]: ...

    def cleanup(self) -> Mapping[str, Any]: ...

    def final_audit(self) -> Mapping[str, Any]: ...


def _fail(message: str) -> None:
    raise PreflightContractError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> Mapping[str, Any]:
    if set(value) != expected:
        _fail(f"{label} fields drifted")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} drifted")


def _true(value: Any, label: str) -> None:
    if value is not True:
        _fail(f"{label} did not pass")


def _zero(value: Any, label: str) -> None:
    if type(value) is not int or value != 0:
        _fail(f"{label} is not zero")


def validate_preflight_snapshot(
    snapshot: Mapping[str, Any], expectations: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the complete static/live admission snapshot.

    Collection is intentionally separate from validation so every negative
    boundary can be tested without weakening the production probes.
    """

    root = _exact_fields(
        _mapping(snapshot, "preflight snapshot"),
        {
            "source",
            "dataset",
            "image_index",
            "model",
            "blob_cache",
            "pod",
            "docker",
            "task4_negative_probes",
            "model_process",
            "vllm",
            "swe_metadata",
            "residue",
            "rootfs",
        },
        "preflight snapshot",
    )
    expected = _mapping(expectations, "preflight expectations")

    source = _exact_fields(
        _mapping(root["source"], "source snapshot"),
        {
            "deployment_commit",
            "inner_commit",
            "deployment_clean",
            "inner_clean",
            "protected_diff_zero",
        },
        "source snapshot",
    )
    _equal(
        source["deployment_commit"],
        expected.get("deployment_commit"),
        "deployment commit",
    )
    _equal(source["inner_commit"], expected.get("inner_commit"), "inner commit")
    for name in ("deployment_clean", "inner_clean", "protected_diff_zero"):
        _true(source[name], f"source {name}")

    dataset = _mapping(root["dataset"], "dataset snapshot")
    _equal(dataset.get("rows"), 500, "dataset row count")
    _equal(
        dataset.get("jsonl_sha256"),
        PRODUCTION_DATASET_PINS.jsonl_sha256,
        "dataset SHA-256",
    )
    _equal(
        dataset.get("id_ledger_sha256"),
        PRODUCTION_DATASET_PINS.id_ledger_sha256,
        "dataset ID ledger",
    )

    images = _mapping(root["image_index"], "image-index snapshot")
    _equal(images.get("rows"), 500, "image-index row count")
    for name in ("index_sha256", "tag_ledger_sha256", "digest_tsv_sha256"):
        _equal(
            images.get(name),
            getattr(PRODUCTION_IMAGE_INDEX_PINS, name),
            f"image-index {name}",
        )

    model = _mapping(root["model"], "model snapshot")
    _equal(model.get("file_count"), 14, "model file count")
    if not isinstance(model.get("file_ledger_sha256"), str) or not model[
        "file_ledger_sha256"
    ]:
        _fail("model file ledger is missing")

    blobs = _exact_fields(
        _mapping(root["blob_cache"], "blob-cache snapshot"),
        {
            "certificate_sha256",
            "revalidation_sha256",
            "descriptor_count",
            "file_count",
            "total_bytes",
            "downloaded_count",
            "verified_bad_count",
        },
        "blob-cache snapshot",
    )
    _equal(
        blobs["certificate_sha256"],
        expected.get("blob_certificate_sha256"),
        "blob certificate",
    )
    _equal(
        blobs["revalidation_sha256"],
        expected.get("blob_revalidation_sha256"),
        "blob revalidation",
    )
    _equal(blobs["descriptor_count"], 1158, "blob descriptor count")
    _equal(blobs["file_count"], 1158, "blob file count")
    _equal(blobs["total_bytes"], 117637519356, "blob byte count")
    _zero(blobs["downloaded_count"], "blob network downloads")
    _zero(blobs["verified_bad_count"], "bad blob count")

    pod = _exact_fields(
        _mapping(root["pod"], "pod snapshot"),
        {"job", "pod", "hostname", "boot_id", "gpu_uuid", "gpu_count"},
        "pod snapshot",
    )
    for name in ("job", "pod", "hostname", "boot_id", "gpu_uuid"):
        _equal(pod[name], expected.get(name), f"pod {name}")
    _equal(pod["gpu_count"], 1, "pod GPU count")

    docker = _exact_fields(
        _mapping(root["docker"], "Docker snapshot"),
        {
            "receipt_sha256",
            "daemon_id",
            "pid",
            "start_ticks",
            "version",
            "api_version",
            "cgroup_version",
            "cgroup_driver",
            "storage_driver",
            "containers",
            "images",
            "volumes",
        },
        "Docker snapshot",
    )
    _equal(
        docker["receipt_sha256"],
        expected.get("docker_receipt_sha256"),
        "Docker receipt",
    )
    for field, expected_name in (
        ("daemon_id", "docker_daemon_id"),
        ("pid", "docker_pid"),
        ("start_ticks", "docker_start_ticks"),
    ):
        _equal(docker[field], expected.get(expected_name), f"Docker {field}")
    _equal(docker["version"], "27.5.1", "Docker version")
    _equal(docker["api_version"], "1.47", "Docker API version")
    _equal(docker["cgroup_version"], "1", "Docker cgroup version")
    _equal(docker["cgroup_driver"], "cgroupfs", "Docker cgroup driver")
    _equal(docker["storage_driver"], "vfs", "Docker storage driver")
    for name in ("containers", "images", "volumes"):
        _zero(docker[name], f"Docker {name}")

    probes = _exact_fields(
        _mapping(root["task4_negative_probes"], "Task-4 probe snapshot"),
        {
            "receipt_sha256",
            "schema",
            "status",
            "network_downloads",
            "memory_exhaustion_blocked",
            "fork_exhaustion_blocked",
            "byte_quota_blocked",
            "inode_quota_blocked",
            "rootfs_mutation_detected",
            "cgroup_residue_absent",
            "tmpfs_residue_absent",
            "docker_residue_absent",
        },
        "Task-4 probe snapshot",
    )
    _equal(
        probes["receipt_sha256"],
        expected.get("task4_receipt_sha256"),
        "Task-4 probe receipt",
    )
    _equal(
        probes["schema"],
        "amg_swebench_task4_live_negative_probes_v1",
        "Task-4 probe schema",
    )
    _equal(probes["status"], "PASS", "Task-4 probe status")
    _zero(probes["network_downloads"], "Task-4 probe network downloads")
    for name in (
        "memory_exhaustion_blocked",
        "fork_exhaustion_blocked",
        "byte_quota_blocked",
        "inode_quota_blocked",
        "rootfs_mutation_detected",
        "cgroup_residue_absent",
        "tmpfs_residue_absent",
        "docker_residue_absent",
    ):
        _true(probes[name], f"Task-4 {name}")

    process = _exact_fields(
        _mapping(root["model_process"], "model-process snapshot"),
        {"pid", "start_ticks", "alive", "command_matches"},
        "model-process snapshot",
    )
    _equal(process["pid"], expected.get("model_pid"), "model PID")
    _equal(
        process["start_ticks"],
        expected.get("model_start_ticks"),
        "model PID start ticks",
    )
    _true(process["alive"], "model process liveness")
    _true(process["command_matches"], "model command identity")

    vllm = _exact_fields(
        _mapping(root["vllm"], "vLLM snapshot"),
        {
            "model_id",
            "prompt_token_ids",
            "response_token_ids",
            "repeat_prompt_token_ids",
            "repeat_response_token_ids",
            "repeat_text_equal",
        },
        "vLLM snapshot",
    )
    _equal(vllm["model_id"], expected.get("model_id"), "vLLM model ID")
    for name in (
        "prompt_token_ids",
        "response_token_ids",
        "repeat_prompt_token_ids",
        "repeat_response_token_ids",
    ):
        values = vllm[name]
        if (
            not isinstance(values, list)
            or not values
            or any(type(item) is not int or item < 0 for item in values)
        ):
            _fail(f"vLLM {name} is malformed")
    _equal(
        vllm["repeat_prompt_token_ids"],
        vllm["prompt_token_ids"],
        "vLLM deterministic prompt token IDs",
    )
    _equal(
        vllm["repeat_response_token_ids"],
        vllm["response_token_ids"],
        "vLLM deterministic response token IDs",
    )
    _true(vllm["repeat_text_equal"], "vLLM deterministic response text")

    metadata = _exact_fields(
        _mapping(root["swe_metadata"], "SWE metadata"),
        {
            "schema",
            "task_count",
            "full_benchmark_task_count",
            "supported_arms",
            "active_slot_count",
            "active_workspace_count",
            "official_grading_inside_adapter",
            "evaluation_max_policy_turns",
            "max_native_actions",
            "max_observation_tokens",
        },
        "SWE metadata",
    )
    _equal(
        metadata["schema"],
        "swebench_verified_external_patch_episode_v1",
        "SWE metadata schema",
    )
    for name in ("task_count", "full_benchmark_task_count"):
        _equal(metadata[name], 500, f"SWE {name}")
    _equal(metadata["supported_arms"], list(ARMS), "SWE arm lattice")
    _zero(metadata["active_slot_count"], "active SWE slots")
    _zero(metadata["active_workspace_count"], "active SWE workspaces")
    if metadata["official_grading_inside_adapter"] is not False:
        _fail("official grading entered the policy adapter")
    for name, value in (
        ("evaluation_max_policy_turns", 250),
        ("max_native_actions", 250),
        ("max_observation_tokens", 8192),
    ):
        _equal(metadata[name], value, f"SWE {name}")

    residue = _exact_fields(
        _mapping(root["residue"], "owned residue"),
        {
            "active_owned_processes",
            "active_cgroups",
            "active_tmpfs_mounts",
            "active_mounts",
            "active_scratch_paths",
            "loaded_task_images",
            "owned_containers",
        },
        "owned residue",
    )
    for name, value in residue.items():
        _zero(value, name)

    rootfs = _exact_fields(
        _mapping(root["rootfs"], "rootfs snapshot"),
        {"path", "pod_local"},
        "rootfs snapshot",
    )
    _true(rootfs["pod_local"], "pod-local rootfs")
    prefix = expected.get("rootfs_prefix")
    if not isinstance(prefix, str) or not str(rootfs["path"]).startswith(prefix):
        _fail("active rootfs escaped the pod-local prefix")

    return {
        "schema": PREFLIGHT_PASS_SCHEMA,
        "status": "PASS",
        "snapshot_sha256": sha256_json(root),
        "deployment_commit": source["deployment_commit"],
        "inner_commit": source["inner_commit"],
        "boot_id": pod["boot_id"],
        "gpu_uuid": pod["gpu_uuid"],
        "docker_daemon_id": docker["daemon_id"],
        "model_id": vllm["model_id"],
    }


def read_private_json(root: Path | str, protected_ref: str) -> Any:
    """Dereference one digest-addressed JSON value without trusting the row."""

    match = EVIDENCE_REFERENCE_RE.fullmatch(protected_ref or "")
    if match is None:
        _fail("protected evidence reference is malformed")
    category, digest = match.groups()
    evidence_root = Path(root)
    path = evidence_root / category / f"{digest}.json"
    try:
        info = path.lstat()
    except OSError as error:
        raise PreflightContractError("protected evidence is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _fail("protected evidence must be a real regular file")
    if info.st_size <= 0 or info.st_size > MAX_PRIVATE_JSON_BYTES:
        _fail("protected evidence size is invalid")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        _fail("protected evidence digest drifted")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PreflightContractError("protected evidence is invalid JSON") from error
    if canonical_json_bytes(value) != payload:
        _fail("protected JSON is not canonical")
    return value


def _load_accepted(store: CellStateStore, key: CellKey) -> Mapping[str, Any]:
    try:
        return store.read_accepted(key)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError("accepted cell record is invalid") from error


def _load_attempt_boundaries(
    store: CellStateStore, accepted: Mapping[str, Any], key: CellKey
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    try:
        artifacts = store.accepted_artifacts(key, accepted)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError("accepted attempt boundaries are invalid") from error
    values = tuple(artifacts[name] for name in ("endpoint", "prediction", "handoff"))
    if any(not isinstance(value, Mapping) for value in values):
        raise RuntimeError("accepted attempt boundary is invalid")
    return values


def _load_official_outcome(
    store: CellStateStore, key: CellKey
) -> Mapping[str, Any]:
    try:
        return store.read_official_outcome(key)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError("official outcome record is invalid") from error


class LifecycleDriver:
    """Fenced gate/full driver over an injected deployment operations layer."""

    def __init__(
        self,
        *,
        root: Path | str,
        configs: Sequence[RunConfig],
        owner: OwnerIdentity,
        owner_is_alive: Callable[[OwnerIdentity], bool],
        operations: LifecycleOperations,
        evidence_root: Path | str,
        endpoint_validator: Callable[[Any], None],
        triad_validator: Callable[[Sequence[Mapping[str, Any]]], Any],
        preflight_expectations: Mapping[str, Any],
        clock: Callable[[], float] = time.monotonic,
        assigned_task_indices: Sequence[int] | None = None,
        lease_registry: DriverLeaseRegistry | None = None,
    ) -> None:
        self.root = ensure_private_directory(root)
        self.configs = tuple(configs)
        self.operations = operations
        self.owner = owner
        self.owner_is_alive = owner_is_alive
        self.evidence_root = Path(evidence_root)
        if not callable(triad_validator):
            raise TypeError("lifecycle triad validator must be callable")
        self.triad_validator = triad_validator
        self.preflight_expectations = dict(preflight_expectations)
        self.clock = clock
        self.by_task = self._index_configs(self.configs)
        assigned = (
            tuple(sorted(self.by_task))
            if assigned_task_indices is None
            else tuple(assigned_task_indices)
        )
        if (
            not assigned
            or tuple(sorted(set(assigned))) != assigned
            or any(task_index not in self.by_task for task_index in assigned)
        ):
            raise ValueError("lifecycle assigned shard is invalid")
        if lease_registry is not None and (
            not isinstance(lease_registry, DriverLeaseRegistry)
            or lease_registry.owner != owner
            or lease_registry.assigned_task_indices != assigned
        ):
            raise ValueError("lifecycle lease registry does not bind its shard")
        self.assigned_task_indices = assigned
        self.lease_registry = lease_registry
        self.by_key = {
            CellKey(config.task.task_index, config.capability.arm.value): config
            for config in self.configs
        }
        manifest = tuple(
            ManifestCell(
                CellKey(config.task.task_index, config.capability.arm.value),
                config.task.task_id,
                config.full_config_sha256,
            )
            for config in self.configs
        )
        self.store = CellStateStore(
            self.root / "state",
            manifest=manifest,
            owner=owner,
            owner_is_alive=owner_is_alive,
            endpoint_validator=endpoint_validator,
        )
        ensure_private_directory(self.root / "control")
        ensure_private_directory(self.root / "gate")
        ensure_private_directory(self.root / "full")

    def ensure_driver_lease(self) -> None:
        if self.lease_registry is None:
            return
        self.lease_registry.start_heartbeat()
        self.lease_registry.assert_healthy()

    def acquire_runtime_lane(self, task_index: int | None) -> None:
        if self.lease_registry is None:
            return
        self.ensure_driver_lease()
        self.lease_registry.acquire_lane(task_index=task_index)

    def release_runtime_lane(self) -> None:
        if self.lease_registry is not None:
            self.lease_registry.release_lane()

    @staticmethod
    def _index_configs(
        configs: Sequence[RunConfig],
    ) -> dict[int, tuple[RunConfig, ...]]:
        if not configs or any(not isinstance(config, RunConfig) for config in configs):
            raise ValueError("lifecycle requires typed run configurations")
        by_task: dict[int, list[RunConfig]] = {}
        for config in configs:
            by_task.setdefault(config.task.task_index, []).append(config)
        expected_indices = list(range(len(by_task)))
        if sorted(by_task) != expected_indices:
            raise ValueError("manifest task indices are not contiguous from zero")
        result: dict[int, tuple[RunConfig, ...]] = {}
        for task_index in expected_indices:
            rows = tuple(by_task[task_index])
            if len(rows) != 3:
                raise ValueError("manifest task does not contain exactly three arms")
            if tuple(row.capability.arm.value for row in rows) != ARMS:
                raise ValueError("manifest task arm order drifted")
            if len({row.task.task_id for row in rows}) != 1:
                raise ValueError("manifest triad task identity drifted")
            if len({row.treatment_excluded_config_sha256 for row in rows}) != 1:
                raise ValueError("manifest triad treatment exclusion drifted")
            result[task_index] = rows
        return result

    @property
    def preflight_path(self) -> Path:
        return self.root / "control" / "preflight-PASS.json"

    @property
    def gate_path(self) -> Path:
        return self.root / "gate" / "PASS.json"

    @property
    def preflight_snapshot_path(self) -> Path:
        return self.root / "control" / "preflight-snapshot.json"

    def preflight(self) -> dict[str, Any]:
        snapshot = self.operations.preflight()
        receipt = validate_preflight_snapshot(
            snapshot, self.preflight_expectations
        )
        atomic_write_json(self.preflight_snapshot_path, snapshot)
        write_immutable_json(self.preflight_path, receipt)
        return receipt

    def _read_validated_preflight(self) -> dict[str, Any]:
        if not self.preflight_path.exists() or not self.preflight_snapshot_path.exists():
            raise RuntimeError("validated preflight receipt is missing")
        try:
            snapshot = read_json(self.preflight_snapshot_path)
            receipt = read_json(self.preflight_path)
            expected = validate_preflight_snapshot(
                snapshot, self.preflight_expectations
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError("validated preflight receipt is invalid") from error
        if receipt != expected:
            raise RuntimeError("validated preflight receipt drifted")
        return expected

    def reconcile_dead_work(self) -> list[dict[str, Any]]:
        """Fence dead drivers and reconcile their exact physical namespaces."""

        receipts: list[dict[str, Any]] = []
        for cell in self.store.manifest:
            key = cell.key
            if key.task_index not in self.assigned_task_indices:
                continue
            if self.store.accepted_path(key).exists():
                _load_accepted(self.store, key)
                continue
            claim_path = self.store.claim_path(key)
            if not claim_path.exists():
                continue
            previous = self.store.read_claim(claim_path, key)
            if previous.owner != self.owner and self.owner_is_alive(previous.owner):
                raise RuntimeError(f"cell recovery found a live owner: {key.slug}")
            token = self.store.acquire(key)
            receipt = self.operations.reconcile_cell(
                self.by_key[key],
                generation=token.generation,
                before_preflight=True,
            )
            if not isinstance(receipt, Mapping):
                raise RuntimeError("cell recovery receipt is invalid")
            accepted = self.store.reconcile_complete_attempt(token)
            receipts.append(
                {
                    "cell": key.to_payload(),
                    "generation": token.generation,
                    "accepted_recovered": accepted is not None,
                    "runtime": dict(receipt),
                }
            )

        for cell in self.store.manifest:
            key = cell.key
            if key.task_index not in self.assigned_task_indices:
                continue
            if not self.store.accepted_path(key).exists():
                continue
            accepted = _load_accepted(self.store, key)
            if self.store.outcome_path(key).exists():
                _load_official_outcome(self.store, key)
                continue
            if not self.store.grade_claim_path(key).exists():
                continue
            _, prediction, handoff = _load_attempt_boundaries(
                self.store, accepted, key
            )
            token = self.store.acquire_grade(key)
            receipt = self.operations.reconcile_grade(
                key=key,
                accepted=accepted,
                prediction=prediction,
                handoff=handoff,
            )
            if not isinstance(receipt, Mapping):
                raise RuntimeError("grader recovery receipt is invalid")
            receipts.append(
                {
                    "cell": key.to_payload(),
                    "grade_claim_generation": token.generation,
                    "grader": dict(receipt),
                }
            )
        startup = self.operations.reconcile_startup(
            task_indices=self.assigned_task_indices
        )
        if not isinstance(startup, Mapping):
            raise RuntimeError("startup reconciliation receipt is invalid")
        receipts.append({"startup": dict(startup)})
        atomic_write_json(
            self.root / "control" / "latest-reconciliation.json",
            {
                "schema": "swebench_triad_reconciliation_v1",
                "receipts": receipts,
            },
        )
        return receipts

    def live_preflight(self) -> dict[str, Any]:
        self.acquire_runtime_lane(None)
        try:
            self.reconcile_dead_work()
            result = self.preflight()
        except BaseException:
            raise
        else:
            self.release_runtime_lane()
            return result

    def _accepted_or_run(
        self, config: RunConfig, stage: Any
    ) -> Mapping[str, Any]:
        key = CellKey(config.task.task_index, config.capability.arm.value)
        if self.store.accepted_path(key).exists():
            return _load_accepted(self.store, key)
        try:
            token = self.store.acquire(key)
        except AlreadyAcceptedError:
            return _load_accepted(self.store, key)
        runtime_reconciliation = self.operations.reconcile_cell(
            config,
            generation=token.generation,
            before_preflight=False,
        )
        if not isinstance(runtime_reconciliation, Mapping):
            raise RuntimeError("cell runtime reconciliation receipt is invalid")
        reconciled = self.store.reconcile_complete_attempt(token)
        if reconciled is not None:
            return reconciled

        row = self.operations.run_cell(
            config,
            stage,
            generation=token.generation,
        )
        self.store.record_endpoint(token, row)
        artifact = _mapping(row.get("final_artifact"), "endpoint artifact")
        prediction = read_private_json(
            self.evidence_root, artifact.get("protected_ref")
        )
        scorer = _mapping(row.get("scorer"), "endpoint scorer")
        scorer_receipt = read_private_json(
            self.evidence_root, scorer.get("receipt_ref")
        )
        scorer_receipt = _mapping(scorer_receipt, "protected scorer receipt")
        queue = _mapping(
            scorer_receipt.get("grader_receipt"), "protected grader queue"
        )
        prediction_digest = sha256_json(prediction)
        _equal(
            queue.get("artifact_sha256"),
            prediction_digest,
            "queued prediction digest",
        )
        if queue.get("official_resolved") is not None:
            _fail("queued handoff claimed an official outcome")
        grader = _mapping(queue.get("grader"), "queued grader identity")
        _equal(grader.get("revision"), HARNESS_COMMIT, "queued grader revision")
        handoff = {
            "prediction_sha256": prediction_digest,
            "official_resolved": None,
            "grader_revision": HARNESS_COMMIT,
        }
        self.store.record_prediction(token, prediction)
        self.store.record_handoff(token, handoff)
        return self.store.accept_current_attempt(token)

    def _grade_if_missing(self, key: CellKey) -> Mapping[str, Any]:
        if self.store.outcome_path(key).exists():
            return _load_official_outcome(self.store, key)
        accepted = _load_accepted(self.store, key)
        try:
            grade_token = self.store.acquire_grade(key)
        except AlreadyGradedError:
            return _load_official_outcome(self.store, key)
        _, prediction, handoff = _load_attempt_boundaries(
            self.store, accepted, key
        )
        outcome = self.operations.grade(
            key=key,
            accepted=accepted,
            prediction=prediction,
            handoff=handoff,
        )
        self.store.record_official_outcome(grade_token, outcome)
        return _load_official_outcome(self.store, key)

    def task_complete(self, task_index: int) -> bool:
        for arm in ARMS:
            key = CellKey(task_index, arm)
            if (
                not self.store.accepted_path(key).exists()
                or not self.store.outcome_path(key).exists()
            ):
                return False
            _load_accepted(self.store, key)
            _load_official_outcome(self.store, key)
        return True

    def validate_task_triad(self, task_index: int) -> dict[str, Any]:
        if task_index not in self.by_task:
            raise ValueError("task index is outside the manifest")
        endpoint_rows: list[Mapping[str, Any]] = []
        for arm in ARMS:
            key = CellKey(task_index, arm)
            if not self.store.accepted_path(key).exists():
                raise RuntimeError("task triad is missing an accepted endpoint")
            accepted = _load_accepted(self.store, key)
            endpoint, _, _ = _load_attempt_boundaries(
                self.store, accepted, key
            )
            endpoint_rows.append(endpoint)
        try:
            verification = self.triad_validator(endpoint_rows)
        except Exception as error:
            raise RuntimeError("task triad validation failed") from error
        if not isinstance(verification, Mapping):
            raise RuntimeError("task triad validation receipt is invalid")
        return dict(verification)

    def task_completion_path(self, task_index: int) -> Path:
        if task_index not in self.by_task:
            raise ValueError("task index is outside the manifest")
        return self.root / "full" / f"task-{task_index:04d}.json"

    def load_task_completion(self, task_index: int) -> dict[str, Any]:
        completion = read_json(self.task_completion_path(task_index))
        if (
            not isinstance(completion, Mapping)
            or completion.get("schema")
            != "swebench_triad_task_completion_v1"
            or completion.get("task_index") != task_index
            or completion.get("instance_id")
            != self.by_task[task_index][0].task.task_id
            or completion.get("accepted_cells") != 3
            or completion.get("official_outcomes") != 3
            or not isinstance(completion.get("triad_verification"), Mapping)
            or not isinstance(completion.get("eviction"), Mapping)
        ):
            raise RuntimeError("task completion receipt is invalid")
        return dict(completion)

    @staticmethod
    def require_zero_residue(residue: Mapping[str, Any]) -> None:
        for name in (
            "active_slots",
            "active_workspaces",
            "containers",
            "processes",
            "cgroups",
            "tmpfs_mounts",
            "mounts",
        ):
            if residue.get(name) != 0:
                raise RuntimeError(f"task residue is nonzero: {name}")
        if residue.get("rootfs_attested") is not True:
            raise RuntimeError("task rootfs was not re-attested")

    def task_result(
        self,
        task_index: int,
        *,
        gate: bool,
        residue: Mapping[str, Any],
        eviction: Mapping[str, Any],
        triad_verification: Mapping[str, Any],
        wall_seconds: float,
    ) -> dict[str, Any]:
        outcomes = [
            _load_official_outcome(self.store, CellKey(task_index, arm))
            for arm in ARMS
        ]
        result = {
            "schema": "swebench_triad_task_completion_v1",
            "task_index": task_index,
            "gate_task": gate,
            "instance_id": self.by_task[task_index][0].task.task_id,
            "accepted_cells": 3,
            "official_outcomes": 3,
            "triad_verification": dict(triad_verification),
            "resolved": [row["resolved"] for row in outcomes],
            "residue": dict(residue),
            "eviction": dict(eviction),
            "wall_seconds": wall_seconds,
        }
        atomic_write_json(self.task_completion_path(task_index), result)
        return result

    def run_task(self, task_index: int, *, gate: bool = False) -> dict[str, Any]:
        if task_index not in self.assigned_task_indices:
            raise ValueError("task index is outside the assigned shard")
        self.acquire_runtime_lane(task_index)
        try:
            result = self._run_task_in_lane(task_index, gate=gate)
        except BaseException:
            raise
        else:
            self.release_runtime_lane()
            return result

    def _run_task_in_lane(
        self, task_index: int, *, gate: bool = False
    ) -> dict[str, Any]:
        if task_index not in self.by_task:
            raise ValueError("task index is outside the manifest")
        if self.task_complete(task_index):
            completion_path = self.task_completion_path(task_index)
            if completion_path.exists():
                return self.load_task_completion(task_index)
            residue = self.operations.audit_residue(task_index)
            recovery_stage = None
            if residue.get("rootfs_attested") is not True:
                recovery_stage = self.operations.stage_task(task_index)
                residue = self.operations.audit_residue(task_index)
            self.require_zero_residue(residue)
            accepted_rows = [
                _load_accepted(self.store, CellKey(task_index, arm))
                for arm in ARMS
            ]
            endpoint_rows = [
                _load_attempt_boundaries(
                    self.store,
                    accepted,
                    CellKey(task_index, arm),
                )[0]
                for arm, accepted in zip(ARMS, accepted_rows)
            ]
            triad_verification = self.triad_validator(endpoint_rows)
            if not isinstance(triad_verification, Mapping):
                raise RuntimeError("triad verification receipt is invalid")
            eviction = self.operations.evict_task(task_index, recovery_stage)
            return self.task_result(
                task_index,
                gate=gate,
                residue=residue,
                eviction=eviction,
                triad_verification=triad_verification,
                wall_seconds=0.0,
            )
        stage = self.operations.stage_task(task_index)
        started = self.clock()
        accepted_rows = []
        for config in self.by_task[task_index]:
            accepted = self._accepted_or_run(config, stage)
            accepted_rows.append(dict(accepted))
        endpoint_rows = [
            _load_attempt_boundaries(
                self.store,
                accepted,
                CellKey(task_index, config.capability.arm.value),
            )[0]
            for config, accepted in zip(
                self.by_task[task_index], accepted_rows
            )
        ]
        triad_verification = self.triad_validator(endpoint_rows)
        if not isinstance(triad_verification, Mapping):
            raise RuntimeError("triad verification receipt is invalid")
        outcomes = []
        for config in self.by_task[task_index]:
            outcomes.append(
                dict(
                    self._grade_if_missing(
                        CellKey(task_index, config.capability.arm.value)
                    )
                )
            )
        if [row.get("cell", {}).get("arm") for row in accepted_rows] != list(ARMS):
            raise RuntimeError("accepted triad arm lattice drifted")
        if [row.get("arm") for row in outcomes] != list(ARMS):
            raise RuntimeError("official triad arm lattice drifted")
        residue = self.operations.audit_residue(task_index)
        self.require_zero_residue(residue)
        eviction = self.operations.evict_task(task_index, stage)
        return self.task_result(
            task_index,
            gate=gate,
            residue=residue,
            eviction=eviction,
            triad_verification=triad_verification,
            wall_seconds=max(0.0, self.clock() - started),
        )

    def _validate_gate(
        self,
        value: Any,
        *,
        preflight: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_fields = {
            "schema",
            "status",
            "task_index",
            "canonical_cells",
            "accepted_cells",
            "official_outcomes",
            "triad_verification_sha256",
            "preflight_sha256",
            "standalone_benchmark_score",
        }
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise RuntimeError("gate PASS receipt fields are not canonical")
        if (
            value.get("schema") != GATE_PASS_SCHEMA
            or value.get("status") != "PASS"
            or value.get("task_index") != 0
            or value.get("canonical_cells")
            != [{"task_index": 0, "arm": arm} for arm in ARMS]
            or value.get("accepted_cells") != len(ARMS)
            or value.get("official_outcomes") != len(ARMS)
            or value.get("standalone_benchmark_score") is not False
        ):
            raise RuntimeError("gate PASS receipt is invalid")
        if value.get("preflight_sha256") != sha256_json(preflight):
            raise RuntimeError("gate PASS preflight binding drifted")

        endpoint_rows: list[Mapping[str, Any]] = []
        for arm in ARMS:
            key = CellKey(0, arm)
            accepted = _load_accepted(self.store, key)
            endpoint, _, _ = _load_attempt_boundaries(
                self.store, accepted, key
            )
            _load_official_outcome(self.store, key)
            endpoint_rows.append(endpoint)
        try:
            triad_verification = self.triad_validator(endpoint_rows)
        except BaseException as error:
            raise RuntimeError("canonical gate triad validation failed") from error
        if not isinstance(triad_verification, Mapping):
            raise RuntimeError("canonical gate triad validation is invalid")
        if value.get("triad_verification_sha256") != sha256_json(
            triad_verification
        ):
            raise RuntimeError("gate PASS triad binding drifted")
        return dict(value)

    def gate(self, *, auto_run_full: bool = False) -> dict[str, Any]:
        if 0 not in self.assigned_task_indices:
            raise RuntimeError("canonical gate task is outside the assigned shard")
        preflight = self.live_preflight()
        if self.gate_path.exists():
            gate = self._validate_gate(
                read_json(self.gate_path), preflight=preflight
            )
        else:
            task = self.run_task(0, gate=True)
            gate = {
                "schema": GATE_PASS_SCHEMA,
                "status": "PASS",
                "task_index": 0,
                "canonical_cells": [
                    {"task_index": 0, "arm": arm} for arm in ARMS
                ],
                "accepted_cells": task["accepted_cells"],
                "official_outcomes": task["official_outcomes"],
                "triad_verification_sha256": sha256_json(
                    task["triad_verification"]
                ),
                "preflight_sha256": sha256_json(preflight),
                "standalone_benchmark_score": False,
            }
            gate = self._validate_gate(gate, preflight=preflight)
            write_immutable_json(self.gate_path, gate)
        if auto_run_full:
            return self.run_full()
        return dict(gate)

    def _require_gate(
        self, *, preflight: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if not self.gate_path.exists():
            raise RuntimeError("full run requires the canonical gate PASS receipt")
        return self._validate_gate(
            read_json(self.gate_path), preflight=preflight
        )

    def run_full(self) -> dict[str, Any]:
        preflight = self.live_preflight()
        self._require_gate(preflight=preflight)
        for task_index in self.assigned_task_indices:
            self.run_task(task_index, gate=task_index == 0)
            self.write_heartbeat()
        if not all(self.task_complete(task_index) for task_index in self.by_task):
            result = {
                "schema": "swebench_triad_shard_completion_v1",
                "status": "PASS",
                "assigned_task_indices": list(self.assigned_task_indices),
                "completed_tasks": sum(
                    self.task_complete(task_index)
                    for task_index in self.assigned_task_indices
                ),
                "global": self.status(),
            }
            if self.lease_registry is not None:
                self.lease_registry.release()
            return result
        rows = self.store.assemble_results()
        output = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        atomic_write_bytes(self.root / "full" / "results.jsonl", output)
        summary = self.store.official_summary()
        atomic_write_json(self.root / "full" / "official-summary.json", summary)
        if self.lease_registry is not None:
            self.lease_registry.release()
        return summary

    def resume(self) -> dict[str, Any]:
        return self.run_full()

    def grade_all(self) -> dict[str, Any]:
        self.live_preflight()
        for task_index in self.assigned_task_indices:
            accepted_presence = [
                self.store.accepted_path(CellKey(task_index, arm)).exists()
                for arm in ARMS
            ]
            if not all(accepted_presence):
                continue
            self.validate_task_triad(task_index)
            missing = []
            for arm in ARMS:
                key = CellKey(task_index, arm)
                _load_accepted(self.store, key)
                if self.store.outcome_path(key).exists():
                    _load_official_outcome(self.store, key)
                else:
                    missing.append(key)
            if not missing:
                continue
            self.acquire_runtime_lane(task_index)
            try:
                stage = self.operations.stage_task(task_index)
                for key in missing:
                    self._grade_if_missing(key)
                residue = self.operations.audit_residue(task_index)
                self.require_zero_residue(residue)
                self.operations.evict_task(task_index, stage)
            except BaseException:
                raise
            else:
                self.release_runtime_lane()
        return self.status()

    def task_status(self, task_index: int) -> dict[str, Any]:
        accepted = 0
        official = 0
        for arm in ARMS:
            key = CellKey(task_index, arm)
            if self.store.accepted_path(key).exists():
                _load_accepted(self.store, key)
                accepted += 1
            if self.store.outcome_path(key).exists():
                _load_official_outcome(self.store, key)
                official += 1
        return {
            "task_index": task_index,
            "accepted": accepted,
            "official": official,
        }

    def gate_pass_status(self) -> bool:
        if not self.gate_path.exists():
            return False
        preflight = self._read_validated_preflight()
        self._validate_gate(read_json(self.gate_path), preflight=preflight)
        return True

    def status(self) -> dict[str, Any]:
        accepted = 0
        official = 0
        for cell in self.store.manifest:
            if self.store.accepted_path(cell.key).exists():
                _load_accepted(self.store, cell.key)
                accepted += 1
            if self.store.outcome_path(cell.key).exists():
                _load_official_outcome(self.store, cell.key)
                official += 1
        return {
            "schema": "swebench_triad_status_v1",
            "manifest_cells": len(self.store.manifest),
            "accepted_cells": accepted,
            "official_outcomes": official,
            "remaining_cells": len(self.store.manifest) - accepted,
            "remaining_outcomes": len(self.store.manifest) - official,
            "gate_pass": self.gate_pass_status(),
            "assigned_task_indices": list(self.assigned_task_indices),
        }

    def write_heartbeat(self) -> dict[str, Any]:
        if self.lease_registry is not None and self.lease_registry.lease_id is not None:
            self.lease_registry.refresh()
            self.lease_registry.assert_healthy()
        status = self.status()
        payload = {
            **status,
            "schema": HEARTBEAT_SCHEMA,
            "monotonic_seconds": self.clock(),
        }
        atomic_write_json(self.root / "full" / "heartbeat.json", payload)
        return payload

    def validate_complete_manifest(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        try:
            verification = self.triad_validator(rows)
        except Exception as error:
            raise RuntimeError("global triad validation failed") from error
        if not isinstance(verification, Mapping):
            raise RuntimeError("global triad validation receipt is invalid")
        expected_counts = {
            "row_count": len(self.store.manifest),
            "triad_count": len(self.by_task),
        }
        for field, expected in expected_counts.items():
            actual = verification.get(field)
            if type(actual) is not int or actual != expected:
                raise RuntimeError(
                    f"global triad validation {field} drifted"
                )
        return dict(verification)

    @staticmethod
    def require_canonical_artifact(path: Path, expected: bytes) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(
                f"canonical public artifact is missing: {path}"
            ) from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError(
                f"canonical public artifact is not a regular file: {path}"
            )
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"canonical public artifact is unreadable: {path}"
            ) from error
        if actual != expected:
            raise RuntimeError(
                f"canonical public artifact bytes drifted: {path}"
            )

    def audit(self) -> dict[str, Any]:
        status = self.status()
        complete = (
            status["accepted_cells"] == status["manifest_cells"]
            and status["official_outcomes"] == status["manifest_cells"]
        )
        if complete and status["gate_pass"] is not True:
            raise RuntimeError("complete audit requires the canonical gate PASS")
        triad_verification: dict[str, Any] = {}
        official_summary: dict[str, Any] = {}
        if complete:
            rows = self.store.assemble_results()
            triad_verification = self.validate_complete_manifest(rows)
            official_summary = self.store.official_summary()
            expected_artifacts = {
                self.root / "full" / "results.jsonl": b"".join(
                    canonical_json_bytes(row) + b"\n" for row in rows
                ),
                self.root / "full" / "official-summary.json": (
                    canonical_json_bytes(official_summary) + b"\n"
                ),
            }
            for path, expected in expected_artifacts.items():
                self.require_canonical_artifact(path, expected)
        result = {
            "schema": "swebench_triad_audit_v1",
            "status": "PASS" if complete else "INCOMPLETE",
            **status,
        }
        if complete:
            result["matched_cells"] = triad_verification["row_count"]
            result["matched_triads"] = triad_verification["triad_count"]
            result["matched_triads_sha256"] = sha256_json(
                triad_verification
            )
            result["official_summary"] = official_summary
            runtime = self.operations.final_audit()
            if not isinstance(runtime, Mapping):
                raise RuntimeError("final runtime audit is invalid")
            result["runtime"] = dict(runtime)
            result["privacy"] = self.privacy_audit()
        atomic_write_json(self.root / "full" / "audit.json", result)
        return result

    def privacy_audit(self) -> dict[str, Any]:
        forbidden = {
            "model_patch",
            "gold_patch",
            "test_patch",
            "problem_statement",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        }
        paths = (
            self.root / "full" / "results.jsonl",
            self.root / "full" / "official-summary.json",
        )
        hits: list[dict[str, str]] = []
        for path in paths:
            try:
                info = path.lstat()
            except FileNotFoundError as error:
                raise RuntimeError(
                    f"privacy audit requires public artifact: {path}"
                ) from error
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise RuntimeError(
                    f"privacy audit requires a regular public artifact: {path}"
                )
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise RuntimeError(
                    f"privacy audit cannot read public artifact: {path}"
                ) from error
            for needle in sorted(forbidden):
                if needle in text:
                    hits.append({"path": str(path), "needle": needle})
        if hits:
            raise RuntimeError("public result artifacts contain protected fields")
        return {
            "schema": "swebench_triad_privacy_audit_v1",
            "status": "PASS",
            "scanned_paths": [str(path) for path in paths],
            "forbidden_field_hits": 0,
        }

    def cleanup(self) -> dict[str, Any]:
        if self.assigned_task_indices != tuple(sorted(self.by_task)):
            raise RuntimeError("global cleanup requires the full task shard")
        self.acquire_runtime_lane(None)
        if self.lease_registry is not None:
            self.lease_registry.assert_no_other_live_drivers()
        result = self.operations.cleanup()
        if not isinstance(result, Mapping):
            raise RuntimeError("cleanup receipt is invalid")
        if result.get("owned_residue") != 0:
            raise RuntimeError("owned cleanup residue is nonzero")
        if result.get("allocation_retained") is not True:
            raise RuntimeError("cleanup did not retain the allocation")
        atomic_write_json(self.root / "control" / "cleanup.json", result)
        self.release_runtime_lane()
        if self.lease_registry is not None:
            self.lease_registry.release()
        return dict(result)


def load_json_object(path: Path | str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load JSON object: {path}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON document must be an object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swebench-triad-eval")
    parser.set_defaults(task_range=None)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "preflight",
        "gate",
        "run",
        "resume",
        "grade",
        "status",
        "audit",
        "cleanup",
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name != "gate":
            command.add_argument(
                "--task-range",
                type=parse_task_range,
                help="explicit half-open shard START:STOP (default: 0:500)",
            )
        if name == "gate":
            command.add_argument("--auto-run-full", action="store_true")
    return parser


def parse_task_range(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+:[0-9]+", value) is None:
        raise argparse.ArgumentTypeError("task range must be START:STOP")
    start_text, stop_text = value.split(":", 1)
    start = int(start_text)
    stop = int(stop_text)
    if not 0 <= start < stop <= 500:
        raise argparse.ArgumentTypeError("task range must satisfy 0 <= START < STOP <= 500")
    return tuple(range(start, stop))


def driver_from_config(
    path: Path,
    *,
    owner: OwnerIdentity | None = None,
    owner_is_alive: Callable[[OwnerIdentity], bool] | None = None,
    operations_factory: Callable[[Any, Sequence[RunConfig]], Any] | None = None,
    assigned_task_indices: Sequence[int] | None = None,
) -> LifecycleDriver:
    """Bind the sealed production description without performing pod actions."""

    from paired_eval.verifier import validate_result_row, verify_pair_completeness

    from .production import (
        ProductionLifecycleOperations,
        ProductionRunConfig,
        current_owner_identity,
        owner_is_alive as production_owner_is_alive,
    )

    production = ProductionRunConfig.load(path)
    selected_owner = current_owner_identity() if owner is None else owner
    if not isinstance(selected_owner, OwnerIdentity):
        raise TypeError("production owner must be OwnerIdentity")
    local_liveness = (
        production_owner_is_alive if owner_is_alive is None else owner_is_alive
    )
    if not callable(local_liveness):
        raise TypeError("production owner liveness probe must be callable")
    selected_tasks = (
        tuple(range(500))
        if assigned_task_indices is None
        else tuple(assigned_task_indices)
    )
    lease_registry = DriverLeaseRegistry(
        production.run_root / "state" / "leases",
        owner=selected_owner,
        assigned_task_indices=selected_tasks,
        local_owner_is_alive=local_liveness,
    )
    factory = operations_factory or ProductionLifecycleOperations
    operations = factory(production, production.configs)
    return LifecycleDriver(
        root=production.run_root,
        configs=production.configs,
        owner=selected_owner,
        owner_is_alive=lease_registry.owner_is_alive,
        operations=operations,
        evidence_root=production.evidence_root,
        endpoint_validator=validate_result_row,
        triad_validator=verify_pair_completeness,
        preflight_expectations=production.preflight_expectations,
        assigned_task_indices=selected_tasks,
        lease_registry=lease_registry,
    )


def write_stdout(value: Any) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value) + b"\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    driver = driver_from_config(
        arguments.config,
        assigned_task_indices=arguments.task_range,
    )
    if arguments.command == "preflight":
        result = driver.preflight()
    elif arguments.command == "gate":
        result = driver.gate(auto_run_full=True)
    elif arguments.command == "run":
        result = driver.run_full()
    elif arguments.command == "resume":
        result = driver.resume()
    elif arguments.command == "grade":
        result = driver.grade_all()
    elif arguments.command == "status":
        result = driver.status()
    elif arguments.command == "audit":
        result = driver.audit()
    else:
        result = driver.cleanup()
    write_stdout(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LifecycleDriver",
    "PreflightContractError",
    "build_parser",
    "driver_from_config",
    "main",
    "parse_task_range",
    "read_private_json",
    "validate_preflight_snapshot",
]
