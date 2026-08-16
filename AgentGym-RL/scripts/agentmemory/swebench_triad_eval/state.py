"""Fenced, resumable cell state and exact official-outcome joins."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import ARMS
from .atomic import (
    atomic_write_json,
    canonical_json_bytes,
    ensure_private_directory,
    exclusive_lock,
    read_json,
    write_immutable_json,
)


class ClaimBusyError(RuntimeError):
    pass


class FenceViolationError(RuntimeError):
    pass


class AlreadyAcceptedError(RuntimeError):
    pass


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, order=True)
class CellKey:
    task_index: int
    arm: str

    def __post_init__(self) -> None:
        if isinstance(self.task_index, bool) or not isinstance(self.task_index, int):
            raise TypeError("cell task index must be an integer")
        if self.task_index < 0:
            raise ValueError("cell task index must be non-negative")
        if self.arm not in ARMS:
            raise ValueError("cell arm is unsupported")

    @property
    def slug(self) -> str:
        return f"{self.task_index:04d}-{self.arm}"

    def to_payload(self) -> dict[str, Any]:
        return {"task_index": self.task_index, "arm": self.arm}


@dataclass(frozen=True)
class ManifestCell:
    key: CellKey
    instance_id: str
    manifest_cell_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, CellKey):
            raise TypeError("manifest cell key must be a CellKey")
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise ValueError("manifest instance ID must be nonempty text")
        require_sha256(self.manifest_cell_sha256, "manifest cell")


@dataclass(frozen=True)
class OwnerIdentity:
    host_id: str
    boot_id: str
    pid: int
    pid_start_ticks: int

    def __post_init__(self) -> None:
        if not isinstance(self.host_id, str) or not self.host_id:
            raise ValueError("owner host ID must be nonempty text")
        if not isinstance(self.boot_id, str) or not self.boot_id:
            raise ValueError("owner boot ID must be nonempty text")
        for name in ("pid", "pid_start_ticks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"owner {name} must be a positive integer")

    def to_payload(self) -> dict[str, Any]:
        return {
            "host_id": self.host_id,
            "boot_id": self.boot_id,
            "pid": self.pid,
            "pid_start_ticks": self.pid_start_ticks,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OwnerIdentity":
        expected = {"host_id", "boot_id", "pid", "pid_start_ticks"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("claim owner identity is invalid")
        return cls(**{name: payload[name] for name in expected})


@dataclass(frozen=True)
class ClaimToken:
    key: CellKey
    generation: int
    manifest_cell_sha256: str
    owner: OwnerIdentity


class CellStateStore:
    def __init__(
        self,
        root: Path | str,
        *,
        manifest: Sequence[ManifestCell],
        owner: OwnerIdentity,
        owner_is_alive: Callable[[OwnerIdentity], bool],
        endpoint_validator: Callable[[Any], None],
    ) -> None:
        cells = tuple(manifest)
        if not cells or any(not isinstance(cell, ManifestCell) for cell in cells):
            raise ValueError("state manifest must contain typed cells")
        keys = [cell.key for cell in cells]
        if len(keys) != len(set(keys)):
            raise ValueError("state manifest contains duplicate cells")
        self.root = ensure_private_directory(root)
        self.manifest = cells
        self.cells = {cell.key: cell for cell in cells}
        self.owner = owner
        self.owner_is_alive = owner_is_alive
        self.endpoint_validator = endpoint_validator
        for name in ("claims", "locks", "attempts", "accepted", "outcomes"):
            ensure_private_directory(self.root / name)

    def cell(self, key: CellKey) -> ManifestCell:
        try:
            return self.cells[key]
        except KeyError as error:
            raise ValueError("cell is absent from the immutable manifest") from error

    def claim_path(self, key: CellKey) -> Path:
        return self.root / "claims" / f"{key.slug}.json"

    def lock_path(self, key: CellKey) -> Path:
        return self.root / "locks" / f"{key.slug}.lock"

    def accepted_path(self, key: CellKey) -> Path:
        return self.root / "accepted" / f"{key.slug}.json"

    def outcome_path(self, key: CellKey) -> Path:
        return self.root / "outcomes" / f"{key.slug}.json"

    def attempt_directory(self, key: CellKey, generation: int) -> Path:
        return self.root / "attempts" / key.slug / f"{generation:08d}"

    def artifact_path(
        self,
        key: CellKey,
        generation: int,
        artifact: str,
    ) -> Path:
        return self.attempt_directory(key, generation) / f"{artifact}.json"

    def acquire(self, key: CellKey) -> ClaimToken:
        cell = self.cell(key)
        with exclusive_lock(self.lock_path(key)):
            if self.accepted_path(key).exists():
                raise AlreadyAcceptedError(f"cell is already accepted: {key.slug}")
            path = self.claim_path(key)
            generation = 1
            if path.exists():
                claim = self.read_claim(path, key)
                if claim.manifest_cell_sha256 != cell.manifest_cell_sha256:
                    raise FenceViolationError("claim manifest digest drifted")
                if claim.owner == self.owner:
                    return claim
                if self.owner_is_alive(claim.owner):
                    raise ClaimBusyError(f"cell has a live owner: {key.slug}")
                generation = claim.generation + 1
            token = ClaimToken(
                key=key,
                generation=generation,
                manifest_cell_sha256=cell.manifest_cell_sha256,
                owner=self.owner,
            )
            atomic_write_json(path, self.claim_payload(token))
            return token

    @staticmethod
    def claim_payload(token: ClaimToken) -> dict[str, Any]:
        return {
            "schema": "swebench_triad_fenced_claim_v1",
            "cell": token.key.to_payload(),
            "generation": token.generation,
            "manifest_cell_sha256": token.manifest_cell_sha256,
            "owner": token.owner.to_payload(),
        }

    @staticmethod
    def read_claim(path: Path, expected_key: CellKey) -> ClaimToken:
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            raise FenceViolationError("claim is not an object")
        expected_fields = {
            "schema",
            "cell",
            "generation",
            "manifest_cell_sha256",
            "owner",
        }
        if set(payload) != expected_fields:
            raise FenceViolationError("claim fields are not canonical")
        cell_payload = payload["cell"]
        if not isinstance(cell_payload, Mapping):
            raise FenceViolationError("claim cell is invalid")
        key = CellKey(cell_payload.get("task_index"), cell_payload.get("arm"))
        if key != expected_key:
            raise FenceViolationError("claim cell identity drifted")
        generation = payload["generation"]
        if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
            raise FenceViolationError("claim generation is invalid")
        return ClaimToken(
            key=key,
            generation=generation,
            manifest_cell_sha256=require_sha256(
                payload["manifest_cell_sha256"], "claim manifest cell"
            ),
            owner=OwnerIdentity.from_payload(payload["owner"]),
        )

    def assert_fence(self, token: ClaimToken) -> None:
        if not isinstance(token, ClaimToken):
            raise TypeError("state write requires a claim token")
        current = self.read_claim(self.claim_path(token.key), token.key)
        if current != token:
            raise FenceViolationError("claim generation or owner was fenced")
        cell = self.cell(token.key)
        if token.manifest_cell_sha256 != cell.manifest_cell_sha256:
            raise FenceViolationError("claim token has the wrong manifest digest")

    def record_endpoint(self, token: ClaimToken, row: Mapping[str, Any]) -> Path:
        self.assert_fence(token)
        self.validate_endpoint(token.key, row)
        return write_immutable_json(
            self.artifact_path(token.key, token.generation, "endpoint"),
            row,
        )

    def record_prediction(
        self,
        token: ClaimToken,
        prediction: Mapping[str, Any],
    ) -> Path:
        self.assert_fence(token)
        self.validate_prediction(token.key, prediction)
        return write_immutable_json(
            self.artifact_path(token.key, token.generation, "prediction"),
            prediction,
        )

    def record_handoff(
        self,
        token: ClaimToken,
        handoff: Mapping[str, Any],
    ) -> Path:
        self.assert_fence(token)
        self.validate_handoff(token, handoff)
        return write_immutable_json(
            self.artifact_path(token.key, token.generation, "handoff"),
            handoff,
        )

    def prediction_sha256(self, token: ClaimToken) -> str:
        self.assert_fence(token)
        prediction_value = read_json(
            self.artifact_path(token.key, token.generation, "prediction")
        )
        return sha256_json(prediction_value)

    def validate_endpoint(self, key: CellKey, row: Any) -> None:
        self.endpoint_validator(row)
        if not isinstance(row, Mapping):
            raise ValueError("endpoint row must be an object")
        cell = self.cell(key)
        if row.get("instance_id") != cell.instance_id or row.get("arm") != key.arm:
            raise ValueError("endpoint row cell identity drifted")
        if row.get("comparable") is not True:
            raise ValueError("non-comparable endpoint rows cannot be accepted")
        failure = row.get("failure")
        if not isinstance(failure, Mapping) or failure.get("class") is not None:
            raise ValueError("failed endpoint rows remain retryable")
        if row.get("final_artifact") is None or row.get("scorer") is None:
            raise ValueError("endpoint row lacks artifact or queued handoff")
        lifecycle = row.get("lifecycle")
        if not isinstance(lifecycle, Mapping) or not lifecycle.get("close_receipt_ref"):
            raise ValueError("endpoint row lacks lifecycle close evidence")

    def validate_prediction(self, key: CellKey, prediction: Any) -> None:
        if not isinstance(prediction, Mapping) or set(prediction) != {
            "instance_id",
            "model_name_or_path",
            "model_patch",
        }:
            raise ValueError("prediction fields are not canonical")
        if prediction["instance_id"] != self.cell(key).instance_id:
            raise ValueError("prediction instance ID drifted")
        if not isinstance(prediction["model_name_or_path"], str) or not prediction[
            "model_name_or_path"
        ]:
            raise ValueError("prediction model identity is invalid")
        if not isinstance(prediction["model_patch"], str):
            raise ValueError("prediction patch must be text")

    def validate_handoff(self, token: ClaimToken, handoff: Any) -> None:
        if not isinstance(handoff, Mapping) or set(handoff) != {
            "prediction_sha256",
            "official_resolved",
            "grader_revision",
        }:
            raise ValueError("grader handoff fields are not canonical")
        if handoff["official_resolved"] is not None:
            raise ValueError("queued handoff cannot claim an official outcome")
        require_sha256(handoff["prediction_sha256"], "handoff prediction")
        if handoff["prediction_sha256"] != self.prediction_sha256(token):
            raise ValueError("grader handoff prediction digest drifted")
        if handoff["grader_revision"] != (
            "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
        ):
            raise ValueError("grader handoff revision drifted")

    def complete_attempt(self, key: CellKey, generation: int) -> dict[str, Any] | None:
        paths = {
            name: self.artifact_path(key, generation, name)
            for name in ("endpoint", "prediction", "handoff")
        }
        if not all(path.exists() for path in paths.values()):
            return None
        endpoint = read_json(paths["endpoint"])
        prediction = read_json(paths["prediction"])
        handoff = read_json(paths["handoff"])
        self.validate_endpoint(key, endpoint)
        self.validate_prediction(key, prediction)
        prediction_sha256 = sha256_json(prediction)
        if (
            not isinstance(handoff, Mapping)
            or handoff.get("official_resolved") is not None
            or handoff.get("prediction_sha256") != prediction_sha256
            or handoff.get("grader_revision")
            != "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
        ):
            raise ValueError("durable grader handoff is invalid")
        return {
            "endpoint": endpoint,
            "prediction": prediction,
            "handoff": handoff,
        }

    def accepted_payload(
        self,
        key: CellKey,
        generation: int,
        artifacts: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema": "swebench_triad_accepted_cell_v1",
            "cell": key.to_payload(),
            "instance_id": self.cell(key).instance_id,
            "manifest_cell_sha256": self.cell(key).manifest_cell_sha256,
            "attempt_generation": generation,
            "endpoint_sha256": sha256_json(artifacts["endpoint"]),
            "prediction_sha256": sha256_json(artifacts["prediction"]),
            "handoff_sha256": sha256_json(artifacts["handoff"]),
        }

    def accept_current_attempt(self, token: ClaimToken) -> dict[str, Any]:
        existing_path = self.accepted_path(token.key)
        if existing_path.exists():
            existing = read_json(existing_path)
            if not isinstance(existing, dict):
                raise ValueError("accepted cell record is invalid")
            return existing
        self.assert_fence(token)
        artifacts = self.complete_attempt(token.key, token.generation)
        if artifacts is None:
            raise ValueError("cell attempt is missing a durable boundary")
        accepted = self.accepted_payload(token.key, token.generation, artifacts)
        write_immutable_json(existing_path, accepted)
        return accepted

    def reconcile_complete_attempt(
        self,
        token: ClaimToken,
    ) -> dict[str, Any] | None:
        if self.accepted_path(token.key).exists():
            value = read_json(self.accepted_path(token.key))
            if not isinstance(value, dict):
                raise ValueError("accepted cell record is invalid")
            return value
        self.assert_fence(token)
        for generation in range(token.generation - 1, 0, -1):
            artifacts = self.complete_attempt(token.key, generation)
            if artifacts is None:
                continue
            accepted = self.accepted_payload(token.key, generation, artifacts)
            write_immutable_json(self.accepted_path(token.key), accepted)
            return accepted
        return None

    def assemble_results(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cell in self.manifest:
            path = self.accepted_path(cell.key)
            if not path.exists():
                raise ValueError("accepted endpoint denominator is incomplete")
            accepted = read_json(path)
            if not isinstance(accepted, Mapping):
                raise ValueError("accepted cell record is invalid")
            generation = accepted.get("attempt_generation")
            endpoint = read_json(self.artifact_path(cell.key, generation, "endpoint"))
            if sha256_json(endpoint) != accepted.get("endpoint_sha256"):
                raise ValueError("accepted endpoint digest drifted")
            if not isinstance(endpoint, dict):
                raise ValueError("accepted endpoint row is invalid")
            rows.append(endpoint)
        return rows

    def record_official_outcome(
        self,
        key: CellKey,
        outcome: Mapping[str, Any],
    ) -> Path:
        cell = self.cell(key)
        accepted_path = self.accepted_path(key)
        if not accepted_path.exists():
            raise ValueError("official outcome cannot precede endpoint acceptance")
        expected = {
            "instance_id",
            "arm",
            "resolved",
            "failure_class",
            "report_sha256",
        }
        if not isinstance(outcome, Mapping) or set(outcome) != expected:
            raise ValueError("official outcome fields are not canonical")
        if outcome["instance_id"] != cell.instance_id or outcome["arm"] != key.arm:
            raise ValueError("official outcome cell identity drifted")
        if type(outcome["resolved"]) is not bool:
            raise ValueError("official outcome must be boolean")
        if outcome["failure_class"] is not None and (
            not isinstance(outcome["failure_class"], str)
            or not outcome["failure_class"]
        ):
            raise ValueError("official failure class is invalid")
        require_sha256(outcome["report_sha256"], "official report")
        accepted = read_json(accepted_path)
        payload = {
            "schema": "swebench_triad_official_outcome_v1",
            **dict(outcome),
            "prediction_sha256": accepted["prediction_sha256"],
            "attempt_generation": accepted["attempt_generation"],
        }
        return write_immutable_json(self.outcome_path(key), payload)

    def official_summary(self) -> dict[str, Any]:
        by_arm: dict[str, list[bool]] = {arm: [] for arm in ARMS}
        for cell in self.manifest:
            path = self.outcome_path(cell.key)
            if not path.exists():
                raise ValueError("official outcome denominator is incomplete")
            outcome = read_json(path)
            if not isinstance(outcome, Mapping) or type(outcome.get("resolved")) is not bool:
                raise ValueError("official outcome is invalid")
            if (
                outcome.get("instance_id") != cell.instance_id
                or outcome.get("arm") != cell.key.arm
            ):
                raise ValueError("official outcome manifest join drifted")
            by_arm[cell.key.arm].append(outcome["resolved"])
        denominators = {arm: len(values) for arm, values in by_arm.items()}
        if len(set(denominators.values())) != 1 or not next(iter(denominators.values())):
            raise ValueError("official arm denominators drifted")
        denominator = next(iter(denominators.values()))
        scores = {
            arm: sum(values) / denominator for arm, values in by_arm.items()
        }
        return {
            "schema": "swebench_triad_official_summary_v1",
            "denominator_per_arm": denominator,
            "scores": scores,
            "contrasts": {
                "compaction_only-native": (
                    scores["amg_compaction_only"] - scores["native"]
                ),
                "amg_memory-compaction_only": (
                    scores["amg_memory"] - scores["amg_compaction_only"]
                ),
                "amg_memory-native": scores["amg_memory"] - scores["native"],
            },
        }


__all__ = [
    "AlreadyAcceptedError",
    "CellKey",
    "CellStateStore",
    "ClaimBusyError",
    "ClaimToken",
    "FenceViolationError",
    "ManifestCell",
    "OwnerIdentity",
]
