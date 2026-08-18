"""Fenced, resumable cell state and exact official-outcome joins."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from paired_eval.serialization import sha256_json as paired_sha256_json

from . import ARMS
from .atomic import (
    atomic_write_json,
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


class AlreadyGradedError(RuntimeError):
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
    return paired_sha256_json(value)


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


@dataclass(frozen=True)
class GradeClaimToken:
    key: CellKey
    generation: int
    accepted_sha256: str
    owner: OwnerIdentity


@dataclass(frozen=True)
class RuntimeLaneToken:
    """Generation-bound authority for one deterministic runtime slot."""

    driver_key: str
    lease_id: str
    owner: OwnerIdentity
    task_index: int | None
    slot_index: int
    server_port: int
    generation: int
    fencing_token: str

    def __post_init__(self) -> None:
        require_sha256(self.driver_key, "runtime lane driver key")
        if not isinstance(self.lease_id, str) or len(self.lease_id) != 64:
            raise ValueError("runtime lane lease ID is invalid")
        if not isinstance(self.owner, OwnerIdentity):
            raise TypeError("runtime lane owner is invalid")
        if self.task_index is not None and (
            type(self.task_index) is not int or self.task_index < 0
        ):
            raise ValueError("runtime lane task index is invalid")
        if type(self.slot_index) is not int or self.slot_index < 0:
            raise ValueError("runtime lane slot index is invalid")
        if (
            type(self.server_port) is not int
            or not 1 <= self.server_port <= 65535
        ):
            raise ValueError("runtime lane server port is invalid")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("runtime lane generation is invalid")
        if (
            not isinstance(self.fencing_token, str)
            or len(self.fencing_token) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.fencing_token
            )
        ):
            raise ValueError("runtime lane fencing token is invalid")


class DriverLeaseRegistry:
    """Cross-host driver, shard, and deterministic runtime-slot leases.

    Driver heartbeats are the cross-host liveness source.  Task records are
    explicit shard assignments.  Exactly one generation-bound record exists
    for each configured slot/port, so a stale worker cannot publish, clean up,
    or release a successor's slot.  All takeovers are serialized and require
    the previous driver heartbeat to be proven dead.
    """

    DRIVER_SCHEMA = "swebench_triad_driver_lease_v1"
    TASK_SCHEMA = "swebench_triad_task_lease_v1"
    LANE_SCHEMA = "swebench_triad_runtime_lane_lease_v2"

    def __init__(
        self,
        root: Path | str,
        *,
        owner: OwnerIdentity,
        assigned_task_indices: Sequence[int],
        now_ns: Callable[[], int] = time.time_ns,
        ttl_ns: int = 90_000_000_000,
        heartbeat_interval_seconds: float = 15.0,
        local_owner_is_alive: Callable[[OwnerIdentity], bool] | None = None,
        slot_ports: Sequence[int] = (18100,),
    ) -> None:
        if not isinstance(owner, OwnerIdentity):
            raise TypeError("driver lease owner must be an OwnerIdentity")
        tasks = tuple(assigned_task_indices)
        if (
            not tasks
            or any(type(task) is not int or task < 0 for task in tasks)
            or tuple(sorted(set(tasks))) != tasks
        ):
            raise ValueError("driver shard must be a sorted unique nonempty task list")
        if not callable(now_ns):
            raise TypeError("driver lease clock must be callable")
        if type(ttl_ns) is not int or ttl_ns <= 0:
            raise ValueError("driver lease TTL must be positive")
        if heartbeat_interval_seconds <= 0 or (
            heartbeat_interval_seconds * 1_000_000_000 >= ttl_ns
        ):
            raise ValueError("driver heartbeat interval must be shorter than its TTL")
        if local_owner_is_alive is not None and not callable(local_owner_is_alive):
            raise TypeError("local driver liveness probe must be callable")
        ports = tuple(slot_ports)
        if (
            not ports
            or len(ports) > 2
            or any(type(port) is not int or not 1 <= port <= 65535 for port in ports)
            or len(set(ports)) != len(ports)
        ):
            raise ValueError("runtime slot ports must be one or two unique ports")
        self.root = ensure_private_directory(root)
        self.drivers_root = ensure_private_directory(self.root / "drivers")
        self.tasks_root = ensure_private_directory(self.root / "tasks")
        self.lanes_root = ensure_private_directory(self.root / "lanes")
        self.owner = owner
        self.assigned_task_indices = tasks
        self.now_ns = now_ns
        self.ttl_ns = ttl_ns
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.local_owner_is_alive = local_owner_is_alive or (lambda _owner: True)
        self.slot_ports = ports
        self.driver_key = sha256_json(owner.to_payload())
        self.lease_id: str | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error: BaseException | None = None
        self._liveness_descriptor: int | None = None

    @property
    def lock_path(self) -> Path:
        return self.root / "registry.lock"

    def driver_path(self, driver_key: str | None = None) -> Path:
        value = self.driver_key if driver_key is None else driver_key
        require_sha256(value, "driver lease key")
        return self.drivers_root / f"{value}.json"

    def liveness_lock_path(self, driver_key: str | None = None) -> Path:
        value = self.driver_key if driver_key is None else driver_key
        require_sha256(value, "driver liveness key")
        return self.drivers_root / f"{value}.liveness.lock"

    def _acquire_process_liveness_lock(self) -> bool:
        if self._liveness_descriptor is not None:
            return False
        path = self.liveness_lock_path()
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise ClaimBusyError("driver identity has a live cross-host lock") from error
        except BaseException:
            os.close(descriptor)
            raise
        self._liveness_descriptor = descriptor
        return True

    def _close_process_liveness_lock(self) -> None:
        descriptor = self._liveness_descriptor
        self._liveness_descriptor = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _driver_lock_is_held(self, driver_key: str) -> bool:
        if driver_key == self.driver_key and self._liveness_descriptor is not None:
            return True
        path = self.liveness_lock_path(driver_key)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                return False
        finally:
            os.close(descriptor)

    def task_path(self, task_index: int) -> Path:
        if type(task_index) is not int or task_index < 0:
            raise ValueError("task lease index is invalid")
        return self.tasks_root / f"task-{task_index:04d}.json"

    def lane_path(self, slot_index: int) -> Path:
        if type(slot_index) is not int or not 0 <= slot_index < len(self.slot_ports):
            raise ValueError("runtime lane slot is outside the configured lattice")
        return self.lanes_root / f"slot-{slot_index}.json"

    @staticmethod
    def _driver_fields() -> set[str]:
        return {
            "schema",
            "lease_id",
            "driver_key",
            "owner",
            "assigned_task_indices",
            "acquired_at_ns",
            "heartbeat_at_ns",
            "expires_at_ns",
            "heartbeat_sequence",
            "status",
        }

    def _validate_driver(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != self._driver_fields():
            raise FenceViolationError("driver lease fields are not canonical")
        if value.get("schema") != self.DRIVER_SCHEMA:
            raise FenceViolationError("driver lease schema drifted")
        lease_id = value.get("lease_id")
        driver_key = value.get("driver_key")
        if (
            not isinstance(lease_id, str)
            or len(lease_id) != 64
            or any(character not in "0123456789abcdef" for character in lease_id)
        ):
            raise FenceViolationError("driver lease ID is invalid")
        require_sha256(driver_key, "driver lease key")
        owner = OwnerIdentity.from_payload(value.get("owner"))
        if sha256_json(owner.to_payload()) != driver_key:
            raise FenceViolationError("driver lease owner digest drifted")
        tasks = value.get("assigned_task_indices")
        if (
            not isinstance(tasks, list)
            or not tasks
            or any(type(task) is not int or task < 0 for task in tasks)
            or sorted(set(tasks)) != tasks
        ):
            raise FenceViolationError("driver lease shard is invalid")
        for name in (
            "acquired_at_ns",
            "heartbeat_at_ns",
            "expires_at_ns",
            "heartbeat_sequence",
        ):
            if type(value.get(name)) is not int or value[name] < 0:
                raise FenceViolationError(f"driver lease {name} is invalid")
        if value["heartbeat_sequence"] <= 0:
            raise FenceViolationError("driver heartbeat sequence is invalid")
        if not (
            value["acquired_at_ns"] <= value["heartbeat_at_ns"]
            < value["expires_at_ns"]
        ):
            raise FenceViolationError("driver lease time bounds are invalid")
        if value.get("status") not in {"active", "released"}:
            raise FenceViolationError("driver lease status is invalid")
        return dict(value)

    def _read_driver(self, driver_key: str) -> dict[str, Any] | None:
        path = self.driver_path(driver_key)
        if not path.exists():
            return None
        return self._validate_driver(read_json(path))

    def _driver_is_live(self, value: Mapping[str, Any]) -> bool:
        driver = self._validate_driver(value)
        if driver["status"] != "active":
            return False
        # The held cross-host advisory lock is the destructive-cleanup fence.
        # Heartbeat expiry is evidence, but never overrules a still-held lock.
        return self._driver_lock_is_held(driver["driver_key"])

    def _validate_task_pointer(self, value: Any, task_index: int) -> dict[str, Any]:
        expected = {
            "schema",
            "task_index",
            "driver_key",
            "lease_id",
            "owner",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise FenceViolationError("task lease fields are not canonical")
        if value.get("schema") != self.TASK_SCHEMA or value.get("task_index") != task_index:
            raise FenceViolationError("task lease identity drifted")
        require_sha256(value.get("driver_key"), "task lease driver key")
        lease_id = value.get("lease_id")
        if not isinstance(lease_id, str) or len(lease_id) != 64:
            raise FenceViolationError("task lease ID is invalid")
        owner = OwnerIdentity.from_payload(value.get("owner"))
        if sha256_json(owner.to_payload()) != value["driver_key"]:
            raise FenceViolationError("task lease owner digest drifted")
        return dict(value)

    def _task_pointer(self, task_index: int) -> dict[str, Any] | None:
        path = self.task_path(task_index)
        if not path.exists():
            return None
        return self._validate_task_pointer(read_json(path), task_index)

    def _pointer_driver_is_live(self, pointer: Mapping[str, Any]) -> bool:
        driver = self._read_driver(pointer["driver_key"])
        if driver is None or driver["lease_id"] != pointer["lease_id"]:
            return False
        return self._driver_is_live(driver)

    def acquire(self) -> str:
        opened_here = self._acquire_process_liveness_lock()
        try:
            with exclusive_lock(self.lock_path):
                current = self._read_driver(self.driver_key)
                if (
                    current is not None
                    and self.lease_id is not None
                    and current["lease_id"] == self.lease_id
                    and current["status"] == "active"
                ):
                    self._refresh_locked(current)
                    return self.lease_id
                for task_index in self.assigned_task_indices:
                    pointer = self._task_pointer(task_index)
                    if pointer is not None and self._pointer_driver_is_live(pointer):
                        raise ClaimBusyError(
                            f"task shard has a live driver lease: {task_index:04d}"
                        )
                now = self.now_ns()
                lease_id = secrets.token_hex(32)
                driver = {
                    "schema": self.DRIVER_SCHEMA,
                    "lease_id": lease_id,
                    "driver_key": self.driver_key,
                    "owner": self.owner.to_payload(),
                    "assigned_task_indices": list(self.assigned_task_indices),
                    "acquired_at_ns": now,
                    "heartbeat_at_ns": now,
                    "expires_at_ns": now + self.ttl_ns,
                    "heartbeat_sequence": 1,
                    "status": "active",
                }
                atomic_write_json(self.driver_path(), driver)
                for task_index in self.assigned_task_indices:
                    atomic_write_json(
                        self.task_path(task_index),
                        {
                            "schema": self.TASK_SCHEMA,
                            "task_index": task_index,
                            "driver_key": self.driver_key,
                            "lease_id": lease_id,
                            "owner": self.owner.to_payload(),
                        },
                    )
                self.lease_id = lease_id
                return lease_id
        except BaseException:
            if opened_here:
                self._close_process_liveness_lock()
            raise

    def _refresh_locked(self, current: Mapping[str, Any]) -> None:
        if self.lease_id is None or current.get("lease_id") != self.lease_id:
            raise FenceViolationError("driver lease was fenced")
        if current.get("status") != "active":
            raise FenceViolationError("driver lease is no longer active")
        now = self.now_ns()
        refreshed = dict(current)
        refreshed["heartbeat_at_ns"] = now
        refreshed["expires_at_ns"] = now + self.ttl_ns
        refreshed["heartbeat_sequence"] += 1
        atomic_write_json(self.driver_path(), refreshed)

    def refresh(self) -> None:
        with exclusive_lock(self.lock_path):
            current = self._read_driver(self.driver_key)
            if current is None:
                raise FenceViolationError("driver lease disappeared")
            self._refresh_locked(current)

    def start_heartbeat(self) -> str:
        lease_id = self.acquire()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self.assert_healthy()
            return lease_id
        self._heartbeat_stop.clear()
        self._heartbeat_error = None

        def heartbeat() -> None:
            while not self._heartbeat_stop.wait(self.heartbeat_interval_seconds):
                try:
                    self.refresh()
                except BaseException as error:
                    self._heartbeat_error = error
                    return

        self._heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"swebench-triad-lease-{self.driver_key[:12]}",
            daemon=True,
        )
        self._heartbeat_thread.start()
        return lease_id

    def assert_healthy(self) -> None:
        if self._heartbeat_error is not None:
            raise FenceViolationError("driver lease heartbeat failed") from self._heartbeat_error
        if self.lease_id is None:
            raise FenceViolationError("driver lease has not been acquired")

    def owner_is_alive(self, owner: OwnerIdentity) -> bool:
        if not isinstance(owner, OwnerIdentity):
            raise TypeError("lease liveness requires an OwnerIdentity")
        key = sha256_json(owner.to_payload())
        driver = self._read_driver(key)
        if driver is None or driver["owner"] != owner.to_payload():
            return False
        if not self._driver_is_live(driver):
            return False
        for task_index in driver["assigned_task_indices"]:
            pointer = self._task_pointer(task_index)
            if (
                pointer is None
                or pointer["driver_key"] != key
                or pointer["lease_id"] != driver["lease_id"]
            ):
                return False
        return True

    def _validate_lane(self, value: Any) -> dict[str, Any]:
        expected = {
            "schema",
            "driver_key",
            "lease_id",
            "owner",
            "task_index",
            "slot_index",
            "server_port",
            "generation",
            "fencing_token",
            "acquired_at_ns",
            "status",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise FenceViolationError("runtime lane lease fields are not canonical")
        if value.get("schema") != self.LANE_SCHEMA:
            raise FenceViolationError("runtime lane lease schema drifted")
        require_sha256(value.get("driver_key"), "runtime lane driver key")
        owner = OwnerIdentity.from_payload(value.get("owner"))
        if sha256_json(owner.to_payload()) != value["driver_key"]:
            raise FenceViolationError("runtime lane owner digest drifted")
        if not isinstance(value.get("lease_id"), str) or len(value["lease_id"]) != 64:
            raise FenceViolationError("runtime lane lease ID is invalid")
        task_index = value.get("task_index")
        if task_index is not None and (type(task_index) is not int or task_index < 0):
            raise FenceViolationError("runtime lane task index is invalid")
        slot_index = value.get("slot_index")
        if (
            type(slot_index) is not int
            or not 0 <= slot_index < len(self.slot_ports)
        ):
            raise FenceViolationError("runtime lane slot index is invalid")
        if value.get("server_port") != self.slot_ports[slot_index]:
            raise FenceViolationError("runtime lane server port drifted")
        if type(value.get("generation")) is not int or value["generation"] <= 0:
            raise FenceViolationError("runtime lane generation is invalid")
        fencing_token = value.get("fencing_token")
        if (
            not isinstance(fencing_token, str)
            or len(fencing_token) != 64
            or any(
                character not in "0123456789abcdef"
                for character in fencing_token
            )
        ):
            raise FenceViolationError("runtime lane fencing token is invalid")
        if type(value.get("acquired_at_ns")) is not int or value["acquired_at_ns"] < 0:
            raise FenceViolationError("runtime lane acquisition time is invalid")
        if value.get("status") not in {"active", "released"}:
            raise FenceViolationError("runtime lane status is invalid")
        return dict(value)

    @staticmethod
    def _lane_token(value: Mapping[str, Any]) -> RuntimeLaneToken:
        return RuntimeLaneToken(
            driver_key=value["driver_key"],
            lease_id=value["lease_id"],
            owner=OwnerIdentity.from_payload(value["owner"]),
            task_index=value["task_index"],
            slot_index=value["slot_index"],
            server_port=value["server_port"],
            generation=value["generation"],
            fencing_token=value["fencing_token"],
        )

    def acquire_lane(
        self,
        *,
        task_index: int | None,
        slot_index: int,
    ) -> RuntimeLaneToken:
        if task_index is not None and task_index not in self.assigned_task_indices:
            raise ValueError("runtime lane task is outside the assigned shard")
        path = self.lane_path(slot_index)
        self.assert_healthy() if self.lease_id is not None else self.acquire()
        with exclusive_lock(self.lock_path):
            current_driver = self._read_driver(self.driver_key)
            if current_driver is None:
                raise FenceViolationError("runtime lane driver lease disappeared")
            self._refresh_locked(current_driver)
            previous_generation = 0
            if path.exists():
                lane = self._validate_lane(read_json(path))
                if lane["slot_index"] != slot_index:
                    raise FenceViolationError("runtime lane path identity drifted")
                previous_generation = lane["generation"]
                if (
                    lane["status"] == "active"
                    and lane["task_index"] == task_index
                    and lane["driver_key"] == self.driver_key
                    and lane["lease_id"] == self.lease_id
                ):
                    return self._lane_token(lane)
                if lane["status"] == "active" and self._pointer_driver_is_live(lane):
                    raise ClaimBusyError("runtime lane has a live driver lease")
            lane = {
                "schema": self.LANE_SCHEMA,
                "driver_key": self.driver_key,
                "lease_id": self.lease_id,
                "owner": self.owner.to_payload(),
                "task_index": task_index,
                "slot_index": slot_index,
                "server_port": self.slot_ports[slot_index],
                "generation": previous_generation + 1,
                "fencing_token": secrets.token_hex(32),
                "acquired_at_ns": self.now_ns(),
                "status": "active",
            }
            atomic_write_json(path, lane)
            return self._lane_token(lane)

    def assert_lane(self, token: RuntimeLaneToken) -> None:
        if not isinstance(token, RuntimeLaneToken):
            raise TypeError("runtime lane fence requires a lane token")
        with exclusive_lock(self.lock_path):
            path = self.lane_path(token.slot_index)
            if not path.exists():
                raise FenceViolationError("runtime lane lease disappeared")
            lane = self._validate_lane(read_json(path))
            if lane["status"] != "active" or self._lane_token(lane) != token:
                raise FenceViolationError("runtime lane token was fenced")
            current_driver = self._read_driver(self.driver_key)
            if current_driver is None:
                raise FenceViolationError("runtime lane driver lease disappeared")
            self._refresh_locked(current_driver)

    def release_lane(self, token: RuntimeLaneToken) -> None:
        if not isinstance(token, RuntimeLaneToken):
            raise TypeError("runtime lane release requires a lane token")
        with exclusive_lock(self.lock_path):
            path = self.lane_path(token.slot_index)
            if not path.exists():
                raise FenceViolationError("runtime lane lease disappeared")
            lane = self._validate_lane(read_json(path))
            if lane["status"] != "active" or self._lane_token(lane) != token:
                raise FenceViolationError("refusing to release a fenced runtime lane")
            lane["status"] = "released"
            atomic_write_json(path, lane)

    def assert_no_other_live_drivers(self) -> None:
        for path in sorted(self.drivers_root.glob("*.json")):
            driver = self._validate_driver(read_json(path))
            if driver["driver_key"] == self.driver_key:
                continue
            if self._driver_is_live(driver):
                raise ClaimBusyError("cleanup found another live driver lease")

    def release(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self.heartbeat_interval_seconds * 2))
        try:
            with exclusive_lock(self.lock_path):
                if self.lease_id is None:
                    return
                for slot_index in range(len(self.slot_ports)):
                    path = self.lane_path(slot_index)
                    if not path.exists():
                        continue
                    lane = self._validate_lane(read_json(path))
                    if (
                        lane["status"] == "active"
                        and lane["driver_key"] == self.driver_key
                        and lane["lease_id"] == self.lease_id
                    ):
                        lane["status"] = "released"
                        atomic_write_json(path, lane)
                for task_index in self.assigned_task_indices:
                    path = self.task_path(task_index)
                    if not path.exists():
                        continue
                    pointer = self._validate_task_pointer(read_json(path), task_index)
                    if (
                        pointer["driver_key"] == self.driver_key
                        and pointer["lease_id"] == self.lease_id
                    ):
                        path.unlink()
                driver = self._read_driver(self.driver_key)
                if driver is not None and driver["lease_id"] == self.lease_id:
                    now = self.now_ns()
                    driver["heartbeat_at_ns"] = max(
                        driver["acquired_at_ns"],
                        min(now, driver["expires_at_ns"] - 1),
                    )
                    driver["expires_at_ns"] = max(
                        driver["heartbeat_at_ns"] + 1, now + 1
                    )
                    driver["heartbeat_sequence"] += 1
                    driver["status"] = "released"
                    atomic_write_json(self.driver_path(), driver)
                self.lease_id = None
        finally:
            self._close_process_liveness_lock()


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
        for name in (
            "claims",
            "locks",
            "attempts",
            "accepted",
            "outcomes",
            "grade-claims",
            "grade-locks",
        ):
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

    def grade_claim_path(self, key: CellKey) -> Path:
        return self.root / "grade-claims" / f"{key.slug}.json"

    def grade_lock_path(self, key: CellKey) -> Path:
        return self.root / "grade-locks" / f"{key.slug}.lock"

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
                self.read_accepted(key)
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

    def acquire_grade(self, key: CellKey) -> GradeClaimToken:
        self.cell(key)
        with exclusive_lock(self.grade_lock_path(key)):
            if self.outcome_path(key).exists():
                self.read_official_outcome(key)
                raise AlreadyGradedError(f"cell is already graded: {key.slug}")
            accepted = self.read_accepted(key)
            accepted_sha256 = sha256_json(accepted)
            path = self.grade_claim_path(key)
            generation = 1
            if path.exists():
                claim = self.read_grade_claim(path, key)
                if claim.accepted_sha256 != accepted_sha256:
                    raise FenceViolationError(
                        "grading claim accepted-cell digest drifted"
                    )
                if claim.owner == self.owner:
                    return claim
                if self.owner_is_alive(claim.owner):
                    raise ClaimBusyError(
                        f"official grading has a live owner: {key.slug}"
                    )
                generation = claim.generation + 1
            token = GradeClaimToken(
                key=key,
                generation=generation,
                accepted_sha256=accepted_sha256,
                owner=self.owner,
            )
            atomic_write_json(path, self.grade_claim_payload(token))
            return token

    @staticmethod
    def grade_claim_payload(token: GradeClaimToken) -> dict[str, Any]:
        return {
            "schema": "swebench_triad_fenced_grade_claim_v1",
            "cell": token.key.to_payload(),
            "generation": token.generation,
            "accepted_sha256": token.accepted_sha256,
            "owner": token.owner.to_payload(),
        }

    @staticmethod
    def read_grade_claim(path: Path, expected_key: CellKey) -> GradeClaimToken:
        payload = read_json(path)
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema",
            "cell",
            "generation",
            "accepted_sha256",
            "owner",
        }:
            raise FenceViolationError("grading claim fields are not canonical")
        if payload.get("schema") != "swebench_triad_fenced_grade_claim_v1":
            raise FenceViolationError("grading claim schema drifted")
        cell_payload = payload.get("cell")
        if not isinstance(cell_payload, Mapping):
            raise FenceViolationError("grading claim cell is invalid")
        key = CellKey(cell_payload.get("task_index"), cell_payload.get("arm"))
        if key != expected_key:
            raise FenceViolationError("grading claim cell identity drifted")
        generation = payload.get("generation")
        if type(generation) is not int or generation <= 0:
            raise FenceViolationError("grading claim generation is invalid")
        return GradeClaimToken(
            key=key,
            generation=generation,
            accepted_sha256=require_sha256(
                payload.get("accepted_sha256"), "grading claim accepted cell"
            ),
            owner=OwnerIdentity.from_payload(payload.get("owner")),
        )

    def assert_grade_fence(self, token: GradeClaimToken) -> None:
        if not isinstance(token, GradeClaimToken):
            raise TypeError("official outcome write requires a grading claim token")
        current = self.read_grade_claim(
            self.grade_claim_path(token.key), token.key
        )
        if current != token:
            raise FenceViolationError(
                "grading claim generation or owner was fenced"
            )
        accepted = self.read_accepted(token.key)
        if sha256_json(accepted) != token.accepted_sha256:
            raise FenceViolationError("accepted cell changed after grading claim")

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
        instance_id = row.get("instance_id")
        task_id = row.get("task_id")
        if instance_id is not None and task_id is not None and instance_id != task_id:
            raise ValueError("endpoint row cell identity drifted")
        endpoint_id = task_id if task_id is not None else instance_id
        if endpoint_id != cell.instance_id or row.get("arm") != key.arm:
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
        prediction = read_json(
            self.artifact_path(token.key, token.generation, "prediction")
        )
        self.validate_handoff_artifact(prediction, handoff)

    @staticmethod
    def validate_handoff_artifact(prediction: Any, handoff: Any) -> None:
        if not isinstance(handoff, Mapping) or set(handoff) != {
            "prediction_sha256",
            "official_resolved",
            "grader_revision",
        }:
            raise ValueError("grader handoff fields are not canonical")
        if handoff["official_resolved"] is not None:
            raise ValueError("queued handoff cannot claim an official outcome")
        require_sha256(handoff["prediction_sha256"], "handoff prediction")
        if handoff["prediction_sha256"] != sha256_json(prediction):
            raise ValueError("grader handoff prediction digest drifted")
        if handoff["grader_revision"] != (
            "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
        ):
            raise ValueError("grader handoff revision drifted")

    def _validate_accepted_and_artifacts(
        self,
        key: CellKey,
        value: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cell = self.cell(key)
        expected = {
            "schema",
            "cell",
            "instance_id",
            "manifest_cell_sha256",
            "attempt_generation",
            "endpoint_sha256",
            "prediction_sha256",
            "handoff_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("accepted cell fields are not canonical")
        if value.get("schema") != "swebench_triad_accepted_cell_v1":
            raise ValueError("accepted cell schema drifted")
        if value.get("cell") != key.to_payload():
            raise ValueError("accepted cell identity drifted")
        if value.get("instance_id") != cell.instance_id:
            raise ValueError("accepted cell instance ID drifted")
        manifest_sha256 = require_sha256(
            value.get("manifest_cell_sha256"), "accepted manifest cell"
        )
        if manifest_sha256 != cell.manifest_cell_sha256:
            raise ValueError("accepted cell manifest digest drifted")
        generation = value.get("attempt_generation")
        if type(generation) is not int or generation <= 0:
            raise ValueError("accepted attempt generation is invalid")
        expected_digests = {}
        for name in ("endpoint", "prediction", "handoff"):
            expected_digests[name] = require_sha256(
                value.get(f"{name}_sha256"), f"accepted {name}"
            )

        artifacts: dict[str, Any] = {}
        for name in ("endpoint", "prediction", "handoff"):
            try:
                artifacts[name] = read_json(
                    self.artifact_path(key, generation, name)
                )
            except (OSError, RuntimeError) as error:
                raise ValueError(f"accepted {name} artifact is unavailable") from error
            if sha256_json(artifacts[name]) != expected_digests[name]:
                raise ValueError(f"accepted {name} digest drifted")

        try:
            self.validate_endpoint(key, artifacts["endpoint"])
            self.validate_prediction(key, artifacts["prediction"])
            self.validate_handoff_artifact(
                artifacts["prediction"], artifacts["handoff"]
            )
        except (KeyError, RuntimeError, TypeError) as error:
            raise ValueError("accepted attempt artifact is invalid") from error
        return dict(value), artifacts

    def validate_accepted(self, key: CellKey, value: Any) -> dict[str, Any]:
        accepted, _ = self._validate_accepted_and_artifacts(key, value)
        return accepted

    def accepted_artifacts(
        self,
        key: CellKey,
        value: Any,
    ) -> dict[str, Any]:
        _, artifacts = self._validate_accepted_and_artifacts(key, value)
        return artifacts

    def read_accepted(self, key: CellKey) -> dict[str, Any]:
        path = self.accepted_path(key)
        if not path.exists():
            raise ValueError("cell does not have an accepted attempt")
        try:
            value = read_json(path)
        except (OSError, RuntimeError) as error:
            raise ValueError("accepted cell record is unavailable") from error
        return self.validate_accepted(key, value)

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
        self.validate_handoff_artifact(prediction, handoff)
        if handoff["prediction_sha256"] != prediction_sha256:
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
            return self.read_accepted(token.key)
        self.assert_fence(token)
        artifacts = self.complete_attempt(token.key, token.generation)
        if artifacts is None:
            raise ValueError("cell attempt is missing a durable boundary")
        accepted = self.accepted_payload(token.key, token.generation, artifacts)
        write_immutable_json(existing_path, accepted)
        return self.validate_accepted(token.key, accepted)

    def reconcile_complete_attempt(
        self,
        token: ClaimToken,
    ) -> dict[str, Any] | None:
        if self.accepted_path(token.key).exists():
            return self.read_accepted(token.key)
        self.assert_fence(token)
        for generation in range(token.generation - 1, 0, -1):
            artifacts = self.complete_attempt(token.key, generation)
            if artifacts is None:
                continue
            accepted = self.accepted_payload(token.key, generation, artifacts)
            write_immutable_json(self.accepted_path(token.key), accepted)
            return self.validate_accepted(token.key, accepted)
        return None

    def assemble_results(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cell in self.manifest:
            path = self.accepted_path(cell.key)
            if not path.exists():
                raise ValueError("accepted endpoint denominator is incomplete")
            accepted = self.read_accepted(cell.key)
            endpoint = self.accepted_artifacts(cell.key, accepted)["endpoint"]
            rows.append(endpoint)
        return rows

    def validate_official_outcome(
        self,
        key: CellKey,
        value: Any,
    ) -> dict[str, Any]:
        cell = self.cell(key)
        expected = {
            "schema",
            "instance_id",
            "arm",
            "resolved",
            "failure_class",
            "report_sha256",
            "prediction_sha256",
            "attempt_generation",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("official outcome fields are not canonical")
        if value.get("schema") != "swebench_triad_official_outcome_v1":
            raise ValueError("official outcome schema drifted")
        if (
            value.get("instance_id") != cell.instance_id
            or value.get("arm") != key.arm
        ):
            raise ValueError("official outcome cell identity drifted")
        if type(value.get("resolved")) is not bool:
            raise ValueError("official outcome must be boolean")
        failure_class = value.get("failure_class")
        if failure_class is not None and (
            not isinstance(failure_class, str) or not failure_class
        ):
            raise ValueError("official failure class is invalid")
        require_sha256(value.get("report_sha256"), "official report")
        require_sha256(value.get("prediction_sha256"), "official prediction")
        generation = value.get("attempt_generation")
        if type(generation) is not int or generation <= 0:
            raise ValueError("official outcome attempt generation is invalid")
        accepted = self.read_accepted(key)
        if value["prediction_sha256"] != accepted["prediction_sha256"]:
            raise ValueError("official outcome prediction digest drifted")
        if generation != accepted["attempt_generation"]:
            raise ValueError("official outcome attempt generation drifted")
        return dict(value)

    def read_official_outcome(self, key: CellKey) -> dict[str, Any]:
        path = self.outcome_path(key)
        if not path.exists():
            raise ValueError("cell does not have an official outcome")
        try:
            value = read_json(path)
        except (OSError, RuntimeError) as error:
            raise ValueError("official outcome record is unavailable") from error
        return self.validate_official_outcome(key, value)

    def record_official_outcome(
        self,
        token: GradeClaimToken,
        outcome: Mapping[str, Any],
    ) -> Path:
        if not isinstance(token, GradeClaimToken):
            raise TypeError("official outcome write requires a grading claim token")
        key = token.key
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
        accepted = self.read_accepted(key)
        payload = {
            "schema": "swebench_triad_official_outcome_v1",
            **dict(outcome),
            "prediction_sha256": accepted["prediction_sha256"],
            "attempt_generation": accepted["attempt_generation"],
        }
        self.validate_official_outcome(key, payload)
        with exclusive_lock(self.grade_lock_path(key)):
            self.assert_grade_fence(token)
            return write_immutable_json(self.outcome_path(key), payload)

    def official_summary(self) -> dict[str, Any]:
        by_arm: dict[str, list[bool]] = {arm: [] for arm in ARMS}
        for cell in self.manifest:
            path = self.outcome_path(cell.key)
            if not path.exists():
                raise ValueError("official outcome denominator is incomplete")
            outcome = self.read_official_outcome(cell.key)
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
    "DriverLeaseRegistry",
    "FenceViolationError",
    "ManifestCell",
    "OwnerIdentity",
]
