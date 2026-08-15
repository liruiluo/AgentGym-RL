#!/usr/bin/env python3
"""Verify the public OpenMLE-fast resident endpoint contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Protocol
import urllib.error
import urllib.request


MANIFEST_SCHEMA = "openmle_fast_public_manifest_v1"
METADATA_SCHEMA = "openmle_fast_public_metadata_v1"
DOMAIN_ID = "openmle_fast"
CONTRACT_VERSION = "openmle_fast_v1"
EXPECTED_OPENMLE_TASKS_REVISION = "f56e4b31252a9b81d95fea100098cd49b7290398"
REQUIRED_CONTRACTS = (
    "action",
    "observation",
    "horizon",
    "workspace",
    "executor",
    "grader_boundary",
    "cleanup",
)
COUNTER_KEYS = (
    "action_count",
    "execution_action_count",
    "execution_attempt_count",
    "execution_completed_count",
    "nested_subprocess_count",
    "fit_count",
    "grading_count",
)
FORBIDDEN_PUBLIC_KEYS = (
    "credential",
    "detail_token",
    "grader_socket",
    "private_manifest",
    "private_path",
    "private_root",
    "secret",
    "traceback",
)


class EndpointProtocol(Protocol):
    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def request_status(self, method: str, path: str) -> tuple[int, bytes]: ...


class Endpoint:
    """Small JSON-over-HTTP adapter used by the command-line verifier."""

    def __init__(self, base_url: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> tuple[int, bytes]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read()

    def request_status(self, method: str, path: str) -> tuple[int, bytes]:
        return self._request(method, path, None)

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        status, raw = self._request(method, path, payload)
        if status < 200 or status >= 300:
            detail = raw.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                f"OpenMLE-fast {method} {path} failed: HTTP {status}: {detail}"
            )
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"OpenMLE-fast {method} {path} returned invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"OpenMLE-fast {method} {path} returned non-object JSON")
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--indices", default="0,1")
    parser.add_argument("--expected-outer-commit", required=True)
    parser.add_argument("--expected-inner-commit", required=True)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--client-timeout-seconds", type=int, required=True)
    parser.add_argument("--timeout-margin-seconds", type=int, required=True)
    parser.add_argument("--forbidden-canaries-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_probe_indices(raw: str) -> list[int]:
    try:
        indices = [int(value) for value in raw.split(",") if value != ""]
    except ValueError as exc:
        raise ValueError("probe indices must be comma-separated integers") from exc
    if len(indices) != 2:
        raise ValueError("the isolation probe requires exactly 2 indices")
    if any(index < 0 for index in indices):
        raise ValueError("probe indices must be nonnegative")
    if len(set(indices)) != len(indices):
        raise ValueError("probe indices must be distinct")
    return indices


def canonical_sha256(document: dict[str, Any]) -> str:
    raw = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def load_forbidden_canaries(path: Path) -> list[str]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("forbidden canaries must come from a real file")
    if path.stat().st_mode & 0o077:
        raise ValueError("forbidden canaries file must have mode 0600 or stricter")
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [line for line in raw.splitlines() if line]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError("forbidden canaries file must contain nonempty strings")
    if len(value) != len(set(value)):
        raise ValueError("forbidden canaries must be distinct")
    return value


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, bool):
        matches = value is expected
    else:
        matches = value == expected
    if not matches:
        raise AssertionError(f"{label} mismatch: {value!r} != {expected!r}")


def require_public_safe(
    value: Any,
    *,
    label: str,
    forbidden_canaries: list[str],
) -> None:
    """Reject secrets, private-location keys, and caller-provided canaries."""

    def walk(current: Any, path: str) -> None:
        if isinstance(current, dict):
            for key, child in current.items():
                key_text = str(key)
                lowered = key_text.lower()
                if any(fragment in lowered for fragment in FORBIDDEN_PUBLIC_KEYS):
                    raise AssertionError(
                        f"{label} is not public-safe: forbidden key {path}.{key_text}"
                    )
                walk(child, f"{path}.{key_text}")
            return
        if isinstance(current, list):
            for index, child in enumerate(current):
                walk(child, f"{path}[{index}]")
            return
        if isinstance(current, str):
            for canary in forbidden_canaries:
                if canary and canary in current:
                    raise AssertionError(
                        f"{label} is not public-safe: canary at {path}"
                    )

    walk(value, "$")


def require_idle(metadata: dict[str, Any], *, label: str) -> None:
    for key in (
        "active_slot_count",
        "active_environment_count",
        "active_workspace_count",
    ):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise AssertionError(f"{label} {key} is not an integer: {value!r}")
        if value != 0:
            raise AssertionError(f"{label} {key} is not zero: {value!r}")


def require_counters(
    step: dict[str, Any],
    *,
    expected_action_count: int,
    expected_execution_count: int | None = None,
    previous: dict[str, int] | None = None,
) -> dict[str, int]:
    info = step.get("info")
    if not isinstance(info, dict):
        raise AssertionError("step info is missing")
    counters = info.get("counters")
    if not isinstance(counters, dict):
        raise AssertionError("step counters are missing")

    normalized: dict[str, int] = {}
    for key in COUNTER_KEYS:
        value = counters.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AssertionError(f"counter {key} is not a nonnegative integer")
        normalized[key] = value
    _require_exact(
        normalized["action_count"],
        expected_action_count,
        label="action_count",
    )
    if expected_execution_count is not None:
        for key in (
            "execution_action_count",
            "execution_attempt_count",
            "execution_completed_count",
        ):
            _require_exact(
                normalized[key],
                expected_execution_count,
                label=key,
            )
    if normalized["execution_action_count"] > normalized["action_count"]:
        raise AssertionError("execution_action_count exceeds action_count")
    if normalized["execution_completed_count"] > normalized["execution_attempt_count"]:
        raise AssertionError(
            "execution_completed_count exceeds execution_attempt_count"
        )
    if normalized["grading_count"] > 1:
        raise AssertionError("grading_count exceeds one terminal grade")
    if previous is not None:
        for key in COUNTER_KEYS:
            if normalized[key] < previous[key]:
                raise AssertionError(f"counter {key} is not monotone")
    return normalized


def require_step_identity(
    step: dict[str, Any],
    record: dict[str, Any],
    manifest_sha256: str,
    *,
    expected_action_count: int,
    expected_execution_count: int | None = None,
    previous_counters: dict[str, int] | None = None,
) -> dict[str, int]:
    reward = step.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise AssertionError("reward is not numeric")
    if not math.isfinite(float(reward)):
        raise AssertionError("reward is not finite")
    _require_exact(float(reward), 0.0, label="probe reward")
    _require_exact(step.get("done"), False, label="probe done")
    _require_exact(step.get("truncated"), False, label="probe truncated")
    info = step.get("info")
    if not isinstance(info, dict):
        raise AssertionError("step info is missing")
    for key in ("data_idx", "task_id", "source_family"):
        _require_exact(info.get(key), record[key], label=key)
    _require_exact(
        info.get("task_manifest_sha256"),
        manifest_sha256,
        label="task_manifest_sha256",
    )
    counters = require_counters(
        step,
        expected_action_count=expected_action_count,
        expected_execution_count=expected_execution_count,
        previous=previous_counters,
    )
    if counters["grading_count"]:
        raise AssertionError("grading occurred during a nonterminal probe")
    return counters


def _shell_action(command: str) -> str:
    return "shell_command " + json.dumps(
        {"command": command, "timeout_ms": 20000},
        separators=(",", ":"),
    )


def _managed_python_action(source: str) -> str:
    return _shell_action("python -c " + json.dumps(source))


def _create_slot(endpoint: EndpointProtocol) -> int:
    created = endpoint.request("POST", "create", {})
    slot_id = created.get("id")
    if isinstance(slot_id, bool) or not isinstance(slot_id, int):
        raise AssertionError(f"create returned invalid slot id: {slot_id!r}")
    return slot_id


def _close_slots(
    endpoint: EndpointProtocol,
    slot_ids: list[int],
) -> list[str]:
    errors: list[str] = []
    for slot_id in slot_ids:
        try:
            _close_slot(endpoint, slot_id)
        except Exception as exc:  # Preserve every cleanup failure for diagnosis.
            errors.append(f"slot {slot_id}: {type(exc).__name__}: {exc}")
    return errors


def _close_slot(
    endpoint: EndpointProtocol,
    slot_id: int,
    *,
    already_closed: bool = False,
) -> None:
    receipt = endpoint.request("POST", "close", {"id": slot_id})
    _require_exact(
        receipt.get("schema"),
        "openmle_fast_cleanup_receipt_v1",
        label="close receipt schema",
    )
    _require_exact(
        receipt.get("closed"),
        not already_closed,
        label="close receipt",
    )
    _require_exact(
        receipt.get("already_closed"),
        already_closed,
        label="idempotent close receipt",
    )


def _validate_manifest(
    document: dict[str, Any],
    manifest_sha256: str,
    probe_indices: list[int],
    manifest_bytes: bytes | None = None,
) -> dict[int, dict[str, Any]]:
    _require_exact(document.get("schema"), MANIFEST_SCHEMA, label="manifest schema")
    _require_exact(
        (
            canonical_sha256(document)
            if manifest_bytes is None
            else hashlib.sha256(manifest_bytes).hexdigest()
        ),
        manifest_sha256,
        label="manifest SHA-256",
    )
    _require_exact(
        document.get("openmle_tasks_revision"),
        EXPECTED_OPENMLE_TASKS_REVISION,
        label="manifest OpenMLE revision",
    )
    records = document.get("records")
    if not isinstance(records, list):
        raise AssertionError("manifest records are missing")
    _require_exact(document.get("task_count"), len(records), label="task_count")
    by_index: dict[int, dict[str, Any]] = {}
    for expected_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise AssertionError(f"manifest record {expected_index} is not an object")
        data_idx = record.get("data_idx")
        if isinstance(data_idx, bool) or data_idx != expected_index:
            raise AssertionError(
                f"manifest data_idx {data_idx!r} is not contiguous at {expected_index}"
            )
        for key in ("task_id", "source_family"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise AssertionError(f"manifest record {expected_index} lacks {key}")
        by_index[expected_index] = record
    for index in probe_indices:
        if index not in by_index:
            raise AssertionError(f"probe data_idx is outside the manifest: {index}")
    return by_index


def _validate_metadata(
    metadata: dict[str, Any],
    document: dict[str, Any],
    manifest_sha256: str,
    *,
    expected_outer_commit: str,
    expected_inner_commit: str,
    expected_prompt_sha256: str,
    client_timeout_seconds: int,
    timeout_margin_seconds: int,
) -> None:
    expected = {
        "schema": METADATA_SCHEMA,
        "domain_id": DOMAIN_ID,
        "contract_version": CONTRACT_VERSION,
        "panel_id": document["panel_id"],
        "role": document["role"],
        "task_count": document["task_count"],
        "task_manifest_sha256": manifest_sha256,
        "openmle_tasks_revision": document["openmle_tasks_revision"],
        "task_id_list_sha256": document["task_id_list_sha256"],
        "compact_panel_sha256": document["compact_panel_sha256"],
        "policy_prompt_sha256": expected_prompt_sha256,
    }
    for key, value in expected.items():
        _require_exact(metadata.get(key), value, label=f"metadata {key}")
    runtime_source = metadata.get("runtime_source")
    if not isinstance(runtime_source, dict):
        raise AssertionError("metadata runtime_source is missing")
    _require_exact(
        runtime_source.get("outer_commit"),
        expected_outer_commit,
        label="metadata outer_commit",
    )
    _require_exact(
        runtime_source.get("inner_commit"),
        expected_inner_commit,
        label="metadata inner_commit",
    )

    contracts = metadata.get("contracts")
    if not isinstance(contracts, dict):
        raise AssertionError("metadata contracts are missing")
    for key in REQUIRED_CONTRACTS:
        value = contracts.get(key)
        if not isinstance(value, str) or not value:
            raise AssertionError(f"metadata contract {key} is empty")

    limits = metadata.get("limits")
    if not isinstance(limits, dict):
        raise AssertionError("metadata limits are missing")
    _require_exact(
        limits.get("max_policy_actions"),
        document["max_policy_actions"],
        label="metadata max_policy_actions",
    )
    max_request_wall_seconds = limits.get("max_request_wall_seconds")
    if (
        isinstance(max_request_wall_seconds, bool)
        or not isinstance(max_request_wall_seconds, (int, float))
        or max_request_wall_seconds <= 0
    ):
        raise AssertionError("metadata max_request_wall_seconds is invalid")
    if timeout_margin_seconds < 0:
        raise AssertionError("timeout margin must be nonnegative")
    required_timeout = float(max_request_wall_seconds) + timeout_margin_seconds
    if client_timeout_seconds <= required_timeout:
        raise AssertionError(
            "client timeout inequality failed: "
            f"{client_timeout_seconds} <= {required_timeout}"
        )


def verify_resident_endpoint(
    endpoint: EndpointProtocol,
    document: dict[str, Any],
    manifest_sha256: str,
    *,
    probe_indices: list[int],
    expected_outer_commit: str,
    expected_inner_commit: str,
    expected_prompt_sha256: str,
    client_timeout_seconds: int,
    timeout_margin_seconds: int,
    forbidden_canaries: list[str],
    manifest_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Run public identity, isolation, reset, and cleanup probes."""

    if len(probe_indices) != 2 or len(set(probe_indices)) != 2:
        raise AssertionError("resident endpoint probe requires two distinct indices")
    if not forbidden_canaries:
        raise AssertionError("at least one out-of-band forbidden canary is required")
    records = _validate_manifest(
        document,
        manifest_sha256,
        probe_indices,
        manifest_bytes,
    )

    metadata_before = endpoint.request("GET", "metadata")
    require_public_safe(
        metadata_before,
        label="metadata",
        forbidden_canaries=forbidden_canaries,
    )
    _validate_metadata(
        metadata_before,
        document,
        manifest_sha256,
        expected_outer_commit=expected_outer_commit,
        expected_inner_commit=expected_inner_commit,
        expected_prompt_sha256=expected_prompt_sha256,
        client_timeout_seconds=client_timeout_seconds,
        timeout_margin_seconds=timeout_margin_seconds,
    )
    require_idle(metadata_before, label="before")

    detail_status, detail_body = endpoint.request_status("GET", "detail?id=0")
    if detail_status != 404:
        raise AssertionError(
            f"public /detail route must return 404, got {detail_status}"
        )
    require_public_safe(
        detail_body.decode("utf-8", errors="replace"),
        label="/detail response",
        forbidden_canaries=forbidden_canaries,
    )

    slot_ids: list[int] = []
    reset_steps: list[dict[str, Any]] = []
    active_metadata: dict[str, Any] | None = None
    cleanup_errors: list[str] = []
    try:
        isolation_slots = []
        for _ in range(2):
            slot_id = _create_slot(endpoint)
            isolation_slots.append(slot_id)
            slot_ids.append(slot_id)
        if len(set(isolation_slots)) != 2:
            raise AssertionError("create returned duplicate slot ids")

        markers = [
            "OPENMLE_FAST_SLOT_"
            + hashlib.sha256(f"{slot_id}:{index}".encode("utf-8")).hexdigest()
            for slot_id, index in zip(
                isolation_slots,
                probe_indices,
            )
        ]
        for slot_id, index in zip(
            isolation_slots,
            probe_indices,
        ):
            step = endpoint.request(
                "POST",
                "reset",
                {"id": slot_id, "data_idx": index},
            )
            require_step_identity(
                step,
                records[index],
                manifest_sha256,
                expected_action_count=0,
            )
            require_public_safe(
                step,
                label="reset response",
                forbidden_canaries=forbidden_canaries,
            )
            reset_steps.append(step)

        write_counters: list[dict[str, int]] = []
        for slot_id, index, marker in zip(
            isolation_slots,
            probe_indices,
            markers,
        ):
            step = endpoint.request(
                "POST",
                "step",
                {
                    "id": slot_id,
                    "action": _managed_python_action(
                        "from pathlib import Path; "
                        "Path('.openmle_endpoint_canary').write_text("
                        f"{marker!r}, encoding='utf-8')"
                    ),
                },
            )
            counters = require_step_identity(
                step,
                records[index],
                manifest_sha256,
                expected_action_count=1,
                expected_execution_count=1,
                previous_counters=require_counters(
                    reset_steps[len(write_counters)],
                    expected_action_count=0,
                ),
            )
            require_public_safe(
                step,
                label="write response",
                forbidden_canaries=forbidden_canaries,
            )
            write_counters.append(counters)

        for offset, (slot_id, index, marker) in enumerate(
            zip(isolation_slots, probe_indices, markers)
        ):
            step = endpoint.request(
                "POST",
                "step",
                {
                    "id": slot_id,
                    "action": _managed_python_action(
                        "from pathlib import Path; "
                        "print(Path('.openmle_endpoint_canary').read_text("
                        "encoding='utf-8'), end='')"
                    ),
                },
            )
            require_step_identity(
                step,
                records[index],
                manifest_sha256,
                expected_action_count=2,
                expected_execution_count=2,
                previous_counters=write_counters[offset],
            )
            observation = step.get("observation")
            if not isinstance(observation, str) or marker not in observation:
                raise AssertionError("slot did not retain its own isolation canary")
            if any(other in observation for other in markers if other != marker):
                raise AssertionError("cross-slot canary leak detected")
            require_public_safe(
                step,
                label="read response",
                forbidden_canaries=forbidden_canaries,
            )

        active_metadata = endpoint.request("GET", "metadata")
        require_public_safe(
            active_metadata,
            label="active metadata",
            forbidden_canaries=forbidden_canaries,
        )
        for key in (
            "active_slot_count",
            "active_environment_count",
            "active_workspace_count",
        ):
            _require_exact(active_metadata.get(key), 2, label=key)

        for slot_id in isolation_slots:
            _close_slot(endpoint, slot_id)
            slot_ids.remove(slot_id)

        reset_slot = _create_slot(endpoint)
        slot_ids.append(reset_slot)
        first_index, second_index = reversed(probe_indices)
        first_reset = endpoint.request(
            "POST",
            "reset",
            {"id": reset_slot, "data_idx": first_index},
        )
        first_counters = require_step_identity(
            first_reset,
            records[first_index],
            manifest_sha256,
            expected_action_count=0,
        )
        reset_marker = (
            "OPENMLE_FAST_CROSS_RESET_"
            + hashlib.sha256(f"{reset_slot}:{first_index}".encode("utf-8")).hexdigest()
        )
        wrote = endpoint.request(
            "POST",
            "step",
            {
                "id": reset_slot,
                "action": _managed_python_action(
                    "from pathlib import Path; "
                    "Path('.openmle_endpoint_canary').write_text("
                    f"{reset_marker!r}, encoding='utf-8')"
                ),
            },
        )
        require_step_identity(
            wrote,
            records[first_index],
            manifest_sha256,
            expected_action_count=1,
            expected_execution_count=1,
            previous_counters=first_counters,
        )

        second_reset = endpoint.request(
            "POST",
            "reset",
            {"id": reset_slot, "data_idx": second_index},
        )
        second_counters = require_step_identity(
            second_reset,
            records[second_index],
            manifest_sha256,
            expected_action_count=0,
        )
        clean_probe = endpoint.request(
            "POST",
            "step",
            {
                "id": reset_slot,
                "action": _managed_python_action(
                    "from pathlib import Path; "
                    "p = Path('.openmle_endpoint_canary'); "
                    "print(p.read_text(encoding='utf-8') if p.exists() "
                    "else 'OPENMLE_FAST_RESET_CLEAN', end='')"
                ),
            },
        )
        require_step_identity(
            clean_probe,
            records[second_index],
            manifest_sha256,
            expected_action_count=1,
            expected_execution_count=1,
            previous_counters=second_counters,
        )
        clean_observation = clean_probe.get("observation")
        if (
            not isinstance(clean_observation, str)
            or "OPENMLE_FAST_RESET_CLEAN" not in clean_observation
            or reset_marker in clean_observation
        ):
            raise AssertionError("cross-reset canary leak detected")
        require_public_safe(
            clean_probe,
            label="cross-reset response",
            forbidden_canaries=forbidden_canaries,
        )
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors.extend(_close_slots(endpoint, slot_ids))
        if slot_ids:
            try:
                _close_slot(endpoint, slot_ids[-1], already_closed=True)
            except Exception as exc:
                cleanup_errors.append(f"idempotent close: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            message = "OpenMLE-fast slot cleanup failed: " + "; ".join(cleanup_errors)
            if active_error is not None:
                if hasattr(active_error, "add_note"):
                    active_error.add_note(message)
                else:
                    raise RuntimeError(message) from active_error
            else:
                raise RuntimeError(message)

    metadata_after = endpoint.request("GET", "metadata")
    require_public_safe(
        metadata_after,
        label="metadata after cleanup",
        forbidden_canaries=forbidden_canaries,
    )
    _validate_metadata(
        metadata_after,
        document,
        manifest_sha256,
        expected_outer_commit=expected_outer_commit,
        expected_inner_commit=expected_inner_commit,
        expected_prompt_sha256=expected_prompt_sha256,
        client_timeout_seconds=client_timeout_seconds,
        timeout_margin_seconds=timeout_margin_seconds,
    )
    require_idle(metadata_after, label="after")

    return {
        "schema": "openmle_fast_resident_endpoint_probe_v1",
        "status": "pass",
        "manifest_sha256": manifest_sha256,
        "probe_indices": probe_indices,
        "metadata_before": metadata_before,
        "metadata_active": active_metadata,
        "metadata_after": metadata_after,
        "reset_count": 4,
        "slot_cleanup_count": 3,
        "idempotent_close_verified": True,
    }


def main() -> None:
    args = parse_args()
    manifest_bytes = args.manifest.read_bytes()
    document = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(document, dict):
        raise TypeError("manifest must contain a JSON object")
    endpoint = Endpoint(args.base_url, args.client_timeout_seconds)
    evidence = verify_resident_endpoint(
        endpoint,
        document,
        args.manifest_sha256,
        probe_indices=parse_probe_indices(args.indices),
        expected_outer_commit=args.expected_outer_commit,
        expected_inner_commit=args.expected_inner_commit,
        expected_prompt_sha256=args.expected_prompt_sha256,
        client_timeout_seconds=args.client_timeout_seconds,
        timeout_margin_seconds=args.timeout_margin_seconds,
        forbidden_canaries=load_forbidden_canaries(args.forbidden_canaries_file),
        manifest_bytes=manifest_bytes,
    )
    evidence["base_url"] = args.base_url
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "output": str(args.output)}))


if __name__ == "__main__":
    main()
