"""Immutable endpoint and reset-identity contract for CAMG held-out eval.

The training orchestrator intentionally requires optimizer-gate receipts.  A
native held-out run has a different authority boundary: frozen assets,
launchers, clean source commits, and a live route-specific reset identity.  This
module owns that boundary without importing Ray, torch, veRL, or any domain
environment implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .multitask_orchestrator import OrchestratorError
from .routes import RouteRegistry


HELDOUT_ENDPOINT_SCHEMA = "camg_heldout_endpoint_registry_v1"
RESET_IDENTITY_SCHEMA = "camg_heldout_reset_identity_v1"
CANONICAL_ROUTES = (
    "webshop",
    "swesmith",
    "literesearcher",
    "openmle_fast",
)
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_HEX = frozenset("0123456789abcdef")
_RESERVED_ENVIRONMENT_PREFIX = "CAMG_HELDOUT_"
_LOADER_OWNED_ENVIRONMENT_KEYS = frozenset({"SWESMITH_DETAIL_TOKEN"})
_REQUIRED_ASSETS = {
    "webshop": frozenset(
        {"heldout_episodes", "product_pool", "routing", "runtime_manifest"}
    ),
    "swesmith": frozenset(
        {
            "admitted_pool_manifest",
            "admission_certificate",
            "extension_pool_manifest",
            "formal_eval_selection",
            "heldout_manifest",
            "image_bindings",
            "image_manifest",
            "mirror_bundles_manifest",
            "routing",
            "runtime_manifest",
        }
    ),
    "literesearcher": frozenset(
        {
            "heldout_manifest",
            "loader_receipt",
            "retrieval_and_grader_manifest",
            "routing",
            "runtime_rows",
        }
    ),
    "openmle_fast": frozenset(
        {
            "heldout_manifest",
            "private_grader_bindings",
            "routing",
            "runtime_manifest",
        }
    ),
}
_RUNTIME_MANIFEST_ASSET = {
    "webshop": "runtime_manifest",
    "swesmith": "runtime_manifest",
    "literesearcher": "retrieval_and_grader_manifest",
    "openmle_fast": "runtime_manifest",
}
_RUNTIME_MANIFEST_SCHEMA = {
    "webshop": "camg_shop_complete_heldout_runtime_manifest_v2",
    "swesmith": "camg_swesmith_formal_eval_runtime_manifest_v5",
    "literesearcher": "camg_literesearcher_heldout_runtime_binding_v1",
    "openmle_fast": "camg_openmle_fast_heldout_runtime_manifest_v1",
}
_RUNTIME_TASK_COUNT_FIELD = {
    "webshop": "task_count",
    "swesmith": "task_count",
    "literesearcher": "test_items",
    "openmle_fast": "task_count",
}


@dataclass(frozen=True)
class HeldoutSource:
    name: str
    root: Path
    commit: str


@dataclass(frozen=True)
class HeldoutAsset:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class HeldoutIdentityProbe:
    create_path: str
    reset_path: str
    close_path: str
    detail_path: str | None = None
    detail_token_file: Path | None = None
    detail_token_sha256: str | None = None


@dataclass(frozen=True)
class HeldoutEndpointSpec:
    route_id: str
    task_count: int
    route_attestation_sha256: str
    endpoint: str
    sources: tuple[HeldoutSource, ...]
    assets: tuple[HeldoutAsset, ...]
    launcher_path: Path
    launcher_sha256: str
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    working_directory: Path
    readiness_url: str
    readiness_expected: Mapping[str, Any]
    readiness_sha256: str | None
    ready_timeout_seconds: float
    poll_seconds: float
    request_timeout_seconds: float
    identity_probe: HeldoutIdentityProbe
    cleanup_timeout_seconds: float


JsonRequester = Callable[..., Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, *, field: str) -> str:
    rendered = str(value or "").strip().lower()
    if len(rendered) != 64 or any(character not in _HEX for character in rendered):
        raise OrchestratorError(f"{field} must be a lowercase SHA-256 digest")
    return rendered


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OrchestratorError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OrchestratorError(f"{field} must be a sequence")
    return value


def _positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise OrchestratorError(f"{field} must be a positive number")
    return float(value)


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OrchestratorError(f"{field} must be a positive integer")
    return value


def _regular_file(value: Any, *, field: str, executable: bool = False) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise OrchestratorError(f"{field} must be an absolute regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise OrchestratorError(f"{field} is not executable: {path}")
    return path.resolve()


def _directory(value: Any, *, field: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise OrchestratorError(f"{field} must be an absolute directory: {path}")
    return path.resolve()


def _json_file(path: Path, *, field: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestratorError(f"{field} is not valid JSON: {path}") from exc
    return _mapping(payload, field=field)


def _parse_endpoint(value: Any, *, field: str) -> tuple[str, str, int]:
    rendered = str(value or "").rstrip("/")
    parsed = urlparse(rendered)
    try:
        port = parsed.port
    except ValueError as exc:
        raise OrchestratorError(f"{field} has an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in _LOOPBACK_HOSTS
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise OrchestratorError(f"{field} must be a same-Pod loopback HTTP endpoint")
    return rendered, str(parsed.hostname), int(port)


def _verify_source(value: Any, *, route_id: str) -> HeldoutSource:
    source = _mapping(value, field=f"{route_id} source")
    name = str(source.get("name", ""))
    if name not in {"outer", "inner"}:
        raise OrchestratorError(f"{route_id} source name is invalid: {name!r}")
    root = _directory(source.get("root"), field=f"{route_id} {name} source root")
    commit = str(source.get("commit", ""))
    if len(commit) != 40 or any(character not in _HEX for character in commit):
        raise OrchestratorError(f"{route_id} {name} source commit is invalid")
    try:
        observed = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise OrchestratorError(
            f"cannot verify {route_id} {name} source: {exc.output.strip()}"
        ) from exc
    if observed != commit:
        raise OrchestratorError(
            f"{route_id} {name} source commit mismatch: {observed} != {commit}"
        )
    if dirty:
        raise OrchestratorError(f"{route_id} {name} source tree is dirty: {dirty}")
    return HeldoutSource(name=name, root=root, commit=commit)


def _verify_asset(value: Any, *, route_id: str) -> HeldoutAsset:
    asset = _mapping(value, field=f"{route_id} asset")
    name = str(asset.get("name", "")).strip()
    if not name:
        raise OrchestratorError(f"{route_id} asset name is empty")
    path = _regular_file(asset.get("path"), field=f"{route_id} asset {name!r}")
    expected = _digest(asset.get("sha256"), field=f"{route_id} asset {name!r} sha256")
    observed = _sha256(path)
    if observed != expected:
        raise OrchestratorError(
            f"{route_id} asset {name!r} sha256 mismatch: {observed} != {expected}"
        )
    return HeldoutAsset(name=name, path=path, sha256=expected)


def _verify_runtime_manifest(
    route_id: str, assets: Mapping[str, HeldoutAsset]
) -> int:
    name = _RUNTIME_MANIFEST_ASSET[route_id]
    payload = _json_file(assets[name].path, field=f"{route_id} runtime manifest")
    if payload.get("schema") != _RUNTIME_MANIFEST_SCHEMA[route_id]:
        raise OrchestratorError(f"{route_id} held-out runtime manifest schema mismatch")
    if str(payload.get("status", "")).lower() not in {"ready", "pass"}:
        raise OrchestratorError(f"{route_id} held-out runtime manifest is not ready")
    if payload.get("heldout_evaluation_run") is not False:
        raise OrchestratorError(
            f"{route_id} runtime manifest must precede held-out evaluation"
        )
    task_count = _positive_int(
        payload.get(_RUNTIME_TASK_COUNT_FIELD[route_id]),
        field=f"{route_id} runtime task count",
    )
    if route_id == "literesearcher":
        heldout_pool = _mapping(
            payload.get("heldout_pool"), field="literesearcher heldout_pool"
        )
        rows = _mapping(heldout_pool.get("rows"), field="literesearcher rows")
        _digest(rows.get("sha256"), field="literesearcher runtime rows sha256")
    if route_id == "swesmith":
        if (
            payload.get("selection")
            != "deterministic complete-repository subset of the "
            "exact-runtime-admitted held-out candidate pool"
            or payload.get("active_training_inputs_modified") is not False
        ):
            raise OrchestratorError("swesmith formal Eval selection contract drifted")
        complete_count = _positive_int(
            payload.get("complete_admitted_pool_task_count"),
            field="swesmith complete admitted-pool task count",
        )
        extension_count = _positive_int(
            payload.get("extension_pool_task_count"),
            field="swesmith extension-pool task count",
        )
        if task_count + extension_count != complete_count:
            raise OrchestratorError(
                "swesmith formal and extension task counts do not partition admission"
            )
        files = _mapping(payload.get("files"), field="swesmith runtime files")
        bindings = {
            "routing": "routing",
            "manifest": "heldout_manifest",
            "formal_eval_selection": "formal_eval_selection",
            "admitted_pool_manifest": "admitted_pool_manifest",
            "extension_pool_manifest": "extension_pool_manifest",
            "image_bindings": "image_bindings",
            "image_manifest": "image_manifest",
        }
        for field, asset_name in bindings.items():
            binding = _mapping(files.get(field), field=f"swesmith runtime {field}")
            asset = assets[asset_name]
            bound_path = Path(str(binding.get("path", "")))
            if not bound_path.is_absolute():
                bound_path = (assets[name].path.parent / bound_path).resolve()
            else:
                bound_path = bound_path.resolve()
            if (
                bound_path != asset.path
                or _digest(binding.get("sha256"), field=f"swesmith {field} sha256")
                != asset.sha256
                or _positive_int(
                    binding.get("bytes"), field=f"swesmith {field} bytes"
                )
                != asset.path.stat().st_size
            ):
                raise OrchestratorError(
                    f"swesmith runtime manifest binds a different {field}"
                )

        selection = _json_file(
            assets["formal_eval_selection"].path,
            field="swesmith formal Eval selection",
        )
        admitted_pool = _json_file(
            assets["admitted_pool_manifest"].path,
            field="swesmith admitted-pool manifest",
        )
        extension_pool = _json_file(
            assets["extension_pool_manifest"].path,
            field="swesmith extension-pool manifest",
        )
        heldout_manifest = _json_file(
            assets["heldout_manifest"].path,
            field="swesmith formal Eval dataset manifest",
        )
        heldout_selection = _mapping(
            heldout_manifest.get("selection"),
            field="swesmith formal Eval dataset selection",
        )
        if (
            selection.get("schema") != "camg_swesmith_formal_eval_selection_v5"
            or selection.get("status") != "frozen"
            or selection.get("formal_eval_task_count") != task_count
            or selection.get("complete_admitted_heldout_pool_task_count")
            != complete_count
            or selection.get("extension_pool_task_count") != extension_count
            or selection.get("selection_depends_on_model_output_or_reward") is not False
            or selection.get("active_training_inputs_modified") is not False
            or selection.get("heldout_evaluation_run") is not False
            or admitted_pool.get("schema")
            != "camg_swesmith_admitted_heldout_pool_manifest_v5"
            or admitted_pool.get("status") != "complete"
            or admitted_pool.get("task_count") != complete_count
            or admitted_pool.get("formal_evaluation_role") is not False
            or admitted_pool.get("training_role") is not False
            or extension_pool.get("schema")
            != "camg_swesmith_extension_pool_manifest_v5"
            or extension_pool.get("status") != "frozen"
            or extension_pool.get("task_count") != extension_count
            or extension_pool.get("formal_evaluation_role") is not False
            or extension_pool.get("training_role") is not False
            or heldout_manifest.get("role") != "formal_heldout"
            or heldout_selection.get("count") != task_count
            or heldout_selection.get("source_admitted_pool_count") != complete_count
        ):
            raise OrchestratorError("swesmith formal Eval package contract drifted")
        routing_rows = assets["routing"].path.read_text(encoding="utf-8").splitlines()
        try:
            routing_indices = [json.loads(line)["data_idx"] for line in routing_rows]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise OrchestratorError("swesmith formal Eval routing is invalid") from exc
        if routing_indices != list(range(task_count)):
            raise OrchestratorError("swesmith formal Eval routing is not dense")
    if route_id == "openmle_fast":
        # The runtime manifest binds the public task manifest but does not
        # duplicate its role.  Read the bound public manifest itself rather
        # than inventing a top-level ``manifest_role`` field that the frozen
        # publication never emitted.
        binding = _mapping(
            payload.get("heldout_manifest"),
            field="openmle_fast heldout manifest binding",
        )
        heldout_asset = assets["heldout_manifest"]
        bound_path = Path(str(binding.get("path", "")))
        if not bound_path.is_absolute():
            bound_path = (assets[name].path.parent / bound_path).resolve()
        else:
            bound_path = bound_path.resolve()
        if bound_path != heldout_asset.path:
            raise OrchestratorError(
                "openmle_fast runtime manifest binds a different heldout manifest"
            )
        if _digest(
            binding.get("sha256"), field="openmle_fast heldout manifest sha256"
        ) != heldout_asset.sha256:
            raise OrchestratorError(
                "openmle_fast runtime manifest heldout digest mismatch"
            )
        if _positive_int(
            binding.get("bytes"), field="openmle_fast heldout manifest bytes"
        ) != heldout_asset.path.stat().st_size:
            raise OrchestratorError(
                "openmle_fast runtime manifest heldout byte count mismatch"
            )
        heldout_manifest = _json_file(
            heldout_asset.path, field="openmle_fast heldout manifest"
        )
        if heldout_manifest.get("schema") != "openmle_fast_public_manifest_v1":
            raise OrchestratorError("openmle_fast heldout manifest schema mismatch")
        if heldout_manifest.get("role") != "heldout":
            raise OrchestratorError("openmle_fast heldout manifest role must be heldout")
        if _positive_int(
            heldout_manifest.get("task_count"),
            field="openmle_fast heldout manifest task count",
        ) != task_count:
            raise OrchestratorError(
                "openmle_fast runtime and heldout manifest task counts differ"
            )
    return task_count


def _verified_launcher_environment(
    *,
    route_id: str,
    task_count: int,
    endpoint: str,
    route_attestation_sha256: str,
    sources: Sequence[HeldoutSource],
    assets: Sequence[HeldoutAsset],
    identity_probe: HeldoutIdentityProbe,
    registry_environment: Mapping[str, Any],
) -> dict[str, str]:
    """Build the only environment namespace allowed to carry verified inputs.

    Registry-authored runtime knobs may use their native environment names, but
    they cannot impersonate source or asset bindings.  Every ``CAMG_HELDOUT_*``
    value is derived after the corresponding path, commit, and digest has been
    verified by this loader.
    """

    environment = {str(key): str(value) for key, value in registry_environment.items()}
    if any(key.startswith(_RESERVED_ENVIRONMENT_PREFIX) for key in environment):
        raise OrchestratorError(
            f"{route_id} endpoint environment uses reserved CAMG_HELDOUT_ names"
        )
    loader_owned = sorted(_LOADER_OWNED_ENVIRONMENT_KEYS.intersection(environment))
    if loader_owned:
        raise OrchestratorError(
            f"{route_id} endpoint environment overrides loader-owned environment: "
            f"{loader_owned!r}"
        )
    if any(
        not key or "=" in key or "\0" in key or "\0" in value
        for key, value in environment.items()
    ):
        raise OrchestratorError(f"{route_id} endpoint environment is unsafe")

    verified = {
        "CAMG_HELDOUT_ROUTE_ID": route_id,
        "CAMG_HELDOUT_ROLE": "heldout",
        "CAMG_HELDOUT_TASK_COUNT": str(task_count),
        "CAMG_HELDOUT_ENDPOINT": endpoint,
        "CAMG_HELDOUT_ROUTE_ATTESTATION_SHA256": route_attestation_sha256,
    }
    for source in sources:
        prefix = f"CAMG_HELDOUT_SOURCE_{source.name.upper()}"
        verified[f"{prefix}_ROOT"] = str(source.root)
        verified[f"{prefix}_COMMIT"] = source.commit
    for asset in assets:
        prefix = f"CAMG_HELDOUT_ASSET_{asset.name.upper()}"
        verified[f"{prefix}_PATH"] = str(asset.path)
        verified[f"{prefix}_SHA256"] = asset.sha256
    if route_id == "swesmith":
        token_file = identity_probe.detail_token_file
        if token_file is None:
            raise OrchestratorError("swesmith verified detail token is missing")
        verified["SWESMITH_DETAIL_TOKEN"] = token_file.read_text(
            encoding="utf-8"
        ).strip()
    environment.update(verified)
    return environment


def _safe_path(value: Any, *, field: str) -> str:
    rendered = str(value or "")
    if not rendered.startswith("/") or any(c in rendered for c in ("\0", "\n", "\r", "?", "#")):
        raise OrchestratorError(f"{field} must be an absolute HTTP path")
    return rendered


def _load_identity_probe(route_id: str, value: Any) -> HeldoutIdentityProbe:
    probe = _mapping(value, field=f"{route_id} identity probe")
    if probe.get("schema") != RESET_IDENTITY_SCHEMA:
        raise OrchestratorError(f"{route_id} identity probe schema mismatch")
    create_path = _safe_path(probe.get("create_path"), field=f"{route_id} create_path")
    reset_path = _safe_path(probe.get("reset_path"), field=f"{route_id} reset_path")
    close_path = _safe_path(probe.get("close_path"), field=f"{route_id} close_path")
    if len({create_path, reset_path, close_path}) != 3:
        raise OrchestratorError(f"{route_id} lifecycle paths must be distinct")
    detail_path: str | None = None
    token_file: Path | None = None
    token_sha: str | None = None
    if route_id == "swesmith":
        detail_path = _safe_path(
            probe.get("detail_path"), field="swesmith detail_path"
        )
        if detail_path in {create_path, reset_path, close_path}:
            raise OrchestratorError("swesmith detail_path must be distinct")
        token_file = _regular_file(
            probe.get("detail_token_file"), field="swesmith detail token"
        )
        if stat.S_IMODE(token_file.stat().st_mode) != 0o600:
            raise OrchestratorError("swesmith detail token must have mode 0600")
        token_sha = _digest(
            probe.get("detail_token_sha256"), field="swesmith detail token sha256"
        )
        if _sha256(token_file) != token_sha:
            raise OrchestratorError("swesmith detail token sha256 mismatch")
        if not token_file.read_text(encoding="utf-8").strip():
            raise OrchestratorError("swesmith detail token is empty")
    elif any(
        probe.get(field) is not None
        for field in ("detail_path", "detail_token_file", "detail_token_sha256")
    ):
        raise OrchestratorError(f"{route_id} must not bind a private detail token")
    return HeldoutIdentityProbe(
        create_path=create_path,
        reset_path=reset_path,
        close_path=close_path,
        detail_path=detail_path,
        detail_token_file=token_file,
        detail_token_sha256=token_sha,
    )


def load_heldout_endpoint_registry(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    route_registry: RouteRegistry,
) -> tuple[tuple[HeldoutEndpointSpec, ...], dict[str, Any]]:
    """Validate held-out endpoint authority before any process is spawned."""

    registry_path = _regular_file(path, field="held-out endpoint registry")
    expected = _digest(expected_sha256, field="held-out endpoint registry sha256")
    observed = _sha256(registry_path)
    if observed != expected:
        raise OrchestratorError(
            f"held-out endpoint registry sha256 mismatch: {observed} != {expected}"
        )
    payload = _json_file(registry_path, field="held-out endpoint registry")
    if payload.get("schema") != HELDOUT_ENDPOINT_SCHEMA or payload.get("status") != "pass":
        raise OrchestratorError("held-out endpoint registry is not a completed v1 registry")
    route_order = tuple(
        str(value)
        for value in _sequence(
            payload.get("route_order"), field="held-out endpoint route_order"
        )
    )
    if route_order != CANONICAL_ROUTES or route_order != route_registry.route_ids:
        raise OrchestratorError(f"held-out endpoint route order mismatch: {route_order!r}")
    routes = _sequence(payload.get("routes"), field="held-out endpoint routes")
    if len(routes) != len(CANONICAL_ROUTES):
        raise OrchestratorError("held-out endpoint registry must contain four routes")

    specs: list[HeldoutEndpointSpec] = []
    source_report: dict[str, Any] = {}
    asset_report: dict[str, Any] = {}
    launcher_report: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    task_counts: dict[str, int] = {}
    listeners: set[tuple[str, int]] = set()
    for position, (expected_route, raw_route) in enumerate(zip(CANONICAL_ROUTES, routes)):
        route = _mapping(raw_route, field=f"held-out endpoint route {position}")
        route_id = str(route.get("route_id", ""))
        if route_id != expected_route:
            raise OrchestratorError(
                f"held-out endpoint route order mismatch at {position}: {route_id!r}"
            )
        registered = route_registry.resolve(route_id)
        attestation = _digest(
            route.get("route_attestation_sha256"),
            field=f"{route_id} route attestation sha256",
        )
        if attestation != registered.route_attestation_sha256:
            raise OrchestratorError(f"{route_id} route attestation mismatch")
        endpoint, host, port = _parse_endpoint(route.get("endpoint"), field=f"{route_id} endpoint")
        if endpoint != str(registered.client_config["env_addr"]).rstrip("/"):
            raise OrchestratorError(f"{route_id} endpoint differs from route registry")
        if (host, port) in listeners:
            raise OrchestratorError(f"duplicate held-out endpoint listener {host}:{port}")
        listeners.add((host, port))

        sources = tuple(
            _verify_source(value, route_id=route_id)
            for value in _sequence(route.get("sources"), field=f"{route_id} sources")
        )
        if tuple(source.name for source in sources) != ("outer", "inner"):
            raise OrchestratorError(f"{route_id} must bind outer then inner source")

        assets = tuple(
            _verify_asset(value, route_id=route_id)
            for value in _sequence(route.get("assets"), field=f"{route_id} assets")
        )
        asset_by_name = {asset.name: asset for asset in assets}
        if len(asset_by_name) != len(assets):
            raise OrchestratorError(f"{route_id} has duplicate asset names")
        if frozenset(asset_by_name) != _REQUIRED_ASSETS[route_id]:
            raise OrchestratorError(
                f"{route_id} held-out assets mismatch: {sorted(asset_by_name)!r}"
            )
        task_count = _verify_runtime_manifest(route_id, asset_by_name)

        if route_id == "openmle_fast":
            expected_role = registered.client_config.get("expected_role")
            client_identity = _mapping(
                route.get("client_identity", {}), field="openmle_fast client_identity"
            )
            bound_role = client_identity.get("expected_role", expected_role)
            if expected_role != "heldout" or bound_role != "heldout":
                raise OrchestratorError("openmle_fast client expected_role must be heldout")

        identity_probe = _load_identity_probe(route_id, route.get("identity_probe"))
        launcher = _mapping(
            route.get("endpoint_launcher"), field=f"{route_id} endpoint launcher"
        )
        launcher_path = _regular_file(
            launcher.get("path"), field=f"{route_id} endpoint launcher", executable=True
        )
        launcher_sha = _digest(
            launcher.get("sha256"), field=f"{route_id} endpoint launcher sha256"
        )
        if _sha256(launcher_path) != launcher_sha:
            raise OrchestratorError(f"{route_id} endpoint launcher sha256 mismatch")
        if launcher.get("process_contract") != "foreground_supervisor_v1":
            raise OrchestratorError(
                f"{route_id} endpoint launcher must own a foreground supervisor"
            )
        argv = tuple(
            str(value)
            for value in _sequence(launcher.get("argv", ()), field=f"{route_id} argv")
        )
        if any(any(c in value for c in ("\0", "\n", "\r")) for value in argv):
            raise OrchestratorError(f"{route_id} endpoint argv contains unsafe text")
        raw_environment = _mapping(
            launcher.get("environment", {}), field=f"{route_id} environment"
        )
        environment = _verified_launcher_environment(
            route_id=route_id,
            task_count=task_count,
            endpoint=endpoint,
            route_attestation_sha256=attestation,
            sources=sources,
            assets=assets,
            identity_probe=identity_probe,
            registry_environment=raw_environment,
        )
        working_directory = _directory(
            launcher.get("working_directory"), field=f"{route_id} working directory"
        )

        readiness = _mapping(route.get("readiness"), field=f"{route_id} readiness")
        readiness_url, _, _ = _parse_endpoint(
            readiness.get("url"), field=f"{route_id} readiness URL"
        )
        if not readiness_url.startswith(endpoint + "/"):
            raise OrchestratorError(f"{route_id} readiness URL is outside endpoint")
        readiness_expected = _mapping(
            readiness.get("expected"), field=f"{route_id} readiness expected"
        )
        if not readiness_expected:
            raise OrchestratorError(f"{route_id} readiness expected is empty")
        readiness_sha_raw = readiness.get("response_sha256")
        readiness_sha = (
            _digest(readiness_sha_raw, field=f"{route_id} readiness response sha256")
            if readiness_sha_raw is not None
            else None
        )
        spec = HeldoutEndpointSpec(
            route_id=route_id,
            task_count=task_count,
            route_attestation_sha256=attestation,
            endpoint=endpoint,
            sources=sources,
            assets=assets,
            launcher_path=launcher_path,
            launcher_sha256=launcher_sha,
            argv=argv,
            environment=environment,
            working_directory=working_directory,
            readiness_url=readiness_url,
            readiness_expected=dict(readiness_expected),
            readiness_sha256=readiness_sha,
            ready_timeout_seconds=_positive_float(
                readiness.get("timeout_seconds"), field=f"{route_id} readiness timeout"
            ),
            poll_seconds=_positive_float(
                readiness.get("poll_seconds"), field=f"{route_id} readiness poll"
            ),
            request_timeout_seconds=_positive_float(
                readiness.get("request_timeout_seconds"), field=f"{route_id} request timeout"
            ),
            identity_probe=identity_probe,
            cleanup_timeout_seconds=_positive_float(
                route.get("cleanup_timeout_seconds"), field=f"{route_id} cleanup timeout"
            ),
        )
        specs.append(spec)
        source_report[route_id] = [
            {"name": source.name, "root": str(source.root), "commit": source.commit}
            for source in sources
        ]
        asset_report[route_id] = [
            {"name": asset.name, "path": str(asset.path), "sha256": asset.sha256}
            for asset in assets
        ]
        launcher_report[route_id] = {
            "path": str(launcher_path),
            "sha256": launcher_sha,
        }
        identities[route_id] = {
            "schema": RESET_IDENTITY_SCHEMA,
            "create_path": identity_probe.create_path,
            "reset_path": identity_probe.reset_path,
            "close_path": identity_probe.close_path,
            "private_detail": route_id == "swesmith",
        }
        task_counts[route_id] = task_count

    return tuple(specs), {
        "schema": HELDOUT_ENDPOINT_SCHEMA,
        "status": "pass",
        "path": str(registry_path),
        "sha256": observed,
        "route_order": list(route_order),
        "task_counts": task_counts,
        "sources": source_report,
        "assets": asset_report,
        "launchers": launcher_report,
        "identity_probes": identities,
    }


def _request_json(
    method: str,
    url: str,
    *,
    body: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> Any:
    method = method.upper()
    request_url = url
    data: bytes | None = None
    request_headers = dict(headers)
    if method == "GET":
        if body:
            request_url += ("&" if "?" in request_url else "?") + urllib.parse.urlencode(body)
    else:
        data = (json.dumps(dict(body), separators=(",", ":")) + "\n").encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        request_url, data=data, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except Exception as exc:
        raise OrchestratorError(f"held-out endpoint request failed: {method} {url}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OrchestratorError(f"held-out endpoint returned invalid JSON: {url}") from exc


def _required_extra(row: Mapping[str, Any], *, route_id: str) -> Mapping[str, Any]:
    if row.get("route_id") != route_id:
        raise OrchestratorError(f"{route_id} probe row route identity mismatch")
    extra = _mapping(row.get("extra_info"), field=f"{route_id} probe extra_info")
    return _mapping(
        extra.get("source_extra_info"),
        field=f"{route_id} probe extra_info.source_extra_info",
    )


def _assert_identity(actual: Mapping[str, Any], expected: Mapping[str, Any], *, route_id: str) -> None:
    drift = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if drift:
        raise OrchestratorError(f"{route_id} held-out reset identity mismatch: {drift!r}")


def probe_heldout_reset_identity(
    spec: HeldoutEndpointSpec,
    row: Mapping[str, Any],
    *,
    request_json: JsonRequester = _request_json,
) -> dict[str, Any]:
    """Create one slot, verify its frozen held-out identity, and always close it."""

    extra = _required_extra(row, route_id=spec.route_id)
    data_idx = row.get("data_idx")
    if isinstance(data_idx, bool) or not isinstance(data_idx, int) or data_idx < 0:
        raise OrchestratorError(f"{spec.route_id} probe data_idx is invalid")
    probe = spec.identity_probe
    created = request_json(
        "POST",
        spec.endpoint + probe.create_path,
        body={},
        headers={},
        timeout_seconds=spec.request_timeout_seconds,
    )
    if not isinstance(created, Mapping) or isinstance(created.get("id"), bool) or not isinstance(created.get("id"), int):
        raise OrchestratorError(f"{spec.route_id} create response has no integer id")
    slot = int(created["id"])
    actual: dict[str, Any] = {}
    close_result: Any = None
    try:
        reset_body = {"id": slot, "data_idx": data_idx}
        reset = request_json(
            "POST",
            spec.endpoint + probe.reset_path,
            body=reset_body,
            headers={},
            timeout_seconds=spec.request_timeout_seconds,
        )
        reset_payload = _mapping(reset, field=f"{spec.route_id} reset response")
        info = _mapping(reset_payload.get("info"), field=f"{spec.route_id} reset info")
        if spec.route_id == "webshop":
            actual = {
                "data_idx": info.get("data_idx"),
                "scenario_id": info.get("scenario_id"),
                "orbit_index": info.get("orbit_index"),
            }
            expected = {
                "data_idx": data_idx,
                "scenario_id": extra.get("scenario_id"),
                "orbit_index": extra.get("orbit_index"),
            }
        elif spec.route_id == "swesmith":
            assert probe.detail_path is not None
            assert probe.detail_token_file is not None
            token = probe.detail_token_file.read_text(encoding="utf-8").strip()
            detail = request_json(
                "GET",
                spec.endpoint + probe.detail_path,
                body={"id": slot},
                headers={"X-SWESMITH-Detail-Token": token},
                timeout_seconds=spec.request_timeout_seconds,
            )
            detail_payload = _mapping(detail, field="swesmith private detail")
            actual = {
                "data_idx": detail_payload.get("data_idx"),
                "instance_id": detail_payload.get("instance_id"),
                "base_repository": detail_payload.get("base_repository"),
            }
            expected = {
                "data_idx": data_idx,
                "instance_id": extra.get("instance_id"),
                "base_repository": extra.get("base_repository"),
            }
        elif spec.route_id == "literesearcher":
            actual = {
                "data_idx": info.get("data_idx"),
                "row_identity": info.get("row_identity"),
                "source_pool_index": info.get("source_pool_index"),
            }
            expected = {
                "data_idx": data_idx,
                "row_identity": extra.get("row_identity"),
                "source_pool_index": extra.get("source_pool_index"),
            }
        elif spec.route_id == "openmle_fast":
            actual = {
                "data_idx": info.get("data_idx"),
                "task_id": info.get("task_id"),
                "source_family": info.get("source_family"),
                "manifest_role": info.get("manifest_role"),
                "manifest_sha256": info.get("manifest_sha256"),
            }
            expected = {
                "data_idx": data_idx,
                "task_id": extra.get("task_id"),
                "source_family": extra.get("source_family"),
                "manifest_role": extra.get("role"),
                "manifest_sha256": extra.get("manifest_sha256"),
            }
        else:  # pragma: no cover - the dataclass is created by the strict loader.
            raise OrchestratorError(f"unsupported held-out route: {spec.route_id}")
        _assert_identity(actual, expected, route_id=spec.route_id)
    finally:
        close_body = {"id": slot}
        close_result = request_json(
            "POST",
            spec.endpoint + probe.close_path,
            body=close_body,
            headers={},
            timeout_seconds=spec.cleanup_timeout_seconds,
        )

    return {
        "schema": RESET_IDENTITY_SCHEMA,
        "status": "pass",
        "route_id": spec.route_id,
        "data_idx": data_idx,
        "slot_id": slot,
        "verified_identity": actual,
        "closed": close_result is not None,
    }


__all__ = [
    "CANONICAL_ROUTES",
    "HELDOUT_ENDPOINT_SCHEMA",
    "HeldoutAsset",
    "HeldoutEndpointSpec",
    "HeldoutIdentityProbe",
    "HeldoutSource",
    "load_heldout_endpoint_registry",
    "probe_heldout_reset_identity",
]
