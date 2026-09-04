from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from agentmemorygym_verl.multitask_orchestrator import (
    OrchestratorError,
    load_endpoint_registry,
)
from agentmemorygym_verl.routes import load_route_registry


ROUTES = ("webshop", "swesmith", "literesearcher", "openmle_fast")
PORTS = {
    "webshop": 65121,
    "swesmith": 65124,
    "literesearcher": 65122,
    "openmle_fast": 65123,
}
TASK_COUNTS = {
    "webshop": 1746,
    "swesmith": 933,
    "literesearcher": 5319,
    "openmle_fast": 169,
}
ASSET_NAMES = {
    "webshop": {
        "heldout_episodes",
        "product_pool",
        "routing",
        "runtime_manifest",
    },
    "swesmith": {
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
    },
    "literesearcher": {
        "heldout_manifest",
        "loader_receipt",
        "retrieval_and_grader_manifest",
        "routing",
        "runtime_rows",
    },
    "openmle_fast": {
        "heldout_manifest",
        "private_grader_bindings",
        "routing",
        "runtime_manifest",
    },
}

LAUNCHER_ROOT = (
    Path(__file__).resolve().parents[1] / "scripts" / "heldout_endpoints"
)
if str(LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(LAUNCHER_ROOT))
openmle_supervisor = importlib.import_module("openmle_heldout_supervisor")


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value) -> str:
    return _write(path, _json_bytes(value))


def _git_source(path: Path, marker: str) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Heldout Test"],
        check=True,
    )
    (path / "SOURCE").write_text(marker + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "SOURCE"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "fixture"], check=True
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _route_registry(root: Path) -> tuple[Path, str]:
    routes = []
    for index, route_id in enumerate(ROUTES, start=1):
        client = {
            "task_name": "agentmemory" if route_id == "webshop" else route_id,
            "env_addr": f"http://127.0.0.1:{PORTS[route_id]}",
            "timeout": 30,
            "max_retries": 1,
        }
        if route_id == "agentmemory":
            client["policy_system_prompt"] = "test"
        if route_id == "openmle_fast":
            client.update(
                {
                    "expected_manifest_sha256": "a" * 64,
                    "expected_release_revision": "b" * 40,
                    "expected_outer_commit": "c" * 40,
                    "expected_inner_commit": "d" * 40,
                    "expected_role": "heldout",
                    "expected_executor_runtime_digest": "sha256:" + "e" * 64,
                    "expected_materializer_sha256": "f" * 64,
                    "expected_actions_sha256": "1" * 64,
                    "expected_max_observation_tokens": 8192,
                }
            )
        routes.append(
            {
                "route_id": route_id,
                "max_rounds": 40 if route_id == "literesearcher" else 30,
                "max_observation_tokens": (
                    12288 if route_id == "literesearcher" else 8192
                ),
                "policy_framing_sha256": f"{index:x}" * 64,
                "route_attestation_sha256": f"{index + 4:x}" * 64,
                "client": client,
            }
        )
    path = root / "route-registry.json"
    digest = _write_json(
        path,
        {
            "schema": "amg_route_registry_v1",
            "agent_name": "amg_task_neutral_async",
            "routes": routes,
        },
    )
    return path, digest


def _runtime_manifest(route_id: str) -> dict:
    schemas = {
        "webshop": "camg_shop_complete_heldout_runtime_manifest_v2",
        "swesmith": "camg_swesmith_formal_eval_runtime_manifest_v5",
        "literesearcher": "camg_literesearcher_heldout_runtime_binding_v1",
        "openmle_fast": "camg_openmle_fast_heldout_runtime_manifest_v1",
    }
    payload = {
        "schema": schemas[route_id],
        "status": "ready",
        "heldout_evaluation_run": False,
    }
    if route_id == "literesearcher":
        payload["heldout_pool"] = {"rows": {"sha256": "a" * 64}}
        payload["test_items"] = TASK_COUNTS[route_id]
    else:
        payload["task_count"] = TASK_COUNTS[route_id]
    if route_id == "swesmith":
        payload.update(
            {
                "selection": (
                    "deterministic complete-repository subset of the "
                    "exact-runtime-admitted held-out candidate pool"
                ),
                "active_training_inputs_modified": False,
                "complete_admitted_pool_task_count": 7450,
                "extension_pool_task_count": 6517,
            }
        )
    return payload


def _fixture(root: Path):
    from agentmemorygym_verl.heldout_endpoints import HELDOUT_ENDPOINT_SCHEMA

    route_path, route_sha = _route_registry(root)
    route_registry = load_route_registry(
        route_path, expected_sha256=route_sha, expected_route_ids=ROUTES
    )
    sources = {}
    for name in ("outer", "inner"):
        source_root = root / f"source-{name}"
        sources[name] = {
            "name": name,
            "root": str(source_root),
            "commit": _git_source(source_root, name),
        }
    literesearcher_endpoint_root = root / "source-literesearcher-endpoint"
    literesearcher_endpoint_source = {
        "name": "endpoint",
        "root": str(literesearcher_endpoint_root),
        "commit": _git_source(literesearcher_endpoint_root, "literesearcher-endpoint"),
    }
    token = root / "swesmith-detail.token"
    token.write_text("private-token\n", encoding="utf-8")
    token.chmod(0o600)

    registry_routes = []
    for route in route_registry.routes:
        route_id = route.route_id
        launcher = root / "launchers" / f"{route_id}.sh"
        launcher_sha = _write(launcher, b"#!/bin/sh\nexec sleep 1\n")
        launcher.chmod(0o755)
        assets = []
        for name in sorted(ASSET_NAMES[route_id]):
            path = root / "assets" / route_id / f"{name}.json"
            if route_id == "swesmith" and name == "routing":
                digest = _write(
                    path,
                    b"".join(
                        (
                            json.dumps(
                                {
                                    "item_id": f"swesmith_{index}",
                                    "data_idx": index,
                                    "extra_info": {"index": index},
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")
                        for index in range(TASK_COUNTS[route_id])
                    ),
                )
                assets.append(
                    {"name": name, "path": str(path), "sha256": digest}
                )
                continue
            if route_id == "swesmith" and name == "formal_eval_selection":
                payload = {
                    "schema": "camg_swesmith_formal_eval_selection_v5",
                    "status": "frozen",
                    "formal_eval_task_count": 933,
                    "complete_admitted_heldout_pool_task_count": 7450,
                    "extension_pool_task_count": 6517,
                    "selection_depends_on_model_output_or_reward": False,
                    "active_training_inputs_modified": False,
                    "heldout_evaluation_run": False,
                }
            elif route_id == "swesmith" and name == "admitted_pool_manifest":
                payload = {
                    "schema": "camg_swesmith_admitted_heldout_pool_manifest_v5",
                    "status": "complete",
                    "task_count": 7450,
                    "formal_evaluation_role": False,
                    "training_role": False,
                }
            elif route_id == "swesmith" and name == "extension_pool_manifest":
                payload = {
                    "schema": "camg_swesmith_extension_pool_manifest_v5",
                    "status": "frozen",
                    "task_count": 6517,
                    "formal_evaluation_role": False,
                    "training_role": False,
                }
            elif route_id == "swesmith" and name == "heldout_manifest":
                payload = {
                    "role": "formal_heldout",
                    "selection": {
                        "count": 933,
                        "source_admitted_pool_count": 7450,
                    },
                }
            elif route_id == "openmle_fast" and name == "heldout_manifest":
                payload = {
                    "schema": "openmle_fast_public_manifest_v1",
                    "role": "heldout",
                    "task_count": TASK_COUNTS[route_id],
                }
            elif name in {"runtime_manifest", "retrieval_and_grader_manifest"}:
                payload = _runtime_manifest(route_id)
                if route_id == "openmle_fast":
                    heldout = next(
                        item for item in assets if item["name"] == "heldout_manifest"
                    )
                    heldout_path = Path(heldout["path"])
                    payload["heldout_manifest"] = {
                        "path": heldout_path.name,
                        "bytes": heldout_path.stat().st_size,
                        "sha256": heldout["sha256"],
                    }
                if route_id == "swesmith":
                    asset_by_name = {item["name"]: item for item in assets}
                    payload["files"] = {}
                    for field, asset_name in {
                        "routing": "routing",
                        "manifest": "heldout_manifest",
                        "formal_eval_selection": "formal_eval_selection",
                        "admitted_pool_manifest": "admitted_pool_manifest",
                        "extension_pool_manifest": "extension_pool_manifest",
                        "image_bindings": "image_bindings",
                        "image_manifest": "image_manifest",
                    }.items():
                        bound_asset = asset_by_name[asset_name]
                        bound_path = Path(bound_asset["path"])
                        payload["files"][field] = {
                            "path": bound_path.name,
                            "bytes": bound_path.stat().st_size,
                            "sha256": bound_asset["sha256"],
                        }
            else:
                payload = {"route_id": route_id, "name": name}
            assets.append(
                {"name": name, "path": str(path), "sha256": _write_json(path, payload)}
            )
        identity_probe = {
            "schema": "camg_heldout_reset_identity_v1",
            "create_path": "/create",
            "reset_path": "/reset",
            "close_path": "/close",
        }
        if route_id == "swesmith":
            identity_probe.update(
                {
                    "detail_path": "/detail",
                    "detail_token_file": str(token),
                    "detail_token_sha256": hashlib.sha256(
                        token.read_bytes()
                    ).hexdigest(),
                }
            )
        route_sources = copy.deepcopy(list(sources.values()))
        if route_id == "literesearcher":
            route_sources.append(copy.deepcopy(literesearcher_endpoint_source))
        registry_routes.append(
            {
                "route_id": route_id,
                "route_attestation_sha256": route.route_attestation_sha256,
                "endpoint": route.client_config["env_addr"],
                "sources": route_sources,
                "assets": assets,
                "endpoint_launcher": {
                    "path": str(launcher),
                    "sha256": launcher_sha,
                    "argv": [],
                    "environment": {},
                    "working_directory": str(root),
                    "process_contract": "foreground_supervisor_v1",
                },
                "readiness": {
                    "url": f"{route.client_config['env_addr']}/metadata",
                    "expected": {"status": "ready", "route_id": route_id},
                    "timeout_seconds": 30,
                    "poll_seconds": 0.1,
                    "request_timeout_seconds": 2,
                },
                "identity_probe": identity_probe,
                "cleanup_timeout_seconds": 30,
            }
        )
    registry = {
        "schema": HELDOUT_ENDPOINT_SCHEMA,
        "status": "pass",
        "route_order": list(ROUTES),
        "routes": registry_routes,
    }
    registry_path = root / "heldout-endpoints.json"
    registry_sha = _write_json(registry_path, registry)
    return route_registry, registry_path, registry_sha, registry


class HeldoutEndpointRegistryTests(unittest.TestCase):
    def test_training_loader_rejects_heldout_registry_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_registry, path, digest, _ = _fixture(root)
            with self.assertRaisesRegex(
                OrchestratorError, "completed v1 registry"
            ):
                load_endpoint_registry(
                    path,
                    expected_sha256=digest,
                    route_registry=route_registry,
                )

    def test_loads_without_training_gate_receipts_and_verifies_all_bindings(self):
        from agentmemorygym_verl.heldout_endpoints import (
            load_heldout_endpoint_registry,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_registry, path, digest, _ = _fixture(root)
            specs, report = load_heldout_endpoint_registry(
                path,
                expected_sha256=digest,
                route_registry=route_registry,
            )
            self.assertEqual(tuple(spec.route_id for spec in specs), ROUTES)
            self.assertEqual(report["schema"], "camg_heldout_endpoint_registry_v1")
            self.assertNotIn("gate_receipts", report)
            for spec in specs:
                self.assertEqual(spec.task_count, TASK_COUNTS[spec.route_id])
                self.assertEqual(
                    set(asset.name for asset in spec.assets), ASSET_NAMES[spec.route_id]
                )
                expected_source_names = (
                    ("outer", "inner", "endpoint")
                    if spec.route_id == "literesearcher"
                    else ("outer", "inner")
                )
                self.assertEqual(
                    tuple(source.name for source in spec.sources), expected_source_names
                )
                self.assertEqual(
                    spec.environment["CAMG_HELDOUT_ROUTE_ID"], spec.route_id
                )
                self.assertEqual(
                    spec.environment["CAMG_HELDOUT_TASK_COUNT"],
                    str(TASK_COUNTS[spec.route_id]),
                )
                self.assertEqual(
                    spec.environment["CAMG_HELDOUT_ROUTE_ATTESTATION_SHA256"],
                    spec.route_attestation_sha256,
                )
                self.assertEqual(spec.environment["CAMG_HELDOUT_ROLE"], "heldout")
                for source in spec.sources:
                    prefix = f"CAMG_HELDOUT_SOURCE_{source.name.upper()}"
                    self.assertEqual(spec.environment[f"{prefix}_ROOT"], str(source.root))
                    self.assertEqual(spec.environment[f"{prefix}_COMMIT"], source.commit)
                for asset in spec.assets:
                    prefix = f"CAMG_HELDOUT_ASSET_{asset.name.upper()}"
                    self.assertEqual(spec.environment[f"{prefix}_PATH"], str(asset.path))
                    self.assertEqual(spec.environment[f"{prefix}_SHA256"], asset.sha256)
                if spec.route_id == "swesmith":
                    self.assertEqual(
                        spec.environment["SWESMITH_DETAIL_TOKEN"],
                        "private-token",
                    )
                else:
                    self.assertNotIn("SWESMITH_DETAIL_TOKEN", spec.environment)
            self.assertEqual(report["task_counts"], TASK_COUNTS)

    def test_literesearcher_requires_independent_endpoint_source(self):
        from agentmemorygym_verl.heldout_endpoints import (
            load_heldout_endpoint_registry,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_registry, path, _digest, registry = _fixture(root)
            literesearcher = next(
                route
                for route in registry["routes"]
                if route["route_id"] == "literesearcher"
            )
            literesearcher["sources"] = literesearcher["sources"][:2]
            digest = _write_json(path, registry)
            with self.assertRaisesRegex(
                OrchestratorError,
                "literesearcher must bind outer, inner, then endpoint source",
            ):
                load_heldout_endpoint_registry(
                    path,
                    expected_sha256=digest,
                    route_registry=route_registry,
                )

    def test_registry_cannot_override_loader_owned_heldout_environment(self):
        from agentmemorygym_verl.heldout_endpoints import (
            load_heldout_endpoint_registry,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_registry, path, _digest, registry = _fixture(root)
            registry["routes"][0]["endpoint_launcher"]["environment"] = {
                "CAMG_HELDOUT_ASSET_RUNTIME_MANIFEST_PATH": "/tmp/unverified.json"
            }
            digest = _write_json(path, registry)
            with self.assertRaisesRegex(OrchestratorError, "reserved CAMG_HELDOUT_"):
                load_heldout_endpoint_registry(
                    path,
                    expected_sha256=digest,
                    route_registry=route_registry,
                )

    def test_registry_cannot_override_loader_owned_swesmith_detail_token(self):
        from agentmemorygym_verl.heldout_endpoints import (
            load_heldout_endpoint_registry,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route_registry, path, _digest, registry = _fixture(root)
            swesmith = next(
                route for route in registry["routes"] if route["route_id"] == "swesmith"
            )
            swesmith["endpoint_launcher"]["environment"] = {
                "SWESMITH_DETAIL_TOKEN": "registry-controlled-secret"
            }
            digest = _write_json(path, registry)
            with self.assertRaisesRegex(OrchestratorError, "loader-owned environment"):
                load_heldout_endpoint_registry(
                    path,
                    expected_sha256=digest,
                    route_registry=route_registry,
                )

    def test_rejects_source_launcher_asset_and_role_drift(self):
        from agentmemorygym_verl.heldout_endpoints import (
            load_heldout_endpoint_registry,
        )

        mutations = ("dirty-source", "launcher-mode", "missing-asset", "train-role")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                route_registry, path, _digest, registry = _fixture(root)
                if mutation == "dirty-source":
                    source = Path(registry["routes"][0]["sources"][0]["root"])
                    (source / "UNTRACKED").write_text("dirty\n", encoding="utf-8")
                elif mutation == "launcher-mode":
                    Path(registry["routes"][0]["endpoint_launcher"]["path"]).chmod(0o644)
                elif mutation == "missing-asset":
                    registry["routes"][2]["assets"] = registry["routes"][2]["assets"][:-1]
                    _write_json(path, registry)
                else:
                    registry["routes"][3]["client_identity"] = {
                        "expected_role": "train_pool"
                    }
                    _write_json(path, registry)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises((OrchestratorError, ValueError)):
                    load_heldout_endpoint_registry(
                        path,
                        expected_sha256=digest,
                        route_registry=route_registry,
                    )

    def test_rejects_missing_or_nonpositive_runtime_task_count(self):
        from agentmemorygym_verl.heldout_endpoints import (
            load_heldout_endpoint_registry,
        )

        for route_id in ROUTES:
            with self.subTest(route=route_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                route_registry, path, _digest, registry = _fixture(root)
                route = next(item for item in registry["routes"] if item["route_id"] == route_id)
                manifest_name = (
                    "retrieval_and_grader_manifest"
                    if route_id == "literesearcher"
                    else "runtime_manifest"
                )
                asset = next(item for item in route["assets"] if item["name"] == manifest_name)
                manifest_path = Path(asset["path"])
                manifest = json.loads(manifest_path.read_text())
                manifest["test_items" if route_id == "literesearcher" else "task_count"] = 0
                asset["sha256"] = _write_json(manifest_path, manifest)
                digest = _write_json(path, registry)
                with self.assertRaisesRegex(OrchestratorError, "positive integer"):
                    load_heldout_endpoint_registry(
                        path,
                        expected_sha256=digest,
                        route_registry=route_registry,
                    )


class HeldoutEndpointLauncherTests(unittest.TestCase):
    def test_launchers_are_executable_and_parse_as_bash(self):
        for name in ("common", *ROUTES):
            with self.subTest(name=name):
                path = LAUNCHER_ROOT / f"{name}.sh"
                self.assertTrue(path.is_file(), path)
                if name != "common":
                    self.assertTrue(os.access(path, os.X_OK), path)
                subprocess.run(
                    ["bash", "-n", str(path)],
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                self.assertFalse(
                    any(line.startswith("+") for line in path.read_text().splitlines()),
                    f"{path} contains leaked patch markers",
                )

    def test_asset_environment_check_is_safe_with_nounset(self):
        with tempfile.TemporaryDirectory() as temporary:
            asset = Path(temporary) / "asset.json"
            digest = _write_json(asset, {"status": "pass"})
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    """
set -euo pipefail
source "$1"
export CAMG_HELDOUT_ASSET_FIXTURE_PATH="$2"
export CAMG_HELDOUT_ASSET_FIXTURE_SHA256="$3"
heldout_assert_asset_env FIXTURE "fixture asset"
""",
                    "bash",
                    str(LAUNCHER_ROOT / "common.sh"),
                    str(asset),
                    digest,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_executable_check_accepts_venv_style_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "python3.12"
            target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            target.chmod(0o700)
            entrypoint = root / "python3"
            entrypoint.symlink_to(target.name)
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -euo pipefail; source "$1"; '
                    'heldout_assert_executable "$2" "fixture Python"',
                    "bash",
                    str(LAUNCHER_ROOT / "common.sh"),
                    str(entrypoint),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_launchers_use_registry_git_roots_without_double_outer_component(self):
        common = (LAUNCHER_ROOT / "common.sh").read_text()
        self.assertIn('basename -- "$CAMG_HELDOUT_SOURCE_OUTER_ROOT"', common)
        self.assertIn(
            '"$CAMG_HELDOUT_SOURCE_OUTER_ROOT/AgentGym"', common
        )
        for route_id in ROUTES:
            with self.subTest(route=route_id):
                source = (LAUNCHER_ROOT / f"{route_id}.sh").read_text()
                for stale in (
                    "$CAMG_HELDOUT_SOURCE_OUTER_ROOT/AgentGym-RL",
                    "$OUTER_SOURCE_ROOT/AgentGym-RL",
                ):
                    self.assertNotIn(stale, source)
                self.assertNotIn(
                    "SOURCE_OUTER=$CAMG_HELDOUT_SOURCE_OUTER_ROOT/AgentGym-RL",
                    source,
                )

    def test_each_route_uses_verified_heldout_contract(self):
        for route_id in ROUTES:
            with self.subTest(route=route_id):
                source = (LAUNCHER_ROOT / f"{route_id}.sh").read_text()
                self.assertIn('source "$HERE/common.sh"', source)
                self.assertIn(
                    f"heldout_assert_base_contract {route_id}", source
                )
                self.assertIn("heldout_assert_parent", source)
                self.assertIn("CAMG_HELDOUT_TASK_COUNT", source)
                self.assertIn("CAMG_HELDOUT_SOURCE_OUTER_ROOT", source)
                self.assertIn("CAMG_HELDOUT_SOURCE_INNER_ROOT", source)
                for asset in ASSET_NAMES[route_id]:
                    self.assertIn(
                        f"heldout_assert_asset_env {asset.upper()}", source
                    )

    def test_shop_uses_sparse_fixed_heldout_window(self):
        source = (LAUNCHER_ROOT / "webshop.sh").read_text()
        self.assertIn("fixed_window_sparse_routing", source)
        self.assertIn('--procedural-provider-mode "$provider_mode"', source)
        self.assertIn('--procedural-task-count "$provider_task_count"', source)
        self.assertIn('--split "$provider_split"', source)
        self.assertIn(
            "provider_task_count > (end_orbit - start_orbit) * 2", source
        )
        self.assertIn("max(routing_idx) != provider_task_count - 1", source)
        self.assertNotIn("provider_task_count != end_orbit * 2", source)
        for stale in ("--procedural-task-count 6400", "reseeded_stream", "--split train"):
            self.assertNotIn(stale, source)

    def test_shop_preserves_frozen_training_memory_contract(self):
        source = (LAUNCHER_ROOT / "webshop.sh").read_text()
        for frozen_argument in (
            "--memory-first-add-reward 0",
            "--memory-first-later-retrieve-reward 0",
            "--memory-exact-repeat-reward 0",
            "--ltm-inventory-mode hidden",
            "--ltm-transition-notice-mode none",
            "--memory-prompt-mode natural_filesystem",
            "--action-listing-mode separate",
        ):
            self.assertIn(frozen_argument, source)

    def test_swesmith_uses_formal_eval_subset_and_run_scoped_rootfs(self):
        source = (LAUNCHER_ROOT / "swesmith.sh").read_text()
        self.assertIn("camg_swesmith_formal_eval_runtime_manifest_v5", source)
        self.assertIn("CAMG_HELDOUT_ASSET_FORMAL_EVAL_SELECTION_PATH", source)
        self.assertIn("CAMG_HELDOUT_ASSET_ADMITTED_POOL_MANIFEST_PATH", source)
        self.assertIn("CAMG_HELDOUT_ASSET_EXTENSION_POOL_MANIFEST_PATH", source)
        self.assertIn("prepare_swesmith_oci_rootfs.py", source)
        self.assertIn(
            "PREPARE_ROOTFS=$SOURCE_OUTER/AgentGym-RL/scripts/agentmemory/prepare_swesmith_oci_rootfs.py",
            source,
        )
        self.assertNotIn(
            "PREPARE_ROOTFS=$SOURCE_OUTER/scripts/agentmemory/prepare_swesmith_oci_rootfs.py",
            source,
        )
        self.assertNotIn(
            "PREPARE_ROOTFS=$SOURCE_ROOT/scripts/agentmemory/prepare_swesmith_oci_rootfs.py",
            source,
        )
        self.assertNotIn("mapfile -d", source)
        self.assertIn("while IFS= read -r -d '' binding_arg; do", source)
        project_root = LAUNCHER_ROOT.parents[2]
        self.assertTrue(
            (
                project_root
                / "AgentGym-RL/scripts/agentmemory/prepare_swesmith_oci_rootfs.py"
            ).is_file()
        )
        self.assertIn("mirror-bundles-manifest", source)
        self.assertIn("SWESMITH_DETAIL_TOKEN", source)
        self.assertNotIn("full-pool-formal100", source)
        self.assertNotIn("-eq 110", source)
        self.assertIn(
            "ROOTFS_RUN_ROOT=/tmp/agentmemorygym-swesmith-heldout-$RUN_KEY",
            source,
        )
        self.assertNotIn(
            "ROOTFS_RUN_ROOT=/dev/shm/agentmemorygym-swesmith-heldout-$RUN_KEY",
            source,
        )
        self.assertIn("--materialize-profile-image", source)
        self.assertIn("selected_repository_task_counts", source)
        self.assertIn("bamboo-proxy.jd.com:80", source)
        self.assertIn("--fallback-transport-prefix dockerproxy.net", source)
        self.assertIn("--fallback-transport-prefix docker.1panel.live", source)
        self.assertNotIn("--fallback-transport-prefix docker.io", source)

    def test_literesearcher_uses_all_heldout_rows_and_identity_source(self):
        source = (LAUNCHER_ROOT / "literesearcher.sh").read_text()
        self.assertIn("camg_literesearcher_heldout_runtime_binding_v1", source)
        self.assertIn("CAMG_HELDOUT_ASSET_RUNTIME_ROWS_PATH", source)
        self.assertIn("CAMG_HELDOUT_SOURCE_ENDPOINT_ROOT", source)
        self.assertIn("CAMG_HELDOUT_SOURCE_ENDPOINT_COMMIT", source)
        self.assertIn(
            "$LITERESEARCHER_ENDPOINT_SOURCE_ROOT/agentenv-agentmemory",
            source,
        )
        self.assertIn("LITERESEARCHER_CAMG_ROLE=heldout", source)
        self.assertNotIn("literesearcher-only-r46.jsonl", source)

    def test_openmle_uses_isolated_heldout_contract(self):
        source = (LAUNCHER_ROOT / "openmle_fast.sh").read_text()
        self.assertIn("camg_openmle_fast_heldout_runtime_manifest_v1", source)
        self.assertIn("OPENMLE_FAST_MANIFEST_ROLE=heldout", source)
        self.assertIn("OPENMLE_FAST_TASK_MANIFEST", source)
        self.assertNotIn("formal100", source)
        runtime_source = (LAUNCHER_ROOT / "openmle_runtime_base.py").read_text()
        self.assertNotIn("def parse_args", runtime_source)
        self.assertNotIn("def main", runtime_source)
        self.assertNotIn('choices=("gate1", "formal100")', runtime_source)

    def test_missing_verified_environment_fails_before_route_start(self):
        for route_id in ROUTES:
            with self.subTest(route=route_id):
                completed = subprocess.run(
                    [str(LAUNCHER_ROOT / f"{route_id}.sh")],
                    env={"PATH": os.environ.get("PATH", "")},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 64, completed.stdout)
                self.assertIn("missing environment variable", completed.stdout)


class OpenMleHeldoutValidatorTests(unittest.TestCase):
    def test_child_environment_drops_heldout_control_plane_bindings(self):
        inherited = {
            "PATH": "/usr/bin",
            "CAMG_HELDOUT_ASSET_PRIVATE_GRADER_BINDINGS_PATH": "/private.jsonl",
            "CAMG_HELDOUT_SOURCE_OUTER_ROOT": "/source",
            "OPENMLE_FAST_PRIVATE_RUNNER": "/runner.py",
        }
        with mock.patch.dict(os.environ, inherited, clear=True):
            sanitized = openmle_supervisor.runtime.sanitized_environment()
        self.assertEqual(sanitized, {"PATH": "/usr/bin"})

    def _fixture(self, root: Path):
        outer = root / "AgentGym-RL"
        inner = outer / "AgentGym"
        inner.mkdir(parents=True)
        selected = {}
        for index in range(10):
            relative = (
                "agentenv-openmle-fast/agentenv_openmle_fast/"
                f"fixture_{index}.py"
            )
            path = inner / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture {index}\n", encoding="utf-8")
            selected[f"inner:{relative}"] = hashlib.sha256(path.read_bytes()).hexdigest()

        digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        task_id = "heldout-task-0"
        shared = {
            "task_id": task_id,
            "source_family": "family-0",
            "grader_binding": "grader-0",
            "grader_binding_sha256": digest("grader"),
            "package_identity_sha256": digest("package"),
            "task_spec_sha256": digest("task"),
        }
        private = {
            "schema": "openmle_fast_fullpool_private_grader_manifest_v1",
            "runtime_digest": digest("runtime"),
            "records": [
                {
                    **shared,
                    "answer_sha256": digest("answer"),
                    "metric_sha256": digest("metric"),
                }
            ],
        }
        pod_root = root / "installed"
        private_path = pod_root / "private.json"
        private_sha = _write_json(private_path, private)
        heldout = {
            "schema": "openmle_fast_public_manifest_v1",
            "role": "heldout",
            "task_count": 1,
            "source_family_count": 1,
            "release_revision": "release-revision",
            "records": [shared],
        }
        heldout_path = root / "heldout.json"
        heldout_sha = _write_json(heldout_path, heldout)
        binding_path = root / "bindings.jsonl"
        binding_path.write_text(
            json.dumps(private["records"][0], sort_keys=True) + "\n",
            encoding="utf-8",
        )
        routing_path = root / "routing.jsonl"
        routing_path.write_text(
            json.dumps(
                {
                    "data_idx": 0,
                    "item_id": "openmle_fast_0",
                    "extra_info": {
                        "index": 0,
                        "role": "heldout",
                        "task_id": task_id,
                        "source_family": "family-0",
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        source_lock = root / "source-lock.json"
        source_lock.write_text("{}\n", encoding="utf-8")
        document = {
            "runtime_source": {
                "outer_commit": "a" * 40,
                "inner_commit": "b" * 40,
                "openmle_tasks_revision": "release-revision",
                "selected_files": selected,
            },
            "integration": {
                "pod_root": str(pod_root),
                "private_manifest": {
                    "relpath": private_path.name,
                    "sha256": private_sha,
                },
                "manifests": {
                    "heldout": {
                        "role": "heldout",
                        "task_count": 1,
                        "sha256": heldout_sha,
                    }
                },
            },
            "exact_runtime": {"runtime_digest": digest("runtime")},
            "launch_contracts": {},
        }
        runtime = {
            "schema": "camg_openmle_fast_heldout_runtime_manifest_v1",
            "status": "ready",
            "heldout_evaluation_run": False,
            "task_count": 1,
            "heldout_manifest": {
                "path": str(heldout_path),
                "bytes": heldout_path.stat().st_size,
                "sha256": heldout_sha,
            },
            "private_grader_bindings": {
                "path": str(binding_path),
                "bytes": binding_path.stat().st_size,
                "sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
            },
            "routing": {
                "path": str(routing_path),
                "bytes": routing_path.stat().st_size,
                "sha256": hashlib.sha256(routing_path.read_bytes()).hexdigest(),
            },
            "source": {
                "outer_commit": "a" * 40,
                "inner_commit": "b" * 40,
                "source_locks": [
                    {
                        "path": str(source_lock),
                        "sha256": hashlib.sha256(source_lock.read_bytes()).hexdigest(),
                    }
                ],
            },
        }
        runtime_path = root / "runtime.json"
        _write_json(runtime_path, runtime)
        args = SimpleNamespace(
            runtime_schema="camg_openmle_fast_heldout_runtime_manifest_v1",
            runtime_manifest=runtime_path,
            heldout_manifest=heldout_path,
            private_grader_bindings=binding_path,
            routing=routing_path,
            source_lock=source_lock,
            task_count=1,
            inner_root=inner,
        )
        contract_module = SimpleNamespace(
            load_source_lock=lambda path, require_final_runtime: copy.deepcopy(document)
        )
        return args, contract_module, outer, inner

    def test_real_validator_accepts_bound_heldout_fixture_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            args, contract_module, outer, inner = self._fixture(Path(directory))
            overlay, receipt = openmle_supervisor.validate_and_overlay(
                args, contract_module
            )
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(receipt["task_count"], 1)
            self.assertEqual(
                overlay["launch_contracts"]["heldout_eval"],
                {"manifest_role": "heldout", "task_count": 1},
            )
            self.assertEqual(
                openmle_supervisor._workspace_root(outer, inner), outer.parent.resolve()
            )

            routing = json.loads(args.routing.read_text())
            routing["extra_info"]["source_family"] = "tampered-family"
            args.routing.write_text(json.dumps(routing) + "\n", encoding="utf-8")
            runtime = json.loads(args.runtime_manifest.read_text())
            runtime["routing"]["bytes"] = args.routing.stat().st_size
            runtime["routing"]["sha256"] = hashlib.sha256(
                args.routing.read_bytes()
            ).hexdigest()
            _write_json(args.runtime_manifest, runtime)
            with self.assertRaisesRegex(RuntimeError, "source family drifted"):
                openmle_supervisor.validate_and_overlay(args, contract_module)

    def test_openmle_source_topology_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outer = root / "AgentGym-RL"
            inner = outer / "AgentGym"
            inner.mkdir(parents=True)
            self.assertEqual(
                openmle_supervisor._workspace_root(outer, inner), root.resolve()
            )
            with self.assertRaisesRegex(RuntimeError, "AgentGym-RL checkout"):
                openmle_supervisor._workspace_root(root / "outer", inner)
            with self.assertRaisesRegex(RuntimeError, "AgentGym submodule"):
                openmle_supervisor._workspace_root(outer, root / "other")

class _FakeEndpoint:
    def __init__(self, route_id: str, expected: dict) -> None:
        self.route_id = route_id
        self.expected = expected
        self.calls = []

    def __call__(self, method, url, *, body, headers, timeout_seconds):
        self.calls.append((method, url, body, headers, timeout_seconds))
        path = "/" + url.split("/", 3)[-1] if url.count("/") >= 3 else "/"
        if path == "/create":
            return {"id": 7}
        if path == "/close":
            return {"closed": True, "id": 7}
        if path == "/detail":
            return {
                "data_idx": self.expected["data_idx"],
                "instance_id": self.expected["instance_id"],
                "base_repository": self.expected["base_repository"],
            }
        if path != "/reset":
            raise AssertionError(path)
        if self.route_id == "webshop":
            return {
                "info": {
                    "data_idx": self.expected["data_idx"],
                    "scenario_id": self.expected["scenario_id"],
                    "orbit_index": self.expected["orbit_index"],
                }
            }
        if self.route_id == "literesearcher":
            return {
                "info": {
                    "data_idx": self.expected["data_idx"],
                    "row_identity": self.expected["row_identity"],
                    "source_pool_index": self.expected["source_pool_index"],
                }
            }
        if self.route_id == "openmle_fast":
            return {
                "info": {
                    "data_idx": self.expected["data_idx"],
                    "task_id": self.expected["task_id"],
                    "source_family": self.expected["source_family"],
                    "manifest_role": "heldout",
                    "manifest_sha256": self.expected["manifest_sha256"],
                }
            }
        return {"info": {}}


class HeldoutEndpointProbeTests(unittest.TestCase):
    def test_create_reset_identity_and_close_all_four_routes(self):
        from agentmemorygym_verl.heldout_endpoints import (
            load_heldout_endpoint_registry,
            probe_heldout_reset_identity,
        )

        rows = {
            "webshop": {
                "route_id": "webshop",
                "data_idx": 9200,
                "extra_info": {
                    "scenario_id": "stale-direct-value",
                    "orbit_index": 999999,
                    "source_extra_info": {
                        "scenario_id": "baking",
                        "orbit_index": 4600,
                    },
                },
            },
            "swesmith": {
                "route_id": "swesmith",
                "data_idx": 3,
                "extra_info": {
                    "instance_id": "stale-direct-value",
                    "base_repository": "stale-direct-value",
                    "source_extra_info": {
                        "instance_id": "pallets.flask.issue-3",
                        "base_repository": "pallets",
                    },
                },
            },
            "literesearcher": {
                "route_id": "literesearcher",
                "data_idx": 12,
                "extra_info": {
                    "row_identity": "0" * 64,
                    "source_pool_index": 999999,
                    "source_extra_info": {
                        "row_identity": "f" * 64,
                        "source_pool_index": 77,
                    },
                },
            },
            "openmle_fast": {
                "route_id": "openmle_fast",
                "data_idx": 5,
                "extra_info": {
                    "task_id": "stale-direct-value",
                    "source_family": "stale-direct-value",
                    "role": "train_pool",
                    "manifest_sha256": "0" * 64,
                    "source_extra_info": {
                        "task_id": "competition@5",
                        "source_family": "KAGGLE_DATASET:a/b",
                        "role": "heldout",
                        "manifest_sha256": "a" * 64,
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            route_registry, path, digest, _ = _fixture(Path(directory))
            specs, _ = load_heldout_endpoint_registry(
                path,
                expected_sha256=digest,
                route_registry=route_registry,
            )
            for spec in specs:
                expected = {
                    "data_idx": rows[spec.route_id]["data_idx"],
                    **rows[spec.route_id]["extra_info"]["source_extra_info"],
                }
                fake = _FakeEndpoint(spec.route_id, expected)
                receipt = probe_heldout_reset_identity(
                    spec,
                    rows[spec.route_id],
                    request_json=fake,
                )
                self.assertEqual(receipt["status"], "pass")
                self.assertEqual(fake.calls[0][0:2], ("POST", spec.endpoint + "/create"))
                self.assertEqual(fake.calls[-1][0:2], ("POST", spec.endpoint + "/close"))
                reset = next(call for call in fake.calls if call[1].endswith("/reset"))
                self.assertEqual(reset[2], {"id": 7, "data_idx": expected["data_idx"]})
                self.assertEqual(fake.calls[-1][2], {"id": 7})
                if spec.route_id == "swesmith":
                    detail = next(call for call in fake.calls if call[1].endswith("/detail"))
                    self.assertEqual(
                        detail[3], {"X-SWESMITH-Detail-Token": "private-token"}
                    )

    def test_identity_mismatch_still_closes_slot(self):
        from agentmemorygym_verl.heldout_endpoints import (
            load_heldout_endpoint_registry,
            probe_heldout_reset_identity,
        )

        with tempfile.TemporaryDirectory() as directory:
            route_registry, path, digest, _ = _fixture(Path(directory))
            specs, _ = load_heldout_endpoint_registry(
                path,
                expected_sha256=digest,
                route_registry=route_registry,
            )
            spec = next(item for item in specs if item.route_id == "webshop")
            row = {
                "route_id": "webshop",
                "data_idx": 9200,
                "extra_info": {
                    "source_extra_info": {
                        "scenario_id": "baking",
                        "orbit_index": 4600,
                    }
                },
            }
            fake = _FakeEndpoint(
                "webshop",
                {
                    "data_idx": 9200,
                    "scenario_id": "electronics",
                    "orbit_index": 4600,
                },
            )
            with self.assertRaisesRegex(OrchestratorError, "identity mismatch"):
                probe_heldout_reset_identity(spec, row, request_json=fake)
            self.assertTrue(fake.calls[-1][1].endswith("/close"))

    def test_probe_rejects_derived_or_missing_native_identity_fields(self):
        from agentmemorygym_verl.heldout_endpoints import (
            load_heldout_endpoint_registry,
            probe_heldout_reset_identity,
        )

        with tempfile.TemporaryDirectory() as directory:
            route_registry, path, digest, _ = _fixture(Path(directory))
            specs, _ = load_heldout_endpoint_registry(
                path,
                expected_sha256=digest,
                route_registry=route_registry,
            )
            rows = {
                "webshop": {
                    "route_id": "webshop",
                    "data_idx": 9200,
                    "extra_info": {
                        "source_extra_info": {
                            "scenario_id": "baking",
                            "orbit_index": 4600,
                        }
                    },
                },
                "swesmith": {
                    "route_id": "swesmith",
                    "data_idx": 3,
                    "extra_info": {
                        "source_extra_info": {
                            "instance_id": "pallets.flask.issue-3",
                            "base_repository": "pallets",
                        }
                    },
                },
            }
            for route_id in ("webshop", "swesmith"):
                spec = next(item for item in specs if item.route_id == route_id)
                expected = {
                    "data_idx": rows[route_id]["data_idx"],
                    **rows[route_id]["extra_info"]["source_extra_info"],
                }
                fake = _FakeEndpoint(route_id, expected)
                original = fake.__call__

                def missing_explicit_field(method, url, **kwargs):
                    payload = original(method, url, **kwargs)
                    path_only = "/" + url.split("/", 3)[-1]
                    if route_id == "webshop" and path_only == "/reset":
                        payload["info"].pop("orbit_index")
                    if route_id == "swesmith" and path_only == "/detail":
                        payload.pop("base_repository")
                    return payload

                with self.subTest(route=route_id), self.assertRaisesRegex(
                    OrchestratorError, "identity mismatch"
                ):
                    probe_heldout_reset_identity(
                        spec,
                        rows[route_id],
                        request_json=missing_explicit_field,
                    )
                self.assertTrue(fake.calls[-1][1].endswith("/close"))


if __name__ == "__main__":
    unittest.main()
