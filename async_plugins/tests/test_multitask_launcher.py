# ruff: noqa: B023, SIM117
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from agentmemorygym_verl.identity import EXPECTED_VERL_COMMIT
from agentmemorygym_verl.multitask_orchestrator import (
    EXPECTED_ROUTE_IDS,
    EndpointLaunchSpec,
    ExactProcessSupervisor,
    HolderLease,
    LaunchPlan,
    LocalBackend,
    MarkerLease,
    OrchestratorError,
    ProcessLease,
    _atomic_json,
    _execute_local,
    assert_ports_available,
    build_generic_launch_command,
    build_launch_plan,
    execute_launch_plan,
    load_endpoint_registry,
    load_orchestrator_config,
    process_start_ticks,
    start_endpoint_processes,
)
from agentmemorygym_verl.routes import (
    canonical_policy_framing_sha256,
    load_route_registry,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "async_plugins/config/amg_multitask400.yaml"
CONFIG_131K = ROOT / "async_plugins/config/amg_multitask400_131k.yaml"
CONFIG_RESUME = ROOT / "async_plugins/config/amg_multitask200_resume.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def _git_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Launcher Test"],
        check=True,
    )
    (path / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _gate_receipt(
    route_id: str,
    launcher_sha256: str,
    runtime_sha256: str,
    source_commit: str,
) -> dict:
    environment = "openmle-fast" if route_id == "openmle_fast" else route_id
    receipt = {
        "schema": "amg_single_card_optimizer_update_gate_v1",
        "environment": environment,
        "status": "pass",
        "run_id": f"gate-{route_id}",
        "execution": {
            "gpu_count": 1,
            "gpu_indices": [0],
            "pod_host": "gate-pod",
        },
        "training": {
            "optimizer_update_count": 1,
            "trainer_exit_code": 0,
            "update1_completed": True,
            "actor_parameter_delta_nonzero": True,
            "critic_parameter_delta_nonzero": True,
            "actor_parameter_delta_l2": 0.1,
            "critic_parameter_delta_l2": 0.2,
            "trajectory_row_count": 8,
        },
        "runtime": {
            "environment_ready": True,
            "asset_hashes_verified": True,
            "fatal_error_count": 0,
            "forwarding_process_count": 0,
            "listener_scope": "same_pod_loopback_only",
        },
        "cleanup": {
            "residue_after_cleanup": 0,
            "markers_cleared": True,
            "checkpoint_readback": True,
        },
        "source": {
            "launcher_sha256": launcher_sha256,
            "runtime_manifest_sha256": runtime_sha256,
            "environment_source_commit": source_commit,
            "environment_outer_source_commit": source_commit,
            "shared_runtime_source_commit": source_commit,
        },
    }
    if route_id == "swesmith":
        receipt["runtime"].update(
            {
                "formal_eligible": True,
                "sandbox_backend": "LinuxNamespaceEpisodeSandbox",
                "sandbox_contract": "swesmith_linux_namespace_oci_rootfs_v1",
                "rootfs_contract": ("digest_pinned_oci_profile_rootfs_read_only"),
                "network_contract": ("new_namespace_loopback_only_no_external_routes"),
                "formal_episode_audit_count": 8,
            }
        )
        receipt["cleanup"].update(
            {
                "sandbox_mount_count_after_cleanup": 0,
                "holders_restored": True,
                "temporary_path_count_after_cleanup": 0,
            }
        )
        receipt["source"].update(
            {
                "source_worktree_dirty": False,
                "environment_outer_source_commit": "a" * 40,
                "environment_source_detached": True,
                "shared_runtime_source_detached": True,
                "shared_runtime_worktree_dirty": False,
            }
        )
        receipt["timing"] = {
            "startup_seconds": 1.0,
            "optimizer_update_wall_seconds": 2.0,
            "total_wall_seconds": 3.0,
        }
    return receipt


class RegistryFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source_root = root / "source"
        self.source_commit = _git_repo(self.source_root)
        self.route_registry_path = root / "routes.json"
        self.route_payload = {
            "schema": "amg_route_registry_v1",
            "agent_name": "amg_task_neutral_async",
            "routes": [
                {
                    "route_id": route_id,
                    "max_rounds": 30,
                    "max_observation_tokens": 8192,
                    "policy_framing_sha256": canonical_policy_framing_sha256(
                        [{"role": "system", "content": route_id}]
                    ),
                    "route_attestation_sha256": str(index + 1) * 64,
                    "client": {
                        "task_name": route_id,
                        "env_addr": f"http://127.0.0.1:{49100 + index}",
                        "timeout": 240,
                        "max_retries": 2,
                    },
                }
                for index, route_id in enumerate(EXPECTED_ROUTE_IDS)
            ],
        }
        self.route_registry_sha256 = _write_json(
            self.route_registry_path, self.route_payload
        )
        self.route_registry = load_route_registry(
            self.route_registry_path,
            expected_sha256=self.route_registry_sha256,
        )
        self.registry_path = root / "endpoint-registry.json"
        entries = []
        for index, route_id in enumerate(EXPECTED_ROUTE_IDS):
            endpoint_launcher = root / f"start-{route_id}.sh"
            endpoint_launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            endpoint_launcher.chmod(0o755)
            gate_launcher = root / f"gate-{route_id}.sh"
            gate_launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            gate_launcher.chmod(0o755)
            runtime_manifest = root / f"runtime-{route_id}.json"
            runtime_manifest.write_text("{}\n", encoding="utf-8")
            asset = root / f"asset-{route_id}.bin"
            asset.write_bytes(route_id.encode("utf-8"))
            gate_receipt = root / f"receipt-{route_id}.json"
            gate_receipt_sha256 = _write_json(
                gate_receipt,
                _gate_receipt(
                    route_id,
                    _sha256(gate_launcher),
                    _sha256(runtime_manifest),
                    self.source_commit,
                ),
            )
            entries.append(
                {
                    "route_id": route_id,
                    "route_attestation_sha256": str(index + 1) * 64,
                    "endpoint": f"http://127.0.0.1:{49100 + index}",
                    "gate_receipt": {
                        "path": str(gate_receipt),
                        "sha256": gate_receipt_sha256,
                        "expected": {},
                    },
                    "gate_launcher": {
                        "path": str(gate_launcher),
                        "sha256": _sha256(gate_launcher),
                    },
                    "runtime_manifest": {
                        "path": str(runtime_manifest),
                        "sha256": _sha256(runtime_manifest),
                    },
                    "sources": [
                        {
                            "name": "outer",
                            "root": str(self.source_root),
                            "commit": self.source_commit,
                            "receipt_field": "source.shared_runtime_source_commit",
                        },
                        {
                            "name": "inner",
                            "root": str(self.source_root),
                            "commit": self.source_commit,
                            "receipt_field": "source.environment_source_commit",
                        },
                    ],
                    "assets": [{"path": str(asset), "sha256": _sha256(asset)}],
                    "endpoint_launcher": {
                        "path": str(endpoint_launcher),
                        "sha256": _sha256(endpoint_launcher),
                        "argv": ["--run-dir", "@ENDPOINT_RUN_DIR@"],
                        "environment": {},
                        "working_directory": str(root),
                        "process_contract": "foreground_supervisor_v1",
                    },
                    "readiness": {
                        "url": f"http://127.0.0.1:{49100 + index}/metadata",
                        "expected": {"status": "ready", "route_id": route_id},
                        "timeout_seconds": 60,
                        "poll_seconds": 0.1,
                        "request_timeout_seconds": 1,
                    },
                    "cleanup_timeout_seconds": 30,
                }
            )
        self.registry_payload = {
            "schema": "amg_multitask_endpoint_registry_v1",
            "status": "pass",
            "route_order": list(EXPECTED_ROUTE_IDS),
            "routes": entries,
        }
        self.registry_sha256 = _write_json(self.registry_path, self.registry_payload)

    def rewrite(self) -> str:
        return _write_json(self.registry_path, self.registry_payload)


class ProductionResolveFixture:
    """Hermetic external-runtime fixture for the real two-shell launch chain."""

    OUTER_COMMIT = "4d8ce04b5d40c2e79abb01b46051a230c7ab3973"
    INNER_COMMIT = "bf12f1d74bf38b5a94afc7c3f913702398c0ec21"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.outer = root / "outer"
        self.verl = root / "verl"
        self.run_dir = root / "run"
        self.endpoint_sentinel = root / "endpoint-started"
        self.trainer_sentinel = root / "trainer-started"
        self.resolve_sentinel = root / "hydra-resolved"
        self._copy_launch_surface()
        self._write_fake_verl_and_runtime()
        self.route_registry, self.route_registry_sha256, ports = (
            self._write_route_registry()
        )
        self.schedule, self.schedule_sha256, sources = self._write_schedule()
        self.schedule_certificate = self._write_schedule_certificate(sources)
        self.source_lock = self._write_source_lock()
        self.endpoint_registry, self.endpoint_registry_sha256 = (
            self._write_endpoint_registry(ports)
        )
        self.fake_bin = self._write_fake_git()

    def _copy_launch_surface(self) -> None:
        plugin_root = self.outer / "async_plugins"
        shutil.copytree(
            ROOT / "async_plugins/agentmemorygym_verl",
            plugin_root / "agentmemorygym_verl",
        )
        (plugin_root / "config").mkdir()
        shutil.copy2(CONFIG, plugin_root / "config/amg_multitask400.yaml")
        (plugin_root / "scripts").mkdir()
        for name in (
            "launch_amg_multitask_fully_async.sh",
            "launch_amg_fully_async.sh",
        ):
            shutil.copy2(ROOT / "async_plugins/scripts" / name, plugin_root / "scripts")
        (plugin_root / "vendor").mkdir()
        shutil.copy2(
            ROOT / "async_plugins/vendor/trl-0.9.6-py3-none-any.whl",
            plugin_root / "vendor",
        )
        inner_file = self.outer / "AgentGym/agentenv/agentenv/envs/fixture.py"
        inner_file.parent.mkdir(parents=True)
        inner_file.write_text("VALUE = 1\n", encoding="utf-8")
        (self.outer / "AgentGym/agentenv-openmle-fast").mkdir(parents=True)

    def _write_fake_verl_and_runtime(self) -> None:
        module = self.verl / "verl/experimental/fully_async_policy/fully_async_main.py"
        module.parent.mkdir(parents=True)
        for parent in (module.parents[2], module.parents[1], module.parent):
            (parent / "__init__.py").write_text("", encoding="utf-8")
        module.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "import yaml\n"
            f"RESOLVE_SENTINEL = Path({str(self.resolve_sentinel)!r})\n"
            f"TRAINER_SENTINEL = Path({str(self.trainer_sentinel)!r})\n"
            "def assign(root, dotted, value):\n"
            "    target = root\n"
            "    parts = dotted.split('.')\n"
            "    for part in parts[:-1]:\n"
            "        target = target.setdefault(part, {})\n"
            "    target[parts[-1]] = value\n"
            "if '--cfg' not in sys.argv:\n"
            "    TRAINER_SENTINEL.write_text('started\\n', encoding='utf-8')\n"
            "    raise SystemExit(91)\n"
            "config = {}\n"
            "for token in sys.argv[1:]:\n"
            "    if '=' not in token:\n"
            "        continue\n"
            "    key, raw = token.split('=', 1)\n"
            "    assign(config, key.lstrip('+'), yaml.safe_load(raw))\n"
            "RESOLVE_SENTINEL.write_text('resolved\\n', encoding='utf-8')\n"
            "print(yaml.safe_dump(config, sort_keys=False), end='')\n",
            encoding="utf-8",
        )

        model_root = self.root / "model"
        model_root.mkdir()
        model_names = (
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
        model_hashes = {}
        for name in model_names:
            path = model_root / name
            path.write_bytes((name + "\n").encode("utf-8"))
            model_hashes[name] = _sha256(path)
        (self.verl / "sitecustomize.py").write_text(
            "import agentmemorygym_verl.identity as identity\n"
            f"identity.LOCKED_MODEL_FILE_SHA256.clear()\n"
            f"identity.LOCKED_MODEL_FILE_SHA256.update({model_hashes!r})\n",
            encoding="utf-8",
        )
        self.model_root = model_root
        self.site_packages = self.root / "runtime/site-packages"
        self.site_packages.mkdir(parents=True)
        self.bundle_sha256 = "b" * 64
        self.bundle_sha256_file = self.root / "runtime/runtime.sha256"
        self.bundle_sha256_file.write_text(
            f"{self.bundle_sha256}  runtime.tar.zst\n", encoding="utf-8"
        )

    def _write_route_registry(self) -> tuple[Path, str, tuple[int, ...]]:
        reservations = []
        try:
            for _ in EXPECTED_ROUTE_IDS:
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.bind(("127.0.0.1", 0))
                reservations.append(listener)
            ports = tuple(listener.getsockname()[1] for listener in reservations)
        finally:
            for listener in reservations:
                listener.close()
        payload = {
            "schema": "amg_route_registry_v1",
            "agent_name": "amg_task_neutral_async",
            "routes": [
                {
                    "route_id": route_id,
                    "max_rounds": 30,
                    "max_observation_tokens": 8192,
                    "policy_framing_sha256": canonical_policy_framing_sha256(
                        [{"role": "system", "content": route_id}]
                    ),
                    "route_attestation_sha256": str(index + 1) * 64,
                    "client": {
                        "task_name": route_id,
                        "env_addr": f"http://127.0.0.1:{ports[index]}",
                        "timeout": 240,
                        "max_retries": 2,
                    },
                }
                for index, route_id in enumerate(EXPECTED_ROUTE_IDS)
            ],
        }
        path = self.root / "route-registry.json"
        return path, _write_json(path, payload), ports

    def _write_schedule(self) -> tuple[Path, str, dict[str, dict[str, object]]]:
        manifest_digest = "d" * 64
        sources = {
            route_id: {
                "schedule_sha256": str(index + 5) * 64,
                "route_attestation_sha256": str(index + 1) * 64,
                "source_manifest_digest": "9abc"[index] * 64,
                "source_panel_id": f"source-{route_id}",
                "source_row_count": 6400,
                "allow_repetition": False,
            }
            for index, route_id in enumerate(EXPECTED_ROUTE_IDS)
        }
        schedule = self.root / "multitask400.jsonl"
        with schedule.open("w", encoding="utf-8") as handle:
            for position in range(25_600):
                route_id = EXPECTED_ROUTE_IDS[position % len(EXPECTED_ROUTE_IDS)]
                source = sources[route_id]
                row = {
                    "item_id": f"{route_id}:{position // 4}",
                    "data_idx": position // 4,
                    "index": position,
                    "route_id": route_id,
                    "agent_name": "amg_task_neutral_async",
                    "data_source": route_id,
                    "extra_info": {
                        "schedule_position": position,
                        "index": position,
                        "route_id": route_id,
                        "role": "train_pool",
                        "manifest_digest": manifest_digest,
                        "panel_id": "production-chain-fixture",
                        "route_registry_sha256": self.route_registry_sha256,
                        "route_attestation_sha256": source["route_attestation_sha256"],
                        "source_schedule_sha256": source["schedule_sha256"],
                        "source_manifest_digest": source["source_manifest_digest"],
                        "source_panel_id": source["source_panel_id"],
                    },
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return schedule, _sha256(schedule), sources

    def _write_schedule_certificate(
        self, sources: dict[str, dict[str, object]]
    ) -> Path:
        path = self.root / "multitask-schedule-certificate.json"
        _write_json(
            path,
            {
                "schema": "amg_multitask_schedule_certificate_v1",
                "spec_sha256": "d" * 64,
                "schedule_sha256": self.schedule_sha256,
                "route_registry_sha256": self.route_registry_sha256,
                "role": "train_pool",
                "panel_id": "production-chain-fixture",
                "agent_name": "amg_task_neutral_async",
                "optimizer_updates": 400,
                "samples_per_update": 64,
                "row_count": 25_600,
                "route_order": list(EXPECTED_ROUTE_IDS),
                "per_route_rows": {route_id: 6400 for route_id in EXPECTED_ROUTE_IDS},
                "sources": {
                    route_id: {
                        "schedule_sha256": source["schedule_sha256"],
                        "route_attestation_sha256": source["route_attestation_sha256"],
                        "source_row_count": source["source_row_count"],
                        "allow_repetition": source["allow_repetition"],
                    }
                    for route_id, source in sources.items()
                },
            },
        )
        return path

    def _write_source_lock(self) -> Path:
        outer_file = self.outer / "async_plugins/agentmemorygym_verl/routes.py"
        inner_file = self.outer / "AgentGym/agentenv/agentenv/envs/fixture.py"
        path = self.root / "multitask-source-lock.json"
        _write_json(
            path,
            {
                "schema": "amg_multitask_launcher_source_lock_v1",
                "status": "pass",
                "runtime_source": {
                    "outer_commit": self.OUTER_COMMIT,
                    "inner_commit": self.INNER_COMMIT,
                    "verl_commit": EXPECTED_VERL_COMMIT,
                    "selected_files": {
                        "outer:async_plugins/agentmemorygym_verl/routes.py": _sha256(
                            outer_file
                        ),
                        "inner:agentenv/agentenv/envs/fixture.py": _sha256(inner_file),
                    },
                },
                "training_runtime": {
                    "base_model": str(self.model_root),
                    "python": sys.executable,
                    "site_packages": str(self.site_packages),
                    "bundle_sha256": self.bundle_sha256,
                    "bundle_sha256_file": str(self.bundle_sha256_file),
                    "gpu_count": 8,
                    "gpu_type": "B300",
                },
                "integration": {
                    "route_registry": {
                        "sha256": self.route_registry_sha256,
                        "route_ids": list(EXPECTED_ROUTE_IDS),
                    },
                    "schedule_certificate": {
                        "sha256": _sha256(self.schedule_certificate),
                        "schedule_sha256": self.schedule_sha256,
                    },
                },
            },
        )
        return path

    def _write_endpoint_registry(self, ports: tuple[int, ...]) -> tuple[Path, str]:
        routes = []
        for index, route_id in enumerate(EXPECTED_ROUTE_IDS):
            gate_launcher = self.root / f"gate-{route_id}.sh"
            gate_launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            gate_launcher.chmod(0o755)
            runtime_manifest = self.root / f"runtime-{route_id}.json"
            runtime_manifest.write_text("{}\n", encoding="utf-8")
            endpoint_launcher = self.root / f"endpoint-{route_id}.sh"
            endpoint_launcher.write_text(
                "#!/bin/sh\n"
                f"printf 'started\\n' > {str(self.endpoint_sentinel)!r}\n"
                "exit 97\n",
                encoding="utf-8",
            )
            endpoint_launcher.chmod(0o755)
            asset = self.root / f"asset-{route_id}.bin"
            asset.write_bytes(route_id.encode("utf-8"))
            receipt = _gate_receipt(
                route_id,
                _sha256(gate_launcher),
                _sha256(runtime_manifest),
                self.INNER_COMMIT,
            )
            receipt["source"]["environment_outer_source_commit"] = self.OUTER_COMMIT
            receipt["source"]["shared_runtime_source_commit"] = self.OUTER_COMMIT
            receipt_path = self.root / f"receipt-{route_id}.json"
            receipt_sha256 = _write_json(receipt_path, receipt)
            endpoint = f"http://127.0.0.1:{ports[index]}"
            routes.append(
                {
                    "route_id": route_id,
                    "route_attestation_sha256": str(index + 1) * 64,
                    "endpoint": endpoint,
                    "gate_receipt": {
                        "path": str(receipt_path),
                        "sha256": receipt_sha256,
                        "expected": {},
                    },
                    "gate_launcher": {
                        "path": str(gate_launcher),
                        "sha256": _sha256(gate_launcher),
                    },
                    "runtime_manifest": {
                        "path": str(runtime_manifest),
                        "sha256": _sha256(runtime_manifest),
                    },
                    "sources": [
                        {
                            "name": "outer",
                            "root": str(self.outer),
                            "commit": self.OUTER_COMMIT,
                            "receipt_field": "source.shared_runtime_source_commit",
                        },
                        {
                            "name": "inner",
                            "root": str(self.outer / "AgentGym"),
                            "commit": self.INNER_COMMIT,
                            "receipt_field": "source.environment_source_commit",
                        },
                    ],
                    "assets": [{"path": str(asset), "sha256": _sha256(asset)}],
                    "endpoint_launcher": {
                        "path": str(endpoint_launcher),
                        "sha256": _sha256(endpoint_launcher),
                        "argv": [],
                        "environment": {},
                        "working_directory": str(self.root),
                        "process_contract": "foreground_supervisor_v1",
                    },
                    "readiness": {
                        "url": f"{endpoint}/metadata",
                        "expected": {"status": "ready", "route_id": route_id},
                        "timeout_seconds": 1,
                        "poll_seconds": 0.01,
                        "request_timeout_seconds": 0.1,
                    },
                    "cleanup_timeout_seconds": 1,
                }
            )
        path = self.root / "endpoint-registry.json"
        digest = _write_json(
            path,
            {
                "schema": "amg_multitask_endpoint_registry_v1",
                "status": "pass",
                "route_order": list(EXPECTED_ROUTE_IDS),
                "routes": routes,
            },
        )
        return path, digest

    def _write_fake_git(self) -> Path:
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        script = fake_bin / "git"
        script.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" != '-C' ]; then exit 98; fi\n"
            "root=$2\n"
            "shift 2\n"
            'case "$1" in\n'
            "  status|diff|ls-files) exit 0 ;;\n"
            "  merge-base) exit 0 ;;\n"
            "  rev-parse)\n"
            f'    if [ "$root" = {str(self.outer.resolve())!r} ] && [ "$2" = \'HEAD:AgentGym\' ]; then echo {self.INNER_COMMIT}; exit 0; fi\n'
            f'    if [ "$root" = {str(self.outer.resolve())!r} ]; then echo {self.OUTER_COMMIT}; exit 0; fi\n'
            f'    if [ "$root" = {str((self.outer / "AgentGym").resolve())!r} ]; then echo {self.INNER_COMMIT}; exit 0; fi\n'
            f'    if [ "$root" = {str(self.verl.resolve())!r} ]; then echo {EXPECTED_VERL_COMMIT}; exit 0; fi\n'
            "    ;;\n"
            "esac\n"
            "exit 99\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return fake_bin


class TestMultitaskOrchestratorContract(unittest.TestCase):
    def test_reviewed_config_freezes_formal400_r38(self) -> None:
        config = load_orchestrator_config(CONFIG)

        self.assertEqual(
            EXPECTED_VERL_COMMIT, "f3ac28fe54c945e092b9630030f44d236a106a11"
        )
        self.assertEqual(config.route_order, EXPECTED_ROUTE_IDS)
        self.assertEqual(config.optimizer_updates, 400)
        self.assertEqual(config.samples_per_update, 64)
        self.assertEqual(config.total_episodes, 25_600)
        self.assertEqual(
            config.total_episodes,
            config.optimizer_updates * config.samples_per_update,
        )
        self.assertEqual((config.trainer_gpus, config.standalone_rollout_gpus), (6, 2))
        self.assertEqual(config.rollout_n, 1)
        self.assertEqual(config.learner_token_budget_profile, "default-65536-v1")
        self.assertEqual(config.actor_train_token_budget, 65_536)
        self.assertEqual(config.critic_train_token_budget, 65_536)
        self.assertEqual(config.critic_infer_token_budget, 32_768)
        self.assertEqual(config.trigger_parameter_sync_step, 1)
        self.assertFalse(config.actor_use_fused_kernels)
        self.assertFalse(config.critic_use_fused_kernels)
        self.assertFalse(config.require_exact_per_update_route_split)
        self.assertEqual(config.sampling_order, "round_robin")

    def test_reviewed_resume_config_freezes_formal200_successor_budget(self) -> None:
        config = load_orchestrator_config(CONFIG_RESUME)

        self.assertEqual(config.optimizer_updates, 200)
        self.assertEqual(config.total_episodes, 12_800)
        self.assertEqual(config.schedule_capacity_episodes, 25_600)
        self.assertEqual(config.resume_start_update, 30)
        self.assertEqual(config.resume_target_update, 200)
        self.assertEqual(config.resume_sampler_samples_yielded, 2119)
        self.assertEqual(config.invocation_optimizer_updates, 170)
        self.assertEqual(config.invocation_episodes, 10_880)

    def test_resume_command_forwards_exact_checkpoint_and_prefix(self) -> None:
        config = load_orchestrator_config(CONFIG_RESUME)
        plan = replace(
            LaunchPlan.for_test(resolve_only=False, config=config),
            resume_from_path=Path("/prefix/checkpoints/global_step_30"),
            resume_prefix_run_dir=Path("/prefix"),
        )

        command = build_generic_launch_command(
            plan,
            resolve_only=False,
            orchestrator_preflight=Path("/run/orchestrator-preflight.json"),
        )
        rendered = " ".join(command)

        self.assertIn("--resume-from-path /prefix/checkpoints/global_step_30", rendered)
        self.assertIn("--resume-prefix-run-dir /prefix", rendered)
        self.assertIn("--resume-start-update 30", rendered)
        self.assertIn("--resume-target-update 200", rendered)
        self.assertIn("--resume-sampler-samples-yielded 2119", rendered)

        with self.assertRaisesRegex(OrchestratorError, "resume launch plan is incomplete"):
            build_generic_launch_command(
                replace(plan, resume_prefix_run_dir=None),
                resolve_only=False,
                orchestrator_preflight=Path("/run/orchestrator-preflight.json"),
            )

    def test_reviewed_131k_profile_is_explicit_and_forwarded(self) -> None:
        config = load_orchestrator_config(CONFIG_131K)
        self.assertEqual(config.learner_token_budget_profile, "multitask-131072-v1")
        self.assertEqual(config.actor_train_token_budget, 131_072)
        self.assertEqual(config.critic_train_token_budget, 131_072)
        plan = LaunchPlan.for_test(resolve_only=False, config=config)
        command = build_generic_launch_command(
            plan,
            resolve_only=False,
            orchestrator_preflight=Path("/run/orchestrator-preflight.json"),
        )
        rendered = " ".join(command)
        self.assertIn("--learner-token-budget-profile multitask-131072-v1", rendered)
        self.assertIn("--actor-train-token-budget 131072", rendered)
        self.assertIn("--critic-train-token-budget 131072", rendered)

    def test_rejects_mismatched_learner_token_budget_profile(self) -> None:
        payload = CONFIG_131K.read_text(encoding="utf-8").replace(
            "actor_train_token_budget: 131072",
            "actor_train_token_budget: 65536",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(OrchestratorError, "actor train token budget"):
                load_orchestrator_config(path)

    def test_holder_acquisition_builds_complete_lifecycle_marker_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker_path = root / "yield.marker"
            holder_lease = HolderLease(
                source_path=root / "holder-lease.json",
                sha256="0" * 64,
                markers=(
                    MarkerLease(
                        name="gpu",
                        path=marker_path,
                        original_value=None,
                        original_pid=0,
                        original_start_ticks="",
                    ),
                ),
                yield_checks=(),
                restore_checks=(),
            )
            config = replace(
                LaunchPlan.for_test(resolve_only=False).config,
                holder_lock_path=root / "holder.lock",
            )
            plan = replace(
                LaunchPlan.for_test(resolve_only=False, config=config),
                run_dir=root / "run",
                holder_lease=holder_lease,
            )
            backend = LocalBackend()
            backend.parent_start_ticks = "parent-ticks"
            captured: dict[str, object] = {}

            def capture_markers(**kwargs: object) -> None:
                captured.update(kwargs)
                raise RuntimeError("stop after marker capture")

            with (
                mock.patch(
                    "agentmemorygym_verl.multitask_orchestrator.prepare_marker_transaction",
                    side_effect=capture_markers,
                ),
                self.assertRaisesRegex(RuntimeError, "stop after marker capture"),
            ):
                backend.acquire_holders(plan)

            marker = captured["markers"][0]  # type: ignore[index]
            self.assertFalse(marker["acquire_started"])
            self.assertFalse(marker["acquired"])
            self.assertFalse(marker["restore_started"])
            self.assertFalse(marker["restore_target_set"])
            self.assertIsNone(marker["restore_target"])
            self.assertFalse(marker["restored"])

    def test_endpoint_registry_binds_receipts_assets_sources_and_route_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RegistryFixture(Path(directory))
            specs, report = load_endpoint_registry(
                fixture.registry_path,
                expected_sha256=fixture.registry_sha256,
                route_registry=fixture.route_registry,
            )

            self.assertEqual(tuple(spec.route_id for spec in specs), EXPECTED_ROUTE_IDS)
            self.assertEqual(report["route_order"], list(EXPECTED_ROUTE_IDS))
            self.assertEqual(set(report["gate_receipts"]), set(EXPECTED_ROUTE_IDS))
            self.assertTrue(all(not spec.launcher_path.is_symlink() for spec in specs))

    def test_missing_or_mismatched_receipt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RegistryFixture(Path(directory))
            receipt = Path(
                fixture.registry_payload["routes"][2]["gate_receipt"]["path"]
            )
            receipt.unlink()
            with self.assertRaisesRegex(OrchestratorError, "gate receipt"):
                load_endpoint_registry(
                    fixture.registry_path,
                    expected_sha256=fixture.registry_sha256,
                    route_registry=fixture.route_registry,
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = RegistryFixture(Path(directory))
            receipt = Path(
                fixture.registry_payload["routes"][1]["gate_receipt"]["path"]
            )
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["environment"] = "webshop"
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            fixture.registry_payload["routes"][1]["gate_receipt"]["sha256"] = _sha256(
                receipt
            )
            registry_sha256 = fixture.rewrite()
            with self.assertRaisesRegex(OrchestratorError, "environment"):
                load_endpoint_registry(
                    fixture.registry_path,
                    expected_sha256=registry_sha256,
                    route_registry=fixture.route_registry,
                )

    def test_mismatched_asset_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RegistryFixture(Path(directory))
            asset = Path(fixture.registry_payload["routes"][0]["assets"][0]["path"])
            asset.write_bytes(b"substituted")
            with self.assertRaisesRegex(OrchestratorError, "asset sha256 mismatch"):
                load_endpoint_registry(
                    fixture.registry_path,
                    expected_sha256=fixture.registry_sha256,
                    route_registry=fixture.route_registry,
                )

    def test_endpoint_registry_rejects_exact_outer_or_inner_source_drift(self) -> None:
        for source_name in ("outer", "inner"):
            with (
                self.subTest(source_name=source_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = RegistryFixture(Path(directory))
                sources = fixture.registry_payload["routes"][0]["sources"]
                next(item for item in sources if item["name"] == source_name)[
                    "commit"
                ] = "f" * 40
                registry_sha256 = fixture.rewrite()
                with self.assertRaisesRegex(OrchestratorError, source_name):
                    load_endpoint_registry(
                        fixture.registry_path,
                        expected_sha256=registry_sha256,
                        route_registry=fixture.route_registry,
                    )

    def test_gate_receipt_source_commits_are_cross_bound_to_registry(self) -> None:
        cases = (
            ("outer", "shared_runtime_source_commit"),
            ("inner", "environment_source_commit"),
        )
        for source_name, receipt_field in cases:
            with (
                self.subTest(source_name=source_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = RegistryFixture(Path(directory))
                route = fixture.registry_payload["routes"][0]
                receipt_path = Path(route["gate_receipt"]["path"])
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt["source"][receipt_field] = "f" * 40
                route["gate_receipt"]["sha256"] = _write_json(receipt_path, receipt)
                registry_sha256 = fixture.rewrite()

                with self.assertRaisesRegex(
                    OrchestratorError,
                    rf"{source_name}.*receipt source commit mismatch",
                ):
                    load_endpoint_registry(
                        fixture.registry_path,
                        expected_sha256=registry_sha256,
                        route_registry=fixture.route_registry,
                    )

    def test_source_receipt_fields_are_locked_to_outer_and_inner_roles(self) -> None:
        cases = (
            ("outer", "source.environment_source_commit"),
            ("inner", "source.shared_runtime_source_commit"),
            ("inner", "source.environment_outer_source_commit"),
        )
        for source_name, receipt_field in cases:
            with (
                self.subTest(source_name=source_name, receipt_field=receipt_field),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = RegistryFixture(Path(directory))
                source = next(
                    item
                    for item in fixture.registry_payload["routes"][0]["sources"]
                    if item["name"] == source_name
                )
                source["receipt_field"] = receipt_field
                registry_sha256 = fixture.rewrite()

                with self.assertRaisesRegex(
                    OrchestratorError, rf"{source_name}.*receipt_field"
                ):
                    load_endpoint_registry(
                        fixture.registry_path,
                        expected_sha256=registry_sha256,
                        route_registry=fixture.route_registry,
                    )

    def test_missing_receipt_commit_can_use_immutable_source_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RegistryFixture(Path(directory))
            route = fixture.registry_payload["routes"][2]
            receipt_path = Path(route["gate_receipt"]["path"])
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["source"].pop("environment_outer_source_commit")
            receipt["source"].pop("shared_runtime_source_commit")
            route["gate_receipt"]["sha256"] = _write_json(receipt_path, receipt)

            source_lock = fixture.root / "literesearcher-source-lock.json"
            source_lock_sha256 = _write_json(
                source_lock,
                {"outer_commit": fixture.source_commit},
            )
            outer = next(
                source for source in route["sources"] if source["name"] == "outer"
            )
            outer.pop("receipt_field")
            outer["source_lock"] = {
                "path": str(source_lock),
                "sha256": source_lock_sha256,
                "commit_field": "outer_commit",
            }
            registry_sha256 = fixture.rewrite()

            specs, report = load_endpoint_registry(
                fixture.registry_path,
                expected_sha256=registry_sha256,
                route_registry=fixture.route_registry,
            )
            self.assertEqual(len(specs), 4)
            evidence = report["sources"]["literesearcher"]
            self.assertEqual(evidence[0]["evidence_kind"], "source_lock")
            self.assertEqual(evidence[1]["evidence_kind"], "gate_receipt")

    def test_all_immutable_successor_sources_require_explicit_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RegistryFixture(Path(directory))
            route = fixture.registry_payload["routes"][2]
            source_lock = fixture.root / "successor-source-lock.json"
            source_lock_sha256 = _write_json(
                source_lock,
                {
                    "outer_commit": fixture.source_commit,
                    "inner_commit": fixture.source_commit,
                },
            )
            for source in route["sources"]:
                source.pop("receipt_field")
                source["source_lock"] = {
                    "path": str(source_lock),
                    "sha256": source_lock_sha256,
                    "commit_field": f"{source['name']}_commit",
                }

            registry_sha256 = fixture.rewrite()
            with self.assertRaisesRegex(
                OrchestratorError, "explicitly selects the all-immutable-lock"
            ):
                load_endpoint_registry(
                    fixture.registry_path,
                    expected_sha256=registry_sha256,
                    route_registry=fixture.route_registry,
                )

            route["source_evidence_policy"] = (
                "base_gate_plus_all_immutable_source_locks_v1"
            )
            registry_sha256 = fixture.rewrite()
            specs, report = load_endpoint_registry(
                fixture.registry_path,
                expected_sha256=registry_sha256,
                route_registry=fixture.route_registry,
            )
            self.assertEqual(len(specs), 4)
            evidence = report["sources"]["literesearcher"]
            self.assertEqual(
                [item["evidence_kind"] for item in evidence],
                ["source_lock", "source_lock"],
            )

    def test_immutable_source_lock_commit_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RegistryFixture(Path(directory))
            route = fixture.registry_payload["routes"][2]
            source_lock = fixture.root / "literesearcher-source-lock.json"
            source_lock_sha256 = _write_json(
                source_lock,
                {"outer_commit": "f" * 40},
            )
            outer = next(
                source for source in route["sources"] if source["name"] == "outer"
            )
            outer.pop("receipt_field")
            outer["source_lock"] = {
                "path": str(source_lock),
                "sha256": source_lock_sha256,
                "commit_field": "outer_commit",
            }
            registry_sha256 = fixture.rewrite()

            with self.assertRaisesRegex(
                OrchestratorError, "outer source lock source commit mismatch"
            ):
                load_endpoint_registry(
                    fixture.registry_path,
                    expected_sha256=registry_sha256,
                    route_registry=fixture.route_registry,
                )

    def test_gate_receipt_requires_nonempty_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RegistryFixture(Path(directory))
            route = fixture.registry_payload["routes"][0]
            receipt_path = Path(route["gate_receipt"]["path"])
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["run_id"] = ""
            route["gate_receipt"]["sha256"] = _write_json(receipt_path, receipt)
            registry_sha256 = fixture.rewrite()

            with self.assertRaisesRegex(OrchestratorError, "run_id is empty"):
                load_endpoint_registry(
                    fixture.registry_path,
                    expected_sha256=registry_sha256,
                    route_registry=fixture.route_registry,
                )

    def test_endpoint_registry_rejects_route_order_and_non_loopback_substitution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RegistryFixture(Path(directory))
            fixture.registry_payload["routes"][0]["endpoint"] = "http://example.com:80"
            registry_sha256 = fixture.rewrite()
            with self.assertRaisesRegex(OrchestratorError, "route registry endpoint"):
                load_endpoint_registry(
                    fixture.registry_path,
                    expected_sha256=registry_sha256,
                    route_registry=fixture.route_registry,
                )

    def test_port_collision_is_rejected_before_launch(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            spec = EndpointLaunchSpec.for_test(
                route_id="webshop", endpoint=f"http://127.0.0.1:{port}"
            )
            with self.assertRaisesRegex(OrchestratorError, "port collision"):
                assert_ports_available((spec,))
        finally:
            listener.close()

    def test_partial_endpoint_startup_rolls_back_exact_started_leases(self) -> None:
        specs = tuple(
            EndpointLaunchSpec.for_test(
                route_id=route_id, endpoint=f"http://127.0.0.1:{49200 + index}"
            )
            for index, route_id in enumerate(EXPECTED_ROUTE_IDS)
        )

        class FakeSupervisor:
            def __init__(self) -> None:
                self.started: list[str] = []
                self.stopped: list[ProcessLease] = []

            def start(self, spec: EndpointLaunchSpec, **_: object) -> ProcessLease:
                self.started.append(spec.route_id)
                if spec.route_id == "literesearcher":
                    raise OrchestratorError("synthetic startup failure")
                return ProcessLease(
                    name=spec.route_id,
                    pid=100 + len(self.started),
                    start_ticks=str(900 + len(self.started)),
                    process=object(),
                    log_handle=None,
                )

            def stop(self, lease: ProcessLease, *, timeout_seconds: float) -> None:
                self.stopped.append(lease)

        supervisor = FakeSupervisor()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(OrchestratorError, "synthetic startup failure"):
                start_endpoint_processes(
                    specs,
                    run_dir=Path(directory),
                    parent_pid=1,
                    parent_start_ticks="1",
                    supervisor=supervisor,
                )
        self.assertEqual(supervisor.started, ["webshop", "swesmith", "literesearcher"])
        self.assertEqual(
            [
                (lease.name, lease.pid, lease.start_ticks)
                for lease in supervisor.stopped
            ],
            [("swesmith", 102, "902"), ("webshop", 101, "901")],
        )

    def test_stale_leader_never_signals_reused_process_group(self) -> None:
        supervisor = ExactProcessSupervisor()
        lease = ProcessLease(
            name="webshop",
            pid=4242,
            start_ticks="stale-ticks",
            process=mock.Mock(),
            log_handle=None,
        )
        with (
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator._process_identity_state",
                return_value=None,
            ),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator._active_process_group_identities",
                return_value=((4243, "other-ticks"),),
            ),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator.os.killpg"
            ) as killpg,
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator._signal_process_identity"
            ) as signal_identity,
            self.assertRaisesRegex(OrchestratorError, "leader identity.*refusing"),
        ):
            supervisor.stop(lease, timeout_seconds=0)

        killpg.assert_not_called()
        signal_identity.assert_not_called()

    def test_exact_zombie_leader_anchor_cleans_live_descendant(self) -> None:
        supervisor = ExactProcessSupervisor()
        process = mock.Mock()
        lease = ProcessLease(
            name="webshop",
            pid=4242,
            start_ticks="leader-ticks",
            process=process,
            log_handle=None,
        )
        supervisor._register(lease)
        group_snapshots = iter(
            (
                ((4243, "child-ticks"),),
                (),
                (),
                (),
            )
        )
        with (
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator._process_identity_state",
                return_value=("Z", 4242),
            ),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator._active_process_group_identities",
                side_effect=lambda _pgid: next(group_snapshots),
            ),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator._signal_process_identity"
            ) as signal_identity,
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator.os.killpg"
            ) as killpg,
        ):
            supervisor.stop(lease, timeout_seconds=1)

        signal_identity.assert_called_once_with(4243, "child-ticks", signal.SIGTERM)
        killpg.assert_not_called()
        process.wait.assert_called_once_with(timeout=5)
        self.assertEqual(supervisor.owned_leases, ())

    def test_poll_never_reaps_zombie_anchor_with_live_descendant(self) -> None:
        supervisor = ExactProcessSupervisor()
        process = mock.Mock()
        lease = ProcessLease(
            name="webshop",
            pid=4242,
            start_ticks="leader-ticks",
            process=process,
            log_handle=None,
        )
        with (
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator._process_identity_state",
                return_value=("Z", 4242),
            ),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator._active_process_group_identities",
                return_value=((4243, "child-ticks"),),
            ),
            self.assertRaisesRegex(OrchestratorError, "live descendants"),
        ):
            supervisor.poll(lease)

        process.poll.assert_not_called()
        process.wait.assert_not_called()

    def test_managed_spawn_requires_default_sigchld_disposition(self) -> None:
        supervisor = ExactProcessSupervisor()
        log_handle = mock.Mock(closed=False)
        with (
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator.signal.getsignal",
                return_value=signal.SIG_IGN,
            ),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator.subprocess.Popen",
                side_effect=AssertionError("managed child was spawned"),
            ) as popen,
            self.assertRaisesRegex(OrchestratorError, "SIGCHLD"),
        ):
            supervisor.start_command(
                name="test-command",
                command=("/bin/true",),
                working_directory=Path("/tmp"),
                environment={},
                log_handle=log_handle,
                identity_path=Path("/tmp/process-identity.json"),
                cleanup_timeout_seconds=1,
            )

        popen.assert_not_called()

    def test_tick_capture_failure_never_uses_unauthenticated_signals(self) -> None:
        supervisor = ExactProcessSupervisor()
        process = mock.Mock(pid=4242)
        process.poll.return_value = 1
        with (
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator.process_start_ticks",
                return_value=None,
            ),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator.os.killpg"
            ) as killpg,
            self.assertRaisesRegex(OrchestratorError, "failed to capture"),
        ):
            supervisor._capture_ticks(process)

        killpg.assert_not_called()
        process.kill.assert_not_called()

    def test_command_bootstrap_is_registered_before_ack_and_signal_unmask(self) -> None:
        supervisor = ExactProcessSupervisor()
        events: list[str] = []
        process = mock.Mock(pid=4242)
        log_handle = mock.Mock(closed=False)

        @contextlib.contextmanager
        def blocked_signals():
            events.append("signals-blocked")
            try:
                yield
            finally:
                events.append("signals-restored")

        original_register = supervisor._register

        def register(lease: ProcessLease) -> None:
            events.append("lease-registered")
            original_register(lease)

        with (
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator._blocked_termination_signals",
                blocked_signals,
            ),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator.subprocess.Popen",
                side_effect=lambda *_args, **_kwargs: (
                    events.append("spawned") or process
                ),
            ),
            mock.patch.object(
                supervisor,
                "_capture_ticks",
                side_effect=lambda _process: events.append("ticks-captured") or "99",
            ),
            mock.patch.object(supervisor, "_register", side_effect=register),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator._atomic_json",
                side_effect=lambda *_args, **_kwargs: events.append(
                    "identity-published"
                ),
            ),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator.os.pipe",
                return_value=(10, 11),
            ),
            mock.patch("agentmemorygym_verl.multitask_orchestrator.os.close"),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator.os.write",
                side_effect=lambda *_args: events.append("acknowledged") or 1,
            ),
            mock.patch(
                "agentmemorygym_verl.multitask_orchestrator.process_start_ticks",
                return_value="parent-ticks",
            ),
        ):
            lease = supervisor.start_command(
                name="test-command",
                command=("/bin/true",),
                working_directory=Path("/tmp"),
                environment={},
                log_handle=log_handle,
                identity_path=Path("/tmp/process-identity.json"),
                cleanup_timeout_seconds=1,
            )

        self.assertEqual(
            events,
            [
                "signals-blocked",
                "spawned",
                "ticks-captured",
                "lease-registered",
                "identity-published",
                "acknowledged",
                "signals-restored",
            ],
        )
        self.assertEqual(lease.start_ticks, "99")
        self.assertEqual(supervisor.owned_leases, (lease,))
        supervisor._release(lease)

    def test_endpoint_start_exports_derived_endpoint_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = replace(
                EndpointLaunchSpec.for_test(
                    route_id="webshop", endpoint="http://127.0.0.1:49249"
                ),
                working_directory=root,
            )
            supervisor = ExactProcessSupervisor()
            lease = ProcessLease(
                name="webshop",
                pid=101,
                start_ticks="202",
                process=object(),
                log_handle=None,
            )
            with (
                mock.patch(
                    "agentmemorygym_verl.multitask_orchestrator.process_start_ticks",
                    return_value="parent-ticks",
                ),
                mock.patch.object(
                    supervisor, "start_command", return_value=lease
                ) as start_command,
            ):
                result = supervisor.start(
                    spec,
                    run_dir=root / "run",
                    parent_pid=os.getpid(),
                    parent_start_ticks="parent-ticks",
                )

            self.assertIs(result, lease)
            environment = start_command.call_args.kwargs["environment"]
            start_command.call_args.kwargs["log_handle"].close()
            self.assertEqual(environment["AMG_MULTITASK_ENDPOINT_HOST"], "127.0.0.1")
            self.assertEqual(environment["AMG_MULTITASK_ENDPOINT_PORT"], "49249")
            self.assertEqual(
                environment["AMG_MULTITASK_ENDPOINT_URL"],
                "http://127.0.0.1:49249",
            )

    @unittest.skipUnless(Path("/proc/self/stat").is_file(), "requires Linux /proc")
    def test_failed_identity_publication_never_releases_child_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executed = root / "executed"
            launcher = root / "launcher.py"
            launcher.write_text(
                "from pathlib import Path\n"
                "import time\n"
                f"Path({str(executed)!r}).write_text('started')\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            spec = replace(
                EndpointLaunchSpec.for_test(
                    route_id="webshop", endpoint="http://127.0.0.1:49250"
                ),
                launcher_path=Path(sys.executable),
                argv=(str(launcher),),
                working_directory=root,
            )
            supervisor = ExactProcessSupervisor()

            def fail_identity(path: Path, payload: object) -> None:
                if path.name == "process-identity.json":
                    time.sleep(0.2)
                    raise OSError("synthetic identity publication failure")
                _atomic_json(path, payload)

            with (
                mock.patch(
                    "agentmemorygym_verl.multitask_orchestrator._atomic_json",
                    side_effect=fail_identity,
                ),
                self.assertRaisesRegex(OSError, "identity publication"),
            ):
                supervisor.start(
                    spec,
                    run_dir=root / "run",
                    parent_pid=os.getpid(),
                    parent_start_ticks=str(process_start_ticks(os.getpid())),
                )

            self.assertFalse(executed.exists())
            self.assertEqual(supervisor.owned_leases, ())

    @unittest.skipUnless(Path("/proc/self/stat").is_file(), "requires Linux /proc")
    def test_bootstrap_leader_outlives_entrypoint_and_cleans_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_pid_path = root / "child.pid"
            launcher = root / "launcher.sh"
            launcher.write_text(
                "#!/bin/sh\n"
                "sleep 60 &\n"
                f"printf '%s\\n' \"$!\" > {str(child_pid_path)!r}\n"
                "exit 0\n",
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            spec = replace(
                EndpointLaunchSpec.for_test(
                    route_id="webshop", endpoint="http://127.0.0.1:49251"
                ),
                launcher_path=launcher,
                working_directory=root,
            )
            supervisor = ExactProcessSupervisor()
            lease = supervisor.start(
                spec,
                run_dir=root / "run",
                parent_pid=os.getpid(),
                parent_start_ticks=str(process_start_ticks(os.getpid())),
            )
            try:
                deadline = time.monotonic() + 5
                while not child_pid_path.is_file() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(child_pid_path.is_file())
                child_pid = int(child_pid_path.read_text(encoding="utf-8"))
                self.assertTrue(supervisor.alive(lease))
                supervisor.stop(lease, timeout_seconds=1)
                deadline = time.monotonic() + 2
                while (
                    process_start_ticks(child_pid) is not None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                self.assertIsNone(process_start_ticks(child_pid))
                self.assertEqual(supervisor.owned_leases, ())
            finally:
                supervisor.stop_all()

    def test_cli_immutable_paths_reject_symlinks_before_resolution(self) -> None:
        config = load_orchestrator_config(CONFIG)
        cases = (
            ("config", "config"),
            ("route_registry", "route registry"),
            ("schedule", "multitask schedule"),
            ("multitask_source_lock", "multitask source lock"),
            (
                "multitask_schedule_certificate",
                "multitask schedule certificate",
            ),
            ("endpoint_registry", "endpoint registry"),
            ("holder_lease", "holder lease"),
        )
        for attribute, label in cases:
            with (
                self.subTest(attribute=attribute),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                outer = root / "outer"
                generic = outer / "async_plugins/scripts/launch_amg_fully_async.sh"
                generic.parent.mkdir(parents=True)
                generic.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                generic.chmod(0o755)
                verl = root / "verl"
                verl.mkdir()
                immutable = {}
                for name in (
                    "config",
                    "route_registry",
                    "schedule",
                    "multitask_source_lock",
                    "multitask_schedule_certificate",
                    "endpoint_registry",
                    "holder_lease",
                ):
                    path = root / f"{name}.json"
                    path.write_text("{}\n", encoding="utf-8")
                    immutable[name] = path
                target = immutable[attribute]
                symlink = root / f"{attribute}.link"
                symlink.symlink_to(target)
                immutable[attribute] = symlink
                resolve_only = attribute != "holder_lease"
                args = argparse.Namespace(
                    config=immutable["config"],
                    outer_root=outer,
                    verl_root=verl,
                    schedule=immutable["schedule"],
                    route_registry=immutable["route_registry"],
                    route_registry_sha256="1" * 64,
                    multitask_source_lock=immutable["multitask_source_lock"],
                    multitask_schedule_certificate=immutable[
                        "multitask_schedule_certificate"
                    ],
                    endpoint_registry=immutable["endpoint_registry"],
                    endpoint_registry_sha256="2" * 64,
                    holder_lease=(
                        immutable["holder_lease"] if not resolve_only else None
                    ),
                    holder_lease_sha256=("3" * 64 if not resolve_only else None),
                    run_dir=root / "run",
                    experiment_name="symlink-negative",
                    resolve_only=resolve_only,
                )
                route_registry = mock.Mock(
                    sha256="1" * 64, route_ids=EXPECTED_ROUTE_IDS
                )
                identity = {
                    "budget_contract": {
                        "optimizer_updates": 400,
                        "samples_per_update": 64,
                        "episodes": 25_600,
                        "trigger_parameter_sync_step": 1,
                    }
                }
                with (
                    mock.patch(
                        "agentmemorygym_verl.multitask_orchestrator.load_orchestrator_config",
                        return_value=config,
                    ),
                    mock.patch(
                        "agentmemorygym_verl.multitask_orchestrator.load_route_registry",
                        return_value=route_registry,
                    ),
                    mock.patch(
                        "agentmemorygym_verl.multitask_orchestrator.inspect_schedule",
                        return_value={"sha256": "4" * 64, "count": 25_600},
                    ),
                    mock.patch(
                        "agentmemorygym_verl.multitask_orchestrator._load_multitask_identity",
                        return_value=identity,
                    ),
                    mock.patch(
                        "agentmemorygym_verl.multitask_orchestrator.load_endpoint_registry",
                        return_value=((), {}),
                    ),
                    mock.patch(
                        "agentmemorygym_verl.multitask_orchestrator.load_holder_lease",
                        return_value=mock.Mock(),
                    ),
                    mock.patch(
                        "agentmemorygym_verl.multitask_orchestrator._source_clean"
                    ),
                    self.assertRaisesRegex(OrchestratorError, label),
                ):
                    build_launch_plan(args)

    def test_resolve_only_never_acquires_holders_or_spawns_endpoints_or_trainer(
        self,
    ) -> None:
        calls: list[str] = []

        class FakeBackend:
            def resolve(self, _plan: LaunchPlan) -> None:
                calls.append("resolve")

            def acquire_holders(self, _plan: LaunchPlan) -> object:
                calls.append("holders")
                return object()

            def start_endpoints(self, _plan: LaunchPlan) -> object:
                calls.append("endpoints")
                return object()

            def start_trainer(self, _plan: LaunchPlan) -> object:
                calls.append("trainer")
                return object()

        plan = LaunchPlan.for_test(resolve_only=True)
        self.assertEqual(execute_launch_plan(plan, backend=FakeBackend()), 0)
        self.assertEqual(calls, ["resolve"])

    def test_orchestrator_boundary_resolve_only_invokes_only_generic_stub(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "generic-invocations.log"
            launcher = root / "generic-launcher.sh"
            launcher.write_text(
                f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{trace}'\nexit 0\n",
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            plan = replace(
                LaunchPlan.for_test(resolve_only=True),
                outer_root=root,
                generic_launcher=launcher,
                run_dir=root / "run",
            )

            with (
                mock.patch.object(
                    LocalBackend,
                    "acquire_holders",
                    side_effect=AssertionError("resolve-only acquired holders"),
                ),
                mock.patch.object(
                    LocalBackend,
                    "start_endpoints",
                    side_effect=AssertionError("resolve-only started endpoints"),
                ),
                mock.patch.object(
                    LocalBackend,
                    "start_trainer_with_holder",
                    side_effect=AssertionError("resolve-only started trainer"),
                ),
            ):
                self.assertEqual(_execute_local(plan), 0)

            invocations = trace.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(invocations), 1)
            self.assertIn("--resolve-only --skip-runtime-preflight", invocations[0])
            receipt = json.loads(
                (plan.run_dir / "resolve-only-receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["endpoints_spawned"], 0)
            self.assertFalse(receipt["trainer_spawned"])

    @unittest.skipUnless(shutil.which("jq"), "production shell requires jq")
    def test_production_one_click_chain_resolves_without_runtime_spawns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProductionResolveFixture(Path(directory))
            command = [
                str(
                    fixture.outer
                    / "async_plugins/scripts/launch_amg_multitask_fully_async.sh"
                ),
                "--verl-root",
                str(fixture.verl),
                "--schedule",
                str(fixture.schedule),
                "--route-registry",
                str(fixture.route_registry),
                "--route-registry-sha256",
                fixture.route_registry_sha256,
                "--multitask-source-lock",
                str(fixture.source_lock),
                "--multitask-schedule-certificate",
                str(fixture.schedule_certificate),
                "--endpoint-registry",
                str(fixture.endpoint_registry),
                "--endpoint-registry-sha256",
                fixture.endpoint_registry_sha256,
                "--run-dir",
                str(fixture.run_dir),
                "--experiment-name",
                "production-chain-resolve-only",
                "--resolve-only",
            ]
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            environment["PATH"] = os.pathsep.join(
                (str(fixture.fake_bin), environment.get("PATH", ""))
            )
            completed = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                env=environment,
                timeout=60,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(fixture.resolve_sentinel.is_file())
            self.assertFalse(fixture.endpoint_sentinel.exists())
            self.assertFalse(fixture.trainer_sentinel.exists())
            orchestrator_receipt = json.loads(
                (fixture.run_dir / "resolve-only-receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(orchestrator_receipt["endpoints_spawned"], 0)
            self.assertFalse(orchestrator_receipt["trainer_spawned"])
            generic_receipt = json.loads(
                (fixture.run_dir / "resolve-only/launch-receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                generic_receipt["entrypoint"],
                "verl.experimental.fully_async_policy.fully_async_main",
            )
            self.assertEqual(generic_receipt["budget"]["optimizer_updates"], 400)
            self.assertEqual(generic_receipt["budget"]["episodes"], 25_600)
            self.assertEqual(
                generic_receipt["runtime_artifacts"]["trainer_log"],
                str((fixture.run_dir / "resolve-only" / "trainer.log").resolve()),
            )

    def test_wait_trainer_polls_through_exact_supervisor(self) -> None:
        backend = LocalBackend()
        process = mock.Mock()
        process.poll.side_effect = AssertionError("direct Popen.poll reaped anchor")
        trainer = ProcessLease(
            name="trainer",
            pid=101,
            start_ticks="201",
            process=process,
            log_handle=None,
        )
        endpoint = ProcessLease(
            name="webshop",
            pid=102,
            start_ticks="202",
            process=mock.Mock(),
            log_handle=None,
        )
        watcher = ProcessLease(
            name="holder-watcher",
            pid=103,
            start_ticks="203",
            process=mock.Mock(),
            log_handle=None,
        )
        holder = mock.Mock(watcher=watcher)
        backend.supervisor = mock.Mock()
        backend.supervisor.poll.side_effect = (None, 0)
        backend.supervisor.alive.return_value = True

        with mock.patch("agentmemorygym_verl.multitask_orchestrator.time.sleep"):
            return_code = backend.wait_trainer(
                LaunchPlan.for_test(resolve_only=False),
                trainer,
                (endpoint,),
                holder,
            )

        self.assertEqual(return_code, 0)
        self.assertEqual(backend.supervisor.poll.call_count, 2)
        process.poll.assert_not_called()

    def test_holder_restore_waits_through_exact_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "watcher-exit.json"
            _write_json(receipt, {"status": "pass"})
            process = mock.Mock()
            process.wait.side_effect = AssertionError("direct Popen.wait reaped anchor")
            watcher = ProcessLease(
                name="holder-watcher",
                pid=103,
                start_ticks="203",
                process=process,
                log_handle=None,
            )
            holder = mock.Mock(
                state_path=root / "state.json",
                watcher_receipt=receipt,
                watcher=watcher,
            )
            plan = replace(LaunchPlan.for_test(resolve_only=False), holder_lease=None)
            backend = LocalBackend()
            backend.supervisor = mock.Mock()
            backend.supervisor.wait.return_value = 0

            with mock.patch(
                "agentmemorygym_verl.multitask_orchestrator.restore_marker_transaction"
            ):
                backend.restore_holders(plan, holder)

            backend.supervisor.wait.assert_called_once_with(watcher, timeout_seconds=15)
            backend.supervisor.stop.assert_called_once_with(watcher, timeout_seconds=5)
            process.wait.assert_not_called()

    def test_holder_acquisition_rollback_waits_through_exact_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = mock.Mock()
            process.wait.side_effect = AssertionError("direct Popen.wait reaped anchor")
            watcher = ProcessLease(
                name="holder-watcher",
                pid=103,
                start_ticks="203",
                process=process,
                log_handle=None,
            )
            supervisor = mock.Mock()

            def start_watcher(**kwargs: object) -> ProcessLease:
                identity_path = Path(str(kwargs["identity_path"]))
                _write_json(
                    identity_path.parent / "watcher-ready.json",
                    {"status": "ready", "signal_handlers_installed": True},
                )
                return watcher

            supervisor.start_command.side_effect = start_watcher
            supervisor.wait.return_value = 0
            plan = replace(
                LaunchPlan.for_test(resolve_only=False),
                run_dir=root / "run",
                holder_lease=mock.Mock(markers=(), yield_checks=()),
            )
            backend = LocalBackend()
            backend.parent_start_ticks = "parent-ticks"
            backend.supervisor = supervisor

            with (
                mock.patch(
                    "agentmemorygym_verl.multitask_orchestrator.prepare_marker_transaction"
                ),
                mock.patch(
                    "agentmemorygym_verl.multitask_orchestrator.acquire_marker_transaction",
                    side_effect=RuntimeError("synthetic acquisition failure"),
                ),
                mock.patch(
                    "agentmemorygym_verl.multitask_orchestrator.restore_marker_transaction"
                ),
                self.assertRaisesRegex(RuntimeError, "synthetic acquisition failure"),
            ):
                backend.acquire_holders(plan)

            supervisor.wait.assert_called_once_with(watcher, timeout_seconds=15)
            supervisor.stop.assert_called_once_with(watcher, timeout_seconds=5)
            process.wait.assert_not_called()

    def test_execute_local_cleans_backend_owned_resources_after_interruption(
        self,
    ) -> None:
        stages = ("holder", "endpoints", "trainer")
        for stage in stages:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                events: list[str] = []

                class InterruptedBackend:
                    def __init__(self) -> None:
                        self.holder_handle = None
                        self.endpoint_leases: tuple[ProcessLease, ...] = ()
                        self.trainer = None

                    def resolve(self, plan: LaunchPlan) -> None:
                        plan.run_dir.mkdir(parents=True)

                    def acquire_holders(self, _plan: LaunchPlan) -> object:
                        self.holder_handle = object()
                        if stage == "holder":
                            raise RuntimeError("interrupted after holder acquisition")
                        return self.holder_handle

                    def start_endpoints(
                        self, _plan: LaunchPlan
                    ) -> tuple[ProcessLease, ...]:
                        self.endpoint_leases = (
                            ProcessLease(
                                name="webshop",
                                pid=101,
                                start_ticks="1",
                                process=object(),
                                log_handle=None,
                            ),
                        )
                        if stage == "endpoints":
                            raise RuntimeError("interrupted after endpoint startup")
                        return self.endpoint_leases

                    def start_trainer_with_holder(
                        self, _plan: LaunchPlan, _holder: object
                    ) -> ProcessLease:
                        self.trainer = ProcessLease(
                            name="trainer",
                            pid=102,
                            start_ticks="2",
                            process=object(),
                            log_handle=None,
                        )
                        raise RuntimeError("interrupted after trainer startup")

                    def stop_trainer(
                        self, _plan: LaunchPlan, _trainer: ProcessLease
                    ) -> None:
                        events.append("trainer")
                        self.trainer = None

                    def stop_endpoints(
                        self,
                        _plan: LaunchPlan,
                        _endpoints: tuple[ProcessLease, ...],
                    ) -> None:
                        events.append("endpoints")
                        self.endpoint_leases = ()

                    def restore_holders(
                        self, _plan: LaunchPlan, _holder: object
                    ) -> None:
                        events.append("holder")
                        self.holder_handle = None

                plan = replace(
                    LaunchPlan.for_test(resolve_only=False),
                    run_dir=Path(directory) / "run",
                )
                backend = InterruptedBackend()
                with (
                    mock.patch(
                        "agentmemorygym_verl.multitask_orchestrator.LocalBackend",
                        return_value=backend,
                    ),
                    self.assertRaisesRegex(RuntimeError, "interrupted"),
                ):
                    _execute_local(plan)

                expected = {
                    "holder": ["holder"],
                    "endpoints": ["endpoints", "holder"],
                    "trainer": ["trainer", "endpoints", "holder"],
                }[stage]
                self.assertEqual(events, expected)

    def test_generic_command_has_only_reviewed_multitask_inputs(self) -> None:
        config = load_orchestrator_config(CONFIG)
        plan = LaunchPlan.for_test(resolve_only=False, config=config)
        command = build_generic_launch_command(
            plan,
            resolve_only=False,
            orchestrator_preflight=Path("/run/orchestrator-preflight.json"),
        )
        rendered = " ".join(command)
        self.assertIn("--trainer-gpus 6", rendered)
        self.assertIn("--standalone-rollout-gpus 2", rendered)
        self.assertIn("--multitask-orchestrator-preflight", rendered)
        self.assertNotIn("--actor-use-fused-kernels", command)
        self.assertNotIn("--critic-use-fused-kernels", command)
        self.assertNotIn("--env-addr", command)

    def test_one_click_shell_uses_locked_runtime_and_thin_orchestrator(self) -> None:
        script = ROOT / "async_plugins/scripts/launch_amg_multitask_fully_async.sh"
        source = script.read_text(encoding="utf-8")
        self.assertTrue(script.stat().st_mode & 0o111)
        self.assertIn(".training_runtime.python", source)
        self.assertIn("-m agentmemorygym_verl.multitask_orchestrator", source)
        self.assertIn("PYTHONPATH is an identity conflict", source)
        self.assertNotIn("fully_async_main", source)
        self.assertNotIn("start_webshop", source)


if __name__ == "__main__":
    unittest.main()
