from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from agentmemorygym_verl.routes import (
    RouteRegistry,
    canonical_policy_framing_sha256,
    load_route_registry,
    route_registry_from_agentgym_config,
)


ROUTE_IDS = ("webshop", "swesmith", "literesearcher", "openmle_fast")


def _framing(route_id: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": f"system:{route_id}"},
        {"role": "assistant", "content": "Understood."},
    ]


def _registry_payload() -> dict:
    ports = {
        "webshop": 65101,
        "swesmith": 65102,
        "literesearcher": 65103,
        "openmle_fast": 65104,
    }
    task_names = {
        "webshop": "webshop",
        "swesmith": "swesmith",
        "literesearcher": "agentmemory",
        "openmle_fast": "openmle_fast",
    }
    return {
        "schema": "amg_route_registry_v1",
        "agent_name": "amg_task_neutral_async",
        "routes": [
            {
                "route_id": route_id,
                "max_rounds": 30,
                "max_observation_tokens": 8192,
                "policy_framing_sha256": canonical_policy_framing_sha256(
                    _framing(route_id)
                ),
                "route_attestation_sha256": (str(index + 1) * 64)[:64],
                "client": {
                    "task_name": task_names[route_id],
                    "env_addr": f"http://127.0.0.1:{ports[route_id]}",
                    "timeout": 240,
                    "max_retries": 2,
                },
            }
            for index, route_id in enumerate(ROUTE_IDS)
        ],
    }


class RouteRegistryTests(unittest.TestCase):
    def _write(self, directory: str, payload: dict) -> tuple[Path, str]:
        path = Path(directory) / "routes.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_loads_exact_four_route_registry_and_returns_immutable_specs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self._write(directory, _registry_payload())
            registry = load_route_registry(
                path,
                expected_sha256=digest,
                expected_route_ids=ROUTE_IDS,
            )

            self.assertIsInstance(registry, RouteRegistry)
            self.assertEqual(registry.sha256, digest)
            self.assertEqual(registry.route_ids, ROUTE_IDS)
            self.assertEqual(
                registry.resolve("literesearcher").client_config["task_name"],
                "agentmemory",
            )
            with self.assertRaises(TypeError):
                registry.resolve("webshop").client_config["env_addr"] = "bad"

    def test_rejects_wrong_digest_duplicate_ids_and_non_loopback_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._write(directory, _registry_payload())
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                load_route_registry(path, expected_sha256="0" * 64)

            duplicate = _registry_payload()
            duplicate["routes"][1]["route_id"] = "webshop"
            path, digest = self._write(directory, duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate route_id"):
                load_route_registry(path, expected_sha256=digest)

            public = _registry_payload()
            public["routes"][0]["client"]["env_addr"] = "https://example.com:443"
            path, digest = self._write(directory, public)
            with self.assertRaisesRegex(ValueError, "loopback"):
                load_route_registry(path, expected_sha256=digest)

    def test_rejects_conflicting_route_attestation_locations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = _registry_payload()
            payload["routes"][0]["client"]["route_attestation_sha256"] = "f" * 64
            path, digest = self._write(directory, payload)
            with self.assertRaisesRegex(ValueError, "route attestation drift"):
                load_route_registry(path, expected_sha256=digest)

    def test_file_registry_requires_canonical_four_route_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = _registry_payload()
            payload["routes"] = payload["routes"][:-1]
            path, digest = self._write(directory, payload)
            with self.assertRaisesRegex(ValueError, "canonical route order"):
                load_route_registry(path, expected_sha256=digest)

    def test_rejects_symlink_and_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, digest = self._write(directory, _registry_payload())
            link = root / "routes-link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                load_route_registry(link, expected_sha256=digest)

            payload = _registry_payload()
            del payload["routes"][0]["policy_framing_sha256"]
            path, digest = self._write(directory, payload)
            with self.assertRaisesRegex(ValueError, "policy_framing_sha256"):
                load_route_registry(path, expected_sha256=digest)

    def test_resolve_row_requires_matching_top_level_and_extra_route_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self._write(directory, _registry_payload())
            registry = load_route_registry(path, expected_sha256=digest)

            self.assertEqual(
                registry.resolve_row(
                    {
                        "route_id": "swesmith",
                        "extra_info": {"route_id": "swesmith"},
                    }
                ).route_id,
                "swesmith",
            )
            with self.assertRaisesRegex(ValueError, "route_id drift"):
                registry.resolve_row(
                    {
                        "route_id": "swesmith",
                        "extra_info": {"route_id": "webshop"},
                    }
                )
            with self.assertRaisesRegex(ValueError, "unknown AMG route_id"):
                registry.resolve_row({"route_id": "missing"})

    def test_single_environment_config_is_adapted_without_registry_file(self) -> None:
        config = {
            "task_name": "webshop",
            "env_addr": "http://127.0.0.1:65101",
            "max_rounds": 30,
            "max_observation_tokens": 8192,
            "timeout": 240,
            "max_retries": 2,
        }
        registry = route_registry_from_agentgym_config(config)

        self.assertEqual(registry.route_ids, ("webshop",))
        self.assertEqual(registry.agent_name, "amg_task_neutral_async")
        self.assertEqual(registry.resolve_row({}).route_id, "webshop")
        self.assertEqual(registry.resolve("webshop").client_config["env_addr"], config["env_addr"])

    def test_registry_config_requires_path_and_digest_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "route_registry_path.*route_registry_sha256"):
            route_registry_from_agentgym_config(
                {"route_registry_path": "/tmp/routes.json"}
            )


if __name__ == "__main__":
    unittest.main()
