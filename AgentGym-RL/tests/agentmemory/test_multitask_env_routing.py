from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "verl/utils/agentgym/client.py"

FAKE_AGENTENV = types.ModuleType("agentenv")
FAKE_ENVS = types.ModuleType("agentenv.envs")
for class_name in (
    "AgentMemoryEnvClient",
    "AcademiaEnvClient",
    "AlfWorldEnvClient",
    "BabyAIEnvClient",
    "MazeEnvClient",
    "MovieEnvClient",
    "SciworldEnvClient",
    "SheetEnvClient",
    "SqlGymEnvClient",
    "TextCraftEnvClient",
    "TodoEnvClient",
    "WeatherEnvClient",
    "WebarenaEnvClient",
    "WebshopEnvClient",
    "WordleEnvClient",
    "SearchQAEnvClient",
    "SwesmithEnvClient",
):
    setattr(FAKE_ENVS, class_name, type(class_name, (), {}))

ORIGINAL_AGENTENV = sys.modules.get("agentenv")
ORIGINAL_ENVS = sys.modules.get("agentenv.envs")
sys.modules["agentenv"] = FAKE_AGENTENV
sys.modules["agentenv.envs"] = FAKE_ENVS
try:
    SPEC = importlib.util.spec_from_file_location(
        "agentmemory_multitask_client_for_test",
        MODULE_PATH,
    )
    assert SPEC is not None and SPEC.loader is not None
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
finally:
    if ORIGINAL_AGENTENV is None:
        sys.modules.pop("agentenv", None)
    else:
        sys.modules["agentenv"] = ORIGINAL_AGENTENV
    if ORIGINAL_ENVS is None:
        sys.modules.pop("agentenv.envs", None)
    else:
        sys.modules["agentenv.envs"] = ORIGINAL_ENVS

configured_multitask_env_addrs = MODULE.configured_multitask_env_addrs
configured_multitask_task_names = MODULE.configured_multitask_task_names
env_addr_for_surface_slot = MODULE.env_addr_for_surface_slot
task_name_for_env_addr = MODULE.task_name_for_env_addr


class MultitaskEnvRoutingTests(unittest.TestCase):
    def test_single_surface_route_is_unchanged(self) -> None:
        args = SimpleNamespace(env_addr="http://single:5000")
        self.assertEqual(configured_multitask_env_addrs(args), ())
        self.assertEqual(env_addr_for_surface_slot(args), "http://single:5000")
        self.assertEqual(
            env_addr_for_surface_slot(args, 0),
            "http://single:5000",
        )
        with self.assertRaisesRegex(ValueError, "nonzero surface slot"):
            env_addr_for_surface_slot(args, 1)

    def test_multitask_route_selects_the_exact_ordered_endpoint(self) -> None:
        endpoints = [f"http://127.0.0.1:{65000 + slot}/" for slot in range(8)]
        args = SimpleNamespace(
            env_addr="http://bootstrap:5000",
            multitask_env_addrs=endpoints,
        )
        normalized = tuple(endpoint.rstrip("/") for endpoint in endpoints)
        self.assertEqual(configured_multitask_env_addrs(args), normalized)
        for slot, endpoint in enumerate(normalized):
            with self.subTest(slot=slot):
                self.assertEqual(
                    env_addr_for_surface_slot(args, slot),
                    endpoint,
                )

    def test_multitask_endpoint_and_slot_errors_fail_closed(self) -> None:
        invalid_endpoint_sets = (
            "http://127.0.0.1:65000",
            [],
            ["127.0.0.1:65000"],
            ["ftp://127.0.0.1:65000"],
            ["http://127.0.0.1:65000", "http://127.0.0.1:65000/"],
        )
        for endpoints in invalid_endpoint_sets:
            with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                configured_multitask_env_addrs(
                    SimpleNamespace(multitask_env_addrs=endpoints)
                )

        args = SimpleNamespace(
            env_addr="http://bootstrap:5000",
            multitask_env_addrs=["http://a:1", "http://b:2"],
        )
        for slot in (-1, 2, 1.0, "1", True):
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                env_addr_for_surface_slot(args, slot)

    def test_cross_family_routes_select_the_existing_client_family(self) -> None:
        endpoints = [f"http://127.0.0.1:{65000 + slot}" for slot in range(9)]
        task_names = ["agentmemory"] * 8 + ["swesmith"]
        args = SimpleNamespace(
            task_name="agentmemory",
            env_addr=endpoints[0],
            multitask_env_addrs=endpoints,
            multitask_task_names=task_names,
        )
        self.assertEqual(
            configured_multitask_task_names(args),
            tuple(task_names),
        )
        for slot, endpoint in enumerate(endpoints):
            with self.subTest(slot=slot):
                self.assertEqual(
                    task_name_for_env_addr(args, endpoint),
                    task_names[slot],
                )
        self.assertEqual(task_name_for_env_addr(args), "agentmemory")

    def test_cross_family_task_name_contract_fails_closed(self) -> None:
        endpoints = ["http://a:1", "http://b:2"]
        invalid_task_names = (
            "agentmemory",
            [],
            ["agentmemory"],
            ["agentmemory", "unsupported"],
        )
        for task_names in invalid_task_names:
            with self.subTest(task_names=task_names), self.assertRaises(ValueError):
                configured_multitask_task_names(
                    SimpleNamespace(
                        multitask_env_addrs=endpoints,
                        multitask_task_names=task_names,
                    )
                )

        args = SimpleNamespace(
            task_name="agentmemory",
            env_addr=endpoints[0],
            multitask_env_addrs=endpoints,
            multitask_task_names=["agentmemory", "swesmith"],
        )
        with self.assertRaisesRegex(ValueError, "not present"):
            task_name_for_env_addr(args, "http://missing:3")


if __name__ == "__main__":
    unittest.main()
