from __future__ import annotations

import unittest
from copy import deepcopy
from types import MappingProxyType, SimpleNamespace

from agentmemorygym_verl.dataset import AMGTrajectoryDataset
from agentmemorygym_verl.routes import RouteRegistry, RouteSpec


class TestAMGTrajectoryDataset(unittest.TestCase):
    def _dataset(self, rows):
        dataset = object.__new__(AMGTrajectoryDataset)
        dataset.dataframe = deepcopy(rows)
        framing = [
            {"role": "system", "content": "Use ordinary shell and filesystem actions."}
        ]
        registry = RouteRegistry(
            routes=(
                RouteSpec(
                    route_id="openmle_fast",
                    max_rounds=30,
                    max_observation_tokens=8192,
                    policy_framing_sha256=None,
                    route_attestation_sha256=None,
                    client_config=MappingProxyType(
                        {
                            "task_name": "openmle_fast",
                            "env_addr": "http://127.0.0.1:65524",
                        }
                    ),
                ),
            ),
            sha256=None,
            source_path=None,
        )
        dataset._route_registry = registry
        dataset._policy_framing_by_route = {"openmle_fast": framing}
        dataset.config = SimpleNamespace(
            agentgym=SimpleNamespace(task_name="openmle_fast")
        )
        return dataset

    def test_preserves_frozen_schedule_identity_and_order_without_mutating_source(self):
        rows = [
            {"item_id": "task-b", "data_idx": 9, "extra_info": {"index": 9}},
            {"item_id": "task-a", "data_idx": 3, "extra_info": {"index": 3}},
        ]
        dataset = self._dataset(rows)
        first = dataset[0]
        second = dataset[1]

        self.assertEqual(
            (first["item_id"], first["data_idx"], first["index"]), ("task-b", 9, 9)
        )
        self.assertEqual(
            (second["item_id"], second["data_idx"], second["index"]), ("task-a", 3, 3)
        )
        self.assertEqual(dataset.dataframe, rows)
        first["raw_prompt"][0]["content"] = "mutated"
        self.assertNotEqual(
            first["raw_prompt"], dataset._policy_framing_by_route["openmle_fast"]
        )

    def test_rejects_nonintegral_data_idx(self):
        dataset = self._dataset([{"item_id": "task", "data_idx": 1.5}])
        with self.assertRaisesRegex(ValueError, "data_idx must be an integer"):
            dataset[0]

    def test_rejects_top_level_and_nested_global_index_drift(self):
        dataset = self._dataset(
            [
                {
                    "item_id": "task",
                    "index": 7,
                    "data_idx": 7,
                    "extra_info": {"index": 8},
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "global index drift"):
            dataset[0]

    def test_routes_each_row_to_its_exact_framing_and_shared_agent_loop(self):
        rows = [
            {
                "item_id": "webshop:item-0",
                "route_id": "webshop",
                "index": 0,
                "data_idx": 3,
                "extra_info": {
                    "index": 0,
                    "schedule_position": 0,
                    "route_id": "webshop",
                },
            },
            {
                "item_id": "swesmith:item-0",
                "route_id": "swesmith",
                "index": 1,
                "data_idx": 11,
                "extra_info": {
                    "index": 1,
                    "schedule_position": 1,
                    "route_id": "swesmith",
                },
            },
        ]
        dataset = object.__new__(AMGTrajectoryDataset)
        dataset.dataframe = deepcopy(rows)
        dataset._route_registry = RouteRegistry(
            routes=(
                RouteSpec(
                    route_id="webshop",
                    max_rounds=30,
                    max_observation_tokens=8192,
                    policy_framing_sha256=None,
                    route_attestation_sha256=None,
                    client_config=MappingProxyType(
                        {
                            "task_name": "webshop",
                            "env_addr": "http://127.0.0.1:65101",
                        }
                    ),
                ),
                RouteSpec(
                    route_id="swesmith",
                    max_rounds=30,
                    max_observation_tokens=8192,
                    policy_framing_sha256=None,
                    route_attestation_sha256=None,
                    client_config=MappingProxyType(
                        {
                            "task_name": "swesmith",
                            "env_addr": "http://127.0.0.1:65102",
                        }
                    ),
                ),
            ),
            sha256="a" * 64,
            source_path=None,
        )
        dataset._policy_framing_by_route = {
            "webshop": [{"role": "system", "content": "shop"}],
            "swesmith": [{"role": "system", "content": "code"}],
        }
        dataset.config = SimpleNamespace(agentgym=SimpleNamespace())

        webshop = dataset[0]
        swesmith = dataset[1]

        self.assertEqual(webshop["raw_prompt"][0]["content"], "shop")
        self.assertEqual(swesmith["raw_prompt"][0]["content"], "code")
        self.assertEqual(webshop["data_idx"], 3)
        self.assertEqual(swesmith["data_idx"], 11)
        self.assertEqual(webshop["index"], 0)
        self.assertEqual(swesmith["index"], 1)
        self.assertEqual(webshop["extra_info"]["index"], 0)
        self.assertEqual(swesmith["extra_info"]["index"], 1)
        self.assertEqual(webshop["route_id"], "webshop")
        self.assertEqual(swesmith["route_id"], "swesmith")
        self.assertEqual(webshop["extra_info"]["route_id"], "webshop")
        self.assertEqual(swesmith["extra_info"]["route_id"], "swesmith")
        self.assertEqual(webshop["agent_name"], "amg_task_neutral_async")
        self.assertEqual(swesmith["agent_name"], "amg_task_neutral_async")
        self.assertEqual(webshop["data_source"], "webshop")
        self.assertEqual(swesmith["data_source"], "swesmith")

        webshop["raw_prompt"][0]["content"] = "mutated"
        self.assertEqual(
            dataset._policy_framing_by_route["webshop"][0]["content"], "shop"
        )

    def test_multienvironment_row_requires_explicit_global_index(self):
        rows = [
            {
                "item_id": "webshop:item-0",
                "route_id": "webshop",
                "data_idx": 3,
                "extra_info": {"route_id": "webshop"},
            }
        ]
        dataset = object.__new__(AMGTrajectoryDataset)
        dataset.dataframe = deepcopy(rows)
        dataset._route_registry = RouteRegistry(
            routes=(
                RouteSpec(
                    route_id="webshop",
                    max_rounds=30,
                    max_observation_tokens=8192,
                    policy_framing_sha256=None,
                    route_attestation_sha256=None,
                    client_config=MappingProxyType(
                        {
                            "task_name": "webshop",
                            "env_addr": "http://127.0.0.1:65101",
                        }
                    ),
                ),
                RouteSpec(
                    route_id="swesmith",
                    max_rounds=30,
                    max_observation_tokens=8192,
                    policy_framing_sha256=None,
                    route_attestation_sha256=None,
                    client_config=MappingProxyType(
                        {
                            "task_name": "swesmith",
                            "env_addr": "http://127.0.0.1:65102",
                        }
                    ),
                ),
            ),
            sha256="a" * 64,
            source_path=None,
        )
        dataset._policy_framing_by_route = {
            "webshop": [{"role": "system", "content": "shop"}],
            "swesmith": [{"role": "system", "content": "code"}],
        }
        dataset.config = SimpleNamespace(agentgym=SimpleNamespace())

        with self.assertRaisesRegex(ValueError, "global index"):
            dataset[0]

    def test_rejects_unknown_or_drifting_route_identity(self):
        dataset = self._dataset(
            [
                {
                    "item_id": "task",
                    "route_id": "missing",
                    "data_idx": 0,
                    "extra_info": {"index": 0, "route_id": "missing"},
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "unknown AMG route_id"):
            dataset[0]

        dataset.dataframe[0]["route_id"] = "openmle_fast"
        with self.assertRaisesRegex(ValueError, "route_id drift"):
            dataset[0]

    def test_rejects_row_that_selects_a_different_agent_loop(self):
        dataset = self._dataset(
            [
                {
                    "item_id": "task",
                    "data_idx": 0,
                    "agent_name": "domain_specific_loop",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "agent_name"):
            dataset[0]

    def test_rejects_schedule_provenance_that_conflicts_with_registry(self):
        dataset = self._dataset(
            [
                {
                    "item_id": "task",
                    "data_idx": 0,
                    "extra_info": {
                        "index": 0,
                        "route_attestation_sha256": "b" * 64,
                        "route_registry_sha256": "c" * 64,
                    },
                }
            ]
        )
        route = dataset._route_registry.routes[0]
        dataset._route_registry = RouteRegistry(
            routes=(
                RouteSpec(
                    route_id=route.route_id,
                    max_rounds=route.max_rounds,
                    max_observation_tokens=route.max_observation_tokens,
                    policy_framing_sha256=route.policy_framing_sha256,
                    route_attestation_sha256="a" * 64,
                    client_config=route.client_config,
                ),
            ),
            sha256="d" * 64,
            source_path=None,
        )

        with self.assertRaisesRegex(ValueError, "route_attestation_sha256 drift"):
            dataset[0]

        dataset.dataframe[0]["extra_info"]["route_attestation_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "route_registry_sha256 drift"):
            dataset[0]

    def test_rejects_explicit_null_schedule_provenance(self):
        dataset = self._dataset(
            [
                {
                    "item_id": "task",
                    "data_idx": 0,
                    "extra_info": {
                        "index": 0,
                        "route_attestation_sha256": None,
                        "route_registry_sha256": None,
                    },
                }
            ]
        )
        route = dataset._route_registry.routes[0]
        dataset._route_registry = RouteRegistry(
            routes=(
                RouteSpec(
                    route_id=route.route_id,
                    max_rounds=route.max_rounds,
                    max_observation_tokens=route.max_observation_tokens,
                    policy_framing_sha256=route.policy_framing_sha256,
                    route_attestation_sha256="a" * 64,
                    client_config=route.client_config,
                ),
            ),
            sha256="d" * 64,
            source_path=None,
        )

        with self.assertRaisesRegex(ValueError, "route_attestation_sha256 drift"):
            dataset[0]

        dataset.dataframe[0]["extra_info"]["route_attestation_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "route_registry_sha256 drift"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
