from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agentenv.controller.env import BaseEnvClient
from agentenv.controller.types import StepOutput
from agentenv.envs.agemem import AGEMEM_PROMPT_MARKER, AgeMemEnvClientAdapter
from agentenv.envs.letta_code import LettaCodeEnvClientAdapter
from agentenv.envs.mem0 import Mem0EnvClientAdapter
from agentmemorygym_verl import env_client


class FakeClient(BaseEnvClient):
    instances: list["FakeClient"] = []

    def __init__(self, **kwargs) -> None:
        super().__init__("react")
        self.kwargs = kwargs
        self.closed = False
        self.info = {"observation": "task"}
        self.episode_source_identity = None
        self.__class__.instances.append(self)

    def __len__(self) -> int:
        return 2

    def observe(self) -> str:
        return "task"

    def policy_framing(self):
        return [{"role": "system", "content": "native"}]

    def step(self, action):
        return StepOutput(state=action, reward=0.0, done=False, info={})

    def reset(self, idx: int):
        self.episode_source_identity = {
            "schema": "camg_native_episode_source_identity_v1",
            "route_id": "webshop",
            "data_idx": idx,
        }
        return None

    def close(self):
        self.closed = True
        return True


def config(task_name: str, *, adapter: object = ...):
    value = {"task_name": task_name, "env_addr": "http://127.0.0.1:1234"}
    if adapter is ...:
        value["memory_adapter"] = {
            "schema": "camg_agemem_style_route_v1",
            "name": "agemem_style",
            "config": {"max_memories": 7},
        }
    elif adapter is not None:
        value["memory_adapter"] = adapter
    return value


class AgeMemEnvClientConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeClient.instances.clear()

    def test_two_distinct_routes_use_the_same_task_neutral_adapter(self) -> None:
        with patch.object(
            env_client,
            "_client_classes",
            return_value={"webshop": FakeClient, "swesmith": FakeClient},
        ):
            shop = env_client.create_env_client(config("webshop"))
            coding = env_client.create_env_client(config("swesmith"))
        self.assertIsInstance(shop, AgeMemEnvClientAdapter)
        self.assertIsInstance(coding, AgeMemEnvClientAdapter)
        self.assertEqual(type(shop), type(coding))
        self.assertEqual(shop.config.max_memories, 7)
        self.assertEqual(coding.config.max_memories, 7)
        self.assertIn(AGEMEM_PROMPT_MARKER, shop.policy_framing()[0]["content"])

        shop.reset(0)
        action = (
            '<agemem_tool_call>[{"name":"Add_memory","arguments":'
            '{"content":"shop-only"}}]</agemem_tool_call>'
        )
        shop.step(action)
        coding.reset(0)
        retrieved = coding.step(
            '<agemem_tool_call>[{"name":"Retrieve_memory","arguments":'
            '{"query":"shop-only"}}]</agemem_tool_call>'
        )
        payload = json.loads(retrieved.state.split("\n", 1)[1])
        self.assertEqual(payload["memories"], [])

    def test_absent_adapter_preserves_native_client(self) -> None:
        with patch.object(
            env_client, "_client_classes", return_value={"webshop": FakeClient}
        ):
            client = env_client.create_env_client(config("webshop", adapter=None))
        self.assertIsInstance(client, FakeClient)
        self.assertNotIsInstance(client, AgeMemEnvClientAdapter)

    def test_mem0_and_letta_routes_use_their_task_neutral_adapters(self) -> None:
        adapters = (
            (
                {
                    "schema": "camg_mem0_route_v1",
                    "name": "mem0",
                    "config": {
                        "runtime_root": "/tmp/camg-mem0-test",
                        "llm_base_url": "http://127.0.0.1:65201/v1",
                        "embedding_base_url": "http://127.0.0.1:65202/v1",
                    },
                },
                Mem0EnvClientAdapter,
            ),
            (
                {
                    "schema": "camg_letta_code_route_v1",
                    "name": "letta_code",
                    "config": {"runtime_root": "/tmp/camg-letta-test"},
                },
                LettaCodeEnvClientAdapter,
            ),
        )
        with patch.object(
            env_client, "_client_classes", return_value={"webshop": FakeClient}
        ):
            for adapter, expected_type in adapters:
                with self.subTest(expected_type=expected_type.__name__):
                    client = env_client.create_env_client(
                        config("webshop", adapter=adapter)
                    )
                    self.assertIsInstance(client, expected_type)
                    client.close()

    def test_mismatched_schema_name_pair_fails_closed(self) -> None:
        with patch.object(
            env_client, "_client_classes", return_value={"webshop": FakeClient}
        ):
            with self.assertRaisesRegex(ValueError, "schema/name pair"):
                env_client.create_env_client(
                    config(
                        "webshop",
                        adapter={
                            "schema": "camg_mem0_route_v1",
                            "name": "letta_code",
                            "config": {},
                        },
                    )
                )
        self.assertTrue(FakeClient.instances[-1].closed)

    def test_swesmith_private_detail_token_binding_is_forwarded(self) -> None:
        value = config("swesmith", adapter=None)
        value.update(
            {
                "detail_token_path": "/run/heldout/swesmith-detail.token",
                "detail_token_sha256": "a" * 64,
            }
        )
        with patch.object(
            env_client, "_client_classes", return_value={"swesmith": FakeClient}
        ):
            client = env_client.create_env_client(value)
        self.assertEqual(
            client.kwargs["detail_token_path"],
            "/run/heldout/swesmith-detail.token",
        )
        self.assertEqual(client.kwargs["detail_token_sha256"], "a" * 64)

        incomplete = config("swesmith", adapter=None)
        incomplete["detail_token_path"] = "/run/heldout/swesmith-detail.token"
        with patch.object(
            env_client, "_client_classes", return_value={"swesmith": FakeClient}
        ), self.assertRaisesRegex(ValueError, "incomplete"):
            env_client.create_env_client(incomplete)

    def test_invalid_adapter_fails_closed_and_closes_native_client(self) -> None:
        with patch.object(
            env_client, "_client_classes", return_value={"webshop": FakeClient}
        ):
            with self.assertRaises(ValueError):
                env_client.create_env_client(
                    config(
                        "webshop",
                        adapter={"schema": "wrong", "name": "agemem_style"},
                    )
                )
        self.assertTrue(FakeClient.instances[-1].closed)


if __name__ == "__main__":
    unittest.main()
