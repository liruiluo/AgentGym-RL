from __future__ import annotations

import unittest
from unittest import mock

from agentmemorygym_verl.env_client import (
    _CLIENT_CLASS_NAMES,
    _OPENMLE_IDENTITY_FIELDS,
    create_env_client,
)


class _RecordingClient:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)


class _RecordingAgentMemoryClient(_RecordingClient):
    configured_prompts: list[str] = []

    def configure_policy_system_prompt(self, prompt: str) -> None:
        self.configured_prompts.append(prompt)


class TestClientRegistryAndPromptBinding(unittest.TestCase):
    def setUp(self):
        _RecordingClient.calls.clear()
        _RecordingAgentMemoryClient.configured_prompts.clear()

    def test_literesearcher_client_is_registered(self):
        self.assertEqual(
            _CLIENT_CLASS_NAMES["literesearcher"],
            "LiteResearcherEnvClient",
        )

    def test_agentmemory_prompt_is_bound_after_construction(self):
        config = {
            "task_name": "agentmemory",
            "env_addr": "http://127.0.0.1:65101",
            "timeout": 17,
            "max_retries": 0,
            "policy_system_prompt": "  exact formal prompt  ",
        }
        with mock.patch(
            "agentmemorygym_verl.env_client._client_classes",
            return_value={"agentmemory": _RecordingAgentMemoryClient},
        ):
            create_env_client(config)

        self.assertEqual(
            _RecordingAgentMemoryClient.calls,
            [
                {
                    "env_server_base": "http://127.0.0.1:65101",
                    "data_len": None,
                    "timeout": 17.0,
                }
            ],
        )
        self.assertEqual(
            _RecordingAgentMemoryClient.configured_prompts,
            ["exact formal prompt"],
        )

    def test_agentmemory_prompt_is_required_before_construction(self):
        config = {
            "task_name": "agentmemory",
            "env_addr": "http://127.0.0.1:65101",
            "max_retries": 0,
        }
        with (
            mock.patch(
                "agentmemorygym_verl.env_client._client_classes",
                return_value={"agentmemory": _RecordingAgentMemoryClient},
            ),
            self.assertRaisesRegex(ValueError, "policy_system_prompt"),
        ):
            create_env_client(config)
        self.assertEqual(_RecordingAgentMemoryClient.calls, [])


class TestOpenMLEClientIdentityForwarding(unittest.TestCase):
    def setUp(self):
        _RecordingClient.calls.clear()

    def test_exactly_nine_identity_fields_are_forwarded_unchanged(self):
        identity = {
            "expected_manifest_sha256": "a" * 64,
            "expected_release_revision": "b" * 40,
            "expected_outer_commit": "c" * 40,
            "expected_inner_commit": "d" * 40,
            "expected_role": "gate_only",
            "expected_executor_runtime_digest": "sha256:" + "e" * 64,
            "expected_materializer_sha256": "f" * 64,
            "expected_actions_sha256": "1" * 64,
            "expected_max_observation_tokens": 8192,
        }
        config = {
            "task_name": "openmle_fast",
            "env_addr": "http://127.0.0.1:65525/",
            "timeout": 240,
            "max_retries": 0,
            "ignored_caller_field": "must-not-forward",
            **identity,
        }
        with mock.patch(
            "agentmemorygym_verl.env_client._client_classes",
            return_value={"openmle_fast": _RecordingClient},
        ):
            create_env_client(config)

        self.assertEqual(len(_OPENMLE_IDENTITY_FIELDS), 9)
        self.assertEqual(len(_RecordingClient.calls), 1)
        forwarded = _RecordingClient.calls[0]
        self.assertEqual(
            set(forwarded),
            {"env_server_base", "data_len", "timeout", *_OPENMLE_IDENTITY_FIELDS},
        )
        for field in _OPENMLE_IDENTITY_FIELDS:
            self.assertEqual(forwarded[field], identity[field])
        self.assertNotIn("ignored_caller_field", forwarded)

    def test_non_openmle_route_does_not_receive_openmle_identity_fields(self):
        config = {
            "task_name": "webshop",
            "env_addr": "http://127.0.0.1:65101",
            "timeout": 17,
            "max_retries": 0,
            **{
                field: f"must-not-forward-{field}"
                for field in _OPENMLE_IDENTITY_FIELDS
            },
        }
        with mock.patch(
            "agentmemorygym_verl.env_client._client_classes",
            return_value={"webshop": _RecordingClient},
        ):
            create_env_client(config)

        self.assertEqual(
            _RecordingClient.calls,
            [
                {
                    "env_server_base": "http://127.0.0.1:65101",
                    "data_len": None,
                    "timeout": 17.0,
                }
            ],
        )

    def test_each_missing_identity_field_fails_before_client_construction(self):
        identity = {
            "expected_manifest_sha256": "a" * 64,
            "expected_release_revision": "b" * 40,
            "expected_outer_commit": "c" * 40,
            "expected_inner_commit": "d" * 40,
            "expected_role": "train_pool",
            "expected_executor_runtime_digest": "sha256:" + "e" * 64,
            "expected_materializer_sha256": "f" * 64,
            "expected_actions_sha256": "1" * 64,
            "expected_max_observation_tokens": 8192,
        }
        for field in _OPENMLE_IDENTITY_FIELDS:
            with self.subTest(field=field):
                config = {
                    "task_name": "openmle_fast",
                    "env_addr": "http://127.0.0.1:65525",
                    "max_retries": 0,
                    **identity,
                }
                del config[field]
                with (
                    mock.patch(
                        "agentmemorygym_verl.env_client._client_classes",
                        return_value={"openmle_fast": _RecordingClient},
                    ),
                    self.assertRaisesRegex(ValueError, field),
                ):
                    create_env_client(config)


if __name__ == "__main__":
    unittest.main()
