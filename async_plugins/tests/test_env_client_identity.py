from __future__ import annotations

import unittest
from unittest import mock

from agentmemorygym_verl.env_client import (
    _OPENMLE_IDENTITY_FIELDS,
    create_env_client,
)


class _RecordingClient:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)


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
