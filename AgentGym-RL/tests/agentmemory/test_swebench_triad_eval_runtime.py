from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_paired_eval_support import Arm, make_config

from paired_eval.contracts import (
    ArtifactResult,
    FinalizationContext,
    ScorerResult,
    capability_for_arm,
)
from paired_eval.evidence import PrivateEvidenceStore
from paired_eval.model_client import ModelClientError, OpenAICompatibleModelClient
from paired_eval.registry import DEFAULT_ADAPTER_SPECS
from swebench_triad_eval.model_transport import ExactTokenVllmTransport
from swebench_triad_eval.runtime_factory import (
    SwebenchRuntimeEndpoint,
    finalize_swebench_artifact,
    handoff_swebench_grader,
    make_swebench_runtime_factory,
)


class RecordingTransport:
    def __init__(self) -> None:
        self.calls = []
        self.tokenize_response = {"tokens": [11, 12, 13]}
        self.chat_response = {
            "prompt_token_ids": [11, 12, 13],
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                    "token_ids": [21, 22],
                }
            ],
        }

    def post(self, url, payload, *, timeout_seconds):
        self.calls.append((url, payload, timeout_seconds))
        if url.endswith("/tokenize"):
            return self.tokenize_response
        return self.chat_response


def tokenize_payload() -> dict:
    return {
        "model": "served-model",
        "messages": [{"role": "user", "content": "task"}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def chat_payload() -> dict:
    return {
        "model": "served-model",
        "messages": [{"role": "user", "content": "task"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32,
        "seed": 0,
        "n": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


class ExactTokenVllmTransportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.delegate = RecordingTransport()
        self.transport = ExactTokenVllmTransport(self.delegate)

    def bind_tokenization(self) -> None:
        payload = tokenize_payload()
        response = self.transport.post(
            "http://127.0.0.1:8000/tokenize",
            payload,
            timeout_seconds=5.0,
        )
        self.assertIs(response, self.delegate.tokenize_response)
        self.assertIs(self.delegate.calls[-1][1], payload)

    def test_tokenize_payload_is_forwarded_without_mutation(self) -> None:
        payload = tokenize_payload()
        before = dict(payload)

        self.transport.post(
            "http://127.0.0.1:8000/tokenize",
            payload,
            timeout_seconds=5.0,
        )

        self.assertEqual(payload, before)
        self.assertIs(self.delegate.calls[-1][1], payload)

    def test_chat_only_adds_return_token_ids_and_validates_both_streams(
        self,
    ) -> None:
        self.bind_tokenization()
        payload = chat_payload()
        before = dict(payload)

        response = self.transport.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            payload,
            timeout_seconds=5.0,
        )

        sent = self.delegate.calls[-1][1]
        self.assertEqual(payload, before)
        self.assertEqual(sent, {**payload, "return_token_ids": True})
        self.assertIs(response, self.delegate.chat_response)

    def test_exact_tokens_are_invariant_across_c1_c2_reverse_and_resume(self) -> None:
        def execute(task_index: int, generation: int):
            transport = ExactTokenVllmTransport(
                RecordingTransport(),
                request_context={
                    "run_id": f"task-{task_index}",
                    "task_index": task_index,
                    "arm": "native",
                    "generation": generation,
                },
            )
            transport.post(
                "http://127.0.0.1:8000/tokenize",
                tokenize_payload(),
                timeout_seconds=5.0,
            )
            transport.post(
                "http://127.0.0.1:8000/v1/chat/completions",
                chat_payload(),
                timeout_seconds=5.0,
            )
            event = transport.events[-1]
            return {
                "prompt": event["prompt_token_ids"],
                "response": event["response_token_ids"],
                "semantic": event["semantic_request_sha256"],
                "request_id": event["request_id"],
            }

        serial = {task: execute(task, 1) for task in (0, 1)}
        with ThreadPoolExecutor(max_workers=2) as executor:
            concurrent = dict(zip((0, 1), executor.map(lambda task: execute(task, 1), (0, 1))))
        with ThreadPoolExecutor(max_workers=2) as executor:
            reversed_rows = list(executor.map(lambda task: execute(task, 1), (1, 0)))
        reversed_arrival = dict(zip((1, 0), reversed_rows))
        resumed = execute(0, 2)

        for task in (0, 1):
            expected = {
                name: serial[task][name]
                for name in ("prompt", "response", "semantic")
            }
            self.assertEqual(
                {name: concurrent[task][name] for name in expected}, expected
            )
            self.assertEqual(
                {name: reversed_arrival[task][name] for name in expected},
                expected,
            )
        self.assertEqual(resumed["prompt"], serial[0]["prompt"])
        self.assertEqual(resumed["response"], serial[0]["response"])
        self.assertEqual(resumed["semantic"], serial[0]["semantic"])
        self.assertNotEqual(resumed["request_id"], serial[0]["request_id"])

    def test_missing_or_malformed_prompt_ids_fail_closed(self) -> None:
        malformed = (
            {},
            {"tokens": []},
            {"tokens": [11, True]},
            {"tokens": [-1]},
            {"tokens": ["11"]},
        )
        for response in malformed:
            with self.subTest(response=response):
                delegate = RecordingTransport()
                delegate.tokenize_response = response
                transport = ExactTokenVllmTransport(delegate)
                with self.assertRaises(ModelClientError):
                    transport.post(
                        "http://127.0.0.1:8000/tokenize",
                        tokenize_payload(),
                        timeout_seconds=5.0,
                    )

    def test_missing_or_malformed_chat_token_ids_fail_closed(self) -> None:
        self.bind_tokenization()
        malformed = (
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                        "token_ids": [21],
                    }
                ]
            },
            {"prompt_token_ids": [11, 12, 13], "choices": []},
            {
                "prompt_token_ids": [11, 12, 13],
                "choices": [{"token_ids": []}],
            },
            {
                "prompt_token_ids": [11, 12, 13],
                "choices": [{"token_ids": [21, True]}],
            },
        )
        for response in malformed:
            with self.subTest(response=response):
                self.delegate.chat_response = response
                with self.assertRaises(ModelClientError):
                    self.transport.post(
                        "http://127.0.0.1:8000/v1/chat/completions",
                        chat_payload(),
                        timeout_seconds=5.0,
                    )

    def test_chat_prompt_ids_must_match_the_tokenize_endpoint(self) -> None:
        self.bind_tokenization()
        self.delegate.chat_response["prompt_token_ids"] = [11, 12, 99]

        with self.assertRaises(ModelClientError):
            self.transport.post(
                "http://127.0.0.1:8000/v1/chat/completions",
                chat_payload(),
                timeout_seconds=5.0,
            )

    def test_chat_cannot_bypass_or_override_the_exact_token_request(self) -> None:
        self.bind_tokenization()
        for value in (False, 1, "true"):
            with self.subTest(value=value):
                payload = chat_payload()
                payload["return_token_ids"] = value
                with self.assertRaises(ModelClientError):
                    self.transport.post(
                        "http://127.0.0.1:8000/v1/chat/completions",
                        payload,
                        timeout_seconds=5.0,
                    )


class FakeSwebenchClient:
    def __init__(self, env_server_base, *, arm, **kwargs) -> None:
        self.env_server_base = env_server_base
        self.arm = arm
        self.kwargs = kwargs
        self.info = {"done": False}
        self.horizon_calls = 0
        self.prediction_row = {
            "instance_id": "owner__repo-1",
            "model_name_or_path": f"amg-swebench-{arm}",
            "model_patch": "diff --git a/value.py b/value.py\n",
        }

    def finalize_policy_horizon(self):
        self.horizon_calls += 1
        self.info = {"done": True}
        return object()

    def prediction(self):
        return dict(self.prediction_row)


def swebench_config(arm: Arm = Arm.NATIVE):
    spec = DEFAULT_ADAPTER_SPECS[1]
    config = make_config(
        benchmark="swebench_verified",
        arm=arm,
        artifact_type="patch",
    )
    return replace(
        config,
        task=replace(
            config.task,
            task_id="owner__repo-1",
            native_tools=spec.native_tools,
        ),
        capability=capability_for_arm(arm),
    )


class SwebenchRuntimeFactoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = PrivateEvidenceStore(
            Path(self.temporary.name) / "evidence"
        )

    def test_factory_binds_published_client_and_exact_model_transport(self) -> None:
        config = swebench_config()
        endpoint = SwebenchRuntimeEndpoint(
            env_server_base="http://127.0.0.1:9100",
            private_run_id="private-attempt-1",
            run_capability="private-capability",
            image_manifest_sha256="a" * 64,
            task_index=config.task.task_index,
            arm=config.capability.arm.value,
            generation=1,
        )
        delegate = RecordingTransport()
        seen = []

        def resolve(value):
            seen.append(value)
            return endpoint

        factory = make_swebench_runtime_factory(
            evidence_store=self.store,
            endpoint_resolver=resolve,
            model_base_url="http://127.0.0.1:8000/v1",
            transport_factory=lambda: delegate,
            model_timeout_seconds=7.0,
            environment_timeout_seconds=1800,
        )
        with patch(
            "paired_eval.registry.AdapterSpec.resolve_client_type",
            return_value=FakeSwebenchClient,
        ):
            bindings = factory(config)

        self.assertEqual(seen, [config])
        self.assertIsInstance(bindings.adapter.raw_client, FakeSwebenchClient)
        self.assertEqual(
            bindings.adapter.raw_client.env_server_base,
            endpoint.env_server_base,
        )
        self.assertEqual(
            bindings.adapter.raw_client.kwargs,
            {
                "run_id": endpoint.private_run_id,
                "run_capability": endpoint.run_capability,
                "image_manifest_sha256": endpoint.image_manifest_sha256,
                "data_len": 500,
                "timeout": 1800,
            },
        )
        self.assertIsInstance(bindings.model, OpenAICompatibleModelClient)
        self.assertIsInstance(bindings.model.transport, ExactTokenVllmTransport)
        self.assertIs(bindings.model.transport.delegate, delegate)
        self.assertEqual(bindings.model.model_config, config.model)

    def test_artifact_forces_only_nonterminal_horizon_and_keeps_patch_private(
        self,
    ) -> None:
        config = swebench_config()
        client = FakeSwebenchClient("http://127.0.0.1", arm="native")
        context = FinalizationContext(
            termination_reason="policy_horizon",
            horizon_cause="policy_turn_limit",
            failure_class=None,
            timed_out=False,
            policy_turns=250,
            tool_calls=250,
        )

        artifact = finalize_swebench_artifact(
            client,
            context,
            config,
            self.store,
        )

        self.assertIsInstance(artifact, ArtifactResult)
        self.assertEqual(client.horizon_calls, 1)
        self.assertEqual(artifact.artifact_type, "patch")
        self.assertNotIn("model_patch", artifact.receipt)
        self.assertEqual(artifact.receipt["instance_id"], config.task.task_id)
        self.assertEqual(
            artifact.receipt["model_patch_bytes"],
            len(client.prediction_row["model_patch"].encode("utf-8")),
        )

        client.info = {"done": True}
        finalize_swebench_artifact(client, context, config, self.store)
        self.assertEqual(client.horizon_calls, 1)

    def test_artifact_rejects_wrong_task_and_grader_handoff_stays_queued(
        self,
    ) -> None:
        config = swebench_config(Arm.AMG_MEMORY)
        client = FakeSwebenchClient(
            "http://127.0.0.1",
            arm=Arm.AMG_MEMORY.value,
        )
        client.prediction_row["instance_id"] = "wrong-task"
        context = FinalizationContext(
            termination_reason="terminal",
            horizon_cause=None,
            failure_class=None,
            timed_out=False,
            policy_turns=1,
            tool_calls=1,
        )
        with self.assertRaises(RuntimeError):
            finalize_swebench_artifact(
                client,
                context,
                config,
                self.store,
            )

        client.prediction_row["instance_id"] = config.task.task_id
        artifact = finalize_swebench_artifact(
            client,
            context,
            config,
            self.store,
        )
        score = handoff_swebench_grader(
            client,
            artifact,
            config,
            self.store,
        )
        self.assertIsInstance(score, ScorerResult)
        self.assertEqual(score.public_metrics, {"official_resolved": None})
        self.assertEqual(score.receipt["status"], "queued")
        self.assertIsNone(score.receipt["official_resolved"])
        self.assertEqual(score.receipt["artifact_sha256"], artifact.sha256)


if __name__ == "__main__":
    unittest.main()
