from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from test_paired_eval_support import SHA_A

from paired_eval.contracts import DecodingConfig, ModelConfig
from paired_eval.evidence import PrivateEvidenceStore
from paired_eval.model_client import (
    ModelClientError,
    OpenAICompatibleModelClient,
)


class FakeTransport:
    def __init__(self, *, omit_response_tokens: bool = False) -> None:
        self.omit_response_tokens = omit_response_tokens
        self.calls = []

    def post(self, url, payload, *, timeout_seconds):
        self.calls.append((url, payload, timeout_seconds))
        if url.endswith("/tokenize"):
            return {"token_ids": [11, 12, 13]}
        choice = {
            "message": {"role": "assistant", "content": "deterministic output"},
            "finish_reason": "stop",
        }
        if not self.omit_response_tokens:
            choice["token_ids"] = [21, 22]
        return {"id": "response-id", "choices": [choice]}


class OpenAICompatibleModelClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store = PrivateEvidenceStore(Path(self.temp_dir.name) / "evidence")
        self.model_config = ModelConfig(
            model_id="served-model",
            revision="revision-1",
            tokenizer_sha256=SHA_A,
        )
        self.decoding = DecodingConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=32,
        )
        self.messages = (
            {"role": "system", "content": "public instruction"},
            {"role": "user", "content": "public task"},
        )

    def test_exact_deterministic_request_and_response_evidence(self) -> None:
        transport = FakeTransport()
        client = OpenAICompatibleModelClient(
            base_url="http://127.0.0.1:8000/v1",
            model_config=self.model_config,
            transport=transport,
            evidence_store=self.store,
            timeout_seconds=5.0,
        )

        self.assertEqual(client.count_prompt_tokens(self.messages), 3)
        output = client.complete(self.messages, self.decoding, seed=17)

        completion_payload = next(
            payload
            for url, payload, _ in transport.calls
            if url.endswith("/chat/completions")
        )
        self.assertEqual(completion_payload["seed"], 17)
        self.assertEqual(completion_payload["temperature"], 0.0)
        self.assertEqual(completion_payload["top_p"], 1.0)
        self.assertEqual(completion_payload["n"], 1)
        self.assertFalse(completion_payload["stream"])
        self.assertEqual(output.prompt_token_ids, (11, 12, 13))
        self.assertEqual(output.response_token_ids, (21, 22))
        self.assertEqual(output.finish_reason, "stop")
        self.assertTrue(output.request_ref.protected_ref.startswith("evidence://"))
        self.assertTrue(output.response_ref.protected_ref.startswith("evidence://"))

    def test_missing_exact_response_tokens_fails_closed(self) -> None:
        client = OpenAICompatibleModelClient(
            base_url="http://127.0.0.1:8000/v1",
            model_config=self.model_config,
            transport=FakeTransport(omit_response_tokens=True),
            evidence_store=self.store,
            timeout_seconds=5.0,
        )

        with self.assertRaises(ModelClientError):
            client.complete(self.messages, self.decoding, seed=17)


if __name__ == "__main__":
    unittest.main()
