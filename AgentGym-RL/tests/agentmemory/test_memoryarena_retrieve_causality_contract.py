from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_CONTEXT = ROOT / "verl/utils/agentgym/rollout_context.py"
ROLLOUT_SCHEMAS = ROOT / "verl/workers/rollout/schemas.py"


class MemoryArenaRetrieveCausalityContractTest(unittest.TestCase):
    def test_webshop_formal_records_require_latest_observation_only_prompt(self) -> None:
        source = ROLLOUT_CONTEXT.read_text(encoding="utf-8")

        self.assertIn('record["prompt_history_policy"] != "latest_observation_only"', source)
        self.assertIn('record["raw_prior_messages_visible"]', source)
        self.assertIn('record["latest_observation"] not in record["visible_prompt"]', source)
        self.assertIn('record["single_observation_prompt_digest"]', source)
        self.assertIn('"Formal visible prompt omits the latest observation', source)
        self.assertIn('"Formal prompt history policy mismatch', source)
        self.assertIn('"Formal prompt exposes raw prior messages', source)

    def test_webshop_formal_records_require_raw_history_clear_on_session_advance(self) -> None:
        source = ROLLOUT_CONTEXT.read_text(encoding="utf-8")

        self.assertIn('after_trace = record["env_info_after"].get("session_trace")', source)
        self.assertIn('expected_history_cleared = bool(session_advanced and not after_trace)', source)
        self.assertIn('record["raw_history_cleared"] != expected_history_cleared', source)
        self.assertIn('session_advanced and not record["raw_history_cleared"]', source)
        self.assertIn('"Formal session advanced without clearing raw history', source)

    def test_prompt_builder_uses_only_system_prompt_and_latest_observation(self) -> None:
        source = ROLLOUT_SCHEMAS.read_text(encoding="utf-8")
        start = source.index("def get_latest_observation_prompt")
        end = source.index("def ", start + len("def get_latest_observation_prompt"))
        function_source = source[start:end]

        self.assertIn('{"role": "system", "content": system_prompt}', function_source)
        self.assertIn("latest_user_message = self.messages[-1]", function_source)
        self.assertIn('latest_user_message.role == "user"', function_source)
        self.assertIn("latest_user_message.to_dict()", function_source)
        self.assertNotIn('self.messages +', function_source)
        self.assertNotIn('self.messages[:-1]', function_source)
        self.assertNotIn('self.messages +', function_source)


if __name__ == "__main__":
    unittest.main()
