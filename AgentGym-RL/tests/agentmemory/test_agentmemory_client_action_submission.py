from __future__ import annotations

import unittest

from agentenv.controller.types import ActionFormat
from agentenv.envs.agentmemory import AgentMemoryAdapter, AgentMemoryEnvClient


class AgentMemoryClientActionSubmissionTest(unittest.TestCase):
    def client(self, submitted: list[str]) -> AgentMemoryEnvClient:
        client = AgentMemoryEnvClient.__new__(AgentMemoryEnvClient)
        client.is_v3 = False
        client.action_format = ActionFormat.REACT
        client.adapter_cls = AgentMemoryAdapter
        client.metadata = {}

        def post(path, payload):
            self.assertEqual(path, "step")
            submitted.append(payload["action"])
            return {
                "observation": "next",
                "reward": 0.0,
                "done": False,
                "info": {},
            }

        client.post = post
        return client

    def test_records_react_output_and_submitted_native_action(self):
        submitted: list[str] = []
        client = self.client(submitted)
        raw_output = "Thought:\ncheck the page\n\nAction:\nclick[Buy Now]"

        client.step(raw_output)

        self.assertEqual(submitted, ["click[Buy Now]"])
        self.assertEqual(
            client.last_action_submission,
            {
                "raw_policy_output": raw_output,
                "submitted_action": "click[Buy Now]",
                "parser_status": "adapter_parsed",
            },
        )

    def test_records_raw_fallback_for_unsupported_wrapper(self):
        submitted: list[str] = []
        client = self.client(submitted)
        raw_output = '{"action": "click[Buy Now]"}'

        client.step(raw_output)

        self.assertEqual(submitted, [raw_output])
        self.assertEqual(
            client.last_action_submission,
            {
                "raw_policy_output": raw_output,
                "submitted_action": raw_output,
                "parser_status": "raw_fallback",
            },
        )

    def test_records_eos_only_empty_raw_fallback(self):
        submitted: list[str] = []
        client = self.client(submitted)

        client.step("")

        self.assertEqual(submitted, [""])
        self.assertEqual(
            client.last_action_submission,
            {
                "raw_policy_output": "",
                "submitted_action": "",
                "parser_status": "raw_fallback",
            },
        )

    def test_raw_fallback_removes_only_one_terminal_textual_eos(self):
        submitted: list[str] = []
        client = self.client(submitted)

        client.step("</s></s>")

        self.assertEqual(submitted, ["</s>"])
        self.assertEqual(
            client.last_action_submission,
            {
                "raw_policy_output": "</s></s>",
                "submitted_action": "</s>",
                "parser_status": "raw_fallback",
            },
        )


if __name__ == "__main__":
    unittest.main()
