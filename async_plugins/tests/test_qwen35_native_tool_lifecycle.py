from __future__ import annotations

import os
import unittest

from agentmemorygym_verl.agent_loop import (
    _native_tool_observation_messages,
    _strict_qwen_tool_chat_template,
)


class TestNativeToolObservationMessages(unittest.TestCase):
    def test_valid_tool_call_uses_native_tool_role(self) -> None:
        prepared = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        action = (
            "<tool_call>\n<function=search>\n<parameter=query>\nabc\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        current = prepared + [
            {"role": "assistant", "content": action},
            {"role": "user", "content": "search result"},
        ]

        updated = _native_tool_observation_messages(
            current,
            expected_assistant_messages=prepared + [
                {"role": "assistant", "content": action}
            ],
            tool_name="search",
            observation="search result",
        )

        self.assertEqual(updated[:-1], current[:-1])
        self.assertEqual(
            updated[-1],
            {"role": "tool", "name": "search", "content": "search result"},
        )

    def test_nonappend_transition_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "append exactly one observation"):
            _native_tool_observation_messages(
                [{"role": "system", "content": "replacement"}],
                expected_assistant_messages=[
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "action"},
                ],
                tool_name="search",
                observation="result",
            )


@unittest.skipUnless(
    os.environ.get("AMG_QWEN35_MODEL_PATH"),
    "set AMG_QWEN35_MODEL_PATH for the pinned real-tokenizer contract test",
)
class TestRealQwen35NativeToolLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoTokenizer
        from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder

        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.environ["AMG_QWEN35_MODEL_PATH"],
            trust_remote_code=True,
            local_files_only=True,
        )
        cls.template = _strict_qwen_tool_chat_template(cls.tokenizer.chat_template)
        cls.builder = QwenContinuousTokenBuilder(
            cls.tokenizer,
            chat_template_kwargs={
                "enable_thinking": False,
                "chat_template": cls.template,
            },
        )
        cls.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

    def test_template_has_one_unambiguous_direct_call_contract(self) -> None:
        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Only direct calls."},
                {"role": "user", "content": "task"},
            ],
            tools=self.tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            chat_template=self.template,
        )
        self.assertNotIn("optional reasoning", rendered)
        self.assertNotIn("answer the question like normal", rendered)
        self.assertIn("exactly one function call", rendered)
        self.assertIn("no natural-language reasoning", rendered)

    def test_sampled_call_then_tool_result_matches_full_rerender(self) -> None:
        from verl.utils.tokenizer import normalize_token_ids

        base = [
            {"role": "system", "content": "Only direct calls."},
            {"role": "user", "content": "task"},
        ]
        action = (
            "<tool_call>\n<function=search>\n<parameter=query>\nabc\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        assistant = {"role": "assistant", "content": action}
        updated = base + [
            assistant,
            {"role": "tool", "name": "search", "content": "RESULT"},
        ]

        prompt_ids = normalize_token_ids(
            self.builder.build_initial_tokens(base, tools=self.tools)
        )
        sampled_ids = self.tokenizer.encode(action, add_special_tokens=False) + [
            self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]
        runtime_ids = self.builder.merge_assistant_tokens(
            prompt_ids, sampled_ids
        ).token_ids
        runtime_ids = self.builder.merge_non_assistant_tokens(
            base + [assistant],
            updated,
            runtime_ids,
            tools=self.tools,
        ).token_ids
        full_ids = normalize_token_ids(
            self.builder.build_initial_tokens(updated, tools=self.tools)
        )

        self.assertEqual(runtime_ids, full_ids)
        rendered = self.tokenizer.decode(runtime_ids)
        self.assertIn("<tool_response>\nRESULT\n</tool_response>", rendered)
        self.assertNotIn("<|im_start|>user\nRESULT<|im_end|>", rendered)


if __name__ == "__main__":
    unittest.main()
