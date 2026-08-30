from __future__ import annotations

import unittest

from agentmemorygym_verl.policy_action_serialization import parse_shell_command_text


class TestParseShellCommandText(unittest.TestCase):
    def test_canonical_json_shell_command(self):
        raw = (
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md",'
            '"workdir":".","timeout_ms":120000}'
        )
        self.assertEqual(
            parse_shell_command_text(raw),
            "cat .agent_memory/CONTINUATION.md",
        )

    def test_qwen_native_shell_command_from_gate_ledger(self):
        raw = """<tool_call>
<function=shell_command>
<parameter=command>
cat .agent_memory/CONTINUATION.md 2>/dev/null || echo "No continuation file found"
</parameter>
<parameter=workdir>
.
</parameter>
<parameter=timeout_ms>
120000
</parameter>
</function>
</tool_call>"""
        self.assertEqual(
            parse_shell_command_text(raw),
            'cat .agent_memory/CONTINUATION.md 2>/dev/null || echo "No continuation file found"',
        )

    def test_one_trailing_eos_matches_endpoint_normalization(self):
        raw = """<tool_call>
<function=shell_command>
<parameter=command>
pwd
</parameter>
</function>
</tool_call></s>"""
        self.assertEqual(parse_shell_command_text(raw), "pwd")

    def test_malformed_native_blocks_remain_rejected(self):
        cases = (
            """<tool_call>
<function=shell_command>
<parameter=command>
pwd
</parameter>
</parameter>
</function>
</tool_call>""",
            """<tool_call>
<function=shell_command>
<parameter=command>
pwd
</parameter>
</tool_call>""",
            """<tool_call>
<function=write_file>
<parameter=command>
pwd
</parameter>
</function>
</tool_call>""",
            "reason first\n<tool_call>\n<function=shell_command>\n"
            "<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>",
        )
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_shell_command_text(raw))

    def test_invalid_shell_arguments_remain_rejected(self):
        cases = (
            'shell_command {"command":""}',
            'shell_command {"command":"pwd","unknown":1}',
            'shell_command {"command":"pwd","timeout_ms":true}',
            """<tool_call>
<function=shell_command>
<parameter=command>
pwd
</parameter>
<parameter=timeout_ms>
nope
</parameter>
</function>
</tool_call>""",
        )
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_shell_command_text(raw))


if __name__ == "__main__":
    unittest.main()
