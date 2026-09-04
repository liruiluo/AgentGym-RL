from __future__ import annotations

import unittest

from agentmemorygym_verl.heldout_method_evidence import summarize_method_step_records


CONTEXT = {
    "schema": "agentmemory_task_neutral_context_transition_v1",
    "operation": "append_observation",
    "messages": [],
}


def row(adapter_key=None, adapter=None):
    wrapper = {}
    if adapter_key is not None:
        wrapper[adapter_key] = adapter
    return {
        "trajectory_uid": "trajectory-0",
        "context_transition": dict(CONTEXT),
        "wrapper_evidence": wrapper,
    }


def mem0(*, boundary=False):
    return {
        "schema": "camg_mem0_adapter_v1",
        "event": "native_action_passthrough",
        "episode_private": True,
        "official_pipeline": True,
        "source_revision": "71fba8d46436f88569d600f81a55208c38ad30b5",
        "version": "2.0.19",
        "boundary_pipeline": boundary,
        "operation_counts": (
            {"add": 1, "search": 1, "added": 1, "retrieved": 1}
            if boundary
            else {}
        ),
        "hidden_model_calls": 2 if boundary else 0,
        "hidden_input_tokens": 20 if boundary else 0,
        "hidden_output_tokens": 10 if boundary else 0,
        "hidden_latency_ms": 50 if boundary else 0,
    }


def letta(event="native_action_passthrough"):
    value = {
        "schema": "camg_letta_code_adapter_v1",
        "event": event,
        "episode_private": True,
        "git_backed": True,
        "source_revision": "787b856f9db9f5030dc2976618e1d1f909f61612",
        "hidden_model_calls": 0,
        "hidden_input_tokens": 0,
        "hidden_output_tokens": 0,
        "hidden_latency_ms": 0,
    }
    if event == "memory_tool_action":
        value.update(
            operation="create",
            accepted=True,
            reason="remember exact fact",
            commit_sha="a" * 40,
        )
    if event == "memory_filesystem_read":
        value.update(
            operation="read",
            accepted=True,
            read_path="reference/fact.md",
            read_bytes=12,
        )
    return value


class HeldoutMethodEvidenceTests(unittest.TestCase):
    def test_frozen_qwen_rejects_any_method_adapter(self):
        clean = summarize_method_step_records([row()], method_id="qwen35_4b")
        self.assertEqual(clean["status"], "PASS")
        contaminated = summarize_method_step_records(
            [row("mem0_adapter", mem0())], method_id="qwen35_4b"
        )
        self.assertEqual(contaminated["status"], "FAIL")
        self.assertIn("method adapter evidence", contaminated["violations"][0])

    def test_mem0_requires_complete_per_row_official_pipeline_evidence(self):
        summary = summarize_method_step_records(
            [
                row("mem0_adapter", mem0()),
                row("mem0_adapter", mem0(boundary=True)),
            ],
            method_id="mem0",
        )
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["totals"]["hidden_model_calls"], 2)
        self.assertEqual(summary["operation_counts"]["add"], 1)

        missing = summarize_method_step_records([row()], method_id="mem0")
        self.assertEqual(missing["status"], "FAIL")
        wrong_revision = mem0()
        wrong_revision["source_revision"] = "b" * 40
        drift = summarize_method_step_records(
            [row("mem0_adapter", wrong_revision)], method_id="mem0"
        )
        self.assertEqual(drift["status"], "FAIL")

    def test_letta_accepts_native_write_and_filesystem_read_without_hidden_calls(self):
        summary = summarize_method_step_records(
            [
                row("letta_code_adapter", letta()),
                row("letta_code_adapter", letta("memory_tool_action")),
                row("letta_code_adapter", letta("memory_filesystem_read")),
            ],
            method_id="letta_code",
        )
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["totals"]["git_commits"], 1)
        self.assertEqual(summary["operation_counts"]["read"], 1)

        hidden = letta()
        hidden["hidden_model_calls"] = 1
        bad = summarize_method_step_records(
            [row("letta_code_adapter", hidden)], method_id="letta_code"
        )
        self.assertEqual(bad["status"], "FAIL")

    def test_adapter_identity_is_exclusive(self):
        contaminated = row("mem0_adapter", mem0())
        contaminated["wrapper_evidence"]["letta_code_adapter"] = letta()
        summary = summarize_method_step_records([contaminated], method_id="mem0")
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any("foreign adapter" in item for item in summary["violations"]))


if __name__ == "__main__":
    unittest.main()
