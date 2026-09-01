from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from agentmemorygym_verl.agemem_evidence import main, summarize_agemem_step_records


def adapter_evidence(
    *,
    event: str,
    operation: str | None = None,
    accepted: bool | None = None,
    action_index: int = 0,
    before: int | None = None,
    after: int = 0,
    **extra: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "camg_agemem_style_adapter_v1",
        "event": event,
        "episode_private": True,
        "hidden_model_calls": 0,
        "context_operation": extra.pop("context_operation", "append_observation"),
    }
    if event == "memory_tool_action":
        value.update(
            {
                "operation": operation,
                "accepted": accepted,
                "memory_action_index": action_index,
                "memory_size_before": before,
                "memory_size_after": after,
            }
        )
    else:
        value.update(
            {
                "memory_action_count": action_index,
                "memory_size_after": after,
            }
        )
    value.update(extra)
    return value


def row(
    order: int,
    adapter: dict[str, object] | None,
    *,
    route: str = "webshop",
    trajectory: str = "t0",
    context_operation: str = "append_observation",
    terminal: bool = False,
) -> dict[str, object]:
    return {
        "route_id": route,
        "data_source": route,
        "trajectory_uid": trajectory,
        "trajectory_row_order": order,
        "trajectory_terminal": terminal,
        "outcome": "native_success" if terminal else "running",
        "context_transition": {
            "schema": "agentmemory_task_neutral_context_transition_v1",
            "operation": context_operation,
            "messages": [] if context_operation != "replace_messages" else [
                {"role": "system", "content": "replacement"}
            ],
        },
        "wrapper_evidence": {} if adapter is None else {"agemem_adapter": adapter},
    }


class AgeMemEvidenceTests(unittest.TestCase):
    def test_valid_chain_and_cross_context_retrieval_pass(self) -> None:
        records = [
            row(
                0,
                adapter_evidence(
                    event="memory_tool_action",
                    operation="Add_memory",
                    accepted=True,
                    action_index=1,
                    before=0,
                    after=1,
                    memory_id="m000001",
                ),
            ),
            row(
                1,
                adapter_evidence(
                    event="memory_tool_action",
                    operation="Summary_context",
                    accepted=True,
                    action_index=2,
                    before=1,
                    after=1,
                    summarized_message_count=2,
                    context_operation="replace_messages",
                ),
                context_operation="replace_messages",
            ),
            row(
                2,
                adapter_evidence(
                    event="memory_tool_action",
                    operation="Retrieve_memory",
                    accepted=True,
                    action_index=3,
                    before=1,
                    after=1,
                    retrieved_memory_ids=["m000001"],
                    retrieved_memory_count=1,
                ),
                terminal=True,
            ),
            row(
                0,
                adapter_evidence(
                    event="native_action_passthrough",
                    action_index=0,
                    after=0,
                ),
                route="swesmith",
                trajectory="t1",
                terminal=True,
            ),
        ]
        summary = summarize_agemem_step_records(
            records, expected_routes=["webshop", "swesmith"]
        )
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["totals"]["rows"], 4)
        self.assertEqual(summary["totals"]["cross_context_retrievals"], 1)
        self.assertEqual(summary["operation_counts"]["Add_memory"], 1)

    def test_missing_evidence_and_hidden_call_fail_closed(self) -> None:
        missing = row(0, None)
        hidden = row(
            0,
            adapter_evidence(
                event="native_action_passthrough",
                action_index=0,
                after=0,
                hidden_model_calls=1,
            ),
            trajectory="t1",
        )
        summary = summarize_agemem_step_records([missing, hidden])
        self.assertEqual(summary["status"], "FAIL")
        self.assertGreaterEqual(summary["violation_count"], 2)
        self.assertEqual(summary["totals"]["hidden_model_calls"], 1)

    def test_cli_reads_rollout_envelopes_and_writes_receipt(self) -> None:
        first = row(
            0,
            adapter_evidence(
                event="memory_tool_action",
                operation="Add_memory",
                accepted=True,
                action_index=1,
                before=0,
                after=1,
                memory_id="m000001",
            ),
        )
        second = row(
            1,
            adapter_evidence(
                event="native_action_passthrough",
                action_index=1,
                after=1,
            ),
            terminal=True,
        )
        next_update = row(
            0,
            adapter_evidence(
                event="native_action_passthrough",
                action_index=0,
                after=0,
            ),
            terminal=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rollout = root / "1.jsonl"
            rollout.write_text(
                # Async completion order is intentionally reversed here.
                "\n".join(
                    json.dumps(
                        {"step": 1, "step_record_json": json.dumps(record)}
                    )
                    for record in (second, first)
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "2.jsonl").write_text(
                json.dumps(
                    {"step": 2, "step_record_json": json.dumps(next_update)}
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "summary.json"
            with redirect_stdout(StringIO()):
                return_code = main(
                    [
                        "--input",
                        str(root),
                        "--through-update",
                        "2",
                        "--expect-route",
                        "webshop",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(return_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["totals"]["trajectories"], 2)
            self.assertEqual(payload["input_manifest"][0]["step_records"], 2)


if __name__ == "__main__":
    unittest.main()
