from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "agentmemory"
    / "analyze_filesystem_behavior_io.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_filesystem_behavior_io", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _snapshot(files: list[dict[str, object]]) -> dict[str, object]:
    return {"files": files}


def _step(
    turn: int,
    phase_before: int,
    phase_after: int,
    *,
    model_text: str,
    user_text: str = "",
    event: dict[str, object] | None = None,
) -> dict[str, object]:
    files = []
    if event is not None:
        files = list(event.get("snapshot_files", []))
        event = {key: value for key, value in event.items() if key != "snapshot_files"}
    return {
        "turn": turn,
        "model_text": model_text,
        "action_submitted": model_text,
        "request_messages": [{"role": "user", "content": user_text}],
        "env_info_before": {"current_subtask_index": phase_before},
        "env_info_after": {
            "current_subtask_index": phase_after,
            "workspace_latest_event": event,
            "workspace_snapshot": _snapshot(files),
        },
        "reward_components": [],
        "env_response": {"observation": "ok"},
    }


class AnalyzeFilesystemBehaviorIoTest(unittest.TestCase):
    def test_parse_add_file_contents(self) -> None:
        patch = (
            "apply_patch\n*** Begin Patch\n*** Add File: notes/fact.md\n"
            "+cake flavor: vanilla\n+detail\n*** End Patch"
        )
        self.assertEqual(
            MODULE._parse_add_file_contents(patch),
            {"notes/fact.md": "cake flavor: vanilla\ndetail\n"},
        )

    def test_persisted_file_without_content_read_is_not_strict(self) -> None:
        version = {"path": "notes/fact.md", "sha256": "a" * 64, "bytes": 21}
        patch = (
            "apply_patch\n*** Begin Patch\n*** Add File: notes/fact.md\n"
            "+listed cake flavor: vanilla\n*** End Patch"
        )
        write_event = {
            "op": "APPLY_PATCH",
            "workspace_diff": {
                "added": [version],
                "modified": [],
                "deleted": [],
            },
            "snapshot_files": [version],
        }
        buy_input = (
            "Customer-approved product cards:\n"
            "- Product: Vanilla Cake\n"
            "  Confirmed listed cake flavor: vanilla [SEP] Back to Search "
            "[SEP] Vanilla Cake [SEP] Price: $10"
        )
        ls_event = {
            "op": "SHELL_COMMAND",
            "command": "ls notes",
            "stdout": "fact.md\n",
            "stderr": "",
            "exit_code": 0,
            "workspace_diff": {"added": [], "modified": [], "deleted": []},
            "snapshot_files": [version],
        }
        steps = [
            _step(1, 0, 0, model_text=patch, event=write_event),
            _step(2, 0, 1, model_text="click[Buy Now]", user_text=buy_input),
            _step(
                3,
                1,
                1,
                model_text='shell_command {"command":"ls notes"}',
                event=ls_event,
            ),
            _step(4, 1, 2, model_text="click[Buy Now]"),
        ]
        buys = MODULE._correct_buys(steps)
        sources = MODULE._source_writes(steps, buys)
        links = MODULE._later_shell_links(steps, buys, sources)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["command"], "ls notes")
        self.assertFalse(links[0]["source_content_observed_in_stdout"])
        self.assertFalse(links[0]["strict_content_chain"])

    def test_same_content_read_before_correct_buy_is_strict(self) -> None:
        version = {"path": "notes/fact.md", "sha256": "b" * 64, "bytes": 21}
        patch = (
            "apply_patch\n*** Begin Patch\n*** Add File: notes/fact.md\n"
            "+listed cake flavor: vanilla\n*** End Patch"
        )
        write_event = {
            "op": "APPLY_PATCH",
            "workspace_diff": {
                "added": [version],
                "modified": [],
                "deleted": [],
            },
            "snapshot_files": [version],
        }
        buy_input = (
            "Customer-approved product cards:\n"
            "- Product: Vanilla Cake\n"
            "  Confirmed listed cake flavor: vanilla [SEP] Back to Search "
            "[SEP] Vanilla Cake [SEP] Price: $10"
        )
        cat_event = {
            "op": "SHELL_COMMAND",
            "command": "cat notes/fact.md",
            "stdout": "listed cake flavor: vanilla\n",
            "stderr": "",
            "exit_code": 0,
            "workspace_diff": {"added": [], "modified": [], "deleted": []},
            "snapshot_files": [version],
        }
        steps = [
            _step(1, 0, 0, model_text=patch, event=write_event),
            _step(2, 0, 1, model_text="click[Buy Now]", user_text=buy_input),
            _step(
                3,
                1,
                1,
                model_text='shell_command {"command":"cat notes/fact.md"}',
                event=cat_event,
            ),
            _step(4, 1, 2, model_text="click[Buy Now]"),
        ]
        buys = MODULE._correct_buys(steps)
        sources = MODULE._source_writes(steps, buys)
        links = MODULE._later_shell_links(steps, buys, sources)
        self.assertEqual(sources[0]["semantic_evidence"]["status"], "exact_field_value")
        self.assertEqual(links[0]["command"], "cat notes/fact.md")
        self.assertTrue(links[0]["source_content_observed_in_stdout"])
        self.assertTrue(links[0]["strict_content_chain"])

    def test_markdown_uses_current_counts_and_semantic_statuses(self) -> None:
        summary = {
            "strict_content_chain_count": 0,
            "source_write_action_count": 2,
            "source_write_trajectory_count": 1,
            "trajectory_count": 2,
            "later_shell_timing_action_count": 0,
            "later_shell_timing_link_count": 0,
            "later_shell_source_content_observed_count": 0,
            "apply_patch_mention_count": 2,
            "accepted_apply_patch_count": 2,
            "submitted_apply_patch_prefix_count": 2,
            "shell_command_mention_count": 0,
            "accepted_shell_command_count": 0,
            "submitted_shell_command_prefix_count": 0,
            "post_transition_content_write_count": 0,
            "session1_failure_trajectory_count": 0,
            "wrong_buy_count": 1,
            "source_semantic_status_counts": {"exact_field_value": 2},
        }
        rendered = MODULE.render_markdown(
            {"run_dir": "/tmp/run", "summary": summary, "episodes": []}
        )
        self.assertIn("2 reported source writes are actions", rendered)
        self.assertNotIn("six reported source writes", rendered)
        self.assertIn("Every source-write timing candidate", rendered)
        self.assertNotIn("generic certified/natural labels", rendered)
        self.assertIn("content writes in sessions 2+", rendered)
        self.assertNotIn("preceding session trace", rendered)

    def test_training_post_adv_layout_preserves_exact_io_and_strict_chain(self) -> None:
        version = {"path": ".agent_memory/fact.md", "sha256": "c" * 64, "bytes": 39}
        patch = (
            "apply_patch\n*** Begin Patch\n*** Add File: .agent_memory/fact.md\n"
            "+Confirmed listed cake flavor: vanilla\n*** End Patch"
        )
        buy_input = (
            "Customer-approved product cards:\n"
            "- Product: Vanilla Cake\n"
            "  Confirmed listed cake flavor: vanilla [SEP] Back to Search "
            "[SEP] Vanilla Cake [SEP] Price: $10"
        )

        def row(
            order: int,
            before: int,
            after: int,
            action: str,
            *,
            latest: str = "",
            event: dict[str, object] | None = None,
        ) -> dict[str, object]:
            before_info = {"current_subtask_index": before}
            after_info: dict[str, object] = {
                "current_subtask_index": after,
                "episode_success": after == 2,
                "reward_components": [],
                "workspace_snapshot": {"files": [version] if order > 0 else []},
            }
            if event is not None:
                after_info["workspace_latest_event"] = event
            return {
                "ppo_valid_sample": True,
                "formal_step_record": {
                    "trajectory_uid": "trajectory-0",
                    "trajectory_row_order": order,
                    "trajectory_return": 2.0,
                    "parent_index": 7,
                    "item_id": "agentmemory-7",
                    "visible_prompt": f"exact-visible-prompt-{order}",
                    "latest_observation": latest,
                    "action": action,
                    "action_submission": {
                        "raw_policy_output": action,
                        "submitted_action": action,
                        "parser_status": "raw_fallback",
                    },
                    "env_info_before": before_info,
                    "env_info_after": after_info,
                    "env_result": "Done!",
                    "immediate_reward": 0.0,
                    "done": after == 2,
                    "generation_response_digest": f"digest-{order}",
                    "generation_response_length": 1,
                },
            }

        patch_event = {
            "op": "APPLY_PATCH",
            "workspace_diff": {"added": [version], "modified": [], "deleted": []},
        }
        shell_action = (
            'shell_command {"command":"rg --hidden -n \'^Confirmed \' .",'
            '"workdir":"."}'
        )
        shell_event = {
            "op": "SHELL_COMMAND",
            "command": "rg --hidden -n '^Confirmed ' .",
            "stdout": "./.agent_memory/fact.md:1:Confirmed listed cake flavor: vanilla\n",
            "stderr": "",
            "exit_code": 0,
            "workspace_diff": {"added": [], "modified": [], "deleted": []},
        }
        payload = {
            "global_step": 1,
            "rows": [
                row(0, 0, 0, patch, event=patch_event),
                row(1, 0, 1, "click[Buy Now]", latest=buy_input),
                row(2, 1, 1, shell_action, event=shell_event),
                row(3, 1, 2, "click[Buy Now]"),
                {"ppo_valid_sample": False, "formal_step_record": None},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            diagnostic_dir = run_dir / "diagnostics"
            diagnostic_dir.mkdir()
            (diagnostic_dir / "ppo_batch_step1_post_adv.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            audit = MODULE.analyze_run(run_dir)

        self.assertEqual(audit["summary"]["source_layout"], "training_post_adv")
        self.assertEqual(audit["summary"]["trajectory_count"], 1)
        self.assertEqual(audit["summary"]["strict_content_chain_count"], 1)
        self.assertEqual(
            audit["summary"]["strict_content_chain_count_by_rollout_step"], {"1": 1}
        )
        step_summary = audit["summary"]["summary_by_rollout_step"]["1"]
        self.assertEqual(step_summary["trajectory_count"], 1)
        self.assertEqual(step_summary["strict_content_chain_count"], 1)
        self.assertEqual(step_summary["strict_content_chain_trajectory_count"], 1)
        self.assertEqual(
            step_summary["source_semantic_status_counts"], {"exact_field_value": 1}
        )
        rendered = MODULE.render_markdown(audit)
        self.assertIn("Step 1 (base policy): 1 strict chains in 1/1 trajectories", rendered)
        episode = audit["episodes"][0]
        self.assertEqual(episode["policy_updates_before_rollout"], 0)
        self.assertEqual(episode["filesystem_io"][0]["exact_model_input"], "exact-visible-prompt-0")
        self.assertEqual(
            episode["filesystem_io"][0]["raw_model_response"]["raw_policy_output"],
            patch,
        )
        self.assertEqual(
            episode["filesystem_io"][0]["action_submission"]["submitted_action"],
            patch,
        )


if __name__ == "__main__":
    unittest.main()
