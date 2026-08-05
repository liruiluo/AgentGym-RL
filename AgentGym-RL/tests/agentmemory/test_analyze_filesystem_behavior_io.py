from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
