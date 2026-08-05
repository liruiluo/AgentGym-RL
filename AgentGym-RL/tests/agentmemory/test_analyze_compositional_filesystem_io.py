from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "agentmemory"
    / "analyze_compositional_filesystem_io.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_compositional_filesystem_io", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


CUSTOMER = "shopper.train.0123456789abcdef"
TOKEN_A = "pt.0123456789abcdef"
TOKEN_B = "pt.fedcba9876543210"


def _io(phase: int, text: str) -> dict[str, object]:
    return {
        "phase_before": phase,
        "exact_model_input": text,
        "request_messages": [{"role": "user", "content": text}],
    }


def _episode() -> dict[str, object]:
    mapping_source = {
        "turn": 1,
        "phase": 0,
        "files": [
            {
                "path": "mapping.txt",
                "content": f"Customer-to-profile: {CUSTOMER} -> {TOKEN_A}\n",
            }
        ],
    }
    directory_source = {
        "turn": 7,
        "phase": 1,
        "files": [
            {
                "path": "directory.txt",
                "content": (
                    f"Profile-directory: {TOKEN_A} color red; "
                    f"{TOKEN_B} color black\n"
                ),
            }
        ],
    }
    mapping_input = (
        f"Customer profile: {CUSTOMER}\n"
        f"The customer's active shopping profile token is {TOKEN_A}."
    )
    directory_input = (
        "The current profile directory is:\n"
        f"- {TOKEN_A}: color is red\n"
        f"- {TOKEN_B}: color is black\n"
    )
    links = [
        {
            "source_turn": 1,
            "source_phase": 0,
            "shell_turn": 12,
            "shell_phase": 2,
            "source_content_observed_in_stdout": True,
            "later_correct_buy_turn": 15,
        },
        {
            "source_turn": 7,
            "source_phase": 1,
            "shell_turn": 13,
            "shell_phase": 2,
            "source_content_observed_in_stdout": True,
            "later_correct_buy_turn": 15,
        },
    ]
    return {
        "episode_path": "diagnostics/step1.json#trajectory-0",
        "rollout_step": 1,
        "policy_updates_before_rollout": 0,
        "trajectory_uid": "trajectory-0",
        "data_idx": 0,
        "final_phase_progress": 3,
        "episode_success": False,
        "filesystem_io": [_io(0, mapping_input), _io(1, directory_input)],
        "source_write_timing_candidates": [mapping_source, directory_source],
        "later_shell_timing_links": links,
    }


class AnalyzeCompositionalFilesystemIoTest(unittest.TestCase):
    def test_two_shell_reads_before_same_buy_form_one_strict_chain(self) -> None:
        audit = {
            "schema": "agentmemory_filesystem_exact_io_audit_v1",
            "run_dir": "/tmp/run",
            "episodes": [_episode()],
        }
        result = MODULE.analyze_audit(audit)
        summary = result["summary"]
        self.assertEqual(summary["exact_two_source_hop_trajectory_count"], 1)
        self.assertEqual(summary["strict_two_hop_chain_count"], 1)
        chain = result["episodes"][0]["strict_two_hop_chains"][0]
        self.assertEqual(chain["shell_turns"], [12, 13])
        self.assertEqual(chain["later_correct_buy_turn"], 15)
        self.assertEqual(
            result["episodes"][0]["exact_directory_sources"][0]["files"][0][
                "path"
            ],
            "directory.txt",
        )

    def test_collapsed_direct_preference_is_not_an_exact_mapping(self) -> None:
        episode = _episode()
        episode["source_write_timing_candidates"][0]["files"][0]["content"] = (
            f"Customer-to-profile: {CUSTOMER} -> {TOKEN_A}; color red\n"
        )
        result = MODULE.analyze_audit(
            {
                "schema": "agentmemory_filesystem_exact_io_audit_v1",
                "run_dir": "/tmp/run",
                "episodes": [episode],
            }
        )
        self.assertEqual(result["summary"]["exact_mapping_source_trajectory_count"], 0)
        self.assertEqual(result["summary"]["strict_two_hop_chain_count"], 0)

    def test_markdown_states_natural_two_read_boundary(self) -> None:
        result = MODULE.analyze_audit(
            {
                "schema": "agentmemory_filesystem_exact_io_audit_v1",
                "run_dir": "/tmp/run",
                "episodes": [_episode()],
            }
        )
        rendered = MODULE.render_markdown(result)
        self.assertIn("one or multiple shell commands", rendered)
        self.assertIn("single-task RL admission gate", rendered)
        self.assertIn("Strict two-hop chains: 1", rendered)


if __name__ == "__main__":
    unittest.main()
