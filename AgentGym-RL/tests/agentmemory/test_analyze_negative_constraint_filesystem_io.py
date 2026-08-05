from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "agentmemory"
    / "analyze_negative_constraint_filesystem_io.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_negative_constraint_filesystem_io", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _episode(*, later_buy: int | None = 8) -> dict[str, object]:
    source = {
        "turn": 1,
        "phase": 0,
        "files": [
            {
                "path": "rules.txt",
                "content": "Standing exclusions: geometric, solid pattern\n",
            }
        ],
    }
    link = {
        "source_turn": 1,
        "source_phase": 0,
        "shell_turn": 6,
        "shell_phase": 1,
        "source_content_observed_in_stdout": True,
        "later_correct_buy_turn": later_buy,
    }
    return {
        "episode_path": "diagnostics/step1.json#trajectory-0",
        "rollout_step": 1,
        "policy_updates_before_rollout": 0,
        "trajectory_uid": "trajectory-0",
        "data_idx": 0,
        "final_phase_progress": 2 if later_buy is not None else 1,
        "episode_success": False,
        "source_write_timing_candidates": [source],
        "later_shell_timing_links": [link],
        "filesystem_io": [
            {
                "env_response": {
                    "info": {"branch_kind": "allow_floral"}
                }
            }
        ],
    }


class AnalyzeNegativeConstraintFilesystemIoTest(unittest.TestCase):
    def test_exact_rule_requires_axis_both_forbidden_and_no_allowed_value(self) -> None:
        kwargs = {
            "axis": "pattern",
            "allowed": "floral",
            "forbidden": ("geometric", "solid"),
        }
        self.assertTrue(
            MODULE._contains_exact_rule(
                ["Standing exclusions: geometric, solid pattern\n"], **kwargs
            )
        )
        self.assertFalse(
            MODULE._contains_exact_rule(
                ["Standing exclusions: geometric, solid\n"], **kwargs
            )
        )
        self.assertFalse(
            MODULE._contains_exact_rule(
                ["Standing exclusions: floral, geometric, solid pattern\n"], **kwargs
            )
        )

    def test_strict_chain_requires_later_correct_buy(self) -> None:
        audit = {
            "schema": "agentmemory_filesystem_exact_io_audit_v1",
            "run_dir": "/tmp/run",
            "episodes": [_episode()],
        }
        result = MODULE.analyze_audit(audit)
        self.assertEqual(result["summary"]["exact_source_rule_trajectory_count"], 1)
        self.assertEqual(result["summary"]["exact_rule_readback_trajectory_count"], 1)
        self.assertEqual(result["summary"]["strict_exclusion_chain_count"], 1)
        self.assertEqual(
            result["episodes"][0]["exact_source_rules"][0]["files"][0]["path"],
            "rules.txt",
        )
        self.assertEqual(
            result["summary_by_rollout_step"]["1"][
                "strict_exclusion_chain_trajectory_count"
            ],
            1,
        )

        no_buy = MODULE.analyze_audit(
            {**audit, "episodes": [_episode(later_buy=None)]}
        )
        self.assertEqual(no_buy["summary"]["exact_rule_readback_count"], 1)
        self.assertEqual(no_buy["summary"]["strict_exclusion_chain_count"], 0)

    def test_markdown_preserves_single_task_claim_boundary(self) -> None:
        result = MODULE.analyze_audit(
            {
                "schema": "agentmemory_filesystem_exact_io_audit_v1",
                "run_dir": "/tmp/run",
                "episodes": [_episode()],
            }
        )
        rendered = MODULE.render_markdown(result)
        self.assertIn("single-task RL admission gate", rendered)
        self.assertIn("not a held-out or multitask endpoint", rendered)
        self.assertIn("Strict exclusion chains: 1", rendered)


if __name__ == "__main__":
    unittest.main()
