from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "verl/utils/agentgym/context_policy.py"
SPEC = importlib.util.spec_from_file_location("context_policy_for_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ContextPolicyCompactionTests(unittest.TestCase):
    def test_agentmemory_compaction_requires_explicit_diagnostic_flag(self) -> None:
        config = {
            "task_name": "agentmemory",
            "rollout_context_policy": "policy_authored_compaction",
        }
        with self.assertRaisesRegex(RuntimeError, "diagnostic-only"):
            MODULE.assert_rollout_context_supported(config)

    def test_agentmemory_compaction_is_allowed_when_explicitly_marked(self) -> None:
        config = {
            "task_name": "agentmemory",
            "rollout_context_policy": "policy_authored_compaction",
            "allow_policy_authored_compaction_for_agentmemory": True,
        }
        MODULE.assert_rollout_context_supported(config)

    def test_latest_observation_default_remains_unchanged(self) -> None:
        config = {"task_name": "agentmemory"}
        MODULE.assert_rollout_context_supported(config)
        self.assertEqual(
            MODULE.rollout_context_policy(config), "latest_observation_only"
        )


if __name__ == "__main__":
    unittest.main()
