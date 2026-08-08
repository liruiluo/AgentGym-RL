from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ROLLOUT = (
    ROOT
    / "verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py"
)


class SharedRolloutActiveSourceTest(unittest.TestCase):
    def test_rollout_has_one_task_neutral_entrypoint(self) -> None:
        source = ROLLOUT.read_text(encoding="utf-8")
        forbidden = (
            "generate_agentmemory_latest_observation",
            "_build_formal_webshop_step_v2",
            "task_name == \"agentmemory\"",
            "FORMAL_WEBSHOP_SCHEMA_V2",
        )
        hits = [marker for marker in forbidden if marker in source]
        self.assertEqual(
            hits,
            [],
            "active shared rollout still contains domain-specific paths: "
            + ", ".join(hits),
        )
        self.assertIn("item_id=raw_item_id", source)
        self.assertNotIn("item_id=parsed_item_id", source)


if __name__ == "__main__":
    unittest.main()
