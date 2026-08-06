from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    from verl.utils.agent_dataset.procedural_index import (
        FILESYSTEM_MULTITASK_CYCLE_SIZE,
        PROVIDER_MODE_RESEEDED_STREAM,
        TaskBalancedMultitaskIndexSource,
        UniformMultitaskIndexSource,
    )
    from verl.utils.agent_dataset.rl_dataset import RLHFDataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on minimal hosts
    IMPORT_ERROR = exc
    RLHFDataset = None
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    RLHFDataset is None,
    f"full RL dataset runtime is unavailable: {IMPORT_ERROR}",
)
class MultitaskDatasetPromptRoutingTests(unittest.TestCase):
    def test_build_messages_uses_each_rows_exact_surface_prompt(self) -> None:
        source = TaskBalancedMultitaskIndexSource(
            task_count=FILESYSTEM_MULTITASK_CYCLE_SIZE,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=FILESYSTEM_MULTITASK_CYCLE_SIZE,
        )
        dataset = RLHFDataset.__new__(RLHFDataset)
        dataset.procedural_index_source = source
        dataset.env_client = SimpleNamespace(
            conversation_start=[
                {"value": "slot-0-user"},
                {"value": "slot-0-assistant"},
            ]
        )
        dataset.multitask_conversation_starts = [
            [
                {"value": f"slot-{slot}-user"},
                {"value": f"slot-{slot}-assistant"},
            ]
            for slot in range(8)
        ]

        observed_user_prompts = []
        for slot in range(8):
            row = source.row_for_position(slot * 12)
            messages, rendered = dataset._build_messages(row)

            expected_user = f"slot-{slot}-user"
            expected_assistant = f"slot-{slot}-assistant"
            self.assertEqual(row["data_source"], "agentmemory")
            self.assertEqual(
                messages,
                [
                    {"role": "user", "content": expected_user},
                    {"role": "assistant", "content": expected_assistant},
                ],
            )
            self.assertIn(expected_user, rendered)
            self.assertIn(expected_assistant, rendered)
            if slot:
                self.assertNotIn("slot-0-user", rendered)
                self.assertNotIn("slot-0-assistant", rendered)
            observed_user_prompts.append(messages[0]["content"])

        self.assertEqual(len(set(observed_user_prompts)), 8)

    def test_build_messages_routes_uniform_rows_by_seeded_surface_slot(self) -> None:
        source = UniformMultitaskIndexSource(
            task_count=64,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=1,
            sampling_seed=17,
            local_task_count=12,
        )
        dataset = RLHFDataset.__new__(RLHFDataset)
        dataset.procedural_index_source = source
        dataset.env_client = SimpleNamespace(
            conversation_start=[
                {"value": "slot-0-user"},
                {"value": "slot-0-assistant"},
            ]
        )
        dataset.multitask_conversation_starts = [
            [
                {"value": f"slot-{slot}-user"},
                {"value": f"slot-{slot}-assistant"},
            ]
            for slot in range(8)
        ]

        row = source.row_for_position(37)
        messages, rendered = dataset._build_messages(row)
        slot = row["agentmemory_surface_slot"]
        self.assertEqual(messages[0]["content"], f"slot-{slot}-user")
        self.assertEqual(messages[1]["content"], f"slot-{slot}-assistant")
        self.assertIn(f"slot-{slot}-user", rendered)


if __name__ == "__main__":
    unittest.main()
