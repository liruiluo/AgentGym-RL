from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    from verl.utils.agent_dataset.procedural_index import (
        FILESYSTEM_MULTITASK_CYCLE_SIZE,
        PROVIDER_MODE_RESEEDED_STREAM,
        TaskBalancedMultitaskIndexSource,
        UniformMultitaskIndexSource,
        WebshopSwesmithFamilyBalancedIndexSource,
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

    def test_build_messages_prefers_canonical_policy_framing(self) -> None:
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
                {"value": "legacy-slot-0-user"},
                {"value": "legacy-slot-0-assistant"},
            ]
        )
        dataset.multitask_conversation_starts = [
            [
                {"value": f"legacy-slot-{slot}-user"},
                {"value": f"legacy-slot-{slot}-assistant"},
            ]
            for slot in range(8)
        ]
        dataset.policy_framing = [
            {"role": "system", "content": "canonical-slot-0"}
        ]
        dataset.multitask_policy_framings = [
            [{"role": "system", "content": f"canonical-slot-{slot}"}]
            for slot in range(8)
        ]

        row = source.row_for_position(37)
        messages, rendered = dataset._build_messages(row)
        slot = row["agentmemory_surface_slot"]

        self.assertEqual(
            messages,
            [{"role": "system", "content": f"canonical-slot-{slot}"}],
        )
        self.assertIn(f"canonical-slot-{slot}", rendered)
        self.assertNotIn("legacy-slot", rendered)
        self.assertNotIn("Ok.", rendered)

    def test_build_messages_routes_webshop_and_swesmith_policy_framing(self) -> None:
        source = WebshopSwesmithFamilyBalancedIndexSource(
            task_count=64,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=1,
            sampling_seed=17,
            webshop_local_task_count=11,
            swesmith_local_task_count=8,
        )
        dataset = RLHFDataset.__new__(RLHFDataset)
        dataset.procedural_index_source = source
        dataset.env_client = SimpleNamespace(
            conversation_start=[
                {"value": "legacy-slot-0-user"},
                {"value": "legacy-slot-0-assistant"},
            ]
        )
        dataset.multitask_conversation_starts = [
            [
                {"value": f"legacy-slot-{slot}-user"},
                {"value": f"legacy-slot-{slot}-assistant"},
            ]
            for slot in range(9)
        ]
        dataset.policy_framing = [
            {"role": "system", "content": "webshop-slot-0"}
        ]
        dataset.multitask_policy_framings = [
            [
                {
                    "role": "system",
                    "content": (
                        f"webshop-slot-{slot}"
                        if slot < 8
                        else "swesmith-coding-policy"
                    ),
                }
            ]
            for slot in range(9)
        ]

        rows = [source.row_for_position(position) for position in range(64)]
        webshop_row = next(
            row for row in rows if row["agentmemory_surface_slot"] < 8
        )
        swesmith_row = next(
            row for row in rows if row["agentmemory_surface_slot"] == 8
        )
        webshop_messages, _ = dataset._build_messages(webshop_row)
        swesmith_messages, _ = dataset._build_messages(swesmith_row)

        self.assertEqual(webshop_row["data_source"], "agentmemory")
        self.assertEqual(swesmith_row["data_source"], "swesmith")
        self.assertTrue(
            webshop_messages[0]["content"].startswith("webshop-slot-")
        )
        self.assertEqual(
            swesmith_messages,
            [{"role": "system", "content": "swesmith-coding-policy"}],
        )


if __name__ == "__main__":
    unittest.main()
