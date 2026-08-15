from __future__ import annotations

from dataclasses import replace
import unittest

from test_paired_eval_support import Arm, make_config, with_arm

from paired_eval.contracts import (
    AMG_MEMORY_CAPABILITY,
    NATIVE_CAPABILITY,
    PairKey,
    capability_for_arm,
)


class PairedEvalContractTest(unittest.TestCase):
    def test_pair_key_excludes_arm_exactly(self) -> None:
        native = make_config(arm=Arm.NATIVE)
        memory = with_arm(native, Arm.AMG_MEMORY)

        self.assertEqual(native.pair_key, memory.pair_key)
        self.assertEqual(
            PairKey.from_config(native).to_payload(),
            {
                "run_id": "paired-test-run",
                "benchmark": "fake_benchmark",
                "protocol": "fake_protocol@1",
                "task_id": "task-001",
                "seed": 7,
            },
        )
        self.assertNotIn("arm", PairKey.from_config(native).to_payload())

    def test_configs_differ_only_by_frozen_combined_capability(self) -> None:
        native = make_config(arm=Arm.NATIVE)
        memory = with_arm(native, Arm.AMG_MEMORY)
        native_payload = native.to_payload()
        memory_payload = memory.to_payload()
        differing = {
            key
            for key in native_payload
            if native_payload[key] != memory_payload[key]
        }

        self.assertEqual(differing, {"capability"})
        self.assertEqual(
            native.treatment_excluded_config_sha256,
            memory.treatment_excluded_config_sha256,
        )
        self.assertNotEqual(native.full_config_sha256, memory.full_config_sha256)
        self.assertEqual(native.capability, NATIVE_CAPABILITY)
        self.assertEqual(memory.capability, AMG_MEMORY_CAPABILITY)
        self.assertFalse(native.capability.policy_authored_compaction)
        self.assertFalse(native.capability.external_read_write_memory)
        self.assertTrue(memory.capability.policy_authored_compaction)
        self.assertTrue(memory.capability.external_read_write_memory)
        self.assertEqual(
            memory.capability.tools,
            ("WRITE(key,value)", "READ(key)"),
        )
        self.assertFalse(memory.capability.implicit_retrieval)
        self.assertFalse(memory.capability.hidden_context_injection)
        self.assertEqual(
            native.capability.allowed_routes,
            (("benchmark_task", "benchmark_task"),),
        )
        self.assertEqual(
            memory.capability.allowed_routes,
            (
                ("benchmark_task", "benchmark_task"),
                ("external_memory", "external_memory"),
                ("policy_compaction", "policy_context"),
            ),
        )

    def test_invalid_capability_combinations_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            replace(
                capability_for_arm(Arm.NATIVE),
                external_read_write_memory=True,
            )
        with self.assertRaises(ValueError):
            replace(
                capability_for_arm(Arm.AMG_MEMORY),
                hidden_context_injection=True,
            )


if __name__ == "__main__":
    unittest.main()
