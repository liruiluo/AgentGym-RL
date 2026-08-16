from __future__ import annotations

from dataclasses import replace
import unittest

from test_paired_eval_support import Arm, make_config, with_arm

from paired_eval.contracts import (
    AMG_COMPACTION_ONLY_CAPABILITY,
    AMG_MEMORY_CAPABILITY,
    EXTERNAL_MEMORY_CAPABILITY_SURFACES,
    NATIVE_CAPABILITY,
    PairKey,
    capability_for_arm,
)


class PairedEvalContractTest(unittest.TestCase):
    def test_pair_key_excludes_arm_exactly(self) -> None:
        native = make_config(arm=Arm.NATIVE)
        compaction_only = with_arm(native, Arm.AMG_COMPACTION_ONLY)
        memory = with_arm(native, Arm.AMG_MEMORY)

        self.assertEqual(native.pair_key, compaction_only.pair_key)
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

    def test_configs_differ_only_by_exact_capability_lattice(self) -> None:
        native = make_config(arm=Arm.NATIVE)
        compaction_only = with_arm(native, Arm.AMG_COMPACTION_ONLY)
        memory = with_arm(native, Arm.AMG_MEMORY)
        configs = (native, compaction_only, memory)

        for left in configs:
            for right in configs:
                differing = {
                    key
                    for key in left.to_payload()
                    if left.to_payload()[key] != right.to_payload()[key]
                }
                if left.capability.arm is right.capability.arm:
                    self.assertEqual(differing, set())
                else:
                    self.assertEqual(differing, {"capability"})
                self.assertEqual(
                    left.treatment_excluded_config_sha256,
                    right.treatment_excluded_config_sha256,
                )

        self.assertEqual(native.capability, NATIVE_CAPABILITY)
        self.assertEqual(
            compaction_only.capability,
            AMG_COMPACTION_ONLY_CAPABILITY,
        )
        self.assertEqual(memory.capability, AMG_MEMORY_CAPABILITY)
        self.assertEqual(
            [
                (
                    config.capability.policy_authored_compaction,
                    config.capability.external_read_write_memory,
                )
                for config in configs
            ],
            [(False, False), (True, False), (True, True)],
        )
        self.assertEqual(compaction_only.capability.tools, ())
        self.assertEqual(compaction_only.capability.external_memory_surfaces, ())
        self.assertNotIn("WRITE", compaction_only.capability.prompt_declaration)
        self.assertNotIn("READ", compaction_only.capability.prompt_declaration)
        self.assertEqual(
            memory.capability.tools,
            ("external_memory_read", "external_memory_write"),
        )
        self.assertEqual(
            memory.capability.prompt_declaration,
            "adapter_owned_external_memory_declaration_v1",
        )
        self.assertEqual(
            memory.capability.external_memory_surfaces,
            EXTERNAL_MEMORY_CAPABILITY_SURFACES,
        )
        self.assertFalse(memory.capability.implicit_retrieval)
        self.assertFalse(memory.capability.hidden_context_injection)
        self.assertEqual(
            native.capability.allowed_routes,
            (("benchmark_task", "benchmark_task"),),
        )
        self.assertEqual(
            compaction_only.capability.allowed_routes,
            (
                ("benchmark_task", "benchmark_task"),
                ("policy_compaction", "policy_context"),
            ),
        )
        self.assertEqual(
            memory.capability.allowed_routes,
            (
                ("benchmark_task", "benchmark_task"),
                ("external_memory", "external_memory"),
                ("policy_compaction", "policy_context"),
            ),
        )

    def test_compaction_contract_explicitly_matches_between_enabled_arms(self) -> None:
        compaction_only = make_config(arm=Arm.AMG_COMPACTION_ONLY)
        memory = with_arm(compaction_only, Arm.AMG_MEMORY)

        self.assertEqual(compaction_only.compaction, memory.compaction)
        self.assertEqual(
            {
                "trigger": compaction_only.compaction.trigger,
                "summary_instruction_sha256": (
                    compaction_only.compaction.summary_instruction_sha256
                ),
                "context_pressure_policy_sha256": (
                    compaction_only.compaction.context_pressure_policy_sha256
                ),
                "context_transition_schema": (
                    compaction_only.compaction.context_transition_schema
                ),
                "action_accounting": compaction_only.compaction.action_accounting,
            },
            {
                "trigger": memory.compaction.trigger,
                "summary_instruction_sha256": (
                    memory.compaction.summary_instruction_sha256
                ),
                "context_pressure_policy_sha256": (
                    memory.compaction.context_pressure_policy_sha256
                ),
                "context_transition_schema": memory.compaction.context_transition_schema,
                "action_accounting": memory.compaction.action_accounting,
            },
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
        for surface in EXTERNAL_MEMORY_CAPABILITY_SURFACES:
            with self.subTest(surface=surface), self.assertRaises(ValueError):
                replace(
                    capability_for_arm(Arm.AMG_COMPACTION_ONLY),
                    external_memory_surfaces=(surface,),
                )
        with self.assertRaises(ValueError):
            replace(
                capability_for_arm(Arm.AMG_MEMORY),
                external_memory_surfaces=("mount",),
            )


if __name__ == "__main__":
    unittest.main()
