from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "agentmemory"
    / "attest_effective_memory_prompt.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "attest_effective_memory_prompt_for_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EffectiveMemoryPromptAttestationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.lifecycle_prompt = " ".join(self.module.LIFECYCLE_SOP_FRAGMENTS)
        self.latent_preference_prompt = " ".join(
            (
                *self.module.LIFECYCLE_SOP_FRAGMENTS,
                *self.module.LATENT_PREFERENCE_SOP_FRAGMENTS,
            )
        )

    def test_lifecycle_prompt_passes_and_records_hash(self) -> None:
        result = self.module.build_attestation(
            prompt=self.lifecycle_prompt,
            memory_prompt_mode="legacy",
            ltm_inventory_mode="keys",
            thinking_enabled=False,
            reasoning_enabled=True,
            require_lifecycle_sop=True,
        )
        self.assertTrue(result["lifecycle_sop_present"])
        self.assertEqual(result["missing_lifecycle_sop_fragments"], [])
        self.assertEqual(result["memory_prompt_mode"], "legacy")
        self.assertEqual(len(result["system_prompt_sha256"]), 64)

    def test_neutral_prompt_fails_required_lifecycle_gate(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "missing the required memory lifecycle SOP"
        ):
            self.module.build_attestation(
                prompt="Across shopping sessions, preserve facts needed later.",
                memory_prompt_mode="neutral_horizon_responsibility",
                ltm_inventory_mode="keys",
                thinking_enabled=False,
                reasoning_enabled=True,
                require_lifecycle_sop=True,
            )

    def test_each_required_fragment_is_fail_closed(self) -> None:
        for missing_fragment in self.module.LIFECYCLE_SOP_FRAGMENTS:
            with self.subTest(missing_fragment=missing_fragment):
                prompt = " ".join(
                    fragment
                    for fragment in self.module.LIFECYCLE_SOP_FRAGMENTS
                    if fragment != missing_fragment
                )
                with self.assertRaisesRegex(RuntimeError, "missing the required"):
                    self.module.build_attestation(
                        prompt=prompt,
                        memory_prompt_mode="legacy",
                        ltm_inventory_mode="keys",
                        thinking_enabled=False,
                        reasoning_enabled=True,
                        require_lifecycle_sop=True,
                    )

    def test_neutral_prompt_can_be_attested_without_sop_requirement(self) -> None:
        result = self.module.build_attestation(
            prompt="Across shopping sessions, preserve facts needed later.",
            memory_prompt_mode="neutral_horizon_responsibility",
            ltm_inventory_mode="keys",
            thinking_enabled=False,
            reasoning_enabled=True,
            require_lifecycle_sop=False,
        )
        self.assertFalse(result["lifecycle_sop_present"])
        self.assertTrue(result["missing_lifecycle_sop_fragments"])

    def test_latent_preference_prompt_passes_both_fail_closed_gates(self) -> None:
        result = self.module.build_attestation(
            prompt=self.latent_preference_prompt,
            memory_prompt_mode="latent_preference_sop",
            ltm_inventory_mode="keys",
            thinking_enabled=False,
            reasoning_enabled=True,
            require_lifecycle_sop=True,
            require_latent_preference_sop=True,
        )
        self.assertTrue(result["lifecycle_sop_present"])
        self.assertTrue(result["latent_preference_sop_present"])
        self.assertEqual(result["missing_latent_preference_sop_fragments"], [])
        self.assertTrue(result["require_latent_preference_sop"])

    def test_generic_lifecycle_prompt_fails_latent_preference_gate(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "latent-preference SOP"):
            self.module.build_attestation(
                prompt=self.lifecycle_prompt,
                memory_prompt_mode="latent_preference_sop",
                ltm_inventory_mode="keys",
                thinking_enabled=False,
                reasoning_enabled=True,
                require_lifecycle_sop=True,
                require_latent_preference_sop=True,
            )

    def test_each_latent_preference_fragment_is_fail_closed(self) -> None:
        for missing_fragment in self.module.LATENT_PREFERENCE_SOP_FRAGMENTS:
            with self.subTest(missing_fragment=missing_fragment):
                prompt = " ".join(
                    (
                        *self.module.LIFECYCLE_SOP_FRAGMENTS,
                        *(
                            fragment
                            for fragment in self.module.LATENT_PREFERENCE_SOP_FRAGMENTS
                            if fragment != missing_fragment
                        ),
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "latent-preference SOP"):
                    self.module.build_attestation(
                        prompt=prompt,
                        memory_prompt_mode="latent_preference_sop",
                        ltm_inventory_mode="keys",
                        thinking_enabled=False,
                        reasoning_enabled=True,
                        require_lifecycle_sop=True,
                        require_latent_preference_sop=True,
                    )
