from __future__ import annotations

import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "verl/utils/agentgym/formal_domain_v3.py"
SPEC = importlib.util.spec_from_file_location("formal_domain_v3_for_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

SYSTEM_PROMPT = "SERVER DOMAIN PROMPT"
LATEST_OBSERVATION = "latest"
VISIBLE_PROMPT = f"{SYSTEM_PROMPT}\n{LATEST_OBSERVATION}"


def filesystem_contract_metadata(*, surface=None):
    workspace_limits = {
        "max_path_chars": 240,
        "max_files": 64,
        "max_directories": 64,
        "max_file_bytes": 65_536,
        "max_total_bytes": 524_288,
        "max_command_chars": 32_768,
        "max_patch_bytes": 262_144,
        "default_timeout_ms": 10_000,
        "max_timeout_ms": 30_000,
        "cpu_seconds": 10,
        "address_space_bytes": 1_073_741_824,
        "max_processes": 32,
        "max_open_files": 64,
        "stdout_bytes": 16_384,
        "stderr_bytes": 16_384,
        "tmp_bytes": 67_108_864,
        "tmp_inodes": 512,
    }
    resource_limits = {
        name: workspace_limits[name]
        for name in MODULE._WORKSPACE_SANDBOX_SHARED_LIMIT_FIELDS
    }
    resource_limits.update(
        {
            "workspace_bytes": workspace_limits["max_total_bytes"],
            "workspace_inodes": (
                workspace_limits["max_files"]
                + workspace_limits["max_directories"]
                + 1
            ),
        }
    )
    return {
        "surface": surface or MODULE.FORMAL_WEBSHOP_FILESYSTEM_SURFACE_V2,
        "paper_eligible": False,
        "memory_prompt_mode": "natural_filesystem",
        "memory_management": "policy_managed_persistent_workspace",
        "workspace_surface": "codex_workspace_v2",
        "workspace_tool_contract": "codex_shell_command_apply_patch_v1",
        "workspace_persistence": "episode_across_sessions",
        "workspace_episode_isolation": True,
        "workspace_shell_enabled": True,
        "workspace_apply_patch_enabled": True,
        "workspace_host_path_exposed": False,
        "workspace_tool_ops": ["SHELL_COMMAND", "APPLY_PATCH"],
        "workspace_limits": workspace_limits,
        "workspace_sandbox": {
            **MODULE._WORKSPACE_SANDBOX_FIELDS,
            "ripgrep_sha256": "c" * 64,
            "ripgrep_expected_sha256": "c" * 64,
            "ripgrep_version": "ripgrep 15.1.0",
            "ripgrep_startup_fingerprint": {
                "device": 1,
                "inode": 2,
                "mode": 33_237,
                "size": 5_000_000,
                "mtime_ns": 1,
                "ctime_ns": 1,
            },
            "resource_limits": resource_limits,
        },
        "reward_contract": {
            "workspace_action_reward": 0.0,
            "shell_command_reward": 0.0,
            "apply_patch_reward": 0.0,
            "memory_specific_shaping": "none",
        },
    }


def env_info(*, phase=0, reward=0.0, done=False):
    return {
        "formal_schema_version": "agentmemory_formal_step_v3",
        "domain_id": "fake",
        "surface": "fake_v3",
        "contract_id": "fake_v1",
        "contract_sha256": "a" * 64,
        "phase_index": phase,
        "phase_count": 2,
        "episode_success": False,
        "done": done,
        "action_execution": {
            "raw_policy_output": "Action: ADVANCE {}",
            "submitted_action": "ADVANCE {}",
            "op": "ADVANCE",
            "status": "executed",
            "step": 1,
        },
        "tool_ops": [{"op": "ADVANCE", "step": 1}],
        "reward_components": [
            {"name": "phase_advance", "value": reward, "op": "ADVANCE", "step": 1}
        ],
        "domain_evidence": {"fixture": True},
        "sample_excluded": False,
    }


def generation_record():
    return {
        "response_token_count": 3,
        "max_response_tokens": 64,
        "finish_reason": "stop",
        "finish_reason_source": "vllm",
        "stop_reason": None,
        "backend_source": "vllm",
        "configured_eos_token_ids": [1],
        "primary_eos_token_id": 1,
        "tokenizer_pad_token_id": 0,
        "token_ids_are_exact": True,
        "backend_token_ids_are_exact": True,
        "truncated": False,
    }


def build_record(
    *,
    reward=1.0,
    phase_after=1,
    done=False,
    before_updates=None,
    after_updates=None,
    generation_updates=None,
):
    before = env_info(phase=0, reward=0.0)
    before["reward_components"] = []
    before["action_execution"] = {}
    after = env_info(phase=phase_after, reward=reward, done=done)
    if before_updates:
        before.update(before_updates)
    if after_updates:
        after.update(after_updates)
    sampled_generation = generation_record()
    if generation_updates:
        sampled_generation.update(generation_updates)
    return MODULE.build_formal_domain_step_v3(
        content="Action: ADVANCE {}",
        score=reward,
        task_round=1,
        done=done,
        item_id="0",
        parent_index=0,
        parent_group_uid="parent",
        replica_index=0,
        trajectory_uid="trajectory",
        exact_state_uid="state",
        prompt_token_ids=[11, 12],
        response_token_ids=[21, 22, 23],
        latest_observation=LATEST_OBSERVATION,
        visible_prompt=VISIBLE_PROMPT,
        system_prompt=SYSTEM_PROMPT,
        single_observation_prompt_digest="b" * 64,
        env_result="domain result",
        generation_record=sampled_generation,
        env_info_before=before,
        env_info_after=after,
    )


class FormalDomainV3Test(unittest.TestCase):
    def test_builds_domain_neutral_record(self):
        record = build_record()
        self.assertEqual(record["schema_version"], "agentmemory_formal_step_v3")
        self.assertTrue(record["phase_advanced"])
        self.assertNotIn("purchase_correct", record)
        self.assertNotIn("search_result_count", record)

    def test_builder_rejects_non_boolean_protocol_inputs(self):
        cases = {
            "step done": {"done": 0},
            "before episode_success": {
                "before_updates": {"episode_success": 0}
            },
            "after episode_success": {
                "after_updates": {"episode_success": 0}
            },
            "before sample_excluded": {
                "before_updates": {"sample_excluded": 0}
            },
            "after sample_excluded": {
                "after_updates": {"sample_excluded": 0}
            },
            "after done": {"after_updates": {"done": 0}},
            "generation exactness": {
                "generation_updates": {"token_ids_are_exact": 1}
            },
            "backend exactness": {
                "generation_updates": {"backend_token_ids_are_exact": 1}
            },
            "truncated": {"generation_updates": {"truncated": 0}},
        }
        for label, kwargs in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    MODULE.FormalDomainV3Error,
                    "must be boolean",
                ):
                    build_record(**kwargs)

    def test_validator_rejects_truthy_and_falsy_non_boolean_fields(self):
        invalid_values = {
            "done": 0,
            "phase_advanced": 1,
            "episode_success": 0,
            "raw_prior_messages_visible": 0,
            "generation_token_ids_are_exact": 1,
            "backend_token_ids_are_exact": 1,
            "truncated": 0,
            "sample_excluded": 0,
        }
        for field, invalid in invalid_values.items():
            with self.subTest(field=field):
                record = build_record()
                record[field] = invalid
                with self.assertRaisesRegex(
                    MODULE.FormalDomainV3Error,
                    f"{field} must be boolean",
                ):
                    MODULE.validate_formal_domain_step_v3(record)

    def test_rejects_reward_ledger_mismatch(self):
        record = build_record()
        record["reward_components"][0]["value"] = 0.5
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "ledger sum"):
            MODULE.validate_formal_domain_step_v3(record)

    def test_rejects_raw_policy_output_mismatch(self):
        record = build_record()
        record["action_execution"]["raw_policy_output"] = "different"
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "sampled content"):
            MODULE.validate_formal_domain_step_v3(record)

    def test_accepts_empty_sample_as_authoritative_invalid_action(self):
        record = build_record(reward=0.0, phase_after=0)
        execution = {
            "raw_policy_output": "",
            "submitted_action": "",
            "op": "INVALID",
            "status": "invalid",
            "step": 1,
        }
        record["content"] = ""
        record["action_execution"] = deepcopy(execution)
        record["env_info_after"]["action_execution"] = deepcopy(execution)
        MODULE.validate_formal_domain_step_v3(record)

    def test_rejects_empty_sample_for_non_invalid_action(self):
        record = build_record(reward=0.0, phase_after=0)
        record["content"] = ""
        record["action_execution"]["raw_policy_output"] = ""
        record["action_execution"]["submitted_action"] = ""
        record["env_info_after"]["action_execution"] = deepcopy(
            record["action_execution"]
        )
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "only for an INVALID"):
            MODULE.validate_formal_domain_step_v3(record)

    def test_rejects_phase_jump(self):
        before = env_info(phase=0, reward=0.0)
        before["reward_components"] = []
        before["action_execution"] = {}
        after = env_info(phase=2, reward=1.0)
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "advance exactly once"):
            MODULE.build_formal_domain_step_v3(
                content="Action: ADVANCE {}",
                score=1.0,
                task_round=1,
                done=False,
                item_id="0",
                parent_index=0,
                parent_group_uid="parent",
                replica_index=0,
                trajectory_uid="trajectory",
                exact_state_uid="state",
                prompt_token_ids=[11],
                response_token_ids=[21],
                latest_observation=LATEST_OBSERVATION,
                visible_prompt=VISIBLE_PROMPT,
                system_prompt=SYSTEM_PROMPT,
                single_observation_prompt_digest="b" * 64,
                env_result="domain result",
                generation_record=generation_record(),
                env_info_before=before,
                env_info_after=after,
            )

    def test_binds_generic_timeout_additively(self):
        record = build_record(reward=0.0, phase_after=0)
        record["task_round"] = 3
        record["action_execution"]["step"] = 3
        record["env_info_after"]["action_execution"]["step"] = 3
        record["reward_components"][0]["step"] = 3
        record["env_info_after"]["reward_components"][0]["step"] = 3
        MODULE.bind_generic_timeout_v3(
            record,
            max_policy_turns=3,
            penalty=-0.01,
        )
        self.assertTrue(record["done"])
        self.assertEqual(record["outcome"], "terminal_failure")
        self.assertAlmostEqual(record["score"], -0.01)
        self.assertEqual(
            record["reward_components"][-1]["name"],
            "policy_turn_ceiling_failure",
        )

    def test_validation_does_not_mutate_record(self):
        record = build_record()
        before = deepcopy(record)
        MODULE.validate_formal_domain_step_v3(record)
        self.assertEqual(record, before)

    def test_v3_contract_uses_exact_server_prompt(self):
        metadata = {
            "formal_schema_version": MODULE.FORMAL_DOMAIN_SCHEMA_V3,
            "surface": "fake_v3",
            "system_prompt": "  canonical prompt with boundary whitespace  ",
        }
        schema, prompt, source = MODULE.resolve_formal_runtime_contract(
            metadata,
            webshop_v2_system_prompt="webshop prompt",
        )
        self.assertEqual(schema, MODULE.FORMAL_DOMAIN_SCHEMA_V3)
        self.assertEqual(prompt, metadata["system_prompt"])
        self.assertEqual(source, "server_metadata")

    def test_v3_contract_rejects_missing_server_prompt(self):
        metadata = {
            "formal_schema_version": MODULE.FORMAL_DOMAIN_SCHEMA_V3,
            "surface": "fake_v3",
        }
        with self.assertRaisesRegex(
            MODULE.FormalDomainV3Error,
            "requires a non-empty system_prompt",
        ):
            MODULE.resolve_formal_runtime_contract(
                metadata,
                webshop_v2_system_prompt="webshop prompt",
            )

    def test_v2_contract_accepts_only_recognized_webshop_surfaces(self):
        for surface in (
            MODULE.FORMAL_WEBSHOP_SURFACE_V2,
            MODULE.FORMAL_WEBSHOP_PROCEDURAL_SURFACE_V2,
            MODULE.FORMAL_WEBSHOP_FILESYSTEM_SURFACE_V2,
            MODULE.FORMAL_WEBSHOP_LATENT_PREFERENCE_SURFACE_V2,
            MODULE.FORMAL_WEBSHOP_RECENCY_OVERRIDE_SURFACE_V2,
            MODULE.FORMAL_WEBSHOP_RECENCY_OVERRIDE_FILESYSTEM_SURFACE_V2,
            MODULE.FORMAL_WEBSHOP_DISTRACTOR_ROBUSTNESS_SURFACE_V2,
            MODULE.FORMAL_WEBSHOP_COMPOSITIONAL_RECALL_SURFACE_V2,
            MODULE.FORMAL_WEBSHOP_INTENT_CLARIFICATION_SURFACE_V2,
            MODULE.FORMAL_WEBSHOP_SELECTIVE_MEMORY_USE_SURFACE_V2,
        ):
            with self.subTest(surface=surface):
                schema, prompt, source = MODULE.resolve_formal_runtime_contract(
                    {"surface": surface},
                    webshop_v2_system_prompt="webshop prompt",
                )
                self.assertEqual(schema, MODULE.FORMAL_WEBSHOP_SCHEMA_V2)
                self.assertEqual(prompt, "webshop prompt")
                self.assertEqual(source, "rollout_webshop_v2")

        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "WebShop v2"):
            MODULE.resolve_formal_runtime_contract(
                {"surface": "travel_v3"},
                webshop_v2_system_prompt="webshop prompt",
            )

    def test_webshop_ltm_inventory_mode_requires_prompt_server_parity(self):
        MODULE.validate_webshop_ltm_inventory_mode(
            {"ltm_inventory_mode": "keys"},
            expected_mode="keys",
        )
        MODULE.validate_webshop_ltm_inventory_mode({}, expected_mode="hidden")

        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "disagree"):
            MODULE.validate_webshop_ltm_inventory_mode(
                {"ltm_inventory_mode": "keys"},
                expected_mode="hidden",
            )
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "missing"):
            MODULE.validate_webshop_ltm_inventory_mode({}, expected_mode="keys")

    def test_webshop_memory_prompt_mode_requires_server_rollout_parity(self):
        MODULE.validate_webshop_memory_prompt_mode(
            {"memory_prompt_mode": "neutral"},
            expected_mode="neutral",
        )
        MODULE.validate_webshop_memory_prompt_mode({}, expected_mode="legacy")
        MODULE.validate_webshop_memory_prompt_mode(
            {"memory_prompt_mode": "neutral_horizon"},
            expected_mode="neutral_horizon",
        )
        MODULE.validate_webshop_memory_prompt_mode(
            {"memory_prompt_mode": "neutral_horizon_responsibility"},
            expected_mode="neutral_horizon_responsibility",
        )
        MODULE.validate_webshop_memory_prompt_mode(
            {"memory_prompt_mode": "latent_preference_sop"},
            expected_mode="latent_preference_sop",
        )

        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "disagree"):
            MODULE.validate_webshop_memory_prompt_mode(
                {"memory_prompt_mode": "neutral"},
                expected_mode="legacy",
            )
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "missing"):
            MODULE.validate_webshop_memory_prompt_mode(
                {},
                expected_mode="neutral",
            )

    def test_filesystem_surface_requires_exact_workspace_contract(self):
        metadata = filesystem_contract_metadata()
        MODULE.validate_webshop_filesystem_surface(
            metadata,
            expected_prompt_mode="natural_filesystem",
        )

        tampered = deepcopy(metadata)
        tampered["workspace_shell_enabled"] = False
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "contract mismatch"):
            MODULE.validate_webshop_filesystem_surface(
                tampered,
                expected_prompt_mode="natural_filesystem",
            )

        tampered = deepcopy(metadata)
        tampered["workspace_apply_patch_enabled"] = False
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "contract mismatch"):
            MODULE.validate_webshop_filesystem_surface(
                tampered,
                expected_prompt_mode="natural_filesystem",
            )

        tampered = deepcopy(metadata)
        tampered["workspace_tool_contract"] = "legacy"
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "contract mismatch"):
            MODULE.validate_webshop_filesystem_surface(
                tampered,
                expected_prompt_mode="natural_filesystem",
            )

        tampered = deepcopy(metadata)
        tampered["reward_contract"]["workspace_action_reward"] = 0.1
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "nonzero"):
            MODULE.validate_webshop_filesystem_surface(
                tampered,
                expected_prompt_mode="natural_filesystem",
            )

        sandbox_tampering = {
            "missing sandbox": lambda item: item.pop("workspace_sandbox"),
            "host network": lambda item: item["workspace_sandbox"].__setitem__(
                "network", "host"
            ),
            "shared uid": lambda item: item["workspace_sandbox"].__setitem__(
                "model_identity", "shared"
            ),
            "ripgrep mismatch": lambda item: item["workspace_sandbox"].__setitem__(
                "ripgrep_expected_sha256", "d" * 64
            ),
            "resource mismatch": lambda item: item["workspace_sandbox"][
                "resource_limits"
            ].__setitem__("max_processes", 31),
        }
        for label, mutate in sandbox_tampering.items():
            with self.subTest(label=label):
                tampered = deepcopy(metadata)
                mutate(tampered)
                with self.assertRaises(MODULE.FormalDomainV3Error):
                    MODULE.validate_webshop_filesystem_surface(
                        tampered,
                        expected_prompt_mode="natural_filesystem",
                    )

        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "only valid"):
            MODULE.validate_webshop_filesystem_surface(
                {
                    "surface": MODULE.FORMAL_WEBSHOP_PROCEDURAL_SURFACE_V2,
                    "memory_prompt_mode": "natural_filesystem",
                },
                expected_prompt_mode="natural_filesystem",
            )

    def test_recency_filesystem_surface_uses_shared_workspace_validator(self):
        metadata = filesystem_contract_metadata(
            surface=MODULE.FORMAL_WEBSHOP_RECENCY_OVERRIDE_FILESYSTEM_SURFACE_V2,
        )
        MODULE.validate_webshop_filesystem_surface(
            metadata,
            expected_prompt_mode="natural_filesystem",
        )
        tampered = deepcopy(metadata)
        tampered["workspace_tool_ops"] = ["SHELL_COMMAND"]
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "exactly"):
            MODULE.validate_webshop_filesystem_surface(
                tampered,
                expected_prompt_mode="natural_filesystem",
            )

    def test_webshop_action_listing_mode_requires_server_rollout_parity(self):
        MODULE.validate_webshop_action_listing_mode(
            {"action_listing_mode": "unified"},
            expected_mode="unified",
        )
        MODULE.validate_webshop_action_listing_mode({}, expected_mode="separate")

        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "disagree"):
            MODULE.validate_webshop_action_listing_mode(
                {"action_listing_mode": "unified"},
                expected_mode="separate",
            )
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "missing"):
            MODULE.validate_webshop_action_listing_mode(
                {},
                expected_mode="unified",
            )

    def test_schema_mismatch_fails_closed(self):
        with self.assertRaisesRegex(MODULE.FormalDomainV3Error, "does not match"):
            MODULE.validate_formal_env_schema(
                MODULE.FORMAL_DOMAIN_SCHEMA_V3,
                {"formal_schema_version": MODULE.FORMAL_WEBSHOP_SCHEMA_V2},
                boundary="pre-action",
            )

    def test_rejects_authoritative_action_drift(self):
        record = build_record()
        record["action_execution"]["submitted_action"] = "different"
        with self.assertRaisesRegex(
            MODULE.FormalDomainV3Error,
            "authoritative env_info_after",
        ):
            MODULE.validate_formal_domain_step_v3(record)


if __name__ == "__main__":
    unittest.main()
