from __future__ import annotations

from copy import deepcopy
import importlib.util
import pickle
import sys
import unittest
from pathlib import Path

try:
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - exercised in the production Torch env.
    DataLoader = None


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "verl/utils/agent_dataset/procedural_index.py"
SPEC = importlib.util.spec_from_file_location(
    "agentmemory_procedural_index_for_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PROVIDER_MODE_FIXED_WINDOW = MODULE.PROVIDER_MODE_FIXED_WINDOW
PROVIDER_MODE_RESEEDED_STREAM = MODULE.PROVIDER_MODE_RESEEDED_STREAM
ProceduralIndexError = MODULE.ProceduralIndexError
ProceduralIndexSource = MODULE.ProceduralIndexSource
TaskBalancedMultitaskIndexSource = MODULE.TaskBalancedMultitaskIndexSource
UniformMultitaskIndexSource = MODULE.UniformMultitaskIndexSource
StatefulProceduralStreamSampler = MODULE.StatefulProceduralStreamSampler
FILESYSTEM_MULTITASK_CYCLE_SIZE = MODULE.FILESYSTEM_MULTITASK_CYCLE_SIZE
FILESYSTEM_MULTITASK_KIND = MODULE.FILESYSTEM_MULTITASK_KIND
FILESYSTEM_MULTITASK_UNIFORM_KIND = MODULE.FILESYSTEM_MULTITASK_UNIFORM_KIND
MULTITASK_LOCAL_DATA_INDEX_KEY = MODULE.MULTITASK_LOCAL_DATA_INDEX_KEY
MULTITASK_LOCAL_TASK_COUNT_KEY = MODULE.MULTITASK_LOCAL_TASK_COUNT_KEY
MULTITASK_ROUTE_KIND_KEY = MODULE.MULTITASK_ROUTE_KIND_KEY
MULTITASK_SAMPLING_SEED_KEY = MODULE.MULTITASK_SAMPLING_SEED_KEY
MULTITASK_SURFACE_SLOT_KEY = MODULE.MULTITASK_SURFACE_SLOT_KEY
build_stream_checkpoint = MODULE.build_stream_checkpoint
generation_non_tensor_keys = MODULE.generation_non_tensor_keys
procedural_index_source_from_config = MODULE.procedural_index_source_from_config
promote_data_idx_for_rollout = MODULE.promote_data_idx_for_rollout
restore_stream_checkpoint = MODULE.restore_stream_checkpoint
validate_paired_batch_indices = MODULE.validate_paired_batch_indices
validate_orbit_batch_indices = MODULE.validate_orbit_batch_indices
validate_multitask_route_triplet = MODULE.validate_multitask_route_triplet
validate_rollout_parent_coverage = MODULE.validate_rollout_parent_coverage


class UnboundedRows:
    def __len__(self) -> int:
        return 8

    def __getitem__(self, position: int) -> int:
        return position


def server_metadata(*, generator_seed: int = 233) -> dict:
    return {
        "surface": "agentmemory_webshop_procedural_natural_chain_train_v1",
        "source": "agentmemory_programmatic_generator",
        "paper_eligible": False,
        "task_count": 64,
        "provider_mode": PROVIDER_MODE_RESEEDED_STREAM,
        "accepted_index_domain": "all_nonnegative_integers",
        "memory_prompt_mode": "neutral",
        "reward_contract": {"correct_purchase": 1.0},
        "provider": {
            "schema": "agentmemory_verified_natural_chain_provider_v4",
            "tasks_per_orbit": 2,
            "provider_mode": PROVIDER_MODE_RESEEDED_STREAM,
            "task_count": 64,
            "accepted_index_domain": "all_nonnegative_integers",
            "candidate_count_per_phase": 2,
            "phase_count_per_task": 6,
            "generator_version": "natural_chain_v3",
            "generator_base_seed": generator_seed,
            "product_pool_sha256": "a" * 64,
            "semantic_period_orbits": 100,
            "semantic_period_tasks": 200,
            "reseeded_stream": {
                "tasks_per_seed_epoch": 200,
                "orbits_per_seed_epoch": 100,
                "counterfactual_pair_never_crosses_seed_epoch": True,
                "seed_epoch_zero_uses_base_seed": True,
                "collision_free_within_complete_seed_epoch": True,
                "semantic_uniqueness_guaranteed_through_task_index": 199,
                "cross_seed_epoch_semantic_uniqueness_guaranteed": False,
            },
            "human_review_required": False,
            "llm_judge_required": False,
            "task_prompt_product_identity": "complete_native_title",
            "target_asin_in_task_prompt": False,
            "native_search_result_asin_handles_visible": True,
            "native_click_action_uses_asin_handle": True,
        },
        "backend": {
            "price_seed": 233,
            "price_table_sha256": "b" * 64,
        },
    }


def filesystem_server_metadata(*, generator_seed: int = 233) -> dict:
    metadata = server_metadata(generator_seed=generator_seed)
    metadata.update(
        {
            "surface": (
                "agentmemory_webshop_procedural_natural_chain_filesystem_v2"
            ),
            "memory_prompt_mode": "natural_filesystem",
            "workspace_surface": "codex_workspace_v2",
            "workspace_tool_contract": "codex_shell_command_apply_patch_v1",
            "workspace_tool_ops": ["SHELL_COMMAND", "APPLY_PATCH"],
            "workspace_shell_enabled": True,
            "workspace_apply_patch_enabled": True,
            "workspace_host_path_exposed": False,
            "source_pairing": "xor_lsb_within_orbit_v1",
            "tasks_per_orbit": 2,
            "workspace_prompt_family": "natural_attribute_chain_filesystem_v2",
            "workspace_seed_contract": "none",
            "workspace_evaluation_contract": (
                "directional_counterfactual_separation_v1"
            ),
            "workspace_intervention_control": {
                "allowed_arms": [
                    "correct",
                    "blank",
                    "swapped",
                    "no_workspace",
                ],
                "boundary_session_index": 1,
                "source_state": "policy_authored_workspace_only",
                "hidden_answer_injection": False,
            },
            "reward_contract": {
                "workspace_action_reward": 0.0,
                "shell_command_reward": 0.0,
                "apply_patch_reward": 0.0,
            },
        }
    )
    return metadata


def latent_preference_server_metadata(*, generator_seed: int = 233) -> dict:
    metadata = server_metadata(generator_seed=generator_seed)
    metadata["surface"] = "agentmemory_webshop_latent_preference_train_v1"
    metadata["memory_prompt_mode"] = "latent_preference_sop"
    metadata["provider"] = {
        **metadata["provider"],
        "schema": "agentmemory_verified_latent_preference_provider_v1",
        "supporting_evidence_counts": [1, 2, 3],
        "resolution_step": 1,
        "preference_hypothesis": "one_value_on_one_natural_attribute_axis",
        "counterfactual_pairing": False,
        "answer_preserving_robustness_pairing": True,
        "application_observation_identity": True,
        "application_target_flip": True,
        "purchase_receipt_asin_verification": True,
    }
    return metadata


def recency_override_server_metadata(*, generator_seed: int = 233) -> dict:
    metadata = server_metadata(generator_seed=generator_seed)
    metadata["surface"] = "agentmemory_webshop_recency_override_train_v1"
    metadata["memory_prompt_mode"] = "latent_preference_sop"
    metadata["provider"] = {
        **metadata["provider"],
        "schema": "agentmemory_verified_recency_override_provider_v1",
        "phase_schedule": [
            "evidence",
            "application",
            "override",
            "application",
            "application",
            "application",
        ],
        "override_phase_index": 2,
        "canonical_memory_key": "user_preference",
        "counterfactual_pairing": True,
        "stay_branch": "old preference remains active",
        "flip_branch": "new preference replaces old canonical state",
        "update_contract": "UPDATE same memory_id or DELETE old then ADD new",
        "application_observation_identity": True,
        "application_target_flip": True,
        "purchase_receipt_asin_verification": True,
    }
    return metadata


def recency_override_filesystem_server_metadata(
    *, generator_seed: int = 233
) -> dict:
    metadata = recency_override_server_metadata(generator_seed=generator_seed)
    filesystem = filesystem_server_metadata(generator_seed=generator_seed)
    metadata.update(
        {
            key: deepcopy(filesystem[key])
            for key in (
                "workspace_surface",
                "workspace_tool_contract",
                "workspace_tool_ops",
                "workspace_shell_enabled",
                "workspace_apply_patch_enabled",
                "workspace_host_path_exposed",
                "source_pairing",
                "tasks_per_orbit",
                "workspace_prompt_family",
                "workspace_evaluation_contract",
                "workspace_intervention_control",
                "reward_contract",
            )
        }
    )
    metadata["surface"] = "agentmemory_webshop_recency_override_filesystem_v2"
    metadata["memory_prompt_mode"] = "natural_filesystem"
    metadata["workspace_prompt_family"] = "recency_override_filesystem_v2"
    metadata["workspace_intervention_control"]["allowed_arms"].insert(
        3,
        "stale",
    )
    metadata["workspace_intervention_control"]["boundary_session_index"] = 3
    return metadata


def distractor_robustness_server_metadata(*, generator_seed: int = 233) -> dict:
    metadata = latent_preference_server_metadata(generator_seed=generator_seed)
    metadata["surface"] = (
        "agentmemory_webshop_distractor_robustness_top1_train_v1"
    )
    metadata["provider"] = {
        **metadata["provider"],
        "schema": "agentmemory_verified_distractor_robustness_provider_v1",
        "counterfactual_pairing": True,
        "branch_order": ["clean", "distracted"],
        "correct_memory_preloaded": False,
        "correct_memory_policy_authored_after_evidence": True,
        "retrieve_policy": "query_top1",
        "memory_id_lookup_allowed": False,
        "initial_memory_inventory_visible": False,
        "strict_top1_certified": True,
        "purchase_receipt_asin_verification": True,
    }
    return metadata


def distractor_robustness_filesystem_server_metadata(
    *, generator_seed: int = 233
) -> dict:
    metadata = distractor_robustness_server_metadata(generator_seed=generator_seed)
    filesystem = filesystem_server_metadata(generator_seed=generator_seed)
    for key in (
        "workspace_surface",
        "workspace_tool_contract",
        "workspace_tool_ops",
        "workspace_shell_enabled",
        "workspace_apply_patch_enabled",
        "workspace_host_path_exposed",
        "source_pairing",
        "tasks_per_orbit",
        "workspace_prompt_family",
        "workspace_seed_contract",
        "workspace_evaluation_contract",
        "workspace_intervention_control",
        "reward_contract",
    ):
        metadata[key] = deepcopy(filesystem[key])
    metadata.update(
        {
            "surface": "agentmemory_webshop_distractor_robustness_filesystem_v2",
            "memory_prompt_mode": "natural_filesystem",
            "source_pairing": "xor_distractor_condition_within_orbit_v1",
            "workspace_prompt_family": "distractor_robustness_filesystem_v2",
            "workspace_seed_contract": (
                "branch_conditioned_ordinary_profile_files_v1"
            ),
            "workspace_intervention_control": {
                "allowed_arms": [
                    "correct",
                    "blank",
                    "no_workspace",
                ],
                "boundary_session_index": 1,
                "source_state": (
                    "policy_authored_current_record_plus_branch_distractors"
                ),
                "hidden_answer_injection": False,
            },
            "workspace_evaluation_contract": "paired_distractor_robustness_v1",
        }
    )
    return metadata


def compositional_recall_server_metadata(*, generator_seed: int = 233) -> dict:
    metadata = latent_preference_server_metadata(generator_seed=generator_seed)
    metadata["surface"] = (
        "agentmemory_webshop_compositional_recall_top1_train_v1"
    )
    provider = metadata["provider"]
    provider.update(
        {
            "schema": "agentmemory_verified_compositional_recall_provider_v1",
            "tasks_per_orbit": 4,
            "semantic_period_tasks": 400,
            "factorial_coordinates": [
                ["token_a", "identity"],
                ["token_a", "swapped"],
                ["token_b", "identity"],
                ["token_b", "swapped"],
            ],
            "canonical_memory_count": 2,
            "retrieve_policy": "query_top1",
            "required_sequential_retrievals": 2,
            "memory_id_lookup_allowed": False,
            "ltm_inventory_visible": False,
            "leave_one_memory_out_certified": True,
            "purchase_receipt_asin_verification": True,
        }
    )
    provider["reseeded_stream"] = {
        **provider["reseeded_stream"],
        "tasks_per_seed_epoch": 400,
        "factorial_orbit_never_crosses_seed_epoch": True,
    }
    provider["reseeded_stream"].pop(
        "counterfactual_pair_never_crosses_seed_epoch"
    )
    provider["reseeded_stream"][
        "semantic_uniqueness_guaranteed_through_task_index"
    ] = 399
    return metadata


def compositional_recall_filesystem_server_metadata(
    *, generator_seed: int = 233
) -> dict:
    metadata = compositional_recall_server_metadata(generator_seed=generator_seed)
    filesystem = filesystem_server_metadata(generator_seed=generator_seed)
    for key in (
        "workspace_surface",
        "workspace_tool_contract",
        "workspace_tool_ops",
        "workspace_shell_enabled",
        "workspace_apply_patch_enabled",
        "workspace_host_path_exposed",
        "reward_contract",
    ):
        metadata[key] = deepcopy(filesystem[key])
    metadata.update(
        {
            "surface": (
                "agentmemory_webshop_compositional_recall_filesystem_v2"
            ),
            "memory_prompt_mode": "natural_filesystem",
            "source_pairing": "xor_lsb_within_orbit_v1",
            "tasks_per_orbit": 4,
            "workspace_prompt_family": "compositional_recall_filesystem_v2",
            "workspace_intervention_control": {
                "allowed_arms": [
                    "correct",
                    "blank",
                    "swapped",
                    "no_workspace",
                ],
                "boundary_session_index": 2,
                "source_state": "policy_authored_workspace_only",
                "hidden_answer_injection": False,
            },
            "workspace_evaluation_contract": (
                "directional_counterfactual_separation_v1"
            ),
        }
    )
    return metadata


def intent_clarification_server_metadata(*, generator_seed: int = 233) -> dict:
    metadata = latent_preference_server_metadata(generator_seed=generator_seed)
    metadata["surface"] = "agentmemory_webshop_intent_clarification_train_v1"
    metadata["provider"] = {
        **metadata["provider"],
        "schema": "agentmemory_verified_intent_clarification_provider_v1",
        "counterfactual_pairing": True,
        "pre_ask_observation_identity": True,
        "all_targets_flip_after_clarification": True,
        "required_action": "ASK",
        "clarification_event": "CLARIFY",
        "ask_allowed_session": 0,
        "max_successful_asks": 1,
        "purchase_before_clarification_allowed": False,
        "canonical_memory_count": 1,
        "retrieve_policy": "query_top1",
        "memory_id_lookup_allowed": False,
        "ltm_inventory_visible": False,
        "training_ready": True,
    }
    return metadata


def selective_memory_use_server_metadata(*, generator_seed: int = 233) -> dict:
    metadata = server_metadata(generator_seed=generator_seed)
    metadata["surface"] = (
        "agentmemory_webshop_selective_memory_use_top1_train_v1"
    )
    metadata["memory_prompt_mode"] = "selective_memory_sop"
    provider = metadata["provider"]
    provider.update(
        {
            "schema": "agentmemory_verified_selective_memory_use_provider_v1",
            "tasks_per_orbit": 4,
            "semantic_period_tasks": 400,
        }
    )
    provider["reseeded_stream"] = {
        **provider["reseeded_stream"],
        "tasks_per_seed_epoch": 400,
        "factorial_orbit_never_crosses_seed_epoch": True,
        "semantic_uniqueness_guaranteed_through_task_index": 399,
    }
    provider["reseeded_stream"].pop(
        "counterfactual_pair_never_crosses_seed_epoch"
    )
    return metadata


def selective_memory_use_filesystem_server_metadata(
    *, generator_seed: int = 233
) -> dict:
    metadata = selective_memory_use_server_metadata(generator_seed=generator_seed)
    filesystem = filesystem_server_metadata(generator_seed=generator_seed)
    for key in (
        "workspace_surface",
        "workspace_tool_contract",
        "workspace_tool_ops",
        "workspace_shell_enabled",
        "workspace_apply_patch_enabled",
        "workspace_host_path_exposed",
        "reward_contract",
    ):
        metadata[key] = deepcopy(filesystem[key])
    metadata.update(
        {
            "surface": "agentmemory_webshop_selective_memory_use_filesystem_v2",
            "memory_prompt_mode": "natural_filesystem",
            "source_pairing": "xor_preference_coordinate_within_factorial_v1",
            "tasks_per_orbit": 4,
            "workspace_prompt_family": "selective_memory_use_filesystem_v2",
            "workspace_seed_contract": (
                "branch_conditioned_initial_profile_files_v1"
            ),
            "workspace_intervention_control": {
                "allowed_arms": [
                    "correct",
                    "blank",
                    "swapped",
                    "no_workspace",
                ],
                "boundary_session_index": 1,
                "source_state": (
                    "harness_seeded_branch_profile_with_optional_policy_edits"
                ),
                "hidden_answer_injection": False,
            },
            "workspace_evaluation_contract": (
                "selective_required_separation_not_required_invariance_v1"
            ),
        }
    )
    return metadata


def negative_constraint_server_metadata(*, generator_seed: int = 233) -> dict:
    metadata = server_metadata(generator_seed=generator_seed)
    metadata["surface"] = "agentmemory_webshop_negative_constraint_top1_train_v1"
    metadata["task_count"] = 72
    metadata["provider"] = {
        **metadata["provider"],
        "schema": "agentmemory_verified_negative_constraint_provider_v1",
        "task_count": 72,
        "virtual_task_count": 72,
        "tasks_per_orbit": 3,
        "candidate_count_per_phase": 3,
        "distinct_values_per_phase": 3,
        "counterfactual_branches": 3,
        "retrieve_policy": "query_top1",
        "memory_id_lookup_allowed": False,
        "initial_memory_inventory_visible": False,
        "native_certified": True,
        "training_ready": True,
        "semantic_period_orbits": 100,
        "semantic_period_tasks": 300,
        "human_review_required": False,
        "llm_judge_required": False,
        "task_prompt_product_identity": "complete_native_title",
        "target_asin_in_task_prompt": False,
        "native_search_result_asin_handles_visible": True,
        "native_click_action_uses_asin_handle": True,
        "reseeded_stream": {
            "tasks_per_seed_epoch": 300,
            "orbits_per_seed_epoch": 100,
            "counterfactual_orbit_never_crosses_seed_epoch": True,
            "seed_epoch_zero_uses_base_seed": True,
            "collision_free_within_complete_seed_epoch": True,
            "semantic_uniqueness_guaranteed_through_task_index": 299,
            "cross_seed_epoch_semantic_uniqueness_guaranteed": False,
        },
    }
    return metadata


def negative_constraint_filesystem_server_metadata(
    *, generator_seed: int = 233
) -> dict:
    metadata = negative_constraint_server_metadata(generator_seed=generator_seed)
    filesystem = filesystem_server_metadata(generator_seed=generator_seed)
    for key in (
        "workspace_surface",
        "workspace_tool_contract",
        "workspace_tool_ops",
        "workspace_shell_enabled",
        "workspace_apply_patch_enabled",
        "workspace_host_path_exposed",
        "reward_contract",
    ):
        metadata[key] = deepcopy(filesystem[key])
    metadata.update(
        {
            "surface": "agentmemory_webshop_negative_constraint_filesystem_v2",
            "memory_prompt_mode": "natural_filesystem",
            "source_pairing": "cyclic_next_within_orbit_v1",
            "tasks_per_orbit": 3,
            "workspace_prompt_family": "negative_constraint_filesystem_v2",
            "workspace_intervention_control": {
                "allowed_arms": [
                    "correct",
                    "blank",
                    "swapped",
                    "no_workspace",
                ],
                "boundary_session_index": 1,
                "source_state": "policy_authored_workspace_only",
                "hidden_answer_injection": False,
            },
            "workspace_evaluation_contract": (
                "directional_counterfactual_separation_v1"
            ),
        }
    )
    return metadata


def _with_filesystem_contract(
    metadata: dict,
    *,
    surface: str,
    prompt_family: str,
    tasks_per_orbit: int,
) -> dict:
    filesystem = filesystem_server_metadata()
    for key in (
        "memory_management",
        "workspace_surface",
        "workspace_tool_contract",
        "workspace_tool_ops",
        "workspace_persistence",
        "workspace_episode_isolation",
        "workspace_shell_enabled",
        "workspace_apply_patch_enabled",
        "workspace_host_path_exposed",
        "workspace_seed_contract",
        "workspace_evaluation_contract",
        "workspace_intervention_control",
        "workspace_limits",
        "workspace_sandbox",
        "reward_contract",
    ):
        if key in filesystem:
            metadata[key] = deepcopy(filesystem[key])
    metadata.update(
        {
            "surface": surface,
            "memory_prompt_mode": "natural_filesystem",
            "source_pairing": "xor_lsb_within_orbit_v1",
            "tasks_per_orbit": tasks_per_orbit,
            "workspace_prompt_family": prompt_family,
        }
    )
    return metadata


def filesystem_multitask_server_metadatas() -> list[dict]:
    return [
        filesystem_server_metadata(),
        _with_filesystem_contract(
            latent_preference_server_metadata(),
            surface="agentmemory_webshop_latent_preference_filesystem_v2",
            prompt_family="latent_preference_filesystem_v2",
            tasks_per_orbit=2,
        ),
        recency_override_filesystem_server_metadata(),
        distractor_robustness_filesystem_server_metadata(),
        compositional_recall_filesystem_server_metadata(),
        negative_constraint_filesystem_server_metadata(),
        _with_filesystem_contract(
            intent_clarification_server_metadata(),
            surface="agentmemory_webshop_intent_clarification_filesystem_v2",
            prompt_family="intent_clarification_filesystem_v2",
            tasks_per_orbit=2,
        ),
        selective_memory_use_filesystem_server_metadata(),
    ]


class ProceduralIndexSourceTests(unittest.TestCase):
    def test_explicit_data_index_is_promoted_for_environment_reset(self) -> None:
        indices = [200_000, 200_001]
        non_tensor_batch = {
            "item_id": ["opaque-a", "opaque-b"],
            "raw_prompt": [[], []],
            "data_idx": indices,
        }
        self.assertEqual(
            generation_non_tensor_keys(non_tensor_batch),
            ["item_id", "raw_prompt", "data_idx"],
        )
        self.assertTrue(promote_data_idx_for_rollout(non_tensor_batch))
        self.assertNotIn("data_idx", non_tensor_batch)
        self.assertIs(non_tensor_batch["rollout_data_indices"], indices)

    def test_legacy_generation_batch_remains_unchanged(self) -> None:
        non_tensor_batch = {"item_id": ["0"], "raw_prompt": [[]]}
        self.assertEqual(
            generation_non_tensor_keys(non_tensor_batch),
            ["item_id", "raw_prompt"],
        )
        self.assertFalse(promote_data_idx_for_rollout(non_tensor_batch))

    def test_task_balanced_multitask_cycle_routes_equal_complete_blocks(self) -> None:
        source = TaskBalancedMultitaskIndexSource(
            task_count=FILESYSTEM_MULTITASK_CYCLE_SIZE * 3,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=FILESYSTEM_MULTITASK_CYCLE_SIZE,
        )
        first_cycle = [
            source.row_for_position(position)
            for position in range(FILESYSTEM_MULTITASK_CYCLE_SIZE)
        ]
        for surface_slot in range(8):
            routed = [
                row
                for row in first_cycle
                if row[MULTITASK_SURFACE_SLOT_KEY] == surface_slot
            ]
            self.assertEqual(len(routed), 12)
            self.assertEqual(
                [row[MULTITASK_LOCAL_DATA_INDEX_KEY] for row in routed],
                list(range(12)),
            )
        next_cycle = source.row_for_position(FILESYSTEM_MULTITASK_CYCLE_SIZE)
        self.assertEqual(next_cycle[MULTITASK_SURFACE_SLOT_KEY], 0)
        self.assertEqual(next_cycle[MULTITASK_LOCAL_DATA_INDEX_KEY], 12)
        self.assertEqual(next_cycle["data_idx"], FILESYSTEM_MULTITASK_CYCLE_SIZE)
        self.assertEqual(
            generation_non_tensor_keys(
                {
                    "item_id": [first_cycle[0]["item_id"]],
                    "raw_prompt": [[]],
                    "data_idx": [0],
                    MULTITASK_SURFACE_SLOT_KEY: [0],
                    MULTITASK_LOCAL_DATA_INDEX_KEY: [0],
                }
            ),
            [
                "item_id",
                "raw_prompt",
                "data_idx",
                MULTITASK_SURFACE_SLOT_KEY,
                MULTITASK_LOCAL_DATA_INDEX_KEY,
            ],
        )

    def test_task_balanced_route_triplets_are_exact_and_fail_closed(self) -> None:
        source = TaskBalancedMultitaskIndexSource(
            task_count=FILESYSTEM_MULTITASK_CYCLE_SIZE * 3,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=FILESYSTEM_MULTITASK_CYCLE_SIZE,
        )
        for position in (0, 11, 12, 47, 95, 96, 191):
            row = source.row_for_position(position)
            self.assertEqual(
                validate_multitask_route_triplet(
                    row["data_idx"],
                    row[MULTITASK_SURFACE_SLOT_KEY],
                    row[MULTITASK_LOCAL_DATA_INDEX_KEY],
                ),
                (
                    row["data_idx"],
                    row[MULTITASK_SURFACE_SLOT_KEY],
                    row[MULTITASK_LOCAL_DATA_INDEX_KEY],
                ),
            )

        for values in (
            (12, 0, 0),
            (12, 1, 1),
            (96, 0, 0),
            (0, 0, 1),
            (0, 0.0, 0),
            (0, "0", 0),
            (0, False, 0),
            (-1, 0, 0),
        ):
            with self.subTest(values=values), self.assertRaises(
                ProceduralIndexError
            ):
                validate_multitask_route_triplet(*values)

    def test_task_balanced_multitask_config_and_batch_are_fail_closed(self) -> None:
        source = procedural_index_source_from_config(
            {
                "procedural_index": {
                    "enabled": True,
                    "kind": FILESYSTEM_MULTITASK_KIND,
                    "task_count": FILESYSTEM_MULTITASK_CYCLE_SIZE * 3,
                    "provider_mode": PROVIDER_MODE_RESEEDED_STREAM,
                }
            }
        )
        self.assertIsInstance(source, TaskBalancedMultitaskIndexSource)
        source.validate_training_batch_size(FILESYSTEM_MULTITASK_CYCLE_SIZE)
        for invalid_batch_size in (64, 192 - 1):
            with self.subTest(batch_size=invalid_batch_size), self.assertRaises(
                ProceduralIndexError
            ):
                source.validate_training_batch_size(invalid_batch_size)
        with self.assertRaisesRegex(ProceduralIndexError, "cycle size"):
            TaskBalancedMultitaskIndexSource(
                task_count=64,
                provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
                tasks_per_orbit=2,
            )

    def test_task_balanced_multitask_attests_every_server_in_order(self) -> None:
        source = TaskBalancedMultitaskIndexSource(
            task_count=FILESYSTEM_MULTITASK_CYCLE_SIZE * 3,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=FILESYSTEM_MULTITASK_CYCLE_SIZE,
        )
        metadatas = filesystem_multitask_server_metadatas()
        source.validate_server_metadatas(metadatas)
        identity = source.training_identity(
            server_metadata=metadatas,
            train_batch_size=FILESYSTEM_MULTITASK_CYCLE_SIZE,
        )
        self.assertTrue(identity["training_geometry"]["task_balanced"])
        self.assertEqual(len(identity["server_metadata"]), 8)
        self.assertEqual(source.required_local_task_count, 36)
        self.assertEqual(
            identity["index_source"]["required_local_task_count"],
            36,
        )

        swapped = deepcopy(metadatas)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaisesRegex(ProceduralIndexError, "route order"):
            source.validate_server_metadatas(swapped)
        with self.assertRaisesRegex(ProceduralIndexError, "each of 8"):
            source.validate_server_metadatas(metadatas[:-1])

        undersized = deepcopy(metadatas)
        undersized[3]["task_count"] = 24
        undersized[3]["provider"]["task_count"] = 24
        with self.assertRaisesRegex(
            ProceduralIndexError,
            "cannot cover the frozen stream",
        ):
            source.validate_server_metadatas(undersized)

    def test_uniform_multitask_config_freezes_batch64_without_orbit_coupling(
        self,
    ) -> None:
        source = procedural_index_source_from_config(
            {
                "procedural_index": {
                    "enabled": True,
                    "kind": FILESYSTEM_MULTITASK_UNIFORM_KIND,
                    "task_count": 6_400,
                    "provider_mode": PROVIDER_MODE_RESEEDED_STREAM,
                    "tasks_per_orbit": 1,
                    "sampling_seed": 17,
                    # Deliberately not divisible by the provider semantic
                    # orbit sizes (2/3/4): learner sampling must not require
                    # a complete counterfactual orbit in any window.
                    "local_task_count": 11,
                }
            }
        )
        self.assertIsInstance(source, UniformMultitaskIndexSource)
        source.validate_training_batch_size(64)
        for invalid_batch_size in (32, 96):
            with self.subTest(batch_size=invalid_batch_size), self.assertRaisesRegex(
                ProceduralIndexError,
                "frozen at 64",
            ):
                source.validate_training_batch_size(invalid_batch_size)
        self.assertFalse(source.metadata()["orbit_members_coupled"])
        self.assertEqual(source.metadata()["source_pool_size"], 88)
        self.assertTrue(source.metadata()["task_balanced_in_expectation"])
        self.assertEqual(
            source.metadata()["sampling_unit"],
            "independent_certified_source_row",
        )
        self.assertFalse(source.metadata()["window_coverage_required"])
        self.assertFalse(
            source.metadata()["counterfactual_window_coverage_required"]
        )
        self.assertEqual(
            source.metadata()["coverage_audit"],
            "posthoc_distribution_only",
        )

    def test_uniform_multitask_accepts_non_orbit_aligned_local_pool(self) -> None:
        source = UniformMultitaskIndexSource(
            task_count=6_400,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=1,
            sampling_seed=17,
            local_task_count=11,
        )
        source.validate_server_metadatas(filesystem_multitask_server_metadatas())
        observed = [
            (
                source.row_for_position(position)[MULTITASK_SURFACE_SLOT_KEY],
                source.row_for_position(position)[MULTITASK_LOCAL_DATA_INDEX_KEY],
            )
            for position in range(88)
        ]
        self.assertEqual(len(set(observed)), 88)

    def test_uniform_multitask_pool_is_a_seeded_permutation_of_independent_rows(
        self,
    ) -> None:
        source = UniformMultitaskIndexSource(
            task_count=64,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=1,
            sampling_seed=17,
            local_task_count=12,
        )
        first_epoch = [source.row_for_position(position) for position in range(96)]
        observed = {
            (
                row[MULTITASK_SURFACE_SLOT_KEY],
                row[MULTITASK_LOCAL_DATA_INDEX_KEY],
            )
            for row in first_epoch
        }
        self.assertEqual(
            observed,
            {
                (surface_slot, local_data_idx)
                for surface_slot in range(8)
                for local_data_idx in range(12)
            },
        )
        self.assertEqual(len(observed), len(first_epoch))
        second_epoch = [
            (
                source.row_for_position(position)[MULTITASK_SURFACE_SLOT_KEY],
                source.row_for_position(position)[MULTITASK_LOCAL_DATA_INDEX_KEY],
            )
            for position in range(96, 192)
        ]
        self.assertNotEqual(
            [
                (
                    row[MULTITASK_SURFACE_SLOT_KEY],
                    row[MULTITASK_LOCAL_DATA_INDEX_KEY],
                )
                for row in first_epoch
            ],
            second_epoch,
        )

    def test_uniform_multitask_route_identity_is_exact_and_fail_closed(
        self,
    ) -> None:
        source = UniformMultitaskIndexSource(
            task_count=64,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=1,
            sampling_seed=17,
            local_task_count=12,
        )
        row = source.row_for_position(37)
        route_kwargs = {
            "route_kind": row[MULTITASK_ROUTE_KIND_KEY],
            "sampling_seed": row[MULTITASK_SAMPLING_SEED_KEY],
            "local_task_count": row[MULTITASK_LOCAL_TASK_COUNT_KEY],
        }
        self.assertEqual(
            validate_multitask_route_triplet(
                row["data_idx"],
                row[MULTITASK_SURFACE_SLOT_KEY],
                row[MULTITASK_LOCAL_DATA_INDEX_KEY],
                **route_kwargs,
            ),
            (
                row["data_idx"],
                row[MULTITASK_SURFACE_SLOT_KEY],
                row[MULTITASK_LOCAL_DATA_INDEX_KEY],
            ),
        )
        with self.assertRaisesRegex(ProceduralIndexError, "seeded global-index"):
            validate_multitask_route_triplet(
                row["data_idx"],
                row[MULTITASK_SURFACE_SLOT_KEY],
                (row[MULTITASK_LOCAL_DATA_INDEX_KEY] + 1) % 12,
                **route_kwargs,
            )
        with self.assertRaises(ProceduralIndexError):
            validate_multitask_route_triplet(
                row["data_idx"],
                row[MULTITASK_SURFACE_SLOT_KEY],
                row[MULTITASK_LOCAL_DATA_INDEX_KEY],
                route_kind=FILESYSTEM_MULTITASK_UNIFORM_KIND,
            )
        self.assertEqual(
            generation_non_tensor_keys(
                {
                    "item_id": [row["item_id"]],
                    "raw_prompt": [[]],
                    "data_idx": [row["data_idx"]],
                    MULTITASK_SURFACE_SLOT_KEY: [
                        row[MULTITASK_SURFACE_SLOT_KEY]
                    ],
                    MULTITASK_LOCAL_DATA_INDEX_KEY: [
                        row[MULTITASK_LOCAL_DATA_INDEX_KEY]
                    ],
                    MULTITASK_ROUTE_KIND_KEY: [row[MULTITASK_ROUTE_KIND_KEY]],
                    MULTITASK_SAMPLING_SEED_KEY: [
                        row[MULTITASK_SAMPLING_SEED_KEY]
                    ],
                    MULTITASK_LOCAL_TASK_COUNT_KEY: [
                        row[MULTITASK_LOCAL_TASK_COUNT_KEY]
                    ],
                }
            ),
            [
                "item_id",
                "raw_prompt",
                "data_idx",
                MULTITASK_SURFACE_SLOT_KEY,
                MULTITASK_LOCAL_DATA_INDEX_KEY,
                MULTITASK_ROUTE_KIND_KEY,
                MULTITASK_SAMPLING_SEED_KEY,
                MULTITASK_LOCAL_TASK_COUNT_KEY,
            ],
        )

    def test_uniform_multitask_attests_servers_and_fixed_compute_identity(
        self,
    ) -> None:
        source = UniformMultitaskIndexSource(
            task_count=6_400,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=1,
            sampling_seed=17,
            local_task_count=12,
        )
        metadatas = filesystem_multitask_server_metadatas()
        source.validate_server_metadatas(metadatas)
        identity = source.training_identity(
            server_metadata=metadatas,
            train_batch_size=64,
        )
        geometry = identity["training_geometry"]
        self.assertEqual(geometry["train_batch_size"], 64)
        self.assertTrue(geometry["fixed_compute_budget"])
        self.assertTrue(geometry["task_balanced_in_expectation"])
        self.assertFalse(geometry["orbit_members_coupled"])

    def test_conflicting_explicit_rollout_indices_fail_closed(self) -> None:
        with self.assertRaisesRegex(ProceduralIndexError, "both data_idx"):
            promote_data_idx_for_rollout(
                {"data_idx": [0], "rollout_data_indices": [1]}
            )

    def test_source_is_lazy_deterministic_and_carries_absolute_index(self) -> None:
        source = ProceduralIndexSource(
            task_count=100_000,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            start_index=200_000,
        )
        self.assertEqual(source.metadata()["materialized_rows"], 0)
        self.assertEqual(
            source.row_for_position(17),
            {
                "item_id": "agentmemory_200017",
                "data_idx": 200_017,
                "extra_info": {"index": 200_017},
            },
        )
        self.assertEqual(source.row_for_position(17), source.row_for_position(17))
        self.assertEqual(
            source.row_for_position(1_000_017)["item_id"],
            "agentmemory_1200017",
        )

    def test_fixed_window_rejects_out_of_range_positions(self) -> None:
        source = ProceduralIndexSource(
            task_count=10,
            provider_mode=PROVIDER_MODE_FIXED_WINDOW,
        )
        self.assertEqual(source.row_for_position(9)["data_idx"], 9)
        with self.assertRaises(IndexError):
            source.row_for_position(10)

    def test_pair_alignment_and_config_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(ProceduralIndexError, "multiple"):
            ProceduralIndexSource(
                task_count=9,
                provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            )
        with self.assertRaisesRegex(ProceduralIndexError, "start_index"):
            ProceduralIndexSource(
                task_count=10,
                provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
                start_index=1,
            )
        self.assertIsNone(procedural_index_source_from_config({}))
        source = procedural_index_source_from_config(
            {
                "procedural_index": {
                    "enabled": True,
                    "task_count": 64,
                    "provider_mode": PROVIDER_MODE_RESEEDED_STREAM,
                }
            }
        )
        self.assertIsNotNone(source)
        self.assertEqual(len(source), 64)  # type: ignore[arg-type]

    def test_training_batch_must_preserve_every_counterfactual_pair(self) -> None:
        source = ProceduralIndexSource(
            task_count=12,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
        )
        source.validate_training_batch_size(4)
        for invalid in (True, 0, 3):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ProceduralIndexError, "complete orbits"
            ):
                source.validate_training_batch_size(invalid)
        with self.assertRaisesRegex(ProceduralIndexError, "divisible"):
            source.validate_training_batch_size(8)

        fixed_window = ProceduralIndexSource(
            task_count=12,
            provider_mode=PROVIDER_MODE_FIXED_WINDOW,
        )
        with self.assertRaisesRegex(ProceduralIndexError, "reseeded_stream"):
            fixed_window.validate_training_batch_size(4)

    def test_server_contract_must_match_dataset_contract(self) -> None:
        source = ProceduralIndexSource(
            task_count=64,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
        )
        metadata = server_metadata()
        source.validate_server_metadata(metadata)
        bad_task_count = deepcopy(metadata)
        bad_task_count["provider"]["task_count"] = 62
        with self.assertRaisesRegex(ProceduralIndexError, "task counts disagree"):
            source.validate_server_metadata(bad_task_count)
        bad_period = deepcopy(metadata)
        bad_period["provider"]["semantic_period_tasks"] = 198
        with self.assertRaisesRegex(ProceduralIndexError, "semantic period"):
            source.validate_server_metadata(bad_period)
        bad_stream = deepcopy(metadata)
        bad_stream["provider"]["reseeded_stream"][
            "tasks_per_seed_epoch"
        ] = 198
        with self.assertRaisesRegex(ProceduralIndexError, "stream epoch"):
            source.validate_server_metadata(bad_stream)
        for field, value, error in (
            ("target_asin_in_task_prompt", True, "target ASIN"),
            (
                "native_search_result_asin_handles_visible",
                False,
                "search-result ASIN handles",
            ),
            (
                "native_click_action_uses_asin_handle",
                False,
                r"click\[ASIN\]",
            ),
        ):
            with self.subTest(field=field):
                bad_native_contract = deepcopy(metadata)
                bad_native_contract["provider"][field] = value
                with self.assertRaisesRegex(ProceduralIndexError, error):
                    source.validate_server_metadata(bad_native_contract)

    def test_server_contract_accepts_only_approved_surface_schema_pairs(self) -> None:
        source = ProceduralIndexSource(
            task_count=64,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
        )
        source.validate_server_metadata(server_metadata())
        source.validate_server_metadata(filesystem_server_metadata())
        source.validate_server_metadata(latent_preference_server_metadata())
        source.validate_server_metadata(recency_override_server_metadata())
        source.validate_server_metadata(recency_override_filesystem_server_metadata())
        source.validate_server_metadata(
            distractor_robustness_filesystem_server_metadata()
        )
        source.validate_server_metadata(distractor_robustness_server_metadata())
        source.validate_server_metadata(intent_clarification_server_metadata())
        compositional_source = ProceduralIndexSource(
            task_count=64,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=4,
        )
        compositional_source.validate_server_metadata(
            compositional_recall_server_metadata()
        )
        compositional_source.validate_server_metadata(
            compositional_recall_filesystem_server_metadata()
        )
        compositional_source.validate_server_metadata(
            selective_memory_use_server_metadata()
        )
        compositional_source.validate_server_metadata(
            selective_memory_use_filesystem_server_metadata()
        )
        negative_source = ProceduralIndexSource(
            task_count=72,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=3,
        )
        negative_source.validate_training_batch_size(72)
        negative_source.validate_server_metadata(
            negative_constraint_server_metadata()
        )
        negative_source.validate_server_metadata(
            negative_constraint_filesystem_server_metadata()
        )

        unknown_surface = latent_preference_server_metadata()
        unknown_surface["surface"] = "agentmemory_webshop_unknown_train_v1"
        with self.assertRaisesRegex(ProceduralIndexError, "unsupported server surface"):
            source.validate_server_metadata(unknown_surface)

        mismatched_schema = latent_preference_server_metadata()
        mismatched_schema["provider"]["schema"] = (
            "agentmemory_verified_natural_chain_provider_v4"
        )
        with self.assertRaisesRegex(ProceduralIndexError, "approved pair"):
            source.validate_server_metadata(mismatched_schema)

        recency_mismatched_schema = recency_override_server_metadata()
        recency_mismatched_schema["provider"]["schema"] = (
            "agentmemory_verified_latent_preference_provider_v1"
        )
        with self.assertRaisesRegex(ProceduralIndexError, "approved pair"):
            source.validate_server_metadata(recency_mismatched_schema)

        recency_filesystem_mismatched_schema = (
            recency_override_filesystem_server_metadata()
        )
        recency_filesystem_mismatched_schema["provider"]["schema"] = (
            "agentmemory_verified_natural_chain_provider_v4"
        )
        with self.assertRaisesRegex(ProceduralIndexError, "approved pair"):
            source.validate_server_metadata(recency_filesystem_mismatched_schema)

        negative_mismatched_schema = negative_constraint_server_metadata()
        negative_mismatched_schema["provider"]["schema"] = (
            "agentmemory_verified_selective_memory_use_provider_v1"
        )
        with self.assertRaisesRegex(ProceduralIndexError, "approved pair"):
            negative_source.validate_server_metadata(negative_mismatched_schema)

    def test_filesystem_server_contract_rejects_prompt_tool_and_reward_drift(self) -> None:
        source = ProceduralIndexSource(
            task_count=64,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
        )
        metadata = filesystem_server_metadata()
        source.validate_server_metadata(metadata)
        mutations = {
            "prompt": lambda value: value.update(memory_prompt_mode="neutral"),
            "tools": lambda value: value.update(workspace_tool_ops=["READ"]),
            "shell": lambda value: value.update(workspace_shell_enabled=False),
            "patch": lambda value: value.update(workspace_apply_patch_enabled=False),
            "contract": lambda value: value.update(workspace_tool_contract="legacy"),
            "shaping": lambda value: value["reward_contract"].update(
                workspace_action_reward=0.1
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                tampered = deepcopy(metadata)
                mutate(tampered)
                with self.assertRaises(ProceduralIndexError):
                    source.validate_server_metadata(tampered)

        legacy = server_metadata()
        legacy["memory_prompt_mode"] = "natural_filesystem"
        with self.assertRaisesRegex(ProceduralIndexError, "bound"):
            source.validate_server_metadata(legacy)

        recency_filesystem = recency_override_filesystem_server_metadata()
        source.validate_server_metadata(recency_filesystem)
        recency_filesystem["workspace_apply_patch_enabled"] = False
        with self.assertRaisesRegex(ProceduralIndexError, "apply_patch"):
            source.validate_server_metadata(recency_filesystem)

        compositional_filesystem = compositional_recall_filesystem_server_metadata()
        compositional_source = ProceduralIndexSource(
            task_count=64,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=4,
        )
        compositional_source.validate_server_metadata(compositional_filesystem)
        compositional_filesystem["source_pairing"] = "cyclic_next_within_orbit_v1"
        with self.assertRaisesRegex(ProceduralIndexError, "source-pairing"):
            compositional_source.validate_server_metadata(compositional_filesystem)

        selective_filesystem = selective_memory_use_filesystem_server_metadata()
        compositional_source.validate_server_metadata(selective_filesystem)
        for field, value in (
            ("workspace_seed_contract", "none"),
            (
                "workspace_intervention_control.source_state",
                "policy_authored_workspace_only",
            ),
        ):
            with self.subTest(selective_tamper=field):
                tampered = deepcopy(selective_filesystem)
                if field == "workspace_seed_contract":
                    tampered[field] = value
                else:
                    tampered["workspace_intervention_control"]["source_state"] = value
                with self.assertRaisesRegex(
                    ProceduralIndexError,
                    "intervention-boundary",
                ):
                    compositional_source.validate_server_metadata(tampered)

        negative_filesystem = negative_constraint_filesystem_server_metadata()
        negative_source = ProceduralIndexSource(
            task_count=72,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
            tasks_per_orbit=3,
        )
        negative_source.validate_server_metadata(negative_filesystem)
        negative_filesystem["workspace_intervention_control"][
            "boundary_session_index"
        ] = 2
        with self.assertRaisesRegex(ProceduralIndexError, "intervention-boundary"):
            negative_source.validate_server_metadata(negative_filesystem)

    def test_batch_indices_preserve_complete_adjacent_orbits(self) -> None:
        validate_paired_batch_indices([200, 201, 202, 203])
        for invalid in (
            [],
            [200],
            [201, 202],
            [200, 202],
            [200, 201, 204, 205],
            [200.9, 201.9],
            ["200", "201"],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(
                ProceduralIndexError
            ):
                validate_paired_batch_indices(invalid)

        validate_orbit_batch_indices(
            [200, 201, 202, 203, 204, 205, 206, 207],
            tasks_per_orbit=4,
        )
        for invalid in ([200, 201], [202, 203, 204, 205], [200, 201, 203, 204]):
            with self.subTest(factorial_invalid=invalid), self.assertRaises(
                ProceduralIndexError
            ):
                validate_orbit_batch_indices(invalid, tasks_per_orbit=4)

    def test_rollout_requires_every_parent_and_replica(self) -> None:
        complete = {
            "rollout_parent_indices": [0, 0, 1, 1, 2, 2, 3, 3],
            "agentmemory_replica_index": [0, 1, 0, 1, 0, 1, 0, 1],
        }
        validate_rollout_parent_coverage(
            complete,
            expected_parent_count=4,
            expected_replicas=2,
        )

        missing_parent = deepcopy(complete)
        missing_parent["rollout_parent_indices"] = [0, 0, 1, 1, 2, 2]
        missing_parent["agentmemory_replica_index"] = [0, 1, 0, 1, 0, 1]
        with self.assertRaisesRegex(ProceduralIndexError, "partial PPO update"):
            validate_rollout_parent_coverage(
                missing_parent,
                expected_parent_count=4,
                expected_replicas=2,
            )

        missing_replica = deepcopy(complete)
        missing_replica["rollout_parent_indices"] = [0, 0, 1, 1, 2, 2, 3]
        missing_replica["agentmemory_replica_index"] = [0, 1, 0, 1, 0, 1, 0]
        with self.assertRaisesRegex(ProceduralIndexError, "partial PPO update"):
            validate_rollout_parent_coverage(
                missing_replica,
                expected_parent_count=4,
                expected_replicas=2,
            )

        with self.assertRaisesRegex(ProceduralIndexError, "not integral"):
            validate_rollout_parent_coverage(
                {
                    "rollout_parent_indices": [0.9],
                    "agentmemory_replica_index": [0.9],
                },
                expected_parent_count=1,
                expected_replicas=1,
            )


class StatefulProceduralStreamSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = ProceduralIndexSource(
            task_count=8,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
        )

    def test_epochs_are_disjoint_contiguous_windows(self) -> None:
        sampler = StatefulProceduralStreamSampler(self.source)
        self.assertEqual(list(sampler), list(range(8)))
        self.assertEqual(list(sampler), list(range(8, 16)))
        self.assertEqual(sampler.next_position, 16)

    def test_pickle_resume_continues_at_exact_cursor(self) -> None:
        sampler = StatefulProceduralStreamSampler(self.source)
        iterator = iter(sampler)
        self.assertEqual([next(iterator) for _ in range(4)], [0, 1, 2, 3])
        restored = pickle.loads(pickle.dumps(sampler))
        self.assertEqual(restored.next_position, 4)
        self.assertEqual(list(restored), list(range(4, 12)))

    @unittest.skipIf(DataLoader is None, "PyTorch is not installed")
    def test_real_dataloader_pickle_resumes_after_a_complete_even_batch(self) -> None:
        sampler = StatefulProceduralStreamSampler(self.source)
        loader = DataLoader(  # type: ignore[operator]
            UnboundedRows(),
            batch_size=4,
            drop_last=True,
            sampler=sampler,
            num_workers=0,
        )
        iterator = iter(loader)
        self.assertEqual(next(iterator).tolist(), [0, 1, 2, 3])
        self.assertEqual(loader.sampler.next_position, 4)

        restored = pickle.loads(pickle.dumps(loader))
        self.assertEqual(restored.sampler.next_position, 4)
        self.assertEqual(next(iter(restored)).tolist(), [4, 5, 6, 7])
        self.assertEqual(restored.sampler.next_position, 8)

    def test_state_load_rejects_unpaired_cursor(self) -> None:
        sampler = StatefulProceduralStreamSampler(self.source)
        with self.assertRaisesRegex(ProceduralIndexError, "next_position"):
            sampler.load_state_dict(
                {
                    "schema": "agentmemory_procedural_stream_sampler_state_v1",
                    "next_position": 3,
                    "samples_per_epoch": 8,
                }
            )

    def test_checkpoint_restores_only_an_identity_bound_cursor(self) -> None:
        source = ProceduralIndexSource(
            task_count=64,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
        )
        identity = source.training_identity(
            server_metadata=server_metadata(),
            train_batch_size=4,
        )
        sampler = StatefulProceduralStreamSampler(source)
        iterator = iter(sampler)
        self.assertEqual([next(iterator) for _ in range(4)], [0, 1, 2, 3])
        checkpoint = build_stream_checkpoint(sampler, identity)

        restored = StatefulProceduralStreamSampler(source)
        restore_stream_checkpoint(restored, identity, checkpoint)
        self.assertEqual(restored.next_position, 4)
        self.assertEqual([next(iter(restored)) for _ in range(2)], [4, 5])

    def test_checkpoint_rejects_changed_server_or_training_identity(self) -> None:
        source = ProceduralIndexSource(
            task_count=64,
            provider_mode=PROVIDER_MODE_RESEEDED_STREAM,
        )
        original = source.training_identity(
            server_metadata=server_metadata(generator_seed=233),
            train_batch_size=4,
        )
        checkpoint = build_stream_checkpoint(
            StatefulProceduralStreamSampler(source),
            original,
        )

        changed_seed = source.training_identity(
            server_metadata=server_metadata(generator_seed=234),
            train_batch_size=4,
        )
        with self.assertRaisesRegex(ProceduralIndexError, "does not match"):
            restore_stream_checkpoint(
                StatefulProceduralStreamSampler(source),
                changed_seed,
                checkpoint,
            )

        tampered = deepcopy(checkpoint)
        tampered["stream_identity"]["server_metadata"]["memory_prompt_mode"] = (
            "legacy"
        )
        with self.assertRaisesRegex(ProceduralIndexError, "digest is invalid"):
            restore_stream_checkpoint(
                StatefulProceduralStreamSampler(source),
                original,
                tampered,
            )


if __name__ == "__main__":
    unittest.main()
