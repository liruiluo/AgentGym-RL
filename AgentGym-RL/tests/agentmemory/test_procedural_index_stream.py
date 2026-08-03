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
StatefulProceduralStreamSampler = MODULE.StatefulProceduralStreamSampler
build_stream_checkpoint = MODULE.build_stream_checkpoint
generation_non_tensor_keys = MODULE.generation_non_tensor_keys
procedural_index_source_from_config = MODULE.procedural_index_source_from_config
promote_data_idx_for_rollout = MODULE.promote_data_idx_for_rollout
restore_stream_checkpoint = MODULE.restore_stream_checkpoint
validate_paired_batch_indices = MODULE.validate_paired_batch_indices
validate_orbit_batch_indices = MODULE.validate_orbit_batch_indices
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
        "counterfactual_pairing": True,
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
            "factorial_coordinates": [[0, 0], [0, 1], [1, 0], [1, 1]],
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
        source.validate_server_metadata(latent_preference_server_metadata())
        source.validate_server_metadata(recency_override_server_metadata())
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
            selective_memory_use_server_metadata()
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

        negative_mismatched_schema = negative_constraint_server_metadata()
        negative_mismatched_schema["provider"]["schema"] = (
            "agentmemory_verified_selective_memory_use_provider_v1"
        )
        with self.assertRaisesRegex(ProceduralIndexError, "approved pair"):
            negative_source.validate_server_metadata(negative_mismatched_schema)

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
