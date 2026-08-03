from __future__ import annotations

import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from agentenv.controller.types import ActionFormat
from agentenv.envs.agentmemory import (
    AgentMemoryAdapter,
    AgentMemoryEnvClient,
    build_procedural_conversation_start,
)


def procedural_metadata():
    return {
        "surface": "agentmemory_webshop_procedural_natural_chain_train_v1",
        "source": "agentmemory_programmatic_generator",
        "paper_eligible": False,
        "task_count": 64,
        "provider_mode": "reseeded_stream",
        "accepted_index_domain": "all_nonnegative_integers",
        "memory_prompt_mode": "neutral",
        "provider": {
            "schema": "agentmemory_verified_natural_chain_provider_v4",
            "tasks_per_orbit": 2,
            "provider_mode": "reseeded_stream",
            "task_count": 64,
            "accepted_index_domain": "all_nonnegative_integers",
            "candidate_count_per_phase": 2,
            "phase_count_per_task": 6,
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
    }


def latent_preference_metadata():
    metadata = procedural_metadata()
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


def recency_override_metadata():
    metadata = latent_preference_metadata()
    metadata["surface"] = "agentmemory_webshop_recency_override_train_v1"
    metadata["task_count"] = 10_000
    metadata["provider_mode"] = "fixed_window"
    metadata["accepted_index_domain"] = "0_to_9999_inclusive"
    metadata["provider"] = {
        **metadata["provider"],
        "schema": "agentmemory_verified_recency_override_provider_v1",
        "provider_mode": "fixed_window",
        "task_count": 10_000,
        "accepted_index_domain": "0_to_9999_inclusive",
        "semantic_period_orbits": 131_072,
        "semantic_period_tasks": 262_144,
        "reseeded_stream": None,
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


def distractor_robustness_metadata():
    metadata = latent_preference_metadata()
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


def compositional_recall_metadata():
    metadata = latent_preference_metadata()
    metadata["surface"] = (
        "agentmemory_webshop_compositional_recall_top1_train_v1"
    )
    metadata["provider"].update(
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
    metadata["provider"]["reseeded_stream"] = {
        **metadata["provider"]["reseeded_stream"],
        "tasks_per_seed_epoch": 400,
        "factorial_orbit_never_crosses_seed_epoch": True,
        "semantic_uniqueness_guaranteed_through_task_index": 399,
    }
    metadata["provider"]["reseeded_stream"].pop(
        "counterfactual_pair_never_crosses_seed_epoch"
    )
    return metadata


def intent_clarification_metadata():
    metadata = latent_preference_metadata()
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


def selective_memory_use_metadata():
    metadata = procedural_metadata()
    metadata["surface"] = (
        "agentmemory_webshop_selective_memory_use_top1_train_v1"
    )
    metadata["memory_prompt_mode"] = "selective_memory_sop"
    metadata["provider"] = {
        **metadata["provider"],
        "schema": "agentmemory_verified_selective_memory_use_provider_v1",
        "tasks_per_orbit": 4,
        "semantic_period_tasks": 400,
        "memory_required_fraction": 0.5,
        "memory_not_required_fraction": 0.5,
        "required_branch_seeded_memory_state": "current",
        "not_required_branch_seeded_memory_state": "stale_opposite",
        "retrieve_policy": "query_top1",
        "memory_id_lookup_allowed": False,
        "ltm_inventory_visible": False,
        "memory_action_positive_shaping_allowed": False,
        "unnecessary_memory_action_penalty": -0.01,
        "memory_required_without_memory_counterfactually_ambiguous": True,
        "memory_not_required_current_request_explicit": True,
        "purchase_receipt_asin_verification": True,
    }
    metadata["provider"]["reseeded_stream"] = {
        **metadata["provider"]["reseeded_stream"],
        "tasks_per_seed_epoch": 400,
        "factorial_orbit_never_crosses_seed_epoch": True,
        "semantic_uniqueness_guaranteed_through_task_index": 399,
    }
    metadata["provider"]["reseeded_stream"].pop(
        "counterfactual_pair_never_crosses_seed_epoch"
    )
    return metadata


class AgentMemoryClientActionSubmissionTest(unittest.TestCase):
    def client(
        self,
        submitted: list[str],
        *,
        step_response: dict | None = None,
    ) -> AgentMemoryEnvClient:
        client = AgentMemoryEnvClient.__new__(AgentMemoryEnvClient)
        client.is_v3 = False
        client.action_format = ActionFormat.REACT
        client.adapter_cls = AgentMemoryAdapter
        client.metadata = {}

        response = step_response or {
            "observation": "next",
            "reward": 0.0,
            "done": False,
            "info": {},
        }

        def post(path, payload):
            self.assertEqual(path, "step")
            submitted.append(payload["action"])
            return deepcopy(response)

        client.post = post
        return client

    def test_records_react_output_and_submitted_native_action(self):
        submitted: list[str] = []
        client = self.client(submitted)
        raw_output = "Thought:\ncheck the page\n\nAction:\nclick[Buy Now]"

        client.step(raw_output)

        self.assertEqual(submitted, ["click[Buy Now]"])
        self.assertEqual(
            client.last_action_submission,
            {
                "raw_policy_output": raw_output,
                "submitted_action": "click[Buy Now]",
                "parser_status": "adapter_parsed",
            },
        )

    def test_records_raw_fallback_for_unsupported_wrapper(self):
        submitted: list[str] = []
        client = self.client(submitted)
        raw_output = '{"action": "click[Buy Now]"}'

        client.step(raw_output)

        self.assertEqual(submitted, [raw_output])
        self.assertEqual(
            client.last_action_submission,
            {
                "raw_policy_output": raw_output,
                "submitted_action": raw_output,
                "parser_status": "raw_fallback",
            },
        )

    def test_records_eos_only_empty_raw_fallback(self):
        submitted: list[str] = []
        client = self.client(submitted)

        client.step("")

        self.assertEqual(submitted, [""])
        self.assertEqual(
            client.last_action_submission,
            {
                "raw_policy_output": "",
                "submitted_action": "",
                "parser_status": "raw_fallback",
            },
        )

    def test_raw_fallback_removes_only_one_terminal_textual_eos(self):
        submitted: list[str] = []
        client = self.client(submitted)

        client.step("</s></s>")

        self.assertEqual(submitted, ["</s>"])
        self.assertEqual(
            client.last_action_submission,
            {
                "raw_policy_output": "</s></s>",
                "submitted_action": "</s>",
                "parser_status": "raw_fallback",
            },
        )

    def test_formal_buy_evidence_is_internal_and_not_part_of_model_state(self):
        submitted: list[str] = []
        client = self.client(
            submitted,
            step_response={
                "observation": "Purchase recorded. The next shopping session is ready.",
                "reward": 1.0,
                "done": False,
                "info": {
                    "tool_ops": [
                        {
                            "op": "BUY",
                            "step": 3,
                            "committed": True,
                            "purchase_correct": True,
                            "session_advanced": True,
                            "terminal": False,
                        }
                    ]
                },
            },
        )

        output = client.step("Thought: done\nAction: click[Buy Now]")

        self.assertEqual(
            output.state,
            "Purchase recorded. The next shopping session is ready.",
        )
        self.assertNotIn("purchase_correct", output.state)
        self.assertTrue(
            client.info["env_info"]["tool_ops"][0]["purchase_correct"]
        )


class ProceduralAgentMemoryClientContractTest(unittest.TestCase):
    def create_client(
        self,
        metadata: dict,
        *,
        action_format: ActionFormat = ActionFormat.REACT,
    ) -> AgentMemoryEnvClient:
        create_response = Mock(status_code=200)
        create_response.json.return_value = {
            "id": 17,
            "observation": "programmatic reset",
            "reward": 0.0,
            "done": False,
            "info": {},
        }
        with (
            patch.object(
                AgentMemoryEnvClient,
                "get_metadata",
                return_value=metadata,
            ),
            patch(
                "agentenv.envs.agentmemory.requests.post",
                return_value=create_response,
            ),
        ):
            return AgentMemoryEnvClient(
                "http://programmatic.test",
                None,
                action_format=action_format,
            )

    def test_prompt_names_the_generated_surface_without_claiming_memoryarena_parity(
        self,
    ) -> None:
        prompt = build_procedural_conversation_start(
            ActionFormat.REACT,
            "neutral",
        )[0]["value"]
        self.assertIn("programmatically generated AgentMemoryGym WebShop", prompt)
        self.assertIn("six separate shopping sessions", prompt)
        self.assertNotIn("original MemoryArena WebShop", prompt)
        self.assertNotIn("paper", prompt.casefold())

    def test_bad_procedural_metadata_is_rejected_before_environment_creation(
        self,
    ) -> None:
        mutations = {
            "source": lambda value: value.update(source="frozen_memoryarena"),
            "paper": lambda value: value.update(paper_eligible=True),
            "schema": lambda value: value["provider"].update(schema="unknown"),
            "candidate_count": lambda value: value["provider"].update(
                candidate_count_per_phase=5
            ),
            "phase_count": lambda value: value["provider"].update(
                phase_count_per_task=5
            ),
            "human_gate": lambda value: value["provider"].update(
                human_review_required=True
            ),
            "llm_judge": lambda value: value["provider"].update(
                llm_judge_required=True
            ),
            "target_asin_leak": lambda value: value["provider"].update(
                target_asin_in_task_prompt=True
            ),
            "hidden_search_handles": lambda value: value["provider"].update(
                native_search_result_asin_handles_visible=False
            ),
            "non_native_click": lambda value: value["provider"].update(
                native_click_action_uses_asin_handle=False
            ),
            "provider_mode": lambda value: value.update(provider_mode="fixed_window"),
            "task_count": lambda value: value["provider"].update(task_count=62),
            "accepted_index_domain": lambda value: value["provider"].update(
                accepted_index_domain="bounded"
            ),
            "semantic_period": lambda value: value["provider"].update(
                semantic_period_tasks=198
            ),
            "stream_epoch": lambda value: value["provider"][
                "reseeded_stream"
            ].update(tasks_per_seed_epoch=198),
        }
        for name, mutate in mutations.items():
            metadata = deepcopy(procedural_metadata())
            mutate(metadata)
            post = Mock()
            with (
                self.subTest(case=name),
                patch.object(
                    AgentMemoryEnvClient,
                    "get_metadata",
                    return_value=metadata,
                ),
                patch("agentenv.envs.agentmemory.requests.post", post),
                self.assertRaises(RuntimeError),
            ):
                AgentMemoryEnvClient(
                    "http://procedural.invalid",
                    None,
                    action_format=ActionFormat.REACT,
                )
            post.assert_not_called()

    def test_latent_preference_surface_uses_dedicated_sop(self) -> None:
        metadata = latent_preference_metadata()
        create_response = Mock(status_code=200)
        create_response.json.return_value = {
            "id": 7,
            "observation": "latent preference reset",
            "reward": 0.0,
            "done": False,
            "info": {},
        }
        with (
            patch.object(
                AgentMemoryEnvClient,
                "get_metadata",
                return_value=metadata,
            ),
            patch(
                "agentenv.envs.agentmemory.requests.post",
                return_value=create_response,
            ) as post,
        ):
            client = AgentMemoryEnvClient(
                "http://latent-preference.test",
                None,
                action_format=ActionFormat.REACT,
            )

        self.assertTrue(client.is_procedural)
        self.assertEqual(client.memory_prompt_mode, "latent_preference_sop")
        prompt = client.conversation_start[0]["value"]
        for fragment in (
            "confirmed choice as preference evidence",
            "customer-profile memory",
            "preference axis",
            "inferred value",
            "Do not assume a fixed number",
            "use ADD before click[Buy Now]",
            "use UPDATE",
            "At the start of every later shopping session",
            "use RETRIEVE",
            "later application sessions",
        ):
            self.assertIn(fragment, prompt)
        self.assertNotIn("compatibility-relevant attributes", prompt)
        post.assert_called_once()

    def test_latent_preference_rejects_non_preference_prompt_mode_before_create(
        self,
    ) -> None:
        metadata = latent_preference_metadata()
        metadata["memory_prompt_mode"] = "legacy"
        post = Mock()
        with (
            patch.object(
                AgentMemoryEnvClient,
                "get_metadata",
                return_value=metadata,
            ),
            patch("agentenv.envs.agentmemory.requests.post", post),
            self.assertRaisesRegex(RuntimeError, "latent_preference_sop"),
        ):
            AgentMemoryEnvClient(
                "http://latent-preference.invalid",
                None,
                action_format=ActionFormat.REACT,
            )
        post.assert_not_called()

    def test_recency_override_surface_uses_preference_sop(self) -> None:
        metadata = recency_override_metadata()
        create_response = Mock(status_code=200)
        create_response.json.return_value = {
            "id": 8,
            "observation": "recency override reset",
            "reward": 0.0,
            "done": False,
            "info": {},
        }
        with (
            patch.object(
                AgentMemoryEnvClient,
                "get_metadata",
                return_value=metadata,
            ),
            patch(
                "agentenv.envs.agentmemory.requests.post",
                return_value=create_response,
            ) as post,
        ):
            client = AgentMemoryEnvClient(
                "http://recency-override.test",
                None,
                action_format=ActionFormat.REACT,
            )

        self.assertTrue(client.is_procedural)
        self.assertTrue(client.is_recency_override)
        self.assertEqual(client.memory_prompt_mode, "latent_preference_sop")
        prompt = client.conversation_start[0]["value"]
        self.assertIn("use UPDATE", prompt)
        self.assertIn("use RETRIEVE", prompt)
        post.assert_called_once()

    def test_distractor_surface_uses_query_only_top1_without_ask(self) -> None:
        client = self.create_client(distractor_robustness_metadata())
        self.assertTrue(client.is_distractor_robustness)
        prompt = client.conversation_start[0]["value"]
        self.assertIn("RETRIEVE requires exactly query:string", prompt)
        self.assertIn("one highest-ranked matching memory", prompt)
        self.assertIn("memory_id and top_k are forbidden", prompt)
        self.assertNotIn("ASK", prompt)
        self.assertNotIn("CLARIFY", prompt)

    def test_compositional_surface_accepts_four_task_orbit_contract(self) -> None:
        client = self.create_client(compositional_recall_metadata())
        self.assertTrue(client.is_compositional_recall)
        self.assertEqual(client.metadata["provider"]["tasks_per_orbit"], 4)
        prompt = client.conversation_start[0]["value"]
        self.assertIn("RETRIEVE requires exactly query:string", prompt)
        self.assertNotIn("ASK", prompt)

    def test_intent_surface_exposes_ask_only_on_that_surface(self) -> None:
        client = self.create_client(intent_clarification_metadata())
        self.assertTrue(client.is_intent_clarification)
        prompt = client.conversation_start[0]["value"]
        self.assertIn('ASK {"field":"..."}', prompt)
        self.assertIn("CLARIFY observation", prompt)

    def test_selective_surface_uses_decide_then_use_or_abstain_sop(self) -> None:
        client = self.create_client(selective_memory_use_metadata())
        self.assertTrue(client.is_selective_memory_use)
        self.assertEqual(client.memory_prompt_mode, "selective_memory_sop")
        self.assertEqual(client.metadata["provider"]["tasks_per_orbit"], 4)
        prompt = client.conversation_start[0]["value"]
        for fragment in (
            "First decide whether the current request already states every attribute",
            "explicit current requirements override profile history",
            "should not ADD or RETRIEVE merely by habit",
            "current request omits the customer's profile preference",
            "use RETRIEVE to expose the saved current profile",
            "Store new memory only when",
            "RETRIEVE requires exactly query:string",
            "memory_id and top_k are forbidden",
        ):
            self.assertIn(fragment, prompt)
        self.assertNotIn("confirmed choice as preference evidence", prompt)
        self.assertNotIn("ASK", prompt)

    def test_selective_surface_rejects_wrong_prompt_and_reward_contract(self) -> None:
        mutations = {
            "prompt": lambda value: value.update(
                memory_prompt_mode="latent_preference_sop"
            ),
            "positive_shaping": lambda value: value["provider"].update(
                memory_action_positive_shaping_allowed=True
            ),
            "fraction": lambda value: value["provider"].update(
                memory_not_required_fraction=0.25
            ),
            "penalty": lambda value: value["provider"].update(
                unnecessary_memory_action_penalty=0.0
            ),
        }
        for name, mutate in mutations.items():
            metadata = selective_memory_use_metadata()
            mutate(metadata)
            post = Mock()
            with (
                self.subTest(case=name),
                patch.object(
                    AgentMemoryEnvClient,
                    "get_metadata",
                    return_value=metadata,
                ),
                patch("agentenv.envs.agentmemory.requests.post", post),
                self.assertRaises(RuntimeError),
            ):
                AgentMemoryEnvClient(
                    "http://selective-memory.invalid",
                    None,
                    action_format=ActionFormat.REACT,
                )
            post.assert_not_called()

    def test_query_top1_function_schema_has_no_top_k_or_memory_id(self) -> None:
        query_top1 = self.create_client(
            distractor_robustness_metadata(),
            action_format=ActionFormat.FUNCTION_CALLING,
        )
        query_prompt = query_top1.conversation_start[0]["value"]
        self.assertIn("exactly the highest-ranked memory", query_prompt)
        self.assertNotIn('"top_k"', query_prompt)
        retrieve_schema = query_prompt.split('"name": "retrieve"', 1)[1].split(
            '"name": "summary"', 1
        )[0]
        self.assertIn('"query"', retrieve_schema)
        self.assertNotIn('"memory_id"', retrieve_schema)

        ordinary_prompt = build_procedural_conversation_start(
            ActionFormat.FUNCTION_CALLING,
            "neutral",
        )[0]["value"]
        self.assertIn('"top_k"', ordinary_prompt)

    def test_bad_recency_override_metadata_is_rejected_before_create(self) -> None:
        mutations = {
            "schema": lambda value: value["provider"].update(schema="unknown"),
            "schedule": lambda value: value["provider"].update(
                phase_schedule=["evidence"] * 6
            ),
            "override_index": lambda value: value["provider"].update(
                override_phase_index=3
            ),
            "update_contract": lambda value: value["provider"].update(
                update_contract="ADD another memory"
            ),
            "observation_identity": lambda value: value["provider"].update(
                application_observation_identity=False
            ),
            "target_flip": lambda value: value["provider"].update(
                application_target_flip=False
            ),
        }
        for name, mutate in mutations.items():
            metadata = deepcopy(recency_override_metadata())
            mutate(metadata)
            post = Mock()
            with (
                self.subTest(case=name),
                patch.object(
                    AgentMemoryEnvClient,
                    "get_metadata",
                    return_value=metadata,
                ),
                patch("agentenv.envs.agentmemory.requests.post", post),
                self.assertRaises(RuntimeError),
            ):
                AgentMemoryEnvClient(
                    "http://recency-override.invalid",
                    None,
                    action_format=ActionFormat.REACT,
                )
            post.assert_not_called()

    def test_bad_latent_preference_metadata_is_rejected_before_create(self) -> None:
        mutations = {
            "schema": lambda value: value["provider"].update(schema="unknown"),
            "evidence_counts": lambda value: value["provider"].update(
                supporting_evidence_counts=[2]
            ),
            "resolution": lambda value: value["provider"].update(resolution_step=2),
            "hypothesis": lambda value: value["provider"].update(
                preference_hypothesis="fixed_three_shot"
            ),
            "counterfactual": lambda value: value["provider"].update(
                counterfactual_pairing=False
            ),
            "observation_identity": lambda value: value["provider"].update(
                application_observation_identity=False
            ),
            "target_flip": lambda value: value["provider"].update(
                application_target_flip=False
            ),
            "receipt": lambda value: value["provider"].update(
                purchase_receipt_asin_verification=False
            ),
        }
        for name, mutate in mutations.items():
            metadata = deepcopy(latent_preference_metadata())
            mutate(metadata)
            post = Mock()
            with (
                self.subTest(case=name),
                patch.object(
                    AgentMemoryEnvClient,
                    "get_metadata",
                    return_value=metadata,
                ),
                patch("agentenv.envs.agentmemory.requests.post", post),
                self.assertRaises(RuntimeError),
            ):
                AgentMemoryEnvClient(
                    "http://latent-preference.invalid",
                    None,
                    action_format=ActionFormat.REACT,
                )
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
