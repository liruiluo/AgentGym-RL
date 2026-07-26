from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/agentmemory/eval_v3_openai.py"
SPEC = importlib.util.spec_from_file_location("agentmemory_eval_v3_openai", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frozen_dataset_provenance(config: str) -> dict:
    spec = MODULE.MEMORYARENA_FROZEN_DATASETS[config]
    payload = {
        "mode": "frozen_public_hf_dataset",
        "dataset_config": config,
        "split": "test",
        "repo_id": MODULE.MEMORYARENA_HF_REPO,
        "revision": MODULE.MEMORYARENA_HF_REVISION,
        "repo_path": spec["repo_path"],
        "sha256": spec["sha256"],
        "record_count": spec["record_count"],
        "phase_count": spec["phase_count"],
        "phase_field": spec["phase_field"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **payload,
        "attestation_sha256": hashlib.sha256(encoded).hexdigest(),
    }


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _FakeOpen:
    def __init__(self, metadata, *, model_text="Action: ADVANCE {}"):
        self.metadata = metadata
        self.model_text = model_text
        self.requests = []
        self.authorization_headers = []
        self.env_info_before = {
            "formal_schema_version": MODULE.FORMAL_SCHEMA_V3,
            "phase_index": 0,
            "phase_count": 1,
            "workflow_progress": 0.0,
            "episode_success": False,
            "reward_components": [],
        }
        self.env_info_after = {
            "formal_schema_version": MODULE.FORMAL_SCHEMA_V3,
            "phase_index": 1,
            "phase_count": 1,
            "workflow_progress": 1.0,
            "episode_success": True,
            "reward_components": [
                {"name": "phase_advance", "value": 1.0, "op": "ADVANCE"}
            ],
        }

    def __call__(self, request, timeout):
        del timeout
        body = json.loads(request.data.decode("utf-8")) if request.data else None
        self.requests.append((request.get_method(), request.full_url, body))
        self.authorization_headers.append(
            (request.full_url, request.get_header("Authorization"))
        )
        url = request.full_url
        if request.get_method() == "GET" and url.endswith("/metadata"):
            return _Response(self.metadata)
        if url.endswith("/create"):
            return _Response(
                {
                    "id": 7,
                    "observation": "phase zero",
                    "reward": 0.0,
                    "done": False,
                    "info": self.env_info_before,
                }
            )
        if url.endswith("/reset"):
            return _Response(
                {
                    "observation": "phase zero",
                    "reward": 0.0,
                    "done": False,
                    "info": self.env_info_before,
                }
            )
        if url.endswith("/tokenize"):
            return _Response({"tokens": [101, 102, 103], "count": 3})
        if url.endswith("/chat/completions"):
            return _Response(
                {
                    "id": "chatcmpl-test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": self.model_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
        if url.endswith("/step"):
            return _Response(
                {
                    "observation": "terminal",
                    "reward": 1.0,
                    "done": True,
                    "info": self.env_info_after,
                }
            )
        if url.endswith("/close"):
            return _Response(True)
        raise AssertionError(f"unexpected URL: {url}")


def _travel_metadata(*, surface: str | None = None) -> dict:
    return {
        "surface": surface or MODULE.TRAVEL_PAPER_EVAL_SURFACE,
        "domain_id": "travel_planner",
        "task_count": MODULE.TRAVEL_RECORD_COUNT,
        "phase_count": MODULE.TRAVEL_PHASE_COUNT,
        "contract_mode": "paper_eval",
        "dataset_sha256": MODULE.MEMORYARENA_FROZEN_DATASETS[
            "group_travel_planner"
        ]["sha256"],
        "dataset_provenance": _frozen_dataset_provenance(
            "group_travel_planner"
        ),
        "paper_evaluation": {
            "id": MODULE.TRAVEL_PAPER_METRIC_CONTRACT,
            "dataset_scope": MODULE.TRAVEL_PAPER_DATASET_SCOPE,
            "available": True,
            "canonical_semantics": True,
            "paper_panel_complete": True,
            "paper_column_eligible": True,
            "separate_from_online_reward": True,
        },
    }


def _travel_episode(
    position: int,
    *,
    people: int = 7,
    passed: int | None = None,
    constraint_people: int = 1,
    constraint_rate: float | None = 1.0,
) -> dict:
    if passed is None:
        passed = people
    group_success = passed == people
    source_id = position + 1
    paper_evaluation = {
        "metric_contract": MODULE.TRAVEL_PAPER_METRIC_CONTRACT,
        "dataset_scope": MODULE.TRAVEL_PAPER_DATASET_SCOPE,
        "source_id": source_id,
        "complete": True,
        "full_pass_people": passed,
        "total_people": people,
        "group_success": group_success,
        "group_constraint_rate": constraint_rate,
        "constraint_people": constraint_people,
        "online_reward_is_separate": True,
    }
    return {
        "data_idx": position,
        "steps": [
            {
                "env_info_after": {
                    "domain_evidence": {
                        "dataset_position": position,
                        "source_id": source_id,
                        "paper_evaluation": paper_evaluation,
                    }
                }
            }
        ],
        "episode_return": 0.0,
        "done": True,
        # Deliberately independent: Travel paper SR comes from its paper ledger.
        "episode_success": False,
        "timed_out": False,
        "final_phase_progress": {
            "phase_index_after": people,
            "phase_count": people,
        },
    }


def _search_metadata(*, surface: str | None = None) -> dict:
    embedding_config = {
        "provider": "openai",
        "model": MODULE.SEARCH_FROZEN_EMBEDDING_MODEL,
        "endpoint_sha256": "c" * 64,
        "route_variant": "paper_eval_openai_embedding_v1",
    }
    judge_config = {
        "mode": "upstream_memoryarena_judge",
        "backend": "openai_responses",
        "model": MODULE.SEARCH_JUDGE_MODEL,
        "max_tokens": MODULE.SEARCH_JUDGE_MAX_TOKENS,
        "endpoint_sha256": "d" * 64,
        "prompt_template_sha256": MODULE.SEARCH_JUDGE_PROMPT_TEMPLATE_SHA256,
    }
    return {
        "surface": surface or MODULE.SEARCH_PAPER_EVAL_SURFACE,
        "domain_id": "progressive_search",
        "contract_id": MODULE.SEARCH_PAPER_CONTRACT_ID,
        "contract_sha256": MODULE.SEARCH_PAPER_CONTRACT_SHA256,
        "system_prompt_sha256": MODULE.SEARCH_PAPER_SYSTEM_PROMPT_SHA256,
        "contract_mode": "paper_eval",
        "semantic_variant": (
            "paper_metric_evaluation_continue_on_incorrect_one_action_v1"
        ),
        "reward_contract": "evaluation_only_zero_reward_not_for_training",
        "reward_overlay": "none",
        "max_steps": 811,
        "max_total_actions": 811,
        "task_count": MODULE.SEARCH_RECORD_COUNT,
        "phase_count": MODULE.SEARCH_PHASE_COUNT,
        "dataset_sha256": MODULE.MEMORYARENA_FROZEN_DATASETS[
            "progressive_search"
        ]["sha256"],
        "dataset_provenance": _frozen_dataset_provenance("progressive_search"),
        "total_action_budget": {
            "limit": 811,
            "enforced_by": "agentmemory_runtime_wrapper",
            "counts": ["native", "memory", "invalid"],
            "legacy_max_steps_field_is_same_limit": True,
            "native_action_allowance": 555,
            "memory_action_allowance": 256,
            "memory_action_allowance_per_phase": 16,
        },
        "native_iteration_budget": {
            "subquery_per_phase": 35,
            "final_phase": 30,
            "counts": ["native", "invalid"],
            "memory_actions_consume_budget": False,
            "separately_tracked_from_total_action_budget": True,
            "upstream_batched_model_turn_parity": False,
        },
        "native_tool_ops": ["search", "get_document"],
        "native_search_k": 5,
        "native_snippet_max_tokens": 512,
        "snippet_tokenizer": copy.deepcopy(MODULE.SEARCH_SNIPPET_TOKENIZER),
        "judge": "memoryarena_browsecomp_gpt_judge_v1",
        "judge_provenance": {
            **judge_config,
            "config_sha256": MODULE._canonical_json_sha256(judge_config),
        },
        "upstream_provenance": {
            "mode": "pinned_pristine_upstream",
            "memoryarena_commit": MODULE.SEARCH_MEMORYARENA_COMMIT,
            "source_files_sha256": dict(
                MODULE.SEARCH_UPSTREAM_SOURCE_FILES_SHA256
            ),
            "source_bundle_sha256": MODULE.SEARCH_UPSTREAM_SOURCE_BUNDLE_SHA256,
        },
        "search_asset_provenance": {
            "mode": "frozen_public_assets",
            "embedding_model": MODULE.SEARCH_FROZEN_EMBEDDING_MODEL,
            "embedding_dimension": MODULE.SEARCH_FROZEN_INDEX_DIMENSION,
            "document_count": MODULE.SEARCH_FROZEN_DOCUMENT_COUNT,
            "index_repository": MODULE.SEARCH_FROZEN_INDEX_REPOSITORY,
            "index_revision": MODULE.SEARCH_FROZEN_INDEX_REVISION,
            "index_shards": [
                dict(item) for item in MODULE.SEARCH_FROZEN_INDEX_SHARDS
            ],
            "corpus_repository": MODULE.SEARCH_FROZEN_CORPUS_REPOSITORY,
            "corpus_revision": MODULE.SEARCH_FROZEN_CORPUS_REVISION,
            "corpus_source_shards": [
                {
                    "name": name,
                    "sha256": digest,
                    "size_bytes": size,
                    "row_count": rows,
                }
                for name, digest, size, rows in (
                    MODULE.SEARCH_FROZEN_CORPUS_SOURCE_SHARDS
                )
            ],
            "corpus_sha256": MODULE.SEARCH_FROZEN_CORPUS_SHA256,
            "corpus_manifest_sha256": (
                MODULE.SEARCH_FROZEN_CORPUS_MANIFEST_SHA256
            ),
        },
        "embedding_route_provenance": {
            "mode": "explicit_hashed_embedding_route",
            **embedding_config,
            "config_sha256": MODULE._canonical_json_sha256(embedding_config),
        },
        "paper_evaluation": {
            "id": MODULE.SEARCH_PAPER_METRIC_CONTRACT,
            "dataset_scope": MODULE.SEARCH_PAPER_DATASET_SCOPE,
            "available": True,
            "metrics": ["PS", "SR@k", "SR"],
            "metric_scale": "unit_interval",
            "paper_panel_complete": False,
            "public_task_count": MODULE.SEARCH_RECORD_COUNT,
            "paper_task_count": MODULE.SEARCH_PAPER_TASK_COUNT,
            "separate_from_online_reward": True,
        },
    }


def _search_episode(
    position: int,
    correct_flags: list[bool],
    *,
    query_id: str | None = None,
) -> dict:
    query_id = query_id or str(position)
    verdicts = []
    for phase_index, correct in enumerate(correct_flags):
        verdicts.append(
            {
                "phase_index": phase_index,
                "phase_kind": (
                    "final" if phase_index == len(correct_flags) - 1 else "subquery"
                ),
                "correct": correct,
                "verdict_source": "memoryarena_llm_judge",
                "answer_sha256": "a" * 64,
                "judge_response_sha256": "b" * 64,
                "judge_confidence": 0.9,
                "judge_parse_error": False,
                "retrieved_docids": [f"doc-{phase_index}"],
            }
        )
    correct_count = sum(correct_flags)
    phase_count = len(correct_flags)
    paper_evaluation = {
        "metric_contract": MODULE.SEARCH_PAPER_METRIC_CONTRACT,
        "dataset_scope": MODULE.SEARCH_PAPER_DATASET_SCOPE,
        "query_id": query_id,
        "complete": True,
        "metric_scale": "unit_interval",
        "phase_verdicts": verdicts,
        "completed_phase_count": phase_count,
        "process_score_numerator": correct_count,
        "process_score_denominator": phase_count,
        "process_score": correct_count / phase_count,
        "sr_at_k": {
            str(depth): {
                "correct": correct,
                "numerator": int(correct),
                "denominator": 1,
            }
            for depth, correct in enumerate(correct_flags, start=1)
        },
        "final_sr_numerator": int(correct_flags[-1]),
        "final_sr_denominator": 1,
        "final_success": correct_flags[-1],
        "online_reward_is_separate": True,
    }
    return {
        "data_idx": position,
        "initial_env_info": {
            "phase_index": 0,
            "phase_count": phase_count,
            "domain_evidence": {
                "query_id": query_id,
                "contract_mode": "paper_eval",
            },
        },
        "steps": [
            {
                "env_info_after": {
                    "phase_index": phase_count,
                    "phase_count": phase_count,
                    "sample_excluded": False,
                    "domain_evidence": {
                        "query_id": query_id,
                        "contract_mode": "paper_eval",
                        "phase_verdict_ledger": verdicts,
                        "paper_evaluation": paper_evaluation,
                    }
                }
            }
        ],
        "episode_return": 0.0,
        "done": True,
        "episode_success": correct_flags[-1],
        "timed_out": False,
        "final_phase_progress": {
            "phase_index_after": phase_count,
            "phase_count": phase_count,
        },
    }


def _shopping_metadata() -> dict:
    provenance = {
        "schema": "memoryarena_raw_dataset_provenance_v1",
        "raw_dataset_path": "/frozen/bundled_shopping/data.jsonl",
        "raw_dataset_sha256": MODULE.SHOPPING_RAW_DATASET_SHA256,
        "memoryarena_commit": MODULE.SHOPPING_MEMORYARENA_COMMIT,
        "domain_data_sha256": MODULE.SHOPPING_DOMAIN_DATA_SHA256,
        "action_surface_version": MODULE.WEBSHOP_V2_SURFACE,
        "split_strategy": MODULE.SHOPPING_SPLIT_STRATEGY,
        "split_manifest_sha256": MODULE.SHOPPING_SPLIT_MANIFEST_SHA256,
        "split_counts": {"train": 120, "dev": 15, "test": 15},
        "bundle_count": MODULE.SHOPPING_BUNDLE_COUNT,
        "sessions_per_bundle": MODULE.SHOPPING_SESSIONS_PER_BUNDLE,
        "session_count": MODULE.SHOPPING_SESSION_COUNT,
        "target_asin_membership_verified": True,
    }
    return {
        "surface": MODULE.WEBSHOP_V2_SURFACE,
        "task_count": MODULE.SHOPPING_BUNDLE_COUNT,
        "splits": ["dev", "test", "train"],
        "dataset_sha256": MODULE.SHOPPING_RAW_DATASET_SHA256,
        "raw_dataset_sha256": MODULE.SHOPPING_RAW_DATASET_SHA256,
        "split_manifest_sha256": MODULE.SHOPPING_SPLIT_MANIFEST_SHA256,
        "memoryarena_commit": MODULE.SHOPPING_MEMORYARENA_COMMIT,
        "domain_data_sha256": MODULE.SHOPPING_DOMAIN_DATA_SHA256,
        "dataset_provenance": provenance,
        "annotation_gate_allowed_task_count": MODULE.SHOPPING_BUNDLE_COUNT,
        "annotation_gate_allowed_task_ids_sha256": (
            MODULE.SHOPPING_TASK_IDS_SHA256
        ),
    }


def _shopping_episode(position: int, *, success: bool = True) -> dict:
    task_id = MODULE._shopping_task_id(position)
    final_index = MODULE.SHOPPING_SESSIONS_PER_BUNDLE if success else 0
    purchase_count = MODULE.SHOPPING_SESSIONS_PER_BUNDLE if success else 1
    purchases = []
    for session_index in range(purchase_count):
        correct = success or session_index < final_index
        purchases.append(
            {
                "op": "BUY",
                "committed": True,
                "purchase_correct": correct,
                "session_advanced": correct,
                "session_index": session_index,
            }
        )
    initial_info = {
        "task_id": task_id,
        "phase_count": MODULE.SHOPPING_SESSIONS_PER_BUNDLE,
        "subtask_count": MODULE.SHOPPING_SESSIONS_PER_BUNDLE,
        "current_subtask_index": 0,
        "episode_success": False,
        "sample_excluded": False,
        "purchase_history": [],
    }
    final_info = {
        "task_id": task_id,
        "phase_count": MODULE.SHOPPING_SESSIONS_PER_BUNDLE,
        "subtask_count": MODULE.SHOPPING_SESSIONS_PER_BUNDLE,
        "current_subtask_index": final_index,
        "episode_success": success,
        "sample_excluded": False,
        "purchase_history": purchases,
    }
    return {
        "data_idx": position,
        "initial_env_info": initial_info,
        "steps": [{"env_info_after": final_info}],
        "episode_return": float(success),
        "done": True,
        "episode_success": success,
        "timed_out": False,
        "final_phase_progress": {
            "phase_index_after": final_index,
            "phase_count": MODULE.SHOPPING_SESSIONS_PER_BUNDLE,
        },
    }


def _formal_metadata(surface: str) -> dict:
    config = MODULE.FORMAL_SURFACE_DATASETS[surface]
    spec = MODULE.MEMORYARENA_FROZEN_DATASETS[config]
    runtime = MODULE.FORMAL_RUNTIME_CONTRACTS[surface]
    judge_config = {
        "mode": "upstream_memoryarena_judge",
        "backend": "openai",
        "model": MODULE.FORMAL_JUDGE_MODEL,
        "temperature": MODULE.FORMAL_JUDGE_TEMPERATURE,
        "max_tokens": MODULE.FORMAL_JUDGE_MAX_TOKENS,
        "endpoint_sha256": "e" * 64,
        "prompt_template_sha256": MODULE.FORMAL_JUDGE_PROMPT_TEMPLATE_SHA256,
    }
    metadata = {
        "formal_schema_version": MODULE.FORMAL_SCHEMA_V3,
        "source": "MemoryArena",
        "surface": surface,
        "domain_id": config,
        "dataset_config": config,
        "task_count": spec["record_count"],
        "phase_count": spec["phase_count"],
        "dataset_sha256": spec["sha256"],
        "dataset_provenance": _frozen_dataset_provenance(config),
        "contract_id": runtime["contract_id"],
        "contract_sha256": runtime["contract_sha256"],
        "system_prompt": runtime["system_prompt"],
        "system_prompt_sha256": runtime["system_prompt_sha256"],
        "native_action_descriptions": ["<final answer text>"],
        "max_steps": 64,
        "judge": "memoryarena_llm_math_equivalence_v1",
        "judge_provenance": {
            **judge_config,
            "config_sha256": MODULE._canonical_json_sha256(judge_config),
        },
        "contract_mode": runtime["contract_mode"],
        "semantic_variant": runtime["semantic_variant"],
        "phase_transition": runtime["phase_transition"],
        "episode_success": runtime["episode_success"],
        "upstream_provenance": {
            "mode": "pinned_pristine_upstream_scopes",
            "memoryarena_commit": MODULE.FORMAL_MEMORYARENA_COMMIT,
            "pristine_git_scopes": ["env", "agent/math.py", "run_math.py"],
            "env_git_tree_oid": MODULE.FORMAL_ENV_GIT_TREE_OID,
            "runtime_import_entry_files_sha256": dict(
                MODULE.FORMAL_RUNTIME_SOURCE_FILES_SHA256
            ),
            "reference_entrypoint_files_sha256": dict(
                MODULE.FORMAL_REFERENCE_SOURCE_FILES_SHA256
            ),
            "selected_files_bundle_sha256": (
                MODULE.FORMAL_SELECTED_SOURCE_BUNDLE_SHA256
            ),
        },
        "memory_reward_policy": {
            "first_add": 0.0,
            "first_later_phase_retrieve": 0.0,
            "exact_repeat": 0.0,
            "invalid_action": 0.0,
        },
        "reward_overlay": "none",
    }
    if runtime["contract_mode"] == "paper_eval":
        metadata["paper_evaluation"] = {
            "id": MODULE.FORMAL_PAPER_METRIC_CONTRACT,
            "dataset_scope": MODULE.FORMAL_PAPER_DATASET_SCOPES[surface],
            "available": True,
            "metrics": ["PS", "SR"],
            "metric_scale": "unit_interval",
            "canonical_semantics": True,
            "paper_panel_complete": True,
            "paper_column_eligible": True,
            "continue_after_incorrect": True,
            "separate_from_online_reward": True,
        }
    return metadata


def _formal_episode(
    position: int,
    *,
    surface: str,
    success: bool = True,
    results: list[bool] | None = None,
) -> dict:
    phase_count = MODULE.FORMAL_TASK_PHASE_COUNTS[surface][position]
    metadata = _formal_metadata(surface)
    task_id = str(position)
    paper_name = f"paper-{position}"
    contract_mode = metadata["contract_mode"]
    if results is not None:
        phase_results = list(results)
    elif contract_mode == "paper_eval":
        phase_results = (
            [True] * phase_count
            if success
            else [True] * (phase_count - 1) + [False]
        )
    else:
        phase_results = [True] * phase_count if success else [False]
    initial_domain = {"task_id": task_id, "paper_name": paper_name}
    initial_info = {
        "formal_schema_version": MODULE.FORMAL_SCHEMA_V3,
        "domain_id": metadata["domain_id"],
        "surface": surface,
        "contract_id": metadata["contract_id"],
        "contract_sha256": metadata["contract_sha256"],
        "phase_index": 0,
        "phase_count": phase_count,
        "episode_success": False,
        "sample_excluded": False,
        "domain_evidence": initial_domain,
    }
    steps = []
    previous = initial_info
    observed_results = []
    for turn, passed in enumerate(phase_results, start=1):
        before_index = previous["phase_index"]
        phase_advanced = passed or contract_mode == "paper_eval"
        after_index = before_index + int(phase_advanced)
        terminal = after_index == phase_count or (
            contract_mode == "failfast" and not passed
        )
        observed_results.append(passed)
        reward = float(passed)
        component_name = (
            "formal_reasoning_answer_correct"
            if passed
            else "formal_reasoning_answer_incorrect"
        )
        current = {
            "formal_schema_version": MODULE.FORMAL_SCHEMA_V3,
            "domain_id": metadata["domain_id"],
            "surface": surface,
            "contract_id": metadata["contract_id"],
            "contract_sha256": metadata["contract_sha256"],
            "phase_index": after_index,
            "phase_count": phase_count,
            "episode_success": terminal and passed and after_index == phase_count,
            "sample_excluded": False,
            "action_execution": {
                "op": "ANSWER",
                "status": "committed_correct" if passed else "committed_incorrect",
                "step": turn,
            },
            "tool_ops": [
                {
                    "op": "ANSWER",
                    "step": turn,
                    "committed": True,
                    "submission_correct": passed,
                    "phase_index": before_index,
                    "phase_advanced": phase_advanced,
                    "terminal": terminal,
                    "answer_sha256": "a" * 64,
                }
            ],
            "reward_components": [
                {
                    "name": component_name,
                    "value": reward,
                    "op": "ANSWER",
                    "step": turn,
                }
            ],
            "domain_evidence": {
                "task_id": task_id,
                "paper_name": paper_name,
                "judge_id": "memoryarena_llm_math_equivalence_v1",
                "answer_sha256": "a" * 64,
                "ground_truth_sha256": "b" * 64,
                "judge_output_sha256": "c" * 64,
                "correct_count": sum(observed_results),
                "phase_results": list(observed_results),
            },
        }
        steps.append(
            {
                "turn": turn,
                "env_info_before": copy.deepcopy(previous),
                "env_info_after": copy.deepcopy(current),
                "reward": reward,
                "done": terminal,
                "phase_progress": {
                    "phase_index_before": before_index,
                    "phase_index_after": after_index,
                    "phase_count": phase_count,
                    "phase_advanced": phase_advanced,
                },
            }
        )
        previous = current
    if contract_mode == "paper_eval" and previous["phase_index"] == phase_count:
        final_success = bool(observed_results[-1])
        previous["domain_evidence"]["paper_evaluation"] = {
            "metric_contract": MODULE.FORMAL_PAPER_METRIC_CONTRACT,
            "dataset_scope": MODULE.FORMAL_PAPER_DATASET_SCOPES[surface],
            "task_id": task_id,
            "paper_name": paper_name,
            "complete": True,
            "phase_results": list(observed_results),
            "completed_phase_count": phase_count,
            "process_score_numerator": sum(observed_results),
            "process_score_denominator": phase_count,
            "process_score": sum(observed_results) / phase_count,
            "final_sr_numerator": int(final_success),
            "final_sr_denominator": 1,
            "final_success": final_success,
            "online_reward_is_separate": True,
        }
        steps[-1]["env_info_after"] = copy.deepcopy(previous)
    final_index = previous["phase_index"]
    return {
        "data_idx": position,
        "initial_env_info": initial_info,
        "steps": steps,
        "episode_return": float(sum(phase_results)),
        "done": True,
        "episode_success": bool(previous["episode_success"]),
        "timed_out": False,
        "final_phase_progress": {
            "phase_index_after": final_index,
            "phase_count": phase_count,
        },
    }


def _paper_manifest(metadata: dict, episodes: list[dict]) -> dict:
    return {
        "environment": {"metadata": metadata},
        "episodes": episodes,
        "summary": MODULE.summarize_paper_surface(episodes, metadata),
    }


class EvalV3OpenAITest(unittest.TestCase):
    def test_v3_rejects_ambiguous_progress_score(self):
        before = {
            "formal_schema_version": MODULE.FORMAL_SCHEMA_V3,
            "phase_index": 0,
            "phase_count": 2,
            "progress_score": 0.0,
        }
        after = {
            "formal_schema_version": MODULE.FORMAL_SCHEMA_V3,
            "phase_index": 1,
            "phase_count": 2,
            "progress_score": 0.5,
        }
        with self.assertRaisesRegex(MODULE.EvalError, "workflow_progress"):
            MODULE._phase_progress(before, after)

    def test_paper_macro_requires_all_five_surface_columns(self):
        rates = {
            "Shopping": 0.1,
            "Travel": 0.2,
            "Search": 0.3,
            "Math": 0.4,
            "Physics": 0.5,
        }
        self.assertAlmostEqual(MODULE.compute_paper_macro5(rates), 0.3)
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            MODULE.compute_paper_macro5(
                {
                    "Shopping": 0.1,
                    "Travel": 0.2,
                    "Search": 0.3,
                    "Formal Reasoning": 0.45,
                }
            )

    def test_v3_metadata_covers_four_nonshopping_paper_surfaces(self):
        surfaces = {
            "Travel": (MODULE.TRAVEL_PAPER_EVAL_SURFACE, "travel_planner"),
            "Search": (MODULE.SEARCH_PAPER_EVAL_SURFACE, "progressive_search"),
            "Math": (
                MODULE.MATH_PAPER_EVAL_SURFACE,
                "formal_reasoning_math",
            ),
            "Physics": (
                MODULE.PHYS_PAPER_EVAL_SURFACE,
                "formal_reasoning_phys",
            ),
        }
        for label, (surface, domain_id) in surfaces.items():
            with self.subTest(label=label):
                metadata = {
                    "surface": surface,
                    "domain_id": domain_id,
                }
                registration = MODULE.resolve_paper_surface(metadata)
                self.assertEqual(registration["paper_column"], label)
                self.assertEqual(registration["surface"], surface)

    def test_formal_failfast_wrong_answer_contract_is_preserved(self):
        for surface in (MODULE.MATH_FAILFAST_SURFACE, MODULE.PHYS_FAILFAST_SURFACE):
            with self.subTest(surface=surface):
                metadata = _formal_metadata(surface)
                episode = _formal_episode(0, surface=surface, success=False)
                metrics = MODULE.aggregate_formal_panel_evidence(
                    [episode], metadata
                )
                self.assertEqual(metrics["metric_contract"], "episode_success")
                self.assertEqual(
                    episode["final_phase_progress"]["phase_index_after"], 0
                )

                continued = copy.deepcopy(episode)
                continued["steps"][0]["env_info_after"]["phase_index"] = 1
                continued["steps"][0]["done"] = False
                with self.assertRaisesRegex(MODULE.EvalError, "fail fast"):
                    MODULE.aggregate_formal_panel_evidence([continued], metadata)

    def test_formal_paper_eval_wrong_answer_continues_for_both_domains(self):
        for surface in (
            MODULE.MATH_PAPER_EVAL_SURFACE,
            MODULE.PHYS_PAPER_EVAL_SURFACE,
        ):
            with self.subTest(surface=surface):
                phase_count = MODULE.FORMAL_TASK_PHASE_COUNTS[surface][0]
                results = [False] + [True] * (phase_count - 1)
                metadata = _formal_metadata(surface)
                episode = _formal_episode(0, surface=surface, results=results)
                metrics = MODULE.aggregate_formal_panel_evidence(
                    [episode], metadata
                )
                first = episode["steps"][0]
                self.assertFalse(first["done"])
                self.assertEqual(
                    first["env_info_after"]["phase_index"], 1
                )
                self.assertTrue(episode["episode_success"])
                self.assertAlmostEqual(
                    metrics["process_score"], (phase_count - 1) / phase_count
                )
                self.assertEqual(metrics["final_success_rate"], 1.0)

                failfast_forgery = copy.deepcopy(episode)
                failfast_forgery["steps"][0]["env_info_after"]["phase_index"] = 0
                with self.assertRaisesRegex(
                    MODULE.EvalError, "paper-eval continuation"
                ):
                    MODULE.aggregate_formal_panel_evidence(
                        [failfast_forgery], metadata
                    )

    def test_travel_paper_ledger_uses_people_and_group_weighting(self):
        self.assertEqual(
            MODULE.TRAVEL_PAPER_DATASET_SCOPE,
            "memoryarena_group_travel_planner_frozen270",
        )
        episodes = [
            _travel_episode(
                0,
                people=5,
                passed=4,
                constraint_people=2,
                constraint_rate=0.2,
            ),
            _travel_episode(
                1,
                people=8,
                passed=8,
                constraint_people=3,
                constraint_rate=0.8,
            ),
            _travel_episode(
                2,
                people=6,
                passed=1,
                constraint_people=0,
                constraint_rate=None,
            ),
        ]
        metrics = MODULE.aggregate_travel_paper_metrics(
            episodes,
            _travel_metadata(),
        )
        self.assertAlmostEqual(metrics["ps"], 100.0 * 13 / 19)
        self.assertAlmostEqual(metrics["sps"], 50.0)
        self.assertAlmostEqual(metrics["sr"], 100.0 / 3)
        self.assertEqual(metrics["groups_with_constraint_people"], 2)
        self.assertEqual(metrics["total_constraint_people"], 5)

        summary = MODULE.summarize_paper_surface(episodes, _travel_metadata())
        self.assertEqual(summary["paper_column"], "Travel")
        self.assertAlmostEqual(summary["paper_success_rate"], 1.0 / 3)
        self.assertFalse(summary["panel_complete"])
        self.assertFalse(summary["paper_macro_eligible"])

    def test_travel_paper_ledger_fails_closed_on_missing_or_conflicting_fields(self):
        base = _travel_episode(0, people=5, passed=4)
        mutations = {
            "missing field": lambda item: item["steps"][0]["env_info_after"][
                "domain_evidence"
            ]["paper_evaluation"].pop("total_people"),
            "extra field": lambda item: item["steps"][0]["env_info_after"][
                "domain_evidence"
            ]["paper_evaluation"].update({"derived_sr": 0.0}),
            "position source mismatch": lambda item: item["steps"][0][
                "env_info_after"
            ]["domain_evidence"]["paper_evaluation"].update({"source_id": 2}),
            "contradictory success": lambda item: item["steps"][0][
                "env_info_after"
            ]["domain_evidence"]["paper_evaluation"].update(
                {"group_success": True}
            ),
            "too many constraint people": lambda item: item["steps"][0][
                "env_info_after"
            ]["domain_evidence"]["paper_evaluation"].update(
                {"constraint_people": 6}
            ),
            "scope mismatch": lambda item: item["steps"][0]["env_info_after"][
                "domain_evidence"
            ]["paper_evaluation"].update({"dataset_scope": "other"}),
            "incomplete": lambda item: item["steps"][0]["env_info_after"][
                "domain_evidence"
            ]["paper_evaluation"].update({"complete": False}),
            "integer completion flag": lambda item: item["steps"][0][
                "env_info_after"
            ]["domain_evidence"]["paper_evaluation"].update({"complete": 1}),
            "boolean dataset position": lambda item: item["steps"][0][
                "env_info_after"
            ]["domain_evidence"].update({"dataset_position": False}),
            "timed out": lambda item: item.update({"timed_out": True}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                episode = copy.deepcopy(base)
                mutate(episode)
                with self.assertRaises(MODULE.EvalError):
                    MODULE.aggregate_travel_paper_metrics(
                        [episode],
                        _travel_metadata(),
                    )

        metadata = _travel_metadata()
        metadata["paper_evaluation"]["dataset_scope"] = "coordinated-drift"
        episode = copy.deepcopy(base)
        episode["steps"][0]["env_info_after"]["domain_evidence"][
            "paper_evaluation"
        ]["dataset_scope"] = "coordinated-drift"
        with self.assertRaisesRegex(MODULE.EvalError, "dataset_scope mismatch"):
            MODULE.aggregate_travel_paper_metrics([episode], metadata)
        with self.assertRaisesRegex(MODULE.EvalError, "repeats source_id"):
            MODULE.aggregate_travel_paper_metrics(
                [copy.deepcopy(base), copy.deepcopy(base)],
                _travel_metadata(),
            )

    def test_complete_travel_panel_is_macro_eligible_and_checks_phase_total(self):
        episodes = [
            _travel_episode(position, people=(7 if position < 249 else 6))
            for position in range(MODULE.TRAVEL_RECORD_COUNT)
        ]
        summary = MODULE.summarize_paper_surface(episodes, _travel_metadata())
        self.assertTrue(summary["panel_complete"])
        self.assertTrue(summary["paper_macro_eligible"])
        self.assertEqual(summary["paper_success_rate"], 1.0)
        self.assertEqual(
            summary["paper_metrics"]["total_people"],
            MODULE.TRAVEL_PHASE_COUNT,
        )

        invalid = [
            _travel_episode(position, people=7)
            for position in range(MODULE.TRAVEL_RECORD_COUNT)
        ]
        with self.assertRaisesRegex(MODULE.EvalError, "phase count"):
            MODULE.aggregate_travel_paper_metrics(invalid, _travel_metadata())

    def test_travel_failfast_surface_is_diagnostic_only(self):
        self.assertEqual(
            MODULE.TRAVEL_FAILFAST_SURFACE,
            "memoryarena_travel_planner_failfast_one_action_v3",
        )
        self.assertEqual(
            MODULE.TRAVEL_PAPER_EVAL_SURFACE,
            "memoryarena_travel_planner_paper_eval_one_action_v3",
        )
        episodes = [
            _travel_episode(position, people=(7 if position < 249 else 6))
            for position in range(MODULE.TRAVEL_RECORD_COUNT)
        ]
        metadata = _travel_metadata(surface=MODULE.TRAVEL_FAILFAST_SURFACE)
        summary = MODULE.summarize_paper_surface(episodes, metadata)
        self.assertTrue(summary["panel_complete"])
        self.assertTrue(summary["paper_panel_complete"])
        self.assertIsNone(summary["paper_success_rate"])
        self.assertIsNone(summary["paper_metrics"])
        self.assertEqual(
            summary["paper_metric_contract"],
            "travel_failfast_diagnostic_only",
        )
        self.assertFalse(summary["paper_macro_eligible"])
        with self.assertRaisesRegex(MODULE.EvalError, "paper-eval runtime surface"):
            MODULE.aggregate_travel_paper_metrics(episodes, metadata)
        with self.assertRaisesRegex(MODULE.EvalError, "not macro5 eligible"):
            MODULE.compute_paper_macro5_from_manifests(
                [_paper_manifest(metadata, episodes)]
            )

    def test_search_paper_ledger_is_task_macro_and_depth_specific(self):
        episodes = [
            _search_episode(0, [False] + [True] * 8),
            _search_episode(1, [True] * 11 + [False]),
        ]
        metrics = MODULE.aggregate_search_paper_metrics(
            episodes,
            _search_metadata(),
        )
        self.assertAlmostEqual(metrics["process_score"], ((8 / 9) + (11 / 12)) / 2)
        self.assertEqual(
            metrics["sr_at_k"]["1"],
            {"correct_tasks": 1, "eligible_tasks": 2, "rate": 0.5},
        )
        self.assertEqual(
            metrics["sr_at_k"]["10"],
            {"correct_tasks": 1, "eligible_tasks": 1, "rate": 1.0},
        )
        self.assertEqual(metrics["final_success_rate"], 0.5)
        self.assertFalse(metrics["public_panel_complete"])
        self.assertFalse(metrics["paper_panel_complete"])

        summary = MODULE.summarize_paper_surface(episodes, _search_metadata())
        self.assertEqual(summary["paper_column"], "Search")
        self.assertEqual(summary["paper_success_rate"], 0.5)
        self.assertFalse(summary["panel_complete"])
        self.assertFalse(summary["paper_macro_eligible"])

    def test_search_paper_ledger_fails_closed_on_conflicting_evidence(self):
        base = _search_episode(0, [False] + [True] * 8)
        ledger_path = lambda item: item["steps"][-1]["env_info_after"][
            "domain_evidence"
        ]["paper_evaluation"]
        mutations = {
            "missing field": lambda item: ledger_path(item).pop("final_success"),
            "extra field": lambda item: ledger_path(item).update({"derived_sr": 1}),
            "query conflict": lambda item: ledger_path(item).update(
                {"query_id": "different"}
            ),
            "reset query conflict": lambda item: item["initial_env_info"][
                "domain_evidence"
            ].update({"query_id": "different"}),
            "reset phase count conflict": lambda item: item["initial_env_info"].update(
                {"phase_count": 8}
            ),
            "phase ledger conflict": lambda item: item["steps"][-1][
                "env_info_after"
            ]["domain_evidence"].update({"phase_verdict_ledger": []}),
            "contract mode conflict": lambda item: item["steps"][-1][
                "env_info_after"
            ]["domain_evidence"].update({"contract_mode": "failfast"}),
            "phase index gap": lambda item: ledger_path(item)["phase_verdicts"][
                1
            ].update({"phase_index": 3}),
            "phase index wrong type": lambda item: ledger_path(item)[
                "phase_verdicts"
            ][0].update({"phase_index": 0.0}),
            "process score conflict": lambda item: ledger_path(item).update(
                {"process_score": 1.0}
            ),
            "SR@k conflict": lambda item: ledger_path(item)["sr_at_k"]["2"].update(
                {"numerator": 0}
            ),
            "final SR conflict": lambda item: ledger_path(item).update(
                {"final_success": False}
            ),
            "online reward conflict": lambda item: item.update(
                {"episode_return": 1.0}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                episode = copy.deepcopy(base)
                mutate(episode)
                with self.assertRaises(MODULE.EvalError):
                    MODULE.aggregate_search_paper_metrics(
                        [episode],
                        _search_metadata(),
                    )

        duplicate_idx = [
            _search_episode(0, [True] * MODULE.SEARCH_TASK_PHASE_COUNTS[0]),
            _search_episode(0, [True] * MODULE.SEARCH_TASK_PHASE_COUNTS[0]),
        ]
        with self.assertRaisesRegex(MODULE.EvalError, "repeats data_idx"):
            MODULE.aggregate_search_paper_metrics(duplicate_idx, _search_metadata())
        wrong_query = _search_episode(
            0,
            [True] * MODULE.SEARCH_TASK_PHASE_COUNTS[0],
            query_id="same",
        )
        with self.assertRaisesRegex(MODULE.EvalError, "query_id mismatch"):
            MODULE.aggregate_search_paper_metrics([wrong_query], _search_metadata())

        metadata = _search_metadata()
        metadata["paper_evaluation"]["paper_panel_complete"] = True
        with self.assertRaisesRegex(MODULE.EvalError, "metadata mismatch"):
            MODULE.aggregate_search_paper_metrics([base], metadata)

    def test_search_metadata_binds_runtime_assets_routes_judge_and_prompt(self):
        episode = _search_episode(
            0,
            [True] * MODULE.SEARCH_TASK_PHASE_COUNTS[0],
        )
        mutations = {
            "missing upstream": lambda item: item.pop("upstream_provenance"),
            "wrong source bundle": lambda item: item["upstream_provenance"].update(
                {"source_bundle_sha256": "0" * 64}
            ),
            "wrong corpus": lambda item: item["search_asset_provenance"].update(
                {"corpus_sha256": "0" * 64}
            ),
            "wrong index": lambda item: item["search_asset_provenance"][
                "index_shards"
            ][0].update({"index_sha256": "0" * 64}),
            "wrong embedding model": lambda item: item[
                "embedding_route_provenance"
            ].update({"model": "different-embedding"}),
            "wrong judge": lambda item: item["judge_provenance"].update(
                {"model": "different-judge"}
            ),
            "wrong prompt": lambda item: item.update(
                {"system_prompt_sha256": "0" * 64}
            ),
            "wrong contract": lambda item: item.update(
                {"contract_id": "different-contract"}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                metadata = _search_metadata()
                mutate(metadata)
                with self.assertRaises(MODULE.EvalError):
                    MODULE.aggregate_search_paper_metrics([episode], metadata)

    def test_search_episode_binds_frozen_row_phase_count(self):
        wrong = _search_episode(
            0,
            [True] * (MODULE.SEARCH_TASK_PHASE_COUNTS[0] - 1),
        )
        with self.assertRaisesRegex(MODULE.EvalError, "phase_count mismatch"):
            MODULE.aggregate_search_paper_metrics([wrong], _search_metadata())

    def test_search_public221_panel_and_canonical_macro5_are_fail_closed(self):
        self.assertEqual(
            MODULE.SEARCH_PAPER_EVAL_SURFACE,
            "memoryarena_progressive_search_paper_eval_public221_one_action_v3",
        )
        self.assertEqual(
            MODULE.SEARCH_FAILFAST_SURFACE,
            "memoryarena_progressive_search_failfast_public221_one_action_v3",
        )
        search_episodes = [
            _search_episode(
                position,
                [True] * MODULE.SEARCH_TASK_PHASE_COUNTS[position],
            )
            for position in range(MODULE.SEARCH_RECORD_COUNT)
        ]
        search_metadata = _search_metadata()
        search_summary = MODULE.summarize_paper_surface(
            search_episodes,
            search_metadata,
        )
        self.assertTrue(search_summary["panel_complete"])
        self.assertFalse(
            MODULE.PAPER_SURFACE_REGISTRY[MODULE.SEARCH_PAPER_EVAL_SURFACE][
                "canonical_macro_candidate"
            ]
        )
        self.assertFalse(search_summary["paper_panel_complete"])
        self.assertFalse(search_summary["paper_macro_eligible"])
        self.assertEqual(search_summary["paper_success_rate"], 1.0)
        self.assertEqual(
            search_summary["paper_metrics"]["phase_count"],
            MODULE.SEARCH_PHASE_COUNT,
        )
        self.assertFalse(search_summary["paper_metrics"]["paper_panel_complete"])

        invalid_phase_total = [
            _search_episode(
                position,
                [True]
                * (
                    MODULE.SEARCH_TASK_PHASE_COUNTS[position] - 1
                    if position == 94
                    else MODULE.SEARCH_TASK_PHASE_COUNTS[position]
                ),
            )
            for position in range(MODULE.SEARCH_RECORD_COUNT)
        ]
        with self.assertRaisesRegex(MODULE.EvalError, "phase_count mismatch"):
            MODULE.aggregate_search_paper_metrics(
                invalid_phase_total,
                search_metadata,
            )

        search_manifest = _paper_manifest(search_metadata, search_episodes)
        with self.assertRaisesRegex(MODULE.EvalError, "complete paper panel"):
            MODULE.compute_paper_macro5_from_manifests([search_manifest])

        tampered = copy.deepcopy(search_manifest)
        tampered["summary"]["paper_macro_eligible"] = True
        tampered["summary"]["paper_panel_complete"] = True
        with self.assertRaisesRegex(MODULE.EvalError, "disagrees"):
            MODULE.compute_paper_macro5_from_manifests([tampered])

        diagnostic_metadata = _search_metadata(surface=MODULE.SEARCH_FAILFAST_SURFACE)
        diagnostic_manifest = _paper_manifest(
            diagnostic_metadata,
            search_episodes,
        )
        self.assertFalse(diagnostic_manifest["summary"]["paper_panel_complete"])
        self.assertFalse(diagnostic_manifest["summary"]["paper_macro_eligible"])
        with self.assertRaisesRegex(MODULE.EvalError, "complete paper panel"):
            MODULE.compute_paper_macro5_from_manifests([diagnostic_manifest])

    def test_shopping_macro_requires_frozen_provenance_ids_and_phase_ledgers(self):
        episodes = [
            _shopping_episode(position)
            for position in range(MODULE.SHOPPING_BUNDLE_COUNT)
        ]
        metadata = _shopping_metadata()
        summary = MODULE.summarize_paper_surface(episodes, metadata)
        self.assertTrue(summary["panel_complete"])
        self.assertTrue(summary["paper_panel_complete"])
        self.assertTrue(summary["paper_macro_eligible"])
        self.assertEqual(
            summary["paper_metrics"]["phase_count"],
            MODULE.SHOPPING_SESSION_COUNT,
        )
        self.assertEqual(
            summary["paper_metrics"]["unique_task_id_count"],
            MODULE.SHOPPING_BUNDLE_COUNT,
        )

        fake_metadata = _shopping_metadata()
        fake_metadata["task_count"] = 1
        with self.assertRaisesRegex(MODULE.EvalError, "top-level"):
            MODULE.summarize_paper_surface([episodes[0]], fake_metadata)

        provenance_drift = _shopping_metadata()
        provenance_drift["dataset_provenance"]["raw_dataset_sha256"] = "0" * 64
        with self.assertRaisesRegex(MODULE.EvalError, "provenance mismatch"):
            MODULE.summarize_paper_surface(episodes, provenance_drift)

        task_id_drift = copy.deepcopy(episodes)
        task_id_drift[0]["initial_env_info"]["task_id"] = "forged"
        with self.assertRaisesRegex(MODULE.EvalError, "task_id mismatch"):
            MODULE.summarize_paper_surface(task_id_drift, metadata)

        phase_drift = copy.deepcopy(episodes)
        phase_drift[0]["steps"][-1]["env_info_after"]["phase_count"] = 5
        with self.assertRaisesRegex(MODULE.EvalError, "phase ledger mismatch"):
            MODULE.summarize_paper_surface(phase_drift, metadata)

    def test_formal_macro_requires_frozen_provenance_ids_and_phase_map(self):
        for surface in (MODULE.MATH_SURFACE, MODULE.PHYS_SURFACE):
            with self.subTest(surface=surface):
                phase_counts = MODULE.FORMAL_TASK_PHASE_COUNTS[surface]
                metadata = _formal_metadata(surface)
                episodes = [
                    _formal_episode(position, surface=surface)
                    for position in range(len(phase_counts))
                ]
                summary = MODULE.summarize_paper_surface(episodes, metadata)
                self.assertTrue(summary["panel_complete"])
                self.assertTrue(summary["paper_panel_complete"])
                self.assertFalse(summary["paper_macro_eligible"])
                self.assertEqual(
                    summary["paper_metrics"]["phase_count"],
                    sum(phase_counts),
                )

                fake_metadata = _formal_metadata(surface)
                fake_metadata["task_count"] = 1
                with self.assertRaisesRegex(MODULE.EvalError, "task_count"):
                    MODULE.summarize_paper_surface([episodes[0]], fake_metadata)

                task_id_drift = copy.deepcopy(episodes)
                task_id_drift[0]["steps"][-1]["env_info_after"][
                    "domain_evidence"
                ]["task_id"] = "forged"
                with self.assertRaisesRegex(MODULE.EvalError, "task_id mismatch"):
                    MODULE.summarize_paper_surface(task_id_drift, metadata)

                phase_drift = copy.deepcopy(episodes)
                phase_drift[0]["steps"][-1]["env_info_after"]["phase_count"] += 1
                with self.assertRaisesRegex(MODULE.EvalError, "phase_count mismatch"):
                    MODULE.summarize_paper_surface(phase_drift, metadata)

                provenance_drift = _formal_metadata(surface)
                provenance_drift["dataset_provenance"]["sha256"] = "0" * 64
                with self.assertRaisesRegex(MODULE.EvalError, "provenance mismatch"):
                    MODULE.summarize_paper_surface(episodes, provenance_drift)

    def test_formal_paper_eval_full_panels_are_macro_eligible(self):
        for surface in (
            MODULE.MATH_PAPER_EVAL_SURFACE,
            MODULE.PHYS_PAPER_EVAL_SURFACE,
        ):
            with self.subTest(surface=surface):
                phase_counts = MODULE.FORMAL_TASK_PHASE_COUNTS[surface]
                episodes = [
                    _formal_episode(position, surface=surface)
                    for position in range(len(phase_counts))
                ]
                first_results = [False] + [True] * (phase_counts[0] - 1)
                final_wrong_results = [True] * (phase_counts[1] - 1) + [False]
                episodes[0] = _formal_episode(
                    0, surface=surface, results=first_results
                )
                episodes[1] = _formal_episode(
                    1, surface=surface, results=final_wrong_results
                )

                summary = MODULE.summarize_paper_surface(
                    episodes, _formal_metadata(surface)
                )
                expected_ps = 1.0 - (
                    (1.0 / phase_counts[0]) + (1.0 / phase_counts[1])
                ) / len(phase_counts)
                expected_sr = (len(phase_counts) - 1) / len(phase_counts)
                self.assertTrue(summary["panel_complete"])
                self.assertTrue(summary["paper_panel_complete"])
                self.assertTrue(summary["paper_macro_eligible"])
                self.assertEqual(
                    summary["paper_metric_contract"],
                    MODULE.FORMAL_PAPER_METRIC_CONTRACT,
                )
                self.assertAlmostEqual(
                    summary["paper_metrics"]["process_score"], expected_ps
                )
                self.assertAlmostEqual(summary["paper_success_rate"], expected_sr)

    def test_formal_panel_rejects_runtime_provenance_and_step_forgery(self):
        surface = MODULE.MATH_SURFACE
        metadata = _formal_metadata(surface)
        episode = _formal_episode(0, surface=surface)
        mutations = {
            "wrong contract": lambda item: item.update(
                {"contract_id": "forged-formal-contract"}
            ),
            "wrong judge": lambda item: item["judge_provenance"].update(
                {"model": "different-judge"}
            ),
            "wrong source tree": lambda item: item["upstream_provenance"].update(
                {"env_git_tree_oid": "0" * 40}
            ),
            "reward overlay": lambda item: item.update(
                {"reward_overlay": "unexpected-shaping"}
            ),
        }
        for label, mutate in mutations.items():
            drift = copy.deepcopy(metadata)
            mutate(drift)
            with self.subTest(label=label), self.assertRaises(MODULE.EvalError):
                MODULE.summarize_paper_surface([episode], drift)

        single_jump = copy.deepcopy(episode)
        terminal = copy.deepcopy(single_jump["steps"][-1])
        terminal["turn"] = 1
        terminal["env_info_before"] = copy.deepcopy(
            single_jump["initial_env_info"]
        )
        terminal["env_info_after"]["action_execution"]["step"] = 1
        terminal["env_info_after"]["reward_components"][0]["step"] = 1
        terminal["env_info_after"]["tool_ops"][0]["step"] = 1
        single_jump["steps"] = [terminal]
        with self.assertRaisesRegex(MODULE.EvalError, "phase jump"):
            MODULE.summarize_paper_surface([single_jump], metadata)

        wrong_reward = copy.deepcopy(episode)
        wrong_reward["steps"][0]["reward"] = 0.0
        with self.assertRaisesRegex(MODULE.EvalError, "reward ledger mismatch"):
            MODULE.summarize_paper_surface([wrong_reward], metadata)

    def test_cli_model_key_is_not_sent_to_environment(self):
        system_prompt = "Canonical test prompt: use ADVANCE {}."
        metadata = {
            "formal_schema_version": MODULE.FORMAL_SCHEMA_V3,
            "surface": "formal_reasoning_math_v3",
            "domain_id": "formal_reasoning_math",
            "contract_id": "formal_reasoning_math_v3",
            "contract_sha256": "a" * 64,
            "system_prompt": system_prompt,
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
        }
        fake = _FakeOpen(metadata)
        original_urlopen = MODULE.urlopen
        MODULE.urlopen = fake
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with redirect_stdout(io.StringIO()):
                    result = MODULE.main(
                        [
                            "--env-url",
                            "http://env.test",
                            "--model-url",
                            "http://model.test/v1",
                            "--model",
                            "test-model",
                            "--indices",
                            "0",
                            "--max-policy-turns",
                            "1",
                            "--output-dir",
                            tmp,
                            "--api-key",
                            "model-secret",
                        ]
                    )
        finally:
            MODULE.urlopen = original_urlopen
        self.assertEqual(result, 0)
        for url, authorization in fake.authorization_headers:
            if url.startswith("http://env.test"):
                self.assertIsNone(authorization)
            elif url.startswith("http://model.test"):
                self.assertEqual(authorization, "Bearer model-secret")

    def test_webshop_v2_fallback_matches_training_prompt_exactly(self):
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.PreTrainedTokenizer = object
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object
        original_transformers = sys.modules.get("transformers")
        original_torch = sys.modules.get("torch")
        sys.modules["transformers"] = transformers_stub
        sys.modules["torch"] = torch_stub
        try:
            schemas_path = ROOT / "verl/workers/rollout/schemas.py"
            spec = importlib.util.spec_from_file_location(
                "eval_v3_prompt_schema_for_test", schemas_path
            )
            schemas = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(schemas)
        finally:
            if original_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = original_transformers
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch
        self.assertEqual(
            MODULE.LEGACY_WEBSHOP_SYSTEM_PROMPT,
            schemas.agentmemory_action_system_prompt(),
        )

    def test_readme_pins_five_surface_paper_macro(self):
        readme = (ROOT / "scripts/agentmemory/README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "macro5 = mean(Shopping_SR, Travel_SR, Search_SR, Math_SR, Physics_SR)",
            readme,
        )
        self.assertIn("auxiliary diagnostic", readme)
        self.assertIn("must not replace the five-column macro", readme)
        self.assertIn("must not be presented as the original MemoryArena", readme)
        self.assertIn("failfast_v3", readme)
        self.assertIn("paper_macro_eligible=false", readme)
        self.assertIn(MODULE.TRAVEL_FAILFAST_SURFACE, readme)
        self.assertIn(MODULE.TRAVEL_PAPER_EVAL_SURFACE, readme)

    def test_one_episode_records_latest_prompt_and_reward_ledger(self):
        system_prompt = "Canonical test prompt: use ADVANCE {}."
        metadata = {
            "formal_schema_version": MODULE.FORMAL_SCHEMA_V3,
            "surface": "formal_reasoning_math_v3",
            "domain_id": "formal_reasoning_math",
            "contract_id": "formal_reasoning_math_v3",
            "contract_sha256": "a" * 64,
            "system_prompt": system_prompt,
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "native_action_descriptions": ["ADVANCE {}"],
            "task_count": 1,
        }
        fake = _FakeOpen(metadata)
        transport = MODULE.JsonHttp(opener=fake)
        env = MODULE.AgentMemoryEnvClient("http://env.test", transport)
        model = MODULE.OpenAIChatClient(
            "http://model.test/v1",
            "test-model",
            transport,
            max_tokens=32,
            temperature=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = MODULE.EvalRunner(
                env,
                model,
                indices=[0],
                max_policy_turns=3,
                output_dir=Path(tmp),
            ).run()
            episode = manifest["episodes"][0]
            step = episode["steps"][0]
            self.assertEqual(episode["episode_return"], 1.0)
            self.assertTrue(episode["episode_success"])
            self.assertTrue(step["prompt_token_ids_exact"])
            self.assertEqual(step["prompt_token_ids"], [101, 102, 103])
            self.assertEqual(
                step["prompt_token_ids_hash"], MODULE.token_ids_hash([101, 102, 103])
            )
            self.assertFalse(step["response_token_ids_exact"])
            self.assertEqual(step["reward_components_sum"], 1.0)
            self.assertTrue(step["reward_components_match"])
            self.assertEqual(step["phase_progress"]["phase_index_before"], 0)
            self.assertEqual(step["phase_progress"]["phase_index_after"], 1)
            self.assertEqual(step["phase_progress"]["workflow_progress"], 1.0)
            self.assertNotIn("progress_score", step["phase_progress"])
            self.assertEqual(step["action_submitted"], "Action: ADVANCE {}")
            self.assertEqual(
                manifest["summary"]["final_phase_progress_distribution"],
                {"0/1": 0, "1/1": 1},
            )
            self.assertEqual(json.loads((Path(tmp) / "manifest.json").read_text())["schema_version"], MODULE.EVAL_SCHEMA)

        chat_requests = [item for item in fake.requests if item[1].endswith("/chat/completions")]
        self.assertEqual(len(chat_requests), 1)
        messages = chat_requests[0][2]["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], "phase zero")
        self.assertNotIn("ADVANCE", messages[-1]["content"])
        self.assertEqual(
            chat_requests[0][2]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        tokenize_requests = [
            item for item in fake.requests if item[1].endswith("/tokenize")
        ]
        self.assertEqual(
            tokenize_requests[0][2]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertFalse(manifest["model"]["enable_thinking"])

    def test_native_thinking_flag_is_bound_to_both_model_requests(self):
        fake = _FakeOpen(
            {
                "formal_schema_version": MODULE.FORMAL_SCHEMA_V3,
                "surface": "formal_reasoning_math_v3",
                "domain_id": "formal_reasoning_math",
                "task_count": 1,
                "contract_id": "formal_reasoning_math_failfast_v3",
                "contract_sha256": "a" * 64,
                "system_prompt": "Canonical test prompt.",
                "system_prompt_sha256": hashlib.sha256(
                    b"Canonical test prompt."
                ).hexdigest(),
            }
        )
        transport = MODULE.JsonHttp(opener=fake)
        env = MODULE.AgentMemoryEnvClient("http://env.test", transport)
        model = MODULE.OpenAIChatClient(
            "http://model.test/v1",
            "test-model",
            transport,
            max_tokens=8,
            temperature=0.0,
            enable_thinking=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = MODULE.EvalRunner(
                env,
                model,
                indices=[0],
                max_policy_turns=1,
                output_dir=Path(tmp),
            ).run()
        model_requests = [
            body
            for method, url, body in fake.requests
            if method == "POST"
            and (url.endswith("/tokenize") or url.endswith("/chat/completions"))
        ]
        self.assertEqual(len(model_requests), 2)
        self.assertTrue(manifest["model"]["enable_thinking"])
        for request in model_requests:
            self.assertEqual(
                request["chat_template_kwargs"], {"enable_thinking": True}
            )

    def test_tokenize_endpoint_is_fail_closed(self):
        model = MODULE.OpenAIChatClient(
            "http://model.test/v1",
            "test-model",
            MODULE.JsonHttp(opener=lambda request, timeout: _Response({})),
            max_tokens=8,
            temperature=0.0,
        )
        with self.assertRaises(MODULE.TokenizationError):
            model.tokenize([{"role": "system", "content": "x"}])

    def test_missing_episode_success_is_fail_closed(self):
        metadata = {
            "formal_schema_version": MODULE.FORMAL_SCHEMA_V3,
            "surface": "formal_reasoning_math_v3",
            "domain_id": "formal_reasoning_math",
            "contract_id": "formal_reasoning_math_v3",
            "contract_sha256": "a" * 64,
            "system_prompt": "Canonical test prompt.",
            "system_prompt_sha256": hashlib.sha256(
                b"Canonical test prompt."
            ).hexdigest(),
        }
        fake = _FakeOpen(metadata)
        fake.env_info_after.pop("episode_success")
        env = MODULE.AgentMemoryEnvClient(
            "http://env.test", MODULE.JsonHttp(opener=fake)
        )
        model = MODULE.OpenAIChatClient(
            "http://model.test/v1",
            "test-model",
            MODULE.JsonHttp(opener=fake),
            max_tokens=8,
            temperature=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                MODULE.EvalError, "missing authoritative episode_success"
            ):
                MODULE.EvalRunner(
                    env,
                    model,
                    indices=[0],
                    max_policy_turns=1,
                    output_dir=Path(tmp),
                ).run()

    def test_summary_missing_episode_success_is_fail_closed(self):
        with self.assertRaisesRegex(MODULE.EvalError, "missing authoritative"):
            MODULE.summarize_episodes([{"episode_return": 0.0}])

    def test_indices_support_ranges_without_duplicates(self):
        self.assertEqual(MODULE.parse_indices("0,2-4,3,7"), [0, 2, 3, 4, 7])
        with self.assertRaises(ValueError):
            MODULE.parse_indices("4-2")

    def test_turn_limit_defaults_to_attested_runtime_not_twenty(self):
        self.assertEqual(
            MODULE.resolve_max_policy_turns({"max_steps": 811}, None),
            811,
        )
        self.assertEqual(
            MODULE.resolve_max_policy_turns(
                {"surface": MODULE.WEBSHOP_V2_SURFACE},
                None,
            ),
            MODULE.LEGACY_WEBSHOP_MAX_POLICY_TURNS,
        )
        self.assertEqual(
            MODULE.resolve_max_policy_turns({"max_steps": 811}, 37),
            37,
        )
        with self.assertRaisesRegex(MODULE.EvalError, "lacks a positive max_steps"):
            MODULE.resolve_max_policy_turns({"surface": "unknown"}, None)

    def test_native_webshop_v2_uses_canonical_fallback_and_parses_react(self):
        metadata = {
            "surface": MODULE.WEBSHOP_V2_SURFACE,
            "task_count": 1,
        }
        fake = _FakeOpen(metadata, model_text="search[item]")
        transport = MODULE.JsonHttp(opener=fake)
        env = MODULE.AgentMemoryEnvClient("http://env.test", transport)
        self.assertEqual(env.system_prompt_source, "webshop_v2_rollout_fallback")
        self.assertEqual(env.metadata["ltm_inventory_mode"], "hidden")
        self.assertEqual(env.metadata["memory_prompt_mode"], "legacy")
        self.assertNotIn("key-only long-term memory inventory", env.system_prompt)
        self.assertEqual(
            MODULE.extract_webshop_v2_action(
                "Thought: search\n\nAction:\nsearch[wireless headphones]"
            ),
            "search[wireless headphones]",
        )
        self.assertEqual(
            MODULE.extract_webshop_v2_action(
                'Thought: save\n\nAction:\nADD {"key":"tv","value":"42 inch"}'
            ),
            'ADD {"key": "tv", "value": "42 inch"}',
        )
        malformed = "prefix search[wireless headphones] suffix"
        self.assertEqual(
            MODULE.extract_webshop_v2_action(malformed),
            malformed,
        )
        model = MODULE.OpenAIChatClient(
            "http://model.test/v1",
            "test-model",
            transport,
            max_tokens=16,
            temperature=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            runner = MODULE.EvalRunner(
                env,
                model,
                indices=[0],
                max_policy_turns=1,
                output_dir=Path(tmp),
            )
            episode = runner.run_episode(0)
            env.close()
        self.assertTrue(episode["episode_success"])
        step_requests = [
            body
            for method, url, body in fake.requests
            if method == "POST" and url.endswith("/step")
        ]
        self.assertEqual(step_requests[-1]["action"], "search[item]")

    def test_native_webshop_v2_derives_key_inventory_prompt_from_server(self):
        metadata = {
            "surface": MODULE.WEBSHOP_V2_SURFACE,
            "task_count": 1,
            "ltm_inventory_mode": "keys",
        }
        env = MODULE.AgentMemoryEnvClient(
            "http://env.test",
            MODULE.JsonHttp(opener=_FakeOpen(metadata)),
        )

        self.assertEqual(env.metadata["ltm_inventory_mode"], "keys")
        self.assertIn("key-only long-term memory inventory", env.system_prompt)
        self.assertIn("at most 24", env.system_prompt)

    def test_native_webshop_v2_rejects_unknown_inventory_mode(self):
        metadata = {
            "surface": MODULE.WEBSHOP_V2_SURFACE,
            "task_count": 1,
            "ltm_inventory_mode": "values",
        }
        with self.assertRaisesRegex(MODULE.EvalError, "ltm_inventory_mode"):
            MODULE.AgentMemoryEnvClient(
                "http://env.test",
                MODULE.JsonHttp(opener=_FakeOpen(metadata)),
            )

    def test_native_webshop_v2_derives_neutral_prompt_from_server_mode(self):
        metadata = {
            "surface": MODULE.WEBSHOP_V2_SURFACE,
            "task_count": 1,
            "memory_prompt_mode": "neutral",
        }
        env = MODULE.AgentMemoryEnvClient(
            "http://env.test",
            MODULE.JsonHttp(opener=_FakeOpen(metadata)),
        )

        self.assertEqual(env.metadata["memory_prompt_mode"], "neutral")
        self.assertIn("ADD requires key:string", env.system_prompt)
        self.assertIn("RETRIEVE requires query:string", env.system_prompt)
        self.assertNotIn("use ADD before click[Buy Now]", env.system_prompt)
        self.assertNotIn(
            "At the start of every later shopping session",
            env.system_prompt,
        )

    def test_native_webshop_v2_derives_neutral_horizon_prompt_from_server_mode(self):
        metadata = {
            "surface": MODULE.WEBSHOP_V2_SURFACE,
            "task_count": 1,
            "memory_prompt_mode": "neutral_horizon",
        }
        env = MODULE.AgentMemoryEnvClient(
            "http://env.test",
            MODULE.JsonHttp(opener=_FakeOpen(metadata)),
        )

        self.assertEqual(env.metadata["memory_prompt_mode"], "neutral_horizon")
        self.assertIn("six sequential shopping sessions", env.system_prompt)
        self.assertIn(
            "may refer to products purchased in earlier sessions",
            env.system_prompt,
        )
        self.assertNotIn("use ADD before click[Buy Now]", env.system_prompt)
        self.assertNotIn(
            "At the start of every later shopping session",
            env.system_prompt,
        )

    def test_native_webshop_v2_rejects_unknown_memory_prompt_mode(self):
        metadata = {
            "surface": MODULE.WEBSHOP_V2_SURFACE,
            "task_count": 1,
            "memory_prompt_mode": "instruction",
        }
        with self.assertRaisesRegex(MODULE.EvalError, "memory_prompt_mode"):
            MODULE.AgentMemoryEnvClient(
                "http://env.test",
                MODULE.JsonHttp(opener=_FakeOpen(metadata)),
            )

    def test_native_webshop_v2_rejects_prompt_drift(self):
        with self.assertRaisesRegex(MODULE.EvalError, "disagrees"):
            MODULE.AgentMemoryEnvClient(
                "http://env.test",
                MODULE.JsonHttp(
                    opener=_FakeOpen(
                        {
                            "surface": MODULE.WEBSHOP_V2_SURFACE,
                            "system_prompt": "different prompt",
                        }
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
