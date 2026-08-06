from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
_MISSING = object()
_ROLLOUT_MODULE_NAMES = (
    "verl",
    "verl.utils",
    "verl.utils.agentgym",
    "verl.utils.agentgym.formal_grpo_credit",
    "verl.utils.agentgym.formal_domain_v3",
    "formal_domain_rollout_context_for_test",
)


def _restore_module(name: str, previous) -> None:
    if previous is _MISSING:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def load_module(name: str, relative_path: str, *, keep_registered: bool = False):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    previous = sys.modules.get(name, _MISSING)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if not keep_registered:
            _restore_module(name, previous)
    return module


def load_rollout_context():
    previous = {
        name: sys.modules.get(name, _MISSING) for name in _ROLLOUT_MODULE_NAMES
    }
    try:
        verl_module = types.ModuleType("verl")
        verl_module.__path__ = []
        utils_module = types.ModuleType("verl.utils")
        utils_module.__path__ = []
        agentgym_module = types.ModuleType("verl.utils.agentgym")
        agentgym_module.__path__ = []
        setattr(verl_module, "utils", utils_module)
        setattr(utils_module, "agentgym", agentgym_module)
        sys.modules["verl"] = verl_module
        sys.modules["verl.utils"] = utils_module
        sys.modules["verl.utils.agentgym"] = agentgym_module
        formal_grpo = load_module(
            "verl.utils.agentgym.formal_grpo_credit",
            "verl/utils/agentgym/formal_grpo_credit.py",
            keep_registered=True,
        )
        formal_domain = load_module(
            "verl.utils.agentgym.formal_domain_v3",
            "verl/utils/agentgym/formal_domain_v3.py",
            keep_registered=True,
        )
        rollout_context = load_module(
            "formal_domain_rollout_context_for_test",
            "verl/utils/agentgym/rollout_context.py",
            keep_registered=True,
        )
        return formal_grpo, formal_domain, rollout_context
    finally:
        for name in reversed(_ROLLOUT_MODULE_NAMES):
            _restore_module(name, previous[name])


FORMAL_GRPO, FORMAL_DOMAIN, ROLLOUT_CONTEXT = load_rollout_context()


def env_info(*, phase: int, done: bool, episode_success: bool, reward: float):
    action_execution = {
        "raw_policy_output": "Action: ADVANCE {}",
        "submitted_action": "ADVANCE {}",
        "op": "ADVANCE",
        "status": "executed",
        "step": 1,
    }
    return {
        "formal_schema_version": FORMAL_DOMAIN.FORMAL_DOMAIN_SCHEMA_V3,
        "domain_id": "fake",
        "surface": "fake_v3",
        "contract_id": "fake_v1",
        "contract_sha256": "a" * 64,
        "phase_index": phase,
        "phase_count": 2,
        "episode_success": episode_success,
        "done": done,
        "action_execution": action_execution,
        "tool_ops": [{"op": "ADVANCE", "step": 1}],
        "reward_components": [
            {"name": "phase_advance", "value": reward, "op": "ADVANCE", "step": 1}
        ],
        "domain_evidence": {"phase_advanced": True},
        "sample_excluded": False,
    }


QWEN35_EOS_TOKEN_IDS = [248044, 248046]
QWEN35_PRIMARY_EOS_TOKEN_ID = 248046


def packed_v3_record(
    *,
    terminal_eos_token_id: int = QWEN35_PRIMARY_EOS_TOKEN_ID,
    stop_reason: int | None = None,
):
    system_prompt = "SERVER CANONICAL PROMPT: use ADVANCE with exact JSON grammar."
    latest_observation = "Current phase observation."
    visible_prompt = f"<system>{system_prompt}</system>\n{latest_observation}"
    prompt_token_ids = [101, 102, 103]
    prompt_digest = ROLLOUT_CONTEXT.prompt_token_digest(prompt_token_ids)
    response_token_ids = [201, terminal_eos_token_id]
    response_digest = ROLLOUT_CONTEXT.prompt_token_digest(response_token_ids)
    exact_state_uid = f"0:turn1:statev1:{prompt_digest}"
    trajectory_uid = "agentmemory:parentv1:0:replica0"
    before = env_info(phase=1, done=False, episode_success=False, reward=0.0)
    before["action_execution"] = {}
    before["tool_ops"] = []
    before["reward_components"] = []
    before["domain_evidence"] = {"phase_advanced": False}
    after = env_info(phase=2, done=True, episode_success=True, reward=1.0)
    record = FORMAL_DOMAIN.build_formal_domain_step_v3(
        content="Action: ADVANCE {}",
        score=1.0,
        task_round=1,
        done=True,
        item_id="0",
        parent_index=0,
        parent_group_uid="agentmemory:parentv1:0",
        replica_index=0,
        trajectory_uid=trajectory_uid,
        exact_state_uid=exact_state_uid,
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
        latest_observation=latest_observation,
        visible_prompt=visible_prompt,
        system_prompt=system_prompt,
        single_observation_prompt_digest=prompt_digest,
        env_result="terminal observation",
        generation_record={
            "response_token_count": len(response_token_ids),
            "max_response_tokens": 8,
            "finish_reason": "stop",
            "finish_reason_source": "official_vllm:backend",
            "stop_reason": stop_reason,
            "backend_source": "official_vllm",
            "configured_eos_token_ids": QWEN35_EOS_TOKEN_IDS,
            "primary_eos_token_id": QWEN35_PRIMARY_EOS_TOKEN_ID,
            "tokenizer_pad_token_id": 248044,
            "token_ids_are_exact": True,
            "backend_token_ids_are_exact": True,
            "truncated": False,
        },
        env_info_before=before,
        env_info_after=after,
    )
    record.pop("prompt_token_ids")
    record.update(
        {
            "trajectory_row_uid": FORMAL_GRPO.build_row_uid(trajectory_uid, 0),
            "trajectory_row_order": 0,
            "trajectory_terminal": True,
            "action": "Action: ADVANCE {}",
            "immediate_reward": 1.0,
            "suffix_return": 1.0,
            "suffix_credit_applied": False,
            "trajectory_return": 1.0,
            "generation_prompt_length": len(prompt_token_ids),
            "generation_prompt_digest": prompt_digest,
            "packed_prompt_length": len(prompt_token_ids),
            "packed_prompt_digest": prompt_digest,
            "generation_response_length": len(response_token_ids),
            "generation_response_digest": response_digest,
            "packed_response_length": len(response_token_ids),
            "packed_response_digest": response_digest,
        }
    )
    return record


def packed_webshop_v2_record(
    *,
    terminal_eos_token_id: int = QWEN35_PRIMARY_EOS_TOKEN_ID,
    stop_reason: int | None = None,
):
    latest_observation = "Current WebShop observation."
    visible_prompt = f"<system>WebShop tools</system>\n{latest_observation}"
    prompt_token_ids = [101, 102, 103]
    prompt_digest = ROLLOUT_CONTEXT.prompt_token_digest(prompt_token_ids)
    response_token_ids = [201, terminal_eos_token_id]
    response_digest = ROLLOUT_CONTEXT.prompt_token_digest(response_token_ids)
    trajectory_uid = "agentmemory:parentv1:0:replica0"
    exact_state_uid = f"0:turn1:statev1:{prompt_digest}"
    action = 'search["displayed product"]'
    tool_op = {
        "op": "SEARCH",
        "step": 1,
        "raw_action": action,
        "result_count": 1,
    }
    reward_component = {
        "name": "search_transition",
        "op": "SEARCH",
        "step": 1,
        "value": 0.0,
    }
    env_info_before = {
        "current_subtask_index": 0,
        "episode_success": False,
        "session_trace": [],
        "tool_ops": [],
        "reward_components": [],
    }
    env_info_after = {
        "current_subtask_index": 0,
        "episode_success": False,
        "session_trace": [action],
        "tool_ops": [tool_op],
        "reward_components": [reward_component],
    }
    return {
        "schema_version": FORMAL_DOMAIN.FORMAL_WEBSHOP_SCHEMA_V2,
        "item_id": "0",
        "exact_state_uid": exact_state_uid,
        "trajectory_uid": trajectory_uid,
        "trajectory_row_uid": FORMAL_GRPO.build_row_uid(trajectory_uid, 0),
        "trajectory_row_order": 0,
        "trajectory_terminal": True,
        "task_round": 1,
        "session_index": 0,
        "subtask_index": 0,
        "next_session_index": 0,
        "subtask_index_before": 0,
        "subtask_index_after": 0,
        "visible_prompt": visible_prompt,
        "latest_observation": latest_observation,
        "prompt_history_policy": "latest_observation_only",
        "raw_prior_messages_visible": False,
        "single_observation_prompt_digest": prompt_digest,
        "action": action,
        "response_token_ids": response_token_ids,
        "response_token_count": len(response_token_ids),
        "max_response_tokens": 8,
        "finish_reason": "stop",
        "finish_reason_source": "official_vllm:backend",
        "stop_reason": stop_reason,
        "generation_backend_source": "official_vllm",
        "generation_stop_reason": stop_reason,
        "generation_eos_token_ids": QWEN35_EOS_TOKEN_IDS,
        "tokenizer_primary_eos_token_id": QWEN35_PRIMARY_EOS_TOKEN_ID,
        "tokenizer_pad_token_id": 248044,
        "generation_token_ids_are_exact": True,
        "backend_token_ids_are_exact": True,
        "truncated": False,
        "env_result": "Search results.",
        "env_info_before": env_info_before,
        "env_info_after": env_info_after,
        "action_submission": {
            "raw_policy_output": action,
            "submitted_action": action,
            "parser_status": "adapter_parsed",
        },
        "committed_purchase": False,
        "purchase_correct": None,
        "accepted_purchase": False,
        "session_advanced": False,
        "buy_committed": False,
        "buy_accepted": False,
        "subtask_advanced": False,
        "raw_history_cleared": False,
        "search_result_count": 1,
        "immediate_reward": 0.0,
        "suffix_return": 0.0,
        "suffix_credit_applied": False,
        "trajectory_return": 0.0,
        "done": False,
        "outcome": "continue",
        "generation_prompt_length": len(prompt_token_ids),
        "generation_prompt_digest": prompt_digest,
        "packed_prompt_length": len(prompt_token_ids),
        "packed_prompt_digest": prompt_digest,
        "generation_response_length": len(response_token_ids),
        "generation_response_digest": response_digest,
        "packed_response_length": len(response_token_ids),
        "packed_response_digest": response_digest,
    }


def packed_intent_clarification_v2_record(
    *,
    surface: str = FORMAL_DOMAIN.FORMAL_WEBSHOP_INTENT_CLARIFICATION_FILESYSTEM_SURFACE_V2,
):
    record = packed_webshop_v2_record()
    action = 'ASK {"field":"color"}'
    record["action"] = action
    record["action_submission"] = {
        "raw_policy_output": action,
        "submitted_action": action,
        "parser_status": "adapter_parsed",
    }
    record["env_result"] = "CLARIFY: For color, I want gray."
    record["env_info_before"]["surface"] = surface
    record["env_info_after"].update(
        {
            "surface": surface,
            "session_trace": [action],
            "tool_ops": [
                {
                    "op": "CLARIFY",
                    "request_op": "ASK",
                    "field": "color",
                    "clarification_received": True,
                    "step": 1,
                    "session_index": 0,
                }
            ],
            "reward_components": [
                {
                    "name": "clarify_transition",
                    "op": "CLARIFY",
                    "step": 1,
                    "value": 0.0,
                }
            ],
        }
    )
    record["search_result_count"] = None
    return record


def invalid_webshop_v2_record(raw_output: str, submitted_action: str):
    record = packed_webshop_v2_record()
    record["action"] = raw_output
    record["action_submission"] = {
        "raw_policy_output": raw_output,
        "submitted_action": submitted_action,
        "parser_status": "raw_fallback",
    }
    record["env_info_after"]["tool_ops"] = []
    record["env_info_after"]["reward_components"] = [
        {
            "name": "invalid_action",
            "op": "INVALID",
            "step": 1,
            "value": -0.01,
            "raw_action": submitted_action.strip(),
            "error": "unsupported action",
        }
    ]
    record["search_result_count"] = None
    for field in ("immediate_reward", "suffix_return", "trajectory_return"):
        record[field] = -0.01
    return record


def validate_one_record(record: dict):
    return ROLLOUT_CONTEXT.validate_formal_runtime_evidence_rows(
        exact_state_uids=[record["exact_state_uid"]],
        trajectory_uids=[record["trajectory_uid"]],
        trajectory_row_uids=[record["trajectory_row_uid"]],
        trajectory_row_orders=[record["trajectory_row_order"]],
        trajectory_terminals=[record["trajectory_terminal"]],
        task_rounds=[record["task_round"]],
        immediate_rewards=[record["immediate_reward"]],
        trajectory_returns=[record["trajectory_return"]],
        action_texts=[record["action"]],
        done_flags=[record["done"]],
        generation_prompt_lengths=[record["generation_prompt_length"]],
        generation_prompt_digests=[record["generation_prompt_digest"]],
        packed_prompt_lengths=[record["packed_prompt_length"]],
        packed_prompt_digests=[record["packed_prompt_digest"]],
        generation_response_lengths=[record["generation_response_length"]],
        generation_response_digests=[record["generation_response_digest"]],
        packed_response_lengths=[record["packed_response_length"]],
        packed_response_digests=[record["packed_response_digest"]],
        suffix_credit_applied=[record["suffix_credit_applied"]],
        suffix_returns=[record["suffix_return"]],
        step_record_jsons=[json.dumps(record)],
        valid_mask=[True],
        expected_suffix_credit=False,
        expected_prompt_width=16,
    )


class FormalDomainRolloutV3Test(unittest.TestCase):
    def test_isolated_loaders_restore_sys_modules(self):
        before = {
            name: (name in sys.modules, sys.modules.get(name))
            for name in _ROLLOUT_MODULE_NAMES
        }
        formal_grpo, formal_domain, rollout_context = load_rollout_context()
        self.assertTrue(callable(formal_grpo.build_row_uid))
        self.assertEqual(
            formal_domain.FORMAL_DOMAIN_SCHEMA_V3,
            "agentmemory_formal_step_v3",
        )
        self.assertTrue(callable(rollout_context.prompt_token_digest))
        after = {
            name: (name in sys.modules, sys.modules.get(name))
            for name in _ROLLOUT_MODULE_NAMES
        }
        for name in _ROLLOUT_MODULE_NAMES:
            with self.subTest(name=name):
                self.assertEqual(after[name][0], before[name][0])
                self.assertIs(after[name][1], before[name][1])

        probe_name = "formal_domain_v3_isolation_probe"
        probe_before = sys.modules.get(probe_name, _MISSING)
        loaded = load_module(
            probe_name,
            "verl/utils/agentgym/formal_domain_v3.py",
        )
        self.assertEqual(loaded.FORMAL_DOMAIN_SCHEMA_V3, "agentmemory_formal_step_v3")
        self.assertIs(sys.modules.get(probe_name, _MISSING), probe_before)

    def test_runtime_validator_accepts_domain_v3_without_buy_semantics(self):
        summary = validate_one_record(packed_v3_record())
        self.assertEqual(summary["valid_rows"], 1)
        self.assertEqual(summary["trajectory_count"], 1)

    def test_real_qwen35_primary_and_alternate_eos_pass_both_packed_schemas(self):
        cases = (
            (QWEN35_PRIMARY_EOS_TOKEN_ID, None),
            (248044, 248044),
        )
        for record_factory in (packed_webshop_v2_record, packed_v3_record):
            for terminal_eos_token_id, stop_reason in cases:
                with self.subTest(
                    schema=record_factory.__name__,
                    terminal_eos_token_id=terminal_eos_token_id,
                ):
                    summary = validate_one_record(
                        record_factory(
                            terminal_eos_token_id=terminal_eos_token_id,
                            stop_reason=stop_reason,
                        )
                    )
                    self.assertEqual(summary["valid_rows"], 1)

    def test_invalid_qwen35_primary_eos_metadata_fails_both_packed_schemas(self):
        for record_factory in (packed_webshop_v2_record, packed_v3_record):
            with self.subTest(schema=record_factory.__name__, case="missing"):
                record = record_factory()
                record.pop("tokenizer_primary_eos_token_id")
                with self.assertRaisesRegex(ValueError, "missing field"):
                    validate_one_record(record)

            with self.subTest(schema=record_factory.__name__, case="outside"):
                record = record_factory()
                record["tokenizer_primary_eos_token_id"] = 999999
                with self.assertRaisesRegex(ValueError, "primary EOS"):
                    validate_one_record(record)

    def test_swapped_qwen35_stop_reason_fails_both_packed_schemas(self):
        for record_factory in (packed_webshop_v2_record, packed_v3_record):
            cases = (
                (QWEN35_PRIMARY_EOS_TOKEN_ID, 248044),
                (248044, None),
            )
            for terminal_eos_token_id, stop_reason in cases:
                with self.subTest(
                    schema=record_factory.__name__,
                    terminal_eos_token_id=terminal_eos_token_id,
                ):
                    record = record_factory(
                        terminal_eos_token_id=terminal_eos_token_id,
                        stop_reason=stop_reason,
                    )
                    with self.assertRaisesRegex(ValueError, "EOS"):
                        validate_one_record(record)

    def test_runtime_validator_rejects_prompt_digest_drift(self):
        record = packed_v3_record()
        record["single_observation_prompt_digest"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "one latest observation"):
            validate_one_record(record)

    def test_runtime_validator_accepts_canonical_unicode_prompt_roundtrip(self):
        decomposed = "BEAUTe\u0301DERM latest observation."
        composed = "BEAUT\u00e9DERM latest observation."
        for record_factory in (packed_webshop_v2_record, packed_v3_record):
            with self.subTest(schema=record_factory.__name__):
                record = record_factory()
                record["latest_observation"] = decomposed
                system_prompt = record.get("system_prompt", "tools")
                record["visible_prompt"] = (
                    f"<system>{system_prompt}</system>\n{composed}"
                )
                summary = validate_one_record(record)
                self.assertEqual(summary["valid_rows"], 1)

    def test_runtime_validator_accepts_chat_template_trimmed_observation_whitespace(self):
        for record_factory in (packed_webshop_v2_record, packed_v3_record):
            with self.subTest(schema=record_factory.__name__):
                record = record_factory()
                record["latest_observation"] = "latest observation\n"
                record["visible_prompt"] = (
                    f"<system>{record.get('system_prompt', 'tools')}</system>\n"
                    "latest observation"
                )
                summary = validate_one_record(record)
                self.assertEqual(summary["valid_rows"], 1)

    def test_runtime_validator_accepts_canonical_unicode_system_prompt(self):
        record = packed_v3_record()
        decomposed = "Use the CAFE\u0301 tool contract."
        composed = "Use the CAF\u00c9 tool contract."
        record["system_prompt"] = decomposed
        record["system_prompt_sha256"] = hashlib.sha256(
            decomposed.encode("utf-8")
        ).hexdigest()
        record["visible_prompt"] = (
            f"<system>{composed}</system>\n{record['latest_observation']}"
        )
        summary = validate_one_record(record)
        self.assertEqual(summary["valid_rows"], 1)

    def test_runtime_validator_still_rejects_missing_latest_observation(self):
        for record_factory in (packed_webshop_v2_record, packed_v3_record):
            with self.subTest(schema=record_factory.__name__):
                record = record_factory()
                record["latest_observation"] = "Actually missing observation."
                with self.assertRaisesRegex(ValueError, "omits the latest observation"):
                    validate_one_record(record)

    def test_runtime_validator_rejects_server_raw_action_drift(self):
        record = packed_v3_record()
        record["action_execution"]["raw_policy_output"] = "different"
        record["env_info_after"]["action_execution"]["raw_policy_output"] = "different"
        with self.assertRaisesRegex(ValueError, "sampled content"):
            validate_one_record(record)

    def test_webshop_v2_binds_qwen_think_output_to_submitted_native_action(self):
        record = packed_webshop_v2_record()
        raw_output = (
            "<think>\nI need to find a relevant item. "
            "Let me search for red velvet cake mix first.\n</think>\n\n"
            "search[red velvet cake mix]"
        )
        record["action"] = raw_output
        record["action_submission"]["raw_policy_output"] = raw_output
        record["action_submission"]["submitted_action"] = (
            "search[red velvet cake mix]"
        )
        record["env_info_after"]["tool_ops"][0]["raw_action"] = (
            "search[red velvet cake mix]"
        )
        summary = validate_one_record(record)
        self.assertEqual(summary["valid_rows"], 1)

    def test_webshop_v2_binds_apply_patch_to_tool_event(self):
        record = packed_webshop_v2_record()
        action = (
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Add File: .agent_memory/MEMORY.md\n"
            "+color=gray\n"
            "*** End Patch"
        )
        record["action"] = action
        record["action_submission"] = {
            "raw_policy_output": action,
            "submitted_action": action,
            "parser_status": "adapter_parsed",
        }
        record["env_result"] = "Done!"
        record["env_info_after"]["session_trace"] = [action]
        record["env_info_after"]["tool_ops"] = [
            {
                "op": "APPLY_PATCH",
                "step": 1,
                "raw_action": action,
                "path": ".agent_memory/MEMORY.md",
                "workspace_tree_sha256_before": "0" * 64,
                "workspace_tree_sha256_after": "1" * 64,
            }
        ]
        record["env_info_after"]["reward_components"] = [
            {
                "name": "apply_patch_transition",
                "op": "APPLY_PATCH",
                "step": 1,
                "value": 0.0,
            }
        ]
        record["search_result_count"] = None
        summary = validate_one_record(record)
        self.assertEqual(summary["valid_rows"], 1)

    def test_webshop_v2_accepts_intent_ask_bound_to_clarify(self):
        for surface in (
            FORMAL_DOMAIN.FORMAL_WEBSHOP_INTENT_CLARIFICATION_SURFACE_V2,
            FORMAL_DOMAIN.FORMAL_WEBSHOP_INTENT_CLARIFICATION_FILESYSTEM_SURFACE_V2,
        ):
            with self.subTest(surface=surface):
                summary = validate_one_record(
                    packed_intent_clarification_v2_record(surface=surface)
                )
                self.assertEqual(summary["valid_rows"], 1)

    def test_webshop_v2_rejects_clarify_outside_intent_surface(self):
        record = packed_intent_clarification_v2_record(
            surface=FORMAL_DOMAIN.FORMAL_WEBSHOP_FILESYSTEM_SURFACE_V2
        )
        with self.assertRaisesRegex(ValueError, "unsupported tool operations"):
            validate_one_record(record)

    def test_webshop_v2_rejects_intent_surface_transition(self):
        record = packed_intent_clarification_v2_record()
        record["env_info_before"]["surface"] = (
            FORMAL_DOMAIN.FORMAL_WEBSHOP_FILESYSTEM_SURFACE_V2
        )
        with self.assertRaisesRegex(ValueError, "surface changed"):
            validate_one_record(record)

    def test_webshop_v2_rejects_incomplete_clarify_binding(self):
        mutations = {
            "request_op": lambda event: event.update(request_op="RETRIEVE"),
            "field": lambda event: event.update(field="size"),
            "receipt": lambda event: event.update(clarification_received=False),
            "session": lambda event: event.update(session_index=1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                record = packed_intent_clarification_v2_record()
                mutate(record["env_info_after"]["tool_ops"][0])
                with self.assertRaisesRegex(ValueError, "Formal CLARIFY"):
                    validate_one_record(record)

    def test_webshop_v2_rejects_shell_command_tool_misbinding(self):
        record = packed_webshop_v2_record()
        action = (
            'shell_command {"command":"cat notes.md","workdir":".",'
            '"timeout_ms":10000}'
        )
        record["action"] = action
        record["action_submission"] = {
            "raw_policy_output": action,
            "submitted_action": action,
            "parser_status": "adapter_parsed",
        }
        record["env_info_after"]["tool_ops"][0]["raw_action"] = action
        with self.assertRaisesRegex(
            ValueError,
            "Formal SHELL_COMMAND action is bound to SEARCH",
        ):
            validate_one_record(record)

    def test_webshop_v2_accepts_server_authoritative_invalid_wrapper(self):
        raw_output = '{"action": "click[Buy Now]"}'
        summary = validate_one_record(
            invalid_webshop_v2_record(raw_output, raw_output)
        )
        self.assertEqual(summary["valid_rows"], 1)

    def test_webshop_v2_accepts_eos_only_empty_invalid_submission(self):
        summary = validate_one_record(invalid_webshop_v2_record("", ""))
        self.assertEqual(summary["valid_rows"], 1)

    def test_webshop_v2_raw_fallback_removes_exactly_one_terminal_textual_eos(self):
        cases = (
            ("</s>", ""),
            ("</s></s>", "</s>"),
            ("   </s>", "   "),
        )
        for raw_output, submitted_action in cases:
            with self.subTest(raw_output=raw_output):
                summary = validate_one_record(
                    invalid_webshop_v2_record(raw_output, submitted_action)
                )
                self.assertEqual(summary["valid_rows"], 1)

    def test_webshop_v2_rejects_forged_raw_fallback_submission(self):
        with self.assertRaisesRegex(ValueError, "raw fallback"):
            validate_one_record(
                invalid_webshop_v2_record(
                    "raw-model-output",
                    "forged-submission",
                )
            )

    def test_webshop_v2_empty_submission_requires_authoritative_invalid_ledger(self):
        adapter_parsed = invalid_webshop_v2_record("", "")
        adapter_parsed["action_submission"]["parser_status"] = "adapter_parsed"
        with self.subTest(case="adapter_parsed"):
            with self.assertRaisesRegex(ValueError, "submitted action is empty"):
                validate_one_record(adapter_parsed)

        tool_event = invalid_webshop_v2_record("", "")
        tool_event["env_info_after"]["tool_ops"] = [
            {
                "op": "SEARCH",
                "step": 1,
                "raw_action": "",
                "result_count": 0,
            }
        ]
        with self.subTest(case="tool_event"):
            with self.assertRaisesRegex(ValueError, "claims a tool operation"):
                validate_one_record(tool_event)

        non_invalid_ledger = invalid_webshop_v2_record("", "")
        non_invalid_ledger["env_info_after"]["reward_components"][0]["name"] = (
            "format_penalty"
        )
        with self.subTest(case="non_invalid_ledger"):
            with self.assertRaisesRegex(ValueError, "lacks one invalid-action"):
                validate_one_record(non_invalid_ledger)

    def test_webshop_v2_rejects_forged_submitted_action_binding(self):
        record = packed_webshop_v2_record()
        record["action_submission"]["submitted_action"] = "click[Buy Now]"
        with self.assertRaisesRegex(ValueError, "bound to SEARCH|raw_action binding"):
            validate_one_record(record)

    def test_real_prompt_builder_receives_exact_metadata_prompt(self):
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.PreTrainedTokenizer = object
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object
        original_transformers = sys.modules.get("transformers")
        original_torch = sys.modules.get("torch")
        sys.modules["transformers"] = transformers_stub
        sys.modules["torch"] = torch_stub
        try:
            schemas = load_module(
                "formal_domain_schemas_for_test",
                "verl/workers/rollout/schemas.py",
            )
        finally:
            if original_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = original_transformers
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        class CapturingTokenizer:
            def __init__(self):
                self.conversations = None

            def apply_chat_template(self, conversations, **kwargs):
                self.conversations = deepcopy(conversations)
                return [7, 8, 9]

        system_prompt = "  COMPLETE SERVER PROMPT WITH Action: GRAMMAR  "
        observation = "latest observation"
        handler = schemas.RolloutHandler.__new__(schemas.RolloutHandler)
        handler.messages = [schemas.Message(role="user", content=observation)]
        tokenizer = CapturingTokenizer()
        token_ids = handler.get_latest_observation_prompt(
            tokenizer,
            system_prompt=system_prompt,
        )
        self.assertEqual(token_ids, [7, 8, 9])
        self.assertEqual(
            tokenizer.conversations,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": observation},
            ],
        )

    def test_rollout_source_keeps_v3_schema_and_skips_buy_validator(self):
        source_path = (
            ROOT
            / "verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py"
        )
        source = source_path.read_text(encoding="utf-8")
        pack_start = source.index("def pack_rollout_handlers")
        pack_end = source.index("def latest_observation_prompt_from_text", pack_start)
        pack_source = source[pack_start:pack_end]
        self.assertIn('schema_version = record.get(', pack_source)
        self.assertIn('"schema_version": schema_version', pack_source)
        self.assertNotIn(
            '"schema_version": "agentmemory_formal_step_v2"',
            pack_source,
        )

        rollout_start = source.index("def generate_agentmemory_latest_observation")
        rollout_end = source.index("@torch.no_grad()", rollout_start)
        rollout_source = source[rollout_start:rollout_end]
        v3_start = rollout_source.index(
            "if formal_schema_version == FORMAL_DOMAIN_SCHEMA_V3:"
        )
        v3_end = rollout_source.index("else:", v3_start)
        v3_source = rollout_source[v3_start:v3_end]
        self.assertIn("build_formal_domain_step_v3", v3_source)
        self.assertNotIn("validate_formal_buy_transition", v3_source)
        self.assertIn(
            "system_prompt=rollout_handler.formal_system_prompt",
            rollout_source,
        )
        self.assertIn("system_prompt=formal_system_prompt", rollout_source)
        self.assertIn("except FormalRuntimeEvidenceError:", rollout_source)


if __name__ == "__main__":
    unittest.main()
