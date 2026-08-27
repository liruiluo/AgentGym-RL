#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "agentmemory" / "verify_swesmith_ppo_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_swesmith_ppo_gate", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load SWE-smith PPO verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_successor(checkpoint_sha256: str, size: int = 128) -> str:
    return (
        "task observation\n\nEarlier conversation was removed after the "
        "continuation snapshot write succeeded. The workspace persists, but "
        "`.agent_memory/CONTINUATION.md` was not copied into this prompt. Use "
        "the next normal action to read that file, then continue from its "
        "evidence and next action. Other workspace files remain available and "
        "may still be read or updated normally. Verified receipt: "
        f"size_bytes={size}, sha256={checkpoint_sha256}."
    )


class SwesmithPpoGateRowEvidenceTests(unittest.TestCase):
    @staticmethod
    def _response_cap_record() -> dict:
        return {
            "truncated": True,
            "finish_reason": "length",
            "finish_reason_source": "official_vllm:backend",
            "generation_stop_reason": None,
            "stop_reason": None,
            "response_token_count": 2048,
            "max_response_tokens": 2048,
            "generation_response_length": 2048,
            "packed_response_length": 2048,
            "generation_token_ids_are_exact": True,
            "backend_token_ids_are_exact": True,
            "done": False,
            "trajectory_terminal": False,
            "outcome": "continue",
            "immediate_reward": 0.0,
            "task_round": 18,
        }

    def test_accepts_exact_backend_response_cap_as_negative_row(self) -> None:
        module = load_module()
        result = module.verify_response_cap_truncation(
            self._response_cap_record(),
            parent_index=43,
        )
        self.assertEqual(result["kind"], "exact_backend_response_cap")
        self.assertEqual(result["parent_index"], 43)
        self.assertEqual(result["response_token_count"], 2048)

    def test_rejects_harness_or_context_truncation(self) -> None:
        module = load_module()
        record = self._response_cap_record()
        record["finish_reason_source"] = "task_neutral_harness"
        with self.assertRaises(AssertionError):
            module.verify_response_cap_truncation(record, parent_index=43)

    def test_rejects_inexact_or_short_response_cap(self) -> None:
        module = load_module()
        for key, value in (
            ("backend_token_ids_are_exact", False),
            ("response_token_count", 2047),
        ):
            with self.subTest(key=key):
                record = self._response_cap_record()
                record[key] = value
                with self.assertRaises(AssertionError):
                    module.verify_response_cap_truncation(record, parent_index=43)

    def test_verifies_native_and_compaction_step_continuity(self) -> None:
        module = load_module()
        native = {
            "wrapper_evidence": {
                "event": module.NATIVE_EVENT,
                "workspace_continuity_id": 9,
            },
            "env_info_before": {"step": 4},
            "env_info_after": {"step": 5, "action_kind": "shell_command"},
            "context_transition": {
                "schema": "agentmemory_task_neutral_context_transition_v1",
                "operation": "append_observation",
                "messages": [],
            },
            "action_submission": {"submitted_action": "shell_command"},
            "immediate_reward": 0.0,
            "trajectory_terminal": False,
            "done": False,
            "outcome": "continue",
        }
        result = module.verify_wrapper_transition(native, previous_native_step=4)
        self.assertEqual(result["native_step_after"], 5)
        self.assertEqual(result["action_kind"], "shell_command")

        checkpoint_sha256 = "a" * 64
        checkpoint_receipt = {
            "schema": module.FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA,
            "path": module.FILESYSTEM_CHECKPOINT_PATH,
            "action_kind": "shell_command",
            "action_completed": True,
            "changed": True,
            "exists": True,
            "regular_file": True,
            "size_bytes": 128,
            "sha256": checkpoint_sha256,
        }
        action = (
            'shell_command {"command":"write .agent_memory/CONTINUATION.md"}'
        )
        compaction = {
            "action": action,
            "wrapper_evidence": {
                "event": module.COMPACTION_EVENT,
                "workspace_continuity_id": 9,
                "continuation_path": module.FILESYSTEM_CHECKPOINT_PATH,
                "continuation_max_bytes": module.FILESYSTEM_CHECKPOINT_MAX_BYTES,
                "continuation_persisted": True,
                "checkpoint_receipt": checkpoint_receipt,
                "checkpoint_failure_reason": None,
                "context_replaced": True,
                "retry_pending": False,
                "checkpoint_retry_observation_bounded": False,
                "preserved_policy_output": True,
                "preserved_native_observation": True,
                "checkpoint_action_in_successor_context": False,
                "checkpoint_observation_in_successor_context": False,
                "checkpoint_content_in_successor_context": False,
            },
            "env_info_before": {"step": 5},
            "env_info_after": {
                "step": 6,
                "action_kind": "shell_command",
                "filesystem_checkpoint": checkpoint_receipt,
            },
            "native_step_before": 5,
            "native_step_after": 6,
            "native_call_count_before": 5,
            "native_call_count_after": 6,
            "policy_step_before": 5,
            "policy_step_after": 6,
            "context_epoch_before": 0,
            "context_epoch_after": 1,
            "context_transition": {
                "schema": "agentmemory_task_neutral_context_transition_v1",
                "operation": "replace_messages",
                "messages": [
                    {"role": "system", "content": "task framing"},
                    {
                        "role": "user",
                        "content": checkpoint_successor(checkpoint_sha256),
                    },
                ],
            },
            "action_submission": {"raw_policy_output": action},
            "control_request": "Persist continuation state before compaction.",
            "immediate_reward": 0.0,
            "trajectory_terminal": False,
            "done": False,
        }
        result = module.verify_wrapper_transition(compaction, previous_native_step=5)
        self.assertEqual(result["native_step_after"], 6)
        self.assertEqual(result["action_kind"], "shell_command")
        self.assertTrue(result["context_replaced"])

    def test_accepts_failed_checkpoint_as_bounded_retry_without_replacement(self) -> None:
        module = load_module()
        action = 'shell_command {"command":"false"}'
        record = {
            "action": action,
            "wrapper_evidence": {
                "event": module.COMPACTION_EVENT,
                "workspace_continuity_id": 9,
                "continuation_path": module.FILESYSTEM_CHECKPOINT_PATH,
                "continuation_max_bytes": module.FILESYSTEM_CHECKPOINT_MAX_BYTES,
                "continuation_persisted": False,
                "checkpoint_receipt": None,
                "checkpoint_failure_reason": "action_not_completed",
                "context_replaced": False,
                "retry_pending": True,
                "checkpoint_retry_observation_bounded": True,
                "preserved_policy_output": False,
                "preserved_native_observation": False,
                "checkpoint_action_in_successor_context": False,
                "checkpoint_observation_in_successor_context": False,
                "checkpoint_content_in_successor_context": False,
            },
            "env_info_before": {"step": 5},
            "env_info_after": {"step": 6, "action_kind": "shell_command"},
            "native_step_before": 5,
            "native_step_after": 6,
            "native_call_count_before": 5,
            "native_call_count_after": 6,
            "policy_step_before": 5,
            "policy_step_after": 6,
            "context_epoch_before": 0,
            "context_epoch_after": 0,
            "context_transition": {
                "schema": "agentmemory_task_neutral_context_transition_v1",
                "operation": "append_observation",
                "messages": [],
            },
            "action_submission": {"raw_policy_output": action},
            "immediate_reward": -0.01,
            "trajectory_terminal": False,
            "done": False,
        }
        result = module.verify_wrapper_transition(record, previous_native_step=5)
        self.assertEqual(result["native_step_after"], 6)
        self.assertFalse(result["context_replaced"])

    def test_rejects_checkpoint_counter_or_successor_marker_mismatch(self) -> None:
        module = load_module()
        checkpoint_sha256 = "a" * 64
        receipt = {
            "schema": module.FILESYSTEM_CHECKPOINT_RECEIPT_SCHEMA,
            "path": module.FILESYSTEM_CHECKPOINT_PATH,
            "action_kind": "shell_command",
            "action_completed": True,
            "changed": True,
            "exists": True,
            "regular_file": True,
            "size_bytes": 128,
            "sha256": checkpoint_sha256,
        }
        base = {
            "action": "shell_command {}",
            "wrapper_evidence": {
                "event": module.COMPACTION_EVENT,
                "workspace_continuity_id": 9,
                "continuation_path": module.FILESYSTEM_CHECKPOINT_PATH,
                "continuation_max_bytes": module.FILESYSTEM_CHECKPOINT_MAX_BYTES,
                "continuation_persisted": True,
                "checkpoint_receipt": receipt,
                "checkpoint_failure_reason": None,
                "context_replaced": True,
                "retry_pending": False,
                "checkpoint_retry_observation_bounded": False,
                "preserved_policy_output": True,
                "preserved_native_observation": True,
                "checkpoint_action_in_successor_context": False,
                "checkpoint_observation_in_successor_context": False,
                "checkpoint_content_in_successor_context": False,
            },
            "env_info_before": {"step": 5},
            "env_info_after": {
                "step": 6,
                "action_kind": "shell_command",
                "filesystem_checkpoint": receipt,
            },
            "native_step_before": 5,
            "native_step_after": 6,
            "native_call_count_before": 5,
            "native_call_count_after": 6,
            "policy_step_before": 5,
            "policy_step_after": 6,
            "context_epoch_before": 0,
            "context_epoch_after": 1,
            "context_transition": {
                "schema": "agentmemory_task_neutral_context_transition_v1",
                "operation": "replace_messages",
                "messages": [
                    {
                        "role": "user",
                        "content": checkpoint_successor(checkpoint_sha256),
                    }
                ],
            },
            "action_submission": {"raw_policy_output": "shell_command {}"},
            "done": False,
        }
        for mutate in (
            lambda value: value.update(policy_step_after=5),
            lambda value: value["context_transition"]["messages"][0].update(
                content="Read .agent_memory/CONTINUATION.md."
            ),
            lambda value: value["env_info_after"].pop("filesystem_checkpoint"),
            lambda value: value["wrapper_evidence"].pop(
                "preserved_policy_output"
            ),
            lambda value: value["context_transition"]["messages"][0].update(
                content=(
                    value["context_transition"]["messages"][0]["content"]
                    + "\n"
                    + value["action"]
                )
            ),
        ):
            with self.subTest(mutate=mutate):
                record = json.loads(json.dumps(base))
                mutate(record)
                with self.assertRaises(AssertionError):
                    module.verify_wrapper_transition(record, previous_native_step=5)

    @staticmethod
    def _checkpoint_read_record(module, *, digest: str = "a" * 64) -> dict:
        receipt = {
            "schema": module.FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA,
            "path": module.FILESYSTEM_CHECKPOINT_PATH,
            "observed": True,
            "size_bytes": 128,
            "sha256": digest,
        }
        action = (
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md"}'
        )
        return {
            "action": action,
            "wrapper_evidence": {
                "event": module.NATIVE_EVENT,
                "workspace_continuity_id": 9,
                "memory_event": "read",
                "document_read_observed": True,
                "filesystem_checkpoint_read": receipt,
            },
            "env_info_before": {"step": 6},
            "env_info_after": {
                "step": 7,
                "action_kind": "shell_command",
                "filesystem_checkpoint_read": receipt,
            },
            "native_step_before": 6,
            "native_step_after": 7,
            "native_call_count_before": 6,
            "native_call_count_after": 7,
            "policy_step_before": 6,
            "policy_step_after": 7,
            "context_epoch_before": 1,
            "context_epoch_after": 1,
            "context_transition": {
                "schema": "agentmemory_task_neutral_context_transition_v1",
                "operation": "append_observation",
                "messages": [],
            },
            "action_submission": {"raw_policy_output": action},
            "immediate_reward": 0.0,
            "trajectory_terminal": False,
            "done": False,
            "outcome": "continue",
        }

    def test_accepts_endpoint_attested_checkpoint_read(self) -> None:
        module = load_module()
        record = self._checkpoint_read_record(module)
        result = module.verify_wrapper_transition(record, previous_native_step=6)
        self.assertEqual(
            result["checkpoint_read_receipt"]["sha256"],
            "a" * 64,
        )
        self.assertEqual(result["context_epoch_after"], 1)

    def test_rejects_unbound_or_drifted_checkpoint_read(self) -> None:
        module = load_module()
        mutations = (
            lambda value: value["env_info_after"].pop(
                "filesystem_checkpoint_read"
            ),
            lambda value: value["env_info_after"][
                "filesystem_checkpoint_read"
            ].update(sha256="b" * 64),
            lambda value: value.update(native_call_count_after=6),
            lambda value: value.update(context_epoch_after=2),
            lambda value: value["action_submission"].update(
                raw_policy_output="shell_command {}"
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                record = json.loads(
                    json.dumps(self._checkpoint_read_record(module))
                )
                mutate(record)
                with self.assertRaises(AssertionError):
                    module.verify_wrapper_transition(
                        record, previous_native_step=6
                    )

    def test_checkpoint_chain_requires_immediate_matching_read(self) -> None:
        module = load_module()
        write = {
            "size_bytes": 128,
            "sha256": "a" * 64,
        }
        pending, chain = module.advance_checkpoint_read_chain(
            None,
            {
                "checkpoint_write_receipt": write,
                "checkpoint_read_receipt": None,
                "context_epoch_after": 1,
            },
            parent_index=4,
            task_round=6,
        )
        self.assertIsNone(chain)
        self.assertIsNotNone(pending)
        pending, chain = module.advance_checkpoint_read_chain(
            pending,
            {
                "checkpoint_write_receipt": None,
                "checkpoint_read_receipt": {
                    "size_bytes": 128,
                    "sha256": "a" * 64,
                },
                "context_epoch_after": 1,
            },
            parent_index=4,
            task_round=7,
        )
        self.assertIsNone(pending)
        self.assertEqual(
            chain,
            {
                "parent_index": 4,
                "write_task_round": 6,
                "read_task_round": 7,
                "size_bytes": 128,
                "sha256": "a" * 64,
            },
        )

        pending, _ = module.advance_checkpoint_read_chain(
            None,
            {
                "checkpoint_write_receipt": write,
                "checkpoint_read_receipt": None,
                "context_epoch_after": 1,
            },
            parent_index=4,
            task_round=6,
        )
        pending, chain = module.advance_checkpoint_read_chain(
            pending,
            {
                "checkpoint_write_receipt": None,
                "checkpoint_read_receipt": None,
                "context_epoch_after": 1,
            },
            parent_index=4,
            task_round=7,
        )
        self.assertIsNone(pending)
        self.assertIsNone(chain)
        pending, chain = module.advance_checkpoint_read_chain(
            pending,
            {
                "checkpoint_write_receipt": None,
                "checkpoint_read_receipt": {
                    "size_bytes": 128,
                    "sha256": "a" * 64,
                },
                "context_epoch_after": 1,
            },
            parent_index=4,
            task_round=8,
        )
        self.assertIsNone(pending)
        self.assertIsNone(chain)

    def test_checkpoint_chain_rejects_digest_or_epoch_drift(self) -> None:
        module = load_module()
        for read_digest, read_epoch in (("b" * 64, 1), ("a" * 64, 2)):
            with self.subTest(read_digest=read_digest, read_epoch=read_epoch):
                pending = {
                    "receipt": {"size_bytes": 128, "sha256": "a" * 64},
                    "task_round": 6,
                    "context_epoch_after": 1,
                }
                pending, chain = module.advance_checkpoint_read_chain(
                    pending,
                    {
                        "checkpoint_write_receipt": None,
                        "checkpoint_read_receipt": {
                            "size_bytes": 128,
                            "sha256": read_digest,
                        },
                        "context_epoch_after": read_epoch,
                    },
                    parent_index=4,
                    task_round=7,
                )
                self.assertIsNone(pending)
                self.assertIsNone(chain)

    def test_rejects_native_step_gap(self) -> None:
        module = load_module()
        record = {
            "wrapper_evidence": {
                "event": module.NATIVE_EVENT,
                "workspace_continuity_id": 9,
            },
            "env_info_before": {"step": 7},
            "env_info_after": {"step": 8, "action_kind": "shell_command"},
            "context_transition": {
                "schema": "agentmemory_task_neutral_context_transition_v1",
                "operation": "append_observation",
                "messages": [],
            },
            "action_submission": {"submitted_action": "shell_command"},
            "immediate_reward": 0.0,
            "trajectory_terminal": False,
            "done": False,
            "outcome": "continue",
        }
        with self.assertRaises(AssertionError):
            module.verify_wrapper_transition(record, previous_native_step=6)

    def test_accepts_successful_native_submission_row(self) -> None:
        module = load_module()
        record = {
            "wrapper_evidence": {
                "event": module.NATIVE_EVENT,
                "workspace_continuity_id": 9,
            },
            "env_info_before": {"step": 29},
            "env_info_after": {
                "step": 30,
                "action_kind": "final",
                "episode_success": True,
                "terminal": True,
            },
            "context_transition": {
                "schema": "agentmemory_task_neutral_context_transition_v1",
                "operation": "append_observation",
                "messages": [],
            },
            "action_submission": {"submitted_action": "final"},
            "immediate_reward": 1.0,
            "trajectory_terminal": True,
            "done": True,
            "outcome": "success",
        }
        result = module.verify_wrapper_transition(record, previous_native_step=29)
        self.assertEqual(result["action_kind"], "final")

    def test_accepts_one_native_invalid_penalty_and_rejects_double_penalty(self) -> None:
        module = load_module()
        record = {
            "wrapper_evidence": {
                "event": module.NATIVE_EVENT,
                "workspace_continuity_id": 9,
                "actor_credit": {
                    "schema": "task_neutral_actor_credit_v1",
                    "positive_eligible": False,
                    "basis": "parser_rejected",
                },
            },
            "env_info_before": {"step": 4},
            "env_info_after": {
                "step": 5,
                "action_kind": "parser_error",
                "episode_success": False,
                "terminal": True,
            },
            "context_transition": {
                "schema": "agentmemory_task_neutral_context_transition_v1",
                "operation": "append_observation",
                "messages": [],
            },
            "action_submission": {"raw_policy_output": "malformed"},
            "immediate_reward": -0.01,
            "trajectory_terminal": True,
            "done": True,
            "outcome": "terminal_failure",
        }
        result = module.verify_wrapper_transition(record, previous_native_step=4)
        self.assertEqual(result["action_kind"], "parser_error")
        record["immediate_reward"] = -0.02
        with self.assertRaises(AssertionError):
            module.verify_wrapper_transition(record, previous_native_step=4)

    def test_accepts_negative_horizon_failure_attached_to_last_native_row(self) -> None:
        module = load_module()
        record = {
            "task_round": 30,
            "wrapper_evidence": {
                "event": module.NATIVE_EVENT,
                "workspace_continuity_id": 9,
            },
            "env_info_before": {"step": 29},
            "env_info_after": {
                "step": 30,
                "action_kind": "shell_command",
                "episode_success": False,
                "terminal": False,
            },
            "context_transition": {
                "schema": "agentmemory_task_neutral_context_transition_v1",
                "operation": "append_observation",
                "messages": [],
            },
            "action_submission": {"submitted_action": "shell_command"},
            "immediate_reward": -0.01,
            "trajectory_terminal": True,
            "done": True,
            "outcome": "terminal_failure",
            "horizon_finalization": {
                "state": "Episode ended without a successful official submission.",
                "reward": -0.01,
                "done": True,
                "info": {
                    "env_info": {
                        "schema": "agentmemory_swesmith_native_episode_v1",
                        "step": 30,
                        "action_kind": "policy_turn_horizon",
                        "terminal": True,
                        "episode_success": False,
                    },
                    "action_submission": {"control_action": "horizon"},
                    "native_step_before": 30,
                    "native_step_after": 30,
                    "policy_step_after": 30,
                    "wrapper_evidence": {"event": "horizon_finalization"},
                },
            },
        }
        result = module.verify_wrapper_transition(record, previous_native_step=29)
        self.assertEqual(result["action_kind"], "shell_command")

    def test_rejects_horizon_reward_mismatch(self) -> None:
        module = load_module()
        record = {
            "task_round": 30,
            "wrapper_evidence": {
                "event": module.NATIVE_EVENT,
                "workspace_continuity_id": 9,
            },
            "env_info_before": {"step": 29},
            "env_info_after": {
                "step": 30,
                "action_kind": "shell_command",
                "episode_success": False,
                "terminal": False,
            },
            "context_transition": {
                "schema": "agentmemory_task_neutral_context_transition_v1",
                "operation": "append_observation",
                "messages": [],
            },
            "action_submission": {"submitted_action": "shell_command"},
            "immediate_reward": -0.01,
            "trajectory_terminal": True,
            "done": True,
            "outcome": "terminal_failure",
            "horizon_finalization": {
                "reward": 0.0,
                "done": True,
                "info": {
                    "env_info": {
                        "step": 30,
                        "terminal": True,
                        "episode_success": False,
                    },
                    "action_submission": {"control_action": "horizon"},
                    "native_step_before": 30,
                    "native_step_after": 30,
                    "policy_step_after": 30,
                    "wrapper_evidence": {"event": "horizon_finalization"},
                },
            },
        }
        with self.assertRaises(AssertionError):
            module.verify_wrapper_transition(record, previous_native_step=29)

    def test_allows_natural_short_episode_without_compaction(self) -> None:
        module = load_module()
        module.verify_event_coverage(
            module.Counter({module.NATIVE_EVENT: 10}),
            module.Counter({"shell_command": 8, "final": 2}),
            require_compaction=False,
            successful_compactions=0,
            successful_checkpoint_read_chains=0,
        )

    def test_preserves_strict_compaction_gate_when_requested(self) -> None:
        module = load_module()
        for successful_compactions, read_chains in ((0, 0), (1, 0)):
            with self.subTest(
                successful_compactions=successful_compactions,
                read_chains=read_chains,
            ):
                with self.assertRaises(AssertionError):
                    module.verify_event_coverage(
                        module.Counter(
                            {module.NATIVE_EVENT: 10, module.COMPACTION_EVENT: 1}
                        ),
                        module.Counter({"shell_command": 8, "final": 2}),
                        require_compaction=True,
                        successful_compactions=successful_compactions,
                        successful_checkpoint_read_chains=read_chains,
                    )
        module.verify_event_coverage(
            module.Counter(
                {module.NATIVE_EVENT: 10, module.COMPACTION_EVENT: 1}
            ),
            module.Counter({"shell_command": 8, "final": 2}),
            require_compaction=True,
            successful_compactions=1,
            successful_checkpoint_read_chains=1,
        )

    def test_accepts_representative_endpoint_probe_for_train64(self) -> None:
        module = load_module()
        indices = module.parse_endpoint_probe_indices(
            "0,1,13,14,30,31,47,48"
        )
        module.verify_endpoint_probe_indices(
            {"indices": indices},
            indices,
            set(range(64)),
        )

    def test_rejects_endpoint_probe_outside_training_curriculum(self) -> None:
        module = load_module()
        indices = module.parse_endpoint_probe_indices(
            "0,1,13,14,30,31,47,64"
        )
        with self.assertRaises(AssertionError):
            module.verify_endpoint_probe_indices(
                {"indices": indices},
                indices,
                set(range(64)),
            )

    def test_rejects_unpinned_endpoint_probe_order(self) -> None:
        module = load_module()
        expected = module.parse_endpoint_probe_indices(
            "0,1,13,14,30,31,47,48"
        )
        with self.assertRaises(AssertionError):
            module.verify_endpoint_probe_indices(
                {"indices": list(reversed(expected))},
                expected,
                set(range(64)),
            )

    def test_online_prefix_accepts_probe_from_full_routing_pool(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            routing_file = Path(raw_root) / "routing.jsonl"
            rows = [
                {
                    "data_idx": index,
                    "extra_info": {
                        "index": index,
                        "schedule_position": index,
                    },
                    "item_id": f"swesmith_{index}",
                }
                for index in range(128)
            ]
            routing_file.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            routing = module.load_routing_contract(
                routing_file,
                train_batch_size=64,
                first_global_step=1,
                global_step=1,
            )
            probe_indices = list(range(64, 72))
            module.verify_endpoint_probe_indices(
                {"indices": probe_indices},
                probe_indices,
                routing["training_data_indices"],
            )

    def test_accepts_formal_step_records_and_binds_all_indices(self) -> None:
        module = load_module()
        rows = [
            {
                "schema_version": module.STEP_SCHEMA,
                "parent_index": index,
                "item_id": f"swesmith_{index}",
            }
            for index in range(8)
        ]
        self.assertEqual(
            module.verify_row_evidence(
                {
                    "row_evidence": {
                        "schema": "agentmemory_formal_step_records_v1",
                        "task_name": "swesmith",
                        "rows": rows,
                    },
                    "formal_step_records": rows,
                },
                set(range(8)),
            ),
            {
                "schema": "agentmemory_formal_step_records_v1",
                "dataset_indices": list(range(8)),
                "row_count": 8,
            },
        )

    def test_accepts_generic_dataset_rows(self) -> None:
        module = load_module()
        self.assertEqual(
            module.verify_row_evidence(
                {
                    "row_evidence": {
                        "schema": "generic_task_dataset_rows_v1",
                        "task_name": "swesmith",
                        "index_field": "index",
                        "dataset_indices": list(range(8)),
                    }
                },
                set(range(8)),
            )["dataset_indices"],
            list(range(8)),
        )

    def test_rejects_incomplete_index_coverage(self) -> None:
        module = load_module()
        with self.assertRaises(AssertionError):
            module.verify_row_evidence(
                {
                    "row_evidence": {
                        "schema": "generic_task_dataset_rows_v1",
                        "task_name": "swesmith",
                        "index_field": "index",
                        "dataset_indices": list(range(7)),
                    }
                },
                set(range(8)),
            )

    def test_loads_nonrepeating_fullpool_routing_segment(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "routing.jsonl"
            rows = [
                {
                    "item_id": f"swesmith_{position}",
                    "data_idx": 1000 + position,
                    "extra_info": {
                        "index": 1000 + position,
                        "schedule_position": position,
                    },
                }
                for position in range(16)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            contract = module.load_routing_contract(
                path,
                train_batch_size=4,
                first_global_step=2,
                global_step=3,
            )

        self.assertEqual(contract["routing_row_count"], 16)
        self.assertEqual(contract["segment_schedule_positions"], list(range(4, 12)))
        self.assertEqual(contract["current_schedule_positions"], list(range(8, 12)))
        self.assertEqual(
            contract["current_item_ids"],
            {index: f"swesmith_{8 + index}" for index in range(4)},
        )
        self.assertEqual(
            contract["segment_data_idx_counts"],
            module.Counter({index: 1 for index in range(1004, 1012)}),
        )

    def test_rejects_noncontiguous_routing_positions(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "routing.jsonl"
            path.write_text(
                json.dumps({
                    "item_id": "swesmith_9",
                    "data_idx": 12,
                    "extra_info": {"index": 12, "schedule_position": 9},
                })
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(AssertionError):
                module.load_routing_contract(
                    path,
                    train_batch_size=1,
                    first_global_step=1,
                    global_step=1,
                )

    def test_binds_current_fullpool_item_ids_in_readback_rows(self) -> None:
        module = load_module()
        rows = [
            {
                "schema_version": module.STEP_SCHEMA,
                "parent_index": index,
                "item_id": f"swesmith_{64 + index}",
            }
            for index in range(8)
        ]
        result = module.verify_row_evidence(
            {
                "row_evidence": {
                    "schema": "agentmemory_formal_step_records_v1",
                    "task_name": "swesmith",
                    "rows": rows,
                },
                "formal_step_records": rows,
            },
            set(range(8)),
            {index: f"swesmith_{64 + index}" for index in range(8)},
        )
        self.assertEqual(result["dataset_indices"], list(range(8)))


class SwesmithPpoGateAuditSelectionTests(unittest.TestCase):
    @staticmethod
    def _audit(*, audit_id: str, index: int, slot: int, started_at: str) -> dict:
        resolved = index == 0
        return {
            "schema": "agentmemory_swesmith_private_episode_audit_v1",
            "audit_id": audit_id,
            "data_idx": index,
            "slot_id": slot,
            "started_at": started_at,
            "close_reason": "client_close",
            "done": True,
            "reward": 1.0 if resolved else 0.0,
            "grade": {
                "resolution_status": "RESOLVED_YES" if resolved else "RESOLVED_NO"
            },
            "step_count": 3,
            "sample_excluded": False,
            "sample_exclusion_reason": None,
            "evidence": [
                {
                    "event": "policy_step",
                    "termination_reason": (
                        "submission_sentinel" if resolved else "grader_unresolved"
                    ),
                    "action": {"kind": "shell_command"},
                    "actor_credit": {
                        "schema": "task_neutral_actor_credit_v1",
                        "positive_eligible": True,
                        "basis": "terminal_submission",
                    },
                    "observation_after": "graded",
                }
            ],
        }

    def test_excludes_stale_preflight_audits_from_reused_endpoint(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            current_probe_ids = []
            for index in range(8):
                stale_id = f"{index + 1:032x}"
                probe_id = f"{index + 101:032x}"
                trainer_id = f"{index + 201:032x}"
                current_probe_ids.append(probe_id)
                fixtures = (
                    self._audit(
                        audit_id=stale_id,
                        index=index,
                        slot=index,
                        started_at="2026-08-09T00:00:00Z",
                    ),
                    self._audit(
                        audit_id=probe_id,
                        index=index,
                        slot=index,
                        started_at="2026-08-09T00:30:00Z",
                    ),
                    self._audit(
                        audit_id=trainer_id,
                        index=index,
                        slot=index,
                        started_at="2026-08-09T01:00:01Z",
                    ),
                )
                for payload in fixtures:
                    (root / f"episode-{payload['audit_id']}.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )

            result = module.verify_audits(
                root,
                {"audit_ids": current_probe_ids},
                set(range(8)),
                module.parse_time("2026-08-09T01:00:00Z", "test.started_at"),
            )

        self.assertEqual(result["audit_count"], 8)
        self.assertEqual(result["dataset_indices"], list(range(8)))
        self.assertEqual(result["stale_audit_count"], 8)
        self.assertEqual(result["selection"], "run-start-time-minus-current-probe")

    def test_rejects_naive_run_timestamp(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(AssertionError, "must include a timezone"):
            module.parse_time("2026-08-09T01:00:00", "test.started_at")

    def test_counts_only_updates_in_a_resume_segment(self) -> None:
        module = load_module()
        self.assertEqual(module.optimizer_update_count(11, 100), 90)
        self.assertEqual(module.optimizer_update_count(1, 10), 10)
        with self.assertRaises(AssertionError):
            module.optimizer_update_count(11, 10)

    def test_accepts_return_signal_from_an_earlier_resume_update(self) -> None:
        module = load_module()
        endpoint_payload = {
            "global_step": 20,
            "stage": "post_adv",
            "rows": [{"ppo_valid_sample": True, "return_nonzero": 0}],
        }
        with tempfile.TemporaryDirectory() as raw_root:
            run_dir = Path(raw_root)
            diagnostics = run_dir / "diagnostics"
            diagnostics.mkdir()
            (diagnostics / "ppo_batch_step14_post_adv.json").write_text(
                json.dumps({
                    "global_step": 14,
                    "stage": "post_adv",
                    "rows": [
                        {"ppo_valid_sample": True, "return_nonzero": 1},
                        {"ppo_valid_sample": False, "return_nonzero": 1},
                    ],
                }),
                encoding="utf-8",
            )
            result = module.verify_segment_return_signal(
                run_dir,
                first_global_step=14,
                global_step=20,
                endpoint_payload=endpoint_payload,
                endpoint_nonzero_return_rows=0,
            )

        self.assertEqual(result["source_global_step"], 14)
        self.assertEqual(result["nonzero_return_rows"], 1)
        self.assertFalse(result["endpoint_has_return_signal"])

    def test_rejects_resume_segment_without_any_return_signal(self) -> None:
        module = load_module()
        endpoint_payload = {
            "global_step": 3,
            "stage": "post_adv",
            "rows": [{"ppo_valid_sample": True, "return_nonzero": 0}],
        }
        with tempfile.TemporaryDirectory() as raw_root:
            run_dir = Path(raw_root)
            diagnostics = run_dir / "diagnostics"
            diagnostics.mkdir()
            for step in (1, 2):
                (diagnostics / f"ppo_batch_step{step}_post_adv.json").write_text(
                    json.dumps({
                        "global_step": step,
                        "stage": "post_adv",
                        "rows": [
                            {"ppo_valid_sample": True, "return_nonzero": 0}
                        ],
                    }),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(
                AssertionError,
                "declared optimizer-update segment has no nonzero PPO return rows",
            ):
                module.verify_segment_return_signal(
                    run_dir,
                    first_global_step=1,
                    global_step=3,
                    endpoint_payload=endpoint_payload,
                    endpoint_nonzero_return_rows=0,
                )

    def test_accepts_return_signal_from_verified_parent_run(self) -> None:
        module = load_module()
        endpoint_payload = {
            "global_step": 21,
            "stage": "post_adv",
            "rows": [{"ppo_valid_sample": True, "return_nonzero": 0}],
        }
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run_dir = root / "resume"
            parent_run_dir = root / "parent"
            (run_dir / "diagnostics").mkdir(parents=True)
            (parent_run_dir / "diagnostics").mkdir(parents=True)
            (parent_run_dir / "diagnostics" / "ppo_batch_step1_post_adv.json").write_text(
                json.dumps({
                    "global_step": 1,
                    "stage": "post_adv",
                    "rows": [
                        {"ppo_valid_sample": True, "return_nonzero": 1},
                        {"ppo_valid_sample": True, "return_nonzero": 0},
                    ],
                }),
                encoding="utf-8",
            )
            result = module.verify_segment_return_signal(
                run_dir,
                first_global_step=21,
                global_step=21,
                endpoint_payload=endpoint_payload,
                endpoint_nonzero_return_rows=0,
                parent_run_dir=parent_run_dir,
            )

        self.assertEqual(result["source"], "verified_parent_run")
        self.assertEqual(result["source_global_step"], 1)
        self.assertEqual(result["nonzero_return_rows"], 1)
        self.assertFalse(result["endpoint_has_return_signal"])

    def test_endpoint_probe_slots_are_validated_without_fixing_service_offsets(self) -> None:
        module = load_module()
        self.assertEqual(
            module.parse_endpoint_probe_slots({"slot_ids": list(range(8))}),
            list(range(8)),
        )

    def test_accepts_repeated_task_indices_for_batch64(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for parent_index in range(64):
                task_index = parent_index % 8
                audit_id = f"{parent_index + 1:032x}"
                payload = self._audit(
                    audit_id=audit_id,
                    index=task_index,
                    slot=parent_index + 9,
                    started_at="2026-08-09T01:00:01Z",
                )
                (root / f"episode-{audit_id}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            result = module.verify_audits(
                root,
                {"audit_ids": []},
                set(range(8)),
                module.parse_time("2026-08-09T01:00:00Z", "test.started_at"),
                expected_audit_count=64,
                expected_data_idx_counts=module.Counter({index: 8 for index in range(8)}),
                expected_slot_cardinality=64,
            )

        self.assertEqual(result["audit_count"], 64)
        self.assertEqual(result["data_idx_counts"], {str(index): 8 for index in range(8)})

    def test_accepts_fresh_service_slot_per_episode_across_updates(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for update in range(2):
                for index in range(8):
                    audit_number = update * 8 + index
                    audit_id = f"{audit_number + 1:032x}"
                    payload = self._audit(
                        audit_id=audit_id,
                        index=index,
                        slot=100 + audit_number,
                        started_at=f"2026-08-09T01:00:{audit_number + 1:02d}Z",
                    )
                    (root / f"episode-{audit_id}.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )

            result = module.verify_audits(
                root,
                {"audit_ids": []},
                set(range(8)),
                module.parse_time("2026-08-09T01:00:00Z", "test.started_at"),
                expected_audit_count=16,
                expected_data_idx_counts=module.Counter(
                    {index: 2 for index in range(8)}
                ),
                expected_slot_cardinality=16,
            )

        self.assertEqual(result["audit_count"], 16)
        self.assertEqual(len(result["slot_counts"]), 16)
        self.assertEqual(set(result["slot_counts"].values()), {1})

    def test_online_prefix_accepts_only_declared_future_indices(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for offset, index in enumerate((10, 11, 12)):
                payload = self._audit(
                    audit_id=f"{offset + 1:032x}",
                    index=index,
                    slot=offset + 100,
                    started_at=f"2026-08-09T01:00:0{offset + 1}Z",
                )
                (root / f"episode-{payload['audit_id']}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            result = module.verify_audits(
                root,
                {"audit_ids": []},
                {10, 11},
                module.parse_time("2026-08-09T01:00:00Z", "test.started_at"),
                expected_audit_count=2,
                expected_data_idx_counts=module.Counter({10: 1, 11: 1}),
                expected_slot_cardinality=2,
                allowed_future_indices={12, 13},
            )
            self.assertEqual(result["future_distinct_audit_count"], 1)

    def test_online_prefix_rejects_undeclared_index(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            payload = self._audit(
                audit_id="1" * 32,
                index=99,
                slot=100,
                started_at="2026-08-09T01:00:01Z",
            )
            (root / "episode-unknown.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(AssertionError, "unexpected in-run"):
                module.verify_audits(
                    root,
                    {"audit_ids": []},
                    {10},
                    module.parse_time(
                        "2026-08-09T01:00:00Z", "test.started_at"
                    ),
                    allowed_future_indices={11, 12},
                )

    def test_accepts_ungraded_authoritative_executor_rejection(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            payload = self._audit(
                audit_id="1" * 32,
                index=3,
                slot=17,
                started_at="2026-08-09T01:00:01Z",
            )
            payload.update({
                "reward": -0.01,
                "grade": None,
                "evidence": [{
                    "event": "policy_step",
                    "termination_reason": "executor_rejected",
                    "action": {"kind": "shell_command"},
                    "actor_credit": {
                        "schema": "task_neutral_actor_credit_v1",
                        "positive_eligible": False,
                        "basis": "executor_rejected",
                    },
                    "observation_after": (
                        "shell_command failed: workspace contains an absolute symlink"
                    ),
                }],
            })
            (root / "episode-rejected.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            result = module.verify_audits(
                root,
                {"audit_ids": []},
                {3},
                module.parse_time("2026-08-09T01:00:00Z", "test.started_at"),
            )

        self.assertEqual(result["audit_count"], 1)
        self.assertEqual(result["graded_audit_count"], 0)
        self.assertEqual(
            result["ungraded_terminal_rejections"],
            [{
                "audit_id": "1" * 32,
                "data_idx": 3,
                "slot_id": 17,
                "step_count": 3,
                "actor_credit_basis": "executor_rejected",
                "action_kind": "shell_command",
                "termination_reason": "executor_rejected",
            }],
        )

    def test_rejects_ungraded_non_authoritative_client_close(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            payload = self._audit(
                audit_id="2" * 32,
                index=3,
                slot=17,
                started_at="2026-08-09T01:00:01Z",
            )
            payload.update({
                "reward": 0.0,
                "grade": None,
                "evidence": [{
                    "event": "policy_step",
                    "termination_reason": "",
                    "action": {"kind": "shell_command"},
                    "actor_credit": {
                        "schema": "task_neutral_actor_credit_v1",
                        "positive_eligible": True,
                        "basis": "shell_executed",
                    },
                    "observation_after": "shell_command exit_code=0",
                }],
            })
            (root / "episode-incomplete.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaises(AssertionError):
                module.verify_audits(
                    root,
                    {"audit_ids": []},
                    {3},
                    module.parse_time(
                        "2026-08-09T01:00:00Z", "test.started_at"
                    ),
                )

    def test_rejects_reused_slots_when_service_contract_is_fresh_per_episode(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for update in range(2):
                for slot in range(8):
                    audit_id = f"{update * 8 + slot + 1:032x}"
                    payload = self._audit(
                        audit_id=audit_id,
                        index=slot,
                        slot=slot,
                        started_at=f"2026-08-09T01:00:{update * 8 + slot + 1:02d}Z",
                    )
                    (root / f"episode-{audit_id}.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )

            with self.assertRaises(AssertionError):
                module.verify_audits(
                    root,
                    {"audit_ids": []},
                    set(range(8)),
                    module.parse_time(
                        "2026-08-09T01:00:00Z", "test.started_at"
                    ),
                    expected_audit_count=16,
                    expected_data_idx_counts=module.Counter(
                        {index: 2 for index in range(8)}
                    ),
                    expected_slot_cardinality=16,
                )

    def test_accepts_reused_slots_across_optimizer_updates(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for update in range(2):
                for slot in range(8):
                    audit_id = f"{update * 8 + slot + 1:032x}"
                    payload = self._audit(
                        audit_id=audit_id,
                        index=slot,
                        slot=slot,
                        started_at=f"2026-08-09T01:00:{update + 1:02d}Z",
                    )
                    (root / f"episode-{audit_id}.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )

            result = module.verify_audits(
                root,
                {"audit_ids": []},
                set(range(8)),
                module.parse_time("2026-08-09T01:00:00Z", "test.started_at"),
                expected_audit_count=16,
                expected_data_idx_counts=module.Counter(
                    {index: 2 for index in range(8)}
                ),
                expected_slot_counts=module.Counter(
                    {slot: 2 for slot in range(8)}
                ),
            )

        self.assertEqual(result["audit_count"], 16)
        self.assertEqual(
            result["slot_counts"], {str(slot): 2 for slot in range(8)}
        )


if __name__ == "__main__":
    unittest.main()
