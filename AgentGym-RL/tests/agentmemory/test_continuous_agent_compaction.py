from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "verl/utils/agentgym/continuous_agent_v1.py"
SPEC = importlib.util.spec_from_file_location("continuous_agent_v1_for_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

COMPACTION_ROW = MODULE.COMPACTION_ROW
CONTINUOUS_AGENT_CONTEXT_POLICY_V1 = MODULE.CONTINUOUS_AGENT_CONTEXT_POLICY_V1
ENVIRONMENT_ACTION_ROW = MODULE.ENVIRONMENT_ACTION_ROW
POLICY_COMPACTION_REQUEST = MODULE.POLICY_COMPACTION_REQUEST
POLICY_CONTINUATION_MARKER = MODULE.POLICY_CONTINUATION_MARKER
ContinuousAgentV1Error = MODULE.ContinuousAgentV1Error
build_continuous_agent_step_v1 = MODULE.build_continuous_agent_step_v1
build_compaction_evidence_v1 = MODULE.build_compaction_evidence_v1
build_horizon_evidence_v1 = MODULE.build_horizon_evidence_v1
build_observation_evidence_v1 = MODULE.build_observation_evidence_v1
build_policy_generation_loss_masks = MODULE.build_policy_generation_loss_masks
text_sha256 = MODULE.text_sha256
token_digest = MODULE.token_digest
validate_neutral_compaction_text = MODULE.validate_neutral_compaction_text
validate_continuous_agent_trajectory_v1 = (
    MODULE.validate_continuous_agent_trajectory_v1
)
continuous_prompt_capacity = MODULE.continuous_prompt_capacity
should_request_policy_compaction = MODULE.should_request_policy_compaction


def generation_record(token_ids: list[int]) -> dict:
    return {
        "token_ids": token_ids,
        "response_token_count": len(token_ids),
        "max_response_tokens": 64,
        "finish_reason": "stop",
        "finish_reason_source": "official_vllm:backend",
        "stop_reason": None,
        "truncated": False,
        "configured_eos_token_ids": [2],
        "primary_eos_token_id": 2,
        "tokenizer_pad_token_id": 0,
        "backend_source": "official_vllm",
        "backend_token_ids_are_exact": True,
        "token_ids_are_exact": True,
    }


def build_row(row_kind: str) -> dict:
    prompt = [10, 11, 12]
    response = [20, 2]
    is_compaction = row_kind == COMPACTION_ROW
    evidence = None
    if is_compaction:
        evidence = build_compaction_evidence_v1(
            pre_request_action_prompt_token_ids=[10, 11],
            pre_compaction_prompt_token_ids=prompt,
            immutable_framing_token_ids=[1, 2, 3],
            summary_token_ids=response,
            post_compaction_prompt_token_ids=[30, 31],
            workspace_continuity_id="7",
        )
    observation = None
    environment_result = ""
    if not is_compaction:
        environment_result = "ok"
        observation = build_observation_evidence_v1(
            full_text=environment_result,
            full_token_ids=[30],
            policy_visible_text=environment_result,
            policy_visible_token_ids=[30],
            post_observation_prompt_token_ids=[40, 41],
            max_observation_tokens=8,
            truncated=False,
            head_token_count=1,
            tail_token_count=0,
            truncation_marker=None,
        )
    return build_continuous_agent_step_v1(
        row_kind=row_kind,
        task_name="swesmith",
        content="summary" if is_compaction else "shell_command({})",
        score=0.0,
        item_id="swesmith_4",
        data_idx=4,
        parent_index=0,
        parent_group_uid="parent:0",
        replica_index=0,
        trajectory_uid="parent:0:replica:0",
        exact_state_uid="0:turn1:statev1:" + token_digest(prompt),
        prompt_token_ids=prompt,
        response_token_ids=response,
        sampled_token_logprobs=[-0.1, -0.2],
        generation_record=generation_record(response),
        environment_id=7,
        environment_step_before=1,
        environment_step_after=2,
        native_environment_call_count_before=1,
        native_environment_call_count_after=1 if is_compaction else 2,
        context_epoch_before=0,
        context_epoch_after=1 if is_compaction else 0,
        done=False,
        environment_result=environment_result,
        compaction_evidence=evidence,
        observation_evidence=observation,
    )


class ContinuousAgentCompactionTests(unittest.TestCase):
    def test_control_text_is_neutral_and_does_not_inject_a_memory_path(self) -> None:
        validate_neutral_compaction_text()
        combined = f"{POLICY_COMPACTION_REQUEST}\n{POLICY_CONTINUATION_MARKER}".lower()
        self.assertNotIn("memory.md", combined)
        self.assertNotIn(".agent_memory", combined)
        self.assertNotIn("next step is", combined)

    def test_compaction_is_a_trainable_row_that_consumes_a_step(self) -> None:
        row = build_row(COMPACTION_ROW)
        self.assertEqual(row["row_kind"], COMPACTION_ROW)
        self.assertEqual(
            row["prompt_history_policy"], CONTINUOUS_AGENT_CONTEXT_POLICY_V1
        )
        self.assertFalse(row["environment_action_dispatched"])
        self.assertEqual(
            row["environment_step_after"], row["environment_step_before"] + 1
        )
        self.assertEqual(
            row["native_environment_call_count_before"],
            row["native_environment_call_count_after"],
        )
        self.assertEqual(
            len(row["response_token_ids"]), len(row["sampled_token_logprobs"])
        )

    def test_compaction_request_is_masked_but_policy_summary_is_trainable(self) -> None:
        loss_mask, prompt_mask, response_mask = build_policy_generation_loss_masks(
            [10, 11, 12],
            [20, 21],
        )
        self.assertEqual(prompt_mask, [0, 0, 0])
        self.assertEqual(response_mask, [1, 1])
        self.assertEqual(loss_mask, [0, 0, 0, 1, 1])

    def test_environment_action_advances_only_the_environment(self) -> None:
        row = build_row(ENVIRONMENT_ACTION_ROW)
        self.assertTrue(row["environment_action_dispatched"])
        self.assertEqual(
            row["environment_step_after"], row["environment_step_before"] + 1
        )
        self.assertEqual(
            row["native_environment_call_count_after"],
            row["native_environment_call_count_before"] + 1,
        )
        self.assertEqual(row["context_epoch_before"], row["context_epoch_after"])

    def test_compaction_has_no_separate_count_limit(self) -> None:
        row = build_row(COMPACTION_ROW)
        self.assertNotIn("max_compactions", row)
        self.assertNotIn("compaction_count_limit", row)

    def test_capacity_trigger_uses_existing_prompt_model_and_response_limits(self) -> None:
        common = {
            "max_prompt_tokens": 30720,
            "max_model_tokens": 32768,
            "max_response_tokens": 2048,
            "max_observation_tokens": 1024,
            "action_observation_envelope_tokens": 16,
        }
        self.assertFalse(
            should_request_policy_compaction(
                action_prompt_token_count=26000,
                compaction_prompt_token_count=26100,
                **common,
            )
        )
        self.assertTrue(
            should_request_policy_compaction(
                action_prompt_token_count=28600,
                compaction_prompt_token_count=28700,
                **common,
            )
        )

    def test_capacity_contract_recomputes_from_runtime_limits(self) -> None:
        self.assertEqual(
            continuous_prompt_capacity(
                max_prompt_tokens=30720,
                max_model_tokens=32768,
                max_response_tokens=2048,
            ),
            30720,
        )
        self.assertEqual(
            continuous_prompt_capacity(
                max_prompt_tokens=60000,
                max_model_tokens=65536,
                max_response_tokens=4096,
            ),
            60000,
        )
        self.assertEqual(
            continuous_prompt_capacity(
                max_prompt_tokens=65536,
                max_model_tokens=65536,
                max_response_tokens=4096,
            ),
            61440,
        )

    def test_capacity_trigger_fails_closed_if_compaction_was_requested_too_late(self) -> None:
        with self.assertRaisesRegex(ContinuousAgentV1Error, "prompt cap"):
            should_request_policy_compaction(
                action_prompt_token_count=30700,
                compaction_prompt_token_count=30750,
                max_prompt_tokens=30720,
                max_model_tokens=32768,
                max_response_tokens=2048,
                max_observation_tokens=1024,
                action_observation_envelope_tokens=16,
            )

    def test_compaction_rejects_harness_workspace_identity_drift(self) -> None:
        row = build_row(COMPACTION_ROW)
        row["compaction"]["workspace_continuity_id"] = "another-workspace"
        with self.assertRaisesRegex(ContinuousAgentV1Error, "continuity"):
            MODULE.validate_continuous_agent_step_v1(row)

    def test_sampled_logprob_count_is_bound_to_summary_tokens(self) -> None:
        row = build_row(COMPACTION_ROW)
        row["sampled_token_logprobs"] = [-0.1]
        with self.assertRaisesRegex(ContinuousAgentV1Error, "exactly 2"):
            MODULE.validate_continuous_agent_step_v1(row)

    def test_compaction_rejects_fabricated_immutable_framing_digest(self) -> None:
        row = build_row(COMPACTION_ROW)
        row["compaction"]["immutable_framing_digest"] = "a" * 64
        with self.assertRaisesRegex(ContinuousAgentV1Error, "framing digest"):
            MODULE.validate_continuous_agent_step_v1(row)

    def test_compaction_rejects_post_prompt_drift(self) -> None:
        row = build_row(COMPACTION_ROW)
        row["compaction"]["post_compaction_prompt_token_ids"] = [30, 32]
        with self.assertRaisesRegex(ContinuousAgentV1Error, "post-compaction prompt"):
            MODULE.validate_continuous_agent_step_v1(row)

    def test_truncated_observation_binds_full_and_visible_evidence(self) -> None:
        marker = (
            "\n[OBSERVATION TRUNCATED: original_tokens=6 omitted_tokens=3 "
            "sha256=abc]\n"
        )
        visible = "head" + marker + "tail"
        evidence = build_observation_evidence_v1(
            full_text="head-middle-tail",
            full_token_ids=[1, 2, 3, 4, 5, 6],
            policy_visible_text=visible,
            policy_visible_token_ids=[1, 7, 8, 6],
            post_observation_prompt_token_ids=[20, 21],
            max_observation_tokens=4,
            truncated=True,
            head_token_count=2,
            tail_token_count=1,
            truncation_marker=marker,
        )
        self.assertTrue(evidence["truncated"])
        self.assertEqual(evidence["policy_visible_token_count"], 4)
        self.assertNotEqual(
            evidence["full_text_sha256"],
            evidence["policy_visible_text_sha256"],
        )

    def test_horizon_can_credit_a_final_compaction_without_an_extra_policy_step(self) -> None:
        row = build_row(COMPACTION_ROW)
        row["done"] = True
        row["score"] = 1.0
        row["horizon_finalization"] = build_horizon_evidence_v1(
            environment_id=row["environment_id"],
            environment_step=row["environment_step_after"],
            native_environment_call_count=row[
                "native_environment_call_count_after"
            ],
            policy_step_reward=0.0,
            horizon_reward=1.0,
            environment_result="resolved",
        )
        MODULE.validate_continuous_agent_step_v1(row)

    def test_trajectory_validator_binds_prompt_and_counter_continuity(self) -> None:
        common = {
            "task_name": "swesmith",
            "item_id": "swesmith_4",
            "data_idx": 4,
            "parent_index": 0,
            "parent_group_uid": "parent:0",
            "replica_index": 0,
            "trajectory_uid": "parent:0:replica:0",
            "environment_id": 7,
        }

        def env_row(
            *,
            prompt: list[int],
            response: list[int],
            post_prompt: list[int],
            step: int,
            native_before: int,
            epoch: int,
            score: float = 0.0,
            done: bool = False,
        ) -> dict:
            result = f"observation-{step}"
            observation = build_observation_evidence_v1(
                full_text=result,
                full_token_ids=[100 + step],
                policy_visible_text=result,
                policy_visible_token_ids=[100 + step],
                post_observation_prompt_token_ids=post_prompt,
                max_observation_tokens=8,
                truncated=False,
                head_token_count=1,
                tail_token_count=0,
                truncation_marker=None,
            )
            return build_continuous_agent_step_v1(
                row_kind=ENVIRONMENT_ACTION_ROW,
                content="shell_command({})",
                score=score,
                exact_state_uid=(
                    f"0:turn{step + 1}:statev1:" + token_digest(prompt)
                ),
                prompt_token_ids=prompt,
                response_token_ids=response,
                sampled_token_logprobs=[-0.1] * len(response),
                generation_record=generation_record(response),
                environment_step_before=step,
                environment_step_after=step + 1,
                native_environment_call_count_before=native_before,
                native_environment_call_count_after=native_before + 1,
                context_epoch_before=epoch,
                context_epoch_after=epoch,
                done=done,
                environment_result=result,
                observation_evidence=observation,
                **common,
            )

        first = env_row(
            prompt=[10],
            response=[20],
            post_prompt=[30],
            step=0,
            native_before=0,
            epoch=0,
        )
        compact_prompt = [30, 31]
        compact_evidence = build_compaction_evidence_v1(
            pre_request_action_prompt_token_ids=[30],
            pre_compaction_prompt_token_ids=compact_prompt,
            immutable_framing_token_ids=[10],
            summary_token_ids=[21],
            post_compaction_prompt_token_ids=[40],
            workspace_continuity_id=7,
        )
        compact = build_continuous_agent_step_v1(
            row_kind=COMPACTION_ROW,
            content="summary",
            score=0.0,
            exact_state_uid="0:turn2:statev1:" + token_digest(compact_prompt),
            prompt_token_ids=compact_prompt,
            response_token_ids=[21],
            sampled_token_logprobs=[-0.2],
            generation_record=generation_record([21]),
            environment_step_before=1,
            environment_step_after=2,
            native_environment_call_count_before=1,
            native_environment_call_count_after=1,
            context_epoch_before=0,
            context_epoch_after=1,
            done=False,
            environment_result="",
            compaction_evidence=compact_evidence,
            **common,
        )
        final = env_row(
            prompt=[40],
            response=[22],
            post_prompt=[50],
            step=2,
            native_before=1,
            epoch=1,
            score=1.0,
            done=True,
        )
        validate_continuous_agent_trajectory_v1([first, compact, final])
        final["prompt_token_ids"] = [41]
        final["exact_state_uid"] = (
            "0:turn3:statev1:" + token_digest(final["prompt_token_ids"])
        )
        with self.assertRaisesRegex(
            ContinuousAgentV1Error, "prompt history drifted"
        ):
            validate_continuous_agent_trajectory_v1([first, compact, final])

    def test_compaction_cannot_fabricate_terminal_reward(self) -> None:
        row = build_row(COMPACTION_ROW)
        row["done"] = True
        row["score"] = 1.0
        with self.assertRaisesRegex(
            ContinuousAgentV1Error,
            "cannot receive immediate environment reward",
        ):
            MODULE.validate_continuous_agent_step_v1(row)

    def test_official_vllm_logprobs_select_the_sampled_token(self) -> None:
        class Logprob:
            def __init__(self, value: float) -> None:
                self.logprob = value

        selected = MODULE.extract_sampled_token_logprobs(
            [20, 2],
            [{20: Logprob(-0.1), 99: Logprob(-4.0)}, {2: Logprob(-0.2)}],
        )
        self.assertEqual(selected, [-0.1, -0.2])

    def test_official_vllm_logprobs_fail_if_the_sampled_token_is_absent(self) -> None:
        with self.assertRaisesRegex(ContinuousAgentV1Error, "absent"):
            MODULE.extract_sampled_token_logprobs([20], [{99: -0.1}])


if __name__ == "__main__":
    unittest.main()
