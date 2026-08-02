from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT_PATH = (
    _REPO_ROOT / "verl/utils/agentgym/rollout_logprob_reuse.py"
)
_ROLLOUT_PATH = (
    _REPO_ROOT
    / "verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py"
)
_TRAINER_PATH = _REPO_ROOT / "verl/agent_trainer/ppo/ray_trainer.py"


def _load_contract_module():
    spec = importlib.util.spec_from_file_location(
        "rollout_logprob_reuse_under_test", _CONTRACT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeLogprob:
    def __init__(self, value):
        self.logprob = value


class EqualityTrap:
    def __eq__(self, other):
        raise AssertionError("sampling contract must not probe arbitrary equality")


class RolloutLogprobContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_contract_module()

    def raw_sampling_params(self, **overrides):
        values = {
            "n": 1,
            "logprobs": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "min_tokens": 0,
            "best_of": None,
            "ignore_eos": False,
            "logprobs_mode": "raw_logprobs",
            "logits_processors": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_mode_is_explicit_and_off_by_default(self):
        self.assertEqual(self.module.resolve_rollout_logprob_mode({}), "off")
        self.assertEqual(
            self.module.resolve_rollout_logprob_mode(
                {self.module.ROLLOUT_LOGPROB_MODE_ENV: " Compare "}
            ),
            "compare",
        )
        self.assertEqual(
            self.module.resolve_rollout_logprob_mode(
                {self.module.ROLLOUT_LOGPROB_MODE_ENV: "bypass"}
            ),
            "bypass",
        )
        with self.assertRaisesRegex(ValueError, "bypass, compare, off"):
            self.module.resolve_rollout_logprob_mode(
                {self.module.ROLLOUT_LOGPROB_MODE_ENV: "auto"}
            )

    def test_compare_requires_raw_single_sample_distribution(self):
        readback = self.module.validate_rollout_logprob_sampling_contract(
            self.raw_sampling_params(), "compare"
        )
        self.assertEqual(readback["logprobs"], 1)
        self.assertEqual(readback["top_k"], -1)
        self.assertEqual(
            self.module.validate_rollout_logprob_sampling_contract(
                self.raw_sampling_params(top_k=0), "compare"
            )["top_k"],
            0,
        )
        for field, value in (
            ("temperature", 0.8),
            ("top_p", 0.95),
            ("top_k", 50),
            ("min_p", 0.05),
            ("presence_penalty", 0.1),
            ("frequency_penalty", 0.1),
            ("repetition_penalty", 1.1),
            ("min_tokens", 1),
            ("n", 2),
            ("logprobs", 0),
        ):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.module.validate_rollout_logprob_sampling_contract(
                    self.raw_sampling_params(**{field: value}), "compare"
                )
        with self.assertRaisesRegex(RuntimeError, "logits_processors"):
            self.module.validate_rollout_logprob_sampling_contract(
                self.raw_sampling_params(logits_processors=[object()]), "bypass"
            )
        with self.assertRaisesRegex(RuntimeError, "truncate_prompt_tokens"):
            self.module.validate_rollout_logprob_sampling_contract(
                self.raw_sampling_params(truncate_prompt_tokens=128), "compare"
            )
        with self.assertRaisesRegex(RuntimeError, "ignore_eos=false"):
            self.module.validate_rollout_logprob_sampling_contract(
                self.raw_sampling_params(ignore_eos=True), "compare"
            )
        with self.assertRaisesRegex(RuntimeError, "raw_logprobs"):
            self.module.validate_official_vllm_engine_logprob_contract(
                SimpleNamespace(
                    model_config=SimpleNamespace(
                        logprobs_mode="processed_logprobs"
                    )
                ),
                "compare",
            )
        with self.assertRaisesRegex(RuntimeError, "could not read"):
            self.module.validate_official_vllm_engine_logprob_contract(
                SimpleNamespace(),
                "compare",
            )

    def test_engine_contract_reads_back_raw_mode_from_all_visible_configs(self):
        direct = SimpleNamespace(logprobs_mode="raw_logprobs")
        nested = SimpleNamespace(logprobs_mode="raw_logprobs")
        readback = self.module.validate_official_vllm_engine_logprob_contract(
            SimpleNamespace(
                model_config=direct,
                llm_engine=SimpleNamespace(model_config=nested),
            ),
            "bypass",
        )
        self.assertEqual(readback["logprobs_mode"], "raw_logprobs")
        self.assertEqual(
            readback["readback_paths"],
            ["llm_engine.model_config", "model_config"],
        )

    def test_off_does_not_constrain_sampling(self):
        self.assertEqual(
            self.module.validate_rollout_logprob_sampling_contract(
                self.raw_sampling_params(logprobs=0, temperature=0.2), "off"
            ),
            {"mode": "off"},
        )

    def test_training_scope_leaves_non_agentmemory_off_path_unchanged(self):
        self.assertFalse(
            self.module.validate_rollout_logprob_training_scope(
                task_name="gsm8k", adv_estimator="grpo", mode="off"
            )
        )
        self.assertTrue(
            self.module.validate_rollout_logprob_training_scope(
                task_name=" AgentMemory ", adv_estimator="gae", mode="compare"
            )
        )
        with self.assertRaisesRegex(ValueError, "scoped to AgentMemoryGym"):
            self.module.validate_rollout_logprob_training_scope(
                task_name="gsm8k", adv_estimator="gae", mode="compare"
            )
        with self.assertRaisesRegex(ValueError, "requires PPO/GAE"):
            self.module.validate_rollout_logprob_training_scope(
                task_name="agentmemory", adv_estimator="grpo", mode="bypass"
            )

    def test_sampling_empty_fields_do_not_use_arbitrary_equality(self):
        self.module.validate_rollout_logprob_sampling_contract(
            self.raw_sampling_params(logits_processors=[]), "compare"
        )
        with self.assertRaisesRegex(RuntimeError, "logits_processors"):
            self.module.validate_rollout_logprob_sampling_contract(
                self.raw_sampling_params(logits_processors=EqualityTrap()),
                "compare",
            )

    def test_extracts_exact_sampled_token_from_each_vllm_row(self):
        actual = self.module.extract_sampled_token_logprobs(
            [11, 22, 33],
            [
                {11: FakeLogprob(-0.1), 91: FakeLogprob(-2.0)},
                {22: {"logprob": -0.2}},
                {33: -0.3},
            ],
        )
        self.assertEqual(actual, [-0.1, -0.2, -0.3])

    def test_extraction_fails_on_missing_or_ambiguous_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "length mismatch"):
            self.module.extract_sampled_token_logprobs([11, 22], [{11: -0.1}])
        with self.assertRaisesRegex(RuntimeError, "missing the sampled token"):
            self.module.extract_sampled_token_logprobs([11], [{22: -0.1}])
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            self.module.extract_sampled_token_logprobs(
                [11], [{11: FakeLogprob(math.nan)}]
            )
        with self.assertRaisesRegex(RuntimeError, "raw list\[int\]"):
            self.module.extract_sampled_token_logprobs((11,), [{11: -0.1}])

    def test_bound_rows_must_align_and_be_finite(self):
        evidence = [
            self.module.build_sampled_token_logprob_evidence(
                prompt_token_ids=[101],
                response_token_ids=[1, 2],
                logprob_rows=[{1: -0.1}, {2: -0.2}],
            ),
            self.module.build_sampled_token_logprob_evidence(
                prompt_token_ids=[202],
                response_token_ids=[3],
                logprob_rows=[{3: -0.3}],
            ),
        ]
        self.assertEqual(
            self.module.validate_rollout_logprob_rows(
                [[101], [202]], [[1, 2], [3]], evidence
            ),
            [[-0.1, -0.2], [-0.3]],
        )
        with self.assertRaisesRegex(RuntimeError, "token-count mismatch"):
            broken = dict(evidence[0])
            broken["log_probs"] = [-0.1]
            self.module.validate_rollout_logprob_rows(
                [[101]], [[1, 2]], [broken]
            )
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            broken = dict(evidence[0])
            broken["log_probs"] = [math.inf, -0.2]
            self.module.validate_rollout_logprob_rows(
                [[101]], [[1, 2]], [broken]
            )

    def test_fake_official_request_output_binds_prompt_response_and_logprobs(self):
        candidate = SimpleNamespace(
            token_ids=[11, 22],
            logprobs=[{11: FakeLogprob(-0.1)}, {22: FakeLogprob(-0.2)}],
        )
        request_output = SimpleNamespace(
            prompt_token_ids=[101, 102],
            outputs=[candidate],
        )
        evidence = self.module.build_official_vllm_sampled_logprob_evidence(
            request_output=request_output,
            expected_prompt_token_ids=[101, 102],
            normalized_response_token_ids=[11, 22],
        )
        self.assertEqual(evidence["prompt_token_ids"], [101, 102])
        self.assertEqual(evidence["response_token_ids"], [11, 22])
        self.assertEqual(evidence["log_probs"], [-0.1, -0.2])

        with self.assertRaisesRegex(RuntimeError, "different prompt tokens"):
            self.module.build_official_vllm_sampled_logprob_evidence(
                request_output=request_output,
                expected_prompt_token_ids=[999],
                normalized_response_token_ids=[11, 22],
            )
        with self.assertRaisesRegex(RuntimeError, "response tokens changed"):
            self.module.build_official_vllm_sampled_logprob_evidence(
                request_output=request_output,
                expected_prompt_token_ids=[101, 102],
                normalized_response_token_ids=[22, 11],
            )
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            self.module.build_official_vllm_sampled_logprob_evidence(
                request_output=SimpleNamespace(
                    prompt_token_ids=[101, 102],
                    outputs=[candidate, candidate],
                ),
                expected_prompt_token_ids=[101, 102],
                normalized_response_token_ids=[11, 22],
            )

    def test_bound_rows_fail_closed_on_reorder_or_filter_misalignment(self):
        evidence = [
            self.module.build_sampled_token_logprob_evidence(
                prompt_token_ids=[101],
                response_token_ids=[1, 2],
                logprob_rows=[{1: -0.1}, {2: -0.2}],
            ),
            self.module.build_sampled_token_logprob_evidence(
                prompt_token_ids=[202],
                response_token_ids=[3, 4],
                logprob_rows=[{3: -0.3}, {4: -0.4}],
            ),
        ]
        with self.assertRaisesRegex(RuntimeError, "prompt binding mismatch"):
            self.module.validate_rollout_logprob_rows(
                [[202], [101]], [[3, 4], [1, 2]], evidence
            )
        with self.assertRaisesRegex(RuntimeError, "response binding mismatch"):
            self.module.validate_rollout_logprob_rows(
                [[101]], [[3, 4]], [evidence[0]]
            )


class RolloutLogprobWiringTests(unittest.TestCase):
    def test_rollout_collects_only_in_explicit_mode_and_packs_zero_padding(self):
        source = _ROLLOUT_PATH.read_text(encoding="utf-8")
        self.assertIn("resolve_rollout_logprob_mode()", source)
        self.assertIn("build_official_vllm_sampled_logprob_evidence(", source)
        self.assertIn("expected_prompt_token_ids=generation_prompt_idxs", source)
        self.assertIn("isinstance(sampled_action_logprobs, Mapping)", source)
        self.assertIn("validate_rollout_logprob_rows(", source)
        self.assertIn("validate_rollout_logprob_sampling_contract(", source)
        self.assertIn("ROLLOUT_LOGPROB_BATCH_KEY", source)
        self.assertIn("padding_value=0.0", source)
        self.assertIn("Rollout logprob padding must be exactly zero", source)

    def test_trainer_compare_keeps_recomputed_values_and_bypass_skips_forward(self):
        source = _TRAINER_PATH.read_text(encoding="utf-8")
        self.assertIn("self.rollout_logprob_mode", source)
        self.assertIn("self.agentmemory_task = validate_rollout_logprob_training_scope", source)
        self.assertIn("if self.agentmemory_task:", source)
        self.assertIn("_compare_rollout_and_recomputed_logprobs(", source)
        self.assertIn("ROLLOUT_LOGPROB_MODE_COMPARE", source)
        self.assertIn("ROLLOUT_LOGPROB_MODE_BYPASS", source)
        self.assertIn("batch.batch['old_log_probs'] = rollout_log_probs", source)
        self.assertIn("self.actor_rollout_wg.compute_log_prob(batch)", source)
        self.assertIn("batch.batch.pop(ROLLOUT_LOGPROB_BATCH_KEY)", source)


if __name__ == "__main__":
    unittest.main()
