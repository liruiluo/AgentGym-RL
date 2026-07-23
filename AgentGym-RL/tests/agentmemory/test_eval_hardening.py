import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from verl import DataProto


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl/agent_trainer/main_generation.py"
)
_SPEC = importlib.util.spec_from_file_location("main_generation_for_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class EvalHardeningTest(unittest.TestCase):
    @staticmethod
    def _formal_record(*, episode_success, domain_id="formal_reasoning_math"):
        return json.dumps({
            "schema_version": "agentmemory_formal_step_v3",
            "domain_id": domain_id,
            "episode_success": episode_success,
        })

    def test_explicit_jsonl_dataset_and_safe_category_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "travel_eval.jsonl"
            dataset_path.write_text(
                '{"item_id": "agentmemory_0"}\n'
                '{"item_id": "agentmemory_1"}\n',
                encoding="utf-8",
            )
            (root / "city.json").write_text(
                json.dumps([{"item_id": "agentmemory_0"}]),
                encoding="utf-8",
            )
            (root / "region.jsonl").write_text(
                '{"item_id": "agentmemory_1"}\n',
                encoding="utf-8",
            )
            (root / "metadata.json").write_text(
                json.dumps({"description": "not a category record"}),
                encoding="utf-8",
            )
            (root / "search.index").write_bytes(b"not JSON")
            (root / "nested").mkdir()

            resolved = _MODULE._resolve_eval_dataset_path(
                {"path": str(root), "file": "travel_eval.jsonl"},
                {"task_name": "agentmemory"},
            )
            self.assertEqual(resolved, dataset_path.resolve())
            records = _MODULE._read_json_records(resolved)
            self.assertEqual([row["item_id"] for row in records], [
                "agentmemory_0",
                "agentmemory_1",
            ])
            category_files, category_map = _MODULE._load_category_map(
                str(root), resolved, "agentmemory"
            )
            self.assertEqual(category_files, ["city.json", "region.jsonl"])
            self.assertEqual(category_map, {
                "agentmemory_0": "city",
                "agentmemory_1": "region",
            })

    def test_dataset_legacy_fallback_and_policy_turn_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "agentmemory_test.json"
            legacy.write_text(json.dumps([{"item_id": "agentmemory_0"}]), encoding="utf-8")
            resolved = _MODULE._resolve_eval_dataset_path(
                {"path": str(root)},
                {"task_name": "agentmemory"},
            )
            self.assertEqual(resolved, legacy.resolve())

        self.assertEqual(
            _MODULE._resolve_max_policy_turns(
                {"max_policy_turns": 17, "max_rounds": 3}
            ),
            17,
        )
        self.assertEqual(
            _MODULE._resolve_max_policy_turns({"max_policy_turns": None, "max_rounds": "3"}),
            3,
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _MODULE._resolve_max_policy_turns({"max_policy_turns": 0})

    def test_prompt_key_and_data_indices_use_row_positions_by_default(self):
        dataset = _MODULE.pd.DataFrame.from_records(
            [
                {"id": 17, "question": "first"},
                {"id": "opaque-source-id", "question": "second"},
            ]
        )
        # A stale item_id setting must not make a valid MemoryArena ``id``
        # column unusable.
        self.assertEqual(
            _MODULE._resolve_eval_prompt_key(
                {"prompt_key": "item_id"}, dataset
            ),
            "id",
        )
        self.assertEqual(
            _MODULE._resolve_eval_data_indices(dataset, "id"),
            [0, 1],
        )

        dataset = _MODULE.pd.DataFrame.from_records(
            [
                {"source_id": 1, "question": "first traveler group"},
                {"source_id": 270, "question": "last traveler group"},
            ]
        )
        self.assertEqual(
            _MODULE._resolve_eval_data_indices(dataset, "source_id"),
            [0, 1],
        )

    def test_explicit_data_idx_is_strict_integer_and_in_range(self):
        dataset = _MODULE.pd.DataFrame.from_records(
            [
                {"item_id": "first", "data_idx": 1},
                {"item_id": "second", "data_idx": 0},
            ]
        )
        self.assertEqual(
            _MODULE._resolve_eval_data_indices(dataset, "item_id"),
            [1, 0],
        )

        for invalid in (True, "0", 0.0, -1, 1, None):
            with self.subTest(invalid=invalid):
                invalid_dataset = _MODULE.pd.DataFrame.from_records(
                    [{"item_id": "only", "data_idx": invalid}]
                )
                with self.assertRaisesRegex(ValueError, "data_idx"):
                    _MODULE._resolve_eval_data_indices(
                        invalid_dataset,
                        "item_id",
                    )

    def test_formal_eval_scrubs_train_only_rollout_flags(self):
        env = {
            "AGENTMEMORY_LATEST_OBS_SUFFIX_CREDIT": "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            removed = _MODULE._scrub_training_rollout_flags()
            self.assertEqual(sorted(removed), sorted(env))
            for name in env:
                self.assertNotIn(name, os.environ)

        with mock.patch.dict(
            os.environ,
            {
                **env,
                "AGENTMEMORY_ALLOW_TRAIN_ROLLOUT_FLAGS_IN_EVAL": "1",
            },
            clear=True,
        ):
            self.assertEqual(_MODULE._scrub_training_rollout_flags(), [])
            for name in env:
                self.assertIn(name, os.environ)

    def test_tiny_eval_batch_pads_to_dp_size(self):
        data = DataProto.from_dict(
            tensors={"row": torch.tensor([[7]])},
            non_tensors={"item_id": np.array(["only"], dtype=object)},
        )
        padded, dummy_count = _MODULE._pad_dataproto_for_dp(data, dp_size=8)
        self.assertEqual(len(padded), 8)
        self.assertEqual(dummy_count, 7)
        torch.testing.assert_close(padded.batch["row"], torch.full((8, 1), 7))

    def test_episode_aggregation_rejects_misaligned_rows(self):
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[1.0], [2.0]])},
            non_tensors={
                "rollout_parent_indices": np.array([0, 0], dtype=object),
                "rollout_done_flags": np.array([True, True], dtype=object),
            },
        )
        output.non_tensor_batch["rollout_parent_indices"] = np.array(
            [0], dtype=object
        )
        with self.assertRaisesRegex(ValueError, "misaligned"):
            _MODULE._aggregate_episode_scores(output, real_batch_size=1)

    def test_terminal_positive_partial_progress_is_not_pass(self):
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[0.99]])},
            non_tensors={
                "rollout_parent_indices": np.array([0], dtype=object),
                "rollout_done_flags": np.array([True], dtype=object),
                "agentmemory_step_record_json": np.array([
                    self._formal_record(episode_success=False)
                ], dtype=object),
            },
        )

        scores, passes, progress, info = _MODULE._aggregate_episode_scores(
            output, real_batch_size=1
        )

        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0], 0.99, places=6)
        self.assertEqual(passes, [False])
        self.assertEqual(progress, [False])
        self.assertEqual(info["pass_source"], "formal_episode_success")

    def test_formal_episode_rows_require_authoritative_success(self):
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[1.0]])},
            non_tensors={
                "rollout_done_flags": np.array([True], dtype=object),
                "agentmemory_step_record_json": np.array(
                    [json.dumps({"env_info_after": {}})], dtype=object
                ),
            },
        )
        with self.assertRaisesRegex(ValueError, "missing authoritative"):
            _MODULE._aggregate_episode_scores(output, real_batch_size=1)

    def test_formal_episode_rows_require_done_evidence(self):
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[0.0]])},
            non_tensors={
                "agentmemory_step_record_json": np.array(
                    [self._formal_record(episode_success=False)], dtype=object
                ),
            },
        )
        with self.assertRaisesRegex(ValueError, "missing rollout_done_flags"):
            _MODULE._aggregate_episode_scores(output, real_batch_size=1)

    def test_formal_episode_rows_positive_reward_without_success_is_rejected(self):
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[5.0]])},
            non_tensors={
                "rollout_done_flags": np.array([True], dtype=object),
                "agentmemory_step_record_json": np.array(
                    [json.dumps({"episode_success": "yes"})], dtype=object
                ),
            },
        )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            _MODULE._aggregate_episode_scores(output, real_batch_size=1)

    def test_formal_episode_rows_require_boolean_done_flags(self):
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[0.0]])},
            non_tensors={
                "rollout_done_flags": np.array([1], dtype=object),
                "agentmemory_step_record_json": np.array(
                    [self._formal_record(episode_success=False)], dtype=object
                ),
            },
        )
        with self.assertRaisesRegex(ValueError, "rollout_done_flags must be boolean"):
            _MODULE._aggregate_episode_scores(output, real_batch_size=1)

    def test_formal_episode_rows_use_authoritative_success_not_return(self):
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[5.0]])},
            non_tensors={
                "rollout_done_flags": np.array([True], dtype=object),
                "agentmemory_step_record_json": np.array(
                    [self._formal_record(episode_success=False)], dtype=object
                ),
            },
        )
        scores, passes, progress, info = _MODULE._aggregate_episode_scores(
            output, real_batch_size=1
        )
        self.assertEqual(scores, [5.0])
        self.assertEqual(passes, [False])
        self.assertEqual(progress, [False])
        self.assertEqual(info["pass_source"], "formal_episode_success")
        self.assertEqual(info["progress_source"], "unknown")
        self.assertEqual(info["progress_unknown_count"], 1)

    def test_formal_episode_rows_authoritative_success_is_pass(self):
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[0.0]])},
            non_tensors={
                "rollout_done_flags": np.array([True], dtype=object),
                "agentmemory_step_record_json": np.array(
                    [self._formal_record(episode_success=True)], dtype=object
                ),
            },
        )
        _, passes, _, info = _MODULE._aggregate_episode_scores(
            output, real_batch_size=1
        )
        self.assertEqual(passes, [True])
        self.assertEqual(info["mode"], "formal_episode_rows")

    def test_phase_histogram_includes_zero_bins_and_unknown(self):
        records = [
            json.dumps(
                {
                    "episode_success": False,
                    "phase_index_after": 2,
                    "phase_count": 3,
                }
            ),
            json.dumps({"episode_success": False}),
        ]
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[0.0], [0.0]])},
            non_tensors={
                "rollout_parent_indices": np.array([0, 1], dtype=object),
                "rollout_done_flags": np.array([True, True], dtype=object),
                "agentmemory_step_record_json": np.array(records, dtype=object),
            },
        )
        self.assertEqual(
            _MODULE._formal_phase_progress_distribution(output, real_batch_size=2),
            {"0/3": 0, "1/3": 0, "2/3": 1, "3/3": 0, "unknown": 1},
        )

    def test_travel_last_phase_correct_after_prior_failure_is_not_pass(self):
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[0.0], [1.0]])},
            non_tensors={
                "rollout_parent_indices": np.array([0, 0], dtype=object),
                "rollout_done_flags": np.array([False, True], dtype=object),
                "agentmemory_step_record_json": np.array([
                    json.dumps({
                        "episode_success": False,
                        "domain_id": "travel_planner",
                        "phase_index_before": 0,
                        "phase_index_after": 0,
                        "phase_count": 2,
                    }),
                    json.dumps({
                        "episode_success": False,
                        "domain_id": "travel_planner",
                        "phase_index_before": 0,
                        "phase_index_after": 1,
                        "phase_count": 2,
                    }),
                ], dtype=object),
            },
        )

        scores, passes, progress, info = _MODULE._aggregate_episode_scores(
            output, real_batch_size=1
        )

        self.assertEqual(scores, [1.0])
        self.assertEqual(passes, [False])
        self.assertEqual(progress, [True])
        self.assertEqual(info["pass_source"], "formal_episode_success")

    def test_authoritative_terminal_success_is_pass(self):
        output = DataProto.from_dict(
            tensors={"task_scores": torch.tensor([[1.0]])},
            non_tensors={
                "rollout_parent_indices": np.array([0], dtype=object),
                "rollout_done_flags": np.array([True], dtype=object),
                "agentmemory_step_record_json": np.array([
                    self._formal_record(episode_success=True)
                ], dtype=object),
            },
        )

        _, passes, _, _ = _MODULE._aggregate_episode_scores(
            output, real_batch_size=1
        )

        self.assertEqual(passes, [True])


if __name__ == "__main__":
    unittest.main()
