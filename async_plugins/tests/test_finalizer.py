from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import yaml
from agentmemorygym_verl.config_contract import verify_resolved_config
from agentmemorygym_verl.finalizer import finalize_run
from finalizer_fixture import build_valid_run, mutate_json, sha256

JsonMutation = Callable[[dict], None]


def mutate_runtime(fixture: dict, mutation: JsonMutation) -> None:
    def mutate_wrapper(wrapper: dict) -> None:
        mutation(wrapper["data"])

    mutate_json(fixture["runtime_path"], mutate_wrapper)


def mutate_runtime_statistics(
    fixture: dict,
    component: str,
    mutation: JsonMutation,
) -> None:
    def mutate_receipt(receipt: dict) -> None:
        for boundary in ("before_clear", "after_clear"):
            statistics = receipt["snapshots"][boundary][component]["statistics"]
            mutation(statistics)

    mutate_runtime(fixture, mutate_receipt)


def rewrite_first_rollout(fixture: dict, mutation: JsonMutation) -> None:
    path = sorted(fixture["rollout_dir"].glob("*.jsonl"))[0]
    lines = path.read_text(encoding="utf-8").splitlines()
    document = json.loads(lines[0])
    mutation(document)
    lines[0] = json.dumps(document, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rewrite_first_step_record(fixture: dict, mutation: JsonMutation) -> None:
    def mutate_document(document: dict) -> None:
        record = json.loads(document["step_record_json"])
        mutation(record)
        document["step_record_json"] = json.dumps(record, sort_keys=True)

    rewrite_first_rollout(fixture, mutate_document)


def rewrite_chain_record(
    fixture: dict, order: int, mutation: JsonMutation
) -> None:
    path = sorted(fixture["rollout_dir"].glob("*.jsonl"))[0]
    documents = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    records = [json.loads(document["step_record_json"]) for document in documents]
    chain_uids = {
        record["trajectory_uid"]
        for record in records
        if record.get("wrapper_evidence", {}).get("event") == "context_compaction"
    }
    if len(chain_uids) != 1:
        raise AssertionError(f"expected one chain trajectory, got {chain_uids!r}")
    chain_uid = chain_uids.pop()
    matches = 0
    for document, record in zip(documents, records):
        if (
            record.get("trajectory_uid") == chain_uid
            and record.get("trajectory_row_order") == order
        ):
            mutation(record)
            document["output"] = record["action"]
            document["step_record_json"] = json.dumps(record, sort_keys=True)
            matches += 1
    if matches != 1:
        raise AssertionError(
            f"expected one chain row at order {order}, found {matches}"
        )
    path.write_text(
        "\n".join(json.dumps(document, sort_keys=True) for document in documents)
        + "\n",
        encoding="utf-8",
    )


def rewrite_metrics(fixture: dict, mutation: JsonMutation) -> None:
    path = fixture["metrics_path"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutation(rows[0])
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def make_post_initial_action_rows_stale(fixture: dict) -> int:
    """Make every action row after update 1 exactly one policy version stale."""

    stale_action_rows = 0
    stale_cumulative_by_update: dict[int, int] = {}
    for path in sorted(
        fixture["rollout_dir"].glob("*.jsonl"), key=lambda item: int(item.stem)
    ):
        update = int(path.stem)
        documents = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        current_version = update - 1
        for document in documents:
            record = json.loads(document["step_record_json"])
            if update > 1:
                record["min_global_steps"] = current_version - 1
                record["max_global_steps"] = current_version - 1
                stale_action_rows += 1
            document["step_record_json"] = json.dumps(record, sort_keys=True)
        path.write_text(
            "\n".join(json.dumps(document, sort_keys=True) for document in documents)
            + "\n",
            encoding="utf-8",
        )
        stale_cumulative_by_update[update] = stale_action_rows

    metric_rows = [
        json.loads(line)
        for line in fixture["metrics_path"].read_text(encoding="utf-8").splitlines()
    ]
    for row in metric_rows:
        update = int(row["step"])
        row["data"]["fully_async/count/current_param_version"] = update - 1
        row["data"]["fully_async/count/stale_trajectory_processed"] = (
            stale_cumulative_by_update[update]
        )
    fixture["metrics_path"].write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in metric_rows) + "\n",
        encoding="utf-8",
    )
    mutate_runtime_statistics(
        fixture,
        "trainer",
        lambda statistics: statistics.update(
            stale_trajectory_processed=stale_action_rows
        ),
    )
    return stale_action_rows


class FinalizerTestCase(unittest.TestCase):
    def build(self, root: Path, mode: str = "gate") -> dict:
        return build_valid_run(root / "run", mode=mode)

    def assert_failed(
        self,
        run_dir: Path,
        *,
        exit_code: int = 0,
        contains: str | None = None,
    ) -> dict:
        verdict = finalize_run(run_dir, trainer_exit_code=exit_code)
        self.assertEqual(verdict["status"], "fail", verdict)
        finalization_path = run_dir / "finalization.json"
        self.assertTrue(finalization_path.is_file())
        on_disk = json.loads(finalization_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["status"], "fail")
        if contains is not None:
            rendered = "\n".join(on_disk["errors"])
            self.assertIn(contains, rendered)
        return verdict


class TestFinalizerSuccess(FinalizerTestCase):
    def test_real_rich_v8_gate_fixture_passes_all_mechanism_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory), mode="gate")
            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)
            self.assertEqual(verdict["terminal_path"], "success")
            self.assertEqual(verdict["role"], "gate_only")
            self.assertEqual(verdict["counts"]["scheduled_episodes"], 64)
            self.assertEqual(verdict["counts"]["complete_learner_updates"], 1)
            self.assertEqual(verdict["counts"]["publication_cycles"], 1)
            self.assertEqual(verdict["counts"]["memory_chains"], 1)
            self.assertGreater(verdict["counts"]["real_action_rows"], 64)
            self.assertEqual(
                verdict["counts"]["derived_padding_action_rows"],
                fixture["padding_rows"],
            )
            self.assertEqual(
                (
                    verdict["counts"]["real_action_rows"]
                    + verdict["counts"]["derived_padding_action_rows"]
                )
                % 512,
                0,
            )
            self.assertEqual(verdict["trainer_exit_code"], 0)

    def test_real_rich_v8_formal_fixture_passes_exact_6400_episode_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory), mode="formal")
            metric_rows = (
                fixture["metrics_path"].read_text(encoding="utf-8").splitlines()
            )
            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(len(metric_rows), 100)
            self.assertEqual(verdict["status"], "pass", verdict)
            self.assertEqual(verdict["role"], "train_pool")
            self.assertEqual(
                verdict["counts"]["scheduled_episodes"], len(fixture["schedule_rows"])
            )
            self.assertEqual(verdict["counts"]["complete_learner_updates"], 100)
            self.assertEqual(verdict["counts"]["publication_cycles"], 100)
            self.assertEqual(verdict["counts"]["policy_version_min"], 0)
            self.assertEqual(verdict["counts"]["policy_version_max"], 99)
            self.assertEqual(verdict["counts"]["validation_events"], 0)


class TestFinalizerTerminalPaths(FinalizerTestCase):
    def test_trainer_crash_always_fails_even_when_all_artifacts_would_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            verdict = self.assert_failed(
                fixture["run_dir"], exit_code=134, contains="trainer exit code 134"
            )
            self.assertEqual(verdict["terminal_path"], "crash")

    def test_native_crash_receipt_is_classified_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))

            def make_crash(receipt: dict) -> None:
                receipt.update(
                    outcome="crash",
                    status="failed",
                    exception={
                        "type": "RuntimeError",
                        "module": "builtins",
                        "message": "trainer exploded",
                    },
                )

            mutate_runtime(fixture, make_crash)
            verdict = self.assert_failed(
                fixture["run_dir"], exit_code=1, contains="native runtime outcome"
            )
            self.assertEqual(verdict["terminal_path"], "crash")

    def test_terminal_underfill_partial_lineage_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))

            def make_partial(receipt: dict) -> None:
                receipt.update(outcome="terminal_underfill", status="partial")
                for boundary in ("before_clear", "after_clear"):
                    statistics = receipt["snapshots"][boundary]["trainer"]["statistics"]
                    statistics["terminal_underfill_events"] = 1
                    statistics["terminal_underfill_samples"] = 1

            mutate_runtime(fixture, make_partial)
            verdict = self.assert_failed(
                fixture["run_dir"], contains="terminal_underfill"
            )
            self.assertEqual(verdict["terminal_path"], "partial")

    def test_stale_success_is_replaced_by_later_failed_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            self.assertEqual(
                finalize_run(fixture["run_dir"], trainer_exit_code=0)["status"],
                "pass",
            )
            fixture["runtime_path"].unlink()
            self.assert_failed(fixture["run_dir"], contains="native runtime receipt")


class TestFinalizerMissingArtifacts(FinalizerTestCase):
    def test_every_required_native_artifact_is_mandatory(self):
        cases = {
            "launch receipt": lambda fixture: fixture["launch_path"].unlink(),
            "native runtime receipt": lambda fixture: fixture["runtime_path"].unlink(),
            "FileLogger": lambda fixture: fixture["metrics_path"].unlink(),
            "resolved config": lambda fixture: fixture["resolved_path"].unlink(),
            "Hydra config": lambda fixture: fixture["hydra_path"].unlink(),
            "rollout JSONL": lambda fixture: next(
                fixture["rollout_dir"].glob("*.jsonl")
            ).unlink(),
            "checkpoint tracker": lambda fixture: (
                fixture["checkpoint_root"] / "latest_checkpointed_iteration.txt"
            ).unlink(),
            "dataloader checkpoint": lambda fixture: (
                fixture["checkpoint_root"] / "global_step_1" / "data.pt"
            ).unlink(),
        }
        for expected, remove in cases.items():
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                remove(fixture)
                self.assert_failed(fixture["run_dir"], contains=expected)

    def test_missing_or_unavailable_runtime_components_fail_closed(self):
        cases: tuple[tuple[str, JsonMutation], ...] = (
            ("snapshots mapping", lambda receipt: receipt.pop("snapshots")),
            (
                "before_clear.trainer",
                lambda receipt: receipt["snapshots"]["before_clear"].pop("trainer"),
            ),
            (
                "after_clear.rollouter statistics are unavailable",
                lambda receipt: receipt["snapshots"]["after_clear"].__setitem__(
                    "rollouter",
                    {
                        "timestamp": "2026-08-18T00:00:02+00:00",
                        "available": False,
                        "error": {
                            "type": "RuntimeError",
                            "module": "builtins",
                            "message": "statistics unavailable",
                        },
                    },
                ),
            ),
            (
                "before_clear.queue statistics are unavailable",
                lambda receipt: receipt["snapshots"]["before_clear"].__setitem__(
                    "queue", {"available": False, "error": {"type": "RuntimeError"}}
                ),
            ),
        )
        for expected, mutation in cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                mutate_runtime(fixture, mutation)
                self.assert_failed(fixture["run_dir"], contains=expected)


class TestFinalizerRuntimeReceipt(FinalizerTestCase):
    def test_native_stale_counter_tracks_action_rows_not_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory), mode="formal")
            stale_action_rows = make_post_initial_action_rows_stale(fixture)

            self.assertGreater(stale_action_rows, len(fixture["schedule_rows"]))
            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)
            self.assertEqual(
                verdict["counts"]["stale_action_rows"], stale_action_rows
            )

    def test_wrapper_and_terminal_metadata_mismatches_fail(self):
        wrapper_cases: tuple[tuple[str, JsonMutation], ...] = (
            ("wrapper step", lambda wrapper: wrapper.update(step=0)),
            ("data mapping", lambda wrapper: wrapper.update(data=[])),
        )
        for expected, mutation in wrapper_cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                mutate_json(fixture["runtime_path"], mutation)
                self.assert_failed(fixture["run_dir"], contains=expected)

        receipt_cases: tuple[tuple[str, JsonMutation], ...] = (
            ("schema_version", lambda receipt: receipt.update(schema_version=2)),
            ("native runtime outcome", lambda receipt: receipt.update(outcome="crash")),
            ("native runtime status", lambda receipt: receipt.update(status="partial")),
            (
                "native runtime exception",
                lambda receipt: receipt.update(exception={"type": "RuntimeError"}),
            ),
            (
                "finalization_errors",
                lambda receipt: receipt.update(
                    finalization_errors=[
                        {"stage": "queue.clear", "exception": {"type": "OSError"}}
                    ]
                ),
            ),
            ("timestamps", lambda receipt: receipt["timestamps"].pop("finalized_at")),
            ("trainer_step", lambda receipt: receipt.update(trainer_step=0)),
        )
        for expected, mutation in receipt_cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                mutate_runtime(fixture, mutation)
                self.assert_failed(fixture["run_dir"], contains=expected)

    def test_trainer_counter_and_bounded_evidence_mismatches_fail(self):
        cases: tuple[tuple[str, JsonMutation], ...] = (
            ("global_steps", lambda stats: stats.update(global_steps=1)),
            (
                "current_param_version",
                lambda stats: stats.update(current_param_version=0),
            ),
            ("total_train_steps", lambda stats: stats.update(total_train_steps=2)),
            ("local_trigger_step", lambda stats: stats.update(local_trigger_step=2)),
            ("processed_samples", lambda stats: stats.update(processed_samples=63)),
            (
                "stale_trajectory_processed",
                lambda stats: stats.update(stale_trajectory_processed=65),
            ),
            (
                "terminal_underfill_events",
                lambda stats: stats.update(terminal_underfill_events=1),
            ),
            (
                "terminal_underfill_samples",
                lambda stats: stats.update(terminal_underfill_samples=1),
            ),
            (
                "pending_rollout_dump_writes",
                lambda stats: stats.update(pending_rollout_dump_writes=1),
            ),
            (
                "bypass real-token count",
                lambda stats: stats["latest_bypass_log_prob_evidence"].update(
                    **{"rollout_corr/bypass_real_token_count": 0}
                ),
            ),
            (
                "old/rollout logprob",
                lambda stats: stats["latest_bypass_log_prob_evidence"].update(
                    **{"rollout_corr/bypass_max_abs_diff": 0.01}
                ),
            ),
            (
                "actor parameter-update probe",
                lambda stats: stats["latest_parameter_update_probe"]["actor"].update(
                    changed=False, changed_elements=0, max_abs_diff=0.0
                ),
            ),
            (
                "critic parameter-update probe",
                lambda stats: stats["latest_parameter_update_probe"]["critic"].update(
                    changed=False, changed_elements=0, max_abs_diff=0.0
                ),
            ),
        )
        for expected, mutation in cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                mutate_runtime_statistics(fixture, "trainer", mutation)
                self.assert_failed(fixture["run_dir"], contains=expected)

    def test_rollouter_counter_and_pending_work_mismatches_fail(self):
        cases: tuple[tuple[str, JsonMutation], ...] = (
            (
                "active_tasks_size",
                lambda stats: stats.update(**{"monitor/active_tasks_size": 1}),
            ),
            (
                "pending_queue_size",
                lambda stats: stats.update(**{"monitor/queue/pending_queue_size": 1}),
            ),
            (
                "mq_queue_size",
                lambda stats: stats.update(**{"monitor/queue/mq_queue_size": 1}),
            ),
            (
                "total_generated_samples",
                lambda stats: stats.update(**{"count/total_generated_samples": 63}),
            ),
            (
                "staleness_samples",
                lambda stats: stats.update(**{"count/staleness_samples": 65}),
            ),
            (
                "dropped_stale_samples",
                lambda stats: stats.update(**{"count/dropped_stale_samples": 1}),
            ),
            (
                "required_samples",
                lambda stats: stats.update(**{"static/required_samples": 63}),
            ),
            (
                "max_required_samples",
                lambda stats: stats.update(**{"static/max_required_samples": 69}),
            ),
            (
                "max_queue_size",
                lambda stats: stats.update(**{"static/max_queue_size": 69}),
            ),
            (
                "max_concurrent_samples",
                lambda stats: stats.update(**{"static/max_concurrent_samples": 0}),
            ),
        )
        for expected, mutation in cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                mutate_runtime_statistics(fixture, "rollouter", mutation)
                self.assert_failed(fixture["run_dir"], contains=expected)

    def test_queue_conservation_count_and_terminal_state_mismatches_fail(self):
        cases: tuple[tuple[str, JsonMutation], ...] = (
            ("total_produced", lambda stats: stats.update(total_produced=63)),
            ("total_consumed", lambda stats: stats.update(total_consumed=63)),
            ("real_enqueued", lambda stats: stats.update(real_enqueued=63)),
            ("real_consumed", lambda stats: stats.update(real_consumed=63)),
            ("real_evicted", lambda stats: stats.update(real_evicted=1)),
            ("real_cleared", lambda stats: stats.update(real_cleared=1)),
            ("real_resident", lambda stats: stats.update(real_resident=1)),
            ("queue_size", lambda stats: stats.update(queue_size=1)),
            ("dropped_samples", lambda stats: stats.update(dropped_samples=1)),
            ("closed", lambda stats: stats.update(closed=False)),
            (
                "control_signals_enqueued",
                lambda stats: stats.update(control_signals_enqueued=1),
            ),
        )
        for expected, mutation in cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                mutate_runtime_statistics(fixture, "queue", mutation)
                self.assert_failed(fixture["run_dir"], contains=expected)

    def test_queue_receipt_flags_and_before_after_stability_are_verified(self):
        cases: tuple[tuple[str, JsonMutation], ...] = (
            (
                "queue_conservation.before_clear",
                lambda receipt: receipt["queue_conservation"].update(
                    before_clear=False
                ),
            ),
            (
                "queue_conservation.after_clear",
                lambda receipt: receipt["queue_conservation"].update(after_clear=False),
            ),
            (
                "clear_delta_matches_resident",
                lambda receipt: receipt["queue_conservation"].update(
                    clear_delta_matches_resident=False
                ),
            ),
            (
                "before/after trainer statistics",
                lambda receipt: receipt["snapshots"]["after_clear"]["trainer"][
                    "statistics"
                ].update(global_steps=3),
            ),
        )
        for expected, mutation in cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                mutate_runtime(fixture, mutation)
                self.assert_failed(fixture["run_dir"], contains=expected)


class TestFinalizerFileLogger(FinalizerTestCase):
    def test_step_zero_rollouter_bootstrap_and_split_publication_rows_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            original = json.loads(
                fixture["metrics_path"].read_text(encoding="utf-8").splitlines()[0]
            )
            learner = dict(original["data"])
            bypass = {
                key: learner.pop(key)
                for key in (
                    "rollout_corr/bypass_real_token_count",
                    "rollout_corr/bypass_max_abs_diff",
                )
            }
            rows = [
                {
                    "step": 0,
                    "data": {
                        "fully_async/rollouter/active_time": 1.0,
                        "dynamic_resource/rollout_resource_utilization": 0.5,
                    },
                },
                {"step": 1, "data": bypass},
                {"step": 1, "data": learner},
            ]
            fixture["metrics_path"].write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)

    def test_step_zero_must_be_rollouter_only(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            with fixture["metrics_path"].open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"step": 0, "data": {"actor/grad_norm": 1.0}})
                    + "\n"
                )
            self.assert_failed(
                fixture["run_dir"], contains="step 0 is not rollouter-only"
            )

    def test_each_publication_has_learner_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory), mode="formal")
            rows = [
                json.loads(line)
                for line in fixture["metrics_path"].read_text(encoding="utf-8").splitlines()
            ]
            rows[1]["data"].pop("actor/grad_norm")
            fixture["metrics_path"].write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            self.assert_failed(
                fixture["run_dir"], contains="publication step 2 has no unique nonzero actor update metric"
            )

    def test_publication_cycle_evidence_is_complete_and_exact(self):
        cases: tuple[tuple[str, JsonMutation], ...] = (
            (
                "compared real-token count",
                lambda row: row["data"].update(
                    **{"rollout_corr/bypass_real_token_count": 0}
                ),
            ),
            (
                "old/rollout logprob mismatch",
                lambda row: row["data"].update(
                    **{"rollout_corr/bypass_max_abs_diff": 0.001}
                ),
            ),
            (
                "actor parameter-update probe",
                lambda row: row["data"].update(
                    **{"parameter_update_probe/actor/changed": False}
                ),
            ),
            (
                "critic parameter-update probe",
                lambda row: row["data"].update(
                    **{"parameter_update_probe/critic/changed": False}
                ),
            ),
            (
                "actor update metric",
                lambda row: row["data"].update(**{"actor/grad_norm": 0.0}),
            ),
            (
                "critic update metric",
                lambda row: row["data"].update(**{"critic/grad_norm": 0.0}),
            ),
            (
                "current parameter versions",
                lambda row: row["data"].update(
                    **{"fully_async/count/current_param_version": 1}
                ),
            ),
            (
                "stale action-row count mismatch",
                lambda row: row["data"].update(
                    **{"fully_async/count/stale_trajectory_processed": 1}
                ),
            ),
        )
        for expected, mutation in cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                rewrite_metrics(fixture, mutation)
                self.assert_failed(fixture["run_dir"], contains=expected)

    def test_update_metrics_require_unique_canonical_grad_norms(self):
        cases = (
            ("actor", "distractor/value"),
            ("critic", "critic/values/max"),
        )
        for role, decoy in cases:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                fixture = self.build(Path(directory))
                rows = [
                    json.loads(line)
                    for line in fixture["metrics_path"]
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                rows[0]["data"].pop(f"{role}/grad_norm")
                rows[0]["data"][decoy] = 1.0
                fixture["metrics_path"].write_text(
                    "\n".join(json.dumps(row, sort_keys=True) for row in rows)
                    + "\n",
                    encoding="utf-8",
                )
                self.assert_failed(
                    fixture["run_dir"],
                    contains=f"no unique nonzero {role} update metric",
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            with fixture["metrics_path"].open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"step": 1, "data": {"actor/grad_norm": 2.0}},
                        sort_keys=True,
                    )
                    + "\n"
                )
            self.assert_failed(
                fixture["run_dir"],
                contains="no unique nonzero actor update metric",
            )

    def test_file_logger_token_sum_and_validation_count_are_cross_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            rewrite_metrics(
                fixture,
                lambda row: row["data"].update(
                    **{"rollout_corr/bypass_real_token_count": 1}
                ),
            )
            self.assert_failed(fixture["run_dir"], contains="real-token total")

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            with fixture["metrics_path"].open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"step": 1, "data": {"val-core/score/mean@1": 1.0}})
                    + "\n"
                )
            self.assert_failed(fixture["run_dir"], contains="validation metric")


class TestFinalizerRollouts(FinalizerTestCase):
    def test_async_completion_order_is_a_multiset_not_schedule_order(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            path = fixture["rollout_dir"] / "1.jsonl"
            documents = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            first_uid = json.loads(documents[0]["step_record_json"])["trajectory_uid"]
            first_episode = [
                document
                for document in documents
                if json.loads(document["step_record_json"])["trajectory_uid"]
                == first_uid
            ]
            other_episodes = [
                document for document in documents if document not in first_episode
            ]
            path.write_text(
                "\n".join(
                    json.dumps(document, sort_keys=True)
                    for document in other_episodes + first_episode
                )
                + "\n",
                encoding="utf-8",
            )
            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)
            self.assertEqual(verdict["status"], "pass", verdict)

    def test_action_rows_may_be_physically_interleaved_within_an_update(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            path = fixture["rollout_dir"] / "1.jsonl"
            documents = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            documents.sort(
                key=lambda document: (
                    -json.loads(document["step_record_json"])["trajectory_row_order"],
                    json.loads(document["step_record_json"])["trajectory_uid"],
                )
            )
            path.write_text(
                "\n".join(
                    json.dumps(document, sort_keys=True) for document in documents
                )
                + "\n",
                encoding="utf-8",
            )

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)

    def test_duplicate_or_missing_action_order_still_fails(self):
        for bad_order in (1, 99):
            with (
                self.subTest(bad_order=bad_order),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                rewrite_first_step_record(
                    fixture,
                    lambda record: record.update(trajectory_row_order=bad_order),
                )
                self.assert_failed(
                    fixture["run_dir"], contains="action rows are not contiguous"
                )

    def test_cross_update_uid_reuse_and_early_terminal_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory), mode="formal")
            first_path = fixture["rollout_dir"] / "1.jsonl"
            second_path = fixture["rollout_dir"] / "2.jsonl"
            first_record = json.loads(
                json.loads(first_path.read_text(encoding="utf-8").splitlines()[0])[
                    "step_record_json"
                ]
            )
            documents = [
                json.loads(line)
                for line in second_path.read_text(encoding="utf-8").splitlines()
            ]
            record = json.loads(documents[0]["step_record_json"])
            record["trajectory_uid"] = first_record["trajectory_uid"]
            documents[0]["step_record_json"] = json.dumps(record, sort_keys=True)
            second_path.write_text(
                "\n".join(json.dumps(document, sort_keys=True) for document in documents)
                + "\n",
                encoding="utf-8",
            )
            self.assert_failed(
                fixture["run_dir"], contains="appears in multiple updates"
            )

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            rewrite_chain_record(
                fixture,
                2,
                lambda record: record.update(
                    trajectory_terminal=True, rollout_done_flag=True
                ),
            )
            rewrite_chain_record(
                fixture,
                3,
                lambda record: record.update(
                    trajectory_terminal=False, rollout_done_flag=False
                ),
            )
            self.assert_failed(
                fixture["run_dir"], contains="terminal row is not the maximum action order"
            )

    def test_rollout_files_must_have_numeric_optimizer_step_names(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            (fixture["rollout_dir"] / "1.jsonl").rename(
                fixture["rollout_dir"] / "update.jsonl"
            )
            self.assert_failed(
                fixture["run_dir"], contains="filenames must be numeric optimizer steps"
            )

    def test_policy_version_span_and_real_token_fields_are_verified(self):
        cases: tuple[tuple[str, JsonMutation], ...] = (
            ("policy-version", lambda record: record.update(max_global_steps=99)),
            (
                "response_token_count",
                lambda record: record.update(response_token_count=0),
            ),
            (
                "trajectory identity",
                lambda record: record.update(item_id="substitution"),
            ),
        )
        for expected, mutation in cases:
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                rewrite_first_step_record(fixture, mutation)
                self.assert_failed(fixture["run_dir"], contains=expected)

    def test_dumped_padding_and_per_collection_sample_conservation_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            rewrite_first_rollout(
                fixture, lambda document: document.update(is_padding=True)
            )
            self.assert_failed(fixture["run_dir"], contains="synthetic padding row")

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            path = next(fixture["rollout_dir"].glob("*.jsonl"))
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            self.assert_failed(
                fixture["run_dir"], contains="terminal trajectories per learner update"
            )

    def test_missing_or_synthetic_memory_chain_is_not_accepted(self):
        cases = (
            "missing_compaction",
            "unpersisted_compaction",
            "empty_read",
            "failed_execute",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = self.build(Path(directory))
                path = next(fixture["rollout_dir"].glob("*.jsonl"))
                rewritten = []
                for raw in path.read_text(encoding="utf-8").splitlines():
                    document = json.loads(raw)
                    record = json.loads(document["step_record_json"])
                    if record["trajectory_row_order"] == 0:
                        if case == "missing_compaction":
                            record["wrapper_evidence"].pop("event", None)
                        elif case == "unpersisted_compaction":
                            record["wrapper_evidence"]["continuation_persisted"] = False
                    elif record["trajectory_row_order"] == 1 and case == "empty_read":
                        record["env_info_after"]["execution"]["stdout"] = ""
                    elif (
                        record["trajectory_row_order"] == 3 and case == "failed_execute"
                    ):
                        record["env_info_after"]["execution"]["exit_code"] = 1
                    document["step_record_json"] = json.dumps(record, sort_keys=True)
                    rewritten.append(json.dumps(document, sort_keys=True))
                path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
                self.assert_failed(
                    fixture["run_dir"],
                    contains="policy-authored external-document chain",
                )


    def test_apply_patch_can_persist_the_compaction_document(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            action = (
                "apply_patch\n*** Begin Patch\n"
                "*** Add File: .agent_memory/OPENMLE_CONTINUATION.md\n"
                "+objective: improve validation\n"
                "+measured_validation_or_failure: validation_mae=1.0\n"
                "+conclusion: update the model\n"
                "+code_path: train.py\n"
                "+next_action: edit train.py before rerunning\n"
                "*** End Patch"
            )

            def mutate(record: dict) -> None:
                record["action"] = action
                record["action_submission"]["raw_policy_output"] = action
                info = record["env_info_after"]
                info["action_kind"] = "apply_patch"
                record["wrapper_evidence"]["native_action_kind"] = "apply_patch"
                execution = info["execution"]
                execution["action_kind"] = "apply_patch"
                execution["exit_code"] = None

            rewrite_chain_record(fixture, 0, mutate)

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)

    def test_action_decoys_do_not_form_a_memory_chain(self):
        orders = {
            "echo_read": 1,
            "commented_execute": 3,
            "subpath_read": 1,
            "subpath_edit": 2,
            "subpath_execute": 3,
        }
        for case, order in orders.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = self.build(Path(directory))

                def mutate(record: dict) -> None:
                    if case == "echo_read":
                        action = (
                            "shell_command {\"command\":\"printf 'objective: improve "
                            "validation\\nmeasured_validation_or_failure: validation_mae=1.0"
                            "\\nconclusion: update the model\\ncode_path: train.py"
                            "\\nnext_action: edit train.py before rerunning\\n' # "
                            ".agent_memory/OPENMLE_CONTINUATION.md\",\"workdir\":\".\","
                            "\"timeout_ms\":20000}"
                        )
                    elif case == "commented_execute":
                        action = (
                            "shell_command {\"command\":\"python other.py # train.py\","
                            "\"workdir\":\".\",\"timeout_ms\":20000}"
                        )
                    elif case == "subpath_read":
                        action = (
                            "shell_command {\"command\":\"cat nested/.agent_memory/"
                            "OPENMLE_CONTINUATION.md\",\"workdir\":\".\","
                            "\"timeout_ms\":20000}"
                        )
                    elif case == "subpath_edit":
                        action = (
                            "apply_patch\n*** Begin Patch\n"
                            "*** Update File: nested/train.py\n"
                            "@@\n-print(1)\n+print(2)\n*** End Patch"
                        )
                        record["env_info_after"]["execution"]["changed_paths"] = [
                            "nested/train.py"
                        ]
                    else:
                        action = (
                            "shell_command {\"command\":\"python nested/train.py\","
                            "\"workdir\":\".\",\"timeout_ms\":20000}"
                        )
                    record["action"] = action
                    record["action_submission"]["raw_policy_output"] = action

                rewrite_chain_record(fixture, order, mutate)
                self.assert_failed(
                    fixture["run_dir"],
                    contains="policy-authored external-document chain",
                )

    def test_continuation_fields_and_completed_execution_are_required(self):
        cases = ("missing_objective", "not_completed")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = self.build(Path(directory))
                order = 1 if case == "missing_objective" else 3

                def mutate(record: dict) -> None:
                    if case == "missing_objective":
                        execution = record["env_info_after"]["execution"]
                        execution["stdout"] = "\n".join(
                            line
                            for line in execution["stdout"].splitlines()
                            if not line.startswith("objective:")
                        )
                    else:
                        info = record["env_info_after"]
                        info["counter_delta"]["execution_completed_count"] = 0
                        info["execution"]["execution_completed_delta"] = 0

                rewrite_chain_record(fixture, order, mutate)
                self.assert_failed(
                    fixture["run_dir"],
                    contains="policy-authored external-document chain",
                )


class TestFinalizerConfigAndCheckpoint(FinalizerTestCase):
    def test_six_trainer_checkpoint_and_probe_world_size_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            config = yaml.safe_load(
                fixture["resolved_path"].read_text(encoding="utf-8")
            )
            config["trainer"]["n_gpus_per_node"] = 6
            config["rollout"]["n_gpus_per_node"] = 2
            config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 510
            config["critic"]["ppo_mini_batch_size"] = 510
            config["async_training"]["require_batches"] = 64 / 510
            text = yaml.safe_dump(config, sort_keys=True)
            fixture["resolved_path"].write_text(text, encoding="utf-8")
            fixture["hydra_path"].write_text(text, encoding="utf-8")

            launch = json.loads(
                fixture["launch_path"].read_text(encoding="utf-8")
            )
            launch["budget"] = verify_resolved_config(
                config,
                mode="gate",
                expected_budget=launch["budget_contract"],
            )
            launch["resolved_config"]["sha256"] = sha256(
                fixture["resolved_path"]
            )
            fixture["launch_path"].write_text(
                json.dumps(launch, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            checkpoint = fixture["checkpoint_root"] / "global_step_1"
            for role in ("actor", "critic"):
                role_dir = checkpoint / role
                for path in role_dir.glob("*_world_size_4_rank_*.pt"):
                    path.unlink()
                for kind in ("model", "optim", "extra_state"):
                    for rank in range(6):
                        (role_dir / f"{kind}_world_size_6_rank_{rank}.pt").write_bytes(
                            f"{role}:{kind}:{rank}".encode()
                        )

            def set_worker_count(stats: dict) -> None:
                for evidence in stats["latest_parameter_update_probe"].values():
                    evidence["worker_count"] = 6

            mutate_runtime_statistics(fixture, "trainer", set_worker_count)

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)

    def test_hydra_interpolation_is_resolved_before_drift_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            config = yaml.safe_load(fixture["hydra_path"].read_text(encoding="utf-8"))
            config["trainer"]["n_gpus_per_node"] = (
                "${oc.select:rollout.n_gpus_per_node}"
            )
            fixture["hydra_path"].write_text(
                yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
            )

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)

    def test_resolved_config_mismatch_and_hydra_drift_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            config = yaml.safe_load(
                fixture["resolved_path"].read_text(encoding="utf-8")
            )
            config["trainer"]["test_freq"] = 1
            text = yaml.safe_dump(config, sort_keys=True)
            fixture["resolved_path"].write_text(text, encoding="utf-8")
            fixture["hydra_path"].write_text(text, encoding="utf-8")
            mutate_json(
                fixture["launch_path"],
                lambda value: value["resolved_config"].update(
                    sha256=sha256(fixture["resolved_path"])
                ),
            )
            self.assert_failed(fixture["run_dir"], contains="test_freq")

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            config = yaml.safe_load(fixture["hydra_path"].read_text(encoding="utf-8"))
            config["trainer"]["n_gpus_per_node"] = 8
            fixture["hydra_path"].write_text(
                yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
            )
            self.assert_failed(fixture["run_dir"], contains="Hydra config")

    def test_incomplete_actor_or_critic_checkpoint_shards_fail(self):
        for role in ("actor", "critic"):
            for kind in ("model", "optim", "extra_state"):
                with (
                    self.subTest(role=role, kind=kind),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    fixture = self.build(Path(directory))
                    target = (
                        fixture["checkpoint_root"]
                        / "global_step_1"
                        / role
                        / f"{kind}_world_size_4_rank_3.pt"
                    )
                    target.unlink()
                    self.assert_failed(
                        fixture["run_dir"], contains=f"checkpoint {role}"
                    )

    def test_checkpoint_tracker_and_runtime_artifact_binding_fail_on_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            tracker = fixture["checkpoint_root"] / "latest_checkpointed_iteration.txt"
            tracker.write_text("0", encoding="utf-8")
            self.assert_failed(fixture["run_dir"], contains="checkpoint tracker step")

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            mutate_json(
                fixture["launch_path"],
                lambda receipt: receipt["runtime_artifacts"].update(
                    native_receipt="/tmp/caller-selected-runtime-receipt.json"
                ),
            )
            self.assert_failed(
                fixture["run_dir"], contains="runtime artifact native_receipt"
            )


if __name__ == "__main__":
    unittest.main()
