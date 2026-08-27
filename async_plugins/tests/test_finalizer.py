from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import yaml
from agentmemorygym_verl.config_contract import verify_resolved_config
from agentmemorygym_verl.finalizer import finalize_run
from finalizer_fixture import (
    MULTITASK_ROUTES,
    build_valid_multitask_run,
    build_valid_run,
    mutate_final_statistics,
    mutate_json,
    messages_sha256,
    sha256,
)

JsonMutation = Callable[[dict], None]


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


def rewrite_chain_record(fixture: dict, order: int, mutation: JsonMutation) -> None:
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


def rewrite_multitask_source_lock(fixture: dict, mutation: JsonMutation) -> None:
    receipt = json.loads(fixture["launch_path"].read_text(encoding="utf-8"))
    source_lock_path = Path(receipt["launch_identity"]["source_lock_path"])
    mutate_json(source_lock_path, mutation)
    receipt["launch_identity"]["source_lock_sha256"] = sha256(source_lock_path)
    fixture["launch_path"].write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rewrite_multitask_certificate(
    fixture: dict, mutation: JsonMutation, *, mirror_contract: bool = True
) -> None:
    receipt = json.loads(fixture["launch_path"].read_text(encoding="utf-8"))
    identity = receipt["launch_identity"]
    certificate_path = Path(identity["schedule_certificate_path"])
    source_lock_path = Path(identity["source_lock_path"])
    mutate_json(certificate_path, mutation)
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate_digest = sha256(certificate_path)
    mutate_json(
        source_lock_path,
        lambda source_lock: source_lock["integration"]["schedule_certificate"].update(
            sha256=certificate_digest
        ),
    )
    identity["schedule_certificate_sha256"] = certificate_digest
    identity["source_lock_sha256"] = sha256(source_lock_path)
    if mirror_contract:
        identity["formal_schedule_contract"] = certificate
    fixture["launch_path"].write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
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

    def test_legacy_receipt_cannot_be_reinterpreted_as_multitask(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            mutate_json(
                fixture["launch_path"],
                lambda receipt: receipt.update(
                    schema="amg_verl_fully_async_multitask_launch_receipt_v1"
                ),
            )

            self.assert_failed(fixture["run_dir"], contains="multitask")


class TestFinalizerTerminalPaths(FinalizerTestCase):
    def test_trainer_crash_always_fails_even_when_all_artifacts_would_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            verdict = self.assert_failed(
                fixture["run_dir"], exit_code=134, contains="trainer exit code 134"
            )
            self.assertEqual(verdict["terminal_path"], "crash")

    def test_incomplete_native_rollout_budget_is_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            path = sorted(fixture["rollout_dir"].glob("*.jsonl"))[0]
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            first_record = json.loads(rows[0]["step_record_json"])
            uid = first_record["trajectory_uid"]
            rows = [
                row
                for row in rows
                if json.loads(row["step_record_json"])["trajectory_uid"] != uid
            ]
            path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )

            verdict = self.assert_failed(
                fixture["run_dir"], contains="terminal trajectories per learner update"
            )
            self.assertEqual(verdict["terminal_path"], "partial")

    def test_stale_success_is_replaced_by_later_failed_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            self.assertEqual(
                finalize_run(fixture["run_dir"], trainer_exit_code=0)["status"],
                "pass",
            )
            fixture["metrics_path"].unlink()
            self.assert_failed(fixture["run_dir"], contains="FileLogger JSONL")

    def test_legacy_shadow_finalization_cannot_leave_stale_canonical_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            canonical = fixture["run_dir"] / "finalization.json"
            self.assertEqual(
                finalize_run(fixture["run_dir"], trainer_exit_code=0)["status"],
                "pass",
            )
            mutate_json(
                fixture["launch_path"],
                lambda receipt: receipt["runtime_artifacts"].update(
                    finalization=str(fixture["run_dir"] / "shadow" / "verdict.json")
                ),
            )

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "fail")
            self.assertEqual(
                json.loads(canonical.read_text(encoding="utf-8")), verdict
            )
            self.assertFalse((fixture["run_dir"] / "shadow" / "verdict.json").exists())


class TestFinalizerMissingArtifacts(FinalizerTestCase):
    def test_every_required_native_artifact_is_mandatory(self):
        cases = {
            "launch receipt": lambda fixture: fixture["launch_path"].unlink(),
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


class TestFinalizerFileLogger(FinalizerTestCase):
    def test_step_zero_rollouter_bootstrap_and_split_publication_rows_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            original = json.loads(
                fixture["metrics_path"].read_text(encoding="utf-8").splitlines()[0]
            )
            learner = dict(original["data"])
            correction = {
                key: learner.pop(key)
                for key in (
                    "rollout_corr/kl",
                    "rollout_corr/k3_kl",
                    "rollout_corr/log_ppl_abs_diff",
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
                {"step": 1, "data": correction},
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
                    json.dumps({"step": 0, "data": {"actor/grad_norm": 1.0}}) + "\n"
                )
            self.assert_failed(
                fixture["run_dir"], contains="step 0 is not rollouter-only"
            )

    def test_each_publication_has_native_learner_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory), mode="formal")
            rows = [
                json.loads(line)
                for line in fixture["metrics_path"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rows[1]["data"].pop("actor/grad_norm")
            fixture["metrics_path"].write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            self.assert_failed(
                fixture["run_dir"],
                contains="publication step 2 has no unique nonzero actor/grad_norm",
            )

    def test_publication_cycle_native_evidence_is_complete_and_exact(self):
        cases: tuple[tuple[str, JsonMutation], ...] = (
            (
                "rollout_corr/kl",
                lambda row: row["data"].pop("rollout_corr/kl"),
            ),
            (
                "rollout_corr/k3_kl",
                lambda row: row["data"].update(**{"rollout_corr/k3_kl": float("nan")}),
            ),
            (
                "actor/grad_norm",
                lambda row: row["data"].update(**{"actor/grad_norm": 0.0}),
            ),
            (
                "critic/grad_norm",
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
            (
                "required_samples mismatch",
                lambda row: row["data"].update(
                    **{"fully_async/static/required_samples": 63}
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
        cases = (("actor", "distractor/value"), ("critic", "critic/values/max"))
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
                    "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                    encoding="utf-8",
                )
                self.assert_failed(
                    fixture["run_dir"], contains=f"no unique nonzero {role}/grad_norm"
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            with fixture["metrics_path"].open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"step": 1, "data": {"actor/grad_norm": 2.0}}, sort_keys=True
                    )
                    + "\n"
                )
            self.assert_failed(
                fixture["run_dir"], contains="no unique nonzero actor/grad_norm"
            )

    def test_native_counters_are_monotone_and_validation_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory), mode="formal")
            rows = [
                json.loads(line)
                for line in fixture["metrics_path"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rows[10]["data"]["fully_async/count/total_generated_samples"] = 1
            fixture["metrics_path"].write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            self.assert_failed(
                fixture["run_dir"],
                contains="total-generated-samples count is not cumulative",
            )

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

    def test_finalizer_rejects_schedule_data_idx_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            for order in range(4):
                rewrite_chain_record(
                    fixture,
                    order,
                    lambda record: record.update(data_idx=1),
                )
            self.assert_failed(
                fixture["run_dir"], contains="item_id/data_idx occurrences"
            )

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
                "\n".join(
                    json.dumps(document, sort_keys=True) for document in documents
                )
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
                fixture["run_dir"],
                contains="terminal row is not the maximum action order",
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
                        record["wrapper_evidence"].pop("document_read_observed", None)
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
                record["wrapper_evidence"]["checkpoint_receipt"][
                    "action_kind"
                ] = "apply_patch"
                execution = info["execution"]
                execution["action_kind"] = "apply_patch"
                execution["exit_code"] = None
                execution["filesystem_checkpoint"][
                    "action_kind"
                ] = "apply_patch"

            rewrite_chain_record(fixture, 0, mutate)

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)

    def test_action_text_does_not_override_emitted_memory_events(self):
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
                            'shell_command {"command":"printf \'objective: improve '
                            "validation\\nmeasured_validation_or_failure: validation_mae=1.0"
                            "\\nconclusion: update the model\\ncode_path: train.py"
                            "\\nnext_action: edit train.py before rerunning\\n' # "
                            '.agent_memory/OPENMLE_CONTINUATION.md","workdir":".",'
                            '"timeout_ms":20000}'
                        )
                    elif case == "commented_execute":
                        action = (
                            'shell_command {"command":"python other.py # train.py",'
                            '"workdir":".","timeout_ms":20000}'
                        )
                    elif case == "subpath_read":
                        action = (
                            'shell_command {"command":"cat nested/.agent_memory/'
                            'OPENMLE_CONTINUATION.md","workdir":".",'
                            '"timeout_ms":20000}'
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
                        record["wrapper_evidence"]["workspace_changed_paths"] = [
                            "nested/train.py"
                        ]
                    else:
                        action = (
                            'shell_command {"command":"python nested/train.py",'
                            '"workdir":".","timeout_ms":20000}'
                        )
                    record["action"] = action
                    record["action_submission"]["raw_policy_output"] = action

                rewrite_chain_record(fixture, order, mutate)
                verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)
                self.assertEqual(verdict["status"], "pass", verdict)

    def test_document_path_text_is_not_a_shared_evidence_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))

            def rewrite_path(record: dict) -> None:
                record["action"] = record["action"].replace("train.py", "dummy.py")
                record["action_submission"]["raw_policy_output"] = record[
                    "action_submission"
                ]["raw_policy_output"].replace("train.py", "dummy.py")
                execution = record["env_info_after"]["execution"]
                if isinstance(execution.get("stdout"), str):
                    execution["stdout"] = execution["stdout"].replace(
                        "train.py", "dummy.py"
                    )
                if isinstance(execution.get("changed_paths"), list):
                    execution["changed_paths"] = [
                        path.replace("train.py", "dummy.py")
                        for path in execution["changed_paths"]
                    ]
                evidence = record.get("wrapper_evidence", {})
                if isinstance(evidence.get("workspace_changed_paths"), list):
                    evidence["workspace_changed_paths"] = [
                        path.replace("train.py", "dummy.py")
                        for path in evidence["workspace_changed_paths"]
                    ]

            for order in range(4):
                rewrite_chain_record(fixture, order, rewrite_path)
            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)
            self.assertEqual(verdict["status"], "pass", verdict)

    def test_trusted_assistant_framing_is_accepted_without_substring_heuristics(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))

            def mutate(record: dict) -> None:
                successor = record["context_transition"]["messages"]
                marker = successor[-1]["content"].split(
                    "\n\nEarlier conversation was removed", 1
                )[1]
                framing = [
                    {"role": "system", "content": "task framing"},
                    {
                        "role": "assistant",
                        "content": record["control_request"],
                    },
                    {"role": "user", "content": "task observation"},
                ]
                record["wrapper_evidence"][
                    "checkpoint_framing_sha256"
                ] = messages_sha256(framing)
                record["context_transition"]["messages"] = [
                    framing[0],
                    framing[1],
                    {
                        "role": "user",
                        "content": (
                            framing[2]["content"]
                            + "\n\nEarlier conversation was removed"
                            + marker
                        ),
                    },
                ]

            rewrite_chain_record(fixture, 0, mutate)
            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)
            self.assertEqual(verdict["status"], "pass", verdict)

    def test_malformed_checkpoint_receipt_or_counter_drift_breaks_chain(self):
        cases = (
            "legacy_path",
            "missing_receipt_key",
            "unchanged_receipt",
            "boolean_size",
            "native_call_counter",
            "context_epoch_counter",
            "missing_successor_digest",
            "missing_preservation_field",
            "missing_read_requirement",
            "missing_endpoint_receipt",
            "endpoint_boolean_action_completed",
            "endpoint_float_size",
            "endpoint_null_copy",
            "successor_action_leak",
            "successor_marker_only_after_user",
            "successor_partial_checkpoint_leak",
            "missing_framing_digest",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = self.build(Path(directory))

                def mutate(record: dict) -> None:
                    receipt = record["wrapper_evidence"]["checkpoint_receipt"]
                    if case == "legacy_path":
                        receipt["path"] = ".agent_memory/OPENMLE_CONTINUATION.md"
                    elif case == "missing_receipt_key":
                        receipt.pop("regular_file")
                    elif case == "unchanged_receipt":
                        receipt["changed"] = False
                    elif case == "boolean_size":
                        receipt["size_bytes"] = True
                    elif case == "native_call_counter":
                        record["native_call_count_after"] = record[
                            "native_call_count_before"
                        ]
                    elif case == "context_epoch_counter":
                        record["context_epoch_after"] = record[
                            "context_epoch_before"
                        ]
                    elif case == "missing_preservation_field":
                        record["wrapper_evidence"].pop("preserved_policy_output")
                    elif case == "missing_read_requirement":
                        record["wrapper_evidence"].pop(
                            "checkpoint_read_required_after"
                        )
                    elif case == "missing_endpoint_receipt":
                        record["env_info_after"]["execution"].pop(
                            "filesystem_checkpoint"
                        )
                    elif case == "endpoint_boolean_action_completed":
                        record["env_info_after"]["execution"][
                            "filesystem_checkpoint"
                        ]["action_completed"] = 1
                    elif case == "endpoint_float_size":
                        record["env_info_after"]["execution"][
                            "filesystem_checkpoint"
                        ]["size_bytes"] = float(receipt["size_bytes"])
                    elif case == "endpoint_null_copy":
                        record["env_info_after"][
                            "filesystem_checkpoint"
                        ] = dict(receipt)
                        record["env_info_after"]["execution"][
                            "filesystem_checkpoint"
                        ] = None
                    elif case == "successor_action_leak":
                        record["context_transition"]["messages"][-1]["content"] += (
                            "\n" + record["action"]
                        )
                    elif case == "successor_marker_only_after_user":
                        messages = record["context_transition"]["messages"]
                        content = messages[-1]["content"]
                        marker = content.split(
                            "\n\nEarlier conversation was removed", 1
                        )[1]
                        messages[-1]["content"] = content.split(
                            "\n\nEarlier conversation was removed", 1
                        )[0]
                        messages.append(
                            {
                                "role": "user",
                                "content": "Earlier conversation was removed" + marker,
                            }
                        )
                    elif case == "successor_partial_checkpoint_leak":
                        marker = "Earlier conversation was removed"
                        content = record["context_transition"]["messages"][-1][
                            "content"
                        ]
                        record["context_transition"]["messages"][-1]["content"] = (
                            content.replace(
                                marker,
                                "objective: improve validation\n" + marker,
                                1,
                            )
                        )
                    elif case == "missing_framing_digest":
                        record["wrapper_evidence"].pop(
                            "checkpoint_framing_sha256"
                        )
                    else:
                        for message in record["context_transition"]["messages"]:
                            message["content"] = message["content"].replace(
                                receipt["sha256"], "digest-omitted"
                            )

                rewrite_chain_record(fixture, 0, mutate)
                self.assert_failed(
                    fixture["run_dir"],
                    contains="policy-authored external-document chain",
                )

    def test_malformed_read_receipt_or_counter_drift_breaks_chain(self):
        cases = (
            "legacy_path",
            "not_observed",
            "missing_key",
            "policy_counter",
            "epoch",
            "missing_endpoint_receipt",
            "endpoint_boolean_observed",
            "endpoint_float_size",
            "endpoint_null_copy",
            "different_checkpoint_digest",
            "missing_read_requirement",
            "wrong_expected_digest",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = self.build(Path(directory))

                def mutate(record: dict) -> None:
                    receipt = record["wrapper_evidence"][
                        "filesystem_checkpoint_read"
                    ]
                    if case == "legacy_path":
                        receipt["path"] = ".agent_memory/OPENMLE_CONTINUATION.md"
                    elif case == "not_observed":
                        receipt["observed"] = False
                    elif case == "missing_key":
                        receipt.pop("sha256")
                    elif case == "policy_counter":
                        record["policy_step_after"] = record["policy_step_before"]
                    elif case == "missing_endpoint_receipt":
                        record["env_info_after"]["execution"].pop(
                            "filesystem_checkpoint_read"
                        )
                    elif case == "endpoint_boolean_observed":
                        record["env_info_after"]["execution"][
                            "filesystem_checkpoint_read"
                        ]["observed"] = 1
                    elif case == "endpoint_float_size":
                        record["env_info_after"]["execution"][
                            "filesystem_checkpoint_read"
                        ]["size_bytes"] = float(receipt["size_bytes"])
                    elif case == "endpoint_null_copy":
                        record["env_info_after"][
                            "filesystem_checkpoint_read"
                        ] = dict(receipt)
                        record["env_info_after"]["execution"][
                            "filesystem_checkpoint_read"
                        ] = None
                    elif case == "different_checkpoint_digest":
                        receipt["sha256"] = "c" * 64
                        record["env_info_after"]["execution"][
                            "filesystem_checkpoint_read"
                        ]["sha256"] = "c" * 64
                    elif case == "missing_read_requirement":
                        record["wrapper_evidence"].pop(
                            "checkpoint_read_required"
                        )
                    elif case == "wrong_expected_digest":
                        record["wrapper_evidence"][
                            "checkpoint_read_expected_sha256"
                        ] = "d" * 64
                    else:
                        record["context_epoch_after"] = (
                            record["context_epoch_before"] + 1
                        )

                rewrite_chain_record(fixture, 1, mutate)
                self.assert_failed(
                    fixture["run_dir"],
                    contains="policy-authored external-document chain",
                )

    def test_nested_workspace_execution_receipt_completes_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            rewrite_chain_record(
                fixture,
                3,
                lambda record: record.update(
                    env_info_after={
                        "wrapper_evidence": {"workspace_action_completed": True}
                    }
                ),
            )
            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)
            self.assertEqual(verdict["status"], "pass", verdict)

    def test_missing_read_or_unsuccessful_execute_event_breaks_the_chain(self):
        cases = (
            "missing_read_event",
            "missing_execute_outcome",
            "failed_execute_outcome",
            "contradictory_action_status",
            "contradictory_completion_counter",
            "contradictory_execution_status",
            "contradictory_exit_code",
            "contradictory_execution_delta",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = self.build(Path(directory))
                order = 1 if case == "missing_read_event" else 3

                def mutate(record: dict) -> None:
                    if case == "missing_read_event":
                        record["wrapper_evidence"].pop("memory_event", None)
                    elif case == "missing_execute_outcome":
                        record["wrapper_evidence"].pop("outcome", None)
                    elif case == "failed_execute_outcome":
                        record["wrapper_evidence"]["outcome"] = "failed"
                    elif case == "contradictory_action_status":
                        record["env_info_after"]["action_status"] = "failed"
                    elif case == "contradictory_completion_counter":
                        record["env_info_after"]["counter_delta"][
                            "execution_completed_count"
                        ] = 0
                    elif case == "contradictory_execution_status":
                        record["env_info_after"]["execution"]["status"] = "failed"
                    elif case == "contradictory_exit_code":
                        record["env_info_after"]["execution"]["exit_code"] = 1
                    else:
                        record["env_info_after"]["execution"][
                            "execution_completed_delta"
                        ] = 0

                rewrite_chain_record(fixture, order, mutate)
                self.assert_failed(
                    fixture["run_dir"],
                    contains="policy-authored external-document chain",
                )

    def test_minimal_task_neutral_execute_evidence_completes_the_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            rewrite_chain_record(
                fixture,
                3,
                lambda record: record.update(
                    env_info_after={"shell_action_succeeded": True}
                ),
            )

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)

    def test_plural_memory_events_cannot_forge_a_complete_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build(Path(directory))
            path = next(fixture["rollout_dir"].glob("*.jsonl"))
            documents = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
            chain_uids = {
                json.loads(document["step_record_json"])["trajectory_uid"]
                for document in documents
                if json.loads(document["step_record_json"])
                .get("wrapper_evidence", {})
                .get("event")
                == "context_compaction"
            }
            self.assertEqual(len(chain_uids), 1)
            chain_uid = chain_uids.pop()
            for document in documents:
                record = json.loads(document["step_record_json"])
                if record.get("trajectory_uid") == chain_uid:
                    record["wrapper_evidence"] = {}
                    if record.get("trajectory_row_order") == 0:
                        record["wrapper_evidence"]["memory_events"] = [
                            "write",
                            "compaction",
                            "read",
                            "modify",
                            "execute",
                        ]
                    document["step_record_json"] = json.dumps(record, sort_keys=True)
            path.write_text(
                "\n".join(json.dumps(document, sort_keys=True) for document in documents)
                + "\n",
                encoding="utf-8",
            )

            self.assert_failed(
                fixture["run_dir"],
                contains="policy-authored external-document chain",
            )


class TestMultitaskFinalizer(FinalizerTestCase):
    def build_multitask(
        self,
        root: Path,
        *,
        updates: int = 8,
        route_counts_by_update: list[dict[str, int]] | None = None,
    ) -> dict:
        return build_valid_multitask_run(
            root / "run",
            updates=updates,
            route_counts_by_update=route_counts_by_update,
        )

    def test_four_opaque_routes_pass_with_exact_owner_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)
            self.assertEqual(
                verdict["launch_receipt_schema"],
                "amg_verl_fully_async_multitask_launch_receipt_v1",
            )
            self.assertEqual(set(verdict["routes"]), set(MULTITASK_ROUTES))
            self.assertEqual(verdict["rolling_8_episode_share"]["status"], "pass")
            self.assertEqual(
                verdict["final_accounting"]["optimizer_consumed"]["episodes"],
                8 * 64,
            )
            for route_id in MULTITASK_ROUTES:
                route = verdict["routes"][route_id]
                self.assertEqual(route["optimizer_consumed_episodes"], 128)
                self.assertEqual(route["native_successes"], 128)
                self.assertEqual(route["document_writes"], 8)
                self.assertEqual(route["compactions"], 8)
                self.assertEqual(route["document_reads"], 8)
                self.assertEqual(route["memory_reuses_or_modifications"], 8)
                self.assertEqual(route["executions"], 8)
                self.assertEqual(route["complete_memory_chains"], 8)

    def test_route_local_max_rounds_horizon_is_a_complete_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_valid_multitask_run(
                Path(directory) / "run",
                updates=8,
                horizon_route_id=MULTITASK_ROUTES[0],
            )

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)
            self.assertEqual(verdict["counts"]["completed_episodes"], 512)
            self.assertEqual(
                verdict["routes"][MULTITASK_ROUTES[0]]["native_successes"],
                120,
            )

    def test_false_done_without_exact_max_rounds_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_valid_multitask_run(
                Path(directory) / "run",
                updates=8,
                horizon_route_id=MULTITASK_ROUTES[0],
            )
            path = fixture["rollout_dir"] / "1.jsonl"
            documents = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            for document in documents:
                record = json.loads(document["step_record_json"])
                if (
                    record.get("route_id") == MULTITASK_ROUTES[0]
                    and record.get("trajectory_terminal") is True
                    and record.get("rollout_done_flag") is False
                ):
                    record["outcome"] = "continue"
                    document["step_record_json"] = json.dumps(record, sort_keys=True)
                    break
            else:
                self.fail("fixture omitted the route-local horizon terminal row")
            path.write_text(
                "\n".join(
                    json.dumps(document, sort_keys=True) for document in documents
                )
                + "\n",
                encoding="utf-8",
            )

            self.assert_failed(
                fixture["run_dir"],
                contains="terminal row is not done or a valid max_rounds horizon",
            )

    def test_multitask_runtime_artifacts_follow_receipt_declared_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            launch = json.loads(fixture["launch_path"].read_text(encoding="utf-8"))
            replacements = {
                "file_logger": fixture["run_dir"] / "telemetry" / "metrics.jsonl",
                "rollout_data": fixture["run_dir"] / "telemetry" / "episodes",
            }
            for field, replacement in replacements.items():
                replacement.parent.mkdir(parents=True, exist_ok=True)
                Path(launch["runtime_artifacts"][field]).rename(replacement)
                launch["runtime_artifacts"][field] = str(replacement)
            fixture["launch_path"].write_text(
                json.dumps(launch, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)

    def test_duplicate_file_logger_route_metric_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            rows = [
                json.loads(line)
                for line in fixture["metrics_path"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            key = (
                "fully_async/sum/optimizer_consumed_episodes/data_source/"
                + MULTITASK_ROUTES[0]
            )
            rows.insert(1, {"step": 1, "data": {key: rows[0]["data"][key]}})
            fixture["metrics_path"].write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )

            self.assert_failed(
                fixture["run_dir"],
                contains="optimizer-consumed episodes route totals mismatch",
            )

    def test_receipt_schema_identity_digest_and_budget_drift_fail_closed(self):
        cases: tuple[tuple[str, JsonMutation], ...] = (
            (
                "unknown receipt schema",
                lambda receipt: receipt.update(schema="unknown-receipt"),
            ),
            (
                "legacy schema cannot reinterpret multitask fields",
                lambda receipt: receipt.update(
                    schema="amg_verl_fully_async_launch_receipt_v5"
                ),
            ),
            (
                "veRL identity",
                lambda receipt: receipt["source"].update(verl_commit="0" * 40),
            ),
            (
                "outer identity",
                lambda receipt: receipt["source"].update(outer_commit="e" * 40),
            ),
            (
                "inner identity",
                lambda receipt: receipt["source"].update(agentgym_commit="e" * 40),
            ),
            (
                "dirty outer source",
                lambda receipt: receipt["source"].update(
                    outer_diff_paths=["uncommitted.py"]
                ),
            ),
            (
                "route identity",
                lambda receipt: receipt["launch_identity"].update(
                    route_ids=list(MULTITASK_ROUTES[:3])
                ),
            ),
            (
                "registry digest",
                lambda receipt: receipt["inputs"].update(
                    route_registry_sha256="0" * 64
                ),
            ),
            (
                "schedule digest",
                lambda receipt: receipt["schedule"].update(sha256="0" * 64),
            ),
            (
                "optimizer budget",
                lambda receipt: receipt["budget_contract"].update(episodes=511),
            ),
        )
        for label, mutation in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                fixture = self.build_multitask(Path(directory))
                mutate_json(fixture["launch_path"], mutation)
                self.assert_failed(fixture["run_dir"])

    def test_receipt_bound_artifact_digests_and_runtime_paths_fail_on_drift(self):
        for artifact in ("source_lock", "schedule_certificate", "route_registry"):
            with (
                self.subTest(artifact=artifact),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build_multitask(Path(directory))
                identity = json.loads(
                    fixture["launch_path"].read_text(encoding="utf-8")
                )["launch_identity"]
                path = Path(identity[f"{artifact}_path"])
                path.write_text(
                    path.read_text(encoding="utf-8") + " ", encoding="utf-8"
                )
                self.assert_failed(fixture["run_dir"])

    def test_invalid_finalization_path_cannot_overwrite_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            metrics_before = fixture["metrics_path"].read_bytes()
            mutate_json(
                fixture["launch_path"],
                lambda receipt: receipt["runtime_artifacts"].update(
                    finalization=str(fixture["metrics_path"])
                ),
            )

            verdict = self.assert_failed(
                fixture["run_dir"], contains="paths must be distinct"
            )

            self.assertEqual(fixture["metrics_path"].read_bytes(), metrics_before)
            written = json.loads(
                (fixture["run_dir"] / "finalization.json").read_text(encoding="utf-8")
            )
            self.assertEqual(written, verdict)

        runtime_fields = (
            "file_logger",
            "rollout_data",
            "hydra_config",
            "checkpoints",
            "finalization",
            "trainer_log",
        )
        for field in runtime_fields:
            with (
                self.subTest(runtime_path=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build_multitask(Path(directory))
                mutate_json(
                    fixture["launch_path"],
                    lambda receipt, field=field: receipt["runtime_artifacts"].update(
                        {field: f"/tmp/unbound-{field}"}
                    ),
                )
                self.assert_failed(fixture["run_dir"])

    def test_receipt_cannot_rewrite_the_bound_source_lock_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))

            def mutate(receipt: dict) -> None:
                receipt["launch_identity"]["publication_outer_commit"] = "e" * 40
                receipt["source"]["publication_outer_commit"] = "e" * 40
                receipt["source"]["outer_commit"] = "e" * 40

            mutate_json(fixture["launch_path"], mutate)
            self.assert_failed(fixture["run_dir"], contains="source lock")

    def test_receipt_cannot_ignore_bound_schedule_certificate_content(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            receipt = json.loads(fixture["launch_path"].read_text(encoding="utf-8"))
            identity = receipt["launch_identity"]
            certificate_path = Path(identity["schedule_certificate_path"])
            source_lock_path = Path(identity["source_lock_path"])
            mutate_json(
                certificate_path,
                lambda certificate: certificate.update(optimizer_updates=9),
            )
            certificate_digest = sha256(certificate_path)
            mutate_json(
                source_lock_path,
                lambda source_lock: source_lock["integration"][
                    "schedule_certificate"
                ].update(sha256=certificate_digest),
            )
            identity["schedule_certificate_sha256"] = certificate_digest
            identity["source_lock_sha256"] = sha256(source_lock_path)
            fixture["launch_path"].write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assert_failed(fixture["run_dir"], contains="certificate")

    def test_source_lock_content_is_bound_even_when_its_digest_is_rewritten(self):
        cases = (
            ("schema", lambda lock: lock.update(schema="unknown")),
            ("status", lambda lock: lock.update(status="fail")),
            (
                "outer commit",
                lambda lock: lock["runtime_source"].update(outer_commit="e" * 40),
            ),
            (
                "inner commit",
                lambda lock: lock["runtime_source"].update(inner_commit="e" * 40),
            ),
            (
                "veRL commit",
                lambda lock: lock["runtime_source"].update(verl_commit="e" * 40),
            ),
            (
                "selected files",
                lambda lock: lock["runtime_source"].update(
                    selected_files={"outer:changed.py": "e" * 64}
                ),
            ),
            (
                "training runtime",
                lambda lock: lock["training_runtime"].update(
                    base_model="/models/unbound"
                ),
            ),
            (
                "registry digest",
                lambda lock: lock["integration"]["route_registry"].update(
                    sha256="e" * 64
                ),
            ),
            (
                "registry routes",
                lambda lock: lock["integration"]["route_registry"].update(
                    route_ids=list(reversed(MULTITASK_ROUTES))
                ),
            ),
            (
                "schedule digest",
                lambda lock: lock["integration"]["schedule_certificate"].update(
                    schedule_sha256="e" * 64
                ),
            ),
        )
        for label, mutation in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                fixture = self.build_multitask(Path(directory))
                rewrite_multitask_source_lock(fixture, mutation)
                self.assert_failed(fixture["run_dir"], contains="source lock")

    def test_schedule_certificate_fields_are_bound_to_launch_contract(self):
        cases = (
            ("schema", lambda value: value.update(schema="unknown")),
            (
                "registry digest",
                lambda value: value.update(route_registry_sha256="e" * 64),
            ),
            (
                "schedule digest",
                lambda value: value.update(schedule_sha256="e" * 64),
            ),
            ("manifest digest", lambda value: value.update(spec_sha256="e" * 64)),
            ("role", lambda value: value.update(role="gate_only")),
            ("updates", lambda value: value.update(optimizer_updates=9)),
            ("samples/update", lambda value: value.update(samples_per_update=32)),
            ("row count", lambda value: value.update(row_count=511)),
            (
                "route order",
                lambda value: value.update(
                    route_order=list(reversed(MULTITASK_ROUTES))
                ),
            ),
            (
                "per-route rows",
                lambda value: value["per_route_rows"].update(
                    {MULTITASK_ROUTES[0]: 129}
                ),
            ),
        )
        for label, mutation in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                fixture = self.build_multitask(Path(directory))
                rewrite_multitask_certificate(fixture, mutation)
                self.assert_failed(fixture["run_dir"], contains="certificate")

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            mutate_json(
                fixture["launch_path"],
                lambda receipt: receipt["launch_identity"][
                    "formal_schedule_contract"
                ].update(panel_id="unbound"),
            )
            self.assert_failed(fixture["run_dir"], contains="certificate")

    def test_final_statistics_requires_one_exact_owner_snapshot(self):
        for case in ("missing", "duplicate", "malformed", "wrong_schema"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = self.build_multitask(Path(directory))
                path = fixture["trainer_log"]
                if case == "missing":
                    path.write_text("runtime output\n", encoding="utf-8")
                elif case == "duplicate":
                    line = path.read_text(encoding="utf-8").splitlines()[-1]
                    path.write_text(
                        path.read_text(encoding="utf-8") + line + "\n",
                        encoding="utf-8",
                    )
                elif case == "malformed":
                    path.write_text(
                        "[FullyAsyncTaskRunner][FinalStatistics] {\n",
                        encoding="utf-8",
                    )
                else:
                    mutate_final_statistics(
                        path, lambda value: value.update(schema="unknown")
                    )
                self.assert_failed(fixture["run_dir"], contains="FinalStatistics")

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            mutate_final_statistics(
                fixture["trainer_log"],
                lambda value: value.update(queue_cleanup={"status": "failed"}),
            )
            self.assert_failed(fixture["run_dir"], contains="cleanup")

    def test_final_statistics_accepts_only_direct_or_ansi_ray_log_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            path = fixture["trainer_log"]
            marker_line = path.read_text(encoding="utf-8").splitlines()[-1]
            path.write_text(
                "runtime output\n"
                "\x1b[36m(FullyAsyncTaskRunner pid=17, ip=127.0.0.1)\x1b[0m "
                + marker_line
                + "\n",
                encoding="utf-8",
            )

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            path = fixture["trainer_log"]
            marker_line = path.read_text(encoding="utf-8").splitlines()[-1]
            path.write_text(
                "runtime output\nINFO embedded " + marker_line + "\n",
                encoding="utf-8",
            )

            self.assert_failed(fixture["run_dir"], contains="FinalStatistics")

    def test_final_statistics_owner_shapes_are_exact(self):
        cases: tuple[tuple[str, JsonMutation], ...] = (
            ("top extra", lambda value: value.update(unexpected={})),
            ("top missing", lambda value: value.pop("queue_cleanup")),
            (
                "queue extra",
                lambda value: value["queue"].update(unexpected=0),
            ),
            (
                "queue missing",
                lambda value: value["queue"].pop("max_queue_size"),
            ),
            (
                "rollouter extra",
                lambda value: value["rollouter"].update(unexpected=0),
            ),
            (
                "rollouter missing",
                lambda value: value["rollouter"].pop("count/staleness_samples"),
            ),
            (
                "trainer extra",
                lambda value: value["trainer"].update(unexpected=0),
            ),
            (
                "trainer missing",
                lambda value: value["trainer"].pop("current_param_version"),
            ),
        )
        for label, mutation in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                fixture = self.build_multitask(Path(directory))
                mutate_final_statistics(fixture["trainer_log"], mutation)
                self.assert_failed(fixture["run_dir"], contains="FinalStatistics")

    def test_final_statistics_counters_reject_integral_floats(self):
        route_id = MULTITASK_ROUTES[0]
        cases: tuple[tuple[str, JsonMutation], ...] = (
            (
                "queue total",
                lambda value: value["queue"].update(total_produced=512.0),
            ),
            (
                "queue route",
                lambda value: value["queue"]["enqueued_by_data_source"].update(
                    {route_id: 128.0}
                ),
            ),
            (
                "rollouter total",
                lambda value: value["rollouter"].update(
                    {"count/total_generated_samples": 512.0}
                ),
            ),
            (
                "rollouter route",
                lambda value: value["rollouter"].update(
                    {f"count/rollout_completed/data_source/{route_id}": 128.0}
                ),
            ),
            (
                "trainer total",
                lambda value: value["trainer"].update(
                    optimizer_consumed_episodes=512.0
                ),
            ),
            (
                "trainer route",
                lambda value: value["trainer"][
                    "optimizer_consumed_episodes_by_data_source"
                ].update({route_id: 128.0}),
            ),
        )
        for label, mutation in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                fixture = self.build_multitask(Path(directory))
                mutate_final_statistics(fixture["trainer_log"], mutation)
                self.assert_failed(fixture["run_dir"], contains="FinalStatistics")

    def test_final_statistics_staleness_threshold_requires_a_json_number(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            mutate_final_statistics(
                fixture["trainer_log"],
                lambda value: value["rollouter"].update(
                    {"static/staleness_threshold": "0.1"}
                ),
            )

            self.assert_failed(
                fixture["run_dir"],
                contains="staleness_threshold is invalid",
            )

    def test_conserved_nonzero_cleanup_cleared_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            route_id = MULTITASK_ROUTES[0]

            def add_cleanup_surplus(value: dict) -> None:
                queue = value["queue"]
                queue["total_produced"] += 1
                queue["total_cleared"] = 1
                queue["enqueued_by_data_source"][route_id] += 1
                queue["cleared_by_data_source"][route_id] = 1

                rollouter = value["rollouter"]
                for field in (
                    "count/total_generated_samples",
                    "count/rollout_dispatched_samples",
                    "count/rollout_completed_samples",
                    "count/queue_enqueued_samples",
                ):
                    rollouter[field] += 1
                rollouter["count/queue_cleared_samples"] = 1
                for event in (
                    "rollout_dispatched",
                    "rollout_completed",
                    "queue_enqueued",
                ):
                    rollouter[f"count/{event}/data_source/{route_id}"] += 1
                rollouter[f"count/queue_cleared/data_source/{route_id}"] = 1

            mutate_final_statistics(fixture["trainer_log"], add_cleanup_surplus)

            verdict = finalize_run(fixture["run_dir"], trainer_exit_code=0)

            self.assertEqual(verdict["status"], "pass", verdict)
            self.assertEqual(
                verdict["final_accounting"]["queue"]["cleanup_cleared"], 1
            )
            self.assertEqual(
                verdict["final_accounting"]["queue_by_route"]["cleanup_cleared"][
                    route_id
                ],
                1,
            )

    def test_every_global_final_statistics_counter_is_checked(self):
        fields = (
            ("queue", "total_produced"),
            ("queue", "total_consumed"),
            ("queue", "dropped_samples"),
            ("queue", "total_cleared"),
            ("queue", "queue_size"),
            ("rollouter", "count/rollout_dispatched_samples"),
            ("rollouter", "count/rollout_inflight_samples"),
            ("rollouter", "count/rollout_completed_samples"),
            ("rollouter", "count/rollout_failed_samples"),
            ("rollouter", "count/rollout_cancelled_samples"),
            ("rollouter", "count/queue_enqueued_samples"),
            ("rollouter", "count/queue_dequeued_samples"),
            ("rollouter", "count/queue_overflow_evictions"),
            ("rollouter", "count/queue_cleared_samples"),
            ("rollouter", "count/queue_resident_samples"),
            ("rollouter", "monitor/active_tasks_size"),
            ("rollouter", "monitor/queue/pending_queue_size"),
            ("rollouter", "monitor/queue/mq_queue_size"),
            ("rollouter", "count/dropped_stale_samples"),
            ("trainer", "optimizer_consumed_episodes"),
            ("trainer", "optimizer_consumed_action_rows"),
            ("trainer", "optimizer_consumed_policy_response_tokens"),
            ("trainer", "stale_action_rows"),
            ("trainer", "current_param_version"),
        )
        for owner, field in fields:
            with (
                self.subTest(owner=owner, field=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build_multitask(Path(directory))

                def mutate(value: dict, owner: str = owner, field: str = field) -> None:
                    value[owner][field] += 1

                mutate_final_statistics(fixture["trainer_log"], mutate)
                self.assert_failed(fixture["run_dir"])

    def test_every_per_route_final_statistics_counter_is_checked(self):
        queue_fields = (
            "enqueued_by_data_source",
            "consumed_by_data_source",
            "evicted_by_data_source",
            "cleared_by_data_source",
            "resident_by_data_source",
        )
        rollouter_events = (
            "rollout_dispatched",
            "rollout_inflight",
            "rollout_completed",
            "rollout_failed",
            "rollout_cancelled",
            "queue_enqueued",
            "queue_dequeued",
            "queue_overflow_evicted",
            "queue_cleared",
            "queue_resident",
        )
        trainer_fields = (
            "optimizer_consumed_episodes_by_data_source",
            "optimizer_consumed_action_rows_by_data_source",
            "optimizer_consumed_policy_response_tokens_by_data_source",
            "stale_action_rows_by_data_source",
        )
        route_id = MULTITASK_ROUTES[0]
        cases = [
            (
                f"queue.{field}",
                lambda value, field=field: value["queue"][field].__setitem__(
                    route_id, value["queue"][field].get(route_id, 0) + 1
                ),
            )
            for field in queue_fields
        ]
        cases.extend(
            (
                f"rollouter.{event}",
                lambda value, event=event: value["rollouter"].__setitem__(
                    f"count/{event}/data_source/{route_id}",
                    value["rollouter"].get(f"count/{event}/data_source/{route_id}", 0)
                    + 1,
                ),
            )
            for event in rollouter_events
        )
        cases.extend(
            (
                f"trainer.{field}",
                lambda value, field=field: value["trainer"][field].__setitem__(
                    route_id, value["trainer"][field].get(route_id, 0) + 1
                ),
            )
            for field in trainer_fields
        )
        for label, mutation in cases:
            with (
                self.subTest(counter=label),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build_multitask(Path(directory))
                mutate_final_statistics(fixture["trainer_log"], mutation)
                self.assert_failed(fixture["run_dir"])

    def test_cross_owner_route_shift_fails_even_when_each_owner_conserves(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            left, right = MULTITASK_ROUTES[:2]

            def mutate(value: dict) -> None:
                for field in (
                    "enqueued_by_data_source",
                    "consumed_by_data_source",
                ):
                    value["queue"][field][left] -= 1
                    value["queue"][field][right] += 1
                for event in ("queue_enqueued", "queue_dequeued"):
                    left_key = f"count/{event}/data_source/{left}"
                    right_key = f"count/{event}/data_source/{right}"
                    value["rollouter"][left_key] -= 1
                    value["rollouter"][right_key] += 1

            mutate_final_statistics(fixture["trainer_log"], mutate)
            self.assert_failed(fixture["run_dir"], contains="route")

    def test_missing_padding_label_and_synthetic_padding_both_fail(self):
        for case in ("missing", "synthetic"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = self.build_multitask(Path(directory))
                if case == "missing":
                    rewrite_first_rollout(
                        fixture, lambda document: document.pop("is_padding")
                    )
                else:
                    rewrite_first_rollout(
                        fixture, lambda document: document.update(is_padding=True)
                    )
                self.assert_failed(fixture["run_dir"], contains="padding")

    def test_rollout_filename_is_bound_to_optimizer_update(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(Path(directory))
            first = fixture["rollout_dir"] / "1.jsonl"
            first.rename(fixture["rollout_dir"] / "01.jsonl")
            self.assert_failed(fixture["run_dir"], contains="filename")

    def test_rolling_eight_route_share_rejects_a_skewed_window(self):
        balanced_skew = [
            {
                MULTITASK_ROUTES[0]: 10,
                MULTITASK_ROUTES[1]: 18,
                MULTITASK_ROUTES[2]: 18,
                MULTITASK_ROUTES[3]: 18,
            }
            for _ in range(8)
        ]
        balanced_skew.append(
            {
                MULTITASK_ROUTES[0]: 64,
                MULTITASK_ROUTES[1]: 0,
                MULTITASK_ROUTES[2]: 0,
                MULTITASK_ROUTES[3]: 0,
            }
        )
        balanced_skew.extend(
            [{route_id: 16 for route_id in MULTITASK_ROUTES} for _ in range(11)]
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.build_multitask(
                Path(directory),
                updates=20,
                route_counts_by_update=balanced_skew,
            )
            verdict = self.assert_failed(fixture["run_dir"], contains="rolling-8")
            self.assertEqual(
                verdict["rolling_8_episode_share"]["windows"][0]["status"],
                "fail",
            )


class TestFinalizerConfigAndCheckpoint(FinalizerTestCase):
    def test_six_trainer_checkpoint_world_size_pass(self):
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

            launch = json.loads(fixture["launch_path"].read_text(encoding="utf-8"))
            launch["budget"] = verify_resolved_config(
                config,
                mode="gate",
                expected_budget=launch["budget_contract"],
            )
            launch["resolved_config"]["sha256"] = sha256(fixture["resolved_path"])
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

    def test_checkpoint_tracker_and_native_artifact_binding_fail_on_drift(self):
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
                    file_logger="/tmp/caller-selected-metrics.jsonl"
                ),
            )
            self.assert_failed(
                fixture["run_dir"], contains="runtime artifact file_logger"
            )

    def test_legacy_runtime_artifacts_reject_in_tree_shadow_paths(self):
        cases = (
            ("file_logger", "metrics-shadow.jsonl", shutil.copy2),
            ("rollout_data", "rollout-data-shadow", shutil.copytree),
        )
        for field, relative_path, copy in cases:
            with (
                self.subTest(runtime_path=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = self.build(Path(directory))
                launch = json.loads(
                    fixture["launch_path"].read_text(encoding="utf-8")
                )
                original = Path(launch["runtime_artifacts"][field])
                shadow = fixture["run_dir"] / relative_path
                copy(original, shadow)
                launch["runtime_artifacts"][field] = str(shadow)
                fixture["launch_path"].write_text(
                    json.dumps(launch, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                self.assert_failed(
                    fixture["run_dir"],
                    contains=f"legacy runtime artifact {field}",
                )


if __name__ == "__main__":
    unittest.main()
