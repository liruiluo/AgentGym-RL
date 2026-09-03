from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agentmemorygym_verl.heldout_eval_contract import (
    ACTION_ROW_SCHEMA,
    AGENT_NAME,
    BATCH_SCHEMA,
    CANONICAL_ROUTES,
    RUN_SCHEMA,
    aggregate_episode_metrics,
    commit_batch,
    compose_heldout_schedule,
    finalize_run_metrics,
    initialize_run_contract,
    inspect_heldout_schedule,
    inspect_resume_state,
    materialize_generated_batch,
    native_success_metric,
    pad_batch_rows,
    sha256_file,
    verify_batch_directory,
    verify_swesmith_formal_eval_authority,
)
from agentmemorygym_verl.heldout_eval import (
    MODEL_MANIFEST_SCHEMA,
    derive_eval_config,
    load_eval_plan,
    run_contract,
    run_evaluation,
    verify_runtime_dataset_row,
)


ROUTE_MAX_ROUNDS = {route_id: 30 for route_id in CANONICAL_ROUTES}
NATIVE_SOURCE_IDENTITY_SCHEMA = "camg_native_episode_source_identity_v1"
TEST_ROUTE_COUNTS = {
    "webshop": 2,
    "swesmith": 23,
    "literesearcher": 3,
    "openmle_fast": 2,
}
TEST_EPISODE_COUNT = sum(TEST_ROUTE_COUNTS.values())


def _source_extra(route_id: str, position: int, data_idx: int) -> dict:
    if route_id == "webshop":
        return {
            "index": data_idx,
            "route_id": route_id,
            "scenario_id": f"scenario-{position}",
            "orbit_index": 500 + position,
        }
    if route_id == "swesmith":
        return {
            "index": data_idx,
            "route_id": route_id,
            "instance_id": f"repository-{position}.issue-{position}",
            "base_repository": f"repository-{position}",
        }
    if route_id == "literesearcher":
        return {
            "index": data_idx,
            "route_id": route_id,
            "row_identity": f"{position + 1:064x}",
            "source_pool_index": 10_000 + position,
        }
    if route_id == "openmle_fast":
        return {
            "index": data_idx,
            "route_id": route_id,
            "task_id": f"competition@{position}",
            "source_family": f"KAGGLE_DATASET:owner/dataset-{position}",
            "role": "heldout",
            "manifest_sha256": "c" * 64,
        }
    raise AssertionError(route_id)


def _episode_source_identity(input_row: dict) -> dict:
    route_id = input_row["route_id"]
    source = input_row["extra_info"]["source_extra_info"]
    identity = {
        "schema": NATIVE_SOURCE_IDENTITY_SCHEMA,
        "route_id": route_id,
        "data_idx": input_row["data_idx"],
    }
    if route_id == "webshop":
        identity.update(
            scenario_id=source["scenario_id"], orbit_index=source["orbit_index"]
        )
    elif route_id == "swesmith":
        identity.update(
            instance_id=source["instance_id"],
            base_repository=source["base_repository"],
        )
    elif route_id == "literesearcher":
        identity.update(
            row_identity=source["row_identity"],
            source_pool_index=source["source_pool_index"],
        )
    elif route_id == "openmle_fast":
        identity.update(
            task_id=source["task_id"],
            source_family=source["source_family"],
            manifest_role=source["role"],
            manifest_sha256=source["manifest_sha256"],
        )
    else:
        raise AssertionError(route_id)
    return identity


def _run_contract_fixture():
    return {
        "schema": RUN_SCHEMA,
        "checkpoint_step": 200,
        "route_registry": {"max_rounds": ROUTE_MAX_ROUNDS},
    }


def _json_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value) -> str:
    path.write_bytes(_json_bytes(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows) -> str:
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_spec(
    root: Path,
    counts=(2, 1, 3, 2),
    *,
    route_registry_sha256: str = "a" * 64,
) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    routes = []
    if len(counts) != len(CANONICAL_ROUTES):
        raise ValueError("fixture count vector must cover every canonical route")
    for route_id, count in zip(CANONICAL_ROUTES, counts):
        rows = [
            {
                "item_id": f"{route_id}-source-{position}",
                "data_idx": position if route_id == "swesmith" else 100 + position,
                "index": position if route_id == "swesmith" else 100 + position,
                "route_id": route_id,
                "data_source": route_id,
                "agent_name": AGENT_NAME,
                "extra_info": _source_extra(
                    route_id,
                    position,
                    position if route_id == "swesmith" else 100 + position,
                ),
            }
            for position in range(count)
        ]
        schedule_path = root / f"{route_id}.jsonl"
        schedule_sha256 = _write_jsonl(schedule_path, rows)
        route = {
            "route_id": route_id,
            "schedule": schedule_path.name,
            "schedule_sha256": schedule_sha256,
            "expected_rows": count,
        }
        if route_id == "swesmith":
            active_train = 4 * count
            extension_count = 2 * count
            admitted_count = count + extension_count
            formal_repositories = 1
            extension_repositories = 1
            admitted_repositories = 2
            admission_summary = root / "swesmith-admission-summary.json"
            admission_summary_sha256 = _write_json(
                admission_summary,
                {
                    "schema": "camg_swesmith_complete_admission_summary_v2",
                    "status": "pass",
                    "heldout_evaluation_run": False,
                    "raw_assigned_task_count": 10_381,
                    "classified_task_count": 10_381,
                    "admitted_task_count": admitted_count,
                    "excluded_task_count": 10_381 - admitted_count,
                    "pending_infrastructure_task_count": 0,
                    "raw_heldout_repository_count": 36,
                    "count_cap": None,
                    "top_k": None,
                    "early_stop": False,
                },
            )
            selection = root / "swesmith-formal-eval-selection.json"
            selection_sha256 = _write_json(
                selection,
                {
                    "schema": "camg_swesmith_formal_eval_selection_v5",
                    "status": "frozen",
                    "active_train_task_count": active_train,
                    "active_training_inputs_modified": False,
                    "complete_admitted_heldout_pool_task_count": admitted_count,
                    "complete_admitted_heldout_pool_repository_count": admitted_repositories,
                    "formal_eval_task_count": count,
                    "formal_eval_repository_count": formal_repositories,
                    "formal_eval_fraction": count / (active_train + count),
                    "extension_pool_task_count": extension_count,
                    "extension_pool_repository_count": extension_repositories,
                    "extension_pool_formal_evaluation_role": False,
                    "extension_pool_training_role": False,
                    "selection_depends_on_model_output_or_reward": False,
                    "heldout_evaluation_run": False,
                },
            )
            pool = root / "swesmith-admitted-pool-manifest.json"
            pool_sha256 = _write_json(
                pool,
                {
                    "schema": "camg_swesmith_admitted_heldout_pool_manifest_v5",
                    "status": "complete",
                    "role": "admitted_heldout_candidate_pool",
                    "task_count": admitted_count,
                    "repository_count": admitted_repositories,
                    "formal_evaluation_role": False,
                    "training_role": False,
                    "files": {},
                },
            )
            extension = root / "swesmith-extension-pool-manifest.json"
            extension_sha256 = _write_json(
                extension,
                {
                    "schema": "camg_swesmith_extension_pool_manifest_v5",
                    "status": "frozen",
                    "role": "admitted_heldout_extension_pool",
                    "task_count": extension_count,
                    "repository_count": extension_repositories,
                    "formal_evaluation_role": False,
                    "training_role": False,
                    "files": {},
                },
            )
            heldout_manifest = root / "swesmith-heldout-manifest.json"
            heldout_manifest_sha256 = _write_json(
                heldout_manifest,
                {
                    "schema_version": "swesmith_jsonl_manifest_v1",
                    "role": "formal_heldout",
                    "selection": {
                        "mode": "instance_ids",
                        "count": count,
                        "repository_count": formal_repositories,
                        "source_admitted_pool_count": admitted_count,
                        "source_admitted_pool_repository_count": admitted_repositories,
                        "active_training_inputs_modified": False,
                        "count_cap": None,
                        "top_k": None,
                        "early_stop": False,
                    },
                },
            )
            runtime = root / "swesmith-runtime-manifest.json"
            runtime_sha256 = _write_json(
                runtime,
                {
                    "schema": "camg_swesmith_formal_eval_runtime_manifest_v5",
                    "status": "ready",
                    "task_count": count,
                    "repository_count": formal_repositories,
                    "raw_assigned_task_count": 10_381,
                    "raw_assigned_repository_count": 36,
                    "semantic_exclusion_count": 10_381 - admitted_count,
                    "complete_admitted_pool_task_count": admitted_count,
                    "complete_admitted_pool_repository_count": admitted_repositories,
                    "extension_pool_task_count": extension_count,
                    "extension_pool_repository_count": extension_repositories,
                    "active_training_inputs_modified": False,
                    "selection": (
                        "deterministic complete-repository subset of the "
                        "exact-runtime-admitted held-out candidate pool"
                    ),
                    "files": {
                        "routing": {
                            "path": schedule_path.name,
                            "bytes": schedule_path.stat().st_size,
                            "sha256": schedule_sha256,
                        },
                        "admission_summary": {
                            "path": admission_summary.name,
                            "bytes": admission_summary.stat().st_size,
                            "sha256": admission_summary_sha256,
                        },
                        "formal_eval_selection": {
                            "path": selection.name,
                            "bytes": selection.stat().st_size,
                            "sha256": selection_sha256,
                        },
                        "admitted_pool_manifest": {
                            "path": pool.name,
                            "bytes": pool.stat().st_size,
                            "sha256": pool_sha256,
                        },
                        "extension_pool_manifest": {
                            "path": extension.name,
                            "bytes": extension.stat().st_size,
                            "sha256": extension_sha256,
                        },
                        "manifest": {
                            "path": heldout_manifest.name,
                            "bytes": heldout_manifest.stat().st_size,
                            "sha256": heldout_manifest_sha256,
                        },
                    },
                    "count_cap": None,
                    "top_k": None,
                    "early_stop": False,
                    "heldout_evaluation_run": False,
                },
            )
            route["formal_eval_runtime_manifest"] = runtime.name
            route["formal_eval_runtime_manifest_sha256"] = runtime_sha256
        routes.append(route)
    route_counts = dict(zip(CANONICAL_ROUTES, counts))
    split_contract = root / "complete-split-contract.json"
    split_contract_sha256 = _write_json(
        split_contract,
        {
            "schema": "camg_formal_eval_split_contract_v5",
            "status": "ready",
            "package_id": "fixture-complete-split-v2",
            "global_invariants": {
                "active_training_inputs_modified": False,
                "heldout_evaluation_run": False,
                "aggregate_metric": (
                    "unweighted macro-average of the four environment-level "
                    "primary success metrics"
                ),
            },
            "environments": {
                "shop": {
                    "preparation_status": "ready",
                    "formal_heldout_episode_count": route_counts["webshop"],
                },
                "coding": {
                    "preparation_status": "ready",
                    "formal_eval_tasks": route_counts["swesmith"],
                },
                "deep_research": {
                    "preparation_status": "ready",
                    "formal_heldout_task_count": route_counts["literesearcher"],
                },
                "auto_research": {
                    "preparation_status": "ready",
                    "formal_heldout_task_count": route_counts["openmle_fast"],
                },
            },
        },
    )
    spec = {
        "schema": "camg_heldout_schedule_spec_v2",
        "agent_name": AGENT_NAME,
        "panel_id": "camg-native-heldout-v1",
        "route_registry_sha256": route_registry_sha256,
        "complete_split_contract": split_contract.name,
        "complete_split_contract_sha256": split_contract_sha256,
        "routes": routes,
    }
    path = root / "heldout-spec.json"
    return path, _write_json(path, spec)


def _compose(root: Path, counts=(2, 1, 3, 2)):
    spec_path, spec_sha256 = _make_spec(root, counts)
    schedule = root / "heldout.jsonl"
    manifest = root / "heldout-manifest.json"
    report = compose_heldout_schedule(
        spec_path,
        expected_spec_sha256=spec_sha256,
        output_path=schedule,
        manifest_path=manifest,
    )
    rows = inspect_heldout_schedule(
        schedule,
        expected_sha256=report["schedule_sha256"],
        expected_count=sum(counts),
    )
    return rows, report, schedule, manifest


def _native_env_info(input_row: dict, *, success: bool):
    route_id = input_row["route_id"]
    if route_id == "webshop":
        native = {"current_subtask_index": 6 if success else 2, "subtask_count": 6}
    if route_id in {"swesmith", "literesearcher"}:
        native = {"episode_success": success}
    if route_id == "openmle_fast":
        native = {
            "grade": {
                "submission_valid": success,
                "improved_over_baseline": True if success else None,
            }
        }
    if route_id not in CANONICAL_ROUTES:
        raise AssertionError(route_id)
    native["episode_source_identity"] = _episode_source_identity(input_row)
    return native


def _action_row(
    input_row,
    *,
    order: int,
    length: int,
    success: bool,
    horizon: bool = False,
    informational_outcome: str = "deliberately-wrong-generic-outcome",
):
    uid = input_row["uid"]
    final = order == length - 1
    env_info = _native_env_info(input_row, success=success)
    row = {
        "schema": ACTION_ROW_SCHEMA,
        "trajectory_uid": uid,
        "trajectory_row_uid": f"{uid}-row-{order}",
        "trajectory_row_order": order,
        "trajectory_terminal": final,
        "trajectory_return": 1.25 if success else -0.25,
        "sample_reschedule_attempt": 0,
        "route_id": input_row["route_id"],
        "data_source": input_row["route_id"],
        "item_id": input_row["item_id"],
        "data_idx": input_row["data_idx"],
        "min_global_steps": 200,
        "max_global_steps": 200,
        "rollout_done_flag": final,
        "env_info_after": (
            env_info
            if not horizon
            else {
                "stale": True,
                "episode_source_identity": _episode_source_identity(input_row),
            }
        ),
        "context_transition": {
            "schema": "agentmemory_task_neutral_context_transition_v1",
            "operation": "append_observation",
        },
        "wrapper_evidence": {
            "agemem_adapter": {
                "schema": "camg_agemem_style_adapter_v1",
                "episode_private": True,
                "hidden_model_calls": 0,
                "event": "native_action_passthrough",
                "memory_action_count": 0,
                "memory_size_after": 0,
                "context_operation": "append_observation",
            }
        },
        "horizon_finalization": (
            {"env_info": env_info} if final and horizon else None
        ),
        "outcome": informational_outcome,
    }
    return row


def _materialized(input_rows, *, success_by_uid=None, horizon_uids=()):
    success_by_uid = success_by_uid or {}
    horizon_uids = set(horizon_uids)
    output = {"uid": [], "eval_padding": [], "step_record_json": []}
    # Interleave different trajectories to prove that UID grouping, rather than
    # positional unpadding, owns the mapping back to input episodes.
    records_by_uid = {}
    for position, row in enumerate(input_rows):
        length = 2 if position % 2 == 0 else 1
        records_by_uid[row["uid"]] = [
            _action_row(
                row,
                order=order,
                length=length,
                success=success_by_uid.get(row["uid"], position % 2 == 0),
                horizon=row["uid"] in horizon_uids,
            )
            for order in range(length)
        ]
    max_length = max(map(len, records_by_uid.values()))
    for order in range(max_length):
        for row in reversed(input_rows):
            records = records_by_uid[row["uid"]]
            if order >= len(records):
                continue
            output["uid"].append(row["uid"])
            output["eval_padding"].append(row["eval_padding"])
            output["step_record_json"].append(
                json.dumps(records[order], sort_keys=True)
            )
    return materialize_generated_batch(
        output,
        input_rows,
        expected_global_step=200,
        route_max_rounds=ROUTE_MAX_ROUNDS,
    )


def _materialize_records(input_row, records):
    return materialize_generated_batch(
        {
            "uid": [input_row["uid"]] * len(records),
            "eval_padding": [input_row["eval_padding"]] * len(records),
            "step_record_json": [json.dumps(record) for record in records],
        },
        [input_row],
        expected_global_step=200,
        route_max_rounds=ROUTE_MAX_ROUNDS,
    )


class TestHeldoutSchedule(unittest.TestCase):
    def test_compose_is_deterministic_exhaustion_aware_and_preserves_local_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, report, schedule, manifest = _compose(root)
            self.assertEqual(
                [row["route_id"] for row in rows],
                [
                    "webshop",
                    "swesmith",
                    "literesearcher",
                    "openmle_fast",
                    "webshop",
                    "literesearcher",
                    "openmle_fast",
                    "literesearcher",
                ],
            )
            self.assertEqual([row["index"] for row in rows], list(range(8)))
            self.assertEqual([row["data_idx"] for row in rows], [100, 0, 100, 100, 101, 101, 101, 102])
            self.assertEqual(len({row["uid"] for row in rows}), len(rows))
            self.assertEqual(report["row_count"], 8)
            self.assertEqual(report["schedule_sha256"], sha256_file(schedule))
            self.assertTrue(manifest.is_file())

            schedule_two = root / "heldout-two.jsonl"
            manifest_two = root / "heldout-manifest-two.json"
            spec_path = root / "heldout-spec.json"
            report_two = compose_heldout_schedule(
                spec_path,
                expected_spec_sha256=sha256_file(spec_path),
                output_path=schedule_two,
                manifest_path=manifest_two,
            )
            self.assertEqual(schedule.read_bytes(), schedule_two.read_bytes())
            self.assertEqual(report, report_two)

    def test_coding_final_panel_may_be_a_frozen_subset_of_formal_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, _ = _make_spec(root, counts=(2, 4, 3, 2))
            spec = json.loads(spec_path.read_text())
            coding = next(
                route for route in spec["routes"] if route["route_id"] == "swesmith"
            )
            coding_path = root / coding["schedule"]
            full_rows = [json.loads(line) for line in coding_path.read_text().splitlines()]
            selected_rows = [full_rows[0], full_rows[2]]
            selected_path = root / "swesmith-final-panel.jsonl"
            coding["schedule"] = selected_path.name
            coding["expected_rows"] = len(selected_rows)
            coding["schedule_sha256"] = _write_jsonl(selected_path, selected_rows)

            split_path = root / spec["complete_split_contract"]
            split = json.loads(split_path.read_text())
            split["environments"]["coding"]["formal_eval_tasks"] = len(selected_rows)
            spec["complete_split_contract_sha256"] = _write_json(split_path, split)
            spec_sha256 = _write_json(spec_path, spec)

            report = compose_heldout_schedule(
                spec_path,
                expected_spec_sha256=spec_sha256,
                output_path=root / "out.jsonl",
                manifest_path=root / "manifest.json",
            )
            authority = report["sources"]["swesmith"]["selection_authority"]
            self.assertEqual(authority["formal_eval_task_count"], 4)
            self.assertEqual(authority["scheduled_task_count"], 2)
            self.assertTrue(authority["scheduled_subset_of_formal_eval"])

            selected_rows[0]["extra_info"]["instance_id"] = "not-in-formal-eval"
            coding["schedule_sha256"] = _write_jsonl(selected_path, selected_rows)
            spec_sha256 = _write_json(spec_path, spec)
            with self.assertRaisesRegex(ValueError, "not a subset"):
                compose_heldout_schedule(
                    spec_path,
                    expected_spec_sha256=spec_sha256,
                    output_path=root / "out-bad.jsonl",
                    manifest_path=root / "manifest-bad.json",
                )

    def test_rejects_source_hash_drift_and_reserved_eval_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, spec_sha256 = _make_spec(root)
            source = root / "webshop.jsonl"
            source.write_text(source.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schedule sha256 mismatch"):
                compose_heldout_schedule(
                    spec_path,
                    expected_spec_sha256=spec_sha256,
                    output_path=root / "out.jsonl",
                    manifest_path=root / "manifest.json",
                )

            spec_path, spec_sha256 = _make_spec(root)
            source_rows = [json.loads(line) for line in source.read_text().splitlines()]
            source_rows[0]["uid"] = "caller-controlled"
            source_sha = _write_jsonl(source, source_rows)
            spec = json.loads(spec_path.read_text())
            spec["routes"][0]["schedule_sha256"] = source_sha
            spec_sha256 = _write_json(spec_path, spec)
            with self.assertRaisesRegex(ValueError, "reserved eval identity"):
                compose_heldout_schedule(
                    spec_path,
                    expected_spec_sha256=spec_sha256,
                    output_path=root / "out.jsonl",
                    manifest_path=root / "manifest.json",
                )

    def test_rejects_coding_runtime_that_is_not_formal_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, _ = _make_spec(root)
            spec = json.loads(spec_path.read_text())
            coding = next(route for route in spec["routes"] if route["route_id"] == "swesmith")
            runtime_path = root / coding["formal_eval_runtime_manifest"]
            runtime = json.loads(runtime_path.read_text())
            runtime["selection"] = "all admitted held-out tasks"
            coding["formal_eval_runtime_manifest_sha256"] = _write_json(
                runtime_path, runtime
            )
            spec_sha256 = _write_json(spec_path, spec)
            with self.assertRaisesRegex(ValueError, "formal Eval subset"):
                compose_heldout_schedule(
                    spec_path,
                    expected_spec_sha256=spec_sha256,
                    output_path=root / "out.jsonl",
                    manifest_path=root / "manifest.json",
                )

    def test_rejects_route_count_not_authorized_by_complete_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, _ = _make_spec(root)
            spec = json.loads(spec_path.read_text())
            split_path = root / spec["complete_split_contract"]
            split = json.loads(split_path.read_text())
            split["environments"]["shop"]["formal_heldout_episode_count"] += 1
            spec["complete_split_contract_sha256"] = _write_json(split_path, split)
            spec_sha256 = _write_json(spec_path, spec)
            with self.assertRaisesRegex(
                ValueError, "counts differ from the complete split contract"
            ):
                compose_heldout_schedule(
                    spec_path,
                    expected_spec_sha256=spec_sha256,
                    output_path=root / "out.jsonl",
                    manifest_path=root / "manifest.json",
                )

    def test_coding_formal_eval_requires_complete_admission_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            routing = root / "routing.jsonl"
            routing_sha256 = _write_jsonl(
                routing,
                [
                    {"data_idx": index, "extra_info": {"index": index}}
                    for index in range(2)
                ],
            )
            summary = root / "admission-summary.json"
            summary_sha256 = _write_json(
                summary,
                {
                    "schema": "camg_swesmith_complete_admission_summary_v2",
                    "status": "pass",
                    "raw_assigned_task_count": 10_381,
                    "classified_task_count": 10_380,
                    "admitted_task_count": 2,
                    "excluded_task_count": 10_378,
                    "pending_infrastructure_task_count": 0,
                    "raw_heldout_repository_count": 36,
                    "count_cap": None,
                    "top_k": None,
                    "early_stop": False,
                    "heldout_evaluation_run": False,
                },
            )
            admission = root / "admission.json"
            selection = root / "selection.json"
            selection_sha256 = _write_json(
                selection,
                {
                    "schema": "camg_swesmith_formal_eval_selection_v5",
                    "status": "frozen",
                    "active_train_task_count": 8,
                    "active_training_inputs_modified": False,
                    "complete_admitted_heldout_pool_task_count": 6,
                    "complete_admitted_heldout_pool_repository_count": 2,
                    "formal_eval_task_count": 2,
                    "formal_eval_repository_count": 1,
                    "formal_eval_fraction": 0.2,
                    "extension_pool_task_count": 4,
                    "extension_pool_repository_count": 1,
                    "extension_pool_formal_evaluation_role": False,
                    "extension_pool_training_role": False,
                    "selection_depends_on_model_output_or_reward": False,
                    "heldout_evaluation_run": False,
                },
            )
            pool = root / "pool.json"
            pool_sha256 = _write_json(pool, {
                "schema": "camg_swesmith_admitted_heldout_pool_manifest_v5",
                "status": "complete", "role": "admitted_heldout_candidate_pool",
                "task_count": 6, "repository_count": 2,
                "formal_evaluation_role": False, "training_role": False,
            })
            extension = root / "extension.json"
            extension_sha256 = _write_json(extension, {
                "schema": "camg_swesmith_extension_pool_manifest_v5",
                "status": "frozen", "role": "admitted_heldout_extension_pool",
                "task_count": 4, "repository_count": 1,
                "formal_evaluation_role": False, "training_role": False,
            })
            heldout = root / "heldout.json"
            heldout_sha256 = _write_json(heldout, {
                "role": "formal_heldout",
                "selection": {
                    "mode": "instance_ids", "count": 2, "repository_count": 1,
                    "source_admitted_pool_count": 6,
                    "source_admitted_pool_repository_count": 2,
                    "active_training_inputs_modified": False,
                    "count_cap": None, "top_k": None, "early_stop": False,
                },
            })
            payload = {
                "schema": "camg_swesmith_formal_eval_runtime_manifest_v5",
                "status": "ready",
                "selection": (
                    "deterministic complete-repository subset of the "
                    "exact-runtime-admitted held-out candidate pool"
                ),
                "heldout_evaluation_run": False,
                "active_training_inputs_modified": False,
                "raw_assigned_repository_count": 36,
                "raw_assigned_task_count": 10_381,
                "task_count": 2,
                "repository_count": 1,
                "complete_admitted_pool_task_count": 6,
                "complete_admitted_pool_repository_count": 2,
                "extension_pool_task_count": 4,
                "extension_pool_repository_count": 1,
                "semantic_exclusion_count": 10_375,
                "count_cap": None,
                "top_k": None,
                "early_stop": False,
                "files": {
                    "routing": {
                        "path": routing.name,
                        "bytes": routing.stat().st_size,
                        "sha256": routing_sha256,
                    },
                    "admission_summary": {
                        "path": summary.name,
                        "bytes": summary.stat().st_size,
                        "sha256": summary_sha256,
                    },
                    "formal_eval_selection": {
                        "path": selection.name, "bytes": selection.stat().st_size,
                        "sha256": selection_sha256,
                    },
                    "admitted_pool_manifest": {
                        "path": pool.name, "bytes": pool.stat().st_size,
                        "sha256": pool_sha256,
                    },
                    "extension_pool_manifest": {
                        "path": extension.name, "bytes": extension.stat().st_size,
                        "sha256": extension_sha256,
                    },
                    "manifest": {
                        "path": heldout.name, "bytes": heldout.stat().st_size,
                        "sha256": heldout_sha256,
                    },
                },
            }
            digest = _write_json(admission, payload)
            with self.assertRaisesRegex(ValueError, "admission summary"):
                verify_swesmith_formal_eval_authority(
                    admission,
                    expected_sha256=digest,
                    expected_routing_sha256=routing_sha256,
                    expected_admitted_task_count=2,
                )

    def test_padding_uses_synthetic_uid_and_explicit_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, _ = _compose(Path(directory))
            padded = pad_batch_rows(
                rows[:3], batch_index=2, size_divisor=4, padding_index_base=10_000
            )
            self.assertEqual(len(padded), 4)
            self.assertEqual(padded[:3], rows[:3])
            synthetic = padded[3]
            self.assertTrue(synthetic["eval_padding"])
            self.assertTrue(synthetic["uid"].startswith("camg-heldout-padding-v1-"))
            self.assertNotIn(synthetic["uid"], {row["uid"] for row in rows})
            self.assertEqual(synthetic["extra_info"]["padding_source_uid"], rows[0]["uid"])
            self.assertEqual(synthetic["index"], 10_008)


class TestHeldoutMaterialization(unittest.TestCase):
    def test_uid_grouping_handles_multi_action_rows_and_filters_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, _ = _compose(Path(directory))
            inputs = pad_batch_rows(
                rows[:3], batch_index=0, size_divisor=4, padding_index_base=10_000
            )
            result = _materialized(inputs)
            self.assertEqual([record["uid"] for record in result["episodes"]], [row["uid"] for row in rows[:3]])
            self.assertEqual(len(result["padding_uids"]), 1)
            self.assertEqual(len(result["padding_action_rows"]), 1)
            self.assertEqual([row["trajectory_row_order"] for row in result["action_rows"][:2]], [0, 1])

    def test_generic_outcome_is_ignored_and_horizon_native_evidence_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, _ = _compose(Path(directory), counts=(1, 1, 1, 1))
            success_by_uid = {row["uid"]: True for row in rows}
            result = _materialized(
                rows,
                success_by_uid=success_by_uid,
                horizon_uids={rows[1]["uid"]},
            )
            metrics = aggregate_episode_metrics(result["episodes"])
            self.assertEqual(metrics["average_success"], 1.0)
            self.assertFalse(metrics["generic_action_outcome_used"])
            self.assertEqual(
                result["episodes"][1]["final_env_info"]["episode_success"],
                True,
            )
            for source, episode in zip(rows, result["episodes"]):
                self.assertEqual(
                    episode["verified_native_source_identity"],
                    _episode_source_identity(source),
                )
            self.assertEqual(
                result["episodes"][0]["terminal_outcome_informational_only"],
                "deliberately-wrong-generic-outcome",
            )

    def test_webshop_budget_terminal_uses_latest_native_progress_before_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, _ = _compose(Path(directory), counts=(1, 1, 1, 1))
            source = rows[0]
            records = [
                _action_row(source, order=order, length=30, success=False)
                for order in range(30)
            ]
            final = records[-1]
            final["rollout_done_flag"] = False
            final["outcome"] = "max_rounds"
            final["env_info_after"] = {
                "episode_source_identity": _episode_source_identity(source)
            }
            final["context_transition"]["operation"] = "preserve"
            final["wrapper_evidence"]["agemem_adapter"] = {
                "schema": "camg_agemem_style_adapter_v1",
                "episode_private": True,
                "hidden_model_calls": 0,
                "event": "memory_tool_action",
                "memory_action_index": 1,
                "memory_size_before": 0,
                "memory_size_after": 0,
                "accepted": True,
                "operation": "Summary_context",
                "context_operation": "preserve",
            }
            episode = _materialize_records(source, records)["episodes"][0]
            self.assertEqual(episode["native_metric"]["value"], 2 / 6)
            self.assertEqual(
                episode["native_metric_evidence"],
                {
                    "source": "latest_native_action.env_info_after",
                    "action_row_uid": records[-2]["trajectory_row_uid"],
                },
            )

    def test_webshop_all_memory_budget_terminal_has_audited_initial_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, _ = _compose(Path(directory), counts=(1, 1, 1, 1))
            source = rows[0]
            records = []
            for order in range(30):
                record = _action_row(
                    source,
                    order=order,
                    length=30,
                    success=False,
                )
                record["env_info_after"] = {
                    "episode_source_identity": _episode_source_identity(source)
                }
                record["context_transition"]["operation"] = "preserve"
                record["wrapper_evidence"]["agemem_adapter"] = {
                    "schema": "camg_agemem_style_adapter_v1",
                    "episode_private": True,
                    "hidden_model_calls": 0,
                    "event": "memory_tool_action",
                    "memory_action_index": order + 1,
                    "memory_size_before": 0,
                    "memory_size_after": 0,
                    "accepted": True,
                    "operation": "Summary_context",
                    "context_operation": "preserve",
                }
                records.append(record)
            records[-1]["rollout_done_flag"] = False
            records[-1]["outcome"] = "max_rounds"
            episode = _materialize_records(source, records)["episodes"][0]
            self.assertEqual(episode["native_metric"]["value"], 0.0)
            self.assertEqual(
                episode["native_metric_evidence"]["source"],
                "initial_state_all_actions_are_memory_tools",
            )

    def test_every_action_row_requires_exact_native_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, _ = _compose(Path(directory), counts=(1, 1, 1, 1))
            for source in rows:
                with self.subTest(route=source["route_id"]):
                    records = [
                        _action_row(source, order=order, length=2, success=True)
                        for order in range(2)
                    ]
                    episode = _materialize_records(source, records)["episodes"][0]
                    self.assertEqual(
                        episode["verified_native_source_identity"],
                        _episode_source_identity(source),
                    )

                    missing = deepcopy(records)
                    missing[0]["env_info_after"].pop("episode_source_identity")
                    with self.assertRaisesRegex(
                        ValueError, "native source identity"
                    ):
                        _materialize_records(source, missing)

                    drifted = deepcopy(records)
                    drifted[1]["env_info_after"]["episode_source_identity"][
                        "data_idx"
                    ] += 1
                    with self.assertRaisesRegex(
                        ValueError, "native source identity drift"
                    ):
                        _materialize_records(source, drifted)

    def test_horizon_native_source_identity_must_match_action_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, _ = _compose(Path(directory), counts=(1, 1, 1, 1))
            source = rows[1]
            record = _action_row(
                source, order=0, length=1, success=True, horizon=True
            )
            record["horizon_finalization"]["env_info"][
                "episode_source_identity"
            ]["instance_id"] = "wrong.issue"
            with self.assertRaisesRegex(ValueError, "native source identity drift"):
                _materialize_records(source, [record])

    def test_budget_terminal_requires_exact_shop_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, _ = _compose(Path(directory), counts=(1, 1, 1, 1))
            shop = rows[0]
            short = [
                _action_row(shop, order=order, length=29, success=False)
                for order in range(29)
            ]
            short[-1]["rollout_done_flag"] = False
            short[-1]["outcome"] = "max_rounds"
            with self.assertRaisesRegex(ValueError, "terminal wrapper transition"):
                _materialize_records(shop, short)

            swe = rows[1]
            not_shop = [
                _action_row(swe, order=order, length=30, success=False)
                for order in range(30)
            ]
            not_shop[-1]["rollout_done_flag"] = False
            not_shop[-1]["outcome"] = "max_rounds"
            with self.assertRaisesRegex(ValueError, "terminal wrapper transition"):
                _materialize_records(swe, not_shop)

    def test_agemem_evidence_is_mandatory_and_hidden_calls_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, _ = _compose(Path(directory), counts=(1, 1, 1, 1))
            source = rows[1]
            missing = _action_row(source, order=0, length=1, success=True)
            missing["wrapper_evidence"] = {}
            with self.assertRaisesRegex(ValueError, "AgeMem evidence failed"):
                _materialize_records(source, [missing])

            hidden = _action_row(source, order=0, length=1, success=True)
            hidden["wrapper_evidence"]["agemem_adapter"]["hidden_model_calls"] = 1
            with self.assertRaisesRegex(ValueError, "hidden model call observed"):
                _materialize_records(source, [hidden])

    def test_rejects_missing_uid_and_noncontiguous_action_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, _ = _compose(Path(directory), counts=(1, 1, 1, 1))
            result = _materialized(rows)
            action_rows = result["action_rows"]
            first_uid = rows[0]["uid"]
            broken = {
                "uid": [row["eval_uid"] for row in action_rows if row["eval_uid"] != first_uid],
                "eval_padding": [row["eval_padding"] for row in action_rows if row["eval_uid"] != first_uid],
                "step_record_json": [json.dumps(row) for row in action_rows if row["eval_uid"] != first_uid],
            }
            with self.assertRaisesRegex(ValueError, "missing expected UIDs"):
                materialize_generated_batch(
                    broken,
                    rows,
                    expected_global_step=200,
                    route_max_rounds=ROUTE_MAX_ROUNDS,
                )

            source = rows[0]
            malformed = _action_row(source, order=1, length=2, success=True)
            fields = {
                "uid": [source["uid"]],
                "eval_padding": [False],
                "step_record_json": [json.dumps(malformed)],
            }
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                materialize_generated_batch(
                    fields,
                    [source],
                    expected_global_step=200,
                    route_max_rounds=ROUTE_MAX_ROUNDS,
                )

    def test_rejects_mixed_checkpoint_or_nonterminal_wrapper_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, _ = _compose(Path(directory), counts=(1, 1, 1, 1))
            source = rows[1]
            record = _action_row(source, order=0, length=1, success=True)
            record["max_global_steps"] = 399
            fields = {
                "uid": [source["uid"]],
                "eval_padding": [False],
                "step_record_json": [json.dumps(record)],
            }
            with self.assertRaisesRegex(ValueError, "mixed policy versions"):
                materialize_generated_batch(
                    fields,
                    [source],
                    expected_global_step=200,
                    route_max_rounds=ROUTE_MAX_ROUNDS,
                )

            record["max_global_steps"] = 200
            record["min_global_steps"] = 399
            record["max_global_steps"] = 399
            with self.assertRaisesRegex(ValueError, "checkpoint step 200"):
                materialize_generated_batch(
                    {
                        **fields,
                        "step_record_json": [json.dumps(record)],
                    },
                    [source],
                    expected_global_step=200,
                    route_max_rounds=ROUTE_MAX_ROUNDS,
                )

            record["min_global_steps"] = 200
            record["max_global_steps"] = 200
            record["rollout_done_flag"] = False
            with self.assertRaisesRegex(ValueError, "terminal wrapper transition"):
                materialize_generated_batch(
                    {
                        **fields,
                        "step_record_json": [json.dumps(record)],
                    },
                    [source],
                    expected_global_step=200,
                    route_max_rounds=ROUTE_MAX_ROUNDS,
                )

    def test_native_metrics_require_registered_fields(self):
        self.assertEqual(
            native_success_metric(
                "webshop", {"current_subtask_index": 3, "subtask_count": 6}
            )["value"],
            0.5,
        )
        with self.assertRaisesRegex(ValueError, "episode_success"):
            native_success_metric("swesmith", {"outcome": "success"})
        with self.assertRaisesRegex(ValueError, "missing grade"):
            native_success_metric("openmle_fast", {"episode_success": True})
        with self.assertRaisesRegex(ValueError, "exactly six sessions"):
            native_success_metric(
                "webshop", {"current_subtask_index": 5, "subtask_count": 5}
            )


class TestHeldoutAtomicResume(unittest.TestCase):
    def test_atomic_batches_resume_and_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, report, _, _ = _compose(root / "source", counts=(2, 2, 2, 2))
            run_dir = root / "run"
            contract = {
                **_run_contract_fixture(),
                "schedule_sha256": report["schedule_sha256"],
            }
            initialize_run_contract(run_dir, contract)
            self.assertEqual(initialize_run_contract(run_dir, contract), contract)
            with self.assertRaisesRegex(ValueError, "differs"):
                initialize_run_contract(
                    run_dir, {**contract, "checkpoint_step": 399}
                )

            first = _materialized(rows[:4])
            receipt0 = commit_batch(
                run_dir,
                batch_index=0,
                schedule_start=0,
                schedule_stop=4,
                materialized=first,
            )
            self.assertEqual(receipt0["schema"], BATCH_SCHEMA)
            self.assertEqual(
                inspect_resume_state(run_dir, schedule_rows=rows),
                {"next_batch_index": 1, "next_schedule_position": 4},
            )
            second = _materialized(rows[4:])
            commit_batch(
                run_dir,
                batch_index=1,
                schedule_start=4,
                schedule_stop=8,
                materialized=second,
            )
            self.assertEqual(
                inspect_resume_state(run_dir, schedule_rows=rows),
                {"next_batch_index": 2, "next_schedule_position": 8},
            )
            final = finalize_run_metrics(run_dir, expected_episode_count=8)
            self.assertEqual(final["episode_count"], 8)
            self.assertEqual(final["average_success_weighting"], "equal_weight_per_environment")

    def test_hash_corruption_and_incomplete_temp_directory_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, _, _, _ = _compose(root / "source", counts=(1, 1, 1, 1))
            run_dir = root / "run"
            initialize_run_contract(run_dir, _run_contract_fixture())
            materialized = _materialized(rows)
            commit_batch(
                run_dir,
                batch_index=0,
                schedule_start=0,
                schedule_stop=4,
                materialized=materialized,
            )
            episode_path = run_dir / "batches/batch-000000/episodes.jsonl"
            episode_path.write_bytes(episode_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "byte count mismatch"):
                verify_batch_directory(run_dir / "batches/batch-000000")

            # Restore from the known in-memory payload before testing the
            # independent interrupted-publication guard.
            _write_jsonl(episode_path, materialized["episodes"])
            incomplete = run_dir / "batches/.batch-000001.crash.tmp"
            incomplete.mkdir()
            with self.assertRaisesRegex(RuntimeError, "incomplete atomic"):
                inspect_resume_state(run_dir, schedule_rows=rows)


class TestHeldoutRunnerPlan(unittest.TestCase):
    def test_runtime_dataset_row_parity_allows_only_standard_runtime_fields(self):
        expected = {
            "uid": "camg-heldout-v1-" + "a" * 64,
            "index": 7,
            "data_idx": 11,
            "item_id": "swesmith:issue-11:heldout-000007",
            "route_id": "swesmith",
            "data_source": "swesmith",
            "agent_name": "amg_task_neutral_async",
            "eval_padding": False,
            "extra_info": {
                "index": 7,
                "route_id": "swesmith",
                "source_item_id": "issue-11",
                "source_extra_info": {
                    "instance_id": "repo.issue-11",
                    "base_repository": "repo",
                },
            },
        }
        attestation = "b" * 64
        processed = deepcopy(expected)
        processed["extra_info"]["route_attestation_sha256"] = attestation
        processed.update(
            {
                "raw_prompt": [{"role": "system", "content": "policy"}],
                "dummy_tensor": object(),
                "tools_kwargs": {},
                "interaction_kwargs": {},
            }
        )
        verify_runtime_dataset_row(
            expected,
            processed,
            expected_route_attestation_sha256=attestation,
            schedule_position=7,
        )
        mutations = {
            "route_id": "webshop",
            "data_idx": 12,
            "item_id": "wrong",
            "uid": "wrong",
            "index": 8,
        }
        for field, value in mutations.items():
            drifted = deepcopy(processed)
            drifted[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                RuntimeError, "runtime dataset"
            ):
                verify_runtime_dataset_row(
                    expected,
                    drifted,
                    expected_route_attestation_sha256=attestation,
                    schedule_position=7,
                )
        nested = deepcopy(processed)
        nested["extra_info"]["source_extra_info"]["instance_id"] = "other.issue"
        with self.assertRaisesRegex(RuntimeError, "extra_info"):
            verify_runtime_dataset_row(
                expected,
                nested,
                expected_route_attestation_sha256=attestation,
                schedule_position=7,
            )
        extra = deepcopy(processed)
        extra["unexpected_runtime_field"] = True
        with self.assertRaisesRegex(RuntimeError, "unexpected top-level"):
            verify_runtime_dataset_row(
                expected,
                extra,
                expected_route_attestation_sha256=attestation,
                schedule_position=7,
            )

    def _plan_fixture(self, root: Path):
        registry = root / "route-registry.json"
        registry_hash = _write_json(
            registry,
            {
                "schema": "amg_route_registry_v1",
                "agent_name": AGENT_NAME,
                "routes": [
                    {
                        "route_id": route_id,
                        "max_rounds": ROUTE_MAX_ROUNDS[route_id],
                        "max_observation_tokens": 8192,
                        "policy_framing_sha256": "a" * 64,
                        "route_attestation_sha256": "b" * 64,
                        "client": {
                            "task_name": route_id,
                            "env_addr": f"http://127.0.0.1:{65101 + position}",
                            "timeout": 240,
                            "max_retries": 2,
                        },
                    }
                    for position, route_id in enumerate(CANONICAL_ROUTES)
                ],
            },
        )
        spec, spec_hash = _make_spec(
            root / "sources",
            counts=tuple(TEST_ROUTE_COUNTS[route] for route in CANONICAL_ROUTES),
            route_registry_sha256=registry_hash,
        )
        schedule = root / "heldout.jsonl"
        schedule_manifest = root / "heldout-manifest.json"
        schedule_report = compose_heldout_schedule(
            spec,
            expected_spec_sha256=spec_hash,
            output_path=schedule,
            manifest_path=schedule_manifest,
        )
        self.assertEqual(schedule_report["row_count"], TEST_EPISODE_COUNT)

        loop_config = root / "agent-loop.yaml"
        loop_config.write_text("- name: amg_task_neutral_async\n", encoding="utf-8")
        model_dir = root / "merged-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
        (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
        model_files = []
        for file_path in sorted(model_dir.iterdir()):
            model_files.append(
                {
                    "path": file_path.name,
                    "bytes": file_path.stat().st_size,
                    "sha256": sha256_file(file_path),
                }
            )
        model_manifest = root / "model-manifest.json"
        commits = {"outer": "1" * 40, "inner": "2" * 40, "verl": "3" * 40}
        model_manifest_hash = _write_json(
            model_manifest,
            {
                "schema": MODEL_MANIFEST_SCHEMA,
                "checkpoint_step": 200,
                "training_run_id": "agemem-formal200",
                "source_commits": commits,
                "model_path": str(model_dir),
                "files": model_files,
            },
        )
        resolved_config = root / "resolved.yaml"
        resolved_config.write_text(
            """actor_rollout_ref:
  agentgym:
    route_registry_path: /old/routes.json
    route_registry_sha256: old
  model:
    path: /old/model
    tokenizer_path: null
    use_shm: false
  rollout:
    name: sglang
    mode: async
    n: 1
    calculate_log_probs: true
    nnodes: 0
    n_gpus_per_node: 6
    tensor_model_parallel_size: 1
    data_parallel_size: 1
    pipeline_model_parallel_size: 1
    load_format: dummy
    skip_tokenizer_init: true
    gpu_memory_utilization: 0.35
    full_determinism: false
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    do_sample: true
    multi_turn:
      enable: true
    val_kwargs:
      temperature: 0
      top_p: 1.0
      top_k: -1
      n: 1
      do_sample: false
    agent:
      num_workers: 64
      default_agent_loop: amg_task_neutral_async
      agent_loop_config_path: /old/agent-loop.yaml
    trace:
      experiment_name: old
    agentgym:
      route_registry_path: /old/routes.json
      route_registry_sha256: old
data:
  train_files: /old/train.jsonl
  val_files: /old/train.jsonl
  train_max_samples: -1
  val_max_samples: 1
  train_batch_size: 0
  gen_batch_size: 1
  val_batch_size: null
  shuffle: false
  validation_shuffle: false
  dataloader_num_workers: 0
  custom_cls:
    path: pkg://agentmemorygym_verl.dataset
    name: AMGTrajectoryDataset
  apply_chat_template_kwargs:
    enable_thinking: false
  agentgym:
    route_registry_path: /old/routes.json
    route_registry_sha256: old
distillation:
  enabled: false
reward:
  reward_model:
    enable: false
trainer:
  experiment_name: old
  validation_data_dir: null
  nnodes: 1
  n_gpus_per_node: 6
""",
            encoding="utf-8",
        )
        kwargs = {
            "run_id": "agemem-heldout-eval-v1",
            "run_dir": root / "run",
            "resolved_config_path": resolved_config,
            "expected_resolved_config_sha256": sha256_file(resolved_config),
            "schedule_path": schedule,
            "expected_schedule_sha256": schedule_report["schedule_sha256"],
            "schedule_manifest_path": schedule_manifest,
            "expected_schedule_manifest_sha256": sha256_file(schedule_manifest),
            "route_registry_path": registry,
            "expected_route_registry_sha256": registry_hash,
            "agent_loop_config_path": loop_config,
            "expected_agent_loop_config_sha256": sha256_file(loop_config),
            "model_manifest_path": model_manifest,
            "expected_model_manifest_sha256": model_manifest_hash,
            "training_run_id": "agemem-formal200",
            "training_outer_commit": commits["outer"],
            "training_inner_commit": commits["inner"],
            "training_verl_commit": commits["verl"],
            "evaluator_outer_commit": "4" * 40,
            "evaluator_inner_commit": "5" * 40,
            "evaluator_verl_commit": commits["verl"],
        }
        return kwargs, model_dir, model_manifest

    def test_plan_verifies_all_inputs_and_derives_greedy_standalone_config(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs, model_dir, _ = self._plan_fixture(Path(directory))
            plan = load_eval_plan(**kwargs)
            config = derive_eval_config(plan)
            self.assertEqual(
                config["actor_rollout_ref"]["model"]["path"], str(model_dir.resolve())
            )
            rollout = config["actor_rollout_ref"]["rollout"]
            self.assertEqual(rollout["load_format"], "auto")
            self.assertEqual(rollout["nnodes"], 1)
            self.assertEqual(rollout["n_gpus_per_node"], 8)
            self.assertFalse(rollout["do_sample"])
            self.assertTrue(rollout["full_determinism"])
            self.assertEqual(rollout["agent"]["num_workers"], 64)
            self.assertEqual(
                config["data"]["agentgym"]["route_registry_sha256"],
                kwargs["expected_route_registry_sha256"],
            )
            config_hash = hashlib.sha256(_json_bytes(config)).hexdigest()
            contract = run_contract(plan, config_hash)
            self.assertEqual(contract["checkpoint_step"], 200)
            self.assertEqual(contract["schedule"]["episode_count"], TEST_EPISODE_COUNT)
            self.assertEqual(contract["schedule"]["per_route_rows"], TEST_ROUTE_COUNTS)
            self.assertEqual(
                contract["schedule"]["complete_split_authority"]["route_counts"],
                TEST_ROUTE_COUNTS,
            )
            self.assertEqual(contract["evaluator_source_commits"]["inner"], "5" * 40)
            self.assertEqual(contract["route_registry"]["max_rounds"], ROUTE_MAX_ROUNDS)
            self.assertFalse(contract["runner"]["generic_action_outcome_used"])

    def test_plan_rejects_early_checkpoint_and_model_payload_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs, model_dir, _ = self._plan_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "only at update200"):
                load_eval_plan(**{**kwargs, "checkpoint_step": 399})
            (model_dir / "config.json").write_text('{"tampered":true}\n')
            with self.assertRaisesRegex(ValueError, "byte count mismatch|sha256 mismatch"):
                load_eval_plan(**kwargs)

    def test_commit_interval_and_receipt_metric_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, _, _, _ = _compose(root / "source", counts=(1, 1, 1, 1))
            run_dir = root / "run"
            initialize_run_contract(run_dir, _run_contract_fixture())
            materialized = _materialized(rows)
            with self.assertRaisesRegex(ValueError, "schedule interval"):
                commit_batch(
                    run_dir,
                    batch_index=0,
                    schedule_start=0,
                    schedule_stop=3,
                    materialized=materialized,
                )
            commit_batch(
                run_dir,
                batch_index=0,
                schedule_start=0,
                schedule_stop=4,
                materialized=materialized,
            )
            receipt_path = run_dir / "batches/batch-000000/receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["batch_metrics"]["average_success"] = 0.123
            _write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "metrics differ"):
                verify_batch_directory(run_dir / "batches/batch-000000")

            receipt = json.loads(receipt_path.read_text())
            receipt["batch_metrics"] = aggregate_episode_metrics(
                materialized["episodes"], require_all_routes=False
            )
            receipt["agemem_evidence"]["real"]["totals"]["hidden_model_calls"] = 1
            _write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "AgeMem evidence differs"):
                verify_batch_directory(run_dir / "batches/batch-000000")

    def test_runtime_uses_standalone_server_and_agent_loop_apis_then_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs, _, _ = self._plan_fixture(root)
            plan = load_eval_plan(**kwargs)
            calls = {
                "ray_init": 0,
                "ray_shutdown": 0,
                "server_create": 0,
                "loop_create": 0,
                "generate": 0,
                "global_steps": [],
            }

            def namespace(value):
                if isinstance(value, dict):
                    return SimpleNamespace(
                        **{key: namespace(item) for key, item in value.items()}
                    )
                if isinstance(value, list):
                    return [namespace(item) for item in value]
                return value

            class FakeDataset:
                def __init__(self, *, data_files, **_kwargs):
                    self.rows = [
                        json.loads(line)
                        for line in Path(data_files).read_text().splitlines()
                    ]

                def __len__(self):
                    return len(self.rows)

                def __getitem__(self, index):
                    row = deepcopy(self.rows[index])
                    row["extra_info"]["route_attestation_sha256"] = "b" * 64
                    row.update(
                        {
                            "raw_prompt": [{"role": "system", "content": "policy"}],
                            "dummy_tensor": object(),
                            "tools_kwargs": {},
                            "interaction_kwargs": {},
                        }
                    )
                    return row

            class FakeDataProto:
                def __init__(self, rows):
                    self.rows = rows
                    self.meta_info = {}

                @classmethod
                def from_single_dict(cls, rows):
                    return cls(rows)

            class RemoteSetStep:
                def remote(self, step):
                    calls["global_steps"].append(step)
                    return step

            class FakeHandle:
                def __init__(self):
                    self.set_global_steps = RemoteSetStep()

            class FakeServer:
                def __init__(self):
                    self.server_handles = [FakeHandle() for _ in range(8)]

                def get_client(self):
                    return object()

            class FakeServerManager:
                @classmethod
                def create(cls, **_kwargs):
                    calls["server_create"] += 1
                    return FakeServer()

            class FakeLoop:
                def generate_sequences(self, prompts):
                    calls["generate"] += 1
                    output = {"uid": [], "eval_padding": [], "step_record_json": []}
                    for row in prompts.rows:
                        record = _action_row(
                            row,
                            order=0,
                            length=1,
                            success=True,
                            informational_outcome="ignored",
                        )
                        output["uid"].append(row["uid"])
                        output["eval_padding"].append(row["eval_padding"])
                        output["step_record_json"].append(json.dumps(record))
                    return SimpleNamespace(non_tensor_batch=output)

            class FakeLoopManager:
                @classmethod
                def create(cls, **_kwargs):
                    calls["loop_create"] += 1
                    return FakeLoop()

            ray = types.ModuleType("ray")
            ray.is_initialized = lambda: False
            ray.init = lambda **_kwargs: calls.__setitem__(
                "ray_init", calls["ray_init"] + 1
            )
            ray.cluster_resources = lambda: {"GPU": 8}
            ray.get = lambda values: values
            ray.shutdown = lambda: calls.__setitem__(
                "ray_shutdown", calls["ray_shutdown"] + 1
            )
            omegaconf = types.ModuleType("omegaconf")
            omegaconf.OmegaConf = SimpleNamespace(create=namespace)
            dataset_module = types.ModuleType("agentmemorygym_verl.dataset")
            dataset_module.AMGTrajectoryDataset = FakeDataset
            verl = types.ModuleType("verl")
            verl.__path__ = []
            experimental = types.ModuleType("verl.experimental")
            experimental.__path__ = []
            loop_module = types.ModuleType("verl.experimental.agent_loop")
            loop_module.AgentLoopManager = FakeLoopManager
            protocol = types.ModuleType("verl.protocol")
            protocol.DataProto = FakeDataProto
            utils = types.ModuleType("verl.utils")
            utils.__path__ = []
            utils.omega_conf_to_dataclass = lambda _value: SimpleNamespace(
                tokenizer="tokenizer", processor=None
            )
            utils_dataset = types.ModuleType("verl.utils.dataset")
            utils_dataset.__path__ = []
            rl_dataset = types.ModuleType("verl.utils.dataset.rl_dataset")
            rl_dataset.collate_fn = lambda rows: rows
            workers = types.ModuleType("verl.workers")
            workers.__path__ = []
            rollout = types.ModuleType("verl.workers.rollout")
            rollout.__path__ = []
            llm_server = types.ModuleType("verl.workers.rollout.llm_server")
            llm_server.LLMServerManager = FakeServerManager
            modules = {
                "ray": ray,
                "omegaconf": omegaconf,
                "agentmemorygym_verl.dataset": dataset_module,
                "verl": verl,
                "verl.experimental": experimental,
                "verl.experimental.agent_loop": loop_module,
                "verl.protocol": protocol,
                "verl.utils": utils,
                "verl.utils.dataset": utils_dataset,
                "verl.utils.dataset.rl_dataset": rl_dataset,
                "verl.workers": workers,
                "verl.workers.rollout": rollout,
                "verl.workers.rollout.llm_server": llm_server,
            }
            environment = {
                "AGENTMEMORY_PROCESS_OWNER": "heldout-test-owner",
                "AGENTMEMORY_RUN_ID": plan.run_id,
            }
            with mock.patch.dict(sys.modules, modules), mock.patch.dict(
                os.environ, environment, clear=False
            ):
                metrics = run_evaluation(plan)
                resumed = run_evaluation(plan)

            self.assertEqual(metrics, resumed)
            self.assertEqual(metrics["episode_count"], TEST_EPISODE_COUNT)
            self.assertEqual(metrics["average_success"], 1.0)
            self.assertEqual(calls["ray_init"], 1)
            self.assertEqual(calls["ray_shutdown"], 1)
            self.assertEqual(calls["server_create"], 1)
            self.assertEqual(calls["loop_create"], 1)
            self.assertEqual(calls["global_steps"], [200] * 8)
            self.assertEqual(calls["generate"], 1)


if __name__ == "__main__":
    unittest.main()
