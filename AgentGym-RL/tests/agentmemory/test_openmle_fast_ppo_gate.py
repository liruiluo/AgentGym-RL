#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import io
import importlib.util
import json
import signal
from pathlib import Path
from contextlib import redirect_stdout
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "agentmemory" / "verify_openmle_fast_ppo_gate.py"
OUTER_COMMIT = "1" * 40
INNER_COMMIT = "2" * 40
PROMPT_SHA256 = "3" * 64


def load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_openmle_fast_ppo_gate",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load OpenMLE-fast PPO verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest() -> dict:
    records = [
        {
            "data_idx": index,
            "task_id": f"task-{index:02d}@1",
            "source_family": f"KAGGLE_DATASET:owner/source-{index:02d}",
            "role": "gate_only",
        }
        for index in range(64)
    ]
    return {
        "schema": "openmle_fast_public_manifest_v1",
        "panel_id": "openmle-fast-integration-v1-g64-gate",
        "role": "gate_only",
        "openmle_tasks_revision": "f56e4b31252a9b81d95fea100098cd49b7290398",
        "task_count": len(records),
        "task_id_list_sha256": "4" * 64,
        "compact_panel_sha256": "5" * 64,
        "max_policy_actions": 30,
        "records": records,
    }


def manifest_sha256(document: dict) -> str:
    raw = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def digest(value) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def counters(
    action_count: int,
    *,
    execution_count: int,
    grading_count: int,
    execution_attempt_count: int | None = None,
    execution_completed_count: int | None = None,
    nested_subprocess_count: int = 0,
    fit_count: int = 0,
) -> dict:
    if execution_attempt_count is None:
        execution_attempt_count = execution_count
    if execution_completed_count is None:
        execution_completed_count = execution_attempt_count
    return {
        "action_count": action_count,
        "execution_action_count": execution_count,
        "execution_attempt_count": execution_attempt_count,
        "execution_completed_count": execution_completed_count,
        "nested_subprocess_count": nested_subprocess_count,
        "fit_count": fit_count,
        "grading_count": grading_count,
    }


def row(
    record: dict,
    task_round: int,
    *,
    terminal: bool,
    manifest_digest: str,
) -> dict:
    index = record["data_idx"]
    episode_id = f"episode-{index:02d}"
    request_id = f"request-{index:02d}"
    submission_sha256 = f"{index:064x}"
    sampled_tokens = [1000 + index, 2000 + task_round]
    sampled_logprobs = [-0.1 - index / 10000, -0.2 - task_round / 10000]
    packed_tokens = [11, 12] + sampled_tokens
    response_mask = [0, 0, 1, 1]
    packed_logprobs = [0.0, 0.0] + sampled_logprobs
    before = counters(
        task_round - 1,
        execution_count=min(task_round - 1, 1),
        grading_count=0,
    )
    after = counters(
        task_round,
        execution_count=min(task_round, 1),
        grading_count=int(terminal),
    )
    action = (
        "submit"
        if terminal
        else "shell_command " + json.dumps({"command": "python train.py"})
    )
    reward = float(index % 2) if terminal else 0.0
    action_submission = {"raw_policy_output": action}
    if terminal:
        action_submission.update(
            {
                "request_id": request_id,
                "episode_id": episode_id,
                "submission_sha256": submission_sha256,
            }
        )
    counter_delta = {key: after[key] - before[key] for key in before}
    value = {
        "schema": "task_neutral_policy_step_v1",
        "parent_index": index,
        "data_idx": index,
        "task_id": record["task_id"],
        "source_family": record["source_family"],
        "task_manifest_sha256": manifest_digest,
        "episode_id": episode_id,
        "item_id": f"openmlefast_{index:06d}_{'a' * 16}",
        "task_round": task_round,
        "action": action,
        "action_submission": action_submission,
        "generation_token_ids_are_exact": True,
        "backend_token_ids_are_exact": True,
        "sampled_response_token_ids": sampled_tokens,
        "packed_token_ids": packed_tokens,
        "response_mask": response_mask,
        "sampled_old_logprobs": sampled_logprobs,
        "packed_old_logprobs": packed_logprobs,
        "sampled_response_token_ids_sha256": digest(sampled_tokens),
        "packed_token_ids_sha256": digest(packed_tokens),
        "response_mask_sha256": digest(response_mask),
        "sampled_old_logprobs_sha256": digest(sampled_logprobs),
        "reward": reward,
        "done": terminal,
        "truncated": False,
        "terminal_reason": "graded_submission" if terminal else None,
        "terminal_classification": "graded" if terminal else None,
        "runtime_success": terminal,
        "episode_success": terminal and reward > 0.0,
        "counters_before": before,
        "counters_after": after,
        "counter_delta": counter_delta,
        "grade_receipt": None,
    }
    if terminal:
        value["grade_receipt"] = {
            "schema": "openmle_fast_grade_response_v1",
            "contract_version": "openmle_fast_v1",
            "request_id": request_id,
            "episode_id": episode_id,
            "task_id": record["task_id"],
            "submission_sha256": submission_sha256,
            "submission_valid": True,
            "native_score": 0.5 + index / 1000,
            "higher_is_better": bool(index % 2),
            "normalized_reward": reward,
            "improved_over_baseline": reward > 0.0,
            "runtime_success": True,
            "terminal_reason": "graded_submission",
            "classification": "graded",
            "audit_digest": f"{index + 1:064x}",
        }
    return value


def endpoint_metadata(document: dict, *, active_count: int = 0) -> dict:
    return {
        "schema": "openmle_fast_public_metadata_v1",
        "domain_id": "openmle_fast",
        "contract_version": "openmle_fast_v1",
        "panel_id": document["panel_id"],
        "role": document["role"],
        "task_count": document["task_count"],
        "task_manifest_sha256": manifest_sha256(document),
        "openmle_tasks_revision": document["openmle_tasks_revision"],
        "task_id_list_sha256": document["task_id_list_sha256"],
        "compact_panel_sha256": document["compact_panel_sha256"],
        "policy_prompt_sha256": PROMPT_SHA256,
        "runtime_source": {
            "outer_commit": OUTER_COMMIT,
            "inner_commit": INNER_COMMIT,
        },
        "active_slot_count": active_count,
        "active_environment_count": active_count,
        "active_workspace_count": active_count,
    }


def update_evidence(document: dict) -> dict:
    rows = []
    digest_value = manifest_sha256(document)
    for record in document["records"]:
        rows.append(row(record, 1, terminal=False, manifest_digest=digest_value))
        rows.append(row(record, 2, terminal=True, manifest_digest=digest_value))
    return {
        "schema": "openmle_fast_ppo_update_evidence_v1",
        "gate_contract": {
            "schema": "openmle_fast_gate_contract_v1",
            "role": "gate_only",
            "optimizer_update_limit": 1,
            "initialization": "fresh_base_checkpoint",
            "resume_checkpoint": None,
            "checkpoint_reuse_allowed": False,
        },
        "manifest_sha256": digest_value,
        "runtime_source": {
            "outer_commit": OUTER_COMMIT,
            "inner_commit": INNER_COMMIT,
        },
        "policy_prompt_sha256": PROMPT_SHA256,
        "update": {
            "first_global_step": 1,
            "last_global_step": 1,
            "optimizer_update_count": 1,
            "rows": rows,
        },
        "optimizer_readback": {
            "role": "same_batch_post_optimizer_readback",
            "global_step": 1,
            "actor": {
                "optimizer_step_before": 0,
                "optimizer_step_after": 1,
                "parameter_delta_l2": 0.25,
                "max_abs_delta": 0.125,
                "parameter_probe_changed_count": 4,
            },
            "critic": {
                "optimizer_step_before": 0,
                "optimizer_step_after": 1,
                "parameter_delta_l2": 0.5,
                "max_abs_delta": 0.25,
                "parameter_probe_changed_count": 3,
            },
        },
        "metadata_after": endpoint_metadata(document),
        "cleanup": {
            "schema": "openmle_fast_owned_cleanup_evidence_v1",
            "run_id": "openmle-fast-gate-run-1",
            "process_owner": "openmle-fast-gate-owner",
            "client_close_count": 64,
            "owned_processes": [
                {
                    "role": "public-environment",
                    "pid": 1001,
                    "start_ticks": 9001,
                    "run_id": "openmle-fast-gate-run-1",
                    "process_owner": "openmle-fast-gate-owner",
                    "alive": False,
                    "exit_code": -int(signal.SIGTERM),
                    "termination_requested": True,
                },
                {
                    "role": "private-grader",
                    "pid": 1002,
                    "start_ticks": 9002,
                    "run_id": "openmle-fast-gate-run-1",
                    "process_owner": "openmle-fast-gate-owner",
                    "alive": False,
                    "exit_code": 0,
                    "termination_requested": True,
                },
            ],
            "process_census": {
                "complete": True,
                "run_id": "openmle-fast-gate-run-1",
                "process_owner": "openmle-fast-gate-owner",
                "matched_pids": [1001, 1002],
                "residual_pids": [],
            },
            "checkpoint_disposition": {
                "schema": "openmle_fast_gate_checkpoint_disposition_v1",
                "policy": "discard_after_readback",
                "checkpoint_reuse_allowed": False,
                "remaining_checkpoint_paths": [],
            },
        },
    }


def endpoint_probe(document: dict) -> dict:
    return {
        "schema": "openmle_fast_resident_endpoint_probe_v1",
        "status": "pass",
        "manifest_sha256": manifest_sha256(document),
        "probe_indices": [0, 63],
        "reset_count": 4,
        "slot_cleanup_count": 3,
        "idempotent_close_verified": True,
        "metadata_before": endpoint_metadata(document),
        "metadata_active": endpoint_metadata(document, active_count=2),
        "metadata_after": endpoint_metadata(document),
    }


def rebind_manifest_digest(evidence: dict, probe: dict, digest_value: str) -> None:
    evidence["manifest_sha256"] = digest_value
    evidence["metadata_after"]["task_manifest_sha256"] = digest_value
    for value in evidence["update"]["rows"]:
        value["task_manifest_sha256"] = digest_value
    probe["manifest_sha256"] = digest_value
    for field in ("metadata_before", "metadata_active", "metadata_after"):
        probe[field]["task_manifest_sha256"] = digest_value


def make_first_terminal_ungraded(evidence: dict) -> dict:
    terminal = evidence["update"]["rows"][1]
    action = 'shell_command {"command":"python train.py"}'
    terminal.update(
        {
            "action": action,
            "action_submission": {"raw_policy_output": action},
            "reward": -1.0,
            "terminal_reason": "episode_wall_limit",
            "terminal_classification": None,
            "runtime_success": False,
            "episode_success": False,
            "grade_receipt": None,
        }
    )
    terminal["counters_after"]["grading_count"] = 0
    terminal["counter_delta"]["grading_count"] = 0
    return terminal


class OpenMLEFastPpoGateTests(unittest.TestCase):
    def verify(
        self,
        evidence: dict,
        document: dict | None = None,
        *,
        endpoint_evidence: dict | None = None,
    ) -> dict:
        module = load_module()
        if document is None:
            document = manifest()
        if endpoint_evidence is None:
            endpoint_evidence = endpoint_probe(document)
        return module.verify_ppo_gate(
            evidence,
            document,
            manifest_sha256(document),
            endpoint_evidence,
            expected_outer_commit=OUTER_COMMIT,
            expected_inner_commit=INNER_COMMIT,
            expected_prompt_sha256=PROMPT_SHA256,
            forbidden_canaries=["NEVER_PUBLIC_CANARY"],
        )

    def test_accepts_exactly_one_complete_g64_update(self) -> None:
        document = manifest()
        attestation = self.verify(update_evidence(document), document)

        self.assertEqual(attestation["status"], "pass")
        self.assertEqual(attestation["optimizer_update_count"], 1)
        self.assertEqual(attestation["task_receipt_count"], 64)
        self.assertEqual(attestation["sampled_action_row_count"], 128)
        self.assertEqual(attestation["graded_terminal_count"], 64)
        self.assertEqual(attestation["ungraded_policy_terminal_count"], 0)
        self.assertFalse(attestation["checkpoint_reuse_allowed"])

    def test_accepts_a_real_policy_terminal_without_forcing_submit(self) -> None:
        document = manifest()
        evidence = update_evidence(document)
        make_first_terminal_ungraded(evidence)

        attestation = self.verify(evidence, document)

        self.assertEqual(attestation["graded_terminal_count"], 63)
        self.assertEqual(attestation["ungraded_policy_terminal_count"], 1)

    def test_accepts_source_backed_policy_terminal_reasons(self) -> None:
        document = manifest()
        policy_reasons = (
            "managed_runtime_limit",
            "episode_wall_limit",
            "wall_timeout",
            "output_limit",
            "memory_limit",
            "pid_limit",
            "disk_limit",
            "file_limit",
            "security_violation",
            "background_process",
            "surviving_background_process",
            "immutable_public_tree_changed",
            "workspace_invariant_violation",
            "policy_resource_violation",
        )
        for reason in policy_reasons:
            with self.subTest(reason=reason):
                evidence = update_evidence(document)
                terminal = make_first_terminal_ungraded(evidence)
                terminal["terminal_reason"] = reason
                attestation = self.verify(evidence, document)
                self.assertEqual(attestation["ungraded_policy_terminal_count"], 1)

    def test_rejects_unrequested_or_forced_signal_cleanup(self) -> None:
        document = manifest()
        for exit_code, termination_requested in (
            (-int(signal.SIGTERM), False),
            (-int(signal.SIGKILL), True),
        ):
            with self.subTest(
                exit_code=exit_code,
                termination_requested=termination_requested,
            ):
                evidence = update_evidence(document)
                process = evidence["cleanup"]["owned_processes"][0]
                process["exit_code"] = exit_code
                process["termination_requested"] = termination_requested
                with self.assertRaisesRegex(AssertionError, "clean managed exit"):
                    self.verify(evidence, document)

    def test_rejects_inconsistent_ungraded_policy_terminals(self) -> None:
        document = manifest()
        for mutation in ("reason", "reward", "fabricated_submission"):
            evidence = update_evidence(document)
            terminal = make_first_terminal_ungraded(evidence)
            if mutation == "reason":
                terminal["terminal_reason"] = "grader_infrastructure_fault"
            elif mutation == "reward":
                terminal["reward"] = 0.0
            else:
                terminal["action_submission"]["request_id"] = "fabricated"
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                self.verify(evidence, document)

    def test_rejects_non_gate_manifest_or_resumable_gate(self) -> None:
        document = manifest()
        for field, value in (("role", "train_pool"), ("panel_id", "other-panel")):
            mutated = json.loads(json.dumps(document))
            mutated[field] = value
            if field == "role":
                for record in mutated["records"]:
                    record["role"] = value
            evidence = update_evidence(mutated)
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.verify(
                    evidence, mutated, endpoint_evidence=endpoint_probe(mutated)
                )

        evidence = update_evidence(document)
        evidence["gate_contract"]["resume_checkpoint"] = "/tmp/checkpoint"
        with self.assertRaisesRegex(AssertionError, "resume checkpoint"):
            self.verify(evidence, document)

        evidence = update_evidence(document)
        evidence["cleanup"]["checkpoint_disposition"]["remaining_checkpoint_paths"] = [
            "global_step_1"
        ]
        with self.assertRaisesRegex(AssertionError, "remaining gate checkpoints"):
            self.verify(evidence, document)

    def test_rejects_a_missing_manifest_task_receipt(self) -> None:
        document = manifest()
        evidence = update_evidence(document)
        evidence["update"]["rows"] = [
            value for value in evidence["update"]["rows"] if value["data_idx"] != 63
        ]
        with self.assertRaisesRegex(AssertionError, "all manifest tasks"):
            self.verify(evidence, document)

    def test_rejects_sampled_and_packed_token_or_logprob_drift(self) -> None:
        document = manifest()
        for field, value in (
            ("generation_token_ids_are_exact", 1),
            ("packed_token_ids", [11, 12, 999, 2001]),
            ("response_mask", [0, 0, 1, 0]),
            ("packed_old_logprobs", [0.0, 0.0, -9.0, -0.2001]),
        ):
            with self.subTest(field=field):
                evidence = update_evidence(document)
                evidence["update"]["rows"][0][field] = value
                with self.assertRaises(AssertionError):
                    self.verify(evidence, document)

    def test_rejects_nonterminal_grading_or_reward(self) -> None:
        document = manifest()
        evidence = update_evidence(document)
        evidence["update"]["rows"][0]["reward"] = 0.5
        evidence["update"]["rows"][0]["grade_receipt"] = {
            "terminal": False,
            "reward": 0.5,
        }
        evidence["update"]["rows"][0]["counters_after"]["grading_count"] = 1
        evidence["update"]["rows"][0]["counter_delta"]["grading_count"] = 1
        with self.assertRaisesRegex(AssertionError, "terminal-only"):
            self.verify(evidence, document)

    def test_rejects_counter_regression(self) -> None:
        document = manifest()
        evidence = update_evidence(document)
        evidence["update"]["rows"][1]["counters_before"][
            "execution_completed_count"
        ] = 0
        with self.assertRaisesRegex(AssertionError, "counter continuity"):
            self.verify(evidence, document)

    def test_rejects_extra_updates_or_zero_parameter_movement(self) -> None:
        document = manifest()
        cases = (
            ("optimizer_update_count", 2),
            ("parameter_delta_l2", 0.0),
            ("max_abs_delta", float("nan")),
        )
        for field, value in cases:
            with self.subTest(field=field):
                evidence = update_evidence(document)
                if field == "optimizer_update_count":
                    evidence["update"][field] = value
                else:
                    evidence["optimizer_readback"]["actor"][field] = value
                with self.assertRaises(AssertionError):
                    self.verify(evidence, document)

    def test_rejects_trajectories_beyond_the_action_horizon(self) -> None:
        document = manifest()
        evidence = update_evidence(document)
        record = document["records"][0]
        evidence["update"]["rows"] = [
            value for value in evidence["update"]["rows"] if value["data_idx"] != 0
        ] + [
            row(
                record,
                task_round,
                terminal=task_round == 31,
                manifest_digest=manifest_sha256(document),
            )
            for task_round in range(1, 32)
        ]
        with self.assertRaisesRegex(AssertionError, "max_policy_actions"):
            self.verify(evidence, document)

    def test_rejects_resource_residue_and_private_canaries(self) -> None:
        document = manifest()
        for mutation in (
            "active_slot",
            "alive_process",
            "non_boolean_alive",
            "private_canary",
        ):
            with self.subTest(mutation=mutation):
                evidence = update_evidence(document)
                if mutation == "active_slot":
                    evidence["metadata_after"]["active_slot_count"] = 1
                elif mutation == "alive_process":
                    evidence["cleanup"]["owned_processes"][0]["alive"] = True
                elif mutation == "non_boolean_alive":
                    evidence["cleanup"]["owned_processes"][0]["alive"] = 0
                else:
                    evidence["metadata_after"]["private_path"] = (
                        "/private/NEVER_PUBLIC_CANARY"
                    )
                with self.assertRaises(AssertionError):
                    self.verify(evidence, document)

    def test_accepts_compound_execution_and_fit_counter_multiplicity(self) -> None:
        document = manifest()
        evidence = update_evidence(document)
        first, terminal = evidence["update"]["rows"][:2]
        first["counters_after"].update(
            {
                "execution_action_count": 1,
                "execution_attempt_count": 3,
                "execution_completed_count": 2,
                "nested_subprocess_count": 5,
                "fit_count": 4,
            }
        )
        first["counter_delta"] = {
            key: first["counters_after"][key] - first["counters_before"][key]
            for key in first["counters_before"]
        }
        terminal["counters_before"] = dict(first["counters_after"])
        terminal["counters_after"] = dict(terminal["counters_before"])
        terminal["counters_after"]["action_count"] += 1
        terminal["counters_after"]["grading_count"] += 1
        terminal["counter_delta"] = {
            key: terminal["counters_after"][key] - terminal["counters_before"][key]
            for key in terminal["counters_before"]
        }

        attestation = self.verify(evidence, document)

        self.assertEqual(attestation["status"], "pass")

    def test_rejects_stale_terminal_grade_bindings(self) -> None:
        document = manifest()
        for field, stale in (
            ("request_id", "stale-request"),
            ("episode_id", "stale-episode"),
            ("submission_sha256", "f" * 64),
        ):
            with self.subTest(field=field):
                evidence = update_evidence(document)
                evidence["update"]["rows"][1]["action_submission"][field] = stale
                with self.assertRaisesRegex(AssertionError, field):
                    self.verify(evidence, document)

    def test_rejects_counter_delta_mismatch(self) -> None:
        document = manifest()
        evidence = update_evidence(document)
        evidence["update"]["rows"][0]["counter_delta"]["execution_attempt_count"] += 1
        with self.assertRaisesRegex(AssertionError, "counter_delta"):
            self.verify(evidence, document)

    def test_rejects_wrong_endpoint_runtime_commit(self) -> None:
        document = manifest()
        probe = endpoint_probe(document)
        probe["metadata_after"]["runtime_source"]["outer_commit"] = "9" * 40
        with self.assertRaisesRegex(AssertionError, "outer_commit"):
            self.verify(
                update_evidence(document),
                document,
                endpoint_evidence=probe,
            )

    def test_rejects_incomplete_owned_process_census(self) -> None:
        document = manifest()
        evidence = update_evidence(document)
        evidence["cleanup"]["process_census"]["complete"] = False
        with self.assertRaisesRegex(AssertionError, "census completeness"):
            self.verify(evidence, document)

    def test_cli_uses_exact_manifest_bytes_and_mode_0600_canaries(self) -> None:
        module = load_module()
        document = manifest()
        raw_manifest = (json.dumps(document, indent=2) + "\n").encode("utf-8")
        digest_value = hashlib.sha256(raw_manifest).hexdigest()
        evidence = update_evidence(document)
        probe = endpoint_probe(document)
        rebind_manifest_digest(evidence, probe, digest_value)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            manifest_path = root / "manifest.json"
            evidence_path = root / "evidence.json"
            probe_path = root / "endpoint.json"
            canaries_path = root / "canaries.json"
            output_path = root / "attestation.json"
            manifest_path.write_bytes(raw_manifest)
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            probe_path.write_text(json.dumps(probe), encoding="utf-8")
            canaries_path.write_text(
                json.dumps(["NEVER_PUBLIC_CANARY"]), encoding="utf-8"
            )
            canaries_path.chmod(0o600)
            argv = [
                str(SCRIPT),
                "--evidence",
                str(evidence_path),
                "--manifest",
                str(manifest_path),
                "--manifest-sha256",
                digest_value,
                "--endpoint-probe",
                str(probe_path),
                "--expected-outer-commit",
                OUTER_COMMIT,
                "--expected-inner-commit",
                INNER_COMMIT,
                "--expected-prompt-sha256",
                PROMPT_SHA256,
                "--forbidden-canaries-file",
                str(canaries_path),
                "--output",
                str(output_path),
            ]
            with mock.patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                module.main()

            self.assertEqual(json.loads(output_path.read_text())["status"], "pass")


if __name__ == "__main__":
    unittest.main()
