from __future__ import annotations

import json
import math
import shutil
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentmemorygym_verl.update1_token_credit_gate import (
    ExpectedRunIdentity,
    GateFailure,
    GatePending,
    OwnerAuthenticationError,
    audit_update1,
    authenticate_owner,
    run_update1_gate,
)
from finalizer_fixture import build_valid_multitask_run, mutate_json, sha256

ROLLOUT_CORRECTION_SUFFIXES = (
    "kl",
    "k3_kl",
    "log_ppl_abs_diff",
    "rollout_is_oob_ratio",
    "rollout_is_ratio_fraction_high",
    "rollout_is_ratio_fraction_low",
    "rollout_is_mean",
    "rollout_is_min",
    "rollout_is_max",
    "rollout_is_std",
    "rollout_is_eff_sample_size",
)


def make_current_owner_metrics(fixture: dict) -> None:
    original = json.loads(
        fixture["metrics_path"].read_text(encoding="utf-8").splitlines()[0]
    )["data"]
    learner = dict(original)
    for suffix in ROLLOUT_CORRECTION_SUFFIXES:
        learner[f"actor/rollout_corr/{suffix}"] = learner.pop(f"rollout_corr/{suffix}")
    # Current learner rows may contain stale rollouter-owned copies.  The gate
    # must select the distinct rollouter row instead.
    learner["fully_async/count/total_generated_samples"] = 0
    learner["fully_async/count/dropped_stale_samples"] = 0
    rollouter = {
        "fully_async/rollouter/active_time": 1.0,
        "fully_async/rollouter/idle_ratio": 0.0,
        "fully_async/rollouter/step_generated_samples": 64,
        "fully_async/rollouter/version_time": 1.0,
        "fully_async/count/total_generated_samples": 64,
        "fully_async/count/rollout_dispatched_samples": 64,
        "fully_async/count/rollout_inflight_samples": 0,
        "fully_async/count/rollout_completed_samples": 64,
        "fully_async/count/rollout_failed_samples": 0,
        "fully_async/count/rollout_cancelled_samples": 0,
        "fully_async/count/queue_enqueued_samples": 64,
        "fully_async/count/queue_dequeued_samples": 64,
        "fully_async/count/queue_overflow_evictions": 0,
        "fully_async/count/queue_cleared_samples": 0,
        "fully_async/count/queue_resident_samples": 0,
        "fully_async/count/staleness_samples": 0,
        "fully_async/count/dropped_stale_samples": 0,
    }
    rows = (
        {
            "step": 0,
            "data": {
                **rollouter,
                "fully_async/rollouter/step_generated_samples": 0,
            },
        },
        {"step": 1, "data": rollouter},
        {"step": 1, "data": learner},
    )
    fixture["metrics_path"].write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def mutate_metric_owner(fixture: dict, owner: str, mutation) -> None:
    rows = [
        json.loads(line)
        for line in fixture["metrics_path"].read_text(encoding="utf-8").splitlines()
    ]
    candidates = []
    for row in rows:
        if row["step"] != 1:
            continue
        data = row["data"]
        is_rollouter = any(key.startswith("fully_async/rollouter/") for key in data)
        if (owner == "rollouter") == is_rollouter:
            candidates.append(data)
    if len(candidates) != 1:
        raise AssertionError(f"expected one {owner} row, got {len(candidates)}")
    mutation(candidates[0])
    fixture["metrics_path"].write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_fake_proc(
    proc_root: Path,
    *,
    pid: int,
    start_ticks: str,
    pgrp: int,
    session: int,
    argv: list[str],
    environment: list[str],
) -> None:
    root = proc_root / str(pid)
    root.mkdir(parents=True, exist_ok=True)
    # Fields after ``comm`` are proc fields 3..22.  starttime is index 19.
    fields = [
        "S",
        "1",
        str(pgrp),
        str(session),
        *("0" for _ in range(15)),
        start_ticks,
    ]
    (root / "stat").write_text(
        f"{pid} (python3) " + " ".join(fields) + "\n", encoding="utf-8"
    )
    (root / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")
    (root / "environ").write_bytes(
        b"\0".join(value.encode() for value in environment) + b"\0"
    )


class GateFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_id = "multitask-fixture"
        self.run_dir = self.root / self.run_id
        self.fixture = build_valid_multitask_run(self.run_dir, updates=1)
        make_current_owner_metrics(self.fixture)
        self.expected = ExpectedRunIdentity(
            run_id=self.run_id,
            outer_commit="c" * 40,
            inner_commit="d" * 40,
            verl_commit="7f359d928fa438c9353c0c6d98941c4638971f05",
        )
        self.owner_path = (
            self.root / "formal-owner" / "orchestrator-process-identity.json"
        )
        self.owner_path.parent.mkdir()
        self.proc_root = self.root / "proc"
        self.pid = 4242
        self.start_ticks = "987654"
        self.bootstrap = "/opt/amg/process_bootstrap.py"
        self.command = [
            "/opt/amg/launch_amg_multitask_fully_async.sh",
            "--config",
            "/opt/amg/amg_multitask200.yaml",
            "--run-dir",
            str(self.run_dir),
            "--experiment-name",
            self.run_id,
        ]
        self.owner_payload = {
            "schema": "amg_fallback_managed_process_identity_v2",
            "name": "outer-multitask-orchestrator",
            "pid": self.pid,
            "start_ticks": self.start_ticks,
            "process_group": self.pid,
            "bootstrap": self.bootstrap,
            "command": self.command,
            "immutable_contract": {"holder_lease_sha256": "a" * 64},
        }
        self._write_owner()
        self.output = self.run_dir / "gates" / "update1-token-credit.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_owner(
        self,
        *,
        pgrp: int | None = None,
        session: int | None = None,
        start_ticks: str | None = None,
        argv: list[str] | None = None,
        environment: list[str] | None = None,
    ) -> None:
        self.owner_path.write_text(
            json.dumps(self.owner_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_fake_proc(
            self.proc_root,
            pid=self.pid,
            start_ticks=start_ticks or self.start_ticks,
            pgrp=self.pid if pgrp is None else pgrp,
            session=self.pid if session is None else session,
            argv=argv
            or [
                "/usr/bin/python3",
                self.bootstrap,
                "--ack-fd",
                "7",
                "--",
                *self.command,
            ],
            environment=environment
            or [
                f"AMG_MULTITASK_RUN_ID={self.run_id}",
                f"AGENTMEMORY_RUN_ID={self.run_id}",
                "PATH=/usr/bin",
            ],
        )

    def audit(self) -> dict:
        return audit_update1(self.run_dir, self.expected)

    def run_gate(self, **kwargs) -> dict:
        return run_update1_gate(
            self.run_dir,
            self.expected,
            owner_receipt=self.owner_path,
            output_path=self.output,
            timeout_seconds=0,
            poll_seconds=0.001,
            stop_timeout_seconds=0,
            proc_root=self.proc_root,
            **kwargs,
        )


class TestUpdate1Evidence(GateFixture):
    def test_pass_covers_all_required_update1_evidence(self):
        evidence = self.audit()

        self.assertEqual(sum(evidence["route_episodes"].values()), 64)
        self.assertTrue(all(evidence["route_episodes"].values()))
        self.assertEqual(
            evidence["token_credit"]["optimizer_execution"]["actor"][
                "ppo_epoch_passes_delta"
            ],
            1,
        )
        self.assertEqual(
            evidence["token_credit"]["optimizer_execution"]["critic"][
                "ppo_epoch_passes_delta"
            ],
            2,
        )
        self.assertEqual(
            evidence["token_credit"]["publication"]["published_parameter_version"],
            1,
        )
        self.assertTrue(
            evidence["critic_parameter_freeze"]["optimizer_membership_exact"]
        )
        self.assertEqual(
            evidence["critic_parameter_freeze"]["frozen_delta"]["changed_count"],
            0,
        )
        self.assertGreater(
            evidence["critic_parameter_freeze"]["trainable_delta"]["changed_count"],
            0,
        )
        self.assertTrue(
            math.isclose(
                evidence["critic_parameter_freeze"]["optimizer_step_learning_rates"][0][
                    0
                ],
                5e-7,
            )
        )

    def test_zero_or_nonfinite_gradients_fail(self):
        for label, role, value in (
            ("zero_actor", "actor", 0.0),
            ("nan_critic", "critic", float("nan")),
        ):
            with self.subTest(case=label):
                original = self.fixture["metrics_path"].read_bytes()
                mutate_metric_owner(
                    self.fixture,
                    "learner",
                    lambda data, role=role, value=value: data.update(
                        {f"{role}/grad_norm": value}
                    ),
                )
                with self.assertRaisesRegex(GateFailure, "finite nonzero"):
                    self.audit()
                self.fixture["metrics_path"].write_bytes(original)

    def test_actor_k1_critic_k2_execution_failures(self):
        cases = (
            ("actor/ppo_epoch_passes_delta", 2),
            ("critic/ppo_epoch_passes_delta", 1),
            ("critic/mini_batches_per_epoch", 2),
            ("critic/optimizer_steps_delta", 1),
        )
        for key, value in cases:
            with self.subTest(key=key):
                original = self.fixture["metrics_path"].read_bytes()
                mutate_metric_owner(
                    self.fixture,
                    "learner",
                    lambda data, key=key, value=value: data.update({key: value}),
                )
                with self.assertRaisesRegex(GateFailure, "actor-K1/critic-K2"):
                    self.audit()
                self.fixture["metrics_path"].write_bytes(original)

    def test_icepop_missing_and_invalid_telemetry_fail(self):
        cases = (
            (
                "missing",
                lambda data: data.pop("actor/rollout_corr/kl"),
                "lacks finite",
            ),
            (
                "zero_ess",
                lambda data: data.update(
                    {"actor/rollout_corr/rollout_is_eff_sample_size": 0.0}
                ),
                "ESS",
            ),
            (
                "oob_identity",
                lambda data: data.update(
                    {"actor/rollout_corr/rollout_is_oob_ratio": 0.16}
                ),
                "high\\+low",
            ),
            (
                "weight_order",
                lambda data: data.update({"actor/rollout_corr/rollout_is_min": 1.1}),
                "applied weights",
            ),
        )
        for label, mutation, message in cases:
            with self.subTest(case=label):
                original = self.fixture["metrics_path"].read_bytes()
                mutate_metric_owner(self.fixture, "learner", mutation)
                with self.assertRaisesRegex(GateFailure, message):
                    self.audit()
                self.fixture["metrics_path"].write_bytes(original)

    def test_freeze_membership_delta_and_lr_fail_closed(self):
        cases = (
            (
                "taxonomy",
                lambda value: value.update(policy="unexpected_freeze_policy"),
                "freeze policy",
            ),
            (
                "optimizer_membership",
                lambda value: value["optimizer"].update(membership_exact=False),
                "optimizer membership",
            ),
            (
                "frozen_changed",
                lambda value: value["first_optimizer_step"]["frozen"].update(
                    changed_count=1, l2=0.1, max_abs=0.1
                ),
                "frozen parameter probe changed",
            ),
            (
                "trainable_unchanged",
                lambda value: value["first_optimizer_step"]["trainable"].update(
                    changed_count=0, l2=0.0, max_abs=0.0
                ),
                "trainable parameter probe did not change",
            ),
            (
                "lr_step_two",
                lambda value: value["optimizer_step_learning_rates"]["steps"][1].update(
                    learning_rates=[2e-6]
                ),
                "step 2 learning rates differ",
            ),
        )
        for label, mutation, message in cases:
            with self.subTest(case=label):
                original = self.fixture["freeze_path"].read_bytes()
                mutate_json(self.fixture["freeze_path"], mutation)
                with self.assertRaisesRegex(GateFailure, message):
                    self.audit()
                self.fixture["freeze_path"].write_bytes(original)

    def test_source_and_config_identity_mismatch_fail(self):
        wrong = ExpectedRunIdentity(
            self.run_id,
            "e" * 40,
            self.expected.inner_commit,
            self.expected.verl_commit,
        )
        with self.assertRaisesRegex(GateFailure, "source identity mismatch"):
            audit_update1(self.run_dir, wrong)

        launch = json.loads(self.fixture["launch_path"].read_text(encoding="utf-8"))
        launch["inputs"]["experiment_name"] = "another-run"
        self.fixture["launch_path"].write_text(
            json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(GateFailure, "run_id mismatch"):
            self.audit()

    def test_resolved_config_drift_fails(self):
        payload = self.fixture["resolved_path"].read_text(encoding="utf-8")
        self.fixture["resolved_path"].write_text(
            payload.replace("lr: 5.0e-06", "lr: 6.0e-06"), encoding="utf-8"
        )
        with self.assertRaisesRegex(GateFailure, "identity/config audit failed"):
            self.audit()

    def test_each_of_four_routes_must_appear_at_update1(self):
        shutil.rmtree(self.run_dir)
        self.fixture = build_valid_multitask_run(
            self.run_dir,
            updates=2,
            route_counts_by_update=[
                {
                    "webshop": 0,
                    "swesmith": 16,
                    "literesearcher": 16,
                    "openmle_fast": 32,
                },
                {
                    "webshop": 32,
                    "swesmith": 16,
                    "literesearcher": 16,
                    "openmle_fast": 0,
                },
            ],
        )
        make_current_owner_metrics(self.fixture)
        with self.assertRaisesRegex(GateFailure, "routes are absent"):
            self.audit()

    def test_63_episodes_remain_pending_and_timeout_to_failure(self):
        rollout = self.fixture["rollout_dir"] / "1.jsonl"
        rows = [json.loads(line) for line in rollout.read_text().splitlines()]
        first_uid = json.loads(rows[0]["step_record_json"])["trajectory_uid"]
        rows = [
            row
            for row in rows
            if json.loads(row["step_record_json"])["trajectory_uid"] != first_uid
        ]
        rollout.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(GatePending, "63/64"):
            self.audit()

        calls = []
        receipt = self.run_gate(
            dry_run=True,
            signal_owner=lambda identity, signum: calls.append((identity, signum)),
        )
        self.assertEqual(receipt["status"], "fail")
        self.assertIn("timed out", receipt["errors"][0])
        self.assertEqual(receipt["termination"]["status"], "suppressed-by-dry-run")
        self.assertEqual(calls, [])

    def test_publication_and_zero_failure_counters_fail(self):
        cases = (
            (
                "consumed_version",
                "learner",
                lambda data: data.update(
                    {"fully_async/count/current_param_version": 1}
                ),
                "consumed parameter version",
            ),
            (
                "version_time",
                "rollouter",
                lambda data: data.update({"fully_async/rollouter/version_time": 0.0}),
                "version_time",
            ),
            (
                "rollout_failed",
                "rollouter",
                lambda data: data.update(
                    {"fully_async/count/rollout_failed_samples": 1}
                ),
                "rollout_failed_samples",
            ),
            (
                "cancelled",
                "rollouter",
                lambda data: data.update(
                    {"fully_async/count/rollout_cancelled_samples": 1}
                ),
                "rollout_cancelled_samples",
            ),
            (
                "overflow",
                "rollouter",
                lambda data: data.update(
                    {"fully_async/count/queue_overflow_evictions": 1}
                ),
                "queue_overflow_evictions",
            ),
            (
                "stale_drop",
                "rollouter",
                lambda data: data.update(
                    {"fully_async/count/dropped_stale_samples": 1}
                ),
                "dropped_stale_samples",
            ),
        )
        for label, owner, mutation, message in cases:
            with self.subTest(case=label):
                original = self.fixture["metrics_path"].read_bytes()
                mutate_metric_owner(self.fixture, owner, mutation)
                with self.assertRaisesRegex(GateFailure, message):
                    self.audit()
                self.fixture["metrics_path"].write_bytes(original)


class TestOwnerAuthentication(GateFixture):
    def test_exact_owner_receipt_pid_group_session_command_and_env_pass(self):
        lease = authenticate_owner(
            self.owner_path, self.run_dir, self.run_id, proc_root=self.proc_root
        )
        self.assertEqual(lease["identity"]["pid"], self.pid)
        self.assertEqual(lease["identity"]["pgrp"], self.pid)
        self.assertEqual(lease["identity"]["session"], self.pid)
        self.assertEqual(lease["binding"]["sha256"], sha256(self.owner_path))

    def test_identity_command_and_environment_mismatches_fail(self):
        cases = (
            (
                "start_ticks",
                lambda: self._write_owner(start_ticks="1"),
                "identity mismatch",
            ),
            (
                "process_group",
                lambda: self._write_owner(pgrp=self.pid + 1),
                "identity mismatch",
            ),
            (
                "session",
                lambda: self._write_owner(session=self.pid + 1),
                "identity mismatch",
            ),
            (
                "command",
                lambda: self._write_owner(
                    argv=[
                        "/usr/bin/python3",
                        self.bootstrap,
                        "--",
                        "/bin/false",
                    ]
                ),
                "command line",
            ),
            (
                "environment",
                lambda: self._write_owner(
                    environment=[f"AMG_MULTITASK_RUN_ID={self.run_id}"]
                ),
                "environment",
            ),
        )
        for label, mutation, message in cases:
            with self.subTest(case=label):
                shutil.rmtree(self.proc_root, ignore_errors=True)
                mutation()
                with self.assertRaisesRegex(OwnerAuthenticationError, message):
                    authenticate_owner(
                        self.owner_path,
                        self.run_dir,
                        self.run_id,
                        proc_root=self.proc_root,
                    )


class TestStopAndReceiptSemantics(GateFixture):
    def test_pass_atomically_replaces_receipt_without_signalling_or_mutating_inputs(
        self,
    ):
        self.output.parent.mkdir()
        self.output.write_text("old\n", encoding="utf-8")
        protected = {
            path: sha256(path)
            for path in (
                self.fixture["launch_path"],
                self.fixture["metrics_path"],
                self.fixture["freeze_path"],
                self.fixture["rollout_dir"] / "1.jsonl",
                self.owner_path,
            )
        }
        calls = []

        receipt = self.run_gate(
            dry_run=False,
            signal_owner=lambda identity, signum: calls.append((identity, signum)),
        )

        self.assertEqual(receipt["decision"], "PASS_CONTINUE")
        self.assertEqual(calls, [])
        self.assertEqual(json.loads(self.output.read_text()), receipt)
        self.assertEqual(protected, {path: sha256(path) for path in protected})
        self.assertFalse(any(self.output.parent.glob(f".{self.output.name}.*.tmp")))

    def test_authenticated_failure_signals_only_exact_identity(self):
        mutate_metric_owner(
            self.fixture,
            "learner",
            lambda data: data.update({"actor/grad_norm": 0.0}),
        )
        calls = []

        def stop(identity, signum):
            calls.append((dict(identity), signum))
            shutil.rmtree(self.proc_root / str(self.pid))
            return True

        receipt = self.run_gate(dry_run=False, signal_owner=stop)

        self.assertEqual(receipt["status"], "fail")
        self.assertEqual(
            calls,
            [
                (
                    {
                        "pid": self.pid,
                        "start_ticks": self.start_ticks,
                        "pgrp": self.pid,
                        "session": self.pid,
                    },
                    signal.SIGTERM,
                )
            ],
        )
        self.assertEqual(receipt["termination"]["status"], "confirmed-stopped")
        self.assertEqual(receipt["termination"]["scope"], "exact-owner-identity")

    def test_dry_run_failure_never_signals(self):
        mutate_metric_owner(
            self.fixture,
            "learner",
            lambda data: data.update({"critic/grad_norm": 0.0}),
        )
        calls = []
        receipt = self.run_gate(
            dry_run=True,
            signal_owner=lambda identity, signum: calls.append((identity, signum)),
        )
        self.assertEqual(receipt["status"], "fail")
        self.assertEqual(calls, [])
        self.assertEqual(receipt["termination"]["status"], "suppressed-by-dry-run")

    def test_owner_receipt_drift_withholds_signal(self):
        mutate_metric_owner(
            self.fixture,
            "learner",
            lambda data: data.update({"actor/grad_norm": 0.0}),
        )
        first = authenticate_owner(
            self.owner_path, self.run_dir, self.run_id, proc_root=self.proc_root
        )
        second = json.loads(json.dumps(first))
        second["binding"]["inode"] += 1
        calls = []
        with mock.patch(
            "agentmemorygym_verl.update1_token_credit_gate.authenticate_owner",
            side_effect=[first, second],
        ):
            receipt = self.run_gate(
                dry_run=False,
                signal_owner=lambda identity, signum: calls.append((identity, signum)),
            )
        self.assertEqual(receipt["status"], "fail")
        self.assertEqual(calls, [])
        self.assertEqual(
            receipt["termination"]["status"],
            "unsafe-to-stop-owner-not-authenticated",
        )
        self.assertIn("drifted before stop", receipt["errors"][0])

    def test_output_cannot_overlap_launch_bound_input(self):
        with self.assertRaisesRegex(GateFailure, "overlaps"):
            run_update1_gate(
                self.run_dir,
                self.expected,
                owner_receipt=self.owner_path,
                output_path=self.fixture["metrics_path"],
                timeout_seconds=0,
                poll_seconds=0.001,
                dry_run=True,
                proc_root=self.proc_root,
            )

    def test_output_cannot_overwrite_owner_identity_receipt(self):
        inside_owner = self.run_dir / "formal-owner-identity.json"
        inside_owner.write_bytes(self.owner_path.read_bytes())
        original = inside_owner.read_bytes()

        with self.assertRaisesRegex(GateFailure, "owner identity receipt"):
            run_update1_gate(
                self.run_dir,
                self.expected,
                owner_receipt=inside_owner,
                output_path=inside_owner,
                timeout_seconds=0,
                poll_seconds=0.001,
                dry_run=True,
                proc_root=self.proc_root,
            )

        self.assertEqual(inside_owner.read_bytes(), original)

    def test_source_has_no_broad_process_kill_primitive(self):
        source = (
            Path(__file__).parents[1]
            / "agentmemorygym_verl"
            / "update1_token_credit_gate.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("pk" + "ill", source)
        self.assertNotIn("kill" + "pg", source)
        self.assertNotIn("os." + "kill", source)


if __name__ == "__main__":
    unittest.main()
