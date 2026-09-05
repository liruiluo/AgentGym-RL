from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agentmemorygym_verl.heldout_eval_orchestrator import (
    ATTEMPT_SCHEMA,
    EvalAttempt,
    HeldoutEvalLocalBackend,
    _attempt_owner_id,
    _remove_empty_runtime_root,
    _verify_endpoint_task_counts,
    classify_inference_argv,
    execute_eval_orchestrator,
)
from agentmemorygym_verl.multitask_orchestrator import OrchestratorError


class _Backend:
    def __init__(self, root: Path, *, stage: str | None = None) -> None:
        self.root = root
        self.stage = stage
        self.events: list[str] = []
        self.evaluator = None
        self.endpoint_leases = ()
        self.holder_handle = None
        self.watch_parent = None

    def resolve(self, _plan):
        self.events.append("resolve")
        directory = self.root / "attempt-000000"
        directory.mkdir()
        return EvalAttempt(
            index=0,
            directory=directory,
            runtime_plan=object(),
            already_complete=False,
            owner_id="eval-run.attempt-000000",
        )

    def prepare_runtime(self, _plan, _attempt):
        self.events.append("prepare")
        if self.stage == "prepare":
            raise RuntimeError("prepare failed")

    def acquire_holders(self, _plan, _attempt):
        self.events.append("holder")
        self.holder_handle = object()
        if self.stage == "holder":
            raise RuntimeError("holder failed")
        return self.holder_handle

    def start_endpoints(self, _plan, _attempt):
        self.events.append("endpoints")
        self.endpoint_leases = (object(),)
        if self.stage == "endpoints":
            raise RuntimeError("endpoints failed")
        return self.endpoint_leases

    def start_watch_parent(self, _plan, _attempt):
        self.events.append("watch-parent")
        self.watch_parent = object()
        if self.stage == "watch-parent":
            raise RuntimeError("watch-parent failed")
        return self.watch_parent

    def start_evaluator(self, _plan, _attempt):
        self.events.append("evaluator")
        self.evaluator = object()
        if self.stage == "evaluator":
            raise RuntimeError("evaluator failed")
        return self.evaluator

    def wait_evaluator(self, _plan, _attempt, _evaluator, _endpoints, _holder):
        self.events.append("wait")
        if self.stage == "wait":
            raise RuntimeError("wait failed")
        return 0

    def stop_evaluator(self, _plan, _evaluator):
        self.events.append("stop-evaluator")
        self.evaluator = None

    def stop_watch_parent(self, _plan, _attempt, _watch_parent):
        self.events.append("stop-watch-parent")
        self.watch_parent = None

    def stop_endpoints(self, _plan, _attempt, _endpoints):
        self.events.append("stop-endpoints")
        self.endpoint_leases = ()

    def restore_holders(self, _plan, _attempt, _holder):
        self.events.append("restore-holder")
        self.holder_handle = None

    def cleanup_audit(self, _plan, _attempt):
        self.events.append("cleanup-audit")
        return {"status": "pass"}


def _plan(root: Path, *, resolve_only: bool = False, episode_count: int = 7_777):
    return SimpleNamespace(
        resolve_only=resolve_only,
        evaluation=SimpleNamespace(
            run_dir=root / "evaluation",
            episode_count=episode_count,
        ),
    )


class HeldoutEvalOrchestratorTests(unittest.TestCase):
    def test_cleanup_removes_only_empty_runtime_directory_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            (root / "memory-episodes" / "nested").mkdir(parents=True)
            self.assertTrue(_remove_empty_runtime_root(root))
            self.assertFalse(root.exists())
            self.assertFalse(_remove_empty_runtime_root(root))

    def test_cleanup_refuses_non_directory_runtime_residue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            root.mkdir()
            residue = root / "unexpected.json"
            residue.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                OrchestratorError, "contains non-directory residue"
            ):
                _remove_empty_runtime_root(root)
            self.assertTrue(residue.is_file())

    def test_cleanup_refuses_symlink_runtime_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            root.symlink_to(Path(directory) / "missing", target_is_directory=True)
            with self.assertRaisesRegex(
                OrchestratorError, "is not a real directory"
            ):
                _remove_empty_runtime_root(root)
            self.assertTrue(root.is_symlink())

    def test_cleanup_audit_removes_empty_attempt_runtime_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            orchestration_dir = root / "orchestration"
            attempt_dir = orchestration_dir / "attempts" / "attempt-000000"
            attempt_dir.mkdir(parents=True)
            runtime_root = root / "runtime"
            (runtime_root / "memory-episodes").mkdir(parents=True)
            plan = SimpleNamespace(
                endpoints=(),
                orchestration_dir=orchestration_dir,
                evaluation=SimpleNamespace(run_id="eval-run"),
            )
            attempt = SimpleNamespace(owner_id="eval-run.attempt-000000")
            backend = HeldoutEvalLocalBackend()
            with (
                mock.patch(
                    "agentmemorygym_verl.heldout_eval_orchestrator."
                    "_attempt_runtime_roots",
                    return_value=(runtime_root, root / "missing-runtime"),
                ),
                mock.patch(
                    "agentmemorygym_verl.heldout_eval_orchestrator."
                    "assert_ports_available"
                ),
                mock.patch(
                    "agentmemorygym_verl.heldout_eval_orchestrator."
                    "foreign_inference_processes",
                    return_value=[],
                ),
                mock.patch(
                    "agentmemorygym_verl.heldout_eval_orchestrator."
                    "run_owned_processes",
                    return_value=[],
                ),
                mock.patch(
                    "agentmemorygym_verl.heldout_eval_orchestrator.mounts_below",
                    return_value=[],
                ),
            ):
                receipt = backend.cleanup_audit(plan, attempt)
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(
                receipt["removed_empty_runtime_roots"], [str(runtime_root)]
            )
            self.assertFalse(runtime_root.exists())

    def test_attempt_owner_is_namespaced_by_eval_run(self):
        self.assertEqual(
            _attempt_owner_id("agemem-native-heldout", 0),
            "agemem-native-heldout.attempt-000000",
        )

    def test_endpoint_task_pools_must_cover_the_eval_schedule(self):
        route_counts = {
            "webshop": 128,
            "swesmith": 128,
            "literesearcher": 128,
            "openmle_fast": 128,
        }
        evaluation = SimpleNamespace(route_counts=route_counts)
        endpoint_counts = {
            "webshop": 1746,
            "swesmith": 933,
            "literesearcher": 5319,
            "openmle_fast": 169,
        }
        endpoints = tuple(
            SimpleNamespace(route_id=route_id, task_count=count)
            for route_id, count in endpoint_counts.items()
        )
        self.assertEqual(
            _verify_endpoint_task_counts(evaluation, endpoints), endpoint_counts
        )
        undersized = endpoints[:-1] + (
            SimpleNamespace(route_id="openmle_fast", task_count=127),
        )
        with self.assertRaisesRegex(OrchestratorError, "cannot cover"):
            _verify_endpoint_task_counts(evaluation, undersized)
        missing = endpoints[:-1]
        with self.assertRaisesRegex(OrchestratorError, "route set differs"):
            _verify_endpoint_task_counts(evaluation, missing)
        self.assertEqual(
            _attempt_owner_id("agemem-native-heldout", 17),
            "agemem-native-heldout.attempt-000017",
        )

    def test_process_classifier_ignores_ray_or_vllm_in_file_arguments(self):
        self.assertEqual(
            classify_inference_argv(
                ["/usr/bin/python3", "/tmp/analyze.py", "/tmp/ray/vllm-report.json"]
            ),
            (),
        )
        self.assertEqual(
            classify_inference_argv(
                ["/usr/bin/python3", "-m", "sglang.launch_server", "--port", "1"]
            ),
            ("python_module:sglang.launch_server",),
        )
        self.assertEqual(classify_inference_argv(["raylet", "--node-ip-address=x"]), ("executable:raylet",))

    def test_resolve_only_never_mutates_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = _Backend(root)
            self.assertEqual(
                execute_eval_orchestrator(
                    _plan(root, resolve_only=True), backend=backend
                ),
                0,
            )
            self.assertEqual(backend.events, ["resolve"])

    def test_runtime_failures_unwind_only_acquired_layers_in_reverse_order(self):
        expected = {
            "prepare": ["resolve", "prepare", "cleanup-audit"],
            "holder": [
                "resolve",
                "prepare",
                "holder",
                "restore-holder",
                "cleanup-audit",
            ],
            "watch-parent": [
                "resolve",
                "prepare",
                "holder",
                "watch-parent",
                "stop-watch-parent",
                "restore-holder",
                "cleanup-audit",
            ],
            "endpoints": [
                "resolve",
                "prepare",
                "holder",
                "watch-parent",
                "endpoints",
                "stop-endpoints",
                "stop-watch-parent",
                "restore-holder",
                "cleanup-audit",
            ],
            "evaluator": [
                "resolve",
                "prepare",
                "holder",
                "watch-parent",
                "endpoints",
                "evaluator",
                "stop-evaluator",
                "stop-endpoints",
                "stop-watch-parent",
                "restore-holder",
                "cleanup-audit",
            ],
            "wait": [
                "resolve",
                "prepare",
                "holder",
                "watch-parent",
                "endpoints",
                "evaluator",
                "wait",
                "stop-evaluator",
                "stop-endpoints",
                "stop-watch-parent",
                "restore-holder",
                "cleanup-audit",
            ],
        }
        for stage, events in expected.items():
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                backend = _Backend(root, stage=stage)
                with self.assertRaisesRegex(RuntimeError, f"{stage} failed"):
                    execute_eval_orchestrator(_plan(root), backend=backend)
                self.assertEqual(backend.events, events)
                receipt = json.loads(
                    (root / "attempt-000000/orchestrator-receipt.json").read_text()
                )
                self.assertEqual(receipt["schema"], ATTEMPT_SCHEMA)
                self.assertEqual(receipt["status"], "fail")

    def test_success_waits_then_cleans_and_reverifies_final_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = _Backend(root)
            with mock.patch(
                "agentmemorygym_verl.heldout_eval_orchestrator.finalize_run_metrics",
                return_value={"status": "pass"},
            ) as finalize:
                self.assertEqual(
                    execute_eval_orchestrator(
                        _plan(root, episode_count=7_777), backend=backend
                    ),
                    0,
                )
            self.assertEqual(
                backend.events,
                [
                    "resolve",
                    "prepare",
                    "holder",
                    "watch-parent",
                    "endpoints",
                    "evaluator",
                    "wait",
                    "stop-evaluator",
                    "stop-endpoints",
                    "stop-watch-parent",
                    "restore-holder",
                    "cleanup-audit",
                ],
            )
            finalize.assert_called_once_with(
                root / "evaluation", expected_episode_count=7_777
            )
            receipt = json.loads(
                (root / "attempt-000000/orchestrator-receipt.json").read_text()
            )
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(receipt["evaluator_exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
