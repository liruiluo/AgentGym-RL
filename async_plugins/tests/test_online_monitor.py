from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agentmemorygym_verl.finalizer as finalizer_module
import agentmemorygym_verl.online_monitor as monitor_module
from agentmemorygym_verl.online_monitor import (
    SNAPSHOT_UPDATES,
    observe_run,
    write_snapshot,
)
from finalizer_fixture import (
    MULTITASK_ROUTES,
    build_valid_multitask_run,
    build_valid_run,
    mutate_json,
)


def file_state(root: Path, *, excluding: set[Path] | None = None) -> dict[str, tuple]:
    excluded = {path.resolve() for path in (excluding or set())}
    state = {}
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        if path.resolve() in excluded:
            continue
        stat = path.stat()
        state[str(path.relative_to(root))] = (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            stat.st_mtime_ns,
            stat.st_size,
        )
    return state


def rewrite_first_rollout(run: dict, mutation) -> None:
    path = run["rollout_dir"] / "1.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    document = json.loads(lines[0])
    mutation(document)
    lines[0] = json.dumps(document, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rewrite_metrics(run: dict, mutation) -> None:
    rows = [
        json.loads(line)
        for line in run["metrics_path"].read_text(encoding="utf-8").splitlines()
    ]
    mutation(rows)
    run["metrics_path"].write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


class TestOnlineMonitor(unittest.TestCase):
    def test_route_local_max_rounds_horizon_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            run = build_valid_multitask_run(
                Path(directory),
                updates=1,
                horizon_route_id=MULTITASK_ROUTES[0],
            )

            snapshot = observe_run(run["run_dir"], 1)

        self.assertEqual(snapshot["status"], "descriptive")
        self.assertEqual(
            snapshot["routes"][MULTITASK_ROUTES[0]]["optimizer_consumed_episodes"],
            16,
        )
        self.assertEqual(
            snapshot["routes"][MULTITASK_ROUTES[0]]["native_successes"],
            15,
        )

    def test_supported_snapshots_cover_1_5_20_40_80(self):
        self.assertEqual(SNAPSHOT_UPDATES, frozenset({1, 5, 20, 40, 80}))
        with tempfile.TemporaryDirectory() as directory:
            run = build_valid_multitask_run(Path(directory), updates=80)
            for update in sorted(SNAPSHOT_UPDATES):
                with self.subTest(update=update):
                    snapshot = observe_run(run["run_dir"], update)
                    expected_status = "descriptive" if update in {1, 5} else "pass"
                    self.assertEqual(snapshot["status"], expected_status)
                    self.assertEqual(snapshot["snapshot_update"], update)
                    self.assertEqual(tuple(snapshot["routes"]), MULTITASK_ROUTES)
                    self.assertEqual(
                        sum(
                            route["optimizer_consumed_episodes"]
                            for route in snapshot["routes"].values()
                        ),
                        update * 64,
                    )
                    rolling = snapshot["rolling_8_episode_share"]
                    self.assertEqual(
                        rolling["status"],
                        "not_applicable" if update < 8 else "pass",
                    )
                    self.assertEqual(len(rolling["windows"]), max(0, update - 7))

    def test_update_5_is_descriptive_but_update_20_enforces_rolling_8(self):
        skewed = [
            {
                MULTITASK_ROUTES[0]: 32,
                MULTITASK_ROUTES[1]: 11,
                MULTITASK_ROUTES[2]: 11,
                MULTITASK_ROUTES[3]: 10,
            }
            for _ in range(8)
        ]
        skewed.extend(
            {
                MULTITASK_ROUTES[0]: 6,
                MULTITASK_ROUTES[1]: 20,
                MULTITASK_ROUTES[2]: 18,
                MULTITASK_ROUTES[3]: 20,
            }
            for _ in range(4)
        )
        skewed.extend(
            {
                MULTITASK_ROUTES[0]: 5,
                MULTITASK_ROUTES[1]: 19,
                MULTITASK_ROUTES[2]: 20,
                MULTITASK_ROUTES[3]: 20,
            }
            for _ in range(8)
        )
        with tempfile.TemporaryDirectory() as directory:
            run = build_valid_multitask_run(
                Path(directory), updates=20, route_counts_by_update=skewed
            )
            prefix = observe_run(run["run_dir"], 5)
            enforced = observe_run(run["run_dir"], 20)

        self.assertEqual(prefix["status"], "descriptive")
        self.assertEqual(prefix["rolling_8_episode_share"]["status"], "not_applicable")
        self.assertEqual(enforced["status"], "fail")
        self.assertTrue(enforced["rolling_8_episode_share"]["violations"])
        first = enforced["rolling_8_episode_share"]["windows"][0]
        self.assertEqual((first["start_update"], first["end_update"]), (1, 8))
        self.assertEqual(first["status"], "fail")

    def test_complete_prefix_ignores_only_an_unterminated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            run = build_valid_multitask_run(Path(directory), updates=5)
            path = run["rollout_dir"] / "5.jsonl"
            with path.open("ab") as handle:
                handle.write(b'{"step": 5')
            snapshot = observe_run(run["run_dir"], 5)
            self.assertEqual(snapshot["status"], "descriptive")

            with path.open("ab") as handle:
                handle.write(b"}\n")
            with self.assertRaisesRegex(ValueError, "rollout update 5"):
                observe_run(run["run_dir"], 5)

    def test_live_prefix_does_not_require_future_or_terminal_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = build_valid_multitask_run(Path(directory), updates=80)
            for path in run["rollout_dir"].glob("*.jsonl"):
                if path.name != "1.jsonl":
                    path.unlink()
            first_metric = (
                run["metrics_path"].read_text(encoding="utf-8").splitlines()[0]
            )
            run["metrics_path"].write_bytes(
                (first_metric + "\n").encode() + b'{"step": 2'
            )
            shutil.rmtree(run["checkpoint_root"])
            run["trainer_log"].unlink()
            mutate_json(
                run["launch_path"],
                lambda receipt: receipt["runtime_artifacts"].pop("trainer_log", None),
            )

            snapshot = observe_run(run["run_dir"], 1)

        self.assertEqual(snapshot["status"], "descriptive")
        self.assertEqual(snapshot["snapshot_update"], 1)

    def test_file_logger_and_rollout_mutations_fail_closed(self):
        cases = (
            (
                "FileLogger route total",
                lambda run: rewrite_metrics(
                    run,
                    lambda rows: rows[0]["data"].__setitem__(
                        "fully_async/sum/optimizer_consumed_episodes/data_source/"
                        + MULTITASK_ROUTES[0],
                        rows[0]["data"][
                            "fully_async/sum/optimizer_consumed_episodes/"
                            "data_source/" + MULTITASK_ROUTES[0]
                        ]
                        + 1,
                    ),
                ),
            ),
            (
                "synthetic padding",
                lambda run: rewrite_first_rollout(
                    run, lambda document: document.update(is_padding=True)
                ),
            ),
            (
                "route/schedule binding",
                lambda run: rewrite_first_rollout(
                    run,
                    lambda document: document.update(
                        step_record_json=json.dumps(
                            {
                                **json.loads(document["step_record_json"]),
                                "route_id": MULTITASK_ROUTES[1],
                                "data_source": MULTITASK_ROUTES[1],
                            },
                            sort_keys=True,
                        )
                    ),
                ),
            ),
        )
        for label, mutation in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                run = build_valid_multitask_run(Path(directory), updates=1)
                mutation(run)
                with self.assertRaises(ValueError):
                    observe_run(run["run_dir"], 1)

    def test_trajectory_identity_cannot_be_reused_across_prefix_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            run = build_valid_multitask_run(Path(directory), updates=5)
            first_path = run["rollout_dir"] / "1.jsonl"
            fifth_path = run["rollout_dir"] / "5.jsonl"
            first_rows = first_path.read_text(encoding="utf-8").splitlines()
            fifth_rows = fifth_path.read_text(encoding="utf-8").splitlines()
            first_document = json.loads(first_rows[4])
            fifth_document = json.loads(fifth_rows[4])
            first_record = json.loads(first_document["step_record_json"])
            fifth_record = json.loads(fifth_document["step_record_json"])
            fifth_record["trajectory_uid"] = first_record["trajectory_uid"]
            fifth_document["step_record_json"] = json.dumps(
                fifth_record, sort_keys=True
            )
            fifth_rows[4] = json.dumps(fifth_document, sort_keys=True)
            fifth_path.write_text("\n".join(fifth_rows) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "trajectory_uid"):
                observe_run(run["run_dir"], 5)

    def test_receipt_identity_and_exact_rollout_filenames_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            run = build_valid_multitask_run(Path(directory), updates=1)
            mutate_json(
                run["launch_path"],
                lambda receipt: receipt["source"].update(outer_commit="e" * 40),
            )
            with self.assertRaisesRegex(ValueError, "launch identity/config audit"):
                observe_run(run["run_dir"], 1)

        with tempfile.TemporaryDirectory() as directory:
            run = build_valid_multitask_run(Path(directory), updates=1)
            (run["rollout_dir"] / "1.jsonl").rename(run["rollout_dir"] / "01.jsonl")
            with self.assertRaisesRegex(ValueError, "rollout prefix"):
                observe_run(run["run_dir"], 1)

    def test_legacy_receipt_and_unlisted_updates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = build_valid_run(Path(directory), mode="gate")
            with self.assertRaisesRegex(ValueError, "multitask receipt"):
                observe_run(legacy["run_dir"], 1)
        with self.assertRaisesRegex(ValueError, "snapshot update"):
            observe_run("/does/not/matter", 8)

    def test_snapshot_write_is_atomic_and_does_not_mutate_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            run = build_valid_multitask_run(Path(directory), updates=1)
            output = run["run_dir"] / "live" / "update-1.json"
            output.parent.mkdir()
            output.write_text('{"old": true}\n', encoding="utf-8")
            old_inode = output.stat().st_ino
            before = file_state(run["run_dir"], excluding={output})

            snapshot = write_snapshot(run["run_dir"], 1, output)

            self.assertEqual(snapshot["status"], "descriptive")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), snapshot)
            self.assertNotEqual(output.stat().st_ino, old_inode)
            self.assertEqual(file_state(run["run_dir"], excluding={output}), before)
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

            outside = Path(directory).parent / "outside-snapshot.json"
            with self.assertRaisesRegex(ValueError, "inside run_dir"):
                write_snapshot(run["run_dir"], 1, outside)
            metrics_before = run["metrics_path"].read_bytes()
            with self.assertRaisesRegex(ValueError, "receipt-bound input"):
                write_snapshot(run["run_dir"], 1, run["metrics_path"])
            self.assertEqual(run["metrics_path"].read_bytes(), metrics_before)
            with self.assertRaisesRegex(ValueError, "receipt-bound input"):
                write_snapshot(run["run_dir"], 1, run["rollout_dir"] / "snapshot.json")

    def test_snapshot_temp_file_cannot_follow_a_predictable_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = build_valid_multitask_run(root / "run", updates=1)
            output = run["run_dir"] / "update-1.json"
            protected = root / "protected.txt"
            protected.write_text("do not overwrite\n", encoding="utf-8")
            predictable = output.with_name(f".{output.name}.{os.getpid()}.tmp")
            predictable.symlink_to(protected)

            snapshot = write_snapshot(run["run_dir"], 1, output)

            self.assertEqual(snapshot["status"], "descriptive")
            self.assertEqual(protected.read_text(encoding="utf-8"), "do not overwrite\n")
            self.assertFalse(output.is_symlink())

    def test_receipt_change_during_observation_aborts_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            run = build_valid_multitask_run(Path(directory), updates=1)
            output = run["run_dir"] / "update-1.json"
            original_load = monitor_module._load_json

            def load_then_rebind(path: Path, label: str):
                receipt = original_load(path, label)
                mutate_json(
                    run["launch_path"],
                    lambda value: value["runtime_artifacts"].update(
                        file_logger=str(output)
                    ),
                )
                return receipt

            with mock.patch.object(
                monitor_module, "_load_json", side_effect=load_then_rebind
            ):
                with self.assertRaisesRegex(ValueError, "changed during observation"):
                    write_snapshot(run["run_dir"], 1, output)

            self.assertFalse(output.exists())

    def test_shared_evidence_code_has_no_route_name_control_flow(self):
        forbidden_routes = {"webshop", "swesmith", "literesearcher", "openmle"}
        legacy_schema = "amg_openmle_publication_identity_v3"
        for module in (finalizer_module, monitor_module):
            source = Path(module.__file__).read_text(encoding="utf-8")
            tree = ast.parse(source)
            condition_strings = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                    condition = node.test
                elif isinstance(node, ast.Match):
                    condition = node.subject
                else:
                    continue
                condition_strings.extend(
                    value.value.casefold()
                    for value in ast.walk(condition)
                    if isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value != legacy_schema
                )
            for route_name in forbidden_routes:
                self.assertFalse(
                    any(route_name in value for value in condition_strings),
                    f"{module.__name__} branches on {route_name}",
                )

    def test_online_monitor_has_no_process_control_surface(self):
        source = Path(monitor_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_modules = {"signal", "subprocess"}
        forbidden_calls = {
            "kill",
            "killpg",
            "pause",
            "send_signal",
            "shutdown",
            "stop",
            "terminate",
        }
        imports = {
            alias.name.split(".", maxsplit=1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        calls.update(
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        )
        self.assertTrue(forbidden_modules.isdisjoint(imports))
        self.assertTrue(forbidden_calls.isdisjoint(calls))


if __name__ == "__main__":
    unittest.main()
