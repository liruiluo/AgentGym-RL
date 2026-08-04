from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_module(
    "test_filesystem_sft_generator",
    ROOT / "scripts" / "agentmemory" / "generate_filesystem_sft_v1.py",
)
SCHEMA = _load_module(
    "test_filesystem_sft_schema",
    ROOT / "verl" / "utils" / "agent_dataset" / "agent_action_schema.py",
)


SURFACE = GENERATOR.SURFACE
TASK_FAMILY = GENERATOR.TASK_FAMILY
TARGET_ASINS = [f"ABCD{index:06d}" for index in range(6)]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _FakeBackend:
    def __init__(self):
        self.active = 0
        self.closed = False
        self.metadata_calls = 0

    def active_session_count(self):
        return self.active

    def metadata(self):
        if self.closed:
            raise AssertionError("metadata was read after backend close")
        self.metadata_calls += 1
        return {"loaded_once": True, "active_session_count": self.active}

    def close(self):
        self.closed = True


class _FakeProvider:
    start_orbit = 0

    def __init__(self, task, proof_sha256):
        self.task = task
        self.proof_sha256 = proof_sha256

    def get(self, data_index):
        return SimpleNamespace(
            task_id=self.task.task_id,
            orbit_id=self.task.orbit_id,
            scenario_id=self.task.scenario_id,
            proof_sha256=self.proof_sha256,
            product_pool_sha256=self.task.product_pool_sha256,
        )

    def proof_for_index(self, data_index):
        return SimpleNamespace(proof_sha256=self.proof_sha256)


class _FakeEnv:
    def __init__(self, task, backend):
        self.task = task
        self.backend = backend
        self._workspace_tmp = tempfile.TemporaryDirectory()
        self.workspace = SimpleNamespace(host_root=Path(self._workspace_tmp.name))
        self.native_page = None
        self.purchase_ledger = []
        self.current_session_index = 0
        self.event_count = 0
        self.step_count = 0
        self.done = False
        self.status = "running"
        self._note = None
        self._tree = _sha256_text("empty tree")

    def _snapshot(self):
        return {
            "schema": "agentmemory_workspace_snapshot_v2",
            "file_count": 0 if self._note is None else 1,
            "directory_count": 0,
            "total_bytes": 0 if self._note is None else len(self._note.encode()),
            "directories": [],
            "files": [] if self._note is None else [{"path": ".agent_memory/MEMORY.md"}],
            "tree_sha256": self._tree,
        }

    def _info(self, *, event=None):
        workspace_event = (
            event if event is not None and event["op"] in {"SHELL_COMMAND", "APPLY_PATCH"}
            else None
        )
        return {
            "surface": SURFACE,
            "task_family": TASK_FAMILY,
            "split": self.task.split,
            "scenario_id": self.task.scenario_id,
            "current_subtask_index": self.current_session_index,
            "phase_count": 6,
            "workspace_snapshot": self._snapshot(),
            "workspace_audit_event_count": self.event_count,
            "workspace_ops": [] if workspace_event is None else [workspace_event],
            "workspace_latest_event": workspace_event,
            "tool_ops": [] if event is None else [event],
        }

    def reset(self, *, data_idx):
        del data_idx
        self.current_session_index = 0
        self.event_count = 0
        self.step_count = 0
        self.done = False
        self.status = "running"
        self.purchase_ledger.clear()
        self.native_page = None
        self._note = None
        self._tree = _sha256_text("empty tree")
        self.backend.active = 1
        return "initial observation", self._info()

    def _workspace_event(self, action, before_tree):
        if action.startswith("shell_command "):
            payload = json.loads(action.removeprefix("shell_command "))
            canonical = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            command = payload["command"]
            stdout = self._note or "<empty>"
            return {
                "op": "SHELL_COMMAND",
                "tool_name": "shell_command",
                "status": "executed",
                "event_id": self.event_count,
                "phase_index": self.current_session_index,
                "episode_id": "fake:episode:1",
                "request_sha256": _sha256_text(canonical),
                "command_sha256": _sha256_text(command),
                "command_bytes": len(command.encode()),
                "exit_code": 0,
                "timed_out": False,
                "stdout": stdout,
                "stderr": "",
                "workspace_tree_sha256_before": before_tree,
                "workspace_tree_sha256_after": before_tree,
            }
        patch_text = action.removeprefix("apply_patch\n")
        added = [line[1:] for line in patch_text.splitlines() if line.startswith("+")]
        self._note = "\n".join(added) + "\n"
        note_path = self.workspace.host_root / ".agent_memory" / "MEMORY.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(self._note, encoding="utf-8")
        self._tree = _sha256_text(self._note)
        digest = _sha256_text(patch_text)
        return {
            "op": "APPLY_PATCH",
            "tool_name": "apply_patch",
            "status": "executed",
            "event_id": self.event_count,
            "phase_index": self.current_session_index,
            "episode_id": "fake:episode:1",
            "request_sha256": digest,
            "patch_sha256": digest,
            "patch_bytes": len(patch_text.encode()),
            "transactional": True,
            "changed_paths": [".agent_memory/MEMORY.md"],
            "workspace_tree_sha256_before": before_tree,
            "workspace_tree_sha256_after": self._tree,
        }

    def step(self, action):
        before_tree = self._tree
        if action.startswith("shell_command ") or action.startswith("apply_patch\n"):
            event = self._workspace_event(action, before_tree)
            self.event_count += 1
            observation = "workspace action executed"
            reward, terminated, truncated = 0.0, False, False
        elif action.startswith("search["):
            phase = self.task.phases[self.current_session_index]
            target = next(item for item in phase.candidates if item.asin == phase.target_asin)
            self.native_page = SimpleNamespace(clickables=[target.asin])
            event = {"op": "SEARCH", "raw_action": action, "result_count": 2}
            observation = "search results"
            reward, terminated, truncated = 0.0, False, False
        elif action.startswith("click[") and action != "click[Buy Now]":
            phase = self.task.phases[self.current_session_index]
            target = next(item for item in phase.candidates if item.asin == phase.target_asin)
            self.native_page = SimpleNamespace(clickables=["Buy Now"])
            event = {"op": "CLICK", "raw_action": action}
            if action != f"click[{target.asin}]":
                raise AssertionError("fake policy clicked the wrong candidate")
            observation = "product page"
            reward, terminated, truncated = 0.0, False, False
        elif action == "click[Buy Now]":
            phase = self.task.phases[self.current_session_index]
            target = next(item for item in phase.candidates if item.asin == phase.target_asin)
            final = self.current_session_index == 5
            event = {
                "op": "BUY",
                "raw_action": action,
                "committed": True,
                "purchase_correct": True,
                "session_advanced": True,
                "terminal": final,
                "step": self.step_count,
                "session_index": self.current_session_index,
            }
            self.purchase_ledger.append(
                {
                    **event,
                    "actual_asin": target.asin,
                    "actual_price_cents": 1000,
                    "selected_options": {},
                    "budget_ok": True,
                }
            )
            self.current_session_index += 1
            self.backend.active = 0 if final else 1
            self.done = final
            self.status = "success" if final else "running"
            observation = "purchase committed"
            reward, terminated, truncated = (2.0 if final else 1.0), final, False
        else:  # pragma: no cover
            raise AssertionError(f"unknown fake action: {action}")
        self.step_count += 1
        info = self._info(event=event)
        return observation, reward, terminated, truncated, info


def _task():
    phases = []
    for index, asin in enumerate(TARGET_ASINS):
        value = f"value-{index}"
        product = SimpleNamespace(
            asin=asin,
            title=f"private title {index}",
            attribute_display_name=value,
            search_query=f"color {value}",
            catalog_record_sha256=f"{index + 1:064x}",
        )
        target = SimpleNamespace(
            asin=asin,
            title=f"private title {index}",
            product=product,
            search_query=f"color {value}",
        )
        distractor = SimpleNamespace(
            asin=f"WXYZ{index:06d}",
            title="distractor",
            product=SimpleNamespace(asin=f"WXYZ{index:06d}", title="distractor"),
            search_query="distractor",
        )
        phases.append(
            SimpleNamespace(
                phase_index=index,
                target_asin=asin,
                attribute_name="color",
                question=f"Select a product with color {value}.",
                candidates=[target, distractor],
            )
        )
    return SimpleNamespace(
        task_id="task_0_a",
        orbit_id="orbit_0",
        orbit_index=0,
        scenario_id="finish",
        split="train",
        product_pool_sha256="a" * 64,
        semantic_sha256="b" * 64,
        phases=phases,
    )


class FilesystemSFTGeneratorTests(unittest.TestCase):
    def test_fake_runtime_produces_exact_expert_action_contract(self):
        task = _task()
        backend = _FakeBackend()
        env = _FakeEnv(task, backend)
        provider = _FakeProvider(task, "c" * 64)
        observation, info = env.reset(data_idx=0)
        del observation, info
        records = GENERATOR.generate_task_records(
            env,
            backend=backend,
            provider=provider,
            task=task,
            data_index=0,
            branch_index=0,
            system_prompt="Use Codex tools in the workspace.",
            source={"outer_source_commit": "1" * 40, "agentgym_source_commit": "2" * 40},
            validate_record=SCHEMA.validate_agent_action_record,
            finalize_record=SCHEMA.finalize_agent_action_record,
            canonical_sha256=SCHEMA.canonical_json_sha256,
            text_sha256=SCHEMA.text_sha256,
        )
        self.assertEqual(len(records), 28)
        self.assertEqual(
            Counter(record["action_kind"] for record in records),
            Counter(
                {
                    "native_search": 6,
                    "native_click": 12,
                    "workspace_shell_command": 5,
                    "workspace_apply_patch": 5,
                }
            ),
        )
        workspace_records = [
            record
            for record in records
            if record["action_kind"].startswith("workspace_")
        ]
        self.assertTrue(workspace_records)
        for event_id, record in enumerate(workspace_records):
            action = record["assistant_action"]
            self.assertNotIn("private title", action)
            self.assertNotRegex(action, r"[A-Z0-9]{10}")
            event = record["workspace_audit"]["event"]
            self.assertEqual(event["episode_id"], "fake:episode:1")
            self.assertEqual(event["event_id"], event_id)

    def test_metadata_is_frozen_before_backend_close(self):
        provider = SimpleNamespace(
            metadata_calls=0,
            metadata=lambda: {"calls": 1},
        )
        backend = _FakeBackend()

        class Sandbox:
            metadata_calls = 0

            @property
            def metadata(self):
                self.metadata_calls += 1
                return {"sandbox": "ready"}

        sandbox = Sandbox()
        frozen = GENERATOR.snapshot_runtime_metadata(
            provider=provider,
            backend=backend,
            shell_sandbox=sandbox,
        )
        backend.close()
        self.assertEqual(frozen["native_backend"]["active_session_count"], 0)
        self.assertEqual(backend.metadata_calls, 1)
        self.assertEqual(sandbox.metadata_calls, 1)
        self.assertEqual(frozen["workspace_sandbox"], {"sandbox": "ready"})

    def test_manifest_failure_rolls_back_dataset_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "records.json"
            manifest = root / "manifest.json"
            writer = GENERATOR.JsonArrayWriter(output)
            writer.write({"record_sha256": "a" * 64, "value": 1})
            writer.seal()

            original_replace = GENERATOR.os.replace

            def fail_manifest_replace(source, destination):
                if Path(destination) == manifest:
                    raise OSError("injected manifest publish failure")
                return original_replace(source, destination)

            with mock.patch.object(GENERATOR.os, "replace", side_effect=fail_manifest_replace):
                with self.assertRaisesRegex(OSError, "manifest publish"):
                    GENERATOR.publish_dataset_and_manifest(
                        writer,
                        manifest,
                        {"dataset_file": str(output)},
                    )
            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_dataset_and_manifest_publish_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "records.json"
            manifest = root / "manifest.json"
            writer = GENERATOR.JsonArrayWriter(output)
            writer.write({"record_sha256": "a" * 64, "value": 1})
            writer.seal()
            expected_hash = GENERATOR.file_sha256(writer.staged_path)
            GENERATOR.publish_dataset_and_manifest(
                writer,
                manifest,
                {"dataset_file": str(output), "dataset_file_sha256": expected_hash},
            )
            self.assertEqual(GENERATOR.file_sha256(output), expected_hash)
            self.assertEqual(json.loads(manifest.read_text())["dataset_file_sha256"], expected_hash)

    def test_json_writer_finish_remains_atomic_for_simple_callers(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.json"
            writer = GENERATOR.JsonArrayWriter(output)
            writer.write({"record_sha256": "a" * 64})
            writer.finish()
            self.assertEqual(json.loads(output.read_text()), [{"record_sha256": "a" * 64}])


if __name__ == "__main__":
    unittest.main()
