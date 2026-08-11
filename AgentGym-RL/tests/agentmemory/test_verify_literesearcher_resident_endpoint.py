#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts"
    / "agentmemory"
    / "verify_literesearcher_resident_endpoint.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_literesearcher_resident_endpoint", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load LiteResearcher endpoint verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiteResearcherResidentEndpointTests(unittest.TestCase):
    def test_accepts_eight_distinct_indices(self) -> None:
        module = load_module()
        self.assertEqual(
            module.parse_probe_indices("0,1,13,14,30,31,47,63"),
            [0, 1, 13, 14, 30, 31, 47, 63],
        )

    def test_rejects_invalid_probe_indices(self) -> None:
        module = load_module()
        for raw, message in (
            ("0,1,2,3,4,5,6", "exactly 8"),
            ("0,1,2,3,4,5,6,6", "distinct"),
            ("0,1,2,3,4,5,6,-1", "nonnegative"),
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                module.parse_probe_indices(raw)

    def test_load_probe_tasks_uses_local_train_positions(self) -> None:
        module = load_module()
        tasks = []
        for index in range(64):
            tasks.append(
                {
                    "index": index + 100,
                    "question": f"question {index}",
                    "targets": [f"answer {index}"],
                    "mask_url": f"https://private/{index}",
                    "public_url": f"https://public/{index}",
                    "page_title": f"title {index}",
                    "resolved_url": f"https://resolved/{index}",
                }
            )
        manifest = {"manifest_sha256": "a" * 64, "train": tasks}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coverage.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, selected = module.load_probe_tasks(
                path, [0, 1, 13, 14, 30, 31, 47, 63]
            )
        self.assertEqual(loaded["manifest_sha256"], "a" * 64)
        self.assertEqual(
            [task["index"] for task in selected],
            [100, 101, 113, 114, 130, 131, 147, 163],
        )

    def test_idle_contract_checks_both_runtime_counts(self) -> None:
        module = load_module()
        module.require_idle(
            {"active_environment_count": 0, "active_workspace_count": 0},
            label="test",
        )
        for key in ("active_environment_count", "active_workspace_count"):
            metadata = {
                "active_environment_count": 0,
                "active_workspace_count": 0,
            }
            metadata[key] = 1
            with self.subTest(key=key), self.assertRaises(AssertionError):
                module.require_idle(metadata, label="test")

    def test_workspace_root_must_be_empty(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module.require_workspace_root_empty(root, label="empty")
            (root / "residue").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "workspace residue"):
                module.require_workspace_root_empty(root, label="dirty")

    def test_builds_native_research_and_workspace_actions(self) -> None:
        module = load_module()
        workspace = module.shell_action("cat .agent_memory/MEMORY.md")
        self.assertTrue(workspace.startswith("shell_command "))
        payload = json.loads(workspace.removeprefix("shell_command "))
        self.assertEqual(payload["workdir"], ".")
        self.assertEqual(payload["timeout_ms"], 30_000)
        search = module.tool_action("search", {"query": "alpha"})
        self.assertEqual(
            json.loads(search.removeprefix("<tool_call>").removesuffix("</tool_call>")),
            {"name": "search", "arguments": {"query": "alpha"}},
        )

    def test_endpoint_accepts_boolean_close_receipt(self) -> None:
        module = load_module()

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self) -> bytes:
                return b"true"

        endpoint = module.Endpoint("http://test.local", timeout=1)
        with mock.patch.object(module.urllib.request, "urlopen", return_value=Response()):
            result = endpoint.request(
                "POST",
                "close",
                {"id": 8},
                expected_type=bool,
            )
        self.assertIs(result, True)

    def test_close_slots_rejects_non_true_receipt(self) -> None:
        module = load_module()

        class Endpoint:
            def request(self, method, path, payload, *, expected_type):
                self.call = (method, path, payload, expected_type)
                return False

        endpoint = Endpoint()
        errors = module.close_slots(endpoint, [8])
        self.assertEqual(len(errors), 1)
        self.assertIn("did not return true", errors[0])
        self.assertEqual(endpoint.call, ("POST", "close", {"id": 8}, bool))


if __name__ == "__main__":
    unittest.main()
