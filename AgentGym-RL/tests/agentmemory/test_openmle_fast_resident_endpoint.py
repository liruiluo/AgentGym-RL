#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "agentmemory" / "verify_openmle_fast_resident_endpoint.py"
OUTER_COMMIT = "1" * 40
INNER_COMMIT = "2" * 40
PROMPT_SHA256 = "3" * 64


def load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_openmle_fast_resident_endpoint",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load OpenMLE-fast endpoint verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest() -> dict:
    return {
        "schema": "openmle_fast_public_manifest_v1",
        "panel_id": "openmle-fast-test-v1",
        "role": "mechanism_gate",
        "openmle_tasks_revision": "f56e4b31252a9b81d95fea100098cd49b7290398",
        "task_count": 2,
        "task_id_list_sha256": "4" * 64,
        "compact_panel_sha256": "5" * 64,
        "max_policy_actions": 30,
        "records": [
            {
                "data_idx": 0,
                "task_id": "alpha@1",
                "source_family": "KAGGLE_DATASET:alpha",
            },
            {
                "data_idx": 1,
                "task_id": "beta@1",
                "source_family": "KAGGLE_DATASET:beta",
            },
        ],
    }


def manifest_sha256(document: dict) -> str:
    raw = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


class FakeEndpoint:
    def __init__(
        self,
        document: dict,
        *,
        detail_exists: bool = False,
        fail_second_create: bool = False,
        leak_across_reset: bool = False,
        omit_execution_counters: bool = False,
        wrong_reset_identity: bool = False,
    ) -> None:
        self.document = document
        self.detail_exists = detail_exists
        self.fail_second_create = fail_second_create
        self.leak_across_reset = leak_across_reset
        self.omit_execution_counters = omit_execution_counters
        self.wrong_reset_identity = wrong_reset_identity
        self.next_slot_id = 10
        self.slots: dict[int, dict] = {}
        self.closed_slots: set[int] = set()
        self.reset_history: list[int] = []
        self.close_history: list[int] = []
        self.metadata_extra: dict = {}

    def metadata(self) -> dict:
        manifest_digest = manifest_sha256(self.document)
        value = {
            "schema": "openmle_fast_public_metadata_v1",
            "domain_id": "openmle_fast",
            "contract_version": "openmle_fast_v1",
            "panel_id": self.document["panel_id"],
            "role": self.document["role"],
            "task_count": self.document["task_count"],
            "task_manifest_sha256": manifest_digest,
            "openmle_tasks_revision": self.document["openmle_tasks_revision"],
            "task_id_list_sha256": self.document["task_id_list_sha256"],
            "compact_panel_sha256": self.document["compact_panel_sha256"],
            "policy_prompt_sha256": PROMPT_SHA256,
            "runtime_source": {
                "outer_commit": OUTER_COMMIT,
                "inner_commit": INNER_COMMIT,
            },
            "contracts": {
                "action": "openmle_fast_three_tool_action_v1",
                "observation": "openmle_fast_bounded_observation_v1",
                "horizon": "openmle_fast_action_horizon_v1",
                "workspace": "openmle_fast_public_workspace_v1",
                "executor": "openmle_fast_isolated_executor_v1",
                "grader_boundary": "openmle_fast_grader_boundary_v1",
                "cleanup": "openmle_fast_owned_cleanup_v1",
            },
            "limits": {
                "max_policy_actions": 30,
                "max_request_wall_seconds": 25,
            },
            "active_slot_count": len(self.slots),
            "active_environment_count": sum(
                slot.get("record") is not None for slot in self.slots.values()
            ),
            "active_workspace_count": sum(
                slot.get("record") is not None for slot in self.slots.values()
            ),
        }
        value.update(copy.deepcopy(self.metadata_extra))
        return value

    def request_status(self, method: str, path: str) -> tuple[int, bytes]:
        if method == "GET" and path.lstrip("/").startswith("detail"):
            if self.detail_exists:
                return 200, b'{"private":"leak"}'
            return 404, b'{"detail":"Not Found"}'
        return 405, b""

    def request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> dict:
        route = path.lstrip("/")
        if method == "GET" and route == "metadata":
            return self.metadata()
        if method == "POST" and route == "create":
            if self.fail_second_create and self.next_slot_id == 11:
                raise RuntimeError("second create failed")
            slot_id = self.next_slot_id
            self.next_slot_id += 1
            self.slots[slot_id] = {
                "record": None,
                "canary": None,
                "counters": self.empty_counters(),
            }
            return {"id": slot_id}
        if method == "POST" and route == "reset":
            assert payload is not None
            slot = self.slots[payload["id"]]
            data_idx = payload["data_idx"]
            self.reset_history.append(data_idx)
            if not self.leak_across_reset:
                slot["canary"] = None
            slot["record"] = self.document["records"][data_idx]
            slot["counters"] = self.empty_counters()
            return self.episode_step(payload["id"])
        if method == "POST" and route == "step":
            assert payload is not None
            slot = self.slots[payload["id"]]
            action = payload["action"]
            command = json.loads(action.removeprefix("shell_command "))["command"]
            argv = shlex.split(command)
            if argv[:2] != ["python", "-c"] or len(argv) != 3:
                raise AssertionError(f"probe did not use managed Python: {command}")
            source = argv[2]
            slot["counters"]["action_count"] += 1
            if not self.omit_execution_counters:
                slot["counters"]["execution_action_count"] += 1
                slot["counters"]["execution_attempt_count"] += 1
                slot["counters"]["execution_completed_count"] += 1
            if ".write_text(" in source:
                marker = re.search(
                    r"OPENMLE_FAST_(?:SLOT|CROSS_RESET)_[0-9a-f]+",
                    source,
                )
                if marker is None:
                    raise AssertionError(f"write probe lacks a marker: {source}")
                slot["canary"] = marker.group(0)
                observation = "write complete"
            elif "OPENMLE_FAST_RESET_CLEAN" in source:
                observation = slot["canary"] or "OPENMLE_FAST_RESET_CLEAN"
            elif ".read_text(" in source:
                observation = slot["canary"] or ""
            else:
                raise AssertionError(f"unexpected probe source: {source}")
            return self.episode_step(payload["id"], observation=observation)
        if method == "POST" and route == "close":
            assert payload is not None
            slot_id = payload["id"]
            self.close_history.append(slot_id)
            if slot_id in self.slots:
                self.slots.pop(slot_id)
                self.closed_slots.add(slot_id)
                return {
                    "schema": "openmle_fast_cleanup_receipt_v1",
                    "closed": True,
                    "already_closed": False,
                }
            if slot_id in self.closed_slots:
                return {
                    "schema": "openmle_fast_cleanup_receipt_v1",
                    "closed": False,
                    "already_closed": True,
                }
            raise AssertionError(f"close requested an unknown slot: {slot_id}")
        raise AssertionError(f"unexpected request: {method} {path} {payload}")

    @staticmethod
    def empty_counters() -> dict:
        return {
            "action_count": 0,
            "execution_action_count": 0,
            "execution_attempt_count": 0,
            "execution_completed_count": 0,
            "nested_subprocess_count": 0,
            "fit_count": 0,
            "grading_count": 0,
        }

    def episode_step(self, slot_id: int, *, observation: str = "ready") -> dict:
        slot = self.slots[slot_id]
        record = slot["record"]
        data_idx = record["data_idx"]
        if self.wrong_reset_identity:
            data_idx = (data_idx + 1) % self.document["task_count"]
        return {
            "observation": observation,
            "reward": 0.0,
            "done": False,
            "truncated": False,
            "info": {
                "data_idx": data_idx,
                "task_id": record["task_id"],
                "source_family": record["source_family"],
                "task_manifest_sha256": manifest_sha256(self.document),
                "counters": copy.deepcopy(slot["counters"]),
            },
        }


class OpenMLEFastResidentEndpointTests(unittest.TestCase):
    def test_verifies_exact_resets_isolation_and_cleanup(self) -> None:
        module = load_module()
        document = manifest()
        endpoint = FakeEndpoint(document)

        evidence = module.verify_resident_endpoint(
            endpoint,
            document,
            manifest_sha256(document),
            probe_indices=[0, 1],
            expected_outer_commit=OUTER_COMMIT,
            expected_inner_commit=INNER_COMMIT,
            expected_prompt_sha256=PROMPT_SHA256,
            client_timeout_seconds=31,
            timeout_margin_seconds=5,
            forbidden_canaries=["NEVER_PUBLIC_CANARY"],
        )

        self.assertEqual(evidence["status"], "pass")
        self.assertEqual(endpoint.reset_history, [0, 1, 1, 0])
        self.assertEqual(endpoint.slots, {})
        self.assertEqual(len(endpoint.close_history), 4)

    def test_rejects_a_public_detail_route(self) -> None:
        module = load_module()
        document = manifest()
        endpoint = FakeEndpoint(document, detail_exists=True)
        with self.assertRaisesRegex(AssertionError, "/detail"):
            module.verify_resident_endpoint(
                endpoint,
                document,
                manifest_sha256(document),
                probe_indices=[0, 1],
                expected_outer_commit=OUTER_COMMIT,
                expected_inner_commit=INNER_COMMIT,
                expected_prompt_sha256=PROMPT_SHA256,
                client_timeout_seconds=31,
                timeout_margin_seconds=5,
                forbidden_canaries=["NEVER_PUBLIC_CANARY"],
            )

    def test_rejects_private_metadata_and_known_canaries(self) -> None:
        module = load_module()
        document = manifest()
        endpoint = FakeEndpoint(document)
        endpoint.metadata_extra = {
            "private_manifest_path": "/private/NEVER_PUBLIC_CANARY.json"
        }
        with self.assertRaisesRegex(AssertionError, "public-safe"):
            module.verify_resident_endpoint(
                endpoint,
                document,
                manifest_sha256(document),
                probe_indices=[0, 1],
                expected_outer_commit=OUTER_COMMIT,
                expected_inner_commit=INNER_COMMIT,
                expected_prompt_sha256=PROMPT_SHA256,
                client_timeout_seconds=31,
                timeout_margin_seconds=5,
                forbidden_canaries=["NEVER_PUBLIC_CANARY"],
            )

    def test_rejects_wrong_reset_identity_and_still_closes_slots(self) -> None:
        module = load_module()
        document = manifest()
        endpoint = FakeEndpoint(document, wrong_reset_identity=True)
        with self.assertRaisesRegex(AssertionError, "data_idx"):
            module.verify_resident_endpoint(
                endpoint,
                document,
                manifest_sha256(document),
                probe_indices=[0, 1],
                expected_outer_commit=OUTER_COMMIT,
                expected_inner_commit=INNER_COMMIT,
                expected_prompt_sha256=PROMPT_SHA256,
                client_timeout_seconds=31,
                timeout_margin_seconds=5,
                forbidden_canaries=["NEVER_PUBLIC_CANARY"],
            )
        self.assertEqual(endpoint.slots, {})

    def test_rejects_cross_reset_canary_leak_and_still_closes_slots(self) -> None:
        module = load_module()
        document = manifest()
        endpoint = FakeEndpoint(document, leak_across_reset=True)
        with self.assertRaisesRegex(AssertionError, "cross-reset"):
            module.verify_resident_endpoint(
                endpoint,
                document,
                manifest_sha256(document),
                probe_indices=[0, 1],
                expected_outer_commit=OUTER_COMMIT,
                expected_inner_commit=INNER_COMMIT,
                expected_prompt_sha256=PROMPT_SHA256,
                client_timeout_seconds=31,
                timeout_margin_seconds=5,
                forbidden_canaries=["NEVER_PUBLIC_CANARY"],
            )
        self.assertEqual(endpoint.slots, {})

    def test_rejects_missing_execution_counter_evidence(self) -> None:
        module = load_module()
        document = manifest()
        endpoint = FakeEndpoint(document, omit_execution_counters=True)
        with self.assertRaisesRegex(AssertionError, "execution_action_count"):
            module.verify_resident_endpoint(
                endpoint,
                document,
                manifest_sha256(document),
                probe_indices=[0, 1],
                expected_outer_commit=OUTER_COMMIT,
                expected_inner_commit=INNER_COMMIT,
                expected_prompt_sha256=PROMPT_SHA256,
                client_timeout_seconds=31,
                timeout_margin_seconds=5,
                forbidden_canaries=["NEVER_PUBLIC_CANARY"],
            )
        self.assertEqual(endpoint.slots, {})

    def test_closes_the_first_slot_when_the_second_create_fails(self) -> None:
        module = load_module()
        document = manifest()
        endpoint = FakeEndpoint(document, fail_second_create=True)
        with self.assertRaisesRegex(RuntimeError, "second create failed"):
            module.verify_resident_endpoint(
                endpoint,
                document,
                manifest_sha256(document),
                probe_indices=[0, 1],
                expected_outer_commit=OUTER_COMMIT,
                expected_inner_commit=INNER_COMMIT,
                expected_prompt_sha256=PROMPT_SHA256,
                client_timeout_seconds=31,
                timeout_margin_seconds=5,
                forbidden_canaries=["NEVER_PUBLIC_CANARY"],
            )
        self.assertEqual(endpoint.slots, {})


if __name__ == "__main__":
    unittest.main()
