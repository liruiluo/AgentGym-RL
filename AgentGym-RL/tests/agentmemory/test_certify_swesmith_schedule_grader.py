from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.agentmemory.certify_swesmith_schedule_grader import (
    SUBMISSION_ACTION,
    load_schedule_indices,
    result_passes,
    run_arm,
)


class FakeRecord:
    instance_id = "owner__repo.01234567.issue"
    instance = {"repo": "swesmith/owner__repo.01234567"}


class FakeEndpoint:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object] | None, bool]] = []

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        private: bool = False,
    ) -> dict[str, object]:
        self.calls.append((method, path, payload, private))
        if path == "create":
            return {"id": 7}
        if path == "reset":
            return {"done": False}
        step_count = len([call for call in self.calls if call[1] == "step"])
        if path == "step" and step_count == 1:
            return {"info": {"action_kind": "apply_patch"}}
        if path == "step":
            return {
                "reward": 0.0,
                "info": {
                    "action_kind": "shell_command",
                    "episode_success": False,
                },
            }
        if path.startswith("detail?"):
            return {"grade": {"resolution_status": "UNRESOLVED"}}
        if path == "close":
            return {}
        raise AssertionError((method, path, payload, private))


class CertifySwesmithScheduleGraderTest(unittest.TestCase):
    def write_routing(self, rows: list[dict[str, object]]) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", delete=False)
        with temporary:
            for row in rows:
                temporary.write(json.dumps(row) + "\n")
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return Path(temporary.name)

    def test_loads_exact_schedule_prefix(self) -> None:
        path = self.write_routing(
            [
                {
                    "item_id": f"swesmith_{position}",
                    "data_idx": data_idx,
                    "extra_info": {
                        "index": data_idx,
                        "schedule_position": position,
                    },
                }
                for position, data_idx in enumerate((4, 1, 7, 2, 9, 3))
            ]
        )
        self.assertEqual(
            load_schedule_indices(path, train_batch_size=2, gate_steps=2),
            [4, 1, 7, 2],
        )

    def test_rejects_duplicate_schedule_prefix(self) -> None:
        path = self.write_routing(
            [
                {
                    "item_id": f"swesmith_{position}",
                    "data_idx": data_idx,
                    "extra_info": {
                        "index": data_idx,
                        "schedule_position": position,
                    },
                }
                for position, data_idx in enumerate((4, 1, 4, 2))
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_schedule_indices(path, train_batch_size=2, gate_steps=2)

    def test_result_contract_separates_positive_and_negative_arms(self) -> None:
        self.assertTrue(
            result_passes(
                {
                    "arm": "gold",
                    "reward": 1.0,
                    "episode_success": True,
                    "resolution_status": "RESOLVED_FULL",
                    "submission_action_kind": "shell_command",
                    "grader_error": None,
                    "errors": [],
                }
            )
        )
        for arm in ("wrong", "tamper"):
            self.assertTrue(
                result_passes(
                    {
                        "arm": arm,
                        "reward": 0.0,
                        "episode_success": False,
                        "resolution_status": "UNRESOLVED",
                        "submission_action_kind": "shell_command",
                        "grader_error": None,
                        "errors": [],
                    }
                )
            )
        self.assertFalse(
            result_passes(
                {
                    "arm": "wrong",
                    "reward": 1.0,
                    "episode_success": True,
                    "resolution_status": "RESOLVED_FULL",
                    "submission_action_kind": "shell_command",
                    "grader_error": None,
                    "errors": [],
                }
            )
        )
        self.assertFalse(
            result_passes(
                {
                    "arm": "gold",
                    "reward": None,
                    "episode_success": False,
                    "resolution_status": None,
                    "submission_action_kind": None,
                    "grader_error": None,
                    "errors": ["reset failed"],
                }
            )
        )

    def test_run_arm_submits_with_upstream_shell_sentinel(self) -> None:
        endpoint = FakeEndpoint()
        result = run_arm(endpoint, [FakeRecord()], data_idx=0, arm="wrong")

        step_payloads = [
            payload
            for _, path, payload, _ in endpoint.calls
            if path == "step"
        ]
        self.assertEqual(len(step_payloads), 2)
        self.assertEqual(step_payloads[1], {"id": 7, "action": SUBMISSION_ACTION})
        self.assertEqual(
            SUBMISSION_ACTION,
            'shell_command {"command":"echo '
            'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT","workdir":"."}',
        )
        self.assertTrue(result_passes(result))


if __name__ == "__main__":
    unittest.main()
