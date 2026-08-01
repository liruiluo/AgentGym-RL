import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from verl.workers.agent_fsdp_workers import (
    OPTIMIZER_PHASE_TELEMETRY_DIR,
    _record_optimizer_snapshot,
    _run_optimizer_transfer,
)


class _OptimizerStub:
    def __init__(self):
        self.state = {
            object(): {
                "exp_avg": torch.ones(4, dtype=torch.float32),
                "step": torch.ones((), dtype=torch.float32),
            }
        }


class OptimizerPhaseTelemetryTest(unittest.TestCase):
    def test_snapshot_and_transfer_record_state_bytes(self):
        optimizer = _OptimizerStub()
        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.dict(
                os.environ,
                {OPTIMIZER_PHASE_TELEMETRY_DIR: output_dir},
            ):
                _record_optimizer_snapshot(
                    optimizer=optimizer,
                    role="actor",
                    action="before_optional_load",
                    update_index=2,
                )
                _run_optimizer_transfer(
                    optimizer=optimizer,
                    role="actor",
                    action="load",
                    update_index=2,
                    transfer=lambda: None,
                )

            files = os.listdir(output_dir)
            self.assertEqual(len(files), 1)
            with open(os.path.join(output_dir, files[0]), encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]

        self.assertEqual(
            [event["action"] for event in events],
            ["before_optional_load", "load"],
        )
        self.assertEqual(events[0]["state"]["total_bytes"], 20)
        self.assertEqual(events[1]["state_before"]["total_bytes"], 20)
        self.assertGreaterEqual(events[1]["elapsed_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
