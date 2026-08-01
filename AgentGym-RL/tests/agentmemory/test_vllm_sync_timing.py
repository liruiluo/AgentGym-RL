import importlib.util
import json
import unittest
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl/workers/sharding_manager/vllm_sync_timing.py"
)
_SPEC = importlib.util.spec_from_file_location("vllm_sync_timing_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

EVENT_TYPE = _MODULE.EVENT_TYPE
LOG_PREFIX = _MODULE.LOG_PREFIX
SyncPhaseTimer = _MODULE.SyncPhaseTimer
format_sync_timing_event = _MODULE.format_sync_timing_event


class FakeClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class SyncPhaseTimerTest(unittest.TestCase):
    def test_accumulates_repeated_phases_and_emits_structured_event(self):
        timer = SyncPhaseTimer(
            enabled=True,
            clock=FakeClock([10.0, 11.0, 13.5, 14.0, 15.25, 18.0]),
        )

        with timer.phase("evidence"):
            pass
        with timer.phase("evidence"):
            pass

        event = timer.build_event(rank=3, sync_file_bytes=123)
        self.assertEqual(event["event_type"], EVENT_TYPE)
        self.assertEqual(event["rank"], 3)
        self.assertEqual(event["sync_file_bytes"], 123)
        self.assertEqual(event["timing_s"]["evidence"], 3.75)
        self.assertEqual(event["timing_s"]["total"], 8.0)

        line = format_sync_timing_event(event)
        self.assertTrue(line.startswith(LOG_PREFIX))
        self.assertEqual(json.loads(line[len(LOG_PREFIX):]), event)

    def test_disabled_timer_is_a_noop(self):
        timer = SyncPhaseTimer(enabled=False, clock=FakeClock([]))
        with timer.phase("save"):
            pass
        self.assertIsNone(timer.build_event(rank=0))
        self.assertIsNone(format_sync_timing_event(None))


if __name__ == "__main__":
    unittest.main()
