from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


_REPO = pathlib.Path(__file__).resolve().parents[2]
_TYPES = (
    _REPO.parent
    / "AgentGym"
    / "agentenv"
    / "agentenv"
    / "controller"
    / "types.py"
)


def _load_types_module():
    spec = importlib.util.spec_from_file_location(
        "agentmemory_task_neutral_types_under_test", _TYPES
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load { _TYPES }")
    module = importlib.util.module_from_spec(spec)
    # dataclasses on Python 3.9 resolve postponed annotations through
    # sys.modules while the dynamically loaded module is being initialized.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TaskNeutralTransitionReceiptTest(unittest.TestCase):
    def test_step_output_carries_wrapper_transition_info(self):
        types = _load_types_module()
        info = {
            "schema": "agentmemory_task_neutral_transition_v1",
            "native_step_before": 2,
            "native_step_after": 3,
            "context_epoch_before": 1,
            "context_epoch_after": 2,
            "wrapper_evidence": {"event": "session_handoff"},
        }

        # This is intentionally the first failing fixture for the migration:
        # the pre-compaction client contract drops all lifecycle information.
        output = types.StepOutput(
            state="next observation",
            reward=0.25,
            done=False,
            info=info,
        )
        self.assertEqual(output.info, info)


if __name__ == "__main__":
    unittest.main()
