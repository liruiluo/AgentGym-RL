from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT / "verl/utils/agentgym/task_neutral_parallel_reset.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "task_neutral_parallel_reset_under_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Message:
    def __init__(self, content: str) -> None:
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": "system", "content": self.content}


class Client:
    def __init__(
        self,
        index: int,
        *,
        barrier: threading.Barrier | None = None,
        delay: float = 0.0,
        error: Exception | None = None,
        sample_excluded: bool = False,
        active: dict[str, int] | None = None,
        active_lock: threading.Lock | None = None,
    ) -> None:
        self.index = index
        self.barrier = barrier
        self.delay = delay
        self.error = error
        self.sample_excluded = sample_excluded
        self.active = active
        self.active_lock = active_lock
        self.reset_indices: list[int] = []

    def reset(self, data_idx: int) -> None:
        self.reset_indices.append(data_idx)
        if self.active is not None and self.active_lock is not None:
            with self.active_lock:
                self.active["current"] += 1
                self.active["maximum"] = max(
                    self.active["maximum"], self.active["current"]
                )
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2.0)
            if self.delay:
                time.sleep(self.delay)
            if self.error is not None:
                raise self.error
        finally:
            if self.active is not None and self.active_lock is not None:
                with self.active_lock:
                    self.active["current"] -= 1

    def observe(self) -> str:
        return f"observation-{self.index}"


def bind_initial_context(client: Client, messages):
    return list(messages) + [
        {"role": "assistant", "content": f"bound-{client.index}"}
    ]


class TaskNeutralParallelResetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    @staticmethod
    def handlers(count: int):
        return [
            SimpleNamespace(
                messages=[Message(f"system-{index}")],
                item_id=f"item-{index}",
                data_idx=100 + index,
                done=False,
            )
            for index in range(count)
        ]

    def test_resets_concurrently_but_returns_batch_index_order(self) -> None:
        handlers = self.handlers(3)
        barrier = threading.Barrier(3)
        active = {"current": 0, "maximum": 0}
        active_lock = threading.Lock()
        clients = [
            Client(
                index,
                barrier=barrier,
                delay=(2 - index) * 0.02,
                active=active,
                active_lock=active_lock,
            )
            for index in range(3)
        ]

        result = self.module.reset_task_neutral_policy_contexts(
            handlers,
            clients,
            resolve_reset_index=lambda handler: handler.data_idx,
            bind_initial_policy_context=bind_initial_context,
        )

        self.assertEqual(active["maximum"], 3)
        self.assertEqual(result.excluded_indices, frozenset())
        self.assertEqual(
            [messages[-1]["content"] for messages in result.policy_messages],
            ["bound-0", "bound-1", "bound-2"],
        )
        self.assertEqual(
            [timing.index for timing in result.item_timings], [0, 1, 2]
        )
        self.assertEqual(
            [client.reset_indices for client in clients],
            [[100], [101], [102]],
        )

    def test_policy_context_binding_runs_on_caller_thread_in_batch_order(
        self,
    ) -> None:
        handlers = self.handlers(3)
        clients = [Client(index, delay=(2 - index) * 0.01) for index in range(3)]
        caller_thread = threading.get_ident()
        bind_calls: list[tuple[int, int]] = []

        def bind(client: Client, messages):
            bind_calls.append((client.index, threading.get_ident()))
            return bind_initial_context(client, messages)

        result = self.module.reset_task_neutral_policy_contexts(
            handlers,
            clients,
            resolve_reset_index=lambda handler: handler.data_idx,
            bind_initial_policy_context=bind,
        )

        self.assertEqual(
            bind_calls,
            [(0, caller_thread), (1, caller_thread), (2, caller_thread)],
        )
        self.assertEqual(
            [messages[-1]["content"] for messages in result.policy_messages],
            ["bound-0", "bound-1", "bound-2"],
        )

    def test_bind_failure_names_original_row_and_item(self) -> None:
        handlers = self.handlers(3)
        clients = [Client(index) for index in range(3)]

        def bind(client: Client, messages):
            if client.index == 1:
                raise ValueError("invalid policy framing")
            return bind_initial_context(client, messages)

        with self.assertRaisesRegex(
            RuntimeError,
            r"row=1 item_id=item-1 error=ValueError: invalid policy framing",
        ):
            self.module.reset_task_neutral_policy_contexts(
                handlers,
                clients,
                resolve_reset_index=lambda handler: handler.data_idx,
                bind_initial_policy_context=bind,
            )

    def test_reset_failure_names_original_row_and_item(self) -> None:
        handlers = self.handlers(3)
        clients = [
            Client(0),
            Client(1, error=ValueError("broken fixture")),
            Client(2),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            r"row=1 item_id=item-1 error=ValueError: broken fixture",
        ):
            self.module.reset_task_neutral_policy_contexts(
                handlers,
                clients,
                resolve_reset_index=lambda handler: handler.data_idx,
                bind_initial_policy_context=bind_initial_context,
            )

    def test_sample_excluded_preserves_other_rows_and_marks_handler_done(self) -> None:
        handlers = self.handlers(3)
        clients = [
            Client(0),
            Client(
                1,
                error=RuntimeError("certified exclusion"),
                sample_excluded=True,
            ),
            Client(2),
        ]

        result = self.module.reset_task_neutral_policy_contexts(
            handlers,
            clients,
            resolve_reset_index=lambda handler: handler.data_idx,
            bind_initial_policy_context=bind_initial_context,
        )

        self.assertEqual(result.excluded_indices, frozenset({1}))
        self.assertEqual(result.policy_messages[1], ())
        self.assertFalse(handlers[0].done)
        self.assertTrue(handlers[1].done)
        self.assertFalse(handlers[2].done)
        self.assertEqual(result.policy_messages[0][-1]["content"], "bound-0")
        self.assertEqual(result.policy_messages[2][-1]["content"], "bound-2")

    def test_client_handler_length_mismatch_fails_before_any_reset(self) -> None:
        handlers = self.handlers(2)
        client = Client(0)
        with self.assertRaisesRegex(ValueError, "client count"):
            self.module.reset_task_neutral_policy_contexts(
                handlers,
                [client],
                resolve_reset_index=lambda handler: handler.data_idx,
                bind_initial_policy_context=bind_initial_context,
            )
        self.assertEqual(client.reset_indices, [])

    def test_worker_limit_bounds_parallelism_without_reordering(self) -> None:
        handlers = self.handlers(4)
        active = {"current": 0, "maximum": 0}
        active_lock = threading.Lock()
        clients = [
            Client(
                index,
                delay=0.02,
                active=active,
                active_lock=active_lock,
            )
            for index in range(4)
        ]

        result = self.module.reset_task_neutral_policy_contexts(
            handlers,
            clients,
            resolve_reset_index=lambda handler: handler.data_idx,
            bind_initial_policy_context=bind_initial_context,
            max_workers=2,
        )

        self.assertEqual(active["maximum"], 2)
        self.assertEqual(
            [messages[-1]["content"] for messages in result.policy_messages],
            ["bound-0", "bound-1", "bound-2", "bound-3"],
        )

    def test_invalid_worker_limit_fails_before_any_reset(self) -> None:
        handlers = self.handlers(1)
        client = Client(0)
        for value in (0, -1, True, 1.5):
            with self.subTest(max_workers=value):
                with self.assertRaises((TypeError, ValueError)):
                    self.module.reset_task_neutral_policy_contexts(
                        handlers,
                        [client],
                        resolve_reset_index=lambda handler: handler.data_idx,
                        bind_initial_policy_context=bind_initial_context,
                        max_workers=value,
                    )
        self.assertEqual(client.reset_indices, [])

    def test_reset_index_failure_keeps_original_error_context(self) -> None:
        handlers = self.handlers(1)
        client = Client(0)

        with self.assertRaisesRegex(
            RuntimeError,
            r"row=0 item_id=item-0 error=ValueError: invalid index",
        ):
            self.module.reset_task_neutral_policy_contexts(
                handlers,
                [client],
                resolve_reset_index=lambda _handler: (_ for _ in ()).throw(
                    ValueError("invalid index")
                ),
                bind_initial_policy_context=bind_initial_context,
            )
        self.assertEqual(client.reset_indices, [])


if __name__ == "__main__":
    unittest.main()
