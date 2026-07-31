from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from verl.workers import fsdp_gradient_accumulation


ROOT = Path(__file__).resolve().parents[2]


class _FakeFSDP:
    def __init__(self):
        self.events = []

    @contextmanager
    def no_sync(self):
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")


def test_nonfinal_microbatch_uses_no_sync_when_enabled() -> None:
    module = _FakeFSDP()
    with mock.patch.object(fsdp_gradient_accumulation, "FSDP", _FakeFSDP):
        with fsdp_gradient_accumulation.fsdp_gradient_sync_context(
            module,
            enabled=True,
            is_last_micro_batch=False,
        ):
            module.events.append("backward")

    assert module.events == ["enter", "backward", "exit"]


def test_final_microbatch_and_default_path_keep_sync_enabled() -> None:
    for enabled, is_last_micro_batch in ((False, False), (True, True)):
        module = _FakeFSDP()
        with mock.patch.object(fsdp_gradient_accumulation, "FSDP", _FakeFSDP):
            with fsdp_gradient_accumulation.fsdp_gradient_sync_context(
                module,
                enabled=enabled,
                is_last_micro_batch=is_last_micro_batch,
            ):
                module.events.append("backward")
        assert module.events == ["backward"]


def test_actor_and_critic_wrap_forward_backward_microbatch_loops() -> None:
    for relative_path, class_name, method_name in (
        (
            "verl/workers/agent_actor/dp_actor.py",
            "DataParallelPPOActor",
            "update_policy",
        ),
        (
            "verl/workers/agent_critic/dp_critic.py",
            "DataParallelPPOCritic",
            "update_critic",
        ),
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        method = next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        method_source = ast.get_source_segment(source, method)
        assert "enumerate(micro_batches)" in method_source
        assert "fsdp_gradient_sync_context(" in method_source
        assert "is_last_micro_batch=is_last_micro_batch" in method_source
        assert "deferred_gradient_sync_microbatches" in method_source


def test_default_config_keeps_actor_and_critic_optimization_disabled() -> None:
    config = (ROOT / "verl/agent_trainer/config/ppo_trainer.yaml").read_text(
        encoding="utf-8"
    )
    assert config.count("use_no_sync_for_gradient_accumulation: False") == 2
    actor_section = config.split("\n  ref:", 1)[0]
    critic_section = config.split("\ncritic:", 1)[1].split("\nreward_model:", 1)[0]
    assert "use_no_sync_for_gradient_accumulation: False" in actor_section
    assert "use_no_sync_for_gradient_accumulation: False" in critic_section
