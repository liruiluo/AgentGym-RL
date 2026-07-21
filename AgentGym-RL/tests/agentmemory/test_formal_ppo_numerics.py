from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[2]


def _masked_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_mask = mask.bool()
    valid = values.masked_select(valid_mask)
    centered = valid - valid.mean()
    scale = torch.sqrt(centered.square().mean() + 1e-8)
    result = torch.zeros_like(values)
    result.masked_scatter_(valid_mask, centered / scale)
    return result


torch_functional_stub = types.ModuleType("verl.utils.torch_functional")
torch_functional_stub.masked_whiten = _masked_whiten
torch_functional_stub.masked_mean = (
    lambda values, mask: values.masked_select(mask.bool()).mean()
)
verl_utils = importlib.import_module("verl.utils")
original_torch_functional = sys.modules.get("verl.utils.torch_functional")
original_torch_functional_attr = getattr(verl_utils, "torch_functional", None)
sys.modules["verl.utils.torch_functional"] = torch_functional_stub
setattr(verl_utils, "torch_functional", torch_functional_stub)

core_spec = importlib.util.spec_from_file_location(
    "agentmemory_v19_core_algos_for_test",
    ROOT / "verl/agent_trainer/ppo/core_algos.py",
)
assert core_spec is not None and core_spec.loader is not None
core_algos = importlib.util.module_from_spec(core_spec)
try:
    core_spec.loader.exec_module(core_algos)
finally:
    if original_torch_functional is None:
        sys.modules.pop("verl.utils.torch_functional", None)
    else:
        sys.modules["verl.utils.torch_functional"] = original_torch_functional
    if original_torch_functional_attr is None:
        delattr(verl_utils, "torch_functional")
    else:
        setattr(verl_utils, "torch_functional", original_torch_functional_attr)

critic_spec = importlib.util.spec_from_file_location(
    "agentmemory_v19_critic_initialization_for_test",
    ROOT / "verl/workers/critic_initialization.py",
)
assert critic_spec is not None and critic_spec.loader is not None
critic_initialization = importlib.util.module_from_spec(critic_spec)
critic_spec.loader.exec_module(critic_initialization)


class DummyCritic(torch.nn.Module):
    def __init__(self, *, num_labels: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_labels=num_labels)
        self.score = torch.nn.Linear(4, 1, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.score(hidden_states)


class FakeDataProto:
    def __init__(self, *, formal: bool = False) -> None:
        self.formal = formal
        self.batch = {
            "values": torch.zeros(2, 2),
            "response_mask": torch.ones(2, 2),
            "token_level_rewards": torch.tensor([[0.0, 0.05], [0.0, -0.10]]),
            "agentmemory_suffix_return": torch.tensor([0.05, -0.10]),
        }
        self.meta_info = {}


def load_compute_advantage_namespace() -> dict:
    trainer_path = ROOT / "verl/agent_trainer/ppo/ray_trainer.py"
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "compute_advantage",
            "_agentmemory_env_flag",
            "_validate_formal_actor_advantage_config",
        }
    ]
    namespace = {
        "DataProto": FakeDataProto,
        "core_algos": core_algos,
        "os": os,
        "requires_formal_trajectory_metadata": lambda data: data.formal,
        "validate_formal_trajectory_metadata": lambda *args, **kwargs: None,
        "AGENTMEMORY_SUFFIX_RETURN": "agentmemory_suffix_return",
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), trainer_path, "exec"),
        namespace,
    )
    return namespace


class CriticInitializationTests(unittest.TestCase):
    def test_formal_scalar_head_is_all_zero_and_emits_zero_values(self) -> None:
        critic = DummyCritic()
        with torch.no_grad():
            critic.score.weight.fill_(0.75)
            critic.score.bias.fill_(-0.25)

        summary = critic_initialization.initialize_critic_value_head(
            critic,
            missing_keys=("score.weight", "score.bias"),
            policy="zero_if_missing",
        )

        self.assertEqual(summary["status"], "zero_initialized")
        self.assertTrue(summary["all_parameters_zero"])
        for parameter in critic.score.parameters():
            self.assertEqual(torch.count_nonzero(parameter).item(), 0)
        torch.testing.assert_close(
            critic(torch.randn(3, 4)),
            torch.zeros(3, 1),
            rtol=0.0,
            atol=0.0,
        )

    def test_non_formal_run_leaves_head_untouched(self) -> None:
        critic = DummyCritic()
        before = critic.score.weight.detach().clone()
        summary = critic_initialization.initialize_critic_value_head(
            critic,
            missing_keys=("score.weight", "score.bias"),
            policy="preserve",
        )
        self.assertIsNone(summary)
        torch.testing.assert_close(critic.score.weight, before)

    def test_non_scalar_or_ambiguous_head_fails_closed(self) -> None:
        critic = DummyCritic(num_labels=2)
        with self.assertRaisesRegex(RuntimeError, "num_labels=1"):
            critic_initialization.zero_initialize_scalar_value_head(critic)

        critic = DummyCritic()
        critic.classifier = torch.nn.Linear(4, 1)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            critic_initialization.zero_initialize_scalar_value_head(critic)

    def test_invalid_initialization_policy_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "value_head_init"):
            critic_initialization.initialize_critic_value_head(
                DummyCritic(),
                missing_keys=("score.weight", "score.bias"),
                policy="maybe",
            )

    def test_pretrained_scalar_head_is_preserved(self) -> None:
        critic = DummyCritic()
        before = {
            name: parameter.detach().clone()
            for name, parameter in critic.score.named_parameters()
        }
        summary = critic_initialization.initialize_critic_value_head(
            critic,
            missing_keys=(),
            policy="zero_if_missing",
        )
        self.assertEqual(summary["status"], "pretrained_head_preserved")
        for name, parameter in critic.score.named_parameters():
            torch.testing.assert_close(parameter, before[name])

    def test_partially_loaded_scalar_head_fails_closed(self) -> None:
        critic = DummyCritic()
        with self.assertRaisesRegex(RuntimeError, "partially loaded"):
            critic_initialization.initialize_critic_value_head(
                critic,
                missing_keys=("score.weight",),
                policy="zero_if_missing",
            )

    def test_meta_rank_defers_to_existing_rank0_fsdp_sync(self) -> None:
        critic = DummyCritic()
        critic.score = torch.nn.Linear(4, 1, device="meta")
        summary = critic_initialization.initialize_critic_value_head(
            critic,
            missing_keys=("score.weight", "score.bias"),
            policy="zero_if_missing",
        )
        self.assertEqual(summary["status"], "meta_deferred_to_rank0_fsdp_sync")
        self.assertIsNone(summary["all_parameters_zero"])

    def test_worker_initializes_before_fsdp_rank_sync(self) -> None:
        worker_path = ROOT / "verl/workers/agent_fsdp_workers.py"
        tree = ast.parse(worker_path.read_text(encoding="utf-8"))
        critic_worker = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CriticWorker"
        )
        builder = next(
            node
            for node in critic_worker.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_critic_model_optimizer"
        )
        source = ast.get_source_segment(
            worker_path.read_text(encoding="utf-8"), builder
        )
        self.assertLess(
            source.index("critic_module.to(torch_dtype)"),
            source.index("initialize_critic_value_head"),
        )
        self.assertLess(
            source.index("initialize_critic_value_head"),
            source.index("critic_module = FSDP("),
        )
        self.assertIn("output_loading_info=True", source)
        self.assertIn('missing_keys=critic_loading_info["missing_keys"]', source)
        self.assertIn("sync_module_states=True", source)


class FormalAdvantageTests(unittest.TestCase):
    def formal_config(self, *, use_kl_loss=True, reward_kl=0.0):
        return SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                actor={
                    "use_kl_loss": use_kl_loss,
                    "kl_loss_coef": 0.01,
                }
            ),
            algorithm=SimpleNamespace(
                gamma=1.0,
                lam=1.0,
                kl_ctrl=SimpleNamespace(kl_coef=reward_kl),
            ),
        )

    def test_rms_preserves_sign_where_mean_centering_flips_it(self) -> None:
        raw = torch.tensor([[-1.0, -0.10, 0.05]])
        mask = torch.ones_like(raw)
        whitened = _masked_whiten(raw, mask)
        rms_scaled = core_algos.masked_rms_scale_advantages(raw, mask)

        self.assertGreater(whitened[0, 1].item(), 0.0)
        self.assertLess(raw[0, 1].item(), 0.0)
        self.assertTrue(
            torch.equal(torch.sign(rms_scaled), torch.sign(raw))
        )

    def test_formal_reward_classes_keep_expected_raw_and_scaled_signs(self) -> None:
        terminal_rewards = torch.tensor([1.0, 0.05, -0.10, -0.05])
        rewards = torch.zeros(4, 3)
        rewards[:, -1] = terminal_rewards
        values = torch.zeros_like(rewards)
        mask = torch.ones_like(rewards)

        scaled, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            eos_mask=mask,
            gamma=1.0,
            lam=1.0,
            advantage_normalization="rms",
        )
        raw = returns - values

        self.assertTrue(torch.all(raw[:2] > 0.0))
        self.assertTrue(torch.all(scaled[:2] > 0.0))
        self.assertTrue(torch.all(raw[2:] < 0.0))
        self.assertTrue(torch.all(scaled[2:] < 0.0))
        torch.testing.assert_close(
            returns,
            terminal_rewards.unsqueeze(-1).expand_as(returns),
        )

    def test_rms_changes_only_advantages_not_returns(self) -> None:
        rewards = torch.tensor([[0.0, 0.0, 0.05], [0.0, 0.0, -0.10]])
        values = torch.tensor([[0.01, -0.02, 0.03], [-0.04, 0.02, -0.01]])
        mask = torch.ones_like(rewards)

        _, whitened_returns = core_algos.compute_gae_advantage_return(
            rewards, values, mask, 1.0, 1.0, advantage_normalization="whiten"
        )
        _, rms_returns = core_algos.compute_gae_advantage_return(
            rewards, values, mask, 1.0, 1.0, advantage_normalization="rms"
        )
        torch.testing.assert_close(rms_returns, whitened_returns, rtol=0.0, atol=0.0)

    def test_masked_padding_is_zero_and_nonfinite_valid_values_fail(self) -> None:
        advantages = torch.tensor([[0.05, -0.10, 99.0]])
        mask = torch.tensor([[1, 1, 0]])
        scaled = core_algos.masked_rms_scale_advantages(advantages, mask)
        self.assertEqual(scaled[0, 2].item(), 0.0)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            core_algos.masked_rms_scale_advantages(
                torch.tensor([[0.05, float("nan")]]),
                torch.ones(1, 2),
            )

    def test_fresh_value_gate_accepts_zero_and_rejects_random_head_scale(self) -> None:
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        self.assertEqual(
            core_algos.validate_near_zero_critic_values(
                torch.zeros(2, 3), mask
            ),
            0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "not near zero"):
            core_algos.validate_near_zero_critic_values(
                torch.tensor([[0.0, 1e-3, 0.0], [0.0, 0.0, 0.0]]),
                mask,
            )

    def test_nonformal_compute_advantage_keeps_legacy_token_axis(self) -> None:
        namespace = load_compute_advantage_namespace()
        original = core_algos.compute_gae_advantage_return
        calls = []

        def recording_compute(*args, **kwargs):
            calls.append(kwargs.get("advantage_normalization", "default"))
            return original(*args, **kwargs)

        with mock.patch.object(
            core_algos, "compute_gae_advantage_return", recording_compute
        ):
            with mock.patch.dict(
                os.environ,
                {"AGENTMEMORY_REQUIRE_FORMAL_RUNTIME_EVIDENCE": "0"},
            ):
                namespace["compute_advantage"](
                    FakeDataProto(), "gae", gamma=1.0, lam=1.0
                )

        self.assertEqual(calls, ["default"])

    def test_formal_config_rejects_legacy_credit_modes(self) -> None:
        namespace = load_compute_advantage_namespace()
        base = {"AGENTMEMORY_REQUIRE_FORMAL_RUNTIME_EVIDENCE": "1"}
        with mock.patch.dict(
            os.environ,
            {
                **base,
                "AGENTMEMORY_FORMAL_ACTOR_ADVANTAGE_MODE": "mc_return_rms",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "legacy suffix/Monte-Carlo"):
                namespace["_validate_formal_actor_advantage_config"](
                    self.formal_config()
                )
        with mock.patch.dict(
            os.environ,
            {
                **base,
                "AGENTMEMORY_FORMAL_ACTOR_ADVANTAGE_MODE": "standard_trajectory_gae",
                "AGENTMEMORY_LATEST_OBS_SUFFIX_CREDIT": "1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "SUFFIX_CREDIT=0"):
                namespace["_validate_formal_actor_advantage_config"](
                    self.formal_config()
                )
        with mock.patch.dict(
            os.environ,
            {
                **base,
                "AGENTMEMORY_FORMAL_ACTOR_ADVANTAGE_MODE": "standard_trajectory_gae",
                "AGENTMEMORY_LATEST_OBS_SUFFIX_CREDIT": "0",
            },
            clear=True,
        ):
            self.assertEqual(
                namespace["_validate_formal_actor_advantage_config"](
                    self.formal_config()
                ),
                "standard_trajectory_gae",
            )
            with self.assertRaisesRegex(RuntimeError, "kl_ctrl.kl_coef=0"):
                namespace["_validate_formal_actor_advantage_config"](
                    self.formal_config(reward_kl=0.01)
                )

    def test_fresh_step_runtime_value_gate_is_wired(self) -> None:
        trainer_path = ROOT / "verl/agent_trainer/ppo/ray_trainer.py"
        source = trainer_path.read_text(encoding="utf-8")
        self.assertIn("AGENTMEMORY_EXPECT_INITIAL_CRITIC_ZERO", source)
        self.assertIn("validate_near_zero_critic_values", source)
        self.assertIn("agentmemory/initial_critic_value_max_abs", source)

    def test_actor_and_critic_honor_configured_ppo_epochs(self) -> None:
        for relative_path, class_name, method_name, metric_prefix in (
            (
                "verl/workers/agent_actor/dp_actor.py",
                "DataParallelPPOActor",
                "update_policy",
                "actor",
            ),
            (
                "verl/workers/agent_critic/dp_critic.py",
                "DataParallelPPOCritic",
                "update_critic",
                "critic",
            ),
        ):
            path = ROOT / relative_path
            source_text = path.read_text(encoding="utf-8")
            tree = ast.parse(source_text)
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
            method_source = ast.get_source_segment(source_text, method)
            self.assertIn("for _ in range(ppo_epochs):", method_source)
            self.assertIn("optimizer_steps += 1", method_source)
            self.assertIn(
                f"'{metric_prefix}/optimizer_steps_per_update'",
                method_source,
            )


if __name__ == "__main__":
    unittest.main()
