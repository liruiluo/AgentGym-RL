from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def _masked_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = values.masked_select(mask.bool())
    centered = valid - valid.mean()
    result = torch.zeros_like(values)
    result.masked_scatter_(mask.bool(), centered / torch.sqrt(centered.square().mean() + 1e-8))
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
    "agentmemory_standard_ppo_core_for_test",
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


def trajectory_gae(
    rewards,
    *,
    values=None,
    trajectory_uids=None,
    row_uids=None,
    row_orders=None,
    terminals=None,
    done_flags=None,
    sample_mask=None,
    gamma=1.0,
    lam=1.0,
):
    rewards = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)
    row_count = rewards.shape[0]
    if values is None:
        values = torch.zeros_like(rewards)
    else:
        values = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    if trajectory_uids is None:
        trajectory_uids = ["trajectory-a"] * row_count
    if row_orders is None:
        row_orders = list(range(row_count))
    if row_uids is None:
        row_uids = [f"row-{uid}-{order}" for uid, order in zip(trajectory_uids, row_orders)]
    if terminals is None:
        terminals = [False] * (row_count - 1) + [True]
    if done_flags is None:
        done_flags = [False] * (row_count - 1) + [True]
    if sample_mask is None:
        sample_mask = [True] * row_count
    return core_algos.compute_trajectory_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        eos_mask=torch.tensor(sample_mask, dtype=torch.float32).reshape(-1, 1),
        trajectory_uids=np.array(trajectory_uids, dtype=object),
        trajectory_row_uids=np.array(row_uids, dtype=object),
        trajectory_row_orders=torch.tensor(row_orders, dtype=torch.long),
        trajectory_terminals=torch.tensor(terminals, dtype=torch.bool),
        done_flags=np.array(done_flags, dtype=object),
        sample_mask=torch.tensor(sample_mask, dtype=torch.bool),
        gamma=gamma,
        lam=lam,
        immediate_rewards=rewards.flatten(),
        advantage_normalization="none",
    )


class StandardTrajectoryGaeTests(unittest.TestCase):
    def test_low_precision_reward_validation_uses_accumulator_dtype(self) -> None:
        rewards = torch.tensor([[0.01]], dtype=torch.bfloat16)
        advantages, returns = core_algos.compute_trajectory_gae_advantage_return(
            token_level_rewards=rewards,
            values=torch.zeros_like(rewards),
            eos_mask=torch.ones_like(rewards),
            trajectory_uids=np.array(["a"], dtype=object),
            trajectory_row_uids=np.array(["a-0"], dtype=object),
            trajectory_row_orders=torch.tensor([0]),
            trajectory_terminals=torch.tensor([True]),
            done_flags=np.array([True], dtype=object),
            sample_mask=torch.ones(1, dtype=torch.bool),
            gamma=1.0,
            lam=1.0,
            immediate_rewards=rewards.flatten(),
            advantage_normalization="none",
        )

        expected = rewards.float()
        self.assertEqual(advantages.dtype, torch.float32)
        self.assertEqual(returns.dtype, torch.float32)
        torch.testing.assert_close(advantages, expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(returns, expected, atol=0.0, rtol=0.0)

    def test_bfloat16_critic_keeps_float32_micro_reward_accumulation(self) -> None:
        row_count = 30
        rewards = torch.full((row_count, 1), 0.01, dtype=torch.float32)
        advantages, returns = core_algos.compute_trajectory_gae_advantage_return(
            token_level_rewards=rewards,
            values=torch.zeros((row_count, 1), dtype=torch.bfloat16),
            eos_mask=torch.ones((row_count, 1), dtype=torch.float32),
            trajectory_uids=np.array(["a"] * row_count, dtype=object),
            trajectory_row_uids=np.array(
                [f"a-{row_index}" for row_index in range(row_count)],
                dtype=object,
            ),
            trajectory_row_orders=torch.arange(row_count),
            trajectory_terminals=torch.tensor(
                [False] * (row_count - 1) + [True], dtype=torch.bool
            ),
            done_flags=np.array(
                [False] * (row_count - 1) + [True], dtype=object
            ),
            sample_mask=torch.ones(row_count, dtype=torch.bool),
            gamma=1.0,
            lam=1.0,
            immediate_rewards=rewards.flatten(),
            advantage_normalization="none",
        )

        expected = torch.arange(
            row_count, 0, -1, dtype=torch.float32
        ).reshape(-1, 1) * 0.01
        self.assertEqual(advantages.dtype, torch.float32)
        self.assertEqual(returns.dtype, torch.float32)
        torch.testing.assert_close(advantages, expected, atol=1e-7, rtol=0.0)
        torch.testing.assert_close(returns, expected, atol=1e-7, rtol=0.0)

    def test_later_correct_buy_gives_earlier_actions_positive_advantage(self) -> None:
        advantages, returns = trajectory_gae([0.0, 0.0, 1.0])
        torch.testing.assert_close(advantages.flatten(), torch.ones(3))
        torch.testing.assert_close(returns.flatten(), torch.ones(3))

    def test_correct_buy_stays_positive_before_small_terminal_penalty(self) -> None:
        advantages, returns = trajectory_gae([1.0, 0.0, -0.5])
        expected = torch.tensor([0.5, -0.5, -0.5])
        torch.testing.assert_close(advantages.flatten(), expected)
        torch.testing.assert_close(returns.flatten(), expected)
        self.assertGreater(returns[0, 0].item(), 0.0)

    def test_gamma_lambda_and_next_row_critic_value_are_aligned(self) -> None:
        advantages, returns = trajectory_gae(
            [0.0, 1.0],
            values=[0.2, 0.4],
            gamma=0.9,
            lam=0.8,
        )
        torch.testing.assert_close(
            advantages.flatten(), torch.tensor([0.592, 0.6]), atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            returns.flatten(), torch.tensor([0.792, 1.0]), atol=1e-6, rtol=0.0
        )

    def test_one_environment_action_advantage_is_shared_by_its_tokens(self) -> None:
        rewards = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        advantages, returns = core_algos.compute_trajectory_gae_advantage_return(
            token_level_rewards=rewards,
            values=torch.tensor([[0.2, 9.0], [0.4, 8.0]]),
            eos_mask=torch.ones_like(rewards),
            trajectory_uids=np.array(["a", "a"], dtype=object),
            trajectory_row_uids=np.array(["a-0", "a-1"], dtype=object),
            trajectory_row_orders=torch.tensor([0, 1]),
            trajectory_terminals=torch.tensor([False, True]),
            done_flags=np.array([False, True], dtype=object),
            sample_mask=torch.ones(2, dtype=torch.bool),
            gamma=0.9,
            lam=0.8,
            immediate_rewards=torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(
            advantages,
            torch.tensor([[0.592, 0.592], [0.6, 0.6]]),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            returns,
            torch.tensor([[0.792, 0.792], [1.0, 1.0]]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_interleaved_trajectories_do_not_leak_value_or_reward(self) -> None:
        advantages, _ = trajectory_gae(
            [0.0, -0.5, 1.0],
            trajectory_uids=["a", "b", "a"],
            row_orders=[0, 0, 1],
            terminals=[False, True, True],
            done_flags=[False, True, True],
        )
        torch.testing.assert_close(
            advantages.flatten(), torch.tensor([1.0, -0.5, 1.0])
        )

    def test_physical_row_reordering_preserves_environment_time(self) -> None:
        advantages, _ = trajectory_gae(
            [0.0, 0.0, 1.0],
            row_orders=[1, 0, 2],
            terminals=[False, False, True],
            done_flags=[False, False, True],
        )
        torch.testing.assert_close(advantages.flatten(), torch.ones(3))

    def test_transport_padding_is_excluded(self) -> None:
        advantages, returns = trajectory_gae(
            [0.0, 1.0, 999.0],
            trajectory_uids=["a", "a", "a"],
            row_uids=["row-a-0", "row-a-1", "row-a-0"],
            row_orders=[0, 1, 0],
            terminals=[False, True, False],
            done_flags=[False, True, False],
            sample_mask=[True, True, False],
        )
        torch.testing.assert_close(advantages.flatten(), torch.tensor([1.0, 1.0, 0.0]))
        torch.testing.assert_close(returns.flatten(), torch.tensor([1.0, 1.0, 0.0]))

    def test_duplicate_gap_and_premature_done_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate trajectory row UID"):
            trajectory_gae([0.0, 1.0], row_uids=["same", "same"])
        with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
            trajectory_gae([0.0, 1.0], row_orders=[0, 2])
        with self.assertRaisesRegex(ValueError, "done appears before"):
            trajectory_gae([0.0, 1.0], done_flags=[True, True])

    def test_suffix_reward_packed_as_immediate_reward_is_rejected(self) -> None:
        rewards = torch.tensor([[1.0], [1.0]])
        with self.assertRaisesRegex(ValueError, "packed environment reward differs"):
            core_algos.compute_trajectory_gae_advantage_return(
                token_level_rewards=rewards,
                values=torch.zeros_like(rewards),
                eos_mask=torch.ones_like(rewards),
                trajectory_uids=np.array(["a", "a"], dtype=object),
                trajectory_row_uids=np.array(["a-0", "a-1"], dtype=object),
                trajectory_row_orders=torch.tensor([0, 1]),
                trajectory_terminals=torch.tensor([False, True]),
                done_flags=np.array([False, True], dtype=object),
                sample_mask=torch.ones(2, dtype=torch.bool),
                gamma=1.0,
                lam=1.0,
                immediate_rewards=torch.tensor([0.0, 1.0]),
            )


class CriticAlignmentTests(unittest.TestCase):
    def test_critic_uses_the_same_pre_token_states_as_actor_log_probs(self) -> None:
        critic_path = ROOT / "verl/workers/agent_critic/dp_critic.py"
        tree = ast.parse(critic_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_select_response_state_values"
        )
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[function], type_ignores=[]), critic_path, "exec"), namespace)
        selected = namespace["_select_response_state_values"](
            torch.tensor([[10.0, 11.0, 20.0, 21.0, 22.0]]),
            torch.ones(1, 3),
        )
        torch.testing.assert_close(selected, torch.tensor([[11.0, 20.0, 21.0]]))


class FakeDataProto:
    def __init__(self) -> None:
        self.batch = {
            "values": torch.zeros(3, 1),
            "response_mask": torch.ones(3, 1),
            "token_level_rewards": torch.tensor([[0.0], [0.0], [1.0]]),
            "agentmemory_immediate_reward": torch.tensor([0.0, 0.0, 1.0]),
            "agentmemory_trajectory_row_order": torch.tensor([0, 1, 2]),
            "agentmemory_trajectory_terminal": torch.tensor([False, False, True]),
            core_algos.PPO_VALID_SAMPLE_MASK: torch.ones(3, dtype=torch.bool),
        }
        self.non_tensor_batch = {
            "agentmemory_trajectory_uid": np.array(["a", "a", "a"], dtype=object),
            "agentmemory_trajectory_row_uid": np.array(["a-0", "a-1", "a-2"], dtype=object),
            "rollout_done_flags": np.array([False, False, True], dtype=object),
        }
        self.meta_info = {}

    def __len__(self) -> int:
        return 3


def load_trainer_functions(validate_calls):
    trainer_path = ROOT / "verl/agent_trainer/ppo/ray_trainer.py"
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    names = {
        "_agentmemory_env_flag",
        "_get_ppo_valid_sample_mask",
        "compute_advantage",
        "_validate_formal_actor_advantage_config",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]

    def validate(*args, **kwargs):
        validate_calls.append(kwargs)
        return {"parent": []}

    namespace = {
        "AGENTMEMORY_IMMEDIATE_REWARD": "agentmemory_immediate_reward",
        "AGENTMEMORY_TRAJECTORY_ROW_ORDER": "agentmemory_trajectory_row_order",
        "AGENTMEMORY_TRAJECTORY_ROW_UID": "agentmemory_trajectory_row_uid",
        "AGENTMEMORY_TRAJECTORY_TERMINAL": "agentmemory_trajectory_terminal",
        "AGENTMEMORY_TRAJECTORY_UID": "agentmemory_trajectory_uid",
        "DataProto": FakeDataProto,
        "core_algos": core_algos,
        "nullcontext": nullcontext,
        "os": os,
        "requires_formal_trajectory_metadata": lambda data: True,
        "torch": torch,
        "validate_formal_trajectory_metadata": validate,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), trainer_path, "exec"), namespace)
    return namespace


class TrainerRoutingTests(unittest.TestCase):
    def test_formal_gae_requires_immediate_rewards_and_uses_environment_time(self) -> None:
        validate_calls = []
        functions = load_trainer_functions(validate_calls)
        data = FakeDataProto()
        functions["compute_advantage"](data, "gae", gamma=1.0, lam=1.0)
        torch.testing.assert_close(data.batch["advantages"], torch.ones(3, 1))
        self.assertFalse(validate_calls[0]["expected_suffix_credit"])
        self.assertEqual(
            data.meta_info["agentmemory_actor_advantage_mode"],
            "standard_trajectory_gae",
        )

    def test_formal_config_rejects_legacy_suffix_and_mc_modes(self) -> None:
        functions = load_trainer_functions([])
        config = SimpleNamespace(
            algorithm=SimpleNamespace(
                kl_ctrl=SimpleNamespace(kl_coef=0.0),
            )
        )
        base = {"AGENTMEMORY_REQUIRE_FORMAL_RUNTIME_EVIDENCE": "1"}
        with mock.patch.dict(
            os.environ,
            {**base, "AGENTMEMORY_FORMAL_ACTOR_ADVANTAGE_MODE": "mc_return_rms"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "legacy suffix/Monte-Carlo"):
                functions["_validate_formal_actor_advantage_config"](config)
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
                functions["_validate_formal_actor_advantage_config"](config)

    def test_no_suffix_timeout_penalty_is_copied_to_packed_handler(self) -> None:
        source = (
            ROOT / "verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py"
        ).read_text(encoding="utf-8")
        sync = "for handler, step in zip(flat_handlers, flat_step_refs):\n                handler.score = float(step[\"score\"])"
        self.assertIn(sync, source)
        self.assertLess(source.index("bind_max_round_timeout_failure("), source.index(sync))
        self.assertLess(source.index(sync), source.index("output = self.pack_rollout_handlers("))


if __name__ == "__main__":
    unittest.main()
