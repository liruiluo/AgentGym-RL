# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
from collections.abc import Sequence

import verl.utils.torch_functional as verl_F
from verl.utils.agentgym.formal_grpo_credit import compute_formal_grpo_credit


PPO_VALID_SAMPLE_MASK = "ppo_valid_sample_mask"


def masked_rms_scale_advantages(
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Scale masked advantages by RMS without subtracting their mean."""

    if advantages.shape != eos_mask.shape:
        raise ValueError(
            "advantages and eos_mask must have identical shapes, got "
            f"{tuple(advantages.shape)} and {tuple(eos_mask.shape)}."
        )
    if not advantages.is_floating_point():
        raise TypeError("advantages must be a floating-point tensor.")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon!r}.")

    mask = eos_mask.to(dtype=torch.bool)
    valid_advantages = advantages.masked_select(mask)
    if valid_advantages.numel() == 0:
        raise ValueError("RMS advantage scaling requires at least one valid token.")
    if not torch.isfinite(valid_advantages).all():
        raise ValueError("RMS advantage scaling received non-finite valid advantages.")

    accumulator_dtype = (
        torch.float64 if advantages.dtype == torch.float64 else torch.float32
    )
    mean_square = valid_advantages.to(accumulator_dtype).square().mean()
    rms = torch.sqrt(mean_square + epsilon)
    if not torch.isfinite(rms) or rms <= 0.0:
        raise ValueError(f"RMS advantage scale is invalid: {rms!r}.")

    scaled_valid = valid_advantages / rms.to(dtype=valid_advantages.dtype)
    if not torch.isfinite(scaled_valid).all():
        raise ValueError("RMS advantage scaling produced non-finite values.")
    if not torch.equal(torch.sign(scaled_valid), torch.sign(valid_advantages)):
        raise RuntimeError("RMS advantage scaling changed a valid advantage sign.")

    scaled = torch.zeros_like(advantages)
    scaled.masked_scatter_(mask, scaled_valid)
    return scaled


def validate_and_scale_monte_carlo_actor_advantages(
    returns: torch.Tensor,
    eos_mask: torch.Tensor,
    declared_suffix_returns: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> torch.Tensor:
    """Use exact action-row return-to-go for actor credit, with RMS scaling."""

    if returns.shape != eos_mask.shape:
        raise ValueError("returns and eos_mask must have identical shapes.")
    if declared_suffix_returns.ndim != 1 or declared_suffix_returns.shape[0] != returns.shape[0]:
        raise ValueError(
            "declared_suffix_returns must have one scalar per action row."
        )
    if not torch.isfinite(declared_suffix_returns).all():
        raise ValueError("declared_suffix_returns contains non-finite values.")
    mask = eos_mask.bool()
    expected = declared_suffix_returns.to(
        device=returns.device,
        dtype=returns.dtype,
    ).unsqueeze(-1).expand_as(returns)
    if not torch.allclose(
        returns.masked_select(mask),
        expected.masked_select(mask),
        rtol=0.0,
        atol=atol,
    ):
        max_error = (
            returns.masked_select(mask) - expected.masked_select(mask)
        ).abs().max().item()
        raise RuntimeError(
            "Formal Monte Carlo actor return does not match declared suffix return: "
            f"max_abs_error={max_error}."
        )
    scaled = masked_rms_scale_advantages(expected, eos_mask)
    valid_scaled = scaled.masked_select(mask)
    valid_expected = expected.masked_select(mask)
    if not torch.equal(torch.sign(valid_scaled), torch.sign(valid_expected)):
        raise RuntimeError("Formal Monte Carlo actor scaling changed a return sign.")
    return scaled


def validate_near_zero_critic_values(
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    atol: float = 1e-6,
) -> float:
    """Fail closed unless the fresh formal critic emits near-zero values."""

    if values.shape != eos_mask.shape:
        raise ValueError(
            "critic values and eos_mask must have identical shapes, got "
            f"{tuple(values.shape)} and {tuple(eos_mask.shape)}."
        )
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(f"atol must be finite and non-negative, got {atol!r}.")
    valid_values = values.masked_select(eos_mask.to(dtype=torch.bool))
    if valid_values.numel() == 0:
        raise ValueError("Fresh critic validation requires at least one valid token.")
    if not torch.isfinite(valid_values).all():
        raise RuntimeError("Fresh formal critic emitted non-finite values.")
    max_abs = float(valid_values.abs().max().item())
    if max_abs > atol:
        raise RuntimeError(
            "Fresh formal critic value head is not near zero: "
            f"max_abs={max_abs:.8g} atol={atol:.8g}."
        )
    return max_abs


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor,
                                 advantage_normalization: str = "whiten"):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        advantage_normalization: `(str)`
            ``whiten`` keeps the legacy centered normalization. ``rms`` scales
            without mean subtraction so raw advantage signs are preserved.

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            # nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            # delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            nextvalues = (values[:, t + 1] * eos_mask[:, t] + (1 - eos_mask[:, t]) * nextvalues) if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] * eos_mask[:, t] + gamma * nextvalues + (1 - eos_mask[:, t]) * (1 - gamma) * nextvalues - eos_mask[:, t] * values[:, t]
            lastgaelam = delta * eos_mask[:, t] + gamma * lam * lastgaelam + (1 - eos_mask[:, t]) * (1 - gamma * lam) * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        if advantage_normalization == "whiten":
            advantages = verl_F.masked_whiten(advantages, eos_mask)
        elif advantage_normalization == "rms":
            advantages = masked_rms_scale_advantages(advantages, eos_mask)
        else:
            raise ValueError(
                "advantage_normalization must be 'whiten' or 'rms', got "
                f"{advantage_normalization!r}."
            )
    return advantages, returns


def compute_trajectory_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    *,
    trajectory_uids: Sequence[object],
    trajectory_row_uids: Sequence[object],
    trajectory_row_orders: torch.Tensor,
    trajectory_terminals: torch.Tensor,
    done_flags: Sequence[object],
    sample_mask: torch.Tensor | None,
    gamma: float,
    lam: float,
    immediate_rewards: torch.Tensor | None = None,
    advantage_normalization: str = "none",
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over ordered environment actions, then expand to tokens.

    Each row is one atomic ``(state, action, reward, next_state, done)`` step.
    Its state value is the critic output aligned with the first response token;
    the next ordered row therefore supplies ``V(next_state)``. One action-level
    advantage and return are expanded across that row's sampled policy tokens
    for the ordinary clipped actor and critic losses.
    """

    if (
        token_level_rewards.ndim != 2
        or values.shape != token_level_rewards.shape
        or eos_mask.shape != token_level_rewards.shape
    ):
        raise ValueError(
            "trajectory GAE requires equal rank-2 reward, value, and mask tensors: "
            f"rewards={tuple(token_level_rewards.shape)} "
            f"values={tuple(values.shape)} mask={tuple(eos_mask.shape)}."
        )
    if not token_level_rewards.is_floating_point() or not values.is_floating_point():
        raise TypeError("trajectory GAE rewards and values must be floating-point tensors.")
    if token_level_rewards.device != values.device or eos_mask.device != values.device:
        raise ValueError("trajectory GAE rewards, values, and mask must share a device.")

    accumulator_dtype = torch.promote_types(
        token_level_rewards.dtype, values.dtype
    )
    if accumulator_dtype in (torch.float16, torch.bfloat16):
        accumulator_dtype = torch.float32

    row_count = token_level_rewards.shape[0]
    named_metadata = {
        "trajectory_uids": trajectory_uids,
        "trajectory_row_uids": trajectory_row_uids,
        "trajectory_row_orders": trajectory_row_orders,
        "trajectory_terminals": trajectory_terminals,
        "done_flags": done_flags,
    }
    lengths = {name: len(value) for name, value in named_metadata.items()}
    if any(length != row_count for length in lengths.values()):
        raise ValueError(
            f"trajectory GAE metadata must align with {row_count} rows: {lengths}."
        )
    if trajectory_row_orders.ndim != 1 or trajectory_terminals.ndim != 1:
        raise ValueError("trajectory row orders and terminal flags must be one-dimensional.")

    try:
        gamma = float(gamma)
        lam = float(lam)
        tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory GAE gamma, lambda, and tolerance must be numeric.") from exc
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"trajectory GAE gamma must be in [0, 1], got {gamma!r}.")
    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError(f"trajectory GAE lambda must be in [0, 1], got {lam!r}.")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"trajectory GAE tolerance must be finite and non-negative, got {tolerance!r}."
        )

    if sample_mask is None:
        sample_mask = torch.ones(row_count, dtype=torch.bool, device=values.device)
    elif sample_mask.ndim != 1 or sample_mask.shape[0] != row_count:
        raise ValueError(
            f"trajectory GAE sample_mask must have shape ({row_count},), "
            f"got {tuple(sample_mask.shape)}."
        )
    else:
        sample_mask = sample_mask.to(device=values.device, dtype=torch.bool)
    if not torch.any(sample_mask):
        raise ValueError("trajectory GAE requires at least one valid action row.")

    if immediate_rewards is not None:
        if immediate_rewards.ndim != 1 or immediate_rewards.shape[0] != row_count:
            raise ValueError(
                f"immediate_rewards must have shape ({row_count},), "
                f"got {tuple(immediate_rewards.shape)}."
            )
        if not immediate_rewards.is_floating_point():
            raise TypeError("immediate_rewards must be a floating-point tensor.")
        immediate_rewards = immediate_rewards.to(
            device=values.device, dtype=accumulator_dtype
        )

    mask_is_binary = torch.logical_or(eos_mask == 0, eos_mask == 1)
    if not bool(mask_is_binary.all().item()):
        raise ValueError("trajectory GAE response mask must contain only zero or one.")
    valid_token_mask = eos_mask.to(dtype=torch.bool)

    row_orders = trajectory_row_orders.detach().cpu().tolist()
    terminal_flags = trajectory_terminals.detach().cpu().tolist()
    trajectories: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen_row_uids: set[str] = set()
    for row_index in range(row_count):
        if not bool(sample_mask[row_index].item()):
            continue
        trajectory_uid = str(trajectory_uids[row_index])
        row_uid = str(trajectory_row_uids[row_index])
        if not trajectory_uid or not row_uid:
            raise ValueError(
                f"trajectory GAE requires non-empty trajectory and row UIDs at row {row_index}."
            )
        if row_uid in seen_row_uids:
            raise ValueError(f"duplicate trajectory row UID at row {row_index}: {row_uid!r}.")
        seen_row_uids.add(row_uid)

        raw_order = row_orders[row_index]
        if isinstance(raw_order, bool) or int(raw_order) != raw_order or int(raw_order) < 0:
            raise ValueError(
                f"trajectory row order must be a non-negative integer at row {row_index}."
            )
        terminal = terminal_flags[row_index]
        done = done_flags[row_index]
        if not isinstance(terminal, (bool, np.bool_)):
            raise ValueError(f"trajectory terminal must be boolean at row {row_index}.")
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError(f"environment done flag must be boolean at row {row_index}.")

        token_indices = torch.nonzero(valid_token_mask[row_index], as_tuple=False).flatten()
        if token_indices.numel() == 0:
            raise ValueError(f"valid trajectory row {row_index} has no response tokens.")
        expected_indices = torch.arange(
            token_indices.numel(), device=token_indices.device, dtype=token_indices.dtype
        )
        if not torch.equal(token_indices, expected_indices):
            raise ValueError(
                f"trajectory row {row_index} response mask is not a contiguous prefix."
            )
        row_rewards = token_level_rewards[row_index, token_indices]
        row_values = values[row_index, token_indices]
        if not torch.isfinite(row_rewards).all() or not torch.isfinite(row_values).all():
            raise ValueError(
                f"trajectory row {row_index} contains non-finite valid rewards or values."
            )
        if immediate_rewards is not None:
            expected_reward = immediate_rewards[row_index]
            if not torch.isfinite(expected_reward):
                raise ValueError(f"immediate reward is non-finite at row {row_index}.")
            if row_rewards.numel() > 1 and not torch.allclose(
                row_rewards[:-1],
                torch.zeros_like(row_rewards[:-1]),
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError(
                    f"environment reward must appear only on the final token at row {row_index}."
                )
            if not torch.isclose(
                row_rewards[-1].to(accumulator_dtype),
                expected_reward,
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError(
                    "packed environment reward differs from the immediate action reward: "
                    f"row={row_index} packed={float(row_rewards[-1].item())} "
                    f"immediate={float(expected_reward.item())}."
                )

        trajectories[trajectory_uid].append(
            {
                "row_index": row_index,
                "row_order": int(raw_order),
                "terminal": bool(terminal),
                "done": bool(done),
                "token_indices": token_indices.tolist(),
            }
        )

    ordered_trajectories = []
    for trajectory_uid, rows in trajectories.items():
        rows.sort(key=lambda row: int(row["row_order"]))
        actual_orders = [int(row["row_order"]) for row in rows]
        expected_orders = list(range(len(rows)))
        if actual_orders != expected_orders:
            raise ValueError(
                "trajectory row order is incomplete or duplicated: "
                f"trajectory={trajectory_uid!r} expected={expected_orders} "
                f"actual={actual_orders}."
            )
        terminal_orders = [
            int(row["row_order"]) for row in rows if bool(row["terminal"])
        ]
        if terminal_orders != [len(rows) - 1]:
            raise ValueError(
                "exactly the final row must be rollout-terminal: "
                f"trajectory={trajectory_uid!r} actual={terminal_orders}."
            )
        premature_done = [
            int(row["row_order"]) for row in rows[:-1] if bool(row["done"])
        ]
        if premature_done:
            raise ValueError(
                "environment done appears before the final trajectory row: "
                f"trajectory={trajectory_uid!r} rows={premature_done}."
            )
        ordered_trajectories.append(rows)

    trajectory_count = len(ordered_trajectories)
    max_action_count = max(len(rows) for rows in ordered_trajectories)
    packed_values = torch.zeros(
        trajectory_count,
        max_action_count,
        device=values.device,
        dtype=accumulator_dtype,
    )
    packed_rewards = torch.zeros_like(packed_values)
    packed_done = torch.zeros(
        trajectory_count,
        max_action_count,
        device=values.device,
        dtype=torch.bool,
    )
    packed_mask = torch.zeros_like(packed_done)
    for trajectory_index, rows in enumerate(ordered_trajectories):
        row_indices = torch.tensor(
            [int(row["row_index"]) for row in rows],
            device=values.device,
            dtype=torch.long,
        )
        state_token_indices = torch.tensor(
            [int(row["token_indices"][0]) for row in rows],
            device=values.device,
            dtype=torch.long,
        )
        action_count = len(rows)
        packed_values[trajectory_index, :action_count] = values[
            row_indices, state_token_indices
        ].to(accumulator_dtype)
        if immediate_rewards is None:
            packed_rewards[trajectory_index, :action_count] = torch.stack(
                [
                    token_level_rewards[
                        int(row["row_index"]), row["token_indices"]
                    ].to(accumulator_dtype).sum()
                    for row in rows
                ]
            )
        else:
            packed_rewards[trajectory_index, :action_count] = immediate_rewards[
                row_indices
            ]
        packed_done[trajectory_index, :action_count] = torch.tensor(
            [bool(row["done"]) for row in rows],
            device=values.device,
            dtype=torch.bool,
        )
        packed_mask[trajectory_index, :action_count] = True

    packed_advantages = torch.zeros_like(packed_values)
    last_gae = torch.zeros(
        trajectory_count, device=values.device, dtype=accumulator_dtype
    )
    with torch.no_grad():
        for action_index in reversed(range(max_action_count)):
            valid = packed_mask[:, action_index]
            if action_index + 1 < max_action_count:
                next_valid = packed_mask[:, action_index + 1]
                next_values = packed_values[:, action_index + 1]
            else:
                next_valid = torch.zeros_like(valid)
                next_values = torch.zeros_like(last_gae)
            continuation = next_valid & ~packed_done[:, action_index]
            delta = (
                packed_rewards[:, action_index]
                + gamma * continuation.to(accumulator_dtype) * next_values
                - packed_values[:, action_index]
            )
            last_gae = torch.where(
                valid,
                delta
                + gamma * lam * continuation.to(accumulator_dtype) * last_gae,
                torch.zeros_like(last_gae),
            )
            packed_advantages[:, action_index] = last_gae
    packed_returns = packed_advantages + packed_values

    advantages = torch.zeros(
        values.shape, device=values.device, dtype=accumulator_dtype
    )
    returns = torch.zeros_like(advantages)
    for trajectory_index, rows in enumerate(ordered_trajectories):
        for action_index, row in enumerate(rows):
            row_index = int(row["row_index"])
            token_indices = row["token_indices"]
            advantages[row_index, token_indices] = packed_advantages[
                trajectory_index, action_index
            ]
            returns[row_index, token_indices] = packed_returns[
                trajectory_index, action_index
            ]

    if advantage_normalization == "none":
        pass
    elif advantage_normalization == "whiten":
        advantages = verl_F.masked_whiten(advantages, valid_token_mask & sample_mask.unsqueeze(-1))
    else:
        raise ValueError(
            "trajectory GAE advantage_normalization must be 'none' or 'whiten', "
            f"got {advantage_normalization!r}."
        )
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: np.ndarray,
                                   sample_mask: torch.Tensor | None = None,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            group identifier for each response.
        sample_mask: `(torch.Tensor)`, optional
            shape: (bs,). False entries are DP padding and do not participate
            in group statistics or receive an advantage.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    if sample_mask is None:
        sample_mask = torch.ones_like(scores, dtype=torch.bool)
    else:
        if sample_mask.ndim != 1 or sample_mask.shape[0] != scores.shape[0]:
            raise ValueError(
                f"sample_mask must have shape ({scores.shape[0]},), got {tuple(sample_mask.shape)}"
            )
        sample_mask = sample_mask.to(device=scores.device, dtype=torch.bool)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if sample_mask[i]:
                id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # A singleton has no within-group relative signal. Returning
                # its raw reward would turn GRPO into uncentered REINFORCE.
                id2mean[idx] = id2score[idx][0]
                id2std[idx] = torch.ones_like(id2score[idx][0])
                continue
            group_scores = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(group_scores)
            id2std[idx] = torch.std(group_scores)
        normalized_scores = torch.zeros_like(scores)
        for i in range(bsz):
            if sample_mask[i]:
                normalized_scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        normalized_scores = normalized_scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return normalized_scores, normalized_scores


def compute_grpo_trajectory_outcome_advantage(
    trajectory_returns: torch.Tensor,
    eos_mask: torch.Tensor,
    parent_group_uids: np.ndarray,
    trajectory_uids: np.ndarray,
    replica_indices: np.ndarray,
    sample_mask: torch.Tensor | None = None,
    expected_replicas: int | None = None,
    epsilon: float = 1e-6,
):
    """Normalize complete trajectory returns once, then broadcast to actions.

    Each action row carries its trajectory's scalar return.  Group statistics
    are computed over unique trajectories, so a 50-action continuation has the
    same weight as a 3-action continuation from the same initial task.
    """

    if trajectory_returns.ndim != 1:
        raise ValueError(
            "trajectory_returns must be one-dimensional, got "
            f"{tuple(trajectory_returns.shape)}"
        )
    batch_size = trajectory_returns.shape[0]
    if eos_mask.ndim != 2 or eos_mask.shape[0] != batch_size:
        raise ValueError(
            "eos_mask must have shape (batch, response_length), got "
            f"{tuple(eos_mask.shape)} for batch {batch_size}"
        )
    metadata = {
        "parent_group_uids": parent_group_uids,
        "trajectory_uids": trajectory_uids,
        "replica_indices": replica_indices,
    }
    metadata_lengths = {name: len(values) for name, values in metadata.items()}
    if any(length != batch_size for length in metadata_lengths.values()):
        raise ValueError(
            "Trajectory GRPO metadata length mismatch: "
            f"batch={batch_size} metadata={metadata_lengths}"
        )
    if sample_mask is None:
        sample_mask = torch.ones(
            batch_size, dtype=torch.bool, device=trajectory_returns.device
        )
    else:
        if sample_mask.ndim != 1 or sample_mask.shape[0] != batch_size:
            raise ValueError(
                f"sample_mask must have shape ({batch_size},), "
                f"got {tuple(sample_mask.shape)}"
            )
        sample_mask = sample_mask.to(
            device=trajectory_returns.device, dtype=torch.bool
        )
    if not torch.any(sample_mask):
        raise ValueError("Trajectory GRPO batch has no valid action rows.")
    if expected_replicas is not None and expected_replicas <= 0:
        raise ValueError(
            f"expected_replicas must be positive, got {expected_replicas}."
        )

    trajectory_records = {}
    parent_to_trajectories = defaultdict(dict)
    with torch.no_grad():
        for row_index in range(batch_size):
            if not bool(sample_mask[row_index].item()):
                continue
            parent_group_uid = str(parent_group_uids[row_index])
            trajectory_uid = str(trajectory_uids[row_index])
            try:
                replica_index = int(replica_indices[row_index])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid trajectory replica index at row {row_index}: "
                    f"{replica_indices[row_index]!r}"
                ) from exc
            trajectory_return = trajectory_returns[row_index]
            if not torch.isfinite(trajectory_return):
                raise ValueError(
                    f"Non-finite trajectory return at row {row_index}."
                )
            owner = (parent_group_uid, replica_index)
            previous_uid = parent_to_trajectories[parent_group_uid].setdefault(
                replica_index, trajectory_uid
            )
            if previous_uid != trajectory_uid:
                raise ValueError(
                    "One parent/replica pair maps to multiple trajectories: "
                    f"owner={owner!r} uids={previous_uid!r},{trajectory_uid!r}"
                )
            record = trajectory_records.setdefault(
                trajectory_uid,
                {
                    "owner": owner,
                    "return": trajectory_return,
                },
            )
            if record["owner"] != owner:
                raise ValueError(
                    f"Trajectory UID {trajectory_uid!r} crosses parent groups."
                )
            if not torch.isclose(
                record["return"], trajectory_return, rtol=epsilon, atol=epsilon
            ):
                raise ValueError(
                    "Conflicting trajectory return within one trajectory: "
                    f"uid={trajectory_uid!r}"
                )

        normalized_by_trajectory = {}
        for parent_group_uid, replica_map in parent_to_trajectories.items():
            replica_set = set(replica_map)
            if expected_replicas is not None:
                expected_set = set(range(expected_replicas))
                if replica_set != expected_set:
                    raise ValueError(
                        "Trajectory GRPO parent group has incomplete replicas: "
                        f"group={parent_group_uid!r} "
                        f"expected={sorted(expected_set)} actual={sorted(replica_set)}"
                    )
            ordered_replica_indices = sorted(replica_map)
            group_returns = torch.stack(
                [
                    trajectory_records[replica_map[replica_index]]["return"]
                    for replica_index in ordered_replica_indices
                ]
            )
            group_mean = group_returns.mean()
            if len(group_returns) == 1:
                group_std = torch.ones_like(group_mean)
            else:
                group_std = group_returns.std()
            for replica_index, trajectory_return in zip(
                ordered_replica_indices, group_returns
            ):
                trajectory_uid = replica_map[replica_index]
                normalized_by_trajectory[trajectory_uid] = (
                    trajectory_return - group_mean
                ) / (group_std + epsilon)

        row_advantages = torch.zeros_like(trajectory_returns)
        for row_index in range(batch_size):
            if bool(sample_mask[row_index].item()):
                row_advantages[row_index] = normalized_by_trajectory[
                    str(trajectory_uids[row_index])
                ]
        token_advantages = row_advantages.unsqueeze(-1) * eos_mask
    return token_advantages, token_advantages


def compute_formal_grpo_complete_trajectory_advantage(
    immediate_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    parent_group_uids: np.ndarray,
    trajectory_uids: np.ndarray,
    trajectory_row_uids: np.ndarray,
    trajectory_row_orders: torch.Tensor,
    trajectory_terminals: torch.Tensor,
    declared_trajectory_returns: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
    expected_group_size: int | None = None,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
):
    """Recompute complete returns and broadcast parent-group GRPO credit.

    The declared return is checked as evidence but is never used as the target.
    Batch rows may have been permuted by sequence-length balancing; explicit
    trajectory row identities recover environment time and reject column drift.
    """

    if immediate_rewards.ndim != 1:
        raise ValueError(
            "immediate_rewards must be one-dimensional, got "
            f"{tuple(immediate_rewards.shape)}."
        )
    batch_size = immediate_rewards.shape[0]
    if eos_mask.ndim != 2 or eos_mask.shape[0] != batch_size:
        raise ValueError(
            "eos_mask must have shape (batch, response_length), got "
            f"{tuple(eos_mask.shape)}."
        )
    if expected_group_size is None:
        raise ValueError("Formal GRPO requires an explicit expected_group_size.")
    tensor_metadata = {
        "trajectory_row_orders": trajectory_row_orders,
        "trajectory_terminals": trajectory_terminals,
        "declared_trajectory_returns": declared_trajectory_returns,
    }
    for name, values in tensor_metadata.items():
        if values.ndim != 1 or values.shape[0] != batch_size:
            raise ValueError(
                f"{name} must have shape ({batch_size},), got {tuple(values.shape)}."
            )
    array_metadata = {
        "parent_group_uids": parent_group_uids,
        "trajectory_uids": trajectory_uids,
        "trajectory_row_uids": trajectory_row_uids,
    }
    for name, values in array_metadata.items():
        if np.asarray(values, dtype=object).ndim != 1 or len(values) != batch_size:
            raise ValueError(
                f"{name} must be one-dimensional with {batch_size} rows."
            )
    if sample_mask is None:
        sample_mask = torch.ones(
            batch_size, dtype=torch.bool, device=immediate_rewards.device
        )
    elif sample_mask.ndim != 1 or sample_mask.shape[0] != batch_size:
        raise ValueError(
            f"sample_mask must have shape ({batch_size},), got {tuple(sample_mask.shape)}."
        )
    sample_mask = sample_mask.to(device=immediate_rewards.device, dtype=torch.bool)
    valid_indices = sample_mask.nonzero(as_tuple=False).flatten().tolist()
    if not valid_indices:
        raise ValueError("Formal GRPO batch has no valid action rows.")

    reward_values = immediate_rewards.detach().cpu().tolist()
    order_values = trajectory_row_orders.detach().cpu().tolist()
    terminal_values = trajectory_terminals.detach().cpu().tolist()
    declared_values = declared_trajectory_returns.detach().cpu().tolist()
    rows = [
        {
            "parent_uid": parent_group_uids[index],
            "trajectory_uid": trajectory_uids[index],
            "row_uid": trajectory_row_uids[index],
            "row_order": order_values[index],
            "terminal": terminal_values[index],
            "immediate_reward": reward_values[index],
            "declared_trajectory_return": declared_values[index],
        }
        for index in valid_indices
    ]
    credit = compute_formal_grpo_credit(
        rows,
        expected_group_size=int(expected_group_size),
        gamma=float(gamma),
        epsilon=float(epsilon),
    )
    row_advantages = torch.zeros_like(immediate_rewards)
    for credit_row in credit.rows:
        original_index = valid_indices[credit_row.input_index]
        row_advantages[original_index] = credit_row.advantage
    token_advantages = row_advantages.unsqueeze(-1) * eos_mask
    return token_advantages, token_advantages


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            # running_return = token_level_rewards[:, t] + gamma * running_return
            running_return = token_level_rewards[:, t] * eos_mask[:, t] + gamma * running_return + (1 - eos_mask[:, t]) * (1 - gamma) * running_return
            returns[:, t] = running_return * eos_mask[:, t]
            # Reset after EOS
            # running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns

def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.ndarray,
    sample_mask: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)
    if sample_mask is None:
        sample_mask = torch.ones_like(scores, dtype=torch.bool)
    else:
        if sample_mask.ndim != 1 or sample_mask.shape[0] != scores.shape[0]:
            raise ValueError(
                f"sample_mask must have shape ({scores.shape[0]},), got {tuple(sample_mask.shape)}"
            )
        sample_mask = sample_mask.to(device=scores.device, dtype=torch.bool)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if sample_mask[i]:
                id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        masked_scores = torch.zeros_like(scores)
        for i in range(bsz):
            if not sample_mask[i]:
                continue
            response_num = len(id2score[index[i]])
            if response_num > 1:
                masked_scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
            else:
                masked_scores[i] = scores[i]
        masked_scores = masked_scores.unsqueeze(-1) * eos_mask

    return masked_scores, masked_scores

def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    eos_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    # Match the stability guard used by newer VERL: long-horizon agentic
    # rollouts can occasionally produce extreme log-prob deltas, and
    # torch.exp(log_prob - old_log_prob) then overflows before PPO clipping can
    # help. Clamping keeps the ratio finite without changing the clipped PPO
    # objective in the normal range.
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
