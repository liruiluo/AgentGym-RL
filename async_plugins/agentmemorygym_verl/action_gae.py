"""Action-axis GAE for one-response-per-action AgentMemoryGym trajectories.

The surrounding trainer, critic, clipped PPO loss, queue, staleness control,
and weight publication remain upstream veRL.  This module only maps the
ordered environment-action rows emitted by the AMG AgentLoop to GAE and then
broadcasts each action target over that row's sampled policy tokens.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_adv_est

TRAJECTORY_UID = "trajectory_uid"
TRAJECTORY_ROW_UID = "trajectory_row_uid"
TRAJECTORY_ROW_ORDER = "trajectory_row_order"
TRAJECTORY_TERMINAL = "trajectory_terminal"
ROLLOUT_DONE_FLAG = "rollout_done_flag"
IMMEDIATE_REWARD = "immediate_reward"
IS_PADDING = "is_padding"


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(config, name, default)


def _require_metadata(
    non_tensor_batch: Mapping[str, Any], key: str, row_count: int
) -> Sequence[Any]:
    if key not in non_tensor_batch:
        raise ValueError(f"AMG action GAE requires non-tensor metadata {key!r}")
    values = non_tensor_batch[key]
    if len(values) != row_count:
        raise ValueError(
            f"AMG action GAE metadata {key!r} must align with {row_count} rows, "
            f"got {len(values)}"
        )
    return values


def _as_bool(value: Any, *, field: str, row: int) -> bool:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be boolean at row {row}, got {value!r}")
    return bool(value)


def _as_nonnegative_int(value: Any, *, field: str, row: int) -> int:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer at row {row}, got bool")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} must be an integer at row {row}, got {value!r}"
        ) from exc
    try:
        exact = float(value) == float(integer)
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact or integer < 0:
        raise ValueError(
            f"{field} must be a non-negative integer at row {row}, got {value!r}"
        )
    return integer


def _as_finite_float(value: Any, *, field: str, row: int) -> float:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} must be numeric at row {row}, got {value!r}"
        ) from exc
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite at row {row}, got {value!r}")
    return number


@register_adv_est("amg_action_axis_gae")
def compute_amg_action_gae(
    *,
    batch: Mapping[str, torch.Tensor],
    non_tensor_batch: Mapping[str, Any],
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over ordered AMG actions and broadcast it to sampled tokens.

    Expected rows are atomic ``(state, sampled action, reward, next state, done)``
    records from a single ``rollout.n=1`` episode.  ``values[row, first valid
    response token]`` is the causal state value used for that action.  Synthetic
    veRL padding rows are ignored completely.
    """

    required_tensors = (
        "token_level_rewards",
        "values",
        "response_mask",
        "rollout_log_probs",
        "old_log_probs",
    )
    missing = [name for name in required_tensors if name not in batch]
    if missing:
        raise ValueError(f"AMG action GAE is missing tensor fields: {missing}")

    rewards = batch["token_level_rewards"]
    values = batch["values"]
    response_mask = batch["response_mask"]
    if (
        rewards.ndim != 2
        or values.shape != rewards.shape
        or response_mask.shape != rewards.shape
    ):
        raise ValueError(
            "AMG action GAE requires equal rank-2 reward/value/response-mask tensors: "
            f"rewards={tuple(rewards.shape)} values={tuple(values.shape)} "
            f"mask={tuple(response_mask.shape)}"
        )
    if not rewards.is_floating_point() or not values.is_floating_point():
        raise TypeError(
            "AMG action GAE reward and value tensors must be floating point"
        )
    if rewards.device != values.device or response_mask.device != values.device:
        raise ValueError("AMG action GAE tensors must share a device")
    if not bool(torch.logical_or(response_mask == 0, response_mask == 1).all().item()):
        raise ValueError("AMG action GAE response_mask must contain only zero or one")
    rollout_log_probs = batch["rollout_log_probs"]
    old_log_probs = batch["old_log_probs"]
    if rollout_log_probs.shape != rewards.shape or old_log_probs.shape != rewards.shape:
        raise ValueError(
            "AMG PPO requires rollout/old logprob tensors aligned with response tokens: "
            f"rollout={tuple(rollout_log_probs.shape)} "
            f"old={tuple(old_log_probs.shape)} rewards={tuple(rewards.shape)}"
        )
    if (
        not rollout_log_probs.is_floating_point()
        or not old_log_probs.is_floating_point()
    ):
        raise TypeError("AMG rollout and old logprob tensors must be floating point")
    if (
        rollout_log_probs.device != values.device
        or old_log_probs.device != values.device
    ):
        raise ValueError(
            "AMG rollout and old logprob tensors must share the batch device"
        )
    valid_token_mask = response_mask.to(dtype=torch.bool)
    rollout_policy_log_probs = rollout_log_probs[valid_token_mask]
    old_policy_log_probs = old_log_probs[valid_token_mask]
    if not bool(torch.isfinite(rollout_policy_log_probs).all().item()) or not bool(
        torch.isfinite(old_policy_log_probs).all().item()
    ):
        raise ValueError("AMG rollout and old policy-token logprobs must be finite")
    if not torch.equal(old_policy_log_probs, rollout_policy_log_probs):
        mismatch_count = int(
            torch.count_nonzero(old_policy_log_probs != rollout_policy_log_probs).item()
        )
        raise ValueError(
            "AMG bypass PPO requires old_log_probs to be exactly the rollout behavior "
            f"logprobs on every sampled policy token; mismatches={mismatch_count}"
        )

    row_count = rewards.shape[0]
    trajectory_uids = _require_metadata(non_tensor_batch, TRAJECTORY_UID, row_count)
    row_uids = _require_metadata(non_tensor_batch, TRAJECTORY_ROW_UID, row_count)
    row_orders = _require_metadata(non_tensor_batch, TRAJECTORY_ROW_ORDER, row_count)
    terminals = _require_metadata(non_tensor_batch, TRAJECTORY_TERMINAL, row_count)
    done_flags = _require_metadata(non_tensor_batch, ROLLOUT_DONE_FLAG, row_count)
    immediate_rewards = _require_metadata(non_tensor_batch, IMMEDIATE_REWARD, row_count)
    is_padding = non_tensor_batch.get(IS_PADDING, np.zeros(row_count, dtype=bool))
    if len(is_padding) != row_count:
        raise ValueError(
            f"AMG action GAE {IS_PADDING!r} must align with {row_count} rows"
        )

    gamma = float(_config_value(config, "gamma", 1.0))
    lam = float(_config_value(config, "lam", 1.0))
    tolerance = float(_config_value(config, "amg_reward_tolerance", 1e-6))
    normalization = str(_config_value(config, "amg_advantage_normalization", "none"))
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"AMG action GAE gamma must be in [0, 1], got {gamma!r}")
    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError(f"AMG action GAE lambda must be in [0, 1], got {lam!r}")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "AMG action GAE reward tolerance must be finite and non-negative"
        )
    if normalization not in {"none", "upstream_masked_whiten"}:
        raise ValueError(
            "AMG action GAE amg_advantage_normalization must be 'none' or "
            f"'upstream_masked_whiten', got {normalization!r}"
        )

    accumulator_dtype = torch.promote_types(rewards.dtype, values.dtype)
    if accumulator_dtype in (torch.float16, torch.bfloat16):
        accumulator_dtype = torch.float32
    real_policy_mask = valid_token_mask.clone()
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_row_uids: set[str] = set()
    for physical_row in range(row_count):
        if _as_bool(is_padding[physical_row], field=IS_PADDING, row=physical_row):
            real_policy_mask[physical_row] = False
            continue

        trajectory_uid = str(trajectory_uids[physical_row])
        row_uid = str(row_uids[physical_row])
        if not trajectory_uid or not row_uid:
            raise ValueError(
                f"AMG action GAE requires non-empty trajectory/row UID at row {physical_row}"
            )
        if row_uid in seen_row_uids:
            raise ValueError(f"duplicate AMG trajectory row UID {row_uid!r}")
        seen_row_uids.add(row_uid)

        token_indices = torch.nonzero(
            valid_token_mask[physical_row], as_tuple=False
        ).flatten()
        if token_indices.numel() == 0:
            raise ValueError(
                f"real AMG action row {physical_row} has no sampled response token"
            )
        expected_prefix = torch.arange(
            token_indices.numel(),
            device=token_indices.device,
            dtype=token_indices.dtype,
        )
        if not torch.equal(token_indices, expected_prefix):
            raise ValueError(
                f"AMG action row {physical_row} response mask is not a contiguous prefix"
            )
        if not torch.isfinite(values[physical_row, token_indices]).all():
            raise ValueError(
                f"AMG action row {physical_row} contains non-finite values"
            )
        if not torch.isfinite(rewards[physical_row, token_indices]).all():
            raise ValueError(
                f"AMG action row {physical_row} contains non-finite rewards"
            )

        immediate_reward = _as_finite_float(
            immediate_rewards[physical_row], field=IMMEDIATE_REWARD, row=physical_row
        )
        packed_reward = rewards[physical_row, token_indices].to(accumulator_dtype).sum()
        expected_reward = torch.as_tensor(
            immediate_reward, device=values.device, dtype=accumulator_dtype
        )
        if not torch.isclose(packed_reward, expected_reward, rtol=0.0, atol=tolerance):
            raise ValueError(
                "AMG packed token reward differs from the immediate action reward: "
                f"row={physical_row} packed={float(packed_reward.item())} "
                f"immediate={immediate_reward}"
            )

        trajectories[trajectory_uid].append(
            {
                "physical_row": physical_row,
                "row_order": _as_nonnegative_int(
                    row_orders[physical_row],
                    field=TRAJECTORY_ROW_ORDER,
                    row=physical_row,
                ),
                "terminal": _as_bool(
                    terminals[physical_row], field=TRAJECTORY_TERMINAL, row=physical_row
                ),
                "done": _as_bool(
                    done_flags[physical_row], field=ROLLOUT_DONE_FLAG, row=physical_row
                ),
                "reward": expected_reward,
                "token_indices": token_indices,
                "state_token_index": int(token_indices[0].item()),
            }
        )

    if not trajectories:
        raise ValueError("AMG action GAE requires at least one real trajectory row")

    advantages = torch.zeros(
        values.shape, device=values.device, dtype=accumulator_dtype
    )
    returns = torch.zeros_like(advantages)
    with torch.no_grad():
        for trajectory_uid, rows in trajectories.items():
            rows.sort(key=lambda row: row["row_order"])
            actual_orders = [row["row_order"] for row in rows]
            expected_orders = list(range(len(rows)))
            if actual_orders != expected_orders:
                raise ValueError(
                    "AMG trajectory row order is incomplete or duplicated: "
                    f"trajectory={trajectory_uid!r} expected={expected_orders} actual={actual_orders}"
                )
            terminal_orders = [row["row_order"] for row in rows if row["terminal"]]
            if terminal_orders != [len(rows) - 1]:
                raise ValueError(
                    "exactly the final AMG row must be trajectory-terminal: "
                    f"trajectory={trajectory_uid!r} actual={terminal_orders}"
                )
            premature_done = [row["row_order"] for row in rows[:-1] if row["done"]]
            if premature_done:
                raise ValueError(
                    "AMG environment done appears before the final trajectory row: "
                    f"trajectory={trajectory_uid!r} rows={premature_done}"
                )

            next_advantage = torch.zeros(
                (), device=values.device, dtype=accumulator_dtype
            )
            for reverse_index in range(len(rows) - 1, -1, -1):
                row = rows[reverse_index]
                physical_row = row["physical_row"]
                state_value = values[physical_row, row["state_token_index"]].to(
                    accumulator_dtype
                )
                has_next = reverse_index + 1 < len(rows) and not row["done"]
                if has_next:
                    next_row = rows[reverse_index + 1]
                    next_value = values[
                        next_row["physical_row"], next_row["state_token_index"]
                    ].to(accumulator_dtype)
                    continuation = 1.0
                else:
                    next_value = torch.zeros(
                        (), device=values.device, dtype=accumulator_dtype
                    )
                    continuation = 0.0
                delta = row["reward"] + gamma * continuation * next_value - state_value
                action_advantage = delta + gamma * lam * continuation * next_advantage
                action_return = action_advantage + state_value
                advantages[physical_row, row["token_indices"]] = action_advantage
                returns[physical_row, row["token_indices"]] = action_return
                next_advantage = action_advantage

    if normalization == "upstream_masked_whiten":
        advantages = verl_F.masked_whiten(advantages, real_policy_mask)
        advantages = advantages * real_policy_mask.to(dtype=advantages.dtype)

    return advantages, returns
