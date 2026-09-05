"""Token-axis credit assignment for ordered AgentMemoryGym action rows.

The environment and the task-neutral AgentLoop emit one learner row per sampled
policy action.  veRL already supplies token-level values, rewards, masks, and
policy log probabilities for each row.  This module is the thin adapter that
reconstructs the complete episode's *policy-token* timeline before applying
GAE.  Environment observations and context-replacement templates are skipped;
the last token of one action therefore bootstraps directly from the first token
of the next action, as in SAO's skip-observation GAE.

Actor advantages use the VAPO length-adaptive policy-lambda formula adopted by
SAO.  Critic targets are computed by a separate lambda=1 recurrence.
PPO/value losses, rollout correction, queues, and online weight publication
remain native veRL.
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
MIN_GLOBAL_STEPS = "min_global_steps"
MAX_GLOBAL_STEPS = "max_global_steps"
OUTCOME = "outcome"
DECLARED_MAX_ROUNDS = "declared_max_rounds"
TERMINATION_KIND = "termination_kind"
HORIZON_FINALIZER_RECEIPT = "horizon_finalizer_receipt"

_MAX_ROUNDS_FINALIZER_RECEIPTS = {
    "no_terminal_transition:no_hook",
    "no_terminal_transition:returned_none",
}


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
        raise ValueError(f"AMG token GAE requires non-tensor metadata {key!r}")
    values = non_tensor_batch[key]
    if len(values) != row_count:
        raise ValueError(
            f"AMG token GAE metadata {key!r} must align with {row_count} rows, "
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
        exact = float(value) == float(integer)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} must be an integer at row {row}, got {value!r}"
        ) from exc
    if not exact or integer < 0:
        raise ValueError(
            f"{field} must be a non-negative integer at row {row}, got {value!r}"
        )
    return integer


def _as_positive_int(value: Any, *, field: str, row: int) -> int:
    integer = _as_nonnegative_int(value, field=field, row=row)
    if integer == 0:
        raise ValueError(f"{field} must be positive at row {row}, got {value!r}")
    return integer


def _as_nonempty_str(value: Any, *, field: str, row: int) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string at row {row}, got {value!r}")
    return value


def _as_finite_float(value: Any, *, field: str, row: int | None = None) -> float:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    location = "" if row is None else f" at row {row}"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be numeric{location}, got {value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite{location}, got {value!r}")
    return number


def length_adaptive_policy_lambda(token_count: int, *, scale: float = 1.5) -> float:
    """Return ``1 - 1 / (scale * token_count)`` with a fail-closed domain."""

    if isinstance(token_count, bool):
        raise TypeError("token_count must be a positive integer, got bool")
    try:
        parsed_count = int(token_count)
        exact_count = float(token_count) == float(parsed_count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"token_count must be a positive integer, got {token_count!r}"
        ) from exc
    if not exact_count or parsed_count <= 0:
        raise ValueError(
            f"token_count must be a positive integer, got {token_count!r}"
        )
    parsed_scale = _as_finite_float(scale, field="policy lambda scale")
    policy_lambda = 1.0 - 1.0 / (parsed_scale * parsed_count)
    if not 0.0 <= policy_lambda <= 1.0:
        raise ValueError(
            "length-adaptive policy lambda must lie in [0, 1], got "
            f"{policy_lambda} from scale={parsed_scale}, token_count={parsed_count}"
        )
    return policy_lambda


@register_adv_est("amg_sao_token_gae")
def compute_amg_sao_token_gae(
    *,
    batch: Mapping[str, torch.Tensor],
    non_tensor_batch: Mapping[str, Any],
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact token-axis GAE across all action rows in each episode.

    Rows may have been generated by different published policy versions.  The
    rollout behavior probability is carried per token, so policy-version
    changes do not split the credit chain.  Synthetic alignment rows are fully
    excluded from validation, normalization, and returned targets.
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
        raise ValueError(f"AMG token GAE is missing tensor fields: {missing}")

    rewards = batch["token_level_rewards"]
    values = batch["values"]
    response_mask = batch["response_mask"]
    rollout_log_probs = batch["rollout_log_probs"]
    old_log_probs = batch["old_log_probs"]
    if rewards.ndim != 2:
        raise ValueError(
            f"AMG token GAE requires rank-2 tensors, got rewards={tuple(rewards.shape)}"
        )
    for name, tensor in {
        "values": values,
        "response_mask": response_mask,
        "rollout_log_probs": rollout_log_probs,
        "old_log_probs": old_log_probs,
    }.items():
        if tensor.shape != rewards.shape:
            raise ValueError(
                "AMG token GAE tensors must have equal shapes: "
                f"rewards={tuple(rewards.shape)} {name}={tuple(tensor.shape)}"
            )
    if not rewards.is_floating_point() or not values.is_floating_point():
        raise TypeError("AMG token GAE rewards and values must be floating point")
    if not rollout_log_probs.is_floating_point() or not old_log_probs.is_floating_point():
        raise TypeError("AMG token GAE policy log probabilities must be floating point")
    if any(
        tensor.device != values.device
        for tensor in (rewards, response_mask, rollout_log_probs, old_log_probs)
    ):
        raise ValueError("AMG token GAE tensors must share one device")
    if not bool(torch.logical_or(response_mask == 0, response_mask == 1).all().item()):
        raise ValueError("AMG token GAE response_mask must contain only zero or one")

    row_count = rewards.shape[0]
    trajectory_uids = _require_metadata(non_tensor_batch, TRAJECTORY_UID, row_count)
    row_uids = _require_metadata(non_tensor_batch, TRAJECTORY_ROW_UID, row_count)
    row_orders = _require_metadata(non_tensor_batch, TRAJECTORY_ROW_ORDER, row_count)
    terminals = _require_metadata(non_tensor_batch, TRAJECTORY_TERMINAL, row_count)
    done_flags = _require_metadata(non_tensor_batch, ROLLOUT_DONE_FLAG, row_count)
    immediate_rewards = _require_metadata(non_tensor_batch, IMMEDIATE_REWARD, row_count)
    min_global_steps = _require_metadata(
        non_tensor_batch, MIN_GLOBAL_STEPS, row_count
    )
    max_global_steps = _require_metadata(
        non_tensor_batch, MAX_GLOBAL_STEPS, row_count
    )
    outcomes = _require_metadata(non_tensor_batch, OUTCOME, row_count)
    declared_max_rounds = _require_metadata(
        non_tensor_batch, DECLARED_MAX_ROUNDS, row_count
    )
    termination_kinds = _require_metadata(
        non_tensor_batch, TERMINATION_KIND, row_count
    )
    horizon_finalizer_receipts = _require_metadata(
        non_tensor_batch, HORIZON_FINALIZER_RECEIPT, row_count
    )
    is_padding = non_tensor_batch.get(IS_PADDING, np.zeros(row_count, dtype=bool))
    if len(is_padding) != row_count:
        raise ValueError(
            f"AMG token GAE {IS_PADDING!r} must align with {row_count} rows"
        )

    gamma = _as_finite_float(_config_value(config, "gamma", 1.0), field="gamma")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"AMG token GAE gamma must be in [0, 1], got {gamma!r}")
    lambda_mode = str(
        _config_value(config, "amg_policy_lambda_mode", "length_adaptive")
    )
    if lambda_mode != "length_adaptive":
        raise ValueError(
            "AMG token GAE amg_policy_lambda_mode must be 'length_adaptive', "
            f"got {lambda_mode!r}"
        )
    lambda_scale = _as_finite_float(
        _config_value(config, "amg_policy_lambda_scale", 1.5),
        field="policy lambda scale",
    )
    critic_lambda = _as_finite_float(
        _config_value(config, "amg_critic_lambda", 1.0),
        field="critic lambda",
    )
    if not 0.0 <= critic_lambda <= 1.0:
        raise ValueError(
            f"AMG token GAE critic lambda must be in [0, 1], got {critic_lambda!r}"
        )
    tolerance = _as_finite_float(
        _config_value(config, "amg_reward_tolerance", 1e-6),
        field="reward tolerance",
    )
    if tolerance < 0.0:
        raise ValueError("AMG token GAE reward tolerance must be non-negative")
    normalization = str(
        _config_value(config, "amg_advantage_normalization", "none")
    )
    if normalization not in {"none", "upstream_masked_whiten"}:
        raise ValueError(
            "AMG token GAE amg_advantage_normalization must be 'none' or "
            f"'upstream_masked_whiten', got {normalization!r}"
        )

    accumulator_dtype = torch.promote_types(rewards.dtype, values.dtype)
    if accumulator_dtype in (torch.float16, torch.bfloat16):
        accumulator_dtype = torch.float32

    real_policy_mask = response_mask.to(dtype=torch.bool).clone()
    padding_rows = []
    for row in range(row_count):
        padding = _as_bool(is_padding[row], field=IS_PADDING, row=row)
        padding_rows.append(padding)
        if padding:
            real_policy_mask[row] = False

    real_rollout_log_probs = rollout_log_probs[real_policy_mask]
    real_old_log_probs = old_log_probs[real_policy_mask]
    if real_rollout_log_probs.numel() == 0:
        raise ValueError("AMG token GAE requires at least one real policy token")
    if not bool(torch.isfinite(real_rollout_log_probs).all().item()) or not bool(
        torch.isfinite(real_old_log_probs).all().item()
    ):
        raise ValueError("AMG token GAE real policy-token logprobs must be finite")
    if not torch.equal(real_old_log_probs, real_rollout_log_probs):
        mismatch_count = int(
            torch.count_nonzero(real_old_log_probs != real_rollout_log_probs).item()
        )
        raise ValueError(
            "AMG bypass PPO requires old_log_probs to be exactly the rollout behavior "
            f"logprobs on every real policy token; mismatches={mismatch_count}"
        )

    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_row_uids: set[str] = set()
    for physical_row in range(row_count):
        if padding_rows[physical_row]:
            continue

        trajectory_uid = _as_nonempty_str(
            trajectory_uids[physical_row],
            field=TRAJECTORY_UID,
            row=physical_row,
        )
        row_uid = _as_nonempty_str(
            row_uids[physical_row],
            field=TRAJECTORY_ROW_UID,
            row=physical_row,
        )
        if row_uid in seen_row_uids:
            raise ValueError(f"duplicate AMG trajectory row UID {row_uid!r}")
        seen_row_uids.add(row_uid)

        token_indices = torch.nonzero(
            response_mask[physical_row].to(dtype=torch.bool), as_tuple=False
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
        row_values = values[physical_row, token_indices]
        row_rewards = rewards[physical_row, token_indices]
        if not bool(torch.isfinite(row_values).all().item()):
            raise ValueError(f"AMG action row {physical_row} contains non-finite values")
        if not bool(torch.isfinite(row_rewards).all().item()):
            raise ValueError(f"AMG action row {physical_row} contains non-finite rewards")

        immediate_reward = _as_finite_float(
            immediate_rewards[physical_row],
            field=IMMEDIATE_REWARD,
            row=physical_row,
        )
        min_global_step = _as_nonnegative_int(
            min_global_steps[physical_row],
            field=MIN_GLOBAL_STEPS,
            row=physical_row,
        )
        max_global_step = _as_nonnegative_int(
            max_global_steps[physical_row],
            field=MAX_GLOBAL_STEPS,
            row=physical_row,
        )
        if min_global_step > max_global_step:
            raise ValueError(
                "AMG rollout policy-version span must satisfy "
                f"{MIN_GLOBAL_STEPS} <= {MAX_GLOBAL_STEPS}: row={physical_row} "
                f"minimum={min_global_step} maximum={max_global_step}"
            )
        expected_reward = torch.as_tensor(
            immediate_reward,
            device=values.device,
            dtype=accumulator_dtype,
        )
        packed_reward = row_rewards.to(accumulator_dtype).sum()
        if not torch.isclose(packed_reward, expected_reward, rtol=0.0, atol=tolerance):
            raise ValueError(
                "AMG packed token reward differs from the immediate action reward: "
                f"row={physical_row} packed={float(packed_reward.item())} "
                f"immediate={immediate_reward}"
            )
        if token_indices.numel() > 1:
            early_rewards = row_rewards[:-1].to(accumulator_dtype)
            if bool((early_rewards.abs() > tolerance).any().item()):
                raise ValueError(
                    "AMG immediate reward must appear only on the final valid policy token: "
                    f"row={physical_row}"
                )
        final_reward = row_rewards[-1].to(accumulator_dtype)
        if not torch.isclose(final_reward, expected_reward, rtol=0.0, atol=tolerance):
            raise ValueError(
                "AMG immediate reward must appear on the final valid policy token: "
                f"row={physical_row} final={float(final_reward.item())} "
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
                    terminals[physical_row],
                    field=TRAJECTORY_TERMINAL,
                    row=physical_row,
                ),
                "done": _as_bool(
                    done_flags[physical_row],
                    field=ROLLOUT_DONE_FLAG,
                    row=physical_row,
                ),
                "min_global_steps": min_global_step,
                "max_global_steps": max_global_step,
                "outcome": _as_nonempty_str(
                    outcomes[physical_row], field=OUTCOME, row=physical_row
                ),
                "declared_max_rounds": _as_positive_int(
                    declared_max_rounds[physical_row],
                    field=DECLARED_MAX_ROUNDS,
                    row=physical_row,
                ),
                "termination_kind": _as_nonempty_str(
                    termination_kinds[physical_row],
                    field=TERMINATION_KIND,
                    row=physical_row,
                ),
                "horizon_finalizer_receipt": _as_nonempty_str(
                    horizon_finalizer_receipts[physical_row],
                    field=HORIZON_FINALIZER_RECEIPT,
                    row=physical_row,
                ),
                "token_indices": token_indices,
            }
        )

    advantages = torch.zeros(values.shape, device=values.device, dtype=accumulator_dtype)
    returns = torch.zeros_like(advantages)
    with torch.no_grad():
        for trajectory_uid, rows in trajectories.items():
            rows.sort(key=lambda row: row["row_order"])
            actual_orders = [row["row_order"] for row in rows]
            expected_orders = list(range(len(rows)))
            if actual_orders != expected_orders:
                raise ValueError(
                    "AMG trajectory row order is incomplete or duplicated: "
                    f"trajectory={trajectory_uid!r} expected={expected_orders} "
                    f"actual={actual_orders}"
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

            declared_horizons = {row["declared_max_rounds"] for row in rows}
            if len(declared_horizons) != 1:
                raise ValueError(
                    "AMG trajectory declared_max_rounds changed across action rows: "
                    f"trajectory={trajectory_uid!r} values={sorted(declared_horizons)}"
                )
            for row in rows[:-1]:
                if row["termination_kind"] != "in_progress":
                    raise ValueError(
                        "nonterminal AMG row must have termination_kind='in_progress': "
                        f"trajectory={trajectory_uid!r} row={row['row_order']} "
                        f"actual={row['termination_kind']!r}"
                    )
                if row["horizon_finalizer_receipt"] != "not_applicable:nonterminal":
                    raise ValueError(
                        "nonterminal AMG row has an invalid horizon finalizer receipt: "
                        f"trajectory={trajectory_uid!r} row={row['row_order']} "
                        f"actual={row['horizon_finalizer_receipt']!r}"
                    )

            final_row = rows[-1]
            if final_row["done"]:
                valid_done_receipts = {
                    "environment_done": "not_invoked:environment_done",
                    "horizon_finalized": "terminal_transition_applied",
                }
                expected_receipt = valid_done_receipts.get(
                    final_row["termination_kind"]
                )
                if expected_receipt != final_row["horizon_finalizer_receipt"]:
                    raise ValueError(
                        "done AMG terminal row has inconsistent termination/finalizer evidence: "
                        f"trajectory={trajectory_uid!r} "
                        f"termination_kind={final_row['termination_kind']!r} "
                        f"receipt={final_row['horizon_finalizer_receipt']!r}"
                    )
                if final_row["termination_kind"] == "horizon_finalized":
                    declared_horizon = final_row["declared_max_rounds"]
                    if (
                        len(rows) != declared_horizon
                        or final_row["row_order"] != declared_horizon - 1
                    ):
                        raise ValueError(
                            "horizon-finalized AMG terminal row lacks a complete "
                            "declared horizon: "
                            f"trajectory={trajectory_uid!r} rows={len(rows)} "
                            f"declared={declared_horizon} "
                            f"final_order={final_row['row_order']}"
                        )
            else:
                declared_horizon = final_row["declared_max_rounds"]
                if (
                    final_row["termination_kind"] != "max_rounds"
                    or final_row["outcome"] != "max_rounds"
                    or final_row["horizon_finalizer_receipt"]
                    not in _MAX_ROUNDS_FINALIZER_RECEIPTS
                    or len(rows) != declared_horizon
                    or final_row["row_order"] != declared_horizon - 1
                ):
                    raise ValueError(
                        "non-done AMG terminal row lacks exact max_rounds horizon attestation: "
                        f"trajectory={trajectory_uid!r} rows={len(rows)} "
                        f"declared={declared_horizon} order={final_row['row_order']} "
                        f"outcome={final_row['outcome']!r} "
                        f"termination_kind={final_row['termination_kind']!r} "
                        f"receipt={final_row['horizon_finalizer_receipt']!r}"
                    )

            token_locations: list[tuple[int, int]] = []
            for row in rows:
                token_locations.extend(
                    (row["physical_row"], int(token_index.item()))
                    for token_index in row["token_indices"]
                )
            policy_lambda = length_adaptive_policy_lambda(
                len(token_locations), scale=lambda_scale
            )
            next_value = torch.zeros((), device=values.device, dtype=accumulator_dtype)
            next_policy_advantage = torch.zeros_like(next_value)
            next_critic_advantage = torch.zeros_like(next_value)
            for flat_index in range(len(token_locations) - 1, -1, -1):
                physical_row, token_index = token_locations[flat_index]
                value = values[physical_row, token_index].to(accumulator_dtype)
                reward = rewards[physical_row, token_index].to(accumulator_dtype)
                continuation = 1.0 if flat_index + 1 < len(token_locations) else 0.0
                delta = reward + gamma * continuation * next_value - value
                policy_advantage = (
                    delta
                    + gamma
                    * policy_lambda
                    * continuation
                    * next_policy_advantage
                )
                critic_advantage = (
                    delta
                    + gamma
                    * critic_lambda
                    * continuation
                    * next_critic_advantage
                )
                advantages[physical_row, token_index] = policy_advantage
                returns[physical_row, token_index] = critic_advantage + value
                next_value = value
                next_policy_advantage = policy_advantage
                next_critic_advantage = critic_advantage

    if normalization == "upstream_masked_whiten":
        advantages = verl_F.masked_whiten(advantages, real_policy_mask)
        advantages = advantages * real_policy_mask.to(dtype=advantages.dtype)

    return advantages, returns
