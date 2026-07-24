"""Distributed token normalization and batch-contract helpers for PPO."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist


LEGACY_ASYMMETRIC_BATCH_MODE = "legacy_asymmetric_config_compensation_v1"
PPO_BATCH_CONTRACT_META_KEY = "ppo_batch_contract"

_PER_GPU_MICRO_BATCH_FIELDS = (
    "actor",
    "critic",
    "critic_forward",
    "reference_logprob",
    "rollout_logprob",
)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}."
        ) from exc
    if parsed <= 0 or parsed != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return parsed


def build_legacy_asymmetric_batch_contract(
    *,
    actor_mini_batch_size: int,
    critic_mini_batch_size: int,
    rollout_n: int,
    world_size: int,
    actor_sequence_parallel_size: int,
    critic_sequence_parallel_size: int,
    per_gpu_micro_batches: Mapping[str, int],
    legacy_micro_batches: Mapping[str, Optional[int]],
    actor_ppo_epochs: int,
    critic_ppo_epochs: int,
    expected_per_gpu_micro_batch_size: Optional[int] = None,
    expected_per_gpu_micro_batches: Optional[Mapping[str, int]] = None,
) -> dict:
    """Validate and describe the legacy flattened-row compensation mode.

    The legacy actor worker multiplies its configured mini-batch by
    ``rollout.n`` before data-parallel normalization. The critic does not.
    Until both workers use one raw unit, callers must compensate with a critic
    raw mini-batch that produces the same local flattened-row mini-batch.
    """

    actor_raw = _positive_int(actor_mini_batch_size, "actor_mini_batch_size")
    critic_raw = _positive_int(critic_mini_batch_size, "critic_mini_batch_size")
    rollout_n = _positive_int(rollout_n, "rollout_n")
    world_size = _positive_int(world_size, "world_size")
    actor_sp = _positive_int(
        actor_sequence_parallel_size, "actor_sequence_parallel_size"
    )
    critic_sp = _positive_int(
        critic_sequence_parallel_size, "critic_sequence_parallel_size"
    )
    actor_epochs = _positive_int(actor_ppo_epochs, "actor_ppo_epochs")
    critic_epochs = _positive_int(critic_ppo_epochs, "critic_ppo_epochs")

    if world_size % actor_sp != 0 or world_size % critic_sp != 0:
        raise ValueError(
            "world_size must be divisible by actor and critic sequence-parallel "
            f"sizes: world_size={world_size} actor_sp={actor_sp} critic_sp={critic_sp}."
        )
    actor_dp = world_size // actor_sp
    critic_dp = world_size // critic_sp
    actor_numerator = actor_raw * rollout_n
    if actor_numerator % actor_dp != 0:
        raise ValueError(
            "actor raw mini-batch times rollout_n must be divisible by actor DP: "
            f"{actor_raw} * {rollout_n} / {actor_dp}."
        )
    if critic_raw % critic_dp != 0:
        raise ValueError(
            "critic raw mini-batch must be divisible by critic DP: "
            f"{critic_raw} / {critic_dp}."
        )

    actor_local = actor_numerator // actor_dp
    critic_local = critic_raw // critic_dp
    if actor_local != critic_local:
        raise ValueError(
            "actor and critic normalized PPO mini-batches use different "
            "flattened-row units under legacy compensation: "
            f"actor={actor_local} critic={critic_local} "
            f"(raw actor={actor_raw}, raw critic={critic_raw}, "
            f"rollout_n={rollout_n}, actor_dp={actor_dp}, critic_dp={critic_dp})."
        )

    missing = set(_PER_GPU_MICRO_BATCH_FIELDS) - set(per_gpu_micro_batches)
    extra = set(per_gpu_micro_batches) - set(_PER_GPU_MICRO_BATCH_FIELDS)
    if missing or extra:
        raise ValueError(
            "per_gpu_micro_batches must contain exactly "
            f"{_PER_GPU_MICRO_BATCH_FIELDS}; missing={sorted(missing)} "
            f"extra={sorted(extra)}."
        )
    parsed_micro = {
        name: _positive_int(per_gpu_micro_batches[name], f"{name}_per_gpu")
        for name in _PER_GPU_MICRO_BATCH_FIELDS
    }
    if (
        expected_per_gpu_micro_batch_size is not None
        and expected_per_gpu_micro_batches is not None
    ):
        raise ValueError(
            "scalar and role-specific expected per-GPU micro-batch declarations "
            "are mutually exclusive."
        )

    expected_micro_by_role = None
    if expected_per_gpu_micro_batches is not None:
        expected_missing = set(_PER_GPU_MICRO_BATCH_FIELDS) - set(
            expected_per_gpu_micro_batches
        )
        expected_extra = set(expected_per_gpu_micro_batches) - set(
            _PER_GPU_MICRO_BATCH_FIELDS
        )
        if expected_missing or expected_extra:
            raise ValueError(
                "expected_per_gpu_micro_batches must contain exactly "
                f"{_PER_GPU_MICRO_BATCH_FIELDS}; missing={sorted(expected_missing)} "
                f"extra={sorted(expected_extra)}."
            )
        expected_micro_by_role = {
            name: _positive_int(
                expected_per_gpu_micro_batches[name],
                f"expected_{name}_per_gpu",
            )
            for name in _PER_GPU_MICRO_BATCH_FIELDS
        }
        mismatched = {
            name: {
                "configured": parsed_micro[name],
                "declared": expected_micro_by_role[name],
            }
            for name in _PER_GPU_MICRO_BATCH_FIELDS
            if parsed_micro[name] != expected_micro_by_role[name]
        }
        if mismatched:
            raise ValueError(
                "per-GPU micro-batch readback does not match the role-specific "
                f"declaration: {mismatched}."
            )
        expected_micro = None
    elif expected_per_gpu_micro_batch_size is not None:
        expected_micro = _positive_int(
            expected_per_gpu_micro_batch_size,
            "expected_per_gpu_micro_batch_size",
        )
        mismatched = {
            name: value
            for name, value in parsed_micro.items()
            if value != expected_micro
        }
        if mismatched:
            raise ValueError(
                "per-GPU micro-batch readback does not match the declared "
                f"v32 value {expected_micro}: {mismatched}."
            )
    else:
        expected_micro = None

    non_null_legacy = {
        name: value
        for name, value in legacy_micro_batches.items()
        if value is not None
    }
    if non_null_legacy:
        raise ValueError(
            "deprecated global micro-batch fields must be null when per-GPU "
            f"fields are selected: {non_null_legacy}."
        )
    if actor_local % parsed_micro["actor"] != 0:
        raise ValueError(
            f"actor local mini-batch {actor_local} is not divisible by "
            f"actor per-GPU micro-batch {parsed_micro['actor']}."
        )
    if critic_local % parsed_micro["critic"] != 0:
        raise ValueError(
            f"critic local mini-batch {critic_local} is not divisible by "
            f"critic per-GPU micro-batch {parsed_micro['critic']}."
        )

    return {
        "mode": LEGACY_ASYMMETRIC_BATCH_MODE,
        "world_size": world_size,
        "actor_data_parallel_size": actor_dp,
        "critic_data_parallel_size": critic_dp,
        "rollout_n": rollout_n,
        "actor_raw_mini_batch_rows": actor_raw,
        "critic_raw_mini_batch_rows": critic_raw,
        "actor_local_mini_batch_rows": actor_local,
        "critic_local_mini_batch_rows": critic_local,
        "actor_ppo_epochs": actor_epochs,
        "critic_ppo_epochs": critic_epochs,
        "expected_per_gpu_micro_batch_size": expected_micro,
        "expected_per_gpu_micro_batches": expected_micro_by_role,
        "per_gpu_micro_batches": parsed_micro,
    }


def optimizer_step_readback(
    contract: Mapping[str, object], global_rows_after_dp_padding: int
) -> dict:
    """Return the expected actor/critic optimizer steps for one PPO update."""

    world_size = _positive_int(contract["world_size"], "contract.world_size")
    global_rows = _positive_int(
        global_rows_after_dp_padding, "global_rows_after_dp_padding"
    )
    if global_rows % world_size != 0:
        raise ValueError(
            "global_rows_after_dp_padding must be divisible by world_size: "
            f"{global_rows} / {world_size}."
        )
    local_rows = global_rows // world_size
    actor_local = _positive_int(
        contract["actor_local_mini_batch_rows"],
        "contract.actor_local_mini_batch_rows",
    )
    critic_local = _positive_int(
        contract["critic_local_mini_batch_rows"],
        "contract.critic_local_mini_batch_rows",
    )
    actor_epochs = _positive_int(
        contract["actor_ppo_epochs"], "contract.actor_ppo_epochs"
    )
    critic_epochs = _positive_int(
        contract["critic_ppo_epochs"], "contract.critic_ppo_epochs"
    )
    actor_minibatches = math.ceil(local_rows / actor_local)
    critic_minibatches = math.ceil(local_rows / critic_local)
    if actor_minibatches != critic_minibatches:
        raise ValueError(
            "actor and critic mini-batches per epoch differ: "
            f"actor={actor_minibatches} critic={critic_minibatches}."
        )
    return {
        "local_rows": local_rows,
        "minibatches_per_epoch": actor_minibatches,
        "actor_optimizer_steps": actor_minibatches * actor_epochs,
        "critic_optimizer_steps": critic_minibatches * critic_epochs,
    }


def validate_worker_batch_readback(
    contract: Mapping[str, object],
    *,
    role: str,
    normalized_mini_batch_rows: int,
    per_gpu_micro_batch_rows: int,
    forward_per_gpu_micro_batch_rows: Optional[int] = None,
) -> dict:
    """Fail closed if a worker's post-normalization values drift from the driver."""

    if role not in {"actor", "critic"}:
        raise ValueError(f"role must be actor or critic, got {role!r}.")
    expected_mini = _positive_int(
        contract[f"{role}_local_mini_batch_rows"],
        f"contract.{role}_local_mini_batch_rows",
    )
    actual_mini = _positive_int(
        normalized_mini_batch_rows, f"{role}.normalized_mini_batch_rows"
    )
    expected_micro = _positive_int(
        contract["per_gpu_micro_batches"][role],
        f"contract.per_gpu_micro_batches.{role}",
    )
    actual_micro = _positive_int(
        per_gpu_micro_batch_rows, f"{role}.per_gpu_micro_batch_rows"
    )
    if actual_mini != expected_mini or actual_micro != expected_micro:
        raise ValueError(
            f"{role} worker batch readback mismatch: "
            f"mini expected={expected_mini} actual={actual_mini}; "
            f"micro expected={expected_micro} actual={actual_micro}."
        )

    readback = {
        "normalized_mini_batch_rows": actual_mini,
        "per_gpu_micro_batch_rows": actual_micro,
    }
    if role == "critic":
        expected_forward = _positive_int(
            contract["per_gpu_micro_batches"]["critic_forward"],
            "contract.per_gpu_micro_batches.critic_forward",
        )
        actual_forward = _positive_int(
            forward_per_gpu_micro_batch_rows,
            "critic.forward_per_gpu_micro_batch_rows",
        )
        if actual_forward != expected_forward:
            raise ValueError(
                "critic forward micro-batch readback mismatch: "
                f"expected={expected_forward} actual={actual_forward}."
            )
        readback["forward_per_gpu_micro_batch_rows"] = actual_forward
    return readback


def select_response_values(
    full_sequence_values: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Align full-sequence critic outputs with response-only PPO tensors."""

    if full_sequence_values.ndim != 2 or response_mask.ndim != 2:
        raise ValueError(
            "critic values and response_mask must both be rank-2 tensors: "
            f"values_shape={tuple(full_sequence_values.shape)} "
            f"response_shape={tuple(response_mask.shape)}"
        )
    if full_sequence_values.shape[0] != response_mask.shape[0]:
        raise ValueError(
            "critic values and response_mask must have the same batch size: "
            f"values_shape={tuple(full_sequence_values.shape)} "
            f"response_shape={tuple(response_mask.shape)}"
        )
    response_length = response_mask.shape[-1]
    if full_sequence_values.shape[-1] < response_length:
        raise ValueError(
            "critic sequence is shorter than response_mask: "
            f"values_shape={tuple(full_sequence_values.shape)} "
            f"response_shape={tuple(response_mask.shape)}"
        )
    response_values = full_sequence_values[:, -response_length:]
    return response_values * response_mask.to(
        device=response_values.device,
        dtype=response_values.dtype,
    )


def mask_padding_rows(
    response_mask: torch.Tensor,
    valid_sample_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Remove data-parallel transport padding rows from a response mask."""

    if valid_sample_mask is None:
        return response_mask
    if (
        valid_sample_mask.ndim != 1
        or valid_sample_mask.shape[0] != response_mask.shape[0]
    ):
        raise ValueError(
            "valid_sample_mask must be one-dimensional and match the batch: "
            f"mask_shape={tuple(valid_sample_mask.shape)} "
            f"response_shape={tuple(response_mask.shape)}"
        )
    return response_mask * valid_sample_mask.to(
        device=response_mask.device,
        dtype=response_mask.dtype,
    ).unsqueeze(-1)


def valid_response_token_count(response_mask: torch.Tensor) -> torch.Tensor:
    """Return a detached scalar token count on the mask's device."""

    return response_mask.detach().sum(dtype=torch.float64)


def distributed_sum(value: torch.Tensor, group=None) -> torch.Tensor:
    """All-reduce a detached tensor, with a single-process test fallback."""

    total = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    return total


def distributed_world_size(group=None) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=group)
    return 1


def scale_token_mean_loss(
    local_mean_loss: torch.Tensor,
    local_token_count: torch.Tensor,
    global_token_count: torch.Tensor,
    *,
    group=None,
) -> torch.Tensor:
    """Scale a local token mean so rank averaging yields a global token mean."""

    if not torch.isfinite(local_token_count) or local_token_count.item() < 0:
        raise ValueError(f"local token count is invalid: {local_token_count!r}.")
    if not torch.isfinite(global_token_count) or global_token_count.item() <= 0:
        raise ValueError(f"global token count is invalid: {global_token_count!r}.")
    local_count = local_token_count.to(
        device=local_mean_loss.device, dtype=local_mean_loss.dtype
    )
    global_count = global_token_count.to(
        device=local_mean_loss.device, dtype=local_mean_loss.dtype
    )
    safe_local_mean = torch.where(
        local_count > 0,
        local_mean_loss,
        torch.zeros_like(local_mean_loss),
    )
    scale = distributed_world_size(group) * local_count / global_count
    return safe_local_mean * scale


class TokenWeightedMetricAccumulator:
    """Accumulate local token means and reduce them into global metrics."""

    def __init__(self) -> None:
        self._weighted_sums: Dict[str, torch.Tensor] = {}
        self._token_count: Optional[torch.Tensor] = None

    def add(
        self,
        values: Mapping[str, Union[torch.Tensor, float]],
        token_count: torch.Tensor,
    ) -> None:
        count = token_count.detach().to(dtype=torch.float64)
        if self._token_count is None:
            self._token_count = torch.zeros_like(count)
        self._token_count = self._token_count + count
        for key, value in values.items():
            metric = torch.as_tensor(value, device=count.device).detach().to(
                dtype=torch.float64
            )
            if count.item() > 0 and not torch.isfinite(metric):
                raise ValueError(f"non-finite PPO metric {key}: {metric!r}.")
            weighted = torch.where(
                count > 0, metric * count, torch.zeros_like(metric)
            )
            if key not in self._weighted_sums:
                self._weighted_sums[key] = torch.zeros_like(weighted)
            self._weighted_sums[key] = self._weighted_sums[key] + weighted

    def reduce(self, group=None) -> Dict[str, list]:
        if self._token_count is None:
            return {}
        keys = sorted(self._weighted_sums)
        packed = torch.stack(
            [self._weighted_sums[key] for key in keys] + [self._token_count]
        )
        packed = distributed_sum(packed, group=group)
        count = packed[-1]
        if not torch.isfinite(count) or count.item() <= 0:
            raise ValueError(f"global PPO metric token count is invalid: {count!r}.")
        return {
            key: [(packed[index] / count).item()]
            for index, key in enumerate(keys)
        }


def reduce_worker_metrics(
    metrics: Mapping[str, object], group=None
) -> Dict[str, list]:
    """Return rank-global means for worker metrics before RPC collection."""

    if not metrics:
        return {}
    keys = sorted(metrics)
    device = torch.device("cpu")
    if (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_backend(group) == "nccl"
    ):
        device = torch.device("cuda", torch.cuda.current_device())

    local_sums = []
    local_counts = []
    for key in keys:
        value = metrics[key]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            scalars = list(value)
        else:
            scalars = [value]
        local_sums.append(sum(float(item) for item in scalars))
        local_counts.append(len(scalars))

    packed = torch.tensor(
        [local_sums, local_counts], device=device, dtype=torch.float64
    )
    packed = distributed_sum(packed, group=group)
    return {
        key: [(packed[0, index] / packed[1, index].clamp_min(1.0)).item()]
        for index, key in enumerate(keys)
    }
