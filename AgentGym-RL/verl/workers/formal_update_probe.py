"""Bounded parameter-change evidence for formal PPO update readback."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist


DEFAULT_PARAMETER_PROBE_ELEMENTS = 65_536


def _trainable_parameter_layout(module: torch.nn.Module) -> tuple[tuple[int, ...], list[torch.Tensor]]:
    parameters = [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.numel() > 0
    ]
    layout = tuple(int(parameter.numel()) for parameter in parameters)
    if not layout:
        raise RuntimeError("Formal update readback found no trainable parameters to probe.")
    return layout, parameters


def _evenly_spaced_indices(total_elements: int, sample_count: int) -> torch.Tensor:
    if total_elements <= 0 or sample_count <= 0 or sample_count > total_elements:
        raise ValueError(
            "Invalid formal parameter probe dimensions: "
            f"total_elements={total_elements} sample_count={sample_count}."
        )
    if sample_count == 1:
        return torch.zeros(1, dtype=torch.long)
    return (
        torch.arange(sample_count, dtype=torch.long) * (total_elements - 1)
    ) // (sample_count - 1)


def _read_parameter_values(
    parameters: list[torch.Tensor],
    layout: tuple[int, ...],
    sample_count: int,
) -> torch.Tensor:
    total_elements = sum(layout)
    indices = _evenly_spaced_indices(total_elements, sample_count)
    chunks = []
    offset = 0
    for parameter, parameter_elements in zip(parameters, layout):
        right_offset = offset + parameter_elements
        left = int(torch.searchsorted(indices, offset, right=False).item())
        right = int(torch.searchsorted(indices, right_offset, right=False).item())
        if right > left:
            local_indices = (indices[left:right] - offset).to(parameter.device)
            values = parameter.detach().reshape(-1).index_select(0, local_indices)
            chunks.append(values.to(device="cpu", dtype=torch.float32))
        offset = right_offset
    if not chunks:
        raise RuntimeError("Formal update readback parameter probe selected no values.")
    values = torch.cat(chunks)
    if values.numel() != sample_count:
        raise RuntimeError(
            "Formal update readback parameter probe count mismatch: "
            f"expected={sample_count} actual={values.numel()}."
        )
    return values


def capture_parameter_probe(
    module: torch.nn.Module,
    *,
    max_elements: int = DEFAULT_PARAMETER_PROBE_ELEMENTS,
) -> dict[str, Any]:
    """Capture a deterministic bounded spread over local trainable parameters."""

    if max_elements <= 0:
        raise ValueError(f"max_elements must be positive, got {max_elements}.")
    layout, parameters = _trainable_parameter_layout(module)
    total_elements = sum(layout)
    sample_count = min(int(max_elements), total_elements)
    return {
        "layout": layout,
        "total_parameter_count": total_elements,
        "sample_count": sample_count,
        "values": _read_parameter_values(parameters, layout, sample_count),
    }


def _reduction_device(process_group) -> torch.device:
    if (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_backend(process_group) == "nccl"
    ):
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def measure_parameter_probe_delta(
    module: torch.nn.Module,
    before: dict[str, Any],
    *,
    process_group=None,
    label: str,
) -> dict[str, Any]:
    """Aggregate sampled shard deltas and fail unless a real update is visible."""

    layout, parameters = _trainable_parameter_layout(module)
    expected_layout = tuple(before.get("layout", ()))
    if layout != expected_layout:
        raise RuntimeError(
            f"Formal {label} parameter probe layout changed across optimizer update."
        )
    sample_count = int(before.get("sample_count", 0))
    before_values = before.get("values")
    if not isinstance(before_values, torch.Tensor) or before_values.numel() != sample_count:
        raise RuntimeError(f"Formal {label} parameter probe pre-update values are missing.")

    after_values = _read_parameter_values(parameters, layout, sample_count)
    delta = after_values.to(torch.float64) - before_values.to(torch.float64)
    finite = bool(
        torch.isfinite(before_values).all().item()
        and torch.isfinite(after_values).all().item()
        and torch.isfinite(delta).all().item()
    )
    safe_delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))

    device = _reduction_device(process_group)
    sums = torch.tensor(
        [
            safe_delta.square().sum().item(),
            safe_delta.abs().sum().item(),
            safe_delta.ne(0).sum().item(),
            safe_delta.numel(),
            int(before["total_parameter_count"]),
        ],
        dtype=torch.float64,
        device=device,
    )
    maximum = torch.tensor(
        [safe_delta.abs().max().item()], dtype=torch.float64, device=device
    )
    finite_flag = torch.tensor([int(finite)], dtype=torch.int64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=process_group)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=process_group)
        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN, group=process_group)

    squared_l2 = float(sums[0].item())
    summary = {
        "parameter_delta_l2": math.sqrt(max(0.0, squared_l2)),
        "parameter_probe_max_abs_delta": float(maximum.item()),
        "parameter_probe_l1_delta": float(sums[1].item()),
        "parameter_probe_changed_count": int(sums[2].item()),
        "parameter_probe_element_count": int(sums[3].item()),
        "parameter_probe_total_parameter_count": int(sums[4].item()),
        "parameter_probe_max_elements_per_rank": int(sample_count),
        "parameter_probe_finite": bool(finite_flag.item()),
        "parameter_probe_sampling": "evenly_spaced_local_trainable_shards",
    }
    numeric_fields = (
        "parameter_delta_l2",
        "parameter_probe_max_abs_delta",
        "parameter_probe_l1_delta",
    )
    if not summary["parameter_probe_finite"] or not all(
        math.isfinite(summary[field]) for field in numeric_fields
    ):
        raise RuntimeError(f"Formal {label} readback found a non-finite parameter probe.")
    if (
        summary["parameter_delta_l2"] <= 0.0
        or summary["parameter_probe_max_abs_delta"] <= 0.0
        or summary["parameter_probe_l1_delta"] <= 0.0
        or summary["parameter_probe_changed_count"] <= 0
    ):
        raise RuntimeError(f"Formal {label} readback found zero parameter delta.")
    return summary
