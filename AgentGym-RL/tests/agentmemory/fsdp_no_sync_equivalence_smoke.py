#!/usr/bin/env python3
"""Run with torchrun to compare FSDP accumulation synchronization paths."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from verl.workers.fsdp_gradient_accumulation import fsdp_gradient_sync_context


def _build_model(device: torch.device) -> FSDP:
    torch.manual_seed(20260730)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 8),
    ).to(device)
    return FSDP(
        model,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )


def _run_step(
    model: FSDP,
    inputs: tuple[torch.Tensor, ...],
    targets: tuple[torch.Tensor, ...],
    *,
    defer_sync: bool,
) -> list[torch.Tensor]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    for index, (micro_inputs, micro_targets) in enumerate(
        zip(inputs, targets, strict=True)
    ):
        with fsdp_gradient_sync_context(
            model,
            enabled=defer_sync,
            is_last_micro_batch=index == len(inputs) - 1,
        ):
            outputs = model(micro_inputs)
            torch.nn.functional.mse_loss(
                outputs,
                micro_targets,
                reduction="sum",
            ).backward()
    optimizer.step()
    with FSDP.summon_full_params(model):
        return [parameter.detach().cpu().clone() for parameter in model.parameters()]


def main() -> None:
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    generator = torch.Generator(device=device).manual_seed(9000 + rank)
    inputs = tuple(
        torch.randn(2, 32, generator=generator, device=device) for _ in range(4)
    )
    targets = tuple(
        torch.randn(2, 8, generator=generator, device=device) for _ in range(4)
    )
    baseline = _run_step(
        _build_model(device),
        inputs,
        targets,
        defer_sync=False,
    )
    optimized = _run_step(
        _build_model(device),
        inputs,
        targets,
        defer_sync=True,
    )

    max_abs_delta = max(
        float((left - right).abs().max().item())
        for left, right in zip(baseline, optimized, strict=True)
    )
    if max_abs_delta > 1e-6:
        raise AssertionError(f"parameter max abs delta {max_abs_delta} > 1e-6")
    dist.barrier()
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "world_size": dist.get_world_size(),
                    "micro_batches": len(inputs),
                    "parameter_max_abs_delta": max_abs_delta,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
