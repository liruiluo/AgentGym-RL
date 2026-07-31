"""Helpers for deferring FSDP gradient synchronization across microbatches."""

from contextlib import nullcontext

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def should_defer_fsdp_gradient_sync(
    module,
    *,
    enabled: bool,
    is_last_micro_batch: bool,
) -> bool:
    """Return whether this backward should accumulate without synchronization."""

    return bool(enabled and not is_last_micro_batch and isinstance(module, FSDP))


def fsdp_gradient_sync_context(
    module,
    *,
    enabled: bool,
    is_last_micro_batch: bool,
):
    """Synchronize only the final backward in an FSDP accumulation window."""

    if should_defer_fsdp_gradient_sync(
        module,
        enabled=enabled,
        is_last_micro_batch=is_last_micro_batch,
    ):
        return module.no_sync()
    return nullcontext()
