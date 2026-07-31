"""Index planning for response-only causal-LM projections."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.distributed as dist

from verl.utils.ulysses import get_ulysses_sequence_parallel_group


class ResponseProjectionPlan(NamedTuple):
    packed_predecessor_positions: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    output_response_mask: torch.Tensor
    padding_only: bool
    packed_token_count: int


class SequenceParallelResponseProjectionPlan(NamedTuple):
    local_predecessor_positions: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    output_response_mask: torch.Tensor
    padding_only: bool
    local_selected_response_tokens: int
    global_selected_response_tokens: int
    packed_shard_start: int
    packed_shard_end: int


def build_response_projection_plan(
    *,
    unpadded_indices: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    valid_sample_mask: torch.Tensor | None = None,
) -> ResponseProjectionPlan:
    """Map valid response labels to their predecessor states in a packed sequence.

    Causal-LM logits at position ``t - 1`` predict the token at position ``t``.
    ``unpadded_indices`` is the flattened padded-position ledger returned by
    FlashAttention's ``unpad_input``.
    """

    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must both be rank-2 tensors.")
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            "input_ids and attention_mask must have identical shapes: "
            f"input_ids={tuple(input_ids.shape)} attention_mask={tuple(attention_mask.shape)}."
        )
    if responses.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("responses and response_mask must both be rank-2 tensors.")
    if responses.shape != response_mask.shape:
        raise ValueError(
            "responses and response_mask must have identical shapes: "
            f"responses={tuple(responses.shape)} response_mask={tuple(response_mask.shape)}."
        )
    if responses.shape[0] != input_ids.shape[0]:
        raise ValueError("packed inputs and responses must have the same batch size.")
    if unpadded_indices.ndim != 1:
        raise ValueError("unpadded_indices must be a rank-1 flattened-position ledger.")
    if unpadded_indices.numel() == 0:
        raise ValueError("response-only projection received an empty packed sequence.")
    if unpadded_indices.numel() > 1 and not torch.all(
        unpadded_indices[1:] > unpadded_indices[:-1]
    ):
        raise ValueError("unpadded_indices must be strictly increasing.")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must be binary.")

    batch_size, sequence_length = input_ids.shape
    response_length = responses.shape[1]
    if response_length <= 0 or sequence_length <= response_length:
        raise ValueError(
            "response-only projection requires a non-empty prompt and response: "
            f"sequence_length={sequence_length} response_length={response_length}."
        )

    raw_selected_mask = response_mask.to(dtype=torch.bool)
    if valid_sample_mask is None:
        output_response_mask = raw_selected_mask
        padding_only = False
    else:
        if (
            valid_sample_mask.ndim != 1
            or valid_sample_mask.shape[0] != input_ids.shape[0]
        ):
            raise ValueError(
                "valid_sample_mask must be one-dimensional and match the batch: "
                f"mask_shape={tuple(valid_sample_mask.shape)} "
                f"batch_size={input_ids.shape[0]}."
            )
        if not torch.all((valid_sample_mask == 0) | (valid_sample_mask == 1)):
            raise ValueError("valid_sample_mask must be binary.")
        valid_rows = valid_sample_mask.to(
            device=response_mask.device, dtype=torch.bool
        )
        output_response_mask = raw_selected_mask & valid_rows.unsqueeze(-1)
        has_valid_rows = bool(torch.any(valid_rows).item())
        has_output_tokens = bool(torch.any(output_response_mask).item())
        padding_only = not has_valid_rows and not has_output_tokens

    if padding_only:
        raw_selected = raw_selected_mask.nonzero(as_tuple=False)
        if raw_selected.numel() == 0:
            # Actor updates clear transport-padding response masks before the
            # distributed forward. Keep one packed hidden state so every FSDP
            # rank still executes the same LM-head forward/backward collectives.
            dummy_flat_position = unpadded_indices[0].to(dtype=torch.long)
            dummy_label = input_ids.reshape(-1)[dummy_flat_position].reshape(1)
            return ResponseProjectionPlan(
                packed_predecessor_positions=torch.zeros(
                    1, dtype=torch.long, device=unpadded_indices.device
                ),
                labels=dummy_label.to(dtype=torch.long),
                response_mask=torch.zeros_like(raw_selected_mask),
                output_response_mask=output_response_mask,
                padding_only=True,
                packed_token_count=int(unpadded_indices.numel()),
            )
        selected_mask = torch.zeros_like(raw_selected_mask)
        first_row, first_column = raw_selected[0]
        selected_mask[first_row, first_column] = True
    else:
        selected_mask = output_response_mask

    selected = selected_mask.nonzero(as_tuple=False)
    if selected.numel() == 0:
        raise ValueError("response-only projection found no valid response tokens.")

    rows = selected[:, 0]
    response_columns = selected[:, 1]
    target_columns = sequence_length - response_length + response_columns
    predecessor_columns = target_columns - 1
    if torch.any(predecessor_columns < 0):
        raise ValueError("a response token has no causal predecessor state.")

    if not torch.all(attention_mask[rows, target_columns].to(dtype=torch.bool)):
        raise ValueError("response_mask selects a target outside attention_mask.")
    if not torch.all(attention_mask[rows, predecessor_columns].to(dtype=torch.bool)):
        raise ValueError("a selected response target has a padded predecessor state.")

    labels = responses[rows, response_columns].to(dtype=torch.long)
    packed_targets = input_ids[rows, target_columns].to(dtype=torch.long)
    if not torch.equal(labels, packed_targets):
        raise ValueError(
            "responses do not match the response suffix in input_ids; refusing "
            "to project logits with an ambiguous token alignment."
        )

    predecessor_flat = rows * sequence_length + predecessor_columns
    ledger = unpadded_indices.to(dtype=predecessor_flat.dtype)
    packed_positions = torch.searchsorted(ledger, predecessor_flat)
    in_bounds = packed_positions < ledger.numel()
    if not torch.all(in_bounds):
        raise ValueError("a response predecessor is missing from the packed ledger.")
    if not torch.equal(ledger[packed_positions], predecessor_flat):
        raise ValueError("a response predecessor does not map exactly into the packed ledger.")

    if packed_positions.numel() > 1 and not torch.all(
        packed_positions[1:] > packed_positions[:-1]
    ):
        raise ValueError("response predecessor positions must be strictly increasing.")

    return ResponseProjectionPlan(
        packed_predecessor_positions=packed_positions.to(dtype=torch.long),
        labels=labels,
        response_mask=selected_mask,
        output_response_mask=output_response_mask,
        padding_only=padding_only,
        packed_token_count=int(unpadded_indices.numel()),
    )


def shard_response_projection_plan(
    plan: ResponseProjectionPlan,
    *,
    sequence_parallel_size: int,
    sequence_parallel_rank: int,
    padding_size: int,
) -> SequenceParallelResponseProjectionPlan:
    """Map global packed response positions onto one Ulysses token shard."""

    if isinstance(sequence_parallel_size, bool) or sequence_parallel_size <= 0:
        raise ValueError("sequence_parallel_size must be a positive integer.")
    if (
        isinstance(sequence_parallel_rank, bool)
        or sequence_parallel_rank < 0
        or sequence_parallel_rank >= sequence_parallel_size
    ):
        raise ValueError(
            "sequence_parallel_rank must be in [0, sequence_parallel_size)."
        )
    if isinstance(padding_size, bool) or padding_size < 0:
        raise ValueError("padding_size must be a non-negative integer.")

    padded_token_count = plan.packed_token_count + padding_size
    if padded_token_count % sequence_parallel_size != 0:
        raise ValueError(
            "padded packed-token count must be divisible by sequence parallel size: "
            f"tokens={plan.packed_token_count} padding={padding_size} "
            f"sp={sequence_parallel_size}."
        )
    shard_size = padded_token_count // sequence_parallel_size
    if shard_size <= 0:
        raise ValueError("each sequence-parallel rank must receive at least one token.")
    shard_start = sequence_parallel_rank * shard_size
    shard_end = shard_start + shard_size

    global_selected_count = 0 if plan.padding_only else int(plan.labels.numel())
    local_response_mask = torch.zeros_like(plan.output_response_mask, dtype=torch.bool)
    if not plan.padding_only:
        if plan.packed_predecessor_positions.numel() != plan.labels.numel():
            raise ValueError(
                "response positions and labels must have identical lengths before sharding."
            )
        selected_coordinates = plan.response_mask.to(dtype=torch.bool).nonzero(
            as_tuple=False
        )
        if selected_coordinates.shape[0] != plan.labels.numel():
            raise ValueError(
                "response_mask token count must match response positions before sharding."
            )
        owned = (
            (plan.packed_predecessor_positions >= shard_start)
            & (plan.packed_predecessor_positions < shard_end)
        )
        local_positions = (
            plan.packed_predecessor_positions[owned] - shard_start
        ).to(dtype=torch.long)
        local_labels = plan.labels[owned]
        local_coordinates = selected_coordinates[owned]
        if local_coordinates.numel() > 0:
            local_response_mask[
                local_coordinates[:, 0], local_coordinates[:, 1]
            ] = True
        local_selected_count = int(owned.sum().item())
    else:
        local_positions = plan.packed_predecessor_positions.new_empty(
            (0,), dtype=torch.long
        )
        local_labels = plan.labels.new_empty((0,), dtype=torch.long)
        local_selected_count = 0

    if local_selected_count == 0:
        # Every FSDP rank must keep the LM head in the graph even when its token
        # shard owns no response. The dummy row is multiplied by zero downstream.
        local_positions = plan.packed_predecessor_positions.new_zeros(
            (1,), dtype=torch.long
        )
        local_labels = plan.labels[:1].to(dtype=torch.long)
        if local_labels.numel() != 1:
            raise ValueError("response projection has no label for its dummy token.")
        padding_only = True
    else:
        padding_only = False

    return SequenceParallelResponseProjectionPlan(
        local_predecessor_positions=local_positions,
        labels=local_labels,
        response_mask=local_response_mask,
        output_response_mask=local_response_mask,
        padding_only=padding_only,
        local_selected_response_tokens=local_selected_count,
        global_selected_response_tokens=global_selected_count,
        packed_shard_start=shard_start,
        packed_shard_end=shard_end,
    )


class _MergeSequenceParallelResponseOutputs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_outputs: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        ctx.sequence_parallel_size = dist.get_world_size(group=group)
        merged = local_outputs.contiguous().clone()
        dist.all_reduce(merged, op=dist.ReduceOp.SUM, group=group)
        return merged

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # The same gathered loss is evaluated on every SP rank. FSDP averages
        # parameter gradients across those ranks, so preserve VERL's Ulysses
        # gather convention by scaling the owner gradient by the SP world size.
        return grad_output * ctx.sequence_parallel_size, None


def merge_sequence_parallel_response_outputs(
    local_outputs: torch.Tensor,
) -> torch.Tensor:
    """Merge disjoint response grids while preserving Ulysses gradients."""

    group = get_ulysses_sequence_parallel_group()
    if group is None:
        return local_outputs
    return _MergeSequenceParallelResponseOutputs.apply(local_outputs, group)


def scatter_response_outputs(
    selected_values: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter selected response-token values back to the padded response grid."""

    if selected_values.ndim != 1:
        raise ValueError("selected response values must be a rank-1 tensor.")
    mask = response_mask.to(dtype=torch.bool)
    expected = int(mask.sum().item())
    if selected_values.numel() != expected:
        raise ValueError(
            "selected response value count differs from response_mask: "
            f"values={selected_values.numel()} mask={expected}."
        )
    return selected_values.new_zeros(mask.shape).masked_scatter(mask, selected_values)


def zero_padding_response_outputs(
    selected_logits: torch.Tensor,
    output_response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return graph-connected zeros for a transport-padding-only microbatch."""

    if selected_logits.ndim != 2 or selected_logits.shape[0] != 1:
        raise ValueError(
            "padding-only projection must keep exactly one dummy logit row: "
            f"logits_shape={tuple(selected_logits.shape)}."
        )
    if torch.any(output_response_mask):
        raise ValueError("padding-only output mask must not contain valid tokens.")
    dependency = selected_logits.float().sum() * 0.0
    return dependency.expand(output_response_mask.shape)
