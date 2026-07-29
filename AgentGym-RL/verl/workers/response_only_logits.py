"""Index planning for response-only causal-LM projections."""

from __future__ import annotations

from typing import NamedTuple

import torch


class ResponseProjectionPlan(NamedTuple):
    packed_predecessor_positions: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    packed_token_count: int


def build_response_projection_plan(
    *,
    unpadded_indices: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
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

    selected_mask = response_mask.to(dtype=torch.bool)
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
        packed_token_count=int(unpadded_indices.numel()),
    )


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
