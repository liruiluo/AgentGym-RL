from __future__ import annotations

from dataclasses import is_dataclass
import unittest
from unittest.mock import patch

import torch
from torch.distributed.utils import _apply_to_tensors

from verl.models.transformers.qwen3_5 import Qwen3_5ResponseFusedPPOOutput
from verl.utils.experimental import torch_functional as experimental_torch
from verl.workers.response_only_logits import (
    build_response_projection_plan,
    scatter_response_outputs,
    zero_padding_response_outputs,
    zero_padding_selected_outputs,
)


class ResponseFusedOutputContractTest(unittest.TestCase):
    def test_fsdp_can_discover_output_tensors(self):
        log_probs = torch.randn(3, requires_grad=True)
        entropy = torch.randn(3, requires_grad=True)
        output = Qwen3_5ResponseFusedPPOOutput(
            log_probs=log_probs,
            entropy=entropy,
        )
        visited = []

        transformed = _apply_to_tensors(
            lambda tensor: visited.append(tensor) or tensor,
            output,
        )

        self.assertTrue(is_dataclass(transformed))
        self.assertEqual(
            {id(tensor) for tensor in visited},
            {id(log_probs), id(entropy)},
        )

    def test_fsdp_accepts_logprob_only_output(self):
        log_probs = torch.randn(3, requires_grad=True)
        output = Qwen3_5ResponseFusedPPOOutput(
            log_probs=log_probs,
            entropy=None,
        )
        visited = []

        transformed = _apply_to_tensors(
            lambda tensor: visited.append(tensor) or tensor,
            output,
        )

        self.assertTrue(is_dataclass(transformed))
        self.assertEqual([id(tensor) for tensor in visited], [id(log_probs)])


class FusedLinearEntropyGateTest(unittest.TestCase):
    def test_logprob_only_forward_preserves_values_and_gradients(self):
        torch.manual_seed(17)
        hidden_with_entropy = torch.randn(7, 5, requires_grad=True)
        weight_with_entropy = torch.randn(13, 5, requires_grad=True)
        hidden_logprob_only = hidden_with_entropy.detach().clone().requires_grad_(True)
        weight_logprob_only = weight_with_entropy.detach().clone().requires_grad_(True)
        labels = torch.tensor([0, 2, 4, 6, 8, 10, 12])
        fused = experimental_torch.FusedLinearForPPO(chunk_size=3)

        with patch.object(
            experimental_torch,
            "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE",
            False,
        ):
            log_probs_with_entropy, entropy = fused(
                hidden_with_entropy,
                weight_with_entropy,
                labels,
                compute_entropy=True,
            )
            log_probs_only, skipped_entropy = fused(
                hidden_logprob_only,
                weight_logprob_only,
                labels,
                compute_entropy=False,
            )

        self.assertIsNotNone(entropy)
        self.assertIsNone(skipped_entropy)
        torch.testing.assert_close(log_probs_only, log_probs_with_entropy)

        (-log_probs_with_entropy.mean()).backward()
        (-log_probs_only.mean()).backward()
        torch.testing.assert_close(hidden_logprob_only.grad, hidden_with_entropy.grad)
        torch.testing.assert_close(weight_logprob_only.grad, weight_with_entropy.grad)


def _fixture():
    input_ids = torch.tensor(
        [
            [0, 0, 10, 11, 12, 21, 22, 0],
            [0, 30, 31, 32, 33, 41, 42, 43],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    responses = torch.tensor([[21, 22, 0], [41, 42, 43]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    unpadded_indices = attention_mask.flatten().nonzero().flatten()
    return input_ids, attention_mask, responses, response_mask, unpadded_indices


class ResponseProjectionPlanTest(unittest.TestCase):
    def test_maps_response_targets_to_packed_predecessors(self):
        input_ids, attention_mask, responses, response_mask, indices = _fixture()
        plan = build_response_projection_plan(
            unpadded_indices=indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
            responses=responses,
            response_mask=response_mask,
        )

        self.assertEqual(plan.packed_token_count, 12)
        self.assertEqual(plan.labels.tolist(), [21, 22, 41, 42, 43])
        self.assertEqual(
            plan.packed_predecessor_positions.tolist(),
            [2, 3, 8, 9, 10],
        )

    def test_scatter_preserves_response_grid_and_gradient(self):
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        selected = torch.arange(5, dtype=torch.float32, requires_grad=True)
        scattered = scatter_response_outputs(selected, mask)

        self.assertEqual(
            scattered.tolist(),
            [[0.0, 1.0, 0.0], [2.0, 3.0, 4.0]],
        )
        scattered.sum().backward()
        self.assertTrue(torch.equal(selected.grad, torch.ones_like(selected)))

    def test_selected_projection_matches_masked_full_projection_gradients(self):
        input_ids, attention_mask, responses, response_mask, indices = _fixture()
        plan = build_response_projection_plan(
            unpadded_indices=indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
            responses=responses,
            response_mask=response_mask,
        )
        torch.manual_seed(7)
        hidden_full = torch.randn(2, 8, 6, dtype=torch.float64, requires_grad=True)
        weight_full = torch.randn(64, 6, dtype=torch.float64, requires_grad=True)
        hidden_selected = hidden_full.detach().clone().requires_grad_(True)
        weight_selected = weight_full.detach().clone().requires_grad_(True)

        full_logits = hidden_full @ weight_full.t()
        response_logits = full_logits[:, -responses.shape[1] - 1 : -1]
        full_log_probs = response_logits.log_softmax(dim=-1).gather(
            -1, responses.unsqueeze(-1)
        ).squeeze(-1)
        full_probs = response_logits.softmax(dim=-1)
        full_entropy = torch.logsumexp(response_logits, dim=-1) - (
            full_probs * response_logits
        ).sum(dim=-1)
        full_loss = -(
            (full_log_probs + 0.01 * full_entropy) * response_mask
        ).sum()
        full_loss.backward()

        packed_hidden = hidden_selected.reshape(-1, hidden_selected.shape[-1])[indices]
        selected_hidden = packed_hidden[plan.packed_predecessor_positions]
        selected_logits = selected_hidden @ weight_selected.t()
        selected_log_probs = selected_logits.log_softmax(dim=-1).gather(
            -1, plan.labels.unsqueeze(-1)
        ).squeeze(-1)
        selected_probs = selected_logits.softmax(dim=-1)
        selected_entropy = torch.logsumexp(selected_logits, dim=-1) - (
            selected_probs * selected_logits
        ).sum(dim=-1)
        sparse_log_probs = scatter_response_outputs(
            selected_log_probs, plan.response_mask
        )
        sparse_entropy = scatter_response_outputs(selected_entropy, plan.response_mask)
        selected_loss = -(
            (sparse_log_probs + 0.01 * sparse_entropy) * response_mask
        ).sum()
        selected_loss.backward()

        torch.testing.assert_close(sparse_log_probs, full_log_probs * response_mask)
        torch.testing.assert_close(sparse_entropy, full_entropy * response_mask)
        torch.testing.assert_close(hidden_selected.grad, hidden_full.grad)
        torch.testing.assert_close(weight_selected.grad, weight_full.grad)

    def test_response_loss_still_reaches_prompt_processing(self):
        input_ids, attention_mask, responses, response_mask, indices = _fixture()
        plan = build_response_projection_plan(
            unpadded_indices=indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
            responses=responses,
            response_mask=response_mask,
        )
        torch.manual_seed(11)
        token_states = torch.randn(2, 8, 4, requires_grad=True)
        causal_hidden = torch.cumsum(
            token_states * attention_mask.unsqueeze(-1), dim=1
        )
        packed_hidden = causal_hidden.reshape(-1, 4)[indices]
        loss = packed_hidden[plan.packed_predecessor_positions].square().sum()
        loss.backward()

        self.assertGreater(token_states.grad[0, 2:5].abs().sum().item(), 0.0)
        self.assertGreater(token_states.grad[1, 1:5].abs().sum().item(), 0.0)

    def test_rejects_missing_or_misaligned_response_tokens(self):
        input_ids, attention_mask, responses, response_mask, indices = _fixture()
        broken_responses = responses.clone()
        broken_responses[0, 0] = 63
        with self.assertRaisesRegex(ValueError, "do not match"):
            build_response_projection_plan(
                unpadded_indices=indices,
                input_ids=input_ids,
                attention_mask=attention_mask,
                responses=broken_responses,
                response_mask=response_mask,
            )

        no_response = torch.zeros_like(response_mask)
        with self.assertRaisesRegex(ValueError, "no valid response"):
            build_response_projection_plan(
                unpadded_indices=indices,
                input_ids=input_ids,
                attention_mask=attention_mask,
                responses=responses,
                response_mask=no_response,
            )

    def test_padding_only_microbatch_keeps_one_graph_connected_dummy_token(self):
        input_ids, attention_mask, responses, response_mask, indices = _fixture()
        plan = build_response_projection_plan(
            unpadded_indices=indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
            responses=responses,
            response_mask=response_mask,
            valid_sample_mask=torch.zeros(2, dtype=torch.bool),
        )

        self.assertTrue(plan.padding_only)
        self.assertEqual(plan.labels.tolist(), [21])
        self.assertEqual(plan.response_mask.sum().item(), 1)
        self.assertEqual(plan.output_response_mask.sum().item(), 0)

        selected_logits = torch.randn(1, 64, requires_grad=True)
        outputs = zero_padding_response_outputs(
            selected_logits,
            plan.output_response_mask,
        )
        self.assertEqual(tuple(outputs.shape), tuple(response_mask.shape))
        self.assertTrue(torch.equal(outputs, torch.zeros_like(outputs)))
        outputs.sum().backward()
        self.assertIsNotNone(selected_logits.grad)
        self.assertTrue(
            torch.equal(selected_logits.grad, torch.zeros_like(selected_logits))
        )

    def test_padding_only_microbatch_survives_precleared_response_mask(self):
        input_ids, attention_mask, responses, response_mask, indices = _fixture()
        plan = build_response_projection_plan(
            unpadded_indices=indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
            responses=responses,
            response_mask=torch.zeros_like(response_mask),
            valid_sample_mask=torch.zeros(2, dtype=torch.bool),
        )

        self.assertTrue(plan.padding_only)
        self.assertEqual(plan.labels.numel(), 1)
        self.assertEqual(plan.packed_predecessor_positions.tolist(), [0])
        self.assertEqual(plan.response_mask.sum().item(), 0)
        self.assertEqual(plan.output_response_mask.sum().item(), 0)

    def test_padding_only_fused_outputs_keep_graph_connection(self):
        output_mask = torch.zeros(2, 3, dtype=torch.bool)
        selected = torch.randn(1, requires_grad=True)

        outputs = zero_padding_selected_outputs(selected, output_mask)

        self.assertEqual(tuple(outputs.shape), tuple(output_mask.shape))
        self.assertTrue(torch.equal(outputs, torch.zeros_like(outputs)))
        outputs.sum().backward()
        self.assertIsNotNone(selected.grad)
        self.assertTrue(torch.equal(selected.grad, torch.zeros_like(selected)))

    def test_mixed_valid_and_padding_rows_project_only_valid_responses(self):
        input_ids, attention_mask, responses, response_mask, indices = _fixture()
        plan = build_response_projection_plan(
            unpadded_indices=indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
            responses=responses,
            response_mask=response_mask,
            valid_sample_mask=torch.tensor([1, 0], dtype=torch.bool),
        )

        self.assertFalse(plan.padding_only)
        self.assertEqual(plan.labels.tolist(), [21, 22])
        self.assertEqual(
            plan.output_response_mask.tolist(),
            [[True, True, False], [False, False, False]],
        )

    def test_valid_rows_without_response_tokens_still_fail_closed(self):
        input_ids, attention_mask, responses, response_mask, indices = _fixture()
        with self.assertRaisesRegex(ValueError, "no valid response"):
            build_response_projection_plan(
                unpadded_indices=indices,
                input_ids=input_ids,
                attention_mask=attention_mask,
                responses=responses,
                response_mask=torch.zeros_like(response_mask),
                valid_sample_mask=torch.tensor([1, 0], dtype=torch.bool),
            )

    def test_oom_shape_projects_only_response_positions(self):
        valid_lengths = [18_000, 16_000, 17_000, 29_929]
        response_lengths = [400, 400, 400, 628]
        sequence_length = 32_000
        response_width = 2_048
        prompt_width = sequence_length - response_width
        input_ids = torch.zeros(4, sequence_length, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        responses = torch.zeros(4, response_width, dtype=torch.long)
        response_mask = torch.zeros_like(responses)

        for row, (valid_length, response_length) in enumerate(
            zip(valid_lengths, response_lengths, strict=True)
        ):
            prompt_length = valid_length - response_length
            prompt_start = prompt_width - prompt_length
            input_ids[row, prompt_start:prompt_width] = 7
            attention_mask[row, prompt_start:prompt_width] = 1
            responses[row, :response_length] = 9
            response_mask[row, :response_length] = 1
            input_ids[row, prompt_width : prompt_width + response_length] = 9
            attention_mask[row, prompt_width : prompt_width + response_length] = 1

        indices = attention_mask.flatten().nonzero().flatten()
        plan = build_response_projection_plan(
            unpadded_indices=indices,
            input_ids=input_ids,
            attention_mask=attention_mask,
            responses=responses,
            response_mask=response_mask,
        )

        vocab_size = 248_320
        bytes_per_bf16 = 2
        full_projection_bytes = plan.packed_token_count * vocab_size * bytes_per_bf16
        selected_projection_bytes = plan.labels.numel() * vocab_size * bytes_per_bf16
        self.assertEqual(plan.packed_token_count, 80_929)
        self.assertEqual(plan.labels.numel(), 1_828)
        self.assertEqual(full_projection_bytes, 40_192_578_560)
        self.assertEqual(selected_projection_bytes, 907_857_920)
        self.assertGreater(full_projection_bytes / selected_projection_bytes, 44.0)


if __name__ == "__main__":
    unittest.main()
