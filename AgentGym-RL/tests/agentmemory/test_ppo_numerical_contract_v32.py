import os
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from tensordict import TensorDict

from verl.agent_trainer.ppo import core_algos
from verl.utils import torch_functional as verl_F
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.ppo_token_normalization import (
    build_legacy_asymmetric_batch_contract,
    distributed_sum,
    mask_padding_rows,
    optimizer_step_readback,
    scale_token_mean_loss,
    summarize_dynamic_micro_batches,
    validate_dynamic_batch_token_caps,
    valid_response_token_count,
    validate_worker_batch_readback,
)


def _batch_contract(**overrides):
    arguments = {
        "actor_mini_batch_size": 64,
        "critic_mini_batch_size": 512,
        "rollout_n": 8,
        "world_size": 8,
        "actor_sequence_parallel_size": 1,
        "critic_sequence_parallel_size": 1,
        "per_gpu_micro_batches": {
            "actor": 2,
            "critic": 2,
            "critic_forward": 2,
            "reference_logprob": 2,
            "rollout_logprob": 2,
        },
        "legacy_micro_batches": {
            "actor": None,
            "critic": None,
            "critic_forward": None,
            "reference_logprob": None,
            "rollout_logprob": None,
        },
        "actor_ppo_epochs": 1,
        "critic_ppo_epochs": 2,
        "expected_per_gpu_micro_batch_size": 2,
    }
    arguments.update(overrides)
    return build_legacy_asymmetric_batch_contract(**arguments)


def _distributed_token_mean_worker(rank, world_size, rendezvous_path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            model.module.weight.fill_(2.0)

        inputs = torch.tensor([[1.0], [3.0]])
        response_mask = torch.ones(2) if rank == 0 else torch.zeros(2)
        outputs = model(inputs).squeeze(-1)
        local_mean = verl_F.masked_mean(outputs, response_mask)
        local_count = valid_response_token_count(response_mask)
        global_count = distributed_sum(local_count)
        scaled_loss = scale_token_mean_loss(
            local_mean, local_count, global_count
        )
        if rank == 1:
            torch.testing.assert_close(scaled_loss, torch.zeros_like(scaled_loss))

        global_sum = distributed_sum(
            verl_F.masked_sum(outputs.detach(), response_mask)
        )
        global_mean = global_sum / global_count
        torch.testing.assert_close(
            global_mean, torch.tensor(4.0, dtype=global_mean.dtype)
        )

        scaled_loss.backward()
        expected_gradient = torch.tensor([[2.0]])
        torch.testing.assert_close(model.module.weight.grad, expected_gradient)
        if not torch.isfinite(model.module.weight.grad).all():
            raise AssertionError("distributed token-mean gradient is not finite")
    finally:
        dist.destroy_process_group()


class FiniteZeroReductionTest(unittest.TestCase):
    def test_masked_nan_and_all_zero_mask_are_differentiable(self):
        values = torch.tensor([float("nan"), 4.0], requires_grad=True)
        mean = verl_F.masked_mean(values, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(mean, torch.tensor(4.0))
        mean.backward()
        torch.testing.assert_close(values.grad, torch.tensor([0.0, 1.0]))

        all_padding = torch.tensor([2.0, -3.0], requires_grad=True)
        zero_mean = verl_F.masked_mean(all_padding, torch.zeros(2))
        torch.testing.assert_close(zero_mean, torch.tensor(0.0))
        zero_mean.backward()
        torch.testing.assert_close(all_padding.grad, torch.zeros(2))

    def test_padding_only_actor_and_critic_losses_have_zero_gradient(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            with self.subTest(device=device.type):
                parameter = torch.tensor(0.25, device=device, requires_grad=True)
                mask = torch.zeros((2, 3), device=device)
                old_log_prob = torch.zeros((2, 3), device=device)
                log_prob = parameter.expand_as(old_log_prob)
                advantages = torch.ones_like(old_log_prob)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    eos_mask=mask,
                    cliprange=0.2,
                )
                entropy_loss = verl_F.masked_mean(
                    parameter.square().expand_as(mask), mask
                )
                explicit_kl = verl_F.masked_mean(
                    core_algos.kl_penalty(log_prob, old_log_prob, "kl"), mask
                )
                vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                    vpreds=parameter.expand_as(mask),
                    returns=torch.ones_like(mask),
                    values=torch.zeros_like(mask),
                    eos_mask=mask,
                    cliprange_value=0.5,
                )
                losses = (
                    pg_loss,
                    pg_clipfrac,
                    ppo_kl,
                    entropy_loss,
                    explicit_kl,
                    vf_loss,
                    vf_clipfrac,
                )
                for loss in losses:
                    self.assertTrue(torch.isfinite(loss).item())
                    torch.testing.assert_close(loss, torch.zeros_like(loss))

                combined = sum(losses)
                scaled = scale_token_mean_loss(
                    combined,
                    local_token_count=torch.tensor(0.0, device=device),
                    global_token_count=torch.tensor(7.0, device=device),
                )
                scaled.backward()
                self.assertTrue(torch.isfinite(parameter.grad).item())
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))

    def test_global_zero_token_count_fails_closed(self):
        loss = torch.tensor(0.0, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "global token count"):
            scale_token_mean_loss(
                loss,
                local_token_count=torch.tensor(0.0),
                global_token_count=torch.tensor(0.0),
            )


class PaddingInvariantGroupCreditTest(unittest.TestCase):
    def setUp(self):
        self.rewards = torch.tensor(
            [[1.0, 0.0], [3.0, 0.0], [2.0, 0.0], [5.0, 0.0]]
        )
        self.response_mask = torch.ones_like(self.rewards)
        self.indices = np.asarray(["a", "a", "b", "b"], dtype=object)

    def _padded_inputs(self):
        rewards = torch.cat([self.rewards, self.rewards[:2]], dim=0)
        response_mask = torch.cat(
            [self.response_mask, self.response_mask[:2]], dim=0
        )
        indices = np.concatenate([self.indices, self.indices[:2]])
        valid = torch.tensor([True, True, True, True, False, False])
        return rewards, response_mask, indices, valid

    def test_grpo_ignores_padding_rows(self):
        reference, _ = core_algos.compute_grpo_outcome_advantage(
            self.rewards, self.response_mask, self.indices
        )
        rewards, response_mask, indices, valid = self._padded_inputs()
        padded, _ = core_algos.compute_grpo_outcome_advantage(
            rewards, response_mask, indices, sample_mask=valid
        )
        torch.testing.assert_close(padded[:4], reference)
        torch.testing.assert_close(padded[4:], torch.zeros_like(padded[4:]))

    def test_rloo_ignores_padding_rows(self):
        reference, _ = core_algos.compute_rloo_outcome_advantage(
            self.rewards, self.response_mask, self.indices
        )
        rewards, response_mask, indices, valid = self._padded_inputs()
        padded, _ = core_algos.compute_rloo_outcome_advantage(
            rewards, response_mask, indices, sample_mask=valid
        )
        torch.testing.assert_close(padded[:4], reference)
        torch.testing.assert_close(padded[4:], torch.zeros_like(padded[4:]))

    def test_row_mask_is_aligned_and_shape_checked(self):
        masked = mask_padding_rows(
            self.response_mask, torch.tensor([True, False, True, False])
        )
        torch.testing.assert_close(masked[1], torch.zeros_like(masked[1]))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            mask_padding_rows(self.response_mask, torch.ones(4, 1))


class BatchContractTest(unittest.TestCase):
    def test_v32_legacy_compensation_and_step_readback(self):
        contract = _batch_contract()
        self.assertEqual(contract["actor_local_mini_batch_rows"], 64)
        self.assertEqual(contract["critic_local_mini_batch_rows"], 64)
        fixtures = {
            480: (1, 2),
            256: (1, 2),
            672: (2, 4),
            728: (2, 4),
        }
        for global_rows, expected in fixtures.items():
            with self.subTest(global_rows=global_rows):
                readback = optimizer_step_readback(contract, global_rows)
                self.assertEqual(
                    (
                        readback["actor_optimizer_steps"],
                        readback["critic_optimizer_steps"],
                    ),
                    expected,
                )

    def test_sp2_uses_data_parallel_rows_for_step_readback(self):
        contract = _batch_contract(
            actor_sequence_parallel_size=2,
            critic_sequence_parallel_size=2,
        )
        readback = optimizer_step_readback(contract, 640)
        self.assertEqual(contract["actor_data_parallel_size"], 4)
        self.assertEqual(readback["actor_local_rows"], 160)
        self.assertEqual(readback["critic_local_rows"], 160)
        self.assertEqual(readback["actor_minibatches_per_epoch"], 2)
        self.assertEqual(readback["critic_minibatches_per_epoch"], 2)
        self.assertEqual(readback["actor_optimizer_steps"], 2)
        self.assertEqual(readback["critic_optimizer_steps"], 4)

    def test_mixed_batch_unit_mode_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "different flattened-row units"):
            _batch_contract(critic_mini_batch_size=64)

    def test_legacy_and_per_gpu_micro_fields_are_mutually_exclusive(self):
        legacy = {
            "actor": 16,
            "critic": None,
            "critic_forward": None,
            "reference_logprob": None,
            "rollout_logprob": None,
        }
        with self.assertRaisesRegex(ValueError, "deprecated global"):
            _batch_contract(legacy_micro_batches=legacy)

    def test_worker_post_normalization_readback(self):
        contract = _batch_contract()
        actor = validate_worker_batch_readback(
            contract,
            role="actor",
            normalized_mini_batch_rows=64,
            per_gpu_micro_batch_rows=2,
        )
        critic = validate_worker_batch_readback(
            contract,
            role="critic",
            normalized_mini_batch_rows=64,
            per_gpu_micro_batch_rows=2,
            forward_per_gpu_micro_batch_rows=2,
        )
        self.assertEqual(actor["normalized_mini_batch_rows"], 64)
        self.assertEqual(critic["forward_per_gpu_micro_batch_rows"], 2)
        with self.assertRaisesRegex(ValueError, "readback mismatch"):
            validate_worker_batch_readback(
                contract,
                role="critic",
                normalized_mini_batch_rows=8,
                per_gpu_micro_batch_rows=2,
                forward_per_gpu_micro_batch_rows=2,
            )

    def test_role_specific_micro_batch_declaration(self):
        declared = {
            "actor": 2,
            "critic": 4,
            "critic_forward": 4,
            "reference_logprob": 2,
            "rollout_logprob": 2,
        }
        contract = _batch_contract(
            per_gpu_micro_batches=declared,
            expected_per_gpu_micro_batch_size=None,
            expected_per_gpu_micro_batches=declared,
        )

        self.assertIsNone(contract["expected_per_gpu_micro_batch_size"])
        self.assertEqual(
            contract["expected_per_gpu_micro_batches"], declared
        )
        self.assertEqual(
            validate_worker_batch_readback(
                contract,
                role="actor",
                normalized_mini_batch_rows=64,
                per_gpu_micro_batch_rows=2,
            )["per_gpu_micro_batch_rows"],
            2,
        )
        critic = validate_worker_batch_readback(
            contract,
            role="critic",
            normalized_mini_batch_rows=64,
            per_gpu_micro_batch_rows=4,
            forward_per_gpu_micro_batch_rows=4,
        )
        self.assertEqual(critic["per_gpu_micro_batch_rows"], 4)
        self.assertEqual(critic["forward_per_gpu_micro_batch_rows"], 4)

    def test_role_specific_micro_batch_declaration_fails_closed(self):
        configured = {
            "actor": 2,
            "critic": 4,
            "critic_forward": 4,
            "reference_logprob": 2,
            "rollout_logprob": 2,
        }
        declared = dict(configured, critic=2)
        with self.assertRaisesRegex(ValueError, "role-specific declaration"):
            _batch_contract(
                per_gpu_micro_batches=configured,
                expected_per_gpu_micro_batch_size=None,
                expected_per_gpu_micro_batches=declared,
            )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _batch_contract(
                expected_per_gpu_micro_batches=configured,
            )

    def test_dynamic_batch_contract_keeps_optimizer_readback(self):
        dynamic_roles = {
            "actor": True,
            "critic": True,
            "critic_forward": True,
            "reference_logprob": True,
            "rollout_logprob": True,
        }
        token_caps = {
            "actor": 131072,
            "critic": 163840,
            "critic_forward": 262144,
            "reference_logprob": 131072,
            "rollout_logprob": 131072,
        }
        contract = _batch_contract(
            expected_per_gpu_micro_batch_size=None,
            dynamic_roles=dynamic_roles,
            dynamic_max_token_lens=token_caps,
        )
        actor = validate_worker_batch_readback(
            contract,
            role="actor",
            normalized_mini_batch_rows=64,
            per_gpu_micro_batch_rows=999,
        )
        critic = validate_worker_batch_readback(
            contract,
            role="critic",
            normalized_mini_batch_rows=64,
            per_gpu_micro_batch_rows=999,
            forward_per_gpu_micro_batch_rows=999,
        )
        self.assertTrue(actor["dynamic_bsz"])
        self.assertNotIn("per_gpu_micro_batch_rows", actor)
        self.assertTrue(critic["dynamic_forward_bsz"])
        self.assertEqual(
            optimizer_step_readback(contract, 648)["actor_optimizer_steps"],
            2,
        )

    def test_dynamic_batch_token_cap_covers_padded_sequence(self):
        roles = {
            "actor": True,
            "critic": True,
            "critic_forward": True,
            "reference_logprob": False,
            "rollout_logprob": True,
        }
        caps = {
            "actor": 131072,
            "critic": 163840,
            "critic_forward": 262144,
            "reference_logprob": 131072,
            "rollout_logprob": 131072,
        }
        sequence_parallel_sizes = {role: 1 for role in roles}
        validate_dynamic_batch_token_caps(
            dynamic_roles=roles,
            dynamic_max_token_lens=caps,
            sequence_parallel_sizes=sequence_parallel_sizes,
            padded_sequence_length=126976,
        )

        with self.assertRaisesRegex(
            ValueError, "actor.*effective token cap.*81920.*126976"
        ):
            validate_dynamic_batch_token_caps(
                dynamic_roles=roles,
                dynamic_max_token_lens=dict(caps, actor=81920),
                sequence_parallel_sizes=sequence_parallel_sizes,
                padded_sequence_length=126976,
            )

    def test_dynamic_batch_token_cap_accounts_for_sequence_parallelism(self):
        validate_dynamic_batch_token_caps(
            dynamic_roles={"actor": True},
            dynamic_max_token_lens={"actor": 65536},
            sequence_parallel_sizes={"actor": 2},
            padded_sequence_length=126976,
        )

    def test_dynamic_batch_contract_requires_all_enabled_caps(self):
        with self.assertRaisesRegex(ValueError, "require a max token length"):
            _batch_contract(
                expected_per_gpu_micro_batch_size=None,
                dynamic_roles={"actor": True},
                dynamic_max_token_lens={},
            )

    def test_dynamic_batch_partition_and_summary(self):
        lengths = [10, 8, 4, 2]
        attention_mask = torch.zeros((4, 10), dtype=torch.long)
        for row, length in enumerate(lengths):
            attention_mask[row, :length] = 1
        batch = TensorDict(
            {
                "attention_mask": attention_mask,
                "input_ids": torch.arange(40).reshape(4, 10),
            },
            batch_size=[4],
        )
        micro_batches, indices = rearrange_micro_batches(
            batch, max_token_len=12
        )
        self.assertEqual(sorted(i for group in indices for i in group), list(range(4)))
        summary = summarize_dynamic_micro_batches(micro_batches)
        self.assertEqual(summary["micro_batches"], 2)
        self.assertEqual(summary["token_load_total"], sum(lengths))
        self.assertEqual(summary["rows_total"], 4)


class DistributedTokenMeanTest(unittest.TestCase):
    def test_padding_only_rank_matches_unpadded_global_gradient(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is unavailable")
        with tempfile.TemporaryDirectory() as temporary_directory:
            rendezvous_path = os.path.join(temporary_directory, "gloo-rendezvous")
            mp.spawn(
                _distributed_token_mean_worker,
                args=(2, rendezvous_path),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
