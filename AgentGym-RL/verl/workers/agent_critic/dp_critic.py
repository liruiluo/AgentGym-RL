# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""
import itertools

import torch
from torch import nn, optim

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.workers.agent_critic import BasePPOCritic
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from verl.workers.ppo_token_normalization import (
    PPO_BATCH_CONTRACT_META_KEY,
    TokenWeightedMetricAccumulator,
    distributed_sum,
    mask_padding_rows,
    scale_token_mean_loss,
    valid_response_token_count,
    validate_worker_batch_readback,
)
from verl.workers.fsdp_gradient_accumulation import (
    fsdp_gradient_sync_context,
    should_defer_fsdp_gradient_sync,
)
from verl.workers.qwen35_runtime import qwen3_5_packed_forward_kwargs

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOCritic']


def _select_response_state_values(
    full_sequence_values: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Align critic states with the tokens whose log probabilities PPO updates.

    The actor scores response token ``t`` from the causal output at ``t - 1``.
    The critic must use that same pre-token state, including the final prompt
    position for the first response token.
    """

    if full_sequence_values.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("critic values and response_mask must both be rank-2 tensors.")
    if full_sequence_values.shape[0] != response_mask.shape[0]:
        raise ValueError("critic values and response_mask must share a batch size.")
    response_length = response_mask.shape[-1]
    if full_sequence_values.shape[-1] <= response_length:
        raise ValueError(
            "critic sequence must include at least one prompt state before the response."
        )
    response_state_values = full_sequence_values[
        :, -response_length - 1 : -1
    ]
    return response_state_values * response_mask.to(
        device=response_state_values.device,
        dtype=response_state_values.dtype,
    )


class DataParallelPPOCritic(BasePPOCritic):

    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        print(f'Critic use_remove_padding={self.use_remove_padding}')

        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        fsdp_config = self.config.model.get('fsdp_config', {})
        self.use_no_sync_for_gradient_accumulation = bool(
            fsdp_config.get('use_no_sync_for_gradient_accumulation', False)
        )
        print(
            'Critic use_no_sync_for_gradient_accumulation='
            f'{self.use_no_sync_for_gradient_accumulation}'
        )

    def _forward_micro_batch(self, micro_batch):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                packed_forward_kwargs = qwen3_5_packed_forward_kwargs(
                    self.critic_module,
                    cu_seqlens,
                    self.ulysses_sequence_parallel_size,
                )
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                    **packed_forward_kwargs,
                )  # prevent model thinks we are generating
                values_rmpad = output.logits.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outpus_and_unpad(values_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
            else:
                output = self.critic_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=False)  # prevent model thinks we are generating
                values = output.logits.squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        self.critic_optimizer.step()
        return grad_norm

    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info['micro_batch_size']
        select_keys = ['input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            values = values[revert_indices]

        return _select_response_state_values(values, data.batch['response_mask'])

    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        select_keys = ['input_ids', 'attention_mask', 'position_ids', 'values', 'returns', 'response_mask']
        if core_algos.PPO_VALID_SAMPLE_MASK in data.batch.keys():
            select_keys.append(core_algos.PPO_VALID_SAMPLE_MASK)
        batch = data.select(batch_keys=select_keys).batch
        loss_group = data.meta_info.get('ppo_loss_process_group')
        metric_group = data.meta_info.get('ppo_metric_process_group', loss_group)
        batch_contract = data.meta_info.get(PPO_BATCH_CONTRACT_META_KEY)
        batch_readback = None
        if batch_contract is not None:
            batch_readback = validate_worker_batch_readback(
                batch_contract,
                role='critic',
                normalized_mini_batch_rows=self.config.ppo_mini_batch_size,
                per_gpu_micro_batch_rows=self.config.ppo_micro_batch_size_per_gpu,
                forward_per_gpu_micro_batch_rows=self.config.forward_micro_batch_size_per_gpu,
            )
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        raw_ppo_epochs = self.config.ppo_epochs
        ppo_epochs = int(raw_ppo_epochs)
        if isinstance(raw_ppo_epochs, bool) or ppo_epochs <= 0 or ppo_epochs != raw_ppo_epochs:
            raise ValueError(f"critic ppo_epochs must be a positive integer, got {raw_ppo_epochs!r}.")

        token_metrics = TokenWeightedMetricAccumulator()
        optimizer_steps = 0
        deferred_sync_micro_batches = 0
        for _ in range(ppo_epochs):
            for data in dataloader:
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()
                mini_batch_response_mask = mask_padding_rows(
                    mini_batch['response_mask'],
                    mini_batch.get(core_algos.PPO_VALID_SAMPLE_MASK),
                )
                global_token_count = distributed_sum(
                    valid_response_token_count(mini_batch_response_mask),
                    group=loss_group,
                )
                if global_token_count.item() <= 0:
                    raise ValueError("PPO critic mini-batch has no valid response tokens.")

                for micro_batch_index, data in enumerate(micro_batches):
                    is_last_micro_batch = micro_batch_index == len(micro_batches) - 1
                    defer_sync = should_defer_fsdp_gradient_sync(
                        self.critic_module,
                        enabled=self.use_no_sync_for_gradient_accumulation,
                        is_last_micro_batch=is_last_micro_batch,
                    )
                    deferred_sync_micro_batches += int(defer_sync)
                    with fsdp_gradient_sync_context(
                        self.critic_module,
                        enabled=self.use_no_sync_for_gradient_accumulation,
                        is_last_micro_batch=is_last_micro_batch,
                    ):
                        data = data.cuda()  # critic device is cpu when using offload
                        values = data['values']
                        returns = data['returns']
                        eos_mask = mask_padding_rows(
                            data['response_mask'],
                            data.get(core_algos.PPO_VALID_SAMPLE_MASK),
                        )
                        local_token_count = valid_response_token_count(eos_mask)
                        vpreds = _select_response_state_values(
                            self._forward_micro_batch(data), eos_mask
                        )

                        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                            vpreds=vpreds,
                            values=values,
                            returns=returns,
                            eos_mask=eos_mask,
                            cliprange_value=self.config.cliprange_value,
                        )
                        loss = scale_token_mean_loss(
                            vf_loss,
                            local_token_count,
                            global_token_count,
                            group=loss_group,
                        )
                        loss.backward()
                        token_metrics.add(
                            {
                                'critic/vf_loss': vf_loss.detach().item(),
                                'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                                'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
                            },
                            local_token_count,
                        )

                grad_norm = self._optimizer_step()
                optimizer_steps += 1
                metrics.setdefault('critic/grad_norm', []).append(grad_norm.detach().item())
        self.critic_optimizer.zero_grad()
        metrics['critic/ppo_epochs'] = [float(ppo_epochs)]
        metrics['critic/optimizer_steps_per_update'] = [float(optimizer_steps)]
        metrics['critic/minibatches_per_epoch'] = [float(optimizer_steps / ppo_epochs)]
        metrics['critic/fsdp_no_sync_gradient_accumulation'] = [
            float(self.use_no_sync_for_gradient_accumulation)
        ]
        metrics['critic/deferred_gradient_sync_microbatches'] = [
            float(deferred_sync_micro_batches)
        ]
        if batch_readback is not None:
            metrics['critic/normalized_mini_batch_rows'] = [
                float(batch_readback['normalized_mini_batch_rows'])
            ]
            metrics['critic/per_gpu_micro_batch_rows'] = [
                float(batch_readback['per_gpu_micro_batch_rows'])
            ]
            metrics['critic/forward_per_gpu_micro_batch_rows'] = [
                float(batch_readback['forward_per_gpu_micro_batch_rows'])
            ]
        metrics.update(token_metrics.reduce(group=metric_group))
        return metrics
