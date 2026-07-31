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
Single Process Actor
"""

import itertools
import json
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.workers.agent_actor import BasePPOActor
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from verl.workers.ppo_token_normalization import (
    PPO_BATCH_CONTRACT_META_KEY,
    TokenWeightedMetricAccumulator,
    distributed_sum,
    mask_padding_rows,
    scale_token_mean_loss,
    summarize_dynamic_micro_batches,
    valid_response_token_count,
    validate_worker_batch_readback,
)
from verl.models.transformers.qwen3_5 import is_qwen3_5_model_type
from verl.workers.qwen35_runtime import (
    model_type_from_module,
    qwen3_5_packed_forward_kwargs,
)
from verl.workers.response_only_logits import (
    build_response_projection_plan,
    scatter_response_outputs,
    zero_padding_response_outputs,
)
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.use_response_only_logits = bool(
            self.config.get('use_response_only_logits', False)
        )
        if self.use_response_only_logits:
            model_type = model_type_from_module(self.actor_module)
            if not is_qwen3_5_model_type(model_type):
                raise NotImplementedError(
                    "Response-only PPO logits currently support only Qwen3.5, "
                    f"got model_type={model_type!r}."
                )
            if not self.use_remove_padding:
                raise ValueError(
                    "Response-only PPO logits require use_remove_padding=true."
                )
            if self.ulysses_sequence_parallel_size != 1:
                raise NotImplementedError(
                    "Response-only PPO logits require Ulysses sequence parallel size 1."
                )
        print(f'Actor use_response_only_logits={self.use_response_only_logits}')
        self._response_only_readback_logged = False
        self._dynamic_bsz_readback = os.environ.get(
            'AGENTMEMORY_DYNAMIC_BSZ_READBACK', '0'
        ) == '1'
        self._dynamic_logprob_calls = 0

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
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

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                packed_forward_kwargs = qwen3_5_packed_forward_kwargs(
                    self.actor_module,
                    cu_seqlens,
                    self.ulysses_sequence_parallel_size,
                )
                if self.use_response_only_logits:
                    projection = build_response_projection_plan(
                        unpadded_indices=indices,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        responses=micro_batch['responses'],
                        response_mask=micro_batch['response_mask'],
                        valid_sample_mask=micro_batch.get(
                            core_algos.PPO_VALID_SAMPLE_MASK
                        ),
                    )
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        use_cache=False,
                        logits_to_keep=projection.packed_predecessor_positions,
                        **packed_forward_kwargs,
                    )
                    selected_logits = output.logits.squeeze(0)
                    expected_shape = (
                        projection.labels.numel(),
                        selected_logits.shape[-1],
                    )
                    if tuple(selected_logits.shape) != expected_shape:
                        raise RuntimeError(
                            "Qwen3.5 response-only LM head returned an unexpected "
                            f"shape: logits={tuple(selected_logits.shape)} "
                            f"expected={expected_shape}."
                        )
                    if projection.padding_only:
                        zero_outputs = zero_padding_response_outputs(
                            selected_logits,
                            projection.output_response_mask,
                        )
                        entropy = zero_outputs
                        log_probs = zero_outputs
                    else:
                        selected_logits.div_(temperature)
                        selected_entropy = self.compute_entropy_from_logits(
                            selected_logits
                        )
                        selected_log_probs = logprobs_from_logits(
                            logits=selected_logits,
                            labels=projection.labels,
                        )
                        entropy = scatter_response_outputs(
                            selected_entropy,
                            projection.response_mask,
                        )
                        log_probs = scatter_response_outputs(
                            selected_log_probs,
                            projection.response_mask,
                        )
                    if (
                        not projection.padding_only
                        and not self._response_only_readback_logged
                    ):
                        selected_count = projection.labels.numel()
                        role = 'actor' if self.actor_optimizer is not None else 'reference'
                        print(
                            "Response-only PPO logits readback: "
                            f"role={role} packed_tokens={projection.packed_token_count} "
                            f"selected_response_tokens={selected_count} "
                            "projection_reduction_ratio="
                            f"{projection.packed_token_count / selected_count:.6f}"
                        )
                        self._response_only_readback_logged = True
                else:
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        use_cache=False,
                        **packed_forward_kwargs,
                    )  # prevent model thinks we are generating
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                    logits_rmpad.div_(temperature)

                    # compute entropy
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                    # gather log_prob if sp > 1
                    if self.use_ulysses_sp:
                        # gather and unpad for the ulysses sp
                        log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                                gather_dim=0,
                                                                unpad_dim=0,
                                                                padding_size=pad_size)
                    # pad back to (bsz, seqlen)
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                             indices=indices,
                                             batch=batch_size,
                                             seqlen=seqlen)
                    full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                               indices=indices,
                                               batch=batch_size,
                                               seqlen=seqlen)

                    # only return response part:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        if self.use_response_only_logits:
            select_keys.append('response_mask')
            if core_algos.PPO_VALID_SAMPLE_MASK in data.batch.keys():
                select_keys.append(core_algos.PPO_VALID_SAMPLE_MASK)
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            if self._dynamic_bsz_readback:
                self._dynamic_logprob_calls += 1
                summary = summarize_dynamic_micro_batches(micro_batches)
                summary.update({
                    'call': self._dynamic_logprob_calls,
                    'max_token_len': int(max_token_len),
                    'rank': (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else 0
                    ),
                    'role': 'reference_logprob' if self.actor_optimizer is None else 'rollout_logprob',
                })
                print(
                    "AgentMemory PPO dynamic-batch readback: "
                    + json.dumps(summary, sort_keys=True)
                )
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'responses', 'response_mask']
        if core_algos.PPO_VALID_SAMPLE_MASK in data.batch.keys():
            select_keys.append(core_algos.PPO_VALID_SAMPLE_MASK)
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        loss_group = data.meta_info.get('ppo_loss_process_group')
        metric_group = data.meta_info.get('ppo_metric_process_group', loss_group)
        batch_contract = data.meta_info.get(PPO_BATCH_CONTRACT_META_KEY)
        batch_readback = None
        if batch_contract is not None:
            batch_readback = validate_worker_batch_readback(
                batch_contract,
                role='actor',
                normalized_mini_batch_rows=self.config.ppo_mini_batch_size,
                per_gpu_micro_batch_rows=self.config.ppo_micro_batch_size_per_gpu,
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        raw_ppo_epochs = self.config.ppo_epochs
        ppo_epochs = int(raw_ppo_epochs)
        if isinstance(raw_ppo_epochs, bool) or ppo_epochs <= 0 or ppo_epochs != raw_ppo_epochs:
            raise ValueError(f"actor ppo_epochs must be a positive integer, got {raw_ppo_epochs!r}.")

        metrics = {}
        token_metrics = TokenWeightedMetricAccumulator()
        optimizer_steps = 0
        dynamic_summaries = []
        for _ in range(ppo_epochs):
            for data in dataloader:
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                    if self._dynamic_bsz_readback:
                        dynamic_summaries.append(
                            summarize_dynamic_micro_batches(micro_batches)
                        )
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                mini_batch_response_mask = mask_padding_rows(
                    mini_batch['response_mask'],
                    mini_batch.get(core_algos.PPO_VALID_SAMPLE_MASK),
                )
                global_token_count = distributed_sum(
                    valid_response_token_count(mini_batch_response_mask),
                    group=loss_group,
                )
                if global_token_count.item() <= 0:
                    raise ValueError("PPO actor mini-batch has no valid response tokens.")

                for data in micro_batches:
                    data = data.cuda()  # actor device is cpu when using offload
                    response_mask = mask_padding_rows(
                        data['response_mask'],
                        data.get(core_algos.PPO_VALID_SAMPLE_MASK),
                    )
                    local_token_count = valid_response_token_count(response_mask)
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature
                    )
                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        eos_mask=response_mask,
                        cliprange=self.config.clip_ratio,
                    )
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)
                    policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff

                    kl_metric_values = {}
                    if self.config.use_kl_loss:
                        kld = core_algos.kl_penalty(
                            logprob=log_prob,
                            ref_logprob=data['ref_log_prob'],
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = masked_mean(kld, response_mask)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        kl_metric_values = {
                            'actor/kl_loss': kl_loss.detach().item(),
                            'actor/kl_coef': self.config.kl_loss_coef,
                        }

                    loss = scale_token_mean_loss(
                        policy_loss,
                        local_token_count,
                        global_token_count,
                        group=loss_group,
                    )
                    loss.backward()
                    token_metrics.add(
                        {
                            'actor/entropy_loss': entropy_loss.detach().item(),
                            'actor/pg_loss': pg_loss.detach().item(),
                            'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                            'actor/ppo_kl': ppo_kl.detach().item(),
                            **kl_metric_values,
                        },
                        local_token_count,
                    )

                grad_norm = self._optimizer_step()
                optimizer_steps += 1
                metrics.setdefault('actor/grad_norm', []).append(grad_norm.detach().item())
        self.actor_optimizer.zero_grad()
        metrics['actor/ppo_epochs'] = [float(ppo_epochs)]
        metrics['actor/optimizer_steps_per_update'] = [float(optimizer_steps)]
        metrics['actor/minibatches_per_epoch'] = [float(optimizer_steps / ppo_epochs)]
        if batch_readback is not None:
            metrics['actor/normalized_mini_batch_rows'] = [
                float(batch_readback['normalized_mini_batch_rows'])
            ]
            metrics['actor/dynamic_bsz'] = [
                float(batch_readback['dynamic_bsz'])
            ]
            if 'per_gpu_micro_batch_rows' in batch_readback:
                metrics['actor/per_gpu_micro_batch_rows'] = [
                    float(batch_readback['per_gpu_micro_batch_rows'])
                ]
        if dynamic_summaries:
            summary = {
                'micro_batches': sum(
                    item['micro_batches'] for item in dynamic_summaries
                ),
                'token_load_min': min(
                    item['token_load_min'] for item in dynamic_summaries
                ),
                'token_load_max': max(
                    item['token_load_max'] for item in dynamic_summaries
                ),
                'token_load_total': sum(
                    item['token_load_total'] for item in dynamic_summaries
                ),
                'rows_min': min(item['rows_min'] for item in dynamic_summaries),
                'rows_max': max(item['rows_max'] for item in dynamic_summaries),
                'rank': (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else 0
                ),
                'role': 'actor',
            }
            summary['token_load_mean'] = (
                summary['token_load_total'] / summary['micro_batches']
            )
            print(
                "AgentMemory PPO dynamic-batch readback: "
                + json.dumps(summary, sort_keys=True)
            )
            for name, value in summary.items():
                if name not in {'rank', 'role'}:
                    metrics[f'actor/dynamic_{name}'] = [float(value)]
        metrics.update(token_metrics.reduce(group=metric_group))
        return metrics
