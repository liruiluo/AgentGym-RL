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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Type, Dict
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.agent_trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.agent_dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.agent_dataset.procedural_index import (
    PROCEDURAL_STREAM_CHECKPOINT_SCHEMA,
    PROVIDER_MODE_RESEEDED_STREAM,
    StatefulProceduralStreamSampler,
    build_stream_checkpoint,
    generation_non_tensor_keys,
    promote_data_idx_for_rollout,
    restore_stream_checkpoint,
    validate_orbit_batch_indices,
    validate_rollout_parent_coverage,
)
from verl.utils.agentgym.rollout_context import (
    AGENTMEMORY_ACTION_TEXT,
    AGENTMEMORY_EXACT_STATE_UID,
    AGENTMEMORY_GENERATION_PROMPT_DIGEST,
    AGENTMEMORY_GENERATION_PROMPT_LENGTH,
    AGENTMEMORY_IMMEDIATE_REWARD,
    AGENTMEMORY_PACKED_PROMPT_DIGEST,
    AGENTMEMORY_PACKED_PROMPT_LENGTH,
    AGENTMEMORY_PARENT_GROUP_UID,
    AGENTMEMORY_REPLICA_INDEX,
    AGENTMEMORY_STEP_RECORD_JSON,
    AGENTMEMORY_SUFFIX_CREDIT_APPLIED,
    AGENTMEMORY_SUFFIX_RETURN,
    AGENTMEMORY_TRAJECTORY_RETURN,
    AGENTMEMORY_TRAJECTORY_ROW_ORDER,
    AGENTMEMORY_TRAJECTORY_ROW_UID,
    AGENTMEMORY_TRAJECTORY_TERMINAL,
    AGENTMEMORY_TRAJECTORY_UID,
    align_batch_to_rollout,
    requires_formal_trajectory_metadata,
    summarize_update_readback,
    validate_formal_trajectory_metadata,
    validate_state_aware_rollout_uids,
)
from verl.utils.agentgym.formal_training_metrics import (
    summarize_formal_training_rows,
)
from verl.workers.ppo_token_normalization import (
    PPO_BATCH_CONTRACT_META_KEY,
    build_legacy_asymmetric_batch_contract,
    optimizer_step_readback,
    requires_flattened_action_row_batch_contract,
    validate_dynamic_batch_token_caps,
)
from abc import ABC, abstractmethod

WorkerType = Type[Worker]


def _agentmemory_atomic_json_dump(payload: dict, output_path: str) -> None:
    """Publish a complete JSON artifact without exposing a partial file."""

    output_path = os.fspath(output_path)
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    temporary_path = (
        f"{output_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        os.replace(temporary_path, output_path)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def find_latest_ckpt_path_aistudio(path, directory_format="global_step_{}"):
    if path is None:
        return None

    from verl.utils.checkpoint.checkpoint_manager import get_checkpoint_tracker_filename
    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        print("Checkpoint tracker file does not exist: %s", tracker_file)
        return None

    from aistudio_checkpoint.aistudio_base_checkpointer import load_checkpoint
    with open(tracker_file, "r") as f:
        iteration, resuming_path = f.read().split("\n")
    ckpt_path = os.path.join(load_checkpoint(resuming_path=resuming_path), directory_format.format(iteration))
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None

    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    token_level_scores = data.batch['token_level_scores']
    response_mask = data.batch['response_mask']

    # compute kl between ref_policy and current policy
    kl_measured = 'ref_log_prob' in data.batch.keys()
    if kl_measured:
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    valid_samples = _get_ppo_valid_sample_mask(data)
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl[valid_samples], dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=int(valid_samples.sum().item()))
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {
        'critic/kl': current_kl,
        'critic/kl_coeff': beta,
        'critic/kl_measured': float(kl_measured),
    }

    return data, metrics


def _get_ppo_valid_sample_mask(data: DataProto) -> torch.Tensor:
    mask = data.batch.get(core_algos.PPO_VALID_SAMPLE_MASK)
    if mask is None:
        return torch.ones(len(data), dtype=torch.bool, device=data.batch.device)
    if mask.ndim != 1 or mask.shape[0] != len(data):
        raise ValueError(
            f"{core_algos.PPO_VALID_SAMPLE_MASK} must have shape ({len(data)},), got {tuple(mask.shape)}"
        )
    mask = mask.to(dtype=torch.bool)
    if not torch.any(mask):
        raise ValueError("PPO batch must contain at least one non-padding sample.")
    return mask


def _mask_ppo_padding_samples(data: DataProto) -> None:
    valid_samples = _get_ppo_valid_sample_mask(data)
    response_mask = data.batch['response_mask']
    data.batch['response_mask'] = response_mask * valid_samples.unsqueeze(-1).to(response_mask.dtype)


def _actor_positive_credit_eligibility(data: DataProto) -> torch.Tensor:
    raw_records = data.non_tensor_batch.get(AGENTMEMORY_STEP_RECORD_JSON)
    if raw_records is None or len(raw_records) != len(data):
        raise RuntimeError(
            "Positive actor-credit masking requires one step record per action row."
        )
    valid_samples = _get_ppo_valid_sample_mask(data).detach().cpu().tolist()
    eligibility: list[bool] = []
    for row_index, (raw_record, valid) in enumerate(zip(raw_records, valid_samples)):
        if not valid:
            eligibility.append(False)
            continue
        if not isinstance(raw_record, str):
            raise RuntimeError(
                f"Actor-credit step record {row_index} must be JSON text."
            )
        try:
            record = json.loads(raw_record)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Actor-credit step record {row_index} is invalid JSON."
            ) from exc
        wrapper_evidence = record.get("wrapper_evidence")
        receipt = (
            wrapper_evidence.get("actor_credit")
            if isinstance(wrapper_evidence, dict)
            else None
        )
        if not isinstance(receipt, dict):
            raise RuntimeError(
                f"Actor-credit step record {row_index} is missing its receipt."
            )
        if receipt.get("schema") != "task_neutral_actor_credit_v1":
            raise RuntimeError(
                f"Actor-credit step record {row_index} has a schema mismatch."
            )
        positive_eligible = receipt.get("positive_eligible")
        if type(positive_eligible) is not bool:
            raise RuntimeError(
                f"Actor-credit step record {row_index} eligibility must be boolean."
            )
        basis = receipt.get("basis")
        if not isinstance(basis, str) or not basis:
            raise RuntimeError(
                f"Actor-credit step record {row_index} basis must be non-empty text."
            )
        eligibility.append(positive_eligible)
    return torch.tensor(
        eligibility,
        dtype=torch.bool,
        device=data.batch["response_mask"].device,
    )


def _mask_ineligible_positive_actor_advantages(
    data: DataProto,
    advantages: torch.Tensor,
) -> torch.Tensor:
    eligibility = _actor_positive_credit_eligibility(data)
    response_mask = data.batch["response_mask"].to(dtype=torch.bool)
    positive_tokens = advantages > 0
    mask = (~eligibility).unsqueeze(-1) & response_mask & positive_tokens
    masked_rows = torch.any(mask, dim=-1)
    data.meta_info["agentmemory_positive_credit_masked_rows"] = int(
        masked_rows.sum().item()
    )
    data.meta_info["agentmemory_positive_credit_masked_tokens"] = int(
        mask.sum().item()
    )
    data.meta_info["agentmemory_positive_actor_credit_receipt_enabled"] = True
    return torch.where(mask, torch.zeros_like(advantages), advantages)


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        formal_required = requires_formal_trajectory_metadata(data)
        runtime_evidence_required = _agentmemory_env_flag(
            "AGENTMEMORY_REQUIRE_FORMAL_RUNTIME_EVIDENCE"
        )
        formal_groups = validate_formal_trajectory_metadata(
            data,
            expected_replicas=int(num_repeat),
            require=formal_required or runtime_evidence_required,
            require_runtime_evidence=runtime_evidence_required,
            expected_suffix_credit=(
                False if formal_required or runtime_evidence_required else None
            ),
        )
        positive_credit_receipt = _agentmemory_env_flag(
            "AGENTMEMORY_POSITIVE_ACTOR_CREDIT_RECEIPT"
        )
        if positive_credit_receipt and formal_groups is None:
            raise RuntimeError(
                "Positive actor-credit masking requires formal trajectory GAE."
            )
        values = data.batch['values']
        response_mask = data.batch['response_mask']
        token_level_rewards = data.batch['token_level_rewards']
        if formal_groups is None:
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=token_level_rewards,
                values=values,
                eos_mask=response_mask,
                gamma=gamma,
                lam=lam,
            )
        else:
            done_flags = data.non_tensor_batch.get("rollout_done_flags")
            if done_flags is None:
                raise RuntimeError(
                    "Formal trajectory GAE requires one environment done flag per action row."
                )
            advantages, returns = core_algos.compute_trajectory_gae_advantage_return(
                token_level_rewards=token_level_rewards,
                values=values,
                eos_mask=response_mask,
                trajectory_uids=data.non_tensor_batch[AGENTMEMORY_TRAJECTORY_UID],
                trajectory_row_uids=data.non_tensor_batch[
                    AGENTMEMORY_TRAJECTORY_ROW_UID
                ],
                trajectory_row_orders=data.batch[
                    AGENTMEMORY_TRAJECTORY_ROW_ORDER
                ],
                trajectory_terminals=data.batch[AGENTMEMORY_TRAJECTORY_TERMINAL],
                done_flags=done_flags,
                sample_mask=_get_ppo_valid_sample_mask(data),
                gamma=gamma,
                lam=lam,
                immediate_rewards=data.batch[AGENTMEMORY_IMMEDIATE_REWARD],
                advantage_normalization="none",
            )
            data.meta_info[
                "agentmemory_actor_advantage_mode"
            ] = "standard_trajectory_gae"
            if positive_credit_receipt:
                advantages = _mask_ineligible_positive_actor_advantages(
                    data,
                    advantages,
                )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        response_mask = data.batch['response_mask']
        sample_mask = _get_ppo_valid_sample_mask(data)
        formal_required = requires_formal_trajectory_metadata(data)
        runtime_evidence_required = _agentmemory_env_flag(
            "AGENTMEMORY_REQUIRE_FORMAL_RUNTIME_EVIDENCE"
        )
        formal_groups = validate_formal_trajectory_metadata(
            data,
            expected_replicas=int(num_repeat),
            require=formal_required or runtime_evidence_required,
            require_runtime_evidence=runtime_evidence_required,
            expected_suffix_credit=False if runtime_evidence_required else None,
        )
        if formal_groups is not None:
            advantages, returns = core_algos.compute_formal_grpo_complete_trajectory_advantage(
                immediate_rewards=data.batch[AGENTMEMORY_IMMEDIATE_REWARD],
                eos_mask=response_mask,
                parent_group_uids=data.non_tensor_batch[
                    AGENTMEMORY_PARENT_GROUP_UID
                ],
                trajectory_uids=data.non_tensor_batch[AGENTMEMORY_TRAJECTORY_UID],
                trajectory_row_uids=data.non_tensor_batch[
                    AGENTMEMORY_TRAJECTORY_ROW_UID
                ],
                trajectory_row_orders=data.batch[
                    AGENTMEMORY_TRAJECTORY_ROW_ORDER
                ],
                trajectory_terminals=data.batch[AGENTMEMORY_TRAJECTORY_TERMINAL],
                declared_trajectory_returns=data.batch[
                    AGENTMEMORY_TRAJECTORY_RETURN
                ],
                sample_mask=sample_mask,
                expected_group_size=int(num_repeat),
                gamma=gamma,
            )
        else:
            token_level_rewards = data.batch['token_level_rewards']
            index = data.non_tensor_batch['uid']
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                eos_mask=response_mask,
                index=index,
                sample_mask=sample_mask,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'rloo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']
        sample_mask = _get_ppo_valid_sample_mask(data)
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        sample_mask=sample_mask)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        response_mask = data.batch['response_mask']
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data



def _agentmemory_env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _validate_formal_actor_advantage_config(config) -> str:
    mode = os.environ.get(
        "AGENTMEMORY_FORMAL_ACTOR_ADVANTAGE_MODE",
        "standard_trajectory_gae",
    ).strip().lower()
    if not _agentmemory_env_flag("AGENTMEMORY_REQUIRE_FORMAL_RUNTIME_EVIDENCE"):
        return mode
    if mode != "standard_trajectory_gae":
        raise RuntimeError(
            "Formal AgentMemory PPO requires standard_trajectory_gae; legacy "
            f"suffix/Monte-Carlo actor modes are disabled, got {mode!r}."
        )
    if _agentmemory_env_flag("AGENTMEMORY_LATEST_OBS_SUFFIX_CREDIT"):
        raise RuntimeError(
            "Formal AgentMemory PPO requires AGENTMEMORY_LATEST_OBS_SUFFIX_CREDIT=0; "
            "future action rewards are propagated by critic GAE."
        )
    if float(config.algorithm.kl_ctrl.kl_coef) != 0.0:
        raise RuntimeError(
            "Formal trajectory GAE requires algorithm.kl_ctrl.kl_coef=0 so the "
            "packed reward remains the exact environment action reward."
        )
    return mode


def _agentmemory_formal_update_readback_target_steps() -> frozenset[int] | None:
    if not _agentmemory_env_flag("AGENTMEMORY_FORMAL_UPDATE_READBACK"):
        return None
    raw_value = os.environ.get("AGENTMEMORY_FORMAL_UPDATE_READBACK_STEP")
    if raw_value is None or not raw_value.strip():
        raise RuntimeError(
            "AGENTMEMORY_FORMAL_UPDATE_READBACK requires an explicit "
            "AGENTMEMORY_FORMAL_UPDATE_READBACK_STEP."
        )
    raw_steps = raw_value.split(",")
    if any(not raw_step.strip() for raw_step in raw_steps):
        raise RuntimeError(
            "AGENTMEMORY_FORMAL_UPDATE_READBACK_STEP contains an empty target, "
            f"got {raw_value!r}."
        )
    try:
        target_steps = tuple(int(raw_step.strip()) for raw_step in raw_steps)
    except ValueError as exc:
        raise RuntimeError(
            "AGENTMEMORY_FORMAL_UPDATE_READBACK_STEP must be a comma-separated "
            f"set of positive integers, got {raw_value!r}."
        ) from exc
    if any(target_step <= 0 for target_step in target_steps):
        raise RuntimeError(
            "AGENTMEMORY_FORMAL_UPDATE_READBACK_STEP targets must be positive, "
            f"got {target_steps}."
        )
    if len(set(target_steps)) != len(target_steps):
        raise RuntimeError(
            "AGENTMEMORY_FORMAL_UPDATE_READBACK_STEP targets must be unique, "
            f"got {target_steps}."
        )
    return frozenset(target_steps)


def _agentmemory_missing_formal_update_readback_steps(
    target_steps: frozenset[int] | None,
    observed_steps: set[int],
) -> list[int]:
    return sorted((target_steps or frozenset()) - observed_steps)


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _agentmemory_dump_ppo_batch_debug(batch: DataProto, config, global_steps: int, stage: str) -> None:
    if not _agentmemory_env_flag("AGENTMEMORY_BATCH_DEBUG"):
        return
    try:
        action_text_key = "agentmemory_action_text"
        generation_prompt_digest_key = "agentmemory_generation_prompt_digest"
        generation_prompt_length_key = "agentmemory_generation_prompt_length"
        packed_prompt_digest_key = "agentmemory_packed_prompt_digest"
        packed_prompt_length_key = "agentmemory_packed_prompt_length"
        suffix_credit_applied_key = "agentmemory_suffix_credit_applied"
        suffix_return_key = "agentmemory_suffix_return"
        default_dir = str(config.trainer.default_local_dir)
        run_dir = os.path.dirname(default_dir.rstrip("/")) if default_dir else os.getcwd()
        out_dir = os.path.join(run_dir, "diagnostics")
        os.makedirs(out_dir, exist_ok=True)

        response_mask = batch.batch.get("response_mask", None)
        scores = batch.batch.get("scores", None)
        token_rewards = batch.batch.get("token_level_rewards", None)
        advantages = batch.batch.get("advantages", None)
        returns = batch.batch.get("returns", None)
        old_log_probs = batch.batch.get("old_log_probs", None)
        task_rounds = batch.batch.get("task_rounds", None)
        valid_samples = batch.batch.get(core_algos.PPO_VALID_SAMPLE_MASK)
        if valid_samples is None:
            valid_samples = torch.ones(len(batch), dtype=torch.bool)
        else:
            valid_samples = valid_samples.to(dtype=torch.bool).detach().cpu()

        rows = []
        uid_arr = batch.non_tensor_batch.get("uid")
        parent_arr = batch.non_tensor_batch.get("rollout_parent_indices")
        parent_group_arr = batch.non_tensor_batch.get(
            AGENTMEMORY_PARENT_GROUP_UID
        )
        exact_state_arr = batch.non_tensor_batch.get(AGENTMEMORY_EXACT_STATE_UID)
        replica_arr = batch.non_tensor_batch.get(AGENTMEMORY_REPLICA_INDEX)
        trajectory_arr = batch.non_tensor_batch.get(AGENTMEMORY_TRAJECTORY_UID)
        trajectory_row_uid_arr = batch.non_tensor_batch.get(
            AGENTMEMORY_TRAJECTORY_ROW_UID
        )
        trajectory_returns = batch.batch.get(AGENTMEMORY_TRAJECTORY_RETURN)
        trajectory_row_orders = batch.batch.get(AGENTMEMORY_TRAJECTORY_ROW_ORDER)
        trajectory_terminals = batch.batch.get(AGENTMEMORY_TRAJECTORY_TERMINAL)
        immediate_rewards = batch.batch.get(AGENTMEMORY_IMMEDIATE_REWARD)
        suffix_returns = batch.batch.get(suffix_return_key)
        suffix_flags = batch.batch.get(suffix_credit_applied_key)
        generation_prompt_lengths = batch.batch.get(
            generation_prompt_length_key
        )
        packed_prompt_lengths = batch.batch.get(packed_prompt_length_key)
        generation_prompt_digests = batch.non_tensor_batch.get(
            generation_prompt_digest_key
        )
        packed_prompt_digests = batch.non_tensor_batch.get(
            packed_prompt_digest_key
        )
        action_arr = batch.non_tensor_batch.get(action_text_key)
        step_record_arr = batch.non_tensor_batch.get(
            "agentmemory_step_record_json"
        )
        for i in range(len(batch)):
            row = {
                "i": i,
                "ppo_valid_sample": bool(valid_samples[i].item()),
            }
            if uid_arr is not None:
                row["uid"] = str(uid_arr[i])
            if parent_arr is not None:
                row["parent_index"] = int(parent_arr[i])
            if parent_group_arr is not None:
                row[AGENTMEMORY_PARENT_GROUP_UID] = str(parent_group_arr[i])
            if exact_state_arr is not None:
                row[AGENTMEMORY_EXACT_STATE_UID] = str(exact_state_arr[i])
            if replica_arr is not None:
                row[AGENTMEMORY_REPLICA_INDEX] = int(replica_arr[i])
            if trajectory_arr is not None:
                row[AGENTMEMORY_TRAJECTORY_UID] = str(trajectory_arr[i])
            if trajectory_row_uid_arr is not None:
                row[AGENTMEMORY_TRAJECTORY_ROW_UID] = str(
                    trajectory_row_uid_arr[i]
                )
            if trajectory_row_orders is not None:
                row[AGENTMEMORY_TRAJECTORY_ROW_ORDER] = int(
                    trajectory_row_orders[i].item()
                )
            if trajectory_terminals is not None:
                row[AGENTMEMORY_TRAJECTORY_TERMINAL] = bool(
                    trajectory_terminals[i].item()
                )
            if trajectory_returns is not None:
                row[AGENTMEMORY_TRAJECTORY_RETURN] = _safe_float(
                    trajectory_returns[i].item()
                )
            if immediate_rewards is not None:
                row[AGENTMEMORY_IMMEDIATE_REWARD] = _safe_float(
                    immediate_rewards[i].item()
                )
            if suffix_returns is not None:
                row[suffix_return_key] = _safe_float(
                    suffix_returns[i].item()
                )
            if suffix_flags is not None:
                row[suffix_credit_applied_key] = bool(
                    suffix_flags[i].item()
                )
            if task_rounds is not None:
                row["task_round"] = int(task_rounds[i].item())
            if generation_prompt_lengths is not None:
                row[generation_prompt_length_key] = int(
                    generation_prompt_lengths[i].item()
                )
            if packed_prompt_lengths is not None:
                row[packed_prompt_length_key] = int(
                    packed_prompt_lengths[i].item()
                )
            if generation_prompt_digests is not None:
                row[generation_prompt_digest_key] = str(
                    generation_prompt_digests[i]
                )
            if packed_prompt_digests is not None:
                row[packed_prompt_digest_key] = str(
                    packed_prompt_digests[i]
                )
            if action_arr is not None:
                row[action_text_key] = str(action_arr[i])
            if step_record_arr is not None:
                row["formal_step_record"] = json.loads(str(step_record_arr[i]))
            if response_mask is not None:
                row["response_mask_sum"] = _safe_float(response_mask[i].sum().item())
            if scores is not None:
                row["score_sum"] = _safe_float(scores[i].sum().item())
            if token_rewards is not None:
                row["token_reward_sum"] = _safe_float(token_rewards[i].sum().item())
            if advantages is not None:
                vals = advantages[i][response_mask[i].bool()] if response_mask is not None else advantages[i].reshape(-1)
                if vals.numel():
                    row["adv_min"] = _safe_float(vals.min().item())
                    row["adv_max"] = _safe_float(vals.max().item())
                    row["adv_mean"] = _safe_float(vals.mean().item())
                    row["adv_nonzero"] = int((vals != 0).sum().item())
            if returns is not None:
                vals = returns[i][response_mask[i].bool()] if response_mask is not None else returns[i].reshape(-1)
                if vals.numel():
                    row["return_min"] = _safe_float(vals.min().item())
                    row["return_max"] = _safe_float(vals.max().item())
                    row["return_mean"] = _safe_float(vals.mean().item())
                    row["return_nonzero"] = int((vals != 0).sum().item())
            if old_log_probs is not None:
                vals = old_log_probs[i]
                vals = vals[response_mask[i].bool()] if response_mask is not None else vals.reshape(-1)
                if vals.numel():
                    row["old_logprob_mean"] = _safe_float(vals.mean().item())
            rows.append(row)

        uid_counts = {}
        if uid_arr is not None:
            for i, uid in enumerate(uid_arr):
                if not bool(valid_samples[i].item()):
                    continue
                uid = str(uid)
                uid_counts[uid] = uid_counts.get(uid, 0) + 1
        valid_rows = int(valid_samples.sum().item())
        parent_group_summaries = []
        if (
            parent_group_arr is not None
            and replica_arr is not None
            and trajectory_arr is not None
            and trajectory_returns is not None
            and trajectory_row_orders is not None
            and trajectory_terminals is not None
            and advantages is not None
        ):
            grouped_rows = {}
            for i in range(len(batch)):
                if not bool(valid_samples[i].item()):
                    continue
                parent_group_uid = str(parent_group_arr[i])
                trajectory_uid = str(trajectory_arr[i])
                response_values = (
                    advantages[i][response_mask[i].bool()]
                    if response_mask is not None
                    else advantages[i].reshape(-1)
                )
                row_token_mean_advantage = (
                    _safe_float(response_values.mean().item())
                    if response_values.numel()
                    else None
                )
                trajectory = grouped_rows.setdefault(parent_group_uid, {}).setdefault(
                    trajectory_uid,
                    {
                        AGENTMEMORY_TRAJECTORY_UID: trajectory_uid,
                        AGENTMEMORY_REPLICA_INDEX: int(replica_arr[i]),
                        AGENTMEMORY_TRAJECTORY_RETURN: _safe_float(
                            trajectory_returns[i].item()
                        ),
                        "row_count": 0,
                        "action_row_advantages": [],
                    },
                )
                trajectory["row_count"] += 1
                if row_token_mean_advantage is not None:
                    trajectory["action_row_advantages"].append(
                        {
                            "row_order": int(trajectory_row_orders[i].item()),
                            "terminal": bool(trajectory_terminals[i].item()),
                            "token_mean_advantage": row_token_mean_advantage,
                        }
                    )
            for parent_group_uid in sorted(grouped_rows):
                trajectories = []
                replica_indices = set()
                for trajectory_uid in sorted(grouped_rows[parent_group_uid]):
                    trajectory = grouped_rows[parent_group_uid][trajectory_uid]
                    action_rows = sorted(
                        trajectory.pop("action_row_advantages"),
                        key=lambda item: item["row_order"],
                    )
                    values = [item["token_mean_advantage"] for item in action_rows]
                    terminal_rows = [item for item in action_rows if item["terminal"]]
                    trajectory["first_action_row_token_mean_advantage"] = (
                        values[0] if values else None
                    )
                    trajectory["terminal_action_row_token_mean_advantage"] = (
                        terminal_rows[0]["token_mean_advantage"]
                        if len(terminal_rows) == 1
                        else None
                    )
                    trajectory["action_row_token_mean_advantage_min"] = (
                        min(values) if values else None
                    )
                    trajectory["action_row_token_mean_advantage_max"] = (
                        max(values) if values else None
                    )
                    replica_indices.add(trajectory[AGENTMEMORY_REPLICA_INDEX])
                    trajectories.append(trajectory)
                parent_group_summaries.append(
                    {
                        AGENTMEMORY_PARENT_GROUP_UID: parent_group_uid,
                        "unique_replicas": len(replica_indices),
                        "replica_indices": sorted(replica_indices),
                        "trajectories": trajectories,
                    }
                )
        summary = {
            "global_step": int(global_steps),
            "stage": stage,
            "batch_size": len(batch),
            "valid_rows": valid_rows,
            "padding_rows": len(batch) - valid_rows,
            "uid_group_sizes": sorted(uid_counts.values()),
            "agentmemory_parent_groups": parent_group_summaries,
            "suffix_credit_enabled": _agentmemory_env_flag(
                "AGENTMEMORY_LATEST_OBS_SUFFIX_CREDIT"
            ),
            "actor_advantage_mode": batch.meta_info.get(
                "agentmemory_actor_advantage_mode",
                "standard_trajectory_gae",
            ),
            "positive_actor_credit_receipt_enabled": bool(
                batch.meta_info.get(
                    "agentmemory_positive_actor_credit_receipt_enabled",
                    False,
                )
            ),
            "positive_credit_masked_rows": int(
                batch.meta_info.get("agentmemory_positive_credit_masked_rows", 0)
            ),
            "positive_credit_masked_tokens": int(
                batch.meta_info.get("agentmemory_positive_credit_masked_tokens", 0)
            ),
            "prompt_attestation_passed": bool(
                generation_prompt_lengths is not None
                and packed_prompt_lengths is not None
            ),
            "generation_prompt_length_max": (
                int(
                    generation_prompt_lengths.detach().cpu()[valid_samples]
                    .max()
                    .item()
                )
                if generation_prompt_lengths is not None
                else None
            ),
            "packed_prompt_length_max": (
                int(
                    packed_prompt_lengths.detach().cpu()[valid_samples]
                    .max()
                    .item()
                )
                if packed_prompt_lengths is not None
                else None
            ),
            "suffix_formula_mismatch_count": 0 if suffix_returns is not None else None,
            "rows": rows,
        }
        _agentmemory_atomic_json_dump(
            summary,
            os.path.join(out_dir, f"ppo_batch_step{global_steps}_{stage}.json"),
        )
    except Exception as exc:
        print(f"AgentMemory PPO batch debug dump failed: {exc}", flush=True)


def _masked_row_values(
    tensor: torch.Tensor,
    response_mask: torch.Tensor,
    valid_samples: torch.Tensor,
    *,
    reduction: str,
) -> list[float]:
    values = []
    for row_index in range(tensor.shape[0]):
        if not bool(valid_samples[row_index].item()):
            continue
        row = tensor[row_index][response_mask[row_index].bool()]
        if not row.numel():
            raise RuntimeError(
                f"Formal update readback row {row_index} has no response tokens."
            )
        if reduction == "sum":
            value = row.sum()
        elif reduction == "mean":
            value = row.mean()
        else:
            raise ValueError(f"Unsupported readback reduction: {reduction}")
        values.append(float(value.detach().cpu().item()))
    return values


_PARAMETER_PROBE_FIELDS = (
    "parameter_delta_l2",
    "parameter_probe_max_abs_delta",
    "parameter_probe_l1_delta",
    "parameter_probe_changed_count",
    "parameter_probe_element_count",
    "parameter_probe_total_parameter_count",
    "parameter_probe_max_elements_per_rank",
)


def _parameter_probe_from_update_metrics(metrics: dict, *, label: str) -> dict:
    summary = {}
    for field in _PARAMETER_PROBE_FIELDS:
        metric_name = f"{label}/{field}"
        if metric_name not in metrics:
            raise RuntimeError(
                f"Formal {label} update readback is missing {metric_name}."
            )
        value = float(metrics[metric_name])
        if not np.isfinite(value):
            raise RuntimeError(
                f"Formal {label} update readback has non-finite {metric_name}."
            )
        summary[field] = value
    for field in (
        "parameter_probe_changed_count",
        "parameter_probe_element_count",
        "parameter_probe_total_parameter_count",
        "parameter_probe_max_elements_per_rank",
    ):
        summary[field] = int(round(summary[field]))
    if (
        summary["parameter_delta_l2"] <= 0.0
        or summary["parameter_probe_max_abs_delta"] <= 0.0
        or summary["parameter_probe_l1_delta"] <= 0.0
        or summary["parameter_probe_changed_count"] <= 0
        or summary["parameter_probe_element_count"] <= 0
    ):
        raise RuntimeError(
            f"Formal {label} update readback found zero parameter delta."
        )
    summary["parameter_probe_finite"] = True
    summary["parameter_probe_sampling"] = (
        "evenly_spaced_local_trainable_shards_then_fsdp_all_reduce"
    )
    return summary


def _formal_update_readback_row_evidence(
    *,
    non_tensor_batch: dict,
    valid_row_indices: list[int],
    task_name: str,
    response_token_rows,
    response_mask_rows,
    old_logprob_rows,
) -> dict:
    """Return canonical row identities without imposing WebShop metadata on other tasks."""

    normalized_task_name = str(task_name).strip().lower()
    if not normalized_task_name:
        raise RuntimeError("Formal PPO update readback requires a task name.")

    step_record_arr = non_tensor_batch.get("agentmemory_step_record_json")
    if step_record_arr is not None:
        if (
            response_token_rows is None
            or response_mask_rows is None
            or old_logprob_rows is None
        ):
            raise RuntimeError(
                "Formal canonical step records require exact token/logprob tensor evidence."
            )
        tensor_row_count = len(response_token_rows)
        if (
            len(response_mask_rows) != tensor_row_count
            or len(old_logprob_rows) != tensor_row_count
            or len(step_record_arr) != tensor_row_count
        ):
            raise RuntimeError(
                "Formal canonical step records and token/logprob tensors are not row-aligned."
            )

        def canonical_sha256(value) -> str:
            raw = json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            return hashlib.sha256(raw).hexdigest()

        rows = []
        for row_index in valid_row_indices:
            record = json.loads(str(step_record_arr[row_index]))
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Formal canonical step record {row_index} is not an object."
                )
            if not record.get("generation_token_ids_are_exact") or not record.get(
                "backend_token_ids_are_exact"
            ):
                raise RuntimeError(
                    f"Formal canonical step record {row_index} lacks exact token identity."
                )
            try:
                packed_tokens = [int(value) for value in response_token_rows[row_index]]
                packed_mask = [int(value) for value in response_mask_rows[row_index]]
                packed_logprobs = [
                    float(value) for value in old_logprob_rows[row_index]
                ]
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"Formal token/logprob tensor evidence is non-numeric at row {row_index}."
                ) from exc
            if not (
                len(packed_tokens) == len(packed_mask) == len(packed_logprobs)
            ):
                raise RuntimeError(
                    f"Formal token/logprob tensor widths differ at row {row_index}."
                )
            if any(value not in (0, 1) for value in packed_mask):
                raise RuntimeError(
                    f"Formal response mask is not binary at row {row_index}."
                )
            if not all(math.isfinite(value) for value in packed_logprobs):
                raise RuntimeError(
                    f"Formal old logprobs are non-finite at row {row_index}."
                )
            sampled_tokens = [
                token
                for token, visible in zip(packed_tokens, packed_mask)
                if visible == 1
            ]
            sampled_logprobs = [
                value
                for value, visible in zip(packed_logprobs, packed_mask)
                if visible == 1
            ]
            if not sampled_tokens:
                raise RuntimeError(
                    f"Formal canonical step record {row_index} has no sampled tokens."
                )
            raw_response_tokens = record.get("response_token_ids")
            if raw_response_tokens != sampled_tokens:
                raise RuntimeError(
                    "Formal sampled response tokens differ from the canonical step "
                    f"record at row {row_index}."
                )
            if record.get("response_token_count") != len(sampled_tokens):
                raise RuntimeError(
                    f"Formal response token count differs at row {row_index}."
                )

            record.update(
                {
                    "sampled_response_token_ids": sampled_tokens,
                    "packed_token_ids": packed_tokens,
                    "response_mask": packed_mask,
                    "sampled_old_logprobs": sampled_logprobs,
                    "packed_old_logprobs": packed_logprobs,
                    "sampled_response_token_ids_sha256": canonical_sha256(
                        sampled_tokens
                    ),
                    "packed_token_ids_sha256": canonical_sha256(packed_tokens),
                    "response_mask_sha256": canonical_sha256(packed_mask),
                    "sampled_old_logprobs_sha256": canonical_sha256(
                        sampled_logprobs
                    ),
                }
            )
            rows.append(record)
        return {
            "schema": "agentmemory_formal_step_records_v1",
            "task_name": normalized_task_name,
            "rows": rows,
        }
    if normalized_task_name == "agentmemory":
        raise RuntimeError(
            "Formal AgentMemory PPO update readback is missing canonical step records."
        )

    index_field = next(
        (
            field
            for field in ("rollout_data_indices", "data_idx", "index")
            if field in non_tensor_batch
        ),
        None,
    )
    if index_field is None:
        raise RuntimeError(
            "Formal non-AgentMemory PPO update readback is missing a canonical "
            "dataset index field."
        )
    index_values = non_tensor_batch[index_field]
    dataset_indices = []
    for row_index in valid_row_indices:
        value = index_values[row_index]
        if isinstance(value, bool):
            raise RuntimeError(
                "Formal PPO update readback dataset indices must not be bool."
            )
        try:
            normalized_index = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "Formal PPO update readback found a non-integer dataset index: "
                f"row={row_index} value={value!r}."
            ) from exc
        if normalized_index < 0:
            raise RuntimeError(
                "Formal PPO update readback dataset indices must be non-negative: "
                f"row={row_index} value={normalized_index}."
            )
        dataset_indices.append(normalized_index)
    return {
        "schema": "generic_task_dataset_rows_v1",
        "task_name": normalized_task_name,
        "index_field": index_field,
        "dataset_indices": dataset_indices,
    }


def _agentmemory_dump_formal_update_readback(
    *,
    batch: DataProto,
    post_actor_log_probs: DataProto,
    post_critic_values: DataProto | None,
    actor_update_metrics: dict,
    critic_update_metrics: dict,
    config,
    global_steps: int,
) -> dict:
    """Fail closed unless one PPO update changes actor and critic outputs."""

    if not _agentmemory_env_flag("AGENTMEMORY_FORMAL_UPDATE_READBACK"):
        return {}
    if post_critic_values is None:
        raise RuntimeError(
            "Formal PPO update readback requires a post-update critic forward pass."
        )
    response_mask = batch.batch["response_mask"]
    valid_samples = batch.batch.get(core_algos.PPO_VALID_SAMPLE_MASK)
    if valid_samples is None:
        valid_samples = torch.ones(
            len(batch), dtype=torch.bool, device=response_mask.device
        )
    else:
        valid_samples = valid_samples.to(
            dtype=torch.bool, device=response_mask.device
        )
    valid_row_indices = [
        row_index
        for row_index in range(len(batch))
        if bool(valid_samples[row_index].item())
    ]
    task_name = str(
        config.actor_rollout_ref.agentgym.get("task_name", "")
    ).strip().lower()
    row_evidence = _formal_update_readback_row_evidence(
        non_tensor_batch=batch.non_tensor_batch,
        valid_row_indices=valid_row_indices,
        task_name=task_name,
        response_token_rows=batch.batch["responses"].detach().cpu().tolist(),
        response_mask_rows=response_mask.detach().cpu().tolist(),
        old_logprob_rows=batch.batch["old_log_probs"].detach().cpu().tolist(),
    )
    actor_before = _masked_row_values(
        batch.batch["old_log_probs"],
        response_mask,
        valid_samples,
        reduction="sum",
    )
    actor_after = _masked_row_values(
        post_actor_log_probs.batch["old_log_probs"],
        response_mask,
        valid_samples,
        reduction="sum",
    )
    critic_before = _masked_row_values(
        batch.batch["values"],
        response_mask,
        valid_samples,
        reduction="mean",
    )
    critic_after = _masked_row_values(
        post_critic_values.batch["values"],
        response_mask,
        valid_samples,
        reduction="mean",
    )
    actor_summary = summarize_update_readback(
        before=actor_before,
        after=actor_after,
        label="actor_logprob",
    )
    critic_summary = summarize_update_readback(
        before=critic_before,
        after=critic_after,
        label="critic_value",
    )
    actor_parameter_probe = _parameter_probe_from_update_metrics(
        actor_update_metrics, label="actor"
    )
    critic_parameter_probe = _parameter_probe_from_update_metrics(
        critic_update_metrics, label="critic"
    )
    payload = {
        "global_step": int(global_steps),
        "role": "same_batch_post_optimizer_readback",
        "checkpoint_step_labels_are_not_used_as_update_evidence": True,
        "row_evidence": row_evidence,
        "actor": {
            "summary": actor_summary,
            "parameter_delta_l2": actor_parameter_probe["parameter_delta_l2"],
            "parameter_probe": actor_parameter_probe,
            "before_sequence_logprob": actor_before,
            "after_sequence_logprob": actor_after,
        },
        "critic": {
            "summary": critic_summary,
            "parameter_delta_l2": critic_parameter_probe["parameter_delta_l2"],
            "parameter_probe": critic_parameter_probe,
            "before_response_value_mean": critic_before,
            "after_response_value_mean": critic_after,
        },
    }
    if row_evidence["schema"] == "agentmemory_formal_step_records_v1":
        payload["formal_step_records"] = row_evidence["rows"]
    default_dir = str(config.trainer.default_local_dir)
    run_dir = os.path.dirname(default_dir.rstrip("/")) if default_dir else os.getcwd()
    out_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(
        out_dir, f"formal_update_readback_step{global_steps}.json"
    )
    _agentmemory_atomic_json_dump(payload, output_path)
    print(
        "AgentMemory formal PPO update readback: "
        f"actor_max_abs_delta={actor_summary['max_abs_delta']:.8g} "
        f"critic_max_abs_delta={critic_summary['max_abs_delta']:.8g} "
        f"actor_parameter_delta_l2={actor_parameter_probe['parameter_delta_l2']:.8g} "
        f"critic_parameter_delta_l2={critic_parameter_probe['parameter_delta_l2']:.8g} "
        f"path={output_path}",
        flush=True,
    )
    return payload


class RoundsScheduler(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_global_steps(self, global_steps: int):
        raise NotImplementedError

    @abstractmethod
    def get_rounds(self):
        raise NotImplementedError
    

class FixedRoundsScheduler(RoundsScheduler):
    def __init__(self, rounds: int):
        self.max_rounds = rounds

    def step(self):
        pass

    def set_global_steps(self, global_steps: int):
        pass

    def get_rounds(self):
        return self.max_rounds


class StepRoundsScheduler(RoundsScheduler):
    def __init__(self, steps_scaling_inter: int, rounds_ls: List[int]):
        self.rounds_ls = rounds_ls
        self.steps_scaling_inter = steps_scaling_inter
        self.max_rounds = rounds_ls[0]
        self.current_stage = 0
        self.global_steps = 1 # start from 1

    def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps
        if (self.global_steps // self.steps_scaling_inter < len(self.rounds_ls)):
            self.current_stage = self.global_steps // self.steps_scaling_inter
        else:
            self.current_stage = len(self.rounds_ls) - 1
        self.max_rounds = self.rounds_ls[self.current_stage]
    
    def step(self):
        if self.current_stage + 1 < len(self.rounds_ls) and self.global_steps % self.steps_scaling_inter == 0:
            self.current_stage += 1
            self.max_rounds = self.rounds_ls[self.current_stage]
        self.global_steps += 1

    def get_rounds(self):
        return self.max_rounds


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    valid_samples = _get_ppo_valid_sample_mask(batch)
    sequence_score = batch.batch['token_level_scores'].sum(-1)[valid_samples]
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)[valid_samples]
    task_scores = batch.batch["task_scores"].sum(-1)[valid_samples]
    task_rounds = batch.batch["task_rounds"][valid_samples]

    response_length = batch.batch['response_mask'][valid_samples].sum(-1).float()
    prompt_length = batch.batch['attention_mask'][valid_samples].sum(-1).float() - response_length

    advantages = batch.batch['advantages'][valid_samples]
    returns = batch.batch['returns'][valid_samples]

    response_mask = batch.batch['response_mask'][valid_samples].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values'][valid_samples]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # task score
        'critic/task_score/mean':
            torch.mean(task_scores).detach().item(),
        'critic/task_score/max':
            torch.max(task_scores).detach().item(),
        'critic/task_score/min':
            torch.min(task_scores).detach().item(),
        # task round
        'critic/task_round/mean':
            torch.mean(task_rounds).detach().item(),
        'critic/task_round/max':
            torch.max(task_rounds).detach().item(),
        'critic/task_round/min':
            torch.min(task_rounds).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
    }
    if batch.meta_info.get("agentmemory_positive_actor_credit_receipt_enabled"):
        metrics["agentmemory/positive_credit_masked_rows"] = int(
            batch.meta_info.get("agentmemory_positive_credit_masked_rows", 0)
        )
        metrics["agentmemory/positive_credit_masked_tokens"] = int(
            batch.meta_info.get("agentmemory_positive_credit_masked_tokens", 0)
        )
    step_record_arr = batch.non_tensor_batch.get("agentmemory_step_record_json")
    trajectory_uid_arr = batch.non_tensor_batch.get(AGENTMEMORY_TRAJECTORY_UID)
    if step_record_arr is not None or trajectory_uid_arr is not None:
        if step_record_arr is None or trajectory_uid_arr is None:
            raise RuntimeError(
                "Formal AgentMemory metrics require both step records and trajectory UIDs."
            )
        immediate_rewards = batch.batch.get(AGENTMEMORY_IMMEDIATE_REWARD)
        suffix_returns = batch.batch.get(AGENTMEMORY_SUFFIX_RETURN)
        trajectory_returns = batch.batch.get(AGENTMEMORY_TRAJECTORY_RETURN)
        row_orders = batch.batch.get(AGENTMEMORY_TRAJECTORY_ROW_ORDER)
        terminal_flags = batch.batch.get(AGENTMEMORY_TRAJECTORY_TERMINAL)
        required_tensors = {
            AGENTMEMORY_IMMEDIATE_REWARD: immediate_rewards,
            AGENTMEMORY_SUFFIX_RETURN: suffix_returns,
            AGENTMEMORY_TRAJECTORY_RETURN: trajectory_returns,
            AGENTMEMORY_TRAJECTORY_ROW_ORDER: row_orders,
            AGENTMEMORY_TRAJECTORY_TERMINAL: terminal_flags,
        }
        missing = sorted(name for name, value in required_tensors.items() if value is None)
        if missing:
            raise RuntimeError(f"Formal AgentMemory metrics are missing tensors: {missing}.")
        formal_rows = []
        for row_index in range(len(batch)):
            if not bool(valid_samples[row_index].item()):
                continue
            mask = batch.batch["response_mask"][row_index].bool()
            row_advantages = batch.batch["advantages"][row_index][mask]
            if row_advantages.numel() == 0:
                raise RuntimeError(f"Formal AgentMemory row {row_index} has no response tokens.")
            formal_rows.append(
                {
                    "trajectory_uid": str(trajectory_uid_arr[row_index]),
                    "row_order": int(row_orders[row_index].item()),
                    "terminal": bool(terminal_flags[row_index].item()),
                    "immediate_reward": float(immediate_rewards[row_index].item()),
                    "suffix_return": float(suffix_returns[row_index].item()),
                    "trajectory_return": float(trajectory_returns[row_index].item()),
                    "advantage_token_mean": float(row_advantages.mean().item()),
                    "record": json.loads(str(step_record_arr[row_index])),
                }
            )
        formal_summary = summarize_formal_training_rows(formal_rows)
        for name, value in formal_summary.items():
            metric_name = name
            for suffix in ("_mean", "_count", "_rate"):
                if metric_name.endswith(suffix):
                    metric_name = metric_name[: -len(suffix)] + "/" + suffix[1:]
                    break
            metrics[f"agentmemory/{metric_name}"] = value
    return metrics


def compute_timing_metrics(batch, timing_raw):
    valid_samples = _get_ppo_valid_sample_mask(batch)
    num_overall_tokens = torch.sum(batch.batch['attention_mask'][valid_samples]).item()
    num_response_tokens = torch.sum(batch.batch['response_mask'][valid_samples]).item()

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'remax':
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        task_name = str(
            config.actor_rollout_ref.agentgym.get('task_name', '')
        ).strip().lower()
        self.ppo_batch_contract = None

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        if task_name != 'agentmemory':
            assert real_train_batch_size % n_gpus == 0, \
                f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if requires_flattened_action_row_batch_contract(task_name) and self.use_critic:
            dynamic_roles = {
                'actor': bool(config.actor_rollout_ref.actor.use_dynamic_bsz),
                'critic': bool(config.critic.use_dynamic_bsz),
                'critic_forward': bool(config.critic.use_dynamic_bsz),
                'reference_logprob': bool(
                    config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz
                ),
                'rollout_logprob': bool(
                    config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
                ),
            }
            dynamic_max_token_lens = {
                'actor': config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
                'critic': config.critic.ppo_max_token_len_per_gpu,
                'critic_forward': config.critic.forward_max_token_len_per_gpu,
                'reference_logprob': config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu,
                'rollout_logprob': config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu,
            }
            expected_micro_by_role = None
            expected_micro = None
            if not any(dynamic_roles.values()):
                expected_micro_by_role_raw = os.environ.get(
                    'VERL_PPO_EXPECTED_PER_GPU_MICRO_BATCHES'
                )
                if expected_micro_by_role_raw is not None:
                    try:
                        expected_micro_by_role = json.loads(
                            expected_micro_by_role_raw
                        )
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "VERL_PPO_EXPECTED_PER_GPU_MICRO_BATCHES must be a "
                            "JSON object."
                        ) from exc
                    if not isinstance(expected_micro_by_role, dict):
                        raise ValueError(
                            "VERL_PPO_EXPECTED_PER_GPU_MICRO_BATCHES must decode "
                            "to a JSON object."
                        )
                    expected_micro = None
                else:
                    expected_micro_raw = os.environ.get(
                        'VERL_PPO_EXPECTED_PER_GPU_MICRO_BATCH_SIZE', '2'
                    )
                    try:
                        expected_micro = int(expected_micro_raw)
                    except ValueError as exc:
                        raise ValueError(
                            "VERL_PPO_EXPECTED_PER_GPU_MICRO_BATCH_SIZE must be a "
                            f"positive integer, got {expected_micro_raw!r}."
                        ) from exc
            else:
                print(
                    "Action-row PPO dynamic-batch contract: "
                    f"task={task_name} roles={dynamic_roles} "
                    f"max_token_lens={dynamic_max_token_lens}"
                )
            self.ppo_batch_contract = build_legacy_asymmetric_batch_contract(
                actor_mini_batch_size=config.actor_rollout_ref.actor.ppo_mini_batch_size,
                critic_mini_batch_size=config.critic.ppo_mini_batch_size,
                rollout_n=config.actor_rollout_ref.rollout.n,
                world_size=n_gpus,
                actor_sequence_parallel_size=config.actor_rollout_ref.actor.get(
                    'ulysses_sequence_parallel_size', 1
                ),
                critic_sequence_parallel_size=config.critic.get(
                    'ulysses_sequence_parallel_size', 1
                ),
                per_gpu_micro_batches={
                    'actor': config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                    'critic': config.critic.ppo_micro_batch_size_per_gpu,
                    'critic_forward': config.critic.forward_micro_batch_size_per_gpu,
                    'reference_logprob': config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    'rollout_logprob': config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                },
                legacy_micro_batches={
                    'actor': config.actor_rollout_ref.actor.ppo_micro_batch_size,
                    'critic': config.critic.ppo_micro_batch_size,
                    'critic_forward': config.critic.forward_micro_batch_size,
                    'reference_logprob': config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    'rollout_logprob': config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                },
                actor_ppo_epochs=config.actor_rollout_ref.actor.ppo_epochs,
                critic_ppo_epochs=config.critic.ppo_epochs,
                expected_per_gpu_micro_batch_size=expected_micro,
                expected_per_gpu_micro_batches=expected_micro_by_role,
                dynamic_roles=dynamic_roles,
                dynamic_max_token_lens=dynamic_max_token_lens,
            )
            actor_sp = config.actor_rollout_ref.actor.get(
                'ulysses_sequence_parallel_size', 1
            )
            critic_sp = config.critic.get('ulysses_sequence_parallel_size', 1)
            validate_dynamic_batch_token_caps(
                dynamic_roles=self.ppo_batch_contract['dynamic_roles'],
                dynamic_max_token_lens=self.ppo_batch_contract[
                    'dynamic_max_token_lens'
                ],
                sequence_parallel_sizes={
                    'actor': actor_sp,
                    'critic': critic_sp,
                    'critic_forward': critic_sp,
                    'reference_logprob': config.actor_rollout_ref.ref.get(
                        'ulysses_sequence_parallel_size', 1
                    ),
                    'rollout_logprob': actor_sp,
                },
                padded_sequence_length=(
                    config.data.max_prompt_length
                    + config.data.max_response_length
                ),
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(
            data_file=self.config.data.train_file,
            tokenizer=self.tokenizer,
            data_config=self.config.data,
            agentgym_config=self.config.actor_rollout_ref.agentgym,
        )
        procedural_source = self.train_dataset.procedural_index_source
        self.procedural_stream_identity = None
        self.procedural_tasks_per_orbit = None
        if procedural_source is not None:
            if procedural_source.provider_mode != PROVIDER_MODE_RESEEDED_STREAM:
                raise ValueError(
                    "PPO procedural training requires provider_mode="
                    "'reseeded_stream'; fixed_window is reserved for bounded "
                    "generation and evaluation"
                )
            if self.config.data.shuffle:
                raise ValueError(
                    "reseeded procedural index streams require data.shuffle=false; "
                    "the generator already randomizes semantic coordinates"
                )
            procedural_source.validate_training_batch_size(
                self.config.data.train_batch_size
            )
            sampler = StatefulProceduralStreamSampler(procedural_source)
            self.procedural_tasks_per_orbit = procedural_source.tasks_per_orbit
            self.procedural_stream_identity = procedural_source.training_identity(
                server_metadata=self.train_dataset.procedural_server_metadata,
                train_batch_size=self.config.data.train_batch_size,
            )
        # use sampler for better ckpt resume
        elif self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        assert len(self.train_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        if self.config.algorithm.rounds_ctrl.type == 'fixed':
            self.rounds_scheduler = FixedRoundsScheduler(rounds=self.config.algorithm.rounds_ctrl.rounds)
        elif self.config.algorithm.rounds_ctrl.type == 'scaling_inter_stepwise':
            self.rounds_scheduler = StepRoundsScheduler(steps_scaling_inter=self.config.algorithm.rounds_ctrl.steps_scaling_inter,
                                                   rounds_ls=self.config.algorithm.rounds_ctrl.rounds)
        else:
            raise NotImplementedError
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        if self.config.trainer.storage_mode == 'aistudio':
            from aistudio_checkpoint.aistudio_mnt_checkpointer import AistudioMntCheckpointer
            ckpter = AistudioMntCheckpointer()
            save_dir = ckpter.get_save_dir(step=self.global_steps)
            # path: given_path + `/global_step_{global_steps}` + `/actor`
            local_global_step_folder = os.path.join(save_dir,
                                                    f'global_step_{self.global_steps}')
        elif self.config.trainer.storage_mode == 'local':
            # path: given_path + `/global_step_{global_steps}` + `/actor`
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                    f'global_step_{self.global_steps}')
        else:
            raise NotImplementedError
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # Legacy datasets serialize the DataLoader. Procedural streams keep the
        # freshly validated dataset/client and checkpoint only an identity-bound
        # sampler cursor.
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        if self.procedural_stream_identity is not None:
            sampler = self.train_dataloader.sampler
            if not isinstance(sampler, StatefulProceduralStreamSampler):
                raise RuntimeError(
                    "procedural stream DataLoader lost its stateful sampler"
                )
            dataloader_state = build_stream_checkpoint(
                sampler,
                self.procedural_stream_identity,
            )
        else:
            dataloader_state = self.train_dataloader
        torch.save(dataloader_state, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            if self.config.trainer.storage_mode == 'aistudio':
                f.write(str(self.global_steps) + "\n" + ckpter.commit(memo=self.config.trainer.experiment_name))
            elif self.config.trainer.storage_mode == 'local':
                f.write(str(self.global_steps))
            else:
                raise NotImplementedError

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            if self.config.trainer.storage_mode == 'aistudio':
                global_step_folder = find_latest_ckpt_path_aistudio(checkpoint_folder)  # None if no latest
            elif self.config.trainer.storage_mode == 'local':
                global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            else:
                raise NotImplementedError

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])
        self.rounds_scheduler.set_global_steps(self.global_steps)

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        # Native VERL checkpoints are trusted and serialize the DataLoader.
        dataloader_state = torch.load(dataloader_local_path, weights_only=False)
        if self.procedural_stream_identity is not None:
            if not isinstance(dataloader_state, dict):
                raise RuntimeError(
                    "procedural resume refuses a legacy whole-DataLoader checkpoint"
                )
            sampler = self.train_dataloader.sampler
            if not isinstance(sampler, StatefulProceduralStreamSampler):
                raise RuntimeError(
                    "current procedural DataLoader lost its stateful sampler"
                )
            restore_stream_checkpoint(
                sampler,
                self.procedural_stream_identity,
                dataloader_state,
            )
        else:
            if (
                isinstance(dataloader_state, dict)
                and dataloader_state.get("schema")
                == PROCEDURAL_STREAM_CHECKPOINT_SCHEMA
            ):
                raise RuntimeError(
                    "procedural stream checkpoint cannot resume with the current "
                    "non-procedural data configuration"
                )
            self.train_dataloader = dataloader_state
            if isinstance(self.train_dataloader.dataset, RLHFDataset):
                self.train_dataloader.dataset.resume_dataset_state()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        _validate_formal_actor_advantage_config(self.config)

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.config.trainer.storage_mode == 'aistudio':
            self._save_checkpoint()

        # we start from step 1
        self.global_steps += 1
        formal_readback_target_steps = (
            _agentmemory_formal_update_readback_target_steps()
        )
        formal_readback_observed_steps: set[int] = set()
        if formal_readback_target_steps is not None:
            last_reachable_step = max(1, int(self.total_training_steps) - 1)
            unreachable_steps = sorted(
                target_step
                for target_step in formal_readback_target_steps
                if target_step < self.global_steps or target_step > last_reachable_step
            )
            if unreachable_steps:
                raise RuntimeError(
                    "Formal PPO update readback targets are not reachable: "
                    f"targets={unreachable_steps} "
                    f"first={self.global_steps} last={last_reachable_step}."
                )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {
                    'trainer/reference_policy_enabled': float(
                        self.use_reference_policy
                    ),
                }
                timing_raw = {}
                formal_readback_active = (
                    formal_readback_target_steps is not None
                    and self.global_steps in formal_readback_target_steps
                )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=generation_non_tensor_keys(
                        batch.non_tensor_batch
                    ),
                )
                # The environment reset index is an explicit data contract.  Do
                # not rely on parsing a numeric suffix from item_id.
                promote_data_idx_for_rollout(gen_batch.non_tensor_batch)
                if self.procedural_stream_identity is not None:
                    validate_orbit_batch_indices(
                        gen_batch.non_tensor_batch["rollout_data_indices"],
                        tasks_per_orbit=self.procedural_tasks_per_orbit,
                    )
                # Preserve driver-global source identities across DP dispatch.
                # Rollout workers must return these values as parent indices;
                # worker-local indices are ambiguous after per-rank splitting.
                gen_batch.non_tensor_batch['rollout_source_parent_indices'] = np.arange(
                    len(gen_batch), dtype=object
                )
                gen_batch.meta_info['global_steps'] = self.global_steps
                max_policy_turns = self.rounds_scheduler.get_rounds()
                gen_batch.meta_info['max_policy_turns'] = max_policy_turns
                gen_batch.meta_info['max_rounds'] = max_policy_turns
                metrics.update({
                    'max_policy_turns': max_policy_turns,
                    'max_rounds': max_policy_turns,
                })

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    if self.procedural_stream_identity is not None:
                        validate_rollout_parent_coverage(
                            gen_batch_output.non_tensor_batch,
                            expected_parent_count=len(gen_batch),
                            expected_replicas=int(
                                self.config.actor_rollout_ref.rollout.n
                            ),
                        )

                    if self.config.algorithm.adv_estimator == 'remax':
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = batch.batch['rewards']
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    validate_state_aware_rollout_uids(gen_batch_output)
                    formal_required = requires_formal_trajectory_metadata(
                        gen_batch_output
                    )
                    runtime_evidence_required = _agentmemory_env_flag(
                        "AGENTMEMORY_REQUIRE_FORMAL_RUNTIME_EVIDENCE"
                    )
                    expected_suffix_credit = (
                        False
                        if formal_required or runtime_evidence_required
                        else None
                    )
                    validate_formal_trajectory_metadata(
                        gen_batch_output,
                        expected_replicas=int(
                            self.config.actor_rollout_ref.rollout.n
                        ),
                        require=formal_required or runtime_evidence_required,
                        require_runtime_evidence=runtime_evidence_required,
                        expected_suffix_credit=expected_suffix_credit,
                    )
                    # Align source samples with rollout outputs. Standard tasks repeat each
                    # source n times; AgentMemoryGym can return one independent training
                    # sample per agent action and provides rollout_parent_indices.
                    batch = align_batch_to_rollout(
                        batch,
                        gen_batch_output,
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                    )
                    batch = batch.union(gen_batch_output)
                    rollout_uid = batch.non_tensor_batch.get('rollout_uid')
                    if rollout_uid is not None:
                        batch.non_tensor_batch['uid'] = rollout_uid
                    batch.batch[core_algos.PPO_VALID_SAMPLE_MASK] = torch.ones(
                        len(batch), dtype=torch.bool, device=batch.batch.device
                    )
                    world_size = self.actor_rollout_wg.world_size
                    before_pad = len(batch)
                    batch, pad_size = pad_dataproto_to_divisor(batch, world_size)
                    if pad_size:
                        batch.batch[core_algos.PPO_VALID_SAMPLE_MASK][-pad_size:] = False
                    metrics['agentmemory/pad_to_world_size'] = pad_size
                    metrics['agentmemory/batch_size_before_pad'] = before_pad
                    metrics['agentmemory/batch_size_after_pad'] = len(batch)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)
                    validate_formal_trajectory_metadata(
                        batch,
                        expected_replicas=int(
                            self.config.actor_rollout_ref.rollout.n
                        ),
                        require=formal_required or runtime_evidence_required,
                        require_runtime_evidence=runtime_evidence_required,
                        expected_suffix_credit=expected_suffix_credit,
                    )

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    batch.meta_info['agentmemory_formal_update_readback_active'] = (
                        formal_readback_active
                    )
                    if self.ppo_batch_contract is not None:
                        batch.meta_info[PPO_BATCH_CONTRACT_META_KEY] = self.ppo_batch_contract
                        expected_steps = optimizer_step_readback(
                            self.ppo_batch_contract, len(batch)
                        )
                        metrics.update({
                            'ppo_batch/mode_legacy_asymmetric': 1.0,
                            'ppo_batch/actor_raw_mini_batch_rows': self.ppo_batch_contract['actor_raw_mini_batch_rows'],
                            'ppo_batch/critic_raw_mini_batch_rows': self.ppo_batch_contract['critic_raw_mini_batch_rows'],
                            'ppo_batch/actor_local_mini_batch_rows': self.ppo_batch_contract['actor_local_mini_batch_rows'],
                            'ppo_batch/critic_local_mini_batch_rows': self.ppo_batch_contract['critic_local_mini_batch_rows'],
                            'ppo_batch/actor_per_gpu_micro_batch_rows': self.ppo_batch_contract['per_gpu_micro_batches']['actor'],
                            'ppo_batch/critic_per_gpu_micro_batch_rows': self.ppo_batch_contract['per_gpu_micro_batches']['critic'],
                            'ppo_batch/critic_forward_per_gpu_micro_batch_rows': self.ppo_batch_contract['per_gpu_micro_batches']['critic_forward'],
                            'ppo_batch/reference_logprob_per_gpu_micro_batch_rows': (
                                self.ppo_batch_contract['per_gpu_micro_batches']['reference_logprob']
                                if self.use_reference_policy
                                else 0
                            ),
                            'ppo_batch/rollout_logprob_per_gpu_micro_batch_rows': self.ppo_batch_contract['per_gpu_micro_batches']['rollout_logprob'],
                            'ppo_batch/local_rows': expected_steps['local_rows'],
                            'ppo_batch/expected_minibatches_per_epoch': expected_steps['minibatches_per_epoch'],
                            'ppo_batch/expected_actor_optimizer_steps': expected_steps['actor_optimizer_steps'],
                            'ppo_batch/expected_critic_optimizer_steps': expected_steps['critic_optimizer_steps'],
                        })
                        dynamic_roles = self.ppo_batch_contract.get(
                            'dynamic_roles', {}
                        )
                        dynamic_caps = self.ppo_batch_contract.get(
                            'dynamic_max_token_lens', {}
                        )
                        for role in (
                            'actor',
                            'critic',
                            'critic_forward',
                            'reference_logprob',
                            'rollout_logprob',
                        ):
                            metrics[f'ppo_batch/dynamic_{role}'] = float(
                                bool(dynamic_roles.get(role, False))
                            )
                            if role in dynamic_caps:
                                metrics[
                                    f'ppo_batch/{role}_max_token_len_per_gpu'
                                ] = float(dynamic_caps[role])

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                        if (
                            runtime_evidence_required
                            and self.global_steps == 1
                            and _agentmemory_env_flag(
                                "AGENTMEMORY_EXPECT_INITIAL_CRITIC_ZERO"
                            )
                        ):
                            valid_value_mask = batch.batch['response_mask'] * (
                                _get_ppo_valid_sample_mask(batch)
                                .unsqueeze(-1)
                                .to(batch.batch['response_mask'].dtype)
                            )
                            metrics['agentmemory/initial_critic_value_max_abs'] = (
                                core_algos.validate_near_zero_critic_values(
                                    values=batch.batch['values'],
                                    eos_mask=valid_value_mask,
                                )
                            )

                    with _timer('adv', timing_raw):
                        # Keep padding through distributed forward passes so DP
                        # ranks receive equal chunks, then exclude it from every
                        # reward, advantage, loss, and logical metric surface.
                        _mask_ppo_padding_samples(batch)
                        # we combine with rule-based rm
                        reward_tensor = batch.batch['scores']
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                        validate_formal_trajectory_metadata(
                            batch,
                            expected_replicas=int(
                                self.config.actor_rollout_ref.rollout.n
                            ),
                            require=formal_required or runtime_evidence_required,
                            require_runtime_evidence=runtime_evidence_required,
                            expected_suffix_credit=expected_suffix_credit,
                        )
                        _agentmemory_dump_ppo_batch_debug(
                            batch=batch,
                            config=self.config,
                            global_steps=self.global_steps,
                            stage="post_adv",
                        )

                    # update critic
                    post_update_critic_values = None
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        if self.ppo_batch_contract is not None:
                            expected_critic_steps = metrics[
                                'ppo_batch/expected_critic_optimizer_steps'
                            ]
                            actual_critic_steps = critic_output_metrics.get(
                                'critic/optimizer_steps_per_update'
                            )
                            if actual_critic_steps != expected_critic_steps:
                                raise RuntimeError(
                                    "critic optimizer-step readback mismatch: "
                                    f"expected={expected_critic_steps} "
                                    f"actual={actual_critic_steps}."
                                )
                        metrics.update(critic_output_metrics)
                        if formal_readback_active:
                            with _timer('critic_after_update_values', timing_raw):
                                post_update_critic_values = self.critic_wg.compute_values(batch)
                    elif formal_readback_active:
                        raise RuntimeError(
                            "Formal PPO update readback requires an active critic."
                        )

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        if self.ppo_batch_contract is not None:
                            expected_actor_steps = metrics[
                                'ppo_batch/expected_actor_optimizer_steps'
                            ]
                            actual_actor_steps = actor_output_metrics.get(
                                'actor/optimizer_steps_per_update'
                            )
                            if actual_actor_steps != expected_actor_steps:
                                raise RuntimeError(
                                    "actor optimizer-step readback mismatch: "
                                    f"expected={expected_actor_steps} "
                                    f"actual={actual_actor_steps}."
                                )
                            if actor_output_metrics.get(
                                'actor/minibatches_per_epoch'
                            ) != critic_output_metrics.get(
                                'critic/minibatches_per_epoch'
                            ):
                                raise RuntimeError(
                                    "actor and critic mini-batches per epoch differ "
                                    "after worker normalization."
                                )
                        metrics.update(actor_output_metrics)
                        need_formal_readback = _agentmemory_env_flag(
                            "AGENTMEMORY_FORMAL_UPDATE_READBACK"
                        ) and formal_readback_active
                        post_update_log_probs = None
                        if need_formal_readback:
                            with _timer('actor_after_update_log_prob', timing_raw):
                                post_update_log_probs = self.actor_rollout_wg.compute_log_prob(batch)
                        if need_formal_readback:
                            readback = _agentmemory_dump_formal_update_readback(
                                batch=batch,
                                post_actor_log_probs=post_update_log_probs,
                                post_critic_values=post_update_critic_values,
                                actor_update_metrics=actor_output_metrics,
                                critic_update_metrics=critic_output_metrics,
                                config=self.config,
                                global_steps=self.global_steps,
                            )
                            formal_readback_observed_steps.add(self.global_steps)
                            metrics.update(
                                {
                                    "agentmemory/actor_readback_max_abs_delta": readback[
                                        "actor"
                                    ]["summary"]["max_abs_delta"],
                                    "agentmemory/critic_readback_max_abs_delta": readback[
                                        "critic"
                                    ]["summary"]["max_abs_delta"],
                                    "agentmemory/actor_parameter_delta_l2": readback[
                                        "actor"
                                    ]["parameter_delta_l2"],
                                    "agentmemory/critic_parameter_delta_l2": readback[
                                        "critic"
                                    ]["parameter_delta_l2"],
                                }
                            )
                    elif formal_readback_active:
                        raise RuntimeError(
                            "Formal PPO update readback cannot run before actor warmup ends."
                        )

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                self.rounds_scheduler.step()

                if self.global_steps >= self.total_training_steps:

                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    missing_readback_steps = (
                        _agentmemory_missing_formal_update_readback_steps(
                            formal_readback_target_steps,
                            formal_readback_observed_steps,
                        )
                    )
                    if missing_readback_steps:
                        raise RuntimeError(
                            "Formal PPO update readback target steps completed without "
                            f"readback artifacts: {missing_readback_steps}."
                        )
                    return

        missing_readback_steps = _agentmemory_missing_formal_update_readback_steps(
            formal_readback_target_steps,
            formal_readback_observed_steps,
        )
        if missing_readback_steps:
            raise RuntimeError(
                "Formal PPO training ended before configured update readback steps: "
                f"{missing_readback_steps}."
            )
