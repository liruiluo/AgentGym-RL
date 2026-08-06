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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List, Mapping
from omegaconf import DictConfig
import numpy as np
import torch
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict
from torch import nn
from tqdm import tqdm

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

import os
import json
import time
import requests
from copy import deepcopy
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.utils.agentgym.client import (
    configured_multitask_env_addrs,
    env_addr_for_surface_slot,
    init_env_client,
)
from verl.utils.agent_dataset.procedural_index import (
    MULTITASK_LOCAL_DATA_INDEX_KEY,
    MULTITASK_LOCAL_TASK_COUNT_KEY,
    MULTITASK_ROUTE_KIND_KEY,
    MULTITASK_SAMPLING_SEED_KEY,
    MULTITASK_SURFACE_SLOT_KEY,
    ProceduralIndexError,
    validate_multitask_route_triplet,
)
from verl.utils.agentgym.context_policy import assert_rollout_context_supported, read_config, rollout_context_policy
from verl.utils.agentgym.formal_domain_v3 import (
    FORMAL_DOMAIN_SCHEMA_V3,
    FORMAL_WEBSHOP_SCHEMA_V2,
    FormalDomainV3Error,
    bind_generic_timeout_v3,
    build_formal_domain_step_v3,
    resolve_formal_runtime_contract,
    validate_formal_env_schema,
    validate_webshop_action_listing_mode,
    validate_webshop_filesystem_surface,
    validate_webshop_ltm_inventory_mode,
    validate_webshop_memory_prompt_mode,
)
from verl.utils.agentgym.rollout_context import (
    AGENTMEMORY_ACTION_TEXT,
    AGENTMEMORY_GENERATION_PROMPT_DIGEST,
    AGENTMEMORY_GENERATION_PROMPT_LENGTH,
    AGENTMEMORY_GENERATION_RESPONSE_DIGEST,
    AGENTMEMORY_GENERATION_RESPONSE_LENGTH,
    AGENTMEMORY_PACKED_PROMPT_DIGEST,
    AGENTMEMORY_PACKED_PROMPT_LENGTH,
    AGENTMEMORY_PACKED_RESPONSE_DIGEST,
    AGENTMEMORY_PACKED_RESPONSE_LENGTH,
    AGENTMEMORY_SUFFIX_CREDIT_APPLIED,
    AGENTMEMORY_SUFFIX_RETURN,
    AGENTMEMORY_STEP_RECORD_JSON,
    normalize_generation_record,
    validate_official_vllm_generation_record,
    validate_formal_runtime_evidence_rows,
    validate_formal_response_reward_placement,
    validate_formal_sequence_limits,
)
from verl.workers.rollout.agent_vllm_rollout.agentmemory_grouping import (
    AGENTMEMORY_EXACT_STATE_UID,
    AGENTMEMORY_IMMEDIATE_REWARD,
    AGENTMEMORY_PARENT_GROUP_UID,
    AGENTMEMORY_REPLICA_INDEX,
    AGENTMEMORY_TRAJECTORY_RETURN,
    AGENTMEMORY_TRAJECTORY_ROW_ORDER,
    AGENTMEMORY_TRAJECTORY_ROW_UID,
    AGENTMEMORY_TRAJECTORY_TERMINAL,
    AGENTMEMORY_TRAJECTORY_UID,
    build_parent_group_uid,
    build_row_uid,
    build_state_aware_rollout_uid,
    build_trajectory_uid,
    compute_suffix_credit_scores,
    expand_excluded_rollout_parent_groups,
    prompt_state_digest,
    resolve_rollout_parent_index,
    trainable_rollout_row_positions,
    validate_formal_trajectory_rows,
)
from verl.workers.rollout.agent_vllm_rollout.formal_buy_transition import (
    FormalBuyTransitionError,
    validate_formal_buy_transition,
)
from verl.workers.rollout.agent_vllm_rollout.vllm_runtime_config import (
    resolve_official_vllm_compilation_config,
    restore_training_triton_cache_after_vllm,
)
from verl.workers.rollout.schemas import (
    Message,
    RolloutHandler,
    _pre_process_inputs,
    agentmemory_action_listing_mode,
    agentmemory_action_system_prompt,
    agentmemory_ltm_inventory_mode,
    agentmemory_memory_prompt_mode,
)

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


class FormalRuntimeEvidenceError(RuntimeError):
    pass


def _formal_runtime_contract_for_client(env_client) -> tuple[str, str, str]:
    info = getattr(env_client, "info", None)
    if not isinstance(info, Mapping):
        raise FormalRuntimeEvidenceError(
            "Formal AgentMemory client is missing its runtime info payload."
        )
    info_metadata = info.get("metadata")
    client_metadata = getattr(env_client, "metadata", None)
    if not isinstance(info_metadata, Mapping):
        raise FormalRuntimeEvidenceError(
            "Formal AgentMemory client info is missing server metadata."
        )
    if isinstance(client_metadata, Mapping):
        contract_keys = (
            "formal_schema_version",
            "domain_id",
            "surface",
            "contract_id",
            "contract_sha256",
            "system_prompt",
            "ltm_inventory_mode",
            "memory_prompt_mode",
            "action_listing_mode",
        )
        mismatches = [
            key
            for key in contract_keys
            if key in info_metadata
            and key in client_metadata
            and info_metadata[key] != client_metadata[key]
        ]
        if mismatches:
            raise FormalRuntimeEvidenceError(
                "Formal AgentMemory client/server metadata mismatch: "
                f"fields={mismatches}."
            )
    try:
        validate_webshop_ltm_inventory_mode(
            info_metadata,
            expected_mode=agentmemory_ltm_inventory_mode(),
        )
        validate_webshop_memory_prompt_mode(
            info_metadata,
            expected_mode=agentmemory_memory_prompt_mode(),
        )
        validate_webshop_filesystem_surface(
            info_metadata,
            expected_prompt_mode=agentmemory_memory_prompt_mode(),
        )
        validate_webshop_action_listing_mode(
            info_metadata,
            expected_mode=agentmemory_action_listing_mode(),
        )
        return resolve_formal_runtime_contract(
            info_metadata,
            webshop_v2_system_prompt=agentmemory_action_system_prompt(
                surface=info_metadata["surface"]
            ),
        )
    except FormalDomainV3Error as exc:
        raise FormalRuntimeEvidenceError(
            f"Invalid formal AgentMemory runtime contract: {exc}"
        ) from exc


def _validate_runtime_env_schema(
    schema_version: str,
    env_info: Mapping,
    *,
    boundary: str,
) -> None:
    try:
        validate_formal_env_schema(
            schema_version,
            env_info,
            boundary=boundary,
        )
    except FormalDomainV3Error as exc:
        raise FormalRuntimeEvidenceError(str(exc)) from exc


def _build_formal_webshop_step_v2(
    *,
    content: str,
    score: float,
    task_round: int,
    done: bool,
    item_id: str,
    parent_index: int,
    parent_group_uid: str,
    replica_index: int,
    trajectory_uid: str,
    exact_state_uid: str,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    latest_observation: str,
    visible_prompt: str,
    single_observation_prompt_digest: str,
    env_result: str,
    generation_record: Mapping,
    env_info_before: Mapping,
    env_info_after: Mapping,
    action_submission: Mapping | None,
) -> dict:
    if "current_subtask_index" not in env_info_before:
        raise FormalRuntimeEvidenceError(
            f"Formal WebShop v2 step is missing the pre-action session index for item {item_id}."
        )
    if "current_subtask_index" not in env_info_after:
        raise FormalRuntimeEvidenceError(
            f"Formal WebShop v2 step is missing the post-action session index for item {item_id}."
        )
    session_index = int(env_info_before["current_subtask_index"])
    next_session_index = int(env_info_after["current_subtask_index"])
    session_advanced = next_session_index > session_index
    tool_ops = env_info_after.get("tool_ops", [])
    try:
        buy_evidence = validate_formal_buy_transition(
            tool_ops=tool_ops,
            env_step=task_round,
            subtask_index_before=session_index,
            subtask_index_after=next_session_index,
            done=done,
        )
    except FormalBuyTransitionError as exc:
        raise FormalRuntimeEvidenceError(
            f"Formal BUY transition evidence is invalid for item={item_id}: {exc}"
        ) from exc
    committed_purchase = bool(buy_evidence["committed_purchase"])
    purchase_correct = buy_evidence["purchase_correct"]
    accepted_purchase = bool(buy_evidence["accepted_purchase"])
    session_trace_after = env_info_after.get("session_trace")
    if not isinstance(session_trace_after, list):
        raise FormalRuntimeEvidenceError(
            "Formal AgentMemory step is missing post-action session_trace."
        )
    raw_history_cleared = bool(session_advanced and not session_trace_after)
    search_tool_ops = [
        tool_op
        for tool_op in tool_ops
        if isinstance(tool_op, dict)
        and str(tool_op.get("op", "")).upper() == "SEARCH"
        and int(tool_op.get("step", -1)) == task_round
    ]
    search_result_count = None
    if search_tool_ops:
        if len(search_tool_ops) != 1 or "result_count" not in search_tool_ops[0]:
            raise FormalRuntimeEvidenceError(
                "Formal SEARCH tool record is missing result_count."
            )
        search_result_count = int(search_tool_ops[0]["result_count"])
    if "episode_success" not in env_info_after:
        raise FormalRuntimeEvidenceError(
            "Formal WebShop v2 step is missing authoritative episode_success."
        )
    if type(env_info_after["episode_success"]) is not bool:
        raise FormalRuntimeEvidenceError(
            "Formal WebShop v2 episode_success must be a boolean."
        )
    episode_success = env_info_after["episode_success"]
    outcome = (
        "success"
        if done and episode_success
        else "terminal_failure" if done else "continue"
    )
    return {
        "schema_version": FORMAL_WEBSHOP_SCHEMA_V2,
        "prompt_token_ids": prompt_token_ids,
        "content": content,
        "score": float(score),
        "parent_index": parent_index,
        "parent_group_uid": parent_group_uid,
        "replica_index": replica_index,
        "trajectory_uid": trajectory_uid,
        "exact_state_uid": exact_state_uid,
        "task_round": task_round,
        "done": done,
        "item_id": item_id,
        "session_index": session_index,
        "subtask_index": session_index,
        "next_session_index": next_session_index,
        "subtask_index_before": session_index,
        "subtask_index_after": next_session_index,
        "visible_prompt": visible_prompt,
        "latest_observation": latest_observation,
        "prompt_history_policy": "latest_observation_only",
        "raw_prior_messages_visible": False,
        "single_observation_prompt_digest": single_observation_prompt_digest,
        "response_token_ids": response_token_ids,
        "response_token_count": int(generation_record["response_token_count"]),
        "max_response_tokens": int(generation_record["max_response_tokens"]),
        "finish_reason": str(generation_record["finish_reason"]),
        "finish_reason_source": str(generation_record["finish_reason_source"]),
        "stop_reason": generation_record.get("stop_reason"),
        "generation_backend_source": str(generation_record["backend_source"]),
        "generation_stop_reason": generation_record.get("stop_reason"),
        "generation_eos_token_ids": list(
            generation_record["configured_eos_token_ids"]
        ),
        "tokenizer_primary_eos_token_id": generation_record[
            "primary_eos_token_id"
        ],
        "tokenizer_pad_token_id": generation_record["tokenizer_pad_token_id"],
        "generation_token_ids_are_exact": bool(
            generation_record["token_ids_are_exact"]
        ),
        "backend_token_ids_are_exact": bool(
            generation_record["backend_token_ids_are_exact"]
        ),
        "truncated": bool(generation_record["truncated"]),
        "env_result": env_result,
        "env_info_before": deepcopy(dict(env_info_before)),
        "env_info_after": deepcopy(dict(env_info_after)),
        "action_submission": deepcopy(dict(action_submission or {})),
        "committed_purchase": committed_purchase,
        "purchase_correct": purchase_correct,
        "accepted_purchase": accepted_purchase,
        "session_advanced": session_advanced,
        "buy_committed": committed_purchase,
        "buy_accepted": accepted_purchase,
        "subtask_advanced": session_advanced,
        "raw_history_cleared": raw_history_cleared,
        "search_result_count": search_result_count,
        "outcome": outcome,
    }


def _ordered_unique_token_ids(*values) -> list[int]:
    ordered: list[int] = []

    def add(value):
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
            return
        token_id = int(value)
        if token_id not in ordered:
            ordered.append(token_id)

    for value in values:
        add(value)
    return ordered


def _resolve_generation_eos_token_ids(
    *,
    actor_module,
    tokenizer,
    model_hf_config,
    model_path,
    trust_remote_code: bool,
) -> list[int]:
    configured_eos = None
    if model_path is not None:
        try:
            from transformers import GenerationConfig

            configured_eos = GenerationConfig.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
            ).eos_token_id
        except Exception as exc:
            print(
                "GenerationConfig EOS readback failed; using loaded model config: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
    actor_generation_config = getattr(actor_module, "generation_config", None)
    resolved = _ordered_unique_token_ids(
        configured_eos,
        getattr(actor_generation_config, "eos_token_id", None),
        getattr(model_hf_config, "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    )
    if not resolved:
        raise RuntimeError("Generation has no configured EOS token IDs.")
    return resolved




def ensure_transformers_tokenizer_compat():
    """Patch small Transformers/vLLM tokenizer API drift at runtime.

    vLLM 0.11 caches ``all_special_tokens_extended``, but the installed
    Transformers 5.x Qwen2Tokenizer no longer exposes that property.  The
    cached value is only used as tokenizer metadata, so falling back to
    ``all_special_tokens`` is sufficient for text-only AgentMemoryGym rollout.
    """
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return
    if hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
        return

    @property
    def all_special_tokens_extended(self):
        return list(self.all_special_tokens)

    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended




def _agentmemory_debug_enabled() -> bool:
    return os.environ.get("AGENTMEMORY_ROLLOUT_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _agentmemory_debug(message: str) -> None:
    if not _agentmemory_debug_enabled():
        return
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except Exception:
        rank = 0
    print(f"[AgentMemoryRollout][rank={rank}][{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def _agentmemory_latest_observation_suffix_credit_enabled() -> bool:
    return _env_flag("AGENTMEMORY_LATEST_OBS_SUFFIX_CREDIT")


def _agentmemory_state_aware_group_uid_enabled() -> bool:
    # Action-level samples are only comparable within the exact prompt state.
    # Legacy grouping flags remain accepted as no-op compatibility aliases.
    return True


def _agentmemory_batch_debug_enabled() -> bool:
    return _env_flag("AGENTMEMORY_BATCH_DEBUG")


def left_pad_sequence(sequences, padding_value):
    max_len = max(sequence.size(0) for sequence in sequences)
    return left_pad_sequence_to_length(sequences, max_len, padding_value)


def left_pad_sequence_to_length(sequences, target_len, padding_value):
    """Left-pad/truncate 1D tensors to a fixed length.

    AgentMemory latest-observation rollout builds one trainable sample per
    action turn. The prompt for each turn is the current observation, so its
    length can differ across data-parallel rollout workers. Padding only to
    the local worker max makes DataProto.concat fail on multi-GPU runs. Keep a
    fixed prompt width before concatenation, matching the normal VERL prompt
    tensor contract.
    """
    padded = []
    for sequence in sequences:
        if sequence.size(0) > target_len:
            sequence = sequence[-target_len:]
        pad_len = target_len - sequence.size(0)
        if pad_len > 0:
            pad = torch.full((pad_len,), padding_value, dtype=sequence.dtype, device=sequence.device)
            sequence = torch.cat((pad, sequence), dim=0)
        padded.append(sequence)
    return torch.stack(padded, dim=0)


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, rollout_config: DictConfig, agentgym_config: DictConfig, tokenizer, model_hf_config, model_path=None, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = rollout_config
        self.agentgym_config = agentgym_config
        self.actor_module = actor_module
        if str(read_config(agentgym_config, "task_name", "")).lower() == "agentmemory":
            validate_formal_sequence_limits(
                prompt_width=int(rollout_config.prompt_length),
                response_width=int(rollout_config.response_length),
                max_model_len=int(rollout_config.max_model_len),
            )
        self.generation_eos_token_ids = _resolve_generation_eos_token_ids(
            actor_module=actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            model_path=model_path,
            trust_remote_code=bool(
                rollout_config.get("trust_remote_code", False)
            ),
        )
        self.primary_eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.primary_eos_token_id is None:
            raise RuntimeError("Generation tokenizer has no primary EOS token ID.")
        if bool(rollout_config.get('use_hf_generate', False)):
            raise ValueError('AgentMemoryGym formal rollouts require native vLLM, not HF generate.')

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        self._official_vllm = vllm_version not in ('0.3.1', '0.4.2', '0.5.4', '0.6.3')
        if not self._official_vllm:
            assert not (
                not rollout_config.enforce_eager
                and rollout_config.free_cache_engine
            ), "legacy vLLM cannot combine CUDA graph with free_cache_engine"
        if (
            self._official_vllm
            and rollout_config.free_cache_engine
            and not rollout_config.get('enable_sleep_mode', False)
        ):
            raise ValueError(
                'Official vLLM requires enable_sleep_mode=true when '
                'free_cache_engine=true; otherwise rollout weights/KV cache '
                'can remain resident during the PPO optimizer phase.'
            )
        if self._official_vllm:
            # Avoid vLLM V1 startup port races when AgentMemoryGym launches one
            # rollout engine per FSDP/Ray rank on the same host.  vLLM's default
            # get_open_port() probes then releases a random port; concurrent
            # engines can occasionally pick the same TCPStore port and fail with
            # EADDRINUSE.  If AGENTMEMORY_VLLM_PORT_BASE is set, give each rank
            # a disjoint port block while leaving vanilla VERL behavior unchanged.
            port_base = os.environ.get('AGENTMEMORY_VLLM_PORT_BASE')
            if port_base:
                try:
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else (os.getpid() % 64)
                    os.environ['VLLM_PORT'] = str(int(port_base) + int(rank) * 100)
                    print(f'AgentMemoryGym vLLM rank port base: rank={rank} VLLM_PORT={os.environ["VLLM_PORT"]}', flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f'AgentMemoryGym vLLM rank port setup skipped: {type(exc).__name__}: {exc}', flush=True)
            if model_path is None:
                raise ValueError("model_path is required for official vLLM rollout")
            ensure_transformers_tokenizer_compat()
            from verl.workers.qwen35_weight_sync import resolve_vllm_init_load_format

            init_load_format = resolve_vllm_init_load_format(
                model_type=str(getattr(model_hf_config, 'model_type', '')),
                configured_init_load_format=rollout_config.get(
                    'vllm_init_load_format', None
                ),
            )
            hf_overrides = rollout_config.get('hf_overrides', None)
            if hf_overrides is not None and not isinstance(hf_overrides, dict):
                hf_overrides = dict(hf_overrides)
            official_vllm_kwargs = dict(
                model=model_path,
                tokenizer=model_path,
                tensor_parallel_size=tensor_parallel_size,
                dtype=rollout_config.dtype,
                enforce_eager=rollout_config.enforce_eager,
                gpu_memory_utilization=rollout_config.gpu_memory_utilization,
                skip_tokenizer_init=False,
                trust_remote_code=rollout_config.get('trust_remote_code', False),
                hf_overrides=hf_overrides,
                model_impl=rollout_config.get('model_impl', 'auto'),
                max_model_len=rollout_config.get('max_model_len', None),
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=rollout_config.get('max_num_seqs', None),
                disable_log_stats=rollout_config.disable_log_stats,
                enable_chunked_prefill=rollout_config.enable_chunked_prefill,
                enable_sleep_mode=rollout_config.get('enable_sleep_mode', False),
                block_size=rollout_config.get('block_size', None),
                language_model_only=rollout_config.get('language_model_only', False),
            )
            limit_mm_per_prompt = rollout_config.get('limit_mm_per_prompt', None)
            if limit_mm_per_prompt is not None:
                official_vllm_kwargs['limit_mm_per_prompt'] = dict(limit_mm_per_prompt)
            gdn_prefill_backend = rollout_config.get('gdn_prefill_backend', None)
            if gdn_prefill_backend is not None:
                official_vllm_kwargs['gdn_prefill_backend'] = gdn_prefill_backend
            if init_load_format is not None:
                official_vllm_kwargs['load_format'] = init_load_format
            compilation_config = resolve_official_vllm_compilation_config(
                enforce_eager=bool(rollout_config.enforce_eager),
                configured=rollout_config.get('compilation_config', None),
                cudagraph_capture_sizes=rollout_config.get(
                    'cudagraph_capture_sizes', None
                ),
            )
            if compilation_config is not None:
                official_vllm_kwargs['compilation_config'] = compilation_config
            print(
                'AgentMemoryGym official vLLM runtime config: '
                f'version={vllm_version} '
                f'enforce_eager={bool(rollout_config.enforce_eager)} '
                f'free_cache_engine={bool(rollout_config.free_cache_engine)} '
                f'enable_sleep_mode={bool(rollout_config.get("enable_sleep_mode", False))} '
                f'compilation_config={compilation_config}',
                flush=True,
            )
            self.inference_engine = LLM(**official_vllm_kwargs)
            training_triton_cache = restore_training_triton_cache_after_vllm()
            if training_triton_cache is not None:
                print(
                    "AgentMemoryGym training Triton cache: "
                    + json.dumps(training_triton_cache, sort_keys=True),
                    flush=True,
                )
            llm_engine = getattr(self.inference_engine, 'llm_engine', None)
            engine_core = getattr(llm_engine, 'engine_core', None)
            engine_client_type = type(engine_core).__name__
            print(
                f'AgentMemoryGym official vLLM engine client: {engine_client_type}',
                flush=True,
            )
            if (
                os.environ.get('AGENTMEMORY_REQUIRE_VLLM_INPROC', '0') == '1'
                and engine_client_type != 'InprocClient'
            ):
                raise RuntimeError(
                    'AGENTMEMORY_REQUIRE_VLLM_INPROC=1 requires InprocClient, '
                    f'got {engine_client_type}.'
                )
        else:
            self.inference_engine = LLM(
                actor_module,
                tokenizer=tokenizer,
                model_hf_config=model_hf_config,
                tensor_parallel_size=tensor_parallel_size,
                dtype=rollout_config.dtype,
                enforce_eager=rollout_config.enforce_eager,
                gpu_memory_utilization=rollout_config.gpu_memory_utilization,
                skip_tokenizer_init=False,
                load_format=rollout_config.load_format,
                disable_log_stats=rollout_config.disable_log_stats,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=rollout_config.enable_chunked_prefill,
            )

        # Offload vllm model to reduce peak memory usage when the customized
        # VERL vLLM wrapper provides the old method. Official vLLM uses
        # wake_up/sleep and may not support this method.
        if self.inference_engine is not None and hasattr(self.inference_engine, 'offload_model_weights'):
            self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=rollout_config.max_tokens,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in rollout_config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = rollout_config.get(k)
        kwargs["n"] = 1  # because we have repeated task n times

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        self.generation_stop_token_ids = _ordered_unique_token_ids(
            self.generation_eos_token_ids,
            getattr(self.sampling_params, "stop_token_ids", None),
        )
        print(
            "AgentMemory generation token contract: "
            f"eos_token_ids={self.generation_eos_token_ids} "
            f"stop_token_ids={self.generation_stop_token_ids} "
            f"primary_eos_token_id={self.primary_eos_token_id} "
            f"pad_token_id={tokenizer.pad_token_id} "
            f"backend={'official_vllm' if self._official_vllm else 'legacy_vllm'}",
            flush=True,
        )

        self.pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer

    def _maybe_resume_engine(self):
        if self.inference_engine is None:
            return
        if hasattr(self.inference_engine, 'init_cache_engine'):
            self.inference_engine.init_cache_engine()
        elif hasattr(self.inference_engine, 'wake_up'):
            try:
                self.inference_engine.wake_up()
            except Exception as exc:
                print(f"vLLM wake_up skipped/failed: {exc}")

    def _maybe_release_engine(self):
        if self.inference_engine is None:
            return
        if hasattr(self.inference_engine, 'free_cache_engine'):
            self.inference_engine.free_cache_engine()
        elif hasattr(self.inference_engine, 'sleep'):
            try:
                self.inference_engine.sleep(level=1)
            except Exception as exc:
                print(f"vLLM sleep skipped/failed: {exc}")


    def _generate_token_ids(self, generation_prompt_idxs, sampling_params):
        if self._official_vllm:
            prompts = [{"prompt_token_ids": list(ids)} for ids in generation_prompt_idxs]
            return self.inference_engine.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        return self.inference_engine.generate(
            prompts=None,
            prompt_token_ids=generation_prompt_idxs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

    def _output_generation_records(
        self,
        output,
        sampling_params=None,
        *,
        require_exact_metadata: bool = False,
    ):
        """Preserve actual generated tokens and backend termination metadata."""

        sampling_params = sampling_params or self.sampling_params
        max_tokens = int(
            getattr(
                sampling_params,
                "max_tokens",
                self.config.get("max_tokens", self.config.response_length),
            )
        )
        configured_stop_ids = getattr(self, "generation_stop_token_ids", None)
        if configured_stop_ids is None:
            configured_stop_ids = getattr(self, "generation_eos_token_ids", None)
        if configured_stop_ids is None:
            tokenizer = getattr(self, "tokenizer", None)
            tokenizer_eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if tokenizer_eos_token_id is None:
                raise RuntimeError("Generation has no stop-token metadata.")
            configured_stop_ids = [tokenizer_eos_token_id]
        configured_stop_ids = list(configured_stop_ids)

        def normalize_row(
            token_ids,
            *,
            finish_reason=None,
            stop_reason=None,
            source,
            token_ids_are_exact=False,
        ):
            return normalize_generation_record(
                token_ids,
                eos_token_ids=configured_stop_ids,
                primary_eos_token_id=self.primary_eos_token_id,
                pad_token_id=self.pad_token_id,
                max_tokens=max_tokens,
                backend_finish_reason=finish_reason,
                stop_reason=stop_reason,
                finish_reason_source=source,
                token_ids_are_exact=token_ids_are_exact,
            )

        if isinstance(output, tuple):
            if require_exact_metadata:
                raise RuntimeError(
                    "Formal legacy/custom tuple generation lacks per-row token "
                    "length and finish metadata."
                )
            ids = output[0]
            rows = ids.tolist() if hasattr(ids, 'tolist') else ids
            return [normalize_row(row, source="verl_vllm_wrapper") for row in rows]
        if hasattr(output, 'tolist'):
            if require_exact_metadata:
                raise RuntimeError(
                    "Formal tensor generation lacks per-row token length and "
                    "finish metadata."
                )
            return [
                normalize_row(row, source="tensor_generation_output")
                for row in output.tolist()
            ]
        if isinstance(output, list) and (not output or isinstance(output[0], (list, tuple))):
            if require_exact_metadata:
                raise RuntimeError(
                    "Formal list generation lacks per-row token length and "
                    "finish metadata."
                )
            return [normalize_row(row, source="list_generation_output") for row in output]
        if isinstance(output, list) and output and isinstance(output[0], dict):
            if require_exact_metadata:
                raise RuntimeError(
                    "Formal official-vLLM generation rejects self-described "
                    "list/dict token metadata."
                )
            records = [
                normalize_row(
                    row.get("token_ids", []),
                    finish_reason=row.get("finish_reason"),
                    stop_reason=row.get("stop_reason"),
                    source=str(row.get("finish_reason_source", "normalized_generation")),
                    token_ids_are_exact=bool(
                        row.get("token_ids_are_exact", False)
                    ),
                )
                for row in output
            ]
            if require_exact_metadata and not all(
                record["token_ids_are_exact"] for record in records
            ):
                raise RuntimeError(
                    "Formal HF generation could not prove the per-row valid token boundary."
                )
            return records
        records = []
        for request_output in output:
            if not getattr(request_output, 'outputs', None):
                raise RuntimeError(
                    "Formal generation backend returned no candidate output."
                )
            else:
                candidate = request_output.outputs[0]
                candidate_finish_reason = getattr(
                    candidate, "finish_reason", None
                )
                candidate_stop_reason = getattr(candidate, "stop_reason", None)
                if require_exact_metadata and candidate_finish_reason is None:
                    raise RuntimeError(
                        "Formal official-vLLM generation is missing backend finish_reason."
                    )
                if require_exact_metadata and not bool(
                    getattr(self, "_official_vllm", False)
                ):
                    raise RuntimeError(
                        "Formal RequestOutput-like generation is not bound to the "
                        "official vLLM backend."
                    )
                if require_exact_metadata and (
                    not isinstance(candidate.token_ids, list)
                    or any(type(token_id) is not int for token_id in candidate.token_ids)
                ):
                    raise RuntimeError(
                        "Formal official-vLLM generation token IDs are not a raw list[int]."
                    )
                if require_exact_metadata and candidate_stop_reason is not None and type(
                    candidate_stop_reason
                ) is not int:
                    raise RuntimeError(
                        "Formal official-vLLM stop_reason is not a raw int or None."
                    )
                record = normalize_row(
                    candidate.token_ids,
                    finish_reason=candidate_finish_reason,
                    stop_reason=candidate_stop_reason,
                    source="official_vllm",
                    token_ids_are_exact=True,
                )
                if require_exact_metadata:
                    validate_official_vllm_generation_record(record)
                records.append(record)
        return records

    def _output_token_id_lists(self, output):
        return [
            record["token_ids"]
            for record in self._output_generation_records(output)
        ]


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def preprocess_prompt_to_rollout_handler(self, prompts: DataProto, n: int) -> List[RolloutHandler]:
        assert "raw_prompt" in prompts.non_tensor_batch.keys(), "raw_prompt is not in non_tensor_batch, need to set data.return_raw_chat=True"
        handler_list = []
        eval_parent_indices = prompts.non_tensor_batch.get("rollout_eval_parent_indices")
        eval_data_indices = prompts.non_tensor_batch.get("rollout_data_indices")
        multitask_surface_slots = prompts.non_tensor_batch.get(
            MULTITASK_SURFACE_SLOT_KEY
        )
        multitask_local_data_indices = prompts.non_tensor_batch.get(
            MULTITASK_LOCAL_DATA_INDEX_KEY
        )
        multitask_route_kinds = prompts.non_tensor_batch.get(
            MULTITASK_ROUTE_KIND_KEY
        )
        multitask_sampling_seeds = prompts.non_tensor_batch.get(
            MULTITASK_SAMPLING_SEED_KEY
        )
        multitask_local_task_counts = prompts.non_tensor_batch.get(
            MULTITASK_LOCAL_TASK_COUNT_KEY
        )
        source_parent_indices = prompts.non_tensor_batch.get(
            "rollout_source_parent_indices"
        )
        if eval_data_indices is not None and len(eval_data_indices) != len(
            prompts.non_tensor_batch["raw_prompt"]
        ):
            raise RuntimeError(
                "rollout_data_indices must align with raw_prompt rows: "
                f"indices={len(eval_data_indices)} "
                f"prompts={len(prompts.non_tensor_batch['raw_prompt'])}"
            )
        if (multitask_surface_slots is None) != (
            multitask_local_data_indices is None
        ):
            raise RuntimeError(
                "multitask rollout rows must carry both surface slots and local "
                "data indices"
            )
        uniform_route_fields = (
            multitask_route_kinds,
            multitask_sampling_seeds,
            multitask_local_task_counts,
        )
        if any(values is not None for values in uniform_route_fields) and not all(
            values is not None for values in uniform_route_fields
        ):
            raise RuntimeError(
                "uniform multitask rollout rows must carry route kind, sampling "
                "seed, and local task count together"
            )
        if multitask_route_kinds is not None and multitask_surface_slots is None:
            raise RuntimeError(
                "uniform multitask route metadata requires surface/local indices"
            )
        for field, values in (
            (MULTITASK_SURFACE_SLOT_KEY, multitask_surface_slots),
            (MULTITASK_LOCAL_DATA_INDEX_KEY, multitask_local_data_indices),
            (MULTITASK_ROUTE_KIND_KEY, multitask_route_kinds),
            (MULTITASK_SAMPLING_SEED_KEY, multitask_sampling_seeds),
            (MULTITASK_LOCAL_TASK_COUNT_KEY, multitask_local_task_counts),
        ):
            if values is not None and len(values) != len(
                prompts.non_tensor_batch["raw_prompt"]
            ):
                raise RuntimeError(
                    f"{field} must align with raw_prompt rows: "
                    f"values={len(values)} "
                    f"prompts={len(prompts.non_tensor_batch['raw_prompt'])}"
                )
        for i, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            parent_index_for_eval = resolve_rollout_parent_index(
                i,
                source_parent_indices=source_parent_indices,
                eval_parent_indices=eval_parent_indices,
            )
            raw_item_id = str(prompts.non_tensor_batch["item_id"][i])
            try:
                parsed_item_id = int(raw_item_id.rsplit("_", 1)[-1])
            except (TypeError, ValueError):
                # Eval rows may carry an opaque source task id.  The explicit
                # rollout_data_indices value remains authoritative for reset;
                # this integer is only a stable logging/evidence row id.
                parsed_item_id = int(eval_data_indices[i]) if eval_data_indices is not None else i
            for replica_index in range(n):
                # only keep not pad part
                input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch['input_ids'][i])
                attention_mask = _pre_process_inputs(0, prompts.batch['attention_mask'][i])
                position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
                handler = RolloutHandler(
                    messages=[
                        Message(role=prompt["role"], content=prompt["content"]) for prompt in raw_prompt
                    ],
                    task_name=raw_item_id.split("_", 1)[0],
                    item_id=parsed_item_id,
                    score=0,
                    done=False,
                    input_ids=list(input_ids),
                    prompt_ids=list(input_ids),
                    response_ids=[],
                    attention_mask=list(attention_mask),
                    prompt_attention_mask=list(attention_mask),
                    response_attention_mask=[],
                    position_ids=list(position_ids),
                    prompt_position_ids=list(position_ids),
                    response_position_ids=[],
                    loss_mask=[0] * len(input_ids),
                    prompt_loss_mask=[0] * len(input_ids),
                    response_loss_mask=[],
                    max_response_len=self.config.response_length,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length)
                )
                assert len(handler.input_ids) == len(handler.attention_mask) == len(handler.position_ids) == len(handler.loss_mask), f"RolloutHandler has mismatched length: input_ids={len(handler.input_ids)}, attention_mask={len(handler.attention_mask)}, position_ids={len(handler.position_ids)}, loss_mask={len(handler.loss_mask)}"
                handler.parent_index = parent_index_for_eval
                if eval_data_indices is not None:
                    raw_data_index = eval_data_indices[i]
                    if isinstance(raw_data_index, bool):
                        raise RuntimeError(
                            f"rollout_data_indices[{i}] must be an integer, got bool"
                        )
                    try:
                        handler.data_idx = int(raw_data_index)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise RuntimeError(
                            f"rollout_data_indices[{i}] is not an integer: "
                            f"{raw_data_index!r}"
                        ) from exc
                else:
                    handler.data_idx = int(handler.item_id)
                if multitask_surface_slots is not None:
                    route_kwargs = {}
                    if multitask_route_kinds is not None:
                        route_kwargs = {
                            "route_kind": multitask_route_kinds[i],
                            "sampling_seed": multitask_sampling_seeds[i],
                            "local_task_count": multitask_local_task_counts[i],
                        }
                    try:
                        (
                            handler.data_idx,
                            handler.agentmemory_surface_slot,
                            handler.agentmemory_local_data_idx,
                        ) = validate_multitask_route_triplet(
                            handler.data_idx,
                            multitask_surface_slots[i],
                            multitask_local_data_indices[i],
                            **route_kwargs,
                        )
                    except ProceduralIndexError as exc:
                        raise RuntimeError(
                            f"multitask rollout route row {i} is invalid: {exc}"
                        ) from exc
                handler.rollout_replica_index = replica_index
                handler_list.append(handler)
        return handler_list

    def build_rollout_handler_from_prompt(
        self,
        prompt_token_ids: List[int],
        content: str,
        score: float,
        parent_index: int,
        sampled_response_token_ids: List[int] | None = None,
    ) -> RolloutHandler:
        if sampled_response_token_ids is None:
            response_token_ids = self.tokenizer.encode(
                content, add_special_tokens=False
            )
        else:
            response_token_ids = [int(token_id) for token_id in sampled_response_token_ids]
        input_ids = prompt_token_ids + response_token_ids
        attention_mask = [1] * len(input_ids)
        position_ids = list(range(len(input_ids)))
        loss_mask = [0] * len(prompt_token_ids) + [1] * len(response_token_ids)
        return RolloutHandler(
            messages=[],
            task_name="agentmemory",
            item_id=parent_index,
            score=score,
            done=False,
            input_ids=input_ids,
            prompt_ids=list(prompt_token_ids),
            response_ids=list(response_token_ids),
            attention_mask=attention_mask,
            prompt_attention_mask=[1] * len(prompt_token_ids),
            response_attention_mask=[1] * len(response_token_ids),
            position_ids=position_ids,
            prompt_position_ids=list(range(len(prompt_token_ids))),
            response_position_ids=list(range(len(prompt_token_ids), len(input_ids))),
            loss_mask=loss_mask,
            prompt_loss_mask=[0] * len(prompt_token_ids),
            response_loss_mask=[1] * len(response_token_ids),
            max_response_len=self.config.response_length,
            max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
        )

    def pack_rollout_handlers(
        self,
        rollout_handler_ls: List[RolloutHandler],
        cur_device,
        parent_indices: List[int] | None = None,
        done_flags: List[bool] | None = None,
        parent_group_uids: List[str] | None = None,
        exact_state_uids: List[str] | None = None,
        replica_indices: List[int] | None = None,
        trajectory_uids: List[str] | None = None,
        trajectory_returns: List[float] | None = None,
        immediate_rewards: List[float] | None = None,
        trajectory_row_uids: List[str] | None = None,
        trajectory_row_orders: List[int] | None = None,
        trajectory_terminals: List[bool] | None = None,
        task_rounds: List[int] | None = None,
        action_texts: List[str] | None = None,
        suffix_credit_applied: bool | None = None,
        suffix_returns: List[float] | None = None,
        step_records: List[dict] | None = None,
    ) -> DataProto:
        if step_records is not None and len(step_records) != len(rollout_handler_ls):
            raise RuntimeError(
                "Formal step-record count does not match rollout handlers: "
                f"records={len(step_records)} handlers={len(rollout_handler_ls)}."
            )
        response_ids, response_attention_mask, response_position_ids, response_loss_mask = [], [], [], []
        scores = []
        for row_index, rollout_handler in enumerate(rollout_handler_ls):
            expected_sampled_response = None
            if step_records is not None:
                raw_response = step_records[row_index].get("response_token_ids")
                if not isinstance(raw_response, list):
                    raise RuntimeError(
                        "Formal rollout step is missing generated response token IDs: "
                        f"row={row_index}."
                    )
                expected_sampled_response = [int(token_id) for token_id in raw_response]
                if rollout_handler.response_ids != expected_sampled_response:
                    raise RuntimeError(
                        "Generated response tokens differ from the response handler "
                        f"before packing at row {row_index}."
                    )
                if (
                    len(rollout_handler.response_ids) > rollout_handler.max_response_len
                    or len(rollout_handler.input_ids) > rollout_handler.max_model_len
                ):
                    raise RuntimeError(
                        "Formal sampled response would be truncated before PPO packing: "
                        f"row={row_index} prompt={len(rollout_handler.prompt_ids)} "
                        f"response={len(rollout_handler.response_ids)} "
                        f"max_response={rollout_handler.max_response_len} "
                        f"max_model={rollout_handler.max_model_len}."
                    )
            rollout_handler.truncate_output_ids()
            if (
                expected_sampled_response is not None
                and rollout_handler.response_ids != expected_sampled_response
            ):
                raise RuntimeError(
                    "Formal sampled response changed during handler truncation: "
                    f"row={row_index}."
                )
            assert len(rollout_handler.input_ids) == len(rollout_handler.attention_mask) == len(rollout_handler.position_ids) == len(rollout_handler.loss_mask), f"""Rollout Handler has different length of {len(rollout_handler.input_ids)=},
            {len(rollout_handler.attention_mask)=}, {len(rollout_handler.position_ids)=}, {len(rollout_handler.loss_mask)=}"""
            assert len(rollout_handler.input_ids) <= self.config.max_model_len, f"Rollout Handler has sequence length {len(rollout_handler.input_ids)} > max_sequence_length {self.config.max_model_len}"

            response_ids.append(torch.tensor(rollout_handler.response_ids, dtype=torch.int, device=cur_device))
            response_attention_mask.append(torch.tensor(rollout_handler.response_attention_mask, dtype=torch.int, device=cur_device))
            response_position_ids.append(torch.tensor(rollout_handler.response_position_ids, dtype=torch.int, device=cur_device))
            response_loss_mask.append(torch.tensor(rollout_handler.response_loss_mask, dtype=torch.int, device=cur_device))
            scores.append(rollout_handler.score)

        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        response_length = response_ids.size(1)

        prompt_ids = [torch.tensor(handler.prompt_ids, dtype=torch.int, device=cur_device) for handler in rollout_handler_ls]
        prompt_attention_mask = [
            torch.tensor(handler.prompt_attention_mask, dtype=torch.int, device=cur_device)
            for handler in rollout_handler_ls
        ]
        prompt_position_ids = [
            torch.tensor(handler.prompt_position_ids, dtype=torch.int, device=cur_device)
            for handler in rollout_handler_ls
        ]
        prompt_width = self.config.prompt_length
        overlong_prompts = [
            (index, int(prompt.size(0)))
            for index, prompt in enumerate(prompt_ids)
            if prompt.size(0) > prompt_width
        ]
        if overlong_prompts:
            raise RuntimeError(
                "Generation prompt exceeds the packed PPO prompt width; "
                "refusing to train on a different token sequence: "
                f"prompt_width={prompt_width} rows={overlong_prompts[:8]}"
            )
        input_ids = left_pad_sequence_to_length(prompt_ids, prompt_width, self.pad_token_id)
        attention_mask = left_pad_sequence_to_length(prompt_attention_mask, prompt_width, 0)
        packed_prompt_attention_mask = attention_mask
        # Recompute positions after fixed-width padding/truncation so every
        # rollout worker returns the same prompt shape to DataProto.concat.
        position_ids = compute_position_id_with_mask(attention_mask)

        delta_position_ids = torch.arange(1, response_length + 1, device=cur_device)
        delta_position_ids = delta_position_ids.unsqueeze(0).repeat(len(rollout_handler_ls), 1)
        response_position_ids = position_ids[:, -1:] + delta_position_ids

        seq = torch.cat((input_ids, response_ids), dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((position_ids, response_position_ids), dim=-1)

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        valid_response_length = response_loss_mask.sum(dim=-1)
        if immediate_rewards is not None and any(
            int(length.item()) <= 0 for length in valid_response_length
        ):
            raise RuntimeError(
                "Formal reward placement found an action row without sampled assistant tokens."
            )
        for i, score in enumerate(scores):
            reward_tensor[i, valid_response_length[i].item() - 1] = score
        if immediate_rewards is not None:
            validate_formal_response_reward_placement(
                response_masks=response_loss_mask.detach().cpu().tolist(),
                score_rows=reward_tensor.detach().cpu().tolist(),
                expected_rewards=(
                    suffix_returns
                    if bool(suffix_credit_applied)
                    else immediate_rewards
                ),
                valid_mask=[True] * len(rollout_handler_ls),
            )

        packed_prompt_tokens = [
            input_ids[index][packed_prompt_attention_mask[index].bool()]
            .detach()
            .cpu()
            .tolist()
            for index in range(len(rollout_handler_ls))
        ]
        generation_prompt_lengths = [
            len(handler.prompt_ids) for handler in rollout_handler_ls
        ]
        generation_prompt_digests = [
            prompt_state_digest(handler.prompt_ids) for handler in rollout_handler_ls
        ]
        packed_prompt_lengths = [len(tokens) for tokens in packed_prompt_tokens]
        packed_prompt_digests = [
            prompt_state_digest(tokens) for tokens in packed_prompt_tokens
        ]
        packed_response_tokens = [
            response_ids[index][response_loss_mask[index].bool()]
            .detach()
            .cpu()
            .tolist()
            for index in range(len(rollout_handler_ls))
        ]
        packed_response_lengths = [
            len(tokens) for tokens in packed_response_tokens
        ]
        packed_response_digests = [
            prompt_state_digest(tokens) for tokens in packed_response_tokens
        ]

        task_round_tensor = torch.tensor(
            task_rounds if task_rounds is not None else [1] * len(rollout_handler_ls),
            dtype=torch.float32,
            device=input_ids.device,
        )

        batch = TensorDict(
            {
                'prompts': input_ids,
                'responses': response_ids,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'response_mask': response_loss_mask,
                'scores': reward_tensor,
                'task_rounds': task_round_tensor,
                'task_scores': reward_tensor,
            },
            batch_size=len(rollout_handler_ls),
        )
        non_tensor_batch = {}
        if parent_indices is not None:
            non_tensor_batch["rollout_parent_indices"] = np.array(parent_indices, dtype=object)
        if done_flags is not None:
            non_tensor_batch["rollout_done_flags"] = np.array(done_flags, dtype=object)
        formal_metadata = {
            AGENTMEMORY_PARENT_GROUP_UID: parent_group_uids,
            AGENTMEMORY_EXACT_STATE_UID: exact_state_uids,
            AGENTMEMORY_REPLICA_INDEX: replica_indices,
            AGENTMEMORY_TRAJECTORY_UID: trajectory_uids,
            AGENTMEMORY_TRAJECTORY_RETURN: trajectory_returns,
            AGENTMEMORY_IMMEDIATE_REWARD: immediate_rewards,
            AGENTMEMORY_TRAJECTORY_ROW_UID: trajectory_row_uids,
            AGENTMEMORY_TRAJECTORY_ROW_ORDER: trajectory_row_orders,
            AGENTMEMORY_TRAJECTORY_TERMINAL: trajectory_terminals,
            "task_rounds": task_rounds,
            AGENTMEMORY_ACTION_TEXT: action_texts,
            AGENTMEMORY_SUFFIX_RETURN: suffix_returns,
            AGENTMEMORY_STEP_RECORD_JSON: step_records,
        }
        present_formal_keys = {
            key for key, values in formal_metadata.items() if values is not None
        }
        if present_formal_keys:
            if len(present_formal_keys) != len(formal_metadata):
                missing = sorted(set(formal_metadata) - present_formal_keys)
                raise RuntimeError(
                    "Incomplete formal trajectory rollout metadata: "
                    f"missing={missing}"
                )
            metadata_lengths = {
                key: len(values) for key, values in formal_metadata.items()
            }
            if any(
                length != len(rollout_handler_ls)
                for length in metadata_lengths.values()
            ):
                raise RuntimeError(
                    "Formal trajectory rollout metadata length mismatch: "
                    f"rows={len(rollout_handler_ls)} metadata={metadata_lengths}"
                )
            batch[AGENTMEMORY_TRAJECTORY_RETURN] = torch.tensor(
                trajectory_returns, dtype=torch.float32, device=input_ids.device
            )
            batch[AGENTMEMORY_IMMEDIATE_REWARD] = torch.tensor(
                immediate_rewards, dtype=torch.float32, device=input_ids.device
            )
            batch[AGENTMEMORY_TRAJECTORY_ROW_ORDER] = torch.tensor(
                trajectory_row_orders, dtype=torch.long, device=input_ids.device
            )
            batch[AGENTMEMORY_TRAJECTORY_TERMINAL] = torch.tensor(
                trajectory_terminals, dtype=torch.bool, device=input_ids.device
            )
            batch[AGENTMEMORY_GENERATION_PROMPT_LENGTH] = torch.tensor(
                generation_prompt_lengths, dtype=torch.long, device=input_ids.device
            )
            batch[AGENTMEMORY_PACKED_PROMPT_LENGTH] = torch.tensor(
                packed_prompt_lengths, dtype=torch.long, device=input_ids.device
            )
            generation_response_tokens = [
                [int(token_id) for token_id in record["response_token_ids"]]
                for record in step_records
            ]
            generation_response_lengths = [
                len(tokens) for tokens in generation_response_tokens
            ]
            generation_response_digests = [
                prompt_state_digest(tokens) for tokens in generation_response_tokens
            ]
            batch[AGENTMEMORY_GENERATION_RESPONSE_LENGTH] = torch.tensor(
                generation_response_lengths,
                dtype=torch.long,
                device=input_ids.device,
            )
            batch[AGENTMEMORY_PACKED_RESPONSE_LENGTH] = torch.tensor(
                packed_response_lengths,
                dtype=torch.long,
                device=input_ids.device,
            )
            if suffix_credit_applied is None:
                raise RuntimeError(
                    "Formal trajectory rollout is missing suffix-credit readback."
                )
            suffix_credit_flags = [
                bool(suffix_credit_applied)
            ] * len(rollout_handler_ls)
            batch[AGENTMEMORY_SUFFIX_CREDIT_APPLIED] = torch.tensor(
                suffix_credit_flags, dtype=torch.bool, device=input_ids.device
            )
            batch[AGENTMEMORY_SUFFIX_RETURN] = torch.tensor(
                suffix_returns, dtype=torch.float32, device=input_ids.device
            )
            immediate_task_scores = torch.zeros_like(
                response_ids, dtype=torch.float32
            )
            for index, immediate_reward in enumerate(immediate_rewards):
                immediate_task_scores[
                    index, valid_response_length[index].item() - 1
                ] = float(immediate_reward)
            batch["task_scores"] = immediate_task_scores
            non_tensor_batch[AGENTMEMORY_PARENT_GROUP_UID] = np.array(
                parent_group_uids, dtype=object
            )
            non_tensor_batch[AGENTMEMORY_EXACT_STATE_UID] = np.array(
                exact_state_uids, dtype=object
            )
            non_tensor_batch[AGENTMEMORY_REPLICA_INDEX] = np.array(
                replica_indices, dtype=object
            )
            non_tensor_batch[AGENTMEMORY_TRAJECTORY_UID] = np.array(
                trajectory_uids, dtype=object
            )
            non_tensor_batch[AGENTMEMORY_TRAJECTORY_ROW_UID] = np.array(
                trajectory_row_uids, dtype=object
            )
            non_tensor_batch[AGENTMEMORY_ACTION_TEXT] = np.array(
                action_texts, dtype=object
            )
            non_tensor_batch[AGENTMEMORY_GENERATION_PROMPT_DIGEST] = np.array(
                generation_prompt_digests, dtype=object
            )
            non_tensor_batch[AGENTMEMORY_PACKED_PROMPT_DIGEST] = np.array(
                packed_prompt_digests, dtype=object
            )
            non_tensor_batch[AGENTMEMORY_GENERATION_RESPONSE_DIGEST] = np.array(
                generation_response_digests, dtype=object
            )
            non_tensor_batch[AGENTMEMORY_PACKED_RESPONSE_DIGEST] = np.array(
                packed_response_digests, dtype=object
            )
            canonical_step_records = []
            for index, raw_record in enumerate(step_records):
                if not isinstance(raw_record, dict):
                    raise RuntimeError(
                        f"Formal step record {index} must be a dictionary."
                    )
                record = deepcopy(raw_record)
                schema_version = record.get(
                    "schema_version", FORMAL_WEBSHOP_SCHEMA_V2
                )
                if schema_version not in (
                    FORMAL_WEBSHOP_SCHEMA_V2,
                    FORMAL_DOMAIN_SCHEMA_V3,
                ):
                    raise RuntimeError(
                        "Unsupported formal step schema before packing: "
                        f"row={index} schema={schema_version!r}."
                    )
                internal_fields = (
                    ("prompt_token_ids",)
                    if schema_version == FORMAL_DOMAIN_SCHEMA_V3
                    else ("prompt_token_ids", "content", "score")
                )
                for internal_field in internal_fields:
                    record.pop(internal_field, None)
                record.update(
                    {
                        "schema_version": schema_version,
                        "exact_state_uid": str(exact_state_uids[index]),
                        "trajectory_uid": str(trajectory_uids[index]),
                        "trajectory_row_uid": str(trajectory_row_uids[index]),
                        "trajectory_row_order": int(trajectory_row_orders[index]),
                        "trajectory_terminal": bool(trajectory_terminals[index]),
                        "task_round": int(task_rounds[index]),
                        "action": str(action_texts[index]),
                        "immediate_reward": float(immediate_rewards[index]),
                        "suffix_return": float(suffix_returns[index]),
                        "suffix_credit_applied": bool(suffix_credit_applied),
                        "trajectory_return": float(trajectory_returns[index]),
                        "done": bool(done_flags[index]),
                        "generation_prompt_length": int(
                            generation_prompt_lengths[index]
                        ),
                        "generation_prompt_digest": str(
                            generation_prompt_digests[index]
                        ),
                        "packed_prompt_length": int(packed_prompt_lengths[index]),
                        "packed_prompt_digest": str(packed_prompt_digests[index]),
                        "generation_response_length": int(
                            generation_response_lengths[index]
                        ),
                        "generation_response_digest": str(
                            generation_response_digests[index]
                        ),
                        "packed_response_length": int(
                            packed_response_lengths[index]
                        ),
                        "packed_response_digest": str(
                            packed_response_digests[index]
                        ),
                    }
                )
                canonical_step_records.append(
                    json.dumps(
                        record,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            non_tensor_batch[AGENTMEMORY_STEP_RECORD_JSON] = np.array(
                canonical_step_records, dtype=object
            )
            validate_formal_runtime_evidence_rows(
                exact_state_uids=exact_state_uids,
                trajectory_uids=trajectory_uids,
                trajectory_row_uids=trajectory_row_uids,
                trajectory_row_orders=trajectory_row_orders,
                trajectory_terminals=trajectory_terminals,
                task_rounds=task_rounds,
                immediate_rewards=immediate_rewards,
                trajectory_returns=trajectory_returns,
                action_texts=action_texts,
                done_flags=done_flags,
                generation_prompt_lengths=generation_prompt_lengths,
                generation_prompt_digests=generation_prompt_digests,
                packed_prompt_lengths=packed_prompt_lengths,
                packed_prompt_digests=packed_prompt_digests,
                generation_response_lengths=generation_response_lengths,
                generation_response_digests=generation_response_digests,
                packed_response_lengths=packed_response_lengths,
                packed_response_digests=packed_response_digests,
                suffix_credit_applied=suffix_credit_flags,
                suffix_returns=suffix_returns,
                step_record_jsons=canonical_step_records,
                valid_mask=[True] * len(rollout_handler_ls),
                expected_suffix_credit=bool(suffix_credit_applied),
                expected_prompt_width=int(self.config.prompt_length),
            )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def latest_observation_prompt_from_text(
        self,
        observation: str,
        *,
        system_prompt: str | None = None,
    ) -> List[int]:
        temp_handler = RolloutHandler(
            messages=[Message(role="user", content=observation)],
            task_name="agentmemory",
            item_id=0,
            score=0,
            done=False,
            input_ids=[],
            prompt_ids=[],
            response_ids=[],
            attention_mask=[],
            prompt_attention_mask=[],
            response_attention_mask=[],
            position_ids=[],
            prompt_position_ids=[],
            response_position_ids=[],
            loss_mask=[],
            prompt_loss_mask=[],
            response_loss_mask=[],
            max_response_len=self.config.response_length,
            max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
        )
        return temp_handler.get_latest_observation_prompt(
            self.tokenizer,
            system_prompt=system_prompt,
        )

    def generate_agentmemory_latest_observation(
        self,
        rollout_handler_ls: List[RolloutHandler],
        env_clients,
        cur_device,
        max_policy_turns: int,
        sampling_kwargs: dict,
        global_steps,
    ) -> DataProto:
        parent_indices = []
        flat_task_rounds = []
        flat_done_flags = []
        flat_handlers = []
        flat_rollout_indices = []
        excluded_rollout_indices = set()
        suffix_credit = _agentmemory_latest_observation_suffix_credit_enabled()
        state_aware_group_uid = _agentmemory_state_aware_group_uid_enabled()
        formal_trajectory_credit = True
        trajectory_steps = [[] for _ in rollout_handler_ls]
        flat_step_refs = []
        uid_overrides = []
        task_rounds = [0] * len(rollout_handler_ls)
        messages = [[] for _ in rollout_handler_ls]
        for idx, rollout_handler in enumerate(rollout_handler_ls):
            try:
                _agentmemory_debug(f"reset_start idx={idx} item_id={rollout_handler.item_id}")
                reset_started = time.time()
                env_clients[idx].reset(
                    getattr(
                        rollout_handler,
                        "agentmemory_local_data_idx",
                        getattr(
                            rollout_handler,
                            "data_idx",
                            rollout_handler.item_id,
                        ),
                    )
                )
                task = env_clients[idx].observe()
                rollout_handler.add_user_message(self.tokenizer, task)
                (
                    rollout_handler.formal_schema_version,
                    rollout_handler.formal_system_prompt,
                    rollout_handler.formal_system_prompt_source,
                ) = _formal_runtime_contract_for_client(env_clients[idx])
                _agentmemory_debug(f"reset_done idx={idx} item_id={rollout_handler.item_id} seconds={time.time() - reset_started:.2f}")
            except FormalRuntimeEvidenceError:
                raise
            except Exception as e:
                print(f"Reset Error: AgentMemory Env error={e} item id = {rollout_handler.item_id}", flush=True)
                rollout_handler.done = True
                rollout_handler.score = 0
                excluded_rollout_indices.add(idx)

        rounds = 0
        all_done_flag = False
        rollout_bar = tqdm(total=max_policy_turns, desc="Running AgentMemory policy turns", disable=torch.distributed.get_rank() != 0)
        while rounds < max_policy_turns and not all_done_flag:
            generation_prompt_idxs = []
            not_done_idxs = []
            for idx, rollout_handler in enumerate(rollout_handler_ls):
                if not rollout_handler.done:
                    generation_prompt_idxs.append(
                        rollout_handler.get_latest_observation_prompt(
                            self.tokenizer,
                            system_prompt=rollout_handler.formal_system_prompt,
                        )
                    )
                    not_done_idxs.append(idx)

            overlong_generation_prompts = [
                (not_done_idxs[index], len(prompt_token_ids))
                for index, prompt_token_ids in enumerate(generation_prompt_idxs)
                if len(prompt_token_ids) > int(self.config.prompt_length)
            ]
            if overlong_generation_prompts:
                raise RuntimeError(
                    "Generation prompt exceeds data.max_prompt_length before "
                    "sampling; refusing a later packed-prompt mismatch: "
                    f"prompt_width={self.config.prompt_length} "
                    f"rows={overlong_generation_prompts[:8]}"
                )

            rollout_bar.set_description(f"AgentMemory policy turns {rounds + 1}/{max_policy_turns} | Active agents per gpu: {len(not_done_idxs)}")
            _agentmemory_debug(f"round_start round={rounds + 1}/{max_policy_turns} active={len(not_done_idxs)}")
            if len(not_done_idxs) == 0:
                break
            with self.update_sampling_params(**sampling_kwargs):
                output = self._generate_token_ids(
                    generation_prompt_idxs=generation_prompt_idxs,
                    sampling_params=self.sampling_params,
                )
            generation_records = self._output_generation_records(
                output,
                self.sampling_params,
                require_exact_metadata=bool(formal_trajectory_credit),
            )
            if len(generation_records) != len(not_done_idxs):
                raise RuntimeError(
                    "Formal generation result count mismatch: "
                    f"active={len(not_done_idxs)} outputs={len(generation_records)}."
                )
            response_ids = [record["token_ids"] for record in generation_records]
            all_done_flag = True
            time.sleep(self.config.send_interval)
            for i, idx in enumerate(not_done_idxs):
                prompt_token_ids = generation_prompt_idxs[i]
                action_token_ids = response_ids[i]
                generation_record = generation_records[i]
                visible_prompt = self.tokenizer.decode(
                    prompt_token_ids, skip_special_tokens=False
                )
                latest_observation = rollout_handler_ls[idx].messages[-1].content
                formal_schema_version = str(
                    rollout_handler_ls[idx].formal_schema_version
                )
                formal_system_prompt = str(
                    rollout_handler_ls[idx].formal_system_prompt
                )
                single_observation_prompt_ids = (
                    self.latest_observation_prompt_from_text(
                        latest_observation,
                        system_prompt=formal_system_prompt,
                    )
                )
                if single_observation_prompt_ids != list(prompt_token_ids):
                    raise FormalRuntimeEvidenceError(
                        "Formal generation prompt contains context beyond the latest "
                        f"observation for item {rollout_handler_ls[idx].item_id}."
                    )
                env_info_payload = getattr(env_clients[idx], "info", None)
                env_info_before = (
                    deepcopy(env_info_payload.get("env_info", {}))
                    if isinstance(env_info_payload, dict)
                    else {}
                )
                _validate_runtime_env_schema(
                    formal_schema_version,
                    env_info_before,
                    boundary="pre-action",
                )
                if formal_schema_version == FORMAL_WEBSHOP_SCHEMA_V2:
                    if "current_subtask_index" not in env_info_before:
                        raise FormalRuntimeEvidenceError(
                            "Formal AgentMemory step is missing the pre-action "
                            f"session index for item {rollout_handler_ls[idx].item_id}."
                        )
                    session_index = int(env_info_before["current_subtask_index"])
                else:
                    if "phase_index" not in env_info_before:
                        raise FormalRuntimeEvidenceError(
                            "Formal v3 step is missing the pre-action phase_index "
                            f"for item {rollout_handler_ls[idx].item_id}."
                        )
                    phase_index = int(env_info_before["phase_index"])
                content = self.tokenizer.decode(action_token_ids, skip_special_tokens=True)
                rollout_handler_ls[idx].add_assistant_message(self.tokenizer, content)
                task_rounds[idx] += 1
                parent_index = getattr(rollout_handler_ls[idx], "parent_index", idx // self.config.n)
                replica_index = int(
                    getattr(rollout_handler_ls[idx], "rollout_replica_index", 0)
                )
                exact_state_uid = build_state_aware_rollout_uid(
                    parent_index,
                    task_rounds[idx],
                    prompt_token_ids,
                )
                parent_group_uid = build_parent_group_uid(parent_index)
                trajectory_uid = build_trajectory_uid(
                    parent_group_uid, replica_index
                )
                try:
                    _agentmemory_debug(f"step_start round={rounds + 1} local_i={i} idx={idx} item_id={rollout_handler_ls[idx].item_id}")
                    step_started = time.time()
                    step_output = env_clients[idx].step(content)
                    _agentmemory_debug(f"step_done round={rounds + 1} local_i={i} idx={idx} item_id={rollout_handler_ls[idx].item_id} seconds={time.time() - step_started:.2f} reward={step_output.reward} done={step_output.done}")
                    state, rollout_handler_ls[idx].score, rollout_handler_ls[idx].done = (
                        step_output.state,
                        step_output.reward,
                        step_output.done,
                    )
                    if getattr(env_clients[idx], "sample_excluded", False):
                        excluded_rollout_indices.add(idx)
                        rollout_handler_ls[idx].score = 0
                        rollout_handler_ls[idx].done = True
                        _agentmemory_debug(
                            f"step_infra_excluded round={rounds + 1} idx={idx} "
                            f"item_id={rollout_handler_ls[idx].item_id}"
                        )
                        continue
                    env_info_payload = getattr(env_clients[idx], "info", None)
                    env_info_after = (
                        deepcopy(env_info_payload.get("env_info", {}))
                        if isinstance(env_info_payload, dict)
                        else {}
                    )
                    _validate_runtime_env_schema(
                        formal_schema_version,
                        env_info_after,
                        boundary="post-action",
                    )
                    prompt_digest = prompt_state_digest(
                        single_observation_prompt_ids
                    )
                    if formal_schema_version == FORMAL_DOMAIN_SCHEMA_V3:
                        if "phase_index" not in env_info_after:
                            raise FormalRuntimeEvidenceError(
                                "Formal v3 step is missing the post-action phase_index "
                                f"for item {rollout_handler_ls[idx].item_id}."
                            )
                        if int(env_info_after["phase_index"]) < phase_index:
                            raise FormalRuntimeEvidenceError(
                                "Formal v3 phase_index regressed across one action."
                            )
                        try:
                            step_record = build_formal_domain_step_v3(
                                content=content,
                                score=float(rollout_handler_ls[idx].score),
                                task_round=task_rounds[idx],
                                done=bool(step_output.done),
                                item_id=str(rollout_handler_ls[idx].item_id),
                                parent_index=parent_index,
                                parent_group_uid=parent_group_uid,
                                replica_index=replica_index,
                                trajectory_uid=trajectory_uid,
                                exact_state_uid=exact_state_uid,
                                prompt_token_ids=prompt_token_ids,
                                response_token_ids=action_token_ids,
                                latest_observation=latest_observation,
                                visible_prompt=visible_prompt,
                                system_prompt=formal_system_prompt,
                                single_observation_prompt_digest=prompt_digest,
                                env_result=str(step_output.state),
                                generation_record=generation_record,
                                env_info_before=env_info_before,
                                env_info_after=env_info_after,
                            )
                        except FormalDomainV3Error as exc:
                            raise FormalRuntimeEvidenceError(
                                "Formal v3 step evidence is invalid for "
                                f"item={rollout_handler_ls[idx].item_id}: {exc}"
                            ) from exc
                    else:
                        step_record = _build_formal_webshop_step_v2(
                            content=content,
                            score=float(rollout_handler_ls[idx].score),
                            task_round=task_rounds[idx],
                            done=bool(step_output.done),
                            item_id=str(rollout_handler_ls[idx].item_id),
                            parent_index=parent_index,
                            parent_group_uid=parent_group_uid,
                            replica_index=replica_index,
                            trajectory_uid=trajectory_uid,
                            exact_state_uid=exact_state_uid,
                            prompt_token_ids=prompt_token_ids,
                            response_token_ids=list(action_token_ids),
                            latest_observation=latest_observation,
                            visible_prompt=visible_prompt,
                            single_observation_prompt_digest=prompt_digest,
                            env_result=str(step_output.state),
                            generation_record=generation_record,
                            env_info_before=env_info_before,
                            env_info_after=env_info_after,
                            action_submission=getattr(
                                env_clients[idx], "last_action_submission", None
                            ),
                        )
                    if suffix_credit or formal_trajectory_credit:
                        step_record["trajectory_row_order"] = len(
                            trajectory_steps[idx]
                        )
                        step_record["trajectory_row_uid"] = build_row_uid(
                            trajectory_uid, step_record["trajectory_row_order"]
                        )
                        trajectory_steps[idx].append(step_record)
                    if suffix_credit:
                        pass
                    else:
                        flat_handlers.append(
                            self.build_rollout_handler_from_prompt(
                                prompt_token_ids=prompt_token_ids,
                                content=content,
                                score=rollout_handler_ls[idx].score,
                                parent_index=parent_index,
                                sampled_response_token_ids=action_token_ids,
                            )
                        )
                        parent_indices.append(parent_index)
                        flat_task_rounds.append(task_rounds[idx])
                        flat_done_flags.append(bool(step_output.done))
                        flat_rollout_indices.append(idx)
                        if formal_trajectory_credit:
                            flat_step_refs.append(step_record)
                        if state_aware_group_uid:
                            uid_overrides.append(exact_state_uid)
                    rollout_handler_ls[idx].add_user_message(self.tokenizer, state)
                    all_done_flag = all_done_flag and step_output.done
                except FormalRuntimeEvidenceError:
                    raise
                except Exception as e:
                    if getattr(env_clients[idx], "sample_excluded", False):
                        rollout_handler_ls[idx].score = 0
                        rollout_handler_ls[idx].done = True
                        excluded_rollout_indices.add(idx)
                        _agentmemory_debug(
                            f"step_infra_exception_excluded round={rounds + 1} "
                            f"idx={idx} item_id={rollout_handler_ls[idx].item_id} "
                            f"error={type(e).__name__}: {e}"
                        )
                        continue
                    if formal_trajectory_credit:
                        raise FormalRuntimeEvidenceError(
                            "Formal AgentMemory environment step failed before an "
                            "auditable action/reward boundary: "
                            f"item={rollout_handler_ls[idx].item_id} "
                            f"task_round={task_rounds[idx]} action={content!r} "
                            f"error={type(e).__name__}: {e}"
                        ) from e
                    _agentmemory_debug(f"step_error round={rounds + 1} local_i={i} idx={idx} item_id={rollout_handler_ls[idx].item_id} error={e}")
                    rollout_handler_ls[idx].score = 0
                    rollout_handler_ls[idx].done = True
                    error_text = f"{type(e).__name__}: {e}"
                    step_record = {
                        "prompt_token_ids": prompt_token_ids,
                        "content": content,
                        "score": 0.0,
                        "parent_index": parent_index,
                        "parent_group_uid": parent_group_uid,
                        "replica_index": replica_index,
                        "trajectory_uid": trajectory_uid,
                        "exact_state_uid": exact_state_uid,
                        "task_round": task_rounds[idx],
                        "done": True,
                        "item_id": str(rollout_handler_ls[idx].item_id),
                        "session_index": session_index,
                        "subtask_index": session_index,
                        "next_session_index": session_index,
                        "subtask_index_before": session_index,
                        "subtask_index_after": session_index,
                        "visible_prompt": visible_prompt,
                        "latest_observation": latest_observation,
                        "prompt_history_policy": "latest_observation_only",
                        "raw_prior_messages_visible": False,
                        "single_observation_prompt_digest": prompt_state_digest(
                            single_observation_prompt_ids
                        ),
                        "response_token_ids": list(action_token_ids),
                        "response_token_count": int(
                            generation_record["response_token_count"]
                        ),
                        "max_response_tokens": int(
                            generation_record["max_response_tokens"]
                        ),
                        "finish_reason": str(
                            generation_record["finish_reason"]
                        ),
                        "finish_reason_source": str(
                            generation_record["finish_reason_source"]
                        ),
                        "stop_reason": generation_record.get("stop_reason"),
                        "generation_backend_source": str(
                            generation_record["backend_source"]
                        ),
                        "generation_stop_reason": generation_record.get(
                            "stop_reason"
                        ),
                        "generation_eos_token_ids": list(
                            generation_record["configured_eos_token_ids"]
                        ),
                        "tokenizer_primary_eos_token_id": generation_record[
                            "primary_eos_token_id"
                        ],
                        "tokenizer_pad_token_id": generation_record[
                            "tokenizer_pad_token_id"
                        ],
                        "generation_token_ids_are_exact": bool(
                            generation_record["token_ids_are_exact"]
                        ),
                        "backend_token_ids_are_exact": bool(
                            generation_record["backend_token_ids_are_exact"]
                        ),
                        "truncated": bool(generation_record["truncated"]),
                        "env_result": error_text,
                        "env_info_before": env_info_before,
                        "env_info_after": deepcopy(env_info_before),
                        "action_execution": None,
                        "committed_purchase": False,
                        "purchase_correct": None,
                        "accepted_purchase": False,
                        "session_advanced": False,
                        "buy_committed": False,
                        "buy_accepted": False,
                        "subtask_advanced": False,
                        "raw_history_cleared": False,
                        "search_result_count": None,
                        "outcome": "environment_error",
                    }
                    if suffix_credit or formal_trajectory_credit:
                        step_record["trajectory_row_order"] = len(
                            trajectory_steps[idx]
                        )
                        step_record["trajectory_row_uid"] = build_row_uid(
                            trajectory_uid, step_record["trajectory_row_order"]
                        )
                        trajectory_steps[idx].append(step_record)
                    if suffix_credit:
                        pass
                    else:
                        flat_handlers.append(
                            self.build_rollout_handler_from_prompt(
                                prompt_token_ids=prompt_token_ids,
                                content=content,
                                score=0,
                                parent_index=parent_index,
                                sampled_response_token_ids=action_token_ids,
                            )
                        )
                        parent_indices.append(parent_index)
                        flat_task_rounds.append(task_rounds[idx])
                        flat_done_flags.append(True)
                        flat_rollout_indices.append(idx)
                        if formal_trajectory_credit:
                            flat_step_refs.append(step_record)
                        if state_aware_group_uid:
                            uid_overrides.append(exact_state_uid)
                    print(f"AgentMemory rollout step error: {e} item id = {rollout_handler_ls[idx].item_id}")
            rounds += 1
            rollout_bar.update(1)
        rollout_bar.close()

        from agentenv_agentmemory.reward_hierarchy import (
            bind_max_round_timeout_failure,
        )

        if excluded_rollout_indices:
            rollout_parent_indices = [
                int(
                    getattr(
                        handler,
                        "parent_index",
                        index // self.config.n,
                    )
                )
                for index, handler in enumerate(rollout_handler_ls)
            ]
            excluded_rollout_indices = expand_excluded_rollout_parent_groups(
                rollout_parent_indices,
                excluded_rollout_indices,
            )

        for trajectory_index, steps in enumerate(trajectory_steps):
            if trajectory_index in excluded_rollout_indices:
                continue
            if not steps:
                continue
            if not steps[-1]["done"] and rounds >= max_policy_turns:
                if steps[-1].get("schema_version") == FORMAL_DOMAIN_SCHEMA_V3:
                    bind_generic_timeout_v3(
                        steps[-1],
                        max_policy_turns=max_policy_turns,
                    )
                else:
                    bind_max_round_timeout_failure(
                        steps[-1],
                        max_rounds=max_policy_turns,
                    )
                rollout_handler_ls[trajectory_index].score = steps[-1]["score"]
            suffix_scores = compute_suffix_credit_scores(
                [step["score"] for step in steps],
                [step["task_round"] for step in steps],
            )
            trajectory_return = sum(float(step["score"]) for step in steps)
            for step in steps:
                step["trajectory_terminal"] = False
            steps[-1]["trajectory_terminal"] = True
            for step, suffix_score in zip(steps, suffix_scores):
                step["suffix_return"] = float(suffix_score)
                step["trajectory_return"] = float(trajectory_return)

        if formal_trajectory_credit and not suffix_credit:
            for handler, step in zip(flat_handlers, flat_step_refs):
                handler.score = float(step["score"])
                handler.done = bool(step["done"])
            flat_done_flags = [bool(step["done"]) for step in flat_step_refs]

        if suffix_credit:
            for trajectory_index, steps in enumerate(trajectory_steps):
                if trajectory_index in excluded_rollout_indices:
                    continue
                if not steps:
                    continue
                for step in steps:
                    flat_handlers.append(
                        self.build_rollout_handler_from_prompt(
                            prompt_token_ids=step["prompt_token_ids"],
                            content=step["content"],
                            score=step["suffix_return"],
                            parent_index=step["parent_index"],
                            sampled_response_token_ids=step["response_token_ids"],
                        )
                    )
                    parent_indices.append(step["parent_index"])
                    flat_task_rounds.append(step["task_round"])
                    flat_done_flags.append(bool(step["done"]))
                    flat_rollout_indices.append(trajectory_index)
                    if formal_trajectory_credit:
                        flat_step_refs.append(step)
                    if state_aware_group_uid:
                        uid_overrides.append(step["exact_state_uid"])

        if excluded_rollout_indices:
            keep_positions = trainable_rollout_row_positions(
                flat_rollout_indices,
                excluded_rollout_indices,
            )
            flat_handlers = [flat_handlers[position] for position in keep_positions]
            parent_indices = [parent_indices[position] for position in keep_positions]
            flat_task_rounds = [flat_task_rounds[position] for position in keep_positions]
            flat_done_flags = [flat_done_flags[position] for position in keep_positions]
            flat_rollout_indices = [
                flat_rollout_indices[position] for position in keep_positions
            ]
            if formal_trajectory_credit:
                flat_step_refs = [
                    flat_step_refs[position] for position in keep_positions
                ]
            if state_aware_group_uid:
                uid_overrides = [
                    uid_overrides[position] for position in keep_positions
                ]
            print(
                "AgentMemory excluded infrastructure-failed rollout parent "
                "groups from PPO rows: "
                f"indices={sorted(excluded_rollout_indices)}",
                flush=True,
            )

        if not flat_handlers:
            raise RuntimeError("AgentMemory latest-observation rollout produced no trainable action samples.")

        for idx, rollout_handler in enumerate(rollout_handler_ls):
            messages[idx] = rollout_handler.messages
        if global_steps:
            try:
                _agentmemory_debug(f"rollout_log_write_start step={global_steps}")
                os.makedirs(os.path.join(self.config.rollout_log_dir, f"step{global_steps}"), exist_ok=True)
                with open(os.path.join(self.config.rollout_log_dir, f"step{global_steps}/{torch.distributed.get_rank()}.json"), "w") as f:
                    json_msg = []
                    for idx, msgs in enumerate(messages):
                        records = {
                            "item_id": rollout_handler_ls[idx].item_id,
                            "conversations": [msg.to_dict() for msg in msgs],
                            "reward": rollout_handler_ls[idx].score,
                            "task_rounds": task_rounds[idx],
                            "sample_excluded": idx in excluded_rollout_indices,
                        }
                        json_msg.append(records)
                    json.dump(json_msg, f, ensure_ascii=True, indent=4)
                _agentmemory_debug(f"rollout_log_write_done step={global_steps} records={len(json_msg)}")
            except Exception as e:
                print(e, flush=True)
        formal_pack_kwargs = {}
        if formal_trajectory_credit:
            empty_trajectory_indices = [
                index
                for index, steps in enumerate(trajectory_steps)
                if index not in excluded_rollout_indices and not steps
            ]
            if empty_trajectory_indices:
                raise RuntimeError(
                    "Formal trajectory rollout is missing sampled action rows: "
                    f"replica_rows={empty_trajectory_indices[:8]}"
                )
            if len(flat_step_refs) != len(flat_handlers):
                raise RuntimeError(
                    "Formal trajectory row reference count mismatch: "
                    f"refs={len(flat_step_refs)} handlers={len(flat_handlers)}"
                )
            trajectory_return_by_uid = {}
            for steps in trajectory_steps:
                if not steps:
                    continue
                trajectory_uid = steps[0]["trajectory_uid"]
                trajectory_return_by_uid[trajectory_uid] = sum(
                    float(step["score"]) for step in steps
                )
            parent_group_uids = [
                step["parent_group_uid"] for step in flat_step_refs
            ]
            exact_state_uids = [
                step["exact_state_uid"] for step in flat_step_refs
            ]
            replica_indices = [step["replica_index"] for step in flat_step_refs]
            trajectory_uids = [step["trajectory_uid"] for step in flat_step_refs]
            trajectory_returns = [
                trajectory_return_by_uid[step["trajectory_uid"]]
                for step in flat_step_refs
            ]
            immediate_rewards = [float(step["score"]) for step in flat_step_refs]
            trajectory_row_uids = [
                str(step["trajectory_row_uid"]) for step in flat_step_refs
            ]
            trajectory_row_orders = [
                int(step["trajectory_row_order"]) for step in flat_step_refs
            ]
            trajectory_terminals = [
                bool(step["trajectory_terminal"]) for step in flat_step_refs
            ]
            suffix_returns = [
                float(step["suffix_return"]) for step in flat_step_refs
            ]
            action_texts = [str(step["content"]) for step in flat_step_refs]
            validate_formal_trajectory_rows(
                parent_group_uids=parent_group_uids,
                exact_state_uids=exact_state_uids,
                replica_indices=replica_indices,
                trajectory_uids=trajectory_uids,
                trajectory_returns=trajectory_returns,
                immediate_rewards=immediate_rewards,
                trajectory_row_uids=trajectory_row_uids,
                trajectory_row_orders=trajectory_row_orders,
                trajectory_terminals=trajectory_terminals,
                parent_indices=parent_indices,
                rollout_uids=uid_overrides,
                valid_mask=[True] * len(flat_step_refs),
                expected_replicas=int(self.config.n),
            )
            formal_pack_kwargs = {
                "parent_group_uids": parent_group_uids,
                "exact_state_uids": exact_state_uids,
                "replica_indices": replica_indices,
                "trajectory_uids": trajectory_uids,
                "trajectory_returns": trajectory_returns,
                "immediate_rewards": immediate_rewards,
                "trajectory_row_uids": trajectory_row_uids,
                "trajectory_row_orders": trajectory_row_orders,
                "trajectory_terminals": trajectory_terminals,
                "task_rounds": flat_task_rounds,
                "action_texts": action_texts,
                "suffix_credit_applied": bool(suffix_credit),
                "suffix_returns": suffix_returns,
                "step_records": flat_step_refs,
            }
        output = self.pack_rollout_handlers(
            flat_handlers,
            cur_device=cur_device,
            parent_indices=parent_indices,
            done_flags=flat_done_flags,
            **formal_pack_kwargs,
        )
        output.batch['task_rounds'] = torch.tensor(
            flat_task_rounds,
            dtype=torch.float32,
            device=output.batch['input_ids'].device,
        )
        if state_aware_group_uid:
            if len(uid_overrides) != len(output):
                raise RuntimeError(
                    "State-aware AgentMemory grouping produced a uid count mismatch: "
                    f"uids={len(uid_overrides)} rollout_rows={len(output)}"
                )
            output.non_tensor_batch["rollout_uid"] = np.array(uid_overrides, dtype=object)
        return output


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self._maybe_resume_engine()

        global_steps = prompts.meta_info.get('global_steps', None)
        max_policy_turns = prompts.meta_info.get(
            'max_policy_turns',
            prompts.meta_info.get('max_rounds', 10),
        )
        cur_device = prompts.batch["input_ids"].device

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # repeat for self.config.n times to rollout
        batch_size = prompts.batch['input_ids'].size(0)
        batch_size *= self.config.n
        rollout_handler_ls = self.preprocess_prompt_to_rollout_handler(prompts, n=self.config.n)
        assert_rollout_context_supported(self.agentgym_config)
        multitask_env_addrs = configured_multitask_env_addrs(
            self.agentgym_config
        )
        if multitask_env_addrs:
            missing_slots = [
                index
                for index, handler in enumerate(rollout_handler_ls)
                if not hasattr(handler, "agentmemory_surface_slot")
            ]
            if missing_slots:
                raise RuntimeError(
                    "multitask endpoints were configured but rollout rows are "
                    f"missing surface slots: rows={missing_slots[:8]}"
                )
            env_clients = [
                init_env_client(
                    self.agentgym_config,
                    env_addr=env_addr_for_surface_slot(
                        self.agentgym_config,
                        handler.agentmemory_surface_slot,
                    ),
                )
                for handler in rollout_handler_ls
            ]
        else:
            routed_rows = [
                index
                for index, handler in enumerate(rollout_handler_ls)
                if hasattr(handler, "agentmemory_surface_slot")
            ]
            if routed_rows:
                raise RuntimeError(
                    "multitask rollout rows require configured endpoints: "
                    f"rows={routed_rows[:8]}"
                )
            env_clients = [
                init_env_client(self.agentgym_config)
                for _ in range(batch_size)
            ]
        time.sleep(self.config.send_interval) # take a break before sendng request
        task_name = str(read_config(self.agentgym_config, "task_name", "")).lower()
        if task_name == "agentmemory" and rollout_context_policy(self.agentgym_config) == "latest_observation_only":
            output = self.generate_agentmemory_latest_observation(
                rollout_handler_ls=rollout_handler_ls,
                env_clients=env_clients,
                cur_device=cur_device,
                max_policy_turns=max_policy_turns,
                sampling_kwargs=kwargs,
                global_steps=global_steps,
            )
            for close_idx, client in enumerate(env_clients):
                try:
                    _agentmemory_debug(f"close_start idx={close_idx}")
                    close_started = time.time()
                    client.close()
                    _agentmemory_debug(f"close_done idx={close_idx} seconds={time.time() - close_started:.2f}")
                except Exception as e:
                    print(f"Error during closing env idx={close_idx}: {e}", flush=True)
            if self.config.free_cache_engine:
                self._maybe_release_engine()
            return output

        all_done_flag = False
        for idx, rollout_handler in enumerate(rollout_handler_ls):
            try:
                env_clients[idx].reset(rollout_handler.item_id)
                task = env_clients[idx].observe()
                rollout_handler.add_user_message(self.tokenizer, task)
            except TimeoutError:
                print(f"Reset Timeout: Webarena Env Timeout. item id = {rollout_handler.item_id}")
                rollout_handler.done = True
                rollout_handler.score = 0

        rounds = 0
        task_rounds = [0] * batch_size
        rollout_bar = tqdm(total = max_policy_turns, desc="Running rounds", disable=torch.distributed.get_rank() != 0)
        def agent_step(i, idx):
            content = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)
            rollout_handler_ls[idx].add_assistant_message(self.tokenizer, content)
            task_rounds[idx] += 1
            try:
                step_output = env_clients[idx].step(content)
                state, rollout_handler_ls[idx].score, rollout_handler_ls[idx].done = (
                    step_output.state,
                    step_output.reward,
                    step_output.done,
                )
                rollout_handler_ls[idx].add_user_message(self.tokenizer, state)
                return step_output.done
            except Exception as e:
                rollout_handler_ls[idx].score = 0
                rollout_handler_ls[idx].done = True
                print(f"Rollou step Error: {e} item id = {rollout_handler_ls[idx].item_id}")
                return True
        while rounds < max_policy_turns and not all_done_flag:
            # get generation prompt
            generation_prompt_idxs = []
            not_done_idxs = []
            for idx, rollout_handler in enumerate(rollout_handler_ls):
                if not rollout_handler.done:
                    generation_prompt_idxs.append(rollout_handler.get_generation_prompt(self.tokenizer))
                    not_done_idxs.append(idx)

            rollout_bar.set_description(f"Rounds {rounds + 1}/{max_policy_turns} | Active agents per gpu: {len(not_done_idxs)}")
            # users can customize different sampling_params at different run
            with self.update_sampling_params(**kwargs):
                output = self.inference_engine.generate(
                    prompts=None,
                    prompt_token_ids=generation_prompt_idxs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False)
            response_ids = self._output_token_id_lists(output)
            all_done_flag = True
            time.sleep(self.config.send_interval) # take a break before sendng request
            if len(not_done_idxs) > 0:
                with ThreadPoolExecutor(max_workers=len(not_done_idxs)) as executor:
                    step_dones = list(executor.map(
                        lambda args: agent_step(*args), [(i, idx) for i, idx in enumerate(not_done_idxs)]
                    ))
                    all_done_flag = all(step_dones)
            rounds += 1
            rollout_bar.update(1)

        # process ids
        rollout_bar.close()
        response_ids, response_attention_mask, response_position_ids, response_loss_mask = [], [], [], []
        scores, messages = [], []

        for rollout_handler in rollout_handler_ls:
            # check length
            rollout_handler.truncate_output_ids()
            assert len(rollout_handler.input_ids) == len(rollout_handler.attention_mask) == len(rollout_handler.position_ids) == len(rollout_handler.loss_mask), f"""Rollout Handler has different length of {len(rollout_handler.input_ids)=},
            {len(rollout_handler.attention_mask)=}, {len(rollout_handler.position_ids)=}, {len(rollout_handler.loss_mask)=}"""
            assert len(rollout_handler.input_ids) <= self.config.max_model_len, f"Rollout Handler has sequence length {len(rollout_handler.input_ids)} > max_sequence_length {self.config.max_model_len}"

            response_ids.append(torch.tensor(rollout_handler.response_ids, dtype=torch.int, device=cur_device))
            response_attention_mask.append(torch.tensor(rollout_handler.response_attention_mask, dtype=torch.int, device=cur_device))
            response_position_ids.append(torch.tensor(rollout_handler.response_position_ids, dtype=torch.int, device=cur_device))
            response_loss_mask.append(torch.tensor(rollout_handler.response_loss_mask, dtype=torch.int, device=cur_device))
            scores.append(rollout_handler.score)
            messages.append(rollout_handler.messages)

        # pad to length
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        response_length = response_ids.size(1)
        delta_position_ids = torch.arange(1, response_length + 1, device=cur_device)
        delta_position_ids = delta_position_ids.unsqueeze(0).repeat(batch_size, 1)
        input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        prompt_length = input_ids.size(-1)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        input_ids = input_ids.repeat_interleave(self.config.n, dim=0)
        attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
        position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
        response_position_ids = position_ids[:, -1:] + delta_position_ids

        seq = torch.cat((input_ids, response_ids), dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((position_ids, response_position_ids), dim=-1)
        response_mask = response_loss_mask

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32) # (bs, response_length)
        valid_response_length = attention_mask[:, prompt_length:].sum(dim=-1)
        for i in range(len(scores)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        if global_steps:
            try:
                os.makedirs(os.path.join(self.config.rollout_log_dir, f"step{global_steps}"), exist_ok=True)
                with open(os.path.join(self.config.rollout_log_dir, f"step{global_steps}/{torch.distributed.get_rank()}.json"), "w") as f:
                    json_msg = []
                    for idx, msgs in enumerate(messages):
                        records = {
                            "item_id": rollout_handler_ls[idx].item_id,
                            "conversations": [msg.to_dict() for msg in msgs],
                            "reward": scores[idx]
                        }
                        json_msg.append(records)
                    json.dump(json_msg, f, ensure_ascii=True, indent=4)
            except Exception as e:
                print(e)

        # close clients
        for client in env_clients:
            try:
                client.close()
            except Exception as e:
                print(f"Error during closing env: {e}")

        batch = TensorDict(
            {
                'prompts': input_ids,
                'responses': response_ids,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'response_mask': response_mask,
                'scores': reward_tensor,
                'task_rounds': torch.tensor(task_rounds, dtype=torch.float32).to(input_ids.device),
                'task_scores': reward_tensor
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self._maybe_release_engine()

        return DataProto(batch=batch)
