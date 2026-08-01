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

import os
import logging
import time
import numpy as np
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage
from verl.third_party.vllm import vllm_version

from .base import BaseShardingManager
from .vllm_sync_evidence import (
    append_and_readback_event,
    bounded_tensor_fingerprint,
    build_sync_event,
    read_last_event,
    validate_sync_event,
)
from verl.workers.qwen35_weight_sync import (
    map_actor_weight_name_for_vllm,
    validate_qwen35_mapped_source_names,
    validate_qwen35_vllm_load_coverage,
)
from verl.models.transformers.qwen3_5 import is_qwen3_5_model_type

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 sync_weight_format: str = 'dtensor',
                 expected_source_names=None,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        self.sync_weight_format = str(sync_weight_format).lower()
        if self.sync_weight_format not in ('hf', 'dtensor'):
            raise ValueError(f'Unsupported rollout sync format: {self.sync_weight_format}')
        self.expected_source_names = tuple(expected_source_names or ())

        self.full_params = self.sync_weight_format == 'hf'
        if self.full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False))
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self._sync_evidence_dir = os.getenv('VERL_AGENTMEMORY_VLLM_SYNC_EVIDENCE_DIR')
        self._require_post_update_change = (
            os.getenv('VERL_AGENTMEMORY_REQUIRE_VLLM_POST_UPDATE_CHANGE', '0') == '1')
        if self._require_post_update_change and not self._sync_evidence_dir:
            raise ValueError(
                'VERL_AGENTMEMORY_REQUIRE_VLLM_POST_UPDATE_CHANGE=1 requires '
                'VERL_AGENTMEMORY_VLLM_SYNC_EVIDENCE_DIR')
        self._sync_meta_info = None
        self._sync_sequence = 0
        self._sync_state_loaded = False
        self._last_sync_event = None
        self._current_sync_event = None
        self._current_sync_previous = None

    def _sync_rank(self):
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def _sync_evidence_path(self):
        if not self._sync_evidence_dir:
            return None
        return os.path.join(self._sync_evidence_dir, f'vllm_sync_rank{self._sync_rank()}.jsonl')

    def set_sync_context(self, meta_info):
        """Bind the next sync to the exact rollout request that will consume it."""
        self._sync_meta_info = meta_info
        self._current_sync_event = None
        self._current_sync_previous = None
        if self._sync_evidence_dir and 'global_steps' not in meta_info:
            raise RuntimeError(
                'VLLM sync evidence requires prompts.meta_info["global_steps"]')

    def _load_sync_state(self):
        if self._sync_state_loaded or not self._sync_evidence_dir:
            return
        path = self._sync_evidence_path()
        if os.path.exists(path):
            self._last_sync_event = read_last_event(path)
            validate_sync_event(self._last_sync_event)
            self._sync_sequence = int(self._last_sync_event['sync_sequence'])
        self._sync_state_loaded = True

    def _next_sync_identity(self):
        if self._sync_meta_info is None:
            raise RuntimeError('set_sync_context must run before vLLM weight synchronization')
        self._load_sync_state()
        self._sync_sequence += 1
        global_steps = self._sync_meta_info.get('global_steps')
        if hasattr(global_steps, 'item'):
            global_steps = global_steps.item()
        if not isinstance(global_steps, (str, int, float, bool)) and global_steps is not None:
            global_steps = str(global_steps)
        sync_id = f'rank{self._sync_rank()}:pid{os.getpid()}:seq{self._sync_sequence}:step{global_steps}'
        self._sync_meta_info['vllm_sync_sequence'] = self._sync_sequence
        self._sync_meta_info['vllm_sync_id'] = sync_id
        return global_steps, sync_id

    def validate_sync_before_generation(self):
        """Fail closed after sync evidence is durable and before generation starts."""
        if not self._sync_evidence_dir:
            return None
        if self._current_sync_event is None:
            raise RuntimeError('No vLLM sync evidence was recorded for the current generation')
        readback = read_last_event(self._sync_evidence_path())
        if readback.get('sync_id') != self._current_sync_event.get('sync_id'):
            raise RuntimeError('Latest vLLM sync evidence does not belong to the current generation')
        require_change = self._require_post_update_change and readback['sync_sequence'] >= 2
        validate_sync_event(
            readback,
            previous_event=self._current_sync_previous,
            require_change=require_change,
        )
        return readback

    def _get_infer_tp_size(self) -> int:
        """Return rollout tensor parallel size without requiring vLLM TP group in this worker.

        Official vLLM V1 owns its tensor-parallel process group inside the
        engine process. AgentMemoryGym g5a pilots use
        rollout.tensor_model_parallel_size=1, so worker-side data
        gather/broadcast is a no-op and must not query vLLM's uninitialized
        TP group. For TP>1 we still fall back to the original vLLM group path.
        """
        if self.device_mesh is not None:
            candidate_getters = (
                lambda: self.device_mesh["infer_tp"].size(),
                lambda: self.device_mesh.size(mesh_dim="infer_tp"),
                lambda: self.device_mesh.size(-1),
            )
            for getter in candidate_getters:
                try:
                    value = getter()
                    if value is not None:
                        return int(value)
                except Exception:
                    pass
        try:
            return int(vllm_ps.get_tensor_model_parallel_world_size())
        except AssertionError as exc:
            if "tensor model parallel group is not initialized" in str(exc):
                logger.warning(
                    "vLLM tensor model parallel group is not initialized in the FSDP worker; "
                    "assuming infer_tp=1 for worker-side data sharding.")
                return 1
            raise

    def _get_vllm_tp_group(self):
        if vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3"):
            return vllm_ps.get_tensor_model_parallel_group()
        return vllm_ps.get_tensor_model_parallel_group().device_group

    def __enter__(self):
        state_dict_started = time.perf_counter()
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        params = self.module.state_dict()
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        state_dict_seconds = time.perf_counter() - state_dict_started
        # Copy, not share memory
        load_format = self.sync_weight_format
        effective_transport = load_format
        transport_stats = None
        weight_sync_started = time.perf_counter()
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
        else:
            self.inference_engine.wake_up()
            # TODO(ZSL): deal with 'hf' format
            llm_engine = getattr(self.inference_engine, 'llm_engine', None)
            if llm_engine is None:
                raise AttributeError('official vLLM LLM has no llm_engine for weight sync')
            if load_format == 'dtensor':
                from verl.third_party.vllm import load_dtensor_weights
                # vLLM V1 removed llm_engine.model_executor.driver_worker.
                # apply_model runs in the V1 engine worker. This is usable for
                # fully materialized tensors, but DTensor objects may only carry
                # the actor-rank local shard after crossing into the engine
                # process, so keep dtensor sync as a diagnostic path.
                if hasattr(llm_engine, 'apply_model'):
                    def _agentmemory_load_dtensor(model):
                        from verl.third_party.vllm import load_dtensor_weights as _load_dtensor_weights
                        _load_dtensor_weights(params, model)
                        return model.__class__.__name__
                    sync_results = llm_engine.apply_model(_agentmemory_load_dtensor)
                    logger.info('Synced actor DTensor weights into official vLLM via apply_model: %s', sync_results)
                elif hasattr(llm_engine, 'model_executor'):
                    load_dtensor_weights(
                        params, llm_engine.model_executor.driver_worker.worker.model_runner.model)
                else:
                    raise AttributeError(
                        'Unsupported vLLM engine: neither apply_model nor model_executor is available')
            elif load_format == 'hf':
                model_type = str(getattr(self.model_config, 'model_type', ''))

                def materialize_hf_weights():
                    weights = []
                    for name, tensor in params.items():
                        if hasattr(tensor, 'full_tensor'):
                            tensor = tensor.full_tensor()
                        if hasattr(tensor, 'detach'):
                            tensor = tensor.detach()
                        weights.append((
                            map_actor_weight_name_for_vllm(
                                name, model_type=model_type
                            ),
                            tensor.cpu(),
                        ))
                    if is_qwen3_5_model_type(model_type):
                        validate_qwen35_mapped_source_names(
                            (name for name, _ in weights),
                            expected_names=self.expected_source_names,
                        )
                    return weights

                def load_hf_weights(model, weights):
                    if not hasattr(model, 'load_weights'):
                        raise AttributeError(f'{model.__class__.__name__} has no load_weights method')
                    load_weights_started = time.perf_counter()
                    loaded = model.load_weights(weights)
                    load_weights_seconds = time.perf_counter() - load_weights_started
                    try:
                        loaded_count = len(loaded)
                    except Exception:
                        loaded_count = len(list(loaded)) if loaded is not None else -1
                    result = {
                        'model': model.__class__.__name__,
                        'loaded_count': loaded_count,
                        'load_weights_seconds': load_weights_seconds,
                    }
                    if is_qwen3_5_model_type(model_type):
                        result.update(validate_qwen35_vllm_load_coverage(
                            loaded_names=loaded,
                            target_parameter_names=(
                                name for name, _ in model.named_parameters(remove_duplicate=False)
                            ),
                        ))
                    return result

                if hasattr(llm_engine, 'apply_model'):
                    from .vllm_hf_sync_transport import (
                        require_direct_inproc_runtime,
                        resolve_hf_sync_transport,
                    )

                    effective_transport = resolve_hf_sync_transport()
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    transport_stats = {'transport': effective_transport}
                    materialize_started = time.perf_counter()
                    hf_weights = materialize_hf_weights()
                    transport_stats['materialize_seconds'] = (
                        time.perf_counter() - materialize_started)
                    expected_source_names = self.expected_source_names
                    evidence_enabled = bool(self._sync_evidence_dir)
                    source_evidence_started = time.perf_counter()
                    if evidence_enabled:
                        global_steps, sync_id = self._next_sync_identity()
                        source_before = bounded_tensor_fingerprint(hf_weights)
                    else:
                        global_steps = sync_id = source_before = None
                    transport_stats['source_evidence_seconds'] = (
                        time.perf_counter() - source_evidence_started)
                    if effective_transport == 'direct_inproc':
                        transport_stats['runtime_types'] = require_direct_inproc_runtime(
                            llm_engine,
                            infer_tp_size=self._get_infer_tp_size(),
                        )

                        def agentmemory_load_hf_direct(model):
                            loaded_source_started = time.perf_counter()
                            loaded_source = (
                                bounded_tensor_fingerprint(hf_weights)
                                if evidence_enabled else None)
                            loaded_source_seconds = (
                                time.perf_counter() - loaded_source_started)
                            result = load_hf_weights(model, hf_weights)
                            result['loaded_source_evidence_seconds'] = loaded_source_seconds
                            if evidence_enabled:
                                target_evidence_started = time.perf_counter()
                                result.update({
                                    'sync_id': sync_id,
                                    'source_fingerprint_sha256': source_before['sha256'],
                                    'loaded_source_fingerprint_sha256': loaded_source['sha256'],
                                    'target_after': bounded_tensor_fingerprint(
                                        model.named_parameters()),
                                })
                                result['target_evidence_seconds'] = (
                                    time.perf_counter() - target_evidence_started)
                            return result

                        apply_model_started = time.perf_counter()
                        try:
                            sync_results = llm_engine.apply_model(
                                agentmemory_load_hf_direct)
                        finally:
                            transport_stats['apply_model_seconds'] = (
                                time.perf_counter() - apply_model_started)
                        transport_stats['apply_model_results'] = sync_results
                        del hf_weights
                    else:
                        sync_dir = (
                            os.getenv('VERL_AGENTMEMORY_HF_SYNC_DIR')
                            or os.getenv('RAY_TMPDIR')
                            or '/tmp'
                        )
                        os.makedirs(sync_dir, exist_ok=True)
                        sync_file = os.path.join(
                            sync_dir,
                            f'agentmemory_hf_sync_rank{rank}_pid{os.getpid()}.pt',
                        )
                        save_started = time.perf_counter()
                        torch.save(hf_weights, sync_file)
                        transport_stats['save_seconds'] = time.perf_counter() - save_started
                        transport_stats['sync_file_bytes'] = os.path.getsize(sync_file)
                        del hf_weights

                        def agentmemory_load_hf_from_file(model):
                            import torch as _torch
                            import time as _time
                            from verl.models.transformers.qwen3_5 import (
                                is_qwen3_5_model_type as _is_qwen3_5_model_type,
                            )
                            from verl.workers.qwen35_weight_sync import (
                                validate_qwen35_mapped_source_names as _validate_source_names,
                                validate_qwen35_vllm_load_coverage as _validate_load_coverage,
                            )
                            from verl.workers.sharding_manager.vllm_sync_evidence import (
                                bounded_tensor_fingerprint as _bounded_tensor_fingerprint,
                            )
                            file_load_started = _time.perf_counter()
                            weights = _torch.load(sync_file, map_location='cpu', weights_only=False)
                            file_load_seconds = _time.perf_counter() - file_load_started
                            if _is_qwen3_5_model_type(model_type):
                                _validate_source_names(
                                    (name for name, _ in weights),
                                    expected_names=expected_source_names,
                                )
                            loaded_source_started = _time.perf_counter()
                            loaded_source = (
                                _bounded_tensor_fingerprint(weights) if evidence_enabled else None)
                            loaded_source_seconds = (
                                _time.perf_counter() - loaded_source_started)
                            if not hasattr(model, 'load_weights'):
                                raise AttributeError(
                                    f'{model.__class__.__name__} has no load_weights method')
                            load_weights_started = _time.perf_counter()
                            loaded = model.load_weights(weights)
                            load_weights_seconds = (
                                _time.perf_counter() - load_weights_started)
                            try:
                                loaded_count = len(loaded)
                            except Exception:
                                loaded_count = len(list(loaded)) if loaded is not None else -1
                            result = {
                                'model': model.__class__.__name__,
                                'loaded_count': loaded_count,
                                'file_load_seconds': file_load_seconds,
                                'loaded_source_evidence_seconds': loaded_source_seconds,
                                'load_weights_seconds': load_weights_seconds,
                            }
                            if _is_qwen3_5_model_type(model_type):
                                result.update(_validate_load_coverage(
                                    loaded_names=loaded,
                                    target_parameter_names=(
                                        name for name, _ in model.named_parameters(
                                            remove_duplicate=False)
                                    ),
                                ))
                            if evidence_enabled:
                                target_evidence_started = _time.perf_counter()
                                result.update({
                                    'sync_id': sync_id,
                                    'source_fingerprint_sha256': source_before['sha256'],
                                    'loaded_source_fingerprint_sha256': loaded_source['sha256'],
                                    'target_after': _bounded_tensor_fingerprint(
                                        model.named_parameters()),
                                })
                                result['target_evidence_seconds'] = (
                                    _time.perf_counter() - target_evidence_started)
                            return result

                        apply_model_started = time.perf_counter()
                        try:
                            sync_results = llm_engine.apply_model(agentmemory_load_hf_from_file)
                        finally:
                            transport_stats['apply_model_seconds'] = (
                                time.perf_counter() - apply_model_started)
                            try:
                                os.remove(sync_file)
                            except FileNotFoundError:
                                pass
                        transport_stats['apply_model_results'] = sync_results
                    if evidence_enabled:
                        previous_event = self._last_sync_event
                        event = build_sync_event(
                            rank=rank,
                            pid=os.getpid(),
                            global_steps=global_steps,
                            sync_sequence=self._sync_sequence,
                            sync_id=sync_id,
                            source_before=source_before,
                            apply_model_results=sync_results,
                            previous_event=previous_event,
                            transport=effective_transport,
                        )
                        self._current_sync_previous = previous_event
                        self._current_sync_event = append_and_readback_event(
                            self._sync_evidence_path(), event)
                        self._last_sync_event = self._current_sync_event
                    logger.info(
                        'Synced actor HF weights into official vLLM via %s apply_model: %s',
                        effective_transport,
                        sync_results,
                    )
                elif hasattr(llm_engine, 'model_executor'):
                    model = llm_engine.model_executor.driver_worker.worker.model_runner.model
                    sync_results = load_hf_weights(model, materialize_hf_weights())
                    logger.info('Synced actor HF weights into official vLLM model_executor: %s', sync_results)
                else:
                    raise AttributeError(
                        'Unsupported vLLM engine: neither apply_model nor model_executor is available')
            else:
                raise NotImplementedError(f'load_format {load_format} not implemented')
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)
        logger.warning(
            'AGENTMEMORY_VLLM_SYNC_TIMING rank=%s transport=%s '
            'state_dict_s=%.6f weight_sync_s=%.6f transport_stats=%s',
            self._sync_rank(),
            effective_transport,
            state_dict_seconds,
            time.perf_counter() - weight_sync_started,
            transport_stats,
        )

        del params
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        # TODO(ZSL): check this
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp.
        tp_size = self._get_infer_tp_size()
        if tp_size <= 1:
            return data
        group = self._get_vllm_tp_group()

        prev_device = data.batch.device
        data.batch = data.batch.cuda(device=torch.cuda.current_device())
        data.batch = allgather_dict_tensors(data.batch.contiguous(), size=tp_size, group=group, dim=0)
        data.batch = data.batch.to(prev_device)
        # all gather non_tensor_batch
        all_non_tensor_batch = [None for _ in range(tp_size)]
        torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=group)
        data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp.
        local_world_size = self._get_infer_tp_size()
        if local_world_size <= 1:
            return data
        src_rank = (torch.distributed.get_rank() // local_world_size) * local_world_size
        group = self._get_vllm_tp_group()
        broadcast_dict_tensor(data.batch, src=src_rank, group=group)
        dp_rank = torch.distributed.get_rank()
        tp_size = local_world_size
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data
