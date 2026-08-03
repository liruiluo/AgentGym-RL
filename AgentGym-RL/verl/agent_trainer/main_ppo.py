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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.agent_trainer.ppo.ray_trainer import RayPPOTrainer
from verl.agent_trainer.reference_policy import should_create_reference_policy

import os

import ray
import hydra


def _apply_training_triton_cache_env(env):
    requested = env.get('VERL_TRAINING_TRITON_CACHE_DIR')
    if not requested:
        return
    expanded = os.path.expanduser(requested)
    if not os.path.isabs(expanded):
        raise RuntimeError(
            'VERL_TRAINING_TRITON_CACHE_DIR must resolve to an absolute path'
        )
    stable_dir = os.path.realpath(expanded)
    env['VERL_TRAINING_TRITON_CACHE_DIR'] = stable_dir
    env['TRITON_CACHE_DIR'] = stable_dir
    env['FLA_CACHE_RESULTS'] = '1'


def _ray_runtime_env_vars():
    env = {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}
    # AgentMemoryGym / JD 9N compatibility knobs must reach Ray workers before
    # vLLM/VERL modules are imported. Ray's per-actor runtime_env can otherwise
    # hide shell exports such as VLLM_USE_V1=0.
    for key in (
        'VLLM_USE_V1',
        'VLLM_USE_DEEP_GEMM',
        'VLLM_ATTENTION_BACKEND',
        'VLLM_WORKER_MULTIPROC_METHOD',
        'VLLM_USE_MODELSCOPE',
        'VLLM_ALLOW_INSECURE_SERIALIZATION',
        'VERL_AGENTMEMORY_HF_SYNC_DIR',
        'VERL_PPO_LOGGING_LEVEL',
        'VERL_TRAINING_TRITON_CACHE_DIR',
        'FLA_CACHE_RESULTS',
        'AGENTMEMORY_DATA_PATH',
        'AGENTMEMORY_SPLIT',
        'AGENTMEMORY_SPLIT_DIR',
        'AGENTMEMORY_CATALOG_INDEX_PATH',
        'HYDRA_FULL_ERROR',
        'WANDB_MODE',
    ):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    _apply_training_triton_cache_env(env)
    # Positive-control and curriculum knobs can be added faster than this
    # whitelist is updated. Ray runtime_env is otherwise a silent prompt/action
    # contract footgun, so forward every explicit AgentMemoryGym knob.
    for key, value in os.environ.items():
        if key.startswith("AGENTMEMORY_") and value is not None:
            env.setdefault(key, value)
    return env


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': _ray_runtime_env_vars()})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    else:
        raise NotImplementedError

    from verl.agent_trainer.ppo.ray_trainer import ResourcePoolManager, Role

    use_reference_policy = should_create_reference_policy(config)
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }
    if use_reference_policy:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    if use_reference_policy:
        mapping[Role.RefPolicy] = global_pool_id

    print(
        f'[main_task] reference_policy_enabled={use_reference_policy}',
        flush=True,
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls)
    print('[main_task] trainer constructed; init_workers begin', flush=True)
    trainer.init_workers()
    print('[main_task] init_workers done; fit begin', flush=True)
    trainer.fit()
    print('[main_task] fit returned', flush=True)


if __name__ == '__main__':
    # import socket
    # print(socket.gethostbyname(socket.gethostname()))
    # socket.sethostname("localhost")
    # print(socket.gethostbyname(socket.gethostname()))
    main()
