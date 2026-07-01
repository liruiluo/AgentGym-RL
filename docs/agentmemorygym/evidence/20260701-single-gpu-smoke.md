# 2026-07-01 AgentMemoryGym true single-GPU smoke on Jingyan 1xB200

## Scope

This note records the first **true single-GPU** AgentMemoryGym smoke after the
Mac/ZBMac 0-card boundary correction.

It was run on the released Jingyan 9N 1-card workspace:

- 9N task: `luolirui-1-jy1-0630030405-1`
- GPU: `NVIDIA B200`, 1 card
- Python: `/opt/conda/envs/rl/bin/python3`
- torch: `2.11.0+cu130`
- copied source manifest:
  - main `AgentGym-RL` commit: `25ce48d`
  - submodule `AgentGym` commit: `d1a1ebb`
  - remote worktree: `/home/ai-jingyan-train/luolirui.1/post-train/code/AgentGym-RL-agentmemory-smoke`

This smoke validates GPU availability, torch CUDA execution, real AgentGym
adapter import, server metadata, real AgentGym client metadata, and the
`verl.utils.agentgym.client.init_env_client` metadata path. It is **not** a full
model rollout, not a GRPO/PPO training run, and not MemoryArena/WebShop data
conversion evidence.

## Evidence logs on the GPU lane

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/single_gpu_smoke_20260701-184733.log
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/single_gpu_server_client_20260701-184810.log
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/single_gpu_init_env_client_20260701-184840.log
```

## Markers

```text
TORCH_CUDA_OK 2.11.0+cu130 NVIDIA B200
TORCH_B200_MATMUL_OK (4096, 4096) torch.bfloat16
COMPILEALL_GPU_ENV_OK
AGENTMEMORY_DATA_VALIDATE_OK tasks=3 splits=train:1,dev:1,test:1
AGENTMEMORY_DIRECT_SMOKE_OK tv_bundle_75 laptop_bundle_14 monitor_bundle_27
AGENTMEMORY_REAL_ADAPTER_IMPORT_OK True latest_observation_only
AGENTMEMORY_CONTEXT_POLICY_SMOKE_OK
SERVER_METADATA_SINGLE_GPU_OK 3 tv_bundle_75,laptop_bundle_14,monitor_bundle_27
AGENTMEMORY_REAL_CLIENT_METADATA_SINGLE_GPU_OK 3 monitor_bundle_27
VERL_INIT_ENV_CLIENT_AGENTMEMORY_SINGLE_GPU_OK 3 3
```

## Boundary

The earlier Mac checks remain classified as 0-card local validation. This file is
the first true single-GPU evidence because it ran on a real B200 lane with torch
CUDA available and imported the real `agentenv.envs.agentmemory` adapter without
using the local stub.

Remaining Stage 1b gap: run a small model/API rollout after the rollout path can
respect the AgentMemoryGym latest-observation / ephemeral-context contract. The
current raw-history vLLM rollout is intentionally guarded and should not be used
as formal training evidence.

## Lane state after smoke

The workspace warmup guard was still present after the smoke and resumed GPU
warmup when the smoke commands exited:

```text
workspace_cpu_warmup
workspace_gpu_warmup
0, NVIDIA B200, 78, 8905, 183359
```
