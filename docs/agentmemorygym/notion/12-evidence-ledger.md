# 证据台账

## 当前核心 claim

| Claim | 状态 | 支撑来源 | 用法 |
|---|---|---|---|
| 长程 agent 任务需要功能性记忆，而不只是复述历史。 | 保留 | MemoryArena、MemoryAgentBench、长期记忆系统相关工作 | 支撑问题背景。 |
| AgentGym-RL 适合作为后训练骨架。 | 保留 | AgentGym-RL 仓库与本地 fork | 支撑代码承载决策。 |
| 已有 RL-memory 方法存在，不能 claim absolute first。 | 保留 | AgeMem、Memory-R1、MemAct、MEM1、MemAgent、UMA | 支撑 novelty guardrail。 |
| v0 应改为 MemoryArena / 电商捆绑购物优先。 | 新增 | 用户当前研究判断 + MemoryArena 任务思想 | 支撑方向更新。 |
| MemoryAgentBench 不再是第一环境主线。 | 更新 | 当前项目定位调整 | 降级为 baseline / 数据参考。 |
| 代码草稿应保留为 skeleton，不因“先文档再代码”而撤回。 | 新增 | 用户 2026-07-01 纠偏 + 当前 worktree | 支撑执行边界。 |
| 最小购物环境不能让 product_id 直接泄漏关键尺寸答案。 | 新增 | 当前 smoke 代码审查 | 支撑环境可训练性边界。 |

## 已核的一手证据

- MemoryArena 项目页：确认其核心评估思想是多会话 Memory-Agent-Environment 交互、显式相互依赖子任务，以及 `bundled_shopping` / `progressive_search` / `group_travel_planner` / `formal_reasoning_math/phys` 等任务配置。
- AgentGym-RL GitHub：确认其定位是 multi-turn interactive decision-making RL 框架，采用 environment / agent / training 模块化设计，并列出 PPO、GRPO、RLOO、REINFORCE++ 等算法。
- AgeMem ACL PDF：确认其 memory tool taxonomy 为 LTM `ADD/UPDATE/DELETE` 与 STM `RETRIEVE/SUMMARY/FILTER`，与当前 AgentMemoryGym action 设计一致。
- 本地记录：`docs/agentmemorygym/evidence/20260701-source-check.md`。

## 后续仍需补的证据

- MemoryArena/WebShop 真实数据转换代码与 item-id 冻结。
- 可复现 RL-memory baseline 的代码可用性与公平比较协议。
- 用户提到的 `Qwen3.6-4B` 模型名和可用 checkpoint，需要在实际训练前单独核验。

## 本地 smoke 证据

- `docs/agentmemorygym/evidence/20260701-skeleton-smoke.md` 记录了 compile/direct/server-client smoke。
- JSONL loader 已验证，marker 为 `JSONL_LOADER_SMOKE_OK`；data validator marker 为 `AGENTMEMORY_DATA_VALIDATE_OK`；server-client JSONL/split 路径 marker 为 `SERVER_CLIENT_JSONL_SMOKE_OK` / `SERVER_CLIENT_SPLIT_SMOKE_OK`。
- 当前 smoke 只证明最小环境 skeleton 可跑，不证明完整 MemoryArena 转换或 RL 提升。
- Mac/ZBMac 是 0 卡机器；本地 compile/data/server/stub smoke 只能证明代码/schema/API 基本可跑，不能记作单卡 smoke。
- Mac 本机缺少 `torch`，本地 import probe 仍记录为 `AGENTGYM_ADAPTER_IMPORT_FAIL ModuleNotFoundError No module named 'torch'`；该限制已通过 Jingyan 1×B200 的真实 GPU 环境补掉 adapter/client/import 层，但完整模型 rollout 仍需后续验证。
- `docs/agentmemorygym/evidence/20260701-single-gpu-smoke.md` 记录 Jingyan 1×B200 真单卡 smoke：`TORCH_CUDA_OK`、`AGENTMEMORY_REAL_ADAPTER_IMPORT_OK`、`SERVER_METADATA_SINGLE_GPU_OK`、`AGENTMEMORY_REAL_CLIENT_METADATA_SINGLE_GPU_OK`、`VERL_INIT_ENV_CLIENT_AGENTMEMORY_SINGLE_GPU_OK`。这证明 GPU/env/client/init 路径可跑，但不证明完整模型 rollout、正式 RL 训练或 MemoryArena 转换。
- `docs/agentmemorygym/evidence/20260701-latest-observation-rollout.md` 记录 latest-observation rollout context：`AGENTMEMORY_LATEST_OBSERVATION_PROMPT_SMOKE_OK`、`AGENTMEMORY_ROLLOUT_CONTEXT_ALIGNMENT_SMOKE_OK`。这证明训练数据路径不再把 raw history 暴露给 actor/ref logprob 重算，但仍不等于完整模型 rollout 或 RL 训练结果。
- `docs/agentmemorygym/evidence/20260701-latest-observation-policy-smoke.md` 记录 scripted-policy rollout smoke：`AGENTMEMORY_LATEST_OBSERVATION_POLICY_SMOKE_OK`。它验证 latest-observation + memory tool contract 能闭合三条 bundled-shopping smoke 任务，但不是 LLM/vLLM rollout。
