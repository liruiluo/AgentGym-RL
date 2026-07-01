# 训练框架方案比较与代码承载决策

## 结论

训练后端继续采用 AgentGym-RL / verl。原因是本项目要做的是 agentic memory 后训练 Gym，而不是重新实现 PPO/GRPO trainer。AgentGym-RL 已经提供多轮 agent rollout、环境 client、verl 训练入口和 PPO/GRPO 类算法接口，适合作为第一版训练骨架。

## 当前采用方式

- 主仓：`code/AgentGym-RL`
- 环境 submodule：`code/AgentGym-RL/AgentGym`
- 新环境：当前保留并整理 `agentenv-agentmemory` skeleton
- 任务名：已在 AgentGym-RL client registry 中注册为 `agentmemory`

## 不采用的路线

- 不新建独立 `agentmemorygym_rl` 主训练仓。
- 不重写 AgentGym-RL 的 trainer。
- 不把 WebShop/WebArena 原生环境直接改坏成 memory env。
- 不在 v0 阶段提前把 CMA-GRPO 或其它新算法写成主贡献。

## 训练阶段安排

1. 文档与 Notion 对齐。
2. 单卡 smoke：环境、client、rollout 小样本；当前已完成 compile/direct/server-client smoke，完整 AgentGym/verl rollout 仍需单卡依赖环境。
3. 基线 smoke：no-memory、full-context、fixed-RAG、heuristic memory manager。
4. 8 卡正式后训练：等新机器，不占用 continual-reasoning gym 的 8 卡。
5. 第一轮行为分析后，再决定是否需要新优化目标或 credit assignment 机制。
