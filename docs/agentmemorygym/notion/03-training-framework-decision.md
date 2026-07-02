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
- 不借 AgeMem 的三阶段课程学习路线；AgeMem 只作为 STM/LTM memory-tool 语义参考。

## 训练阶段安排

1. 文档与 Notion 对齐。
2. 0 卡本地检查：compile、data/schema、direct env、server API、rollout guard；这只证明代码/schema 基本可跑，不算单卡测试。
3. 真正单卡 smoke：Jingyan 1×B200 已跑通环境、client、Qwen3-4B rollout、SEARCH-aware prompt smoke；这些只算链路/接口证据。
4. 基线 smoke：scripted SEARCH baseline / heuristic memory manager 已先跑通。semanticfix5 诊断结果为 no-memory `0/15`、full-context `6/15`、memory-tool strict no-retry `6/15`、memory-tool retry5 `13/15`、memory-tool soft-fallback verifier diagnostic `15/15`。这些只说明接口可解和任务依赖记忆，不是 RL 提升。
5. 训练上下文边界：latest-observation 不是无短期记忆，而是当前 observation 内包含本 session 自动 STM；trainer 不应把上一 session 的 raw history 拼回 actor/ref logprob 输入。
6. Bounded RL pilot：SEARCH/metadata 路线已经能证明 frozen dev 可解，不能再无限停在诊断阶段。拿到授权 GPU lane 后应尽快跑小规模 GRPO/PPO pilot，用来暴露训练链路和策略失败。
7. 8 卡正式后训练：两台新 8 卡已提交申请并由 watcher 接管排队状态，正式实验等新机器 RUNNING 且平台 holder + pod 内 auto-yield 占卡守护验证后启动，不占用 continual-reasoning gym 的旧 8 卡。
8. 第一轮行为分析后，再决定是否需要新优化目标或 credit assignment 机制。
