# AgentMemoryGym 文档入口

本目录是 AgentMemoryGym 当前方向的本地文档源。当前项目已经从旧的 **MemoryAgentBench-first** 启动包，调整为 **MemoryArena / 电商捆绑购物优先的 Agentic RL Memory 后训练 Gym**。

## 当前短结论

- 目标：构建一个能通过 RL 后训练提升 agent memory policy 的 Gym，而不是只做新 benchmark 或先押一个新算法。
- Hero 环境：电商捆绑/序列购物。后续购买必须记住前面已购商品属性，例如电视尺寸、重量、VESA、接口等。
- 训练骨架：继续复用 AgentGym-RL / verl；memory env 落在 AgentGym submodule fork 中。
- Memory 工具：参考 AgeMem/Agentic Memory，把 `ADD / UPDATE / DELETE / RETRIEVE / SUMMARY / FILTER` 做成 policy 可选动作。
- 奖励：最终任务成功 + 子任务进度 + 兼容约束满足，扣除过度 memory/tool 操作成本。
- 执行顺序：**先改本地文档与 Notion 文档，再改代码**。这只是优先级/验收顺序，不等于撤回已有代码草稿；当前 `agentenv-agentmemory` skeleton 可以保留在工作树，但在正式收口前必须重新验证并明确标注为草稿。
- 资源边界：Mac/ZBMac 仍只做 0 卡静态/数据/schema/API 检查；大 MemoryArena product DB 和 SQLite/FTS 搜索索引都放 Jingyan 共享盘，不落开发机本地盘。Jingyan 1×B200 已用于真实单卡 smoke 和 scripted SEARCH dev baseline；8 卡机器当前给 continual-reasoning gym，AgentMemoryGym 等新 8 卡再做正式后训练。

## 文件

- `agentic-memory-gym-design.md`：英文/技术设计初稿。
- `agentic-memory-gym-design.zh.md`：中文主设计文档。
- `notion/`：同步到 Notion 分页面版的本地源文件。
- `evidence/20260701-skeleton-smoke.md`：当前 skeleton compile/direct/server-client smoke 证据。
- `evidence/20260701-rollout-guard-smoke.md`：dataset/split metadata 与 raw-history rollout guard 证据。
- `evidence/20260701-single-gpu-smoke.md`：Jingyan 1×B200 上的真实单卡 GPU/env/client/init smoke 证据。
- `evidence/20260701-latest-observation-rollout.md`：latest-observation rollout context 与 per-action PPO 样本展开证据。
- `evidence/20260701-latest-observation-policy-smoke.md`：latest-observation scripted-policy rollout smoke；验证 memory tool contract，不冒充 LLM rollout。
- `evidence/20260701-memoryarena-converter.md`：MemoryArena bundled-shopping converter / freeze / full product DB / SEARCH index 证据；当前 frozen train/dev/test 为 `120/15/15`，target match 已 `asin_catalog=900 / ambiguous=0`。
- `evidence/20260702-qwen3-4b-rollout-smoke.md`：Qwen3-4B 真单卡 rollout、enriched metadata diagnostic、SEARCH-aware prompt smoke；均是链路/接口证据，不是 RL 提升结果。
- `evidence/20260702-scripted-search-baseline.md`：scripted SEARCH baseline / heuristic memory manager；full dev no-retry 为 `5/15`、retry diagnostic 为 `10/15`，证明 SEARCH 接口有可解性但不是 RL/memory 提升结果。

## 当前 Notion 页面

Notion 入口页：`AgentMemoryGym-RL 项目文档（分页面版，2026-06-21）`。

本地到 Notion 的写入策略：本地 Markdown 是当前同步源；Notion 是人类可读镜像。同步时保留父页 child page，不删除子页结构。
