# 代码实现方案当前决策

## 当前决策

主代码仍采用 `code/AgentGym-RL` 这个 AgentGym-RL fork，新增 memory environment 仍落在 `code/AgentGym-RL/AgentGym` submodule fork 中。项目不新建平行的 `agentmemorygym_rl` 主训练仓。

但执行顺序更新为：**先改本地文档和 Notion 文档，再改代码**。这只是优先级和验收顺序，不代表撤回已有代码草稿；当前已有代码草稿保留在工作树，并进入 skeleton 整理与 smoke 验证。

## 当前主线变化

旧方案：先把 MemoryAgentBench AR/CR 样本包装成 AgentGym custom env。

新方案：先做 **MemoryArena / 电商捆绑购物** 的可训练 Gym skeleton。MemoryAgentBench 降级为 baseline、数据和评测参考，不再作为第一个环境主线。

## 代码边界

```text
code/AgentGym-RL/                 # 主训练框架 fork：trainer、rollout、配置、环境接入
code/AgentGym-RL/AgentGym/        # submodule fork：新增 agentenv-agentmemory
code/AgentGym-RL/docs/agentmemorygym/  # 当前文档与 Notion 同步源
```

后续新增代码应保持小步可验证：

1. 新增 `agentenv-agentmemory` package。
2. 先实现 bundled shopping smoke 环境。
3. 再注册 `AgentMemoryEnvClient` 和 `task_name=agentmemory`。
4. Mac/ZBMac 只做 0 卡本地检查：compile、data/schema、direct env、server API；不能写成单卡测试。
5. 大 MemoryArena product DB、catalog scan 和 SQLite/FTS index 都放 Jingyan 共享盘，不放开发机/Mac 本地盘。共享盘容量不是限制，可以全量下载、全量建索引；不要因为本机 0 卡或本机无 DB 而降级成“本机最小依赖”。
6. Jingyan 1×B200 已用于 direct env / client / Qwen3-4B rollout smoke 和 scripted SEARCH baseline；semanticfix5 当前诊断为 no-memory `0/15`、full-context `6/15`、memory-tool no-retry `6/15`、retry5 `13/15`、soft-fallback verifier diagnostic `15/15`，只算接口/可解性和 memory-dependence 证据。
7. 环境记忆边界固定为：当前 session 自动 STM trace 可见；成功 `BUY` 进入下一 session 时清空 raw history；跨 session 只通过 LTM `ADD/RETRIEVE` 传递必要信息。
8. LTM `RETRIEVE` 默认后端固定为本地 BM25；policy 调用仍是 `RETRIEVE {"query": "...", "top_k": 3}`，不接 DashScope/OpenAI embedding API，也不接 helper model。
9. 新 8 卡机器已提交申请：`luolirui-1-amg-g5a-0702` 和 `luolirui-1-amg-g5b-0702`，当前排队中。devbox 外层 watcher 会在任务 RUNNING 后安装 pod 内 auto-yield GPU/CPU 守护，保证训练结束后自动回到占卡状态。正式 8 卡后训练等这些新机器到位；排队期间继续收口文档、基线和 bounded RL pilot 准备。

## 资源边界

- 当前 0 卡本地检查不等于单卡 smoke。
- Jingyan 1×B200 已由用户释放并用于 AgentMemoryGym smoke；继续只做 smoke/诊断，不冒充正式训练资源。
- 现有 8 卡机器当前给 continual-reasoning gym 项目用。
- AgentMemoryGym 的两台新 8 卡机器已在 2026-07-02 提交申请，当前 QUEUEING；不能把排队当成已占住，必须等平台 holder RUNNING 且 pod 内 GPU workload/auto-yield guard 验证后才算占卡成功。

## 已有代码草稿状态

代码草稿保留，但此页只记录方向，不把草稿视为已完成实现。正式完成代码前必须重新跑 compile/import/server-client smoke，并检查 AgentGym-RL registry、submodule 状态和 diff。当前最小环境应使用中性 product_id，避免 `mount_large_75` 这类 ID 直接泄漏答案。
