# 00 Workspace README

## 当前项目定位

AgentMemoryGym 现在是 **Agentic RL Memory 后训练 Gym** 项目。主线已经从 MemoryAgentBench-first 调整为 MemoryArena / 电商捆绑购物优先。

## 当前目录理解

```text
code/AgentGym-RL/                  主训练 fork
code/AgentGym-RL/AgentGym/         环境 submodule fork
code/AgentGym-RL/docs/agentmemorygym/  当前文档与 Notion 同步源
```

## 当前优先级

1. 文档和 Notion 先对齐。
2. 代码草稿可以保留在工作树；“先文档再代码”是优先级，不是撤回草稿。
3. 当前整理并验证 `agentenv-agentmemory` skeleton。
4. Mac/ZBMac 只做 0 卡本地检查；大 product DB 与 SQLite/FTS SEARCH index 放 Jingyan 共享盘，不放开发机/Mac 本地盘。共享盘容量不是限制，可以全量下载、全量建索引；不要因为本机 0 卡或本机无 DB 而降级成“本机最小依赖”。
5. Jingyan 1×B200 已完成真单卡 env/client/Qwen3-4B rollout smoke；这证明链路可跑，不证明 RL memory 提升。
6. 现有 8 卡继续给 continual-reasoning gym；AgentMemoryGym 等新 8 卡到位后再跑正式后训练。

## 关键边界

- MemoryAgentBench 仍是 baseline / 数据参考，不是第一环境主线。
- 电商捆绑购物是第一版 hero environment。
- AgentGym-RL / verl 仍是训练后端。
- session 内自动保留 STM trace；成功 `BUY` 进入下一 session 时清空 raw history，跨 session 只靠 LTM `ADD/RETRIEVE`。
- 不 claim 第一个 RL-memory 方法。
