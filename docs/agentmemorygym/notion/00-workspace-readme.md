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
4. 先单卡 compile/direct/server-client smoke。
5. 明天新 8 卡到位后再跑正式后训练。

## 关键边界

- MemoryAgentBench 仍是 baseline / 数据参考，不是第一环境主线。
- 电商捆绑购物是第一版 hero environment。
- AgentGym-RL / verl 仍是训练后端。
- 不 claim 第一个 RL-memory 方法。
