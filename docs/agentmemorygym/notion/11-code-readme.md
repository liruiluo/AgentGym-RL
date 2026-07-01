# 代码目录说明

```text
code/AgentGym-RL/              # 主训练框架 fork
code/AgentGym-RL/AgentGym/     # 环境 submodule fork
code/AgentGym-RL/docs/agentmemorygym/  # 当前文档与 Notion 同步源
```

当前 skeleton：

```text
AgentGym/agentenv-agentmemory/
AgentGym/agentenv/agentenv/envs/agentmemory.py
AgentGym-RL/verl/utils/agentgym/client.py 中注册 task_name=agentmemory
```

当前状态：代码草稿可以保留在工作树，但执行顺序仍是先文档/Notion，再代码验证。正式代码完成前必须重新跑 compile/import/server-client smoke。

Skeleton 当前只覆盖 handcrafted MemoryArena/WebShop-style bundled shopping smoke：

- 默认数据入口：`agentenv_agentmemory/data/bundled_shopping_smoke.jsonl`。
- 默认 split 文件：`agentenv_agentmemory/data/splits/{train,dev,test}.txt`，当前各一个 smoke item。
- 中性 product_id，例如 `tv_b`、`mount_b`、`console_b`，避免直接把 `75` / `large` 写进 ID。
- `info` 记录 `task_id`、`split`、`source`、`difficulty`、`memory_dependency`、`progress_score`、`episode_success`、`memory_ops`、`memory_state_diff`、`compatibility_violations`、`purchase_history`。
- AgentGym client 侧保留 server 返回的 `env_info`，后续可用于 rollout 日志和 behavior analysis。
- data validator：`PYTHONPATH=agentenv-agentmemory python3 agentenv-agentmemory/scripts/validate_agentmemory_data.py`。
- direct smoke helper：`PYTHONPATH=agentenv-agentmemory python3 agentenv-agentmemory/scripts/smoke_agentmemory.py`。
- JSONL loader smoke marker：`JSONL_LOADER_SMOKE_OK`。
- validator marker：`AGENTMEMORY_DATA_VALIDATE_OK`。
- 仍不代表完整 MemoryArena 数据转换；下一层需要 real MemoryArena/WebShop converter、train/dev/test split、raw-history leakage guard 和小模型/API rollout smoke。
