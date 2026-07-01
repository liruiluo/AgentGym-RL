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
- dataset/split 配置：`AGENTMEMORY_DATA_PATH`、`AGENTMEMORY_SPLIT`、`AGENTMEMORY_SPLIT_DIR`。
- server metadata：`/metadata` 返回 `task_count`、`task_ids`、`splits`、`source`。
- 中性 product_id，例如 `tv_b`、`mount_b`、`console_b`，避免直接把 `75` / `large` 写进 ID。
- `info` 记录 `task_id`、`split`、`source`、`difficulty`、`memory_dependency`、`progress_score`、`episode_success`、`memory_ops`、`memory_state_diff`、`compatibility_violations`、`purchase_history`。
- AgentGym client 侧保留 server 返回的 `env_info`，后续可用于 rollout 日志和 behavior analysis。
- data validator：`PYTHONPATH=agentenv-agentmemory python3 agentenv-agentmemory/scripts/validate_agentmemory_data.py`。
- direct smoke helper：`PYTHONPATH=agentenv-agentmemory python3 agentenv-agentmemory/scripts/smoke_agentmemory.py`。
- JSONL loader smoke marker：`JSONL_LOADER_SMOKE_OK`。
- validator marker：`AGENTMEMORY_DATA_VALIDATE_OK`。
- rollout guard marker：`AGENTMEMORY_CONTEXT_POLICY_SMOKE_OK`。`task_name=agentmemory` 在当前全历史拼接式 vLLM rollout 中默认 fail-fast，防止正式训练绕过 memory tools；显式 raw-history override 只能用于 diagnostic smoke。
- 当前 Mac/ZBMac 上的检查是 0 卡本地检查，不是单卡 smoke。
- Jingyan 1×B200 已通过真实单卡 env/client/init smoke：torch CUDA、real AgentGym adapter import、server metadata、real client metadata、`init_env_client` metadata path。
- latest-observation rollout context 已实现：`agentmemory` 默认只用当前 observation 生成 action，多轮 episode 展平成 per-action PPO 样本，`rollout_parent_indices` 负责 trainer batch 对齐。
- latest-observation scripted-policy rollout smoke 已通过：`AGENTMEMORY_LATEST_OBSERVATION_POLICY_SMOKE_OK`。
- MemoryArena bundled-shopping converter 已新增：`memoryarena_converter.py`、`convert_memoryarena_bundled_shopping.py`、`smoke_memoryarena_converter.py`；public 150 条转换 smoke 为 `train/dev/test=120/15/15`，validator 通过。
- converter 现在支持 `--catalog-path` 传入 MemoryArena product DB JSON 文件或目录：优先用 `target_asin -> catalog title` 消歧，再 fallback 到属性匹配。
- 无 catalog 时 full data 有 12/900 个 ambiguous matches；Jingyan 共享盘 4 个相关 catalog shard 验证后 summary 为 `rows=900 / ambiguous=0 / catalog=450 / fallback=450 / min_match=7`。
- 大 product DB 镜像位置：`/home/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/`；不落开发机本地盘。
- 仍不代表完整 LLM/vLLM rollout 或 RL 训练结果；下一层需要真实小模型/API rollout smoke。
