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
- converter 也支持 `--enrich-candidate-metadata` / `--candidate-metadata-min-score` / `--candidate-metadata-catalog-scope`：仅当同一 subtask 的所有候选都有达标 catalog match 时，才把 `average_rating / price_usd / total_reviews` 写入 candidate attributes；ASIN/source path 只留在 report/manifest。
- 无 catalog 时 full data 有 12/900 个 ambiguous matches；Jingyan 共享盘 4 个相关 catalog shard 验证后 summary 为 `rows=900 / ambiguous=0 / catalog=450 / fallback=450 / min_match=7`。
- 大 product DB 镜像位置：`/home/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/`；不落开发机本地盘。当前全量镜像已校验完成：`135 files / 13,517,161,526 bytes`，extra/missing/mismatch/part 均为 0。
- 新增 freeze helper：`freeze_memoryarena_bundled_shopping.py` 会先按 target ASIN 快速筛出相关 catalog shard，再调用 converter + validator 并写 `freeze_manifest.json`。正式 freeze 产物：`memoryarena_formal_freeze_20260701-234045`，`tasks=150 / rows=900 / train/dev/test=120/15/15 / asin_catalog=900 / ambiguous=0`。
- Qwen3-4B / Transformers 真实单卡 rollout smoke 已在 Jingyan 1×B200 跑通，证据为 `AGENTMEMORY_QWEN3_4B_LATEST_OBSERVATION_PROGRESS_ROLLOUT_SMOKE_OK`；frozen dev 2 条样本产生 `20` 个 valid env steps，但 `progress_score=0.0`，handcrafted smoke 可到 `progress_score=0.3333`。
- 当前严格 enriched freeze 只覆盖 `285/900` 个 step 的 all-candidate metadata；用全量 product_catalog 67 个 shard 复跑仍是 `285/900`，所以缺口不是共享盘存储或下载范围，而是可靠 option-to-catalog 对齐/部分 metadata 缺字段。
- 已新增 product-catalog `SEARCH` 工具草稿：环境支持 `SEARCH {"query":"...","top_k":3}`，返回 title + `average_rating / price_usd / total_reviews / match_score`；`build_memoryarena_catalog_search_index.py` 已在 Jingyan 共享盘上把全量 product DB 构成 SQLite/FTS index，路径为 `/home/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/agentmemory_catalog_search.sqlite`，`products=1,031,654`、约 `479M`。ASIN/source path 不进入 observation。
- Qwen3-4B metadata-aware diagnostic prompt 在 enriched dev 上产生非零进度（`0.1667,0.0`），但普通 prompt 仍会 `RETRIEVE highest rated` loop；SEARCH-aware prompt 会调用 `SEARCH`，但重复占位 query `visible candidate title`，没有 `ADD/BUY`，`progress_score=0.0,0.0`。这些仍不代表完整 vLLM/verl rollout 或 RL 训练结果。
- 新增 scripted SEARCH baseline：`agentenv-agentmemory/scripts/run_scripted_search_baseline.py`。脚本只用 visible candidate titles、当前 instruction、自己的 `ADD/RETRIEVE` memory 和 `SEARCH` 返回的公开 metadata；`--include-target-audit` 只写审计字段，不参与 action selection。Jingyan dev no-retry 结果 `5/15`，`max_buy_attempts=5` verifier-feedback retry diagnostic 结果 `10/15`。新增 `--compatibility-fallback ranked-all-after-compatible` 后，soft-fallback verifier diagnostic 达到 `15/15`。配套 `analyze_scripted_search_failures.py` 会输出 `AGENTMEMORY_SCRIPTED_SEARCH_FAILURE_AUDIT_OK`。这些都只是接口/可解性证据，不是 RL 训练。
