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
- `docs/agentmemorygym/evidence/20260701-memoryarena-converter.md` 记录 MemoryArena bundled-shopping converter：synthetic fixture marker `AGENTMEMORY_MEMORYARENA_CONVERTER_SMOKE_OK`；public full data marker `AGENTMEMORY_MEMORYARENA_CONVERT_OK tasks=150 splits=train:120,dev:15,test:15 min_match_score=2 ambiguous_matches=12` 与 `AGENTMEMORY_DATA_VALIDATE_OK`。这证明数据转换入口可跑，但 12/900 个 step 仍需 WebShop catalog / ASIN map 消歧。
- 同一 converter 现已支持 catalog / ASIN resolver。Jingyan 共享盘上用 4 个相关 MemoryArena product-catalog shard 重跑 public conversion，marker 为 `AGENTMEMORY_MEMORYARENA_CONVERT_OK tasks=150 splits=train:120,dev:15,test:15 min_match_score=7 ambiguous_matches=0`，validator 仍为 `AGENTMEMORY_DATA_VALIDATE_OK`；summary 为 `rows=900 / ambiguous=0 / catalog=450 / found=450 / missing=450`。
- 大文件存储与证据路径：MemoryArena product DB 已全量镜像到 `/home/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/`，最终校验 `135 files / 13,517,161,526 bytes`，extra/missing/mismatch/part 均为 0；catalog-assisted conversion 证据在 `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_catalog_priority_convert_20260701-221648`。
- 正式 freeze 证据：`/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_formal_freeze_20260701-234045`，marker 为 `AGENTMEMORY_MEMORYARENA_FORMAL_FREEZE_OK`，`tasks=150 / rows=900 / train/dev/test=120/15/15 / asin_catalog=900 / ambiguous=0 / catalog_paths=11`，source sha256 `4411a2da528a33dc6aca519b49cc225895363f18b2d19b191fddb501200134ef`。这仍是数据转换证据，不是 RL 训练或 memory 能力提升证据。
- Qwen3-4B 真单卡模型 rollout 证据：`docs/agentmemorygym/evidence/20260702-qwen3-4b-rollout-smoke.md`。Jingyan 1×B200 / Transformers Qwen3-4B frozen dev run marker 为 `AGENTMEMORY_QWEN3_4B_LATEST_OBSERVATION_PROGRESS_ROLLOUT_SMOKE_OK`，`env_steps=20 / parse_successes=20 / any_episode_success=False / progress_score=0.0`；handcrafted sanity run marker 相同，`env_steps=12 / parse_successes=12 / progress_score=0.3333`。这证明真实模型链路可跑；同时说明原始 frozen observation 的商品 metadata 信息面不足。后续 SEARCH 工具已补接口，但基础 Qwen prompt 仍不会正确使用。
- Candidate metadata enriched freeze 证据：`memoryarena_enriched_freeze_20260702-014308`，`asin_catalog=900 / ambiguous=0` 保持不变；严格阈值 90 下 `candidate_metadata_status_counts={full:285, partial:605, none:10}`。这证明 leakage-safe metadata 接口已接入，但还没有覆盖全部 step。
- Full-catalog enriched freeze 证据：`memoryarena_fullcatalog_enriched_freeze_20260702-024824`，扫描 Jingyan 共享盘全量 product_catalog 67 个 shard；结果仍是 `candidate_metadata_status_counts={full:285, partial:605, none:10}`、`candidate_metadata_full_steps=285/900`。结论：全量 DB 已够，缺口在可靠匹配/字段面，不能靠“再下载更多到开发机”解决。
- SEARCH 工具代码证据：`agentenv_agentmemory/catalog_search.py`、`scripts/build_memoryarena_catalog_search_index.py`、`AgentMemoryEnv.action_search`，本地 smoke marker `AGENTMEMORY_MEMORYARENA_CONVERTER_SMOKE_OK` 已覆盖 SEARCH 小 fixture。全量 SQLite/FTS index 已建在 Jingyan 共享盘：`/home/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/agentmemory_catalog_search.sqlite`；build marker `AGENTMEMORY_CATALOG_SEARCH_INDEX_OK products=1031654`，index 约 `479M`。
- Enriched dev Qwen3-4B rerun：普通 prompt 证据 `qwen3_4b_enriched_metadata_rollout_20260702-023225` 仍 `progress_score=0.0,0.0`；metadata-aware diagnostic prompt 证据 `qwen3_4b_metadata_prompt_rollout_20260702-023436` 达到 `progress_score=0.1667,0.0`。这是接口/链路证据，不是 RL/memory 能力结果。
- SEARCH-aware Qwen3-4B smoke 证据：`qwen3_4b_search_prompt_rollout_20260702-043335`，marker 为 `AGENTMEMORY_QWEN3_4B_LATEST_OBSERVATION_PROGRESS_ROLLOUT_SMOKE_OK`，`env_steps=24 / parse_successes=24 / any_episode_success=False / progress_score=0.0,0.0`。轨迹显示 24 步全是 `SEARCH {"query":"visible candidate title"}`，说明基础 Qwen3-4B 能跑 SEARCH 动作但不会自动用真实候选标题、也不会转入 ADD/BUY。
- Scripted SEARCH baseline 证据：`docs/agentmemorygym/evidence/20260702-scripted-search-baseline.md`，Jingyan no-retry full dev 目录 `scripted_search_baseline_dev_20260702-051721`，marker `AGENTMEMORY_SCRIPTED_SEARCH_BASELINE_OK`，`episodes=15 / successes=5 / success_rate=0.3333 / mean_progress_score=0.5444 / search_calls=265 / rejected_buys=10`。
- SEARCH + verifier-feedback retry diagnostic 证据：`scripted_search_baseline_dev_retry5_20260702-052710`，`episodes=15 / successes=10 / success_rate=0.6667 / mean_progress_score=0.8222 / search_calls=365 / buy_calls=88 / rejected_buys=14 / max_buy_attempts=5`。这是接口/可解性诊断，不是 one-shot policy 质量，也不是 RL/memory 能力提升。
- Failure audit 证据：`scripted_search_failure_audit_retry5_20260702-055120`，marker `AGENTMEMORY_SCRIPTED_SEARCH_FAILURE_AUDIT_OK`，残余失败类型全是 `compatibility_filter_excluded_target`。
- Soft-fallback verifier diagnostic 证据：`scripted_search_baseline_dev_softretry7_20260702-055137`，`episodes=15 / successes=15 / success_rate=1.0 / mean_progress_score=1.0 / search_calls=420 / buy_calls=109 / rejected_buys=19 / compatibility_fallback=ranked-all-after-compatible`；combined audit `scripted_search_failure_audit_softretry7_20260702-055940` 显示该 run 无 failed steps。这证明接口可解，不是 RL/memory 能力提升。
