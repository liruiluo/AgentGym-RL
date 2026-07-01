# 阶段计划

## Stage 0：方向重定位与文档收口

目标：把项目从 MemoryAgentBench-first 改成 MemoryArena / 电商捆绑购物优先。

完成标准：

- 本地文档更新。
- Notion 分页面更新。
- 父页短结论不再写 MemoryAgentBench AR/CR first。
- 阶段计划明确“先文档与 Notion，再代码”。

## Stage 1a：0 卡本地代码/schema/server 检查

目标：在 Mac/ZBMac 这种 0 卡机器上，只验证 `agentenv-agentmemory` 最小环境的静态、数据、schema、server API 和 guard 逻辑。此阶段不算单卡 smoke。

完成标准：

- direct environment scripted smoke 通过。
- compile/import 通过。
- server-client smoke 通过。
- 默认 smoke 任务从 `agentenv_agentmemory/data/bundled_shopping_smoke.jsonl` 读取。
- smoke split 文件存在：`train/dev/test` 各一个 item。
- 数据 validator 通过：`AGENTMEMORY_DATA_VALIDATE_OK`。
- candidate product id 不直接泄漏关键尺寸答案。
- server `/metadata` 返回 task_count / task_ids / splits。
- AgentGym-RL raw-history guard 通过：`AGENTMEMORY_CONTEXT_POLICY_SMOKE_OK`。
- 明确当前限制：完整 AgentGym/verl rollout 未在 GPU/torch 单卡环境验证。

## Stage 1b：真实单卡 GPU smoke

目标：在有 GPU 且装好 torch/AgentGym/verl 的干净 lane 上验证 client import、rollout 小样本和 guard 行为。

完成标准：

- 默认不占用用户已分配给 continual-reasoning gym、Jingyan 或 CRL eval 的现有 lane；本轮 Jingyan 1×B200 已由用户明确释放并授权用于 AgentMemoryGym smoke。
- 已在 Jingyan 1×B200 上通过 torch CUDA、real AgentGym adapter import、server metadata、real client metadata、`init_env_client` metadata smoke。
- 证据：`TORCH_CUDA_OK`、`AGENTMEMORY_REAL_ADAPTER_IMPORT_OK`、`SERVER_METADATA_SINGLE_GPU_OK`、`AGENTMEMORY_REAL_CLIENT_METADATA_SINGLE_GPU_OK`、`VERL_INIT_ENV_CLIENT_AGENTMEMORY_SINGLE_GPU_OK`。
- diagnostic rollout 若显式允许 raw-history，必须标记不可计入正式结果。
- latest-observation rollout context 已实现并通过 smoke：`AGENTMEMORY_LATEST_OBSERVATION_PROMPT_SMOKE_OK`、`AGENTMEMORY_ROLLOUT_CONTEXT_ALIGNMENT_SMOKE_OK`。
- formal rollout 不再走 raw-history；多轮 episode 展平成每个 action 一条 PPO 样本，trainer 用 `rollout_parent_indices` 对齐原 batch。
- Qwen3-4B / Transformers 小模型 rollout 已在 Jingyan 1×B200 上跑通真实模型→action parser→env step 链路，证据 marker 为 `AGENTMEMORY_QWEN3_4B_LATEST_OBSERVATION_PROGRESS_ROLLOUT_SMOKE_OK`；但 frozen MemoryArena dev 样本 `progress_score=0.0`，还不是任务成功或训练收益证据。

## Stage 2：MemoryArena 电商数据转换

目标：把 bundled web shopping 任务改成可训练 item。

当前进展：

- 已新增 MemoryArena bundled-shopping converter 入口，可把 public `bundled_shopping/data.jsonl` 转成 AgentMemoryGym JSONL。
- 全量 150 条 public bundled-shopping smoke 已可转换为 `train/dev/test = 120/15/15`，并通过 data validator。
- converter 会生成 target-match audit report；无 catalog 时 heuristic 在 900 个 step 中有 12 个 tied/ambiguous match。
- 已新增 catalog / ASIN resolver；在 Jingyan 共享盘的 4 个相关 product-catalog shard 上重跑 public conversion，900 个 step 的 ambiguous match 已降为 0。
- MemoryArena product DB 已全量镜像到 Jingyan 共享盘：`/home/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/`，不落开发机本地盘；最终校验为 `135 files / 13,517,161,526 bytes`，extra/missing/mismatch/part 均为 0。
- 正式 freeze 已完成：`memoryarena_formal_freeze_20260701-234045`，`train/dev/test=120/15/15`，`rows=900 / ambiguous=0 / asin_catalog=900 / catalog_paths=11`，source sha256 为 `4411a2da528a33dc6aca519b49cc225895363f18b2d19b191fddb501200134ef`。
- Qwen3-4B frozen dev rollout 暴露下一层代码缺口：当前 converted observation 只含候选标题和 source-option，不含所有候选的 rating/price/review 等 product DB 字段，也没有 product-catalog `SEARCH` 工具；因此 highest-rated / highest-priced / budget 类任务对模型不公平，不能直接进入正式训练结果口径。

完成标准：

- item schema。
- train/dev/test split；正式 freeze 已给出 `120/15/15` item-id 文件。
- accept-reject / compatibility map。
- reward decomposition。
- normalized trajectory info。
- WebShop catalog / ASIN map 或官方 option-to-ASIN 对齐源消掉 ambiguous target matches；正式 freeze 已做到 `asin_catalog=900 / ambiguous=0`。
- 下一步需把 product DB 元数据或 SEARCH 工具接入 observation/action space，再重跑小模型 rollout，至少验证 BUY / memory / feedback loop 能在 frozen dev 上产生非零进度。

## Stage 3：基线 smoke

目标：建立 no-memory、full-context、fixed-RAG、heuristic memory manager 的小规模结果。

完成标准：

- 每个 baseline 至少跑一组电商小样本。
- 报告成功率、进度、兼容率和 memory cost。

## Stage 4：正式后训练

目标：等明天新 8 卡机器后启动正式训练。当前 8 卡机器不占用，因为正在给 continual-reasoning gym 项目使用。

完成标准：

- 8 卡资源确认。
- 训练配置固定。
- baseline 和 reward 不再临时漂移。
- 训练日志和评测链路可复现。
