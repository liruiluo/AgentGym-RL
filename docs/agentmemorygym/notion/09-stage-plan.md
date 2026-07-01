# 阶段计划

## Stage 0：方向重定位与文档收口

目标：把项目从 MemoryAgentBench-first 改成 MemoryArena / 电商捆绑购物优先。

完成标准：

- 本地文档更新。
- Notion 分页面更新。
- 父页短结论不再写 MemoryAgentBench AR/CR first。
- 阶段计划明确“先文档与 Notion，再代码”。

## Stage 1：单卡环境 smoke

目标：在单卡上验证 `agentenv-agentmemory` 最小环境、server-client、AgentGym-RL client registry。

完成标准：

- direct environment scripted smoke 通过。
- compile/import 通过。
- server-client smoke 通过。
- 默认 smoke 任务从 `agentenv_agentmemory/data/bundled_shopping_smoke.jsonl` 读取。
- smoke split 文件存在：`train/dev/test` 各一个 item。
- 数据 validator 通过：`AGENTMEMORY_DATA_VALIDATE_OK`。
- candidate product id 不直接泄漏关键尺寸答案。
- 记录当前限制：AgentGym-RL 原始 rollout 可能仍保留完整历史，需要后续处理 raw-history leakage。

## Stage 2：MemoryArena 电商数据转换

目标：把 bundled web shopping 任务改成可训练 item。

完成标准：

- item schema。
- train/dev/test split；当前 smoke split 只是占位，真实转换后需要重新冻结 item-id。
- accept-reject / compatibility map。
- reward decomposition。
- normalized trajectory info。

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
