# 评估设计

## 评估原则

记忆必须是完成任务的功能性组成部分。Agent 只是能复述过去不够；它必须能用记忆指导下一步行动，例如买到兼容商品、排出满足约束的旅行计划、推进依赖搜索或复用前文推导结论。

## 主指标

- 成功率 / 任务完成率。
- 进度得分。
- 兼容约束满足率。
- Memory ablation drop。
- 每成功任务的 memory cost。
- 错误类型分布。

## 电商指标

- Bundle completion rate。
- Compatibility satisfaction rate。
- Wrong purchase rate。
- Cross-session attribute recall/use rate。
- 支架、电视柜、线缆等后续商品的条件命中率。

## 评测切分

必须区分 train/dev/test。不能把最终测试集用于 reward 调参或 prompt/harness 调参。MemoryArena 转换数据进入训练前，应先固定 item id、任务族、难度和依赖距离字段。

## 报告要求

每个结果同时报告任务指标和 memory 行为指标。只报告成功率不足以说明 memory policy 是否真的学会了记忆。
