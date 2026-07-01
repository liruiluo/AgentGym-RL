# 研究问题与创新点

## 核心问题

如何把短期上下文管理、长期记忆维护、任务行动、可验证奖励、记忆成本和行为归因组织到同一个可训练的 agent memory Gym 中？

## 可证伪假设

1. 随着交互轮数、会话跨度和信息依赖距离增加，显式 memory policy 的收益应更明显。
2. 合适的 memory state 可以降低 raw-history dependence，让后续决策不完全依赖原始长历史。
3. RL 后训练可能学到比固定 harness 更好的 memory action 选择，但这必须通过 baseline 和 ablation 证明。

## 创新点写法

- 不是“第一个 RL memory 方法”。
- 是一个统一的 agentic memory post-training Gym。
- 重点在环境改写、动作空间、奖励、split、baseline 和行为分析。
- Hero 场景是电商捆绑购物，直接对应真实产品价值。

## 需要避免的过强表述

- 不说现有 benchmark 不能评估 memory；正确说法是它们多是 evaluation-first，还缺统一训练闭环。
- 不说 harness 没用；正确说法是 harness 保下限，RL 后训练尝试提上限。
- 不说 smoke 结果证明 RL 有效；RL 改善必须等正式训练和 baseline 对比。
