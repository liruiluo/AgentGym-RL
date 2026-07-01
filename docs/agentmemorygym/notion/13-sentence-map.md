# 逐句对照表

本页暂改为方向更新后的轻量 claim map。正式论文写作前再扩成逐句证据表。

| 文档位置 | 判断 | Claim | 备注 |
|---|---|---|---|
| 项目概述 | AgentMemoryGym 是 agentic memory post-training Gym。 | 框架定位 | 不是新静态 benchmark，也不是先押单算法。 |
| 项目概述 | 电商捆绑购物是 hero environment。 | 环境主线 | 对应真实多轮序列推荐/购买需求。 |
| 框架设计 | LTM 默认隐藏，RETRIEVE 后进入 STM。 | 状态建模 | 更接近 POMDP。 |
| 评估设计 | 成功率必须配合 memory 行为指标。 | 评估原则 | 防止把长上下文或固定检索误判为 learned memory。 |
| 基线与相关工作 | 不能 claim first RL-memory。 | Novelty guardrail | 已有 AgeMem、Memory-R1、MemAct 等。 |
| 代码目录说明 | 代码草稿保留为 skeleton，不撤回但也不当完成实现。 | 执行边界 | 对应用户“代码草稿没必要撤”的纠偏。 |
| 阶段计划 | product_id 不应直接泄漏尺寸/兼容答案。 | 环境质量 gate | 避免最小环境被字符串捷径解掉。 |
