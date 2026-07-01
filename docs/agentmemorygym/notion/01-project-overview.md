# 项目概述

## 当前定位

AgentMemoryGym 现在定位为 **Agentic RL Memory 后训练 Gym**：把长程、多会话、依赖历史状态的 agent 任务，改写成可训练、可评测、可归因的 RL 环境，用来提升模型自己的 memory policy。

本项目不再把 MemoryAgentBench AR/CR 当作第一主线。MemoryAgentBench 仍是 baseline、数据和评测参考；第一主线改为 **MemoryArena 式任务，尤其电商捆绑/序列购物**。

## 为什么要做

现实里的复杂 agent 任务通常不是一次性信息充分的单轮任务。以电商为例，用户可能先买电视，过几轮才买支架、电视柜、线缆或音响。后续购买必须记住前面电视的尺寸、重量、接口和摆放限制，否则会买到不兼容商品。用户一般无法第一轮就把全套需求说完，因此 agent 需要跨轮维护可用记忆。

固定 harness 可以保证下限，例如固定写入、固定检索、固定摘要和人工兼容表。但 harness 主要依赖人类先验，不能直接提升模型选择 memory action 的上限。本项目关注 RL 后训练能否让 agent 学会何时写入、更新、删除、检索、摘要和过滤。

## 为什么是 Gym

现有 memory benchmark 已经很多，继续只做静态榜单边际收益有限。现有 RL-memory 方法也不少，但常绑定在各自狭窄任务和记忆形式里，难以公平比较、复现和 scaling 分析。

AgentMemoryGym 要补的是中间层：统一环境接口、统一 memory tool action space、可验证奖励、训练/评估切分、基线协议和行为分析字段。

## Hero 环境：捆绑网络购物

第一版 demo 和训练环境优先做电商捆绑购物：

```text
Session 1：买 75 英寸电视。
Session 2：买兼容这台电视的支架，但当前请求不再重复电视尺寸。
Session 3：买兼容同一台电视的电视柜。
```

环境维护 hidden bundle state，例如 `tv_size=75`、`tv_weight_kg=32`、`vesa=400x400`。Agent 只看到当前 active context 和当前候选商品；长期记忆必须通过 memory tools 写入和取回。

## 第一版贡献写法

- AgentMemoryGym：面向 agent memory policy 的 RL/Gym 后训练框架。
- MemoryArena/e-commerce-first 的可训练环境改写方案。
- 统一 STM/LTM memory action space。
- 可验证奖励、轨迹字段、评测指标和 baseline suite。
- 行为分析：判断收益来自有效记忆，而不是长上下文、固定检索或过度工具调用。

不要写成“第一个 RL 训练 agent memory 的方法”。AgeMem、Memory-R1、MemAct、MEM1、MemAgent、UMA 等已经覆盖局部 RL-memory 路线。
