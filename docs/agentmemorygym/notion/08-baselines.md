# 基线与相关工作

## Baseline families

1. No-memory agent：只看当前 observation。
2. Budgeted full-context：有限窗口保留历史。
3. Fixed RAG memory：固定写入和相似度检索。
4. Heuristic memory manager：规则触发 ADD/UPDATE/RETRIEVE，并可在电商环境中调用公开 product-catalog `SEARCH`。当前 scripted SEARCH baseline 已在 dev split 得到 no-memory `0/15`、full-context `6/15`、memory-tool no-retry `6/15`、memory-tool retry5 `13/15`、soft-fallback verifier diagnostic `15/15`。这些只作为接口/可解性和 memory-dependence baseline，不是 RL 提升结果。
5. Engineering memory systems：MemGPT、Mem0、HippoRAG 等作为系统参考。
6. Existing RL-memory methods：AgeMem、Memory-R1、MemAct、MEM1、MemAgent、UMA 等。
7. Learned AgentMemoryGym policy：在同一环境、奖励、split 和 tool schema 下训练。

## Novelty guardrail

已有工作已经覆盖局部 RL-memory 路线，所以本项目不能写成“首次用 RL 训练 agent memory”。合理贡献是统一 Gym/RL-style 后训练环境、评测协议、基线和行为分析。

## 当前参考关系

- MemoryArena：第一优先环境来源和评估思想来源。
- AgentGym-RL：训练后端和多轮环境接口来源。
- AgeMem：仅作为 STM/LTM memory tool 语义来源；不借其三阶段课程学习路线。
- MemoryAgentBench：baseline / 数据参考 / 对比评测来源，不再是第一环境主线。
