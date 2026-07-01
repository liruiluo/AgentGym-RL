# AgentMemoryGym Notion ↔ Local Map

## Source of truth split

- **Local docs source**: `code/AgentGym-RL/docs/agentmemorygym/`
- **Notion mirror**: `AgentMemoryGym-RL 项目文档（分页面版，2026-06-21）`
- **Code source**: `code/AgentGym-RL/` plus `code/AgentGym-RL/AgentGym/` submodule.
- **Current policy**: local Markdown is the editable source for the current direction; Notion is the human-readable mirror. No continuous auto-sync is installed yet; sync is manual/API-driven.

## Notion page map

| Notion page | Page id | Local source |
|---|---|---|
| Project home | `3861baa5-b6c9-8184-a164-f564cc487bc6` | patched summary blocks only |
| 00 Workspace README | `3861baa5-b6c9-81d5-9379-f782b6be2d91` | `notion/00-workspace-readme.md` |
| 01 项目概述 | `3861baa5-b6c9-8126-9638-e1b029bdd6f0` | `notion/01-project-overview.md` |
| 02 代码实现方案当前决策 | `3861baa5-b6c9-8136-b27d-e6055eacf18b` | `notion/02-implementation-decision.md` |
| 03 训练框架方案比较与代码承载决策 | `3861baa5-b6c9-8162-b793-e1e1216aeade` | `notion/03-training-framework-decision.md` |
| 04 框架设计 v0 | `3861baa5-b6c9-8173-b716-e9b3526dceab` | `notion/04-framework-design-v0.md` |
| 05 研究问题与创新点 | `3861baa5-b6c9-818e-8769-f19afdb4e99d` | `notion/05-research-questions.md` |
| 06 实验分析设计 | `3861baa5-b6c9-8163-baf6-c5ffc8793975` | `notion/06-experiment-analysis.md` |
| 07 评估设计 | `3861baa5-b6c9-818d-9e4f-f847da1eba60` | `notion/07-evaluation-design.md` |
| 08 基线与相关工作 | `3861baa5-b6c9-81e2-bd86-f69574b71484` | `notion/08-baselines.md` |
| 09 阶段计划 | `3861baa5-b6c9-8172-90fa-e7e40b96d921` | `notion/09-stage-plan.md` |
| 10 下一步执行清单 | `3861baa5-b6c9-81e5-9b8d-e7c33cc5bd8e` | `notion/10-next-actions.md` |
| 11 代码目录说明 | `3861baa5-b6c9-81bf-a3e3-cb5ea17165f5` | `notion/11-code-readme.md` |
| 12 证据台账 | `3861baa5-b6c9-819e-8216-f81dd1af60c5` | `notion/12-evidence-ledger.md` |
| 13 逐句对照表 | `3861baa5-b6c9-81c7-be3d-c4a68b9fe0b1` | `notion/13-sentence-map.md` |

## Last sync

- Date: 2026-07-01
- Backend: official Notion REST API via local `~/.config/myagent/notion-api.env` token.
- Verification: readback markers checked for parent page plus pages 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10 in the first sync; pages 00, 02, 03, 04, 09, 10, 11, 12, 13 were refreshed after the code-skeleton boundary update; pages 09, 10, 11, 12 were refreshed again after JSONL split/validator updates; pages 00, 02, 03, 09, 10, 11, 12 were refreshed after the 0-card-vs-single-GPU boundary correction; pages 09, 10, 11, 12 were refreshed after the Jingyan 1×B200 single-GPU env/client/init smoke.
- Important current markers: `MemoryArena / 电商捆绑购物优先`, `MemoryAgentBench 降级`, `先改本地文档和 Notion 文档，再改代码`, `代码草稿可以保留在工作树`, `中性 product_id`, `bundled_shopping_smoke.jsonl`, `AGENTMEMORY_DATA_VALIDATE_OK`, `SERVER_CLIENT_JSONL_SMOKE_OK`, `SERVER_CLIENT_SPLIT_SMOKE_OK`, `Stage 1a：0 卡本地代码/schema/server 检查`, `Stage 1b：真实单卡 GPU smoke`, `VERL_INIT_ENV_CLIENT_AGENTMEMORY_SINGLE_GPU_OK`, `AGENTMEMORY_ROLLOUT_CONTEXT_ALIGNMENT_SMOKE_OK`.
