# AgentMemoryGym Scripted SEARCH Baseline Plan

> For agentic workers: inline verified execution. Do not report this as RL improvement.

**Goal:** Add and run a scripted SEARCH baseline / heuristic memory manager for frozen MemoryArena bundled shopping, proving whether the fair `SEARCH` interface can support task progress before 8-card RL.

**Architecture:** Implement a repo-side script under `AgentGym/agentenv-agentmemory/scripts/` that drives `AgentMemoryEnv` through real `SEARCH`, `ADD`, `RETRIEVE`, and `BUY` actions. The policy reads only visible candidate titles, current instruction text, its own memory, and public metadata returned by `SEARCH`; it must not use `target_product_id` for action selection. It writes JSONL trajectories and a summary under an evidence directory on Jingyan shared disk.

**Tech Stack:** Python stdlib + existing `agentenv_agentmemory` environment and SQLite/FTS `SEARCH` index.

## Tasks

- [x] Create `scripts/run_scripted_search_baseline.py`.
- [x] Parse visible candidates and call environment `SEARCH` for actual candidate titles, not placeholders.
- [x] Maintain lightweight memory via `ADD`/`RETRIEVE` after each purchase.
- [x] Choose candidates by instruction preference (`highest/lowest-rated`, `highest/lowest-priced`) after filtering by compatibility notes when parseable; fall back to metric-only if compatibility is ambiguous.
- [x] Save `summary.json`, `episodes.jsonl`, and `actions.jsonl` with task success/progress/memory cost.
- [x] Run local 0-card compile/smoke.
- [x] Sync to Jingyan runtime copy and run on formal dev split with shared-disk full SEARCH index.
- [x] Update evidence docs/Notion sources and commit submodule then parent. (docs + Notion sync done; commit pending in next step)
