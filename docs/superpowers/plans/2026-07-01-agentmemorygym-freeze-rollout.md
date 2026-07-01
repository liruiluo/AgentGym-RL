# AgentMemoryGym Formal Data Freeze and Rollout Smoke Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline verified execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current AgentMemoryGym shopping environment from converter smoke into a reviewable formal-data freeze plus a real single-GPU small-model/API rollout smoke path.

**Architecture:** Keep large MemoryArena product DB files on the Jingyan shared disk. Generate immutable converted JSONL/split/report artifacts under a timestamped evidence directory, record a manifest with exact inputs and validation markers, and add minimal repo-side scripts/docs so future training can consume the frozen data without copying large files into the repo. Then run a bounded rollout smoke on Jingyan 1×B200 using the existing environment/server contract.

**Tech Stack:** Python stdlib, existing `agentenv-agentmemory` converter/validator/environment, Jingyan shared disk, Notion REST sync for human-facing docs.

---

### Task 1: Freeze catalog-assisted public bundled-shopping data

**Files/paths:**
- Read: `AgentGym/agentenv-agentmemory/scripts/convert_memoryarena_bundled_shopping.py`
- Read: `AgentGym/agentenv-agentmemory/scripts/validate_agentmemory_data.py`
- Remote output: `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_formal_freeze_<timestamp>/`

- [x] Run the converter on public `bundled_shopping/data.jsonl` with `--catalog-path /home/.../memoryarena-product-db`.
- [x] Validate output with `validate_agentmemory_data.py`.
- [x] Write `freeze_manifest.json` with source URL, repo commits, product DB verification, task/split/report counts, and ambiguity summary.
- [x] Verify `ambiguous=0`, split counts, and no `.part` files.

### Task 2: Add a repo-side freeze helper if the command sequence is reusable

**Files:**
- Maybe create: `AgentGym/agentenv-agentmemory/scripts/freeze_memoryarena_bundled_shopping.py`
- Modify: `AgentGym/agentenv-agentmemory/README.md`

- [x] Keep helper small: parse input/catalog/output root, call existing converter/validator, summarize report.
- [x] Do not commit datasets or product DB into repo.
- [x] Compile and smoke.

### Task 3: Single-GPU rollout smoke

**Files/paths:**
- Remote repo: `/home/ai-jingyan-train/luolirui.1/post-train/code/AgentGym-RL-agentmemory-smoke`
- Evidence: `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/<rollout-run>/`

- [x] Confirm Jingyan 1×B200 lane is usable and not the 8-card continual-reasoning lane.
- [x] Run a bounded true rollout path using frozen MemoryArena dev data with Transformers Qwen3-4B.
- [x] Record exact command, logs, marker, and whether this is model/API rollout or scripted policy.
- [x] Record negative result honestly: frozen dev rollout produced valid env steps but `progress_score=0.0`; handcrafted smoke progressed to `1/3` but did not finish.
- [x] Identify next code gap: converted MemoryArena observation needs product DB metadata for all candidates or a product-catalog `SEARCH` tool before formal training/eval claims.

### Task 4: Docs, Notion, commits, memory

**Files:**
- Modify: `docs/agentmemorygym/evidence/20260701-memoryarena-converter.md`
- Modify: `docs/agentmemorygym/notion/09-stage-plan.md`
- Modify: `docs/agentmemorygym/notion/10-next-actions.md`
- Modify: `docs/agentmemorygym/notion/11-code-readme.md`
- Modify: `docs/agentmemorygym/notion/12-evidence-ledger.md`
- Modify: `docs/agentmemorygym/notion-local-map.md`

- [x] Record freeze path and markers.
- [x] Sync Notion pages 09/10/11/12 and verify formal-freeze markers.
- [x] Commit submodule first if code changed; then main repo docs/pointer.
- [x] Sync Notion pages 09/10/11/12 again after Qwen3-4B rollout smoke and verify markers.
- [ ] Commit rollout docs and memory.

### Task 5: Candidate metadata enrichment before formal RL

**Files:**
- Modify: `AgentGym/agentenv-agentmemory/agentenv_agentmemory/memoryarena_converter.py`
- Modify: `AgentGym/agentenv-agentmemory/scripts/convert_memoryarena_bundled_shopping.py`
- Modify: `AgentGym/agentenv-agentmemory/scripts/freeze_memoryarena_bundled_shopping.py`
- Modify: `AgentGym/agentenv-agentmemory/scripts/smoke_memoryarena_converter.py`
- Modify after evidence: `docs/agentmemorygym/evidence/20260702-qwen3-4b-rollout-smoke.md`
- Modify after evidence: `docs/agentmemorygym/notion/{09-stage-plan.md,10-next-actions.md,11-code-readme.md,12-evidence-ledger.md}`

- [x] Add converter flags `--enrich-candidate-metadata` and `--candidate-metadata-min-score`.
- [x] During enriched conversion, scan the product DB from the Jingyan shared disk; do not copy product DB files into Mac/devbox or commit them.
- [x] Attach comparable metadata only when every visible candidate in a subtask has a catalog match above the threshold, so target-only leakage is impossible.
- [x] Expose only task-relevant comparable fields in observations: `average_rating`, `price_usd`, and `total_reviews`; keep ASIN/source paths in report/manifest only.
- [x] Extend report/manifest with metadata coverage and match-score audit fields.
- [x] Add smoke tests covering full metadata enrichment and unmatched no-partial-leak behavior.
- [x] Run compileall and converter smoke locally as 0-card validation.
- [x] Run enriched formal freeze on Jingyan shared disk using the Jingyan shared-disk product DB root and explicit relevant catalog shards.
- [ ] Rerun Qwen3-4B true single-GPU rollout smoke on the enriched dev split and record it as plumbing/interface evidence, not RL improvement.
- [x] Sync Notion pages 09/10/11/12 from local markdown after evidence is written.
