# MemoryArena Catalog Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or inline verified execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a catalog/ASIN-aware resolver so MemoryArena bundled-shopping tasks can be converted into AgentMemoryGym data with fewer or zero ambiguous target-option matches.

**Architecture:** Keep the existing heuristic converter as fallback. Add a lightweight catalog index that maps ASIN to product title/metadata from MemoryArena product DB JSON/JSONL files, then resolve prompt options by exact/fuzzy overlap against the catalog title before falling back to answer-attribute overlap. Large product DB lives on Jingyan shared disk; local Mac only runs code/static and small fixture validation.

**Tech Stack:** Python stdlib JSON/argparse/dataclasses, existing AgentGym `agentenv-agentmemory` package, Jingyan shared disk for full data.

---

### Task 1: Inventory current converter and data shape

**Files:**
- Read: `AgentGym/agentenv-agentmemory/agentenv_agentmemory/memoryarena_converter.py`
- Read: `AgentGym/agentenv-agentmemory/scripts/convert_memoryarena_bundled_shopping.py`
- Read: `AgentGym/agentenv-agentmemory/scripts/smoke_memoryarena_converter.py`

- [ ] Inspect current target matching and report schema.
- [ ] Inspect one MemoryArena task and one product catalog shard.

### Task 2: Add catalog index and resolver

**Files:**
- Modify: `AgentGym/agentenv-agentmemory/agentenv_agentmemory/memoryarena_converter.py`
- Modify: `AgentGym/agentenv-agentmemory/scripts/convert_memoryarena_bundled_shopping.py`

- [ ] Add `CatalogProduct` / catalog loading helpers supporting JSON arrays, JSONL, and nested product dicts.
- [ ] Add CLI flags for catalog JSON files/directories.
- [ ] Resolve target option via `target_asin -> catalog title/name -> option text`, with deterministic tie handling and heuristic fallback.
- [ ] Extend audit report with resolver/match/source fields.

### Task 3: Add focused smoke coverage

**Files:**
- Modify: `AgentGym/agentenv-agentmemory/scripts/smoke_memoryarena_converter.py`
- Maybe create small temp catalog fixture inside the smoke script.

- [ ] Test catalog resolver beats ambiguous heuristic on a small fixture.
- [ ] Keep existing marker `AGENTMEMORY_MEMORYARENA_CONVERTER_SMOKE_OK`.

### Task 4: Download product DB to shared disk and run full conversion there

**Paths:**
- Shared data root: `/home/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/`
- Smoke repo: `/home/ai-jingyan-train/luolirui.1/post-train/code/AgentGym-RL-agentmemory-smoke`

- [ ] Create shared data root on Jingyan.
- [ ] Use Mac/JD only as control plane/streaming source; avoid devbox persistent storage.
- [ ] Download/sync full MemoryArena product DB into shared disk.
- [ ] Run converter against public bundled shopping JSONL + catalog resolver.

### Task 5: Docs, Notion, commit

**Files:**
- Modify: `docs/agentmemorygym/evidence/20260701-memoryarena-converter.md`
- Modify: `docs/agentmemorygym/notion/09-stage-plan.md`
- Modify: `docs/agentmemorygym/notion/10-next-actions.md`
- Modify: `docs/agentmemorygym/notion/11-code-readme.md`
- Modify: `docs/agentmemorygym/notion/12-evidence-ledger.md`
- Modify: `docs/agentmemorygym/notion-local-map.md` if boundaries changed
- Modify: `AgentGym/agentenv-agentmemory/README.md`

- [ ] Record exact ambiguity/result counts and shared-disk path.
- [ ] Sync Notion pages 09/10/11/12 and verify markers.
- [ ] Commit submodule first, then main repo pointer/docs, then memory if needed.
