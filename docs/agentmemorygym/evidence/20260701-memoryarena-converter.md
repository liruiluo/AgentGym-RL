# 2026-07-01 MemoryArena bundled-shopping converter smoke

## Scope

This evidence records the first real MemoryArena/WebShop-style data conversion
entrypoint for AgentMemoryGym. It converts MemoryArena `bundled_shopping`
records from their public JSONL shape:

```text
{id, category, questions[], answers[]}
```

into the AgentMemoryGym `ShoppingTask` JSONL shape consumed by
`agentenv_agentmemory.environment.AgentMemoryEnv`.

The converter does **not** commit the MemoryArena dataset itself into this repo.
It accepts a local path or HTTPS URL and writes converted JSONL, split files, and
a target-match audit report.

## Added code

```text
AgentGym/agentenv-agentmemory/agentenv_agentmemory/memoryarena_converter.py
AgentGym/agentenv-agentmemory/scripts/convert_memoryarena_bundled_shopping.py
AgentGym/agentenv-agentmemory/scripts/smoke_memoryarena_converter.py
AgentGym/agentenv-agentmemory/agentenv_agentmemory/data/fixtures/memoryarena_bundled_shopping_sample.jsonl
```

## Local fixture smoke

```bash
cd AgentGym
python3 -m compileall -q \
  agentenv-agentmemory/agentenv_agentmemory/memoryarena_converter.py \
  agentenv-agentmemory/scripts/convert_memoryarena_bundled_shopping.py \
  agentenv-agentmemory/scripts/smoke_memoryarena_converter.py
PYTHONPATH=agentenv-agentmemory \
  python3 agentenv-agentmemory/scripts/smoke_memoryarena_converter.py
```

Marker:

```text
AGENTMEMORY_MEMORYARENA_CONVERTER_SMOKE_OK
```

The smoke converts a synthetic MemoryArena-format fixture, writes one item per
split, reloads the generated split files through `load_task_dataset`, runs a
straight target-buy plan, and verifies that a wrong first purchase is rejected.

## Public MemoryArena full-data conversion smoke

Command:

```bash
TMP=$(mktemp -d)
PYTHONPATH=agentenv-agentmemory \
  python3 agentenv-agentmemory/scripts/convert_memoryarena_bundled_shopping.py \
  --input https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/main/bundled_shopping/data.jsonl \
  --output "$TMP/memoryarena_agentmemory.jsonl" \
  --split-dir "$TMP/splits" \
  --report "$TMP/report.jsonl"
PYTHONPATH=agentenv-agentmemory \
  python3 agentenv-agentmemory/scripts/validate_agentmemory_data.py \
  --data "$TMP/memoryarena_agentmemory.jsonl" \
  --split-dir "$TMP/splits"
```

Markers / summary:

```text
AGENTMEMORY_MEMORYARENA_CONVERT_OK tasks=150 splits=train:120,dev:15,test:15 min_match_score=2 ambiguous_matches=12
AGENTMEMORY_DATA_VALIDATE_OK tasks=150 splits=train:120,dev:15,test:15
FULL_MEMORYARENA_CONVERT_SUMMARY tasks 150 rows 900 min_score 2 max_score 55 ambiguous 12
```

## Important boundary

MemoryArena answers provide target ASINs plus attributes, while the prompt lists
natural-language option descriptions. This v0 converter infers the target option
from answer-attribute overlap and records the audit row. On the full public
bundled-shopping file, 12 of 900 step-level matches are tied/ambiguous under this
heuristic.

Therefore this closes the **data-converter entrypoint** gap, but it is not yet a
formal frozen training/evaluation dataset. Before formal results, the ambiguous
rows should be resolved with a WebShop catalog / ASIN map or a stronger official
option-to-ASIN alignment source.

## Catalog / ASIN resolver update

The converter now supports optional product-catalog inputs via repeated
`--catalog-path`. It builds a `target_asin -> catalog title` index, ranks prompt
options by catalog-title similarity with attribute overlap as tie-breaker, and
falls back to the old attribute heuristic when the target ASIN is not present in
the provided catalog shards.

New local fixture marker remains:

```text
AGENTMEMORY_MEMORYARENA_CONVERTER_SMOKE_OK
```

The smoke now includes a small catalog tiebreak case where attribute-only
matching is ambiguous but the ASIN catalog title selects the correct option.

## Jingyan shared-disk product DB mirror

Per the storage boundary, large MemoryArena product DB files have been mirrored
to the Jingyan shared disk, not to the devbox local disk:

```text
/home/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/
```

The full source is:

```text
https://huggingface.co/datasets/ai-hyz/MemoryArena-product-db
```

The full repository contains 135 files / 13,517,161,526 expected bytes. The
mirror job streamed Mac network downloads directly into the Jingyan shared disk.
Final mirror verification at 2026-07-01 22:45 CST:

```text
STATUS complete file_count=135 expected_bytes=13517161526
REMOTE_EXACT_VERIFY api_files=135 api_bytes=13517161526 remote_files=135 remote_bytes=13517161526 extra=0 missing=0 mismatch=0 parts=0
```

Four catalog shards were priority-mirrored first for converter validation:

```text
product_catalog/electronics_accessories_supplies.json
product_catalog/electronics_television_video.json
product_catalog/grocery_gourmet_food_pantry_staples.json
product_catalog/grocery_gourmet_food_snacks_sweets.json
```

Remote priority-catalog conversion:

```bash
PYTHONPATH=AgentGym/agentenv-agentmemory \
  python3 AgentGym/agentenv-agentmemory/scripts/convert_memoryarena_bundled_shopping.py \
  --input /home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_bundled_shopping_data_20260701.jsonl \
  --output "$RUN/memoryarena_agentmemory.jsonl" \
  --split-dir "$RUN/splits" \
  --report "$RUN/report.jsonl" \
  --catalog-path "$ROOT/product_catalog/electronics_accessories_supplies.json" \
  --catalog-path "$ROOT/product_catalog/electronics_television_video.json" \
  --catalog-path "$ROOT/product_catalog/grocery_gourmet_food_pantry_staples.json" \
  --catalog-path "$ROOT/product_catalog/grocery_gourmet_food_snacks_sweets.json"
```

Remote evidence directory:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_catalog_priority_convert_20260701-221648
```

Markers / summary:

```text
AGENTMEMORY_MEMORYARENA_CONVERT_OK tasks=150 splits=train:120,dev:15,test:15 min_match_score=7 ambiguous_matches=0
AGENTMEMORY_DATA_VALIDATE_OK tasks=150 splits=train:120,dev:15,test:15
SUMMARY rows 900 ambiguous 0 catalog 450 found 450 missing 450 min_match 7
```

Interpretation:

- The previous 12/900 heuristic ties are eliminated under the catalog-assisted
  converter on the public bundled-shopping file.
- 450/900 steps are resolved through `asin_catalog`; 450/900 still fall back to
  attribute matching because the priority shard subset does not cover every
  target ASIN.
- This closes the target-match ambiguity in the current conversion smoke, but it
  is still data-conversion evidence, not a formal RL result or proof of improved
  memory ability.

## Jingyan 1×B200 container verification

The Jingyan container could not fetch HuggingFace directly in this run because
its HTTPS tunnel returned `503 Service Unavailable`, so the public JSONL was
downloaded on Mac and copied to the Jingyan evidence directory. The converter was
then run inside the Jingyan 1×B200 container with local input:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_bundled_shopping_data_20260701.jsonl
```

Remote evidence directory:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_convert_20260701-212422
```

Remote markers matched local full-data conversion:

```text
AGENTMEMORY_MEMORYARENA_CONVERT_OK tasks=150 splits=train:120,dev:15,test:15 min_match_score=2 ambiguous_matches=12
AGENTMEMORY_DATA_VALIDATE_OK tasks=150 splits=train:120,dev:15,test:15
```
