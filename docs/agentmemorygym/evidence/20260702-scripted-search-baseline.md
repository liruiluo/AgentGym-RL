# 2026-07-02 scripted SEARCH baseline / heuristic memory manager

## Scope

This evidence records a **scripted SEARCH baseline** for AgentMemoryGym
MemoryArena bundled shopping. It is a heuristic memory manager that uses the
real `AgentMemoryEnv` action interface (`SEARCH`, `BUY`, `ADD`, `RETRIEVE`) on
the frozen dev split and the full shared-disk product-catalog SQLite/FTS index.

It is **not** RL training, not a learned policy result, and not evidence that
memory ability has improved. The purpose is to separate environment/interface
solvability from untrained Qwen prompt behavior before 8-card RL.

## Code path

```text
AgentGym/agentenv-agentmemory/scripts/run_scripted_search_baseline.py
```

Policy boundary:

- Reads only visible candidate titles, current instruction text, its own
  `ADD`/`RETRIEVE` memory, and public product metadata returned by `SEARCH`.
- Calls `SEARCH` with actual visible candidate titles, not placeholders.
- Does not use `target_product_id` for action selection. `--include-target-audit`
  only writes target ids into saved audit rows.
- Chooses by parsed preference (`highest/lowest-rated`,
  `highest/lowest-priced`) after compatibility filtering when parseable.

## Data / index

Formal freeze:

```text
/media/cfs/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_formal_freeze_20260701-234045
train/dev/test=120/15/15
asin_catalog=900 / ambiguous=0
```

Full SEARCH index on Jingyan shared disk:

```text
/media/cfs/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/agentmemory_catalog_search.sqlite
AGENTMEMORY_CATALOG_SEARCH_INDEX_OK products=1031654
size ~= 479M
```

Current cpu9n writeable evidence root:

```text
/media/cfs/luolirui.1/agentmemorygym-smoke-evidence/
```

The product DB and index stay on shared disk; they are not copied to the Mac or
devbox. The `/media/cfs/...` paths are the canonical cpu9n/shared-disk access
paths; older `/home/ai-jingyan-train/...` paths in historical logs are Jingyan
GPU-container aliases.

## Local 0-card validation

Local Mac/ZBMac validation was limited to static code checks:

```text
PYTHONPATH=agentenv-agentmemory python3 -m py_compile agentenv-agentmemory/scripts/run_scripted_search_baseline.py
focused matcher checks:
  6 Outlet matches "6 outlets"
  12 Outlet does not match "12ft cord ... 6 outlets"
  Vitamin E no longer matches source_option=e
  Berry does not match strawberry
  broad color-set labels match "9 vibrant colors"
```

This is 0-card validation only, not a single-GPU smoke.

## Current strict no-retry dev run: semantic matcher fixed

Evidence directory:

```text
/media/cfs/luolirui.1/agentmemorygym-smoke-evidence/scripted_search_baseline_dev_noretry_semanticfix5_20260702-074234
```

Marker / summary:

```text
AGENTMEMORY_SCRIPTED_SEARCH_BASELINE_OK
episodes=15
successes=6
success_rate=0.4000
mean_progress_score=0.5778
search_calls=275
buy_calls=61
rejected_buys=9
max_buy_attempts=1
```

Failure audit:

```text
/media/cfs/luolirui.1/agentmemorygym-smoke-evidence/scripted_search_failure_audit_dev_noretry_semanticfix5_20260702-075010
AGENTMEMORY_SCRIPTED_SEARCH_FAILURE_AUDIT_OK
failure_type_counts={"attempt_budget_below_target_rank": 7, "compatibility_filter_excluded_target": 2}
```

Interpretation:

- One-shot/no-retry remains a weak heuristic baseline: it often selects a
  plausible but verifier-rejected top-ranked option.
- The current semantic matcher no longer leaks through `source_option`, no
  longer treats token substrings such as `Berry` in `strawberry` as a match, and
  still handles narrow plural/phrase cases plus broad color-set descriptions.

## Current retry diagnostic: SEARCH + verifier-feedback retry

A second run allowed up to five ranked BUY attempts per subtask. This should be
reported as **SEARCH + verifier-feedback retry diagnostic**, not as one-shot
policy quality.

Evidence directory:

```text
/media/cfs/luolirui.1/agentmemorygym-smoke-evidence/scripted_search_baseline_dev_retry5_semanticfix5_20260702-074234
```

Marker / summary:

```text
AGENTMEMORY_SCRIPTED_SEARCH_BASELINE_OK
episodes=15
successes=13
success_rate=0.8667
mean_progress_score=0.9000
total_env_steps=612
search_calls=385
buy_calls=91
rejected_buys=10
max_buy_attempts=5
```

Failure audit:

```text
/media/cfs/luolirui.1/agentmemorygym-smoke-evidence/scripted_search_failure_audit_dev_retry5_semanticfix5_20260702-075320
AGENTMEMORY_SCRIPTED_SEARCH_FAILURE_AUDIT_OK
failure_type_counts={"compatibility_filter_excluded_target": 2}
```

Residual failures:

```text
ck step1 target ma_ck_b_e excluded by allowed=Compact
cu step2 target ma_cu_c_b excluded by allowed=Raisin
```

Interpretation:

- Current strict retry5 improves over the earlier retry diagnostic from `10/15`
  to `13/15` after semantic matcher fixes.
- The remaining two failures are not storage/download issues and not hidden target
  freeze issues. They are real interface/normalization gaps: the MemoryArena
  compatibility label (`Compact`, `Raisin`) is not recoverable from the correct
  visible option title / top SEARCH result.
- This still does not justify any claim about RL improving memory. It is an
  interface and solvability probe before formal training.

## Historical diagnostics kept for comparison

Earlier runs before semantic matcher fixes:

```text
scripted_search_baseline_dev_20260702-051721
  no-retry: 5/15, mean_progress=0.5444
scripted_search_baseline_dev_retry5_20260702-052710
  retry5: 10/15, mean_progress=0.8222
scripted_search_failure_audit_retry5_20260702-055120
  failure_type_counts={"compatibility_filter_excluded_target": 5}
```

Root causes found from those audits:

- `source_option=e` is non-semantic and must not make labels like `Vitamin E`
  match.
- Unordered token-bag matching made `12 Outlet` match `12ft cord ... 6 outlets`;
  multi-token labels must match as adjacent phrases with only narrow plural
  variants.
- Raw substring matching made labels such as `Berry` match `strawberry`; use
  token matching instead.
- Some generated tasks describe a broad set, e.g. `one of: Red, Blue, Green,
  Yellow, Pink`, while the correct option says `9 vibrant colors`; keep a narrow
  broad-color-set exception instead of reopening arbitrary substring matching.

## Soft-fallback verifier diagnostic

A third diagnostic run keeps the same visible candidate titles and public SEARCH
metadata, but changes the scripted heuristic from strict compatibility filtering
to explicit soft fallback:

```text
--compatibility-fallback ranked-all-after-compatible
--max-buy-attempts 7
```

The policy first tries candidates that match the parsed compatibility label; if
those are rejected by the environment verifier, it tries the remaining visible
candidates ranked by the same SEARCH metadata. This is an **exhaustive
verifier-feedback diagnostic**, not a deployable one-shot policy and not a
learned policy result.

Current semanticfix5 evidence directory:

```text
/media/cfs/luolirui.1/agentmemorygym-smoke-evidence/scripted_search_baseline_dev_softretry7_semanticfix5_20260702-075423
```

Current marker / summary:

```text
AGENTMEMORY_SCRIPTED_SEARCH_BASELINE_OK
episodes=15
successes=15
success_rate=1.0000
mean_progress_score=1.0000
total_env_steps=674
search_calls=420
buy_calls=104
rejected_buys=14
max_buy_attempts=7
compatibility_fallback=ranked-all-after-compatible
```

Failure audit:

```text
/media/cfs/luolirui.1/agentmemorygym-smoke-evidence/scripted_search_failure_audit_dev_softretry7_semanticfix5_20260702-075423
AGENTMEMORY_SCRIPTED_SEARCH_FAILURE_AUDIT_OK
failure_type_counts={}
```

Interpretation:

- The frozen MemoryArena dev items are completable through the fair environment
  action surface when the policy can use SEARCH plus verifier feedback.
- Strict title-level compatibility filtering is too brittle to be the final
  heuristic memory manager.
- RL training should learn when to trust compatibility labels, when to broaden
  candidate search, and how much verifier-feedback exploration is worth; the
  soft-fallback diagnostic only proves that the environment/tool interface can
  support that behavior.

## Next closure bar

Before 8-card RL, keep this baseline as a reproducible reference and use its
failure cases to tighten one of the following:

1. product option-to-catalog normalization / metadata extraction;
2. richer `SEARCH` result fields or candidate-level all-metadata exposure; or
3. a learned policy that can use `SEARCH`, `ADD`, `RETRIEVE`, and verifier
   feedback without target leakage.
