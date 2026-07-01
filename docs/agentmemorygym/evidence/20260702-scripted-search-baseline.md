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
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_formal_freeze_20260701-234045
train/dev/test=120/15/15
asin_catalog=900 / ambiguous=0
```

Full SEARCH index on Jingyan shared disk:

```text
/home/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/agentmemory_catalog_search.sqlite
AGENTMEMORY_CATALOG_SEARCH_INDEX_OK products=1031654
size ~= 479M
```

## Local 0-card validation

Local Mac/ZBMac validation was limited to static code checks:

```text
python3 -m compileall -q agentenv-agentmemory/scripts/run_scripted_search_baseline.py
PYTHONPATH=agentenv-agentmemory python3 agentenv-agentmemory/scripts/run_scripted_search_baseline.py --help
```

This is 0-card validation only, not a single-GPU smoke.

## Jingyan full dev run: one BUY attempt per subtask

Evidence directory:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/scripted_search_baseline_dev_20260702-051721
```

Marker / summary:

```text
AGENTMEMORY_SCRIPTED_SEARCH_BASELINE_OK
episodes=15
successes=5
success_rate=0.3333
mean_progress_score=0.5444
total_env_steps=412
search_calls=265
buy_calls=59
rejected_buys=10
```

Interpretation:

- The fair `SEARCH` interface plus a simple heuristic memory manager produces
  meaningful nonzero task progress on frozen MemoryArena dev.
- This is clearly above the SEARCH-aware Qwen3-4B prompt smoke (`0/2`, progress
  `0.0,0.0`), which repeatedly queried the literal placeholder text `visible
  candidate title` and never reached `ADD` or `BUY`.
- The one-shot heuristic is still far from solved. Top-1 SEARCH metadata can map
  a visible option to a nearby catalog variant with different price/rating, and
  some compatibility labels are not recoverable from public title-level
  metadata.

## Retry diagnostic: SEARCH + verifier-feedback retry

A second run allowed up to five ranked BUY attempts per subtask. This should be
reported as **SEARCH + verifier-feedback retry diagnostic**, not as one-shot
policy quality.

Evidence directory:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/scripted_search_baseline_dev_retry5_20260702-052710
```

Marker / summary:

```text
AGENTMEMORY_SCRIPTED_SEARCH_BASELINE_OK
episodes=15
successes=10
success_rate=0.6667
mean_progress_score=0.8222
total_env_steps=581
search_calls=365
buy_calls=88
rejected_buys=14
max_buy_attempts=5
```

Per-episode outcome:

```text
i=True/1.0000, s=True/1.0000, ac=True/1.0000, am=True/1.0000,
aw=False/0.3333, bg=True/1.0000, bq=False/0.8333, ca=True/1.0000,
ck=False/0.1667, cu=False/0.3333, de=True/1.0000, do=False/0.6667,
dy=True/1.0000, ei=True/1.0000, es=True/1.0000
```

Interpretation:

- Retry feedback raises dev success from `5/15` to `10/15`, so the environment
  and SEARCH interface are not fundamentally dead.
- The remaining failures are useful diagnostics for the next environment/policy
  iteration: exact target-like variants can remain ambiguous, catalog SEARCH may
  return missing/variant ratings or prices, and some compatibility constraints
  require richer product normalization than title-level retrieval.
- This still does not justify any claim about RL improving memory. It is an
  interface and solvability probe before formal training.


## Failure audit of retry diagnostic

The retry5 failure audit is stored at:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/scripted_search_failure_audit_retry5_20260702-055120
AGENTMEMORY_SCRIPTED_SEARCH_FAILURE_AUDIT_OK
failure_type_counts={"compatibility_filter_excluded_target": 5}
```

All five remaining retry5 failures were caused by the strict compatibility
label filter excluding the target candidate from the ranked pool:

```text
aw step2 target ma_aw_c_e excluded by allowed=Niacinamide
bq step5 target ma_bq_f_b excluded by allowed=6 Outlet
ck step1 target ma_ck_b_e excluded by allowed=Compact
cu step2 target ma_cu_c_b excluded by allowed=Raisin
do step4 target ma_do_e_c excluded by allowed=Lemon
```

This means the main blocker is not the hidden target freeze or the shared-disk
SEARCH index. It is the brittle text-level compatibility parser: MemoryArena
rules often name a compatibility label that the correct visible option does not
repeat verbatim in its title/SEARCH result.

## Soft-fallback verifier diagnostic

A third diagnostic run kept the same visible candidate titles and public SEARCH
metadata, but changed the scripted heuristic from strict compatibility filtering
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

Evidence directory:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/scripted_search_baseline_dev_softretry7_20260702-055137
```

Marker / summary:

```text
AGENTMEMORY_SCRIPTED_SEARCH_BASELINE_OK
episodes=15
successes=15
success_rate=1.0000
mean_progress_score=1.0000
total_env_steps=679
search_calls=420
buy_calls=109
rejected_buys=19
max_buy_attempts=7
compatibility_fallback=ranked-all-after-compatible
```

Combined failure audit:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/scripted_search_failure_audit_softretry7_20260702-055940
AGENTMEMORY_SCRIPTED_SEARCH_FAILURE_AUDIT_OK
retry5 failure_type_counts={"compatibility_filter_excluded_target": 5}
softretry7 failure_type_counts={}
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
