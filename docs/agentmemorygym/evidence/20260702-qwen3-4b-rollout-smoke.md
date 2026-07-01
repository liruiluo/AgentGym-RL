# 2026-07-02 Qwen3-4B latest-observation rollout smoke

## Scope

This evidence records a **true single-GPU model rollout smoke** for
AgentMemoryGym on the released Jingyan 1×B200 lane. It is not an RL training
run, not a vLLM/verl throughput run, and not evidence that memory ability has
improved.

The smoke uses a local Transformers load of Qwen3-4B and prompts the model with
only the latest environment observation. No raw-history override and no scripted
target policy were used.

## Hardware / runtime

```text
Host/container: luolirui-1-jy1-0630030405-1-master-0
GPU: NVIDIA B200
Python: /opt/conda/envs/rl/bin/python3
torch: 2.11.0+cu130
transformers: 5.6.0
model: /home/ai-jingyan-train/luolirui.1/post-train/models/Qwen3-4B
```

The temporary GPU warmup guard yielded/was paused during the model run; the
Jingyan GPU guard was active again after the smoke.

## Frozen MemoryArena dev run

Data:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_formal_freeze_20260701-234045/memoryarena_agentmemory.jsonl
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_formal_freeze_20260701-234045/splits
```

Evidence directory:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/qwen3_4b_latest_observation_progress_rollout_20260702-001520
```

Marker / summary:

```text
AGENTMEMORY_QWEN3_4B_LATEST_OBSERVATION_PROGRESS_ROLLOUT_SMOKE_OK env_steps 20 parse_successes 20 any_episode_success False
episodes=2
task_ids=memoryarena_bundled_shopping_i,memoryarena_bundled_shopping_s
progress_score=0.0,0.0
```

Interpretation:

- Qwen3-4B loaded on the real B200 and generated valid AgentMemoryGym tool
  actions that were parsed and executed by the environment.
- The model did **not** complete any frozen MemoryArena task and made no
  progress on the two dev items. It repeatedly chose `RETRIEVE {"query":
  "highest rated", ...}` while long-term memory was empty.
- This exposes a real next-code gap: the current converted observation only
  shows option titles and source-option ids. It does not expose the product DB
  fields needed by instructions such as highest-rated / highest-priced /
  budget, nor does the environment provide a `SEARCH` tool over the product DB.

## Handcrafted bundled-shopping sanity run

To confirm that the same model path can produce `BUY` actions and receive
environment rewards on a solvable visible-attribute task, a bounded run was also
executed on the repo smoke data.

Evidence directory:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/qwen3_4b_handcrafted_smoke_rollout_20260702-001623
```

Marker / summary:

```text
AGENTMEMORY_QWEN3_4B_LATEST_OBSERVATION_PROGRESS_ROLLOUT_SMOKE_OK env_steps 12 parse_successes 12 any_episode_success False
task_id=tv_bundle_75
progress_score=0.3333333333333333
```

The model bought `tv_b` correctly on step 0, then repeatedly attempted the
incompatible `mount_a`. This sanity run proves the real model→action parser→env
reward path on GPU, but it also shows that untrained Qwen3-4B does not yet use
the memory/feedback loop well enough to finish the task.

## Next implementation gap

Before claiming a meaningful AgentMemoryGym training/evaluation result, the
shopping environment needs one of these interfaces:

1. expose comparable product DB metadata for **all visible candidates** in the
   observation; or
2. add a product-catalog `SEARCH` tool and train/evaluate policies that retrieve
   rating/price/review information before buying.

The frozen item-id split and target resolver are usable, but the current
converted observation is not yet a fair train/eval surface for
rating/price-driven MemoryArena shopping tasks.

## Enriched metadata freeze + Qwen3-4B rerun

Strict metadata-enriched freeze data:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/memoryarena_enriched_freeze_20260702-014308
```

Freeze summary:

```text
AGENTMEMORY_MEMORYARENA_FORMAL_FREEZE_OK
tasks=150 / rows=900 / train:120,dev:15,test:15
resolver_counts={"asin_catalog": 900}
ambiguous=0
candidate_metadata_status_counts={"full":285,"partial":605,"none":10}
candidate_metadata_full_steps=285/900
```

Two Qwen3-4B reruns were made on the enriched dev split:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/qwen3_4b_enriched_metadata_rollout_20260702-023225
AGENTMEMORY_QWEN3_4B_LATEST_OBSERVATION_PROGRESS_ROLLOUT_SMOKE_OK
env_steps=20 / parse_successes=20 / any_episode_success=False / progress_score=0.0,0.0
```

With the old generic prompt, Qwen3-4B still looped on
`RETRIEVE {"query":"highest rated"}` despite the first dev observation exposing
`average_rating / price_usd / total_reviews`. This means metadata exposure alone
does not solve the policy behavior.

A second diagnostic prompt explicitly told the model to compare visible
`average_rating / price_usd / total_reviews` and that `RETRIEVE` cannot fetch
product-catalog metadata:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/qwen3_4b_metadata_prompt_rollout_20260702-023436
AGENTMEMORY_QWEN3_4B_LATEST_OBSERVATION_PROGRESS_ROLLOUT_SMOKE_OK
env_steps=20 / parse_successes=20 / any_episode_success=False
progress_score=0.16666666666666666,0.0
```

This second run bought the correct first item on dev episode 0 (`ma_i_a_b`) and
then failed on later subtasks. It is useful plumbing/interface evidence: the
metadata-enriched observation can induce at least one real `BUY`/reward/progress
step on frozen MemoryArena dev. It is still not an RL result and not evidence of
improved memory ability.

Current closure bar before formal training/eval: strict candidate metadata only
covers 285/900 steps, so the environment still needs either better all-candidate
metadata matching or an explicit product-catalog `SEARCH` tool.

## Product-catalog SEARCH tool smoke

A product-catalog `SEARCH` tool was added after the strict candidate-metadata
route plateaued at `285/900` fully enriched steps even with all 67 product
catalog shards. The full SQLite/FTS index was built on the Jingyan shared disk,
not on the Mac/devbox local disk:

```text
/home/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/agentmemory_catalog_search.sqlite
AGENTMEMORY_CATALOG_SEARCH_INDEX_OK products=1031654
index size ~= 479M
```

A remote env smoke on the formal freeze confirmed that `SEARCH` can be called
through the environment with `AGENTMEMORY_CATALOG_INDEX_PATH` configured:

```text
RESET_OK memoryarena_bundled_shopping_i True
SEARCH_OK reward -0.01 done False
```

Example result for a visible candidate title:

```text
SEARCH query: A gluten-free carrot cake mix...
- Gluten-Free Carrot Cake Mix (average_rating=4.5, price_usd=14.99, total_reviews=59, match_score=111)
```

## SEARCH-aware Qwen3-4B prompt smoke

Evidence directory:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/qwen3_4b_search_prompt_rollout_20260702-043335
```

Marker / summary:

```text
AGENTMEMORY_QWEN3_4B_LATEST_OBSERVATION_PROGRESS_ROLLOUT_SMOKE_OK
env_steps=24 / parse_successes=24 / any_episode_success=False
progress_score=0.0,0.0
```

Transcript inspection shows both dev episodes executed valid environment steps,
but every parsed action was the placeholder query:

```text
SEARCH {"query":"visible candidate title","top_k":3}
```

The resulting catalog hit was unrelated (`LIPSTICK QUEEN Visible Lip Liner...`),
and the model never transitioned to `ADD` or `BUY`. This is still useful
plumbing evidence: the environment/action parser/model loop can execute SEARCH
on the real shared-disk index. It is not a task-success result. The next useful
engineering step is a scripted SEARCH baseline / heuristic memory manager that
queries actual visible candidate titles, stores the retrieved metadata, and
then buys, so we can separate environment solvability from untrained Qwen prompt
behavior before RL.
