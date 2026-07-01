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
