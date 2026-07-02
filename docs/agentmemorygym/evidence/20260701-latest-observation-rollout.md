# 2026-07-01 AgentMemoryGym latest-observation rollout context path

## Scope

This note records the rollout-context code layer after the first true single-GPU
env/client smoke. The goal is to prevent formal AgentMemoryGym training from
bypassing memory tools by reading raw prior observations/actions in the rollout
history.

Implemented behavior:

- `agentmemory` defaults to `rollout_context_policy=latest_observation_only`.
- Explicit `allow_raw_history_for_agentmemory=true` still selects raw-history
  mode for diagnostic smoke only.
- In latest-observation mode, each generation prompt contains only the current
  environment observation and the assistant generation prefix. The environment
  observation may include current-session STM trace; it must not include raw
  observations/actions from previous sessions.
- Multi-round AgentMemory episodes are flattened into one PPO training sample per
  assistant action. Each sample's `input_ids` contains the latest-observation
  prompt plus that action only, so actor/ref log-prob recomputation does not see
  raw previous-session observations or actions.
- The rollout output carries `rollout_parent_indices`; the PPO trainer aligns the
  original batch to flattened action samples through those parent indices instead
  of blindly using `repeat(n)`.

This is a rollout data-path / leakage-boundary implementation. It is not meant
to delete same-session working context: current-session STM is part of the
environment observation. It is still not a full model rollout smoke or RL
training result.

## Local verification

```text
AGENTMEMORY_CONTEXT_POLICY_SMOKE_OK
AGENTMEMORY_LATEST_OBSERVATION_PROMPT_SMOKE_OK
```

Commands:

```bash
python3 -m compileall -q \
  AgentGym-RL/verl/utils/agentgym/context_policy.py \
  AgentGym-RL/verl/utils/agentgym/rollout_context.py \
  AgentGym-RL/verl/workers/rollout/schemas.py \
  AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py \
  AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py \
  docs/agentmemorygym/scripts/smoke_context_policy.py \
  docs/agentmemorygym/scripts/smoke_latest_observation_prompt.py \
  docs/agentmemorygym/scripts/smoke_rollout_context_alignment.py
python3 docs/agentmemorygym/scripts/smoke_context_policy.py
python3 docs/agentmemorygym/scripts/smoke_latest_observation_prompt.py
```

## Torch-backed single-GPU environment verification

Run on Jingyan 1×B200 (`luolirui-1-jy1-0630030405-1`) under
`/opt/conda/envs/rl/bin/python3` after syncing the modified minimal source tree
into:

```text
/home/ai-jingyan-train/luolirui.1/post-train/code/AgentGym-RL-agentmemory-smoke
```

Markers:

```text
AGENTMEMORY_CONTEXT_POLICY_SMOKE_OK
AGENTMEMORY_LATEST_OBSERVATION_PROMPT_SMOKE_OK
AGENTMEMORY_ROLLOUT_CONTEXT_ALIGNMENT_SMOKE_OK
```

The remote `AGENTMEMORY_ROLLOUT_CONTEXT_ALIGNMENT_SMOKE_OK` covers parent-index
alignment, including the `n>1` case and duplicated parent `uid` values needed for
GRPO/RLOO grouping.

## Remaining gap

Next step is a real small-model or API rollout smoke using this latest-observation
path. Do not use the raw-history override as formal evidence.
