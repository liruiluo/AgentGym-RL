# 2026-07-01 AgentMemoryGym latest-observation scripted-policy rollout smoke

## Scope

Jingyan 1×B200 currently has torch/transformers/ray but lacks full AgentGym-RL
vLLM training dependencies (`vllm`, `tensordict`, `codetiming`). Installing them
inside the released Jingyan lane would be too invasive for a quick smoke.

This smoke therefore validates the environment contract with a deterministic
scripted policy rather than a language model:

- the policy receives only the current observation each step;
- it maintains its own external memory store;
- it explicitly emits `ADD` actions before the context-discontinuity purchase and `RETRIEVE` before dependent accessory purchases;
- all three bundled-shopping smoke tasks finish successfully;
- a no-memory wrong-purchase baseline is rejected with compatibility violations.

This is not an LLM/vLLM rollout, not a model-quality result, and not formal RL
training evidence.

## Commands

Local:

```bash
PYTHONPATH=AgentGym/agentenv-agentmemory \
  python3 AgentGym/agentenv-agentmemory/scripts/smoke_latest_observation_policy.py
```

Jingyan 1×B200:

```bash
PYTHONPATH=/home/ai-jingyan-train/luolirui.1/post-train/code/AgentGym-RL-agentmemory-smoke/AgentGym/agentenv-agentmemory \
  /opt/conda/envs/rl/bin/python3 \
  /home/ai-jingyan-train/luolirui.1/post-train/code/AgentGym-RL-agentmemory-smoke/AgentGym/agentenv-agentmemory/scripts/smoke_latest_observation_policy.py
```

## Markers

```text
AGENTMEMORY_LATEST_OBSERVATION_POLICY_SMOKE_OK tv_bundle_75 laptop_bundle_14 monitor_bundle_27
```

Remote log:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-smoke-evidence/latest_observation_policy_smoke_20260701-210634.log
```

## Dependency audit

Jingyan 1×B200 audit at the time of smoke:

```text
torch True
transformers True
tensordict False
vllm False
ray True
omegaconf True
codetiming False
pandas True
requests True
fastapi True
uvicorn True
torch 2.11.0+cu130 cuda True NVIDIA B200
model dir: /home/ai-jingyan-train/luolirui.1/post-train/models/Qwen3-4B
```

Next step for real LLM rollout: prepare a non-invasive AgentGym-RL runtime with
`vllm`, `tensordict`, and `codetiming`, or run the LLM rollout in a lane where
those dependencies already exist. Do not count this scripted-policy smoke as the
full LLM rollout gap closure.
