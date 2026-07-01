# 2026-07-01 AgentMemoryGym dataset metadata + rollout guard smoke

## Scope

This note validates the next code layer after the initial environment skeleton:

- split-aware dataset loading through `load_task_dataset(split=...)` and env vars
  `AGENTMEMORY_DATA_PATH`, `AGENTMEMORY_SPLIT`, `AGENTMEMORY_SPLIT_DIR`;
- server `/metadata` reporting `task_count`, `task_ids`, `splits`, and `source`;
- `AgentMemoryEnvClient(data_len=None)` reading server metadata instead of relying
  on `data_len=1`;
- AgentGym-RL vLLM rollout fail-fast guard that blocks AgentMemory formal rollout
  when it would otherwise expose full raw conversation history to the policy.

This is 0-card local validation on Mac/ZBMac. It is still not a single-GPU
smoke, not a full MemoryArena conversion, and not RL-improvement evidence.

## Commands run

```bash
python3 -m compileall -q \
  AgentGym/agentenv-agentmemory \
  AgentGym/agentenv/agentenv/envs/agentmemory.py \
  AgentGym/agentenv/agentenv/envs/__init__.py \
  AgentGym-RL/verl/utils/agentgym/client.py \
  AgentGym-RL/verl/utils/agentgym/context_policy.py \
  AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py \
  docs/agentmemorygym/scripts/smoke_context_policy.py

PYTHONPATH=AgentGym/agentenv-agentmemory \
  python3 AgentGym/agentenv-agentmemory/scripts/validate_agentmemory_data.py

PYTHONPATH=AgentGym/agentenv-agentmemory \
  python3 AgentGym/agentenv-agentmemory/scripts/smoke_agentmemory.py

python3 docs/agentmemorygym/scripts/smoke_context_policy.py

PYTHONPATH=AgentGym/agentenv-agentmemory python3 - <<'PY'
from agentenv_agentmemory.environment import load_task_dataset
for split in ['train', 'dev', 'test']:
    tasks = load_task_dataset(split=split)
    assert len(tasks) == 1, (split, tasks)
    assert tasks[0].split == split, (split, tasks[0])
print('AGENTMEMORY_SPLIT_LOADER_SMOKE_OK')
PY
```

Output:

```text
AGENTMEMORY_DATA_VALIDATE_OK tasks=3 splits=train:1,dev:1,test:1
AGENTMEMORY_DIRECT_SMOKE_OK tv_bundle_75 laptop_bundle_14 monitor_bundle_27
AGENTMEMORY_CONTEXT_POLICY_SMOKE_OK
AGENTMEMORY_SPLIT_LOADER_SMOKE_OK
```

## Server/client metadata smoke

Server metadata:

```text
SERVER_METADATA_SMOKE_OK 3 tv_bundle_75,laptop_bundle_14,monitor_bundle_27
```

AgentMemory client metadata smoke uses a small local stub for `agentenv.controller`
so this 0-card Mac can validate the metadata path without installing `torch`.
This does not replace a real AgentGym/verl single-GPU import/rollout check:

```text
AGENTMEMORY_CLIENT_METADATA_SMOKE_OK 3 monitor_bundle_27
```

## Rollout guard boundary

`AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py` now calls
`assert_rollout_context_supported(self.agentgym_config)` before creating env
clients. For `task_name=agentmemory`, the guard raises unless either:

- `agentgym.allow_raw_history_for_agentmemory=true`, or
- `AGENTMEMORY_ALLOW_RAW_HISTORY=1`.

Those overrides are diagnostic-only. Formal AgentMemoryGym training still needs
a future latest-observation / ephemeral-context rollout implementation so the
policy cannot bypass memory tools by reading raw prior observations or actions.
