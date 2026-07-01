# 2026-07-01 AgentMemoryGym skeleton smoke

## Scope

Validated the current `agentenv-agentmemory` skeleton only. This is not a full MemoryArena conversion and not RL-improvement evidence.

## Current code shape

- Environment package: `AgentGym/agentenv-agentmemory/`
- Default smoke data: `AgentGym/agentenv-agentmemory/agentenv_agentmemory/data/bundled_shopping_smoke.jsonl`
- Smoke split files: `AgentGym/agentenv-agentmemory/agentenv_agentmemory/data/splits/{train,dev,test}.txt`
- AgentGym adapter/client: `AgentGym/agentenv/agentenv/envs/agentmemory.py`
- verl registry: `AgentGym-RL/verl/utils/agentgym/client.py`, task name `agentmemory`
- Product IDs in smoke tasks are neutral (`tv_b`, `mount_b`, `console_b`, etc.) rather than leaking size labels such as `mount_large_75`.

## Commands run

```bash
python3 -m compileall -q \
  AgentGym/agentenv-agentmemory \
  AgentGym/agentenv/agentenv/envs/agentmemory.py \
  AgentGym/agentenv/agentenv/envs/__init__.py \
  AgentGym-RL/verl/utils/agentgym/client.py

PYTHONPATH=AgentGym/agentenv-agentmemory \
  python3 AgentGym/agentenv-agentmemory/scripts/validate_agentmemory_data.py

PYTHONPATH=AgentGym/agentenv-agentmemory \
  python3 AgentGym/agentenv-agentmemory/scripts/smoke_agentmemory.py
```

Output:

```text
AGENTMEMORY_DATA_VALIDATE_OK tasks=3 splits=train:1,dev:1,test:1
AGENTMEMORY_DIRECT_SMOKE_OK tv_bundle_75 laptop_bundle_14 monitor_bundle_27
```

Server-client smoke used the temporary local venv at `/tmp/agentmemorygym-smoke-venv` with FastAPI/Uvicorn/Requests and the updated TV plan:

```text
CREATE 0 False
STEP BUY 1.0 False 0.3333333333333333
STEP ADD -0.01 False 0.3333333333333333
STEP RETRIEVE -0.01 False 0.3333333333333333
STEP BUY 1.0 False 0.6666666666666666
STEP BUY 2.0 True 1.0
SERVER_CLIENT_SMOKE_OK
```

After the final neutral-ID/prompt cleanup, the server-client path was rerun and
returned:

```text
SERVER_CLIENT_SMOKE_OK_FINAL
```

After moving default smoke tasks behind the JSONL loader, the server-client path
was rerun and returned:

```text
SERVER_CLIENT_JSONL_SMOKE_OK
```

After adding smoke split files and the data validator, the server-client path
was rerun on the `test` split item (`monitor_bundle_27`) and returned:

```text
SERVER_CLIENT_SPLIT_SMOKE_OK monitor_bundle_27 test
```

## Known local limitation

Full AgentGym adapter import was not run successfully on this Mac because the local smoke venv lacks `torch`:

```text
AGENTGYM_ADAPTER_IMPORT_FAIL ModuleNotFoundError No module named 'torch'
```

This is an environment dependency limitation, not yet evidence of an adapter code failure. Full AgentGym/verl import and rollout should be tested on the approved single-card environment with AgentGym dependencies installed.
