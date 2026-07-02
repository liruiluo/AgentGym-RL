# 2026-07-02 AgentMemoryGym session STM boundary

## Scope

This note records the code/docs correction that `latest-observation` should not
mean "no short-term memory." Formal AgentMemoryGym rollout should expose the
current environment observation, and that observation may include the current
session's automatic STM trace. What remains blocked is cross-session raw
conversation/action history.

Implemented environment behavior:

- `AgentMemoryEnv` records current-session `session_trace` entries for
  non-boundary actions such as `ADD`, `RETRIEVE`, `SEARCH`, invalid actions, and
  rejected `BUY` attempts.
- `session_trace` is rendered as `Current session short-term history`.
- `short_term_context` is rendered as `Active retrieved/summary context` to avoid
  confusing LTM retrieval output with the whole STM.
- Successful `BUY` that advances to the next shopping session clears both
  `session_trace` and active retrieved/summary context.
- `info["session_trace"]` is emitted for debugging and behavior analysis.

Additional tool-contract cleanup after the AgeMem boundary correction:

- Visible context items now have observation-local IDs: `S0`, `S1`, ... for
  current-session STM trace entries, and `C0`, `C1`, ... for active
  retrieved/summary context entries.
- `SUMMARY` now supports deterministic context summarization from selected
  visible context (`span=session|active|all`) as well as explicit policy-provided
  summary text. The clean RL path is `SUMMARY {"text": "...", "source_ids":
  [...]}`: the current policy model writes the summary tokens; the environment
  only validates optional visible source IDs and applies the state change. It
  does not read hidden state or call an external judge/model.
- `FILTER` now supports `scope=active|session|all`, so a policy can filter the
  active retrieved/summary context and/or current-session trace. The clean RL
  path is model-authored `keep_ids` / `drop_ids`; query-based filtering remains
  only as a deterministic scaffold/baseline.
- `SEARCH` is explicitly separated from memory tools: it is recorded in
  `info["tool_ops"]` as a catalog tool, while `info["memory_ops"]` remains
  restricted to `ADD/UPDATE/DELETE/RETRIEVE/SUMMARY/FILTER`.
- `ADD/UPDATE/DELETE/RETRIEVE/SUMMARY/FILTER` are covered by direct smoke checks
  for state diffs, active context changes, deletion, and post-delete retrieval.

## Local verification

Run from `code/AgentGym-RL`:

```bash
python3 -m compileall -q AgentGym/agentenv-agentmemory
PYTHONPATH=AgentGym/agentenv-agentmemory python3 AgentGym/agentenv-agentmemory/scripts/smoke_agentmemory.py
PYTHONPATH=AgentGym/agentenv-agentmemory python3 AgentGym/agentenv-agentmemory/scripts/smoke_latest_observation_policy.py
PYTHONPATH=AgentGym/agentenv-agentmemory python3 AgentGym/agentenv-agentmemory/scripts/smoke_memoryarena_converter.py
python3 docs/agentmemorygym/scripts/smoke_context_policy.py
python3 docs/agentmemorygym/scripts/smoke_latest_observation_prompt.py
```

Markers:

```text
AGENTMEMORY_DIRECT_SMOKE_OK tv_bundle_75 laptop_bundle_14 monitor_bundle_27
AGENTMEMORY_LATEST_OBSERVATION_POLICY_SMOKE_OK tv_bundle_75 laptop_bundle_14 monitor_bundle_27
AGENTMEMORY_MEMORYARENA_CONVERTER_SMOKE_OK
AGENTMEMORY_CONTEXT_POLICY_SMOKE_OK
AGENTMEMORY_LATEST_OBSERVATION_PROMPT_SMOKE_OK
```

These are still environment / policy-contract smoke tests. They do not claim RL
training or memory ability improvement.
