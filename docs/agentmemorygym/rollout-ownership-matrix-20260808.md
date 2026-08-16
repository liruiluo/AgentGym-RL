# Shared Rollout Ownership Matrix

Status: shared source migration is statically and fixture-verified; the real
WebShop/SWE-smith environment-server gate is still pending.

| Surface | Wrapper-owned transition | Ordinary policy payload | Task-neutral receipt consumed by the runner |
| --- | --- | --- | --- |
| WebShop filesystem | A successful non-terminal BUY sets `pending_session_handoff`; the next policy turn is a wrapper-local locator handoff. The native server is not called for that turn. | The exact sampled text is passed to `BaseEnvClient.step(policy_output)`. | `StepOutput.info`: `schema`, `context_transition`, counters, `action_submission`, and opaque `wrapper_evidence`; the runner only applies the declared context operation and preserves the raw row. |
| SWE-smith | The wrapper measures token pressure and selects a policy-authored context compaction turn. Compaction does not call the native server. | The exact sampled text is passed to `BaseEnvClient.step(policy_output)`. | The same receipt fields; the runner does not interpret `context_compaction` or coding semantics. |
| OpenMLE-fast | The wrapper measures token pressure, selects the next ordinary OpenMLE action as the compaction turn, sends it to the native server so it consumes the same 30-action ledger, then replaces old history with immutable task framing plus that exact sampled action and its bounded native observation. A successful optional write to `.agent_memory/OPENMLE_CONTINUATION.md` is recorded but is not a precondition for safe replacement. | The exact sampled text is passed once to `BaseEnvClient.step(policy_output)` and once to the native OpenMLE `/step`; parser failure, rejection, and completion all remain ordinary charged action outcomes. | The same task-neutral receipt fields. The runner applies only `replace_messages`; it does not inspect OpenMLE action syntax, workspace paths, parser status, or persistence evidence. |

The shared entrypoint must perform only:

1. reset and observe;
2. bind the current message list to the wrapper;
3. measure prompt pressure and ask the wrapper whether a control request is
   needed;
4. sample one policy output;
5. call `env.step` once with that output;
6. mechanically apply the receipt's context transition;
7. pack the exact prompt/response tokens, reward, and opaque receipt into PPO.

The active-source audit is clean on this candidate: `vllm_rollout.py` exposes
one `generate_task_neutral_policy` implementation and does not dispatch on a
domain or call a native server for a control turn.  The remote py312 fixture
has exercised WebShop handoff and SWE-smith compaction together, with exact
sampled/packed token and reward readback.  No GPU gate or formal run has been
started from this candidate yet.
