# Shared Rollout Ownership Matrix

Status: the unified filesystem-checkpoint candidate is statically and
fixture-verified; the four-environment real-model activation gate is pending.

All four environments use the same task-neutral transition shape.  A boundary
request does not create a second sampler or a free summary row.  The next sampled
output is one ordinary policy action, is sent through the existing `env.step`, and
keeps its exact tokens/logprobs and reward in PPO.  The wrapper may clear old
messages only after the native environment attests that this action changed the
exact fixed path `.agent_memory/CONTINUATION.md` into a non-empty regular file of
at most 8,192 bytes.  The successor contains immutable task framing plus the
fixed path, size, SHA256, and a read-next instruction; it contains neither the
checkpoint body nor the sampled write action/native observation.  A failed write
keeps the old context and returns a short bounded retry observation.

| Surface | Wrapper-owned transition | Ordinary policy payload | Task-neutral receipt consumed by the runner |
| --- | --- | --- | --- |
| WebShop filesystem | A successful non-terminal BUY creates a session boundary.  On the next policy turn, the wrapper requests the fixed checkpoint write, sends it to the native WebShop/filesystem server, verifies the write receipt, then replaces the prior-session message history. | The exact sampled `shell_command` or `apply_patch` is passed once to `BaseEnvClient.step(policy_output)` and consumes one policy/native action. | `StepOutput.info` carries counters, `action_submission`, the normalized filesystem-checkpoint receipt, and optional `replace_messages`; the runner does not inspect WebShop/session/file semantics. |
| SWE-smith | The wrapper measures token pressure and requests the same fixed checkpoint write.  The native coding environment executes it before any replacement. | The exact sampled action is passed once to `BaseEnvClient.step(policy_output)` and consumes one policy/native action. | The same receipt fields; replacement is emitted only for a verified write, otherwise context is preserved for retry. |
| LiteResearcher | The wrapper measures token pressure with the route's larger bounded Visit observation and requests the same fixed checkpoint write.  The LiteResearcher workspace service executes and attests it. | The exact sampled action is passed once to `BaseEnvClient.step(policy_output)` and consumes one policy/native action. | The same receipt fields; the runner does not inspect Search, Visit, answer, workspace, or checkpoint semantics. |
| OpenMLE-fast | The wrapper measures token pressure and requests the same fixed checkpoint write.  The native OpenMLE sandbox executes it under the existing 30-action budget before any replacement. | The exact sampled action is passed once to `BaseEnvClient.step(policy_output)` and once to the native OpenMLE `/step`; parser failure, rejection, and completion remain ordinary charged outcomes. | The same receipt fields; the runner applies only the declared context operation and does not inspect OpenMLE actions, paths, task IDs, or grading policy. |

The shared entrypoint must perform only:

1. reset and observe;
2. bind the current message list to the wrapper;
3. measure prompt pressure and ask the wrapper whether a control request is
   needed;
4. sample one policy output;
5. call `env.step` once with that output;
6. mechanically apply the receipt's context transition;
7. pack the exact prompt/response tokens, reward, and opaque receipt into PPO.

The active-source audit for this candidate keeps
`async_plugins/agentmemorygym_verl/agent_loop.py` and
`AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py` unchanged
and environment-agnostic.  CPU/remote-py312 fixtures cover all four wrappers and
shared AgentLoop packing.  A GPU formal may start only after the exact deployed
commits pass a real-model four-route `write -> replace -> read` activation gate.
