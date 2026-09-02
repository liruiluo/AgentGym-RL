# CompactionRL matched baseline: ownership matrix

Date: 2026-09-01

Exact implementation base:

- outer AgentGym-RL: `bc4d6a9ac18d78e18e4d2c1d90d77346c100c09c`
- inner AgentGym: `b2c149eadff0e7657d567bd7070cebb0268ab394`
- inner candidate: `311ec3062bddfbddb88f71f35e159ddf1e8a7c61`
- veRL: `f3ac28fe54c945e092b9630030f44d236a106a11`

This branch implements the core CompactionRL mechanism as a matched CAMG
baseline.  It does not claim a method-faithful reproduction of the paper's
optimizer or training stack.

## Ownership

| Concern | Owner | Contract |
|---|---|---|
| When compaction is due | Environment wrapper through the shared `PolicyContextPressure` contract | Trigger before the next ordinary action/observation could exhaust the fixed prompt capacity. |
| Summary request and validation | Generic inner `context_compaction.py` helper selected by each wrapper | The same policy emits one non-empty bounded plain-text summary.  It is a normal sampled policy row and is never sent to the native environment. |
| Context reconstruction | Generic inner helper, invoked by the wrapper | Rebuild from immutable policy framing, a fixed resume message containing the summary, and the latest two complete action/observation pairs; reduce the retained tail only if the exact rendered prompt would exceed capacity. |
| Native task transition | Each environment wrapper | Ordinary policy actions alone are sent to the native endpoint.  Native reward, termination, and task semantics are unchanged. |
| Context transition receipt | Each environment wrapper using the existing task-neutral receipt | Return `replace_messages`; no domain name, parser, session counter, or task-specific compaction logic enters shared rollout. |
| PPO row, sampled tokens, behavior log probabilities, action-axis GAE, and token-level PPO | Existing `AMGTaskNeutralAgentLoop` and veRL learner | Summary rows follow the same sampled-response path as task-action rows.  There is no summary reward and no advantage surgery. |
| Voluntary file-memory surface | Baseline policy framing | `.agent_memory/**` is not used as a voluntary memory mechanism.  Normal task filesystem work remains available. |
| Experiment selection | Route-registry client configuration | All four routes set `context_memory_mode=compactionrl`, `compaction_recent_steps=2`, and one shared summary bound. |

## Shared-loop decision

The exact active loop already performs:

`prepare_policy_turn -> sample policy output -> env.step(policy_output) -> record sampled tokens/logprobs -> mechanically apply context_transition`.

Therefore the CompactionRL lifecycle difference is wrapper-owned and **does
not require a separate rollout**.  The expected diff for
`async_plugins/agentmemorygym_verl/agent_loop.py` is zero.  A change to that
file is forbidden unless a minimal failing fixture proves a task-neutral
receipt gap.

## Required prelaunch evidence

1. Unit tests for trigger accounting, non-empty/bounded summaries, retry, exact
   context reconstruction, and adaptive two-pair tail retention.
2. Integration tests showing at least two environment wrappers use the same
   helper and the same outer AgentLoop entrypoint.
3. Active-source audit on the exact launch commit: no environment name,
   domain parser, native server call, or dedicated `generate_*` path in the
   shared AgentLoop.
4. Four-route attestation proving the route registry selects
   `context_memory_mode=compactionrl` and hashes the effective policy framing.
5. A real four-environment update-1 gate before continuing the declared
   formal run.

## Matched experiment contract

- fresh Qwen3.5-4B base
- formal endpoint at 200 learner updates, 64 consumed episodes/update, 12,800
  total episodes
- four routes with 3,200 consumed episodes each under the frozen round-robin
  schedule; the immutable launcher retains a longer 400-update schedule only
  as launch provenance, and unconsumed rows are not an endpoint requirement
- six learner/hybrid GPUs plus two standalone rollout GPUs
- `rollout.n=1`, same decoder, native rewards, graders, action budgets, and
  fully asynchronous PPO/GAE path as the current joint run
- shared-storage checkpoints every 10 updates, including a verified step-200
  terminal checkpoint
- no native held-out evaluation before the declared training endpoint
