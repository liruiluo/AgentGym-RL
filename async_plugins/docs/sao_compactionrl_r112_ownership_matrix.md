# r112 SAO + CompactionRL capability and ownership matrix

Status: implementation contract for
`experiment/camg-sao-compactionrl-r112-20260906`.

Pinned source identities at the start of the migration:

- outer: `26ccd2cddeab471117f8e63d6fb41700398e820d`
- AgentGym submodule: `0988e41e68892559f0fa49dd22227c1434c049ee`
- veRL: `f3ac28fe54c945e092b9630030f44d236a106a11`
- upstream audit reference: `23af6a7a2e8d6efeeb2adbe5d1689c7a24f503a3`

## Frozen scientific boundary

This branch changes optimization only. It must not change an environment
prompt, action grammar, parser, reward, grader, task panel, context-boundary
trigger, action budget, or retry/termination semantics. A correctness repair to
the r112 environment contract must be isolated in the environment-maintenance lineage, independently
reviewed, and then explicitly adopted; it must not enter this branch as an
algorithm convenience.

The matched comparator is r112’s current
`amg_action_axis_gae + token-mean PPO` path. Both arms must use the same base
model, ordered task occurrences, update/episode/action budget, decoder, hardware
topology, and environment source lock. r92 is a historical target, not the
canonical matched baseline.

## Ownership matrix

| Concern | Owner | Exact contract | Algorithm-branch change |
|---|---|---|---|
| Environment transition timing | Four existing wrappers | Each wrapper decides when an action changes state, terminates, or requests context replacement. | None. |
| Policy action transport | `AMGTaskNeutralAgentLoop` | Every sampled assistant response is an ordinary policy action passed to `env.step`; its token ids and rollout log probabilities are preserved. | None. |
| Context replacement | Wrapper receipt + task-neutral AgentLoop plumbing | Observation/context tokens are not optimized; replacement does not terminate the episode or the policy-token credit chain. | None. |
| Shared rollout entrypoint | `amg_task_neutral_async` | Shop, Coding, DeepResearch, and AutoResearch use the same implementation. No environment-named sampler or rollout branch is allowed. | None; `vllm_rollout.py` remains untouched. |
| Immediate reward placement | Native AgentLoop collation | Each action reward is placed once on the final valid token of that action row; the token sum must equal `immediate_reward`. | Validate fail-closed before GAE. |
| Episode token ordering | New task-neutral estimator adapter | Sort by `(trajectory_uid, trajectory_row_order)`, then concatenate every real response token in row/token order. | Replace action-level reduction/broadcast with exact token-chain packing. |
| Policy credit | New `amg_sao_token_gae` estimator | Skip observations while linking the last token of action `i` directly to the first token of action `i+1`. Use one length-adaptive policy lambda per complete episode. | New estimator; no trainer or rollout fork. |
| Critic targets | New estimator over the same token chain | Compute a separate return recurrence with `lambda_critic=1`; never derive critic targets from the policy lambda. | New estimator output only. |
| Advantage normalization | veRL-compatible token mask | Whiten once over all real sampled policy tokens in the learner batch; padding is excluded and remains zero. | Preserve upstream masked-whitening semantics at token granularity. |
| Actor objective | veRL native policy loss | `token-mean` aggregation and bypass-mode REINFORCE use `pi_current / pi_rollout`; tokens outside the registered double-sided interval are masked independently of advantage sign. | Configuration only after a numerical semantics fixture passes. |
| Value objective | veRL native value loss | Token-mean value regression over real response tokens. The Qwen3.5 critic keeps embeddings, norms, MLPs, and the scalar value head trainable while freezing self-attention, linear-attention, and the unused visual tower. Its LM head is weight-tied to the trainable token embedding, so the runtime records that exact alias rather than claiming an impossible independent freeze. | Set two critic epochs for each one actor epoch; no custom value loss. |
| Async queue and publication | veRL fully-async trainer | Queueing, policy-version metadata, online publication, backpressure, stale/drop accounting, and checkpoint recovery remain upstream-owned. | Keep `trigger_parameter_sync_step=1`; no new scheduler. |
| Environment routing | Existing immutable route registry | Route selection chooses a wrapper client, not a rollout implementation. | None. |

## Token-chain contract

For each trajectory, let the valid generated policy tokens after ordering and
concatenation be `y_0, ..., y_(L-1)`, with value predictions `V_t` and rewards
`r_t`. Observation tokens and reset templates are absent from this chain. The
last token of one action is adjacent to the first token of the next action for
credit assignment, including across a context replacement.

The policy trace parameter is fixed by the complete policy-token length:

```text
lambda_policy(L) = 1 - 1 / (1.5 * L)
```

The actor advantage is the reverse token recurrence

```text
delta_t = r_t + gamma * V_(t+1) - V_t
A_t     = delta_t + gamma * lambda_policy(L) * A_(t+1)
```

with `V_L = A_L = 0`. Critic returns are computed independently with
`lambda_critic=1`:

```text
A^V_t = delta_t + gamma * A^V_(t+1)
R^V_t = A^V_t + V_t
```

For `gamma=1`, this makes `R^V_t` the undiscounted cumulative future reward,
while the policy retains the length-adaptive bias/variance trade-off. Rewards
are never copied to earlier segments or repeated at context boundaries.

## Current active-source evidence

Launcher:
`async_plugins/scripts/launch_amg_multitask_fully_async.sh` at outer commit
`26ccd2cddeab471117f8e63d6fb41700398e820d` requires an explicit `--verl-root`
and constructs `PYTHONPATH` from that root before invoking the task-neutral
orchestrator.

A static AST audit of the exact veRL worktree
`7f359d928fa438c9353c0c6d98941c4638971f05` and outer worktree passed for the
fully-async main/rollouter/trainer/queue, upstream AgentLoop, AMG AgentLoop,
dataset, and current advantage estimator. The route registry declares all four
routes while the AgentLoop YAML contains exactly one implementation:
`agentmemorygym_verl.agent_loop.AMGTaskNeutralAgentLoop`.

The active-source test binds the accepted veRL root to the exact commit named
by this branch.  The final launch must update that binding to the newly pushed
veRL algorithm commit and re-run the audit against the immutable launch roots.

## Required evidence before launch

1. Numeric fixtures cover single-action token GAE, cross-action observation
   gaps, context replacement, reward conservation, terminal/padding behavior,
   mixed rollout policy versions, length-adaptive policy lambda, independent
   critic lambda, and double-sided importance-ratio masking.
2. Resolved config proves the new estimator, token-mean actor/critic losses,
   bypass-mode token DIS, actor epoch 1, critic epoch 2, and online publication
   every optimizer update.
3. Exact-current active-source audit passes and at least two environment
   fixtures traverse `amg_task_neutral_async`.
4. No diff exists in environment wrappers, prompt/reward/grader/task inputs,
   or a shared/domain-specific rollout implementation.
5. A GPU run may use only a naturally available one of the three existing
   eight-card allocations. No fourth allocation is requested.

## Runtime and holder safety successor contract

The post-r112 runtime-safety successor changes orchestration only.  It does not
authorize deployment or a formal run, and it does not modify the environment,
rollout, optimizer, reward, grader, schedule, or frozen r112 package.

- The platform fallback wrapper bootstraps with the hash-pinned
  `/opt/conda/envs/py312/bin/python3` runtime.  It deliberately ignores the
  training launcher's volatile `PYTHON` value under `/dev/shm`; the training
  process remains bound to its separately frozen runtime.
- A persisted `pending_resume` binds both the exact pause-marker inode/bytes and
  the optional release-request inode/bytes.  A replacement supervisor finishes
  that authorization idempotently.  It may publish `holding` only after marker,
  request, and pending transaction state have all been closed and the holder
  resource attestation has passed.
- The marker transaction's append-only `drain_identities` payload is the
  authority for PID/start-ticks/process-group liveness.  Identity bytes,
  metadata, and digest are read through one descriptor.  A current directory
  scan can discover additional identities, but unlinking or replacing a
  registered path cannot erase the saved lease from the drain decision.
- CPU regression evidence used for a launch review must be produced by
  `scripts/run_exact_source_tests.py`.  The harness binds every named checkout
  to an expected clean HEAD and selected-file manifest, records the real
  interpreter and digest plus the exact command and environment allowlist, and
  repeats HEAD/status/file hashing after the command.  A dirty tree, byte drift,
  nonzero command, or post-run mutation publishes a failed receipt.

These source changes still require a newly frozen package, a fresh independent
runtime review, and an actual same-boundary Pod recovery receipt before any
r112/r113 formal launch can be authorized.

The existing frozen r112 deploy runner is not compatible with this successor:
it uses one `/dev/shm` `PY` value both for `migrate` / `formal --python` and for
the training command.  A successor package must use two explicitly named,
independently hashed values.  The stable `/opt` interpreter must execute
`fallback_supervisor.py` and be passed through `--python` so its immutable
contract exactly matches the platform-restarted `supervise` process.  The
source-locked `/dev/shm` interpreter remains the first executable inside the
formal command after `--`.  Package validation must reject a runner that binds
the fallback contract back to the training interpreter, even when both current
binaries happen to have the same digest.  This runner split, package freeze,
and target-Pod restart test are launch-review gates rather than edits to the
immutable r112 package.
