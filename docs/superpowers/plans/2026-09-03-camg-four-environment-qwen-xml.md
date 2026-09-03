# CAMG Four-Environment Strict Qwen XML Formal200 Plan

**Date:** 2026-09-03

**Owner:** the foreground replacement csub bound in
`20260903-amg-four-env-xml-formal200.json`

**Objective:** replace the superseded r90c lineage with one fresh Qwen3.5-base,
latest-veRL, fully-asynchronous PPO formal200 whose WebShop, SWE-smith,
LiteResearcher, and OpenMLE policy surfaces all accept exactly one Qwen XML
`<tool_call>` per turn. Supervise it through update200 and terminal checkpoint
closure, then run the frozen matched 4x128 held-out evaluation.

## Live execution status (2026-09-03 22:21 CST)

- [x] r91c launched on `6.235.112.253` as the frozen pre-deconfliction control.
  It has committed update 40 with all four routes present, nonzero actor/critic
  updates, per-update publication, and zero rollout failures, stale drops, or
  queue-overflow evictions. Keep it online until r92 launch assets are fully
  ready; it is not the candidate to merge.
- [x] Independent reviewers4--6 exposed and drove closure of the prompt-projection
  failure family: nested/dynamic legacy action records, non-idempotence, incorrect
  `search[keywords]` semantics, exact opaque-payload corruption, endpoint-valid
  WebShop whitespace variants, and identifier-prefixed `my_apply_patch` collisions.
  The current repair projects policy-visible text only, preserves decoded opaque
  values byte-for-byte, and leaves reward, PPO/GAE, endpoint audit state, and the
  shared rollout unchanged.
- [x] Current isolated B300 inner suite: 87 tests plus 112 subtests pass. Targeted
  outer wrapper/routing suite: 47 tests plus 56 subtests pass with the dedicated
  vLLM compatibility test separately passing. Replaying r91c updates 1--45 through
  the current projector covers 32,180 rows and 28,704 policy-visible user messages;
  it leaves zero executable-looking legacy forms, zero non-idempotent projections,
  zero pseudo-XML imbalance, zero protected-fact loss, and does not alter source
  rollout files. An expanded 1,000-case adversarial exact/idempotence fuzz also
  passes for both WebShop and OpenMLE.
- [x] Reviewer7 found one final identifier-prefix plus backticked-`apply_patch`
  collision. The inert encoder and both final projectors now use indivisible
  quoted/plain-token boundaries, with exact shell and patch regressions for both
  WebShop and OpenMLE. Reviewer8's final narrow review reports Critical/High/
  Important/Medium = 0 and `VERDICT: PASS`; its only Low note about stderr capture
  was closed by a fresh runtime log containing the unittest `OK` footer.
- [x] Inner `58c52228d356c18cdd0e24546c866df4d56829e8` and outer
  `a9825c69ecf6a7d21c0d5e45ab78abea7648aff8` are committed and pushed. The
  immutable r92 source trees are clean, resolve-only PASS recorded zero endpoint
  or trainer spawn, r91c was retired after serving as the same-B300 control, and
  fresh r92 is running on `6.235.112.253`.
- [x] r92 update1 and updates1--5 truth gates PASS: every update consumed 64
  complete trajectories with all four routes, nonzero actor/critic updates,
  per-update parameter publication, and zero rollout failure, stale drop, or
  queue overflow. Qwen XML normalization improved over r91c on all four routes;
  the repair remains approximately speed-neutral on the same B300.
- [x] Through update8, same-stage r92 exceeds r91c on Shop PS, SWE success,
  LiteResearcher success, OpenMLE BBR/VSR, and all four route returns. Within r92,
  updates6--8 show clear early lift for SWE and LiteResearcher, slight OpenMLE
  lift, and flat Shop PS with materially lower invalid actions. This is not yet
  evidence of four-route stable lift or a new historical best.
- [x] Through update8, all four routes exhibit real executable checkpoint chains.
  Behavioral write -> replace -> read -> later-action trajectories are Shop
  105/130, SWE 60/128, LiteResearcher 65/129, and OpenMLE 61/125. These are
  system-requested checkpoint chains, not evidence of autonomous long-term-note
  creation.
- [ ] At update10 verify the complete six-rank checkpoint, exact publication and
  owned-process health, then produce the same-hardware quality/format/memory/case
  report while training remains online.
- [ ] At update20 decide stable four-route training lift from multi-update blocks;
  do not stop for an isolated weak policy window and do not run held-out eval.

## Guardrails

- Never resume r90c, checkpoint140, or any old optimizer/frontier state.
- Keep the current eight-GPU allocation; do not request or release another one.
- Keep `AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py`
  byte-identical to the current committed version.
- Route-specific parsing and canonicalization belong in environment clients;
  endpoint legacy grammars remain internal execution formats.
- Malformed, prose-wrapped, multiple, unknown, or bare policy actions must reach
  endpoints only as the common impossible-action sentinel and must consume one
  recoverable-invalid step.
- LiteResearcher's policy-facing terminal action is the XML `answer` function;
  the client maps it to the frozen endpoint's native `<answer>...</answer>` form.
- No held-out evaluation before verified update200 and terminal checkpoint.

## Phase 1: freeze the evidence boundary

1. Reconfirm the sole owner process, holder-only allocation, no trainer/Ray/vLLM
   residue, and exact remote LiteResearcher endpoint source.
2. Record the current inner/outer commits and the committed hash of the shared
   rollout entrypoint.
3. Treat any pre-existing uncommitted implementation as a candidate patch:
   inspect it, preserve it, and accept it only after tests and independent review.

Evidence:

```bash
git -C "$LOCAL/AgentGym" status --short --branch
git -C "$LOCAL" status --short --branch
git -C "$LOCAL/AgentGym" hash-object \
  agentenv/agentenv/envs/verl_qwen_tool_parser.py
git -C "$LOCAL" hash-object \
  AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py
```

## Phase 2: write contract tests before accepting implementation

Add focused tests for all policy-facing routes:

1. Common parser: one exact envelope, schema-backed typed arguments, optional
   decoded EOS handling, and rejection of prose, duplicate calls, unknown tools,
   malformed parameters, bare JSON, and bare actions.
2. WebShop: prompt/tool manifest/checkpoint write/read use XML; search, click,
   shell_command, apply_patch, and conditional ask normalize to the existing
   endpoint grammar; all bare legacy forms become the sentinel; raw and submitted
   actions remain separately evidenced.
3. SWE-smith: prompt/checkpoint/terminal sentinel examples use XML; shell and
   patch calls normalize to the endpoint's canonical grammar; bare forms and
   mixed prose become the sentinel.
4. LiteResearcher: add the XML `answer` schema/example; validate and normalize
   search, visit, workspace calls; map XML answer to native `<answer>`; reject the
   old bare answer, bare workspace actions, legacy JSON tool calls, and prose.
5. OpenMLE: preserve its existing XML prompt but make the client fail closed for
   every non-XML policy action and record the same parser evidence schema.
6. Update checkpoint/prompt AST tests so they load shared constants explicitly
   and continue to run without importing the heavyweight runtime.

Run the red tests in the CUDA-capable pod Python environment if the Mac lacks
PyTorch. Define `LOCAL` inside this block so it is independently runnable:

```bash
LOCAL=/Users/luolirui.1/Projects/amg-action-contract-20260903
PYTHONPATH="$LOCAL/AgentGym/agentenv:$LOCAL/AgentGym-RL" \
  python -m unittest discover -s "$LOCAL/AgentGym/agentenv/tests" \
  -p 'test_*qwen*py'
```

## Phase 3: minimal implementation

1. Reuse `parse_single_qwen3_tool_call`; do not maintain another XML parser.
2. Centralize the strict single-call text and invalid-action sentinel beside
   that parser.
3. Keep small route-owned semantic validators/canonicalizers because endpoint
   grammars differ.
4. Convert WebShop checkpoint guidance and post-read cues to XML.
5. Add only the LiteResearcher terminal translation needed by its frozen endpoint.
6. Tighten OpenMLE's existing normalizer to the same fail-closed boundary.
7. Do not modify endpoint source unless a client-boundary test proves that the
   frozen endpoint cannot preserve the required reward/terminal semantics.

## Phase 4: verification and source closure

Run focused client tests, wrapper-policy-turn tests, plugin tests, and compile
checks in the proven runtime. Then run:

```bash
LOCAL=/Users/luolirui.1/Projects/amg-action-contract-20260903
git -C "$LOCAL/AgentGym" diff --check
git -C "$LOCAL" diff --check
git -C "$LOCAL" diff --exit-code -- \
  AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py
PYTHONPATH="$LOCAL/AgentGym/agentenv:$LOCAL/AgentGym-RL:$LOCAL/async_plugins" \
  python -m unittest discover -s "$LOCAL/async_plugins/tests" -p 'test_*.py'
```

Also generate a four-route prompt/parser audit that records, for each route,
the policy schema, accepted XML examples, rejected legacy examples, submitted
endpoint action, parser evidence, and terminal semantics. Run the existing
active-source AST audit and save hashes in the future run directory.

After independent review passes, commit and push the AgentGym inner branch.
Then commit and push the outer branch with the new inner pointer and this plan.

## Phase 5: fresh formal200 launch gate

1. Reconfirm sole owner and holder-only target pod.
2. Materialize exact pushed sources and source locks; attest inner, outer,
   plugin, latest-veRL, endpoint, dataset, Qwen3.5 base-model, task/seed/tool/
   budget/grader, and shared-rollout hashes.
3. Use a new run name and empty run directory. Reject any checkpoint/resume
   argument or pre-existing optimizer/frontier state.
4. Launch exactly one tmux/orchestrator lineage on the existing allocation.
5. At update1 prove 64/64 consumed episodes, all four routes present, nonzero
   actor and critic updates, zero rollout failure/dropped-stale/queue-overflow,
   zero fatal errors, exact owner/process identities, and a healthy holder yield.

## Phase 6: continuous supervision and milestones

At u80, u120, u160, and u200 produce cumulative and latest-20 reports with:

- WebShop progress_score;
- SWE and LiteResearcher `env_info_after.episode_success`;
- OpenMLE BBR and VSR;
- async queue/frontier/staleness/drop/failure health;
- exact six-rank actor/critic model, optimizer, and extra-state checkpoint checks;
- four-route terminal/reward/case health and representative raw success/failure
  trajectories;
- strict write → replace_messages → exact read → subsequent dependent-action
  evidence;
- matched prior-control comparisons where task IDs overlap.

Continue automatically while infrastructure and environment integrity are
healthy. Policy mistakes or one weak metric window require diagnosis, not a
live-contract edit or stop.

At u200 verify 64/64 consumed episodes, nonzero actor/critic update, zero
unacceptable loss/overflow/fatal errors, complete terminal checkpoint and
persisted async state, no update201 commit, orderly run-owned cleanup, holder
restoration, and retained allocation.

## Phase 7: post-terminal evaluation and closure

Only after the u200/no-u201 gate passes, launch the frozen matched held-out
4x128 evaluation from the terminal checkpoint. Verify all 512 terminal rows,
route identities, task IDs, seeds, tools, budgets, graders, and official/native
metrics. Then synchronize `experiments/AMG_TODO.md`, targeted Notion experiment
fields, daily memory, a final evidence manifest, and the supervision state.

Run a second-order fallout and task-closure audit over sibling docs, launch/
analysis scripts, source locks, manifests, active entrypoints, compatibility
aliases, TODO/Notion rows, and residual processes before declaring completion.
