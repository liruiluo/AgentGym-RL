# AMG latest-veRL Four-Environment Main Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the accepted latest-veRL fully-asynchronous PPO stack into the canonical AMG main path and launch one fresh 400-update shared-policy run over WebShop, SWE-smith, LiteResearcher, and OpenMLE-fast without duplicating upstream rollout, queue, trainer, or optimizer machinery.

**Architecture:** Keep one upstream `fully_async_main`, one `amg_task_neutral_async` AgentLoop, and one actor/critic/checkpoint lineage. A frozen interleaved JSONL manifest carries a per-row `route_id`; an immutable route registry maps that ID to wrapper-owned endpoint and attestation settings. The dataset resolves only policy framing and row identity, the AgentLoop creates the selected wrapper client and records route-labelled evidence, and upstream veRL retains FIFO completion/dequeue, staleness, parameter publication, and PPO ownership.

**Tech Stack:** Python 3.12, Hydra/OmegaConf, latest veRL fully-async policy, Ray, SGLang, FSDP2, pytest/unittest, JSONL manifests, shell orchestration.

---

## Non-negotiable contracts

- Do not read or derive implementation truth from the paper; use current source, launch receipts, trajectory ledgers, and environment attestations.
- No domain branch or parser in upstream/shared rollout. Environment lifecycle, action parsing, compaction, filesystem memory, and reward stay in wrappers.
- Use upstream per-row metadata, custom dataset, stateful sequential sampling, FIFO queue, staleness accounting, parameter publication, checkpointing, and PPO trainer.
- One row produces one complete multi-action episode bundle; `rollout.n=1`; one shared actor/critic/optimizer.
- Exact per-update 16/16/16/16 is not required. Dispatch is frozen 1:1:1:1 round-robin; learner-consumed exposure is checked over rolling 8-update windows.
- Do not start a disposable Multitask20 lineage. The fresh Multitask400 run uses in-run update1/5/20/40 gates and continues when they pass.
- No external evaluation before update400. Training-side trajectory and reward evidence drives stop-loss.
- Do not merge rejected infra experiment history. Later rebase/cherry-pick only accepted r51c/other infra commits.

## File map

- Create `async_plugins/agentmemorygym_verl/routes.py`: immutable route-registry loading, digest validation, route selection, and single-environment compatibility conversion.
- Modify `async_plugins/agentmemorygym_verl/env_client.py`: construct a client from one already-resolved route specification; keep wrapper-specific identity forwarding here, outside shared rollout.
- Modify `async_plugins/agentmemorygym_verl/dataset.py`: validate per-row route identity, attach the same AgentLoop name, select route-specific policy framing, and preserve route-local `data_idx`.
- Modify `async_plugins/agentmemorygym_verl/agent_loop.py`: resolve `route_id` per episode, create the selected client, use route-local horizon/observation limits, and put route identity on every action row without domain branching.
- Create `async_plugins/agentmemorygym_verl/multitask_manifest.py`: compose and verify a frozen round-robin manifest from four independently attested schedule files.
- Modify `async_plugins/agentmemorygym_verl/config_contract.py`: validate route-registry identity, multi-environment schedule accounting, one shared AgentLoop, and exact optimizer budget while preserving legacy single-environment checks.
- Modify `async_plugins/agentmemorygym_verl/launch.py`: accept a frozen route registry and multi-environment schedule, emit generic Hydra overrides, and retain the existing OpenMLE single-environment entrypoint as a compatibility path until cutover.
- Create `async_plugins/scripts/launch_amg_multitask_fully_async.sh`: thin one-command wrapper around the Python launcher; it must not reimplement endpoint lifecycle.
- Create `async_plugins/config/amg_multitask400.yaml`: reviewed default topology/budget and references to the frozen manifest/route registry; environment endpoints are injected by the one-click orchestrator.
- Modify `async_plugins/agentmemorygym_verl/finalizer.py`: per-route conservation, composition, staleness/drop, action/token, reward, compaction, document write/read, and complete-memory-chain summaries.
- Create `async_plugins/agentmemorygym_verl/online_monitor.py`: non-blocking update1/5/20/40 snapshots that can read a live run without requiring terminal finalization.
- Make the smallest environment-neutral changes required in the pinned external veRL `message_queue.py`, `fully_async_rollouter.py`, and `fully_async_trainer.py` for metadata-labelled queue events and non-evicting end-of-stream semantics. Use an existing upstream hook instead if source inspection finds one; never add AMG route names or reward logic there.
- Replace the legacy-source audit with runtime-resolved source-path and AST checks for the exact imported veRL/plugin stack.
- Add focused tests under both `async_plugins/tests/` and the pinned veRL worktree for each boundary.

### Task 1: Immutable route registry

**Files:**
- Create: `async_plugins/agentmemorygym_verl/routes.py`
- Test: `async_plugins/tests/test_routes.py`

- [x] Write failing tests for exactly four unique route IDs, regular-file/no-symlink loading, SHA-256 pinning, required endpoint/config fields, local loopback endpoint policy, immutable normalized output, unknown route failure, and single-environment compatibility.
- [x] Run `python -m pytest -q async_plugins/tests/test_routes.py` and confirm the new tests fail for the missing module.
- [x] Implement a small `RouteRegistry`/`RouteSpec` API; do not add scheduling, lifecycle, or domain action logic.
- [x] Run the focused tests and compile the module.
- [x] Commit only route-registry code and tests.

### Task 2: Per-row dataset framing and identity

**Files:**
- Modify: `async_plugins/agentmemorygym_verl/dataset.py`
- Modify: `async_plugins/agentmemorygym_verl/config_contract.py`
- Test: `async_plugins/tests/test_dataset.py`
- Test: `async_plugins/tests/test_config_contract.py`

- [x] Add failing tests showing two rows with distinct route IDs receive the selected wrapper’s exact framing, route-local `data_idx`, globally unique `item_id`, explicit `agent_name=amg_task_neutral_async`, and route-labelled `data_source` without mutating source rows.
- [x] Keep globally unique rollout `index` separate from route-local `data_idx`; preserve legacy one-route schedules while requiring explicit global indices for multi-route rows.
- [x] Add negative tests for missing/unknown/mismatched route IDs and framing failures; assert every bootstrap client closes.
- [x] Refactor dataset initialization to load the registry once and cache only immutable framing by route; preserve the current one-route behavior.
- [x] Run focused tests, then all async plugin CPU tests available in the locked runtime.
- [x] Commit the dataset change separately.

### Task 3: One task-neutral AgentLoop over four wrappers

**Files:**
- Modify: `async_plugins/agentmemorygym_verl/agent_loop.py`
- Modify: `async_plugins/agentmemorygym_verl/env_client.py`
- Test: `async_plugins/tests/test_agent_loop_contract.py`
- Test: `async_plugins/tests/test_env_client_identity.py`

- [x] Add failing fake-client tests with four distinct routes and deliberately different lifecycle behavior. Every row must create only its selected wrapper and close it on success, exclusion, and error.
- [x] Resolve route-local `max_rounds` and `max_observation_tokens` before the episode. Do not key any executable branch on a concrete environment name in `agent_loop.py`.
- [x] Add `route_id` to every action row and `AgentLoopOutput.extra_fields`; preserve exact sampled tokens/logprobs and the existing context-transition path.
- [x] Keep wrapper-specific constructor/identity adaptation in `env_client.py`; prove OpenMLE identity fields remain exact and other routes do not receive them.
- [x] Run focused tests and an AST no-domain-branch check.
- [x] Commit the AgentLoop/client change separately.

### Task 3b: Real four-wrapper bootstrap contract

**Files:**
- Modify only after gate-owner handoff: the wrapper-owned client/config surfaces identified by the four one-update receipts.
- Test: focused wrapper contract tests plus `async_plugins/tests/test_dataset.py` integration fixtures.

- [ ] Consume the gate owner’s exact WebShop, SWE-smith, LiteResearcher, and OpenMLE-fast wrapper commits and endpoint metadata; do not infer them from stale launch scripts.
- [ ] Make every real wrapper expose a nonempty immutable `policy_framing()` with a frozen digest. WebShop must not inherit the base `None`; LiteResearcher must receive and verify its formal prompt/metadata rather than relying on an unconfigured client.
- [ ] Prove each wrapper’s reset/observe/step/context-transition contract, route-local `data_idx`, prompt digest, endpoint attestation, and guaranteed close path through the same dataset plus AgentLoop.
- [ ] Keep all adaptation in wrapper/client-factory code; do not add a fallback framing or concrete environment branch to the dataset/shared loop.

### Task 4: Frozen four-way manifest

**Files:**
- Create: `async_plugins/agentmemorygym_verl/multitask_manifest.py`
- Modify: `async_plugins/agentmemorygym_verl/config_contract.py`
- Test: `async_plugins/tests/test_multitask_manifest.py`
- Test: `async_plugins/tests/test_config_contract.py`

- [x] Correct the dataset/schedule contract so globally unique `index` is independent from route-local `data_idx` while legacy one-route schedules remain valid.
- [ ] Add failing tests that compose four source schedules into exactly 25,600 globally unique rows for formal400, preserving each route’s local `data_idx`, task/source-family provenance, and source digest.
- [ ] Assert every four-row dispatch block contains one row per route, every row selects the same AgentLoop, and rerunning from identical inputs is byte-for-byte deterministic.
- [ ] Reject duplicate global item IDs, route-local index drift, wrong roles, missing source attestations, and source exhaustion not covered by an explicit deterministic repetition contract.
- [ ] Implement streaming composition so large schedules are not loaded repeatedly.
- [ ] Run focused tests and a small synthetic 4×N fixture.
- [ ] Commit manifest code and tests.

### Task 5: Generic launch/config contract

**Files:**
- Modify: `async_plugins/agentmemorygym_verl/config_contract.py`
- Modify: `async_plugins/agentmemorygym_verl/launch.py`
- Test: `async_plugins/tests/test_config_contract.py`
- Test: `async_plugins/tests/test_launcher_contract.py`

- [ ] Add failing tests for route-registry path/hash propagation to both actor and data config, formal400 budget (`400*64=25,600`), one AgentLoop, sequential/non-shuffled input, and no legacy global `task_name/env_addr` in multi-environment mode.
- [ ] Generalize schedule inspection to return per-route counts and provenance while preserving exact single-environment behavior.
- [ ] Split OpenMLE publication parsing from generic launch assembly rather than weakening OpenMLE attestation checks.
- [ ] Build multi-environment overrides from the frozen route registry; keep upstream fully-async defaults and accepted infra values unchanged.
- [ ] Run focused tests and Hydra resolve-only verification against the pinned latest-veRL tree.
- [ ] Commit launch/config changes separately.

### Task 6: Generic asynchronous queue evidence and exact termination

**Files:**
- Inspect first, then minimally modify only if upstream lacks an equivalent hook:
  - pinned latest-veRL `verl/experimental/fully_async_policy/message_queue.py`
  - pinned latest-veRL `verl/experimental/fully_async_policy/fully_async_rollouter.py`
  - pinned latest-veRL `verl/experimental/fully_async_policy/fully_async_trainer.py`
- Modify: `async_plugins/agentmemorygym_verl/finalizer.py`
- Create: `async_plugins/agentmemorygym_verl/online_monitor.py`
- Test: pinned veRL queue/termination tests and `async_plugins/tests/test_finalizer.py`

- [ ] Add generic metadata-labelled events at their true owners for rollout completion, enqueue acceptance, overflow eviction, dequeue, stale rejection, and optimizer consumption. The veRL changes may transport/count opaque labels but must contain no AMG route names, parser, reward, or lifecycle logic.
- [ ] Replace the bounded-queue `put_sample(None)` end signal with a non-evicting shutdown/end-of-stream path. Add an end-to-end queue test proving an exact multiple of 64 accepted episodes yields exactly the requested optimizer-update count and no underfilled final group; fail the formal contract on any drop or missing update.
- [ ] Add finalizer fixtures for per-route dispatch/completion/enqueue/dequeue/failure/overflow/stale accounting and optimizer-consumed episode/action/token shares. A post-run parser must not pretend to infer events that were never emitted.
- [ ] Add rolling 8-update exposure checks requiring every route and 20%--30% optimizer-consumed episode share; at updates 1 and 5 report prefix composition without applying an undefined eight-update verdict. Report action/token shares without forcing equality.
- [ ] Preserve existing global queue conservation and single-route memory checks; generalize reward, compaction, document write/read, and complete-memory-chain summaries by route.
- [ ] Create a read-only online observer that snapshots update1/5/20/40 from a live run without requiring terminal checkpoint/finalization. It must not pause/drain the async pipeline; only a separate exact-PID hard-gate action may stop a conclusively invalid run.
- [ ] Run one single-route and one four-route synthetic queue fixture, then full pinned-veRL/plugin tests.
- [ ] Commit generic veRL instrumentation separately from AMG finalizer/monitor code.

### Task 7: Thin one-click multi-environment launch surface

**Files:**
- Create: `async_plugins/scripts/launch_amg_multitask_fully_async.sh`
- Create: `async_plugins/config/amg_multitask400.yaml`
- Test: `async_plugins/tests/test_multitask_launcher.py`
- Modify only after gate-owner handoff: environment-specific launcher/config files identified by the four single-card receipts.

- [ ] Consume the four gate owner’s exact source commits, endpoint start commands, asset hashes, ports, timeout/resource bounds, and cleanup contracts; do not guess or copy stale scripts.
- [ ] Start all four endpoints concurrently where dependencies allow, prove same-Pod loopback and identity, then exec the generic Python launcher.
- [ ] Install exact PID/start-ticks cleanup for every endpoint and the trainer; retain holder marker transaction and process guard behavior.
- [ ] Add resolve-only, fail-first partial-start cleanup, endpoint-collision, source-drift, and cold/warm startup timing tests.
- [ ] Emit a single launch receipt that binds all four route attestations and the unified schedule.
- [ ] Commit only after all four independent one-update receipts exist.

### Task 8: Clean main-candidate and in-run formal gate

**Files:**
- Create/update: source-lock, run contract, route registry, schedule certificate, and launch receipt under the canonical shared workspace.
- Update: `experiments/AMG_TODO.md`

- [ ] Freeze the r51c update20 verdict and any final accepted profile-guided candidate before selecting the infra commits.
- [ ] Rebase/cherry-pick the multi-environment commits onto the exact accepted outer and veRL commits; resolve no rejected experiment code into the tree.
- [ ] Run full plugin tests, compileall, Hydra resolution, deterministic manifest tests, and four one-click cold/warm checks. Resolve the actual imported module paths at runtime and AST-audit those exact veRL/plugin files; do not scan the inactive nested legacy `AgentGym-RL/verl` tree as evidence.
- [ ] Obtain the four real single-card one-update receipts; verify each requirement field directly.
- [ ] Launch one fresh Multitask400 lineage from the exact clean main-candidate commit. Update1 is the first mixed-environment integration gate and must continue when valid; do not stop/restart merely to rename it formal.
- [ ] After that exact lineage passes update1 source, update-integrity, four-route, and behavior checks, fast-forward canonical main to the identical commit and push while the trainer continues. Keep the old synchronous line as a tag/rollback point, not an active parallel main.
- [ ] At updates 1/5/20/40, use the live observer to freeze prefix evidence while the trainer continues. Apply rolling-8 composition only from update8 onward. Stop only on a declared semantic, safety, throughput, composition, or same-update-quality failure.
- [ ] At update400, verify checkpoint completeness, per-route training curves/trajectory samples, queue conservation, memory behaviors, cleanup, and only then run matched endpoint evaluation.
