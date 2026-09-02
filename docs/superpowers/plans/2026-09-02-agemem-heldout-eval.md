# AgeMem Native Held-Out Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute this plan task-by-task with test-first changes and verification before each commit. This supervised run has already selected inline execution; do not pause for another execution-mode choice.

**Goal:** Build a fail-closed, resumable evaluator that launches the four frozen CAMG held-out environments in the same Pod and evaluates the AgeMem-style `global_step_200` checkpoint over the final frozen native held-out set.  The current 21-task Coding artifact is only a runtime-readiness panel and is not an admissible final evaluation denominator.

**Architecture:** Keep the existing task-agnostic `AMGTrajectoryDataset -> AgentLoopManager -> env.step -> PPO/evidence` path unchanged. Add a held-out-only endpoint registry and readiness/identity layer outside the shared loop, strengthen schedule-to-runtime and terminal source-identity checks, and give the evaluator its own exact process/watch-parent/cleanup lifecycle. Training gate receipts remain admissible historical environment evidence but are not used as held-out endpoint launch authority.

**Tech Stack:** Python 3.12, unittest/pytest, Bash endpoint launchers, Ray, veRL standalone SGLang, same-Pod loopback HTTP services.

---

## File map

- Create `async_plugins/agentmemorygym_verl/heldout_endpoints.py`: held-out endpoint registry schema, immutable asset/source/launcher verification, per-route reset identity contracts, and readiness receipt helpers.
- Modify `async_plugins/agentmemorygym_verl/heldout_eval.py`: exact runtime dataset parity check and explicit handling of veRL `@auto_await` constructors.
- Modify `async_plugins/agentmemorygym_verl/heldout_eval_contract.py`: terminal native source-identity validation and evidence publication.
- Modify `async_plugins/agentmemorygym_verl/heldout_eval_orchestrator.py`: use held-out registry, exact attempt run IDs, owner/watch-parent lifecycle, route reset probes, and complete cleanup audit.
- Modify `async_plugins/scripts/run_amg_heldout_eval.py`: expose only the new held-out registry contract and owner lifecycle arguments.
- Create `async_plugins/scripts/heldout_endpoints/{webshop,swesmith,literesearcher,openmle_fast}.sh`: route-specific foreground launchers bound to frozen held-out assets and same-Pod loopback.
- Create `async_plugins/tests/test_heldout_endpoints.py`: registry, launcher, readiness, and reset-identity unit contracts.
- Modify `async_plugins/tests/test_heldout_eval.py`: full schedule/runtime parity and terminal source identity tests.
- Modify `async_plugins/tests/test_heldout_eval_orchestrator.py`: attempt ID, process classifier, watch-parent, cleanup, and failure-unwind tests.

## Non-goals and hard gates

- Do not edit `verl/.../agent_loop.py`, `vllm_rollout.py`, or add any environment-specific rollout.
- Do not run held-out or external evaluation before the formal AgeMem run has a verified `global_step_200` publication.
- Do not probe another Pod's ports, create forwarding, stop parallel owners, or release any allocation.
- Do not treat a training one-update gate receipt as proof that the held-out endpoint loaded the held-out task identity.
- Bind the frozen 8,167-episode denominator exactly: Shop 1,746 + Coding 933 + DeepResearch 5,319 + AutoResearch 169.  Coding's 933-task formal Eval subset comes from 10 complete repositories selected without model outputs or rewards from the fully classified 7,450-task admitted held-out candidate pool; the remaining 6,517 tasks are extension-only.  Every route count must agree across its runtime manifest, the formal split contract, and the composed evaluation schedule.  Reject the old 640-row Shop panel, the 21-task Coding readiness panel, and the 7,450-task candidate pool as a main-table denominator.

### Task 1: Freeze the held-out registry and reset contract

**Files:**
- Create: `async_plugins/agentmemorygym_verl/heldout_endpoints.py`
- Create: `async_plugins/tests/test_heldout_endpoints.py`

- [x] Write fixtures for schema `camg_heldout_endpoint_registry_v1` with exactly four canonical routes.
- [x] Assert registry hash, route order, loopback endpoint, unique port, route attestation, launcher hash/executable bit, immutable source roots/commits, held-out runtime manifest hashes, and all route-specific assets.
- [x] Assert no optimizer/gate receipt is required as launch authority; optionally bind it only as historical evidence.
- [x] Define `/create -> /reset(data_idx) -> route-specific identity -> /close` checks:
  - Shop: `scenario_id`, `orbit_index`.
  - SWE-smith: `instance_id`, `base_repository` from private `/detail` using a 0600 token.
  - LiteResearcher: `row_identity`, `source_pool_index`.
  - AutoResearch: `task_id`, `source_family`, `manifest_role=heldout`, `manifest_sha256`.
- [x] Run the focused test and confirm the old training registry loader fails the new fixture before implementing the new loader.
- [x] Implement the minimal pure-data loader/probe contract and make the test pass.

### Task 2: Prove schedule/runtime and terminal identity parity

**Files:**
- Modify: `async_plugins/agentmemorygym_verl/heldout_eval.py`
- Modify: `async_plugins/agentmemorygym_verl/heldout_eval_contract.py`
- Modify: `async_plugins/tests/test_heldout_eval.py`

- [x] Add a failing test that mutates each of `route_id`, `data_idx`, `item_id`, `uid`, or nested `extra_info` after `AMGTrajectoryDataset.__getitem__`.
- [x] Permit exactly one dataset-owned provenance addition: `extra_info.route_attestation_sha256` from the pinned route registry; reject every other extra-info drift.
- [x] Add failing terminal tests for each route's frozen source identity fields.
- [x] Validate terminal native evidence against `extra_info.source_extra_info` and persist a compact `verified_native_source_identity` receipt in every episode.
- [x] Run focused tests and preserve the existing native metric and AgeMem evidence behavior.

### Task 3: Give the evaluator an exact run-scoped owner

**Files:**
- Modify: `async_plugins/agentmemorygym_verl/heldout_eval_orchestrator.py`
- Modify: `async_plugins/scripts/run_amg_heldout_eval.py`
- Modify: `async_plugins/tests/test_heldout_eval_orchestrator.py`

- [x] Add failing tests proving every attempt uses `<eval-run-id>.attempt-000000`, never the generic `attempt-000000` owner ID.
- [x] Replace substring process detection with an executable/module classifier so file arguments containing `ray`/`vllm` do not trigger false positives.
- [x] Install a separate evaluator parent-death guard bound to PID plus `/proc` start ticks before starting Ray/model servers; require a start receipt.
- [x] Ensure normal unwind and parent-death cleanup target only the exact attempt owner and run ID.
- [x] Audit endpoint ports, exact owner processes, Ray/vLLM processes, `/tmp/agentmemorygym-swesmith-*`, LiteResearcher `/dev/shm/amg-lr-*` roots/mounts/cleanup receipt, and orchestration-root mounts.
- [x] Preserve evaluator -> endpoints -> holder unwind ordering on every failure path.

### Task 4: Implement four held-out endpoint launchers

**Files:**
- Create: `async_plugins/scripts/heldout_endpoints/webshop.sh`
- Create: `async_plugins/scripts/heldout_endpoints/swesmith.sh`
- Create: `async_plugins/scripts/heldout_endpoints/literesearcher.sh`
- Create: `async_plugins/scripts/heldout_endpoints/openmle_fast.sh`
- Test: `async_plugins/tests/test_heldout_endpoints.py`

- [x] Shop: bind `camg_shop_complete_heldout_runtime_manifest_v2`, its sparse routing, all 1,746 fixed held-out episodes, and the frozen product pool.  Read provider bounds/counts from the manifest rather than embedding the obsolete 640-row panel contract.
- [x] SWE-smith: bind `camg_swesmith_formal_eval_runtime_manifest_v5`, its dense 933-task formal routing, the 7,450-task admitted-pool and 6,517-task extension manifests, frozen image assets, a run-scoped `/tmp` sandbox, and a private 0600 detail token.  The extension pool must enter neither training nor the main evaluation denominator.
- [x] LiteResearcher: load exactly held-out rows `0..5318`; retain upstream loader alias `train` while attesting CAMG role `heldout`; use a run-scoped `/dev/shm` trajectory workspace and exact cleanup receipt.
- [x] AutoResearch: bind held-out manifest/private grader, set `expected_role=heldout`, and attest task/source-family/manifest identity.
- [x] Keep every service foreground-supervised and loopback-only; reject missing assets or source drift before listening.

### Task 5: Verify veRL construction and all affected surfaces

**Files:**
- Test: `async_plugins/tests/test_heldout_*.py`
- Audit: `verl/verl/workers/rollout/llm_server/llm_server.py` and `verl/verl/experimental/agent_loop/agent_loop.py`

- [x] Source-audit whether `LLMServerManager.create` and `AgentLoopManager.create` are `@auto_await` wrappers and encode the verified call form explicitly.
- [x] Run focused held-out tests with `PYTHONDONTWRITEBYTECODE=1 python -B`.
- [x] Run the complete available `async_plugins/tests` suite in the pinned Python 3.12 + torch runtime; record unrelated legacy-launcher fixture failures separately rather than weakening the held-out gates.
- [x] Audit active source for environment-specific rollout names/imports/call paths; require one shared `amg_task_neutral_async` entrypoint for at least two environments.
- [x] Confirm `git diff -- verl/.../agent_loop.py .../vllm_rollout.py` is empty.
- [x] Prevent bytecode writes with `PYTHONDONTWRITEBYTECODE=1` / `python -B` and verify no branch-owned `__pycache__` or `.pyc` enters the diff.

### Task 6: Commit, publish, deploy, and supervise

- [ ] Commit the required native source-identity changes in the inner repo first, then update and commit the outer submodule plus evaluator.
- [ ] Fetch the canonical private `github` remotes, verify non-divergence, and push the isolated branches; keep the local bundle `origin` untouched.
- [ ] Build a content-addressed deployment bundle containing source commits, launchers, route registry, schedule, held-out registry, and checksums; verify after upload.
- [x] Reuse the existing r84/B300 lane for AgeMem; do not submit or arm a fourth 8-card allocation.  The active pool is capped at the three existing 8-card lanes unless master explicitly changes it in the current chat.
- [x] Complete the update1 gate for the active reused-lane run (64/64, four routes, nonzero actor/critic gradients, zero failure/drop/overflow/stale drop, complete AgeMem adapter evidence, zero hidden model calls).
- [ ] Continue formal200 without held-out evaluation; monitor immutable checkpoints and runtime health.
- [x] Finish full Coding held-out admission across the frozen 36-repository side and freeze the 933-task complete-repository formal Eval subset separately from the 6,517-task extension pool.
- [ ] After a verified `global_step_200` merged-HF publication, launch exactly one resumable full native held-out evaluation and fill Table 2 only from finalized receipts.
