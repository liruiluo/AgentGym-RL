# CompactionRL Native Held-Out Evaluation Plan

## Task contract

- **Task family:** CAMG native held-out evaluation and paper-result backfill.
- **Unique owner:** `experiment/camg-compactionrl-native-eval-20260902`.
- **Objective:** evaluate the completed CompactionRL `global_step_200` actor on the frozen four-environment CAMG held-out panel and produce the four Table 2 success rates plus their equal-weight average.
- **Write scope:** this worktree, a run-scoped CompactionRL evaluation directory, and the explicitly authorized CompactRL row in the paper after final receipts exist.
- **Run scope:** the existing CompactionRL 8-card allocation only, after the complete update-200 checkpoint reaches its declared endpoint and the training owner has completed cleanup/holder handoff.
- **Non-goals:** no fourth 8-card request; no held-out or external evaluation before update 200; no environment-specific rollout; no changes to training rewards, prompts, tasks, budgets, decoding, or graders; no interference with the other two 8-card owners.
- **Success artifact:** a resumable 6,149-episode evaluation with one terminal receipt per assigned episode, exact source identities, complete failure accounting, a machine-readable metric summary, and a verified paper build containing the CompactRL row.
- **Verification:** focused evaluator tests, complete available `async_plugins/tests`, active-source audit proving the shared task-agnostic AgentLoop, exact schedule and checkpoint hashes, 6,149/6,149 terminal cells, route-level denominator checks, paper checker/build, and pushed commits.
- **Duplicate avoidance:** do not edit the active AgeMem evaluator worktree. Reuse its method-independent evaluator commit once published; add only CompactionRL-specific evidence/configuration in this branch. Until that publication exists, prepare immutable inputs and tests that do not duplicate the generic orchestration implementation.

## Frozen protocol

- Panel: `camg-native-eval-ready-v1-20260901`.
- Counts: Shop 640, Coding 21, DeepResearch 5,319, AutoResearch 169; total 6,149.
- Policy checkpoint: the verified merged-HF publication derived from CompactionRL `global_step_200`.
- Shared entrypoint: `amg_task_neutral_async`; environment lifecycle remains in wrappers/endpoints.
- Batch size: 64 real episodes plus deterministic padding only when needed by the inference runtime. Padding never enters denominators.
- Shop success: `sum(completed_sessions) / (6 * assigned_episodes)`.
- Coding success: resolved issues / assigned episodes.
- DeepResearch success: accepted terminal answers / assigned episodes.
- AutoResearch success: valid submissions whose direction-adjusted private native score strictly beats the frozen generic baseline / assigned episodes.
- Average Success: equal-weight mean of the four route success rates.

## Execution order

1. Keep the live run immutable and continuously supervised; seal registered training milestones and never run held-out data before the declared update-200 endpoint.
2. Consume the published task-agnostic held-out evaluator, then add a CompactionRL evidence adapter and exact checkpoint/model manifest.
3. Compose and byte-verify the 6,149-row schedule from the frozen held-out package; exercise only synthetic/CPU fixtures before update 200.
4. At update 200, verify checkpoint completeness, exact-stop receipt, trainer exit, exact-source hashes, cleanup, and holder restoration; merge the actor to a content-addressed HF directory.
5. On the same retained CompactionRL lane, launch the four loopback-only held-out endpoints and one resumable evaluation.
6. Require 6,149 terminal receipts and exact route denominators; compute Table 2 metrics from finalized native evidence only.
7. Backfill only the CompactRL row, run the paper checker/full restricted-TeX build, synchronize the visible PDF, push, and perform a second-order fallout audit.
