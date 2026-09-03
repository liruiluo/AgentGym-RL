# CompactionRL Native Held-Out Evaluation Plan

## Task contract

- **Task family:** CAMG native held-out evaluation and paper-result backfill.
- **Unique owner:** `experiment/camg-compactionrl-native-eval-20260902`.
- **Objective:** evaluate the completed CompactionRL `global_step_200` actor on the frozen CAMG final panel (128 tasks per environment, 512 total) and produce the four Table 2 success rates plus their equal-weight average.
- **Write scope:** this worktree, a run-scoped CompactionRL evaluation directory, and the explicitly authorized CompactRL row in the paper after final receipts exist.
- **Run scope:** the existing CompactionRL 8-card allocation only, after the complete update-200 checkpoint reaches its declared endpoint and the training owner has completed cleanup/holder handoff.
- **Non-goals:** no fourth 8-card request; no held-out or external evaluation before update 200; no environment-specific rollout; no changes to training rewards, prompts, tasks, budgets, decoding, or graders; no interference with the other two 8-card owners.
- **Success artifact:** a resumable 512-task evaluation with one terminal receipt per assigned task, exact source identities, complete failure accounting, a machine-readable metric summary, and a verified paper build containing the CompactRL row.
- **Verification:** focused evaluator tests, complete available `async_plugins/tests`, active-source audit proving the shared task-agnostic AgentLoop, exact schedule and checkpoint hashes, all 512 terminal cells present, per-route denominator `128`, paper checker/build, and pushed commits.
- **Duplicate avoidance:** do not edit the AgeMem evaluator worktree. Reuse only evaluator commits that have been published and verified as method-independent against the shared 4×128 panel; add only CompactionRL-specific evidence/configuration in this branch. AgeMem-specific evidence parsing or an older 8,167-task deployment schedule must not enter this arm.

## Frozen protocol

- Inputs: canonical package `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/camg-final-heldout-128-v1-20260902` with `manifest.json` SHA256 `d5e9da093103706ba5586fd61121ef25cda1e49513df68179aed703ba2b5d74c`, `evaluation-contract.json` SHA256 `b32760ab2cc416daa3f3987ce287c69c471b25855e59323fd2bcb68083442fdf`, and `SHA256SUMS` SHA256 `42d7a0b721f97b9c1e1afa785371f322810349aa8cdcaa53029033878177671e`.
- Counts: Shop 128; Coding 128; DeepResearch 128; AutoResearch 128; total 512. Every Table 2 method uses byte-identical routing rows, and the primary aggregate is the unweighted macro-average of the four environment-level success rates.
- The larger admitted held-out pools are source/extension pools only. Explicitly reject the obsolete 640-row Shop panel, 21-task Coding readiness panel, 14,684-row complete-pool schedule, and AgeMem-specific 8,167-task deployment schedule as Table 2 denominators.
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
3. Byte-verify the frozen 4×128 package and compose its exact 512-row schedule; exercise only synthetic/CPU fixtures before update 200.
4. At update 200, verify checkpoint completeness, exact-stop receipt, trainer exit, exact-source hashes, cleanup, and holder restoration; merge the actor to a content-addressed HF directory.
5. On the same retained CompactionRL lane, launch the four loopback-only held-out endpoints and one resumable evaluation.
6. Require one terminal receipt for every one of the 512 frozen rows and exact denominator 128 on each route; compute Table 2 metrics from finalized CAMG evidence only.
7. Backfill only the CompactRL row, run the paper checker/full restricted-TeX build, synchronize the visible PDF, push, and perform a second-order fallout audit.
