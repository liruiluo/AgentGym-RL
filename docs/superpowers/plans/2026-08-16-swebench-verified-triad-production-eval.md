# SWE-bench Verified Three-Arm Production Evaluation Implementation Plan

> **Execution ownership:** `amg-swebench-triad-eval-0816` implements this plan sequentially. Independent agents are read-only reviewers under prewritten contracts; they never own execution, remote access, Docker, or GPU state.

**Goal:** Build the smallest external deployment that runs the frozen Qwen3.5-4B SWE-bench Verified one-task triad gate, resumes all 500 tasks across the exact three arms, and produces official v4.1.0 outcomes and matched contrasts.

**Architecture:** Keep `PairedRunner`, shared rollout, benchmark adapter, and pushed integration refs unchanged. Add a deployment-only package that freezes identities, stages one digest-pinned OCI image at a time from the already verified blob cache, binds the published SWE client and exact-token vLLM transport, seals each SWE server in a fenced cgroup-v1 memory+pids envelope with hard tmpfs byte/inode quotas and anchored full-rootfs attestation, commits crash-recoverable cell/triad state atomically, and hands closed predictions to the pinned official grader. A lifecycle CLI owns vLLM, the per-cell SWE server, Docker image staging/eviction, gate-to-full transition, resume/dedupe, heartbeats, grading, holder fail-safe restoration, and cleanup.

**Tech Stack:** Python 3.12, stdlib HTTP/file locking, published `paired_eval` and AgentGym adapters, vLLM OpenAI server, Docker Engine 27.5.1, crane 0.21.9, SWE-bench v4.1.0, unittest.

**Frozen deployment root and identities:** `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/triad-eval-20260816`; outer `aa2e9c80d572b513b5849c6d9b37a8dc4698bbc3`; inner `a0cc3ecf989ee89ba19a8e979617b4ec38909331`; dataset revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, JSONL `392529c5e79ca273bf0b073be35169beb68c604a26d9aef5514912fc584fa6cb`, ID ledger `a6b0fd7c8c2969a0eef892e032250adcfa6d32362d395c246930e61b575ac9b9`; harness `726c5461e2ef52d83cf1ea2107870a8bb3328d57`, tree `f178530b37202c549b1b2b3300db2da90da648db`; tag ledger `b69e618cfcfd2a59c3897e3f4856dbd88c4eeb921a5b24467a90bff6fa48581a`; manifest index `f2c1fb29457b66034cb04067f93707833125c8284b93771c924c10878ad9cd9b`; derived 500-row tag/digest TSV `b327b313612adefbc12161e2bf1e63e54925cbfcdccc26a416c1f7e94686af6b`; certified cache 1,158 descriptors / 117,637,519,356 bytes / bad=0.

**Recovery supersession (2026-08-16):** Continue from deployed outer `218f64d706fd755f99bbaaecabc922328f70d2d6` / inner `a0cc3ecf989ee89ba19a8e979617b4ec38909331`; the older outer SHA above remains provenance for the published integration only. The real task-index-0 `00/10/11` gate must run the pinned official grader. Those three accepted cells and boolean outcomes are the first cells of the canonical 1,500-cell manifest and must be resumed, not rerun, when the full driver continues with the remaining 1,497 cells. The gate is still not a standalone benchmark result.

**Frozen model:** `/home/ai-jingyan-train/luolirui.1/post-train/models/Qwen3.5-4B`; shard 1 (5,329,398,688 bytes) `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61`; shard 2 (3,990,429,408 bytes) `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188`; `model.safetensors.index.json` `cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d`; `config.json` `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`; `tokenizer.json` `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`; `tokenizer_config.json` `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8`; `chat_template.jinja` `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`.

**Literal asset paths:** dataset `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/data/pinned_verified_test.jsonl`; harness `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/src/SWE-bench`; manifest index `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/images/instance-manifest-index.jsonl`; certified blob cache `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/images/blob-cache`; isolated Docker socket `/root/.local/state/amg-external-eval-container-runtime-v1/docker.sock`.

---

## File map

- Create `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/__init__.py`: package version and frozen arm names.
- Create `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/atomic.py`: fsync + atomic JSON/JSONL writes and exclusive cell claims.
- Create `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/identity.py`: source/data/model/runtime certificates and immutable manifest generation.
- Create `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/oci.py`: one-image cached crane materialization, Docker load/tag verification, and mirror update.
- Create `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/resource_guard.py`: cgroup-v1 aggregate memory+pids fencing, hard tmpfs workspace/memory quotas, full-rootfs attestation, and cleanup evidence.
- Create `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/model_transport.py`: add vLLM `return_token_ids=true` and fail closed on endpoint shape drift.
- Create `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/runtime_factory.py`: published registry/client adapter hooks, prediction artifact finalization, and grader queue receipt.
- Create `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/state.py`: resumable attempt WAL, accepted cell records, triad validation, dedupe, and canonical result assembly.
- Create `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/official_grader.py`: pinned single-instance v4.1.0 invocation, outcome parsing, retry-safe outcome records, and aggregate contrasts.
- Create `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/cli.py`: preflight, gate, full run, resume, grade, status, privacy audit, and cleanup lifecycle.
- Create `AgentGym-RL/tests/agentmemory/test_swebench_triad_eval_*.py`: focused unit/integration tests for every deployment boundary.
- Do not modify `AgentGym-RL/scripts/agentmemory/paired_eval/**`, `AgentGym/**`, or any `vllm_rollout.py`.

### Task 1: Freeze exact identities and build immutable manifests

**Files:**
- Create: `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/identity.py`
- Test: `AgentGym-RL/tests/agentmemory/test_swebench_triad_eval_identity.py`

- [ ] Write failing tests that reject wrong outer/inner SHAs, wrong 500-row source hash, unsorted/duplicate IDs, wrong 500-tag digest ledger, model file drift, an arm other than `native/amg_compaction_only/amg_memory`, and any treatment-excluded mismatch.
- [ ] Run `PYTHONPATH=AgentGym-RL/scripts/agentmemory python -m unittest AgentGym-RL/tests/agentmemory/test_swebench_triad_eval_identity.py -v`; expect failures for missing implementation.
- [ ] Implement canonical SHA-256 helpers, the model/tokenizer aggregate receipts, a 500-task manifest builder with seed 0 and task-major arm order, and certificate assertions for exactly 1,500 unique cells and one treatment-excluded hash per task.
- [ ] Freeze decoding/budgets in the manifest: temperature 0, top-p 1, max output 2,048; max policy/tool turns 250; max model/prompt tokens 32,768/30,720; max observation tokens 8,192; total accounting tokens 8,388,608; wall limit 1,800 seconds.
- [ ] Run the focused test and `python -m paired_eval expand` against a generated fixture; expect 1,500 canonical rows and lattice `00/10/11`.
- [ ] Commit only this task.

### Task 2: Add atomic state, attempt WAL, resume, and dedupe

**Files:**
- Create: `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/atomic.py`
- Create: `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/state.py`
- Test: `AgentGym-RL/tests/agentmemory/test_swebench_triad_eval_state.py`

- [ ] Write failing crash-window and two-worker race tests at claim, endpoint row, prediction, accepted row, grader report, official outcome, and eviction boundaries.
- [ ] Implement a fenced state machine with generation, host ID, boot ID, PID, and PID start time. Reclaim only after proving the recorded owner dead; every later write must match the current generation and immutable input digests.
- [ ] Separate retryable attempt rows from accepted endpoint rows. Accept only a lifecycle-closed row with a durable prediction (explicit empty patch allowed) and digest-bound queued grader receipt; pre-artifact infrastructure failures remain retryable.
- [ ] Coordinate existing artifacts by digest after crashes; reject stale-generation writes and a second distinct endpoint/prediction/report/outcome for the same cell.
- [ ] Rebuild `results.jsonl` only from accepted cells in manifest order, then exact-join all 1,500 manifest cells to 1,500 unique boolean official outcomes before final aggregation.
- [ ] Run the focused tests twice to prove resume is idempotent.
- [ ] Commit only this task.

### Task 3: Stage one certified OCI image without redownloading valid blobs

**Files:**
- Create: `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/oci.py`
- Test: `AgentGym-RL/tests/agentmemory/test_swebench_triad_eval_oci.py`

- [ ] Write failing fixtures for manifest/config/layer digest mismatch, unsafe tar paths, incomplete cache reuse, duplicate image aliases, Docker config-ID mismatch, and base-commit mismatch.
- [ ] Implement lookup from the frozen 500-row manifest index; require every referenced blob to match certified size and SHA-256 before use.
- [ ] Run `crane pull --format=tarball --cache_path /home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/images/blob-cache` for only the selected digest, capture its exact manifest/config, and refuse any network-layer download or digest drift.
- [ ] Pipe the verified tarball through `crane export` into an atomic `swesmith_oci_rootfs_cache_v1` directory; record manifest/config hashes plus a canonical manifest of every rootfs path, type, mode, size, link target, inode/ctime fingerprint, and regular-file SHA-256. Re-attest that full manifest before each cell and fail on any mutation/addition/deletion.
- [ ] Load the same tarball into the isolated Docker socket, tag the image by the canonical SWE tag, and require Docker image ID `sha256:<config digest>`.
- [ ] Clone/fetch the image's `/testbed/.git` into the dedicated `owner__repo` mirror and prove the exact dataset `base_commit` resolves before policy reset.
- [ ] After a task's triad and three official outcomes are durable, immediately evict only its loaded tag, containers, extracted rootfs, quota mounts, and scratch; retain certified blobs and the repository mirror. Test two-task transition and crash residue recovery.
- [ ] Run the fixture tests; on the pod run one read-only/reusable gate-image staging probe and prove cached blobs remain unchanged.
- [ ] Commit only this task.

### Task 4: Close the formal sandbox gate and bind the published SWE runtime

**Files:**
- Create: `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/model_transport.py`
- Create: `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/runtime_factory.py`
- Create: `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/resource_guard.py`
- Test: `AgentGym-RL/tests/agentmemory/test_swebench_triad_eval_runtime.py`
- Test: `AgentGym-RL/tests/agentmemory/test_swebench_triad_eval_resource_guard.py`

- [ ] Write failing tests proving chat requests add `return_token_ids=true`, tokenize requests remain unchanged, missing/malformed prompt or response token IDs fail closed, and no benchmark/arm branch enters `PairedRunner`.
- [ ] Implement a transport wrapper over `UrllibJsonTransport` that requests vLLM's exact prompt/response token IDs and validates the returned shape before the published model client consumes it.
- [ ] Launch each per-cell SWE server through the isolated Docker daemon's writable cgroup-v1 mount namespace. Create unique memory+pids cgroups under the owned parent, set limits before attaching/exec, prove all descendants inherit, record peak/fail counters, and require both task lists empty before removal.
- [ ] Use a deployment-only workspace materializer/sandbox subclass: mount exact-size/exact-inode tmpfs filesystems before the published sandbox snapshots the workspace/external-memory roots, and unmount only owned mounts on every close/error path. No change enters `AgentGym/**`.
- [ ] Add live negative tests for descendant memory exhaustion, fork exhaustion, transient byte fill, inode fill, and rootfs mutation. A missing controller, escaped descendant, stale mount, nonempty cgroup, or failed negative test forbids `gate/PASS.json`.
- [ ] Implement the SWE builder through `PairedEvalRegistry` and `ClientEnvironmentAdapter`; pass the exact server URL, per-attempt private run ID/capability, arm, 500-row image-manifest hash, and the frozen model config.
- [ ] Implement artifact finalization by forcing horizon only when the endpoint is nonterminal, fetching the exact prediction row, storing the full patch only in private evidence, and returning an `ArtifactResult`.
- [ ] Implement grader handoff as a digest-bound queued receipt with `official_resolved=None`; queued is not an official result and official grading remains a separate post-close phase.
- [ ] Run published registry/runner tests plus the focused runtime test.
- [ ] Commit only this task.

### Task 5: Implement pinned official grading and matched summaries

**Files:**
- Create: `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/official_grader.py`
- Test: `AgentGym-RL/tests/agentmemory/test_swebench_triad_eval_grader.py`

- [ ] Write failing tests for prediction schema/order, non-pinned harness import, wrong Docker socket/root, duplicate/non-boolean outcome, absent/stale report, timeout, patch-apply failure, queued-receipt misuse, and denominator drift.
- [ ] Invoke exact commit `726c5461e2ef52d83cf1ea2107870a8bb3328d57` with `PYTHONPATH` pinned to the checkout, local dataset JSONL, one instance ID, one prediction, one worker, 1,800-second timeout, namespace `swebench`, and the isolated Docker socket.
- [ ] Bind each harness run ID/output root to task index, arm, fenced generation, and prediction SHA-256. Parse only that immutable official v4.1.0 report into `{instance_id, arm, resolved, failure_class, report_sha256}`; never expose grader-only fields to the policy server or public summary.
- [ ] Aggregate only the official outcome ledger after exact-joining 500 unique boolean outcomes per arm; compute raw scores plus `compaction_only-native`, `amg_memory-compaction_only`, and `amg_memory-native`.
- [ ] Run the focused tests with a fake harness, then a bounded public-container handshake against the real socket.
- [ ] Commit only this task.

### Task 6: Add the lifecycle CLI and fail-closed preflight

**Files:**
- Create: `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/cli.py`
- Create: `AgentGym-RL/scripts/agentmemory/swebench_triad_eval/__init__.py`
- Test: `AgentGym-RL/tests/agentmemory/test_swebench_triad_eval_cli.py`

- [ ] Write failing tests that gate on exact source/model/data/blob/runtime certificates, live pod UUID, Docker identity, resource-guard live negative-test receipt, vLLM health/model ID, SWE metadata shape, zero active workspaces, and owned-process PID files.
- [ ] Implement `preflight`, `gate`, `run`, `resume`, `grade`, `status`, `audit`, and `cleanup` subcommands.
- [ ] Make `gate` execute one real task under all three arms, run the pinned official grader for all three predictions, and validate endpoint reset/step/artifact/close, prompt treatment exclusion, lattice, dedupe, three unique boolean outcomes, and zero workspace/container residue. Accept the cells into the canonical manifest; do not publish them as a standalone benchmark score.
- [ ] Make `run` require the gate PASS certificate, then automatically enter the full 500-task manifest; for each task stage its image, execute missing cells, accept the triad, close policy state, run/retry official grading, evict owned task staging, emit a heartbeat, and continue.
- [ ] Make cleanup stop only owned services, remove only owned task containers/images/scratch, verify residue 0, and leave allocation/Docker assets/model/source intact.
- [ ] Run all deployment tests and the published 42-test paired suite.
- [ ] Commit only this task.

### Task 7: Deploy an isolated Qwen3.5-compatible serving environment

**Files:**
- Runtime-only: `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/triad-eval-20260816/venv-serving/`, `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/triad-eval-20260816/control/serving-receipt.json`, and service logs/PID files.

- [ ] Download/install only to the shared-disk triad venv, pin vLLM 0.27.1 and its resolved dependencies by wheel SHA-256, and record `pip freeze`; do not alter `/opt/conda/envs/py312`.
- [ ] Prove `AutoConfig` and vLLM recognize `qwen3_5`, then launch the model at the exact shared snapshot in bf16, max model length 32,768, tensor parallel 1, seed 0, thinking disabled by requests, and no model download.
- [ ] Call `/v1/models`, `/tokenize`, and a deterministic two-request chat probe with `return_token_ids=true`; require identical token IDs and text.
- [ ] Before yield, install an idempotent EXIT/INT/TERM trap and independent parent-death watchdog. Every exit path restores and verifies both holder layers; record load time, GPU UUID, versions, command hash, model/tokenizer identity, endpoint-shape receipt, and holder transition evidence.

### Task 8: Deploy SWE server and run the real one-task triad gate

**Files:**
- Runtime-only: `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/triad-eval-20260816/gate/**`, service logs, gate manifest, gate certificate.

- [ ] Re-inventory the sole 1-card job and prove both pod-exec and SSH reachability immediately before launch.
- [ ] Materialize/load only sorted task index 0, build its exact mirror, launch the SWE server through the proven cgroup/quota/rootfs guard with private roots and frozen identities, and verify `/metadata` plus a no-secret client reset/close regression.
- [ ] Run all three gate arms through `PairedRunner.run_task`, grade each immutable prediction with the pinned official grader, and validate one triad, three boolean outcomes, fenced crash/recovery/dedupe, cgroup peak+empty state, hard quota cleanup, and rootfs re-attestation before writing `gate/PASS.json`. These are canonical manifest cells `task_index=0` / lattice `00/10/11`; never rerun them after PASS and never report the three-row gate alone as a benchmark score.
- [ ] If any concrete failure occurs, preserve evidence, repair only that failure, restart only owned services, and rerun the gate from a new private attempt namespace.
- [ ] On PASS, automatically start Task 9 without requesting permission.

### Task 9: Run, monitor, repair, and finish all 1,500 cells

**Files:**
- Runtime-only: `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/triad-eval-20260816/full/**`, heartbeat, attempts, accepted cells, predictions, evidence, official outcomes, timing ledger.

- [ ] Continue the full immutable manifest in its supervised tmux binding from the accepted task-0 triad and official outcomes; verify resume selects exactly the remaining 1,497 cells without duplicate rows.
- [ ] After 10 and 20 real tasks, compute cell/task p50 and p95 durations, failure/retry rates, and measured ETA; update supervisor state without pausing the run.
- [ ] Continuously audit PID liveness, heartbeat freshness, GPU process identity, Docker residue, active SWE slots, completed-cell monotonicity, and official outcome count; repair concrete failures and resume automatically.
- [ ] Do not change model, decoding, budgets, order, images, grader, or any arm setting for throughput.
- [ ] Continue until exactly 1,500 accepted cells and exactly 500 official outcomes per arm exist.

### Task 10: Final verification, review, and cleanup

**Files:**
- Runtime-only: frozen results, official summary, diagnostics, privacy report, command/exit ledger, cleanup receipt.
- Review contract: `/Users/luolirui.1/Projects/amg-paired-eval-20260815/contracts/subagents/amg-swebench-triad-closure-reviewer.md`

- [ ] Rebuild and verify canonical 1,500-row results; verify three 500-row prediction ledgers and official outcome ledgers.
- [ ] Produce raw scores, three contrasts, timeout/failure/action/compaction/memory diagnostics, and setup/gate/task timing distributions.
- [ ] Run protected-value/path/key privacy scans and generate a leak-safe public summary.
- [ ] Re-run protected rollout zero-diff, active-source domain/arm grep, source cleanliness, exact refs, and treatment-excluded identity checks.
- [ ] Dispatch one independent read-only closure reviewer under the prewritten contract; resolve all Critical/Important findings and rerun verification.
- [ ] Stop only owned model/SWE/Docker-child processes, remove owned containers/images/scratch, restore GPU+CPU holder mode, and prove residue 0 while retaining the allocation.
- [ ] Push the isolated deployment branch only after tests and review, freeze commit SHAs, update supervision state complete, and report the official result.

## Literal verification and launch commands

From `/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/worktrees/AgentGym-RL-amg-swebench-triad-deploy-20260816` on the assigned pod:

```bash
PYTHONPATH=AgentGym-RL/scripts/agentmemory python3 -m unittest discover -v \
  -s AgentGym-RL/tests/agentmemory -p 'test_swebench_triad_eval_*.py'
PYTHONPATH=AgentGym-RL/scripts/agentmemory python3 -m unittest discover -v \
  -s AgentGym-RL/tests/agentmemory -p 'test_paired_eval_*.py'
PYTHONPATH=AgentGym/agentenv-swebench-verified:AgentGym/agentenv-swesmith:AgentGym/agentenv-agentmemory:AgentGym/agentenv \
  python3 -m unittest discover -v -s AgentGym/agentenv-swebench-verified/tests -p 'test_*.py'
PYTHONPATH=AgentGym-RL/scripts/agentmemory python3 -m swebench_triad_eval.cli preflight \
  --config /home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/triad-eval-20260816/control/run-config.json
PYTHONPATH=AgentGym-RL/scripts/agentmemory python3 -m swebench_triad_eval.cli gate \
  --config /home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/external-evals/swebench-verified-v4.1.0/triad-eval-20260816/control/run-config.json \
  --auto-run-full
```

`preflight` must exit 0 and atomically write `control/preflight-PASS.json`; `gate --auto-run-full` must write `gate/PASS.json` only after all three real cells and sandbox negative probes pass, then exec the resumable full driver. For each cell the driver records the exact v4.1.0 invocation in `full/command-exit-ledger.jsonl`; the command is `venv/bin/python -m swebench.harness.run_evaluation --dataset_name <frozen-jsonl> --split test --predictions_path <immutable-one-row-prediction> --max_workers 1 --timeout 1800 --force_rebuild false --cache_level instance --clean false --namespace swebench --run_id <cell-arm-generation-prediction-digest>`, with all bracketed values resolved to absolute paths/digests in that ledger before execution. Expected durable receipts are one accepted endpoint row, one prediction, one queued handoff, and one boolean official outcome per manifest cell; queued handoff alone never satisfies completion.
