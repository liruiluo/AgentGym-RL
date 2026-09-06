# r112 Runtime / Holder Safety Implementation Plan

> **For agentic workers:** Execute inline in this uniquely owned successor worktree; no parallel editor or subagent may touch this branch.

**Goal:** Close the r112 launch-package restart, release-transaction, business-lease TOCTOU, and exact-byte test-provenance findings without changing environment or learning semantics.

**Architecture:** Keep the existing fallback supervisor and marker transaction ownership. Bind the platform wrapper to a restart-stable Python, make resume finalization idempotent across every crash boundary, and make the append-only business-lease payload—not a second path scan—the authority for live-process drain. Add a reusable fail-closed provenance harness around arbitrary CPU test commands.

**Tech Stack:** Bash, Python 3.12 standard library, `unittest`, git CLI.

---

### Task 1: Restart-stable watchdog bootstrap

**Files:**
- Modify: `async_plugins/scripts/amg_fallback_supervisor_watchdog.sh`
- Modify: `async_plugins/tests/test_fallback_supervisor.py`

- [x] Add a failing clean-environment wrapper fixture with `PYTHON` absent.
- [x] Default to `/opt/conda/envs/py312/bin/python3`, canonicalize it, and preserve the pinned binary digest.
- [x] Prove the test-selected stable path works without the volatile training runtime.

### Task 2: Crash-idempotent release transaction

**Files:**
- Modify: `async_plugins/agentmemorygym_verl/fallback_supervisor.py`
- Modify: `async_plugins/tests/test_fallback_supervisor.py`

- [x] Add failure injection at authorization, marker removal, request consumption, and holder attestation.
- [x] Recover a persisted no-marker `pending_resume` by consuming its exact bound request before holder publication.
- [x] Ensure owner-visible `holding` implies marker/request/pending state are all complete.

### Task 3: Bound-payload business drain authority

**Files:**
- Modify: `async_plugins/agentmemorygym_verl/orchestrator_lifecycle.py`
- Modify: `async_plugins/tests/test_orchestrator_lifecycle.py`

- [x] Add validation-to-live-use unlink and replacement barrier fixtures.
- [x] Read identity bindings with one file descriptor.
- [x] Audit each registered PID/start-ticks/process-group from the saved append-only payload; retain directory scan only for unexpected paths.

### Task 4: Exact-byte regression provenance harness

**Files:**
- Create: `async_plugins/scripts/run_exact_source_tests.py`
- Create: `async_plugins/tests/test_exact_source_tests.py`
- Modify: `async_plugins/docs/sao_compactionrl_r112_ownership_matrix.md`

- [x] Add CLI accepting repo roots, expected HEADs, selected-file manifests, interpreter, command, log, and receipt.
- [x] Fail closed on dirty trees, pre-hash mismatch, command failure, or post-run mutation.
- [x] Atomically publish a receipt containing command/env allowlist/exit/pre/post identities.
- [x] Exercise PASS and source-mutation failure fixtures.

### Task 5: Verification and delivery

- [x] Run focused Mac and JD-devbox tests; catalog the known full-suite dependency/platform exclusions rather than treating them as product failures.
- [x] Run `bash -n`, `compileall`, and pre-commit harness fixtures; the clean-HEAD provenance receipt is emitted after commit.
- [ ] Record exact commands, counts, hashes, and closure matrix in the implementation report.
- [ ] Audit the diff for algorithm/environment isolation, commit, push, and verify clean remote-aligned HEAD.
- [ ] Freeze a successor deploy runner that passes stable `/opt` Python to fallback `migrate` / `formal` while retaining the source-locked `/dev/shm` Python inside the training command; validate it on the target Pod and obtain independent launch review.
