# SWE-smith Episode Contract

Status: frozen for the first AgentMemoryGym SWE-smith PPO update.

## Policy surface

- The policy receives the issue text and a persistent repository workspace.
- The only executable tools are Codex-style `shell_command` and `apply_patch`.
- A normal final response submits the current workspace for grading.
- Tool actions receive zero reward. The terminal reward is `1` only when the
  hidden verifier reports a fully resolved instance; every other terminal
  outcome receives `0`.
- Conversation history is continuous. The harness does not insert artificial
  sessions or a dedicated memory API.

## Data boundary

- The server indexes a frozen JSONL dataset by the exact integer supplied in
  `extra_info.index`; opaque instance identifiers are evidence only.
- Policy-visible data contains the problem statement and public repository
  files. The gold patch, expected answer, verifier command, expected F2P/P2P
  statuses, pristine tests, and control paths remain server-private.
- The policy workspace never contains `.git` or hidden verifier artifacts.

## Workspace lifecycle

- Reset materializes the bug source and restored tests exactly once into an
  episode-exclusive directory. Later actions operate on that same directory;
  the repository is not copied once per action.
- Each episode has an exclusive unprivileged identity. Commands execute in a
  fresh mount, network, PID, IPC, and UTS namespace with a minimal read-only
  system root, a direct read-write bind of the episode workspace, a bounded
  tmpfs, resource limits, no capabilities, and no new privileges.
- Timeout and normal command completion terminate the entire command process
  tree before control returns to the environment.
- Reset and close remove only that episode's workspace. They never release or
  restart an accelerator allocation.

## Mutation boundary

- `apply_patch` accepts only the native `*** Begin Patch` grammar.
- It prevalidates every touched path, rejects path escape, duplicate paths,
  symlinks, hardlinks, non-UTF-8 files, oversized files, and quota violations,
  then applies in place with touched-file backups.
- Any failed patch restores all touched paths byte-for-byte and removes newly
  created paths. A successful patch leaves unrelated files and directories
  untouched.
- Shell-created trees are revalidated after every command before another model
  action or terminal grading is accepted.

## Hidden verifier

- Before grading, every F2P and P2P test file is restored from a hidden pristine
  source captured at reset. Test edits can never contribute to reward.
- The verifier runs the full declared F2P set first. It runs the full declared
  P2P set only when F2P passes; P2P is never sampled.
- Test output is parsed by the frozen SWE-smith repository profile and scored
  with `get_eval_tests_report()` plus `get_resolution_status()`.
- Evidence records the typed policy call, parser result, exit code, bounded
  stdout/stderr, timeout/process cleanup, workspace diff, restored test paths,
  complete F2P/P2P status maps, and final resolution status without exposing
  hidden values to the policy.

## First PPO gate

- The learner runs on all eight B200s in `6.5.167.119` from the first real
  optimizer update. A single-GPU process may preheat independent environment
  dependencies, but it is not a learner gate.
- The first batch uses eight source episodes aligned with eight ranks to reduce
  environment concurrency variables while still exercising full eight-rank
  FSDP actor, critic, optimizer, checkpoint, and readback paths.
- Acceptance requires a completed optimizer update, nonzero actor and critic
  parameter deltas, a verified checkpoint/readback, exact dataset-index routing,
  auditable per-trajectory tool/final/verifier evidence, zero owned Ray/vLLM
  residue after a bounded gate, and restored GPU+CPU holders. Passing the gate
  continues into training rather than falling back to single-GPU PPO.
