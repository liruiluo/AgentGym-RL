# SWE-smith Episode Contract

Status: compaction revision required before the first AgentMemoryGym SWE-smith PPO update.

## Policy surface

- The policy receives the issue text and a persistent repository workspace.
- The only executable tools are Codex-style `shell_command` and `apply_patch`.
- A normal final response submits the current workspace for grading.
- Tool actions receive zero reward. The terminal reward is `1` only when the
  hidden verifier reports a fully resolved instance; every other terminal
  outcome receives `0`.
- Conversation history is continuous. The harness does not insert artificial
  sessions or a dedicated memory API.

## Context lifecycle

- The persistent repository workspace is the policy's external long-term
  state. The policy maintains any durable notes with the same ordinary
  `shell_command` and `apply_patch` tools used for coding.
- Near the configured context limit, the harness may count tokens and issue a
  neutral compaction request with a bounded response budget. The current actor
  policy itself writes the compact continuation summary. No separate model,
  hidden state, gold patch, verifier evidence, harness-authored summary, or
  harness-generated memory-file inventory may supply its semantic content.
- Sampled compaction-summary tokens are retained as trainable policy actions
  under the exact prompt that generated them and receive downstream terminal
  task credit. They are not free observations or detached labels.
- The neutral compaction request, prior dialogue, and environment observations
  are prompt tokens with loss mask `0`. The policy-authored summary is the
  response of its own trajectory row with response mask `1`; action-time GAE
  propagates later terminal task reward to that row. When the same summary is
  carried into the next prompt, that later copy is masked as context without
  detaching the original sampled row.
- The hard-capacity trigger is deterministic harness control, so this version
  does not learn *when* to compact. It learns *what continuation state to
  write*. Learning the trigger would require a separately sampled policy tool
  call and is outside the first native SWE-smith contract.
- A compaction consumes one ordinary episode step, exactly like a
  `shell_command`, `apply_patch`, or final response. The existing
  `max_policy_turns` budget applies uniformly; there is no task-specific
  compaction-count limit. Evidence separately records whether the step called
  the native environment, but that distinction does not make compaction free.
- After compaction, the harness preserves only immutable system/tool/task
  framing, the model-authored continuation summary, and a neutral continuation
  marker. The repository workspace is unchanged. The policy must put any
  external-note paths, current state, or next actions that it needs into its
  own summary and recover details with normal tools. Omitted facts receive no
  semantic fallback.
- Silent token truncation and terminating an otherwise valid episode merely
  because raw history reached the context limit are forbidden in formal
  training. Every compaction boundary records exact pre/post prompt digests,
  sampled summary token IDs and logprobs, token counts, and subsequent file
  reads for audit.

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

- The gate must first prove one real model-authored compaction and continued
  workspace use without silent truncation. A 32K no-compaction run is only an
  integration diagnostic and cannot satisfy the long-horizon training contract.

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
