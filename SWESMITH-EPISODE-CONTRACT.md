# SWE-smith Episode Contract

Status: active AgentMemoryGym SWE-smith RL contract (`filesystem_checkpoint_v3`).

## Pinned upstream parity

The semantic reference is Mini-SWE-Agent repository `SWE-agent/mini-swe-agent`
at commit `a83fcae82d2a08f0ee0c688f9d137b3566c097f8`. SWE-smith repository code
at `9b74ac08118a85c39c356802f7961893af73e07f` supplies the task and image;
Mini-SWE-Agent supplies the interaction and termination contract.

| Surface | Pinned upstream | AMG training adapter | Difference classification |
| --- | --- | --- | --- |
| Action grammar | A response contains a bash tool action; malformed responses are format errors | One sampled turn is exactly one `shell_command` or `apply_patch`; malformed/plain text is a parser error | RL serialization constraint |
| Repair workflow | Inspect, reproduce, make a localized non-test source edit, rerun the reproduction, test edge cases, then submit | The same ordered workflow is stated using the one-action Codex grammar; test/config edits remain forbidden | Semantically preserved with adapted action syntax |
| Submission trigger | A successful shell command whose first stdout line is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` raises `Submitted` | The same sentinel and zero-exit/first-line check are applied after the sandbox command | Semantically identical |
| Submission payload | SWE-bench config emits `echo ... && cat patch.txt`; remaining stdout is the patch | The persistent workspace is graded directly, so the adapter emits the sentinel-only command and records the workspace digest | Deliberate workspace-grade adaptation; not byte-equivalent patch transport |
| Plain text | No bash action means a format error, never a submission | Plain text, including `final`, is a parser error; the turn is penalized and the episode continues while budget remains | Semantically equivalent fail-closed behavior with recoverable RL feedback |
| Turn exhaustion | `LimitsExceeded` exits with an empty submission; no workspace grade is requested | Horizon terminates with reward `0`, `grade=None`, and no hidden-grader call | Explicitly preserves upstream no-submission failure |
| Budget | Pinned SWE-bench config uses `step_limit: 250` | Formal training uses a frozen 30-turn compute curriculum; held-out native evaluation uses 250 | Deliberate training compute bound |
| Workspace | Upstream submits a patch from a `.git` worktree | AMG keeps a persistent no-`.git` workspace and grades it once on the sentinel | Deliberate runtime adaptation |

The source and contract identities above are exposed in `/metadata` and checked
by the procedural index and resident endpoint verifier. Any mismatch is
fail-closed before PPO.

## Policy and reward surface

- The policy receives the issue text and a persistent repository workspace.
- The only executable tools are Codex-style `shell_command` and `apply_patch`.
- A successful sentinel shell command submits the current workspace for grading.
- Accepted nonterminal tool actions, including checkpoint writes and reads,
  receive zero immediate reward. Parser/executor-rejected actions receive the
  frozen `-0.01` immediate reward, consume one turn, and do not terminate early.
- The terminal reward is `1` only when the hidden verifier reports a fully
  resolved instance; other terminal outcomes receive `0` terminal reward.
- Memory behavior has no auxiliary reward. It is trained only through the same
  downstream task return, action-axis GAE, and PPO path as ordinary task actions.

## Interaction budget

- Formal training uses at most 30 policy turns per episode. `shell_command`,
  `apply_patch`, sentinel submission, parser/executor rejection, and each
  policy-authored checkpoint attempt consume exactly one turn. A successful
  submission terminates early.
- The initial observation states the exact configured budget. This is an AMG
  training compute bound, not the upstream default. The pinned upstream config
  uses `step_limit: 250` for SWE-bench/SWE-smith.
- Held-out native evaluation uses the upstream 250-turn reference. Reports must
  separate `resolved`, `training_horizon_exhausted`, and
  `upstream_horizon_exhausted`; update100 must finish before external evaluation.

## Filesystem checkpoint and context replacement

- Context pressure starts a wrapper-owned checkpoint opportunity, not a new
  rollout. The policy still emits an ordinary, logged, trainable
  `shell_command` or `apply_patch` action through the shared task-agnostic loop.
- Reset provisions an empty reserved `.agent_memory/` parent. A collision with
  pre-existing task content fails closed. The reserved directory and
  `.agent_memory/CONTINUATION.md` do not count as task-source modification.
- A valid checkpoint action must create or update the regular file
  `.agent_memory/CONTINUATION.md`, leave it non-empty and at most 8 KiB, and
  produce a receipt for the exact path, digest, size, and completed action.
- Only a valid changed checkpoint receipt authorizes the wrapper's mechanical
  `replace_messages`. The replacement contains neither sampled policy text nor
  native tool output; both remain in the immutable rollout ledger. The compact
  successor states the exact remaining-turn budget and asks the policy to read
  the file, then execute its saved next action rather than restart broad search.
- A failed checkpoint action is retained as an ordinary action plus native
  observation with `append_observation`; workspace mutations are not hidden.
  The wrapper allows at most two attempts for one opportunity. The first attempt
  starts only with capacity for two writes, one read, and four later task turns;
  a retry starts only with capacity for one write, one read, and four later task
  turns. When that capacity or retry budget is absent, normal task interaction
  continues without replacement.
- The mechanism learns checkpoint content, later reads, and downstream use. The
  wrapper-owned timing and `replace_messages` operation are mechanical and have
  no policy tokens or separate gradient.

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
- A submission whose only workspace change is the reserved checkpoint is a
  no-source-change failure and skips the expensive hidden grader.

## Hidden verifier

- Before grading, every declared F2P and P2P test file is restored from the
  hidden pristine source captured at reset. Test edits can never contribute to
  reward.
- The endpoint executes the frozen profile's single full official command once.
  That command covers the complete declared F2P and P2P set; the endpoint does
  not run a separate F2P phase and then repeat those tests in a full phase.
- Test output is parsed by the frozen SWE-smith repository profile and scored
  with `get_eval_tests_report()` plus `get_resolution_status()`.
- Metadata records `grader_execution_contract`, `grader_phase_count=1`, and the
  effective timeout. Evidence records the typed policy call, parser result,
  exit code, bounded stdout/stderr, timeout/process cleanup, workspace diff,
  restored test paths, complete status maps, and final resolution status without
  exposing hidden values to the policy.

## PPO and evidence gate

- A formal run uses all eight accelerators in its declared allocation; a
  one-rank or masked-GPU run inside an eight-card allocation is forbidden.
- The learner batch is 64. Formal lineages start from pristine Qwen3.5-4B
  update0, not a checkpoint already reinforced under an earlier wrapper.
- Shared policy sampling, rollout, PPO, action-axis GAE, reward, the frozen
  30-turn task board, and no-thinking setting remain unchanged by this wrapper
  repair.
- Acceptance requires a completed optimizer update, nonzero actor and critic
  parameter deltas, checkpoint readback, exact dataset-index routing, auditable
  per-trajectory action/final/verifier evidence, zero owned Ray/vLLM residue
  after a bounded gate, and restored GPU+CPU holders.
- Every milestone reports pristine update1 separately and includes real cases:
  native success, ordinary failure, failed checkpoint, and the full
  `write -> replace -> read -> edit/test/submit -> terminal reward` chain when
  present. Exact `cat` reads, broader attested same-file reads, and path mentions
  without a read are separate metrics.

## Resident endpoint

- One resident HTTP process may serve all learner ranks on same-Pod loopback.
  Each client obtains a distinct slot from `/create`; reset binds that slot to
  exactly one dataset index and one exclusive workspace/UID lease.
- Before PPO launch, the eight-slot verifier must bind indices `0..7`, prove
  unique slot, audit, workspace, and production UID identities, exercise
  isolated write/read actions, close every slot, and attest zero active slots,
  environments, workspaces, and workspace residue.
- Endpoint reuse is allowed only when source/data/image/runtime fingerprints
  match the frozen launcher and metadata reports zero active environments.
