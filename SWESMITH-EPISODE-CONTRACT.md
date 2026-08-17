# SWE-smith Episode Contract

Status: active AgentMemoryGym SWE-smith RL contract.

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
| Plain text | No bash action means a format error, never a submission | Plain text, including `final`, is a parser error and leaves the episode running | Semantically equivalent fail-closed behavior |
| Turn exhaustion | `LimitsExceeded` exits with an empty submission; no workspace grade is requested | Horizon terminates with reward `0`, `grade=None`, and no hidden-grader call | Explicitly preserves upstream no-submission failure |
| Budget | Pinned SWE-bench config uses `step_limit: 250` | r4 used a bounded 30-turn curriculum; r5 restores the endpoint's 75-turn training default; held-out native evaluation uses 250 | Deliberate training compute bound |
| Workspace | Upstream submits a patch from a `.git` worktree | AMG keeps a persistent no-`.git` workspace and grades it once on the sentinel | Deliberate runtime adaptation |

The source and contract identities above are exposed in `/metadata` and are
checked by the procedural index and resident endpoint verifier. Any mismatch
is fail-closed before PPO.

## Policy surface

- The policy receives the issue text and a persistent repository workspace.
- The only executable tools are Codex-style `shell_command` and `apply_patch`.
- A successful sentinel shell command submits the current workspace for grading.
- Tool actions receive zero reward. The terminal reward is `1` only when the
  hidden verifier reports a fully resolved instance; every other terminal
  outcome receives `0`.
- Conversation history is continuous. The harness does not insert artificial
  sessions or a dedicated memory API.

## Interaction budget

- Training uses a bounded policy-turn budget. The r4 formal launcher used 30
  turns; r5 uses the endpoint default of 75. `shell_command`, `apply_patch`, the
  sentinel submission, parser errors, and policy-authored context
  compaction each consume exactly one turn. There are no free compactions or
  parser retries, and a successful submission terminates early.
- The initial observation states the exact configured turn budget and that
  compactions consume it, so the policy can verify and submit before an
  otherwise ungraded horizon.
- The configured training value is an AgentMemoryGym compute bound. It is not
  the upstream default. Pinned Mini-SWE-Agent commit
  `a83fcae82d2a08f0ee0c688f9d137b3566c097f8` sets `step_limit: 250` for
  SWE-bench/SWE-smith in
  `src/minisweagent/config/benchmarks/swebench.yaml`.
- Held-out native evaluation uses the upstream 250-turn reference. Reports must
  separate `resolved`, `training_horizon_exhausted`, and
  `upstream_horizon_exhausted` rather than treating a 75-turn timeout as proof
  that the issue is unsolvable.

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

## PPO gate

- The learner runs on all eight B200s in the lane recorded by the frozen
  launcher and read-only inventory at launch time. Pod IPs are runtime
  evidence, never a permanent role assignment. A single-GPU process may
  preheat independent environment dependencies, but it is not a learner gate.
- The learner batch remains 64. The first direct SWE-smith training curriculum
  is sampled from a frozen, diverse, solvable subset of the native train split;
  an eight-instance plumbing panel may validate runtime wiring but must not be
  repeated to stand in for the training distribution.
- The historical `simple_train8` panel is an `oauthlib` plumbing panel.
  `oauthlib` belongs to the frozen held-out side of the repo-disjoint split, so
  that panel is never eligible for PPO training even though its legacy filename
  contains `train`.
- Acceptance requires a completed optimizer update, nonzero actor and critic
  parameter deltas, a verified checkpoint/readback, exact dataset-index routing,
  auditable per-trajectory tool/final/verifier evidence, zero owned Ray/vLLM
  residue after a bounded gate, and restored GPU+CPU holders. Passing the gate
  continues into training rather than falling back to single-GPU PPO.

## Resident endpoint

- One resident HTTP process may serve all learner ranks. Each client obtains a
  distinct slot from `/create`; reset binds that slot to exactly one dataset
  index and one exclusive workspace/UID lease.
- Before PPO launch, the eight-slot verifier must bind indices `0..7`, prove
  unique slot, audit, workspace, and production UID identities, exercise
  isolated write/read actions, close every slot, and attest zero active slots,
  environments, workspaces, and workspace residue.
- Endpoint reuse is allowed only when its source/data/image/runtime fingerprint
  matches the frozen launcher and metadata reports zero active environments.
