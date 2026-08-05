# AgentMemoryGym Codex workspace memory v2

## 1. Decision

The canonical AgentMemoryGym v2 memory surface is an episode-scoped persistent
workspace operated through the same two tools used by Codex:

- `shell_command`: run an ordinary shell command in an isolated workspace.
- `apply_patch`: apply a Codex `*** Begin Patch` transaction to workspace files.

It does not expose dedicated `ADD / RETRIEVE / UPDATE / DELETE / SUMMARY /
FILTER` actions. It also does not expose a synthetic `Read / Write / Edit /
Grep / Glob` tool family. Reading, searching, organizing, summarizing, and
maintaining notes are policy choices expressed through normal shell utilities
or `apply_patch`.

The initial WebShop v2 surface is:

```text
agentmemory_webshop_procedural_natural_chain_filesystem_v2
```

The legacy control remains byte-compatible at:

```text
agentmemory_webshop_procedural_natural_chain_train_v1
```

No existing surface ID may silently change meaning.

## 2. Policy-visible contract

One reply executes exactly one native domain action or one workspace action.
WebShop keeps its native browser actions:

```text
search[keywords]
click[current clickable value]
click[Buy Now]
```

The Codex workspace actions are:

```text
shell_command {"command":"rg -n 'finish|color' .","workdir":".","timeout_ms":10000}
apply_patch
*** Begin Patch
*** Add File: .agent_memory/MEMORY.md
+selected finish: black
*** End Patch
```

`shell_command` requires `command`. `workdir` and `timeout_ms` are optional.
The command is normal shell, not a semantic whitelist. Common uses include
`rg`, `grep`, `cat`, `sed`, `find`, directory creation, and small local scripts.
Paths and workdirs are workspace-relative.

`apply_patch` accepts one multiline Codex patch. It supports Add File, Update
File, Delete File, and Move to. The complete patch is parsed and applied to a
staging tree; a parse error, context mismatch, or quota failure leaves the
workspace unchanged.

Workspace actions receive zero task reward. The prompt does not prescribe a
filename, note schema, write cadence, read cadence, or memory lifecycle.

## 3. Persistence and isolation

- Reset creates a fresh private workspace for one episode.
- The same workspace persists across every shopping session in that episode.
- Native page state and current-session traces may reset; workspace files do not.
- Reset or close destroys the previous episode workspace.
- Workspaces, audit events, and files cannot cross environment IDs or episodes.
- The policy sees logical relative paths, never a host path.
- The workspace begins empty. The environment never preloads task answers or a
  task-specific note template.

## 4. Shell safety boundary

Formal `shell_command` execution uses `linux_namespace_chroot_tmpfs_v1`:

- fresh mount, PID, network, IPC, and UTS namespaces;
- no routes in the private network namespace and no host network access;
- a minimal chroot with read-only system binaries and libraries;
- a bounded tmpfs workspace copied in before execution and validated before
  atomic copy-out;
- a host-wide exclusively leased high UID/GID for each concurrent command,
  `no_new_privileges`, and empty inheritable, permitted, effective, bounding,
  and ambient capability sets;
- wall-time, CPU, address-space, process, open-file, stdout, stderr, file,
  directory, inode, and total-storage limits;
- process-group plus PID-namespace teardown on timeout;
- stdout/stderr pipes that retain at most their declared policy-visible byte
  limits instead of first writing larger host files;
- a pinned static `rg` exposed at `/tools/rg`, with launcher-declared SHA256
  checked before preflight and recorded with its version. The full binary is
  hashed once at startup; its device, inode, mode, size, mtime, and ctime are
  revalidated before every command so drift fails closed without repeatedly
  reading the complete binary from shared storage.

The sandbox launcher must run on Linux as root only to construct namespaces and
then drop privileges. A missing namespace primitive, missing pinned `rg`, or
failed preflight makes the surface unavailable. It never falls back to host
shell execution.

The current shared runtime pin is the official ripgrep 15.1.0
`x86_64-unknown-linux-musl` release:

```text
/home/ai-jingyan-train/luolirui.1/post-train/agentmemorygym-rl-workspace/runtime/tools/ripgrep/15.1.0-x86_64-unknown-linux-musl/rg
SHA256 ebeaf56f8a25e102e9419933423738b3a2a613a444fd749d695e15eba53f71f2
source https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep-15.1.0-x86_64-unknown-linux-musl.tar.gz
```

Every filesystem-v2 server launch must pass both
`--workspace-rg-binary <path>` and `--workspace-rg-sha256 <sha256>`. Passing a
real executable with a different digest fails before namespace preflight. The
formal host must provide `unshare`, `mount`, `chroot`, `capsh`, `setpriv`, and
`prlimit`; a missing primitive is a hard startup error.

After each command, the copied-out tree is rejected if it contains a symlink,
hard link, device, socket, FIFO, non-regular object, oversized file, excessive
path, or quota overflow. The host-side episode directory is replaced only after
the complete staged tree passes validation.

## 5. `apply_patch` safety boundary

`apply_patch` does not invoke a shell or the host `patch` program. The harness
parses its closed Codex patch grammar, normalizes every relative path, rejects
traversal and unsafe file types, applies all operations to a staging copy, and
validates the resulting tree before atomic installation. A multi-file patch is
all-or-nothing.

## 6. Audit and metric contract

Every workspace action records a background event containing:

- episode, environment step, session index, operation, and request digest;
- workspace tree SHA256 before and after;
- exact added, modified, deleted, and directory changes;
- shell exit code, timing, timeout, bounded stdout/stderr digests and truncation;
- patch operation count, changed paths, and transactional status.

The policy receives normal tool output, not this audit ledger. Formal metrics
recompute snapshots, tree hashes, diffs, event ordering, and cross-step
continuity. They reject legacy five-tool events and dedicated memory events on
the v2 surface.

Shell auditing cannot prove which file influenced model reasoning without
syscall-level tracing. The strongest operation-only diagnostic is therefore a
candidate chain:

```text
auditable workspace write
-> correct source-session purchase
-> written version still present in a later session
-> later shell_command
-> correct later-session purchase
```

This chain is not reported as exact retrieval or causal memory use. The legacy
`functional_memory_chain_count` remains reserved for its dedicated-memory-API
evidence contract and is not incremented by workspace candidates.

## 7. Causal capability gate

Memory capability requires a frozen four-arm intervention with identical task
observations and workspace layout where applicable:

- `correct`: the source episode's correct workspace contents;
- `blank`: an empty workspace;
- `swapped`: the paired counterfactual's workspace contents;
- `no_workspace`: both workspace tools unavailable.

The model must change the dependent decision in the direction predicted by the
workspace intervention. Operation counts, scripted actions, metadata, and
candidate chains are plumbing or behavior evidence only.

The intervention server is a separate `intervention_eval` role. Startup
requires a private token file and a frozen runtime source ID. The token is sent
only in the `X-AgentMemory-Intervention-Token` header. Metadata exposes its
SHA-256, never the token. Two authenticated evaluator-only endpoints are
available:

- `POST /workspace-export`: export the exact policy-authored file tree,
  including file bytes, after the first correct source-session purchase.
- `POST /workspace-intervention`: atomically install one frozen causal arm at
  that same boundary.

Neither call is a policy action. Both are absent from the model prompt, receive
zero reward, and do not enter the policy tool ledger. Exported bytes are saved
as evaluator evidence and are never appended to the model observation.

`eval_filesystem_causal_v2.py` implements the real-model protocol:

1. Sample target and exact counterfactual-pair source sessions independently.
2. Require each policy to reach session 1 with a non-empty authored workspace.
3. Create four fresh target environments and replay the target's exact submitted
   source actions in each one.
4. Require exact visible observations, native-state projections, and exported
   workspace bytes to match the target source run before intervention. The
   comparison normalizes only the shell wrapper's measured `Wall time`, which
   is nondeterministic across exact action replays; persisted raw responses
   retain the original timing values.
5. Install the four arms. `correct`, `blank`, and `swapped` retain the identical
   enabled prompt. `no_workspace` receives a dedicated prompt permitting only
   native WebShop actions and explicitly declaring both workspace tools absent.
6. Resample only the dependent sessions and save exact prompts, token IDs,
   actions, environment responses, exported files, purchases, and outcomes.

The paired gate uses `temperature=0`; stochastic decoding would mix memory
effects with independent sampling noise. A typical invocation is:

```bash
python3 scripts/agentmemory/eval_filesystem_causal_v2.py \
  --env-url http://127.0.0.1:PORT \
  --model-url http://127.0.0.1:MODEL_PORT/v1 \
  --model MODEL_ID \
  --indices 0-15 \
  --max-policy-turns 56 \
  --intervention-token-file /private/path/intervention.token \
  --output-dir /shared/evaluations/filesystem-causal-v2
```

An orbit is ineligible rather than silently repaired when either source policy
cannot reach the boundary, writes no workspace, or produces a source action
sequence whose replay is not byte-for-byte reproducible. Runtime metadata,
authentication, state hashes, or intervention-contract mismatches fail the
whole run closed.

Recency override adds the `stale` arm and requires the target source to be the
`flip` member while its paired source is `stay`. Its generator emits each pair
in `(stay, flip)` order, so direct evaluator indices for flip targets are odd
(`1,3,5,...`). The evaluator rejects the opposite orientation before accepting
an orbit.

## 8. Migration gate

1. Validate parser, patch transaction, persistence, isolation, quotas, and audit continuity.
2. Run host-escape, network, privilege, process cleanup, timeout, output, and storage gates.
3. Validate prompt, rollout serialization, formal metrics, and eval conversion.
4. Run scripted native correct/blank/swapped/no-workspace smoke for plumbing.
5. Run a real model four-arm gate and inspect exact prompts, actions, files, and outcomes.
6. Only then start a formal RL run and held-out capability evaluation.

Scripted smoke passing does not establish model memory ability, and engineering
completion does not complete the broader AgentMemoryGym training objective.
