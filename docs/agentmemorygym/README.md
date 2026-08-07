# AgentMemoryGym Repository Boundary

This source repository keeps reusable AgentMemoryGym runtime code, contract
tests, and stable developer documentation only.

Run directories, checkpoints, frozen launch inputs, manifests, model I/O,
diagnostic reports, Notion mirrors, and historical experiment plans belong in
the external AgentMemoryGym workspace. Retired source-tree documents from
2026-07-01 through 2026-07-02 are preserved there under
`archive/retired-source-docs-20260721/` with a SHA-256 manifest.

Current model-facing training and evaluation must use the native MemoryArena
WebShop runtime. The retired SQLite/FTS surrogate and scripted-search evidence
must not be restored as an experiment path.

## Procedural shopping stream

The legacy training/control surface
`agentmemory_webshop_procedural_natural_chain_train_v1` surface generates and
rule-verifies a task when the environment receives its integer reset index. It
does not require a materialized JSON task file, per-item human review, or an
LLM judge. The task text identifies approved products by complete native titles;
normal WebShop search results still use their native `click[ASIN]` handles.

Enable the matching lazy index source in PPO overrides:

```text
data.train_file=null
data.shuffle=false
data.procedural_index.enabled=true
data.procedural_index.provider_mode=reseeded_stream
data.procedural_index.task_count=<positive-even-window-size>
data.procedural_index.start_index=0
```

The environment server's provider mode and task count must match these values.
`task_count` is the number of rows yielded by one dataloader pass, not the size
of the generated task universe. The stateful sampler advances to the next
contiguous index window on every pass and its cursor is serialized in the
normal trainer checkpoint. Keep the start index and train batch size even so
the two branches of every counterfactual pair stay adjacent, and make
`task_count` divisible by the train batch size so `drop_last` cannot consume an
untrained tail. The trainer fails closed on shuffle, invalid batch geometry, or
mismatched server metadata. PPO accepts only `reseeded_stream`; `fixed_window`
is reserved for bounded generation and evaluation rather than training.

The v2 mainline keeps the same proof-carrying task provider and native WebShop
semantics but replaces dedicated memory APIs with an episode-scoped persistent
workspace:

```text
agentmemory_webshop_procedural_natural_chain_filesystem_v2
```

The policy operates ordinary workspace files through Codex-style
`shell_command` and `apply_patch`. Workspace actions receive zero task reward;
the shell runs in a networkless, resource-bounded Linux namespace and exposes
no host path, LTM inventory, or hidden memory-management API. The v1 and v2
surface IDs remain separate so historical results are reproducible. The
complete runtime, safety, evidence, and migration contract is in
[`natural-filesystem-memory-v2.md`](natural-filesystem-memory-v2.md).

The v2 episode contains six **native, manually/sessionized WebShop sessions**;
it is not a continuous full-history conversation. A successful `click[Buy Now]`
advances the native task and clears the prior session's page, cart/budget,
active context, and action/observation transcript. Only the policy workspace
files persist. An optional policy-authored `session_handoff` exists solely to
re-anchor the policy to a file it created in that persistent workspace. Its
output may identify a workspace-relative path or a generic command for finding
and reading that file; it must not carry shopping facts, product choices,
progress, prior actions/observations, or other semantic task state. The harness
never writes the locator, and the handoff never calls the native server. Never
report this WebShop mechanism as context compaction, continuous-history reuse,
or a naturally long trajectory.

#### Terminology invariant

Keep these three mechanisms separate in code, logs, and reports:

| Mechanism | When it happens | What persists | What it is not |
| --- | --- | --- | --- |
| Native WebShop session reset | A correct `click[Buy Now]` advances the bundled task | Native server state is reset; policy workspace files remain | Continuous context compaction |
| Policy handoff | An optional policy step at that reset boundary | Only a model-authored locator for a persistent workspace file | A task-state summary, harness-written content, or old transcript |
| Context compaction | A naturally continuous task reaches its context budget | Model-authored summary tokens plus the persistent workspace | A WebShop session transition |

The shared serialized row schema currently uses
`compaction_mode=webshop_session_handoff` for the second row and
`compaction_mode=context_limit` for the third row. The `compaction_mode` field
name is a compatibility detail, not a semantic classification. Human-facing
WebShop code comments, metrics, launchers, TODO entries, and reports must call
the second mechanism `session_handoff`, never WebShop compaction, and must not
count it as continuous-history reuse. SWE-smith, LiteResearcher, and MLE use
context compaction only inside their native continuous episodes.

The raw policy completion for `session_handoff` is untrusted. Runtime must parse
it fail closed and may forward only one workspace-relative path, one read-only
workspace discovery/read command (`cat`, `rg`, `grep`, `find`, `ls`, `head`,
`tail`, or `sed`), or an explicit no-locator result. Absolute or `file://`
paths, `..` traversal, host paths, prose summaries, multiline or multi-item
outputs, write commands, and shell operators are invalid. An invalid completion
remains the sampled policy row, token/logprob evidence, and RL action, but its
content is not copied into the next session; that prompt receives only the
fresh native observation. The runtime must record the raw digest, parse verdict,
locator kind, and forwarded digest. A gate that merely packs the handoff row or
observes a non-empty string proves PPO plumbing only, not locator-only semantic
closure.

For this stream, `data.pt` contains only a versioned sampler cursor plus a
canonical identity for the index source, complete server metadata, and training
geometry. Resume first builds and validates the current dataset and environment
client, then restores the cursor only when that identity matches exactly. It
rejects a legacy serialized `DataLoader` or any changed generator seed,
provider contract, prompt/reward metadata, task count, or batch geometry.

Each generated PPO batch must contain contiguous even/odd counterfactual pairs.
After rollout, the trainer also requires at least one trainable action row for
every `(source parent, rollout replica)` requested from that batch. If rollout
infrastructure excludes a parent or replica, the step raises before the PPO
update and before a new stream cursor checkpoint can be written; partial batches
are never silently trained or durably skipped.
