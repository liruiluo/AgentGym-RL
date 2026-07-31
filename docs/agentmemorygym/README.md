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

The training-only
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
