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
