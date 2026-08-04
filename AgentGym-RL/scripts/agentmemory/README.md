# AgentMemoryGym Evidence Eval

## Codex filesystem SFT bootstrap

`generate_filesystem_sft_v1.py` creates executed demonstrations for the
`agentmemory_webshop_procedural_natural_chain_filesystem_v2` surface.  It uses
the real WebShop backend and the real namespace-isolated Codex workspace, so a
record is admitted only when the submitted `shell_command` or `apply_patch`
has an authoritative event with matching request bytes, before/after tree
hashes, and a verified native purchase trajectory.  The generator keeps the
1.18M-item catalog and Lucene searcher alive for the entire process; do not
launch it once per task.

The output is a JSON array plus a manifest.  Each task contributes exactly 28
supervised actions (six searches, twelve native clicks, five shell reads, and
five patches), and the manifest records the source commits, provider/catalog
fingerprints, action counts, and record-hash sequence.  Dataset and manifest
publication is rolled back if either side cannot be authenticated.

The SFT adapter uses `data_mode: agent_action_v1`.  The policy-visible input is
the canonical system prompt plus the current environment observation; the
loss mask covers only the assistant action and `<|im_end|>`, never execution
receipts, private ASINs, or verifier fields.  Run the tokenizer-equivalence
and tiny-overfit gates before using the resulting checkpoint for PPO.

Example configuration override:

```bash
python3 scripts/agentmemory/generate_filesystem_sft_v1.py \
  --memoryarena-root /path/to/MemoryArena \
  --memoryarena-base-commit <sha> \
  --items-file /path/to/items_shuffle.json \
  --attributes-file /path/to/attributes.json \
  --search-root /path/to/lucene \
  --java-home /path/to/java \
  --lucene-index-manifest /path/to/lucene-manifest.json \
  --product-pool /path/to/certified-product-pool.json \
  --product-pool-file-sha256 <sha256> \
  --orbit-count 64 \
  --workspace-rg-binary /path/to/rg \
  --workspace-rg-sha256 <sha256> \
  --expected-outer-source-commit <sha> \
  --expected-agentgym-source-commit <sha> \
  --output-json /path/to/filesystem-sft.json \
  --manifest-json /path/to/filesystem-sft.manifest.json
```

For training, set `data.data_mode=agent_action_v1`, point
`data.train_files` at the generated JSON array, and keep `data.truncation`
large enough that the complete action remains supervised.  A source worktree
must be clean and pinned to the commits recorded in the manifest.

`attest_effective_memory_prompt.py` records the exact rollout system prompt,
its SHA-256, and the effective memory/LTM modes. Formal MemoryChain launchers
should pass `--require-lifecycle-sop` when their scientific contract requires
explicit `ADD`-before-buy and later-session `RETRIEVE` timing.

`eval_v3_openai.py` runs a small, auditable behavior evaluation against an
already-running AgentMemory HTTP server and an OpenAI-compatible vLLM server.
It does not import the training stack and has no `torch` dependency.

The driver supports all AgentMemory v3 domain surfaces (Travel, formal
reasoning, and BrowseComp+), the native WebShop v2 surface, and the non-paper
`agentmemory_webshop_procedural_natural_chain_filesystem_v2` evidence surface.
For every policy turn it:

1. Reads the canonical v3 `system_prompt` from `/metadata`; WebShop v2 surfaces
   resolve the exact rollout prompt from `verl/workers/rollout/schemas.py` and
   reject any conflicting server-provided prompt.
2. Builds exactly two messages: `system` plus the latest `user` observation.
   Prior assistant/environment messages are not sent to the model.
3. Calls the model at `/v1/chat/completions`.
4. Calls the vLLM `/tokenize` endpoint for the same messages, recording the
   returned integer IDs and a SHA-256 hash.  If the endpoint or IDs are not
   available, the run fails closed instead of fabricating token evidence.
5. Sends the sampled text to the environment and records both environment
   info payloads, action, reward, reward components, phase progress, `done`,
   and `episode_success`.

Both `/tokenize` and `/chat/completions` receive the same
`chat_template_kwargs.enable_thinking` value. Native thinking is explicitly
disabled by default; pass `--enable-thinking` when evaluating a policy trained
or served with native thinking enabled. Record this choice in the run manifest
and keep it fixed across compared checkpoints.

The normal OpenAI response does not expose completion token IDs.  Accordingly
`response_token_ids_exact` is `false` unless the response carries an explicit
integer ID sequence; response text is never locally tokenized and labeled
exact.

## Invocation

```bash
python3 AgentGym-RL/scripts/agentmemory/eval_v3_openai.py \
  --env-url http://127.0.0.1:8201 \
  --model-url http://127.0.0.1:8100/v1 \
  --model Qwen3-4B-Instruct-2507 \
  --indices 0,2-4 \
  --output-dir /path/to/eval-output
```

`--model-url` may include `/v1` or omit it.  An optional bearer token can be
passed with `--api-key`; it is sent only to model/tokenize endpoints and never
to the environment server.  No key is sent by default.  Each run writes
`manifest.json` and one `episode_<index>.json` under `--output-dir`.  The
manifest includes provenance metadata, exactness flags, aggregate return /
success / timeout counts, and `final_phase_progress_distribution` (for example
`0/6` through `6/6`) so every diagnostic reports how many trajectories stop
at each task phase.

`--indices` names the zero-based dataset positions sent to environment reset.
The Travel Planner test set therefore uses `0..269`; its frozen upstream source
IDs `1..270` are recorded separately in terminal evidence. Formal-reasoning and
progressive-search resets are also zero-based positions.

The per-episode turn cap defaults to the environment's attested `max_steps`
(and to 56 for legacy WebShop). This matters for Progressive Search: its frozen
rows contain up to 16 phases, and the runtime reserves the full native
per-phase quotas plus explicit memory-action headroom. Use
`--max-policy-turns` only for a deliberately shorter diagnostic.

The evaluator is a behavior/evidence smoke harness.  It does not invoke the
outer formal PPO validator and does not claim sampled response-token exactness.

For the natural-filesystem surface, `/metadata` must attest the exact
Codex `shell_command/apply_patch` contract, episode-scoped cross-session
persistence, namespace-isolated networkless shell, no host-path access,
positive workspace quotas, and zero workspace-action or memory-specific
shaping. The evaluator recomputes each workspace manifest's tree SHA-256,
before/after diff, and contiguous audit ledger. Its diagnostic candidate chain
is:

```text
source-session workspace write(path, content_sha256)
-> correct source-session BUY
-> written version remains present at a later-session shell_command
-> correct later-session BUY
```

Shell audit does not establish that a specific file was read or influenced the
decision, so this chain is not counted as `functional_memory_chain_count`.
Workspace-operation counts and this temporal chain are diagnostic behavior
evidence, not a causal memory-capability result. A capability claim additionally
requires the frozen `correct`, `blank`, `swapped`, and `no_workspace`
intervention arms. This surface lives in `EVIDENCE_SURFACE_REGISTRY`, not
`PAPER_SURFACE_REGISTRY`, and always reports `paper_macro_eligible=false`.

Run those four arms with `eval_filesystem_causal_v2.py` against a dedicated
`intervention_eval` environment service. The driver first samples policy-owned
target and paired source workspaces, exports their exact files through the
authenticated evaluator control plane, replays the exact target source actions
into four fresh environments, and only then installs the interventions. It
resamples dependent sessions at `temperature=0`. The private intervention token
is supplied by `--intervention-token-file`; only its SHA-256 appears in runtime
metadata or saved evidence. The `no_workspace` arm uses a separate canonical
system prompt that permits only native WebShop actions, while the other three
arms use the byte-identical enabled Codex prompt.

## Metric Contract

`episode_success` is the only success/pass field.  A positive shaped reward,
an accepted intermediate action, or a nonzero return is not a successful
episode.  Missing or non-boolean `episode_success` is an evaluation error and
the run fails closed.  This is an evidence-integrity check: it does not infer,
rewrite, or substitute a success result for the environment's native task
semantics.

Every manifest also reports `final_phase_progress_distribution`.  Its keys are
the authoritative phase reached by each trajectory (`"i/N"`), and its values
are counts.  If the environment does not expose a valid phase index/count, the
trajectory is recorded under `"unknown"`; the evaluator never infers a phase
count from a task name.  Keep this histogram alongside success rate for every
diagnostic so partial progress (for example `3/6`) is visible.

For paper aggregation, report one success rate per surface and use the
five-column macro:

```text
macro5 = mean(Shopping_SR, Travel_SR, Search_SR, Math_SR, Physics_SR)
```

`compute_paper_macro5` implements this exact contract and rejects a missing,
renamed, extra, non-finite, or out-of-range column instead of silently forming
a four-family average.

The currently registered runtime surfaces are:

| Paper column | Runtime surface | Paper success source |
| --- | --- | --- |
| Shopping | `memoryarena_webshop_native_v1` | terminal `episode_success` |
| Travel | `memoryarena_travel_planner_paper_eval_one_action_v3` | official Travel SR ledger |
| Search | `memoryarena_progressive_search_paper_eval_public221_one_action_v3` | official Search paper ledger |
| Math | `memoryarena_formal_reasoning_math_paper_eval_one_action_v3` | original-style per-paper PS ledger and final-question SR |
| Physics | `memoryarena_formal_reasoning_phys_paper_eval_one_action_v3` | original-style per-paper PS ledger and final-question SR |

`memoryarena_travel_planner_failfast_one_action_v3` and
`memoryarena_progressive_search_failfast_public221_one_action_v3` are explicit
training/diagnostic surfaces. Their success rates are not eligible for the
Travel or Search paper columns. Travel fail-fast ends on an incorrect submitted
plan or a per-traveler action-budget exhaustion without advancing that phase;
Travel paper-eval records the failed prediction and continues so the official
PS/SPS/SR ledger covers every traveler in the group.

Travel paper-eval terminal evidence must contain one complete
`memoryarena_travel_eval_py_ps_sps_sr_v1` contribution ledger per frozen group.
Its exact schema is
`paper_evaluation={metric_contract,dataset_scope,source_id,complete,full_pass_people,total_people,group_success,group_constraint_rate,constraint_people,online_reward_is_separate}`.
The outer evaluator recomputes:

```text
PS  = sum(full_pass_people) / sum(total_people)
SPS = mean(group_constraint_rate for groups with constraint_people > 0)
SR  = sum(group_success) / number_of_groups
```

Travel PS/SPS/SR are reported on a `0..100` percentage scale; Travel's macro5
contribution is `SR / 100`. The evaluator rejects missing, extra, mistyped, or
contradictory ledger fields, position/source-ID drift, duplicate groups, dataset
scope drift, and a complete panel whose people total is not 1,869. A partial
panel may remain useful diagnostic evidence, but it is marked
`paper_macro_eligible=false` and cannot enter `compute_paper_macro5_from_manifests`.

Search paper evaluation uses
`memoryarena_progressive_search_ps_sr_at_k_final_sr_v1`. Each terminal task
ledger records the ordered phase verdicts, task-level process-score numerator
and denominator, one `SR@k` contribution per available depth, and the final
phase success. The outer evaluator recomputes all derived values from the
verdict sequence and aggregates:

```text
PS   = mean(task_correct_phases / task_phase_count)
SR@k = correct_at_depth_k / tasks_with_phase_count_at_least_k
SR   = mean(task_final_success)
```

These Search metrics use a `0..1` unit-interval scale, and Search's macro5
contribution is final `SR`. The evaluator requires the paper-eval surface,
zero online return, complete terminal ledgers, unique dataset positions and
query IDs, consecutive phase indices, exact per-depth contributions, and
cross-consistency among the duplicated terminal evidence fields. A complete
public panel must contain 221 tasks and 1,641 phases.

The released Search data covers `public221_of_paper256`: 35 tasks from the
paper's 256-task panel are not public. `panel_complete=true` in an outer
manifest means all 221 tasks on this named public surface were evaluated;
the nested Search metric remains `paper_panel_complete=false`. Results must
retain this dataset-scope label and must not claim full paper-256 coverage.

Formal paper evaluation uses
`memoryarena_formal_reasoning_ps_final_sr_v1`. Every submitted answer is
privately judged and advances to the next question, including an incorrect
answer. Each terminal paper ledger records the ordered question verdicts,
process-score numerator and denominator, and the final-question success bit.
The outer evaluator recomputes:

```text
PS = mean(correct questions / questions in paper)
SR = papers whose final question is correct / evaluated papers
```

Math and Physics use separate frozen panels and separate paper columns. A
complete Math panel contains 40 papers; a complete Physics panel contains 20.
Partial panels remain diagnostic and are marked `paper_macro_eligible=false`.
Only complete, provenance-verified panels may contribute their final `SR` to
the canonical five-column macro.

The Math+Physics **Formal Reasoning** family average is an auxiliary diagnostic
only. It must not replace the five-column macro or hide a missing surface.

The fail-fast training/diagnostic surfaces remain separately registered as
`memoryarena_formal_reasoning_math_failfast_v3` and
`memoryarena_formal_reasoning_phys_failfast_v3`. A wrong answer terminates
those variants immediately, and success means every question was correct.
Their output must retain the `failfast_v3` label and cannot be reported as the
paper Math or Physics column. This deliberate training contract
must not be presented as the original MemoryArena evaluation protocol.
