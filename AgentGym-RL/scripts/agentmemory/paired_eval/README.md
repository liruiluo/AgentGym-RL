# AMG paired external-evaluation runner

This package is the thin, benchmark-agnostic orchestration layer for the frozen
matched triad: `native`, `amg_compaction_only`, and `amg_memory`.
`PairedRunner.run_task` contains the only sampling loop. It samples one normal
policy output, sends it through the injected task-neutral policy-turn
controller, accounts the ordinary turn/tool/token/wall budgets, and records
digest-addressed private evidence.

The injected wrapper owns reset/close, benchmark lifecycle, its native memory
action parsing and execution, compaction trigger/timing, context-transition
receipts, final artifacts, and official grader handoff. The runner sees only
the abstract external read/write capability and structured operation receipts;
it neither parses an action language nor dispatches on benchmark or arm. The
capability lattice is exact: `native=00`,
`amg_compaction_only=10`, and `amg_memory=11`, where the bits denote
policy-authored compaction and external read/write memory. There is no
memory-only fourth arm.

The two compaction-enabled arms share the exact trigger, summary-instruction
digest, context-pressure-policy digest, context-transition schema, and global
policy-action accounting. External memory is an all-or-none bundle covering
its dedicated namespace/root, mount, endpoint, environment variable, prompt
declaration, tool schema, parser/dispatch path, action receipt, private evidence
store, and cleanup handle. Disabled arms attest none of those surfaces.
Benchmark-native tools and task workspaces remain matched across all arms.

For memory-disabled arms, the full initial prompt must be byte-identical to the
treatment-excluded prompt. For `amg_memory`, the benchmark adapter must strip
and attest its exact frozen memory suffix; the runner then verifies that this is
the only prompt change. Client-side normalization must be idempotent, so an
adapter cannot self-attest a clean base digest while injecting additional
treatment-specific instructions.

## Manifest and CLI

A manifest declares common model, decoding, budget, compaction, source,
runtime, and grader configuration; a nonempty task list; and exactly the three
arms `native`, `amg_compaction_only`, and `amg_memory`. Expansion produces one
immutable `RunConfig` per task/arm combination.

```bash
PYTHONPATH=scripts/agentmemory python3 -m paired_eval expand --manifest paired.json
PYTHONPATH=scripts/agentmemory python3 -m paired_eval run \
  --manifest paired.json \
  --results private/results.jsonl \
  --evidence-dir private/evidence \
  --runtime-factory integration.bindings:build_runtime
PYTHONPATH=scripts/agentmemory python3 -m paired_eval verify --results private/results.jsonl
PYTHONPATH=scripts/agentmemory python3 -m paired_eval public-summary --results private/results.jsonl
```

The integration factory has signature
`build_runtime(config, *, evidence_store) -> RuntimeBindings`. It composes the
appropriate wrapper and model client outside the runner. Production execution
uses `AgentGymPolicyTurnController`, which imports and calls the exact
`bind_initial_policy_context`, `prepare_policy_turn`, and
`complete_policy_turn` functions. The dependency-light controller exists for
stdlib-only contract tests.

In-process callers can use
`make_runtime_factory(builders, evidence_store=store)` to capture the private
store and obtain the one-argument factory accepted by `execute_manifest`.

At reset, a wrapper returns deterministic namespace-and-route-bound lifecycle
root IDs for every route enabled by the frozen capability. Every ordinary step
carries the task-neutral transition schema, its accepted capability/root
identity, monotonic native/context/session/policy counters, and a route-specific
wrapper execution attestation. That attestation is digest-bound to the exact
policy output, while its private execution detail remains in evidence storage.
The runner validates these receipts without parsing policy text. A replacement
transition must use the generic policy-compaction route. `close` returns a typed
attestation covering the same namespace and every declared root; missing,
reused, mismatched, or unclosed roots make the row non-comparable.
Exact-tokenization errors retain a typed model cause even when the policy-turn
controller wraps them.

`run` requires a fresh/empty result path. It validates all rows and every exact
three-arm group before appending the complete batch under one file lock; reuse
of a nonempty path fails closed. Every row includes a successful generic wrapper-close
receipt before it can be comparable, and the wall budget covers final artifact,
grader, and close time as well as sampling and environment steps.

JSONL and evidence directories are mode `0700`; files are mode `0600`.
Messages, policy outputs, observations, wrapper receipts, artifacts, grader
details, and error text stay behind `evidence://<category>/<sha256>` references.
The public-summary command projects only identity fields, termination classes,
comparability, and explicitly scalar public metrics. It reports raw metrics for
all three arms plus `compaction_effect`,
`external_memory_incremental_effect`, and `full_amg_effect` numeric contrasts.
Public identity fields and metric names must be bounded ASCII labels; paths,
URIs, whitespace, and other unrestricted strings fail closed instead of being
copied into a summary.

This package provides the shared orchestration contract plus the frozen-client
registry and lifecycle bridge. Deployment-specific runtime builders, official
graders, datasets and containers, inference services, and benchmark runtime
readiness remain external. Passing local tests establishes the integration code
contract only; it does not establish real benchmark readiness.
