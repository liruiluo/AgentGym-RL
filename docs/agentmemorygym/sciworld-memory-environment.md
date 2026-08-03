# SciWorld memory environment design for AgentMemoryGym

Status: first integration skeleton, not a completed capability result.

## Plain objective

Use SciWorld as a controlled scientific-experiment environment where the agent must:

1. run experiments;
2. decide what to write into its own long-term memory / lab notebook;
3. retrieve that memory later;
4. use it to solve a later task;
5. run another experiment instead of guessing when memory is insufficient.

This is not meant to prove that the model knows elementary science facts. It is meant to train and test memory formation, retrieval, and reuse in an interactive lab.

## Non-negotiable memory boundary

For long-horizon SciWorld tasks, the harness must not hand the model a helpful rolling history window.

Allowed:

- current SciWorld observation and action feedback;
- ordinary model context/token limits;
- policy-facing memory tools such as `ADD`, `UPDATE`, `RETRIEVE`, `SUMMARY`, and `FILTER`;
- context failure if the policy refuses to manage its own notes.

Not allowed for the main memory surface:

- “keep only the latest N steps” as a semantic design;
- environment-written summaries;
- ground-truth lab notes;
- curated recent-window transcripts;
- automatic compression by the harness.

The model itself must choose what to compress, what to store, and what to retrieve. Any environment-provided summary or handcrafted rolling window is only a scaffold/control variant, not evidence of learned long-term memory use.

## First surfaces

### 1. `sciworld_conductivity_memory_v1`

First implementation target.

Why this one:

- simple action procedure;
- clear source-session fact: object/substance X is conductive or non-conductive;
- clear dependent-session use: pick the correct material/component later;
- easy to audit whether success used memory or only current observation.

Memory ability trained:

- experimental fact memory;
- object-property binding;
- “if unsure, test again rather than inventing.”

### 2. `sciworld_meltingpoint_memory_v1`

Second target.

Memory ability trained:

- numeric experimental memory;
- threshold comparison;
- avoiding stale or approximate values when exact measurement is needed.

### 3. `sciworld_friction_memory_v1`

Third target.

Memory ability trained:

- comparative / ranking memory;
- remembering which unnamed surface had higher or lower friction;
- using prior measurements to avoid rerunning the same experiment.

### 4. `sciworld_lab_notebook_longhorizon_v1`

Later target.

This is the long-context surface. It should chain many SciWorld experiments so the raw transcript becomes too long to keep in prompt. The success condition should require a policy-authored external notebook/LTM, not a harness-made recent-N prompt.

Memory ability trained:

- self-managed compression;
- self-managed external lab notebook updates;
- retrieving the right prior experiment from many notes;
- deleting/updating wrong or obsolete notes when the agent discovers a correction.

### 5. `sciworld_rule_memory_v1`

Later target.

This covers multi-experiment rule/fact induction. Color mixing and friction rules belong here. They are not SOP memory by default.

Memory ability trained:

- deriving a reusable scientific fact/rule from multiple experiments;
- storing the rule in a way that can transfer to a later task.

## SOP memory boundary

SOP memory means remembering a reusable procedure, not just remembering a scientific fact.

Examples:

- fact/rule memory: “red + yellow makes orange”; “surface A has more friction than surface B.”
- SOP memory: “to test conductivity, assemble a circuit, insert the material, observe whether the bulb lights, then record the result.”

SciWorld can train both, but they need separate surface IDs and metrics.

## Acceptance bar for the first skeleton

The first code drop should only claim environment-support progress when it has:

1. registered SciWorld surface IDs in the AgentMemoryGym v3 domain registry;
2. exposed a model-facing contract that says memory is policy-managed;
3. included tests that reject manual recent-N / environment-summary wording;
4. kept `scienceworld` as an optional runtime dependency with a clear fail-closed error if missing;
5. provided a deterministic fixture backend so Mac-side tests do not require Java, torch, or the real SciWorld package.

It must not claim full native SciWorld training readiness until the real Java/Python SciWorld runtime smoke passes.
