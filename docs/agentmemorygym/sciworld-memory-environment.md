# SciWorld memory environments for AgentMemoryGym

Status: multi-surface integration skeleton. These surfaces are registered and fixture-tested; native Java/Python SciWorld smoke and PPO training are still separate gates.

## Plain objective

Use SciWorld as a controlled scientific-experiment environment where the agent must:

1. run experiments;
2. decide what to write into its own long-term memory / lab notebook;
3. retrieve that memory later;
4. use it to solve a later task;
5. run another experiment instead of guessing when memory is insufficient.

This is not meant to prove that the model already knows elementary science facts. It is meant to train and test memory formation, retrieval, and reuse in an interactive lab.

## Non-negotiable memory boundary

For long-horizon SciWorld tasks, the harness must not hand the model a helpful rolling history window.

Allowed:

- current SciWorld observation and action feedback;
- ordinary model context/token limits;
- policy-facing memory tools such as `ADD`, `UPDATE`, `RETRIEVE`, `SUMMARY`, and `FILTER`;
- context failure if the policy refuses to manage its own notes.

Not allowed for the main memory surface:

- keeping a fixed small suffix of prior steps as the semantic memory design;
- environment-written summaries;
- ground-truth lab notes;
- curated recent-window transcripts;
- automatic compression by the harness.

The model itself must choose what to compress, what to store, and what to retrieve. Any environment-provided summary or handcrafted rolling transcript is only a scaffold/control variant, not evidence of learned long-term memory use.

## Registered SciWorld memory surfaces

These are memory-training surfaces inside SciWorld. They are not meant to be a
one-to-one copy of the official 30 ScienceWorld task names. A later native smoke
should map each surface to one or more real ScienceWorld task families, while
keeping the memory contract below unchanged.

### 1. `sciworld_conductivity_memory_v1`

What it tests:

- experimental fact memory;
- binding an unknown material to `conductive` / `nonconductive`;
- using the prior lab result in a later circuit-building phase.

Native SciWorld anchor:

- `test-conductivity-of-unknown-substances`.

### 2. `sciworld_meltingpoint_memory_v1`

What it tests:

- numeric experimental memory;
- remembering an exact measured value;
- comparing that value against a later threshold.

Native SciWorld anchor:

- `measure-melting-point-unknown-substance`.

### 3. `sciworld_friction_memory_v1`

What it tests:

- comparative / ranking memory;
- remembering which unnamed surface has higher friction;
- using prior measurements to choose a later surface.

Native SciWorld anchor:

- `inclined-plane-friction-unnamed-surfaces`.

### 4. `sciworld_rule_memory_v1`

What it tests:

- multi-experiment fact/rule induction;
- storing a reusable scientific rule from several observations;
- applying the rule later when the original experiments are not repeated;
- chaining retrieved facts, e.g. first retrieve `red + yellow -> orange`, then
  retrieve `orange + yellow -> amber`, then state the two-step plan.

Boundary:

- color mixing / friction rules are fact/rule memory, not SOP memory by default.

Native SciWorld anchor:

- `chemistry-mix-paint-secondary-color` and `chemistry-mix-paint-tertiary-color`.

### 5. `sciworld_sop_memory_v1`

What it tests:

- procedural memory;
- remembering a reusable lab procedure;
- transferring the procedure to a later task with different objects.

Example:

- remember how to test conductivity: assemble battery, wire, bulb, and sample; observe whether the bulb lights.

Boundary:

- this is different from remembering that a specific material was conductive.

### 6. `sciworld_negative_evidence_memory_v1`

What it tests:

- remembering failed or null experimental results;
- using negative evidence to exclude a candidate later;
- not repeating the same dead-end experiment as if it were unknown.

Example:

- record that powder zeta did not fizz with vinegar, then later choose the other candidate when the task needs a fizzing material.

### 7. `sciworld_hypothesis_tracking_memory_v1`

What it tests:

- keeping competing hypotheses separate;
- remembering which experiment supported or ruled out each hypothesis;
- selecting the supported explanation later.

Example:

- lamp direction is supported; soil color is ruled out.

### 8. `sciworld_calibration_memory_v1`

What it tests:

- remembering instrument calibration;
- applying a stored measurement offset later;
- avoiding treating raw readings as corrected facts.

Example:

- thermometer T reads 5 degrees high; later raw 75 means corrected 70.

### 9. `sciworld_contextual_rule_memory_v1`

What it tests:

- remembering that a rule depends on context;
- applying the rule only under the right condition;
- not flattening conditional science facts into universal facts.

Example:

- sugar dissolves quickly in hot water and slowly in cold water; the useful rule depends on temperature.

### 10. `sciworld_state_change_memory_v1`

What it tests:

- revising an earlier lab note after newer reliable evidence;
- using the latest reliable state rather than the stale preliminary result;
- exercising `UPDATE`-style memory behavior in later runtime tests.

Example:

- a quick strip first suggests riva is acidic; a calibrated pH meter later supersedes it and says neutral.

### 11. `sciworld_goal_progress_memory_v1`

What it tests:

- remembering unfinished experiment progress;
- carrying completed subgoals and the next subgoal across phase boundaries;
- resuming a multi-step lab plan without the harness repeating the checklist.

Example:

- collect sample -> heat sample -> record final color; later remember that only final color remains.

### 12. `sciworld_lab_notebook_longhorizon_v1`

What it tests:

- self-managed external lab notebook;
- many experiment records across phases;
- retrieving the right old note from many policy-written notes;
- succeeding without a harness-generated history summary.

This is the long-context surface. The raw interaction can become too large to rely on ordinary prompt context; success should come from policy-authored `ADD/UPDATE/RETRIEVE`, not environment curation.

## Acceptance bar for this skeleton

This code drop may claim only environment-support progress when it has:

1. registered all twelve SciWorld surface IDs in the AgentMemoryGym v3 domain registry;
2. exposed a model-facing contract that says memory is policy-managed;
3. included tests that reject manual history-window / environment-summary wording;
4. kept `scienceworld` as an optional runtime dependency with a clear fail-closed error if missing;
5. provided deterministic fixture backends so Mac-side tests do not require Java, torch, or the real SciWorld package;
6. fixture-tested at least one minimal memory chain for every registered surface.

It must not claim full native SciWorld training readiness until the real Java/Python SciWorld runtime smoke passes.
