# 2026-07-01 AgentMemoryGym source check

This note records first-party / primary-source checks used by the current
AgentMemoryGym design. It is a source-alignment note, not experiment evidence.

## MemoryArena

Source: <https://memoryarena.github.io/>

Checked points:

- MemoryArena frames agent memory as part of multi-session Memory-Agent-Environment loops, not isolated recall.
- Tasks contain explicitly interdependent subtasks; the agent must acquire useful memory in earlier sessions and use it to solve later subtasks.
- Public dataset configs include `bundled_shopping`, `progressive_search`, `group_travel_planner`, `formal_reasoning_math`, and `formal_reasoning_phys`.
- Dataset rows expose multi-subtask structure via fields such as `id`, `questions`, `answers`, and background/context fields.
- The current AgentMemoryGym v0 direction is therefore aligned with using bundled shopping as the hero environment and converting MemoryArena/WebShop-style tasks into trainable gym items.

## AgentGym-RL

Source: <https://github.com/woooodyy/AgentGym-RL>

Checked points:

- AgentGym-RL is presented as a framework for multi-turn interactive decision-making through RL.
- The repo describes a modular split between environment, agent, and training modules.
- Environment interaction uses a server-client style interface, which matches the current `agentenv-agentmemory` FastAPI server/client skeleton.
- The README lists mainstream online RL algorithms including PPO, GRPO, RLOO, and REINFORCE++, plus complementary methods such as SFT/DPO/AgentEvol.
- Keeping AgentMemoryGym inside the AgentGym-RL / verl stack is therefore the right first engineering path.

## AgeMem

Source: <https://aclanthology.org/2026.acl-long.981.pdf>

Checked points:

- AgeMem exposes explicit memory-management actions as tools.
- The paper's tool taxonomy matches the current AgentMemoryGym action plan:
  - LTM: `ADD`, `UPDATE`, `DELETE`
  - STM: `RETRIEVE`, `SUMMARY`, `FILTER`
- The paper argues for integrating memory tools into the agent action space so the policy can learn when and how to use memory, rather than relying only on external heuristics.
- Boundary correction on 2026-07-02: AgeMem is only a memory-tool taxonomy reference for AgentMemoryGym. Its three-stage curriculum-learning route is not adopted because it is not clean enough for the current Gym/evaluation design.
- This supports AgentMemoryGym's decision to make memory operations part of the RL action space, while still allowing harness/RAG baselines.

## Current boundary

- These source checks support the design direction only.
- They do not prove that the current skeleton is a full MemoryArena conversion.
- They do not prove RL memory improvement.
- They do not verify the user-mentioned `Qwen3.6-4B` model name; keep that as an explicit future verification item.
