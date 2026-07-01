# AgentMemoryGym: Agentic RL Memory Post-Training Gym

## 0. State card

- **研究目标**：构建一个面向 agent memory policy 的 RL 后训练 Gym，把长程、多会话、依赖历史状态的 agent 任务改写成可训练、可评测、可归因的环境。
- **当前阶段**：从旧的 MemoryAgentBench-first 启动包，切换到 **MemoryArena/e-commerce-first 的 Agentic RL Memory Gym**。本阶段先完成设计对齐和最小可运行环境骨架，不声称已经完成训练。
- **已有人类决定**：以电商捆绑/序列购物作为 hero environment；复用 AgentGym-RL / verl 作为训练后端；memory action space 参考 AgeMem/Agentic Memory 工具语义；v0 不把新算法写成主贡献。
- **已有材料**：本仓 `AgentGym-RL` fork 与 `AgentGym` submodule fork；旧 Notion 备份；MemoryArena、AgeMem、AgentGym-RL、MemoryAgentBench 等参考源。
- **本轮交付物**：本设计文档 + `agentenv-agentmemory` 最小环境 skeleton + AgentGym-RL client 注册入口。

## 1. Positioning

AgentMemoryGym 的核心不是再做一个静态 memory benchmark，也不是先提出一个单点新算法，而是做一个 **Agentic RL Memory 后训练框架**：

```text
memory-dependent agent task stream
  -> AgentGym-style interactive environment
  -> explicit STM/LTM memory tools
  -> verifiable task/progress reward
  -> AgentGym-RL / verl PPO-GRPO style post-training
  -> task metrics + memory behavior analysis
```

### 1.1 Why RL post-training for memory

现实中的长程复杂 agent 任务经常不是一次性信息充分的单轮问答。用户会逐步提出需求，环境会逐步暴露反馈，后续决策必须依赖早期已发生的事实。例如电商中的捆绑网络购物：用户先买电视，之后才买电视支架、电视柜和线缆；后续购买必须记住前面电视的尺寸、重量、接口和摆放限制，否则会买到不兼容商品。

Harness / memory module 可以提供下限：固定写入规则、固定检索器、固定摘要器、人工设计的兼容检查都能减少灾难性遗忘。但 harness 主要依赖人类先验，不直接回答“模型能否通过后训练学会更好的 memory policy”。本项目关注的是：在同样的 memory tool 和环境反馈下，RL 是否能提升 agent 对写入、更新、删除、检索、摘要和过滤的决策上限。Harness 与 learned policy 不是互斥关系，v0 中 harness 更适合作为 baseline 和安全兜底。

### 1.2 Why a Gym rather than only a benchmark or an algorithm

- **不是只做 benchmark**：MemoryArena、MemoryAgentBench、StructMemEval、Evo-Memory 等已经覆盖了大量 memory 评测面。新的静态指标边际收益有限。
- **不是先做单算法**：AgeMem、Memory-R1、MemAct、MEM1、MemAgent、UMA 等已经说明 memory operation 可以被训练；问题是各自环境、动作空间、奖励和报告协议不统一，公平比较和 scaling 分析困难。
- **Gym 的空白**：把 agent memory 任务改写成统一可训练环境，提供 train/dev/test split、memory tools、verifiable reward、trajectory schema、baseline suite 和行为归因协议。

因此 v0 贡献应写成：**Agentic memory RL training environment + baseline/evaluation/analysis protocol**，而不是“第一个用 RL 训练 agent memory”。

## 2. Environment families

v0 以 MemoryArena 式“memory is functional to completing the task”为原则：记忆不是为了复述过去，而是为了指导后续 action。第一批环境以四类任务组织，其中 bundled web shopping 是 hero environment。

### 2.1 Bundled web shopping (hero)

场景模拟真实的持续搭配购买。每个 episode 由多个 shopping session 组成，后续 session 的目标缺少关键条件，必须依赖早期购买记录和属性记忆。

Example dependency:

```text
Session 1: buy a 75-inch TV.
Session 2: buy a wall mount compatible with the previously bought TV.
Session 3: buy a media console compatible with the TV size and room constraints.
```

环境维护 hidden bundle state，例如 `tv_size=75`、`tv_weight_kg=32`、`vesa=400x400`。policy 只能看到当前 session 的 active context；长期记忆必须通过 `ADD/UPDATE/DELETE/RETRIEVE` 等工具维护和取回。奖励来自购买是否完成且兼容。

这类任务最贴近产品价值：用户通常无法在第一轮完整描述全套需求，产品也不一定具备多轮序列推荐商品的能力。AgentMemoryGym 应首先把这个场景做成可训练、可验证、可归因的后训练环境。

### 2.2 Group travel planning

从单人行程开始，陆续加入 5--8 个成员。新成员的约束依赖前人行程，例如“第二天晚餐和上一个加入的 Rebecca 一起吃”或“酒店评分比 Rebecca 昨天住的酒店高至少两级”。成功需要维护人名、加入顺序、日期、餐厅、酒店和偏好之间的动态状态。

### 2.3 Progressive information search

把复杂搜索问题拆成严格因果顺序的子查询。后续查询必须使用前面搜索得到的实体或属性，不能靠一次性猜测完成。适合检测 memory tool 与 web/search tool 的协同：写入中间实体、更新歧义实体、检索前置查询结果。

### 2.4 Sequential formal reasoning

面向数学/物理长链条推导。专家把论文核心结论拆成有序中间陈述；推导下一步必须重用前面几页中的参数、定义、引理和约束。此类任务难度最高，v0 可先作为 evaluation-only 或 later-stage environment，不宜在最小骨架中强行实现。

## 3. POMDP and memory action space

AgentMemoryGym 更准确地是一个带外部记忆状态的 POMDP：

```text
hidden state s_t = (task_state_t, long_term_memory_t, short_term_context_t, history_t)
observation o_t = render(short_term_context_t, current_task_view_t, tool_result_t)
action a_t = task action or memory tool action
transition T: (s_t, a_t) -> s_{t+1}
reward r_t = task_success/progress - memory/tool cost - violation penalty
```

- **LTM**：跨 session 持久化，但对 policy 隐藏；只有 `RETRIEVE` 结果会进入 active context。
- **STM / active context**：当前可见上下文，受摘要、检索、过滤和任务 observation 影响。
- **Task state**：例如已买商品、兼容约束、当前成员行程、搜索中间实体、推导中间命题。

### 3.1 Memory tools

v0 采用可解析的文本/JSON action，后续可映射到 function calling：

| Group | Tool | Semantics |
|---|---|---|
| LTM | `ADD {key, value}` | 抽取当前上下文中新且高价值的信息写入长期记忆。 |
| LTM | `UPDATE {memory_id/key, value}` | 当新信息修正旧信息时更新已有记忆。 |
| LTM | `DELETE {memory_id/key}` | 删除过时、错误或有害的记忆。 |
| STM | `RETRIEVE {query, top_k}` | 从 LTM 中检索相关记忆并加入 active context。 |
| STM | `SUMMARY {text}` | 将长上下文压缩为可复用摘要，加入 active context 或 LTM。 |
| STM | `FILTER {query}` | 从 active context 中移除与当前任务无关的信息。 |
| Task | `BUY {product_id}` / `PLAN` / `SEARCH` / `ANSWER` | 环境特定动作。hero env 先实现 `BUY`。 |

关键点：memory tool 是 policy 的 action，不是外部 harness 自动替 policy 做的事。Harness baseline 可以使用相同工具，但工具触发规则固定。

## 4. Reward and trajectory schema

### 4.1 Reward sources

- **Final success**：episode 是否完成全部依赖任务，例如整套购买全部兼容。
- **Progress reward**：每个子任务/购买/计划步骤是否成功。
- **Compatibility / constraint reward**：商品尺寸、承重、接口、酒店评分、日期约束等是否满足。
- **Memory operation cost**：轻微惩罚过度写入、过度检索和无意义摘要。
- **Violation penalty**：错误购买、冲突更新、检索后仍忽略关键记忆。

v0 skeleton 使用简单 dense progress reward；正式训练前需要固定 reward weights，并把 reward decomposition 写入 `info`。

### 4.2 Required `info` fields

每一步至少记录：

```json
{
  "task_id": "tv_bundle_75",
  "task_family": "bundled_shopping",
  "split": "train",
  "source": "memoryarena_webshop_style_handcrafted_v0",
  "difficulty": "smoke_dependency_distance_2",
  "memory_dependency": "tv_size_weight_vesa_reused_across_sessions",
  "progress_score": 0.67,
  "episode_success": false,
  "memory_ops": [{"op": "ADD", "key": "tv_size", "step": 1}],
  "memory_state_diff": {"added": [...], "updated": [...], "deleted": [...]},
  "compatibility_violations": [],
  "purchase_history": [...],
  "current_subtask_index": 2
}
```

这些字段服务三件事：训练日志、评测指标和行为分析。不要只保留最终 reward。

## 5. Evaluation and baselines

### 5.1 Primary metrics

- Success rate / task completion rate.
- Progress score：完成的依赖子任务比例。
- Constraint satisfaction rate：兼容约束满足率。
- Memory ablation drop：移除或打乱关键记忆后的成功率下降。
- Memory cost：每成功任务平均写入、检索、更新、删除、摘要次数。
- Error taxonomy：忘记写入、写错、未检索、检索噪声、检索后未使用、错误更新、过度工具调用。

### 5.2 Baselines

1. No-memory agent：仅当前 observation。
2. Budgeted full-context：有限窗口内保留历史。
3. Fixed RAG memory：固定写入和相似度检索。
4. Heuristic memory manager：规则触发 ADD/UPDATE/RETRIEVE。
5. Existing RL-memory methods where reproducible：AgeMem/Memory-R1/MemAct/MEM1/MemAgent/UMA-style baselines。
6. Learned AgentMemoryGym policy：在同一 environment、reward、split 和 memory tool schema 下训练。

## 6. Implementation plan in this fork

### 6.1 Current code ownership

```text
AgentGym-RL/                         # main training fork
  docs/agentmemorygym/               # design and run records
  AgentGym/                          # submodule environment fork
    agentenv-agentmemory/            # new environment package
      agentenv_agentmemory/data/bundled_shopping_smoke.jsonl
      agentenv_agentmemory/data/splits/{train,dev,test}.txt
    agentenv/agentenv/envs/          # AgentGym client registry
  AgentGym-RL/verl/utils/agentgym/   # verl/AgentGym env client registry
```

### 6.2 v0 implementation steps

1. Add `agentenv-agentmemory` with a minimal bundled-shopping environment.
2. Add `AgentMemoryEnvClient` and register task name `agentmemory`.
3. Run direct environment smoke with scripted memory policy.
4. Add a JSONL item schema for bundled shopping smoke tasks.
5. Add data converters for real MemoryArena/WebShop-style bundled shopping tasks.
6. Replace the smoke split files with real train/dev/test item-id files after MemoryArena/WebShop conversion.
7. Add a bounded GRPO smoke config.
8. Only after baseline behavior analysis, decide whether a new method beyond standard GRPO/PPO is needed.

## 7. Claim guardrails

- Do not claim AgentMemoryGym is the first RL memory method.
- Do not claim MemoryArena is fully implemented until real converted tasks, splits, and evaluation scripts exist.
- Do not claim Qwen3.6-4B as a public model name in outward-facing docs without fresh source verification; internally it can be a placeholder for the target 4B backbone.
- Do not report success from smoke-only scripted runs as evidence of RL improvement.
- Do not run final evaluation on intermediate checkpoints unless explicitly marked diagnostic.

## 8. Sources to keep attached

- AgentGym-RL: https://github.com/WooooDyy/AgentGym-RL
- MemoryArena project page: https://memoryarena.github.io/
- AgeMem ACL Anthology page: https://aclanthology.org/2026.acl-long.981/
- MemoryAgentBench: previously used as v0 reference/source; now demoted from first environment to baseline/data reference.
