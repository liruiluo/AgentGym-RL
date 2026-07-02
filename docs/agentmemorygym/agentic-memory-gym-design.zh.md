# AgentMemoryGym：Agentic RL Memory 后训练 Gym 设计

## 0. 当前状态卡

- **研究目标**：构建一个面向 agent memory policy 的 RL 后训练 Gym，把长程、多会话、依赖历史状态的 agent 任务改写成可训练、可评测、可归因的环境。
- **当前阶段**：MemoryArena / 电商捆绑购物数据与接口打通。已完成 target freeze、全量 product DB 共享盘镜像、SQLite/FTS `SEARCH` 索引、Qwen3-4B 单卡 smoke，以及 scripted SEARCH baseline / heuristic memory manager dev 诊断。semanticfix5 当前结果为：no-memory `0/15`，full-context `6/15`，显式 memory-tool strict no-retry `6/15`，memory-tool retry5 `13/15`，memory-tool soft-fallback verifier diagnostic `15/15` 且 failure audit 无 failed steps。下一步是在授权 GPU lane 上启动 bounded RL pilot，正式 8 卡后训练等新申请的 8 卡机器到位。
- **已确定边界**：复用 AgentGym-RL / verl；memory 工具只参考 AgeMem 的 STM/LTM action 语义；不借 AgeMem 的三阶段课程学习路线；v0 先做 Gym、基线、评测和行为分析，不声称第一个 RL-memory 方法。
- **资源边界**：Mac/ZBMac 只做 0 卡检查；MemoryArena product DB 与 SQLite/FTS index 放 Jingyan 共享盘。Jingyan 1×B200 已跑真单卡 smoke；现有 8 卡仍给 continual-reasoning gym，AgentMemoryGym 等新 8 卡再跑正式后训练。
- **本阶段完成标准**：本地文档和 Notion 页面不再把 MemoryAgentBench AR/CR 写成第一主线；电商捆绑购物成为 hero environment；已有代码草稿保留为 `agentenv-agentmemory` skeleton，并通过 compile/direct/server-client smoke 后仍只标注为草稿。

## 1. 为什么做 Agentic RL Memory 后训练

现实中的复杂 agent 任务往往不是一次性信息充分的单轮问答。用户会逐步提出需求，环境会逐步给出反馈，后续行动必须依赖早期状态。例如电商捆绑购物中，用户先买电视，后面才买电视支架、电视柜、线缆和音响；这些后续商品必须兼容前面电视的尺寸、重量、接口和摆放约束。用户通常无法第一轮就完整描述全套需求，产品也不一定具备多轮序列推荐商品的能力。

固定 harness / memory module 可以保证下限：例如固定写入规则、固定检索、固定摘要、人工兼容表。但 harness 主要依赖人类先验，不能直接提升模型自己选择记忆动作的上限。本项目关注的是：在相同 memory tool 和可验证任务反馈下，RL 后训练能否让 agent 学会什么时候写入、更新、删除、检索、摘要和过滤，从而在长程任务中更稳定地完成目标。Harness 与后训练策略可以协同；v0 中 harness 应作为 baseline 和安全兜底。

## 2. 为什么是 Gym，不只是 benchmark 或算法

当前已经有不少 memory benchmark。MemoryArena、MemoryAgentBench、StructMemEval、Evo-Memory 等都能评估不同层面的 agent memory。再做一个静态 benchmark 的边际收益有限。

另一方面，AgeMem、Memory-R1、MemAct、MEM1、MemAgent、UMA 等已经说明 memory operation 可以被训练，但这些方法往往绑定特定任务、特定记忆形态或特定训练假设，不容易公平比较，也不容易分析 scaling 是否稳定。AgeMem 在这里仅作为 memory tool taxonomy 参考，不作为课程学习或数据组织方案参考。

因此 AgentMemoryGym 的定位是：把 agent memory 任务转成统一可训练 Gym，提供训练/验证/测试切分、memory action space、可验证奖励、轨迹字段、基线套件和行为分析协议。也就是说，项目要解决的是“怎么让 memory 后训练可复用、可比较、可扩展”，而不是只报告一个新榜单或先押一个新算法。

## 3. 环境设计：四类任务，电商优先

### 3.1 捆绑网络购物（Hero environment）

场景模拟真实的“持续搭配购买”。每个 episode 由多个购物 session 组成，后续 session 缺少关键历史信息，必须依赖前面购买记录和属性记忆。

例子：

```text
Session 1：买一台 75 英寸电视。
Session 2：买一个兼容这台电视的支架，但当前请求不再重复电视尺寸。
Session 3：买一个兼容同一台电视的电视柜。
```

环境维护 hidden bundle state，例如 `tv_size=75`、`tv_weight_kg=32`、`vesa=400x400`。Agent 当前能看到当前 session 的 observation、候选商品、自动记录的本 session action/tool-result trace，以及显式 `RETRIEVE/SUMMARY/FILTER` 带来的 active retrieved/summary context；跨 session 的原始对话/action 历史会清空。长期记忆需要通过 `ADD/UPDATE/DELETE/RETRIEVE` 等工具维护和取回。奖励来自购买是否完成且兼容。

这条线最贴近产品价值，也最适合作为第一版 demo 和后训练场景。

### 3.2 团队旅行规划

行程从一个人开始，然后陆续加入 5 到 8 个成员。新成员会提出依赖历史的约束，例如“第二天晚餐要和上一个加入的 Rebecca 一起吃”，或“酒店评分必须比 Rebecca 昨天住的酒店高至少两级”。Agent 必须记住加入顺序、人名、日期、餐厅、酒店和偏好。

### 3.3 渐进式信息搜索

复杂搜索问题被拆成多个严格依赖的子查询。后续查询必须使用前面查到的实体、学校、地点、年份或获奖者。这里测试的是 search tool 与 memory tool 的协同：写入中间实体、更新歧义实体、检索前置结果。

### 3.4 序贯形式化推理

面向数学/物理的长链条推导。专家把论文核心结论拆成有序引理、命题和参数状态；下一步推导必须准确复用前文结论。这类任务难度高，v0 先作为后续 evaluation-only / later-stage 环境，不强行塞进最小实现。

## 4. POMDP 与 memory action space

AgentMemoryGym 更准确地是一个带外部记忆状态的 POMDP：

```text
hidden state s_t = (task_state_t, long_term_memory_t, session_stm_t, retrieved_context_t, history_t)
observation o_t = render(session_stm_t, retrieved_context_t, current_task_view_t, tool_result_t)
action a_t = task action 或 memory tool action
transition T: (s_t, a_t) -> s_{t+1}
reward r_t = task_success/progress - memory/tool cost - violation penalty
```

- **自动短期记忆 STM / session trace**：当前 session 内的 action、observation 摘要和 tool result trace，默认可见；成功 `BUY` 进入下一 session 时清空，防止跨 session raw-history 泄漏。
- **长期记忆 LTM**：跨 session 持久化，但默认对 policy 隐藏；只有 `RETRIEVE` 结果进入 active retrieved/summary context。
- **Active retrieved/summary context**：由 `RETRIEVE/SUMMARY/FILTER` 操作产生的当前可见工作区；它不是全部 STM，而是从 LTM 或摘要工具带入当前 observation 的显式上下文。
- **任务状态**：例如已购商品、兼容约束、旅行成员行程、搜索中间实体、推导中间命题。

v0 memory 工具：

- `ADD {key, value}`：把高价值新信息写入长期记忆。
- `UPDATE {memory_id/key, value}`：当新信息修正旧信息时更新记忆。
- `DELETE {memory_id/key}`：删除过时或错误记忆。
- `RETRIEVE {query, top_k}`：把相关长期记忆拉回 active retrieved/summary context。
- `SUMMARY {text, source_ids?}`：把当前可见上下文压缩成摘要并放入 active retrieved/summary context。正式 RL 路径里摘要文本由当前 policy 模型自己生成，`source_ids` 只引用当前 observation 里可见的 `S*` / `C*` 条目，环境不在后台调用外部 LLM 代写摘要。
- `FILTER {keep_ids|drop_ids, scope}`：由当前 policy 模型选择保留或丢弃哪些可见上下文 ID，例如 `C0`、`S1`；环境只执行确定性的 keep/drop 状态转移。
- `SUMMARY {span}` / `FILTER {query}`：只作为 deterministic scaffold、smoke 和规则 baseline，不能当成隐藏的外部智能模块。
- 环境动作：购物使用 `BUY {product_id}` 和 product-catalog `SEARCH {query, top_k}`；其它环境后续扩展 `PLAN / ANSWER`。`SEARCH` 只返回公开商品 metadata，不暴露 ASIN/source path 或 target。

关键点：memory tool 是 policy 的 action，不是外部 harness 自动替 policy 做的事。尤其 `SUMMARY/FILTER` 不能让环境偷偷调另一个外部 LLM 完成判断，否则这些 token 不进入 rollout/logprob，RL 信用分配不干净。Harness baseline 可以使用同样工具，但触发规则固定。

## 5. 奖励与轨迹字段

奖励来源：

- 最终成功：episode 是否完成全部依赖任务。
- 进度奖励：每个子任务是否完成。
- 兼容/约束奖励：商品尺寸、承重、接口、酒店评分、日期等约束是否满足。
- memory/tool 成本：轻微惩罚过度写入、过度检索和无意义摘要。
- 违规惩罚：错误购买、冲突更新、检索后仍忽略关键记忆。

每一步 `info` 至少记录：

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
  "tool_ops": [{"op": "SEARCH", "tool_family": "catalog", "step": 1}],
  "memory_ops": [{"op": "ADD", "key": "tv_size", "step": 1}],
  "memory_state_diff": {"added": [], "updated": [], "deleted": []},
  "compatibility_violations": [],
  "purchase_history": [],
  "current_subtask_index": 2,
  "session_trace": []
}
```

这些字段服务训练日志、评测指标和行为分析。不能只保留最终 reward。

## 6. 评估与基线

主要指标：

- 成功率 / 任务完成率。
- 进度得分。
- 兼容约束满足率。
- memory ablation drop：移除或打乱关键记忆后的成功率下降。
- memory cost：每成功任务平均写入、检索、更新、删除、摘要次数。
- 错误类型：未写入、写错、未检索、检索噪声、检索后未使用、错误更新、过度工具调用。

基线：

1. No-memory agent：只看当前 observation。
2. Budgeted full-context：有限窗口保留历史。
3. Fixed RAG memory：固定写入与相似度检索。
4. Heuristic memory manager：规则触发 memory tool。
5. 可复现的现有 RL-memory 方法：AgeMem / Memory-R1 / MemAct / MEM1 / MemAgent / UMA。
6. Learned AgentMemoryGym policy：同一环境、奖励、split 和 tool schema 下训练。

## 7. 工程落点

```text
AgentGym-RL/                         # 主训练 fork
  docs/agentmemorygym/               # 当前设计与 Notion 同步源
  AgentGym/                          # submodule 环境 fork
    agentenv-agentmemory/            # 当前最小环境 skeleton
      agentenv_agentmemory/data/bundled_shopping_smoke.jsonl
      agentenv_agentmemory/data/splits/{train,dev,test}.txt
    agentenv/agentenv/envs/          # AgentMemoryEnvClient 注册入口
  AgentGym-RL/verl/utils/agentgym/   # task_name=agentmemory 接 trainer client
```

当前执行顺序已经调整为：

1. 先改本地文档和 Notion 文档。
2. 保留并整理已有代码 skeleton，不撤草稿。
3. 用 JSONL item schema 和 train/dev/test split 文件承载 smoke 任务，作为后续 MemoryArena/WebShop 转换入口。
4. server 支持 dataset/split 配置和 `/metadata`，client 可读真实 `task_count`，避免 trainer 继续硬编码 `data_len=1`。
5. AgentGym-RL vLLM rollout 已加 raw-history guard：`task_name=agentmemory` 默认阻止全历史拼接式 rollout；只有显式 `allow_raw_history_for_agentmemory` / `AGENTMEMORY_ALLOW_RAW_HISTORY=1` 才允许 diagnostic smoke，不得用于正式训练。
6. 当前 0 卡本地检查只作为代码/schema 快速验证，不写成单卡结果。
7. 真单卡 GPU 依赖环境已完成 env/client/init smoke：B200 + torch CUDA、real AgentGym adapter import、server metadata、`init_env_client` metadata path 均通过。
8. rollout 数据路径已推进：AgentMemory 默认 latest-observation；这里的 observation 包含当前 session 自动 STM trace，但不包含跨 session raw history。多轮 episode 展平成每个 action 一条 PPO 样本，并用 `rollout_parent_indices` 对齐原 batch，避免 actor/ref logprob 重算时读到上一 session 的原始历史。
9. Qwen3-4B / Transformers 真单卡 rollout 已跑通真实模型→parser→env step；SEARCH-aware prompt 也能调用 `SEARCH`，但当前会反复用占位 query，没有任务进度。
10. 全量 product DB 已在 Jingyan 共享盘构建 SQLite/FTS index；严格 all-candidate metadata 仍只有 `285/900`，说明缺口不是存储/下载范围，而是可靠 option-to-catalog 对齐与 policy 使用 SEARCH。
11. scripted SEARCH baseline / heuristic memory manager 已跑通 dev split：no-memory 为 `0/15`、`mean_progress=0.1889`；full-context 为 `6/15`、`mean_progress=0.5778`；显式 memory-tool one-shot/no-retry 为 `6/15`、`mean_progress=0.5778`；`max_buy_attempts=5` 的 SEARCH + verifier-feedback retry diagnostic 在 semantic matcher 修复后为 `13/15`、`mean_progress=0.9000`；failure audit 显示当前残余 2 个失败为 `compatibility_filter_excluded_target`。
12. 新增 soft-fallback verifier diagnostic：`--compatibility-fallback ranked-all-after-compatible --max-buy-attempts 7` 后 dev split 达到 `15/15`、`mean_progress=1.0`，说明 fair SEARCH + verifier feedback 足以完成当前 frozen dev，但这仍不是 RL 训练或 memory ability improvement。
13. 下一步是在授权 GPU lane 上启动 bounded GRPO/PPO pilot，用来暴露训练链路和策略失败；正式 memory 能力提升结论必须等新 8 卡上的 protocol、split、metric、checksum 和 baseline 对比收口。

## 8. 声明边界

- 不说 AgentMemoryGym 是第一个 RL memory 方法。
- 不说 MemoryArena 已完整实现，除非真实完成数据转换、split、评测脚本和 readback。
- 不把 smoke-only 的脚本运行说成 RL 提升证据。
- 对外文档中不要在未核验前把 Qwen3.6-4B 写成公开权重名；内部可以暂称目标 4B backbone。
- 中间 checkpoint 不做正式最终评测，除非明确标记 diagnostic。

## 9. 参考入口

- AgentGym-RL：https://github.com/WooooDyy/AgentGym-RL
- MemoryArena：https://memoryarena.github.io/
- AgeMem ACL Anthology：https://aclanthology.org/2026.acl-long.981/
- MemoryAgentBench：降级为 baseline / 数据参考，不再作为第一环境主线。
