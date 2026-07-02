# 框架设计 v0

## 总体循环

```text
memory-dependent task stream
  -> AgentGym-style environment
  -> STM/LTM memory tools
  -> verifiable task/progress reward
  -> AgentGym-RL / verl post-training
  -> task metrics + memory behavior analysis
```

## POMDP 形式

```text
hidden state s_t = (task_state_t, long_term_memory_t, session_stm_t, retrieved_context_t, history_t)
observation o_t = render(session_stm_t, retrieved_context_t, current_task_view_t, tool_result_t)
action a_t = task action 或 memory tool action
transition T: (s_t, a_t) -> s_{t+1}
reward r_t = task_success/progress - memory/tool cost - violation penalty
```

长期记忆默认隐藏，只有检索结果进入 active retrieved/summary context。短期记忆分两层：一层是环境自动维护的当前 session action/tool-result trace，session 内可见，成功 `BUY` 进入下一 session 时清空；另一层是 `RETRIEVE/SUMMARY/FILTER` 产生的 active retrieved/summary context。任务状态包含已购商品、兼容约束、旅行安排、搜索中间实体或推导命题。

## Memory tools

AgeMem 只作为 STM/LTM tool taxonomy 参考；不采用它的三阶段课程学习路线。

- `ADD {key, value}`：写入高价值新信息。
- `UPDATE {memory_id/key, value}`：修正旧记忆。
- `DELETE {memory_id/key}`：删除错误或过时记忆。
- `RETRIEVE {query, top_k}`：取回相关长期记忆，放入 active retrieved/summary context。
- `SUMMARY {text, source_ids?}`：由当前 policy 模型自己写摘要文本，环境只验证可见 `S*` / `C*` source IDs 并替换 active retrieved/summary context。
- `FILTER {keep_ids|drop_ids, scope}`：由当前 policy 模型选择保留/丢弃哪些可见上下文 ID；环境只执行确定性 keep/drop。
- `SUMMARY {span}` / `FILTER {query}`：只作为 deterministic scaffold、smoke 或规则 baseline，不调用外部 LLM / hidden judge。
- 环境动作：购物使用 `BUY {product_id}` 和 product-catalog `SEARCH {query, top_k}`；`SEARCH` 返回公开商品 metadata，不暴露 ASIN/source path/target。其它环境后续扩展 `PLAN / ANSWER`。

关键边界：memory tool 是 policy 的 action，不是外部 harness 自动替 policy 做的事。`SUMMARY/FILTER` 的正式 RL 路径必须让当前 policy 产出摘要 token 或 keep/drop 决策；不能让环境后台调外部 LLM，否则不进 rollout/logprob。`SEARCH` 是商品 catalog 工具，不算 memory tool。

## 第一环境：bundled web shopping

每个 episode 由多个 session 组成。session 内允许自动保留完整/近期 action-observation-tool-result 工作上下文；后续 session 的指令不重复关键历史属性，且不会看到上一 session 的 raw history，必须依赖前面写入 LTM 并检索回来的记忆。

奖励包括：

- 完成当前购买。
- 商品与历史 bundle state 兼容。
- 完成全部子任务。
- 过度 memory/tool 操作轻微扣分。
- 不兼容购买或错误更新扣分。

## 轨迹字段

每一步 `info` 至少记录：

- `task_family`
- `task_id`
- `split`
- `source`
- `difficulty`
- `memory_dependency`
- `progress_score`
- `episode_success`
- `tool_ops`
- `memory_ops`
- `memory_state_diff`
- `compatibility_violations`
- `purchase_history`
- `current_subtask_index`
- `session_trace`

这些字段用于训练日志、评测表和行为分析。不能只保留最终 reward。
