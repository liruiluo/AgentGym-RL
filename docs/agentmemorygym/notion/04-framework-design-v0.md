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
hidden state s_t = (task_state_t, long_term_memory_t, short_term_context_t, history_t)
observation o_t = render(short_term_context_t, current_task_view_t, tool_result_t)
action a_t = task action 或 memory tool action
transition T: (s_t, a_t) -> s_{t+1}
reward r_t = task_success/progress - memory/tool cost - violation penalty
```

长期记忆默认隐藏，只有检索结果进入 active context。短期记忆是当前可见上下文。任务状态包含已购商品、兼容约束、旅行安排、搜索中间实体或推导命题。

## Memory tools

- `ADD {key, value}`：写入高价值新信息。
- `UPDATE {memory_id/key, value}`：修正旧记忆。
- `DELETE {memory_id/key}`：删除错误或过时记忆。
- `RETRIEVE {query, top_k}`：取回相关长期记忆。
- `SUMMARY {text}`：压缩冗长上下文。
- `FILTER {query}`：过滤短期上下文噪声。
- 环境动作：购物使用 `BUY {product_id}` 和 product-catalog `SEARCH {query, top_k}`；`SEARCH` 返回公开商品 metadata，不暴露 ASIN/source path/target。其它环境后续扩展 `PLAN / ANSWER`。

## 第一环境：bundled web shopping

每个 episode 由多个 session 组成。后续 session 的指令不重复关键历史属性，必须依赖前面写入和检索到的记忆。

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
- `memory_ops`
- `memory_state_diff`
- `compatibility_violations`
- `purchase_history`
- `current_subtask_index`

这些字段用于训练日志、评测表和行为分析。不能只保留最终 reward。
