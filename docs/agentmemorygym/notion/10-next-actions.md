# 下一步执行清单

## 现在先做

1. 更新本地文档。
2. 更新 Notion 分页面。
3. Notion readback 验证关键 marker。
4. 检查主仓和 submodule diff，确认代码草稿还只是草稿、不是完整 MemoryArena。

## 当前代码整理

1. 整理 `agentenv-agentmemory` skeleton，不撤已有草稿。
2. 保持 `AgentMemoryEnvClient` / `task_name=agentmemory` 注册。
3. 用中性 product_id 和显式 `info` 字段减少答案泄漏。
4. 用 JSONL item schema 和 smoke split 文件承载任务，为后续 MemoryArena/WebShop 转换留入口。
5. 跑 data validator、`compileall` 和 direct env smoke。
6. 启 server 做 client smoke。
7. 记录 0 卡本地检查结果，不写成单卡测试结果。
8. 等有干净单卡 GPU lane 后，再记录真实单卡 smoke 结果。

## 0 卡本地检查边界

Mac/ZBMac 上的 compile、data validator、direct env、server API、stubbed client metadata、context-policy guard 都只是 0 卡本地检查。它们用于防低级代码/schema 错误，不能叫单卡 smoke。

## 单卡测试边界

真正单卡 smoke 需要有 GPU、torch、AgentGym/verl 依赖和干净资源 lane。不占用当前给 continual-reasoning gym 的 8 卡，也不占用已有 Jingyan / CRL eval lane。AgentMemoryGym 明天拿新 8 卡后再启动正式训练。

## 代码完成前不得声称

- 不得声称已完整实现 MemoryArena。
- 不得声称 RL 已提升 memory 能力。
- 不得把 smoke 结果写成正式实验结果。
- 不得在未核验模型名时对外写死 Qwen3.6-4B。
