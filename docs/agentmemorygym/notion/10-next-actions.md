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
8. 已用 Jingyan 1×B200 记录真实单卡 env/client/init smoke 结果。
9. latest-observation scripted-policy rollout smoke 已通过；下一步仍需补真正小模型/API rollout smoke，不把 raw-history override 计入正式结果。
10. MemoryArena bundled-shopping converter 已有入口和全量 smoke；catalog / ASIN resolver 已接入，Jingyan 共享盘 4 个相关 catalog shard 验证后把 12/900 个 target-match 歧义降到 0/900。
11. Product DB 已全量镜像到 Jingyan 共享盘 `/home/ai-jingyan-train/luolirui.1/post-train/data/memoryarena-product-db/`，最终校验 `135 files / 13,517,161,526 bytes`，extra/missing/mismatch/part 均为 0；不放开发机本地盘。
12. 正式 train/dev/test item-id 已冻结：`memoryarena_formal_freeze_20260701-234045`，`120/15/15`，`asin_catalog=900 / ambiguous=0`。
13. 下一步：在 Jingyan 1×B200 上跑真正小模型/API rollout smoke；不要把 scripted-policy 或 raw-history diagnostic 当正式 rollout 结果。

## 0 卡本地检查边界

Mac/ZBMac 上的 compile、data validator、direct env、server API、stubbed client metadata、context-policy guard 都只是 0 卡本地检查。它们用于防低级代码/schema 错误，不能叫单卡 smoke。

## 单卡测试边界

真正单卡 smoke 需要有 GPU、torch、AgentGym/verl 依赖和干净资源 lane。本轮已获准使用释放出的 Jingyan 1×B200，完成 env/client/init smoke；仍未完成小模型/API rollout smoke。AgentMemoryGym 明天拿新 8 卡后再启动正式训练。

## 代码完成前不得声称

- 不得声称已完整实现 MemoryArena。
- 不得声称 RL 已提升 memory 能力。
- 不得把 smoke 结果写成正式实验结果。
- 不得在未核验模型名时对外写死 Qwen3.6-4B。
