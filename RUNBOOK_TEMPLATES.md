# RUNBOOK_TEMPLATES.md

本文件提供可直接复制的阶段记录、run manifest、结果汇总模板。若仓库已有既定格式，优先沿用仓库格式；本文件只作为最小模板。

## 1. Phase 状态记录模板

```md
## Phase X 状态
- 日期：YYYY-MM-DD
- 状态：已完成 / 未完成 / blocker
- 执行者：Codex CLI / subagent 名称
- 相关文件：
  - ...
- 新确认的仓库事实：
  - ...
- 仍未确认的事项：
  - TODO: inspect repo
- 输出目录 / 结果文件：
  - ...
- 下一步：
  - ...
- blocker（如有）：
  - ...
```

## 2. 单次 run manifest 模板

```md
# Run Manifest

## 基本信息
- run_id:
- phase:
- 日期:
- 结果来源类型: LOCAL_RUN / LOCAL_REEVAL / REFERENCE_ONLY
- 数据划分: dev / test(仅预测，不汇报 F1) / NA
- 模型:
- 相关 checkpoint / 预测文件:
- 输出目录:

## 执行信息
- 启动命令: TODO: inspect repo -> fill actual command
- 配置快照位置:
- 代码状态:
  - commit hash / diff 摘要 / 修改文件列表
- 关键开关:
  - SCV: on / off / lite
  - prompt skeleton: on / off
  - role whitelist: on / off
  - alias map: on / off
  - duplicate role split: on / off
  - normalization: on / off
  - grounding: off / exact / exact+fuzzy / exact+fuzzy+code

## 指标
- strict_f1:
- relaxed_f1 / type_f1:
- parse_error_rate:
- schema_compliance_rate:
- hallucination_rate:
- grounding_rate:
- ungrounded_argument_rate:
- SCV 调用次数:
- SCV wall-clock 占比:
- 训练耗时:
- 推理耗时:
- 显存占用:

## 失败与诊断
- Top 错误类型:
  - ...
- 失败样本位置:
- sidecar diagnostics 位置:
- 与 baseline 的主要差异:
  - ...

## 结论
- 是否进入下一阶段: 是 / 否
- 决策依据:
  - ...
```

## 3. Phase 0 仓库核查记录模板

```md
# Phase 0 Repo Audit

## 数据加载入口
- 文件:
- 已确认事实:
- 未确认事项:
- blocker:

## prompt 构造入口
- 文件:
- 已确认事实:
- 未确认事项:
- blocker:

## JSON 解析/修复入口
- 文件:
- 已确认事实:
- 未确认事项:
- blocker:

## SCV 实现入口
- 文件:
- 已确认事实:
- 未确认事项:
- blocker:

## 训练入口
- 文件:
- 已确认事实:
- 未确认事项:
- blocker:

## 评测入口
- 文件:
- 已确认事实:
- 未确认事项:
- blocker:

## API baseline 入口
- 文件:
- 已确认事实:
- 未确认事项:
- blocker:

## 消融脚本入口
- 文件:
- 已确认事实:
- 未确认事项:
- blocker:

## run 目录规范
- 仓库现有约定:
- 若无现有约定时采用的最小规范:
- blocker:
```

## 4. 最小错误统计模板

```md
# Error Breakdown

| 错误类型 | 计数 | 占比 | 样例文件/样例ID | 备注 |
|---|---:|---:|---|---|
| parse failure | TBD | TBD | TBD | |
| 非法事件类型 | TBD | TBD | TBD | |
| 非法角色 | TBD | TBD | TBD | |
| alias 未回正 | TBD | TBD | TBD | |
| duplicate role 未拆分 | TBD | TBD | TBD | |
| 规范化错误 | TBD | TBD | TBD | |
| grounding 失败 | TBD | TBD | TBD | |
| ungrounded argument 被保留 | TBD | TBD | TBD | |
| 互斥事件冲突 | TBD | TBD | TBD | |
| 共享角色归属不明 | TBD | TBD | TBD | |
```

## 5. 结果汇总速记模板

```md
# Result Snapshot

- baseline(strict_f1):
- scv_off(strict_f1):
- scv_on(strict_f1):
- mvp(strict_f1):
- enhanced(strict_f1):
- api_baseline(strict_f1):

- baseline(schema_compliance_rate):
- mvp(schema_compliance_rate):
- enhanced(schema_compliance_rate):

- baseline(hallucination_rate):
- mvp(hallucination_rate):
- enhanced(hallucination_rate):

结论：
- 是否进入增强版：
- 是否保留 SCV-Lite：
- 是否需要回退：
- 哪些结果只能标 REFERENCE_ONLY：
```

## 6. 论文表格填表注意事项
- 所有 F1 都必须显式标注为 dev。
- test 只能出现在“预测/提交说明”或备注中。
- 没有 provenance 的数字不填表。
- `REFERENCE_ONLY` 条目必须在备注中写清“未直接复现”。
- 若某指标脚本不支持，写 `NA`，不要留空。
