# RA-DPO Academic Metrics Guide

本说明用于论文口径下汇报 `evaluate.py` / `evaluate_api.py` 的评测结果。

## 1. 主要与次要终点

- 主要终点（Primary Endpoint）:
  - `Strict F1`（三元组 `(event_type, role, argument)` 完全匹配）
- 次要终点（Secondary Endpoints）:
  - `Strict Precision / Strict Recall`
  - `Relaxed F1`（论元允许部分匹配）
  - `Type F1`（仅事件类型）
  - `Parse Error Rate`（解析失败率）
  - `Hallucination Rate / Hallucination Entity Rate`
  - `Schema Compliance Rate (Type+Role)`（事件类型合法 + 角色合法）

## 2. 指标定义与解释

- `Strict F1` 反映端到端抽取质量，建议作为主比较指标。
- `Relaxed F1` 反映边界容错下的语义抽取能力，建议作为辅助指标。
- `Type F1` 高而 `Strict F1` 低，通常表示“类型识别较好，但论元角色/边界错误较多”。
- `Schema Compliance Rate (Type+Role)`:
  - 当前实现为严格版本：若 `role` 不在对应 `event_type` 的 schema 角色集合中，则该样本记为不合规。

## 3. 统计报告建议

- 固定报告：`n`（样本数）、`split`、模型版本、prompt 版本、few-shot 设置、并发数。
- 置信区间：报告 95% CI（建议 bootstrap，已在脚本中支持）。
- 随机性控制：至少 3 个随机种子，报告均值与标准差（或 CI）。
- 显著性：模型对比建议使用 paired bootstrap 或近似随机化检验。

## 4. 错误分析建议

- 至少给出 Top-K `FN_*` / `FP_*` 错误项（脚本已有 `error_breakdown`）。
- 区分三类问题：
  - 类型错误（Type）
  - 角色错误（Role，含 schema 外角色）
  - 论元边界/规范化错误（Argument Span）

## 5. 建议表格结构（论文）

- 主表（主结果）:
  - Strict P/R/F1, Relaxed P/R/F1, Type P/R/F1
- 可靠性附表:
  - Parse Error Rate, Hallucination 指标, Schema Compliance
- 误差附表:
  - Top-10 Error Breakdown

## 6. 实验复现最小清单

- 数据集版本与切分
- 配置文件 hash 与命令行
- API 模型名称与响应模型标识
- 温度/最大 token/重试次数/并发数
- prompt hash（system + few-shot examples）
