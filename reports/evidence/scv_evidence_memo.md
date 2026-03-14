# SCV Evidence Memo

## Scope
- 本备忘录只整理“当前样例与摘要层面能支持什么、不能支持什么”。
- 不写最终研究结论，不把个别 case 扩写成总体因果判断。
- 所有数字均回溯到已有 `summary/metrics/jsonl`，本轮未训练、未重跑。

## Current Strongest Supported Evidence
- `academic_full_local` 与 `academic_no_scv_local` 的样例级 strict 对照，不是单边碾压。
  - `full correct / no_scv wrong = 13`
  - `no_scv correct / full wrong = 10`
  - `both wrong but different = 816`
  - 证据来源：`logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047.jsonl`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047.jsonl`
- `full` 这 13 个独占正确样例里，最常见的失误不是 parse fail，而是 `no_scv` 的额外角色和角色缺失。
  - 统计标签计数：
    - `no_scv_role_extra = 10`
    - `no_scv_role_missing = 8`
    - `no_scv_event_type_error = 2`
  - 代表样例：
    - `2b9ed764ea681a7254207b22e39df35a`: `no_scv` 多报 `债权人`
    - `66a55a53915fbdf74bcd5e14f863d40d`: `no_scv` 把 `企业收购` 翻成 `企业融资`
    - `389655d8f75c394dbd62f80eced5ad00`: `no_scv` 漏 `披露时间`
- `no_scv` 这 10 个独占正确样例里，最常见的是 `full` 的角色遗漏和角色槽位错置。
  - 统计标签计数：
    - `full_role_missing = 9`
    - `full_role_extra = 7`
    - `full_canonical_rewrite = 1`
  - 代表样例：
    - `58567406e01800c4d1ff10835ffac74d`: `full` 把 `股票简称` 写成 `被减持方`
    - `7de944ac12541a2b3021d3dbe264ecca`: `full` 把 `披露时间` 写成 `破产时间`
    - `76e991541d5ba919309b396785ea136e`: `full` 漏掉 `披露时间`
- parse 差异存在，但规模有限，且不能解释全部样例差异。
  - 本地 `metrics.json` 给出的 parse 失败数：
    - `academic_full_local = 11/1171 (0.0094)`
    - `academic_no_scv_local = 8/1171 (0.0068)`
  - parse/divergence 样例共 `9` 条。
  - 代表样例：
    - `019e72e570e8082c4ac9f537e221a12d`: `full` `no_json_found`，`no_scv` 虽成功解析但过泛化
    - `fc489b7b91f05e3eaee453063154d7ff`: `no_scv` `no_json_found`，`full` 保留了可比较结构
    - `81bb2a40ae7ec7af28e95ce40abc970e`: `full` parse fail，`no_scv` 解析成功但多事件时间链混乱
  - 证据来源：`logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047_metrics.json`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047_metrics.json`
- role schema / canonicalization 相关错误是真实存在的，而且能在样例与 run-level summary 两端对上。
  - `academic_full_local` summary 中可见：
    - `invalid_role:股份回购|事件时间 = 49`
    - `invalid_role:股东减持|被减持方 = 9`
  - `academic_no_scv_local` summary 中可见：
    - `invalid_role:股份回购|事件时间 = 75`
    - `invalid_role:企业收购|事件时间 = 8`
  - 对应样例：
    - `0a85c1c8ec6d4042be78784839cb4116`: `股份回购|事件时间 -> 回购完成时间`
    - `58567406e01800c4d1ff10835ffac74d`: `股东减持|被减持方 -> 股票简称`
    - `c45668926f8b7a09a9fff66254420147`: full/no_scv/base 都原始输出 `股份回购|事件时间`
  - 证据来源：`logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047_summary.json`, `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047_summary.json`

## What The Current Evidence Can Support
- 可以支持“full 与 no_scv 的差异主要体现在样例级错误模式，而不只是单个总指标数值”。
- 可以支持“样例差异至少包含三类来源”：
  - 额外角色或额外事件
  - 角色槽位混用或 canonicalization 相关错误
  - parse 是否成功，以及成功后是否把背景段落提升为主事件
- 可以支持“多主题、长文本、尾部拼接新闻”是最值得下一轮继续查的高风险样例类型”。
  - 代表样例：`a413e8c72bba3359cc9a5cf6cea4fc8b`, `81bb2a40ae7ec7af28e95ce40abc970e`, `0a85c1c8ec6d4042be78784839cb4116`
- 可以支持“base/API 参照物能帮助判断某个差异更像 isolated failure 还是跨模型共性”。
  - `66a55a...`: base 与 `api_zeroshot` 都跟 `full` 一致，`api_fewshot` 则过预测
  - `585674...`: base 原始层面和 `full` 一样依赖 canonicalization；两条 API 原始层面已直接正确
  - `a413e8...`: base 会把背景竞购放大得比 `no_scv` 更严重，而 API 两线更克制

## What The Current Evidence Does Not Support
- 不能支持“SCV 一定有效”或“SCV 一定无效”的终局判断。
  - 原因：当前仍是单种子视角，且样例级逆转是双向的 `13 vs 10`。
- 不能支持显著性结论。
  - 原因：用户已明确所有评估不能轻易下显著性判断；当前工作区主线仍是单 seed 审计。
- 不能支持“SCV 主要改善 hallucination”这一说法。
  - 原因：本地三组 JSONL 只有 `text_preview`，缺少完整原文，无法对 local run 做可靠的样例级 hallucination 复核。
  - 目前只能使用 run-level `hallucination_breakdown` 做方向提示。
- 不能支持“某条样例一定是 schema violation”的穷尽判断。
  - 原因：本轮没有本地原始 schema 文件，只能用 summary 中已有的 `schema_violation_breakdown` 做 candidate 对照。
- 不能支持“SCV 专门改善了 parse”这一单因解释。
  - 原因：parse 失败数确实有差异，但 `9` 条 parse divergence 不能解释全部 `13/10/816` 的样例分化。

## SCV May Mainly Interact With These Error Modes
- 可能与“抑制额外角色/额外事件”有关。
  - 依据：
    - `full` 独占正确样例里，`no_scv_role_extra` 最多
    - 样例 `2b9ed764...`, `66a55a...`, `0a85c1c8...`
- 可能与“相邻事件类型边界”有关。
  - 依据：
    - `66a55a...` 的 `企业收购 -> 企业融资`
    - `a413e8...` 的“破产主线 + 收购背景”混杂
- 可能与“长文本多事件整合”有关。
  - 依据：
    - `81bb2a40...` 同时含收购与质押
    - `019e72e...` 是聚合式减持新闻，容易把统计背景抬升成事件
- 但也有明显反证，说明不能只写单向叙事。
  - `58567406...`, `7de944ac...`, `76e99154...` 都是 `no_scv` 明显优于 `full`
  - 这些样例主要体现为角色槽位回填、时间角色识别、简单显式时间保留

## Items That Are Still Speculative
- “SCV 更偏 precision 还是 recall”目前只能做弱推测，不能定稿。
  - 样例层面看，`full` 的独占正确样例更常见于 `no_scv` 多报；但 `no_scv` 也有若干独占正确样例来自 `full` 漏报时间或错位角色。
- “SCV 是否专门减少 schema invalid role”也只能说有迹象。
  - `invalid_role:股份回购|事件时间` 在 `no_scv` summary 里比 `full` 更高，但本轮只拿到了有限的样例映射，不能直接推出总体机制。
- “SCV 是否改善了跨句推理”仍需更多样例。
  - `81bb2a40...` 与 `a413e8...` 支持这是重要问题，但还不足以证明改进方向。

## Next Evidence Needed
- 本地三组对应的完整原文，而不是只含 `text_preview` 的 JSONL。
  - 这样才能做样例级 hallucination 复核。
- 原始 schema 文件或等价的 event-role 定义表。
  - 这样才能把 schema violation 从 candidate 提升为正式判定。
- 更完整的 per-sample evaluator export。
  - 若能直接导出每条样例的 strict/relaxed/type TP-FP-FN，将大幅降低人工复核成本。
- 更多 seed 的同口径样例级对照。
  - 当前工作区主线是单 seed 审计，不能承载显著性或稳定性判断。
