# Current Evidence Report

> 生成时间：2026-03-14  
> 工作区：`D:\zimmmtly\Desktop\og-lans-evidence`  
> 报告用途：供后续 ChatGPT / Gemini Deep Research 直接使用的结构化证据报告

## 1. 审计范围与任务边界
- 本轮工作是静态证据审计，不训练、不重跑、不伪造结论。
- 审计对象限定在当前工作区已有日志、summary、metrics、jsonl、manifest、samples、tensorboard、train 产物。
- 本报告的主比较对象是 5 个评估系统：
  - `academic_full_local`
  - `academic_no_scv_local`
  - `base_qwen4b_local`
  - `api_zeroshot_deepseek_chat`
  - `api_fewshot_deepseek_chat`
- 当前主线仍是单 seed 审计。
  - `academic` 两条线的评估产物内部一致地记录为 `3047`
  - `base` 与两条 `api` 线内部一致地记录为 `3407`
- 因此，本报告支持“事实审计”和“证据边界梳理”，不支持显著性结论，也不支持把单次 run 的样例观察升格为总体因果判断。
- 核心依赖文件：
  - `reports/evidence/run_identity_table.csv`
  - `reports/evidence/model_comparison_table.csv`
  - `reports/evidence/error_case_notes.md`
  - `reports/evidence/scv_evidence_memo.md`

## 2. 数据来源与文件映射
### 2.1 评估目录
- `logs/DuEE-Fin/eval_academic/20260305_130645_dev`
  - 用途：本地 academic 评估，全模块版本。
  - 关键文件：
    - `eval_results_dev_seed3047_summary.json`：总指标与误差分布
    - `eval_results_dev_seed3047_metrics.json`：补充指标、parse 统计、诊断字段
    - `eval_results_dev_seed3047.jsonl`：逐样本金标、预测、canonical、响应文本
    - `academic_summary.json` / `academic_summary.md`：展示层学术汇总
    - `run_manifest.json`：运行元信息、命令、checkpoint、seed

- `logs/DuEE-Fin/eval_academic/20260311_163743_dev`
  - 用途：本地 academic 评估，去 SCV 版本。
  - 关键文件类型与上面同构。

- `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev`
  - 用途：本地 base model 对照评估，`qwen_base_local` / `base_only` 组。
  - 关键文件：
    - `eval_results_summary.json`
    - `eval_results_metrics.json`
    - `eval_results.jsonl`
    - `run_manifest.json`

- `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808`
  - 用途：API zeroshot 基线。
  - 关键文件：
    - `eval_summary.json`：总指标、bootstrap CI、token usage、api stats
    - `eval_results.jsonl`：逐样本预测、usage、response_meta
    - `eval_report.txt`：人类可读摘要
    - `run_manifest.json`

- `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514`
  - 用途：API fewshot 基线。
  - 关键文件类型与 zeroshot 同构。

### 2.2 训练与辅助目录
- `logs/DuEE-Fin/samples`
  - 用途：训练侧 LANS/SCV 采样与过滤证据，不是最终评估结果。
  - 关键文件：
    - `lans_generated_samples.jsonl`
    - `lans_sampling_summary.json`
    - `scv_filtered_samples.jsonl`（仅 `main_s3047_v5` 存在）

- `logs/DuEE-Fin/tensorboard`
  - 用途：训练过程曲线与控制台镜像。
  - 关键文件：
    - `events.out.tfevents.*`：二进制 event 流
    - `experiment_*.log`：训练期文本日志

- `logs/DuEE-Fin/train`
  - 用途：训练运行身份、命令、状态、seed、完成情况。
  - 关键文件：
    - `run_manifest.json`
    - `run.log`

### 2.3 文件类型与用途总览
- `summary.json` / `eval_summary.json`
  - 主汇总层，优先用于总指标与总览判断。
- `metrics.json`
  - 指标细化层，优先用于 parse 统计与诊断块。
- `jsonl`
  - 逐样本证据层，优先用于错误样例分析。
- `run_manifest.json`
  - 身份与 provenance 层，优先用于 run identity 对照。
- `academic_summary.*` / `eval_report.txt`
  - 展示层辅助核对，不作为主数据源。

## 3. 运行身份对照与不一致项
### 3.1 统一身份主键
- 后续所有 join 一律使用 `system_id`，而不是目录名、checkpoint 名或单个 seed 字段。
- 规范化后的 5 个系统 ID 为：
  - `academic_full_local`
  - `academic_no_scv_local`
  - `base_qwen4b_local`
  - `api_zeroshot_deepseek_chat`
  - `api_fewshot_deepseek_chat`

### 3.2 seed 与身份来源字段
- `run_identity_table.csv` 显式保留了 5 类 seed 来源：
  - `dirname_seed`
  - `filename_seed`
  - `manifest_seed`
  - `summary_seed`
  - `internal_seed`
- 身份判定策略是：
  - `manifest-first`
  - `summary/metrics-second`
  - `dirname/filename only auxiliary`

### 3.3 已确认的不一致项
- `academic_full_local` 与 `academic_no_scv_local`
  - summary、metrics、文件名内部一致指向 `3047`
  - 对应训练侧目录名也都带 `s3047`
- `base_qwen4b_local` 与两条 `api` 线
  - summary/manifest 内部一致指向 `3407`
- `train/20260303_092232_main_s3047_v5/run_manifest.json`
  - 记录 `seed=3407`
  - 但实验名是 `main_s3047_v5`
  - 与 academic_full 的评估产物 `3047` 存在冲突
- `train/20260306_191921_A2_no_scv_s3047/run_manifest.json`
  - 命令显式包含 `--project.seed 3047`
  - manifest 字段仍写 `seed=3407`
  - `status=running`
  - 但 `run.log` 已写出成功完成
- `train/20260306_191921_A2_no_scv_s3047/run.log`
  - 头部仍显示 `LANS=True | SCV=True`
  - 与命令中的 `--algorithms.scv.enabled false` 冲突
- `analysis.protocol.evaluation.seeds`
  - 是协议配置，不是实际运行身份
  - 不能用于 run identity 归并

### 3.4 哪些字段可用，哪些不能单独用
- 可作为身份依据：
  - `system_id`
  - `run_dir`
  - `summary_file`
  - `metrics_file`
  - `jsonl_file`
  - `checkpoint_path_or_id`
  - `command`
  - `manifest_seed + summary_seed + internal_seed` 的组合
- 不能单独使用：
  - 目录名中的 `s3047` / `seed3407`
  - 单个 seed 字段
  - `analysis.protocol.*`
  - 训练 manifest 的 `seed` 字段，当它与评估本地产物冲突时

### 3.5 full / no_scv 标签的证据链
- `academic_full_local`
  - checkpoint 指向 `main_s3047_v5/checkpoint-660`
  - `train/.../main_s3047_v5/run.log` 含 `LANS=True | SCV=True`、`Loading SCV model`
  - `samples/main_s3047_v5/scv_filtered_samples.jsonl` 存在
  - `samples/main_s3047_v5/lans_sampling_summary.json` 记录 `scv_filtered_count=35537`
- `academic_no_scv_local`
  - checkpoint 指向 `A2_no_scv_s3047`
  - `train/.../A2_no_scv_s3047/run_manifest.json -> command` 含 `--algorithms.scv.enabled false`
  - `samples/A2_no_scv_s3047/lans_sampling_summary.json` 记录 `scv_filtered_count=0`
  - `samples/A2_no_scv_s3047/scv_filtered_samples.jsonl` 不存在
- 结论边界：
  - `full/no_scv` 标签是有证据链支撑的
  - 但这条证据链依赖多源联合判断，不能只看目录名或单个 manifest 字段

## 4. 模型/系统统一比较总表
### 4.1 统一比较对象
- 五个对象都统一到 `eval_split=dev`，主指标都记录为 `strict_f1`。
- 统一表主文件：`reports/evidence/model_comparison_table.csv`
- 字段映射说明文件：`reports/evidence/model_comparison_notes.md`

### 4.2 可直接比较的指标
- `primary_metric_name`
- `primary_metric_value`
- `precision`
- `recall`
- `f1`
- `exact_match_f1`
- `relaxed_f1`
- `type_f1`
- `parse_error_rate`
- `hallucination_rate`
- `schema_compliance_rate`

### 4.3 部分可比较的指标
- `bootstrap_ci_lower`
- `bootstrap_ci_upper`
- `token_usage_prompt`
- `token_usage_completion`
- `token_usage_total`
- `api_cost_or_stats`
- 这些字段只适合 API 内部诊断，不适合直接并入本地模型主质量对比。

### 4.4 暂时不可比较的指标
- `event_f1`
- `arg_f1`
- `schema_violation_rate`
- 当前工作区没有同语义、同口径、可直接引用的统一字段，因此统一表中保留 `NA`。

### 4.5 主要发现摘要
- `academic_full_local`
  - `strict_f1 = 0.3756`
  - `relaxed_f1 = 0.5387`
  - `type_f1 = 0.8475`
  - `parse_error_rate = 0.0094`
  - `hallucination_rate = 0.205`
  - `schema_compliance_rate = 0.7455`
- `academic_no_scv_local`
  - `strict_f1 = 0.3782`
  - `relaxed_f1 = 0.5406`
  - `type_f1 = 0.8525`
  - `parse_error_rate = 0.0068`
  - `hallucination_rate = 0.1913`
  - `schema_compliance_rate = 0.7276`
- `base_qwen4b_local`
  - `strict_f1 = 0.3649`
  - `relaxed_f1 = 0.5280`
  - `type_f1 = 0.8598`
  - `parse_error_rate = 0.0068`
  - `hallucination_rate = 0.2357`
  - `schema_compliance_rate = 0.6541`
- `api_zeroshot_deepseek_chat`
  - `strict_f1 = 0.4636`
  - `bootstrap_ci = [0.4486, 0.4774]`
  - `token_usage_total = 2506452`
  - `failed_calls = 0`
- `api_fewshot_deepseek_chat`
  - `strict_f1 = 0.4668`
  - `bootstrap_ci = [0.4537, 0.4800]`
  - `token_usage_total = 3744900`
  - `failed_calls = 0`

### 4.6 comparability 解释
- 本地三条线与 API 两条线在核心 strict/relaxed/type 指标上可以放入同一统一表。
- 但 API 额外拥有 `bootstrap_ci`、`token_usage`、`api_stats`，这些只应被视为 API 侧附加诊断。
- `bootstrap_ci` 的字段名在 local/base 与 API summary 里都存在，但 local/base 的值是 `null`；只有 API 提供了可用的非空 CI。

## 5. 当前最可靠的关键事实
- full vs no_scv 的双向逆转不是单边碾压。
  - 样例级 strict 对照结果是：
    - `full correct / no_scv wrong = 13`
    - `no_scv correct / full wrong = 10`
    - `both wrong but different = 816`
  - 证据来源：`reports/evidence/scv_evidence_memo.md`

- parse 差异存在，但不能解释全部差异。
  - `academic_full_local` 的 parse 失败数是 `11/1171 (0.0094)`
  - `academic_no_scv_local` 的 parse 失败数是 `8/1171 (0.0068)`
  - 这能解释一部分样例分歧，但不能解释全部 `13/10/816` 的双向逆转结构。
  - 证据来源：`reports/evidence/scv_evidence_memo.md`, `reports/evidence/model_comparison_table.csv`

- role/schema/canonicalization 错误真实存在，而且能在 summary 与样例两端对上。
  - run-level summary 中可见：
    - `invalid_role:股份回购|事件时间`
    - `invalid_role:股东减持|被减持方`
    - `invalid_role:企业收购|事件时间`
  - 样例侧可对上的代表 case：
    - `0a85c1c8ec6d4042be78784839cb4116`
    - `58567406e01800c4d1ff10835ffac74d`
    - `c45668926f8b7a09a9fff66254420147`
  - 证据来源：`reports/evidence/error_case_notes.md`, `reports/evidence/scv_evidence_memo.md`

- API 指标中的 `bootstrap_ci` 与 `token_usage` 只能用于 API 内部诊断，不应与本地主质量指标混比。
  - local/base summary 虽有同名 `bootstrap_ci` 键，但值为 `null`
  - 只有 API 行有可用 CI、真实 token usage 和 api stats
  - 证据来源：`reports/evidence/model_comparison_notes.md`, `reports/evidence/model_comparison_table.csv`

## 6. SCV 模块证据判断
### 6.1 当前证据能支持什么
- 能支持“full 与 no_scv 的差异主要落在样例级错误模式，而不只是一个总指标数值”。
- 能支持“样例差异至少包括三类来源”：
  - 额外角色或额外事件
  - 角色槽位混用与 canonicalization 相关错误
  - parse 是否成功，以及成功后是否把背景段落抬升成主事件
- 能支持“多主题、长文本、尾部拼接新闻”是当前最值得继续追的高风险样例类型。
- 能支持“base/API 参照物能帮助判断某个错误是 isolated failure 还是跨模型共性”。

### 6.2 当前证据不能支持什么
- 不能支持任何关于 SCV 的终局判断。
- 不能支持显著性结论。
- 不能支持“SCV 主要改善 hallucination”这一表述。
  - 原因：本地三组只保留 `text_preview`，没有完整原文。
- 不能支持“某条样例一定是 schema violation”的穷尽判定。
  - 原因：本轮没有原始 schema 文件，只能做 candidate 对照。
- 不能支持“SCV 专门改善 parse”的单因解释。
  - 原因：parse 差异存在，但规模不足以解释全部样例结构。

### 6.3 SCV 可能主要作用于哪些错误模式
- 可能与“抑制额外角色/额外事件”有关。
  - `full` 独占正确样例里，`no_scv_role_extra` 最多。
- 可能与“相邻事件类型边界”有关。
  - 代表样例：`66a55a...` 的 `企业收购 -> 企业融资`
- 可能与“长文本多事件整合”有关。
  - 代表样例：`81bb2a40...`, `019e72e...`, `a413e8...`
- 也可能与“角色槽位回填和时间角色识别”有关。
  - 反例样例显示 `no_scv` 在部分简单显式时间场景上优于 `full`

### 6.4 仍然只是猜测的部分
- “SCV 更偏 precision 还是 recall”目前只能做弱推测。
- “SCV 是否系统性减少 invalid role”目前只能说有迹象，不能定稿。
- “SCV 是否改善跨句推理”仍需更多样例与更细粒度 evaluator 证据。
- “SCV 是否主要影响 hallucination”当前证据链不足。

## 7. 典型错误样例与论文 case study 候选
### 7.1 事件类型边界
- `66a55a53915fbdf74bcd5e14f863d40d`
  - 适合支撑的论点：`企业收购` 与 `企业融资` 在相邻语义下会发生事件类型翻转。
- `a413e8c72bba3359cc9a5cf6cea4fc8b`
  - 适合支撑的论点：多主题新闻中，背景竞购信息可能被错误提升为主事件。

### 7.2 role / schema / canonicalization
- `58567406e01800c4d1ff10835ffac74d`
  - 适合支撑的论点：role alias 错误与 canonicalization 修正之间存在清晰对应关系。
- `0a85c1c8ec6d4042be78784839cb4116`
  - 适合支撑的论点：`事件时间 -> 回购完成时间` 属于可观测的 invalid role / canonical rewrite 模式。
- `df0ae19741f364a0a8430dc98e44f6da`
  - 适合支撑的论点：同一类 role alias 错误会在 full/no_scv 两条本地线上同时出现，并不只属于单一变体。

### 7.3 parse 与长文本多事件
- `019e72e570e8082c4ac9f537e221a12d`
  - 适合支撑的论点：parse fail 与“解析成功但过泛化”是两类不同错误风险。
- `81bb2a40ae7ec7af28e95ce40abc970e`
  - 适合支撑的论点：多事件、多日期、跨句整合失败会把 parse、role、time 三类错误叠加在一条样例上。

## 8. 当前实验的证据边界与风险
- 单 seed 限制
  - 当前主线只能支持单次 run 审计，不能外推稳定性或显著性。
- 缺少完整原文
  - 本地三组 JSONL 只有 `text_preview`，导致 hallucination 只能做保守推断，不能做正式样例级复核。
- 缺少原始 schema 文件
  - schema violation 只能做 breakdown-guided candidate 对照，不能做穷尽重判。
- 缺少更细粒度 per-sample evaluator export
  - 当前只能通过 strict triplet 近似对齐样例，不如直接有 per-sample TP/FP/FN 稳定。
- 训练 manifest / status / seed 等元数据不一致
  - 会影响 provenance 清晰度，也会提高后续 Deep Research 理解实验链路的成本。
- samples / train / eval 三侧证据链需要联合阅读
  - 目录名或单一 manifest 字段不能独立承载 full/no_scv 身份判断。

## 9. 后续 Deep Research 最需要吃进去的信息摘要（500-1000字）
本项目是对 DuEE-Fin 金融事件抽取实验的一次静态证据审计，工作区只保留已有训练、采样、评估日志与产物，没有重新训练或重跑评测。当前需要重点理解 5 个评估对象：两个本地 academic run（`academic_full_local` 与 `academic_no_scv_local`）、一个本地 base run（`base_qwen4b_local`），以及两个 DeepSeek API run（zeroshot / fewshot）。运行身份已经被显式统一到 `system_id`，但目录名、文件名、manifest、summary 内部 seed 并不总是一致；例如 academic 评估产物一致写成 `3047`，base/api 一致写成 `3407`，而训练侧 manifest 又存在目录名、命令 seed、manifest seed、status 互相冲突的情况。因此后续研究必须以 `run_identity_table.csv` 的 `system_id` 为 join key，不能只凭目录名或单个 seed 字段判断实验身份。

在统一指标层面，5 个对象都能对齐到 `dev` split 和 `strict_f1` 主指标，核心 strict/relaxed/type 指标可直接比较，但 API 特有的 `bootstrap_ci`、`token usage`、`api_stats` 只能做 API 内部诊断，不能与本地主质量指标混在一起解释。当前单次 run 中，`academic_full_local` 与 `academic_no_scv_local` 的 strict_f1 很接近，但样例级对照显示差异并非单边碾压：`full correct / no_scv wrong = 13`，`no_scv correct / full wrong = 10`，`both wrong but different = 816`。这意味着讨论 SCV 时不能只盯总分，还要看错误模式。当前最可靠的样例证据表明，差异主要分布在额外角色/额外事件、相邻事件类型翻转、角色槽位混用、canonicalization 可修与不可修的边界、以及多主题长文本下的 parse 与跨句整合失败。parse 差异确实存在，但规模有限，不能解释全部差异；role/schema/canonicalization 错误也是真实存在的，run-level summary 中的 `invalid_role:*` breakdown 能与具体样例对上。与此同时，这些证据还不足以支持任何关于 SCV 的终局判断，也不足以支持显著性、系统性 hallucination 改善、或穷尽的 schema 违规结论。后续 Deep Research 最值得追问的问题是：1）SCV/LANS 类模块在事件抽取里更常被描述为抑制 over-prediction、修正 schema-role 错配，还是改善跨句推理；2）金融事件抽取文献中如何处理 role alias、canonicalization 与 schema filtering 的边界；3）当前样例分化更像已知方法论现象，还是当前实现/标注口径的局部副作用。

## 10. 推荐的后续动作
1. `本地静态`：基于已有 `jsonl` 继续整理 full/no_scv 的 per-sample strict 对照表，把 `13/10/816` 进一步分解成更稳定的错误标签统计。
2. `本地静态`：从 summary breakdown 出发，系统筛出 `invalid_role:*` 与 `canonical_role_rewrites>0` 的交集样例，形成更干净的 schema/canonicalization 子集。
3. `本地静态`：把 `samples / train / eval` 三侧证据链再压缩成一张 provenance 图，专门解释 full/no_scv 标签为何成立以及哪里存在元数据冲突。
4. `联网 Deep Research`：检索 SCV/LANS、schema-constrained event extraction、role alias normalization、financial event extraction error analysis 的近年论文与公开实现。
5. `联网 Deep Research`：检索与“过预测抑制”“事件类型边界混淆”“长文本多事件整合失败”相关的错误分析框架，作为论文讨论区参考。
6. `联网 Deep Research`：检索 DeepSeek 类 API 事件抽取或结构化抽取任务中 few-shot 与 zeroshot 在 token 开销、schema 合规率、bootstrap CI 报告方式上的惯例。
7. `后续补实验/补数据`：补充完整原文与原始 schema 文件，否则 hallucination 与 schema violation 都无法做严格样例级复核。
8. `后续补实验/补训练`：若要回答稳定性、显著性或“SCV 是否系统性改善某类错误”，必须补多 seed 同口径实验与更细粒度 per-sample evaluator export。

## 本报告依赖的中间文件清单
- `reports/evidence/run_identity_table.csv`
- `reports/evidence/run_identity_notes.md`
- `reports/evidence/model_comparison_table.csv`
- `reports/evidence/model_comparison_notes.md`
- `reports/evidence/error_case_notes.md`
- `reports/evidence/scv_evidence_memo.md`
