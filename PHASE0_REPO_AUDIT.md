# PHASE0_REPO_AUDIT.md

用途：供 Codex CLI 或 subagents 先执行只读核查。完成后，把确认结果回填到 `OG-LANS_ExecPlan.md` 与 runbook。

## 0. 执行规则
- 只读核查优先，不先改代码。
- 不联网。
- 不猜命令、不猜路径。
- 找不到就写：`TODO: inspect repo` 或 blocker。

## 1. 数据加载入口
- 目标：确认 DuEE-Fin 的 train/dev/test、schema、样本字段。
- 先找：`src/oglans/data/adapter.py`
- 还要找：数据目录、schema 文件、相关配置文件
- 必须确认：
  - train/dev/test 实际路径
  - 文本字段名
  - `event_list` 字段名
  - 是否有句子/span/document id
- 若未找到：全局搜索 `DuEEFinAdapter` / `duee_fin` / `event_schema`

## 2. prompt 构造入口
- 目标：确认 prompt 如何构造、训练和推理是否共用、输出格式提示在哪里。
- 先找：`src/oglans/data/prompt_builder.py`
- 必须确认：
  - 是否存在 `ChinesePromptBuilder`
  - 是否已有 JSON 模板或 schema 注入位点
  - 新旧 prompt 是否可共存
- 若未找到：全局搜索 `PromptBuilder` / `prompt_builder` / `schema` / `json`

## 3. JSON 解析/修复入口
- 目标：确认 JSON parse / repair / postprocess 主入口。
- 先找：`src/oglans/utils/json_parser.py`
- 必须确认：
  - 主入口函数/类
  - 当前输入输出格式
  - 是否已有 schema 校验、alias、normalization
  - 是否允许 sidecar diagnostics
- 若未找到：全局搜索 `json_parser` / `parse_json` / `repair`

## 4. SCV 实现入口
- 目标：确认 SCV 调用点、on/off 切换方式、缓存/滑窗。
- 先找：`src/oglans/utils/scv.py`
- 必须确认：
  - SCV 在训练/推理/离线清洗中的位置
  - on/off 如何控制
  - 是否能保留 full-SCV 与 SCV-Lite 双路径
- 若未找到：全局搜索 `SCV` / `semantic consistency` / `nli`

## 5. 训练入口
- 目标：确认 baseline 的最小可执行路径。
- 先找：`main.py`
- 还要找：`UnslothDPOTrainerWrapper`、`UnslothSFTTrainerWrapper`
- 必须确认：
  - 模型/数据/输出目录如何指定
  - 是否有 checkpoint 恢复
  - 是否有 SCV 开关
- 若未找到：全局搜索 `TrainerWrapper` / `train` / `main`

## 6. 评测入口
- 目标：确认 dev strict_f1 及可靠性指标如何计算。
- 先找：`evaluate.py`
- 必须确认：
  - 输入预测格式
  - dev/test 如何区分
  - summary 字段名
  - 是否已有 parse/schema/hallucination 指标
- 若未找到：全局搜索 `strict_f1` / `schema_compliance` / `hallucination` / `evaluate`

## 7. API baseline 入口
- 目标：确认 `evaluate_api.py` 是否支持离线复评。
- 先找：`evaluate_api.py`
- 必须确认：
  - 是否只做在线调用
  - 是否能读取本地缓存/manifest/usage
  - 是否已有离线 API 预测文件
- 若未找到：全局搜索 `evaluate_api` / `manifest` / `usage` / `deepseek`

## 8. 消融与编排入口
- 目标：确认哪些自动化脚本能复用。
- 先找：`scripts/ablation_study.py`、`run_*_repro_suite.py`、`validate_academic_artifacts.py`
- 必须确认：
  - 当前支持的实验开关
  - 读取哪些 summary
  - 是否假定固定输出 schema
- 若未找到：全局搜索 `ablation` / `repro_suite` / `academic_artifacts`

## 9. 负样本 / 课程学习入口
- 目标：确认增强版是否能利用 schema 图、LANS、DS-CNS。
- 先找：`build_graph.py`、`src/oglans/utils/ds_cns.py`
- 必须确认：
  - 图缓存
  - 采样接口
  - competence / 扰动接口
  - 是否能接入 chosen/rejected 构造
- 若未找到：增强版不改训练目标

## 10. run 目录与结果留存
- 目标：确认仓库现有 run / outputs / logs / summary / manifest 风格。
- 必须确认：
  - run_id 规则
  - 预测文件名
  - summary 文件名
  - 是否已有 per-run manifest
- 若未找到：在 Phase 0 结束前定义最小留存规范并写入 runbook

## 11. Phase 0 完成判定
满足以下条件才可离开 Phase 0：
- baseline 入口已找到，或 blocker 已写明
- evaluate 入口已找到，或 blocker 已写明
- SCV 入口已找到，或 blocker 已写明
- 至少有一种 baseline 结果获取路径：直接运行或离线复评
- run 目录与结果留存规则已确认
