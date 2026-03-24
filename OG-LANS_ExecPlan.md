# OG-LANS 项目修正实验 ExecPlan（供 Codex CLI 执行）

配套文件：
- `执行说明.md`：本计划的执行说明
- `RUNBOOK_TEMPLATES.md`：run manifest、阶段记录、结果表模板

本文件只允许基于以下两份输入文档与当前工作树编写和执行：
- `plan_A_首选_结构约束解码_证据对齐_v2.md`
- `og-lans-deep-research-report.md`

凡是这两份文档没有明确说明、且无法从当前工作树直接确认的内容，一律写成：
- `TODO: inspect repo`
- `待 Codex CLI 进入仓库后确认`
- `若仓库不支持则降级为……`

禁止事项：
- 禁止假设仓库中一定存在某个脚本、配置键、函数名、目录结构、checkpoint、命令参数
- 禁止把 test 用于 F1 汇报
- 禁止把“结果引用/格式对齐”伪装成“直接复现”

# 0. 使用说明

## 0.1 用途
本文档是 Codex CLI 的主执行计划。目标不是写综述，而是驱动当前工作树完成以下任务：
1. 核查仓库与现有实验入口。
2. 复现或确认当前 baseline。
3. 实施首选改造方案的 MVP。
4. 按条件推进增强版。
5. 组织对比实验并对齐评测。
6. 生成主表、消融表、效率表、可靠性表所需素材。

## 0.2 执行顺序
按以下顺序执行，不得越级：
1. Phase 0：仓库核查与事实确认
2. Phase 1：复现/确认当前基线
3. Phase 2：SCV on/off 与当前主线对比
4. Phase 3：首选方案 MVP
5. Phase 4：首选方案增强版
6. Phase 5：DeepSeek API baseline 对齐
7. Phase 6：外部/经典 baseline 接入或结果对齐
8. Phase 7：汇总主表、消融表、效率表、可靠性表

## 0.3 何时更新本文档
以下情况必须更新本文档或其配套 runbook：
- Phase 0 找到真实入口文件、真实配置键、真实输出目录后。
- 任一 Phase 完成后。
- 出现 blocker、回退、降级、路径变更时。
- 某个 `TODO: inspect repo` 被仓库事实消除时。

更新要求：
- 不覆盖旧结论；保留“原假设 → 仓库事实 → 处理结果”的痕迹。
- 对每个 Phase 记录“已完成 / 未完成 / blocker”。
- 明确 run_id、输出目录、结果来源类型。

## 0.4 如何处理中断后恢复
中断恢复时执行以下固定流程：
1. 重读本文件与 `执行说明.md`。
2. 找到最后一个被明确标记为“已完成”的 Phase。
3. 验证该 Phase 产物仍存在：预测文件、summary、manifest、日志、评测输出。
4. 若产物缺失或无法对齐，回退到上一个可验证完成点。
5. 只从最近的“可验证完成点”继续，不从记忆继续。

## 0.5 何时必须停止并回报 blocker
满足任一条件立即停止，并回写 blocker：
- 无法确认 dev 集定位或评测口径。
- 无法确认 baseline 的训练/推理/评测入口。
- 无法确认预测文件格式，导致 strict_f1 无法对齐。
- 需要联网/新下载才能继续，但当前任务禁止联网。
- 关键改动破坏 baseline 回退路径，且短时间内无法恢复。
- 结果无法追溯到 run、配置、代码状态或本地产物。

# 1. 任务定义与完成标准

## 1.1 最终目标
围绕 DuEE-Fin 中文金融篇章级事件抽取项目，在当前 OG-LANS 主线上产出一套可直接推进的实验执行链路，覆盖：
- baseline 核查与复评
- SCV on/off 对比
- 首选方案 MVP 与增强版
- DeepSeek API baseline 对齐
- 外部/经典 baseline 状态确认
- 面向论文主表、消融表、效率表、可靠性表的素材汇总

## 1.2 主指标
- `strict_f1`（只允许基于 dev）

## 1.3 辅助指标
- `relaxed_f1` 或 `type_f1`（若脚本支持）
- `parse_error_rate`
- `schema_compliance_rate`
- `hallucination_rate`
- `grounding_rate`
- `ungrounded_argument_rate`
- `SCV` 调用次数
- `SCV` wall-clock 占比（若可统计）
- 训练耗时
- 推理耗时
- 显存占用（若可获得）

## 1.4 Done Criteria
满足以下条件即可视为本轮实验完成：
1. Phase 0 已把关键入口核实到文件级，或 blocker 已被明确记录。
2. 当前 OG-LANS 主线 baseline 已在 dev 上复现、复评或对齐到现有本地可审计产物。
3. `SCV-off` 与 `SCV-on` 的 dev 对比已完成，或无法完成的原因已被固定记录。
4. 新方案 MVP 已实现并完成 dev 评测。
5. 增强版是否执行已有明确依据：执行 / 不执行 / 降级。
6. DeepSeek API baseline 已完成离线对齐，或被明确标记为“结果引用/格式对齐”。
7. ProCNet / RAAT / PTPCG / GIT 四项都已拿到状态标签（A/B/C）及其依据。
8. 四类表格素材均可追溯到本地 run 或结果来源说明。
9. 最终交付物中显式写明：**dev 才用于 F1 对比；test 不用于 F1 汇报**。

## 1.5 口径硬约束
- 所有 F1 对比只允许使用 dev。
- test 只可用于预测文件生成、提交或演示，不得进入 F1 表。
- 若仓库已有 test 评测逻辑，也不得把 test F1 混入主表或消融表。

# 2. 输入依据

以下只列出两份输入文档中能够直接确认、且与执行相关的事实。

1. 当前工程主线已经形成：`schema 图驱动负样本构造 → LANS 自适应难度调度 → SCV 语义一致性过滤 → IPO/DPO 偏好优化 → 结构化输出与可靠性评测`，并存在本地评测与 API 评测两条路径。[来源：`og-lans-deep-research-report.md`]
2. DuEE-Fin 当前快照统计为：train=`7015` 文档 / `9498` 事件 / `48891` 论元；dev=`1171` 文档 / `1533` 事件 / `7915` 论元；test split 没有金标 `event_list`，因此不能用于 F1 汇报。[来源：`og-lans-deep-research-report.md`]
3. 当前 schema 文件 `duee_fin_event_schema.json` 覆盖 `13` 类事件、`92` 个角色。[来源：两份输入文档]
4. 当前问题包括：`SCV` 效果不强、`SCV` 训练时间长、`DeepSeek API baseline` 在 F1 上高于项目方法约 `10–15%`、且其可靠性更强。[来源：用户任务说明 + `og-lans-deep-research-report.md`]
5. 本轮首选方向不是继续强化重 SCV，而是：结构约束解码、Schema-Aware 修复、证据对齐 / grounding、SCV-Lite。[来源：用户任务说明 + `plan_A_首选_结构约束解码_证据对齐_v2.md`]
6. 已被明确提到的关键模块包括：
   - `src/oglans/data/adapter.py`
   - `src/oglans/data/prompt_builder.py`
   - `src/oglans/utils/json_parser.py`
   - `src/oglans/utils/scv.py`
   - `build_graph.py`
   - `src/oglans/utils/ds_cns.py`
   - `main.py`
   - `UnslothDPOTrainerWrapper`
   - `UnslothSFTTrainerWrapper`
   - `evaluate.py`
   - `evaluate_api.py`
   - `scripts/ablation_study.py`
   - `run_*_repro_suite.py`
   - `validate_academic_artifacts.py`
   但这些路径、类名、入口关系是否仍与当前工作树一致，全部待仓库确认。[来源：`og-lans-deep-research-report.md`]
7. 当前可审计快照中，最佳本地 `strict_f1` 来自 `SCV-off` 而非 `SCV-on`；该结论必须在当前工作树内重新核查，而不是直接当作已验证事实写入结果结论。[来源：`plan_A_首选_结构约束解码_证据对齐_v2.md`]
8. 明显误差源被归纳为：角色名漂移、边界/归一化错误、多值角色合并成一个字符串、生成 schema 内但文本不支持的伪事件。[来源：`plan_A_首选_结构约束解码_证据对齐_v2.md`]
9. `evaluate.py` 已被描述为支持 strict/relaxed/type、解析诊断、schema compliance、hallucination 等评测；`grounding_rate` / `ungrounded_argument_rate` 需要新增。[来源：两份输入文档]
10. ProCNet、RAAT、PTPCG、GIT 是本轮需要进入主表设计的 baseline 名单，但是否能在当前工作树直接执行仍待仓库确认。[来源：用户任务说明 + `og-lans-deep-research-report.md`]

# 3. 约束与非目标

## 3.1 固定约束
- 基座模型：`Qwen-4B-instruct-2507`
- 环境：`torch2.6.0+cu118`
- 硬件：单卡 `A6000`
- 数据集：`DuEE-Fin`
- 当前工程主线：`OG-LANS`
- F1 只允许基于 dev；test 不得用于 F1 汇报。

## 3.2 不做什么
- 不以“继续强化全量重 SCV”为本轮主线。
- 不联网，不下载新仓库，不新增文档外论文。
- 不伪造外部 baseline 的直接复现。
- 不在未核实入口/路径/配置时直接做大改动。
- 不为追求增强版而跳过 MVP。

## 3.3 不能假设什么
- 不能假设某个脚本、配置键、函数名、输出目录一定存在。
- 不能假设 `prompt_builder.py`、`json_parser.py`、`scv.py`、`evaluate.py` 的当前实现与文档完全一致。
- 不能假设仓库里有 ProCNet / RAAT / PTPCG / GIT 的本地代码。
- 不能假设仓库里保存了 DeepSeek API 的离线缓存或 manifest。
- 不能假设当前推理栈支持真正的 grammar-constrained decoding。

## 3.4 若无仓库支持则不做
- 真正的解码级结构约束：若推理栈不支持，则降级为 prompt 侧 skeleton + parser 侧修复。
- teacher 蒸馏 / 偏好优化增强：若当前工作树没有离线 teacher 产物或训练入口，则不做。
- 外部 baseline 直接复现：若当前工作树没有本地代码或离线结果，则仅保留结果引用/格式对齐。
- 显存占用强制统计：若现有日志或脚本不支持，则记为 `NA`，不阻塞主线。

# 4. 仓库核查清单（Phase 0）

以下 checklist 必须先执行，执行结果必须记录到 runbook。

- [ ] **数据加载入口**
  - 目标：确认 DuEE-Fin 的数据读取、split 命名、schema 加载方式、样本字段。
  - 需要 inspect 的文件/目录：`src/oglans/data/adapter.py`；`TODO: inspect repo` 中所有可能的数据目录和配置目录。
  - 预期确认内容：train/dev/test 实际路径；文本字段名；`event_list` 字段名；是否保留句子/span 信息；schema 文件的定位方式。
  - 若未找到则怎么办：全局搜索 `DuEEFinAdapter`、`duee_fin`、`event_schema`；仍未找到则停止 Phase 1，并写 blocker。

- [ ] **prompt 构造入口**
  - 目标：确认 prompt 如何构造，训练与推理是否共用，以及输出格式约束在哪里定义。
  - 需要 inspect 的文件/目录：`src/oglans/data/prompt_builder.py`；`TODO: inspect repo` 中其他 prompt 模板文件。
  - 预期确认内容：是否已有 JSON 模板；是否区分 train/infer prompt；是否有 schema 注入位置；是否可安全加入 skeleton。
  - 若未找到则怎么办：全局搜索 `PromptBuilder`、`prompt_builder`、`schema`、`json`；若仍未找到，则把 MVP 的 prompt 侧改动降级为 parser 侧改动。

- [ ] **JSON 解析/修复入口**
  - 目标：确认 JSON 解析、修复、容错、非法字段处理在哪里发生。
  - 需要 inspect 的文件/目录：`src/oglans/utils/json_parser.py`；`TODO: inspect repo` 中其他 parser / repair / postprocess 文件。
  - 预期确认内容：主入口函数/类；当前输入输出格式；schema 校验是否已存在；是否已有 alias/normalization；是否支持 sidecar diagnostics。
  - 若未找到则怎么办：全局搜索 `json_parser`、`parse_json`、`repair`；仍未找到则停止 MVP，并写 blocker。

- [ ] **SCV 实现入口**
  - 目标：确认 SCV 在训练、推理、离线清洗中的调用点，以及 on/off 的真实切换方式。
  - 需要 inspect 的文件/目录：`src/oglans/utils/scv.py`；相关训练/推理入口；`TODO: inspect repo`。
  - 预期确认内容：SCV 输入输出；缓存/滑窗/阈值；调用位置；如何保留 full-SCV 与 SCV-Lite 双路径。
  - 若未找到则怎么办：全局搜索 `SCV`、`semantic consistency`、`nli`；仍未找到则 Phase 2 标记 blocker。

- [ ] **训练入口**
  - 目标：确认 baseline 的训练/推理主入口与配置流。
  - 需要 inspect 的文件/目录：`main.py`、`UnslothDPOTrainerWrapper`、`UnslothSFTTrainerWrapper`、`TODO: inspect repo` 中 train/runner/config 相关位置。
  - 预期确认内容：如何指定模型、数据、输出目录、SCV 开关、评测钩子、checkpoint 恢复。
  - 若未找到则怎么办：全局搜索 `TrainerWrapper`、`train`、`main`；若仍无法确定，则停止 Phase 1，并写 blocker。

- [ ] **评测入口**
  - 目标：确认 `strict_f1`、`relaxed_f1`/`type_f1`、`parse_error_rate`、`schema_compliance_rate`、`hallucination_rate` 的实现位置与 summary 输出位置。
  - 需要 inspect 的文件/目录：`evaluate.py`；`TODO: inspect repo` 中 metrics / report / summary 相关位置。
  - 预期确认内容：输入预测格式；是否区分 dev/test；summary 字段名；是否支持扩展 grounding 指标。
  - 若未找到则怎么办：全局搜索 `strict_f1`、`schema_compliance`、`hallucination`、`evaluate`；仍未找到则停止所有 F1 汇总任务。

- [ ] **API baseline 入口**
  - 目标：确认 `evaluate_api.py` 是否支持读取离线缓存/manifest/response，而不是只做在线调用。
  - 需要 inspect 的文件/目录：`evaluate_api.py`；`TODO: inspect repo` 中 `api` / `manifest` / `usage` / `cache` 相关位置。
  - 预期确认内容：离线复评入口；响应缓存格式；usage/retries 记录格式；是否能在不联网条件下重评分。
  - 若未找到则怎么办：全局搜索 `evaluate_api`、`manifest`、`usage`、`deepseek`；若无离线产物，则本轮只保留结果引用/格式对齐。

- [ ] **消融脚本入口**
  - 目标：确认自动化是否能复用来跑 SCV on/off、MVP、增强版消融。
  - 需要 inspect 的文件/目录：`scripts/ablation_study.py`、`run_*_repro_suite.py`、`validate_academic_artifacts.py`。
  - 预期确认内容：支持哪些实验开关；读取哪些 summary；是否假定固定输出 schema。
  - 若未找到则怎么办：全局搜索 `ablation`、`repro_suite`、`academic_artifacts`；若自动化不可用，则改为手工 manifest 汇总。

- [ ] **负样本 / 课程学习入口**
  - 目标：确认 `schema 图`、`LANS`、`DS-CNS` 的实现位置，为增强版做准备。
  - 需要 inspect 的文件/目录：`build_graph.py`、`src/oglans/utils/ds_cns.py`。
  - 预期确认内容：图缓存、采样接口、competence 调度、扰动接口、是否能接入 chosen/rejected 构造。
  - 若未找到则怎么办：增强版不改训练目标，只做推理/后处理增强。

- [ ] **run manifest / summary / outputs 目录结构**
  - 目标：确认结果产物的现有保存约定。
  - 需要 inspect 的文件/目录：`TODO: inspect repo` 中 `runs/`、`outputs/`、`results/`、`summary/`、`manifest/`、`logs/` 等。
  - 预期确认内容：run_id 规则；summary 文件名；预测文件名；是否已有 per-run manifest；是否已有成本/耗时记录。
  - 若未找到则怎么办：在不破坏现有风格的前提下定义最小留存规范，并把该决策写入 runbook。

# 5. 总体实验路线图

## Phase 0：仓库核查与事实确认  [必须执行]
- 输入：当前工作树、两份输入文档、本文件
- 输出：入口地图、事实回填、blocker 列表、最小留存规范
- 必要动作：
  1. 完成第 4 节 checklist。
  2. 确认 dev/test 的真实边界与评测口径。
  3. 确认 baseline、SCV、评测、API baseline、消融脚本的真实位置和调用关系。
  4. 确认当前工作树的结果目录约定。
- 验证动作：
  - 能指出关键入口文件。
  - 能说明哪些内容已确认、哪些仍是 `TODO: inspect repo`。
- 成功标准：关键入口都已找到，或每个缺口都已有 blocker / fallback。
- 失败后的 fallback：若训练入口或评测入口无法确认，则不进入修改阶段，只产出 blocker 说明。

## Phase 1：复现/确认当前基线  [必须执行]
- 输入：Phase 0 确认后的 baseline 入口、dev 评测入口、现有 checkpoint / 预测产物（若有）
- 输出：当前 OG-LANS 主线 baseline 的 dev 结果记录
- 必要动作：
  1. 优先复用仓库已有入口跑 baseline；若命令未知，则 `TODO: inspect repo` 后回填真实命令。
  2. 若训练成本过高或入口不完整，优先复评现有 checkpoint / 预测文件 / summary。
  3. 为该 baseline 保存配置快照、启动命令、代码改动摘要、输出目录。
- 验证动作：
  - `evaluate.py` 能对 baseline 预测产物给出 dev 结果。
  - 结果输出中能明确 split=dev。
- 成功标准：拿到一个可审计的 baseline 结果（本地执行或本地离线复评）。
- 失败后的 fallback：若训练和预测都无法运行，但仓库有旧 summary，则只保留“本地旧结果复核”；若无可复核产物，则停止后续表格填数。

## Phase 2：SCV on/off 与当前主线对比  [必须执行]
- 输入：Phase 1 baseline；Phase 0 确认的 SCV 切换入口
- 输出：`SCV-off` 与 `SCV-on` 的 dev 对比结果与成本记录
- 必要动作：
  1. 找到 SCV 的真实开关或最小可控切换点。
  2. 在其他条件尽量保持一致的前提下，对比 `SCV-off` 与 `SCV-on`。
  3. 记录 `strict_f1`、`parse_error_rate`、`schema_compliance_rate`、`hallucination_rate`、`SCV` 调用次数、`SCV` wall-clock 占比（若可得）。
  4. 保留 full-SCV 原路径，为后续 SCV-Lite 对比提供参照。
- 验证动作：
  - 两组实验的唯一差异是 SCV 状态，或差异点已被写清。
  - 两组结果都来自同一 dev 评测口径。
- 成功标准：确认当前工作树中“SCV-off / SCV-on”谁更优、成本如何。
- 失败后的 fallback：若无法重跑两组，则至少对齐现有离线产物；若没有可对齐产物，则把该结论保留为待核实。

## Phase 3：首选方案 MVP  [必须执行]
- 输入：Phase 0–2 确认后的入口事实与 baseline 对比
- 输出：MVP 代码改动、dev 评测结果、最小错误分析
- 必要动作：
  1. 在 `prompt_builder.py` / `json_parser.py` / `scv.py` / `evaluate.py` 上实现 MVP。
  2. 保留 baseline 可回退路径。
  3. 先做小样本 smoke check，再做完整 dev 评测。
- 验证动作：
  - 预测结果可解析。
  - 非法角色可清理或被记录。
  - grounding 诊断可生成。
  - `evaluate.py` 可在 dev 上完成评测。
- 成功标准：MVP 在 dev 上可运行、可评测、可回退；并且至少在可靠性指标或 parse/schema 指标上可见改进。
- 失败后的 fallback：
  - 若解码级约束不可接入，则降级为 `prompt skeleton + parser repair + grounding + SCV-Lite`。
  - 若 grounding 无法安全写回预测结构，则写入 sidecar log，不阻塞主预测 JSON。
  - 若 prompt 改动与现有 checkpoint 严重失配，则先只启用 parser 侧 MVP。

## Phase 4：首选方案增强版  [可选增强]
- 输入：已通过 Go 规则的 MVP
- 输出：增强版结果，或“不执行增强版”的明确说明
- 必要动作：
  1. 在 MVP 成功后再引入更严格的结构约束。
  2. 增强 schema-aware repair。
  3. 扩大 SCV-Lite 触发策略。
  4. 若仓库支持且资源允许，再考虑 teacher / preference 结合。
- 验证动作：
  - 与 MVP 做同口径 dev 对比。
  - 记录复杂度、耗时、收益差异。
- 成功标准：增强版相对 MVP 有明确收益或明确的负收益结论。
- 失败后的 fallback：增强版只进入附录或 backlog，主线保留 MVP。

## Phase 5：DeepSeek API baseline 对齐  [必须对齐；运行仅限离线产物]
- 输入：`evaluate_api.py` 与当前工作树中已有的 API 预测 / manifest / usage / summary（若有）
- 输出：离线 API baseline 对齐结果，或“结果引用/格式对齐”说明
- 必要动作：
  1. 先确认 `evaluate_api.py` 是否支持本地离线重评分或读取缓存。
  2. 只处理当前工作树已有的本地产物；不得发起新的在线请求。
  3. 若存在离线 API 输出，则按 dev 口径与本地 baseline 对齐。
- 验证动作：
  - 结果来自 dev，且可追溯到本地文件。
  - usage / retries / manifest 若存在，已被保留。
- 成功标准：得到一个离线可审计的 API baseline 结果。
- 失败后的 fallback：若无本地离线产物，则标记为“结果引用/格式对齐”，不得伪装成已运行。

## Phase 6：外部/经典 baseline 接入或结果对齐  [若仓库支持则执行]
- 输入：当前工作树中的 baseline 代码、离线预测、summary（若有）
- 输出：ProCNet / RAAT / PTPCG / GIT 的状态与结果来源说明
- 必要动作：
  1. 清点当前工作树是否已经包含这些 baseline 的本地实现或离线结果。
  2. 若本地可运行，则执行并对齐当前 dev 评测口径。
  3. 若只有离线输出，则做格式转换与复评。
  4. 若什么都没有，则明确降级为“结果引用/格式对齐”。
- 验证动作：
  - 每个 baseline 都有状态：A / B / C。
  - 每个填入表格的数值都有来源说明。
- 成功标准：四个 baseline 全部拿到明确状态与占位。
- 失败后的 fallback：没有本地 artefact 的 baseline 一律按 C 处理。

## Phase 7：汇总主表、消融表、效率表、可靠性表  [必须执行]
- 输入：Phase 1–6 的 summary、manifest、预测文件、错误统计
- 输出：论文表格素材、总结文件、失败与风险说明
- 必要动作：
  1. 汇总每个 run 的指标与 provenance。
  2. 区分“本地执行”“本地离线复评”“结果引用/格式对齐”。
  3. 对所有表格显式标注 dev-only F1。
  4. 生成失败与风险说明、论文主表建议、附录消融建议、后续 backlog。
- 验证动作：
  - 每一格都能追溯到 run_id、结果目录或来源说明。
  - 主表与消融表不混入 test F1。
- 成功标准：四类表格和配套总结完整可交付。
- 失败后的 fallback：未跑通的条目保留 `TBD`，并写明阻塞原因与当前状态。

# 6. 对比实验主表设计

| 实验项 | 实验目的 | 执行状态分类 | 依赖条件 | 输出物 | 关键指标 | 风险与降级方案 |
|---|---|---|---|---|---|---|
| 当前 OG-LANS 主线 | 确认当前仓库主线 baseline | A. 必须在仓库中执行 | 已确认训练/推理/评测入口，或已有本地可复评产物 | baseline summary、预测文件、manifest | dev `strict_f1`；辅以 parse/schema/hallucination | 若无法完整训练，则退化为本地产物复评 |
| SCV-off | 验证去掉 SCV 后的主线表现 | A. 必须在仓库中执行 | 已确认 SCV 开关或最小切换点 | dev 对比 summary | `strict_f1`、可靠性指标 | 若无法直切，则最小代码隔离；仍不行则只做离线对齐 |
| SCV-on | 验证 full-SCV 的当前成本与收益 | A. 必须在仓库中执行 | 已确认 SCV 路径、缓存、调用点 | dev 对比 summary、SCV 成本记录 | `strict_f1`、`SCV` wall-clock 占比 | 若运行过重，则优先复评已有产物 |
| DeepSeek API baseline | 对齐项目外强基线 | A. 必须对齐；运行仅限本地离线产物 | 当前工作树存在离线 API 预测/manifest/usage | API baseline summary、usage/retries/manifest | dev `strict_f1`、可靠性指标 | 若无离线产物，则降级为 C |
| 新方案 MVP | 实施首选改造的最小可行版本 | A. 必须在仓库中执行 | Phase 0–2 已确认关键模块与 baseline | MVP 预测、summary、错误分析、回滚点 | `strict_f1`、parse/schema/hallucination/grounding、SCV 调用数 | 若解码约束难接入，则降级为 parser 主导的 MVP |
| 新方案增强版 | 验证 MVP 之后的增量收益 | B. 若仓库支持则执行 | MVP 通过 Go 规则；仓库支持额外增强 | enhanced summary、与 MVP 的差异报告 | 同 MVP，外加成本增量 | 若收益不稳或复杂度过高，则不进主表主列 |
| ProCNet | 补齐集合级/经典 DEE baseline | B. 若仓库/代码可获得则执行 | 当前工作树已有本地代码或离线输出 | 本地复评结果或引用说明 | dev `strict_f1`、效率（若可得） | 若无本地 artefact，则 C |
| RAAT | 补齐关系增强 DEE baseline | B. 若仓库/代码可获得则执行 | 同上 | 同上 | dev `strict_f1`、关系一致性相关现象 | 若无本地 artefact，则 C |
| PTPCG | 补齐高效非自回归 baseline | B. 若仓库/代码可获得则执行 | 同上 | 同上 | dev `strict_f1`、效率/成本 | 若无本地 artefact，则 C |
| GIT | 补齐图交互 / Tracker baseline | B. 若仓库/代码可获得则执行 | 同上 | 同上 | dev `strict_f1`、多事件相关现象 | 若无本地 artefact，则 C |

执行状态分类解释：
- **A. 必须在仓库中执行**：优先使用当前工作树直接执行；若做不到，至少完成本地离线复评。
- **B. 若仓库/代码可获得则执行**：只有当前工作树已具备本地代码或离线结果时才执行。
- **C. 无法复现时仅保留结果引用或格式对齐**：必须明确标注，禁止伪装成已运行。

# 7. 新方案实验设计（核心）

## 7.1 MVP

### 改动目标
把当前 OG-LANS 从“训练期重 SCV、推理期自由生成”转成“推理期先锁结构、再做轻量证据核验”的最小可行版本。

### 必须包含的能力
1. **结构约束解码**
   - 若推理栈支持，则接入最小可行的结构约束。
   - 若推理栈不支持，则降级为 prompt 端固定 JSON skeleton。
2. **JSON skeleton / schema-controlled generation**
   - 输出层只允许合法事件类型、合法字段层级、合法角色位置。
3. **role whitelist**
   - 基于事件类型过滤非法角色。
4. **alias map**
   - 把角色别名映射回 canonical role；别名来源只能是 schema、当前仓库已有规则、或本仓库错误样本中可归纳出的保守规则。
5. **duplicate role split**
   - 对多值串联的 role 做保守拆分；若不能安全拆分，则保留原值并打标。
6. **时间 / 金额 / 比例 / 公司名规范化**
   - 只做保守规范化，每次变换都应可记录、可回退。
7. **grounding**
   - exact match
   - 去空格 / 去标点后的 fuzzy match
   - 公司简称 / 证券代码映射
   - 规则仅允许依赖原文、当前样本局部信息、仓库内已有资源。
8. **SCV-Lite 触发条件**
   - grounding 失败
   - 同一文本产生互斥事件
   - 多事件共享角色无法判定归属

### 可能涉及的模块
- `src/oglans/data/prompt_builder.py`
- `src/oglans/utils/json_parser.py`
- `src/oglans/utils/scv.py`
- `evaluate.py`
- `src/oglans/data/adapter.py`（若需要暴露原文/句子/span）

### 对训练链路影响
- 默认不改训练主链路，优先以推理/后处理 MVP 为主。
- 若 prompt 训练/推理强耦合，且改 prompt 必须重新训练：
  - 先确认仓库是否支持最小重训或小规模校正；
  - 若不支持，则先仅上线 parser-side MVP。
- 不默认改动 `build_graph.py`、`ds_cns.py`、trainer wrapper，除非 Phase 0 明确显示必须联动。

### 对推理链路影响
- 输出 JSON 结构更固定。
- 解析阶段新增角色白名单、alias、拆分、规范化、grounding 与错误标记。
- SCV 从全量路径降为 Lite 条件触发。

### 预期收益
- 优先提升 `schema_compliance_rate`。
- 下降 `parse_error_rate`。
- 降低 `hallucination_rate`。
- 在不增加重 SCV 成本的前提下争取提升或至少稳住 dev `strict_f1`。

### 主要风险
- 约束过严导致 recall 下滑。
- alias map / 规范化规则修过头。
- grounding 过严导致合法简称被误杀。
- prompt 改动与旧 checkpoint 不匹配。

### 最小验证方法
1. 对少量样本做 smoke check：
   - 输出能被当前 parser 接收。
   - 非法角色被清掉或被记录。
   - 重复 role 在可拆分时被拆分。
   - grounding 成功/失败都能留下诊断。
2. 在完整 dev 上评测：
   - 对比 baseline、SCV-off、SCV-on、MVP。
3. 若新增字段会破坏现有评测格式，则 grounding 诊断写 sidecar log，不改主预测 JSON。

## 7.2 增强版

### 改动目标
在 MVP 成功的前提下，继续提升结构可靠性与证据对齐质量，但不牺牲可回退性。

### 候选增强项
- 更严格的结构约束
- 更强的 schema-aware repair
- 更细粒度 grounding 指标
- 更完整的 SCV-Lite 触发策略
- 与偏好优化 / teacher 数据结合（仅当仓库支持且资源允许）

### 可能涉及的模块
- `prompt_builder.py`
- `json_parser.py`
- `scv.py`
- `evaluate.py`
- `adapter.py`
- `ds_cns.py` / trainer wrappers（仅当仓库支持 teacher / preference 路径）
- `evaluate_api.py`（仅当已有离线 teacher 产物）

### 对训练链路影响
- 默认不扩大训练栈改动。
- 若仓库已有 `UnslothSFTTrainerWrapper` / `UnslothDPOTrainerWrapper` 且支持离线 teacher / chosen-rejected 数据，则可尝试小规模增强。
- 若仓库不支持，则增强版只停留在推理/后处理与评测层。

### 对推理链路影响
- 可能引入更严的结构过滤与更细的 grounding 判定。
- 可能增加 SCV-Lite 触发范围，因此必须同步记录成本。

### 预期收益
- 在 MVP 已改善结构稳定性的前提下，进一步压低 hallucination / ungrounded argument。
- 若仓库支持偏好优化增强，争取补一部分 strict_f1。

### 主要风险
- 成本增加快于收益。
- SCV-Lite 触发范围过大，重新演变为“重 SCV”。
- grounding 指标细化导致日志复杂、评测脚本脆弱。
- 训练链路增强引入新的不稳定因素。

### 最小验证方法
- 每次只增加一个增强点。
- 每个增强点都必须与 MVP 做同口径 dev 对比。
- 任何增强如果导致成本大涨且收益不明显，立即回退到 MVP。

# 8. 逐文件修改计划

> 下列文件名来自输入文档。真实函数名、配置键、调用顺序全部待仓库确认。若当前工作树中不存在对应文件，则记录 blocker，不得脑补替代路径。

## `src/oglans/data/prompt_builder.py`
- 修改目标：把自然生成 JSON 调整为更受 schema 控制的 JSON skeleton 生成。
- 先 inspect 什么：
  - `ChinesePromptBuilder` 是否存在。
  - train/infer 是否共用 prompt。
  - 输出格式提示与 schema 注入位置。
- 建议改什么逻辑：
  - 加入最小 JSON skeleton。
  - 明确事件类型与角色合法范围。
  - 若仓库支持，则加入更强结构约束；若不支持，只保留 prompt 级约束。
- 哪些旧功能必须保持不坏：
  - 原 baseline prompt 路径可恢复。
  - 旧 checkpoint 若仍需评测，必须保留兼容入口。
- 修改后如何做最小验证：
  - 单样本构造 prompt，对比新旧 prompt。
  - 新输出能进入现有 parser / evaluate。
- 可能的回归风险：
  - 旧模型对新 prompt 不适配。
  - 输出字段名变化导致 parser 失配。

## `src/oglans/utils/json_parser.py`
- 修改目标：增加 schema-aware repair、role whitelist、alias map、duplicate role split、规范化、grounding 诊断。
- 先 inspect 什么：
  - 当前 parse / repair 主入口。
  - schema 文件如何读取。
  - 当前 argument 的实际存储结构。
  - 当前是否已有非法字段清理或容错修复。
- 建议改什么逻辑：
  - 事件类型条件化的 role whitelist。
  - alias → canonical role 的保守映射。
  - 多值 role 的可回退拆分。
  - 时间/金额/比例/公司名规范化。
  - grounding exact/fuzzy/code mapping。
  - sidecar diagnostics：记录清理、修复、丢弃、grounding 成败。
- 哪些旧功能必须保持不坏：
  - 对旧输出格式的宽容解析。
  - 旧 baseline 可继续被复评。
- 修改后如何做最小验证：
  - 使用最小错误样例检查：角色漂移、边界归一化、多值合并、伪事件。
- 可能的回归风险：
  - 修过头导致合法值被删。
  - 新诊断字段破坏原评测输入。

## `src/oglans/utils/scv.py`
- 修改目标：保留 full-SCV 路径，同时新增 SCV-Lite，仅在指定触发条件下运行。
- 先 inspect 什么：
  - SCV 在训练、推理、离线清洗中的调用点。
  - 缓存、滑窗、阈值、批处理接口。
  - 当前 on/off 的控制方式。
- 建议改什么逻辑：
  - 保留原 full-SCV 以支持 `SCV-on` baseline。
  - 新增 Lite 触发门：grounding 失败、互斥事件、共享角色归属冲突。
  - 对每次触发记录 reason code 与耗时。
- 哪些旧功能必须保持不坏：
  - 原 SCV-on baseline 可运行。
  - SCV-off 可完全关闭。
- 修改后如何做最小验证：
  - 用同一组样本分别跑 off / on / lite，确认调用次数与触发原因可追踪。
- 可能的回归风险：
  - SCV 与训练链路耦合，绕过后训练报错。
  - Lite 触发过宽，成本重新膨胀。

## `evaluate.py`
- 修改目标：在不破坏现有指标的前提下，新增 grounding 相关指标并固化 dev-only F1 口径。
- 先 inspect 什么：
  - 当前 strict/relaxed/type 的实现。
  - parse/schema/hallucination 的 summary 字段名。
  - 评测输入是否允许 sidecar diagnostics。
- 建议改什么逻辑：
  - 新增 `grounding_rate`。
  - 新增 `ungrounded_argument_rate`。
  - grounding 诊断缺失时输出 `NA`，不得报错。
  - summary 中显式写出 split，防止 test 混入主表。
- 哪些旧功能必须保持不坏：
  - baseline 分数不因新增指标而漂移。
  - 旧 summary 聚合脚本尽可能兼容。
- 修改后如何做最小验证：
  - 对旧结果复评，确认旧指标不变。
  - 对启用 grounding 的结果，确认新指标可落盘。
- 可能的回归风险：
  - summary schema 变化导致 ablation 脚本失配。
  - `NA` 处理不当导致聚合脚本崩溃。

## `src/oglans/data/adapter.py`
- 修改目标：确认是否需要暴露原文/句子/span/teacher 字段给 parser、evaluator 或增强训练链路。
- 先 inspect 什么：
  - `DuEEFinAdapter` 的返回字段。
  - train/dev/test 的样本结构差异。
  - 是否已保留句级切分、候选 span 或 document id。
- 建议改什么逻辑：
  - 若已有原文/句级信息，则暴露给 grounding 使用。
  - 若仓库支持 teacher / chosen-rejected 数据流，再考虑补字段。
- 哪些旧功能必须保持不坏：
  - baseline 数据读取不变。
- 修改后如何做最小验证：
  - 抽样确认字段没有丢失、split 没有串。
- 可能的回归风险：
  - 数据接口改动影响训练 / 推理兼容性。

## `src/oglans/utils/ds_cns.py`
- 修改目标：仅为增强版预留接口；默认不在 MVP 阶段修改。
- 先 inspect 什么：
  - 是否存在 chosen/rejected 或 hard negative 构造逻辑。
  - 是否有 competence、扰动强度、图缓存相关接口。
- 建议改什么逻辑：
  - 若增强版进入训练链路，可在此接入“可靠性优先”的 rejected 构造。
  - 若仓库不支持，则不修改。
- 哪些旧功能必须保持不坏：
  - 当前 LANS / DS-CNS baseline 行为不变。
- 修改后如何做最小验证：
  - 对采样输出做最小样例检查，确保格式不破坏训练入口。
- 可能的回归风险：
  - 训练分布变化导致 baseline 不再可比。

## `build_graph.py`
- 修改目标：仅在增强版需要集合级/关系级一致性扩展时考虑。
- 先 inspect 什么：
  - graph 构建输入、缓存路径、下游依赖。
- 建议改什么逻辑：
  - 默认不动；若增强版需要更多 schema 映射或 alias 来源，可评估是否从这里读取已有图信息。
- 哪些旧功能必须保持不坏：
  - 当前 schema 图构建与缓存行为。
- 修改后如何做最小验证：
  - 图缓存仍可被下游读取。
- 可能的回归风险：
  - 图缓存失效或与既有训练数据不兼容。

## `evaluate_api.py`
- 修改目标：仅用于离线 API baseline 对齐；不得引入新的联网调用。
- 先 inspect 什么：
  - 是否支持离线缓存读取。
  - 是否已输出 manifest / usage / retries。
  - API 输出与本地 `evaluate.py` 的接口关系。
- 建议改什么逻辑：
  - 若已有离线产物读取能力，则补齐 dev-only 复评与 summary 对齐。
  - 若只支持在线调用，则不修改为在线路径；只记录 blocker。
- 哪些旧功能必须保持不坏：
  - 现有离线产物读取逻辑（若有）。
- 修改后如何做最小验证：
  - 用已有离线 API 响应做一次 dry-run 复评。
- 可能的回归风险：
  - 不小心触发在线调用。

## `scripts/ablation_study.py`
- 修改目标：如仓库支持，则接入 MVP / enhanced / SCV-lite 结果汇总。
- 先 inspect 什么：
  - 当前读取哪些 summary 字段。
  - 当前假定哪些实验命名规则。
- 建议改什么逻辑：
  - 若脚本存在且可维护，则新增 grounding、SCV 调用数、结果来源类型等字段。
  - 若脚本不存在或耦合过深，则改为手工 manifest 汇总。
- 哪些旧功能必须保持不坏：
  - 已存在的 ablation summary 聚合。
- 修改后如何做最小验证：
  - 用旧 summary 与新 summary 各跑一次，确认兼容。
- 可能的回归风险：
  - 聚合脚本无法处理 `NA` 或新字段。

# 9. 指标、日志与结果留存规范

每次实验必须留存以下内容；若仓库已有既定格式，则优先沿用既有格式。

## 9.1 每次 run 必存信息
- 配置快照
- 启动命令
- 代码改动摘要
- 训练耗时
- 推理耗时
- 显存占用（若可获得）
- dev 集指标 summary
- 失败样本统计
- 错误类型统计
- API baseline usage / retries / manifest（若适用）

## 9.2 至少跟踪的指标
- `strict_f1`
- `relaxed_f1` 或 `type_f1`（若脚本支持）
- `parse_error_rate`
- `schema_compliance_rate`
- `hallucination_rate`
- `grounding_rate`
- `ungrounded_argument_rate`
- `SCV` 调用次数
- `SCV` wall-clock 占比（若可统计）

## 9.3 provenance 规则
每个数字必须标记来源类型：
- `LOCAL_RUN`：当前工作树直接执行得到
- `LOCAL_REEVAL`：当前工作树对本地已有预测/summary 重新评测得到
- `REFERENCE_ONLY`：结果引用或格式对齐，非直接复现

## 9.4 结果留存原则
- baseline、SCV-on、SCV-off、MVP、增强版必须分别保存，不得覆盖。
- 若仓库没有既定 run 目录：`TODO: inspect repo` 后定义一个与现有风格一致的最小目录规范，并写入 runbook。
- 若仓库是 git：记录 commit hash 或 diff 摘要；若不是 git：记录修改文件列表。
- 若有 sidecar diagnostics，必须与对应预测文件共存，且命名关系清楚。

## 9.5 失败样本与错误统计
至少区分以下错误类型；若脚本不能自动统计，则先人工统计少量代表样本：
- parse failure
- 非法事件类型
- 非法角色
- alias 未回正
- duplicate role 未拆分
- 规范化前后不一致
- grounding 失败
- ungrounded argument 被保留
- 互斥事件冲突
- 共享角色归属不明

# 10. Stop / Go 规则

按以下规则做决策：

1. **Go 到增强版**
   - 若 `schema_compliance_rate` 明显提升且 `parse_error_rate` 明显下降，即使 `strict_f1` 只小幅提升，也允许进入增强版。

2. **优先补 grounding / repair**
   - 若 `strict_f1` 上升但 `hallucination_rate` 恶化，先补 grounding / schema-aware repair，不直接扩大 SCV-Lite。

3. **检查 recall 受损**
   - 若 `hallucination_rate` 下降但 `strict_f1` 不升，优先检查是否约束过严导致 recall 下降。

4. **缩小 SCV-Lite 触发范围**
   - 若 SCV-Lite 成本上升但收益不明显，则缩小触发条件或回退到更轻的 Lite 版本。

5. **增强版停止条件**
   - 若增强版相对 MVP 主要指标无明显改善、但实现和维护成本明显上升，则停止增强版，把 MVP 作为主线结果。

6. **外部 baseline 标注规则**
   - 若外部 baseline 无法完整复现，则标注为“结果引用/格式对齐”，不得伪装为直接复现。

7. **结果可信度优先于结果数量**
   - 若某个结果缺少 manifest、缺少 split 标注、或无法追溯，则该结果不得进入主表。

8. **dev 口径不清立即停止**
   - 若无法确认某结果是否来自 dev，则停止汇总该结果，直到口径查清。

# 11. 推荐执行顺序

按以下线性顺序推进。每一步都必须有完成标志与下一步入口。

1. **核查数据入口**
   - 要做什么：确认 DuEE-Fin 数据、split、schema、字段名。
   - 完成标志：能写出 train/dev/test 的真实定位和关键字段。
   - 下一步入口：核查 prompt / parser / evaluate。

2. **核查 prompt / parser / scv / evaluate 入口**
   - 要做什么：确认 MVP 与评测所需核心文件。
   - 完成标志：能指出真实入口文件与待修改点。
   - 下一步入口：核查训练入口与输出目录。

3. **核查训练入口与输出目录**
   - 要做什么：确认 baseline 的最小可执行/可复评路径。
   - 完成标志：有一个 baseline 可运行或可复评的实际入口。
   - 下一步入口：Phase 1 baseline 复现/确认。

4. **确认 baseline**
   - 要做什么：跑 baseline 或复评已有本地产物。
   - 完成标志：得到一个 dev baseline summary，附带 provenance。
   - 下一步入口：SCV on/off 对比。

5. **确认 SCV on/off**
   - 要做什么：找到 SCV 切换点并完成对比。
   - 完成标志：拿到 `SCV-off`、`SCV-on` 的 dev summary 与成本记录。
   - 下一步入口：MVP 实施。

6. **先做 parser 侧 MVP**
   - 要做什么：优先在 `json_parser.py` 实现 whitelist / alias / split / normalize / grounding。
   - 完成标志：小样本 smoke check 通过。
   - 下一步入口：prompt 侧 skeleton 或解码约束。

7. **再做 prompt 侧 skeleton**
   - 要做什么：在 `prompt_builder.py` 固定输出骨架；若仓库支持再加更强约束。
   - 完成标志：新输出可进入 parser，不破坏 baseline 回退。
   - 下一步入口：SCV-Lite。

8. **接入 SCV-Lite**
   - 要做什么：把 SCV 触发收缩到 grounding 失败 / 互斥事件 / 共享角色冲突。
   - 完成标志：能记录 Lite 触发次数与原因。
   - 下一步入口：evaluate grounding 指标。

9. **扩展 `evaluate.py`**
   - 要做什么：增加 grounding 相关指标，固化 split 标识。
   - 完成标志：旧结果复评不漂移，新结果能产出 grounding 指标或 `NA`。
   - 下一步入口：完整 dev 跑 MVP。

10. **跑 MVP dev 评测**
    - 要做什么：在 dev 上完成 MVP 评测与最小误差分析。
    - 完成标志：获得 MVP 的 dev summary 与错误统计。
    - 下一步入口：按 Stop / Go 规则判断是否进入增强版。

11. **判断是否进入增强版**
    - 要做什么：根据第 10 节决定继续或停止增强版。
    - 完成标志：增强版状态明确：执行 / 不执行。
    - 下一步入口：若执行则进入增强版；否则跳到 API baseline 对齐。

12. **若执行增强版，则逐项增量实现**
    - 要做什么：每次只加一个增强点，并与 MVP 对比。
    - 完成标志：增强版结果稳定或明确失败。
    - 下一步入口：API baseline 对齐。

13. **对齐 DeepSeek API baseline**
    - 要做什么：检查 `evaluate_api.py` 与本地离线产物，完成离线复评或降级说明。
    - 完成标志：API baseline 有 dev 结果或有 `REFERENCE_ONLY` 说明。
    - 下一步入口：外部 baseline 清点。

14. **清点 ProCNet / RAAT / PTPCG / GIT**
    - 要做什么：确认当前工作树是否已有本地代码或离线结果。
    - 完成标志：四项 baseline 全部拿到 A/B/C 状态。
    - 下一步入口：表格汇总。

15. **汇总主表、消融表、效率表、可靠性表**
    - 要做什么：收集所有 summary、manifest、错误统计与 provenance。
    - 完成标志：四类表格草案完成，所有数字可追溯。
    - 下一步入口：撰写最终交付物。

16. **产出最终交付物**
    - 要做什么：生成 summary、表格、风险说明、论文建议、backlog。
    - 完成标志：第 12 节所有交付物齐备。
    - 下一步入口：结束本轮执行。

# 12. 最终交付物

Codex CLI 最终必须产出以下内容；若某项无法完成，必须提供 blocker 与当前状态。

1. **一份实验执行 summary**
   - 内容：做了什么、哪些已完成、哪些未完成、主要 blocker、结果来源类型。

2. **一份主实验结果表**
   - 至少包含：OG-LANS 主线、SCV-off、SCV-on、DeepSeek API baseline、新方案 MVP、新方案增强版、ProCNet、RAAT、PTPCG、GIT。

3. **一份消融结果表**
   - 至少覆盖：whitelist、alias、duplicate split、normalize、grounding、SCV-Lite、prompt skeleton、增强项。

4. **一份效率/成本结果表**
   - 至少覆盖：训练耗时、推理耗时、显存占用（若可得）、SCV 调用次数、SCV 时间占比、API usage/retries（若适用）。

5. **一份可靠性结果表**
   - 至少覆盖：parse error、schema compliance、hallucination、grounding、ungrounded argument。

6. **一份失败与风险说明**
   - 内容：失败阶段、回退动作、未完成原因、可信度风险。

7. **一份论文主表建议**
   - 内容：哪些结果适合放主表，哪些只能放附录，哪些必须标注 `REFERENCE_ONLY`。

8. **一份附录消融建议**
   - 内容：增强版、失败消融、非稳定结果、结果引用/格式对齐条目。

9. **一份后续 backlog**
   - 内容：未做但值得做的事项，前提是仓库支持或资源允许。

# 13. 可并行部分（如适用）

允许并行，但必须遵守：**不可并行修改同一批文件；并行结果必须先合并事实，再进入修改阶段。**

## 13.1 可并行的只读核查任务
以下任务可由 subagents 并行执行，因为默认只读：
- **Subagent A：代码入口核查**
  - 范围：`adapter.py`、`prompt_builder.py`、`json_parser.py`、`scv.py`
  - 目标：确认数据 / prompt / parser / SCV 的真实入口和耦合关系
- **Subagent B：评测与结果目录清点**
  - 范围：`evaluate.py`、`evaluate_api.py`、结果目录、summary / manifest / logs
  - 目标：确认评测口径、离线 API 产物、现有 run 目录规范
- **Subagent C：训练与消融入口核查**
  - 范围：`main.py`、trainer wrappers、`scripts/ablation_study.py`、`run_*_repro_suite.py`
  - 目标：确认 baseline 运行路径与消融可复用性

并行结束后，主 agent 必须先合并事实，再决定改动方案。

## 13.2 可并行的低冲突修改任务
仅在 Phase 0–2 完成后允许：
- **Subagent D：parser / grounding 路径**
  - 文件：`json_parser.py`
- **Subagent E：评测扩展路径**
  - 文件：`evaluate.py`
- **Subagent F：prompt skeleton 路径**
  - 文件：`prompt_builder.py`

限制：
- `scv.py` 改动不要与 `json_parser.py` 同时提交到同一分支，先等 parser 诊断结构稳定。
- 修改 `evaluate.py` 前先锁定 summary schema，否则并行结果难合并。

## 13.3 不建议并行的任务
以下任务默认串行：
- baseline 复现与 SCV on/off 对比
- MVP 全链路评测
- 增强版对比
- 最终表格汇总

原因：这些任务强依赖同一组 summary 口径与同一份回滚点。

# 14. 附录：表格模板

## 14.1 主实验表

| 方法 | 状态(A/B/C) | 结果来源 | 数据划分 | strict_f1 | relaxed_f1/type_f1 | parse_error_rate | schema_compliance_rate | hallucination_rate | grounding_rate | 备注 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| OG-LANS 当前主线 | A | LOCAL_RUN / LOCAL_REEVAL | dev | TBD | TBD | TBD | TBD | TBD | TBD/NA | |
| SCV-off | A | LOCAL_RUN / LOCAL_REEVAL | dev | TBD | TBD | TBD | TBD | TBD | TBD/NA | |
| SCV-on | A | LOCAL_RUN / LOCAL_REEVAL | dev | TBD | TBD | TBD | TBD | TBD | TBD/NA | |
| DeepSeek API baseline | A/C | LOCAL_REEVAL / REFERENCE_ONLY | dev | TBD | TBD | TBD | TBD | TBD | TBD/NA | 离线产物缺失则只能 C |
| 新方案 MVP | A | LOCAL_RUN | dev | TBD | TBD | TBD | TBD | TBD | TBD | |
| 新方案增强版 | B | LOCAL_RUN / N/A | dev | TBD | TBD | TBD | TBD | TBD | TBD | 若未执行写原因 |
| ProCNet | B/C | LOCAL_RUN / LOCAL_REEVAL / REFERENCE_ONLY | dev | TBD | TBD | TBD/NA | TBD/NA | TBD/NA | TBD/NA | |
| RAAT | B/C | LOCAL_RUN / LOCAL_REEVAL / REFERENCE_ONLY | dev | TBD | TBD | TBD/NA | TBD/NA | TBD/NA | TBD/NA | |
| PTPCG | B/C | LOCAL_RUN / LOCAL_REEVAL / REFERENCE_ONLY | dev | TBD | TBD | TBD/NA | TBD/NA | TBD/NA | TBD/NA | |
| GIT | B/C | LOCAL_RUN / LOCAL_REEVAL / REFERENCE_ONLY | dev | TBD | TBD | TBD/NA | TBD/NA | TBD/NA | TBD/NA | |

## 14.2 消融实验表

| 实验设置 | 是否启用 | 结果来源 | strict_f1 | parse_error_rate | schema_compliance_rate | hallucination_rate | grounding_rate | SCV 调用次数 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 是 | TBD | TBD | TBD | TBD | TBD | TBD/NA | TBD | |
| + role whitelist | 是/否 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| + alias map | 是/否 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| + duplicate role split | 是/否 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| + normalize(time/money/ratio/company) | 是/否 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| + grounding exact/fuzzy/code | 是/否 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| + prompt skeleton | 是/否 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| + SCV-Lite | 是/否 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| + enhanced item 1 | 是/否 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| + enhanced item 2 | 是/否 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |

## 14.3 效率/成本表

| 方法/设置 | 结果来源 | 训练耗时 | 推理耗时 | 显存占用 | SCV 调用次数 | SCV wall-clock 占比 | API usage | API retries | 备注 |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| OG-LANS 当前主线 | TBD | TBD | TBD | TBD/NA | TBD | TBD/NA | NA | NA | |
| SCV-off | TBD | TBD | TBD | TBD/NA | 0 | 0/NA | NA | NA | |
| SCV-on | TBD | TBD | TBD | TBD/NA | TBD | TBD | NA | NA | |
| 新方案 MVP | TBD | TBD | TBD | TBD/NA | TBD | TBD | NA | NA | |
| 新方案增强版 | TBD | TBD | TBD | TBD/NA | TBD | TBD | NA | NA | |
| DeepSeek API baseline | TBD | NA | TBD | NA | NA | NA | TBD/NA | TBD/NA | 仅离线产物可统计 |
| ProCNet / RAAT / PTPCG / GIT | TBD | TBD/NA | TBD/NA | TBD/NA | NA | NA | NA | NA | 仅在本地 artefact 可得时填写 |

## 14.4 可靠性表

| 方法/设置 | 结果来源 | parse_error_rate | schema_compliance_rate | hallucination_rate | grounding_rate | ungrounded_argument_rate | 主要错误类型 Top-3 | 备注 |
|---|---|---:|---:|---:|---:|---:|---|---|
| OG-LANS 当前主线 | TBD | TBD | TBD | TBD | TBD/NA | TBD/NA | TBD | |
| SCV-off | TBD | TBD | TBD | TBD | TBD/NA | TBD/NA | TBD | |
| SCV-on | TBD | TBD | TBD | TBD | TBD/NA | TBD/NA | TBD | |
| 新方案 MVP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| 新方案增强版 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| DeepSeek API baseline | TBD | TBD/NA | TBD/NA | TBD/NA | TBD/NA | TBD/NA | TBD | |

