# OG-LANS 方法论说明

## 1. 研究问题

本项目关注中文金融领域的篇章级事件抽取任务，并以 DuEE-Fin 作为当前唯一的实验数据集。相较于句级抽取，篇章级金融事件抽取同时面临长文本、多事件共存、多值论元和角色边界不稳定等问题。对于基于大语言模型的结构化抽取而言，模型即使能识别出事件线索，也未必能稳定输出与 schema 严格一致、且能够被稳定评测的事件结构。

OG-LANS 的核心目标，是在不脱离大语言模型生成范式的前提下，将“领域知识约束”“负样本难度控制”和“偏好优化训练”结合起来，使模型更稳定地输出符合 DuEE-Fin schema 的事件结果。

## 2. 方法总览

当前仓库对应的方法主线可以概括为：

1. 以 DuEE-Fin 事件 schema 构建本体图，提供事件类型之间的语义距离结构。
2. 以中文 prompt 和结构化目标答案组织训练样本，使模型学习面向事件抽取的输出格式。
3. 基于本体图距离与多粒度扰动生成 rejected 样本，并通过 LANS 机制动态调节负样本难度。
4. 使用 SCV 模块过滤可能被误构造成假阴性的负样本，降低偏好学习噪声。
5. 在 Unsloth + TRL 框架下执行 IPO/DPO 风格的偏好优化，并在评测阶段通过 schema 约束、CAT-lite 和反事实检查进一步审查输出质量。

## 3. 本体图驱动的负样本建模

OG-LANS 的第一项核心假设是：事件类型之间并非彼此孤立，而是存在由领域 schema 所诱导的语义邻近关系。仓库中的 `build_graph.py` 与 `src/oglans/utils/ds_cns.py` 以 `duee_fin_event_schema.json` 为输入，构建事件类型图，并用图上最短路径近似表示事件类型间的语义距离。

这一设计的作用有两点：

- 它为负样本“难度”提供了可解释的结构依据，而不完全依赖黑盒打分。
- 它使 rejected 样本不再只是随机错误，而是更接近真实抽取错误分布中的“近邻混淆”。

在实现上，负样本扰动并不限于事件类型层面，还覆盖论元角色层面和论元值层面。也就是说，模型既可能面对“类型接近但错误”的反例，也可能面对“类型正确但角色错配”或“值被扰动”的反例，从而更贴近真实事件抽取中的细粒度失败模式。

## 4. LANS: 基于损失的自适应难度调度

仅仅拥有本体图并不足以保证训练有效。若过早地向模型注入过难的负样本，偏好学习会变得不稳定；若长期只使用过易样本，又会削弱区分能力。为此，OG-LANS 在仓库中实现了 Loss-Aware Adaptive Negative Sampling。

其直观思想是：用当前训练损失反向估计模型能力，再由该能力值决定下一阶段更适合采样何种难度的 rejected 样本。代码中已经实现以下三个机制：

- 能力估计：依据 DPO/IPO 损失构造 competence，并可选地使用 EMA 平滑。
- 难度起搏：随着 competence 上升，逐步提升“语义更近、区分更难”的负样本比例。
- 梯度调节：通过 CGA 机制，在模型能力较弱时适度放大困难样本的学习信号。

因此，OG-LANS 并不是静态地构造一批负样本，而是把负样本生成过程纳入训练动态之中。

## 5. SCV: 语义一致性验证

偏好优化中的一个关键风险，是把“表面上不同、实际上仍然被原文支持”的样本误当作 rejected 反例。若这种假阴性进入训练，会直接污染偏好信号。为此，仓库在 `src/oglans/utils/scv.py` 中实现了 SCV（Semantic Consistency Verification）模块。

SCV 的职责不是生成答案，而是检查候选 rejected 样本是否真的偏离原文语义。当前实现以 NLI 风格判断为核心，并带有缓存与长文档滑窗支持。其研究意义在于：

- 降低由负样本构造带来的标签反转风险；
- 将偏好学习的监督重点从“表面差异”转向“语义不一致”；
- 在保持训练主线不变的前提下，为 rejected 样本提供额外的语义过滤层。

在当前仓库的方法体系中，SCV 是连接“自动构造负样本”与“可靠偏好训练”的关键安全阀。

## 6. IPO/DPO 训练与结构化输出

本仓库的训练入口并不直接追求自由生成，而是将事件抽取任务固定为带有 schema 约束的结构化输出。`DuEEFinAdapter` 与 `ChinesePromptBuilder` 将原始文档转换为中文 instruction 风格输入，并将金标事件组织为严格 JSON 数组形式的 chosen 输出。当前正式契约要求模型直接输出事件列表，不再把 `<thought>` 或 markdown code fence 视为训练/评测主路径的一部分。

在此基础上，训练侧通过 Unsloth/TRL 执行偏好优化。当前实现既支持 preference 模式，也保留了 plain SFT 作为最小可比基线。主线配置默认采用 IPO 风格损失，并允许在代码层面扩展至其他 preference 变体。这里的方法论重点不在于宣称“某一损失函数本身就是创新”，而在于将偏好学习与本体图难度控制、SCV 过滤和严格 JSON 抽取契约联合起来。

在当前工程实现中，训练侧已经与评测侧共享同一套 `single_pass / two_stage` 协议，而不是仅在推理时做阶段划分。具体来说：

- `single_pass` 训练直接学习结构化事件抽取；
- `two_stage` 训练采用联合训练：同一条 `train_fit` 样本会同时展开为 `stage1` 事件类型判别样本与 `stage2` 受类型约束的抽取样本；
- `stage2` 训练中的类型约束来自训练期 teacher-forced 的 gold event types，而评测期仍只允许使用 Stage 1 预测类型，从而避免把评测先验泄漏回训练报告。

这也意味着当前仓库里的 LANS 消融已经不再只是“配置注释上的开关”。`algorithms.lans.enabled` 会真正决定偏好训练是否进入 LANS 负样本调度分支，而训练清单会同时记录 `configured_lans_enabled` 与 `effective_lans_enabled`，用于区分“方法定义”与“实际运行状态”。

## 7. 评测侧约束

OG-LANS 的方法论并不把评测视为训练之后的附属步骤，而是把评测定义为方法完整性的一部分。当前仓库已经在评测端实现：

- 以 `doc_role_micro_f1` 为默认主指标的文档级 record-aware 评测；
- `overall role / classification / instance / combination` 四类文档级学术指标；
- 兼容 `strict / relaxed / type` 的旧口径指标；
- 兼容 trigger / argument 文本代理指标（`ee_text_proxy`），用于与通用 EE 工具链做附表级对照；
- schema compliance 与 hallucination 统计；
- CoT 一致性与反事实一致性检查；
- CAT-lite 这一轻量后处理过滤流程；
- 本地模型与 API 模型两条可比评测路径。

需要单独说明的是，CoT 相关检查当前属于评测侧分析能力，而不是训练输出契约本身。若模型没有显式 thought block，相关指标会按协议记为 skipped，而不是强行要求生成中间推理文本。

当前仓库也已经把 `zeroshot / fewshot` 提示模式收敛为显式的 `prompt_variant + fewshot_num_examples` 契约；训练、本地评测与 API 评测都共享这组语义，而旧的 `use_oneshot / use_fewshot` 仅保留为兼容入口。

为了让 API baseline、本地 Qwen-4B baseline 与后续自研方法保持可比，当前仓库又进一步收敛出一套共享研究协议：

- baseline 设计默认遵循 `train_fit / train_tune / dev_final` 三段式纪律；
- `train_fit` 既用于训练，也用于 few-shot exemplar pool；
- `train_tune` 只用于在全局层面选择 `single_pass / two_stage`、few-shot 检索策略和后处理开关；
- `dev_final` 只在所有 prompt、检索、rerank 和 pipeline 冻结后运行一次；
- 两阶段方案中，Stage 2 只能看到 Stage 1 预测出的事件类型子集，而不能使用任何 gold type。

这意味着仓库当前支持的“单阶段 vs 两阶段”对比，不是把额外人工先验偷偷注入模型，而是在相同 prompt、相同 few-shot 检索池、相同 schema 约束与相同后处理条件下，仅比较“是否显式引入一个事件类型判别阶段”。

还需要强调的是，当前仓库已经区分“论文主表指标”和“工程诊断指标”。对于 DuEE-Fin 这类篇章级事件抽取任务，主表更贴近 DocEE 系列工作中的角色级 Micro-F1、instance 与 combination 指标；而 `strict / relaxed / type`、schema compliance、hallucination、CoT 与 SCV-lite 统计更适合作为辅助分析或附录指标。

同样重要的是，当前仓库已经把“指标表”和“成本表”视为同等重要的研究输出。对于 API baseline 与本地小模型 baseline，summary 与复现实验脚本不仅保留主指标，还会同步记录 token usage、wall-clock 和效率指标（如 `f1_per_1k_tokens` 与 `f1_per_minute`），从而避免只报告最强 F1 而掩盖方法代价。

这意味着项目的方法论并非“只训练一个模型然后看 F1”，而是试图从训练与评测两侧同时约束结构化输出的可靠性。

## 8. 当前表述边界

需要强调的是，当前仓库是一套研究型实验实现，而不是已经完成论文定稿的正式开源包。因此，关于方法的表述应保持以下边界：

- 可以表述“已实现的方法机制”与“预期研究动机”。
- 不应把配置注释中的论文定位、投稿目标或 publication-ready 描述，直接当作已被实验完全验证的事实。
- 不应把仍默认关闭、预留或需要特定环境的模块写成无条件稳定可复现。

在这一边界下，最合适的学术化概括是：

“OG-LANS 将 DuEE-Fin 事件 schema 转化为本体图，以此指导多粒度负样本构造，并通过基于损失的自适应课程调度与 SCV 语义过滤，为 IPO/DPO 风格的结构化事件抽取训练提供更可靠的偏好信号。”
