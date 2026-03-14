# OG-LANS 项目说明（当前实现版）

## 1. 项目定位

OG-LANS 是一个面向金融篇章级事件抽取的研究型 Python 项目。项目目标不是做通用开放生成，而是让大语言模型在固定 Schema 约束下输出结构化事件结果，并用偏好优化、动态负采样和一致性诊断降低论元级错误、Schema 违规和幻觉。

当前仓库实现了三条主线能力：

1. 训练本地模型：以 Qwen3-4B-Instruct-2507 为基座，采用 LoRA + IPO/DPO 风格偏好优化进行微调。
2. 评估本地模型：支持 LoRA 微调模型评估，也支持纯基座模型对照评估。
3. 评估 API 基线：对 DeepSeek/OpenAI 兼容接口做零样本、少样本和多种子复现实验。

项目当前更适合学术实验与论文复现，而不是通用产品部署。

## 2. 当前代码库解决的问题

项目聚焦 DuEE-Fin 这类金融事件抽取任务中的几个核心困难：

1. 负样本过于简单时，偏好学习会退化为模板匹配。
2. 静态课程学习无法反映模型真实能力边界。
3. 长文本和细粒度角色约束容易诱发论元错配、边界错误和幻觉。
4. 仅看主 F1 不足以解释模型为什么失败，因此需要额外的诊断指标。

为此，项目当前采用“训练增强 + 评估诊断 + 复现管控”三层设计。

## 3. 当前仓库结构

核心目录如下：

- `configs/`
  - `config.yaml`：正式训练与评估默认配置
  - `config_debug.yaml`：训练冒烟配置
  - `eval_protocol.yaml`：学术评估协议与诊断开关
- `src/oglans/`
  - `data/`：数据适配、提示词构造、Schema 相关逻辑
  - `trainer/`：Unsloth/TRL 上封装的偏好训练器
  - `utils/`：LANS 调度、SCV、一致性校验、复现工具等
  - `inference/`：CAT-lite 后处理与反事实扰动工具
- `scripts/`
  - 训练、评估、复现实验、并行运行等 shell 包装脚本
- `main.py`
  - 本地训练入口
- `evaluate.py`
  - 本地模型评估入口
- `evaluate_api.py`
  - API 基线评估入口
- `tests/`
  - 配置、评估、协议、指标与新功能的单测

## 4. 当前算法实现

### 4.1 训练目标

当前训练主线是偏好优化框架，代码主体在 `src/oglans/trainer/unsloth_trainer.py`。

当前实现包含：

1. 基础偏好训练器：`CGADPOTrainer`
2. 当前默认损失类型：`IPO`
3. 额外的 RPO/SFT anchor 混合项
4. 可配置的 ODPO 风格偏好相关参数默认值

配置层默认值来自 `configs/config.yaml`：

- `loss_type: "ipo"`
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 32`
- `num_train_epochs: 3`
- `save_steps: 300`

这意味着当前正式配置默认的单卡有效 batch size 为 `1 x 32 = 32`。

### 4.2 LANS 动态负采样

项目的核心算法实现位于 `src/oglans/utils/ds_cns.py` 和 `src/oglans/trainer/unsloth_trainer.py`。

当前思路是：

1. 使用 `LANSScheduler` 跟踪训练损失变化。
2. 用 EMA 平滑的能力值估计当前模型处于什么学习阶段。
3. 根据能力值动态调整负样本难度阈值。
4. 在训练早期更多采 easy/medium negative，能力上升后逐步增加 hard negative。
5. 可选启用 CGA，对低能力阶段的梯度进行对比式放大。

当前配置默认：

- `refresh_start_epoch: 1`
- `use_cga: true`

这表示系统会保留一次初始负样本生成，然后从第 1 个 epoch 开始执行后续刷新。

### 4.3 负样本与 SCV

项目并不只做随机错误注入，而是尝试构造“结构上更接近真实错误”的负样本。相关逻辑分布在：

- `src/oglans/data/prompt_builder.py`
- `src/oglans/utils/ds_cns.py`
- `src/oglans/utils/scv.py`

其中：

1. `prompt_builder.py` 负责构造正负样本的 CoT 与结构化输出模板。
2. `ds_cns.py` 负责负样本策略、难度分层和调度。
3. `scv.py` 负责语义一致性验证，尽量过滤“被错误标成负样本、但语义上仍可能成立”的样本。

SCV 会显著增加刷新与训练前处理开销，但它是当前实现里保证负样本质量的重要组件。

### 4.4 推理与评估增强

当前评估实现不再只有单一路径，而是提供两类增强：

1. CoT 评估模式
   - `self_consistency`
   - `counterfactual`
2. 推理流水线模式
   - `e2e`
   - `cat_lite`

相关实现位于：

- `evaluate.py`
- `evaluate_api.py`
- `src/oglans/inference/cat_lite.py`

`cat_lite.py` 当前提供：

1. `apply_cat_lite_pipeline`
2. `perturb_text_for_counterfactual`

这意味着当前项目不只是输出 F1，还支持检查：

- CoT 与最终抽取结果是否内部一致
- 对输入做单点反事实扰动后，模型的 reasoning/result 是否跟着变化

## 5. 当前评估体系

### 5.1 本地模型评估

本地评估入口是 `evaluate.py`，主要覆盖两类对象：

1. LoRA 微调后的本地模型
2. 纯基座模型对照

当前已经支持 `--base_only` 模式，因此可以在不加载 LoRA adapter 的情况下直接评估本地 Qwen 基座模型。

对应脚本：

- `scripts/run_eval_local.sh`：评估 LoRA 模型
- `scripts/run_eval_base.sh`：评估纯基座模型
- `scripts/run_eval_academic.sh`：按学术口径汇总多 seed / 多 checkpoint 结果

### 5.2 API 评估

API 评估入口是 `evaluate_api.py`，脚本是 `scripts/run_eval_api.sh`。

当前支持：

1. zero-shot
2. few-shot
3. smoke run
4. 多种子 sweep
5. token 使用量与成本统计
6. counterfactual 额外 API 调用统计

### 5.3 当前指标族

项目当前采用“主指标 + 诊断指标”的结构：

主指标：

- strict F1
- relaxed F1
- type F1

诊断指标：

- parse error rate
- hallucination sample rate
- hallucination entity rate
- schema compliance rate
- CoT faithfulness
- CoT type consistency
- CoT argument consistency
- CoT counterfactual pass rate

指标协议由 `configs/eval_protocol.yaml` 与 `docs/METRIC_SPEC.md`、`docs/METRIC_AUDIT.md` 共同约束。

## 6. 当前脚本体系

`scripts/` 目录当前主要脚本如下：

- `run_train.sh`
  - 标准训练入口
- `run_train_tmux.sh`
  - 训练的 tmux 包装
- `run_debug.sh`
  - 快速训练/流程冒烟
- `run_eval_local.sh`
  - LoRA 本地模型评估
- `run_eval_base.sh`
  - 纯基座模型评估
- `run_eval_api.sh`
  - API 评估
- `run_eval_academic.sh`
  - 学术口径汇总评估
- `run_api_repro_suite.py`
  - API 多条件复现实验
- `run_parallel_eval_train.sh`
  - GPU0/GPU1 并行跑基座评估与训练/训练后评估
- `build_graph.py`
  - 本体图构建
- `validate_academic_artifacts.py`
  - 评估产物校验

## 7. 当前默认实验设置

基于 `configs/config.yaml`，当前默认正式配置可概括为：

1. 基座模型：`Qwen/Qwen3-4B-Instruct-2507`
2. 模型源：`modelscope`
3. 正式训练：`3` 个 epoch
4. 单卡 micro-batch：`1`
5. 梯度累积：`32`
6. 保存间隔：`300` step
7. 推理最大生成长度：`2048`
8. 默认推理流水线：`e2e`

调试配置 `configs/config_debug.yaml` 主要差异：

1. `num_train_epochs: 1`
2. `gradient_accumulation_steps: 4`
3. 仍保留相同的基座模型与大部分算法组件

## 8. 当前运行产物

项目运行后会在 `logs/` 下生成不同类型产物：

- `logs/<dataset>/checkpoints/<exp_name>/`
  - checkpoint
  - `resolved_config.yaml`
  - `run_manifest.json`
- `logs/<dataset>/tensorboard/<exp_name>/`
  - TensorBoard 日志
- `logs/<dataset>/eval_local/...`
  - 本地 LoRA 评估结果
- `logs/<dataset>/eval_base/...`
  - 基座模型评估结果
- `logs/<dataset>/eval_api/...`
  - API 评估结果
- `logs/<dataset>/eval_academic/...`
  - 学术汇总评估结果
- `logs/ops/...`
  - 并行任务脚本日志

这些产物是当前仓库复现实验、写论文和做误差分析的主要证据。

## 9. 当前项目的工程特征

从工程角度看，当前项目有几个明显特征：

1. 强配置驱动
   - 训练、评估、协议和诊断大多通过 YAML 与 CLI 参数控制。
2. 研究型脚本化
   - 通过多个 shell wrapper 保证不同实验口径可快速复用。
3. 重诊断而非只重主指标
   - 除主 F1 外，大量记录 hallucination、schema、CoT 指标和 manifest。
4. 明确区分本地模型、基座模型、API 基线
   - 已形成三个相互对照的实验分支。

## 10. 当前项目边界

当前实现已经具备较完整的学术实验链路，但边界也很明确：

1. 项目主要针对 DuEE-Fin 风格的金融事件抽取，不是通用 IE 框架。
2. 当前算法重心仍然是偏好优化和负样本课程学习，不是 RLHF 全流程。
3. CoT 相关指标是诊断工具，不应直接等同于真实因果推理忠实度。
4. SCV、长文本生成和高阶评估会带来明显的运行开销。

## 11. 适合如何使用这份文档

这份文档适合回答以下问题：

1. 当前仓库到底实现了什么？
2. 训练、评估、脚本和输出目录如何对应？
3. 当前版本支持哪些实验口径？
4. 现有代码中的算法模块分别落在哪些文件？

如果要写论文方法章节，应该配合 `docs/PROJECT_DESCRIPTION_PAPER_NARRATIVE.md` 一起看；如果要同时掌握算法叙事和工程映射，应看 `docs/PROJECT_DESCRIPTION_COMBINED.md`。
