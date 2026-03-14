# OG-LANS 项目说明（双层整合版）

## 1. 文档目的

这份文档将项目分成两层同时描述：

1. 方法层：项目为什么这样设计，核心研究假设是什么。
2. 工程层：这些方法在当前仓库里具体落到哪些文件、配置、脚本和运行产物。

如果你只需要看当前代码做了什么，优先看 `docs/PROJECT_DESCRIPTION_CURRENT_IMPLEMENTATION.md`。如果你只需要论文式叙事，优先看 `docs/PROJECT_DESCRIPTION_PAPER_NARRATIVE.md`。这份文档适合需要同时掌握两者的人。

## 2. 一句话概括项目

OG-LANS 是一个面向金融篇章级事件抽取的研究型框架，它以偏好优化为训练主干，用本体图驱动的动态负采样提升结构化抽取能力，并用多维评估协议审查模型在严格结构、幻觉控制和推理一致性上的表现。

## 3. 方法层全景

### 3.1 核心研究假设

项目建立在以下假设之上：

1. 事件抽取中的错误并非同质，模型需要通过高质量负样本学习细粒度边界。
2. 负样本难度必须随模型能力变化，否则课程信号会失效。
3. 仅靠主 F1 无法充分解释模型质量，必须同时观测 schema、hallucination 和 reasoning diagnostics。

### 3.2 方法结构

方法结构可分为四个模块：

1. 偏好优化主干
2. OG-CNS/DS-CNS 负样本构造
3. OG-LANS 动态难度调度
4. SCV + CoT 诊断

它们的职责分别是：

- 主干负责“学偏好”
- 负样本负责“定义错误边界”
- 调度负责“在对的时机给对的难度”
- 诊断负责“解释为什么好或为什么差”

## 4. 工程层全景

### 4.1 目录与责任边界

当前仓库的核心责任边界如下：

- `main.py`
  - 训练入口
- `evaluate.py`
  - 本地模型评估入口
- `evaluate_api.py`
  - API 基线评估入口
- `configs/`
  - 所有核心实验配置与协议
- `src/oglans/data/`
  - 数据适配与 prompt 生成
- `src/oglans/trainer/`
  - 偏好训练器与训练回调
- `src/oglans/utils/`
  - LANS 调度、SCV、解析与复现工具
- `src/oglans/inference/`
  - 评估期后处理和反事实工具
- `scripts/`
  - 不同实验口径的运行脚本

### 4.2 关键文件与方法模块映射

| 方法模块 | 核心职责 | 主要文件 |
|---|---|---|
| 偏好训练主干 | IPO/DPO 风格训练、RPO 混合项、训练损失组织 | `src/oglans/trainer/unsloth_trainer.py` |
| LANS 能力调度 | 能力值更新、阈值映射、CGA 控制 | `src/oglans/utils/ds_cns.py` |
| 负样本构造 | 正负样本 prompt、错误 CoT、结构化输出模板 | `src/oglans/data/prompt_builder.py` |
| 语义一致性校验 | NLI 过滤伪负样本、长文本滑窗 | `src/oglans/utils/scv.py` |
| 本地评估 | 解析、指标、基座/LoRA 对照、反事实与 CAT-lite | `evaluate.py` |
| API 评估 | API 调用、成本统计、协议一致评估 | `evaluate_api.py` |
| 推理增强 | CAT-lite 后处理、文本反事实扰动 | `src/oglans/inference/cat_lite.py` |

## 5. 训练流程

### 5.1 方法层视角

训练流程的研究逻辑是：

1. 从标注事件数据构造正样本。
2. 基于本体图和结构扰动生成多难度负样本。
3. 通过 SCV 过滤高风险伪负样本。
4. 用偏好优化让模型偏向正确输出。
5. 根据训练损失更新能力值，再反过来影响下一轮负样本难度分布。

这形成了一个闭环：模型能力影响训练数据难度，训练数据难度又反过来塑造模型能力。

### 5.2 工程层视角

当前训练实际入口：

1. `main.py`
2. `scripts/run_train.sh`

当前正式默认配置来自 `configs/config.yaml`：

- 基座模型：`Qwen/Qwen3-4B-Instruct-2507`
- 模型源：`modelscope`
- `loss_type: "ipo"`
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 32`
- `num_train_epochs: 3`
- `save_steps: 300`
- `refresh_start_epoch: 1`
- `use_cga: true`

当前调试配置来自 `configs/config_debug.yaml`：

- `num_train_epochs: 1`
- `gradient_accumulation_steps: 4`

## 6. 评估流程

### 6.1 方法层视角

当前评估不是单一 F1 统计，而是三层评估：

1. 主任务性能
   - strict/relaxed/type F1
2. 输出可靠性
   - parse error
   - schema compliance
   - hallucination
3. 推理诊断
   - CoT consistency
   - counterfactual consistency

这种设计的目的，是避免把“格式正确但事实错误”的模型误判为真正有效。

### 6.2 工程层视角

当前评估入口分三类：

1. LoRA 本地模型
   - `evaluate.py`
   - `scripts/run_eval_local.sh`
2. 纯基座模型
   - `evaluate.py --base_only`
   - `scripts/run_eval_base.sh`
3. API 基线
   - `evaluate_api.py`
   - `scripts/run_eval_api.sh`

学术汇总入口：

- `scripts/run_eval_academic.sh`

当前还支持两类评估增强开关：

- `--cot_eval_mode`
  - `self_consistency`
  - `counterfactual`
- `--pipeline_mode`
  - `e2e`
  - `cat_lite`

## 7. 当前脚本体系如何支撑实验设计

| 脚本 | 主要用途 | 典型场景 |
|---|---|---|
| `scripts/run_train.sh` | 标准训练 | 正式训练、单 seed 重跑 |
| `scripts/run_debug.sh` | 最小流程验证 | 训练冒烟、环境检查 |
| `scripts/run_train_tmux.sh` | tmux 启动训练 | 长时间服务器训练 |
| `scripts/run_eval_local.sh` | 评估 LoRA 模型 | 单 checkpoint 验证 |
| `scripts/run_eval_base.sh` | 评估纯基座模型 | 基座对照实验 |
| `scripts/run_eval_api.sh` | 评估商业 API | zero/few-shot 基线 |
| `scripts/run_eval_academic.sh` | 学术口径汇总 | 多 seed 论文主表 |
| `scripts/run_parallel_eval_train.sh` | 双 GPU 并行任务 | 一卡训练，一卡评估 |
| `scripts/run_api_repro_suite.py` | API 复现实验 | 多变体批量对照 |

这套脚本体系的价值不在于“脚本多”，而在于把不同实验口径分开，减少把 debug、正式训练、基座对照和 API 基线混在一起的风险。

## 8. 当前输出与复现资产

### 8.1 训练产物

- `logs/<dataset>/checkpoints/<exp_name>/`
  - checkpoint
  - `resolved_config.yaml`
  - `run_manifest.json`

### 8.2 评估产物

- `logs/<dataset>/eval_local/...`
- `logs/<dataset>/eval_base/...`
- `logs/<dataset>/eval_api/...`
- `logs/<dataset>/eval_academic/...`

### 8.3 辅助产物

- `logs/<dataset>/tensorboard/<exp_name>/`
- `logs/ops/...`

这些产物使当前项目具备较好的可追溯性：同一结果不仅有指标，还有配置、命令、协议与运行元数据。

## 9. 当前版本的新增能力

基于近期实现记录，当前版本相较于更早期状态，已经新增或稳定化了以下能力：

1. `base-only` 本地基座对照评估
2. `cat_lite` 推理流水线
3. `counterfactual` CoT 诊断
4. API counterfactual token 成本统计
5. `run_parallel_eval_train.sh` 并行训练/评估编排
6. 训练与评估脚本的参数同步

这意味着项目现在不只是在“训一个模型”，而是在维护一套完整的实验平台。

## 10. 当前项目适合怎样的协作方式

如果你是不同角色，建议这样使用项目：

1. 工程实现者
   - 先看 `docs/PROJECT_DESCRIPTION_CURRENT_IMPLEMENTATION.md`
   - 再看 `configs/` 和 `scripts/`
2. 论文写作者
   - 先看 `docs/PROJECT_DESCRIPTION_PAPER_NARRATIVE.md`
   - 再看 `OG-LANS项目书.md`
3. 实验执行者
   - 先看 `docs/2026-02-20_impl_log_retrain_eval_plan.md`
   - 再看对应 shell 脚本
4. 指标审查者
   - 看 `docs/METRIC_SPEC.md`
   - 看 `docs/METRIC_AUDIT.md`
   - 看 `ACADEMIC_EVALUATION_PROTOCOL.md`

## 11. 当前状态的准确结论

截至当前代码版本，可以做出的稳妥结论是：

1. 项目已经形成完整的训练、基座对照、本地评估、API 评估和学术汇总链路。
2. 项目的方法核心是“偏好优化 + 动态负采样 + 一致性诊断”。
3. 当前仓库已经把算法主张映射到可运行的工程模块与脚本。
4. 这套系统适合作为学术实验框架，并可继续支持后续方法改进与论文撰写。

## 12. 相关文档导航

若你继续阅读，建议顺序如下：

1. `docs/PROJECT_DESCRIPTION_CURRENT_IMPLEMENTATION.md`
2. `docs/PROJECT_DESCRIPTION_PAPER_NARRATIVE.md`
3. `OG-LANS项目书.md`
4. `docs/2026-02-20_impl_log_retrain_eval_plan.md`
5. `ACADEMIC_EVALUATION_PROTOCOL.md`
6. `docs/METRIC_SPEC.md`
