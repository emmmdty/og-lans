# OG-LANS 工程实现现状

## 1. 当前定位

当前仓库是一个围绕 DuEE-Fin 的学术实验工程，主线目标是验证“本体图 + SCV + IPO/DPO”对中文金融事件抽取的作用。它已经具备可组织训练、离线评测、API 基线评测和复现实验的代码骨架，但仍然是研究型仓库，而不是面向通用用户发布的成品系统。

## 2. 已完成模块

| 模块 | 当前状态 | 已实现内容 | 后续可补充 |
| --- | --- | --- | --- |
| 数据适配与提示构造 | 已实现 | `DuEEFinAdapter` 支持读取 DuEE-Fin JSONL、保留事件类型信息、构造中文 prompt 与 chosen 响应；`ChinesePromptBuilder` 支持 system prompt、few-shot、单阶段/两阶段推理输入负载构造 | 增加更多 prompt 版本管理、样例版本化与自动对比 |
| 训练主线 | 已实现 | `main.py` 支持配置解析、数据/Schema 路径推断、运行清单写入；`UnslothDPOTrainerWrapper` 与 `UnslothSFTTrainerWrapper` 分别覆盖偏好训练与 plain SFT 基线 | 增加更稳定的依赖锁定、训练恢复与更明确的 checkpoint 管理说明 |
| 负样本与课程机制 | 已实现 | `src/oglans/utils/ds_cns.py` 提供基于 schema 图的采样、LANS 能力调度、CGA 权重、多粒度扰动；支持图缓存 | 增加更系统的消融记录、图统计导出和更细粒度可视化 |
| SCV 语义校验 | 已实现 | `src/oglans/utils/scv.py` 提供 NLI 驱动的语义一致性验证、缓存与长文档滑窗检查 | 将当前部分硬编码窗口策略进一步配置化，并补充更多真实模型验证 |
| 本地评测 | 已实现 | `evaluate.py` 支持 strict/relaxed/type 指标、解析诊断、schema compliance、hallucination、CoT 一致性、CAT-lite 与反事实扰动评估；当前已支持 `single_pass / two_stage`、动态 few-shot 检索和固定 `train_fit/train_tune` 划分清单 | 补充统一的结果汇总模板和更明确的论文表格导出约束 |
| API 基线评测 | 已实现 | `evaluate_api.py` 支持 DeepSeek/OpenAI 兼容接口、并发请求、重试、usage 统计、运行清单与 dev 集 bootstrap 统计；当前已支持 `single_pass / two_stage` 共享协议、动态 few-shot 检索、固定 `train_fit/train_tune` 划分清单和成本统计 | 增加对更多供应商别名、成本汇总和结果缓存策略的约束 |
| 实验脚本 | 已实现 | `scripts/` 下已有训练包装、base 模型评测、API 评测、学术汇总、local/API reproducibility suite、图构建、消融实验和产物校验脚本 | 增加统一 CLI 文档、示例输出以及 CI 级 smoke tests |
| 测试支撑 | 已实现且当前环境已验证 | `tests/` 下当前有 35 个 `test_*.py` 文件，覆盖配置、路径推断、LANS、SCV、JSON 解析、评测语义、脚本包装、复现实验工具以及部分 Phase 3 后处理逻辑；当前可在 `uv` 环境下跑通 `uv run pytest` | 建立 CI，把“当前可运行”进一步提升为“持续自动验证” |

## 3. 当前工程边界

从代码本身可以确认，仓库已经不仅是“概念草图”，而是具备以下实际能力：

- 可从 DuEE-Fin 构造训练样本并执行 Unsloth 训练流程。
- 可在本地模型上执行结构化事件评测。
- 可对外部 API 模型做零样本或 few-shot 基线评测。
- 可围绕论文式实验组织复现实验、学术摘要和消融配置。

但它也仍然有明确边界：

- 当前没有仓库级 `README.md`，对外入口仍偏工程内使用。
- 当前没有 `requirements.txt`，但仓库已经包含 `uv.lock`，因此实际依赖安装不再只是“裸 `pyproject.toml`”状态。
- 当前仓库可在 `uv sync --extra dev` 后直接运行 `uv run pytest`；CPU-only 环境下仍会跳过一部分 GPU/集成测试。
- 许多能力依赖较重的可选环境，包括 `unsloth`、`trl`、`transformers`、`networkx`、`openai` 和外部 API 凭证。

## 4. 已实现但需要弱化表述的部分

以下内容在代码中有实现或接口，但当前更适合写成“已具备实验实现”而不是“已完全稳定成熟”：

- `RPO` 相关逻辑存在，但 `configs/config.yaml` 中 `training.rpo.alpha` 默认为 `0.0`，注释也明确表示主线默认关闭。
- `LANSDataCollator` 在 `src/oglans/trainer/unsloth_trainer.py` 中被标为 `Legacy`，当前默认训练路径并不依赖它。
- `SCV` 长文档滑窗参数在配置里有预留注释，但目前主实现仍以代码内逻辑为主。
- 训练与推理主契约已经收敛为“纯 JSON 输出”；CoT 一致性检查目前属于评测侧分析能力，而不是默认训练输出格式。
- `experiment` 区域包含一些“预留扩展”字段，不能当作已完成功能写入对外文档。
- `scripts/ablation_study.py`、`run_*_repro_suite.py` 等脚本说明仓库已支持实验编排，但并不等于已经附带完整论文结果。

## 5. 当前最稳妥的项目表述

如果需要一句话描述当前工程状态，建议使用：

“OG-LANS 是一个面向 DuEE-Fin 的研究型事件抽取实验仓库，已经实现中文提示构造、基于本体图的动态负样本采样、SCV 语义校验、IPO/DPO 训练以及本地/API 评测流程，但整体仍处于论文实验工程阶段，尚未整理为通用发布版。”

## 6. 当前学术评测与复现口径

虽然仓库已删除独立的 metrics / protocol / ethics 文档，但相关有效信息仍需要保留在工程说明中。就当前实现而言，较稳妥的论文汇报口径如下：

- 默认论文主指标已经切换为 `doc_role_micro_f1`；`doc_instance_micro_f1`、`doc_combination_micro_f1`、`doc_event_type_micro_f1` 构成当前推荐的主表指标组。
- `strict_f1`、`relaxed_f1`、`type_f1` 仍保留，用于兼容旧结果和工程侧诊断，但不再默认代表最接近 DuEE-Fin / DocEE 文献的主表口径。
- `evaluate.py` 与 `evaluate_api.py` 都会输出结构化 `summary` / `run_manifest` 信息，适合记录 `config hash`、`prompt hash`、`parser version`、`normalization version`、命令行与时间戳。
- 本地训练、本地评测与 API 评测现在共享同一套 `prompt_variant + fewshot_num_examples` 语义；`use_oneshot/use_fewshot` 仅作为兼容旧 CLI 的别名存在。
- 为保证 baseline 与后续自研方法可比，仓库当前默认采用 `train_fit / train_tune / dev_final` 研究协议：`train_fit` 用于训练与 few-shot exemplar pool，`train_tune` 仅用于方法选择，`dev` 只作为冻结后的最终报告集。
- 当前 `comparison.research_split_manifest_path` 已指向固定的 `train_fit/train_tune` 划分清单；这避免了不同入口在运行时重算 split 导致的隐式漂移。
- 当前 summary / metrics 文件已经区分 `academic_metrics` 与 `legacy_metrics`，前者用于论文口径，后者用于兼容旧脚本和旧汇总逻辑。
- 对 dev 集等带金标场景，当前代码已支持 bootstrap 置信区间；多 seed 复现实验脚本还支持聚合统计和 paired permutation 显著性比较。
- `run_api_repro_suite.py` 与 `run_local_repro_suite.py` 现在都会同时导出主指标和成本/效率指标，包括 `total_tokens`、`avg_tokens_per_sample`、`wall_clock_seconds`、`samples_per_second`、`f1_per_1k_tokens` 与 `f1_per_minute`。
- API 评测场景下，应记录请求模型名、实际返回的 `response_model`，以及实际命中的 `base_url`；这一点很重要，因为上游 API 别名和代理路由都可能漂移。
- 自动指标不能替代人工误差分析。当前代码已经保留 error breakdown、hallucination、CoT 一致性与反事实检查，但论文或报告仍应补充人工案例分析。
- 为保证透明性，运行时环境、token usage、wall-clock 等信息也值得保留；这部分当前已经进入 summary 或 manifest，而不是额外依赖手工记录。

## 7. scripts/ 目录的实际作用

当前 `scripts/` 中值得保留在协作文档里的入口如下：

- `run_train.sh`：包装标准训练流程。
- `run_eval_base.sh`：评测本地基座模型。
- `run_eval_api.sh`：运行 API 基线评测。
- `run_eval_academic.sh`：批量执行学术口径评测。
- `run_api_repro_suite.py`：多 seed、多模式 API 复现实验汇总。
- `run_local_repro_suite.py`：本地 checkpoint 多 seed 复现实验汇总。
- `run_mini_matrix.py`：在小样本子集上串起 base/full/A1-A7、zeroshot/fewshot 与多 seed 的 mini 对比矩阵。
- `validate_academic_artifacts.py`：检查评测摘要字段完整性。
- `build_graph.py`：从 schema 构建事件本体图。
- `ablation_study.py`：按配置覆盖组织消融实验。
- `resolve_config_context.py`：为 shell wrapper 解析统一配置上下文。

## 8. 下一步最值得补充的工程工作

如果后续继续完善仓库，优先级最高的不是继续堆叠算法名词，而是补齐工程基础设施：

1. 增加仓库级 `README.md`，明确安装、数据准备、训练、评测和结果目录。
2. 继续围绕 `uv.lock` 固化依赖环境，并补齐 CI 环境文件或最小安装说明。
3. 在可复现环境中跑通并记录测试结果，最好接入自动化 CI。
4. 为训练、评测、复现实验提供最小可运行示例和示例产物。
5. 对论文口径与工程口径做分层，避免配置注释里的“publication ready”误导真实成熟度判断。
