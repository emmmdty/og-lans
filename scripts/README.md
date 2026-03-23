# Scripts 说明

当前 `scripts/` 目录中的脚本如下，每个文件只保留一句话说明，便于快速查找入口。

| 文件 | 说明 |
|---|---|
| `ablation_study.py` | 批量运行消融实验并汇总各组件对结果的影响。 |
| `build_graph.py` | 根据事件 schema 构建本体图，供 OG-CNS/负采样相关流程使用。 |
| `resolve_config_context.py` | 通过 `ConfigManager` 解析配置并向 shell wrapper 输出统一上下文信息。 |
| `run_api_repro_suite.py` | 编排 API 多种子、zero-shot/few-shot 复现实验并做汇总。 |
| `run_eval_academic.sh` | 以学术汇报口径批量执行多 seed checkpoint 评测并聚合结果。 |
| `run_eval_api.sh` | 包装 `evaluate_api.py`，用于运行、预检或 sweep API 基线评测。 |
| `run_eval_base.sh` | 包装 `evaluate.py --base_only`，用于评测本地基座模型。 |
| `run_train.sh` | 包装 `main.py`，用于启动标准训练流程。 |
| `validate_academic_artifacts.py` | 检查评测摘要文件是否满足论文汇报所需字段完整性。 |
