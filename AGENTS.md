# OG-LANS 仓库协作指南

## 项目结构
- `src/oglans/` 是核心 Python 包。
- `src/oglans/data/` 负责数据适配与提示词构造。
- `src/oglans/inference/` 负责推理后处理逻辑，包括 `cat_lite` 过滤与反事实扰动。
- `src/oglans/trainer/` 封装 Unsloth DPO 训练流程。
- `src/oglans/utils/` 包含 LANS/SCV、评测协议、运行清单、量化判断、路径推断与可复现性工具。
- `configs/` 存放主训练配置与评测相关配置，包括 `config.yaml`、`config_debug.yaml`、`eval_protocol.yaml`、`role_aliases_duee_fin.yaml`。
- `main.py` 是训练入口；`evaluate.py` 负责本地模型与基础模型评测；`evaluate_api.py` 负责 API 基线评测。
- `scripts/` 存放运行包装脚本与研究辅助脚本。主线脚本包括 `run_train.sh`、`run_eval_base.sh`、`run_eval_api.sh`、`run_eval_academic.sh`；复现实验与结果检查脚本包括 `run_api_repro_suite.py`、`validate_academic_artifacts.py`；研究辅助脚本包括 `build_graph.py`、`ablation_study.py` 与 `resolve_config_context.py`。
- `tests/` 是 pytest 测试集，覆盖配置加载、训练调度、SCV/LANS、CAT-lite、指标语义、运行清单、基础评测模式与 API 复现实验。
- `docs/` 存放指标规范、审计说明与项目描述文档，`PROJECT_DESCRIPTION_*` 属于项目说明材料，不是运行入口。
- `data/raw/<dataset>/` 存放原始数据集；`data/processed/` 与 `data/schemas/` 存放派生缓存、图或 schema 产物。
- `logs/<dataset>/` 存放训练和评测产物。当前工作区保留的主线实验目录是 `logs/DuEE-Fin/`。

## 常用命令
- `pip install -e .`：以可编辑模式安装本项目。
- `pip install -r requirements.txt`：安装研究依赖。
- `python main.py --config configs/config.yaml --data_dir ./data/raw/DuEE-Fin`：运行主训练流程。
- `bash scripts/run_train.sh --data_dir ./data/raw/DuEE-Fin`：通过脚本包装训练流程。
- `python evaluate.py --config configs/config.yaml --checkpoint logs/DuEE-Fin/checkpoints/<exp> --output_file logs/DuEE-Fin/eval_checkpoint/<run_id>/eval_results.jsonl`：评测本地 checkpoint。
- `bash scripts/run_eval_base.sh --model-name <model_or_path> --config configs/config.yaml`：评测基础模型模式。
- `python evaluate_api.py --config configs/config.yaml --split dev --model deepseek-chat --concurrency 8`：运行 API 基线评测。
- `python scripts/run_api_repro_suite.py --config configs/config.yaml --split dev --seeds 3407,3408,3409`：运行多种子复现实验汇总。
- `python scripts/validate_academic_artifacts.py --summary logs/DuEE-Fin/eval_api/<run_id>/eval_summary.json`：校验学术评测产物字段完整性。
- `python -m pytest`：运行测试套件。
- `bash scripts/...` 命令默认假设可用的 bash 环境；Python 命令默认假设解释器在 `PATH` 中。

## 编码与测试约定
- 使用 Python 3.10+，4 空格缩进。
- 模块导入保持在 `oglans` 命名空间下，例如 `from oglans.utils import ...`。
- 命名尽量贴近实验与配置语义，例如 `lans_alpha`、`taxonomy_path`、`canonical_metric_mode`。
- 开发工具声明在 `pyproject.toml` 中，当前可见的开发依赖包括 `pytest`、`black`、`isort`、`flake8`。
- `pytest.ini` 已将 `tests/` 设为测试根目录，并将 `src` 加入 `PYTHONPATH`。
- 测试命名遵循 `tests/test_*.py`、`Test*`、`test_*`。
- 修改训练、评测或指标逻辑时，优先补充对应子系统附近的定向测试，而不是只依赖整套回归。

## 实验与产物约定
- 默认数据目录是 `data/raw/<dataset>/`，不要把派生产物混入原始数据目录。
- 训练相关产物通常写入 `logs/<dataset>/checkpoints/`、`logs/<dataset>/tensorboard/`、`logs/<dataset>/samples/`、`logs/<dataset>/train/`。
- 评测相关产物通常写入 `logs/<dataset>/eval_checkpoint/`、`logs/<dataset>/eval_base/`、`logs/<dataset>/eval_api/`、`logs/<dataset>/eval_academic/`。
- 图构建、schema 解析和缓存类中间产物应写入 `data/schemas/` 或 `data/processed/`，保持源码目录干净。
- 评测输出通常同时包含 `jsonl` 预测、`summary.json`、`metrics.json`、`run_manifest.json` 或学术汇总文件；新增脚本时尽量复用这一产物组织方式。

## 文档与提交说明
- `README.md` 提供项目范围、快速开始和复现实验入口。
- `ACADEMIC_EVALUATION_PROTOCOL.md`、`ACADEMIC_METRICS_GUIDE.md`、`REPRODUCIBILITY_CHECKLIST.md` 规定了评测、汇报和复现口径。
- `DATA_STATEMENT.md`、`ETHICS_AND_LIMITATIONS.md`、`CITATION.cff` 属于发布与论文配套元信息。
- `docs/METRIC_SPEC.md` 与 `docs/METRIC_AUDIT.md` 用于指标定义与审计；`docs/PROJECT_DESCRIPTION_*` 用于项目说明与论文叙述参考。
- 当前工作区快照不包含 `.git` 目录，因此看不到仓库内既有提交规范；如需提交说明，优先使用简短、祈使式摘要，并在必要时补充实验范围。
