# OG-LANS 仓库协作指南

## 仓库定位

- 本仓库是一个围绕 DuEE-Fin 的学术实验工程，方法主线为“本体图 + SCV + IPO/DPO”的事件抽取。
- 当前主体代码位于 `src/`，并已经包含训练、离线评测、API 基线评测、脚本编排和测试文件。
- 这是研究型仓库，不应按“通用产品”或“论文终稿配套代码包”来理解。

## 权威文档入口

- `DATASET_DUEE_FIN.md`：当前仓库实际使用的 DuEE-Fin 数据快照与字段说明。
- `ENGINEERING_STATUS.md`：代码工程的已完成能力、当前边界和后续可补充项。
- `OG_LANS_METHODOLOGY.md`：项目方法论的学术化表述。

除上述三份文档外，不再维护独立的 metrics、protocol、ethics 或 scripts README 文档。新增说明应优先并入这三份文档，而不是重新分散。

## 项目结构

- `src/oglans/`：核心 Python 包。
- `src/oglans/data/`：DuEE-Fin 数据适配与中文提示词构造。
- `src/oglans/trainer/`：Unsloth 偏好训练与 plain SFT 基线封装。
- `src/oglans/inference/`：CAT-lite 过滤与反事实扰动等推理后处理。
- `src/oglans/utils/`：LANS、SCV、JSON 解析、评测协议、运行清单、路径推断、模型下载运行时等工具。
- `configs/`：当前存在 `config.yaml`、`config_debug.yaml`、`config_compare_base.yaml`、`config_plain_sft.yaml`、`config_phase3_mvp.yaml`、`eval_protocol.yaml`、`role_aliases_duee_fin.yaml`。
- `main.py`：训练入口。
- `evaluate.py`：本地模型与 checkpoint 评测入口。
- `evaluate_api.py`：外部 API 模型评测入口。
- `scripts/`：训练、评测、复现实验、图构建、消融实验和产物校验脚本。
- `tests/`：pytest 测试集；当前有 35 个 `test_*.py` 文件。
- `data/raw/DuEE-Fin/`：当前仓库内实际使用的数据与 schema。
- `logs/`：训练、评测和实验产物输出根目录。

## 常用命令

- `uv sync`：安装项目运行依赖。
- `uv sync --extra dev`：安装开发与测试依赖。
- `uv run python main.py --config configs/config.yaml --data_dir ./data/raw/DuEE-Fin`：运行主训练流程。
- `bash scripts/run_train.sh --data_dir ./data/raw/DuEE-Fin`：通过 shell wrapper 启动训练。
- `uv run python evaluate.py --config configs/config.yaml --checkpoint logs/DuEE-Fin/checkpoints/<exp> --output_file logs/DuEE-Fin/eval_checkpoint/<run_id>/eval_results.jsonl`：评测本地 checkpoint。
- `bash scripts/run_eval_base.sh --model-name <model_or_path> --config configs/config.yaml`：评测本地基座模型。
- `uv run python evaluate_api.py --config configs/config.yaml --split dev --model deepseek-chat --concurrency 8`：运行 API 基线评测。
- `uv run python scripts/run_api_repro_suite.py --config configs/config.yaml --split dev --seeds 3407,3408,3409`：汇总 API 多 seed 复现实验。
- `uv run python scripts/run_local_repro_suite.py --config configs/config.yaml --base-model Qwen/Qwen3-4B-Instruct-2507 --split dev --checkpoints full=logs/DuEE-Fin/checkpoints/exp1`：汇总本地基座模型与 checkpoint 的多 seed 复现实验。
- `uv run python scripts/ablation_study.py --config configs/config.yaml --experiments A1,A2`：运行指定消融实验。
- `uv run python scripts/run_mini_matrix.py --config configs/config.yaml --base_model ./data/cache/modelscope/Qwen/Qwen3-4B-Instruct-2507 --seeds 3407,3408,3409`：运行 mini 全链路对比矩阵（base/full/A1-A7，可配 zeroshot/fewshot）。
- `uv run python scripts/audit_baseline_matrix.py --suite-summary <suite_summary.json>`：审查本地 baseline 四行矩阵，分析 `single_pass/two_stage` 与 `zeroshot/fewshot` 的差异。
- `uv run python scripts/audit_training_signal.py --checkpoint-manifest <run_manifest.json>`：审查训练侧 `LANS/SCV` 负样本统计与导出口径。
- `bash scripts/run_eval_api.sh --preflight`：检查 API 评测运行前置条件。
- `bash scripts/run_eval_academic.sh --help`：查看多 seed 学术口径评测入口。
- `uv run python scripts/build_graph.py --dataset_name DuEE-Fin`：从 schema 构建本体图缓存。
- `uv run python scripts/resolve_config_context.py --config configs/config.yaml`：输出 shell wrapper 可复用的配置上下文。
- `uv run python scripts/validate_academic_artifacts.py --summary <summary.json>`：校验学术汇总字段完整性。
- `uv run pytest`：运行测试。

说明：

- 当前仓库没有 `requirements.txt`，依赖安装以 `pyproject.toml` 为准。
- 当前仓库已包含 `uv.lock`，优先使用 `uv sync` / `uv run`。
- 当前仓库也没有 `README.md`；对仓库的结构理解请以本文件和三份核心文档为准。
- `pyproject.toml` 当前仍引用了缺失的 `README.md` 作为打包元数据；如果后续处理打包或发布，需要同步修正。
- 若当前环境未安装 dev 依赖，`uv run pytest` 或 `python -m pytest` 都会直接失败。
- API 评测默认依赖根目录 `.env` 或外部环境变量中的 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`。

## 编码与测试约定

- 使用 Python 3.10+，4 空格缩进。
- 模块导入保持在 `oglans` 命名空间下，例如 `from oglans.utils import ...`。
- 命名优先贴近实验语义，例如 `taxonomy_path`、`prompt_variant`、`preference_mode`、`canonical_metric_mode`。
- 开发工具在 `pyproject.toml` 中声明，当前可见 dev 依赖包括 `pytest`、`black`、`isort`、`flake8`。
- `pytest.ini` 已将 `src` 加入 `PYTHONPATH`。
- 测试文件命名遵循 `tests/test_*.py`。
- 修改训练、评测、路径推断或脚本行为时，优先补对应子系统附近的定向测试。

## 实验与产物约定

- 原始数据保留在 `data/raw/`；不要把派生产物回写进原始数据目录。
- 图缓存、schema 派生产物和数据缓存应写入 `data/schemas/` 或 `data/processed/`。
- 训练相关产物通常写入 `logs/<dataset>/checkpoints/`、`logs/<dataset>/tensorboard/`、`logs/<dataset>/samples/`。
- 评测相关产物通常写入 `logs/<dataset>/eval_checkpoint/`、`logs/<dataset>/eval_base/`、`logs/<dataset>/eval_api/`、`logs/<dataset>/eval_academic/`。
- 新增脚本时，尽量复用 `jsonl + summary/metrics + run_manifest` 这一产物组织方式。

## 协作说明

- 当前工作区包含 `.git`，提交说明应使用简短、祈使式摘要。
- 如果你需要描述项目背景、方法或数据，不要在 `AGENTS.md` 内重复展开，直接更新对应的三份核心文档。
- 如果你发现仓库结构再次发生变化，应优先同步本文件中的路径、命令和文档入口。
