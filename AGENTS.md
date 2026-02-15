# Repository Guidelines

## Project Structure & Module Organization
- `src/oglans/` is the core package. Key areas: `data/` (adapters, prompt builders), `trainer/` (Unsloth DPO trainer wrapper), `utils/` (sampling, SCV, logging, reproducibility), and `config.py` (configuration loader).
- `configs/` holds experiment configs (`config.yaml`, `config_debug.yaml`).
- `scripts/` contains runnable utilities (`run_train.sh`, `run_eval_local.sh`, `run_eval_api.sh`, `run_eval_academic.sh`, `build_graph.py`).
- `tests/` contains pytest tests. `main.py` is the training entry point; `evaluate.py` and `evaluate_api.py` handle evaluation.
- Data and outputs live under `data/` and `logs/` (created at runtime).

## Build, Test, and Development Commands
- `pip install -e .` installs the package for local development.
- `pip install -r requirements.txt` installs pinned research dependencies.
- `python main.py --config configs/config.yaml --data_dir ./data/raw/DuEE-Fin` runs training.
- `bash scripts/run_train.sh --data_dir ./data/raw/DuEE-Fin` runs the scripted training wrapper (expects a bash shell).
- `bash scripts/run_eval_local.sh --checkpoint logs/DuEE-Fin/checkpoints/exp1` runs local model evaluation with auto-logged outputs.
- `python -m pytest` runs the test suite.

## Coding Style & Naming Conventions
- Python 3.10+ with 4-space indentation; keep module imports under the `oglans` namespace (e.g., `from oglans.utils import ...`).
- Dev tooling is declared in `pyproject.toml` (`black`, `isort`, `flake8`). No repo-specific config is present, so defaults apply.
- Prefer descriptive, domain-specific names (e.g., `lans_alpha`, `taxonomy_path`) that match config keys.

## Testing Guidelines
- Framework: pytest (see `pytest.ini`).
- Naming: `tests/test_*.py`, classes `Test*`, functions `test_*`.
- Examples: `python -m pytest tests/test_lans.py -v`.

## Commit & Pull Request Guidelines
- This workspace snapshot does not include a `.git` directory, so no established commit convention is visible. Use short, imperative summaries (e.g., “Add SCV threshold guard”) and add a scope when helpful.
- PRs should include: purpose, config changes, training/eval commands run, and links to key logs or metrics outputs (e.g., `logs/.../eval_result_*.jsonl`).

## Configuration & Data Notes
- Default datasets are expected under `data/raw/<dataset>`; processed caches and graphs are written to `data/processed/` and `data/schemas/`.
- Experiment outputs go to `logs/<dataset>/checkpoints` and `logs/<dataset>/tensorboard` as configured in `main.py`.
