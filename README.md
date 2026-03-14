# OG-LANS

OG-LANS (Ontology-Graph Loss-Aware Adaptive Negative Sampling) is a research codebase for document-level financial event extraction with LLMs and direct preference optimization.

This public release keeps the core research code, configuration, tests, and documentation needed to understand and reproduce the framework structure, while excluding private credentials, local synchronization tooling, raw datasets, and experiment artifacts.

## Overview

The repository centers on three research workflows:

- Training with ontology-graph-aware negative sampling and preference optimization.
- Local evaluation for checkpoints or base models.
- API-based baseline evaluation for OpenAI-compatible endpoints.

Key components include:

- `OG-CNS`: ontology-graph-driven negative sample construction.
- `LANS`: loss-aware adaptive negative scheduling.
- `SCV`: semantic consistency verification for filtering noisy negatives.
- Academic evaluation utilities for strict/relaxed metrics, hallucination analysis, and reproducibility manifests.

## Repository Layout

```text
src/oglans/          Core Python package
configs/             Training and evaluation configs
tests/               Pytest suite
docs/                Metric specs and project description material
main.py              Training entrypoint
evaluate.py          Local/base-model evaluation
evaluate_api.py      API-based baseline evaluation
scripts/             Shell wrappers and research utilities
```

The public repository intentionally does not ship the following local-only content:

- Raw DuEE-Fin data under `data/raw/`
- Derived caches under `data/processed/` and `data/schemas/`
- Training/evaluation outputs under `logs/`
- Local backup snapshots under `backup/`
- Secret-bearing local files such as `.env` and sync configuration

## Environment Setup

Python 3.10+ is expected.

```bash
pip install -r requirements.txt
pip install -e .
```

Development tooling declared in `pyproject.toml`:

- `pytest`
- `black`
- `isort`
- `flake8`

## Data Preparation

This repository does not redistribute the DuEE-Fin dataset.

Before running training or evaluation, place the dataset files in the local path below:

```text
data/raw/DuEE-Fin/
```

Expected local files include:

- `duee_fin_train.json`
- `duee_fin_dev.json`
- `duee_fin_test.json`
- `duee_fin_sample.json`
- `duee_fin_event_schema.json`

Please obtain and use the dataset in compliance with the original provider's terms. See `DATA_STATEMENT.md` for repository-specific notes and `DuEE-Fin 数据集详细描述.md` for a local descriptive summary used during development.

## Quick Start

### Train

```bash
python main.py --config configs/config.yaml --data_dir ./data/raw/DuEE-Fin
```

Or via wrapper:

```bash
bash scripts/run_train.sh --data_dir ./data/raw/DuEE-Fin
```

### Evaluate a local checkpoint

```bash
bash scripts/run_eval_local.sh --checkpoint logs/DuEE-Fin/checkpoints/<exp>
```

### Evaluate a base model

```bash
bash scripts/run_eval_base.sh --model-name <model_or_path> --config configs/config.yaml
```

### Evaluate an API baseline

Set credentials in your local shell environment or a local `.env` file that is not committed.

```bash
python evaluate_api.py --config configs/config.yaml --split dev --model deepseek-chat --concurrency 8
```

### Run the API reproducibility suite

```bash
python scripts/run_api_repro_suite.py --config configs/config.yaml --split dev --seeds 3407,3408,3409
```

### Validate academic reporting artifacts

```bash
python scripts/validate_academic_artifacts.py --summary logs/DuEE-Fin/eval_api/<run_id>/eval_summary.json
```

### Run tests

```bash
python -m pytest
```

## Outputs and Reproducibility

When run locally with the required data and dependencies, the project writes outputs to local-only directories such as:

- `logs/<dataset>/checkpoints/`
- `logs/<dataset>/tensorboard/`
- `logs/<dataset>/samples/`
- `logs/<dataset>/train/`
- `logs/<dataset>/eval_local/`
- `logs/<dataset>/eval_base/`
- `logs/<dataset>/eval_api/`
- `logs/<dataset>/eval_academic/`

These outputs are intentionally excluded from the public repository. The codebase includes reproducibility-oriented utilities and documentation, including:

- `ACADEMIC_EVALUATION_PROTOCOL.md`
- `ACADEMIC_METRICS_GUIDE.md`
- `docs/METRIC_SPEC.md`
- `docs/METRIC_AUDIT.md`
- `DATA_STATEMENT.md`
- `ETHICS_AND_LIMITATIONS.md`

## Public Release Scope

This initial GitHub release is a conservative open-source snapshot:

- Included: source code, configs, tests, docs, entry scripts, packaging metadata.
- Excluded: raw data, cached artifacts, checkpoints, logs, backups, local sync scripts, credentials, machine-specific configuration.

The goal is to make the research implementation auditable and reusable without leaking private assets or redistributing materials with uncertain sharing status.

## License

This repository is released under the MIT License. See `LICENSE`.

Dataset, model, and third-party service usage may be subject to separate upstream terms.
