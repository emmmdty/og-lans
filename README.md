# OG-LANS

Ontology-Graph Loss-Aware Adaptive Negative Sampling for LLM-based Event Extraction.

## Scope

This repository provides:

- Training and ablation utilities for OG-LANS.
- Local model evaluation (`evaluate.py`).
- API-based baseline evaluation (`evaluate_api.py`) for DeepSeek/OpenAI-compatible endpoints.

## Quick Start

```bash
pip install -r requirements.txt
```

Set API key (recommended):

```bash
export DEEPSEEK_API_KEY="your_key"
```

Run API baseline on DuEE-Fin dev:

```bash
python evaluate_api.py --config configs/config.yaml --split dev --model deepseek-chat --concurrency 8
```

Run zero-shot vs few-shot reproducibility suite (multi-seed):

```bash
python scripts/run_api_repro_suite.py --config configs/config.yaml --split dev --seeds 3407,3408,3409
```

## Reproducibility Artifacts

- `evaluate_api.py` writes:
  - per-sample predictions (`*.jsonl`)
  - summary report (`*_summary.json`)
  - runtime manifest (embedded in summary)
  - token usage and API failure stats
  - bootstrap confidence intervals (when gold labels exist)
- `scripts/run_api_repro_suite.py` writes:
  - `suite_summary.json` (aggregated mean/std/CI)
  - `suite_summary.md` (publication-ready tables)

For full protocol and checklist:

- `ACADEMIC_EVALUATION_PROTOCOL.md`
- `REPRODUCIBILITY_CHECKLIST.md`

Validate whether a generated summary contains required reporting fields:

```bash
python scripts/validate_academic_artifacts.py --summary logs/DuEE-Fin/eval/eval_summary_<timestamp>.json
```
