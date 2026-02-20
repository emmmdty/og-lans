# Reproducibility Checklist (Repository-Level)

This checklist tracks what is already implemented and what remains required for publication submission packages.

## Experimental Setup

- [x] Fixed random seed configurable from CLI (`evaluate_api.py --seed`)
- [x] Config hash recorded in summary (`config_hash_sha256`)
- [x] Command line recorded in summary (`meta.command`)
- [x] Split and sample count explicitly recorded
- [x] Prompt mode recorded (`zeroshot` / `fewshot`)
- [x] Local academic eval seed policy documented (`train_seed` preferred over weak `eval_seed`)

## Metrics and Statistical Reporting

- [x] Strict / Relaxed / Type metrics reported
- [x] Parse success/failure reported
- [x] API failure statistics reported
- [x] Bootstrap confidence intervals supported (`--compute_ci`, `--bootstrap_samples`)
- [x] Multi-seed aggregation script available (`scripts/run_api_repro_suite.py`)
- [x] Paired permutation significance test across seeds

## Runtime and Environment Traceability

- [x] Python/runtime metadata captured
- [x] Core package versions captured
- [x] Git commit/dirty status captured when available
- [x] API response model ids captured per sample and summary

## Data and Evaluation Integrity

- [x] Schema-compatible event typing and roles in prompt templates
- [x] Prediction-only fallback for unlabeled test split
- [x] Parse diagnostics persisted for every sample

## Remaining for Submission Package (Paper-Level)

- [x] Dataset license/access statement documented (`DATA_STATEMENT.md`)
- [x] Ethical/societal impact statement documented (`ETHICS_AND_LIMITATIONS.md`)
- [x] Compute budget proxy logged (`token_usage`, `runtime.wall_clock_seconds`)
- [x] Error breakdown exported (`metrics.error_breakdown` in summaries)
