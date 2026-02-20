# Shell Scripts Guide

This file is the single source of usage docs for all `scripts/*.sh`.

## Naming Convention

- All shell entrypoints use `run_*.sh`.
- Migration from old names:
  - `scripts/eval_api.sh` -> `scripts/run_eval_api.sh`
  - `scripts/run_eval.sh` -> `scripts/run_eval_local.sh`
  - `scripts/run_academic_eval.sh` -> `scripts/run_eval_academic.sh`

## Quick Start

- Train:
  - `bash scripts/run_train.sh --data_dir ./data/raw/DuEE-Fin`
- Local model eval (single run):
  - `bash scripts/run_eval_local.sh --checkpoint logs/DuEE-Fin/checkpoints/exp1`
- API eval (DeepSeek/OpenAI-compatible):
  - `bash scripts/run_eval_api.sh --action preflight`
  - `bash scripts/run_eval_api.sh --split dev --model deepseek-chat`
- Academic multi-seed eval:
  - `bash scripts/run_eval_academic.sh --checkpoint logs/DuEE-Fin/checkpoints/exp1 --seeds 3407 --split dev --eval-mode both`
- Debug pipeline:
  - `bash scripts/run_debug.sh`
- Train in tmux:
  - `bash scripts/run_train_tmux.sh --session train_main --exp_name main_s3407 --attach`

## Common Runtime Defaults

Most scripts export:
- `PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `WANDB_MODE=offline`
- `PYTHONUNBUFFERED=1`
- Python interpreter is auto-detected (`python` first, fallback to `python3`).

Device defaults:
- All scripts use `CUDA_VISIBLE_DEVICES=0` only as fallback when env var is unset.
- Override with `export CUDA_VISIBLE_DEVICES=<id(s)>` before launching scripts.

## Script Details

### 1) `scripts/run_train.sh`

Purpose:
- Launches `main.py` training with auto experiment naming and terminal log capture.

Common commands:
- Default dataset:
  - `bash scripts/run_train.sh`
- Explicit data dir:
  - `bash scripts/run_train.sh --data_dir ./data/raw/DuEE-Fin`
- Custom experiment name:
  - `bash scripts/run_train.sh --exp_name exp_custom --data_dir ./data/raw/DuEE-Fin`
- Explicit schema path:
  - `bash scripts/run_train.sh --data_dir /path/to/data --schema_path /path/to/schema.json`

Runtime notes:
- `run_train.sh` now prints a startup config summary (`seed/max_steps/LANS/SCV/refresh_start_epoch`).
- `run_train.sh` auto-detects `python`/`python3` and uses the first available interpreter.
- If training appears idle before GPU usage, look for heartbeat logs:
  - `初始负样本进度: ...`
  - `Epoch <n> 负样本刷新进度: ...`
  - `SCV 心跳: ...`

Outputs:
- Checkpoints: `logs/<dataset>/checkpoints/<exp_name>`
- Run directory: `logs/<dataset>/train/<run_id>/`
- Terminal log: `logs/<dataset>/train/<run_id>/run.log`
- Run manifest: `logs/<dataset>/train/<run_id>/run_manifest.json`

### 2) `scripts/run_eval_local.sh`

Purpose:
- Runs `evaluate.py` once for a local checkpoint and auto-saves log + result JSONL.
- For quantized (4bit/8bit) models, `evaluate.py` skips `model.to(...)` and relies on loader device placement.

Required:
- `--checkpoint <path>`

Options:
- `--dataset-name <name>` (optional)
- `--config <path>` (default: `configs/config.yaml`)
- `--protocol <path>` (default: `configs/eval_protocol.yaml`)
- `--role-alias-map <path>` (default: `configs/role_aliases_duee_fin.yaml`)
- `--canonical-mode <off|analysis_only|apply_for_aux_metric>`
- `--primary-metric <name>` (optional)
- `--exp-name <name>` or `--exp_name <name>`
- `--split <name>` (default: `dev`)
- `--batch-size <int>` (default: `16`)
- Any extra args are forwarded to `evaluate.py`.

Examples:
- `bash scripts/run_eval_local.sh --checkpoint logs/DuEE-Fin/checkpoints/exp1`
- `bash scripts/run_eval_local.sh --checkpoint logs/DuEE-Fin/checkpoints/exp1 --split test --num_samples 200`

Outputs:
- Run directory: `logs/<dataset>/eval_local/<run_id>/`
- Log: `logs/<dataset>/eval_local/<run_id>/run.log`
- Result: `logs/<dataset>/eval_local/<run_id>/eval_results.jsonl`
- Metrics: `logs/<dataset>/eval_local/<run_id>/eval_results_metrics.json`
- Summary: `logs/<dataset>/eval_local/<run_id>/eval_results_summary.json`
- Run manifest: `logs/<dataset>/eval_local/<run_id>/run_manifest.json`

### 3) `scripts/run_eval_api.sh`

Purpose:
- Unified launcher for `evaluate_api.py`, supports single run, preflight, and reproducibility sweep.

Actions:
- `--action run` (default)
- `--action preflight`
- `--action sweep`

Key options:
- `--config <path>` (default: `configs/config.yaml`)
- `--protocol <path>` (default: `configs/eval_protocol.yaml`)
- `--split <dev|test|train>` (default: `dev`)
- `--model <name>` (default: `deepseek-chat`)
- `--concurrency <int>` (default: `8`)
- `--fewshot` / `--zeroshot`
- `--seed <int>`
- `--num-samples <int>`
- `--json-mode <auto|on|off>`
- `--role-alias-map <path>`
- `--canonical-mode <off|analysis_only|apply_for_aux_metric>`
- `--primary-metric <name>`
- `--no-ci`
- `--output-file <path>`
- `--summary-file <path>`
- `--background` (run action only)
- `--smoke` (shortcut: `--num-samples 20 --concurrency 2`)

Sweep-only:
- `--seeds <csv>` (default: `3407,3408,3409`)
- `--variants <csv>` (default: `zeroshot,fewshot`)

Examples:
- `bash scripts/run_eval_api.sh --action preflight`
- `bash scripts/run_eval_api.sh --split dev --model deepseek-chat`
- `bash scripts/run_eval_api.sh --fewshot --split dev`
- `bash scripts/run_eval_api.sh --action sweep --split dev --seeds 3407,3408,3409 --variants zeroshot,fewshot`

Notes:
- API key can be provided by env (`DEEPSEEK_API_KEY` / `OPENAI_API_KEY`) or project `.env`.
- Default run artifacts are written to `logs/<dataset>/eval_api/<run_id>/` with:
  `run.log`, `eval_results.jsonl`, `eval_summary.json`, `run_manifest.json`.
- If `--config` points to a debug project path like `logs/debug/...`, API artifacts are routed to `logs/debug/eval_api/...`.

### 4) `scripts/run_eval_academic.sh`

Purpose:
- Academic reporting pipeline for local checkpoint evaluation across multiple seeds.
- Produces per-seed outputs + aggregated `mean/std/min/max`.

Required:
- `--checkpoint <path>`

Core options:
- `--dataset-name <name>` (optional)
- `--config <path>` (default: `configs/config.yaml`)
- `--protocol <path>` (default: `configs/eval_protocol.yaml`)
- `--role-alias-map <path>` (default: `configs/role_aliases_duee_fin.yaml`)
- `--canonical-mode <off|analysis_only|apply_for_aux_metric>`
- `--primary-metric <name>` (optional)
- `--split <dev|test|train>` (default: `dev`)
- `--batch-size <int>` (default: `4`)
- `--eval-mode <strict|relaxed|both>` (default: `both`)
- `--seeds <csv>` (default: `3407,3408,3409`)
- `--seed-policy <train_seed|eval_seed>` (default: `train_seed`)
- `--num-samples <int>`
- `--use-oneshot`
- `--do-sample` (not recommended for paper main numbers)
- `--allow-weak-seed-sweep` (exploratory only for `eval_seed` + deterministic decode)
- `--out-dir <path>`
- `--tag <str>`
- `--continue-on-error`

Examples:
- Recommended paper setting (`train_seed`): one checkpoint per seed:
  `bash scripts/run_eval_academic.sh --seed-policy train_seed --seeds 3407,3408,3409 --checkpoint logs/DuEE-Fin/checkpoints/exp_seed3407,logs/DuEE-Fin/checkpoints/exp_seed3408,logs/DuEE-Fin/checkpoints/exp_seed3409 --split dev --eval-mode both`
- Exploratory setting (`eval_seed`) on one checkpoint:
  `bash scripts/run_eval_academic.sh --seed-policy eval_seed --checkpoint logs/DuEE-Fin/checkpoints/exp1 --seeds 3407,3408,3409 --allow-weak-seed-sweep --split dev`

Outputs:
- `run_manifest.json`
- `run_seed<seed>.log`
- `eval_results_<split>_seed<seed>.jsonl`
- `eval_results_<split>_seed<seed>_metrics.json`
- `academic_summary.json`
- `academic_summary.md`

### 5) `scripts/run_debug.sh`

Purpose:
- Quick end-to-end sanity run (tests + debug train + small eval).

Options:
- `--skip-tests`
- `--skip-training`
- `--skip-eval`
- `--allow-test-fail` (default is strict: tests fail -> script exits)
- `--quick` (equivalent to `--max-steps 5 --eval-samples 5`)
- `--max-steps N` (default: `20`)
- `--eval-samples N` (default: `10`)

Examples:
- `bash scripts/run_debug.sh`
- `bash scripts/run_debug.sh --quick`
- `bash scripts/run_debug.sh --skip-tests`
- `bash scripts/run_debug.sh --skip-eval`

### 6) `scripts/run_train_tmux.sh`

Purpose:
- Launches `run_train.sh` inside a detached tmux session for long-running server jobs.

Key options:
- `--session <name>` tmux session name
- `--env <name>` conda env name (default `tjk_ee`)
- `--gpu <id>` CUDA device id
- `--config <path>` config path
- `--data_dir <path>` dataset dir
- `--exp_name <name>` experiment name
- `--attach` attach after launch
- `--force-restart` replace existing same-name session

Example:
- `bash scripts/run_train_tmux.sh --session train_main_s3407 --env tjk_ee --gpu 0 --config configs/config.yaml --data_dir ./data/raw/DuEE-Fin --exp_name main_s3407 --attach`
