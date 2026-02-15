# Academic Evaluation Protocol

This protocol is designed for reproducible API baseline reporting on DuEE-Fin.

## 1. Preconditions

- Python >= 3.10
- Installed dependencies from `requirements.txt`
- `DEEPSEEK_API_KEY` (or `OPENAI_API_KEY`) configured

## 2. Single Run (One Setting)

Zero-shot:

```bash
python evaluate_api.py \
  --config configs/config.yaml \
  --split dev \
  --model deepseek-chat \
  --seed 3407 \
  --concurrency 8 \
  --json_mode auto
```

Few-shot:

```bash
python evaluate_api.py \
  --config configs/config.yaml \
  --split dev \
  --model deepseek-chat \
  --seed 3407 \
  --concurrency 8 \
  --json_mode auto \
  --use_fewshot
```

Expected outputs under `logs/DuEE-Fin/eval/`:

- `eval_results_deepseek_<split>_<shot>.jsonl`
- `eval_summary_<timestamp>.json`
- `eval_api_<timestamp>.log`

The summary includes:

- strict/relaxed/type metrics
- parse success/failure
- bootstrap confidence intervals (dev with labels)
- token usage
- API failure rate
- runtime manifest (Python/system/package/git)

## 3. Multi-Seed Reproducibility Suite

```bash
python scripts/run_api_repro_suite.py \
  --config configs/config.yaml \
  --split dev \
  --model deepseek-chat \
  --seeds 3407,3408,3409 \
  --modes zeroshot,fewshot \
  --concurrency 8
```

Expected outputs:

- `logs/DuEE-Fin/eval/repro_suite_<timestamp>/suite_summary.json`
- `logs/DuEE-Fin/eval/repro_suite_<timestamp>/suite_summary.md`

`suite_summary.json` includes:

- per-run commands and return codes
- aggregated mean/std/CI across seeds
- paired permutation significance test (`fewshot` vs `zeroshot`)

## 4. Reporting Template (Recommended)

For each setting, report:

- model id in request + actual `response_model` returned by API
- split (`dev`/`test`) and whether gold labels are available
- `strict_f1`, `relaxed_f1`, `type_f1`
- confidence intervals and number of bootstrap samples
- number of seeds and aggregated statistics
- API failures and parse failure rates
- command, config hash, and timestamp

## 5. Important Caveat

`duee_fin_test.json` typically has no gold `event_list` labels in this repository snapshot.
For `--split test`, results are prediction-only and should not be reported as F1.
