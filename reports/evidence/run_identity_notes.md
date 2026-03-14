# Run Identity Notes

## Scope
- This file documents the evidence chain for the 5 audited evaluation runs listed in `run_identity_table.csv`.
- Identity resolution is `manifest-first`, `summary/metrics-second`, `dirname/filename auxiliary`, with conflicts recorded explicitly.
- `internal_seed` means the normalized seed chosen for downstream joins: prefer metrics `_meta.seed` when a metrics file exists, otherwise use summary `meta.seed`.

## Field Sources
- `dirname_seed`: parsed from the evaluation directory basename only when the basename contains `seed####` or `_s####`.
- `filename_seed`: parsed from the main summary/metrics/jsonl filenames only when the filename contains `seed####`.
- `manifest_seed`: `run_manifest.json -> meta.seed`.
- `summary_seed`: main summary file `meta.seed` or `_meta.seed`.
- `internal_seed`: metrics `_meta.seed` for `academic/base`; summary `meta.seed` for `api`.

## academic_full_local
- Evaluation directory: `logs/DuEE-Fin/eval_academic/20260305_130645_dev`
- Primary identity evidence:
  - `logs/DuEE-Fin/eval_academic/20260305_130645_dev/run_manifest.json`
  - `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047_summary.json`
  - `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047_metrics.json`
- Identity chain:
  - `run_manifest.json -> meta.checkpoint` points to `.../checkpoints/main_s3047_v5/checkpoint-660`
  - summary and metrics both carry `seed=3047`
  - file names also carry `seed3047`
- Full-module evidence:
  - `logs/DuEE-Fin/train/20260303_092232_main_s3047_v5/run.log` contains `LANS=True | SCV=True`
  - the same log contains `Loading SCV model`, `SCV 断言测试`, and repeated `SCV 心跳`
  - `logs/DuEE-Fin/samples/main_s3047_v5/scv_filtered_samples.jsonl` exists
  - `logs/DuEE-Fin/samples/main_s3047_v5/lans_sampling_summary.json` reports `scv_filtered_count=35537`
- Conflicts:
  - linked training manifest `logs/DuEE-Fin/train/20260303_092232_main_s3047_v5/run_manifest.json` records `seed=3407`
  - the linked training experiment name is `main_s3047_v5`, which visually suggests `3047`
  - evaluation identity is therefore anchored to the evaluation artifacts themselves, not to the training manifest seed field

## academic_no_scv_local
- Evaluation directory: `logs/DuEE-Fin/eval_academic/20260311_163743_dev`
- Primary identity evidence:
  - `logs/DuEE-Fin/eval_academic/20260311_163743_dev/run_manifest.json`
  - `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047_summary.json`
  - `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047_metrics.json`
- Identity chain:
  - `run_manifest.json -> meta.checkpoint` points to `.../checkpoints/A2_no_scv_s3047`
  - summary and metrics both carry `seed=3047`
  - file names also carry `seed3047`
- No-SCV evidence:
  - `logs/DuEE-Fin/train/20260306_191921_A2_no_scv_s3047/run_manifest.json -> command` includes `--algorithms.scv.enabled false`
  - `logs/DuEE-Fin/samples/A2_no_scv_s3047/lans_sampling_summary.json` reports `scv_filtered_count=0`
  - `logs/DuEE-Fin/samples/A2_no_scv_s3047/scv_filtered_samples.jsonl` does not exist
- Conflicts:
  - the linked training manifest still records `seed=3407`
  - the linked training manifest status is `running`
  - `logs/DuEE-Fin/train/20260306_191921_A2_no_scv_s3047/run.log` later reports `[SUCCESS] Experiment A2_no_scv_s3047 completed.`
  - the same training log header still shows `LANS=True | SCV=True`, which conflicts with the command-level `--algorithms.scv.enabled false`
  - for this run, the no-SCV label is therefore supported by the command string plus sample-side SCV outputs, not by the log header alone

## base_qwen4b_local
- Evaluation directory: `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev`
- Primary identity evidence:
  - `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/run_manifest.json`
  - `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results_summary.json`
  - `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results_metrics.json`
- Identity chain:
  - `run_manifest.json -> meta.model_variant = base_only`
  - `run_manifest.json -> meta.control_group_tag = qwen_base_local`
  - summary and metrics carry `seed=3407`
  - no checkpoint or adapter path is recorded
- Notes:
  - directory and filenames do not carry an explicit seed token
  - seed identity comes from manifest/summary/metrics only

## api_zeroshot_deepseek_chat
- Evaluation directory: `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808`
- Primary identity evidence:
  - `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/run_manifest.json`
  - `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_summary.json`
  - `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_results.jsonl`
- Identity chain:
  - directory basename includes `seed3407`
  - manifest and summary both record `seed=3407`
  - command does not include `--use_fewshot`
  - `eval_summary.json -> meta.use_fewshot = false`
- API-specific fields:
  - `eval_summary.json -> metrics.bootstrap_ci` exists and is non-null
  - `eval_summary.json -> token_usage` exists
  - `eval_summary.json -> api_stats` exists
  - `eval_results.jsonl` sampled row includes `response_meta`

## api_fewshot_deepseek_chat
- Evaluation directory: `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514`
- Primary identity evidence:
  - `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/run_manifest.json`
  - `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_summary.json`
  - `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_results.jsonl`
- Identity chain:
  - directory basename includes `seed3407`
  - manifest and summary both record `seed=3407`
  - command includes `--use_fewshot`
  - `eval_summary.json -> meta.use_fewshot = true`
  - `eval_summary.json -> meta.fewshot_num_examples = 3`
  - `eval_summary.json -> meta.prompt_hashes.fewshot_example_indices = [0,1,2]`
- API-specific fields:
  - `eval_summary.json -> metrics.bootstrap_ci` exists and is non-null
  - `eval_summary.json -> token_usage` exists
  - `eval_summary.json -> api_stats` exists
  - `eval_results.jsonl` sampled row includes `response_meta`

## Do Not Use As Identity Source
- `analysis.protocol.evaluation.seeds` inside evaluation summary files reflects protocol configuration, not the realized run seed
- directory names like `main_s3047_v5` or `A2_no_scv_s3047` are not authoritative on their own
- training manifest `seed` fields are not authoritative for linked evaluation identity when they conflict with evaluation-local artifacts
