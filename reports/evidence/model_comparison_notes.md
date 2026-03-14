# Model Comparison Notes

## Scope
- This file documents how the unified comparison table in `model_comparison_table.csv` was built from the 5 audited evaluation runs.
- All numeric values are copied from run-level `summary` or `metrics` files already linked in `run_identity_table.csv`.
- No total metric was recomputed from `eval_results*.jsonl`.

## System ID Mapping
- `academic_full_local`
- `academic_no_scv_local`
- `base_qwen4b_local`
- `api_zeroshot_deepseek_chat`
  - User shorthand alias in this phase: `api_deepseek_zeroshot`
- `api_fewshot_deepseek_chat`
  - User shorthand alias in this phase: `api_deepseek_fewshot`

## Raw Field Shapes By Source Group

### academic / base
- Main source:
  - `eval_results*_summary.json -> metrics.*`
- Core fields available in summary:
  - `strict_precision`, `strict_recall`, `strict_f1`
  - `relaxed_precision`, `relaxed_recall`, `relaxed_f1`
  - `type_precision`, `type_recall`, `type_f1`
  - `hallucination_rate`, `schema_compliance_rate`
  - `primary_metric`, `primary_metric_value`
  - `bootstrap_ci` key exists but is `null` in the audited local/base runs
- Supporting source:
  - `eval_results*_metrics.json`
- Fields used from metrics JSON:
  - `parse_statistics.parse_error_rate`
  - semantic confirmation that `strict` and `type_identification` are separate metric families
- Auxiliary cross-check only:
  - `academic_summary.json`
  - `academic_summary.md`

### api
- Main source:
  - `eval_summary.json -> metrics.*`
- Core fields available in summary:
  - same strict / relaxed / type fields as local summaries
  - `parse_error_rate`
  - `hallucination_rate`, `schema_compliance_rate`
  - `primary_metric`, `primary_metric_value`
  - non-null `bootstrap_ci`
- API-only fields from the same summary:
  - `token_usage.*`
  - `api_stats.*`
- Auxiliary cross-check only:
  - `eval_report.txt`

## Unified Field Mapping
- `primary_metric_name` <- `summary.metrics.primary_metric`
- `primary_metric_value` <- `summary.metrics.primary_metric_value`
- `precision` <- `summary.metrics.strict_precision`
- `recall` <- `summary.metrics.strict_recall`
- `f1` <- `summary.metrics.strict_f1`
- `exact_match_f1` <- `summary.metrics.strict_f1`
  - justification:
    - `ACADEMIC_METRICS_GUIDE.md` defines `Strict F1` as exact triple match
    - `docs/METRIC_SPEC.md` defines `strict_*` as exact `(event_type, role, normalized_argument)` tuple match
- `relaxed_f1` <- `summary.metrics.relaxed_f1`
- `type_f1` <- `summary.metrics.type_f1`
- `parse_error_rate`
  - local runs: `metrics.json -> parse_statistics.parse_error_rate`
  - API runs: `eval_summary.json -> metrics.parse_error_rate`
- `hallucination_rate` <- `summary.metrics.hallucination_rate`
- `schema_compliance_rate` <- `summary.metrics.schema_compliance_rate`
- `bootstrap_ci_lower/bootstrap_ci_upper`
  - API only: `summary.metrics.bootstrap_ci.strict_f1.ci[0/1]`
  - local runs: `NA`
- `token_usage_prompt/completion/total`
  - API only: `summary.token_usage.prompt_tokens/completion_tokens/total_tokens`
  - local runs: `NA`
- `api_cost_or_stats`
  - API only: formatted from `summary.api_stats`
  - local runs: `NA`

## Intentionally Left As NA
- `event_f1`
  - current artifacts expose `type_f1`, which is event-type set overlap, not a separately named event-level extraction F1
- `arg_f1`
  - no standalone argument-level F1 is reported in the audited artifacts
- `schema_violation_rate`
  - current artifacts expose `schema_compliance_rate` and violation breakdown counts
  - this phase does not derive `1 - compliance_rate`, because the requested field name is not directly reported and should not be backfilled by inference
- `metrics_file` for API runs
  - API runs do not have a separate `*_metrics.json`

## bootstrap_ci Clarification
- The local/base summaries and the API summaries share the same field name: `metrics.bootstrap_ci`.
- For the audited local/base runs:
  - `academic_full_local`
  - `academic_no_scv_local`
  - `base_qwen4b_local`
  - the key is present, but the stored value is `null`.
- For the audited API runs:
  - `api_zeroshot_deepseek_chat`
  - `api_fewshot_deepseek_chat`
  - the value is a populated structure with per-metric bootstrap results such as `strict_f1.ci = [lower, upper]`.
- Therefore, the unified table fills `bootstrap_ci_lower/bootstrap_ci_upper` only when a usable CI value exists.
- The local/base rows are `NA` not because the field name is absent, but because the audited runs did not materialize a non-null CI result.

## Why Non-API Token/API Fields Are NA
- Local/base summaries contain top-level `token_usage` and `api_stats` blocks with placeholder-style zero values.
- These blocks do not represent real API consumption for local evaluation.
- To avoid mixing placeholder values with genuine API usage, the unified table maps all non-API token/API fields to `NA`.

## Comparability Rules
- `comparability_level=directly_comparable` means the core metric subset is comparable under the same:
  - `split=dev`
  - `primary_metric=strict_f1`
  - `canonical_metric_mode=analysis_only`
  - `protocol_hash_sha256=23c21ed1d6dba521600f6625192777cb16a27ce5551fbb079f802fbbaef5e2da`
  - `role_alias_map_hash_sha256=df531d0f51331187345167b08985b27520b19cdc5fcda274365cdadd29a7c79b`
  - `pipeline_mode=e2e`
  - `cot_eval_mode=self_consistency`
  - `prompt_style=qwen`
- This label does not mean every column is comparable across all rows.
- API-only extras remain source-specific diagnostics.

## Directly Comparable Metrics
- `primary_metric_name`
- `primary_metric_value`
- `precision`
- `recall`
- `f1`
- `exact_match_f1`
- `relaxed_f1`
- `type_f1`
- `parse_error_rate`
- `hallucination_rate`
- `schema_compliance_rate`

## Partially Comparable Metrics
- `bootstrap_ci_lower`
- `bootstrap_ci_upper`
- `token_usage_prompt`
- `token_usage_completion`
- `token_usage_total`
- `api_cost_or_stats`

These are meaningful within the API runs and can be compared between API zeroshot and API fewshot, but they are not ordinary local-model quality metrics and should not be folded into the cross-source main comparison.

## Not Directly Comparable Or Not Reported
- `event_f1`
- `arg_f1`
- `schema_violation_rate`

## Best Shared Subset For academic_full_local vs academic_no_scv_local
- `precision`
- `recall`
- `f1`
- `exact_match_f1`
- `relaxed_f1`
- `type_f1`
- `parse_error_rate`
- `hallucination_rate`
- `schema_compliance_rate`

## Source Trace By Row
- `academic_full_local`
  - summary: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047_summary.json`
  - metrics: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/eval_results_dev_seed3047_metrics.json`
  - auxiliary check: `logs/DuEE-Fin/eval_academic/20260305_130645_dev/academic_summary.md`
- `academic_no_scv_local`
  - summary: `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047_summary.json`
  - metrics: `logs/DuEE-Fin/eval_academic/20260311_163743_dev/eval_results_dev_seed3047_metrics.json`
  - auxiliary check: `logs/DuEE-Fin/eval_academic/20260311_163743_dev/academic_summary.md`
- `base_qwen4b_local`
  - summary: `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results_summary.json`
  - metrics: `logs/DuEE-Fin/eval_base/20260222_210712_base_model_dev/eval_results_metrics.json`
- `api_zeroshot_deepseek_chat`
  - summary: `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_summary.json`
  - auxiliary check: `logs/DuEE-Fin/eval_api/20260314_121602_dev_seed3407_zeroshot_deepseek-chat_p1838808/eval_report.txt`
- `api_fewshot_deepseek_chat`
  - summary: `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_summary.json`
  - auxiliary check: `logs/DuEE-Fin/eval_api/20260314_121753_dev_seed3407_fewshot_deepseek-chat_p1839514/eval_report.txt`
