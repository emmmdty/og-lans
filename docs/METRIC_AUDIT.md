# Metric Audit Report

## Scope

Code paths audited:
- `evaluate.py`
- `evaluate_api.py`
- `src/oglans/utils/eval_protocol.py`

## Findings Summary

## Fixed

1. CoT argument consistency previously defaulted to always true when checked.
- Resolution: explicit argument extraction and set-based consistency check added.

2. CoT could return true when thought block was absent.
- Resolution: missing thought now counted as skipped/parse-fail based on protocol settings.

3. Parse success mixed raw/repair without separation.
- Resolution: added `parse_raw_success`, `parse_repair_success`, and extraction-failure counters.

4. Schema failure lacked structured error categories.
- Resolution: added `schema_violation_breakdown`.

5. Hallucination output lacked category-level explainability.
- Resolution: added `hallucination_breakdown` by `event_type|role`.

## Remaining Limitations (Explicitly Declared)

1. CoT parsing is still heuristic (regex-based) and language-style sensitive.
2. Relaxed span matching is approximate for Chinese free-form text.
3. Hallucination detection remains text-grounding based, not NLI-based factual verification.

## Academic Compliance Notes

1. CoT metrics are diagnostic; do not use as primary benchmark claims.
2. Always report metric coverage for any CoT number.
3. Always log protocol + metric version in result artifacts.

