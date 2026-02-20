# Metric Specification (v2.0)

This document defines the evaluation metric semantics used by `evaluate.py` and `evaluate_api.py`.

## Core Metrics

1. `strict_precision/recall/f1`
- Unit: `(event_type, role, normalized_argument)` exact tuple match.

2. `relaxed_precision/recall/f1`
- Unit: `(event_type, role, argument)` with configurable argument matching.
- Config:
  - `metrics.relaxed.match_mode: include_or_char_overlap | span_iou`
  - `metrics.relaxed.char_overlap_threshold`
  - `metrics.relaxed.span_iou_threshold`

3. `type_precision/recall/f1`
- Unit: event type set overlap.

4. Parse diagnostics
- `parse_raw_success`: JSON parsed without repair.
- `parse_repair_success`: JSON parsed after repair steps.
- `parse_errors`: JSON parse failed.
- `parse_extraction_failures`: extraction method `no_json_found`.

5. Schema compliance
- `schema_compliance_rate`: per-sample schema pass rate.
- Config:
  - `metrics.schema.mode: syntax_only | schema_strict`

6. Hallucination
- `hallucination_rate`: samples containing at least one unsupported argument span.
- `hallucination_entity_rate`: hallucinated arguments / total checked arguments.
- Config:
  - `metrics.hallucination.match_mode: normalized_substring | exact_span`

## Diagnostic Metrics (Not Main Claim)

1. `cot_type_consistency`
2. `cot_argument_consistency`
3. `cot_faithfulness`
4. `cot_coverage_rate`

CoT metrics are diagnostic and should not be used as the paper’s primary optimization target.

## CoT Evaluation Rules

1. CoT checks are enabled only when `metrics.cot.enabled=true`.
2. If `metrics.cot.require_thought_block=true`, missing `<thought>...</thought>` is counted as `cot_skipped`.
3. `cot_checked` is incremented only when thought text can be extracted.
4. `cot_faithfulness` is a conjunction-based consistency score over type and argument checks.
5. Coverage must always be reported with CoT metrics:
- `cot_coverage_rate = cot_checked / (cot_checked + cot_skipped + cot_parse_fail)`

## Reporting Rules

1. Main tables should prioritize:
- Strict/Relaxed/Type F1
- Parse diagnostics
- Schema compliance
- Hallucination metrics
2. CoT metrics should be presented in diagnostics/appendix with coverage.
3. Protocol and metric versions must be logged:
- `protocol.version`
- `metrics.version`

