"""
Academic evaluation utilities for reproducible reporting.

This module is intentionally dependency-light so that statistical logic
can be unit-tested without requiring model/runtime dependencies.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CountDict = Dict[str, int]
MIN_SIGNIFICANCE_PAIRS = 2

ACADEMIC_MAIN_TABLE_METRICS = (
    "doc_role_micro_f1",
    "doc_instance_micro_f1",
    "doc_combination_micro_f1",
    "doc_event_type_micro_f1",
)
CORE_DIAGNOSTIC_REPORT_METRICS = (
    "legacy_dueefin_overall_precision",
    "legacy_dueefin_overall_recall",
    "legacy_dueefin_overall_f1",
    "strict_precision",
    "strict_recall",
    "strict_f1",
    "relaxed_f1",
    "type_f1",
    "schema_compliance_rate",
    "hallucination_rate",
)
SUMMARY_DIAGNOSTIC_REPORT_METRICS = (
    "parse_success_rate",
    "parse_error_rate",
    "avg_gold_events",
    "avg_predicted_events",
    "avg_gold_event_types",
    "avg_schema_event_types",
    "stage1_gold_coverage_rate",
    "stage1_exact_match_rate",
    "stage1_miss_rate",
    "stage1_overprediction_rate",
    "avg_stage1_predicted_types",
    "correction_applied_rate",
    "records_split_count",
    "roles_rewritten_count",
    "roles_added_count",
    "events_dropped_after_correction",
)
LOCAL_SUITE_REPORT_METRICS = ACADEMIC_MAIN_TABLE_METRICS + CORE_DIAGNOSTIC_REPORT_METRICS
API_SUITE_REPORT_METRICS = ACADEMIC_MAIN_TABLE_METRICS + (
    "strict_f1",
    "relaxed_f1",
    "type_f1",
    "schema_compliance_rate",
    "hallucination_rate",
)
OPTIONAL_REPORT_METRICS = (
    "parse_error_rate",
    "parse_success_rate",
    "cot_faithfulness",
    "avg_tokens_per_sample",
    "total_tokens",
    "wall_clock_seconds",
    "samples_per_second",
)
COST_REPORT_METRICS = (
    "avg_tokens_per_sample",
    "total_tokens",
    "wall_clock_seconds",
    "samples_per_second",
)
EFFICIENCY_REPORT_METRICS = (
    "f1_per_1k_tokens",
    "f1_per_minute",
)


def _mapping_get(mapping: Mapping[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        overall = value.get("overall")
        if overall is None:
            return None
        return float(overall)
    return float(value)


def extract_report_metrics(
    payload: Mapping[str, Any],
    *,
    required_metrics: Sequence[str] | None = None,
    optional_metrics: Sequence[str] | None = None,
) -> Dict[str, float]:
    """
    Normalize evaluation metrics across metrics.json, summary["metrics"], and legacy blocks.

    Args:
        payload: Metrics dict or full evaluation summary dict.
        required_metrics: Metrics that must be present; missing entries raise ValueError.
        optional_metrics: Metrics to include when available without raising.

    Returns:
        A flat metric mapping suitable for suite aggregation and reporting.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping.")

    metrics_obj = payload
    if isinstance(payload.get("metrics"), Mapping):
        metrics_obj = payload["metrics"]  # type: ignore[index]
    diagnostics = _mapping_get(payload, "diagnostics") or {}
    cost = _mapping_get(payload, "cost") or {}
    token_usage = _mapping_get(payload, "token_usage") or {}
    runtime = _mapping_get(payload, "runtime") or {}

    if not isinstance(metrics_obj, Mapping):
        raise TypeError("metrics payload must be a mapping.")

    academic_metrics = _mapping_get(metrics_obj, "academic_metrics") or {}
    doc_ee = _mapping_get(academic_metrics, "doc_ee") or {}
    legacy_dueefin = _mapping_get(academic_metrics, "legacy_dueefin") or {}
    legacy_dueefin_overall = _mapping_get(legacy_dueefin, "overall") or {}
    overall = _mapping_get(doc_ee, "overall") or {}
    instance = _mapping_get(doc_ee, "instance") or {}
    combination = _mapping_get(doc_ee, "combination") or {}
    classification = _mapping_get(doc_ee, "classification") or {}

    legacy_metrics = _mapping_get(metrics_obj, "legacy_metrics") or {}
    strict = _mapping_get(metrics_obj, "strict") or _mapping_get(legacy_metrics, "strict") or {}
    relaxed = _mapping_get(metrics_obj, "relaxed") or _mapping_get(legacy_metrics, "relaxed") or {}
    type_identification = (
        _mapping_get(metrics_obj, "type_identification")
        or _mapping_get(legacy_metrics, "type_identification")
        or {}
    )
    parse_statistics = (
        _mapping_get(metrics_obj, "parse_statistics")
        or _mapping_get(legacy_metrics, "parse_statistics")
        or {}
    )
    hallucination = _mapping_get(metrics_obj, "hallucination") or _mapping_get(legacy_metrics, "hallucination") or {}
    cot_faithfulness = (
        _mapping_get(metrics_obj, "cot_faithfulness")
        or _mapping_get(legacy_metrics, "cot_faithfulness")
        or None
    )

    candidates: Dict[str, float | None] = {
        "doc_role_micro_f1": _to_float_or_none(metrics_obj.get("doc_role_micro_f1", overall.get("MicroF1"))),
        "doc_instance_micro_f1": _to_float_or_none(
            metrics_obj.get("doc_instance_micro_f1", instance.get("MicroF1"))
        ),
        "doc_combination_micro_f1": _to_float_or_none(
            metrics_obj.get("doc_combination_micro_f1", combination.get("MicroF1"))
        ),
        "doc_event_type_micro_f1": _to_float_or_none(
            metrics_obj.get("doc_event_type_micro_f1", classification.get("MicroF1"))
        ),
        "legacy_dueefin_overall_precision": _to_float_or_none(
            metrics_obj.get(
                "legacy_dueefin_overall_precision",
                legacy_dueefin_overall.get("precision"),
            )
        ),
        "legacy_dueefin_overall_recall": _to_float_or_none(
            metrics_obj.get(
                "legacy_dueefin_overall_recall",
                legacy_dueefin_overall.get("recall"),
            )
        ),
        "legacy_dueefin_overall_f1": _to_float_or_none(
            metrics_obj.get(
                "legacy_dueefin_overall_f1",
                legacy_dueefin_overall.get("f1"),
            )
        ),
        "strict_precision": _to_float_or_none(metrics_obj.get("strict_precision", strict.get("precision"))),
        "strict_recall": _to_float_or_none(metrics_obj.get("strict_recall", strict.get("recall"))),
        "strict_f1": _to_float_or_none(metrics_obj.get("strict_f1", strict.get("f1"))),
        "relaxed_f1": _to_float_or_none(metrics_obj.get("relaxed_f1", relaxed.get("f1"))),
        "type_f1": _to_float_or_none(metrics_obj.get("type_f1", type_identification.get("f1"))),
        "schema_compliance_rate": _to_float_or_none(
            metrics_obj.get("schema_compliance_rate", legacy_metrics.get("schema_compliance_rate"))
        ),
        "hallucination_rate": _to_float_or_none(
            diagnostics.get(
                "hallucination_rate",
                metrics_obj.get("hallucination_rate", hallucination.get("sample_rate")),
            )
        ),
        "parse_error_rate": _to_float_or_none(
            diagnostics.get("parse_error_rate", metrics_obj.get("parse_error_rate", parse_statistics.get("parse_error_rate")))
        ),
        "parse_success_rate": _to_float_or_none(
            diagnostics.get(
                "parse_success_rate",
                metrics_obj.get("parse_success_rate", parse_statistics.get("parse_success_rate")),
            )
        ),
        "avg_gold_events": _to_float_or_none(diagnostics.get("avg_gold_events")),
        "avg_predicted_events": _to_float_or_none(diagnostics.get("avg_predicted_events")),
        "avg_gold_event_types": _to_float_or_none(diagnostics.get("avg_gold_event_types")),
        "avg_schema_event_types": _to_float_or_none(diagnostics.get("avg_schema_event_types")),
        "stage1_gold_coverage_rate": _to_float_or_none(diagnostics.get("stage1_gold_coverage_rate")),
        "stage1_exact_match_rate": _to_float_or_none(diagnostics.get("stage1_exact_match_rate")),
        "stage1_miss_rate": _to_float_or_none(diagnostics.get("stage1_miss_rate")),
        "stage1_overprediction_rate": _to_float_or_none(diagnostics.get("stage1_overprediction_rate")),
        "avg_stage1_predicted_types": _to_float_or_none(diagnostics.get("avg_stage1_predicted_types")),
        "correction_applied_rate": _to_float_or_none(diagnostics.get("correction_applied_rate")),
        "records_split_count": _to_float_or_none(diagnostics.get("records_split_count")),
        "roles_rewritten_count": _to_float_or_none(diagnostics.get("roles_rewritten_count")),
        "roles_added_count": _to_float_or_none(diagnostics.get("roles_added_count")),
        "events_dropped_after_correction": _to_float_or_none(
            diagnostics.get("events_dropped_after_correction")
        ),
        "cot_faithfulness": _to_float_or_none(cot_faithfulness),
        "avg_tokens_per_sample": _to_float_or_none(
            cost.get("avg_tokens_per_sample", token_usage.get("avg_tokens_per_sample"))
        ),
        "total_tokens": _to_float_or_none(cost.get("total_tokens", token_usage.get("total_tokens"))),
        "wall_clock_seconds": _to_float_or_none(runtime.get("wall_clock_seconds")),
        "samples_per_second": _to_float_or_none(runtime.get("samples_per_second")),
    }

    requested_required = tuple(required_metrics or ())
    requested_optional = tuple(optional_metrics or ())
    if not requested_required and not requested_optional:
        requested_required = LOCAL_SUITE_REPORT_METRICS

    flat: Dict[str, float] = {}
    missing: List[str] = []

    for metric_name in requested_required:
        value = candidates.get(metric_name)
        if value is None:
            missing.append(metric_name)
            continue
        flat[metric_name] = float(value)

    if missing:
        raise ValueError(f"Missing required metrics: {', '.join(sorted(missing))}")

    for metric_name in requested_optional:
        value = candidates.get(metric_name)
        if value is not None:
            flat[metric_name] = float(value)

    return flat


def append_efficiency_metrics(
    metric_row: Mapping[str, float],
    *,
    primary_metric: str = "doc_role_micro_f1",
) -> Dict[str, float]:
    row = {str(key): float(value) for key, value in metric_row.items()}
    primary_value = row.get(primary_metric)
    total_tokens = row.get("total_tokens")
    wall_clock_seconds = row.get("wall_clock_seconds")
    if primary_value is not None and total_tokens is not None and total_tokens > 0:
        row["f1_per_1k_tokens"] = (primary_value * 1000.0) / total_tokens
    if primary_value is not None and wall_clock_seconds is not None and wall_clock_seconds > 0:
        row["f1_per_minute"] = (primary_value * 60.0) / wall_clock_seconds
    return row


def build_significance_metadata(
    pair_counts: Sequence[int],
    *,
    min_pairs: int = MIN_SIGNIFICANCE_PAIRS,
) -> Dict[str, Any]:
    normalized_counts = [int(count) for count in pair_counts]
    metadata: Dict[str, Any] = {
        "significance_min_pairs": int(min_pairs),
    }
    if not normalized_counts:
        metadata["significance_status"] = "skipped_no_complete_pairs"
        metadata["significance_skipped_reason"] = "no complete paired comparisons were available"
        return metadata

    observed_min = min(normalized_counts)
    metadata["significance_observed_pairs"] = observed_min
    if observed_min < int(min_pairs):
        metadata["significance_status"] = "skipped_insufficient_pairs"
        metadata["significance_skipped_reason"] = (
            f"paired permutation requires at least {int(min_pairs)} paired observations per comparison; "
            f"observed {observed_min}"
        )
        return metadata

    metadata["significance_status"] = "computed"
    return metadata


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def metric_from_counts(tp: int, pred_total: int, gold_total: int) -> Dict[str, float]:
    precision = safe_div(tp, pred_total)
    recall = safe_div(tp, gold_total)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_sample_counts(sample_counts: Sequence[CountDict]) -> CountDict:
    keys = [
        "strict_tp", "strict_pred_total", "strict_gold_total",
        "relaxed_tp", "relaxed_pred_total", "relaxed_gold_total",
        "type_tp", "type_pred_total", "type_gold_total",
        "doc_role_tp", "doc_role_pred_total", "doc_role_gold_total",
        "doc_event_type_tp", "doc_event_type_pred_total", "doc_event_type_gold_total",
        "doc_instance_tp", "doc_instance_pred_total", "doc_instance_gold_total",
        "doc_combination_tp", "doc_combination_pred_total", "doc_combination_gold_total",
    ]
    aggregated = {k: 0 for k in keys}
    for row in sample_counts:
        for k in keys:
            aggregated[k] += int(row.get(k, 0))
    return aggregated


def metrics_from_sample_counts(sample_counts: Sequence[CountDict]) -> Dict[str, float]:
    agg = aggregate_sample_counts(sample_counts)
    strict = metric_from_counts(agg["strict_tp"], agg["strict_pred_total"], agg["strict_gold_total"])
    relaxed = metric_from_counts(agg["relaxed_tp"], agg["relaxed_pred_total"], agg["relaxed_gold_total"])
    type_m = metric_from_counts(agg["type_tp"], agg["type_pred_total"], agg["type_gold_total"])
    doc_role = metric_from_counts(agg["doc_role_tp"], agg["doc_role_pred_total"], agg["doc_role_gold_total"])
    doc_event_type = metric_from_counts(
        agg["doc_event_type_tp"],
        agg["doc_event_type_pred_total"],
        agg["doc_event_type_gold_total"],
    )
    doc_instance = metric_from_counts(
        agg["doc_instance_tp"],
        agg["doc_instance_pred_total"],
        agg["doc_instance_gold_total"],
    )
    doc_combination = metric_from_counts(
        agg["doc_combination_tp"],
        agg["doc_combination_pred_total"],
        agg["doc_combination_gold_total"],
    )
    return {
        "strict_precision": strict["precision"],
        "strict_recall": strict["recall"],
        "strict_f1": strict["f1"],
        "relaxed_precision": relaxed["precision"],
        "relaxed_recall": relaxed["recall"],
        "relaxed_f1": relaxed["f1"],
        "type_precision": type_m["precision"],
        "type_recall": type_m["recall"],
        "type_f1": type_m["f1"],
        "doc_role_micro_precision": doc_role["precision"],
        "doc_role_micro_recall": doc_role["recall"],
        "doc_role_micro_f1": doc_role["f1"],
        "doc_event_type_micro_precision": doc_event_type["precision"],
        "doc_event_type_micro_recall": doc_event_type["recall"],
        "doc_event_type_micro_f1": doc_event_type["f1"],
        "doc_instance_micro_precision": doc_instance["precision"],
        "doc_instance_micro_recall": doc_instance["recall"],
        "doc_instance_micro_f1": doc_instance["f1"],
        "doc_combination_micro_precision": doc_combination["precision"],
        "doc_combination_micro_recall": doc_combination["recall"],
        "doc_combination_micro_f1": doc_combination["f1"],
    }


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 1:
        return sorted_values[-1]
    idx = int(round((len(sorted_values) - 1) * p))
    return sorted_values[idx]


def bootstrap_confidence_intervals(
    sample_counts: Sequence[CountDict],
    n_bootstrap: int = 1000,
    seed: int = 3407,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float | List[float]]]:
    """
    Bootstrap confidence intervals for strict/relaxed/type precision/recall/F1.
    """
    if not sample_counts:
        return {}

    rng = random.Random(seed)
    n = len(sample_counts)
    metric_keys = [
        "strict_precision", "strict_recall", "strict_f1",
        "relaxed_precision", "relaxed_recall", "relaxed_f1",
        "type_precision", "type_recall", "type_f1",
        "doc_role_micro_precision", "doc_role_micro_recall", "doc_role_micro_f1",
        "doc_event_type_micro_precision", "doc_event_type_micro_recall", "doc_event_type_micro_f1",
        "doc_instance_micro_precision", "doc_instance_micro_recall", "doc_instance_micro_f1",
        "doc_combination_micro_precision", "doc_combination_micro_recall", "doc_combination_micro_f1",
    ]
    trajectories = {k: [] for k in metric_keys}

    for _ in range(max(1, int(n_bootstrap))):
        sampled = [sample_counts[rng.randrange(n)] for _ in range(n)]
        m = metrics_from_sample_counts(sampled)
        for k in metric_keys:
            trajectories[k].append(m[k])

    alpha = (1.0 - confidence) / 2.0
    out: Dict[str, Dict[str, float | List[float]]] = {}
    for k in metric_keys:
        values = trajectories[k]
        values_sorted = sorted(values)
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
        std = math.sqrt(var)
        out[k] = {
            "mean": mean,
            "std": std,
            "ci": [_percentile(values_sorted, alpha), _percentile(values_sorted, 1 - alpha)],
            "confidence": confidence,
            "n_bootstrap": int(n_bootstrap),
        }
    return out


def aggregate_runs(
    run_metrics: Sequence[Dict[str, float]],
    metric_keys: Iterable[str] | None = None,
) -> Dict[str, Dict[str, float | List[float] | int]]:
    if not run_metrics:
        return {}
    if metric_keys is None:
        metric_keys = sorted(run_metrics[0].keys())

    result: Dict[str, Dict[str, float | List[float] | int]] = {}
    n = len(run_metrics)
    for k in metric_keys:
        values = [float(r[k]) for r in run_metrics if k in r]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
        std = math.sqrt(var)
        vals_sorted = sorted(values)
        result[k] = {
            "mean": mean,
            "std": std,
            "min": vals_sorted[0],
            "max": vals_sorted[-1],
            "median": _percentile(vals_sorted, 0.5),
            "ci95": [_percentile(vals_sorted, 0.025), _percentile(vals_sorted, 0.975)],
            "n_runs": n,
        }
    return result


def paired_permutation_pvalue(
    baseline_scores: Sequence[float],
    improved_scores: Sequence[float],
    n_monte_carlo: int = 10000,
    seed: int = 3407,
) -> Dict[str, float | int]:
    """
    Two-sided paired permutation test over per-seed scores.
    """
    if len(baseline_scores) != len(improved_scores):
        raise ValueError("baseline_scores and improved_scores must have equal length.")
    if len(baseline_scores) == 0:
        raise ValueError("Scores must be non-empty.")
    if len(baseline_scores) < MIN_SIGNIFICANCE_PAIRS:
        raise ValueError(f"Scores must contain at least {MIN_SIGNIFICANCE_PAIRS} paired observations.")

    diffs = [float(b) - float(a) for a, b in zip(baseline_scores, improved_scores)]
    observed = abs(sum(diffs) / len(diffs))
    n = len(diffs)

    if n <= 20:
        # Exact test
        all_signs = itertools.product([-1.0, 1.0], repeat=n)
        total = 0
        extreme = 0
        for signs in all_signs:
            total += 1
            stat = abs(sum(d * s for d, s in zip(diffs, signs)) / n)
            if stat >= observed - 1e-12:
                extreme += 1
        p_value = extreme / total
        return {
            "p_value": p_value,
            "observed_mean_diff": sum(diffs) / n,
            "n_pairs": n,
            "method": "exact_permutation",
            "n_permutations": total,
        }

    # Monte-Carlo approximation
    rng = random.Random(seed)
    extreme = 0
    for _ in range(max(1000, int(n_monte_carlo))):
        stat = abs(sum(d * (1.0 if rng.random() > 0.5 else -1.0) for d in diffs) / n)
        if stat >= observed - 1e-12:
            extreme += 1
    p_value = extreme / max(1, int(n_monte_carlo))
    return {
        "p_value": p_value,
        "observed_mean_diff": sum(diffs) / n,
        "n_pairs": n,
        "method": "monte_carlo_permutation",
        "n_permutations": int(n_monte_carlo),
    }
