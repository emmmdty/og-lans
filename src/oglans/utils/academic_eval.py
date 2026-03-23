"""
Academic evaluation utilities for reproducible reporting.

This module is intentionally dependency-light so that statistical logic
can be unit-tested without requiring model/runtime dependencies.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple


CountDict = Dict[str, int]


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
