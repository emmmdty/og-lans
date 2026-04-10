import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "oglans" / "utils" / "academic_eval.py"
spec = importlib.util.spec_from_file_location("academic_eval", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load academic_eval from {MODULE_PATH}")
academic_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(academic_eval)

aggregate_sample_counts = academic_eval.aggregate_sample_counts
append_efficiency_metrics = academic_eval.append_efficiency_metrics
bootstrap_confidence_intervals = academic_eval.bootstrap_confidence_intervals
extract_report_metrics = academic_eval.extract_report_metrics
metrics_from_sample_counts = academic_eval.metrics_from_sample_counts
paired_permutation_pvalue = academic_eval.paired_permutation_pvalue


def test_metrics_from_sample_counts_basic():
    rows = [
        {
            "strict_tp": 2,
            "strict_pred_total": 4,
            "strict_gold_total": 5,
            "relaxed_tp": 3,
            "relaxed_pred_total": 4,
            "relaxed_gold_total": 5,
            "type_tp": 1,
            "type_pred_total": 2,
            "type_gold_total": 2,
        },
        {
            "strict_tp": 1,
            "strict_pred_total": 2,
            "strict_gold_total": 2,
            "relaxed_tp": 1,
            "relaxed_pred_total": 2,
            "relaxed_gold_total": 2,
            "type_tp": 1,
            "type_pred_total": 1,
            "type_gold_total": 2,
        },
    ]
    agg = aggregate_sample_counts(rows)
    assert agg["strict_tp"] == 3
    assert agg["strict_pred_total"] == 6
    assert agg["strict_gold_total"] == 7

    metrics = metrics_from_sample_counts(rows)
    assert 0.0 <= metrics["strict_f1"] <= 1.0
    assert 0.0 <= metrics["relaxed_f1"] <= 1.0
    assert 0.0 <= metrics["type_f1"] <= 1.0


def test_metrics_from_sample_counts_supports_doc_level_primary_metrics():
    rows = [
        {
            "doc_role_tp": 2,
            "doc_role_pred_total": 2,
            "doc_role_gold_total": 4,
            "doc_event_type_tp": 1,
            "doc_event_type_pred_total": 1,
            "doc_event_type_gold_total": 1,
            "doc_instance_tp": 1,
            "doc_instance_pred_total": 1,
            "doc_instance_gold_total": 2,
            "doc_combination_tp": 1,
            "doc_combination_pred_total": 1,
            "doc_combination_gold_total": 2,
        }
    ]

    metrics = metrics_from_sample_counts(rows)
    assert metrics["doc_role_micro_f1"] == 2.0 / 3.0
    assert metrics["doc_event_type_micro_f1"] == 1.0
    assert metrics["doc_instance_micro_f1"] == 2.0 / 3.0
    assert metrics["doc_combination_micro_f1"] == 2.0 / 3.0


def test_bootstrap_confidence_intervals_shape():
    rows = [
        {
            "strict_tp": 1,
            "strict_pred_total": 2,
            "strict_gold_total": 2,
            "relaxed_tp": 1,
            "relaxed_pred_total": 2,
            "relaxed_gold_total": 2,
            "type_tp": 1,
            "type_pred_total": 1,
            "type_gold_total": 2,
        }
        for _ in range(5)
    ]
    out = bootstrap_confidence_intervals(rows, n_bootstrap=100, seed=1, confidence=0.95)
    assert "strict_f1" in out
    assert "ci" in out["strict_f1"]
    assert len(out["strict_f1"]["ci"]) == 2
    assert out["strict_f1"]["n_bootstrap"] == 100


def test_bootstrap_confidence_intervals_include_doc_level_primary_metrics():
    rows = [
        {
            "doc_role_tp": 2,
            "doc_role_pred_total": 2,
            "doc_role_gold_total": 4,
            "doc_event_type_tp": 1,
            "doc_event_type_pred_total": 1,
            "doc_event_type_gold_total": 1,
            "doc_instance_tp": 1,
            "doc_instance_pred_total": 1,
            "doc_instance_gold_total": 2,
            "doc_combination_tp": 1,
            "doc_combination_pred_total": 1,
            "doc_combination_gold_total": 2,
        }
        for _ in range(3)
    ]

    out = bootstrap_confidence_intervals(rows, n_bootstrap=50, seed=1, confidence=0.95)
    assert "doc_role_micro_f1" in out
    assert "doc_instance_micro_f1" in out
    assert "doc_combination_micro_f1" in out


def test_paired_permutation_pvalue_exact_output():
    baseline = [0.4, 0.5, 0.6]
    improved = [0.5, 0.55, 0.65]
    stat = paired_permutation_pvalue(baseline, improved)
    assert 0.0 <= stat["p_value"] <= 1.0
    assert stat["n_pairs"] == 3
    assert stat["method"] == "exact_permutation"


def test_extract_report_metrics_reads_nested_summary_fields():
    payload = {
        "metrics": {
            "academic_metrics": {
                "doc_ee": {
                    "overall": {"MicroF1": 0.61},
                    "instance": {"MicroF1": 0.55},
                    "combination": {"MicroF1": 0.57},
                    "classification": {"MicroF1": 0.72},
                }
            },
            "strict": {"precision": 0.51, "recall": 0.49, "f1": 0.5},
            "relaxed": {"f1": 0.63},
            "type_identification": {"f1": 0.74},
            "schema_compliance_rate": 0.81,
            "hallucination": {"sample_rate": 0.12},
            "parse_statistics": {"parse_error_rate": 0.05},
            "cot_faithfulness": {"overall": 0.9},
        }
    }

    metrics = extract_report_metrics(
        payload,
        required_metrics=(
            "doc_role_micro_f1",
            "doc_instance_micro_f1",
            "doc_combination_micro_f1",
            "doc_event_type_micro_f1",
            "strict_precision",
            "strict_recall",
            "strict_f1",
            "relaxed_f1",
            "type_f1",
            "schema_compliance_rate",
            "hallucination_rate",
        ),
        optional_metrics=("parse_error_rate", "cot_faithfulness"),
    )

    assert metrics["doc_role_micro_f1"] == 0.61
    assert metrics["relaxed_f1"] == 0.63
    assert metrics["type_f1"] == 0.74
    assert metrics["schema_compliance_rate"] == 0.81
    assert metrics["hallucination_rate"] == 0.12
    assert metrics["parse_error_rate"] == 0.05
    assert metrics["cot_faithfulness"] == 0.9


def test_extract_report_metrics_rejects_missing_required_metric():
    with pytest.raises(ValueError, match="relaxed_f1"):
        extract_report_metrics(
            {"metrics": {"strict": {"f1": 0.5}}},
            required_metrics=("strict_f1", "relaxed_f1"),
        )


def test_extract_report_metrics_reads_cost_fields_when_requested():
    payload = {
        "metrics": {
            "doc_role_micro_f1": 0.61,
        },
        "token_usage": {
            "avg_tokens_per_sample": 1234.0,
            "total_tokens": 5678,
        },
        "runtime": {
            "wall_clock_seconds": 12.5,
            "samples_per_second": 8.0,
        },
    }

    metrics = extract_report_metrics(
        payload,
        required_metrics=("doc_role_micro_f1",),
        optional_metrics=(
            "avg_tokens_per_sample",
            "total_tokens",
            "wall_clock_seconds",
            "samples_per_second",
        ),
    )

    assert metrics["doc_role_micro_f1"] == 0.61
    assert metrics["avg_tokens_per_sample"] == 1234.0
    assert metrics["total_tokens"] == 5678.0
    assert metrics["wall_clock_seconds"] == 12.5
    assert metrics["samples_per_second"] == 8.0


def test_paired_permutation_pvalue_rejects_single_pair():
    with pytest.raises(ValueError, match="at least 2"):
        paired_permutation_pvalue([0.4], [0.5])


def test_append_efficiency_metrics_computes_cost_normalized_scores():
    row = append_efficiency_metrics(
        {
            "doc_role_micro_f1": 0.5,
            "total_tokens": 2000.0,
            "wall_clock_seconds": 30.0,
        }
    )

    assert row["f1_per_1k_tokens"] == 0.25
    assert row["f1_per_minute"] == 1.0


def test_extract_report_metrics_prefers_new_diagnostics_and_cost_blocks():
    payload = {
        "metrics": {
            "doc_role_micro_f1": 0.31,
            "doc_instance_micro_f1": 0.21,
            "doc_combination_micro_f1": 0.22,
            "doc_event_type_micro_f1": 0.71,
            "strict_f1": 0.41,
            "relaxed_f1": 0.51,
            "type_f1": 0.61,
            "schema_compliance_rate": 0.81,
            "hallucination_rate": 0.09,
        },
        "diagnostics": {
            "parse_success_rate": 0.98,
            "parse_error_rate": 0.02,
            "avg_gold_events": 1.3,
            "avg_predicted_events": 1.1,
            "correction_applied_rate": 0.4,
            "records_split_count": 3,
            "roles_rewritten_count": 1,
            "roles_added_count": 2,
            "events_dropped_after_correction": 1,
        },
        "cost": {
            "total_tokens": 1234,
            "avg_tokens_per_sample": 12.34,
        },
        "runtime": {
            "wall_clock_seconds": 45.6,
            "samples_per_second": 2.5,
        },
    }

    metrics = extract_report_metrics(
        payload,
        required_metrics=(
            "doc_role_micro_f1",
            "parse_success_rate",
            "total_tokens",
            "wall_clock_seconds",
            "correction_applied_rate",
            "records_split_count",
            "roles_rewritten_count",
            "roles_added_count",
            "events_dropped_after_correction",
        ),
    )

    assert metrics["parse_success_rate"] == 0.98
    assert metrics["total_tokens"] == 1234.0
    assert metrics["wall_clock_seconds"] == 45.6
    assert metrics["correction_applied_rate"] == 0.4
    assert metrics["records_split_count"] == 3.0
    assert metrics["roles_rewritten_count"] == 1.0
    assert metrics["roles_added_count"] == 2.0
    assert metrics["events_dropped_after_correction"] == 1.0
