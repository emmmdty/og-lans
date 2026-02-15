import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "oglans" / "utils" / "academic_eval.py"
spec = importlib.util.spec_from_file_location("academic_eval", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load academic_eval from {MODULE_PATH}")
academic_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(academic_eval)

aggregate_sample_counts = academic_eval.aggregate_sample_counts
bootstrap_confidence_intervals = academic_eval.bootstrap_confidence_intervals
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


def test_paired_permutation_pvalue_exact_output():
    baseline = [0.4, 0.5, 0.6]
    improved = [0.5, 0.55, 0.65]
    stat = paired_permutation_pvalue(baseline, improved)
    assert 0.0 <= stat["p_value"] <= 1.0
    assert stat["n_pairs"] == 3
    assert stat["method"] == "exact_permutation"
