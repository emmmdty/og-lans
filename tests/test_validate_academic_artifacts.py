import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_academic_artifacts.py"
spec = importlib.util.spec_from_file_location("validate_academic_artifacts", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module from {MODULE_PATH}")
validate_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_mod)


def build_valid_summary():
    return {
        "meta": {
            "timestamp": "2026-03-18T00:00:00",
            "model": "test-model",
            "api_response_models": [],
            "seed": 3407,
            "command": "python evaluate_api.py",
            "config_hash_sha256": "a" * 64,
            "protocol_path": "/tmp/protocol.yaml",
            "protocol_hash_sha256": "b" * 64,
            "eval_protocol_path": "/tmp/protocol.yaml",
            "eval_protocol_hash": "c" * 64,
            "role_alias_path": "/tmp/aliases.yaml",
            "role_alias_hash": "d" * 64,
            "prompt_variant": "zeroshot",
            "prompt_builder_version": "route_a_compare_v1",
            "parser_version": "route_a_compare_v1",
            "normalization_version": "route_a_compare_v1",
            "training_mode": "preference",
            "primary_metric": "doc_role_micro_f1",
            "canonical_metric_mode": "analysis_only",
            "generation": {
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            "prompt_hashes": {},
            "has_gold_labels": True,
        },
        "metrics": {
            "doc_role_micro_f1": 0.61,
            "doc_instance_micro_f1": 0.55,
            "doc_combination_micro_f1": 0.57,
            "doc_event_type_micro_f1": 0.72,
            "strict_precision": 0.48,
            "strict_recall": 0.52,
            "strict_f1": 0.5,
            "relaxed_f1": 0.6,
            "type_f1": 0.7,
            "schema_compliance_rate": 0.82,
            "hallucination_rate": 0.14,
            "parse_error_rate": 0.0,
            "parse_success_rate": 1.0,
            "primary_metric": "doc_role_micro_f1",
            "primary_metric_value": 0.61,
            "bootstrap_ci": {"doc_role_micro_f1": [0.58, 0.64]},
            "academic_metrics": {
                "doc_ee": {
                    "overall": {"MicroF1": 0.61},
                    "instance": {"MicroF1": 0.55},
                    "combination": {"MicroF1": 0.57},
                    "classification": {"MicroF1": 0.72},
                },
                "ee_text_proxy": {},
            },
        },
        "token_usage": {
            "total_tokens": 10,
            "avg_tokens_per_sample": 1.0,
        },
        "api_stats": {
            "failed_calls": 0,
        },
        "runtime": {
            "wall_clock_seconds": 1.23,
        },
        "runtime_manifest": {
            "python": {"version": "3.10.0"},
            "system": {"platform": "Windows"},
        },
        "analysis": {
            "primary_metric": "doc_role_micro_f1",
            "protocol": {"version": "1.0"},
        },
    }


def test_validate_summary_accepts_compare_metadata():
    errors = validate_mod.validate_summary(build_valid_summary())
    assert errors == []


def test_validate_summary_rejects_missing_compare_metadata():
    summary = build_valid_summary()
    del summary["meta"]["prompt_variant"]

    errors = validate_mod.validate_summary(summary)

    assert "Missing required field: meta.prompt_variant" in errors


def build_valid_suite_summary(tmp_path: Path):
    child_summary = build_valid_summary()
    child_summary_path = tmp_path / "eval_results_summary.json"
    child_summary_path.write_text(json.dumps(child_summary), encoding="utf-8")
    metrics = child_summary["metrics"]
    metric_names = [
        "doc_role_micro_f1",
        "doc_instance_micro_f1",
        "doc_combination_micro_f1",
        "doc_event_type_micro_f1",
        "strict_f1",
        "relaxed_f1",
        "type_f1",
        "strict_precision",
        "strict_recall",
        "schema_compliance_rate",
        "hallucination_rate",
    ]
    return {
        "timestamp": "2026-04-09T00:00:00",
        "config": "configs/config.yaml",
        "dataset": "DuEE-Fin",
        "split": "dev",
        "seeds": [3407],
        "primary_metric": "doc_role_micro_f1",
        "shared_reference_meta": {},
        "runs": [
            {
                "run_key": "base",
                "seed": 3407,
                "summary_file": str(child_summary_path),
            }
        ],
        "aggregated": {
            "base": {
                "n_success_runs": 1,
                "metrics": {
                    metric: {
                        "mean": float(metrics[metric]),
                        "std": 0.0,
                        "ci95": [float(metrics[metric]), float(metrics[metric])],
                        "n_runs": 1,
                    }
                    for metric in metric_names
                },
            }
        },
        "significance": {},
        "significance_status": "skipped_insufficient_pairs",
        "significance_min_pairs": 2,
        "significance_skipped_reason": "paired permutation requires at least 2 paired observations per comparison; observed 1",
    }


def test_validate_summary_accepts_single_seed_suite_summary(tmp_path):
    errors = validate_mod.validate_summary(build_valid_suite_summary(tmp_path))

    assert errors == []


def test_validate_summary_rejects_suite_missing_skip_metadata(tmp_path):
    summary = build_valid_suite_summary(tmp_path)
    del summary["significance_status"]

    errors = validate_mod.validate_summary(summary)

    assert "Missing suite field: significance_status" in errors
