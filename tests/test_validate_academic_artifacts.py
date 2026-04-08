import importlib.util
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
            "strict_f1": 0.5,
            "relaxed_f1": 0.6,
            "type_f1": 0.7,
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
