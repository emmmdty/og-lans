import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_baseline_matrix.py"
spec = importlib.util.spec_from_file_location("audit_baseline_matrix", str(SCRIPT_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row(
    *,
    event_types,
    stage_mode="single_pass",
    predicted_types=None,
    example_ids=None,
):
    return {
        "id": "sample-1",
        "parse_success": True,
        "ground_truth": [{"event_type": event_type, "arguments": []} for event_type in event_types],
        "prediction": [],
        "prediction_canonical": [],
        "prompt_meta": {
            "fewshot_selection_mode": "dynamic" if example_ids else "none",
            "fewshot_example_ids": list(example_ids or []),
            "fewshot_count": len(example_ids or []),
        },
        "stage_meta": {
            "stage_mode": stage_mode,
            "stage1_predicted_event_types": list(predicted_types or []),
            "stage1_parse_success": True if stage_mode == "two_stage" else None,
            "stage1_parse_error": None,
            "stage2_schema_event_types": list(predicted_types or event_types),
        },
    }


def test_summarize_result_rows_reports_stage1_coverage_and_fewshot_usage():
    rows = [
        _row(event_types=["公司上市"], stage_mode="two_stage", predicted_types=["公司上市"], example_ids=["a", "b", "c"]),
        _row(event_types=["企业融资"], stage_mode="two_stage", predicted_types=[], example_ids=["a", "b", "c"]),
        _row(event_types=["质押"], stage_mode="two_stage", predicted_types=["企业融资"], example_ids=["a", "b", "c"]),
    ]

    summary = mod.summarize_result_rows(rows)

    assert summary["stage1_rows"] == 3
    assert summary["stage1_gold_coverage_rate"] == 1 / 3
    assert summary["stage1_empty_rate"] == 1 / 3
    assert summary["fewshot_unique_example_ids"] == 3
    assert summary["fewshot_unique_combinations"] == 1


def test_audit_suite_builds_pairwise_deltas(tmp_path: Path):
    single_run = tmp_path / "single_run"
    two_stage_run = tmp_path / "two_stage_run"
    single_run.mkdir()
    two_stage_run.mkdir()

    _write_jsonl(
        single_run / "eval_results.jsonl",
        [_row(event_types=["公司上市"])],
    )
    _write_jsonl(
        two_stage_run / "eval_results.jsonl",
        [_row(event_types=["公司上市"], stage_mode="two_stage", predicted_types=["公司上市"])],
    )

    suite_summary = {
        "suite": "demo",
        "model": "Qwen",
        "gpu": 1,
        "variants": {
            "single_pass_zeroshot": {
                "run_dir": str(single_run),
                "summary_file": str(single_run / "eval_results_summary.json"),
                "prompt_variant": "zeroshot",
                "stage_mode": "single_pass",
                "use_fewshot": False,
                "batch_size": 16,
                "metrics": {
                    "doc_role_micro_f1": 0.3,
                    "doc_instance_micro_f1": 0.1,
                    "doc_combination_micro_f1": 0.1,
                    "doc_event_type_micro_f1": 0.7,
                    "strict_f1": 0.4,
                    "relaxed_f1": 0.5,
                    "type_f1": 0.7,
                    "schema_compliance_rate": 0.8,
                    "hallucination_rate": 0.2,
                    "avg_tokens_per_sample": 1000.0,
                    "total_tokens": 1000.0,
                    "wall_clock_seconds": 10.0,
                    "samples_per_second": 1.0,
                    "f1_per_1k_tokens": 0.3,
                    "f1_per_minute": 1.8,
                },
            },
            "two_stage_zeroshot": {
                "run_dir": str(two_stage_run),
                "summary_file": str(two_stage_run / "eval_results_summary.json"),
                "prompt_variant": "zeroshot",
                "stage_mode": "two_stage",
                "use_fewshot": False,
                "batch_size": 16,
                "metrics": {
                    "doc_role_micro_f1": 0.2,
                    "doc_instance_micro_f1": 0.05,
                    "doc_combination_micro_f1": 0.05,
                    "doc_event_type_micro_f1": 0.6,
                    "strict_f1": 0.3,
                    "relaxed_f1": 0.4,
                    "type_f1": 0.6,
                    "schema_compliance_rate": 0.7,
                    "hallucination_rate": 0.25,
                    "avg_tokens_per_sample": 1100.0,
                    "total_tokens": 1100.0,
                    "wall_clock_seconds": 12.0,
                    "samples_per_second": 1.0,
                    "f1_per_1k_tokens": 0.18,
                    "f1_per_minute": 1.0,
                },
            },
        },
    }
    suite_path = tmp_path / "suite_summary.json"
    suite_path.write_text(json.dumps(suite_summary), encoding="utf-8")

    audit = mod.audit_suite(suite_path)

    assert audit["reference_variant"] == "single_pass_zeroshot"
    assert audit["pairwise_deltas"]["two_stage_zeroshot"]["doc_role_micro_f1"] == pytest.approx(-0.1)
    assert "best variant by doc_role_micro_f1 is single_pass_zeroshot." in audit["findings"]
