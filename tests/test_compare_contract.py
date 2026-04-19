import pytest

from oglans.utils.compare_contract import (
    COMPARABLE_CONTRACT_FIELDS,
    build_compare_contract,
    extract_compare_contract,
    build_result_diagnostics,
    validate_compare_contract_match,
)
from oglans.utils.experiment_contract import build_experiment_contract


def _compare_payload(**overrides):
    payload = {
        "model_family": "local_base",
        "model_kind": "base_only",
        "split": "dev",
        "primary_metric": "doc_role_micro_f1",
        "stage_mode": "single_pass",
        "prompt_variant": "zeroshot",
        "fewshot_num_examples": 0,
        "fewshot_selection_mode": "none",
        "fewshot_pool_split": "none",
        "train_tune_ratio": 0.1,
        "research_split_manifest_path": "/tmp/frozen.json",
        "research_split_manifest_hash": "a" * 64,
        "pipeline_mode": "e2e",
        "postprocess_profile": "none",
        "canonical_metric_mode": "analysis_only",
        "prompt_builder_version": "phase3_mvp_v1",
        "parser_version": "phase3_mvp_v1",
        "normalization_version": "phase3_mvp_v1",
        "protocol_hash": "b" * 64,
        "role_alias_hash": "c" * 64,
        "seed": 3407,
        "seed_effective": False,
        "token_usage_kind": "estimated",
    }
    payload.update(overrides)
    return payload


def test_build_compare_contract_requires_expected_fields():
    payload = _compare_payload()
    compare = build_compare_contract(payload)

    for field_name in COMPARABLE_CONTRACT_FIELDS:
        assert field_name in compare
    assert compare["comparable_contract_hash"]


def test_validate_compare_contract_match_rejects_prompt_mismatch():
    left = build_compare_contract(_compare_payload(prompt_variant="zeroshot"))
    right = build_compare_contract(_compare_payload(prompt_variant="fewshot", fewshot_num_examples=3))

    with pytest.raises(ValueError, match="comparable contract mismatch"):
        validate_compare_contract_match([left, right])


def test_build_compare_contract_normalizes_legacy_two_stage_typed_alias():
    compare = build_compare_contract(_compare_payload(stage_mode="two_stage_typed"))

    assert compare["stage_mode"] == "two_stage_per_type"


def test_extract_compare_contract_can_derive_from_experiment_contract_block():
    experiment_contract = build_experiment_contract(
        _compare_payload(prompt_variant="fewshot", fewshot_num_examples=3)
    )

    compare = extract_compare_contract({"experiment_contract": experiment_contract})

    assert compare["prompt_variant"] == "fewshot"
    assert compare["comparable_contract_hash"] == experiment_contract["experiment_contract_hash"]


def test_build_result_diagnostics_reports_two_stage_metrics():
    rows = [
        {
            "parse_success": True,
            "ground_truth": [{"event_type": "公司上市", "arguments": []}],
            "prediction": [{"event_type": "公司上市", "arguments": []}],
            "prompt_meta": {"fewshot_example_ids": ["a", "b", "c"]},
            "stage_meta": {
                "stage_mode": "two_stage",
                "stage1_predicted_event_types": ["公司上市"],
                "stage2_schema_event_types": ["公司上市"],
            },
            "correction_stats": {
                "applied": True,
                "records_split_count": 1,
                "roles_rewritten_count": 0,
                "roles_added_count": 2,
                "events_dropped_after_correction": 0,
                "correction_trigger_breakdown": {"split_record:公司上市": 1},
            },
        },
        {
            "parse_success": False,
            "ground_truth": [{"event_type": "企业融资", "arguments": []}],
            "prediction": [],
            "prompt_meta": {"fewshot_example_ids": ["a", "d", "e"]},
            "stage_meta": {
                "stage_mode": "two_stage",
                "stage1_predicted_event_types": [],
                "stage2_schema_event_types": [],
            },
            "correction_stats": {
                "applied": False,
                "records_split_count": 0,
                "roles_rewritten_count": 1,
                "roles_added_count": 0,
                "events_dropped_after_correction": 1,
                "correction_trigger_breakdown": {"drop:no_arguments_after_correction:企业融资": 1},
            },
        },
    ]

    diagnostics = build_result_diagnostics(rows)

    assert diagnostics["parse_success_rate"] == pytest.approx(0.5)
    assert diagnostics["stage1_gold_coverage_rate"] == pytest.approx(0.5)
    assert diagnostics["stage1_miss_rate"] == pytest.approx(0.5)
    assert diagnostics["fewshot_unique_example_ids"] == 5
    assert diagnostics["fewshot_unique_combinations"] == 2
    assert diagnostics["correction_applied_rate"] == pytest.approx(0.5)
    assert diagnostics["records_split_count"] == 1
    assert diagnostics["roles_rewritten_count"] == 1
    assert diagnostics["roles_added_count"] == 2
    assert diagnostics["events_dropped_after_correction"] == 1


def test_build_result_diagnostics_treats_two_stage_per_type_as_stage1_aware():
    rows = [
        {
            "parse_success": True,
            "ground_truth": [{"event_type": "中标", "arguments": []}],
            "prediction": [{"event_type": "中标", "arguments": []}],
            "prompt_meta": {"fewshot_example_ids": ["a"]},
            "stage_meta": {
                "stage_mode": "two_stage_per_type",
                "stage1_predicted_event_types": ["中标"],
                "stage2_schema_event_types": ["中标"],
            },
        }
    ]

    diagnostics = build_result_diagnostics(rows)

    assert diagnostics["stage1_gold_coverage_rate"] == pytest.approx(1.0)
    assert diagnostics["stage1_exact_match_rate"] == pytest.approx(1.0)
