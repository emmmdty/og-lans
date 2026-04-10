import importlib.util
from pathlib import Path

import pytest
from oglans.utils.compare_contract import build_compare_contract


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_comparable_eval_suite.py"
spec = importlib.util.spec_from_file_location("run_comparable_eval_suite", str(SCRIPT_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_build_local_eval_command_for_checkpoint_and_base():
    base_cmd = mod.build_local_eval_command(
        evaluate_path=Path("evaluate.py"),
        config="configs/config.yaml",
        protocol="configs/eval_protocol.yaml",
        role_alias_map="configs/role_aliases_duee_fin.yaml",
        model_name_or_path="/models/base",
        checkpoint_path=None,
        split="dev",
        seed=3407,
        output_file=Path("base.jsonl"),
        summary_file=Path("base_summary.json"),
        batch_size=8,
        prompt_variant="zeroshot",
        fewshot_num_examples=0,
        stage_mode="single_pass",
        fewshot_selection_mode="dynamic",
        fewshot_pool_split="train_fit",
        train_tune_ratio=0.1,
        research_split_manifest="configs/research_splits/frozen.json",
        report_primary_metric="doc_role_micro_f1",
        canonical_metric_mode="analysis_only",
        base_only=True,
    )
    assert "--base_only" in base_cmd
    assert "--checkpoint" not in base_cmd

    ckpt_cmd = mod.build_local_eval_command(
        evaluate_path=Path("evaluate.py"),
        config="configs/config.yaml",
        protocol="configs/eval_protocol.yaml",
        role_alias_map="configs/role_aliases_duee_fin.yaml",
        model_name_or_path="/models/base",
        checkpoint_path="logs/full",
        split="dev",
        seed=3407,
        output_file=Path("full.jsonl"),
        summary_file=Path("full_summary.json"),
        batch_size=8,
        prompt_variant="fewshot",
        fewshot_num_examples=3,
        stage_mode="two_stage",
        fewshot_selection_mode="dynamic",
        fewshot_pool_split="train_fit",
        train_tune_ratio=0.1,
        research_split_manifest="configs/research_splits/frozen.json",
        report_primary_metric="doc_role_micro_f1",
        canonical_metric_mode="analysis_only",
        base_only=False,
    )
    assert "--checkpoint" in ckpt_cmd
    assert "logs/full" in ckpt_cmd
    assert "--summary_file" in ckpt_cmd


def test_build_api_eval_command_forwards_matrix_controls():
    cmd = mod.build_api_eval_command(
        evaluate_api_path=Path("evaluate_api.py"),
        config="configs/config.yaml",
        protocol="configs/eval_protocol.yaml",
        role_alias_map="configs/role_aliases_duee_fin.yaml",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        split="dev",
        seed=3407,
        output_file=Path("api.jsonl"),
        summary_file=Path("api_summary.json"),
        concurrency=32,
        prompt_variant="fewshot",
        fewshot_num_examples=3,
        stage_mode="two_stage",
        fewshot_selection_mode="dynamic",
        fewshot_pool_split="train_fit",
        train_tune_ratio=0.1,
        research_split_manifest="configs/research_splits/frozen.json",
        report_primary_metric="doc_role_micro_f1",
        canonical_metric_mode="analysis_only",
    )
    assert "--use_fewshot" in cmd
    assert "--base_url" in cmd
    assert "--stage_mode" in cmd
    assert "two_stage" in cmd


def test_validate_family_records_rejects_mismatched_contract_hash():
    records = [
        {
            "family": "api",
            "variant": "single_pass_zeroshot",
            "compare": build_compare_contract(
                {
                    "model_family": "api",
                    "model_kind": "api_model",
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
                    "canonical_metric_mode": "analysis_only",
                    "protocol_hash": "b" * 64,
                    "role_alias_hash": "c" * 64,
                    "seed": 3407,
                    "seed_effective": False,
                    "token_usage_kind": "actual",
                }
            ),
        },
        {
            "family": "checkpoint",
            "variant": "single_pass_zeroshot",
            "compare": build_compare_contract(
                {
                    "model_family": "local_checkpoint",
                    "model_kind": "adapter_checkpoint",
                    "split": "dev",
                    "primary_metric": "doc_role_micro_f1",
                    "stage_mode": "single_pass",
                    "prompt_variant": "fewshot",
                    "fewshot_num_examples": 3,
                    "fewshot_selection_mode": "dynamic",
                    "fewshot_pool_split": "train_fit",
                    "train_tune_ratio": 0.1,
                    "research_split_manifest_path": "/tmp/frozen.json",
                    "research_split_manifest_hash": "a" * 64,
                    "pipeline_mode": "e2e",
                    "canonical_metric_mode": "analysis_only",
                    "protocol_hash": "b" * 64,
                    "role_alias_hash": "c" * 64,
                    "seed": 3407,
                    "seed_effective": False,
                    "token_usage_kind": "estimated",
                }
            ),
        },
    ]

    with pytest.raises(ValueError, match="comparable contract mismatch"):
        mod.validate_family_records(records)


def test_build_table_sections_splits_main_diagnostic_and_cost_rows():
    sections = mod.build_table_sections(
        [
            {
                "family": "api",
                "variant": "single_pass_zeroshot",
                "compare": {"token_usage_kind": "actual"},
                "metrics": {
                    "doc_role_micro_f1": 0.31,
                    "doc_instance_micro_f1": 0.02,
                    "doc_combination_micro_f1": 0.02,
                    "doc_event_type_micro_f1": 0.81,
                    "strict_f1": 0.41,
                    "relaxed_f1": 0.51,
                    "type_f1": 0.61,
                    "schema_compliance_rate": 0.91,
                    "hallucination_rate": 0.05,
                    "parse_success_rate": 0.99,
                    "parse_error_rate": 0.01,
                    "avg_gold_events": 1.2,
                    "avg_predicted_events": 1.0,
                    "avg_gold_event_types": 1.1,
                    "avg_schema_event_types": 1.3,
                    "stage1_gold_coverage_rate": None,
                    "stage1_exact_match_rate": None,
                    "stage1_miss_rate": None,
                    "stage1_overprediction_rate": None,
                    "avg_stage1_predicted_types": None,
                    "total_tokens": 1000.0,
                    "avg_tokens_per_sample": 10.0,
                    "wall_clock_seconds": 20.0,
                    "samples_per_second": 5.0,
                    "seconds_per_100_samples": 200.0,
                    "f1_per_1k_tokens": 0.31,
                    "f1_per_minute": 0.93,
                },
            }
        ]
    )

    assert len(sections["main_table"]) == 1
    assert len(sections["diagnostic_table"]) == 1
    assert len(sections["cost_table"]) == 1
    assert sections["cost_table"][0]["token_usage_kind"] == "actual"


def test_validate_suite_completeness_rejects_missing_variant_family_pair():
    with pytest.raises(ValueError, match="incomplete comparable suite"):
        mod.validate_suite_completeness(
            [
                {
                    "family": "api",
                    "variant": "single_pass_zeroshot",
                    "compare": {},
                    "metrics": {},
                }
            ]
        )
