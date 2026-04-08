import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_local_repro_suite.py"
spec = importlib.util.spec_from_file_location("run_local_repro_suite", str(SCRIPT_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_parse_checkpoint_mapping_supports_expected_keys():
    parsed = mod.parse_checkpoint_mapping("full=logs/full,a2=logs/a2")
    assert parsed == {"full": "logs/full", "a2": "logs/a2"}


def test_validate_eval_artifacts_requires_protocol_and_version_metadata(tmp_path):
    summary = tmp_path / "summary.json"
    manifest = tmp_path / "run_manifest.json"

    summary.write_text(
        json.dumps(
            {
                "meta": {
                    "run_id": "full_seed3407",
                    "checkpoint": "/tmp/full",
                    "protocol_hash_sha256": "abc123",
                    "prompt_builder_version": "route_a_compare_v1",
                    "parser_version": "route_a_compare_v1",
                    "normalization_version": "route_a_compare_v1",
                },
                "metrics": {"strict_f1": 0.4},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "meta": {
                    "checkpoint": "/tmp/full",
                    "protocol_hash_sha256": "abc123",
                    "prompt_builder_version": "route_a_compare_v1",
                    "parser_version": "route_a_compare_v1",
                    "normalization_version": "route_a_compare_v1",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    validated = mod.validate_eval_artifacts(
        summary_file=summary,
        run_manifest_file=manifest,
        run_key="full",
    )

    assert validated["metrics"]["strict_f1"] == 0.4


def test_validate_eval_artifacts_rejects_missing_checkpoint_for_adapter_runs(tmp_path):
    summary = tmp_path / "summary.json"
    manifest = tmp_path / "run_manifest.json"

    summary.write_text(
        json.dumps(
            {
                "meta": {
                    "run_id": "full_seed3407",
                    "protocol_hash_sha256": "abc123",
                    "prompt_builder_version": "route_a_compare_v1",
                    "parser_version": "route_a_compare_v1",
                    "normalization_version": "route_a_compare_v1",
                },
                "metrics": {"strict_f1": 0.4},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "meta": {
                    "protocol_hash_sha256": "abc123",
                    "prompt_builder_version": "route_a_compare_v1",
                    "parser_version": "route_a_compare_v1",
                    "normalization_version": "route_a_compare_v1",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required metadata: checkpoint"):
        mod.validate_eval_artifacts(
            summary_file=summary,
            run_manifest_file=manifest,
            run_key="full",
        )


def test_build_eval_command_uses_base_only_for_base_and_checkpoint_for_adapters():
    base_cmd = mod.build_eval_command(
        evaluate_path=Path("evaluate.py"),
        config="configs/config.yaml",
        protocol="configs/eval_protocol.yaml",
        role_alias_map="configs/role_aliases_duee_fin.yaml",
        run_key="base",
        split="dev",
        seed=3407,
        output_file=Path("base.jsonl"),
        batch_size=4,
        canonical_metric_mode="analysis_only",
        report_primary_metric="doc_role_micro_f1",
        model_name_or_path="/models/base",
        checkpoint_path=None,
    )
    assert "--base_only" in base_cmd
    assert "--model_name_or_path" in base_cmd
    assert "--checkpoint" not in base_cmd

    adapter_cmd = mod.build_eval_command(
        evaluate_path=Path("evaluate.py"),
        config="configs/config.yaml",
        protocol="configs/eval_protocol.yaml",
        role_alias_map="configs/role_aliases_duee_fin.yaml",
        run_key="full",
        split="dev",
        seed=3407,
        output_file=Path("full.jsonl"),
        batch_size=4,
        canonical_metric_mode="analysis_only",
        report_primary_metric="doc_role_micro_f1",
        model_name_or_path="/models/base",
        checkpoint_path="logs/full",
    )
    assert "--checkpoint" in adapter_cmd
    assert "--base_only" not in adapter_cmd


def test_ensure_complete_seed_coverage_fail_fast_on_missing_runs():
    records = [
        mod.RunRecord(
            run_key="base",
            seed=3407,
            command=[],
            output_file="base3407.jsonl",
            summary_file="base3407_summary.json",
            run_manifest_file="base3407_manifest.json",
            returncode=0,
            duration_seconds=1.0,
            ok=True,
        ),
        mod.RunRecord(
            run_key="base",
            seed=3408,
            command=[],
            output_file="base3408.jsonl",
            summary_file="base3408_summary.json",
            run_manifest_file="base3408_manifest.json",
            returncode=0,
            duration_seconds=1.0,
            ok=True,
        ),
        mod.RunRecord(
            run_key="full",
            seed=3407,
            command=[],
            output_file="full3407.jsonl",
            summary_file="full3407_summary.json",
            run_manifest_file="full3407_manifest.json",
            returncode=0,
            duration_seconds=1.0,
            ok=True,
        ),
    ]

    with pytest.raises(ValueError, match="incomplete seed coverage"):
        mod.ensure_complete_seed_coverage(
            records=records,
            run_keys=["base", "full"],
            seeds=[3407, 3408],
        )
