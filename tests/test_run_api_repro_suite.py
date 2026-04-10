import importlib.util
from pathlib import Path

import pytest

from oglans.utils.compare_contract import build_compare_contract


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_api_repro_suite.py"
spec = importlib.util.spec_from_file_location("run_api_repro_suite", str(SCRIPT_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_infer_dataset_name_from_config_taxonomy(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
algorithms:
  ds_cns:
    taxonomy_path: ./data/raw/MyDataset/mydataset_event_schema.json
""".strip(),
        encoding="utf-8",
    )
    assert mod.infer_dataset_name_from_config(str(cfg)) == "MyDataset"


def test_infer_dataset_name_from_config_project_fallback(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
project:
  dataset_cache_dir: ./data/processed/AnotherDS/exp1
""".strip(),
        encoding="utf-8",
    )
    assert mod.infer_dataset_name_from_config(str(cfg)) == "AnotherDS"


def test_load_protocol_merges_defaults(tmp_path):
    protocol = tmp_path / "eval_protocol.yaml"
    protocol.write_text(
        """
primary_metric: doc_role_micro_f1
evaluation:
  split: dev
  seeds: [3407, 3408, 3409]
""".strip(),
        encoding="utf-8",
    )
    loaded = mod.load_protocol(str(protocol))
    assert loaded["primary_metric"] == "doc_role_micro_f1"
    assert loaded["evaluation"]["split"] == "dev"
    assert loaded["evaluation"]["concurrency"] == 8


def test_infer_eval_api_root_from_config_debug_tag(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
project:
  output_dir: ./logs/debug/checkpoints
  logging_dir: ./logs/debug/tensorboard
""".strip(),
        encoding="utf-8",
    )
    root = mod.infer_eval_api_root_from_config(str(cfg), "DuEE-Fin")
    assert root.as_posix().endswith("/logs/debug/eval_api")


def test_infer_eval_api_root_from_config_dataset_fallback(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("project: {}", encoding="utf-8")
    root = mod.infer_eval_api_root_from_config(str(cfg), "DuEE-Fin")
    assert root.as_posix().endswith("/logs/DuEE-Fin/eval_api")


def test_infer_dataset_name_from_config_supports_extends(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    child_cfg = tmp_path / "child.yaml"
    base_cfg.write_text(
        """
algorithms:
  ds_cns:
    taxonomy_path: ./data/raw/DuEE-Fin/duee_fin_event_schema.json
""".strip(),
        encoding="utf-8",
    )
    child_cfg.write_text(
        f"""
extends: {base_cfg.name}
project:
  name: Plain-SFT-Gen
""".strip(),
        encoding="utf-8",
    )
    assert mod.infer_dataset_name_from_config(str(child_cfg)) == "DuEE-Fin"


def _summary(metric_value: float, *, prompt_variant: str = "zeroshot", stage_mode: str = "single_pass", seed: int = 3407) -> dict:
    return {
        "compare": build_compare_contract(
            {
                "model_family": "api",
                "model_kind": "api_model",
                "split": "dev",
                "primary_metric": "doc_role_micro_f1",
                "stage_mode": stage_mode,
                "prompt_variant": prompt_variant,
                "fewshot_num_examples": 0 if prompt_variant == "zeroshot" else 3,
                "fewshot_selection_mode": "none" if prompt_variant == "zeroshot" else "dynamic",
                "fewshot_pool_split": "none" if prompt_variant == "zeroshot" else "train_fit",
                "train_tune_ratio": 0.1,
                "research_split_manifest_path": "/tmp/frozen.json",
                "research_split_manifest_hash": "a" * 64,
                "pipeline_mode": "e2e",
                "canonical_metric_mode": "analysis_only",
                "protocol_hash": "b" * 64,
                "role_alias_hash": "c" * 64,
                "seed": seed,
                "seed_effective": False,
                "token_usage_kind": "actual",
            }
        ),
        "metrics": {
            "doc_role_micro_f1": metric_value,
            "doc_instance_micro_f1": 0.11,
            "doc_combination_micro_f1": 0.12,
            "doc_event_type_micro_f1": 0.91,
            "strict_f1": 0.41,
            "relaxed_f1": 0.51,
            "type_f1": 0.61,
        }
    }


def test_compute_significance_skips_single_seed_and_sets_metadata():
    significance, metadata = mod.compute_significance(
        {
            "zeroshot": {3407: _summary(0.2)},
            "fewshot": {3407: _summary(0.25)},
        },
        report_primary_metric="doc_role_micro_f1",
        expected_seeds=[3407],
    )

    assert significance == {}
    assert metadata["significance_status"] == "skipped_insufficient_pairs"
    assert metadata["significance_min_pairs"] == 2


def test_validate_mode_contracts_returns_hash_per_mode():
    hashes = mod.validate_mode_contracts(
        {
            "zeroshot": {3407: _summary(0.2, prompt_variant="zeroshot")},
            "fewshot": {3407: _summary(0.25, prompt_variant="fewshot")},
        }
    )

    assert set(hashes) == {"zeroshot", "fewshot"}
    assert hashes["zeroshot"] != hashes["fewshot"]


def test_validate_mode_contracts_rejects_mismatch_within_mode():
    with pytest.raises(ValueError, match="comparable contract mismatch"):
        mod.validate_mode_contracts(
            {
                "fewshot": {
                    3407: _summary(0.25, prompt_variant="fewshot", stage_mode="single_pass", seed=3407),
                    3408: _summary(0.26, prompt_variant="fewshot", stage_mode="two_stage", seed=3408),
                }
            }
        )


def test_compute_significance_rejects_incomplete_seed_coverage():
    try:
        mod.compute_significance(
            {
                "zeroshot": {3407: _summary(0.2), 3408: _summary(0.21)},
                "fewshot": {3407: _summary(0.25)},
            },
            report_primary_metric="doc_role_micro_f1",
            expected_seeds=[3407, 3408],
        )
    except ValueError as exc:
        assert "incomplete seed coverage for significance" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incomplete seed coverage")


def test_build_cmd_forwards_base_url():
    cmd = mod.build_cmd(
        evaluate_api_path=Path("/tmp/evaluate_api.py"),
        config="configs/config.yaml",
        protocol="configs/eval_protocol.yaml",
        mode="zeroshot",
        split="dev",
        seed=3407,
        output_file=Path("/tmp/out.jsonl"),
        summary_file=Path("/tmp/summary.json"),
        concurrency=8,
        json_mode="auto",
        model="provider-model",
        base_url="https://provider.example/v1",
        num_samples=32,
        bootstrap_samples=1000,
        role_alias_map="configs/role_aliases_duee_fin.yaml",
        canonical_metric_mode="analysis_only",
        report_primary_metric="doc_role_micro_f1",
        fewshot_num_examples=None,
    )

    assert "--base_url" in cmd
    assert "https://provider.example/v1" in cmd


def test_build_cmd_forwards_stage_and_pool_controls():
    cmd = mod.build_cmd(
        evaluate_api_path=Path("/tmp/evaluate_api.py"),
        config="configs/config.yaml",
        protocol="configs/eval_protocol.yaml",
        mode="fewshot",
        split="dev",
        seed=3407,
        output_file=Path("/tmp/out.jsonl"),
        summary_file=Path("/tmp/summary.json"),
        concurrency=8,
        json_mode="auto",
        model="provider-model",
        base_url=None,
        num_samples=32,
        bootstrap_samples=1000,
        role_alias_map="configs/role_aliases_duee_fin.yaml",
        canonical_metric_mode="analysis_only",
        report_primary_metric="doc_role_micro_f1",
        fewshot_num_examples=3,
        stage_mode="two_stage",
        fewshot_selection_mode="dynamic",
        fewshot_pool_split="train_fit",
        train_tune_ratio=0.1,
        research_split_manifest="configs/research_splits/frozen.json",
    )

    assert "--stage_mode" in cmd
    assert "two_stage" in cmd
    assert "--fewshot_selection_mode" in cmd
    assert "dynamic" in cmd
    assert "--fewshot_pool_split" in cmd
    assert "train_fit" in cmd
    assert "--train_tune_ratio" in cmd
    assert "0.1" in cmd
    assert "--research_split_manifest" in cmd
    assert "configs/research_splits/frozen.json" in cmd
