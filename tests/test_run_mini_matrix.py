import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_mini_matrix.py"
spec = importlib.util.spec_from_file_location("run_mini_matrix", str(SCRIPT_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_parse_experiments_supports_all_keyword():
    experiments = mod.parse_experiments("all")

    assert experiments[0] == "full"
    assert "A7" in experiments


def test_build_training_overrides_include_runtime_paths_and_mini_limits():
    overrides = mod.build_training_overrides(
        seed=3407,
        train_num_samples=12,
        model_name_or_path="./models/qwen",
        model_source="local",
        scv_model_name_or_path="./models/nli",
        scv_source="local",
        train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        max_steps=None,
    )

    assert overrides["project.seed"] == 3407
    assert overrides["project.max_samples"] == 12
    assert overrides["model.base_model"] == "./models/qwen"
    assert overrides["model.source"] == "local"
    assert overrides["algorithms.scv.nli_model"] == "./models/nli"
    assert overrides["training.per_device_train_batch_size"] == 2
    assert overrides["training.gradient_accumulation_steps"] == 4


def test_build_eval_command_switches_between_base_and_checkpoint_modes(tmp_path):
    base_cmd = mod.build_eval_command(
        config_path=tmp_path / "config.yaml",
        protocol_path="configs/eval_protocol.yaml",
        role_alias_map="configs/role_aliases_duee_fin.yaml",
        output_file=tmp_path / "base.jsonl",
        prompt_variant="fewshot",
        fewshot_num_examples=2,
        eval_num_samples=12,
        eval_batch_size=16,
        seed=3407,
        report_primary_metric="doc_role_micro_f1",
        canonical_metric_mode="analysis_only",
        model_name_or_path="./models/qwen",
        model_source="local",
        checkpoint_path=None,
    )

    assert "--base_only" in base_cmd
    assert "--checkpoint" not in base_cmd
    assert "--prompt_variant" in base_cmd
    assert "--fewshot_num_examples" in base_cmd
    assert "--model.source" in base_cmd

    adapter_cmd = mod.build_eval_command(
        config_path=tmp_path / "config.yaml",
        protocol_path="configs/eval_protocol.yaml",
        role_alias_map="configs/role_aliases_duee_fin.yaml",
        output_file=tmp_path / "adapter.jsonl",
        prompt_variant="zeroshot",
        fewshot_num_examples=2,
        eval_num_samples=12,
        eval_batch_size=16,
        seed=3407,
        report_primary_metric="doc_role_micro_f1",
        canonical_metric_mode="analysis_only",
        model_name_or_path="./models/qwen",
        model_source="local",
        checkpoint_path="logs/DuEE-Fin/checkpoints/full_s3407",
    )

    assert "--checkpoint" in adapter_cmd
    assert "--base_only" not in adapter_cmd
    assert "--fewshot_num_examples" not in adapter_cmd
    assert "--model.source" in adapter_cmd


def _nested_summary(doc_role: float, relaxed: float, type_f1: float, schema: float, hallucination: float):
    return {
        "metrics": {
            "academic_metrics": {
                "doc_ee": {
                    "overall": {"MicroF1": doc_role},
                    "instance": {"MicroF1": 0.11},
                    "combination": {"MicroF1": 0.12},
                    "classification": {"MicroF1": 0.91},
                }
            },
            "strict": {"f1": 0.41},
            "relaxed": {"f1": relaxed},
            "type_identification": {"f1": type_f1},
            "schema_compliance_rate": schema,
            "hallucination": {"sample_rate": hallucination},
        }
    }


def test_build_metric_row_preserves_nested_diagnostic_metrics():
    row = mod.build_metric_row(_nested_summary(0.31, 0.62, 0.73, 0.84, 0.05), 3407)

    assert row["doc_role_micro_f1"] == 0.31
    assert row["relaxed_f1"] == 0.62
    assert row["type_f1"] == 0.73
    assert row["schema_compliance_rate"] == 0.84
    assert row["hallucination_rate"] == 0.05
    assert row["seed"] == 3407.0


def test_compute_significance_skips_single_seed_and_returns_metadata():
    seed_map = {
        "base": {"zeroshot": {3407: _nested_summary(0.2, 0.3, 0.4, 0.5, 0.1)}},
        "full": {"zeroshot": {3407: _nested_summary(0.25, 0.35, 0.45, 0.55, 0.08)}},
    }

    significance, metadata = mod.compute_significance(
        by_run_prompt_seed=seed_map,
        prompt_modes=["zeroshot"],
        seeds=[3407],
        primary_metric="doc_role_micro_f1",
    )

    assert significance == {}
    assert metadata["significance_status"] == "skipped_insufficient_pairs"
    assert metadata["significance_min_pairs"] == 2
