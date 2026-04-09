import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ablation_study.py"
spec = importlib.util.spec_from_file_location("ablation_study", str(SCRIPT_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_load_config_supports_extends(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    child_cfg = tmp_path / "child.yaml"
    base_cfg.write_text(
        """
model:
  profile: qwen3_instruct
  source: modelscope
algorithms:
  ds_cns:
    taxonomy_path: ./data/raw/DuEE-Fin/duee_fin_event_schema.json
training:
  mode: preference
comparison:
  prompt_builder_version: phase3_mvp_v1
  parser_version: phase3_mvp_v1
  normalization_version: phase3_mvp_v1
evaluation:
  mode: scored
""".strip(),
        encoding="utf-8",
    )
    child_cfg.write_text(
        f"""
extends: {base_cfg.name}
training:
  mode: sft
project:
  name: Plain-SFT-Gen
""".strip(),
        encoding="utf-8",
    )

    cfg = mod.load_config(str(child_cfg))

    assert cfg["algorithms"]["ds_cns"]["taxonomy_path"] == "./data/raw/DuEE-Fin/duee_fin_event_schema.json"
    assert cfg["training"]["mode"] == "sft"


def test_validate_eval_split_rejects_non_dev():
    with pytest.raises(ValueError, match="dev"):
        mod.validate_eval_split("test")


def test_build_seeded_experiment_name_is_deterministic():
    assert mod.build_seeded_experiment_name("A2_no_scv", 3047) == "A2_no_scv_s3047"


def test_resolve_checkpoint_dir_uses_dataset_logs_root(tmp_path):
    checkpoint_dir = mod.resolve_checkpoint_dir(
        project_root=tmp_path,
        dataset_name="DuEE-Fin",
        experiment_name="A2_no_scv_s3047",
    )

    assert checkpoint_dir == tmp_path / "logs" / "DuEE-Fin" / "checkpoints" / "A2_no_scv_s3047"


def test_require_explicit_seeds_rejects_missing_value():
    with pytest.raises(ValueError, match="explicit --seeds"):
        mod.require_explicit_seeds(None)


def test_require_explicit_seeds_parses_csv():
    assert mod.require_explicit_seeds("3047, 3048") == [3047, 3048]


def test_ablation_script_defaults_to_doc_role_primary_metric():
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'default="strict_f1"' not in script_text
    assert 'default="doc_role_micro_f1"' in script_text


def test_ablation_script_exposes_prompt_mode_controls():
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--prompt_modes" in script_text
    assert "--fewshot_num_examples" in script_text


def test_flatten_metrics_includes_relaxed_type_schema_fields():
    metrics = mod._flatten_metrics(
        {
            "academic_metrics": {
                "doc_ee": {
                    "overall": {"MicroF1": 0.31},
                    "instance": {"MicroF1": 0.22},
                    "combination": {"MicroF1": 0.23},
                    "classification": {"MicroF1": 0.91},
                }
            },
            "strict": {"precision": 0.41, "recall": 0.42, "f1": 0.4},
            "relaxed": {"f1": 0.52},
            "type_identification": {"f1": 0.63},
            "schema_compliance_rate": 0.74,
            "hallucination": {"sample_rate": 0.08},
        }
    )

    assert metrics["relaxed_f1"] == 0.52
    assert metrics["type_f1"] == 0.63
    assert metrics["schema_compliance_rate"] == 0.74


def test_generate_latex_table_uses_academic_first_columns():
    latex = mod.generate_latex_table(
        {
            "full": {
                "status": "success",
                "metrics": {
                    "doc_role_micro_f1": 0.31,
                    "doc_instance_micro_f1": 0.22,
                    "doc_combination_micro_f1": 0.23,
                    "doc_event_type_micro_f1": 0.91,
                    "strict_f1": 0.4,
                    "relaxed_f1": 0.52,
                    "type_f1": 0.63,
                    "schema_compliance_rate": 0.74,
                    "hallucination_rate": 0.08,
                },
            }
        }
    )

    assert "Doc Role F1" in latex
    assert "Strict P" not in latex


def test_generate_latex_table_rejects_missing_aggregated_mean():
    with pytest.raises(ValueError, match="Missing aggregated mean"):
        mod.generate_latex_table(
            {
                "full": {
                    "aggregated": {
                        "metrics": {
                            "doc_role_micro_f1": {"std": 0.0},
                        }
                    }
                }
            }
        )
