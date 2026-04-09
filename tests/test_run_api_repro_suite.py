import importlib.util
from pathlib import Path


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


def _summary(metric_value: float) -> dict:
    return {
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
