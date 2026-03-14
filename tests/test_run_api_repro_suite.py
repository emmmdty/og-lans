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
primary_metric: strict_f1
evaluation:
  split: dev
  seeds: [3407, 3408, 3409]
""".strip(),
        encoding="utf-8",
    )
    loaded = mod.load_protocol(str(protocol))
    assert loaded["primary_metric"] == "strict_f1"
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
