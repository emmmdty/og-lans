import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "oglans" / "utils" / "pathing.py"
spec = importlib.util.spec_from_file_location("oglans_utils_pathing", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
pathing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pathing)


def test_normalize_dataset_name():
    assert pathing.normalize_dataset_name("./data/raw/DuEE-Fin") == "DuEE-Fin"
    assert pathing.normalize_dataset_name("C:/tmp/MyData") == "MyData"


def test_resolve_schema_path_prefers_cli(tmp_path):
    data_dir = tmp_path / "mydata"
    data_dir.mkdir()
    cli_schema = tmp_path / "cli_schema.json"
    cli_schema.write_text("{}", encoding="utf-8")
    auto_schema = data_dir / "mydata_event_schema.json"
    auto_schema.write_text("{}", encoding="utf-8")

    resolved, candidates = pathing.resolve_schema_path(
        data_dir=str(data_dir),
        dataset_name="MyData",
        configured_schema_path=None,
        cli_schema_path=str(cli_schema),
    )
    assert Path(resolved) == cli_schema
    assert Path(candidates[0]) == cli_schema


def test_resolve_schema_path_falls_back_to_configured(tmp_path):
    data_dir = tmp_path / "custom_data_dir"
    data_dir.mkdir()
    configured = tmp_path / "configured_schema.json"
    configured.write_text("{}", encoding="utf-8")

    resolved, candidates = pathing.resolve_schema_path(
        data_dir=str(data_dir),
        dataset_name="AnotherDS",
        configured_schema_path=str(configured),
        cli_schema_path=None,
    )
    assert Path(resolved) == configured
    assert str(configured) in candidates


def test_infer_eval_root_from_config_prefers_log_tag():
    cfg = {
        "project": {
            "output_dir": "./logs/debug/checkpoints",
        }
    }
    root = pathing.infer_eval_root_from_config(cfg, "DuEE-Fin", eval_task="eval_api")
    assert root.replace("\\", "/").endswith("logs/debug/eval_api")


def test_infer_dataset_name_from_config_prefers_dataset_cache_semantics():
    cfg = {
        "project": {
            "dataset_cache_dir": "./data/processed/AnotherDS/exp1",
        }
    }
    assert pathing.infer_dataset_name_from_config(cfg) == "AnotherDS"


def test_build_runtime_context_from_config_path_supports_extends(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    child_cfg = tmp_path / "child.yaml"
    data_dir = tmp_path / "data" / "raw" / "DuEE-Fin"
    data_dir.mkdir(parents=True)
    schema_path = data_dir / "duee_fin_event_schema.json"
    schema_path.write_text("{}", encoding="utf-8")
    base_cfg.write_text(
        """
algorithms:
  ds_cns:
    taxonomy_path: ./data/raw/DuEE-Fin/duee_fin_event_schema.json
project:
  output_dir: ./logs/debug/checkpoints
training:
  mode: sft
""".strip(),
        encoding="utf-8",
    )
    child_cfg.write_text(f"extends: {base_cfg.name}\n", encoding="utf-8")

    ctx = pathing.build_runtime_context_from_config_path(str(child_cfg), project_root=str(tmp_path))
    assert ctx["dataset_name"] == "DuEE-Fin"
    assert ctx["training_mode"] == "sft"
    assert ctx["eval_api_root"].replace("\\", "/").endswith("logs/debug/eval_api")
    assert ctx["eval_checkpoint_root"].replace("\\", "/").endswith("logs/debug/eval_checkpoint")
    assert "eval_local_root" not in ctx
