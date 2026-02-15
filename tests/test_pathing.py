import importlib.util
from pathlib import Path


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
