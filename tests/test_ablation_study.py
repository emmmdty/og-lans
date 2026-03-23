import importlib.util
from pathlib import Path


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
algorithms:
  ds_cns:
    taxonomy_path: ./data/raw/DuEE-Fin/duee_fin_event_schema.json
training:
  mode: preference
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
