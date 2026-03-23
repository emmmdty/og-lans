import importlib.util
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "main.py"


def _install_main_import_stubs(monkeypatch):
    oglans_pkg = types.ModuleType("oglans")
    oglans_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "oglans", oglans_pkg)

    utils_mod = types.ModuleType("oglans.utils")
    utils_mod.setup_logger = lambda *args, **kwargs: None
    utils_mod.collect_runtime_manifest = lambda *args, **kwargs: {}
    utils_mod.compute_file_sha256 = lambda *args, **kwargs: None
    utils_mod.compute_json_sha256 = lambda *args, **kwargs: "x" * 64
    utils_mod.build_run_manifest = lambda *args, **kwargs: {}
    utils_mod.save_json = lambda *args, **kwargs: None
    utils_mod.configure_model_download_runtime = lambda *args, **kwargs: {}
    utils_mod.get_model_download_runtime_snapshot = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "oglans.utils", utils_mod)

    repro_mod = types.ModuleType("oglans.utils.reproducibility")
    repro_mod.set_global_seed = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "oglans.utils.reproducibility", repro_mod)

    data_mod = types.ModuleType("oglans.data")
    data_mod.DuEEFinAdapter = object
    monkeypatch.setitem(sys.modules, "oglans.data", data_mod)

    trainer_mod = types.ModuleType("oglans.trainer")

    class FakeDPOTrainer:
        def __init__(self, config, samples):
            self.config = config
            self.samples = samples

    class FakeSFTTrainer:
        def __init__(self, config, samples):
            self.config = config
            self.samples = samples

    trainer_mod.UnslothDPOTrainerWrapper = FakeDPOTrainer
    trainer_mod.UnslothSFTTrainerWrapper = FakeSFTTrainer
    monkeypatch.setitem(sys.modules, "oglans.trainer", trainer_mod)

    config_mod = types.ModuleType("oglans.config")
    config_mod.ConfigManager = object
    monkeypatch.setitem(sys.modules, "oglans.config", config_mod)

    pathing_mod = types.ModuleType("oglans.utils.pathing")
    pathing_mod.normalize_dataset_name = lambda value: value
    pathing_mod.resolve_schema_path = lambda **kwargs: ("schema.json", ["schema.json"])
    monkeypatch.setitem(sys.modules, "oglans.utils.pathing", pathing_mod)

    prompt_mod = types.ModuleType("oglans.data.prompt_builder")
    prompt_mod.PROMPT_BUILDER_VERSION = "route_a_compare_v1"
    prompt_mod.build_inference_prompt_payload = lambda **kwargs: {
        "prompt_variant": "zeroshot",
        "formatted_text": "",
    }
    monkeypatch.setitem(sys.modules, "oglans.data.prompt_builder", prompt_mod)

    parser_mod = types.ModuleType("oglans.utils.json_parser")
    parser_mod.PARSER_VERSION = "route_a_compare_v1"
    parser_mod.NORMALIZATION_VERSION = "route_a_compare_v1"
    monkeypatch.setitem(sys.modules, "oglans.utils.json_parser", parser_mod)

    yaml_mod = types.ModuleType("yaml")
    yaml_mod.safe_load = lambda *args, **kwargs: {}
    yaml_mod.safe_dump = lambda *args, **kwargs: ""
    monkeypatch.setitem(sys.modules, "yaml", yaml_mod)

    monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))


def _load_main_module(monkeypatch):
    _install_main_import_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location("main_for_test", str(MODULE_PATH))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load main module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_trainer_dispatches_by_training_mode(monkeypatch):
    module = _load_main_module(monkeypatch)

    pref_trainer = module.create_trainer({"training": {"mode": "preference"}}, ["x"])
    sft_trainer = module.create_trainer({"training": {"mode": "sft"}}, ["x"])

    assert type(pref_trainer).__name__ == "FakeDPOTrainer"
    assert type(sft_trainer).__name__ == "FakeSFTTrainer"


def test_create_trainer_rejects_unknown_training_mode(monkeypatch):
    module = _load_main_module(monkeypatch)

    with pytest.raises(ValueError):
        module.create_trainer({"training": {"mode": "invalid"}}, [])
