import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "oglans" / "utils" / "hub_runtime.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("oglans_utils_hub_runtime", str(MODULE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configure_modelscope_runtime_sets_expected_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("MODELSCOPE_CACHE", raising=False)

    hub_runtime = _load_module()
    snapshot = hub_runtime.configure_modelscope_runtime(str(tmp_path))

    expected_cache = tmp_path / "data" / "cache" / "modelscope"
    assert snapshot["MODELSCOPE_CACHE"] == str(expected_cache)
    assert os.environ["MODELSCOPE_CACHE"].endswith(str(Path("data") / "cache" / "modelscope"))


def test_configure_hf_hub_runtime_preserves_existing_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "45")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "15")

    hub_runtime = _load_module()
    snapshot = hub_runtime.configure_hf_hub_runtime(str(tmp_path))

    assert snapshot["HF_HUB_DISABLE_XET"] == "0"
    assert snapshot["HF_HUB_DOWNLOAD_TIMEOUT"] == "45"
    assert snapshot["HF_HUB_ETAG_TIMEOUT"] == "15"


def test_resolve_model_name_or_path_prefers_existing_local_path(tmp_path):
    local_model_dir = tmp_path / "local-model"
    local_model_dir.mkdir()

    hub_runtime = _load_module()
    resolved = hub_runtime.resolve_model_name_or_path(str(local_model_dir))

    assert resolved == str(local_model_dir.resolve())


def test_resolve_model_name_or_path_uses_modelscope_when_available(monkeypatch, tmp_path):
    monkeypatch.delenv("MODELSCOPE_CACHE", raising=False)

    hub_runtime = _load_module()

    fake_modelscope = types.ModuleType("modelscope")
    expected_cache = tmp_path / "data" / "cache" / "modelscope"

    def snapshot_download(model_name_or_path, cache_dir=None):
        assert model_name_or_path == "Qwen/Qwen3-4B-Instruct-2507"
        assert cache_dir == str(expected_cache)
        return "/tmp/downloaded-model"

    fake_modelscope.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)

    resolved = hub_runtime.resolve_model_name_or_path(
        "Qwen/Qwen3-4B-Instruct-2507",
        source="modelscope",
        modelscope_cache_dir=str(expected_cache),
        project_root=str(tmp_path),
    )

    assert resolved == "/tmp/downloaded-model"


def test_resolve_model_name_or_path_raises_when_modelscope_fails(monkeypatch, caplog, tmp_path):
    monkeypatch.delenv("MODELSCOPE_CACHE", raising=False)

    hub_runtime = _load_module()

    fake_modelscope = types.ModuleType("modelscope")
    expected_cache = tmp_path / "data" / "cache" / "modelscope"

    def snapshot_download(model_name_or_path, cache_dir=None):
        raise RuntimeError("network timeout")

    fake_modelscope.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="ModelScope download failed"):
            hub_runtime.resolve_model_name_or_path(
                "Qwen/Qwen3-4B-Instruct-2507",
                source="modelscope",
                modelscope_cache_dir=str(expected_cache),
                project_root=str(tmp_path),
            )

    assert "ModelScope download failed" in caplog.text


def test_resolve_model_name_or_path_allows_explicit_huggingface(monkeypatch, tmp_path):
    hub_runtime = _load_module()
    for key in [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_ASSETS_CACHE",
        "HF_XET_CACHE",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_DOWNLOAD_TIMEOUT",
        "HF_HUB_ETAG_TIMEOUT",
    ]:
        monkeypatch.delenv(key, raising=False)

    resolved = hub_runtime.resolve_model_name_or_path(
        "Qwen/Qwen3-4B-Instruct-2507",
        source="huggingface",
        project_root=str(tmp_path),
    )

    assert resolved == "Qwen/Qwen3-4B-Instruct-2507"
    assert os.environ["HF_HOME"].endswith(str(Path("data") / "cache" / "huggingface"))


def test_local_model_source_requires_existing_local_path(tmp_path):
    hub_runtime = _load_module()

    with pytest.raises(RuntimeError, match="model.source=local requires an existing local filesystem path"):
        hub_runtime.resolve_model_name_or_path(
            str(tmp_path / "missing-model"),
            source="local",
            project_root=str(tmp_path),
        )


def test_build_unsloth_from_pretrained_kwargs_enforces_local_files_only_for_local_source():
    hub_runtime = _load_module()

    kwargs = hub_runtime.build_unsloth_from_pretrained_kwargs(
        model_name="/abs/model",
        max_seq_length=4096,
        load_in_4bit=True,
        source="local",
    )

    assert kwargs["model_name"] == "/abs/model"
    assert kwargs["max_seq_length"] == 4096
    assert kwargs["load_in_4bit"] is True
    assert kwargs["local_files_only"] is True


def test_build_unsloth_from_pretrained_kwargs_keeps_remote_sources_online():
    hub_runtime = _load_module()

    kwargs = hub_runtime.build_unsloth_from_pretrained_kwargs(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        max_seq_length=4096,
        load_in_4bit=True,
        source="modelscope",
    )

    assert kwargs["local_files_only"] is False


def test_build_unsloth_from_pretrained_kwargs_passes_attn_implementation(monkeypatch):
    hub_runtime = _load_module()
    fake_import_utils = types.ModuleType("transformers.utils.import_utils")
    fake_import_utils.is_flash_attn_2_available = lambda: True
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake_import_utils)

    kwargs = hub_runtime.build_unsloth_from_pretrained_kwargs(
        model_name="/abs/model",
        max_seq_length=4096,
        load_in_4bit=True,
        source="local",
        attn_implementation="flash_attention_2",
    )

    assert kwargs["attn_implementation"] == "flash_attention_2"


def test_flash_attention_2_requires_runtime_support(monkeypatch):
    hub_runtime = _load_module()
    fake_import_utils = types.ModuleType("transformers.utils.import_utils")
    fake_import_utils.is_flash_attn_2_available = lambda: False
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake_import_utils)

    with pytest.raises(RuntimeError, match="flash_attention_2"):
        hub_runtime.build_unsloth_from_pretrained_kwargs(
            model_name="/abs/model",
            max_seq_length=4096,
            load_in_4bit=True,
            source="local",
            attn_implementation="flash_attention_2",
        )
