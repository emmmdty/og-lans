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
        project_root=str(tmp_path),
    )

    assert resolved == "/tmp/downloaded-model"


def test_resolve_model_name_or_path_raises_when_modelscope_fails(monkeypatch, caplog, tmp_path):
    hub_runtime = _load_module()

    fake_modelscope = types.ModuleType("modelscope")

    def snapshot_download(model_name_or_path, cache_dir=None):
        raise RuntimeError("network timeout")

    fake_modelscope.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="ModelScope download failed"):
            hub_runtime.resolve_model_name_or_path(
                "Qwen/Qwen3-4B-Instruct-2507",
                source="modelscope",
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
