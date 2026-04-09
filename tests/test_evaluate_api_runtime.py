import importlib.util
import os
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "evaluate_api.py"
spec = importlib.util.spec_from_file_location("evaluate_api", str(SCRIPT_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_resolve_api_runtime_config_cli_base_url_overrides_env_and_config(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.deepseek.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.openai.example/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    resolved = mod.resolve_api_runtime_config(
        base_url_override="https://cli.example/v1",
        model_override="cli-model",
        api_cfg={
            "base_url": "https://config.example/v1",
            "model": "config-model",
            "api_key": None,
        },
        environ=os.environ,
    )

    assert resolved["base_url"] == "https://cli.example/v1"
    assert resolved["base_url_source"] == "cli"
    assert resolved["model_name"] == "cli-model"
    assert resolved["model_source"] == "cli"
    assert resolved["api_key"] == "deepseek-secret"
    assert resolved["api_key_source"] == "env:DEEPSEEK_API_KEY"


def test_resolve_api_runtime_config_env_base_url_overrides_config(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.openai.example/v1")
    monkeypatch.delenv("DEEPSEEK_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    resolved = mod.resolve_api_runtime_config(
        base_url_override=None,
        model_override=None,
        api_cfg={
            "base_url": "https://config.example/v1",
            "model": "config-model",
            "api_key": None,
        },
        environ=os.environ,
    )

    assert resolved["base_url"] == "https://env.openai.example/v1"
    assert resolved["base_url_source"] == "env:OPENAI_BASE_URL"
    assert resolved["model_name"] == "config-model"
    assert resolved["model_source"] == "config"
    assert resolved["api_key"] == "openai-secret"
    assert resolved["api_key_source"] == "env:OPENAI_API_KEY"


def test_resolve_api_runtime_config_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolved = mod.resolve_api_runtime_config(
        base_url_override=None,
        model_override=None,
        api_cfg={
            "base_url": "https://config.example/v1",
            "model": "config-model",
            "api_key": "config-secret",
        },
        environ=os.environ,
    )

    assert resolved["base_url"] == "https://config.example/v1"
    assert resolved["base_url_source"] == "config"
    assert resolved["model_name"] == "config-model"
    assert resolved["model_source"] == "config"
    assert resolved["api_key"] == "config-secret"
    assert resolved["api_key_source"] == "config"


def test_resolve_api_runtime_config_prefers_provider_default_model_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    resolved = mod.resolve_api_runtime_config(
        base_url_override=None,
        model_override=None,
        api_cfg={
            "base_url": None,
            "model": "config-model",
            "api_key": None,
        },
        environ=os.environ,
    )

    assert resolved["base_url"] == "https://api.deepseek.com"
    assert resolved["base_url_source"] == "env:DEEPSEEK_BASE_URL"
    assert resolved["model_name"] == "deepseek-reasoner"
    assert resolved["model_source"] == "env:DEEPSEEK_DEFAULT_MODEL"


def test_resolve_api_runtime_config_prefers_openai_default_model_for_openai_endpoint(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.openai.example/v1")
    monkeypatch.setenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat")
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4.1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    resolved = mod.resolve_api_runtime_config(
        base_url_override=None,
        model_override=None,
        api_cfg={
            "base_url": None,
            "model": "config-model",
            "api_key": None,
        },
        environ=os.environ,
    )

    assert resolved["base_url"] == "https://provider.openai.example/v1"
    assert resolved["base_url_source"] == "env:OPENAI_BASE_URL"
    assert resolved["model_name"] == "gpt-4.1"
    assert resolved["model_source"] == "env:OPENAI_DEFAULT_MODEL"
