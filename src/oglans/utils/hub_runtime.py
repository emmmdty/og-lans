"""
Shared model download runtime helpers.
"""

from __future__ import annotations

import logging
import os
import importlib
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_HF_HUB_DISABLE_XET = "1"
DEFAULT_HF_HUB_DOWNLOAD_TIMEOUT = "120"
DEFAULT_HF_HUB_ETAG_TIMEOUT = "30"
DEFAULT_MODELSCOPE_CACHE_DIRNAME = "modelscope"


def _default_hf_paths(project_root: str | os.PathLike[str]) -> Dict[str, str]:
    root = Path(project_root).resolve()
    hf_home = root / "data" / "cache" / "huggingface"
    return {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HF_ASSETS_CACHE": str(hf_home / "assets"),
        "HF_XET_CACHE": str(hf_home / "xet"),
        "HF_HUB_DISABLE_XET": DEFAULT_HF_HUB_DISABLE_XET,
        "HF_HUB_DOWNLOAD_TIMEOUT": DEFAULT_HF_HUB_DOWNLOAD_TIMEOUT,
        "HF_HUB_ETAG_TIMEOUT": DEFAULT_HF_HUB_ETAG_TIMEOUT,
    }


def _default_modelscope_paths(project_root: str | os.PathLike[str]) -> Dict[str, str]:
    root = Path(project_root).resolve()
    cache_dir = root / "data" / "cache" / DEFAULT_MODELSCOPE_CACHE_DIRNAME
    return {
        "MODELSCOPE_CACHE": str(cache_dir),
    }


def configure_hf_hub_runtime(
    project_root: str | os.PathLike[str],
    *,
    force: bool = False,
) -> Dict[str, str]:
    """
    Populate a stable Hugging Face runtime environment.

    Existing environment variables win unless force=True.
    """
    defaults = _default_hf_paths(project_root)
    snapshot: Dict[str, str] = {}

    for key, value in defaults.items():
        if force or not os.environ.get(key):
            os.environ[key] = value
        snapshot[key] = os.environ[key]

    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_XET_CACHE"):
        Path(snapshot[key]).mkdir(parents=True, exist_ok=True)

    return snapshot


def configure_modelscope_runtime(
    project_root: str | os.PathLike[str],
    *,
    force: bool = False,
) -> Dict[str, str]:
    """
    Populate a stable ModelScope runtime environment.
    """
    defaults = _default_modelscope_paths(project_root)
    snapshot: Dict[str, str] = {}

    for key, value in defaults.items():
        if force or not os.environ.get(key):
            os.environ[key] = value
        snapshot[key] = os.environ[key]

    Path(snapshot["MODELSCOPE_CACHE"]).mkdir(parents=True, exist_ok=True)
    return snapshot


def configure_model_download_runtime(
    project_root: str | os.PathLike[str],
    *,
    source: str = "modelscope",
    force: bool = False,
) -> Dict[str, str]:
    """
    Configure the default download runtime for the requested source.
    """
    source_name = str(source or "modelscope").lower()
    if source_name == "local":
        return {}
    if source_name == "modelscope":
        return configure_modelscope_runtime(project_root, force=force)
    if source_name == "huggingface":
        return configure_hf_hub_runtime(project_root, force=force)
    raise ValueError(f"Unsupported model source: {source}")


def get_hf_runtime_snapshot(project_root: Optional[str | os.PathLike[str]] = None) -> Dict[str, str]:
    """
    Return the active HF runtime values, filling defaults if needed.
    """
    if project_root is not None:
        return configure_hf_hub_runtime(project_root)
    snapshot: Dict[str, str] = {}
    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_ASSETS_CACHE",
        "HF_XET_CACHE",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_DOWNLOAD_TIMEOUT",
        "HF_HUB_ETAG_TIMEOUT",
    ):
        snapshot[key] = os.environ.get(key, "")
    return snapshot


def get_modelscope_runtime_snapshot(
    project_root: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, str]:
    """
    Return the active ModelScope runtime values, filling defaults if needed.
    """
    if project_root is not None:
        return configure_modelscope_runtime(project_root)
    return {
        "MODELSCOPE_CACHE": os.environ.get("MODELSCOPE_CACHE", ""),
    }


def get_model_download_runtime_snapshot(
    project_root: Optional[str | os.PathLike[str]] = None,
    *,
    source: str = "modelscope",
) -> Dict[str, str]:
    """
    Return active runtime settings for the requested source.
    """
    source_name = str(source or "modelscope").lower()
    if source_name == "local":
        return {}
    if source_name == "modelscope":
        return get_modelscope_runtime_snapshot(project_root)
    if source_name == "huggingface":
        return get_hf_runtime_snapshot(project_root)
    raise ValueError(f"Unsupported model source: {source}")


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        logger = logging.getLogger("OGLANS")
    fn = getattr(logger, level, None)
    if callable(fn):
        fn(message)


def resolve_model_name_or_path(
    model_name_or_path: str,
    *,
    source: str = "modelscope",
    logger: Any = None,
    modelscope_cache_dir: Optional[str] = None,
    project_root: Optional[str | os.PathLike[str]] = None,
) -> str:
    """
    Resolve a model identifier to a local path when possible.
    """
    candidate = Path(model_name_or_path).expanduser()
    if candidate.exists():
        resolved = str(candidate.resolve())
        _log(logger, "info", f"Using local model path: {resolved}")
        return resolved

    source_name = str(source or "modelscope").lower()
    if source_name == "local":
        raise RuntimeError(
            "model.source=local requires an existing local filesystem path. "
            f"Got: {model_name_or_path}"
        )
    if source_name == "modelscope":
        runtime = configure_modelscope_runtime(project_root or Path.cwd())
        cache_dir = modelscope_cache_dir or runtime["MODELSCOPE_CACHE"]
        try:
            from modelscope import snapshot_download

            resolved = snapshot_download(model_name_or_path, cache_dir=cache_dir)
            _log(logger, "info", f"ModelScope resolved model to: {resolved}")
            return resolved
        except Exception as exc:  # pragma: no cover - network/backend failure path
            _log(
                logger,
                "error",
                "ModelScope download failed: "
                f"model={model_name_or_path}, cache_dir={cache_dir}, "
                f"error_type={type(exc).__name__}, error={exc}",
            )
            raise RuntimeError(
                "ModelScope download failed. "
                f"model={model_name_or_path}, cache_dir={cache_dir}, "
                "explicitly set model.source=huggingface if you want to use Hugging Face instead."
            ) from exc

    if source_name == "huggingface":
        if project_root is not None:
            configure_hf_hub_runtime(project_root)
        _log(logger, "info", f"Using HuggingFace model source for: {model_name_or_path}")
        return model_name_or_path

    raise ValueError(f"Unsupported model source: {source}")


def validate_attention_implementation(attn_implementation: Optional[str]) -> Optional[str]:
    normalized = str(attn_implementation or "").strip()
    if not normalized:
        return None
    if normalized not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError(f"Unsupported attention implementation: {normalized}")
    if normalized != "flash_attention_2":
        return normalized

    try:
        import_utils = importlib.import_module("transformers.utils.import_utils")
    except ModuleNotFoundError as exc:  # pragma: no cover - transformers missing is a runtime env issue
        raise RuntimeError(
            "flash_attention_2 requested, but transformers is not installed."
        ) from exc

    is_available = getattr(import_utils, "is_flash_attn_2_available", None)
    if callable(is_available) and not bool(is_available()):
        raise RuntimeError(
            "flash_attention_2 requested, but the current runtime does not support it."
        )
    return normalized


def build_unsloth_from_pretrained_kwargs(
    *,
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    source: str,
    dtype: Any = None,
    attn_implementation: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a strict Unsloth loading contract.

    Local model sources must stay fully offline so Unsloth does not attempt
    remote Hugging Face statistics checks or hub lookups.
    """
    source_name = str(source or "modelscope").lower()
    kwargs = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
        "local_files_only": source_name == "local",
    }
    validated_attn = validate_attention_implementation(attn_implementation)
    if validated_attn is not None:
        kwargs["attn_implementation"] = validated_attn
    return kwargs
