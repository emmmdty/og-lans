"""
Path resolution helpers for dataset-specific training/evaluation pipelines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def normalize_dataset_name(data_dir: str) -> str:
    """
    Infer dataset name from data directory path.
    """
    norm = os.path.normpath(data_dir or "")
    base = os.path.basename(norm)
    return base or "DuEE-Fin"


def normalize_dataset_slug(dataset_name: str) -> str:
    return (dataset_name or "DuEE-Fin").lower().replace("-", "_")


def _split_path_parts(raw: str) -> List[str]:
    return [p for p in os.path.normpath(str(raw)).replace("\\", "/").split("/") if p and p != "."]


def _infer_dataset_from_data_path(raw: str) -> Optional[str]:
    parts = _split_path_parts(raw)
    if "data" not in parts:
        return None
    idx = parts.index("data")
    if idx + 2 >= len(parts):
        return None
    data_kind = parts[idx + 1]
    if data_kind not in {"raw", "processed"}:
        return None
    dataset_name = parts[idx + 2]
    return dataset_name or None


def infer_dataset_name_from_config(config: Dict[str, Any]) -> Optional[str]:
    """
    Infer dataset name from config paths when possible.
    """
    ds_cns = config.get("algorithms", {}).get("ds_cns", {})
    taxonomy_path = ds_cns.get("taxonomy_path")
    if taxonomy_path:
        dataset_dir = os.path.basename(os.path.dirname(os.path.normpath(str(taxonomy_path))))
        if dataset_dir:
            return dataset_dir

    project = config.get("project", {})
    cache_dir = project.get("dataset_cache_dir")
    if cache_dir:
        inferred = _infer_dataset_from_data_path(str(cache_dir))
        if inferred:
            return inferred

    for key in ("output_dir", "logging_dir"):
        raw = project.get(key)
        if not raw:
            continue
        p = os.path.normpath(str(raw))
        base = os.path.basename(p)
        if base in {"checkpoints", "tensorboard", "samples", "eval", "logs", "log", "train"}:
            base = os.path.basename(os.path.dirname(p))
        if base:
            return base
    return None


def infer_eval_root_from_config(
    config: Dict[str, Any],
    dataset_name: str,
    *,
    eval_task: str,
) -> str:
    """
    Infer logs/<tag>/<eval_task> from project paths when possible.
    """
    project = config.get("project", {})
    for key in ("output_dir", "logging_dir", "debug_data_dir"):
        raw = project.get(key)
        if not raw:
            continue
        norm = os.path.normpath(str(raw)).replace("\\", "/")
        parts = [p for p in norm.split("/") if p and p != "."]
        if "logs" not in parts:
            continue
        idx = parts.index("logs")
        if idx + 1 < len(parts):
            tag = parts[idx + 1]
            if tag:
                return os.path.normpath(os.path.join("logs", tag, eval_task))
    return os.path.normpath(os.path.join("logs", dataset_name, eval_task))


def build_runtime_context_from_config_path(
    config_path: str,
    *,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load config via ConfigManager so wrapper scripts respect extends/defaults.
    """
    from oglans.config import ConfigManager

    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()

    # Wrapper-facing context only needs extends resolution plus runtime defaults.
    # Full semantic contract validation belongs to execution entrypoints.
    config = ConfigManager.load_config(str(cfg_path), validate_semantic=False)
    dataset_name = infer_dataset_name_from_config(config) or "DuEE-Fin"
    taxonomy_path = config.get("algorithms", {}).get("ds_cns", {}).get("taxonomy_path") or ""
    dataset_dir = os.path.dirname(os.path.normpath(str(taxonomy_path))) if taxonomy_path else os.path.normpath(
        os.path.join("data", "raw", dataset_name)
    )
    split_prefix = "duee_fin"

    return {
        "dataset_name": dataset_name,
        "dataset_dir": os.path.normpath(dataset_dir),
        "schema_path": os.path.normpath(str(taxonomy_path)) if taxonomy_path else "",
        "eval_api_root": infer_eval_root_from_config(config, dataset_name, eval_task="eval_api"),
        "eval_checkpoint_root": infer_eval_root_from_config(config, dataset_name, eval_task="eval_checkpoint"),
        "eval_base_root": infer_eval_root_from_config(config, dataset_name, eval_task="eval_base"),
        "eval_academic_root": infer_eval_root_from_config(config, dataset_name, eval_task="eval_academic"),
        "output_root": os.path.normpath(str(config.get("project", {}).get("output_dir", ""))),
        "training_mode": str(config.get("training", {}).get("mode", "preference")),
        "seed": config.get("project", {}).get("seed", ""),
        "max_steps": config.get("training", {}).get("max_steps", ""),
        "num_train_epochs": config.get("training", {}).get("num_train_epochs", ""),
        "logging_steps": config.get("training", {}).get("logging_steps", ""),
        "lans_enabled": bool(config.get("algorithms", {}).get("lans", {}).get("enabled", False)),
        "scv_enabled": bool(config.get("algorithms", {}).get("scv", {}).get("enabled", False)),
        "rpo_alpha": config.get("training", {}).get("rpo", {}).get("alpha", 0.0),
        "rpo_warmup_steps": config.get("training", {}).get("rpo", {}).get("warmup_steps", 0),
        "preference_mode": str(config.get("training", {}).get("preference", {}).get("mode", "ipo")),
        "offset_source": str(
            config.get("training", {}).get("preference", {}).get("offset_source", "margin_bucket")
        ),
        "offset_static": config.get("training", {}).get("preference", {}).get("offset_static", 0.15),
        "lans_refresh_start_epoch": config.get("algorithms", {}).get("lans", {}).get("refresh_start_epoch", 1),
        "lans_refresh_log_interval": config.get("algorithms", {}).get("lans", {}).get("refresh_log_interval", 200),
        "lans_hard_floor_prob": config.get("algorithms", {}).get("lans", {}).get("strategies", {}).get(
            "hard_floor_prob", 0.0
        ),
        "lans_hard_floor_after_warmup": config.get("algorithms", {}).get("lans", {}).get("strategies", {}).get(
            "hard_floor_after_warmup",
            config.get("algorithms", {}).get("lans", {}).get("strategies", {}).get("hard_floor_prob", 0.0),
        ),
        "split_prefix": split_prefix,
    }


def build_schema_candidates(
    data_dir: str,
    dataset_name: str,
    configured_schema_path: Optional[str] = None,
    cli_schema_path: Optional[str] = None,
) -> List[str]:
    """
    Build candidate schema paths with deterministic priority.
    """
    dataset_slug = normalize_dataset_slug(dataset_name)
    candidates: List[str] = []

    def _add(path: Optional[str]) -> None:
        if not path:
            return
        norm = os.path.normpath(path)
        if norm not in candidates:
            candidates.append(norm)

    _add(cli_schema_path)
    _add(os.path.join(data_dir, f"{dataset_slug}_event_schema.json"))
    _add(configured_schema_path)
    if configured_schema_path:
        _add(os.path.join(data_dir, os.path.basename(configured_schema_path)))
    _add(os.path.join(data_dir, "duee_fin_event_schema.json"))
    return candidates


def resolve_schema_path(
    data_dir: str,
    dataset_name: str,
    configured_schema_path: Optional[str] = None,
    cli_schema_path: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """
    Resolve schema path from candidates; return the first existing path.
    If none exists, returns the highest-priority candidate and full candidate list.
    """
    candidates = build_schema_candidates(
        data_dir=data_dir,
        dataset_name=dataset_name,
        configured_schema_path=configured_schema_path,
        cli_schema_path=cli_schema_path,
    )
    for p in candidates:
        if os.path.exists(p):
            return p, candidates
    return (candidates[0] if candidates else ""), candidates
