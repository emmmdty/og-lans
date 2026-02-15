"""
Path resolution helpers for dataset-specific training/evaluation pipelines.
"""

from __future__ import annotations

import os
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
    for key in ("dataset_cache_dir", "output_dir", "logging_dir"):
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
