#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared run manifest utilities for training/evaluation reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from importlib.metadata import PackageNotFoundError, version as pkg_version
except Exception:  # pragma: no cover
    PackageNotFoundError = Exception  # type: ignore[assignment]

    def pkg_version(_: str) -> str:  # type: ignore[override]
        raise PackageNotFoundError


def _to_path_str(path: str | Path) -> str:
    return str(Path(path))


def get_git_commit(repo_dir: str | Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_to_path_str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip()
    except Exception:
        return None


def get_git_dirty(repo_dir: str | Path) -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_to_path_str(repo_dir),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
        )
        return bool(out.strip())
    except Exception:
        return None


def get_package_versions(package_names: Iterable[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in package_names:
        try:
            versions[name] = pkg_version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def compute_file_sha256(file_path: str | Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_json_sha256(payload: Any) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def build_contract_record(
    *,
    model_profile: Optional[str],
    model_source: Optional[str],
    effective_model_path: Optional[str],
    validation_errors: Optional[Sequence[str]] = None,
    compatibility_contract_version: str = "strict_v1",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    errors = [str(item) for item in (validation_errors or []) if str(item).strip()]
    payload: Dict[str, Any] = {
        "contract_version": str(compatibility_contract_version),
        "model_profile": model_profile,
        "model_source": model_source,
        "effective_model_path": effective_model_path,
        "validation_status": "passed" if not errors else "failed",
        "validation_errors": errors,
    }
    if extra:
        payload.update(extra)
    return payload


def make_validation_error(
    code: str,
    message: str,
    *,
    stage: Optional[str] = None,
    severity: str = "error",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "code": str(code),
        "message": str(message),
        "severity": str(severity),
    }
    if stage:
        record["stage"] = str(stage)
    if details:
        record["details"] = details
    return record


def append_validation_error(
    records: List[Dict[str, Any]],
    *,
    code: str,
    message: str,
    stage: Optional[str] = None,
    severity: str = "error",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = make_validation_error(
        code,
        message,
        stage=stage,
        severity=severity,
        details=details,
    )
    dedupe_key = (
        record.get("code"),
        record.get("stage"),
    )
    for existing in records:
        existing_key = (
            existing.get("code"),
            existing.get("stage"),
        )
        if existing_key == dedupe_key:
            return existing
    records.append(record)
    return record

TRAIN_WRAPPER_VALUE_OPTIONS = {
    "--config",
    "--data_dir",
    "--data-dir",
    "--schema_path",
    "--schema-path",
    "--exp_name",
    "--exp-name",
}


def filter_wrapper_cli_args(
    cli_args: Optional[Sequence[str]],
    *,
    ignored_options: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Drop wrapper-only CLI args so ConfigManager sees the same overrides as main.py.
    """
    args = [str(item) for item in (cli_args or [])]
    ignored = set(ignored_options or TRAIN_WRAPPER_VALUE_OPTIONS)
    filtered: List[str] = []
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--":
            filtered.extend(args[idx + 1 :])
            break
        if token in {"-h", "--help"}:
            idx += 1
            continue
        if token in ignored:
            idx += 2
            continue
        filtered.append(token)
        idx += 1
    return filtered


def load_effective_config_metadata(
    config_path: str | Path,
    *,
    cli_args: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Load the final config seen by main.py, including CLI overrides forwarded by wrapper scripts.
    """
    from oglans.config import ConfigManager

    overrides = filter_wrapper_cli_args(cli_args)
    config = ConfigManager.load_config(str(config_path), overrides)
    training_cfg = config.get("training", {}) or {}
    algorithms = config.get("algorithms", {}) or {}
    return {
        "config": config,
        "config_hash_sha256": compute_json_sha256(config),
        "seed": config.get("project", {}).get("seed"),
        "training_mode": str(training_cfg.get("mode", "preference")),
        "lans_enabled": bool(algorithms.get("lans", {}).get("enabled", False)),
        "scv_enabled": bool(algorithms.get("scv", {}).get("enabled", False)),
        "stage_mode": str(config.get("comparison", {}).get("stage_mode", "single_pass")),
        "fewshot_selection_mode": str(
            config.get("comparison", {}).get("fewshot_selection_mode", "dynamic")
        ),
        "fewshot_pool_split": str(config.get("comparison", {}).get("fewshot_pool_split", "train_fit")),
        "train_tune_ratio": float(config.get("comparison", {}).get("train_tune_ratio", 0.1)),
        "research_split_manifest_path": config.get("comparison", {}).get("research_split_manifest_path"),
    }


def collect_runtime_manifest(
    repo_dir: str | Path,
    package_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": os.path.abspath(os.sys.executable),
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
        },
        "git": {
            "commit": get_git_commit(repo_dir),
            "dirty": get_git_dirty(repo_dir),
        },
    }
    if package_names:
        manifest["packages"] = get_package_versions(package_names)
    return manifest


def build_run_manifest(
    task: str,
    status: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    contract: Optional[Dict[str, Any]] = None,
    runtime: Optional[Dict[str, Any]] = None,
    runtime_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task": task,
        "status": status,
    }
    if meta:
        payload["meta"] = meta
    if artifacts:
        payload["artifacts"] = artifacts
    if contract:
        payload["contract"] = contract
    if runtime:
        payload["runtime"] = runtime
    if runtime_manifest:
        payload["runtime_manifest"] = runtime_manifest
    return payload


def save_json(path: str | Path, payload: Any) -> str:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path_obj)
