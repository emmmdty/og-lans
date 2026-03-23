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
from typing import Any, Dict, Iterable, Optional

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
