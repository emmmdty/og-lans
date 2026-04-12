#!/usr/bin/env python3
"""
Migrate legacy model caches into the canonical repository model root.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


MODEL_MARKER_FILES = (
    "config.json",
    "configuration.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "generation_config.json",
    "model_index.json",
)


def canonical_models_root(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "models"


def is_model_snapshot_dir(path: Path) -> bool:
    return path.is_dir() and any((path / marker).exists() for marker in MODEL_MARKER_FILES)


def discover_model_directories(source_root: str | Path) -> List[Dict[str, Any]]:
    root = Path(source_root).resolve()
    if not root.exists():
        return []

    discovered: List[Dict[str, Any]] = []
    for org_dir in sorted(root.iterdir()):
        if not org_dir.is_dir():
            continue
        for model_dir in sorted(org_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if not is_model_snapshot_dir(model_dir):
                continue
            discovered.append(
                {
                    "identifier": f"{org_dir.name}/{model_dir.name}",
                    "source": model_dir,
                }
            )
    return discovered


def _normalize_source_roots(source_roots: Iterable[str | Path]) -> List[Path]:
    return [Path(root).resolve() for root in source_roots]


def _build_migration_plan(project_root: str | Path, source_roots: Iterable[str | Path]) -> Dict[str, Any]:
    project_root_path = Path(project_root).resolve()
    models_root = canonical_models_root(project_root_path)
    normalized_roots = _normalize_source_roots(source_roots)
    missing_roots = [str(root) for root in normalized_roots if not root.exists()]

    planned: List[Dict[str, Any]] = []
    for source_root in normalized_roots:
        for item in discover_model_directories(source_root):
            identifier = item["identifier"]
            source = Path(item["source"]).resolve()
            target = models_root / identifier
            planned.append(
                {
                    "identifier": identifier,
                    "source": str(source),
                    "source_root": str(source_root),
                    "target": str(target.resolve(strict=False)),
                    "already_canonical": source == target.resolve(strict=False),
                }
            )
    planned.sort(key=lambda item: (item["identifier"], item["source"]))
    return {
        "project_root": str(project_root_path),
        "models_root": str(models_root),
        "source_roots": [str(root) for root in normalized_roots],
        "missing_source_roots": missing_roots,
        "migrations": planned,
    }


def migrate_model_store(
    *,
    project_root: str | Path,
    source_roots: Iterable[str | Path],
    mode: str = "copy",
    dry_run: bool = False,
    link_legacy: bool = False,
) -> Dict[str, Any]:
    normalized_mode = str(mode).lower()
    if normalized_mode not in {"copy", "move"}:
        raise ValueError(f"Unsupported migration mode: {mode}")
    if link_legacy and normalized_mode != "move":
        raise ValueError("link_legacy requires mode='move'")

    summary = _build_migration_plan(project_root, source_roots)
    summary["mode"] = normalized_mode
    summary["dry_run"] = bool(dry_run)
    summary["migrated"] = 0
    summary["linked_legacy"] = 0
    summary["skipped_already_canonical"] = 0

    if dry_run:
        return summary

    models_root = Path(summary["models_root"])
    models_root.mkdir(parents=True, exist_ok=True)

    for item in summary["migrations"]:
        source = Path(item["source"])
        target = Path(item["target"])
        if item["already_canonical"]:
            item["action"] = "skip_already_canonical"
            summary["skipped_already_canonical"] += 1
            continue
        if target.exists():
            raise FileExistsError(f"Target model directory already exists: {target}")

        target.parent.mkdir(parents=True, exist_ok=True)
        if normalized_mode == "copy":
            shutil.copytree(source, target, symlinks=True)
            item["action"] = "copied"
        else:
            shutil.move(str(source), str(target))
            item["action"] = "moved"
            if link_legacy:
                source.parent.mkdir(parents=True, exist_ok=True)
                source.symlink_to(target, target_is_directory=True)
                item["legacy_link"] = str(source)
                summary["linked_legacy"] += 1
        summary["migrated"] += 1

    return summary


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy model caches into ./models")
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Repository root that owns the canonical ./models directory.",
    )
    parser.add_argument(
        "--from",
        dest="source_roots",
        action="append",
        default=None,
        help="Legacy model root to migrate from. Can be provided multiple times.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["copy", "move"],
        default="copy",
        help="Migration mode for discovered model directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report the migration plan without writing files.",
    )
    parser.add_argument(
        "--link-legacy",
        action="store_true",
        help="After move, recreate the legacy path as a symlink to the new canonical path.",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Optional JSON file to write the migration summary to.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    source_roots = args.source_roots or [str(Path(args.project_root) / "data" / "cache" / "modelscope")]
    summary = migrate_model_store(
        project_root=args.project_root,
        source_roots=source_roots,
        mode=args.mode,
        dry_run=args.dry_run,
        link_legacy=args.link_legacy,
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
