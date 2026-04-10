#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build high-confidence teacher-silver DuEE-Fin JSONL from evaluation outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oglans.utils.research_protocol import normalize_research_split_manifest
from oglans.utils.teacher_silver import (
    build_teacher_silver_records,
    load_duee_fin_text_index,
    load_jsonl,
    save_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary_results", required=True, type=str)
    parser.add_argument("--secondary_results", type=str, default=None)
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--research_split_manifest", type=str, default=None)
    parser.add_argument("--output_jsonl", required=True, type=str)
    parser.add_argument("--summary_file", type=str, default=None)
    parser.add_argument("--min_role_overlap", type=float, default=0.5)
    parser.add_argument("--allow_preview_text", action="store_true")
    parser.add_argument("--no_require_consensus", action="store_true")
    parser.add_argument("--no_require_matching_event_types", action="store_true")
    parser.add_argument("--allow_parse_failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_rows = load_jsonl(args.primary_results)
    secondary_rows = load_jsonl(args.secondary_results) if args.secondary_results else None
    text_index = load_duee_fin_text_index(args.dataset_file) if args.dataset_file else None
    allowed_ids = None
    if args.research_split_manifest:
        manifest = normalize_research_split_manifest(args.research_split_manifest, pool_split="train_fit")
        allowed_ids = manifest["fit_ids"]

    records, summary = build_teacher_silver_records(
        primary_rows,
        secondary_rows=secondary_rows,
        text_index=text_index,
        allowed_ids=allowed_ids,
        require_consensus=not args.no_require_consensus and secondary_rows is not None,
        require_matching_event_types=not args.no_require_matching_event_types,
        parse_success_only=not args.allow_parse_failures,
        min_role_overlap=args.min_role_overlap,
        allow_preview_text=args.allow_preview_text,
    )
    output_path = save_jsonl(records, args.output_jsonl)

    summary_payload = {
        **summary,
        "primary_results": str(Path(args.primary_results).resolve()),
        "secondary_results": str(Path(args.secondary_results).resolve()) if args.secondary_results else None,
        "dataset_file": str(Path(args.dataset_file).resolve()) if args.dataset_file else None,
        "research_split_manifest": (
            str(Path(args.research_split_manifest).resolve()) if args.research_split_manifest else None
        ),
        "output_jsonl": str(output_path.resolve()),
    }
    summary_path = Path(args.summary_file) if args.summary_file else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
