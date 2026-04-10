#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit training-side LANS/SCV signals and exported sample semantics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def infer_samples_dir(checkpoint_manifest_path: Path) -> Path:
    manifest = load_json(checkpoint_manifest_path)
    exp_name = str(manifest.get("meta", {}).get("exp_name", "")).strip()
    if not exp_name:
        raise ValueError(f"Cannot infer exp_name from manifest: {checkpoint_manifest_path}")
    checkpoint_dir = checkpoint_manifest_path.parent
    dataset_root = checkpoint_dir.parent.parent
    return dataset_root / "samples" / exp_name


def summarize_filtered_events(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    row_count = 0
    unique_sample_ids = set()
    reason_counter: Counter[str] = Counter()
    strategy_counter: Counter[str] = Counter()
    attempt_counter: Counter[int] = Counter()

    for row in rows:
        row_count += 1
        unique_sample_ids.add(str(row.get("sample_id", "")))
        reason_counter[str(row.get("reason", ""))] += 1
        strategy_counter[str(row.get("strategy", ""))] += 1
        attempt_counter[int(row.get("attempt", 0))] += 1

    return {
        "row_count": row_count,
        "unique_sample_ids": len(unique_sample_ids),
        "reason_distribution": dict(reason_counter),
        "strategy_distribution": dict(strategy_counter),
        "attempt_distribution": {str(k): v for k, v in sorted(attempt_counter.items())},
    }


def audit_training_signal(checkpoint_manifest_path: Path) -> Dict[str, Any]:
    manifest = load_json(checkpoint_manifest_path)
    samples_dir = infer_samples_dir(checkpoint_manifest_path)
    summary_path = samples_dir / "lans_sampling_summary.json"
    generated_path = samples_dir / "lans_generated_samples.jsonl"
    filtered_path = samples_dir / "scv_filtered_samples.jsonl"

    summary = load_json(summary_path)
    generated_rows = sum(1 for _ in iter_jsonl(generated_path))
    filtered_summary = summarize_filtered_events(iter_jsonl(filtered_path))

    filtered_sample_count = int(summary.get("scv_filtered_count", 0))
    filtered_event_count = int(summary.get("scv_filter_event_count", 0))
    unique_sample_ids = int(filtered_summary["unique_sample_ids"])
    event_rows = int(filtered_summary["row_count"])

    explanations: List[str] = []
    if unique_sample_ids == filtered_sample_count and event_rows != filtered_sample_count:
        explanations.append(
            "scv_filtered_count counts generated samples that triggered SCV at least once, "
            "while scv_filtered_samples.jsonl logs every SCV filtering event/attempt."
        )

    return {
        "exp_name": manifest.get("meta", {}).get("exp_name"),
        "training_mode": manifest.get("meta", {}).get("training_mode"),
        "stage_mode": manifest.get("meta", {}).get("stage_mode"),
        "prompt_variant": manifest.get("meta", {}).get("prompt_variant"),
        "configured_train_count": manifest.get("meta", {}).get("configured_train_count"),
        "effective_train_count": manifest.get("meta", {}).get("effective_train_count"),
        "effective_lans_enabled": manifest.get("meta", {}).get("effective_lans_enabled"),
        "effective_scv_enabled": manifest.get("meta", {}).get("effective_scv_enabled"),
        "wall_clock_seconds": manifest.get("runtime", {}).get("wall_clock_seconds"),
        "samples_dir": str(samples_dir),
        "generated_rows": generated_rows,
        "lans_summary": {
            "total_generated": summary.get("total_generated"),
            "scv_filter_event_count": filtered_event_count,
            "scv_filtered_count": filtered_sample_count,
            "scv_filter_event_rate": summary.get("scv_filter_event_rate"),
            "scv_filter_rate": summary.get("scv_filter_rate"),
            "retry_exhausted_count": summary.get("retry_exhausted_count"),
            "strategy_distribution": summary.get("strategy_distribution", {}),
            "post_scv_strategy_distribution": summary.get("post_scv_strategy_distribution", {}),
        },
        "filtered_event_audit": {
            **filtered_summary,
            "avg_filter_events_per_filtered_sample": (
                event_rows / unique_sample_ids if unique_sample_ids else 0.0
            ),
        },
        "explanations": explanations,
    }


def render_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Training Signal Audit",
        "",
        f"- exp_name: `{audit['exp_name']}`",
        f"- training_mode: `{audit['training_mode']}`",
        f"- stage_mode: `{audit['stage_mode']}`",
        f"- prompt_variant: `{audit['prompt_variant']}`",
        f"- configured_train_count: `{audit['configured_train_count']}`",
        f"- effective_train_count: `{audit['effective_train_count']}`",
        f"- effective_lans_enabled: `{audit['effective_lans_enabled']}`",
        f"- effective_scv_enabled: `{audit['effective_scv_enabled']}`",
        "",
        "## LANS / SCV Summary",
        "",
    ]
    ls = audit["lans_summary"]
    fa = audit["filtered_event_audit"]
    lines.extend(
        [
            f"- total_generated: `{ls['total_generated']}`",
            f"- scv_filter_event_count: `{ls['scv_filter_event_count']}`",
            f"- scv_filtered_count: `{ls['scv_filtered_count']}`",
            f"- generated_rows: `{audit['generated_rows']}`",
            f"- filtered_event_rows: `{fa['row_count']}`",
            f"- filtered_unique_sample_ids: `{fa['unique_sample_ids']}`",
            f"- avg_filter_events_per_filtered_sample: `{fa['avg_filter_events_per_filtered_sample']:.4f}`",
            "",
            "## Explanations",
            "",
        ]
    )
    for explanation in audit["explanations"]:
        lines.append(f"- {explanation}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit training LANS/SCV signals.")
    parser.add_argument(
        "--checkpoint-manifest",
        type=str,
        required=True,
        help="Path to training checkpoint run_manifest.json",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_training_signal(Path(args.checkpoint_manifest))

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = render_markdown(audit)
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
