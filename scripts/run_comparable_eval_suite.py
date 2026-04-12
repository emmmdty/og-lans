#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a comparable cross-family evaluation suite across API, local base model,
and local adapter checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

from oglans.utils.academic_eval import (
    ACADEMIC_MAIN_TABLE_METRICS,
    COST_REPORT_METRICS,
    EFFICIENCY_REPORT_METRICS,
    append_efficiency_metrics,
    extract_report_metrics,
)
from oglans.utils.compare_contract import extract_compare_contract, validate_compare_contract_match


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VARIANTS = (
    {"name": "single_pass_zeroshot", "prompt_variant": "zeroshot", "stage_mode": "single_pass"},
    {"name": "single_pass_fewshot", "prompt_variant": "fewshot", "stage_mode": "single_pass"},
    {"name": "two_stage_zeroshot", "prompt_variant": "zeroshot", "stage_mode": "two_stage"},
    {"name": "two_stage_fewshot", "prompt_variant": "fewshot", "stage_mode": "two_stage"},
)
DIAGNOSTIC_METRICS = (
    "strict_f1",
    "relaxed_f1",
    "type_f1",
    "schema_compliance_rate",
    "hallucination_rate",
    "single_event_doc_role_micro_f1",
    "multi_event_doc_role_micro_f1",
    "parse_success_rate",
    "parse_error_rate",
    "avg_gold_events",
    "avg_predicted_events",
    "avg_gold_event_types",
    "avg_schema_event_types",
    "stage1_gold_coverage_rate",
    "stage1_exact_match_rate",
    "stage1_miss_rate",
    "stage1_overprediction_rate",
    "avg_stage1_predicted_types",
    "correction_applied_rate",
    "records_split_count",
    "roles_rewritten_count",
    "roles_added_count",
    "events_dropped_after_correction",
)


class RunRecord(NamedTuple):
    family: str
    variant: str
    command: List[str]
    summary_file: str
    output_file: str
    returncode: int
    duration_seconds: float
    ok: bool
    error: Optional[str] = None


def build_local_eval_command(
    *,
    evaluate_path: Path,
    config: str,
    protocol: str,
    role_alias_map: str,
    model_name_or_path: str,
    checkpoint_path: Optional[str],
    split: str,
    seed: int,
    output_file: Path,
    summary_file: Path,
    batch_size: int,
    prompt_variant: str,
    fewshot_num_examples: int,
    stage_mode: str,
    fewshot_selection_mode: str,
    fewshot_pool_split: str,
    train_tune_ratio: float,
    research_split_manifest: str,
    report_primary_metric: str,
    canonical_metric_mode: str,
    base_only: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        str(evaluate_path),
        "--config", config,
        "--protocol", protocol,
        "--role_alias_map", role_alias_map,
        "--split", split,
        "--seed", str(seed),
        "--batch_size", str(batch_size),
        "--output_file", str(output_file),
        "--summary_file", str(summary_file),
        "--model_name_or_path", model_name_or_path,
        "--prompt_variant", prompt_variant,
        "--stage_mode", stage_mode,
        "--fewshot_selection_mode", fewshot_selection_mode,
        "--fewshot_pool_split", fewshot_pool_split,
        "--train_tune_ratio", str(train_tune_ratio),
        "--research_split_manifest", research_split_manifest,
        "--report_primary_metric", report_primary_metric,
        "--canonical_metric_mode", canonical_metric_mode,
    ]
    if prompt_variant == "fewshot":
        cmd.extend(["--fewshot_num_examples", str(fewshot_num_examples)])
    if base_only:
        cmd.append("--base_only")
    else:
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required when base_only=False")
        cmd.extend(["--checkpoint", checkpoint_path])
    return cmd


def build_api_eval_command(
    *,
    evaluate_api_path: Path,
    config: str,
    protocol: str,
    role_alias_map: str,
    model: str,
    base_url: Optional[str],
    split: str,
    seed: int,
    output_file: Path,
    summary_file: Path,
    concurrency: int,
    prompt_variant: str,
    fewshot_num_examples: int,
    stage_mode: str,
    fewshot_selection_mode: str,
    fewshot_pool_split: str,
    train_tune_ratio: float,
    research_split_manifest: str,
    report_primary_metric: str,
    canonical_metric_mode: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(evaluate_api_path),
        "--config", config,
        "--protocol", protocol,
        "--role_alias_map", role_alias_map,
        "--model", model,
        "--split", split,
        "--seed", str(seed),
        "--concurrency", str(concurrency),
        "--output_file", str(output_file),
        "--summary_file", str(summary_file),
        "--stage_mode", stage_mode,
        "--fewshot_selection_mode", fewshot_selection_mode,
        "--fewshot_pool_split", fewshot_pool_split,
        "--train_tune_ratio", str(train_tune_ratio),
        "--research_split_manifest", research_split_manifest,
        "--report_primary_metric", report_primary_metric,
        "--canonical_metric_mode", canonical_metric_mode,
    ]
    if base_url:
        cmd.extend(["--base_url", base_url])
    if prompt_variant == "fewshot":
        cmd.append("--use_fewshot")
        cmd.extend(["--fewshot_num_examples", str(fewshot_num_examples)])
    return cmd


def run_one(cmd: List[str]) -> tuple[int, float, Optional[str]]:
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    duration = time.time() - started
    if proc.returncode == 0:
        return 0, duration, None
    return proc.returncode, duration, (proc.stderr or proc.stdout or "").strip()[-2000:]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_family_records(records: Sequence[Mapping[str, Any]]) -> str:
    compare_blocks = []
    for record in records:
        compare = record.get("compare")
        if not isinstance(compare, Mapping):
            raise ValueError(f"missing compare block: {record}")
        compare_blocks.append(compare)
    return validate_compare_contract_match(compare_blocks)


def _metric_row(summary: Mapping[str, Any]) -> Dict[str, float]:
    row = extract_report_metrics(
        summary,
        required_metrics=ACADEMIC_MAIN_TABLE_METRICS + DIAGNOSTIC_METRICS[:5],
        optional_metrics=COST_REPORT_METRICS + EFFICIENCY_REPORT_METRICS + DIAGNOSTIC_METRICS[5:],
    )
    return append_efficiency_metrics(row)


def build_table_sections(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    main_table: List[Dict[str, Any]] = []
    diagnostic_table: List[Dict[str, Any]] = []
    cost_table: List[Dict[str, Any]] = []
    for record in records:
        metrics = dict(record.get("metrics", {}))
        main_table.append(
            {
                "variant": record["variant"],
                "family": record["family"],
                **{metric: metrics.get(metric) for metric in ACADEMIC_MAIN_TABLE_METRICS},
            }
        )
        diagnostic_table.append(
            {
                "variant": record["variant"],
                "family": record["family"],
                **{metric: metrics.get(metric) for metric in DIAGNOSTIC_METRICS},
            }
        )
        cost_table.append(
            {
                "variant": record["variant"],
                "family": record["family"],
                "token_usage_kind": record.get("compare", {}).get("token_usage_kind"),
                **{metric: metrics.get(metric) for metric in COST_REPORT_METRICS + EFFICIENCY_REPORT_METRICS},
            }
        )
    return {
        "main_table": main_table,
        "diagnostic_table": diagnostic_table,
        "cost_table": cost_table,
    }


def validate_suite_completeness(records: Sequence[Mapping[str, Any]]) -> None:
    expected_pairs = {
        (family, variant["name"])
        for family in ("api", "base", "checkpoint")
        for variant in VARIANTS
    }
    actual_pairs = {
        (str(record.get("family")), str(record.get("variant")))
        for record in records
    }
    missing_pairs = sorted(expected_pairs - actual_pairs)
    if missing_pairs:
        raise ValueError(f"incomplete comparable suite: missing successful runs for {missing_pairs}")


def render_markdown_table(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str], *, include_token_usage_kind: bool = False) -> List[str]:
    if not rows:
        return ["_No rows_"]
    headers = ["variant", "family"]
    if include_token_usage_kind:
        headers.append("token_usage_kind")
    headers.extend(metrics)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        cells: List[str] = [str(row.get("variant", "")), str(row.get("family", ""))]
        if include_token_usage_kind:
            cells.append(str(row.get("token_usage_kind", "")))
        for metric in metrics:
            value = row.get(metric)
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            elif value is None:
                cells.append("null")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comparable cross-family evaluation suite.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--protocol", type=str, default="configs/eval_protocol.yaml")
    parser.add_argument("--role_alias_map", type=str, default="configs/role_aliases_duee_fin.yaml")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--api_model", type=str, required=True)
    parser.add_argument("--api_base_url", type=str, default=None)
    parser.add_argument("--fewshot_num_examples", type=int, default=3)
    parser.add_argument("--fewshot_selection_mode", type=str, default="dynamic")
    parser.add_argument("--fewshot_pool_split", type=str, default="train_fit")
    parser.add_argument("--train_tune_ratio", type=float, default=0.1)
    parser.add_argument("--research_split_manifest", type=str, required=True)
    parser.add_argument("--report_primary_metric", type=str, default="doc_role_micro_f1")
    parser.add_argument("--canonical_metric_mode", type=str, default="analysis_only")
    parser.add_argument("--local_batch_size", type=int, default=8)
    parser.add_argument("--api_concurrency", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    suite_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "logs" / "comparable_eval" / f"suite_{ts}")
    suite_dir.mkdir(parents=True, exist_ok=True)

    evaluate_path = PROJECT_ROOT / "evaluate.py"
    evaluate_api_path = PROJECT_ROOT / "evaluate_api.py"
    records: List[RunRecord] = []
    summaries: List[Dict[str, Any]] = []

    for family in ("api", "base", "checkpoint"):
        for variant in VARIANTS:
            variant_dir = suite_dir / family / variant["name"]
            variant_dir.mkdir(parents=True, exist_ok=True)
            output_file = variant_dir / "eval_results.jsonl"
            summary_file = variant_dir / "eval_results_summary.json"
            if family == "api":
                cmd = build_api_eval_command(
                    evaluate_api_path=evaluate_api_path,
                    config=args.config,
                    protocol=args.protocol,
                    role_alias_map=args.role_alias_map,
                    model=args.api_model,
                    base_url=args.api_base_url,
                    split=args.split,
                    seed=args.seed,
                    output_file=output_file,
                    summary_file=summary_file,
                    concurrency=args.api_concurrency,
                    prompt_variant=variant["prompt_variant"],
                    fewshot_num_examples=args.fewshot_num_examples,
                    stage_mode=variant["stage_mode"],
                    fewshot_selection_mode=args.fewshot_selection_mode,
                    fewshot_pool_split=args.fewshot_pool_split,
                    train_tune_ratio=args.train_tune_ratio,
                    research_split_manifest=args.research_split_manifest,
                    report_primary_metric=args.report_primary_metric,
                    canonical_metric_mode=args.canonical_metric_mode,
                )
            else:
                cmd = build_local_eval_command(
                    evaluate_path=evaluate_path,
                    config=args.config,
                    protocol=args.protocol,
                    role_alias_map=args.role_alias_map,
                    model_name_or_path=args.base_model,
                    checkpoint_path=(None if family == "base" else args.checkpoint),
                    split=args.split,
                    seed=args.seed,
                    output_file=output_file,
                    summary_file=summary_file,
                    batch_size=args.local_batch_size,
                    prompt_variant=variant["prompt_variant"],
                    fewshot_num_examples=args.fewshot_num_examples,
                    stage_mode=variant["stage_mode"],
                    fewshot_selection_mode=args.fewshot_selection_mode,
                    fewshot_pool_split=args.fewshot_pool_split,
                    train_tune_ratio=args.train_tune_ratio,
                    research_split_manifest=args.research_split_manifest,
                    report_primary_metric=args.report_primary_metric,
                    canonical_metric_mode=args.canonical_metric_mode,
                    base_only=(family == "base"),
                )

            code, duration, error = run_one(cmd)
            records.append(
                RunRecord(
                    family=family,
                    variant=variant["name"],
                    command=cmd,
                    summary_file=str(summary_file),
                    output_file=str(output_file),
                    returncode=code,
                    duration_seconds=duration,
                    ok=(code == 0),
                    error=error,
                )
            )
            if code != 0:
                continue
            summary = load_json(summary_file)
            compare = extract_compare_contract(summary)
            summaries.append({
                "family": family,
                "variant": variant["name"],
                "compare": compare,
                "summary_file": str(summary_file),
                "metrics": _metric_row(summary),
            })

    validate_suite_completeness(summaries)
    variant_contract_hashes: Dict[str, str] = {}
    variants: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_name = variant["name"]
        variant_records = [record for record in summaries if record["variant"] == variant_name]
        if not variant_records:
            continue
        variant_contract_hashes[variant_name] = validate_family_records(variant_records)
        variants[variant_name] = {
            "comparable_contract_hash": variant_contract_hashes[variant_name],
            "records": variant_records,
        }
    table_sections = build_table_sections(summaries)
    suite_summary = {
        "timestamp": ts,
        "config": args.config,
        "protocol": args.protocol,
        "split": args.split,
        "seed": args.seed,
        "variant_contract_hashes": variant_contract_hashes,
        "records": [record._asdict() for record in records],
        "variants": variants,
        **table_sections,
    }
    suite_path = suite_dir / "suite_summary.json"
    suite_path.write_text(json.dumps(suite_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = ["# Comparable Evaluation Suite", ""]
    md_lines.append("## Main Table")
    md_lines.extend(render_markdown_table(table_sections["main_table"], ACADEMIC_MAIN_TABLE_METRICS))
    md_lines.append("")
    md_lines.append("## Diagnostic Table")
    md_lines.extend(render_markdown_table(table_sections["diagnostic_table"], DIAGNOSTIC_METRICS))
    md_lines.append("")
    md_lines.append("## Cost Table")
    md_lines.extend(
        render_markdown_table(
            table_sections["cost_table"],
            COST_REPORT_METRICS + EFFICIENCY_REPORT_METRICS,
            include_token_usage_kind=True,
        )
    )
    md_lines.append("")
    suite_md = suite_dir / "suite_summary.md"
    suite_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[DONE] {suite_path}")
    print(f"[DONE] {suite_md}")


if __name__ == "__main__":
    main()
